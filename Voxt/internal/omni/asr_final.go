package omni

import (
	"context"
	"encoding/json"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func (a *ASR) finalBias(b bias) bias {
	if len(b.hotwords) > 0 {
		if b.context == "" {
			b.context = mossDiarizePrompt
		}
		b.context += " Hotwords: " + strings.Join(b.hotwords, ", ")
	}
	return b
}

// final transcribes quiet-cut chunks in order, with a shared token budget
// for Qwen3-ASR and one budget per Whisper window. The first detected
// language, when available, is passed to subsequent requests. known is of
// the recording's start: the chunks after the first have no part of it.
func (a *ASR) final(ctx context.Context, l *lease, pcm []byte, language string, b bias, known prior, onCandidate func(finalCandidate), measurement *candidateMeasurement) (string, string, error) {
	t := a.spec.ASR
	budget := t.tokenBudget(asr.Duration(pcm, sampleRate))
	shared := t.TokensPerSecond > 0 // else each request has the whole budget
	batchWhisper := a.spec.Kind == kindWhisper && a.draft == ""
	chunks := t.splitChunks(pcm)
	if a.spec.Kind == kindWhisper {
		chunks = trimWhisperSilenceTail(chunks)
	}
	if len(chunks) > 1 {
		known.text = "" // the text is of all the chunks: none of them is all of it
		known.guessAutoLanguage = false
	}
	includeCandidates := onCandidate != nil && a.spec.Kind == kindQwen3ASR && len(chunks) == 1
	if onCandidate != nil {
		event := map[string]any{"event": "voxt_final_candidate_protocol", "enabled": includeCandidates}
		if measurement != nil {
			event["run_id"], event["session_id"] = measurement.runID, measurement.sessionID
		}
		if includeCandidates {
			cadence := "bounded"
			if measurement != nil && measurement.cadence == "dense" {
				cadence = "dense"
			}
			event["cadence"], event["decode_path"] = cadence, "http_final_single_chunk"
		} else if a.spec.Kind != kindQwen3ASR {
			event["reason"], event["decode_path"] = "unsupported_model", "unsupported_model"
		} else {
			event["reason"], event["decode_path"] = "multiple_final_chunks", "http_final_multiple_chunks"
		}
		payload, err := json.Marshal(event)
		if err == nil {
			a.rt.logf("info", "%s", payload)
		}
	}
	if onCandidate != nil && !includeCandidates {
		reason := "unsupported_model"
		if a.spec.Kind == kindQwen3ASR {
			reason = "multiple_final_chunks"
		} else {
		}
		runID, sessionID := "", uint64(0)
		if measurement != nil {
			runID, sessionID = measurement.runID, measurement.sessionID
		}
		logCandidateExclusion(a.rt, runID, sessionID, reason, len(chunks))
	} else {
	}
	var parts []string
	for len(chunks) > 0 {
		if shared && budget <= 0 {
			break
		}
		if batchWhisper {
			batchPCM, chunkWindows := packWhisperChunks(chunks)
			if len(chunkWindows) > 1 {
				req := finalRequest{pcm: batchPCM, language: language, maxTokens: budget}
				texts, detected, err := a.transcribeWhisperBatch(ctx, l, req, chunkWindows)
				if err != nil {
					return "", "", err
				}
				if texts != nil {
					parts = append(parts, texts...)
					if language == "" {
						language = detected
					}
					chunks = chunks[len(chunkWindows):]
					continue
				}
				// A mismatched older runtime cannot split this first batch;
				// redo it per chunk and avoid batching the remaining chunks.
				batchWhisper = false
			}
		}
		c := chunks[0]
		chunks = chunks[1:]
		start := known
		known = prior{}
		requestID := nextOfflineRequestID()
		pcm, ok := a.speechOnlyTimed(ctx, requestID, c.pcm)
		if !ok {
			continue
		}
		candidateCadence := ""
		if measurement != nil {
			candidateCadence = measurement.cadence
		}
		req := finalRequest{pcm: pcm, language: language, context: b.context, maxTokens: budget, layout: t.AudioLayout, known: start, includeCandidates: includeCandidates, candidateCadence: candidateCadence}
		if includeCandidates {
			req.onCandidate = func(candidate finalCandidate) {
				originalText := candidate.text
				text, reason, ok := a.finalCandidateText(candidate.text, b.terms)
				candidate.text = text
				if !ok {
					candidate.text = originalText
					logCandidateRecord(a.rt, "voxt_final_candidate_suppressed", candidate, time.Time{}, reason)
				} else {
					onCandidate(candidate)
				}
			}
		} else {
		}
		r, err := transcribeTimed(ctx, l, req, a.rt.cfg.StallTimeout, requestID)
		if err != nil {
			return "", "", err
		}
		r.text = a.trimEcho(ctx, l, req, a.plainText(r.text), r.length, b.terms)
		if shared {
			budget -= r.tokens
		}
		if language == "" {
			language = r.language
		}
		parts = append(parts, r.text)
	}
	return joinParts(parts), language, nil
}

func (a *ASR) finalCandidateText(text string, terms []string) (string, string, bool) {
	text = a.plainText(text)
	if !utf8.ValidString(text) || strings.ContainsRune(text, '\uFFFD') {
		return "", "invalid_utf8", false
	} else {
	}
	cut, _, certain := echoTail(text, terms, false)
	if cut >= 0 && !certain {
		return "", "uncertain_dictionary_echo", false
	} else if cut >= 0 {
		text = strings.TrimSpace(text[:cut])
	} else {
	}
	if text == "" {
		return "", "empty_after_normalization", false
	} else {
		return text, "", true
	}
}

// transcribeWhisperBatch reassembles each quiet-cut chunk before joinParts
// joins the chunks. The runtime uses spaces between windows of one chunk.
func (a *ASR) transcribeWhisperBatch(ctx context.Context, l *lease, req finalRequest, chunkWindows []int) ([]string, string, error) {
	r, err := transcribe(ctx, l, req, a.rt.cfg.StallTimeout)
	if err != nil || r.windowTexts == nil {
		return nil, "", err
	}
	if len(r.windowTexts) != len(req.pcm)/whisperWindowBytes {
		return nil, "", &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "the Whisper batch returned the wrong number of windows"}
	}
	parts := make([]string, 0, len(chunkWindows))
	for _, count := range chunkWindows {
		var windows []string
		for _, text := range r.windowTexts[:count] {
			if text != "" {
				windows = append(windows, text)
			}
		}
		parts = append(parts, strings.Join(windows, " "))
		r.windowTexts = r.windowTexts[count:]
	}
	return parts, r.language, nil
}
