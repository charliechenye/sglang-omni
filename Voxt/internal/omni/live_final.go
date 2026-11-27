package omni

import (
	"fmt"
	"slices"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// finalPass recognises pcm, the whole recording, once more (what the
// pieces left of it) and ends the stream with the result. The Final
// starts at once: the wrap-up (draining the preview's queue and awaiting
// the server's recognition of it) matters only when the stream's text is
// the fallback, so it runs only for Salvage — the Final failed, or it is
// still running when Salvage is asked — not on every Stop. The socket
// stays open while the Final runs, so the tail's segments still arrive.
func (s *live) finalPass(pcm []byte) {
	if s.conn != nil {
		defer s.conn.CloseNow()
	}
	start := time.Now()
	language := s.language
	finalBias := s.a.finalBias(s.bias)
	var texts []string
	done, busy := 0, time.Duration(0)
	if pc := s.pc; pc != nil {
		s.mu.Lock()
		activePieceDone := pc.activePieceDone
		s.mu.Unlock()
		if activePieceDone != nil {
			<-activePieceDone
		}
		s.mu.Lock()
		done, texts, busy = pc.done, slices.Clone(pc.texts), pc.busy
		if pc.language != "" {
			language = pc.language
		}
		s.mu.Unlock()
	}
	text := ""
	var err error
	finalAttempt := 0
	if tail := pcm[done:]; len(tail) > 0 && (done == 0 || !s.silentTail(tail)) {
		attempt := 1
		finalAttempt = attempt
		var onCandidate func(finalCandidate)
		if s.shadowCandidates {
			onCandidate = func(candidate finalCandidate) {
				candidate.text = joinParts(append(slices.Clone(texts), candidate.text))
				candidate.metadata.SessionID = s.shadowID
				candidate.metadata.Attempt = attempt
				if candidate.text == "" {
					logCandidateRecord(s.a.rt, "voxt_final_candidate_suppressed", candidate, s.stopAt, "empty_after_piece_join")
					return
				}
				s.mu.Lock()
				published := false
				if s.shadowCandidates && s.ctx.Err() == nil && !s.ended {
					published = s.emit(asr.Event{Kind: asr.EventShadowCandidate, Text: candidate.text, Candidate: candidate.metadata})
				}
				s.mu.Unlock()
				if published {
					logCandidateRecord(s.a.rt, "voxt_final_candidate_observed", candidate, s.stopAt, "")
				} else {
					logCandidateRecord(s.a.rt, "voxt_final_candidate_suppressed", candidate, s.stopAt, "session_inactive_or_event_queue_full")
				}
			}
		}
		hint := language
		var known prior
		if s.a.spec.Kind == kindQwen3ASR {
			// The server has prefilled the Final's audio, from the last piece
			// on, and its dictionary (recognisePieces).
			known.session = s.session
			if done == 0 {
				if hint != "" {
					s.mu.Lock()
					known.text = s.asm.text()
					s.mu.Unlock()
				} else if !s.req.LiveResult {
					// note (chenye): an auto-language Final generates a language
					// header. Let the server propose it with the preview text.
					known.guessAutoLanguage = true
				}
			}
		}
		text, language, err = s.a.final(s.ctx, s.l, tail, hint, finalBias, known, onCandidate)
		if err != nil && s.ctx.Err() == nil && !s.l.p.alive() {
			// The server died under the recording (a crash, memory pressure): the audio is all here, so once more on a new one.
			if l, aerr := s.a.rt.acquire(s.ctx, s.a.spec, s.a.dir, s.a.draft); aerr == nil {
				s.a.rt.logf("warning", "%s: its server died during the session (%v): the Final runs again on a new one", s.a.spec.ID, err)
				attempt = 2
				finalAttempt = attempt
				text, language, err = s.a.final(s.ctx, l, tail, hint, finalBias, known, onCandidate)
				l.release()
			}
		}
	}
	took := time.Since(start)
	if err != nil {
		s.mu.Lock()
		s.lateCandidates = false
		s.shadowCandidates = false
		s.mu.Unlock()
	}
	s.logFinal(pcm, done, len(texts), busy, took, err)
	if err != nil {
		if s.ctx.Err() != nil {
			err = asr.NetError(s.ctx, Provider, s.ctx.Err())
		} else {
			// Salvage weighs the preview: deliver the tail still queued
			// for the socket and await its recognition first (bounded,
			// so a stuck server cannot stall the error either).
			s.wrapUpPreview()
		}
		metadata := asr.CandidateMetadata{}
		if s.shadowID != 0 {
			metadata = asr.CandidateMetadata{Source: "http_final", SessionID: s.shadowID, Attempt: finalAttempt, ReceivedAt: time.Now()}
		} else {
		}
		s.end(asr.Event{Kind: asr.EventError, Err: err, Candidate: metadata})
		return
	}
	text = joinParts(append(texts, text))
	res := asr.Result{Text: text, Lang: language, Duration: asr.Duration(pcm, sampleRate), Model: s.a.spec.ID, Latency: took}
	metadata := asr.CandidateMetadata{}
	if s.shadowID != 0 {
		metadata = asr.CandidateMetadata{Source: "http_final", SessionID: s.shadowID, Attempt: finalAttempt, ReceivedAt: time.Now()}
	} else {
	}
	s.end(asr.Event{Kind: asr.EventFinal, Text: text, Result: res, Candidate: metadata})
}

// maxSilentTail is the longest tail after the pieces that is left out when it
// holds no speech.
const maxSilentTail = time.Second

// silentTail says whether the tail the pieces left is short and without speech
// (Silero's): the pieces hold the text, and a model with a dictionary in its
// context reads the dictionary out of the silence.
func (s *live) silentTail(tail []byte) bool {
	if s.a.vad == nil || asr.Duration(tail, sampleRate) > maxSilentTail {
		return false
	}
	segments, err := s.a.vad.Speech(s.ctx, tail, sampleRate)
	return err == nil && len(segments) == 0
}

// slowFinal: a Final after Stop this long gets the Mac's memory in its
// log line (a model swapped out recognises several times slower).
const slowFinal = 5 * time.Second

// logFinal logs the Final after Stop: the audio, the pieces recognised
// while it recorded, the rest, and the Mac's memory when it was slow or
// failed.
func (s *live) logFinal(pcm []byte, done, pieces int, busy, took time.Duration, err error) {
	d := func(b []byte) time.Duration { return asr.Duration(b, sampleRate).Round(100 * time.Millisecond) }
	line := fmt.Sprintf("%s: Final of %v of audio", s.a.spec.ID, d(pcm))
	if pieces > 0 {
		line += fmt.Sprintf(": %d pieces while it recorded (%v of audio in %v), the rest (%v)", pieces, d(pcm[:done]), busy.Round(100*time.Millisecond), d(pcm[done:]))
	}
	level := "info"
	switch {
	case err != nil && s.ctx.Err() != nil:
		level, line = "warning", line+fmt.Sprintf(" stopped after %v", took.Round(time.Millisecond))
	case err != nil:
		level, line = "warning", line+fmt.Sprintf(" failed after %v: %v", took.Round(time.Millisecond), err)
	default:
		line += fmt.Sprintf(" in %v", took.Round(time.Millisecond))
	}
	if err != nil || took > slowFinal {
		line += "; " + memoryNote()
	}
	s.a.rt.logf(level, "%s", line)
}
