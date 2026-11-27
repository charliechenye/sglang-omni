package omni

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// loopback is the client of the servers: never through a proxy, no
// overall timeout (a Final takes as long as the audio needs; contexts and
// the stall watch bound it).
var loopback = &http.Client{Transport: &http.Transport{
	Proxy:               nil,
	MaxIdleConnsPerHost: 4,
	IdleConnTimeout:     30 * time.Second,
}}

// prior is what a live session already has for a Final: the preview's text
// of a recording the Final starts with, whose beginning the server keeps while
// decoding agrees, and the session, whose encoded audio it reuses ("" none).
// Of the rest after the pieces the session has what it prefilled from where
// the last one ended, and no text. All of it only saves work: the text is
// decoding's.
//
// note (chenye): guessAutoLanguage opts into server-side language-header
// and transcript guesses for a whole, auto-language Final.
type prior struct {
	text, session     string
	guessAutoLanguage bool
}

// finalRequest is one request of a Final.
type finalRequest struct {
	pcm       []byte // PCM16 mono at 16 kHz
	language  string // "" detects
	context   string
	maxTokens int
	layout    string
	preview   bool  // a provisional request with faster disconnect detection
	known     prior // what the live session has of the audio
	// pieceEnd is the sample a piece of the live session's recording ends
	// at (0: not a piece): the session decodes it from its Final's start
	// and moves that start here.
	pieceEnd          int
	includeCandidates bool
	onCandidate       func(finalCandidate)
}

// finalResult is a request's answer.
type finalResult struct {
	text        string
	tokens      int    // generated
	language    string // detected (or the hint), "" when none
	length      bool   // the budget ended it
	windowTexts []string
	segments    []struct {
		Start   float64 `json:"start"`
		End     float64 `json:"end"`
		Speaker string  `json:"speaker"`
		Text    string  `json:"text"`
	}
}

type finalCandidate struct {
	text     string
	metadata asr.CandidateMetadata
}

// body is the request's multipart form (Voxt's OmniMultipartBody for
// Qwen3-ASR: streamed, with the generation metadata).
func (r finalRequest) body(model string) (*bytes.Buffer, string) {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	field := func(k, v string) { _ = w.WriteField(k, v) }
	field("model", model)
	field("stream", "true")
	field("response_format", "json")
	if r.language != "" {
		field("language", r.language)
	}
	if r.context != "" {
		field("prompt", r.context)
	}
	if r.maxTokens > 0 {
		field("max_new_tokens", itoa(r.maxTokens))
	}
	field("stop_at_end_of_text", "true")
	field("stop_on_token_loop", "true")
	field("include_generation_metadata", "true")
	if r.includeCandidates {
		field("include_transcript_candidates", "true")
	}
	if r.preview {
		field("preview", "true")
	}
	if r.layout != "" {
		field("audio_layout", r.layout)
	}
	if r.known.text != "" {
		field("guessed_text", r.known.text)
	}
	if r.known.session != "" {
		field("realtime_session", r.known.session)
		if r.pieceEnd > 0 {
			field("realtime_piece_end", itoa(r.pieceEnd))
		}
		if r.known.guessAutoLanguage && r.language == "" && r.known.text == "" && !r.preview {
			field("guess_auto_language", "true")
		}
	}
	fw, _ := w.CreateFormFile("file", "audio.wav")
	_, _ = fw.Write(asr.WAV(r.pcm, sampleRate))
	_ = w.Close()
	return &b, w.FormDataContentType()
}

// transcribe sends one request on l's server and reads its event stream.
// A stream silent for stall (the server's heartbeat is every 250 ms) has
// hung the server: it is killed.
func transcribe(ctx context.Context, l *lease, r finalRequest, stall time.Duration) (finalResult, error) {
	return transcribeTimed(ctx, l, r, stall, 0)
}

func transcribeTimed(ctx context.Context, l *lease, r finalRequest, stall time.Duration, requestID uint64) (finalResult, error) {
	totalStarted := time.Now()
	var httpRequestStarted time.Time
	defer func() { logRequestStage(l.s.rt, requestID, "final", "total", totalStarted, len(r.pcm), 0) }()
	started := time.Now()
	body, ctype := r.body(l.endpoint().model)
	logRequestStage(l.s.rt, requestID, "final", "multipart", started, len(r.pcm), body.Len())
	// The hint goes to the log, never the recognised text: a wrong hint
	// translated Chinese dictation (#3398) and nothing showed it.
	if !r.preview {
		l.s.rt.logf("info", "%s: Final request with language hint %q", l.s.spec.ID, r.language)
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.endpoint().base()+"/v1/audio/transcriptions", body)
	if err != nil {
		return finalResult{}, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Err: err}
	}
	req.Header.Set("Content-Type", ctype)
	var stalled atomic.Bool
	watch := time.AfterFunc(stall, func() { stalled.Store(true); cancel() })
	defer watch.Stop()
	started = time.Now()
	httpRequestStarted = started
	resp, err := loopback.Do(req)
	logRequestStage(l.s.rt, requestID, "final", "http_headers", started, body.Len(), 0)
	if err != nil {
		return finalResult{}, stallOr(ctx, l, stalled.Load(), err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		code, msg := asr.ErrorBody(raw)
		e := asr.HTTPError(Provider, resp.StatusCode, code, msg)
		if resp.StatusCode == http.StatusBadRequest {
			e.Kind = asr.KindProtocol
		}
		return finalResult{}, e
	}
	started = time.Now()
	defer func() { logRequestStage(l.s.rt, requestID, "final", "sse", started, 0, 0) }()
	rd := bufio.NewReaderSize(resp.Body, 64<<10)
	var done *finalResult
	for {
		watch.Reset(stall)
		line, err := rd.ReadString('\n')
		if err != nil {
			if done != nil && errors.Is(err, io.EOF) {
				return *done, nil
			}
			if errors.Is(err, io.EOF) {
				return finalResult{}, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the transcription ended before its result", Retryable: true}
			}
			return finalResult{}, stallOr(ctx, l, stalled.Load(), err)
		}
		payload, ok := strings.CutPrefix(strings.TrimRight(line, "\r\n"), "data:")
		if !ok {
			continue // a heartbeat (":") or a blank line
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			if done == nil {
				return finalResult{}, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the transcription ended before its result", Retryable: true}
			}
			return *done, nil
		}
		res, candidate, err := parseEvent(payload)
		if err != nil {
			return finalResult{}, err
		}
		if res != nil {
			done = res
		} else if candidate != nil && done == nil && r.onCandidate != nil {
			candidate.metadata.RequestID = requestID
			candidate.metadata.ReceivedAt = time.Now()
			candidate.metadata.RequestElapsedMS = float64(candidate.metadata.ReceivedAt.Sub(totalStarted).Microseconds()) / 1000
			candidate.metadata.HTTPRequestStartOffsetMS = float64(httpRequestStarted.Sub(totalStarted).Microseconds()) / 1000
			r.onCandidate(*candidate)
		} else {
		}
	}
}

// parseEvent reads one server-sent event: the result for
// transcript.text.done, nil for others, an error for "error". Only the
// error's code is kept: messages may echo the request.
func parseEvent(payload string) (*finalResult, *finalCandidate, error) {
	var ev struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		Segments []struct {
			Start   float64 `json:"start"`
			End     float64 `json:"end"`
			Speaker string  `json:"speaker"`
			Text    string  `json:"text"`
		} `json:"segments"`
		Metadata *struct {
			Tokens      *int     `json:"generated_token_count"`
			Language    *string  `json:"language"`
			Finish      string   `json:"finish_reason"`
			WindowTexts []string `json:"window_texts"`
		} `json:"generation_metadata"`
		Error *struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(payload), &ev); err != nil || ev.Type == "" {
		return nil, nil, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "an unreadable transcription event"}
	}
	switch ev.Type {
	case "transcript.text.done":
		r := &finalResult{text: ev.Text, segments: ev.Segments}
		if m := ev.Metadata; m != nil {
			if m.Tokens == nil || m.Finish == "" {
				return nil, nil, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "the generation metadata is incomplete"}
			}
			r.tokens, r.length = *m.Tokens, m.Finish == "length"
			r.windowTexts = m.WindowTexts
			if m.Language != nil {
				r.language = *m.Language
			}
		}
		return r, nil, nil
	case "transcript.text.candidate":
		fields := make(map[string]json.RawMessage)
		if json.Unmarshal([]byte(payload), &fields) != nil {
			return nil, nil, nil
		}
		read := func(name string, value any) bool {
			raw, ok := fields[name]
			return ok && json.Unmarshal(raw, value) == nil
		}
		var sequence uint64
		var generatedTokens int
		var elapsedMS float64
		var firstTokenMS float64
		var snapshotBuildMS float64
		var publishedCandidateCount uint64
		var snapshotBuildMSTotal float64
		var text string
		if !read("sequence", &sequence) || sequence == 0 ||
			!read("generated_token_count", &generatedTokens) ||
			generatedTokens <= 0 || !read("elapsed_ms", &elapsedMS) ||
			math.IsNaN(elapsedMS) || math.IsInf(elapsedMS, 0) || elapsedMS < 0 ||
			!read("text", &text) || text == "" {
			return nil, nil, nil
		}
		_ = read("first_transcript_token_elapsed_ms", &firstTokenMS)
		_ = read("snapshot_build_ms", &snapshotBuildMS)
		_ = read("published_candidate_count", &publishedCandidateCount)
		_ = read("snapshot_build_ms_total", &snapshotBuildMSTotal)
		candidate := &finalCandidate{
			text: text,
			metadata: asr.CandidateMetadata{
				Source: "http_final", Sequence: sequence,
				GeneratedTokens: generatedTokens, NativeElapsedMS: elapsedMS,
				FirstTranscriptTokenMS: firstTokenMS, SnapshotBuildMS: snapshotBuildMS,
				PublishedCandidateCount: publishedCandidateCount,
				SnapshotBuildMSTotal:    snapshotBuildMSTotal,
			},
		}
		var language string
		if read("language", &language) {
			candidate.metadata.Language = language
		} else {
		}
		return nil, candidate, nil
	case "error":
		code := "error"
		if ev.Error != nil && ev.Error.Code != "" {
			code = ev.Error.Code
		}
		return nil, nil, &asr.Error{Kind: asr.KindServer, Provider: Provider, Code: code, Detail: "the transcription failed"}
	}
	return nil, nil, nil
}

// stallOr is a transport error's *asr.Error, killing a stalled server.
func stallOr(ctx context.Context, l *lease, stalled bool, err error) error {
	if stalled {
		l.broken("no answer")
		return &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the local speech model stopped answering", Retryable: true, Err: err}
	}
	// A server that crashed breaks the connection a moment before its
	// exit is seen.
	if ctx.Err() == nil && l.p.wait(500*time.Millisecond) {
		return &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: fmt.Sprintf("the local speech model exited (%v)", l.p.exitErr), Retryable: true, Err: err}
	}
	return asr.NetError(ctx, Provider, err)
}
