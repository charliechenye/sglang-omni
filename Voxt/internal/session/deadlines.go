package session

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

// CleanupLatencyPolicy optionally allows ordinary Transcribe to fall back to raw text after Stop.
type CleanupLatencyPolicy struct {
	// A non-positive value disables automatic fallback.
	AutoFallbackAfter time.Duration
}

type requestOriginContextKey struct{}

type requestOriginContextValue struct {
	call *headCall
}

// llmPrefillPerChar is a request's extra time per character of the input
// before its answer starts: a long selection's prompt takes seconds to
// read.
const llmPrefillPerChar = 5 * time.Millisecond

// llmDeadlines is a request over an input of chars characters, its
// deadlines set (behavior.md 5.5): once the answer streams, no gap of
// LLMStall between its pieces; in all LLMBase plus llmPrefillPerChar and
// LLMPerChar per character, at least floor. The first piece has no
// deadline of its own: a model that reasons before it answers, or loads
// on its first request, sends nothing for a while and is not stuck.
func (cfg *Config) llmDeadlines(req llm.Request, chars int, floor time.Duration) llm.Request {
	req.StallTimeout = cfg.LLMStall
	req.Timeout = max(floor, cfg.LLMBase+time.Duration(chars)*(llmPrefillPerChar+cfg.LLMPerChar))
	return req
}

// llmFloor is a request's least deadline: the wait the user chose
// (settings.Models.LLMWait, #3821), else auto.
func (r *run) llmFloor(auto time.Duration) time.Duration {
	return cmp.Or(r.settings.Models.LLMWaitFor(), auto)
}

// TimedOut says err is a language model's deadline passing: llmChat's,
// or the client's own.
func TimedOut(err error) bool {
	return errors.Is(err, context.DeadlineExceeded) || llm.Code(err) == llm.CodeTimeout
}

// llmChat runs req within its deadlines, until SkipLLM; an error says
// which one passed. One kind of picture goes with it: the Rewrite box's
// own (only those, the user's own), else the picture the user copied,
// else the window's.
func (r *run) llmChat(req llm.Request) (llm.Response, error) {
	return r.llmChatContext(r.currentLLMContext(), req)
}

func (r *run) currentLLMContext() context.Context {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	return r.llmCtx
}

func (r *run) llmChatContext(parent context.Context, req llm.Request) (llm.Response, error) {
	r.m.mu.Lock()
	client, choice, deadlineAt := r.svc.LLM, r.cleanupChoice, r.cleanupDeadline
	if parent == nil || parent.Err() != nil || r.ctx.Err() != nil || choice != cleanupUndecided || !r.cleanupDecisionOpenLocked() {
		r.m.mu.Unlock()
		return llm.Response{}, context.Canceled
	}
	if !deadlineAt.IsZero() {
		remaining := deadlineAt.Sub(r.now())
		if remaining <= 0 {
			r.m.mu.Unlock()
			return llm.Response{}, context.DeadlineExceeded
		}
		req.Timeout = min(req.Timeout, remaining)
	}
	r.m.mu.Unlock()
	if client == nil {
		return llm.Response{}, errors.New("no language model is ready")
	}
	ctx, cancel := context.WithTimeout(parent, req.Timeout)
	defer cancel()
	traceTag := llm.TraceTag(parent)
	if traceTag == "" {
		traceTag = "cleanup"
	}
	ctx = llm.WithTraceTag(ctx, traceTag)
	hint := ""
	switch {
	case len(r.req.Images) > 0:
		req.Images, req.ImagesRequired = r.req.Images, true
		hint = "box"
		r.previewImageContext(parent, "", nil)
	case r.clip != nil: // the picture the user copied wins over the window's
		req.Images = [][]byte{r.clip}
		hint = "clipboard"
		r.previewImageContext(parent, "clipboard", r.clip)
	case r.shot != nil:
		if jpeg := r.shot(); jpeg != nil {
			req.Images = [][]byte{jpeg}
			hint = "window"
			r.previewImageContext(parent, "window", jpeg)
		}
	}
	if hint != "" {
		r.m.updateIf(r, func(s *Snapshot) bool {
			if parent.Err() != nil || r.ctx.Err() != nil || !r.cleanupDecisionOpenLocked() {
				return false
			}
			s.ImageHint = hint
			return true
		})
	}
	if parent.Err() != nil || r.ctx.Err() != nil {
		return llm.Response{}, context.Canceled
	}
	if r.shadowTraceEnabled {
		requestStartedAt := time.Now()
		if origin, ok := parent.Value(requestOriginContextKey{}).(requestOriginContextValue); ok {
			stopAt := r.monotonicStoppedAt()
			shouldLog := false
			if origin.call != nil {
				origin.call.mu.Lock()
				origin.call.requestStartedAt = requestStartedAt
				shouldLog = !stopAt.IsZero() && !origin.call.requestStartLogged
				origin.call.requestStartLogged = origin.call.requestStartLogged || shouldLog
				origin.call.mu.Unlock()
			}
			if shouldLog {
				r.logHeadStartRequestStart(origin.call, requestStartedAt, stopAt)
			}
		}
	}
	resp, err := client.Chat(ctx, req)
	if err == nil && resp.ImagesDropped && hint != "" {
		// The model answered from the text alone: the card must not
		// claim a picture went with the request.
		r.m.updateIf(r, func(s *Snapshot) bool {
			if ctx.Err() != nil || parent.Err() != nil || !r.cleanupDecisionOpenLocked() {
				return false
			}
			s.ImageHint = ""
			return true
		})
	}
	if err == nil || r.ctx.Err() != nil {
		return resp, err
	}
	var e *llm.Error
	switch {
	case errors.As(err, &e) && e.Params["wait"] == llm.WaitStall:
		err = fmt.Errorf("no new token for %v after %s pieces: %w", req.StallTimeout, e.Params["deltas"], err)
	case ctx.Err() == context.DeadlineExceeded:
		err = fmt.Errorf("no answer within %v: %w", req.Timeout.Round(time.Second/10), err)
	}
	return resp, err
}

func (r *run) logHeadStartRequestStart(call *headCall, requestStartedAt, stopAt time.Time) {
	if !r.shadowTraceEnabled || call == nil || stopAt.IsZero() {
		return
	}
	r.logShadowEvent(map[string]any{
		"event": "voxt_headstart_request_start", "run_id": r.id,
		"session_id": r.shadowSessionID, "request_id": call.requestID,
		"source": call.origin, "request_origin": call.origin,
		"generation_id":            call.generationID,
		"request_plan_sha256":      r.head.partPlanHash(headPart{req: call.req}),
		"stop_to_request_start_ms": float64(requestStartedAt.Sub(stopAt).Microseconds()) / 1000,
		"request_start_valid":      true,
	})
}

// asrLimit is how long recognition may take after Stop for a recording
// of audio: ASRTimeout plus ASRPerSecond per second of it (behavior.md
// 5.5a).
func (cfg *Config) asrLimit(audio time.Duration) time.Duration {
	return cfg.ASRTimeout + time.Duration(audio.Seconds()*float64(cfg.ASRPerSecond))
}
