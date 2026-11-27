package session

import (
	"context"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

func (h *headStart) submitLate(plan headPlan) lateSubmission {
	if !plan.hasFresh() || !h.lateAllowed() {
		return lateSubmission{dispatchReadyAt: time.Now()}
	}
	dispatchReadyAt := time.Now()
	calls := h.submit(plan, true)
	requestIDs := make([]uint64, 0, len(calls))
	for _, call := range calls {
		requestIDs = append(requestIDs, call.requestID)
	}
	return lateSubmission{
		requestIDs: requestIDs, dispatchReadyAt: dispatchReadyAt,
		dispatchedAt: time.Now(), dispatched: true,
	}
}

func (h *headStart) submit(plan headPlan, late bool) []*headCall {
	calls := make([]*headCall, 0, len(plan.parts))
	for _, part := range plan.parts {
		call := part.reuse
		if call == nil {
			call = h.start(part.req, part.text, late)
		}
		calls = append(calls, call)
	}
	if late {
		h.text = plan.heard
		h.lateAsked = true
	}
	h.push(calls)
	h.r.logf("debug", "asking the language model ahead about %d characters", utf8.RuneCountInString(plan.rest))
	return calls
}

// send sends the requests of text, its dictionary's aliases replaced, and
// gives up those of texts before the last headKeep.
func (h *headStart) send(text string) bool {
	plan, ok, discard := h.prepare(text)
	if !ok {
		if discard {
			h.drop()
		}
		return false
	}
	h.submit(plan, false)
	return true
}

func (h *headStart) start(req llm.Request, text string, late bool) *headCall {
	r := h.r
	ctx, cancel := context.WithCancel(r.currentLLMContext())
	origin := "headstart_recording"
	if late {
		origin = "headstart_late"
	}
	requestID := r.llmRequestSequence.Add(1)
	c := &headCall{req: req, requestID: requestID, text: text, late: late, origin: origin, cancel: cancel, done: make(chan struct{})}
	ctx = context.WithValue(ctx, requestOriginContextKey{}, requestOriginContextValue{call: c})
	ctx = llm.WithTraceTag(ctx, origin)
	req.OnDelta = c.add
	ctx = llm.WithTraceRequestID(ctx, requestID)
	h.mu.Lock()
	h.requestCalls = append(h.requestCalls, c)
	h.mu.Unlock()
	go func() {
		defer close(c.done)
		defer cancel()
		c.resp, c.err = r.llmChatContext(ctx, req)
		r.logShadowEvent(map[string]any{
			"event": "voxt_headstart_request_done", "run_id": r.id,
			"session_id": r.shadowSessionID, "request_id": requestID,
			"source": origin, "succeeded": c.err == nil,
			"request_plan_sha256": r.head.partPlanHash(headPart{req: req}),
		})
		r.m.noteRate(h.service, c.resp, c.err)
	}()
	return c
}

func (h *headStart) logRequestStartsAfterStop(stopAt time.Time) {
	h.mu.Lock()
	calls := append([]*headCall(nil), h.requestCalls...)
	h.mu.Unlock()
	for _, call := range calls {
		call.mu.Lock()
		requestStartedAt := call.requestStartedAt
		shouldLog := !requestStartedAt.IsZero() && !call.requestStartLogged
		call.requestStartLogged = call.requestStartLogged || shouldLog
		call.mu.Unlock()
		if shouldLog {
			h.r.logHeadStartRequestStart(call, requestStartedAt, stopAt)
		}
	}
}
