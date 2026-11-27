package session

import (
	"context"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

// all are the calls of the texts asked about, the newest text's first.
func (h *headStart) all() []*headCall {
	h.mu.Lock()
	defer h.mu.Unlock()
	var calls []*headCall
	for _, gen := range slices.Backward(h.gens) {
		for _, c := range gen {
			if !slices.Contains(calls, c) {
				calls = append(calls, c)
			}
		}
	}
	return calls
}

// find is the call that asks the same as req, nil for none.
func (h *headStart) find(req llm.Request) *headCall {
	calls := h.all()
	i := slices.IndexFunc(calls, func(c *headCall) bool { return h.sameRequest(c.req, req) })
	if i < 0 {
		return nil
	}
	return calls[i]
}

// push keeps the calls of the text asked about now, with those of the
// texts before it (headKeep of them in all), and gives up the others.
func (h *headStart) push(gen []*headCall) {
	before := h.all()
	h.mu.Lock()
	h.gens = append(slices.Clone(h.gens), gen)
	h.gens = h.gens[max(0, len(h.gens)-headKeep):]
	h.mu.Unlock()
	kept := h.all()
	for _, c := range before {
		if !slices.Contains(kept, c) {
			c.cancel()
		}
	}
}

// drop gives up every call.
func (h *headStart) drop() {
	before := h.all()
	h.mu.Lock()
	h.gens = nil
	h.mu.Unlock()
	for _, c := range before {
		c.cancel()
	}
}

// near is a call for a part that says what text says but for a few words
// at most, and the words its answer must be put right for; nil for none.
func (h *headStart) near(req llm.Request, text string) (*headCall, []spokenWord, []headChange) {
	type match struct {
		call    *headCall
		old     []spokenWord
		changes []headChange
		done    bool
		err     error
	}
	var exact, repaired []match
	for _, c := range h.all() {
		done, err := h.callError(c)
		if c.late && done && err != nil {
			continue
		}
		old, changes, ok := h.nearCompatibility(c.req, c.text, req, text)
		if !ok {
			continue
		}
		m := match{call: c, old: old, changes: changes, done: done, err: err}
		if len(changes) == 0 {
			exact = append(exact, m)
		} else {
			repaired = append(repaired, m)
		}
	}
	choose := func(matches []match) *match {
		if len(matches) == 0 {
			return nil
		}
		first := matches[0]
		if first.call.late && !first.done {
			for i := 1; i < len(matches); i++ {
				if matches[i].done && matches[i].err == nil {
					return &matches[i]
				}
			}
		}
		return &first
	}
	chosen := choose(exact)
	if chosen == nil {
		chosen = choose(repaired)
	}
	if chosen == nil {
		return nil, nil, nil
	}
	return chosen.call, chosen.old, chosen.changes
}

func (h *headStart) nearCompatibility(oldRequest llm.Request, oldText string, request llm.Request, text string) ([]spokenWord, []headChange, bool) {
	last := len(request.Messages) - 1
	if len(oldRequest.Messages) != last+1 || last < 0 ||
		!slices.Equal(oldRequest.Messages[:last], request.Messages[:last]) ||
		!h.sameThinking(oldRequest, request) ||
		strings.Replace(oldRequest.Messages[last].Content, oldText, text, 1) != request.Messages[last].Content {
		return nil, nil, false
	} else {
	}
	old := spokenWords(oldText)
	changes, ok := headChanges(old, oldText, text)
	if !ok {
		return nil, nil, false
	} else {
		return old, changes, true
	}
}

// askPart asks for one part's clean-up, with the answer a head start has
// under way when there is one for it. Goroutine-safe.
func (r *run) askPart(ctx context.Context, req llm.Request, text string) (llm.Response, error) {
	h := &r.head
	if c := h.find(req); c != nil {
		source := "recording_websocket_headstart"
		if c.late {
			source = "late_websocket_headstart"
		}
		if c.late && h.failedLate(c) {
			r.logHeadStartAdoption(req, c.requestID, source, "failed_speculative_request", false)
		} else {
			r.logf("debug", "the language model was asked about this part ahead of the final text")
			if !c.late {
				resp, err := c.join(req.OnDelta)
				r.logHeadStartAdoption(req, c.requestID, source, "exact_match", err == nil)
				return resp, err
			}
			resp, err := c.join(nil)
			if err == nil {
				c.mu.Lock()
				deltas := c.held.String()
				c.held.Reset()
				c.mu.Unlock()
				if deltas != "" && req.OnDelta != nil {
					req.OnDelta(deltas)
				}
				r.logHeadStartAdoption(req, c.requestID, source, "exact_match", true)
				return resp, nil
			}
			r.logHeadStartAdoption(req, c.requestID, source, "failed_speculative_request", false)
			if ctx.Err() != nil || r.ctx.Err() != nil || !r.cleanupMayContinue() {
				return resp, err
			}
		}
	}
	if c, old, changes := h.near(req, text); c != nil {
		if resp, err, ok, adopted := h.reuse(c, old, changes, req, text); ok {
			outcome := "fresh_request_fallback"
			if adopted {
				outcome = "near_match_adoption"
				if len(changes) > 0 {
					outcome = "near_match_repair_adoption"
				}
			}
			r.logHeadStartAdoption(req, c.requestID, "websocket_headstart", outcome, adopted)
			return resp, err
		}
		r.logHeadStartAdoption(req, c.requestID, "websocket_headstart", "near_match_repair_failed", false)
	}
	resp, err := r.llmChatContext(ctx, req)
	if h.enabled.Load() {
		r.m.noteRate(h.service, resp, err)
	}
	return resp, err
}

func (r *run) logHeadStartAdoption(req llm.Request, requestID uint64, source, outcome string, adopted bool) {
	if !r.shadowTraceEnabled {
		return
	}
	r.logShadowEvent(map[string]any{
		"event": "voxt_headstart_adoption", "run_id": r.id,
		"session_id": r.shadowSessionID, "request_id": requestID, "source": source,
		"outcome": outcome, "adopted": adopted,
		"request_plan_sha256": r.head.partPlanHash(headPart{req: req}),
	})
}

// reuse is c's answer put right for text, and false when text must be
// asked about. An answer that cannot be put right is found out only once
// it is there: with changes to place, text is asked about meanwhile, and
// the first answer counts.
func (h *headStart) reuse(c *headCall, old []spokenWord, changes []headChange, req llm.Request, text string) (llm.Response, error, bool, bool) {
	right := func() (llm.Response, bool) {
		if c.err != nil {
			return llm.Response{}, false
		}
		resp, ok := c.resp, false
		if resp.Text, ok = putRight(resp.Text, old, changes); ok {
			h.r.logf("debug", "the language model was asked about this part ahead of the final text, %d of its words put right", len(changes))
		}
		return resp, ok
	}
	select {
	case <-c.done:
		resp, ok := right()
		return resp, nil, ok, ok
	default:
	}
	if len(changes) == 0 {
		<-c.done
		resp, ok := right()
		return resp, nil, ok, ok
	}
	fresh := h.start(req, text, false)
	defer fresh.cancel()
	select {
	case <-c.done:
		if resp, ok := right(); ok {
			return resp, nil, true, true
		}
	case <-fresh.done:
	}
	resp, err := fresh.join(req.OnDelta)
	return resp, err, true, false
}

func (h *headStart) callError(c *headCall) (bool, error) {
	select {
	case <-c.done:
		return true, c.err
	default:
		return false, nil
	}
}

func (h *headStart) failedLate(c *headCall) bool {
	done, err := h.callError(c)
	return c.late && done && err != nil
}

// join streams the answer so far and the rest into onDelta, and returns
// the call's end.
func (c *headCall) join(onDelta func(string)) (llm.Response, error) {
	c.mu.Lock()
	if held := c.held.String(); held != "" && onDelta != nil {
		onDelta(held)
	}
	c.onDelta = onDelta
	c.mu.Unlock()
	<-c.done
	return c.resp, c.err
}

func (c *headCall) add(delta string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.onDelta != nil {
		c.onDelta(delta)
		return
	}
	c.held.WriteString(delta)
}

// partRequest is the request for a part of text's clean-up.
func (r *run) partRequest(text string, p compose.Part, think *llm.Thinking, onDelta func(string)) llm.Request {
	req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, Thinking: think, InputChars: utf8.RuneCountInString(text), OnDelta: onDelta}
	return r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.llmFloor(r.m.cfg.LLMTimeout))
}

// sameRequest says whether a and b ask the service the same: the same
// messages, with the same thinking.
func (h *headStart) sameRequest(a, b llm.Request) bool {
	return slices.Equal(a.Messages, b.Messages) && h.sameThinking(a, b)
}

// sameThinking says whether the service's setting gives a and b the same
// thinking.
func (h *headStart) sameThinking(a, b llm.Request) bool {
	return a.Thinking == nil && b.Thinking == nil && (a.InputChars < h.under) == (b.InputChars < h.under)
}

// thinkOffUnder is the length under which a service's setting turns its
// thinking off (0: never).
func (r *run) thinkOffUnder(service string) int {
	for _, provider := range r.settings.Providers.LLM {
		if provider.ID == service {
			return provider.Generation.ThinkingOffUnder
		}
	}
	return 0
}
