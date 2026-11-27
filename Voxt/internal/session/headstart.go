package session

import (
	"context"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	// headKeep is how many texts are asked about at a time: the live text's
	// end often changes once more, and the answer to the text before it may
	// be the Final's.
	headKeep = 3
	// headAsks is the most texts a recording is asked about before Stop.
	headAsks = 20
)

// headStart asks the language model to clean a dictation up before its
// Final is there, from the text known so far (guess): when the user
// pauses, and at Stop. The Final is often that text, and its requests then
// are the ones asked already: they take the answers under way (askPart)
// instead of asking, also when the Final says the same but for punctuation
// or a few words (headmatch.go). The requests the Final does not need are
// given up.
//
// What is asked is the whole text, as after Stop: a stretch cleaned alone
// does not know what is said later, and cannot take back what it wrote.
//
// The live clean-up (livepolish.go) is left out where this runs: it cleans
// the sentences one at a time, and its answers count only when the Final
// repeats the live text word for word.
//
// A server with one queue would answer these before the Final's own, so
// this is a remote service's (Model.Remote).
type headStart struct {
	r       *run
	enabled atomic.Bool
	service string // the language model's id
	under   int    // its ThinkingOffUnder

	// The session goroutine's.
	text      string // the text asked about last
	asks      int    // texts asked about while the user speaks
	lateAsked bool

	mu           sync.Mutex
	final        string        // what the speech model finalised so far (asr.EventPiece)
	gens         [][]*headCall // the requests of the texts asked about last, headKeep at most, the newest last
	requestCalls []*headCall
}

type headPart struct {
	partIndex int
	req       llm.Request
	text      string
	reuse     *headCall
}

type headPlan struct {
	heard string
	rest  string
	parts []headPart
}

type lateSubmission struct {
	requestIDs      []uint64
	dispatchReadyAt time.Time
	dispatchedAt    time.Time
	dispatched      bool
}

func (p headPlan) hasFresh() bool {
	for _, part := range p.parts {
		if part.reuse == nil {
			return true
		}
	}
	return false
}

// headCall is a request asked early. resp and err are there once done is
// closed; the answer so far goes to onDelta, or is kept until a request
// takes the call.
type headCall struct {
	req                llm.Request
	requestID          uint64
	text               string // the part's
	late               bool
	origin             string
	cancel             context.CancelFunc
	done               chan struct{}
	requestStartedAt   time.Time
	requestStartLogged bool
	resp               llm.Response
	err                error

	mu      sync.Mutex
	held    strings.Builder
	onDelta func(string)
}

// prepareHeadInputs allocates partial and pause channels before ASR workers
// start when Head Start could become eligible later.
func (r *run) prepareHeadInputs(endWord bool) bool {
	r.head.r = r
	mode := os.Getenv("VOXT_EXPERIMENT_FINAL_CANDIDATES")
	r.shadowTraceEnabled = mode == "shadow" || mode == "shadow_dense"
	r.shadowDense = mode == "shadow_dense"
	if r.shadowTraceEnabled && r.shadowSessionID == 0 {
		r.shadowSessionID = shadowSessionIDs.Add(1)
	}
	possible := r.req.Kind == store.KindTranscribe && !r.settings.Features.Transcribe.CleanAfterStop && r.shot == nil && len(r.req.Images) == 0
	if possible {
		if r.init.split && !r.init.cleanupDecisionReady {
			possible = r.settings.Features.Transcribe.Enhance || r.sc.cleanupPending
		} else {
			possible = r.transcribeCleanup()
		}
	}
	if possible && (!r.init.split || r.init.llmReady) {
		possible = r.svc.LLM != nil && r.svc.LLMInfo.Remote && !r.m.resting(r.svc.LLMInfo.ID)
	}
	if (endWord || possible) && r.heard == nil {
		r.heard, r.silent = make(chan heardText, 1), make(chan bool, 1)
	}
	if r.shadowCandidates == nil {
		r.shadowCandidates = make(chan shadowCandidate, 1)
	}
	return possible
}

// startHead decides eligibility only after cleanup readiness is known.
// The atomic store publishes the immutable service and Thinking policy to
// stream and cleanup workers.
func (r *run) startHead() {
	if r.head.enabled.Load() {
		return
	}
	if r.head.r == nil {
		r.head.r = r
	}
	service := r.svc.LLMInfo.ID
	if r.req.Kind != store.KindTranscribe || r.settings.Features.Transcribe.CleanAfterStop || r.shot != nil || len(r.req.Images) > 0 ||
		!r.svc.LLMInfo.Remote || r.svc.LLM == nil || !r.transcribeCleanup() || r.m.resting(service) {
		return
	}
	r.head.service = service
	r.head.under = r.thinkOffUnder(service)
	r.head.enabled.Store(true)
}

func (h *headStart) canAsk() bool {
	r := h.r
	if !h.enabled.Load() || r == nil || r.ctx.Err() != nil || r.cancelled.Load() || r.stopped.Load() || r.streamReaderOff.Load() {
		return false
	}
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	return r.cleanupDecisionOpenLocked() && !r.skipped.Load()
}

// pause asks about the live text while the user is silent.
func (h *headStart) pause(live string) {
	if h.asks < headAsks && h.ask(live) {
		h.asks++
	}
}

// hearFinal takes the final text so far of a recording that finalises
// stretches while it goes on: the head start's text begins with it.
func (h *headStart) hearFinal(text string) {
	h.mu.Lock()
	h.final = strings.TrimSpace(text)
	h.mu.Unlock()
}

// guess is the final text as far as it is known: what the speech model
// finalised, which the Final repeats, and the live text after it. False
// when the live text does not show where that ends.
func (h *headStart) guess(live string) (string, bool) {
	h.mu.Lock()
	final := h.final
	h.mu.Unlock()
	live = strings.TrimSpace(live)
	if final == "" {
		return live, live != ""
	}
	tail, ok := tailAfter(live, final)
	if !ok {
		return "", false
	}
	return joinSpeech(final, tail), true
}

// ask sends the requests of the text known so far, unless it has. It
// reports whether it asked anything.
func (h *headStart) ask(live string) bool {
	if !h.canAsk() {
		return false
	}
	heard, cleaned, ok := h.normalize(live)
	if !ok {
		return false
	}
	if heard == h.text {
		return false
	}
	h.text = heard
	return h.send(cleaned)
}

func (h *headStart) normalize(live string) (heard, cleaned string, ok bool) {
	r := h.r
	heard, ok = h.guess(live)
	if !ok {
		return "", "", false
	}
	heard, _ = withoutEndWord(heard, r.endWord())
	if heard == "" {
		return "", "", false
	}
	cleaned = heard
	if r.settings.Dictionary.Enabled {
		cleaned = dict.Replace(heard, r.terms)
	}
	return heard, cleaned, true
}

func (h *headStart) prepare(cleaned string) (headPlan, bool, bool) {
	r := h.r
	plan := headPlan{}
	rest, ok := r.earlyCleanup().rest(cleaned)
	if _, whole := snippet.Whole(cleaned, r.snippets); whole || !ok || strings.TrimSpace(rest) == "" {
		return plan, false, true
	}
	parts, err := compose.BuildParts(r.transcribeInput(rest))
	if err != nil {
		return plan, false, false
	}
	plan.rest = rest
	for partIndex, part := range parts[:min(len(parts), partsTogether)] {
		req := r.partRequest(cleaned, part, nil, nil)
		plan.parts = append(plan.parts, headPart{partIndex: partIndex, req: req, text: part.Text, reuse: h.find(req)})
	}
	return plan, true, false
}

func (h *headStart) prepareLate(live string) (headPlan, bool) {
	plan, ok, _, _, _ := h.prepareLateObserved(live)
	return plan, ok
}

type lateGateState struct {
	BudgetAvailable       bool
	LLMReady              bool
	CleanupEligible       bool
	RecordingGenerations  int
	LateAlreadyDispatched bool
}

func (s lateGateState) eligible() bool {
	return s.BudgetAvailable && s.LLMReady && s.CleanupEligible
}

func (h *headStart) prepareLateObserved(live string) (headPlan, bool, string, lateGateState, lateGateState) {
	before := h.lateGateState()
	if !before.LLMReady || !before.CleanupEligible ||
		(!before.BudgetAvailable && !h.r.shadowTraceEnabled) {
		return headPlan{}, false, lateGateReason(before), before, before
	}
	plan, reason := h.latePlan(live)
	if reason == "" && !plan.hasFresh() {
		reason = "no_fresh_request"
	}
	if reason != "" {
		return headPlan{}, false, reason, before, h.lateGateState()
	}
	after := h.lateGateState()
	if !after.BudgetAvailable {
		return plan, false, lateGateReason(after), before, after
	}
	if !after.eligible() {
		return plan, false, lateGateReason(after), before, after
	}
	return plan, true, "", before, after
}

func (h *headStart) lateAllowed() bool {
	budgetAvailable, llmReady, cleanupEligible := h.lateGates()
	return budgetAvailable && llmReady && cleanupEligible
}

func (h *headStart) lateGates() (bool, bool, bool) {
	state := h.lateGateState()
	return state.BudgetAvailable, state.LLMReady, state.CleanupEligible
}

func (h *headStart) lateGateState() lateGateState {
	state := lateGateState{}
	h.mu.Lock()
	state.LateAlreadyDispatched = h.lateAsked
	state.RecordingGenerations = len(h.gens)
	state.BudgetAvailable = !h.lateAsked && len(h.gens) < headKeep
	h.mu.Unlock()
	ctx := h.r.currentLLMContext()
	state.LLMReady = h.enabled.Load() && ctx != nil && ctx.Err() == nil &&
		!h.r.m.resting(h.service)
	state.CleanupEligible = h.r.ctx.Err() == nil && h.r.cleanupMayContinue()
	return state
}

func lateGateReason(state lateGateState) string {
	switch {
	case !state.BudgetAvailable:
		return "late_generation_budget_unavailable"
	case !state.LLMReady:
		return "llm_not_ready"
	case !state.CleanupEligible:
		return "cleanup_ineligible"
	default:
		return ""
	}
}
