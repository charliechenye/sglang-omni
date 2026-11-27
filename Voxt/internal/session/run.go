package session

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

var shadowSessionIDs atomic.Uint64

// run is one session. Its fields are the session goroutine's, but for
// ended (Manager.mu) and what is atomic or a channel.
type run struct {
	m        *Manager
	id       string
	req      StartRequest
	settings settings.Settings

	ctx       context.Context // the session's: cancelled by Cancel and at the end
	cancel    context.CancelFunc
	cancelled atomic.Bool
	// llmCtx is the language model's requests': SkipLLM, StopLLM and
	// Rethink cancel it (llmStop, under m.mu: the run replaces both to ask
	// again), and skipped, stopped or rethink says which; the session goes
	// on.
	llmCtx      context.Context
	llmStop     context.CancelFunc
	skipped     atomic.Bool
	stopped     atomic.Bool
	rethink     atomic.Bool
	cleanupWake chan struct{}
	// Cleanup arbitration is guarded by m.mu. Workers publish candidates
	// here before waking the session owner; only the owner delivers one.
	cleanupChoice           cleanupChoice
	cleanupOutcome          cleanupOutcome
	cleanupDeadline         time.Time
	cleanupDeadlineNotified bool
	cleanupGeneration       uint64
	cleanupAttempt          uint64
	cleanupWorkerRunning    bool
	cleanupCandidate        *cleanupOutput
	cleanupOutput           cleanupOutput
	// think is the level the requests ask with: StartRequest.Thinking,
	// then Rethink's (guarded by m.mu).
	think           *llm.Thinking
	stop            chan struct{}
	stopAt          time.Time // captured by Stop before recognition or clean-up can begin
	stopMonotonicAt time.Time
	stopAtMu        sync.Mutex
	// recorded is closed once listen's recording loop has the capture's
	// end: the stream's Finish waits for it, so that the Final's answer
	// never comes while that loop still runs.
	recorded chan struct{}
	stopOnce sync.Once
	pcm      *pcmBuffer
	ended    bool // guarded by m.mu
	// historyRetry owns a private snapshot instead of a Manager session.
	// onDraft optionally receives its streaming answer.
	historyRetry bool
	onDraft      func(string)
	own          Snapshot

	svc         Services
	terms       []store.Term
	snippets    []snippet.Snippet
	speechHints []string
	init        runInitialization
	// svcDone gets the compatibility resolver result. On split Transcribe
	// sessions, resolved means the speech lane is ready; cleanup state is
	// published separately by the session owner.
	svcDone  chan svcResult
	resolved bool
	sc       sceneState // app enhancement (scene.go)
	// target is Translate's target language (a code).
	target string
	// asrLimit is recognition's limit after Stop (Config.asrLimit).
	asrLimit time.Duration
	// recognitionFailed: the Final failed or found no text; the history
	// entry keeps the recording for a while (keep), for Retry.
	recognitionFailed bool
	// asrDegraded: the Final failed and the session goes on with the
	// text the stream had (listen); its notice wins over a clean-up's.
	asrDegraded *Problem
	// opening: the speech model is still starting (wait.go); waitDue:
	// Config.WaitNote has passed since the start.
	opening, waitDue atomic.Bool
	// dsp and gainDB are what the sound processing did (noteDSP), for a
	// retry's entry.
	dsp    string
	gainDB float64
	// shot is the picture of the target's window for the language model
	// (window.go); nil: none.
	shot func() []byte
	// clip is the picture on the clipboard for Rewrite's language model
	// (window.go), read before the selection; nil: none.
	clip []byte
	// typed is Rewrite's typed text: StartRequest.Text's, or the input
	// box's (Config.BoxInput) when the answer is asked for.
	typed string
	// heard carries whole-text updates and advisory candidates to the session owner.
	heard                   chan heardText
	shadowCandidates        chan shadowCandidate
	shadowTraceEnabled      bool
	shadowCandidatesEnabled bool
	shadowDense             bool
	shadowSessionID         uint64
	// Shadow analysis is retained in this bounded buffer during Final and
	// emitted only after cleanup and delivery have completed.
	shadowMu                        sync.Mutex
	shadowObservations              []shadowObservation
	shadowHeadStartDecisions        []shadowHeadStartDecision
	shadowHeadStartDecisionsDropped uint64
	shadowObservationCap            bool
	shadowObservationDropped        uint64
	shadowCandidateCoalesced        atomic.Uint64
	shadowFinal                     *streamEnd
	shadowInitialBudget             initialHeadBudget
	shadowInitialBudgetKnown        bool
	shadowReportOnce                sync.Once
	cleanupPlanSequence             atomic.Uint64
	cleanupPlanMu                   sync.Mutex
	cleanupPlans                    []headPlan
	llmRequestSequence              atomic.Uint64
	// svcResolved is closed when the session's services are resolved
	// (useServices): the early clean-up's worker waits for it (and r.ctx).
	svcResolved      chan struct{}
	streamReaderDone chan struct{}
	streamReaderStop context.CancelFunc
	streamReaderOff  atomic.Bool
	stream           asr.Stream // the session owner's current stream
	// early cleans finalized speech stretches before Stop (earlyclean.go).
	earlyMu sync.Mutex
	early   *earlyClean
	// note (luojiaxuan): live cleans the stream's live text up while the user
	// speaks, so Stop leaves only what was said last (livepolish.go); the
	// Once makes it on the first live text or at Stop. liveOff: the stream's
	// own stretches took over (hearPiece). Both are the stream's reader
	// goroutine's until it returns.
	liveOnce sync.Once
	live     *livePolish
	liveOff  bool
	// head asks the language model about the live text ahead of the Final
	// (headstart.go); silent tells the recording loop of the user's pauses.
	head       headStart
	silent     chan bool
	pauseWatch *pauseWatch // written before feed workers start; read by the owner
}

type capResult struct {
	res proto.CaptureResult
	err error
}

type svcResult struct {
	svc      Services
	terms    []store.Term
	snippets []snippet.Snippet
	err      error
}

type openResult struct {
	st  asr.Stream
	err error
}

// streamEnd is a stream's last event: its final result or its error;
// silent when the recording ended with no speech, so nothing was sent
// (the gate held it all) and no result will come.
type streamEnd struct {
	res              asr.Result
	err              error
	silent           bool
	candidate        asr.CandidateMetadata
	finalAvailableAt time.Time
}

func (r *run) logf(level, format string, args ...any) {
	r.m.cfg.Log(level, "session "+r.id+": "+fmt.Sprintf(format, args...))
}

// logText is a text for the log: itself when the settings allow it, else
// its length.
func (r *run) logText(s string) string {
	if r.settings.Privacy.LogText {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprintf("%d chars", len([]rune(s)))
}

func (r *run) now() time.Time { return r.m.cfg.Now() }

func (r *run) recordStopAt(at time.Time) time.Time {
	r.stopAtMu.Lock()
	if r.stopAt.IsZero() {
		r.stopAt = at
		r.stopMonotonicAt = time.Now()
	}
	stoppedAt, monotonicStoppedAt := r.stopAt, r.stopMonotonicAt
	r.stopAtMu.Unlock()
	r.head.logRequestStartsAfterStop(monotonicStoppedAt)
	return stoppedAt
}

func (r *run) stoppedAt() time.Time {
	r.stopAtMu.Lock()
	defer r.stopAtMu.Unlock()
	return r.stopAt
}

func (r *run) stopRecording() {
	r.stopOnce.Do(func() {
		r.recordStopAt(r.now())
		close(r.stop)
	})
}

// firstState is where a session starts: composing with its text,
// reading the selection (Translate), else recording.
func (r *run) firstState() State {
	switch {
	case r.req.Text != nil:
		return Composing
	case r.req.Kind == store.KindTranslate && r.canSelect():
		return Reading
	}
	return Recording
}

// canSelect says whether the session reads the selection: Translate and
// Rewrite with a selector (a Rewrite follow-up too: the user may have
// selected more text since, #3816), not a try, in an app that is not
// VoxtNew.
func (r *run) canSelect() bool {
	cfg := r.m.cfg
	t := r.req.Target
	return cfg.Select != nil && !r.req.Try && t.PID != 0 && (cfg.Self == 0 || t.PID != cfg.Self) &&
		(r.req.Kind == store.KindTranslate || r.req.Kind == store.KindRewrite)
}

// main is the session's goroutine: the selection, recording,
// recognition, composition, delivery, history.
func (r *run) main() {
	defer r.cancel()
	kind := r.req.Kind
	switch kind {
	case store.KindTranscribe, store.KindTranslate, store.KindRewrite:
	default:
		r.fail(&Problem{Code: ProblemUnsupported, Err: fmt.Errorf("unknown feature %q", kind)})
		return
	}
	r.logf("info", "start %s in %s", kind, r.req.Target.BundleID)
	// Rewrite's window picture waits for the clipboard's: the copied
	// picture wins, and a picture that will not be sent is never taken.
	if kind != store.KindRewrite {
		r.shootWindow()
	}

	// The services and the dictionary, meanwhile. A typed text or a
	// selection to translate is not heard: no speech model (#3838).
	selTranslate := kind == store.KindTranslate && r.canSelect()
	r.resolve(r.req.Text == nil && !selTranslate)

	if t := r.req.Text; t != nil {
		r.typed = t.Typed
		if r.awaitServices() {
			r.write(r.spoken(strings.TrimSpace(t.Raw)), t.Selection)
		}
		return
	}
	var selDone chan string
	var listened chan struct{}
	switch {
	case selTranslate:
		sel := r.readSelection(nil)
		if r.ctx.Err() != nil {
			r.endCancelled(nil)
			return
		}
		if sel != "" {
			if r.awaitServices() {
				r.write("", sel)
			}
			return
		}
		// No selection: Translate hears the text, with a speech model.
		r.resolve(true)
		r.m.update(r, func(s *Snapshot) { s.State = Recording })
	case kind == store.KindRewrite && r.canSelect():
		selDone, listened = make(chan string, 1), make(chan struct{})
		go func() {
			r.readClipboardImage() // first: reading the selection may ⌘C over the clipboard
			if !r.req.Box {
				// The box's session shoots when the answer is asked for:
				// pictures pasted into it while it records win, and a
				// picture that will not be sent is never taken.
				r.shootWindow()
			}
			sel := r.readSelection(listened)
			r.previewClip()
			selDone <- sel
		}()
	}

	raw, ok := r.listen()
	if listened != nil {
		close(listened)
	}
	if !ok {
		return
	}
	if r.init.split && !r.awaitTranscribeData() {
		return
	}
	text := r.spoken(raw)
	switch kind {
	case store.KindTranscribe:
		if !r.insertSnippet(text) {
			if r.init.split && !r.awaitCleanupReadiness() {
				if r.ctx.Err() != nil || r.cancelled.Load() {
					return
				}
				r.finish(Done, nil, r.expandSnippets(text, text), r.transcribeOpts())
				return
			}
			r.transcribe(text)
		}
	case store.KindTranslate:
		r.write(text, "")
	case store.KindRewrite:
		sel := ""
		if selDone != nil {
			select {
			case sel = <-selDone:
			case <-r.ctx.Done():
				r.endCancelled(nil)
				return
			}
		}
		r.write(text, sel)
	}
}

// finishOpts say how a result is delivered: card, when set, sends it to
// the answer card without trying to insert it (the reason it shows
// there); replace replaces the selection, only in the app it was made in
// (deliver.Request.Replace).
