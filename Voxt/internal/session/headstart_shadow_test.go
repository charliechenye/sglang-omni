package session

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

func TestShadowCandidateDoesNotDispatchOrChangeVisibleText(t *testing.T) {
	const transcript = "we ship the new build on friday"
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	finishGate := make(chan struct{})
	var finishOnce sync.Once
	releaseFinish := func() { finishOnce.Do(func() { close(finishGate) }) }
	t.Cleanup(releaseFinish)
	streamer := &fakeStreamer{
		final: transcript, silent: true, shadowCandidates: true,
		shadowTexts: []string{"unrelated speculative transcript"},
		finishGate:  finishGate, finishStarted: make(chan struct{}, 1),
	}
	h := newHeadSession(t, streamer, model, nil)
	id := h.start()
	h.speak(id, 10)
	select {
	case <-streamer.finishStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the ASR Final did not start")
	}
	if model.requestCount() != 0 {
		t.Fatalf("shadow candidate started %d speculative requests", model.requestCount())
	}
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.Partial != "" {
			t.Fatalf("shadow candidate changed visible partial text to %q", snapshot.Partial)
		}
	}
	releaseFinish()
	call := waitLiveCall(t, model, "the ordinary Final cleanup")
	call.complete(t, "Cleaned final transcript", nil)
	if end := h.rec.wait(t); end.Result != "Cleaned final transcript" || end.Problem != nil {
		t.Fatalf("end %+v", end)
	}
	if model.requestCount() != 1 {
		t.Fatalf("made %d LLM requests, want only the normal Final cleanup", model.requestCount())
	}
}

func TestShadowPlanCompatibilityUsesPendingCleanupPartIndices(t *testing.T) {
	request := func(content string) llm.Request {
		return llm.Request{Messages: []llm.Message{{Role: llm.User, Content: content}}}
	}
	candidate := headPlan{parts: []headPart{
		{partIndex: 0, req: request("already completed")},
		{partIndex: 1, req: request("pending match")},
		{partIndex: 2, req: request("candidate mismatch")},
	}}
	authoritativePending := headPlan{parts: []headPart{
		{partIndex: 1, req: request("pending match")},
		{partIndex: 2, req: request("different pending request")},
	}}
	if exact, near, repair, compatible := (&headStart{}).plansCompatible(candidate, authoritativePending); exact || near || repair || len(compatible) != 1 || compatible[0] != 1 {
		t.Fatalf("exact=%t near=%t repair=%t compatible parts=%v", exact, near, repair, compatible)
	}
}

func TestFinalCandidateExperimentModesKeepProductionOffUninstrumented(t *testing.T) {
	for _, mode := range []string{"", "instrumented", "shadow", "shadow_dense"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", mode)
			var logs []string
			manager := &Manager{cfg: Config{Log: func(_, message string) { logs = append(logs, message) }}}
			r := &run{m: manager, id: "mode-test"}
			r.head.r = r
			r.configureFinalCandidateExperiment()
			r.prepareHeadInputs(false)
			plan := headPlan{parts: []headPart{{partIndex: 0, req: llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "content"}}}}}}
			r.captureCleanupPlan(plan, "authoritative_cleanup")

			switch mode {
			case "":
				if r.shadowTraceEnabled || r.shadowCandidatesEnabled || r.shadowCandidates != nil || len(r.cleanupPlans) != 0 || len(logs) != 0 {
					t.Fatalf("production OFF allocated or emitted experiment state: trace=%t candidates=%t channel=%t plans=%d logs=%d", r.shadowTraceEnabled, r.shadowCandidatesEnabled, r.shadowCandidates != nil, len(r.cleanupPlans), len(logs))
				}
			case "instrumented":
				if !r.shadowTraceEnabled || r.shadowCandidatesEnabled || r.shadowCandidates != nil || len(r.cleanupPlans) != 1 || len(logs) != 2 {
					t.Fatalf("instrumented baseline must capture traces without candidate transport: trace=%t candidates=%t channel=%t plans=%d logs=%d", r.shadowTraceEnabled, r.shadowCandidatesEnabled, r.shadowCandidates != nil, len(r.cleanupPlans), len(logs))
				}
			default:
				if !r.shadowTraceEnabled || !r.shadowCandidatesEnabled || r.shadowCandidates == nil || len(r.cleanupPlans) != 1 || len(logs) != 2 {
					t.Fatalf("shadow mode must capture traces and allocate its candidate channel: trace=%t candidates=%t channel=%t plans=%d logs=%d", r.shadowTraceEnabled, r.shadowCandidatesEnabled, r.shadowCandidates != nil, len(r.cleanupPlans), len(logs))
				}
			}
		})
	}
}

func TestTerminalShadowReportIsExactlyOnceOnCancellation(t *testing.T) {
	var logs []string
	manager := &Manager{cfg: Config{Log: func(_, message string) { logs = append(logs, message) }}}
	r := &run{
		m: manager, id: "cancel-test", shadowTraceEnabled: true,
		shadowCandidatesEnabled: true, shadowSessionID: 73,
		shadowFinal: &streamEnd{err: errors.New("cancelled before Final")},
	}
	snapshot := Snapshot{State: Cancelled}
	r.finalizeShadowReport(snapshot)
	r.finalizeShadowReport(snapshot)
	count := 0
	for _, message := range logs {
		if strings.Contains(message, `"event":"voxt_final_candidate_summary"`) {
			count++
			if !strings.Contains(message, `"terminal_status":"cancelled"`) {
				t.Fatalf("terminal summary lost cancellation status: %s", message)
			}
		}
	}
	if count != 1 {
		t.Fatalf("got %d terminal summaries, want exactly one: %v", count, logs)
	}
}

func TestTerminalShadowStatusDistinguishesFailureAndEmptyFinal(t *testing.T) {
	r := &run{}
	cases := []struct {
		name     string
		snapshot Snapshot
		end      streamEnd
		want     string
	}{
		{name: "no speech", snapshot: Snapshot{State: Done, Problem: &Problem{Code: ProblemNoSpeech}}, want: "no_speech"},
		{name: "empty final", snapshot: Snapshot{State: Done}, end: streamEnd{finalAvailableAt: time.Now()}, want: "empty_transcript"},
		{name: "ASR error", snapshot: Snapshot{State: Failed, Problem: &Problem{Code: ProblemASR}}, end: streamEnd{err: errors.New("ASR failed")}, want: "asr_failure"},
		{name: "unsupported path", snapshot: Snapshot{State: Recording}, end: streamEnd{err: errors.New("unsupported")}, want: "unsupported_path"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if got := terminalShadowStatus(r, test.snapshot, test.end, false); got != test.want {
				t.Fatalf("terminalShadowStatus() = %q, want %q", got, test.want)
			}
		})
	}
}
