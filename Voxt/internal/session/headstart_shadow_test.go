package session

import (
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
