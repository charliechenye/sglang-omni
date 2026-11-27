package session

import (
	"sync"
	"testing"
	"time"
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
	select {
	case <-h.shadowObserved:
	case <-time.After(5 * time.Second):
		t.Fatal("the shadow candidate was not observed")
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
