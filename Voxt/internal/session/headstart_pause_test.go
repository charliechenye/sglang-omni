package session

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestHeadStartPauseConfiguration(t *testing.T) {
	if got := headStartPauseMilliseconds("", false); got != headHangover {
		t.Fatalf("unset configuration selected %d ms, want %d", got, headHangover)
	}
	for _, tc := range []struct {
		value string
		want  int
	}{
		{value: "100", want: 100},
		{value: "150", want: 150},
		{value: "300", want: 300},
		{value: "500", want: 500},
		{value: " 300 ", want: 300},
		{value: "", want: headHangover},
		{value: "0", want: headHangover},
		{value: "fast", want: headHangover},
		{value: "-100", want: headHangover},
		{value: "301", want: headHangover},
		{value: "60000", want: headHangover},
	} {
		if got := headStartPauseMilliseconds(tc.value, true); got != tc.want {
			t.Errorf("configuration %q selected %d ms, want %d", tc.value, got, tc.want)
		}
	}
}

func TestHeadStartPauseThresholdsUseDetectorFrames(t *testing.T) {
	for _, tc := range []struct {
		milliseconds int
		frames       int
	}{
		{milliseconds: 100, frames: 5},
		{milliseconds: 150, frames: 8},
		{milliseconds: 300, frames: 15},
		{milliseconds: 500, frames: 25},
	} {
		t.Run(fmt.Sprintf("%d ms", tc.milliseconds), func(t *testing.T) {
			r := &run{silent: make(chan bool, 1), headStartPauseMilliseconds: tc.milliseconds}
			watch := r.watchPauses()
			speechFrame := tone(asr.BytesPerSecond/50, 220, 0.14)
			quietFrame := tone(asr.BytesPerSecond/50, 220, 0.0005)
			for range 3 {
				watch.hear(speechFrame)
			}
			if !watch.detector.Speaking() || watch.quiet.Load() {
				t.Fatal("detector did not enter speech")
			}
			<-r.silent
			for range tc.frames - 1 {
				watch.hear(quietFrame)
				if watch.quiet.Load() {
					t.Fatalf("pause triggered before %d detector frames", tc.frames)
				}
			}
			watch.hear(quietFrame)
			if !watch.quiet.Load() {
				t.Fatalf("pause did not trigger after %d detector frames", tc.frames)
			}
			if got := <-r.silent; !got {
				t.Fatal("pause notification was not quiet")
			}
		})
	}
}

func TestHeadStartSpeechResumingBeforePauseThresholdDoesNotTrigger(t *testing.T) {
	r := &run{silent: make(chan bool, 1), headStartPauseMilliseconds: headHangover}
	watch := r.watchPauses()
	speechFrame := tone(asr.BytesPerSecond/50, 220, 0.14)
	quietFrame := tone(asr.BytesPerSecond/50, 220, 0.0005)
	for range 3 {
		watch.hear(speechFrame)
	}
	<-r.silent
	for range 4 {
		watch.hear(quietFrame)
	}
	for range 3 {
		watch.hear(speechFrame)
	}
	if watch.quiet.Load() || !watch.detector.Speaking() {
		t.Fatal("speech resumption did not cancel the pending pause")
	}
	select {
	case got := <-r.silent:
		t.Fatalf("unexpected quiet notification: %v", got)
	default:
	}
}

func TestHeadStartPauseWatchFindsPauseInBackloggedPCM(t *testing.T) {
	r := &run{silent: make(chan bool, 1), headStartPauseMilliseconds: 100}
	watch := r.watchPauses()
	pcm := append(bytes.Repeat(frame, 10), bytes.Repeat(quietFrame, 1)...)
	watch.hear(pcm)
	if !watch.quiet.Load() {
		t.Fatal("the pause was missed when speech and silence arrived together")
	}
	if got := <-r.silent; !got {
		t.Fatal("the backlogged pause notification was not quiet")
	}
}

func TestHeadStartPauseWatchCoalescesBackloggedResumption(t *testing.T) {
	r := &run{silent: make(chan bool, 1), headStartPauseMilliseconds: 100}
	watch := r.watchPauses()
	pcm := append(bytes.Repeat(frame, 10), bytes.Repeat(quietFrame, 1)...)
	pcm = append(pcm, frame...)
	watch.hear(pcm)
	if watch.quiet.Load() {
		t.Fatal("the pause stayed quiet after speech resumed in the same backlog")
	}
	if got := <-r.silent; got {
		t.Fatal("a stale quiet notification escaped the backlogged resumption")
	}
	select {
	case got := <-r.silent:
		t.Fatalf("unexpected duplicate notification: %v", got)
	default:
	}
}

func TestHeadStartExperimentalPauseStartsWhileRecording(t *testing.T) {
	t.Setenv(headStartPauseEnvironment, "100")
	const live = "we ship the new build on friday"
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	h := newHeadSession(t, &fakeStreamer{partials: []string{live}, final: live}, model, nil)
	h.pause()
	r := sessionRun(t, h.harness)
	if r.headStartPauseMilliseconds != 100 {
		t.Fatalf("session captured %d ms, want 100", r.headStartPauseMilliseconds)
	}
	t.Setenv(headStartPauseEnvironment, "300")
	if r.headStartPauseMilliseconds != 100 {
		t.Fatal("session pause configuration changed after the session started")
	}
	call := waitLiveCall(t, model, "the configured pause")
	states := h.rec.all()
	if !r.pauseWatch.quiet.Load() || states[len(states)-1].State != Recording {
		t.Fatal("the configured pause request did not start during recording")
	}
	h.m.Stop(h.id)
	call.complete(t, "We ship the new build on Friday.", nil)
	if end := h.rec.wait(t); end.Result != "We ship the new build on Friday." || end.Problem != nil {
		t.Fatalf("session ended with result %q and problem %+v", end.Result, end.Problem)
	}
}

func TestHeadStartStopAsksBeforePauseThreshold(t *testing.T) {
	t.Setenv(headStartPauseEnvironment, "500")
	const live = "we ship the new build on friday"
	model := newControlledLiveLLM(false)
	t.Cleanup(model.close)
	streamer := &fakeStreamer{partials: []string{live}, final: live}
	h := newHeadSession(t, streamer, model, nil)
	h.audio.file = bytes.Repeat(frame, 10)
	h.audio.fileTick = 0
	h.audio.drained = make(chan struct{})
	h.id = h.start()
	select {
	case <-h.audio.drained:
	case <-time.After(5 * time.Second):
		t.Fatal("recording did not deliver the speech frames")
	}
	h.rec.waitFor(t, "the live partial", func(snapshot Snapshot) bool { return snapshot.Partial == live })
	r := sessionRun(t, h.harness)
	if r.pauseWatch == nil || r.pauseWatch.quiet.Load() {
		t.Fatal("the pause threshold elapsed before Stop")
	}
	h.m.Stop(h.id)
	call := waitLiveCall(t, model, "the Stop request")
	call.complete(t, "We ship the new build on Friday.", nil)
	if end := h.rec.wait(t); end.Result != "We ship the new build on Friday." || end.Problem != nil {
		t.Fatalf("session ended with result %q and problem %+v", end.Result, end.Problem)
	}
}

func TestHeadStartLatePartialDoesNotBenefitFromShorterPauseArms(t *testing.T) {
	for _, milliseconds := range []int{headHangover, 300, 150, 100} {
		t.Run(fmt.Sprintf("%d ms", milliseconds), func(t *testing.T) {
			t.Setenv(headStartPauseEnvironment, fmt.Sprint(milliseconds))
			const live = "we ship the new build on friday"
			streamer := &fakeStreamer{final: live, silent: true}
			streamer.ensureSignals()
			model := newControlledLiveLLM(false)
			t.Cleanup(model.close)
			h := newHeadSession(t, streamer, model, nil)
			h.audio.tick = time.Millisecond
			h.audio.speaks = func(frameIndex int) bool { return frameIndex < 10 }
			h.id = h.start()
			select {
			case <-streamer.firstSend:
			case <-time.After(5 * time.Second):
				t.Fatal("ASR stream did not receive audio")
			}
			r := sessionRun(t, h.harness)
			waitPauseState(t, r, true)
			if n := model.requestCount(); n != 0 {
				t.Fatalf("%d requests started before any ASR partial was available", n)
			}
			streamer.mu.Lock()
			stream := streamer.streams[0]
			streamer.mu.Unlock()
			stream.emit(asr.Event{Kind: asr.EventPartial, Text: live})
			h.rec.waitFor(t, "the delayed ASR partial", func(snapshot Snapshot) bool { return snapshot.Partial == live })
			call := waitLiveCall(t, model, "the request after the delayed partial")
			h.m.Stop(h.id)
			call.complete(t, "We ship the new build on Friday.", nil)
			if end := h.rec.wait(t); end.Result != "We ship the new build on Friday." || end.Problem != nil {
				t.Fatalf("session ended with result %q and problem %+v", end.Result, end.Problem)
			}
			t.Logf("pause_ms=%d request_before_partial=0 request_after_partial=%d", milliseconds, model.requestCount())
		})
	}
}
