package main

import (
	"context"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type benchLLM struct {
	client *llm.Client
	run    *benchRun
}

func (l *benchLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	run := l.run
	call := benchCall{StartAt: time.Now(), RequestID: llm.TraceRequestID(ctx), InputChars: request.InputChars, Kind: llm.TraceTag(ctx)}
	if call.Kind == "" {
		call.Kind = "cleanup"
	}
	index := -1
	if run != nil {
		run.mu.Lock()
		run.LLMCalls = append(run.LLMCalls, call)
		index = len(run.LLMCalls) - 1
		run.mu.Unlock()
	}
	originalDelta := request.OnDelta
	request.OnDelta = func(delta string) {
		if delta != "" && run != nil {
			run.mu.Lock()
			recordFirstToken(&run.LLMCalls[index], time.Now())
			run.mu.Unlock()
		}
		if originalDelta != nil {
			originalDelta(delta)
		}
	}
	if originalDelta == nil {
		request.OnDelta = nil
	}
	response, err := l.client.Chat(ctx, request)
	if run != nil {
		run.mu.Lock()
		run.LLMCalls[index].DoneAt = time.Now()
		run.LLMCalls[index].Usage = response.Usage
		run.LLMCalls[index].Succeeded = err == nil
		run.mu.Unlock()
	}
	return response, err
}

func recordFirstToken(call *benchCall, at time.Time) {
	if call.FirstTokenAt.IsZero() {
		call.FirstTokenAt = at
	}
}

type benchAudio struct {
	mu          sync.Mutex
	pcm         []byte
	frameFactor float64
	ready       chan struct{}
	stop        chan struct{}
	stopOnce    sync.Once
}

func (a *benchAudio) prepare(pcm []byte) <-chan struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pcm = pcm
	a.ready = make(chan struct{})
	a.stop = make(chan struct{})
	a.stopOnce = sync.Once{}
	return a.ready
}

func (a *benchAudio) Record(ctx context.Context, id string, params proto.CaptureParams, fn func(session.AudioEvent)) (proto.CaptureResult, error) {
	a.mu.Lock()
	pcm, factor, ready, stop := a.pcm, a.frameFactor, a.ready, a.stop
	a.mu.Unlock()
	frameMS := params.FrameMS
	if frameMS <= 0 {
		frameMS = 100
	}
	frameBytes := max(2, 16000*frameMS/1000*2)
	fn(session.AudioEvent{Started: &proto.CaptureStartedParams{Job: id, SampleRate: 16000, DeviceName: "benchmark fixture"}})
	frames, samples := 0, 0
	for offset := 0; offset < len(pcm); {
		end := min(offset+frameBytes, len(pcm))
		if end%2 != 0 {
			end--
		}
		if end <= offset {
			break
		}
		frame := append([]byte(nil), pcm[offset:end]...)
		fn(session.AudioEvent{PCM: &proto.PCMParams{Job: id, Seq: frames, Offset: samples, Data: frame}})
		frames++
		samples += len(frame) / 2
		offset = end
		if factor > 0 {
			timer := time.NewTimer(time.Duration(float64(time.Duration(frameMS)*time.Millisecond) * factor))
			select {
			case <-ctx.Done():
				timer.Stop()
				return proto.CaptureResult{}, ctx.Err()
			case <-stop:
				timer.Stop()
				return proto.CaptureResult{Reason: proto.CaptureStopped, Frames: frames, Samples: samples}, nil
			case <-timer.C:
			}
		}
	}
	close(ready)
	select {
	case <-ctx.Done():
		return proto.CaptureResult{}, ctx.Err()
	case <-stop:
		return proto.CaptureResult{Reason: proto.CaptureStopped, Frames: frames, Samples: samples}, nil
	}
}

func (a *benchAudio) Stop(context.Context, string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.stop != nil {
		a.stopOnce.Do(func() { close(a.stop) })
	}
	return nil
}

type benchTextIO struct {
	active func() *benchRun
}

func (benchTextIO) Focus(context.Context) (proto.FocusResult, error) {
	return proto.FocusResult{}, nil
}

func (io benchTextIO) Paste(context.Context, proto.PasteParams) (proto.PasteResult, error) {
	if io.active != nil {
		if run := io.active(); run != nil {
			run.mu.Lock()
			if run.DeliveryRequestedAt.IsZero() {
				run.DeliveryRequestedAt = time.Now()
			}
			if run.DeliveryCompletedAt.IsZero() {
				run.DeliveryCompletedAt = time.Now()
			}
			run.mu.Unlock()
		}
	}
	return proto.PasteResult{Posted: true}, nil
}
