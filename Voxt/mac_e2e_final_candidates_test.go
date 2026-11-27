package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const finalCandidateBenchEnv = "VOXTNEW_FINAL_CANDIDATE_BENCH"

// TestMacE2EFinalCandidateBenchmark runs interleaved OFF and SHADOW sessions
// with local Qwen3-ASR and an OpenAI-compatible cleanup endpoint. It never
// prints transcript text or downloads model files.
func TestMacE2EFinalCandidateBenchmark(t *testing.T) {
	if runtime.GOOS != "darwin" || os.Getenv(finalCandidateBenchEnv) != "1" {
		t.Skip("set VOXTNEW_FINAL_CANDIDATE_BENCH=1 on macOS to run real-model validation")
	}
	modelDir := requiredBenchEnv(t, "VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL_DIR")
	endpoint := requiredBenchEnv(t, "VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_ENDPOINT")
	llmModel := requiredBenchEnv(t, "VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_MODEL")
	modelID := strings.TrimSpace(os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_ASR_MODEL"))
	if modelID == "" {
		modelID = models.ASRAccurate
	}
	scenarios, err := readBenchScenarios()
	if err != nil {
		t.Fatal(err)
	}
	rounds := benchInt(t, "VOXTNEW_FINAL_CANDIDATE_BENCH_ROUNDS", 5)
	warmups := benchInt(t, "VOXTNEW_FINAL_CANDIDATE_BENCH_WARMUPS", 1)
	playbackFactor := benchFloat(t, "VOXTNEW_FINAL_CANDIDATE_BENCH_PLAYBACK_FACTOR", 1)
	if playbackFactor < 0 {
		t.Fatal("playback factor cannot be negative")
	}

	binary, err := omni.Locate(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	logs := &benchLog{}
	rt := omni.NewRuntime(omni.Config{Binary: binary, Log: logs.add})
	defer rt.Close()
	baseASR, err := omni.NewASR(rt, modelID, modelDir)
	if err != nil {
		t.Fatal(err)
	}
	if baseASR.Spec().Kind != "qwen3_asr" {
		t.Fatalf("ASR model %q is %q; the candidate benchmark requires qwen3_asr", modelID, baseASR.Spec().Kind)
	}
	llmRemote := os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_REMOTE") != "0"
	backend := llm.NewOpenAI(llm.Preset{
		ID: llm.Custom, BaseURL: endpoint, DefaultModel: llmModel,
		KeyOptional: true, JSONMode: llm.JSONObjectMode,
	}, os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_LLM_KEY"), &http.Client{Timeout: 5 * time.Minute})
	client := &llm.Client{Backend: backend, Timeout: 5 * time.Minute, Retry: llm.RetryPolicy{MaxAttempts: 1}, Meter: &llm.Meter{}}

	var activeMu sync.Mutex
	var activeSettings settings.Settings
	var activeASR *omni.ASR
	var activeTerms []store.Term
	var activeRun *benchRun
	capture := &benchAudio{frameFactor: playbackFactor}
	timedLLM := &benchLLM{client: client, active: func() *benchRun {
		activeMu.Lock()
		defer activeMu.Unlock()
		return activeRun
	}}
	done := make(chan session.Snapshot, 1)
	manager := session.New(session.Config{
		Audio: capture,
		Text:  benchTextIO{},
		ResolveASR: func(context.Context, session.Kind, settings.Settings) session.ASRReadiness {
			activeMu.Lock()
			provider := activeASR
			activeMu.Unlock()
			return session.ASRReadiness{ASR: provider, ASRInfo: session.Model{ID: modelID, Name: modelID}}
		},
		ResolveLLM: func(context.Context, session.Kind, settings.Settings) session.LLMReadiness {
			return session.LLMReadiness{
				LLM: timedLLM, LLMInfo: session.Model{ID: "benchmark-openai-compatible", Name: llmModel, Remote: llmRemote},
				Outcome: session.LLMAvailable,
			}
		},
		Settings: func() settings.Settings {
			activeMu.Lock()
			defer activeMu.Unlock()
			return activeSettings
		},
		Glossary: func(context.Context) ([]store.Term, error) {
			activeMu.Lock()
			defer activeMu.Unlock()
			return append([]store.Term(nil), activeTerms...), nil
		},
		Locale: func() string { return "en" },
		Log:    logs.add,
		OnChange: func(snapshot session.Snapshot) {
			activeMu.Lock()
			if current := activeRun; current != nil {
				current.observe(snapshot)
			}
			activeMu.Unlock()
			if snapshot.State.Ended() {
				select {
				case done <- snapshot:
				default:
				}
			}
		},
		LLMTimeout: 5 * time.Minute,
		LLMStall:   45 * time.Second,
	})
	defer manager.Close()

	originalMode, modeWasSet := os.LookupEnv("VOXT_EXPERIMENT_FINAL_CANDIDATES")
	defer func() {
		if modeWasSet {
			_ = os.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", originalMode)
		} else {
			_ = os.Unsetenv("VOXT_EXPERIMENT_FINAL_CANDIDATES")
		}
	}()

	firstRun := true
	for _, scenario := range scenarios {
		pcm, audioHash, err := readBenchAudio(scenario.Audio)
		if err != nil {
			t.Fatalf("scenario %q: %v", scenario.Name, err)
		}
		terms := benchTerms(scenario.DictionaryTerms)
		settingsForRun := settings.Defaults()
		settingsForRun.General.MainLanguage = scenario.Language
		settingsForRun.Features.Transcribe.ASR = modelID
		settingsForRun.Features.Transcribe.LLM = "benchmark-openai-compatible"
		settingsForRun.Features.Transcribe.Enhance = true
		settingsForRun.Features.Transcribe.Cleanup = true
		settingsForRun.Features.Transcribe.ShowCard = true
		settingsForRun.Dictionary.Enabled = len(terms) > 0
		settingsForRun.Dictionary.ASRHints = true
		settingsForRun.Dictionary.Glossary = true
		currentASR := baseASR
		if scenario.PiecesAfterMS != 0 || scenario.PieceMinMS != 0 || scenario.PieceTargetMS != 0 || scenario.PieceMaxMS != 0 {
			currentASR = currentASR.WithPieces(omni.Pieces{
				After:  time.Duration(scenario.PiecesAfterMS) * time.Millisecond,
				Min:    time.Duration(scenario.PieceMinMS) * time.Millisecond,
				Target: time.Duration(scenario.PieceTargetMS) * time.Millisecond,
				Max:    time.Duration(scenario.PieceMaxMS) * time.Millisecond,
			})
		}
		draftDir := strings.TrimSpace(os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_DRAFT_DIR"))
		if scenario.Draft {
			if draftDir == "" {
				t.Fatalf("scenario %q requests a draft but VOXTNEW_FINAL_CANDIDATE_BENCH_DRAFT_DIR is unset", scenario.Name)
			}
			currentASR = currentASR.WithDraft(draftDir)
		}
		activeMu.Lock()
		activeSettings, activeASR, activeTerms = settingsForRun, currentASR, terms
		activeMu.Unlock()

		for warmup := 0; warmup < warmups; warmup++ {
			for _, arm := range []string{"off", "shadow"} {
				run := runBenchSession(t, manager, logs, capture, pcm, scenario, arm, warmup, -1, firstRun, done, &activeMu, &activeRun)
				firstRun = false
				emitBenchRun(t, run, audioHash, modelID, llmModel, llmRemote)
			}
		}
		var pairResults []benchPairResult
		for round := 1; round <= rounds; round++ {
			arms := []string{"off", "shadow"}
			if round%2 == 0 {
				arms = []string{"shadow", "off"}
			}
			paired := make(map[string]*benchRun, 2)
			for _, arm := range arms {
				run := runBenchSession(t, manager, logs, capture, pcm, scenario, arm, -1, round, firstRun, done, &activeMu, &activeRun)
				firstRun = false
				paired[arm] = run
				emitBenchRun(t, run, audioHash, modelID, llmModel, llmRemote)
			}
			pairResults = append(pairResults, emitBenchPair(t, scenario, round, paired["off"], paired["shadow"]))
		}
		emitBenchSummary(t, scenario, pairResults)
	}
}

type benchScenario struct {
	Name            string   `json:"name"`
	Audio           string   `json:"audio"`
	Language        string   `json:"language"`
	HeadStartCase   string   `json:"headstart_case"`
	DictionaryTerms []string `json:"dictionary_terms"`
	Draft           bool     `json:"draft"`
	PiecesAfterMS   int      `json:"pieces_after_ms"`
	PieceMinMS      int      `json:"piece_min_ms"`
	PieceTargetMS   int      `json:"piece_target_ms"`
	PieceMaxMS      int      `json:"piece_max_ms"`
}

type benchCall struct {
	StartAt, FirstTokenAt, DoneAt time.Time
	InputChars                    int
	Usage                         llm.Usage
	Succeeded                     bool
}

type benchRun struct {
	mu             sync.Mutex
	Scenario       benchScenario
	Arm            string
	Warmup         bool
	Round          int
	ColdStart      bool
	StartedAt      time.Time
	StoppedAt      time.Time
	Snapshot       session.Snapshot
	RecordingAt    time.Time
	FinalAt        time.Time
	FirstDraftAt   time.Time
	CleanupDoneAt  time.Time
	DeliveryAt     time.Time
	SessionDoneAt  time.Time
	LLMCalls       []benchCall
	ExperimentLogs []json.RawMessage
	PlaybackFactor float64
}

type benchLog struct {
	mu     sync.Mutex
	active *benchRun
}

func (l *benchLog) setActive(run *benchRun) {
	l.mu.Lock()
	l.active = run
	l.mu.Unlock()
}

func (l *benchLog) add(_ string, message string) {
	start := strings.IndexByte(message, '{')
	if start < 0 {
		return
	}
	var event struct {
		Event string `json:"event"`
	}
	raw := json.RawMessage(message[start:])
	if json.Unmarshal(raw, &event) != nil ||
		!(strings.HasPrefix(event.Event, "voxt_final_candidate_") || event.Event == "omni_request_stage") {
		return
	}
	l.mu.Lock()
	if l.active != nil {
		l.active.mu.Lock()
		l.active.ExperimentLogs = append(l.active.ExperimentLogs, append(json.RawMessage(nil), raw...))
		l.active.mu.Unlock()
	}
	l.mu.Unlock()
}

func runBenchSession(t *testing.T, manager *session.Manager, logs *benchLog, audio *benchAudio, pcm []byte, scenario benchScenario, arm string, warmup, round int, cold bool, done <-chan session.Snapshot, activeMu *sync.Mutex, activeRun **benchRun) *benchRun {
	t.Helper()
	mode := ""
	if arm == "shadow" {
		mode = "shadow"
		if err := os.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", mode); err != nil {
			t.Fatal(err)
		}
	} else if err := os.Unsetenv("VOXT_EXPERIMENT_FINAL_CANDIDATES"); err != nil {
		t.Fatal(err)
	}
	run := &benchRun{Scenario: scenario, Arm: arm, Warmup: warmup >= 0, Round: round, ColdStart: cold, StartedAt: time.Now()}
	run.PlaybackFactor = audio.frameFactor
	ready := audio.prepare(pcm)
	select {
	case <-done:
	default:
	}
	activeMu.Lock()
	*activeRun = run
	activeMu.Unlock()
	logs.setActive(run)

	id, err := manager.Start(session.StartRequest{Kind: store.KindTranscribe})
	if err != nil {
		t.Fatalf("start %s session: %v", arm, err)
	}
	select {
	case <-ready:
	case <-time.After(20 * time.Minute):
		t.Fatalf("scenario %q audio did not finish playback", scenario.Name)
	}
	stopAt := time.Now()
	manager.Stop(id)
	run.mu.Lock()
	run.StoppedAt = stopAt
	run.mu.Unlock()
	select {
	case snapshot := <-done:
		run.Snapshot = snapshot
	case <-time.After(20 * time.Minute):
		t.Fatalf("scenario %q %s session did not finish", scenario.Name, arm)
	}
	logs.setActive(nil)
	activeMu.Lock()
	*activeRun = nil
	activeMu.Unlock()
	return run
}

func (r *benchRun) observe(snapshot session.Snapshot) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Snapshot = snapshot
	now := time.Now()
	if snapshot.State == session.Recording && r.RecordingAt.IsZero() {
		r.RecordingAt = now
	}
	if snapshot.Raw != "" && r.FinalAt.IsZero() {
		r.FinalAt = now
	}
	if snapshot.Draft != "" && r.FirstDraftAt.IsZero() {
		r.FirstDraftAt = now
	}
	if snapshot.State == session.Delivering && r.CleanupDoneAt.IsZero() {
		r.CleanupDoneAt = now
	}
	if snapshot.Result != "" && r.DeliveryAt.IsZero() {
		r.DeliveryAt = now
	}
	if snapshot.State.Ended() && r.SessionDoneAt.IsZero() {
		r.SessionDoneAt = now
	}
}

type benchLLM struct {
	client *llm.Client
	active func() *benchRun
}

func (l *benchLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	run := l.active()
	call := benchCall{StartAt: time.Now(), InputChars: request.InputChars}
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
			run.LLMCalls[index].FirstTokenAt = time.Now()
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

type benchTextIO struct{}

func (benchTextIO) Focus(context.Context) (proto.FocusResult, error) {
	return proto.FocusResult{}, nil
}

func (benchTextIO) Paste(context.Context, proto.PasteParams) (proto.PasteResult, error) {
	return proto.PasteResult{Posted: true}, nil
}
