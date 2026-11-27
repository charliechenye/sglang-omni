package main

import (
	"context"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

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
	experimentCommit, runtimeProvenance, err := readBenchRuntimeProvenance(binary)
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
	done := make(chan session.Snapshot, 1)
	manager := session.New(session.Config{
		Audio: capture,
		ResolveASR: func(context.Context, session.Kind, settings.Settings) session.ASRReadiness {
			activeMu.Lock()
			provider := activeASR
			activeMu.Unlock()
			return session.ASRReadiness{ASR: provider, ASRInfo: session.Model{ID: modelID, Name: modelID}}
		},
		ResolveLLM: func(context.Context, session.Kind, settings.Settings) session.LLMReadiness {
			activeMu.Lock()
			owner := activeRun
			activeMu.Unlock()
			return session.LLMReadiness{
				LLM: &benchLLM{client: client, run: owner}, LLMInfo: session.Model{ID: "benchmark-openai-compatible", Name: llmModel, Remote: llmRemote},
				Outcome: session.LLMAvailable,
			}
		},
		Text: benchTextIO{active: func() *benchRun {
			activeMu.Lock()
			defer activeMu.Unlock()
			return activeRun
		}},
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
	allRuns := make([]*benchRun, 0, len(scenarios)*(warmups+rounds)*3)
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
		settingsForRun.Features.Transcribe.ShowCard = false
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

		scenarioRuns := make([]*benchRun, 0, (warmups+rounds)*3)
		for warmup := 0; warmup < warmups; warmup++ {
			for _, arm := range []string{"off", "shadow", "shadow_dense"} {
				run := runBenchSession(t, manager, logs, capture, pcm, scenario, arm, warmup, -1, firstRun, done, &activeMu, &activeRun)
				firstRun = false
				scenarioRuns = append(scenarioRuns, run)
				allRuns = append(allRuns, run)
				emitBenchRun(t, run, audioHash, modelID, llmModel, llmRemote, experimentCommit, runtimeProvenance)
			}
		}
		pairResults := map[string][]benchPairResult{"shadow": {}, "shadow_dense": {}}
		for round := 1; round <= rounds; round++ {
			arms := []string{"off", "shadow", "shadow_dense"}
			rotation := (round - 1) % len(arms)
			arms = append(arms[rotation:], arms[:rotation]...)
			paired := make(map[string]*benchRun, len(arms))
			for _, arm := range arms {
				run := runBenchSession(t, manager, logs, capture, pcm, scenario, arm, -1, round, firstRun, done, &activeMu, &activeRun)
				firstRun = false
				paired[arm] = run
				scenarioRuns = append(scenarioRuns, run)
				allRuns = append(allRuns, run)
				emitBenchRun(t, run, audioHash, modelID, llmModel, llmRemote, experimentCommit, runtimeProvenance)
			}
			for _, treatment := range []string{"shadow", "shadow_dense"} {
				pair := emitBenchPair(t, scenario, round, treatment, paired["off"], paired[treatment])
				pairResults[treatment] = append(pairResults[treatment], pair)
			}
		}
		for _, treatment := range []string{"shadow", "shadow_dense"} {
			emitBenchSummary(t, scenario, treatment, pairResults[treatment])
		}
		if scenario.ProtocolSmoke {
			for _, arm := range []string{"shadow", "shadow_dense"} {
				observed := false
				for _, run := range scenarioRuns {
					if run.Arm != arm {
						continue
					}
					if candidateProtocolCoverage(run.snapshot().ExperimentLogs).HTTPObservedCount > 0 {
						observed = true
						break
					}
				}
				if !observed {
					t.Fatalf("scenario %q did not demonstrate native candidate protocol support in any %s session", scenario.Name, arm)
				}
			}
		}
	}
	emitCandidateCoverageSummary(t, allRuns)
}
