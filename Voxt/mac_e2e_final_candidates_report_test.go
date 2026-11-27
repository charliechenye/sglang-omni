package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type benchRunSnapshot struct {
	Scenario            benchScenario
	RunID               string
	ExperimentSessionID uint64
	Arm                 string
	Warmup              bool
	Round               int
	ColdStart           bool
	ProtocolSmokePassed bool
	StartedAt           time.Time
	StoppedAt           time.Time
	RecordingAt         time.Time
	FinalAt             time.Time
	FirstDraftAt        time.Time
	CleanupDoneAt       time.Time
	ResultReadyAt       time.Time
	DeliveryRequestedAt time.Time
	DeliveryCompletedAt time.Time
	SessionDoneAt       time.Time
	Snapshot            session.Snapshot
	LLMCalls            []benchCall
	ExperimentLogs      []json.RawMessage
	PlaybackFactor      float64
}

type benchRuntimeProvenance struct {
	SchemaVersion       int    `json:"schema_version"`
	BuildCommand        string `json:"build_command"`
	NativeSourcePath    string `json:"native_source_path"`
	ExperimentCommit    string `json:"experiment_commit"`
	NativeSourceCommit  string `json:"native_source_commit"`
	NativeWorktreeClean bool   `json:"native_worktree_clean"`
	RuntimeBinaryPath   string `json:"runtime_binary_path"`
	RuntimeBinarySHA256 string `json:"runtime_binary_sha256"`
}

type benchAdoptionStats struct {
	ExactMatchAdoptionCount       int
	NearMatchRepairAdoptionCount  int
	PartialMultipartAdoption      bool
	HeadStartMiss                 bool
	FailedSpeculativeRequestCount int
	FallbackCount                 int
	NoCleanupRequested            bool
	AdoptedParts                  int
	CleanupParts                  int
	OutcomesByRequestID           map[uint64]string
	AdoptedRequestIDs             map[uint64]bool
}

func readBenchRuntimeProvenance(binaryPath string) (string, benchRuntimeProvenance, error) {
	repositoryRootOutput, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", benchRuntimeProvenance{}, fmt.Errorf("locate experiment repository: %w", err)
	}
	repositoryRoot := strings.TrimSpace(string(repositoryRootOutput))
	commitOutput, err := exec.Command("git", "-C", repositoryRoot, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", benchRuntimeProvenance{}, fmt.Errorf("read experiment commit: %w", err)
	}
	commit := strings.TrimSpace(string(commitOutput))
	statusOutput, err := exec.Command("git", "-C", repositoryRoot, "status", "--porcelain", "--untracked-files=all").Output()
	if err != nil {
		return "", benchRuntimeProvenance{}, fmt.Errorf("read experiment worktree status: %w", err)
	}
	if len(statusOutput) != 0 {
		return "", benchRuntimeProvenance{}, fmt.Errorf("the experiment worktree must be clean before real-model screening")
	}
	binaryPath, err = filepath.Abs(binaryPath)
	if err != nil {
		return "", benchRuntimeProvenance{}, err
	}
	provenancePath := filepath.Join(filepath.Dir(filepath.Dir(binaryPath)), "build-provenance.json")
	encoded, err := os.ReadFile(provenancePath)
	if err != nil {
		return "", benchRuntimeProvenance{}, fmt.Errorf("read native runtime provenance %s: %w", provenancePath, err)
	}
	var provenance benchRuntimeProvenance
	if err := json.Unmarshal(encoded, &provenance); err != nil {
		return "", benchRuntimeProvenance{}, fmt.Errorf("decode native runtime provenance: %w", err)
	}
	binaryBytes, err := os.ReadFile(binaryPath)
	if err != nil {
		return "", benchRuntimeProvenance{}, err
	}
	binaryHash := sha256.Sum256(binaryBytes)
	expectedSource, err := filepath.Abs(filepath.Join(repositoryRoot, "sglang_omni_mlx", "native"))
	if err != nil {
		return "", benchRuntimeProvenance{}, err
	}
	provenanceSource, sourceErr := filepath.Abs(provenance.NativeSourcePath)
	provenanceBinary, binaryErr := filepath.Abs(provenance.RuntimeBinaryPath)
	if provenance.SchemaVersion != 1 || provenance.ExperimentCommit != commit ||
		provenance.NativeSourceCommit != commit || !provenance.NativeWorktreeClean ||
		sourceErr != nil || provenanceSource != expectedSource || binaryErr != nil ||
		provenanceBinary != binaryPath || provenance.RuntimeBinarySHA256 != hex.EncodeToString(binaryHash[:]) {
		return "", benchRuntimeProvenance{}, fmt.Errorf("native runtime provenance does not match clean experiment commit %s", commit)
	}
	return commit, provenance, nil
}

func (r *benchRun) snapshot() benchRunSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	return benchRunSnapshot{
		Scenario: r.Scenario, RunID: r.RunID, ExperimentSessionID: r.ExperimentSessionID,
		Arm: r.Arm, Warmup: r.Warmup, Round: r.Round,
		ColdStart: r.ColdStart, ProtocolSmokePassed: r.ProtocolSmokePassed,
		StartedAt: r.StartedAt, StoppedAt: r.StoppedAt,
		RecordingAt: r.RecordingAt, FinalAt: r.FinalAt, FirstDraftAt: r.FirstDraftAt,
		CleanupDoneAt: r.CleanupDoneAt, ResultReadyAt: r.ResultReadyAt,
		DeliveryRequestedAt: r.DeliveryRequestedAt,
		DeliveryCompletedAt: r.DeliveryCompletedAt,
		SessionDoneAt:       r.SessionDoneAt, Snapshot: r.Snapshot,
		LLMCalls:       append([]benchCall(nil), r.LLMCalls...),
		ExperimentLogs: append([]json.RawMessage(nil), r.ExperimentLogs...),
		PlaybackFactor: r.PlaybackFactor,
	}
}

func emitBenchRun(t *testing.T, run *benchRun, audioHash, modelID, llmModel string, llmRemote bool, experimentCommit string, runtimeProvenance benchRuntimeProvenance) {
	t.Helper()
	data := run.snapshot()
	adoptionStats := measuredHeadstartOutcomes(data)
	rawHash, rawBytes, rawRunes := textStats(data.Snapshot.Raw)
	resultHash, resultBytes, resultRunes := textStats(data.Snapshot.Result)
	calls := make([]map[string]any, 0, len(data.LLMCalls))
	for _, call := range data.LLMCalls {
		calls = append(calls, map[string]any{
			"request_kind":                        call.Kind,
			"request_id":                          call.RequestID,
			"headstart_adoption_outcome":          adoptionStats.OutcomesByRequestID[call.RequestID],
			"headstart_request_adopted":           adoptionStats.AdoptedRequestIDs[call.RequestID],
			"start_offset_stop_ms":                offsetMS(data.StoppedAt, call.StartAt),
			"provider_first_token_offset_stop_ms": offsetMS(data.StoppedAt, call.FirstTokenAt),
			"first_token_valid":                   !call.FirstTokenAt.IsZero(),
			"completion_offset_stop_ms":           offsetMS(data.StoppedAt, call.DoneAt),
			"input_chars":                         call.InputChars,
			"usage":                               call.Usage,
			"succeeded":                           call.Succeeded,
		})
	}
	firstLLMRequestMS, firstLLMRequestValid := firstLLMRequestOffset(data.StoppedAt, data.LLMCalls)
	candidateEvents, nativeCandidateCount, snapshotBuildMSTotal, snapshotBuildMSReceived, firstTranscriptTokenMS := candidateNativeStats(data.ExperimentLogs)
	coverage := candidateProtocolCoverage(data.ExperimentLogs)
	languageMode := "explicit"
	if strings.TrimSpace(data.Scenario.Language) == "" {
		languageMode = "automatic"
	}
	row := map[string]any{
		"event": "final_candidate_benchmark_run", "scenario": data.Scenario.Name,
		"run_id":                              data.RunID,
		"session_id":                          data.ExperimentSessionID,
		"experiment_commit":                   experimentCommit,
		"native_runtime_provenance":           runtimeProvenance,
		"measured_headstart_outcome":          adoptionStats,
		"protocol_smoke_required":             data.Scenario.ProtocolSmoke,
		"protocol_smoke_passed":               data.ProtocolSmokePassed || !data.Scenario.ProtocolSmoke || data.Arm == "off",
		"candidate_protocol_support_observed": coverage.HTTPObservedCount > 0,
		"session_without_candidates":          coverage.ProtocolReported && coverage.ProtocolEnabled && coverage.HTTPObservedCount == 0,
		"language_hint":                       data.Scenario.Language, "language_mode": languageMode, "headstart_case": data.Scenario.HeadStartCase,
		"dictionary_term_count": len(data.Scenario.DictionaryTerms), "draft": data.Scenario.Draft,
		"pieces_after_ms": data.Scenario.PiecesAfterMS, "piece_min_ms": data.Scenario.PieceMinMS,
		"piece_target_ms": data.Scenario.PieceTargetMS, "piece_max_ms": data.Scenario.PieceMaxMS,
		"arm": data.Arm, "warmup": data.Warmup, "round": data.Round, "cold_start": data.ColdStart,
		"playback_factor": data.PlaybackFactor, "asr_model": modelID, "llm_model": llmModel, "llm_remote": llmRemote,
		"audio_sha256": audioHash, "raw_transcript_sha256": rawHash, "raw_transcript_bytes": rawBytes, "raw_transcript_runes": rawRunes,
		"result_sha256": resultHash, "result_bytes": resultBytes, "result_runes": resultRunes,
		"start_to_recording_ms":                   offsetMS(data.StartedAt, data.RecordingAt),
		"stop_to_final_available_ms":              offsetMS(data.StoppedAt, data.FinalAt),
		"stop_to_first_llm_request_ms":            firstLLMRequestMS,
		"stop_to_first_llm_request_valid":         firstLLMRequestValid,
		"stop_to_first_visible_cleanup_text_ms":   offsetMS(data.StoppedAt, data.FirstDraftAt),
		"first_visible_cleanup_text_valid":        !data.FirstDraftAt.IsZero(),
		"stop_to_cleanup_complete_ms":             offsetMS(data.StoppedAt, data.CleanupDoneAt),
		"cleanup_complete_valid":                  !data.CleanupDoneAt.IsZero(),
		"stop_to_result_ready_ms":                 offsetMS(data.StoppedAt, data.ResultReadyAt),
		"result_ready_valid":                      !data.ResultReadyAt.IsZero(),
		"stop_to_delivery_requested_ms":           offsetMS(data.StoppedAt, data.DeliveryRequestedAt),
		"delivery_requested_valid":                !data.DeliveryRequestedAt.IsZero(),
		"stop_to_simulated_delivery_completed_ms": offsetMS(data.StoppedAt, data.DeliveryCompletedAt),
		"simulated_delivery_completed_valid":      !data.DeliveryCompletedAt.IsZero(),
		"delivery_mode":                           "simulated_fake_text_io",
		"stop_to_session_terminal_ms":             offsetMS(data.StoppedAt, data.SessionDoneAt),
		"session_terminal_valid":                  !data.SessionDoneAt.IsZero(),
		"asr_stage_ms":                            durationMS(data.Snapshot.Timings.ASR), "llm_stage_ms": durationMS(data.Snapshot.Timings.LLM),
		"delivery_stage_ms": durationMS(data.Snapshot.Timings.Deliver), "llm_call_count": len(calls), "llm_calls": calls,
		"http_candidate_events_received":           candidateEvents,
		"native_published_candidate_count":         nativeCandidateCount,
		"native_snapshot_build_ms_total":           snapshotBuildMSTotal,
		"native_snapshot_build_ms_received_sum":    snapshotBuildMSReceived,
		"first_transcript_token_native_elapsed_ms": firstTranscriptTokenMS,
		"candidate_coverage":                       coverage,
		"experiment_events":                        data.ExperimentLogs,
		"status":                                   data.Snapshot.State, "problem": data.Snapshot.Problem.Code,
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
}

func measuredHeadstartOutcomes(data benchRunSnapshot) benchAdoptionStats {
	stats := benchAdoptionStats{
		OutcomesByRequestID: make(map[uint64]string),
		AdoptedRequestIDs:   make(map[uint64]bool),
	}
	headstartRequestCount, cleanupRequestCount := 0, 0
	cleanupPartHashes := make(map[string]int)
	adoptedPartHashes := make(map[string]int)
	for _, call := range data.LLMCalls {
		switch call.Kind {
		case "headstart_recording", "headstart_late":
			headstartRequestCount++
		case "cleanup", "early_cleanup":
			cleanupRequestCount++
		}
	}
	for _, raw := range data.ExperimentLogs {
		var event struct {
			Event       string   `json:"event"`
			Source      string   `json:"source"`
			Outcome     string   `json:"outcome"`
			Adopted     bool     `json:"adopted"`
			RequestID   uint64   `json:"request_id"`
			RequestHash string   `json:"request_plan_sha256"`
			PartHashes  []string `json:"part_request_sha256"`
			PartCount   int      `json:"part_count"`
		}
		if json.Unmarshal(raw, &event) != nil {
			continue
		}
		if event.Event == "voxt_headstart_adoption" {
			stats.OutcomesByRequestID[event.RequestID] = event.Outcome
			stats.AdoptedRequestIDs[event.RequestID] = event.Adopted
			if event.Adopted {
				adoptedPartHashes[event.RequestHash]++
			}
			switch event.Outcome {
			case "exact_match":
				if event.Adopted {
					stats.ExactMatchAdoptionCount++
				}
			case "near_match_repair_adoption":
				if event.Adopted {
					stats.NearMatchRepairAdoptionCount++
				}
			case "failed_speculative_request":
				stats.FailedSpeculativeRequestCount++
			case "fresh_request_fallback", "near_match_repair_failed":
				stats.FallbackCount++
			}
		}
		if event.Event == "voxt_cleanup_request_plan" &&
			(event.Source == "authoritative_cleanup" || event.Source == "early_cleanup") {
			for _, hash := range event.PartHashes {
				cleanupPartHashes[hash]++
			}
			if event.PartCount > 0 && len(event.PartHashes) == 0 {
				stats.CleanupParts += event.PartCount
			}
		}
	}
	for hash, partCount := range cleanupPartHashes {
		stats.CleanupParts += partCount
		stats.AdoptedParts += min(partCount, adoptedPartHashes[hash])
	}
	stats.PartialMultipartAdoption = stats.CleanupParts > 1 && stats.AdoptedParts > 0 && stats.AdoptedParts < stats.CleanupParts
	stats.HeadStartMiss = headstartRequestCount > 0 && cleanupRequestCount > 0 &&
		stats.ExactMatchAdoptionCount+stats.NearMatchRepairAdoptionCount == 0
	stats.NoCleanupRequested = cleanupRequestCount == 0
	return stats
}
