package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type benchRunSnapshot struct {
	Scenario       benchScenario
	Arm            string
	Warmup         bool
	Round          int
	ColdStart      bool
	StartedAt      time.Time
	StoppedAt      time.Time
	RecordingAt    time.Time
	FinalAt        time.Time
	FirstDraftAt   time.Time
	CleanupDoneAt  time.Time
	DeliveryAt     time.Time
	SessionDoneAt  time.Time
	Snapshot       session.Snapshot
	LLMCalls       []benchCall
	ExperimentLogs []json.RawMessage
	PlaybackFactor float64
}

func (r *benchRun) snapshot() benchRunSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	return benchRunSnapshot{
		Scenario: r.Scenario, Arm: r.Arm, Warmup: r.Warmup, Round: r.Round,
		ColdStart: r.ColdStart, StartedAt: r.StartedAt, StoppedAt: r.StoppedAt,
		RecordingAt: r.RecordingAt, FinalAt: r.FinalAt, FirstDraftAt: r.FirstDraftAt,
		CleanupDoneAt: r.CleanupDoneAt, DeliveryAt: r.DeliveryAt,
		SessionDoneAt: r.SessionDoneAt, Snapshot: r.Snapshot,
		LLMCalls:       append([]benchCall(nil), r.LLMCalls...),
		ExperimentLogs: append([]json.RawMessage(nil), r.ExperimentLogs...),
		PlaybackFactor: r.PlaybackFactor,
	}
}

func emitBenchRun(t *testing.T, run *benchRun, audioHash, modelID, llmModel string, llmRemote bool) {
	t.Helper()
	data := run.snapshot()
	rawHash, rawBytes, rawRunes := textStats(data.Snapshot.Raw)
	resultHash, resultBytes, resultRunes := textStats(data.Snapshot.Result)
	calls := make([]map[string]any, 0, len(data.LLMCalls))
	for _, call := range data.LLMCalls {
		calls = append(calls, map[string]any{
			"start_offset_stop_ms":       offsetMS(data.StoppedAt, call.StartAt),
			"first_token_offset_stop_ms": offsetMS(data.StoppedAt, call.FirstTokenAt),
			"completion_offset_stop_ms":  offsetMS(data.StoppedAt, call.DoneAt),
			"input_chars":                call.InputChars,
			"usage":                      call.Usage,
			"succeeded":                  call.Succeeded,
		})
	}
	candidateEvents, nativeCandidateCount, snapshotBuildMSTotal, snapshotBuildMSReceived, firstTranscriptTokenMS := candidateNativeStats(data.ExperimentLogs)
	row := map[string]any{
		"event": "final_candidate_benchmark_run", "scenario": data.Scenario.Name,
		"language_hint": data.Scenario.Language, "headstart_case": data.Scenario.HeadStartCase,
		"dictionary_term_count": len(data.Scenario.DictionaryTerms), "draft": data.Scenario.Draft,
		"pieces_after_ms": data.Scenario.PiecesAfterMS, "piece_min_ms": data.Scenario.PieceMinMS,
		"piece_target_ms": data.Scenario.PieceTargetMS, "piece_max_ms": data.Scenario.PieceMaxMS,
		"arm": data.Arm, "warmup": data.Warmup, "round": data.Round, "cold_start": data.ColdStart,
		"playback_factor": data.PlaybackFactor, "asr_model": modelID, "llm_model": llmModel, "llm_remote": llmRemote,
		"audio_sha256": audioHash, "raw_transcript_sha256": rawHash, "raw_transcript_bytes": rawBytes, "raw_transcript_runes": rawRunes,
		"result_sha256": resultHash, "result_bytes": resultBytes, "result_runes": resultRunes,
		"start_to_recording_ms":          offsetMS(data.StartedAt, data.RecordingAt),
		"stop_to_final_available_ms":     offsetMS(data.StoppedAt, data.FinalAt),
		"stop_to_first_llm_request_ms":   firstLLMRequestOffset(data.StoppedAt, data.LLMCalls),
		"stop_to_first_cleanup_token_ms": offsetMS(data.StoppedAt, data.FirstDraftAt),
		"stop_to_cleanup_complete_ms":    offsetMS(data.StoppedAt, data.CleanupDoneAt),
		"stop_to_delivery_ready_ms":      offsetMS(data.StoppedAt, data.DeliveryAt),
		"stop_to_session_done_ms":        offsetMS(data.StoppedAt, data.SessionDoneAt),
		"asr_stage_ms":                   durationMS(data.Snapshot.Timings.ASR), "llm_stage_ms": durationMS(data.Snapshot.Timings.LLM),
		"delivery_stage_ms": durationMS(data.Snapshot.Timings.Deliver), "llm_call_count": len(calls), "llm_calls": calls,
		"http_candidate_events_received":           candidateEvents,
		"native_published_candidate_count":         nativeCandidateCount,
		"native_snapshot_build_ms_total":           snapshotBuildMSTotal,
		"native_snapshot_build_ms_received_sum":    snapshotBuildMSReceived,
		"first_transcript_token_native_elapsed_ms": firstTranscriptTokenMS,
		"experiment_events":                        data.ExperimentLogs,
		"status":                                   data.Snapshot.State, "problem": data.Snapshot.Problem.Code,
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
}

type benchPairResult struct {
	TranscriptEqual   bool
	ResultEqual       bool
	FinalValid        bool
	RequestStartValid bool
	FirstTokenValid   bool
	CleanupValid      bool
	DeliveryValid     bool
	FinalDeltaMS      float64
	RequestStartMS    float64
	FirstTokenMS      float64
	CleanupDeltaMS    float64
	DeliveryDeltaMS   float64
	RequestDelta      int
}

func emitBenchPair(t *testing.T, scenario benchScenario, round int, off, shadow *benchRun) benchPairResult {
	t.Helper()
	offData, shadowData := off.snapshot(), shadow.snapshot()
	offFinal := offsetMS(offData.StoppedAt, offData.FinalAt)
	shadowFinal := offsetMS(shadowData.StoppedAt, shadowData.FinalAt)
	offRequestStart := firstLLMRequestOffset(offData.StoppedAt, offData.LLMCalls)
	shadowRequestStart := firstLLMRequestOffset(shadowData.StoppedAt, shadowData.LLMCalls)
	offFirstToken := offsetMS(offData.StoppedAt, offData.FirstDraftAt)
	shadowFirstToken := offsetMS(shadowData.StoppedAt, shadowData.FirstDraftAt)
	offDelivery := offsetMS(offData.StoppedAt, offData.DeliveryAt)
	shadowDelivery := offsetMS(shadowData.StoppedAt, shadowData.DeliveryAt)
	offCleanup := offsetMS(offData.StoppedAt, offData.CleanupDoneAt)
	shadowCleanup := offsetMS(shadowData.StoppedAt, shadowData.CleanupDoneAt)
	result := benchPairResult{
		TranscriptEqual:   textHash(offData.Snapshot.Raw) == textHash(shadowData.Snapshot.Raw),
		ResultEqual:       textHash(offData.Snapshot.Result) == textHash(shadowData.Snapshot.Result),
		FinalValid:        offFinal >= 0 && shadowFinal >= 0,
		RequestStartValid: offRequestStart >= 0 && shadowRequestStart >= 0,
		FirstTokenValid:   offFirstToken >= 0 && shadowFirstToken >= 0,
		CleanupValid:      offCleanup >= 0 && shadowCleanup >= 0,
		DeliveryValid:     offDelivery >= 0 && shadowDelivery >= 0,
		FinalDeltaMS:      subtractIfValid(shadowFinal, offFinal),
		RequestStartMS:    subtractIfValid(shadowRequestStart, offRequestStart),
		FirstTokenMS:      subtractIfValid(shadowFirstToken, offFirstToken),
		CleanupDeltaMS:    subtractIfValid(shadowCleanup, offCleanup),
		DeliveryDeltaMS:   subtractIfValid(shadowDelivery, offDelivery),
		RequestDelta:      len(shadowData.LLMCalls) - len(offData.LLMCalls),
	}
	row := map[string]any{
		"event": "final_candidate_benchmark_pair", "scenario": scenario.Name, "round": round,
		"transcript_equal":               result.TranscriptEqual,
		"result_equal":                   result.ResultEqual,
		"off_stop_to_final_available_ms": offFinal, "shadow_stop_to_final_available_ms": shadowFinal,
		"final_available_delta_shadow_minus_off_ms": result.FinalDeltaMS,
		"off_stop_to_first_llm_request_ms":          offRequestStart, "shadow_stop_to_first_llm_request_ms": shadowRequestStart,
		"first_llm_request_delta_shadow_minus_off_ms": result.RequestStartMS,
		"off_stop_to_first_cleanup_token_ms":          offFirstToken, "shadow_stop_to_first_cleanup_token_ms": shadowFirstToken,
		"first_cleanup_token_delta_shadow_minus_off_ms": result.FirstTokenMS,
		"off_stop_to_cleanup_complete_ms":               offCleanup, "shadow_stop_to_cleanup_complete_ms": shadowCleanup,
		"cleanup_complete_delta_shadow_minus_off_ms": result.CleanupDeltaMS,
		"off_stop_to_delivery_ready_ms":              offDelivery, "shadow_stop_to_delivery_ready_ms": shadowDelivery,
		"delivery_delta_shadow_minus_off_ms": result.DeliveryDeltaMS,
		"off_llm_request_count":              len(offData.LLMCalls), "shadow_llm_request_count": len(shadowData.LLMCalls),
		"llm_request_delta": result.RequestDelta,
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
	return result
}

func emitBenchSummary(t *testing.T, scenario benchScenario, pairs []benchPairResult) {
	t.Helper()
	finalDeltas, requestStartDeltas, firstTokenDeltas := []float64{}, []float64{}, []float64{}
	cleanupDeltas, deliveryDeltas := []float64{}, []float64{}
	transcriptMatches, resultMatches, requestDelta := 0, 0, 0
	for _, pair := range pairs {
		if pair.TranscriptEqual {
			transcriptMatches++
		}
		if pair.ResultEqual {
			resultMatches++
		}
		requestDelta += pair.RequestDelta
		if pair.FinalValid {
			finalDeltas = append(finalDeltas, pair.FinalDeltaMS)
		}
		if pair.RequestStartValid {
			requestStartDeltas = append(requestStartDeltas, pair.RequestStartMS)
		}
		if pair.FirstTokenValid {
			firstTokenDeltas = append(firstTokenDeltas, pair.FirstTokenMS)
		}
		if pair.CleanupValid {
			cleanupDeltas = append(cleanupDeltas, pair.CleanupDeltaMS)
		}
		if pair.DeliveryValid {
			deliveryDeltas = append(deliveryDeltas, pair.DeliveryDeltaMS)
		}
	}
	row := map[string]any{
		"event": "final_candidate_benchmark_summary", "scenario": scenario.Name,
		"paired_sample_count": len(pairs), "transcript_match_count": transcriptMatches,
		"cleanup_result_match_count": resultMatches, "llm_request_delta_total": requestDelta,
		"final_available_delta_ms":     distribution(finalDeltas),
		"first_llm_request_delta_ms":   distribution(requestStartDeltas),
		"first_cleanup_token_delta_ms": distribution(firstTokenDeltas),
		"cleanup_complete_delta_ms":    distribution(cleanupDeltas),
		"delivery_delta_ms":            distribution(deliveryDeltas),
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
}

func distribution(values []float64) map[string]any {
	if len(values) == 0 {
		return map[string]any{"sample_count": 0, "p50": -1, "p95": -1}
	}
	sort.Float64s(values)
	percentile := func(fraction float64) float64 {
		index := int(float64(len(values))*fraction+0.999999) - 1
		return values[max(0, min(index, len(values)-1))]
	}
	return map[string]any{"sample_count": len(values), "p50": percentile(0.50), "p95": percentile(0.95)}
}

func candidateNativeStats(events []json.RawMessage) (int, uint64, float64, float64, float64) {
	receivedCount := 0
	publishedCount := uint64(0)
	buildMSTotal := float64(-1)
	buildMSReceived := float64(0)
	firstTokenMS := float64(-1)
	for _, raw := range events {
		var event struct {
			Event          string  `json:"event"`
			Source         string  `json:"source"`
			FirstToken     float64 `json:"first_transcript_token_elapsed_ms"`
			SnapshotBuild  float64 `json:"snapshot_build_ms"`
			PublishedCount uint64  `json:"published_candidate_count"`
			BuildMSTotal   float64 `json:"snapshot_build_ms_total"`
		}
		if json.Unmarshal(raw, &event) != nil || event.Event != "voxt_final_candidate_observed" || event.Source != "http_final" {
			continue
		}
		receivedCount++
		buildMSReceived += event.SnapshotBuild
		if event.PublishedCount > publishedCount {
			publishedCount = event.PublishedCount
		}
		if event.BuildMSTotal > buildMSTotal {
			buildMSTotal = event.BuildMSTotal
		}
		if firstTokenMS < 0 && event.FirstToken > 0 {
			firstTokenMS = event.FirstToken
		}
	}
	return receivedCount, publishedCount, buildMSTotal, buildMSReceived, firstTokenMS
}

func firstLLMRequestOffset(stopAt time.Time, calls []benchCall) float64 {
	first := float64(-1)
	for _, call := range calls {
		offset := offsetMS(stopAt, call.StartAt)
		if offset >= 0 && (first < 0 || offset < first) {
			first = offset
		}
	}
	return first
}

func TestCandidateNativeStatsIncludesCoalescedBuildTotals(t *testing.T) {
	events := []json.RawMessage{
		json.RawMessage(`{"event":"voxt_final_candidate_observed","source":"http_final","first_transcript_token_elapsed_ms":8.5,"snapshot_build_ms":0.4,"published_candidate_count":1,"snapshot_build_ms_total":0.4}`),
		json.RawMessage(`{"event":"voxt_final_candidate_observed","source":"http_final","first_transcript_token_elapsed_ms":8.5,"snapshot_build_ms":0.7,"published_candidate_count":3,"snapshot_build_ms_total":1.8}`),
	}
	received, published, total, receivedSum, firstToken := candidateNativeStats(events)
	if received != 2 || published != 3 || total != 1.8 || receivedSum != 1.1 || firstToken != 8.5 {
		t.Fatalf("stats = %d, %d, %.1f, %.1f, %.1f", received, published, total, receivedSum, firstToken)
	}
}

func TestFirstLLMRequestOffsetUsesEarliestCall(t *testing.T) {
	stopAt := time.Now()
	calls := []benchCall{
		{StartAt: stopAt.Add(150 * time.Millisecond)},
		{StartAt: stopAt.Add(25 * time.Millisecond)},
	}
	if offset := firstLLMRequestOffset(stopAt, calls); offset != 25 {
		t.Fatalf("first LLM request offset = %v ms, want 25", offset)
	}
}

func subtractIfValid(left, right float64) float64 {
	if left < 0 || right < 0 {
		return -1
	}
	return left - right
}

func textStats(text string) (string, int, int) {
	if text == "" {
		return "", 0, 0
	}
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:]), len(text), len([]rune(text))
}

func textHash(text string) string {
	hash, _, _ := textStats(text)
	return hash
}

func offsetMS(start, end time.Time) float64 {
	if start.IsZero() || end.IsZero() {
		return -1
	}
	return float64(end.Sub(start).Microseconds()) / 1000
}

func durationMS(duration time.Duration) float64 { return float64(duration.Microseconds()) / 1000 }

func benchTerms(values []string) []store.Term {
	terms := make([]store.Term, 0, len(values))
	for index, value := range values {
		terms = append(terms, store.Term{ID: fmt.Sprintf("bench-%d", index), Term: value, Enabled: true})
	}
	return terms
}

func readBenchAudio(path string) ([]byte, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	pcm, sampleRate, channels, err := asr.ReadWAV(raw)
	if err != nil {
		return nil, "", err
	}
	if sampleRate != 16000 || channels != 1 {
		return nil, "", fmt.Errorf("audio must be 16 kHz mono, not %d Hz x %d channels", sampleRate, channels)
	}
	hash := sha256.Sum256(raw)
	return pcm, hex.EncodeToString(hash[:]), nil
}

func readBenchScenarios() ([]benchScenario, error) {
	if path := strings.TrimSpace(os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS")); path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var scenarios []benchScenario
		if err := json.Unmarshal(raw, &scenarios); err != nil {
			return nil, err
		}
		if len(scenarios) == 0 {
			return nil, fmt.Errorf("scenario manifest must contain at least one scenario")
		}
		base := filepath.Dir(path)
		for index := range scenarios {
			if scenarios[index].Name == "" || scenarios[index].Audio == "" {
				return nil, fmt.Errorf("scenario %d needs name and audio", index+1)
			}
			if !filepath.IsAbs(scenarios[index].Audio) {
				scenarios[index].Audio = filepath.Join(base, scenarios[index].Audio)
			}
		}
		return scenarios, nil
	}
	path := strings.TrimSpace(os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_AUDIO"))
	if path == "" {
		return nil, fmt.Errorf("set VOXTNEW_FINAL_CANDIDATE_BENCH_AUDIO or VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIOS")
	}
	name := strings.TrimSpace(os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_SCENARIO"))
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	return []benchScenario{{
		Name: name, Audio: path, Language: os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_LANGUAGE"),
		HeadStartCase: os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_HEADSTART_CASE"),
	}}, nil
}

func requiredBenchEnv(t *testing.T, name string) string {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		t.Fatalf("%s is required", name)
	}
	return value
}

func benchInt(t *testing.T, name string, fallback int) int {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		t.Fatalf("%s must be a non-negative integer", name)
	}
	return parsed
}

func benchFloat(t *testing.T, name string, fallback float64) float64 {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		t.Fatalf("%s must be a number", name)
	}
	return parsed
}
