package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func firstLLMRequestOffset(stopAt time.Time, calls []benchCall) (float64, bool) {
	if stopAt.IsZero() {
		return 0, false
	}
	var first float64
	found := false
	for _, call := range calls {
		if call.StartAt.IsZero() {
			continue
		}
		offset := offsetMS(stopAt, call.StartAt)
		if !found || offset < first {
			first = offset
			found = true
		}
	}
	return first, found
}

func firstProviderTokenOffset(stopAt time.Time, calls []benchCall) (float64, bool) {
	if stopAt.IsZero() {
		return 0, false
	}
	var first float64
	found := false
	for _, call := range calls {
		if call.FirstTokenAt.IsZero() {
			continue
		}
		offset := offsetMS(stopAt, call.FirstTokenAt)
		if !found || offset < first {
			first = offset
			found = true
		}
	}
	return first, found
}

func subtractWhenValid(left, right float64, leftValid, rightValid bool) float64 {
	if !leftValid || !rightValid {
		return -1
	}
	return left - right
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

func TestCandidateProtocolCoverageKeepsNoCandidateAndSuppressionReasons(t *testing.T) {
	events := []json.RawMessage{
		json.RawMessage(`{"event":"voxt_final_candidate_protocol","enabled":true,"cadence":"dense","decode_path":"http_final_single_chunk"}`),
		json.RawMessage(`{"event":"voxt_final_candidate_suppressed","source":"http_final","reason":"session_inactive"}`),
		json.RawMessage(`{"event":"voxt_final_candidate_eligibility","candidate_processing_ms":1.5,"candidate_gate_ms":0.25}`),
		json.RawMessage(`{"event":"voxt_final_candidate_summary","observations_capped":true,"observations_dropped":2,"candidate_channel_coalesced":3,"candidate_channel_pending":1}`),
	}
	coverage := candidateProtocolCoverage(events)
	if !coverage.ProtocolReported || !coverage.ProtocolEnabled || coverage.Cadence != "dense" {
		t.Fatalf("protocol state = %+v", coverage)
	}
	if coverage.HTTPObservedCount != 0 || coverage.SuppressedCount != 1 || coverage.SuppressionReasons["session_inactive"] != 1 {
		t.Fatalf("candidate coverage = %+v", coverage)
	}
	if !coverage.ObservationsCapped || coverage.ObservationsDropped != 2 || coverage.ChannelCoalesced != 3 || coverage.ChannelPending != 1 {
		t.Fatalf("drop coverage = %+v", coverage)
	}
	if coverage.CandidateProcessingMSTotal != 1.5 || coverage.CandidateGateMSTotal != 0.25 {
		t.Fatalf("Go processing totals = %.2f, %.2f", coverage.CandidateProcessingMSTotal, coverage.CandidateGateMSTotal)
	}
}

func TestMeasuredHeadstartOutcomesPreservesDuplicateMultipartRequests(t *testing.T) {
	data := benchRunSnapshot{ExperimentLogs: []json.RawMessage{
		json.RawMessage(`{"event":"voxt_cleanup_request_plan","source":"authoritative_cleanup","part_count":2,"part_request_sha256":["same","same"]}`),
		json.RawMessage(`{"event":"voxt_headstart_adoption","request_id":4,"request_plan_sha256":"same","adopted":true,"outcome":"exact_match"}`),
		json.RawMessage(`{"event":"voxt_headstart_adoption","request_id":4,"request_plan_sha256":"same","adopted":true,"outcome":"exact_match"}`),
	}}
	stats := measuredHeadstartOutcomes(data)
	if stats.CleanupParts != 2 || stats.AdoptedParts != 2 || stats.PartialMultipartAdoption {
		t.Fatalf("duplicate multipart adoption = %+v", stats)
	}
}

func TestValidateBenchScenariosRequiresExplicitProtocolSmoke(t *testing.T) {
	if err := validateBenchScenarios([]benchScenario{{Name: "auto", Language: ""}}); err == nil {
		t.Fatal("automatic-only scenario list passed without an explicit smoke fixture")
	}
	if err := validateBenchScenarios([]benchScenario{{Name: "smoke", ProtocolSmoke: true}}); err == nil {
		t.Fatal("automatic-language protocol smoke fixture passed")
	}
	if err := validateBenchScenarios([]benchScenario{{Name: "smoke", Language: "en", ProtocolSmoke: true}}); err != nil {
		t.Fatalf("explicit-language smoke fixture failed: %v", err)
	}
}

func TestFirstLLMRequestOffsetUsesEarliestCall(t *testing.T) {
	stopAt := time.Now()
	calls := []benchCall{
		{StartAt: stopAt.Add(150 * time.Millisecond)},
		{StartAt: stopAt.Add(25 * time.Millisecond)},
	}
	if offset, valid := firstLLMRequestOffset(stopAt, calls); !valid || offset != 25 {
		t.Fatalf("first LLM request offset = %v ms (valid %t), want 25 ms", offset, valid)
	}
}

func TestFirstLLMRequestOffsetIncludesPreStopRequests(t *testing.T) {
	stopAt := time.Now()
	calls := []benchCall{
		{StartAt: stopAt.Add(20 * time.Millisecond)},
		{StartAt: stopAt.Add(-75 * time.Millisecond)},
	}
	if offset, valid := firstLLMRequestOffset(stopAt, calls); !valid || offset != -75 {
		t.Fatalf("pre-Stop request offset = %v ms (valid %t), want -75 ms", offset, valid)
	}
	if _, valid := firstLLMRequestOffset(stopAt, nil); valid {
		t.Fatal("missing request timing must be explicitly invalid")
	}
}

func TestFirstProviderTokenTimestampKeepsFirstNonemptyDelta(t *testing.T) {
	started := time.Now()
	call := benchCall{}
	recordFirstToken(&call, started.Add(4*time.Millisecond))
	recordFirstToken(&call, started.Add(19*time.Millisecond))
	if got, want := call.FirstTokenAt, started.Add(4*time.Millisecond); !got.Equal(want) {
		t.Fatalf("first provider token at %v, want %v", got, want)
	}
}

func TestFirstProviderTokenOffsetIncludesPreStopDeltas(t *testing.T) {
	stopAt := time.Now()
	calls := []benchCall{
		{FirstTokenAt: stopAt.Add(20 * time.Millisecond)},
		{FirstTokenAt: stopAt.Add(-75 * time.Millisecond)},
	}
	if offset, valid := firstProviderTokenOffset(stopAt, calls); !valid || offset != -75 {
		t.Fatalf("pre-Stop first provider token offset = %v ms (valid %t), want -75 ms", offset, valid)
	}
	if _, valid := firstProviderTokenOffset(stopAt, nil); valid {
		t.Fatal("missing provider-token timing must be explicitly invalid")
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
		if err := validateBenchScenarios(scenarios); err != nil {
			return nil, err
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
	scenario := benchScenario{
		Name: name, Audio: path, Language: os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_LANGUAGE"),
		HeadStartCase: os.Getenv("VOXTNEW_FINAL_CANDIDATE_BENCH_HEADSTART_CASE"),
	}
	scenario.ProtocolSmoke = strings.TrimSpace(scenario.Language) != ""
	if err := validateBenchScenarios([]benchScenario{scenario}); err != nil {
		return nil, err
	}
	return []benchScenario{scenario}, nil
}

func validateBenchScenarios(scenarios []benchScenario) error {
	hasExplicitSmoke := false
	for index, scenario := range scenarios {
		if scenario.ProtocolSmoke {
			if strings.TrimSpace(scenario.Language) == "" {
				return fmt.Errorf("scenario %q marks protocol_smoke but has automatic language; the protocol smoke fixture needs an explicit language", scenario.Name)
			}
			hasExplicitSmoke = true
		}
		if scenario.Name == "" {
			return fmt.Errorf("scenario %d needs a name", index+1)
		}
	}
	if !hasExplicitSmoke {
		return fmt.Errorf("scenario manifest needs at least one explicit-language protocol_smoke fixture to verify candidate support")
	}
	return nil
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
