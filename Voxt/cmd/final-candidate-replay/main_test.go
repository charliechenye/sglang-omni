package main

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"
)

func TestReplaySelectsOnlyCausallyUsableCandidates(t *testing.T) {
	input := strings.Join([]string{
		`info omni: {"event":"voxt_final_candidate_observed","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":1}`,
		`info omni: {"event":"voxt_final_candidate_eligibility","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":1,"generated_token_count":4,"stop_to_candidate_ms":100,"candidate_eligible":true,"extends_latest_hypothesis":true,"meaningful_text_boundary":true}`,
		`info omni: {"event":"voxt_final_candidate_observed","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":2}`,
		`info omni: {"event":"voxt_final_candidate_eligibility","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":2,"generated_token_count":16,"stop_to_candidate_ms":300,"candidate_eligible":true,"latest_preview_match":true,"extends_latest_hypothesis":true}`,
		`info omni: {"event":"voxt_final_candidate_observed","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":3}`,
		`info omni: {"event":"voxt_final_candidate_eligibility","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":3,"generated_token_count":32,"stop_to_candidate_ms":700,"candidate_eligible":true,"extends_latest_hypothesis":true,"meaningful_text_boundary":true}`,
		`info omni: {"event":"voxt_final_candidate_outcome","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":1,"candidate_usable_before_final":true,"candidate_lead_ms":600}`,
		`info omni: {"event":"voxt_final_candidate_outcome","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":2,"candidate_usable_before_final":true,"candidate_lead_ms":900,"oracle_compatible":true,"oracle_incremental_compatible":true,"exact_transcript_match":true,"exact_cleanup_plan_match":true}`,
		`info omni: {"event":"voxt_final_candidate_outcome","session_id":1,"request_id":10,"attempt":1,"source":"http_final","sequence":3,"candidate_usable_before_final":false,"candidate_lead_ms":-25}`,
		`info omni: {"event":"voxt_final_candidate_observed","session_id":1,"request_id":20,"attempt":1,"source":"websocket_preview","sequence":1}`,
		`info omni: {"event":"voxt_final_candidate_eligibility","session_id":1,"request_id":20,"attempt":1,"source":"websocket_preview","sequence":1,"candidate_eligible":true}`,
		`info omni: {"event":"voxt_final_candidate_outcome","session_id":1,"request_id":20,"attempt":1,"source":"websocket_preview","sequence":1,"candidate_usable_before_final":true,"candidate_lead_ms":800,"oracle_compatible":true,"oracle_incremental_compatible":true}`,
		`info omni: {"event":"voxt_final_candidate_observed","session_id":2,"request_id":30,"attempt":1,"source":"http_final","sequence":1}`,
	}, "\n")
	report, err := replay(strings.NewReader(input), []int{8, 16, 32}, []int{250, 500})
	if err != nil {
		t.Fatal(err)
	}
	if report.SessionCount != 2 || report.ReceivedCandidateCount != 5 || report.SessionEligibilityCount != 4 || report.ObservedWithoutEligibilityCount != 1 {
		t.Fatalf("unexpected event counts: %+v", report)
	}
	if report.OracleCompatibleCandidateCount != 2 || report.OracleIncrementalCandidateCount != 2 {
		t.Fatalf("unexpected oracle counts: %+v", report)
	}
	results := make(map[string]policyResult, len(report.Policies))
	for _, result := range report.Policies {
		results[result.Name+":"+formatThreshold(result.Threshold)] = result
	}
	assertSelection := func(name string, selected, compatible, wasted int) {
		t.Helper()
		result, ok := results[name]
		if !ok {
			t.Fatalf("missing policy %q", name)
		}
		if result.SelectedSessionCount != selected || result.OracleCompatibleSelectionCount != compatible || result.WastedIfDispatchedCount != wasted {
			t.Errorf("%s selected=%d compatible=%d wasted=%d, want %d/%d/%d", name, result.SelectedSessionCount, result.OracleCompatibleSelectionCount, result.WastedIfDispatchedCount, selected, compatible, wasted)
		}
	}
	assertSelection("first_eligible_http:0", 1, 0, 1)
	assertSelection("latest_preview_match:0", 1, 1, 0)
	assertSelection("meaningful_boundary:0", 1, 0, 1)
	assertSelection("extends_latest_hypothesis:0", 1, 0, 1)
	assertSelection("websocket_control:0", 1, 1, 0)
	assertSelection("token_cadence:8", 1, 1, 0)
	assertSelection("token_cadence:16", 1, 1, 0)
	assertSelection("token_cadence:32", 0, 0, 0)
	assertSelection("time_cadence_ms:250", 1, 1, 0)
	assertSelection("time_cadence_ms:500", 0, 0, 0)
}

func TestReplayReadsBenchmarkEventEnvelope(t *testing.T) {
	bench := map[string]any{
		"event": "final_candidate_benchmark_run",
		"experiment_events": []map[string]any{
			{"event": "voxt_final_candidate_observed", "session_id": 8, "request_id": 12, "attempt": 1, "source": "http_final", "sequence": 1},
			{"event": "voxt_final_candidate_eligibility", "session_id": 8, "request_id": 12, "attempt": 1, "source": "http_final", "sequence": 1, "candidate_eligible": true, "generated_token_count": 8},
			{"event": "voxt_final_candidate_outcome", "session_id": 8, "request_id": 12, "attempt": 1, "source": "http_final", "sequence": 1, "candidate_usable_before_final": true, "candidate_lead_ms": 250, "oracle_incremental_compatible": true},
		},
	}
	line, err := json.Marshal(bench)
	if err != nil {
		t.Fatal(err)
	}
	report, err := replay(strings.NewReader("FINAL_CANDIDATE_BENCH "+string(line)), []int{8}, []int{250})
	if err != nil {
		t.Fatal(err)
	}
	if report.ReceivedCandidateCount != 1 || report.SessionEligibilityCount != 1 || report.OracleIncrementalCandidateCount != 1 {
		t.Fatalf("benchmark event envelope was not read: %+v", report)
	}
}

func formatThreshold(value float64) string {
	if value == 0 {
		return "0"
	}
	return strconv.FormatFloat(value, 'f', -1, 64)
}
