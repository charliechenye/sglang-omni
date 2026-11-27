package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func replayLines(t *testing.T, events ...map[string]any) replayReport {
	t.Helper()
	var lines []string
	for _, event := range events {
		raw, err := json.Marshal(event)
		if err != nil {
			t.Fatal(err)
		}
		lines = append(lines, "info omni: "+string(raw))
	}
	report, err := replay(strings.NewReader(strings.Join(lines, "\n")), []int{8, 16}, []int{250, 500})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func httpCandidate(sessionID uint64, requestID uint64, attempt int, sequence uint64, ready float64, parts ...string) map[string]any {
	return map[string]any{
		"event": "voxt_final_candidate_eligibility", "session_id": sessionID,
		"request_id": requestID, "attempt": attempt, "source": "http_final",
		"sequence": sequence, "stop_to_candidate_ms": ready - 5,
		"stop_to_eligibility_ms": ready, "candidate_eligible": true,
		"candidate_eligible_without_budget": true,
		"generated_token_count":             16, "part_request_sha256": parts,
		"recording_headstart_generations": 1,
	}
}

func candidateOutcome(sessionID uint64, requestID uint64, attempt int, source string, sequence uint64, compatible bool) map[string]any {
	return map[string]any{
		"event": "voxt_final_candidate_outcome", "session_id": sessionID,
		"request_id": requestID, "attempt": attempt, "source": source,
		"sequence": sequence, "authoritative_final_available": true,
		"outcome_unresolved": false, "oracle_compatible": compatible,
		"oracle_incremental_compatible": compatible,
		"exact_cleanup_plan_match":      compatible,
	}
}

func finalSummary(sessionID uint64, at float64, status string) map[string]any {
	return map[string]any{
		"event": "voxt_final_candidate_summary", "session_id": sessionID,
		"stop_to_final_available_ms": at, "final_status": status,
	}
}

func policyByName(t *testing.T, report replayReport, name string, threshold float64) policyResult {
	t.Helper()
	for _, current := range report.Policies {
		if current.Name == name && current.Threshold == threshold {
			return current
		}
	}
	t.Fatalf("missing policy %q at threshold %v", name, threshold)
	return policyResult{}
}

func TestReplayDoesNotSkipFailedFirstAttemptUsingFutureOutcome(t *testing.T) {
	first := httpCandidate(1, 10, 1, 1, 100, "a")
	second := httpCandidate(1, 11, 2, 1, 250, "b")
	firstFailure := map[string]any{
		"event": "voxt_final_candidate_outcome", "session_id": 1,
		"request_id": 10, "attempt": 1, "source": "http_final", "sequence": 1,
		"authoritative_final_available": false, "final_status": "failed",
		"outcome_unresolved": true,
	}
	secondSuccess := candidateOutcome(1, 11, 2, "http_final", 1, true)
	report := replayLines(t, first, firstFailure, second, secondSuccess, finalSummary(1, 400, "completed"))
	selected := policyByName(t, report, "http_diagnostic", 0)
	if selected.SelectedSessionCount != 1 || selected.UnresolvedSelectionCount != 1 {
		t.Fatalf("failed first candidate must remain selected and unresolved: %+v", selected)
	}
	if selected.OracleCompatibleSelectionCount != 0 {
		t.Fatalf("replay retroactively selected the second attempt: %+v", selected)
	}
}

func TestCombinedPolicyUsesOneBudgetForHTTPAndWebSocket(t *testing.T) {
	http := httpCandidate(2, 20, 1, 1, 100, "http")
	ws := map[string]any{
		"event": "voxt_headstart_decision", "session_id": 2,
		"source": "websocket_preview", "sequence": 2,
		"candidate_plan_ready_ms": 150, "dispatch_ready_ms": 150,
		"preparation_succeeded": true, "candidate_eligible": false,
		"candidate_eligible_without_budget": true,
		"recording_headstart_generations":   1,
		"part_request_sha256":               []string{"ws"},
	}
	report := replayLines(t, http, ws, finalSummary(2, 300, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 1 || combined.UnresolvedSelectionCount != 1 {
		t.Fatalf("combined policy should spend its only slot on HTTP: %+v", combined)
	}
}

func TestCurrentBaselineUsesActualWebSocketDispatchNotLateShadowObservation(t *testing.T) {
	shadowObservation := map[string]any{
		"event": "voxt_final_candidate_eligibility", "session_id": 3,
		"source": "websocket_preview", "sequence": 1,
		"candidate_eligible": false, "candidate_eligible_without_budget": false,
		"stop_to_eligibility_ms": 80,
	}
	dispatch := map[string]any{
		"event": "voxt_headstart_decision", "session_id": 3,
		"source": "websocket_preview", "sequence": 9,
		"candidate_plan_ready_ms": 60, "dispatch_ready_ms": 60,
		"preparation_succeeded": true, "candidate_eligible": true,
		"dispatched": true, "candidate_eligible_without_budget": true,
		"recording_headstart_generations": 1,
		"part_request_sha256":             []string{"p"},
	}
	requestDone := map[string]any{
		"event": "voxt_headstart_request_done", "session_id": 3,
		"request_plan_sha256": "p", "succeeded": true,
	}
	adopted := map[string]any{
		"event": "voxt_headstart_adoption", "session_id": 3,
		"request_plan_sha256": "p", "adopted": true, "outcome": "exact_match",
	}
	report := replayLines(t, shadowObservation, dispatch, requestDone, adopted, finalSummary(3, 200, "completed"))
	baseline := policyByName(t, report, "current_websocket_baseline", 0)
	if baseline.SelectedSessionCount != 1 || baseline.ActualAdoptionCount != 1 || baseline.ExactAdoptionCount != 1 {
		t.Fatalf("baseline must use actual dispatch and adoption events: %+v", baseline)
	}
}

func TestCombinedComparisonCountsActualBaselineAdoptions(t *testing.T) {
	http := httpCandidate(9, 90, 1, 1, 80, "http-part")
	ws := map[string]any{
		"event": "voxt_headstart_decision", "session_id": 9,
		"source": "websocket_preview", "sequence": 2,
		"candidate_plan_ready_ms": 100, "dispatch_ready_ms": 100,
		"preparation_succeeded": true, "candidate_eligible": true,
		"dispatched": true, "candidate_eligible_without_budget": true,
		"recording_headstart_generations": 1,
		"part_request_sha256":             []string{"ws-part"},
	}
	wsAdoption := map[string]any{
		"event": "voxt_headstart_adoption", "session_id": 9,
		"request_plan_sha256": "ws-part", "adopted": true, "outcome": "exact_match",
	}
	report := replayLines(t, http, candidateOutcome(9, 90, 1, "http_final", 1, true),
		ws, wsAdoption, finalSummary(9, 300, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.IncrementalCompatibleSelections != 0 || combined.PreservedCompatibleSelections != 1 {
		t.Fatalf("real WebSocket adoption must count as baseline-compatible: %+v", combined)
	}
}

func TestCombinedPolicyRetainsEarlierIncompatibleHTTPBudgetConsumption(t *testing.T) {
	http := httpCandidate(10, 100, 1, 1, 80, "http-part")
	ws := map[string]any{
		"event": "voxt_headstart_decision", "session_id": 10,
		"source": "websocket_preview", "sequence": 2,
		"candidate_plan_ready_ms": 120, "dispatch_ready_ms": -1,
		"preparation_succeeded": true, "candidate_eligible": false,
		"dispatched": false, "candidate_eligible_without_budget": true,
		"recording_headstart_generations": 1,
	}
	report := replayLines(t, http, candidateOutcome(10, 100, 1, "http_final", 1, false),
		ws, candidateOutcome(10, 0, 0, "websocket_preview", 2, true),
		finalSummary(10, 300, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 1 || combined.WastedIfDispatchedCount != 1 ||
		combined.OracleCompatibleSelectionCount != 0 {
		t.Fatalf("failed or incompatible first choice must consume the shared budget: %+v", combined)
	}
}

func TestCombinedPolicyKeepsRecordingBudgetAndFinalBoundary(t *testing.T) {
	covered := httpCandidate(11, 110, 1, 1, 20, "covered")
	covered["recording_headstart_generations"] = 3
	finalFirst := httpCandidate(12, 120, 1, 1, 50, "late")
	report := replayLines(t, covered, candidateOutcome(11, 110, 1, "http_final", 1, true),
		finalSummary(11, 100, "completed"), finalFirst,
		candidateOutcome(12, 120, 1, "http_final", 1, true), finalSummary(12, 50, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 0 {
		t.Fatalf("no counterfactual dispatch is allowed when the recording budget is full or Final is already ready: %+v", combined)
	}
}

func TestReplayReportsFinalDuringPreparationCancellationAndMissingOutcome(t *testing.T) {
	finalDuringPrep := httpCandidate(4, 40, 1, 1, 100)
	cancelled := httpCandidate(5, 50, 1, 1, 75)
	missingOutcome := httpCandidate(6, 60, 1, 1, 30)
	report := replayLines(t,
		finalDuringPrep, finalSummary(4, 100, "completed"),
		cancelled, finalSummary(5, 50, "cancelled"),
		missingOutcome, finalSummary(6, 200, "completed"),
	)
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 1 || combined.UnresolvedSelectionCount != 1 {
		t.Fatalf("only the missing-outcome session should dispatch and be unresolved: %+v", combined)
	}
	if report.UnresolvedOutcomeCount != 3 {
		t.Fatalf("missing retrospective outcomes were not counted: %d", report.UnresolvedOutcomeCount)
	}
}

func TestReplayKeepsRecordingTimeHeadStartBudgetAndFiltersWarmups(t *testing.T) {
	fullBudget := httpCandidate(7, 70, 1, 1, 20)
	fullBudget["recording_headstart_generations"] = 3
	line := map[string]any{"event": "final_candidate_benchmark_run", "warmup": true,
		"experiment_events": []map[string]any{httpCandidate(8, 80, 1, 1, 10), finalSummary(8, 100, "completed")}}
	encoded, err := json.Marshal(line)
	if err != nil {
		t.Fatal(err)
	}
	coldStartLine := map[string]any{"event": "final_candidate_benchmark_run", "cold_start": true,
		"experiment_events": []map[string]any{httpCandidate(9, 90, 1, 1, 10), finalSummary(9, 100, "completed")}}
	coldStartEncoded, err := json.Marshal(coldStartLine)
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Join([]string{
		"info omni: " + string(encoded),
		"info omni: " + string(coldStartEncoded),
		fmt.Sprintf("info omni: %s", mustJSON(t, fullBudget)),
		fmt.Sprintf("info omni: %s", mustJSON(t, finalSummary(7, 100, "completed"))),
	}, "\n")
	report, err := replay(strings.NewReader(input), []int{8}, []int{250})
	if err != nil {
		t.Fatal(err)
	}
	if report.SessionCount != 1 {
		t.Fatalf("warmup session was included: %+v", report)
	}
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 0 {
		t.Fatalf("recording-time HeadStart budget was ignored: %+v", combined)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}
