package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func replayLines(t *testing.T, events ...map[string]any) replayReport {
	t.Helper()
	return replayLinesWithBudget(t, true, events...)
}

func replayLinesWithBudget(t *testing.T, addBudget bool, events ...map[string]any) replayReport {
	t.Helper()
	var lines []string
	seenSessions := make(map[uint64]bool)
	for _, event := range events {
		if sessionID, ok := event["session_id"].(uint64); ok && sessionID != 0 {
			seenSessions[sessionID] = true
		}
		if sessionID, ok := event["session_id"].(int); ok && sessionID > 0 {
			seenSessions[uint64(sessionID)] = true
		}
	}
	if addBudget {
		for sessionID := range seenSessions {
			lines = append(lines, "info omni: "+mustJSON(t, initialBudget(sessionID, 1, 1)))
		}
	}
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

func initialBudget(sessionID uint64, generations, allowance int) map[string]any {
	return map[string]any{
		"event": "voxt_headstart_initial_budget", "session_id": sessionID,
		"recording_generation_count":        generations,
		"initial_late_generation_allowance": allowance,
		"late_generation_already_consumed":  allowance == 0,
		"effective_generation_limit":        3, "effective_multipart_limit": 4,
	}
}

func httpCandidate(sessionID uint64, requestID uint64, attempt int, sequence uint64, ready float64, parts ...string) map[string]any {
	return map[string]any{
		"event": "voxt_final_candidate_eligibility", "session_id": sessionID,
		"request_id": requestID, "attempt": attempt, "source": "http_final",
		"sequence": sequence, "stop_to_candidate_ms": ready - 5,
		"stop_to_eligibility_ms": ready, "candidate_eligible": true,
		"candidate_eligible_without_budget": true,
		"generated_token_count":             16, "part_request_sha256": parts,
		"fresh_request_part_count":        len(parts),
		"recording_headstart_generations": 1,
	}
}

func candidateOutcome(sessionID uint64, requestID uint64, attempt int, source string, sequence uint64, compatible bool) map[string]any {
	return map[string]any{
		"event": "voxt_final_candidate_outcome", "session_id": sessionID,
		"request_id": requestID, "attempt": attempt, "source": source,
		"sequence": sequence, "authoritative_final_available": true,
		"outcome_unresolved": false, "oracle_compatible": compatible,
		"oracle_incremental_compatible":  compatible,
		"fresh_compatible_cleanup_parts": boolInt(compatible),
		"exact_cleanup_plan_match":       compatible,
	}
}

func websocketDispatch(sessionID, requestID, sequence uint64, ready float64, candidateHash, planHash string) map[string]any {
	return map[string]any{
		"event": "voxt_headstart_decision", "session_id": sessionID,
		"source": "websocket_preview", "sequence": sequence,
		"candidate_sha256": candidateHash, "request_plan_sha256": planHash,
		"candidate_plan_ready_ms": ready, "dispatch_ready_ms": ready,
		"preparation_succeeded": true, "candidate_eligible": true,
		"dispatched": true, "candidate_eligible_without_budget": true,
		"recording_headstart_generations": 1,
		"request_ids":                     []uint64{requestID},
		"part_request_sha256":             []string{planHash}, "fresh_request_part_count": 1,
	}
}

func websocketDecisionOutcome(sessionID, sequence uint64, candidateHash, planHash string, compatible bool) map[string]any {
	return map[string]any{
		"event": "voxt_headstart_decision_outcome", "session_id": sessionID,
		"source": "websocket_preview", "sequence": sequence,
		"candidate_sha256": candidateHash, "candidate_plan_sha256": planHash,
		"outcome_unresolved": false, "oracle_compatible": compatible,
		"oracle_incremental_compatible":  compatible,
		"fresh_compatible_cleanup_parts": boolInt(compatible),
		"exact_cleanup_plan_match":       compatible,
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
		"candidate_sha256": "candidate-ws", "request_plan_sha256": "plan-ws",
		"candidate_plan_ready_ms": 60, "dispatch_ready_ms": 60,
		"preparation_succeeded": true, "candidate_eligible": true,
		"dispatched": true, "candidate_eligible_without_budget": true,
		"recording_headstart_generations": 1,
		"request_ids":                     []uint64{301},
		"part_request_sha256":             []string{"p"},
	}
	requestDone := map[string]any{
		"event": "voxt_headstart_request_done", "session_id": 3,
		"request_id": 301, "request_plan_sha256": "p", "succeeded": true,
	}
	adopted := map[string]any{
		"event": "voxt_headstart_adoption", "session_id": 3,
		"request_id": 301, "request_plan_sha256": "p", "adopted": true, "outcome": "exact_match",
	}
	report := replayLines(t, shadowObservation, dispatch, requestDone, adopted,
		websocketDecisionOutcome(3, 9, "candidate-ws", "plan-ws", true), finalSummary(3, 200, "completed"))
	baseline := policyByName(t, report, "current_websocket_baseline", 0)
	if baseline.SelectedSessionCount != 1 || baseline.ActualAdoptionCount != 1 || baseline.ExactAdoptionCount != 1 {
		t.Fatalf("baseline must use actual dispatch and adoption events: %+v", baseline)
	}
}

func TestCombinedComparisonUsesComparableCompatibilityAndReportsEarlierLead(t *testing.T) {
	http := httpCandidate(9, 90, 1, 1, 80, "http-part")
	ws := websocketDispatch(9, 901, 2, 100, "ws-candidate", "ws-plan")
	wsAdoption := map[string]any{
		"event": "voxt_headstart_adoption", "session_id": 9,
		"request_id": 901, "request_plan_sha256": "ws-part", "adopted": true, "outcome": "exact_match",
	}
	wsRequestDone := map[string]any{"event": "voxt_headstart_request_done", "session_id": 9,
		"request_id": 901, "succeeded": true}
	report := replayLines(t, http, candidateOutcome(9, 90, 1, "http_final", 1, true),
		ws, wsAdoption, wsRequestDone,
		websocketDecisionOutcome(9, 2, "ws-candidate", "ws-plan", true),
		finalSummary(9, 300, "completed"),
		initialBudget(9, 1, 1))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.IncrementalCompatibleSelections != 0 || combined.PreservedCompatibleSelections != 1 ||
		combined.EarlierCompatibleSelections != 1 || combined.PairedLeadSampleCount != 1 || combined.PairedLeadP50MS != 20 {
		t.Fatalf("same compatibility count must retain the earlier HTTP opportunity: %+v", combined)
	}
	baseline := policyByName(t, report, "current_websocket_baseline", 0)
	if baseline.ActualAdoptionCount != 1 || baseline.OracleCompatibleSelectionCount != 1 {
		t.Fatalf("actual adoption and oracle compatibility must remain separate: %+v", baseline)
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
	report := replayLines(t, initialBudget(11, 3, 0), covered, candidateOutcome(11, 110, 1, "http_final", 1, true),
		finalSummary(11, 100, "completed"), initialBudget(12, 0, 1), finalFirst,
		candidateOutcome(12, 120, 1, "http_final", 1, true), finalSummary(12, 50, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 0 {
		t.Fatalf("no counterfactual dispatch is allowed when the recording budget is full or Final is already ready: %+v", combined)
	}
}

func TestInitialBudgetControlsZeroTwoAndThreeRecordingGenerations(t *testing.T) {
	zero := httpCandidate(31, 310, 1, 1, 20, "zero")
	two := httpCandidate(32, 320, 1, 1, 20, "two")
	three := httpCandidate(33, 330, 1, 1, 20, "three")
	report := replayLines(t,
		initialBudget(31, 0, 1), zero, candidateOutcome(31, 310, 1, "http_final", 1, true), finalSummary(31, 200, "completed"),
		initialBudget(32, 2, 1), two, candidateOutcome(32, 320, 1, "http_final", 1, true), finalSummary(32, 200, "completed"),
		initialBudget(33, 3, 0), three, candidateOutcome(33, 330, 1, "http_final", 1, true), finalSummary(33, 200, "completed"),
	)
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 2 || combined.UnresolvedCounterfactualCount != 0 {
		t.Fatalf("replay did not use the per-session initial allowance: %+v", combined)
	}
}

func TestLateWebSocketDispatchDoesNotRetroactivelyEraseInitialAllowance(t *testing.T) {
	ws := websocketDispatch(34, 341, 1, 120, "ws-34", "ws-plan-34")
	lateHTTP := httpCandidate(34, 342, 1, 2, 160, "http-part")
	lateHTTP["recording_headstart_generations"] = 3 // includes the real late generation
	lateHTTP["counterfactual_state_unresolved"] = true
	report := replayLines(t, initialBudget(34, 2, 1), ws,
		websocketDecisionOutcome(34, 1, "ws-34", "ws-plan-34", true),
		lateHTTP, candidateOutcome(34, 342, 1, "http_final", 2, true), finalSummary(34, 300, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 1 || combined.OracleCompatibleSelectionCount != 1 || combined.UnresolvedCounterfactualCount == 0 {
		t.Fatalf("later mutable state should be unresolved while the initial WebSocket slot remains visible: %+v", combined)
	}
}

func TestMissingInitialBudgetIsUnresolved(t *testing.T) {
	candidate := httpCandidate(35, 350, 1, 1, 20, "missing-budget")
	report := replayLinesWithBudget(t, false, candidate,
		candidateOutcome(35, 350, 1, "http_final", 1, true), finalSummary(35, 200, "completed"))
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.SelectedSessionCount != 0 || combined.UnresolvedCounterfactualCount == 0 || combined.UnresolvedPairCount == 0 {
		t.Fatalf("missing initial budget must not be inferred from later events: %+v", combined)
	}
}

func TestComparisonKeepsMissingFinalAndMissingBaselineOutcomeUnresolved(t *testing.T) {
	withoutFinal := httpCandidate(38, 380, 1, 1, 20, "no-final")
	withoutFinal["counterfactual_state_unresolved"] = true
	missingBaselineOutcome := websocketDispatch(39, 391, 1, 40, "ws-39", "ws-plan-39")
	missingBaselineOutcome["outcome_unresolved"] = true
	report := replayLines(t,
		withoutFinal, candidateOutcome(38, 380, 1, "http_final", 1, true),
		missingBaselineOutcome, finalSummary(39, 100, "completed"),
	)
	combined := policyByName(t, report, "combined_first_eligible", 0)
	if combined.UnresolvedPairCount != 2 || combined.IncrementalCompatibleSelections != 0 ||
		combined.DisplacedCompatibleSelections != 0 {
		t.Fatalf("incomplete baseline or final evidence was scored as zero: %+v", combined)
	}
}

func TestMultipartReuseDoesNotCountAsFreshSpeculation(t *testing.T) {
	candidate := httpCandidate(36, 360, 1, 1, 20, "reuse", "fresh")
	candidate["existing_equivalent_requests"] = 1
	candidate["fresh_request_part_count"] = 1
	outcome := candidateOutcome(36, 360, 1, "http_final", 1, true)
	outcome["fresh_compatible_cleanup_parts"] = 1
	report := replayLines(t, initialBudget(36, 2, 1), candidate, outcome, finalSummary(36, 200, "completed"))
	selected := policyByName(t, report, "http_diagnostic", 0)
	if selected.SelectedSessionCount != 1 || selected.CompatibleFreshRequestParts != 1 || selected.ExistingRecordingCoverageParts != 1 {
		t.Fatalf("reused multipart work was counted as a new request: %+v", selected)
	}
}

func TestComparablePairPreservesNegativeLeadWhenAlternativeIsLater(t *testing.T) {
	trace := map[string]*sessionTrace{
		"s": {
			initialBudgetKnown: true, lateAllowance: 1, generationLimit: 3, multipartLimit: 4,
			finalAvailableKnown: true, finalAvailableMS: 500,
			candidates: []candidate{
				{production: true, eligibility: logEvent{Source: "websocket_preview", Dispatched: true, DispatchReadyMS: 300, PlanReadyMS: 300}, outcome: logEvent{OracleCompatible: true}, hasOutcome: true},
				{eligibility: logEvent{Source: "http_final", StopToEligibilityMS: 350, CandidateEligible: true, PartPlanHashes: []string{"http-37"}, FreshRequestPartCount: 1}, outcome: logEvent{OracleCompatible: true}, hasOutcome: true},
			},
		},
	}
	current := policy{name: "combined_http_only", combined: true, match: func(item candidate) bool { return item.eligibility.Source == "http_final" }}
	result := policyResult{}
	compareToBaseline(&result, current, trace)
	if result.LaterCompatibleSelections != 1 || result.PairedLeadSampleCount != 1 || result.PairedLeadP50MS != -50 {
		t.Fatalf("negative paired lead was lost: %+v", result)
	}
}

func TestBenchmarkArmsScenariosAndAllRunDenominatorsStaySeparate(t *testing.T) {
	rows := []map[string]any{
		{"event": "final_candidate_benchmark_run", "run_id": "off-warm", "arm": "off", "scenario": "en-1", "round": -1, "warmup": true, "status": "done"},
		{"event": "final_candidate_benchmark_run", "run_id": "off-fail", "arm": "off", "scenario": "en-1", "round": 1, "status": "failed", "problem": "asr"},
		{
			"event": "final_candidate_benchmark_run", "run_id": "shadow-one", "arm": "shadow", "scenario": "en-1", "round": 1,
			"audio_sha256": "fixture-en", "language_hint": "English", "language_mode": "explicit", "status": "done",
			"experiment_events": []map[string]any{
				initialBudget(101, 1, 1), httpCandidate(101, 1010, 1, 1, 40, "part"),
				candidateOutcome(101, 1010, 1, "http_final", 1, true), finalSummary(101, 100, "completed"),
			},
		},
		{"event": "final_candidate_benchmark_run", "run_id": "dense-cold", "arm": "shadow_dense", "scenario": "zh-1", "round": -1, "cold_start": true, "status": "done"},
		{
			"event": "final_candidate_benchmark_run", "run_id": "dense-empty", "arm": "shadow_dense", "scenario": "zh-1", "round": 1,
			"language_hint": "Chinese", "language_mode": "explicit", "status": "done",
			"experiment_events": []map[string]any{initialBudget(103, 1, 1), finalSummary(103, 100, "successful_final_no_candidates")},
		},
	}
	var lines []string
	for _, row := range rows {
		lines = append(lines, mustJSON(t, row))
	}
	report, err := replay(strings.NewReader(strings.Join(lines, "\n")), []int{8}, []int{250})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Arms) != 3 || report.SessionCount != 5 {
		t.Fatalf("run identities or arms were lost: %+v", report)
	}
	arm := func(name string) armReport {
		t.Helper()
		for _, current := range report.Arms {
			if current.Arm == name {
				return current
			}
		}
		t.Fatalf("missing arm %q", name)
		return armReport{}
	}
	off := arm("off")
	shadow := arm("shadow")
	dense := arm("shadow_dense")
	if off.SessionCount != 2 || off.WarmupSessionCount != 1 || off.SteadySessionCount != 1 ||
		policyByName(t, replayReport{Policies: off.Policies}, "combined_first_eligible", 0).FailedSessionCount != 1 {
		t.Fatalf("OFF denominator or warmup classification is wrong: %+v", off)
	}
	if shadow.SessionCount != 1 || shadow.Scenarios[0].Scenario != "en-1" || shadow.Runs[0].AudioSHA256 != "fixture-en" {
		t.Fatalf("shadow scenario dimensions were not preserved: %+v", shadow)
	}
	densePolicy := policyByName(t, replayReport{Policies: dense.Policies}, "combined_first_eligible", 0)
	if dense.SessionCount != 2 || dense.ColdStartCount != 1 || dense.SteadySessionCount != 1 || densePolicy.SessionCount != 1 || densePolicy.NoCandidateSessionCount != 1 {
		t.Fatalf("dense zero-candidate and cold-start denominator is wrong: %+v %+v", dense, densePolicy)
	}
	filtered, err := replayForArm(strings.NewReader(strings.Join(lines, "\n")), []int{8}, []int{250}, "shadow")
	if err != nil {
		t.Fatal(err)
	}
	if filtered.SelectedArm != "shadow" || filtered.SessionCount != 1 || len(filtered.Arms) != 1 || filtered.Arms[0].Arm != "shadow" {
		t.Fatalf("arm filter mixed runs: %+v", filtered)
	}
}

func TestReplayReportsFinalDuringPreparationCancellationAndMissingOutcome(t *testing.T) {
	finalDuringPrep := httpCandidate(4, 40, 1, 1, 100, "final-prep")
	cancelled := httpCandidate(5, 50, 1, 1, 75, "cancelled")
	missingOutcome := httpCandidate(6, 60, 1, 1, 30, "missing-outcome")
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
		fmt.Sprintf("info omni: %s", mustJSON(t, initialBudget(7, 3, 0))),
		fmt.Sprintf("info omni: %s", mustJSON(t, fullBudget)),
		fmt.Sprintf("info omni: %s", mustJSON(t, finalSummary(7, 100, "completed"))),
	}, "\n")
	report, err := replay(strings.NewReader(input), []int{8}, []int{250})
	if err != nil {
		t.Fatal(err)
	}
	if report.SessionCount != 3 || len(report.Arms) != 1 || report.Arms[0].SteadySessionCount != 1 ||
		report.Arms[0].WarmupSessionCount != 1 || report.Arms[0].ColdStartCount != 1 {
		t.Fatalf("warmup/cold sessions were lost or pooled into steady metrics: %+v", report)
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
