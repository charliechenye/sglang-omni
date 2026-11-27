package main

import (
	"sort"
	"strconv"
)

func buildPolicies(tokenThresholds, timeThresholds []int) []policy {
	websocket := func(current candidate) bool {
		return current.production &&
			(current.eligibility.Source == "websocket_preview" || current.eligibility.Source == "websocket_candidate")
	}
	httpCandidate := func(current candidate) bool { return current.eligibility.Source == "http_final" }
	policies := []policy{
		{name: "current_websocket_baseline", match: func(current candidate) bool {
			return current.production && websocket(current) && current.eligibility.Dispatched
		}},
		{name: "http_diagnostic", match: httpCandidate},
		{name: "first_eligible_http", match: httpCandidate},
		{name: "latest_preview_match", match: func(current candidate) bool { return httpCandidate(current) && current.eligibility.LatestPreviewMatch }},
		{name: "meaningful_boundary", match: func(current candidate) bool {
			return httpCandidate(current) && current.eligibility.MeaningfulTextBoundary
		}},
		{name: "extends_latest_hypothesis", match: func(current candidate) bool {
			return httpCandidate(current) && current.eligibility.ExtendsLatestHypothesis
		}},
		{name: "websocket_control", match: websocket},
		{name: "combined_first_eligible", combined: true, match: func(current candidate) bool { return websocket(current) || httpCandidate(current) }},
		{name: "combined_preview_match", combined: true, match: func(current candidate) bool {
			return websocket(current) || httpCandidate(current) && current.eligibility.LatestPreviewMatch
		}},
		{name: "combined_boundary", combined: true, match: func(current candidate) bool {
			return websocket(current) || httpCandidate(current) && current.eligibility.MeaningfulTextBoundary
		}},
	}
	for _, threshold := range tokenThresholds {
		threshold := threshold
		policies = append(policies,
			policy{name: "token_cadence", threshold: float64(threshold), match: func(current candidate) bool {
				return httpCandidate(current) && current.eligibility.GeneratedTokenCount >= threshold
			}},
			policy{name: "combined_cadence_tokens", threshold: float64(threshold), combined: true, match: func(current candidate) bool {
				return websocket(current) || httpCandidate(current) && current.eligibility.GeneratedTokenCount >= threshold
			}},
		)
	}
	for _, threshold := range timeThresholds {
		threshold := threshold
		policies = append(policies,
			policy{name: "time_cadence_ms", threshold: float64(threshold), match: func(current candidate) bool {
				return httpCandidate(current) && current.eligibility.StopToCandidateMS >= float64(threshold)
			}},
			policy{name: "combined_cadence_time_ms", threshold: float64(threshold), combined: true, match: func(current candidate) bool {
				return websocket(current) || httpCandidate(current) && current.eligibility.StopToCandidateMS >= float64(threshold)
			}},
		)
	}
	return policies
}

func evaluatePolicy(current policy, sessions map[string]*sessionTrace) policyResult {
	result := policyResult{Name: current.name, Threshold: current.threshold}
	var arrivalLeads, dispatchLeads, actualRequestStarts []float64
	for _, trace := range sessions {
		selected := selectCandidates(current, trace, &result)
		if len(selected) == 0 {
			continue
		}
		result.CandidateSessionCount++
		result.SelectedSessionCount += len(selected)
		for _, chosen := range selected {
			if current.name == "current_websocket_baseline" {
				if !chosen.hasActualOutcome {
					result.UnresolvedSelectionCount++
					continue
				}
				result.ResolvedSelectionCount++
				result.ActualAdoptionCount += boolInt(chosen.actualAdoptionCount > 0)
				result.ExactAdoptionCount += chosen.exactAdoptionCount
				result.NearRepairAdoptionCount += chosen.nearRepairAdoptionCount
				result.PartialMultipartAdoptionCount += boolInt(chosen.partialAdoption)
				result.FailedSpeculativeRequestCount += chosen.failedRequestCount
				if chosen.actualAdoptionCount == 0 {
					result.WastedIfDispatchedCount++
				}
				if chosen.actualRequestStartKnown {
					actualRequestStarts = append(actualRequestStarts, chosen.actualRequestStartMS)
				}
				if chosen.hasOutcome && !chosen.outcome.OutcomeUnresolved {
					if chosen.outcome.OracleCompatible {
						result.OracleCompatibleSelectionCount++
					}
					if chosen.outcome.ExactTranscriptMatch {
						result.ExactTranscriptMatchCount++
					}
					if chosen.outcome.ExactCleanupPlanMatch {
						result.ExactCleanupPlanMatchCount++
					}
					if chosen.outcome.NearCleanupPlanCompatible && chosen.outcome.CleanupRepairEligible {
						result.NearCleanupRepairCount++
					}
				}
				if trace.finalAvailableKnown {
					appendLeadSamples(&arrivalLeads, &dispatchLeads, trace.finalAvailableMS, chosen)
				}
				continue
			}
			if !chosen.hasOutcome || chosen.outcome.OutcomeUnresolved {
				result.UnresolvedSelectionCount++
				continue
			}
			result.ResolvedSelectionCount++
			if chosen.production {
				result.ActualAdoptionCount += boolInt(chosen.actualAdoptionCount > 0)
				result.ExactAdoptionCount += chosen.exactAdoptionCount
				result.NearRepairAdoptionCount += chosen.nearRepairAdoptionCount
				result.PartialMultipartAdoptionCount += boolInt(chosen.partialAdoption)
				result.FailedSpeculativeRequestCount += chosen.failedRequestCount
				if chosen.actualRequestStartKnown {
					actualRequestStarts = append(actualRequestStarts, chosen.actualRequestStartMS)
				}
			}
			outcome := chosen.outcome
			if outcome.OracleCompatible {
				result.OracleCompatibleSelectionCount++
			}
			if outcome.ExactTranscriptMatch {
				result.ExactTranscriptMatchCount++
			}
			if outcome.ExactCleanupPlanMatch {
				result.ExactCleanupPlanMatchCount++
			}
			if outcome.NearCleanupPlanCompatible && outcome.CleanupRepairEligible {
				result.NearCleanupRepairCount++
			}
			if !outcome.OracleIncrementalCompatible {
				result.WastedIfDispatchedCount++
			}
			if trace.finalAvailableKnown {
				appendLeadSamples(&arrivalLeads, &dispatchLeads, trace.finalAvailableMS, chosen)
			}
		}
	}
	result.LeadSampleCount = len(dispatchLeads)
	result.ArrivalLeadP50MS = percentile(arrivalLeads, 0.50)
	result.ArrivalLeadP95MS = percentile(arrivalLeads, 0.95)
	result.DispatchReadyLeadP50MS = percentile(dispatchLeads, 0.50)
	result.DispatchReadyLeadP95MS = percentile(dispatchLeads, 0.95)
	result.ActualRequestStartSampleCount = len(actualRequestStarts)
	result.ActualRequestStartP50MS = percentile(actualRequestStarts, 0.50)
	result.ActualRequestStartP95MS = percentile(actualRequestStarts, 0.95)
	return result
}

func appendLeadSamples(arrivalLeads, dispatchLeads *[]float64, finalAvailableMS float64, chosen candidate) {
	if chosen.outcome.CandidateArrivalValid {
		lead := chosen.outcome.ArrivalLeadMS
		if lead > 0 {
			*arrivalLeads = append(*arrivalLeads, lead)
		}
	}
	lead := finalAvailableMS - candidateReadyMS(chosen)
	if lead > 0 {
		*dispatchLeads = append(*dispatchLeads, lead)
	}
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func selectCandidates(current policy, trace *sessionTrace, result *policyResult) []candidate {
	for _, candidate := range trace.candidates {
		if !current.match(candidate) {
			continue
		}
		if !candidateEligibleWithoutBudget(candidate) {
			continue
		}
		result.EligibleCandidateCount++
	}
	if current.name == "current_websocket_baseline" {
		var selected []candidate
		for _, candidate := range trace.candidates {
			if current.match(candidate) && candidate.eligibility.Dispatched {
				selected = append(selected, candidate)
			}
		}
		return selected
	}
	if current.name == "websocket_control" {
		current.match = func(candidate candidate) bool {
			return candidate.production &&
				(candidate.eligibility.Source == "websocket_preview" || candidate.eligibility.Source == "websocket_candidate")
		}
	}
	remaining := 0
	if trace.recordingGensKnown {
		if trace.recordingGenerations < 3 {
			remaining = 1
		}
	} else {
		result.UnresolvedCounterfactualCount++
		return nil
	}
	if remaining == 0 {
		return nil
	}
	terminalAt := trace.finalAvailableMS
	terminalKnown := trace.finalAvailableKnown
	if trace.cancellationKnown && (!terminalKnown || trace.cancelledAtMS < terminalAt) {
		terminalAt = trace.cancelledAtMS
		terminalKnown = true
	}
	selected := make([]candidate, 0, 1)
	for _, candidate := range trace.candidates {
		if !current.match(candidate) || !candidateEligibleWithoutBudget(candidate) {
			continue
		}
		readyAt := candidateReadyMS(candidate)
		if readyAt < 0 || !terminalKnown {
			result.UnresolvedCounterfactualCount++
			continue
		}
		if terminalAt <= readyAt {
			continue
		}
		selected = append(selected, candidate)
		remaining-- // A failed attempt retains this generation budget.
		if remaining == 0 {
			break
		}
	}
	return selected
}

func compareToBaseline(result *policyResult, current policy, sessions map[string]*sessionTrace) {
	for _, trace := range sessions {
		baseline := selectCandidates(policy{name: "current_websocket_baseline", match: func(candidate candidate) bool { return candidate.production && candidate.eligibility.Dispatched }}, trace, &policyResult{})
		alternative := selectCandidates(current, trace, &policyResult{})
		baseCompatible, baseWasted := scoreCounts(baseline, true)
		altCompatible, altWasted := scoreCounts(alternative, false)
		result.IncrementalCompatibleSelections += max(0, altCompatible-baseCompatible)
		result.PreservedCompatibleSelections += min(altCompatible, baseCompatible)
		result.DisplacedCompatibleSelections += max(0, baseCompatible-altCompatible)
		result.NewWastedGenerations += max(0, altWasted-baseWasted)
		result.AvoidedWastedGenerations += max(0, baseWasted-altWasted)
		if altCompatible > baseCompatible {
			for _, item := range alternative {
				if item.hasOutcome && item.outcome.OracleIncrementalCompatible && trace.finalAvailableKnown {
					lead := trace.finalAvailableMS - candidateReadyMS(item)
					if lead > 0 {
						result.IncrementalDispatchReadyLeadMS += lead
						break
					}
				}
			}
		}
	}
}

func scoreCounts(items []candidate, actualBaseline bool) (compatible, wasted int) {
	for _, item := range items {
		if actualBaseline {
			if !item.hasActualOutcome {
				continue
			}
			if item.actualAdoptionCount > 0 {
				compatible++
			} else {
				wasted++
			}
			continue
		}
		if !item.hasOutcome || item.outcome.OutcomeUnresolved {
			continue
		}
		if item.outcome.OracleIncrementalCompatible {
			compatible++
		} else {
			wasted++
		}
	}
	return compatible, wasted
}

func percentile(values []float64, fraction float64) float64 {
	if len(values) == 0 {
		return -1
	}
	sort.Float64s(values)
	index := int(float64(len(values))*fraction + 0.999999)
	return values[min(len(values)-1, max(0, index-1))]
}

func formatThreshold(value float64) string {
	if value == 0 {
		return "0"
	}
	return strconv.FormatFloat(value, 'f', -1, 64)
}
