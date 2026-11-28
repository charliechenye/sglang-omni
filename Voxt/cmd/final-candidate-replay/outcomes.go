package main

func resolveActualWebSocketOutcome(trace *sessionTrace, current *candidate) {
	reusedRequestIDs := uniqueRequestIDs(current.eligibility.ReusedRequestIDs)
	for _, requestID := range reusedRequestIDs {
		if adoption, ok := trace.adoptions[requestID]; ok && adoption.Adopted {
			current.reusedRecordingAdoptionCount++
		}
	}

	freshRequestIDs := current.eligibility.FreshRequestIDs
	if freshRequestIDs == nil && current.eligibility.GenerationID == 0 {
		freshRequestIDs = current.eligibility.RequestIDs
	}
	freshRequestIDs = uniqueRequestIDs(freshRequestIDs)
	current.freshRequestCount = len(freshRequestIDs)
	if len(freshRequestIDs) == 0 {
		current.actualOutcomeStatus = "unresolved_expected_fresh_requests_missing"
		return
	}

	resolvedRequestCount := 0
	for _, requestID := range freshRequestIDs {
		if adoption, ok := trace.adoptions[requestID]; ok && adoption.Adopted {
			current.actualAdoptionCount++
			if adoption.Outcome == "exact_match" {
				current.exactAdoptionCount++
			}
			if adoption.Outcome == "near_match_repair_adoption" {
				current.nearRepairAdoptionCount++
			}
		}
		if request, ok := trace.requestResultsByID[requestID]; ok {
			resolvedRequestCount++
			if !request.Succeeded && !request.Cancelled {
				current.failedRequestCount++
			}
			if request.Cancelled {
				current.cancelledRequestCount++
			}
		} else if requestID == 0 {
			current.partiallyResolved = true
		} else {
		}
		if requestStart, ok := trace.requestStartsByID[requestID]; ok && requestStart.RequestStartValid &&
			(requestStart.RequestStartMS < current.actualRequestStartMS || !current.actualRequestStartKnown) {
			current.actualRequestStartMS = requestStart.RequestStartMS
			current.actualRequestStartKnown = true
		}
	}
	if resolvedRequestCount != len(freshRequestIDs) {
		current.partiallyResolved = resolvedRequestCount > 0
		if current.partiallyResolved {
			current.actualOutcomeStatus = "partially_resolved"
		} else {
			current.actualOutcomeStatus = "unresolved_request_results_missing"
		}
		return
	}
	if current.actualAdoptionCount > 0 {
		current.hasActualOutcome = true
		current.actualOutcomeStatus = "resolved_fresh_adoption"
		if trace.terminalDecisionComplete && current.actualAdoptionCount < len(freshRequestIDs) {
			current.partialAdoption = true
			current.unadoptedFreshRequestCount = len(freshRequestIDs) - current.actualAdoptionCount
		}
	} else if trace.terminalDecisionComplete {
		current.hasActualOutcome = true
		current.actualOutcomeStatus = "resolved_fresh_miss"
		current.unadoptedFreshRequestCount = len(freshRequestIDs)
	} else {
		current.actualOutcomeStatus = "unresolved_terminal_decision_missing"
	}
}

func uniqueRequestIDs(requestIDs []uint64) []uint64 {
	seen := make(map[uint64]bool, len(requestIDs))
	unique := make([]uint64, 0, len(requestIDs))
	for _, requestID := range requestIDs {
		if seen[requestID] {
			continue
		}
		seen[requestID] = true
		unique = append(unique, requestID)
	}
	return unique
}

func eventKey(session string, event logEvent) candidateKey {
	return candidateKey{session: session, request: event.RequestID, attempt: event.Attempt, source: event.Source, sequence: event.Sequence}
}

func eventKeyForCandidate(sessionKey string, current *candidate) candidateKey {
	return eventKey(sessionKey, current.eligibility)
}

func candidateReadyMS(current candidate) float64 {
	if current.production {
		if current.eligibility.DispatchReadyMS >= 0 {
			return current.eligibility.DispatchReadyMS
		}
		return current.eligibility.PlanReadyMS
	}
	return current.eligibility.StopToEligibilityMS
}

func candidateEligibleWithoutBudget(current candidate) bool {
	if current.eligibility.EligibleWithoutBudget != nil {
		return *current.eligibility.EligibleWithoutBudget
	}
	if current.production {
		return current.eligibility.PreparationSucceeded && current.eligibility.LLMReady && current.eligibility.CleanupEligible
	}
	return current.eligibility.CandidateEligible
}
