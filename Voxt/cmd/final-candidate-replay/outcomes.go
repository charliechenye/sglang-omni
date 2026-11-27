package main

func resolveActualWebSocketOutcome(trace *sessionTrace, current *candidate) {
	requestIDs := append([]uint64(nil), current.eligibility.RequestIDs...)
	seen := make(map[uint64]bool, len(requestIDs))
	for _, requestID := range requestIDs {
		if requestID == 0 || seen[requestID] {
			continue
		}
		seen[requestID] = true
		if adoption, ok := trace.adoptions[requestID]; ok {
			current.hasActualOutcome = true
			if adoption.Adopted {
				current.actualAdoptionCount++
				if adoption.Outcome == "exact_match" {
					current.exactAdoptionCount++
				}
				if adoption.Outcome == "near_match_repair_adoption" {
					current.nearRepairAdoptionCount++
				}
			}
		}
		if request, ok := trace.requestResultsByID[requestID]; ok {
			current.hasActualOutcome = true
			if !request.Succeeded {
				current.failedRequestCount++
			}
		}
		if requestStart, ok := trace.requestStartsByID[requestID]; ok && requestStart.RequestStartValid &&
			(requestStart.RequestStartMS < current.actualRequestStartMS || !current.actualRequestStartKnown) {
			current.actualRequestStartMS = requestStart.RequestStartMS
			current.actualRequestStartKnown = true
		}
	}
	partCount := len(requestIDs)
	if current.actualAdoptionCount > 0 && current.actualAdoptionCount < partCount {
		current.partialAdoption = true
	}
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
