package main

func resolveActualWebSocketOutcome(trace *sessionTrace, current *candidate) {
	seenRequestIDs := make(map[uint64]bool)
	seenPartHashes := make(map[string]bool)
	recordAdoption := func(adoption logEvent, partHash string) {
		if adoption.RequestID != 0 {
			if seenRequestIDs[adoption.RequestID] {
				return
			}
			seenRequestIDs[adoption.RequestID] = true
		} else {
			if seenPartHashes[partHash] {
				return
			}
			seenPartHashes[partHash] = true
		}
		current.hasActualOutcome = true
		if !adoption.Adopted {
			return
		}
		current.actualAdoptionCount++
		if adoption.Outcome == "exact_match" {
			current.exactAdoptionCount++
		}
		if adoption.Outcome == "near_match_repair_adoption" {
			current.nearRepairAdoptionCount++
		}
		if !current.hasOutcome {
			current.outcome = logEvent{
				OracleCompatible: true, OracleIncrementalCompatible: true,
				ExactCleanupPlanMatch:     adoption.Outcome == "exact_match",
				NearCleanupPlanCompatible: adoption.Outcome == "near_match_repair_adoption",
				CleanupRepairEligible:     adoption.Outcome == "near_match_repair_adoption",
			}
			current.hasOutcome = true
		}
	}
	requestIDs := append([]uint64(nil), current.eligibility.RequestIDs...)
	if len(requestIDs) == 0 {
		for _, partHash := range current.eligibility.PartPlanHashes {
			requestID := uint64(0)
			if requestStart, ok := trace.requestStarts[partHash]; ok {
				requestID = requestStart.RequestID
			} else if requestDone, ok := trace.requestResults[partHash]; ok {
				requestID = requestDone.RequestID
			}
			if requestID != 0 {
				requestIDs = append(requestIDs, requestID)
			}
		}
	}
	for _, requestID := range requestIDs {
		if requestID == 0 || seenRequestIDs[requestID] {
			continue
		}
		if adoption, ok := trace.adoptions[requestID]; ok {
			recordAdoption(adoption, "")
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
	for _, partHash := range current.eligibility.PartPlanHashes {
		if adoption, ok := trace.adoptionsByPlanHash[partHash]; ok {
			recordAdoption(adoption, partHash)
		}
		if request, ok := trace.requestResults[partHash]; ok {
			if request.RequestID == 0 || !seenRequestIDs[request.RequestID] {
				current.hasActualOutcome = true
				if !request.Succeeded {
					current.failedRequestCount++
				}
			}
		}
		if requestStart, ok := trace.requestStarts[partHash]; ok && requestStart.RequestStartValid &&
			(requestStart.RequestStartMS < current.actualRequestStartMS || !current.actualRequestStartKnown) {
			current.actualRequestStartMS = requestStart.RequestStartMS
			current.actualRequestStartKnown = true
		}
	}
	partCount := max(len(requestIDs), len(current.eligibility.PartPlanHashes))
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
