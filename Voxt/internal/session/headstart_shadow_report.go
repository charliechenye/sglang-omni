package session

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"
	"unicode/utf8"
)

func (r *run) recordShadowOutcomesAfterDelivery() {
	r.shadowMu.Lock()
	observations := append([]shadowObservation(nil), r.shadowObservations...)
	capped := r.shadowObservationCap
	dropped := r.shadowObservationDropped
	end := r.shadowFinal
	r.shadowMu.Unlock()
	if end != nil {
		r.recordShadowOutcomes(*end, observations, capped, dropped,
			r.shadowCandidateCoalesced.Load(), len(r.shadowCandidates))
	}
}

func (r *run) latestCleanupPlan() (headPlan, bool) {
	r.cleanupPlanMu.Lock()
	defer r.cleanupPlanMu.Unlock()
	if len(r.cleanupPlans) == 0 {
		return headPlan{}, false
	}
	plan := r.cleanupPlans[len(r.cleanupPlans)-1]
	plan.parts = append([]headPart(nil), plan.parts...)
	return plan, true
}

func (r *run) recordShadowOutcomes(end streamEnd, observations []shadowObservation, capped bool,
	dropped, coalesced uint64, pending int) {
	if len(observations) == 0 && end.candidate.SessionID == 0 {
		return
	}
	stopAt := r.monotonicStoppedAt()
	stopOffsetMS := func(at time.Time) float64 {
		if stopAt.IsZero() || at.IsZero() {
			return -1
		}
		return float64(at.Sub(stopAt).Microseconds()) / 1000
	}
	for _, observation := range observations {
		candidate := observation.candidate
		metadata := candidate.metadata
		candidateHash := sha256.Sum256([]byte(candidate.text))
		candidatePlanHash := r.head.planHash(observation.plan)
		partPlanHashes := make([]string, 0, len(observation.plan.parts))
		for _, part := range observation.plan.parts {
			partPlanHashes = append(partPlanHashes, r.head.partPlanHash(part))
		}
		stopToHTTPRequestStartMS := float64(-1)
		if metadata.Source == "http_final" && !metadata.ReceivedAt.IsZero() {
			stopToHTTPRequestStartMS = stopOffsetMS(metadata.ReceivedAt) - metadata.RequestElapsedMS + metadata.HTTPRequestStartOffsetMS
		}
		r.logShadowEvent(map[string]any{
			"event": "voxt_final_candidate_eligibility", "run_id": r.id,
			"session_id": metadata.SessionID, "request_id": metadata.RequestID,
			"attempt": metadata.Attempt, "source": metadata.Source, "sequence": metadata.Sequence,
			"generated_token_count":             metadata.GeneratedTokens,
			"native_elapsed_ms":                 metadata.NativeElapsedMS,
			"first_transcript_token_elapsed_ms": metadata.FirstTranscriptTokenMS,
			"snapshot_build_ms":                 metadata.SnapshotBuildMS,
			"published_candidate_count":         metadata.PublishedCandidateCount,
			"snapshot_build_ms_total":           metadata.SnapshotBuildMSTotal,
			"request_elapsed_ms":                metadata.RequestElapsedMS,
			"http_request_start_offset_ms":      metadata.HTTPRequestStartOffsetMS,
			"stop_to_http_request_start_ms":     stopToHTTPRequestStartMS,
			"candidate_arrived_ms":              stopOffsetMS(metadata.ReceivedAt),
			"stop_to_candidate_ms":              stopOffsetMS(metadata.ReceivedAt),
			"candidate_processing_started_ms":   stopOffsetMS(observation.processingAt),
			"candidate_plan_ready_ms":           stopOffsetMS(observation.planReadyAt),
			"candidate_dispatch_ready_ms":       stopOffsetMS(observation.dispatchReadyAt),
			"stop_to_eligibility_ms":            stopOffsetMS(observation.dispatchReadyAt),
			"candidate_processing_ms":           float64(observation.planReadyAt.Sub(observation.processingAt).Microseconds()) / 1000,
			"candidate_gate_ms":                 float64(observation.dispatchReadyAt.Sub(observation.planReadyAt).Microseconds()) / 1000,
			"transcript_sha256":                 hex.EncodeToString(candidateHash[:]),
			"transcript_runes":                  utf8.RuneCountInString(candidate.text),
			"latest_preview_match":              observation.previewMatch,
			"extends_latest_hypothesis":         observation.extendsPreview,
			"meaningful_text_boundary":          observation.textBoundary,
			"plan_sha256":                       candidatePlanHash,
			"candidate_plan_sha256":             candidatePlanHash,
			"plan_reason":                       observation.reason,
			"part_request_sha256":               partPlanHashes,
			"existing_equivalent_requests":      observation.existingRequests,
			"late_generation_budget_available":  observation.budgetAvailable,
			"llm_readiness_open":                observation.llmReady,
			"cleanup_eligible":                  observation.cleanupEligible,
			"recording_headstart_generations":   observation.recordingGenerations,
			"candidate_eligible_without_budget": observation.reason == "" && observation.plan.hasFresh() &&
				observation.llmReady && observation.cleanupEligible,
			"candidate_eligible": observation.eligible,
		})
	}
	if end.err != nil || end.finalAvailableAt.IsZero() {
		finalStatus := "failed"
		if r.cancelled.Load() || r.ctx.Err() != nil {
			finalStatus = "cancelled"
		}
		for _, observation := range observations {
			metadata := observation.candidate.metadata
			retired := metadata.Source == "http_final" &&
				metadata.SessionID == end.candidate.SessionID &&
				metadata.Attempt == end.candidate.Attempt
			record := map[string]any{
				"event":                         "voxt_final_candidate_outcome",
				"run_id":                        r.id,
				"session_id":                    metadata.SessionID,
				"request_id":                    metadata.RequestID,
				"attempt":                       metadata.Attempt,
				"source":                        metadata.Source,
				"sequence":                      metadata.Sequence,
				"authoritative_final_available": false,
				"final_status":                  finalStatus,
				"outcome_unresolved":            true,
				"retired_attempt":               retired,
				"stale_attempt":                 metadata.Source == "http_final" && !retired,
			}
			payload, err := json.Marshal(record)
			if err == nil {
				r.logf("info", "%s", payload)
			}
		}
		summary, err := json.Marshal(map[string]any{
			"event":                       "voxt_final_candidate_summary",
			"run_id":                      r.id,
			"session_id":                  end.candidate.SessionID,
			"authoritative_attempt":       end.candidate.Attempt,
			"observed_candidate_count":    len(observations),
			"observations_capped":         capped,
			"observations_dropped":        dropped,
			"candidate_channel_coalesced": coalesced,
			"candidate_channel_pending":   pending,
			"final_status":                finalStatus,
			"stop_to_final_available_ms":  shadowStopToFinalMS(r, end),
			"cancellation_at_ms":          shadowCancellationMS(r, end),
		})
		if err == nil {
			r.logf("info", "%s", summary)
		}
		return
	}

	finalPlan, hasAuthoritativePlan := r.latestCleanupPlan()
	finalReason := ""
	finalPlanHash := ""
	if !hasAuthoritativePlan {
		finalReason = "authoritative_cleanup_plan_unavailable"
	} else {
		finalPlanHash = r.head.planHash(finalPlan)
	}
	finalHash := sha256.Sum256([]byte(end.res.Text))
	finalRunes := []rune(end.res.Text)
	earliestUsableLeadMS := float64(-1)
	earliestCompatibleLeadMS := float64(-1)
	earliestHTTPUsableLeadMS := float64(-1)
	earliestWebSocketUsableLeadMS := float64(-1)
	stopToFinalMS := float64(-1)
	if stopAt := r.monotonicStoppedAt(); !stopAt.IsZero() {
		stopToFinalMS = float64(end.finalAvailableAt.Sub(stopAt).Microseconds()) / 1000
	}
	for _, observation := range observations {
		candidate := observation.candidate
		metadata := candidate.metadata
		candidateRunes := []rune(candidate.text)
		candidateHash := sha256.Sum256([]byte(candidate.text))
		candidatePlanHash := r.head.planHash(observation.plan)
		commonPrefixRunes := 0
		for commonPrefixRunes < len(candidateRunes) && commonPrefixRunes < len(finalRunes) &&
			candidateRunes[commonPrefixRunes] == finalRunes[commonPrefixRunes] {
			commonPrefixRunes++
		}
		lengthRatio := float64(0)
		prefixCompleteness := float64(0)
		if len(finalRunes) > 0 {
			lengthRatio = float64(len(candidateRunes)) / float64(len(finalRunes))
			prefixCompleteness = float64(commonPrefixRunes) / float64(len(finalRunes))
		}
		sameSession := end.candidate.SessionID == 0 || metadata.SessionID == end.candidate.SessionID
		attemptMatches := metadata.Source != "http_final" ||
			end.candidate.Attempt > 0 && metadata.Attempt == end.candidate.Attempt
		staleAttempt := metadata.Source == "http_final" && !attemptMatches
		lead := end.finalAvailableAt.Sub(metadata.ReceivedAt)
		arrivedBeforeFinal := !metadata.ReceivedAt.IsZero() && lead > 0
		exactPlan, nearPlan, repairEligible := false, false, false
		compatibleParts := []int{}
		planComparable := observation.reason == "" || observation.reason == "same_as_latest_headstart"
		if planComparable && finalReason == "" {
			exactPlan, nearPlan, repairEligible, compatibleParts = r.head.plansCompatible(observation.plan, finalPlan)
		}
		compatiblePartIndexes := make(map[int]bool, len(compatibleParts))
		for _, partIndex := range compatibleParts {
			compatiblePartIndexes[partIndex] = true
		}
		freshCompatibleParts := 0
		for _, part := range observation.plan.parts {
			if compatiblePartIndexes[part.partIndex] && part.reuse == nil {
				freshCompatibleParts++
			}
		}
		oracleCompatible := sameSession && attemptMatches && arrivedBeforeFinal && len(compatibleParts) > 0
		leadMS := float64(lead.Microseconds()) / 1000
		dispatchReadyBeforeFinal := !observation.dispatchReadyAt.IsZero() && observation.dispatchReadyAt.Before(end.finalAvailableAt)
		usable := observation.eligible && sameSession && attemptMatches && arrivedBeforeFinal && dispatchReadyBeforeFinal
		if usable && (earliestUsableLeadMS < 0 || leadMS > earliestUsableLeadMS) {
			earliestUsableLeadMS = leadMS
		}
		if oracleCompatible && (earliestCompatibleLeadMS < 0 || leadMS > earliestCompatibleLeadMS) {
			earliestCompatibleLeadMS = leadMS
		}
		if usable && metadata.Source == "http_final" &&
			(earliestHTTPUsableLeadMS < 0 || leadMS > earliestHTTPUsableLeadMS) {
			earliestHTTPUsableLeadMS = leadMS
		}
		if usable && metadata.Source == "websocket_preview" &&
			(earliestWebSocketUsableLeadMS < 0 || leadMS > earliestWebSocketUsableLeadMS) {
			earliestWebSocketUsableLeadMS = leadMS
		}
		record := map[string]any{
			"event":                         "voxt_final_candidate_outcome",
			"run_id":                        r.id,
			"session_id":                    metadata.SessionID,
			"request_id":                    metadata.RequestID,
			"attempt":                       metadata.Attempt,
			"authoritative_attempt":         end.candidate.Attempt,
			"source":                        metadata.Source,
			"sequence":                      metadata.Sequence,
			"authoritative_final_available": true,
			"authoritative_final_available_offset_ms": stopToFinalMS,
			"candidate_lead_ms":                       leadMS,
			"dispatch_ready_lead_ms":                  float64(end.finalAvailableAt.Sub(observation.dispatchReadyAt).Microseconds()) / 1000,
			"candidate_processing_started_ms":         stopOffsetMS(observation.processingAt),
			"candidate_plan_ready_ms":                 stopOffsetMS(observation.planReadyAt),
			"candidate_dispatch_ready_ms":             stopOffsetMS(observation.dispatchReadyAt),
			"same_session":                            sameSession,
			"attempt_matches_authoritative":           attemptMatches,
			"stale_attempt":                           staleAttempt,
			"candidate_eligible_when_processed":       observation.eligible,
			"dispatch_ready_before_final":             dispatchReadyBeforeFinal,
			"candidate_usable_before_final":           usable,
			"transcript_sha256":                       hex.EncodeToString(candidateHash[:]),
			"authoritative_transcript_sha256":         hex.EncodeToString(finalHash[:]),
			"exact_transcript_match":                  candidate.text == end.res.Text,
			"candidate_runes":                         len(candidateRunes),
			"authoritative_runes":                     len(finalRunes),
			"transcript_length_ratio":                 lengthRatio,
			"common_prefix_runes":                     commonPrefixRunes,
			"common_prefix_completeness":              prefixCompleteness,
			"candidate_plan_sha256":                   candidatePlanHash,
			"authoritative_plan_sha256":               finalPlanHash,
			"exact_cleanup_plan_match":                exactPlan,
			"near_cleanup_plan_compatible":            nearPlan && !exactPlan,
			"partial_cleanup_plan_compatible":         len(compatibleParts) > 0 && !exactPlan && !nearPlan,
			"cleanup_repair_eligible":                 repairEligible,
			"authoritative_cleanup_plan_available":    hasAuthoritativePlan,
			"outcome_unresolved":                      !hasAuthoritativePlan,
			"authoritative_cleanup_plan_sha256":       finalPlanHash,
			"compatible_cleanup_parts":                len(compatibleParts),
			"compatible_candidate_part_indices":       compatibleParts,
			"candidate_cleanup_part_count":            len(observation.plan.parts),
			"authoritative_cleanup_part_count":        len(finalPlan.parts),
			"fresh_compatible_cleanup_parts":          freshCompatibleParts,
			"fresh_requests_potentially_avoided":      freshCompatibleParts,
			"existing_equivalent_requests":            observation.existingRequests,
			"oracle_compatible":                       oracleCompatible,
			"oracle_incremental_compatible":           oracleCompatible && freshCompatibleParts > 0,
		}
		payload, err := json.Marshal(record)
		if err == nil {
			r.logf("info", "%s", payload)
		}
	}
	summary := map[string]any{
		"event":                                   "voxt_final_candidate_summary",
		"run_id":                                  r.id,
		"session_id":                              end.candidate.SessionID,
		"authoritative_attempt":                   end.candidate.Attempt,
		"observed_candidate_count":                len(observations),
		"observations_capped":                     capped,
		"observations_dropped":                    dropped,
		"candidate_channel_coalesced":             coalesced,
		"candidate_channel_pending":               pending,
		"earliest_usable_lead_ms":                 earliestUsableLeadMS,
		"earliest_oracle_compatible_lead_ms":      earliestCompatibleLeadMS,
		"earliest_http_usable_lead_ms":            earliestHTTPUsableLeadMS,
		"earliest_websocket_usable_lead_ms":       earliestWebSocketUsableLeadMS,
		"http_precedes_websocket_ms":              float64(-1),
		"authoritative_final_available_offset_ms": stopToFinalMS,
		"stop_to_final_available_ms":              stopToFinalMS,
	}
	if earliestHTTPUsableLeadMS >= 0 && earliestWebSocketUsableLeadMS >= 0 {
		summary["http_precedes_websocket_ms"] = earliestHTTPUsableLeadMS - earliestWebSocketUsableLeadMS
	} else {
	}
	payload, err := json.Marshal(summary)
	if err == nil {
		r.logf("info", "%s", payload)
	}
}

func shadowStopToFinalMS(r *run, end streamEnd) float64 {
	stopAt := r.monotonicStoppedAt()
	if stopAt.IsZero() || end.finalAvailableAt.IsZero() {
		return -1
	}
	return float64(end.finalAvailableAt.Sub(stopAt).Microseconds()) / 1000
}

func shadowCancellationMS(r *run, end streamEnd) float64 {
	if end.err == nil || !r.cancelled.Load() {
		return -1
	}
	return shadowStopToFinalMS(r, end)
}

func (h *headStart) plansCompatible(candidate, final headPlan) (bool, bool, bool, []int) {
	fullPlan := len(candidate.parts) == len(final.parts)
	exact, near, repair := fullPlan, fullPlan, false
	compatibleParts := make([]int, 0, min(len(candidate.parts), len(final.parts)))
	finalParts := make(map[int]headPart, len(final.parts))
	for _, finalPart := range final.parts {
		finalParts[finalPart.partIndex] = finalPart
	}
	for _, candidatePart := range candidate.parts {
		finalPart, ok := finalParts[candidatePart.partIndex]
		if !ok {
			exact, near = false, false
			continue
		}
		if h.sameRequest(candidatePart.req, finalPart.req) {
			compatibleParts = append(compatibleParts, candidatePart.partIndex)
			continue
		}
		exact = false
		_, changes, ok := h.nearCompatibility(
			candidatePart.req, candidatePart.text, finalPart.req, finalPart.text)
		if !ok {
			near = false
			continue
		}
		compatibleParts = append(compatibleParts, candidatePart.partIndex)
		if len(changes) > 0 {
			repair = true
		}
	}
	return exact, near, repair, compatibleParts
}
