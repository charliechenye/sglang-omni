package session

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"time"
	"unicode/utf8"
)

func (r *run) monotonicStoppedAt() time.Time {
	r.stopAtMu.Lock()
	defer r.stopAtMu.Unlock()
	return r.stopMonotonicAt
}

func (h *headStart) latePlan(live string) (headPlan, string) {
	plan, reason := h.transcriptPlan(live)
	if reason != "" {
		return headPlan{}, reason
	} else if plan.heard == h.text {
		return headPlan{}, "same_as_latest_headstart"
	} else {
		return plan, ""
	}
}

func (h *headStart) transcriptPlan(live string) (headPlan, string) {
	heard, cleaned, ok := h.normalize(live)
	if !ok {
		return headPlan{}, "normalization_rejected"
	} else {
	}
	plan, ok, _ := h.prepare(cleaned)
	if !ok {
		return headPlan{}, "cleanup_not_applicable"
	} else {
	}
	plan.heard = heard
	return plan, ""
}

type shadowObservation struct {
	candidate        shadowCandidate
	plan             headPlan
	planHash         string
	reason           string
	budgetAvailable  bool
	llmReady         bool
	cleanupEligible  bool
	existingRequests int
	previewMatch     bool
	extendsPreview   bool
	textBoundary     bool
	eligible         bool
}

func (h *headStart) planHash(plan headPlan) string {
	fingerprint := make([]map[string]any, 0, len(plan.parts))
	for _, part := range plan.parts {
		fingerprint = append(fingerprint, map[string]any{
			"messages":     part.req.Messages,
			"thinking_off": part.req.InputChars < h.under,
		})
	}
	planBytes, _ := json.Marshal(fingerprint)
	hash := sha256.Sum256(planBytes)
	return hex.EncodeToString(hash[:])
}

func (r *run) observeShadowCandidate(candidate shadowCandidate) shadowObservation {
	plan, reason := r.head.transcriptPlan(candidate.text)
	if reason == "" && plan.heard == r.head.text {
		reason = "same_as_latest_headstart"
	} else {
	}
	budgetAvailable, llmReady, cleanupEligible := r.head.lateGates()
	existingRequests := 0
	for _, part := range plan.parts {
		if part.reuse != nil {
			existingRequests++
		} else {
		}
	}
	planHash := r.head.planHash(plan)
	textHash := sha256.Sum256([]byte(candidate.text))
	trimmed := strings.TrimSpace(candidate.text)
	lastRune, _ := utf8.DecodeLastRuneInString(trimmed)
	previewMatch := candidate.preview != "" && candidate.text == candidate.preview
	extendsPreview := candidate.preview != "" &&
		len(candidate.text) > len(candidate.preview) &&
		strings.HasPrefix(candidate.text, candidate.preview)
	stopToCandidateMS := float64(0)
	stopAt := r.monotonicStoppedAt()
	if !stopAt.IsZero() && !candidate.metadata.ReceivedAt.IsZero() {
		stopToCandidateMS = float64(candidate.metadata.ReceivedAt.Sub(stopAt).Microseconds()) / 1000
	} else {
	}
	eligibilityAt := time.Now()
	arrivalToEligibilityMS := float64(-1)
	stopToEligibilityMS := float64(-1)
	if !candidate.metadata.ReceivedAt.IsZero() {
		arrivalToEligibilityMS = float64(eligibilityAt.Sub(candidate.metadata.ReceivedAt).Microseconds()) / 1000
	} else {
	}
	if !stopAt.IsZero() {
		stopToEligibilityMS = float64(eligibilityAt.Sub(stopAt).Microseconds()) / 1000
	} else {
	}
	httpStartOffsetMS := float64(-1)
	stopToHTTPRequestStartMS := float64(-1)
	if candidate.metadata.Source == "http_final" {
		httpStartOffsetMS = candidate.metadata.HTTPRequestStartOffsetMS
		stopToHTTPRequestStartMS = stopToCandidateMS - candidate.metadata.RequestElapsedMS + httpStartOffsetMS
	} else {
	}
	record := map[string]any{
		"event":                             "voxt_final_candidate_eligibility",
		"session_id":                        candidate.metadata.SessionID,
		"request_id":                        candidate.metadata.RequestID,
		"attempt":                           candidate.metadata.Attempt,
		"source":                            candidate.metadata.Source,
		"sequence":                          candidate.metadata.Sequence,
		"generated_token_count":             candidate.metadata.GeneratedTokens,
		"native_elapsed_ms":                 candidate.metadata.NativeElapsedMS,
		"first_transcript_token_elapsed_ms": candidate.metadata.FirstTranscriptTokenMS,
		"snapshot_build_ms":                 candidate.metadata.SnapshotBuildMS,
		"published_candidate_count":         candidate.metadata.PublishedCandidateCount,
		"snapshot_build_ms_total":           candidate.metadata.SnapshotBuildMSTotal,
		"request_elapsed_ms":                candidate.metadata.RequestElapsedMS,
		"http_request_start_offset_ms":      httpStartOffsetMS,
		"stop_to_http_request_start_ms":     stopToHTTPRequestStartMS,
		"stop_to_candidate_ms":              stopToCandidateMS,
		"candidate_processing_ms":           arrivalToEligibilityMS,
		"stop_to_eligibility_ms":            stopToEligibilityMS,
		"transcript_sha256":                 hex.EncodeToString(textHash[:]),
		"transcript_runes":                  utf8.RuneCountInString(candidate.text),
		"latest_preview_match":              previewMatch,
		"extends_latest_hypothesis":         extendsPreview,
		"meaningful_text_boundary":          strings.ContainsRune("。.!！?？,，;；:：", lastRune),
		"plan_sha256":                       planHash,
		"plan_reason":                       reason,
		"existing_equivalent_requests":      existingRequests,
		"late_generation_budget_available":  budgetAvailable,
		"llm_readiness_open":                llmReady,
		"cleanup_eligible":                  cleanupEligible,
		"candidate_eligible": reason == "" && plan.hasFresh() &&
			budgetAvailable && llmReady && cleanupEligible,
	}
	payload, err := json.Marshal(record)
	if err == nil {
		r.logf("info", "%s", payload)
	} else {
	}
	return shadowObservation{
		candidate: candidate, plan: plan, planHash: planHash, reason: reason,
		budgetAvailable: budgetAvailable, llmReady: llmReady,
		cleanupEligible: cleanupEligible, existingRequests: existingRequests,
		previewMatch: previewMatch, extendsPreview: extendsPreview,
		textBoundary: strings.ContainsRune("。.!！?？,，;；:：", lastRune),
		eligible:     reason == "" && plan.hasFresh() && budgetAvailable && llmReady && cleanupEligible,
	}
}

func (r *run) recordShadowOutcomes(end streamEnd, observations []shadowObservation, capped bool) {
	for {
		select {
		case candidate := <-r.shadowCandidates:
			if len(observations) < 128 {
				observations = append(observations, r.observeShadowCandidate(candidate))
			} else if !capped {
				r.logf("info", "{\"event\":\"voxt_final_candidate_observation_capped\",\"limit\":128}")
				capped = true
			} else {
			}
		default:
			goto drained
		}
	}

drained:
	if len(observations) == 0 && end.candidate.SessionID == 0 {
		return
	}
	if end.err != nil || end.finalAvailableAt.IsZero() {
		for _, observation := range observations {
			metadata := observation.candidate.metadata
			retired := metadata.Source == "http_final" &&
				metadata.SessionID == end.candidate.SessionID &&
				metadata.Attempt == end.candidate.Attempt
			record := map[string]any{
				"event":                         "voxt_final_candidate_outcome",
				"session_id":                    metadata.SessionID,
				"request_id":                    metadata.RequestID,
				"attempt":                       metadata.Attempt,
				"source":                        metadata.Source,
				"sequence":                      metadata.Sequence,
				"authoritative_final_available": false,
				"final_status":                  "failed_or_cancelled",
				"retired_attempt":               retired,
				"stale_attempt":                 metadata.Source == "http_final" && !retired,
			}
			payload, err := json.Marshal(record)
			if err == nil {
				r.logf("info", "%s", payload)
			}
		}
		summary, err := json.Marshal(map[string]any{
			"event":                    "voxt_final_candidate_summary",
			"session_id":               end.candidate.SessionID,
			"authoritative_attempt":    end.candidate.Attempt,
			"observed_candidate_count": len(observations),
			"observations_capped":      capped,
			"final_status":             "failed_or_cancelled",
		})
		if err == nil {
			r.logf("info", "%s", summary)
		}
		return
	}

	finalPlan, finalReason := r.head.transcriptPlan(end.res.Text)
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
		equivalentParts := 0
		planComparable := observation.reason == "" || observation.reason == "same_as_latest_headstart"
		if planComparable && finalReason == "" {
			exactPlan, nearPlan, repairEligible = r.head.plansCompatible(observation.plan, finalPlan)
			if len(observation.plan.parts) == len(finalPlan.parts) {
				for partIndex, finalPart := range finalPlan.parts {
					if r.head.sameRequest(observation.plan.parts[partIndex].req, finalPart.req) {
						equivalentParts++
					}
				}
			}
		}
		oracleCompatible := sameSession && attemptMatches && arrivedBeforeFinal && nearPlan
		leadMS := float64(lead.Microseconds()) / 1000
		usable := observation.eligible && sameSession && attemptMatches && arrivedBeforeFinal
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
			"session_id":                    metadata.SessionID,
			"request_id":                    metadata.RequestID,
			"attempt":                       metadata.Attempt,
			"authoritative_attempt":         end.candidate.Attempt,
			"source":                        metadata.Source,
			"sequence":                      metadata.Sequence,
			"authoritative_final_available": true,
			"authoritative_final_available_offset_ms": stopToFinalMS,
			"candidate_lead_ms":                       leadMS,
			"same_session":                            sameSession,
			"attempt_matches_authoritative":           attemptMatches,
			"stale_attempt":                           staleAttempt,
			"candidate_eligible_when_processed":       observation.eligible,
			"candidate_usable_before_final":           usable,
			"transcript_sha256":                       hex.EncodeToString(candidateHash[:]),
			"authoritative_transcript_sha256":         hex.EncodeToString(finalHash[:]),
			"exact_transcript_match":                  candidate.text == end.res.Text,
			"candidate_runes":                         len(candidateRunes),
			"authoritative_runes":                     len(finalRunes),
			"transcript_length_ratio":                 lengthRatio,
			"common_prefix_runes":                     commonPrefixRunes,
			"common_prefix_completeness":              prefixCompleteness,
			"candidate_plan_sha256":                   observation.planHash,
			"authoritative_plan_sha256":               r.head.planHash(finalPlan),
			"exact_cleanup_plan_match":                exactPlan,
			"near_cleanup_plan_compatible":            nearPlan && !exactPlan,
			"cleanup_repair_eligible":                 repairEligible,
			"equivalent_cleanup_parts":                equivalentParts,
			"existing_equivalent_requests":            observation.existingRequests,
			"oracle_compatible":                       oracleCompatible,
			"oracle_incremental_compatible":           oracleCompatible && observation.plan.hasFresh(),
		}
		payload, err := json.Marshal(record)
		if err == nil {
			r.logf("info", "%s", payload)
		}
	}
	summary := map[string]any{
		"event":                                   "voxt_final_candidate_summary",
		"session_id":                              end.candidate.SessionID,
		"authoritative_attempt":                   end.candidate.Attempt,
		"observed_candidate_count":                len(observations),
		"observations_capped":                     capped,
		"earliest_usable_lead_ms":                 earliestUsableLeadMS,
		"earliest_oracle_compatible_lead_ms":      earliestCompatibleLeadMS,
		"earliest_http_usable_lead_ms":            earliestHTTPUsableLeadMS,
		"earliest_websocket_usable_lead_ms":       earliestWebSocketUsableLeadMS,
		"http_precedes_websocket_ms":              float64(-1),
		"authoritative_final_available_offset_ms": stopToFinalMS,
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

func (h *headStart) plansCompatible(candidate, final headPlan) (bool, bool, bool) {
	if len(candidate.parts) != len(final.parts) {
		return false, false, false
	} else {
	}
	exact, near, repair := true, true, false
	for i, finalPart := range final.parts {
		candidatePart := candidate.parts[i]
		if h.sameRequest(candidatePart.req, finalPart.req) {
			continue
		} else {
			exact = false
		}
		_, changes, ok := h.nearCompatibility(
			candidatePart.req, candidatePart.text, finalPart.req, finalPart.text)
		if !ok {
			near = false
			return false, false, false
		} else if len(changes) > 0 {
			repair = true
		} else {
		}
	}
	return exact, near, repair
}
