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
	candidate                     shadowCandidate
	plan                          headPlan
	processingAt                  time.Time
	planReadyAt                   time.Time
	dispatchReadyAt               time.Time
	reason                        string
	recordingGenerations          int
	budgetAvailable               bool
	llmReady                      bool
	cleanupEligible               bool
	existingRequests              int
	previewMatch                  bool
	extendsPreview                bool
	textBoundary                  bool
	eligible                      bool
	counterfactualStateUnresolved bool
}

type shadowHeadStartDecision struct {
	update          heardText
	plan            headPlan
	sequence        uint64
	reason          string
	before          lateGateState
	after           lateGateState
	processingAt    time.Time
	planReadyAt     time.Time
	dispatchReadyAt time.Time
	dispatchAt      time.Time
	requestIDs      []uint64
	dispatched      bool
}

func cloneHeadPlan(plan headPlan) headPlan {
	plan.parts = append([]headPart(nil), plan.parts...)
	return plan
}

func (h *headStart) planHash(plan headPlan) string {
	fingerprint := make([]map[string]any, 0, len(plan.parts))
	for _, part := range plan.parts {
		fingerprint = append(fingerprint, map[string]any{
			"part_index":   part.partIndex,
			"messages":     part.req.Messages,
			"thinking_off": part.req.InputChars < h.under,
		})
	}
	planBytes, _ := json.Marshal(fingerprint)
	hash := sha256.Sum256(planBytes)
	return hex.EncodeToString(hash[:])
}

func (h *headStart) partPlanHash(part headPart) string {
	requestBytes, _ := json.Marshal(map[string]any{
		"messages":     part.req.Messages,
		"thinking_off": part.req.InputChars < h.under,
	})
	hash := sha256.Sum256(requestBytes)
	return hex.EncodeToString(hash[:])
}

func (r *run) captureCleanupPlan(plan headPlan, source string) {
	if !r.shadowTraceEnabled || len(plan.parts) == 0 {
		return
	}
	planID := r.cleanupPlanSequence.Add(1)
	if source == "authoritative_cleanup" {
		captured := headPlan{heard: plan.heard, rest: plan.rest, parts: append([]headPart(nil), plan.parts...)}
		r.cleanupPlanMu.Lock()
		r.cleanupPlans = append(r.cleanupPlans, captured)
		r.cleanupPlanMu.Unlock()
	}
	partHashes := make([]string, 0, len(plan.parts))
	for _, part := range plan.parts {
		partHash := r.head.partPlanHash(part)
		partHashes = append(partHashes, partHash)
		r.logShadowEvent(map[string]any{
			"event": "voxt_cleanup_request_plan_part", "run_id": r.id,
			"plan_id": planID, "source": source, "part_index": part.partIndex,
			"part_count": len(plan.parts), "request_plan_sha256": partHash,
		})
	}
	r.logShadowEvent(map[string]any{
		"event": "voxt_cleanup_request_plan", "run_id": r.id,
		"plan_id": planID, "source": source, "part_count": len(plan.parts),
		"request_plan_sha256": r.head.planHash(plan),
		"part_request_sha256": partHashes,
	})
}

func (r *run) logShadowEvent(record map[string]any) {
	if !r.shadowTraceEnabled {
		return
	}
	payload, err := json.Marshal(record)
	if err == nil {
		r.logf("info", "%s", payload)
	}
}

func (r *run) observeShadowCandidate(candidate shadowCandidate) shadowObservation {
	processingAt := time.Now()
	plan, reason := r.head.transcriptPlan(candidate.text)
	planReadyAt := time.Now()
	if reason == "" && plan.heard == r.head.text {
		reason = "same_as_latest_headstart"
	} else {
	}
	budgetAvailable, llmReady, cleanupEligible := r.head.lateGates()
	r.shadowMu.Lock()
	initial := r.shadowInitialBudget
	initialKnown := r.shadowInitialBudgetKnown
	r.shadowMu.Unlock()
	recordingGenerations := 0
	if initialKnown {
		recordingGenerations = initial.RecordingGenerations
	}
	existingRequests := 0
	for _, part := range plan.parts {
		if part.reuse != nil {
			existingRequests++
		} else {
		}
	}
	trimmed := strings.TrimSpace(candidate.text)
	lastRune, _ := utf8.DecodeLastRuneInString(trimmed)
	previewMatch := candidate.preview != "" && candidate.text == candidate.preview
	extendsPreview := candidate.preview != "" &&
		len(candidate.text) > len(candidate.preview) &&
		strings.HasPrefix(candidate.text, candidate.preview)
	dispatchReadyAt := time.Now()
	eligible := reason == "" && plan.hasFresh() && budgetAvailable && llmReady && cleanupEligible
	counterfactualStateUnresolved := false
	r.shadowMu.Lock()
	for _, decision := range r.shadowHeadStartDecisions {
		if decision.dispatched && !decision.dispatchAt.IsZero() && decision.dispatchAt.Before(processingAt) {
			counterfactualStateUnresolved = true
			break
		}
	}
	r.shadowMu.Unlock()
	return shadowObservation{
		candidate: candidate, plan: plan,
		processingAt: processingAt, planReadyAt: planReadyAt, dispatchReadyAt: dispatchReadyAt,
		reason: reason, recordingGenerations: recordingGenerations,
		budgetAvailable: budgetAvailable, llmReady: llmReady, cleanupEligible: cleanupEligible,
		existingRequests: existingRequests, previewMatch: previewMatch, extendsPreview: extendsPreview,
		textBoundary: strings.ContainsRune("。.!！?？,，;；:：", lastRune), eligible: eligible,
		counterfactualStateUnresolved: counterfactualStateUnresolved,
	}
}
