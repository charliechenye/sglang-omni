package session

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"time"
)

type lateDispatchResult struct {
	end             streamEnd
	err             error
	terminal        bool
	dispatched      bool
	requestIDs      []uint64
	dispatchReadyAt time.Time
	dispatchedAt    time.Time
}

// note (chenye): waitFinal consumes whole-text updates while the session owner
// waits for the authoritative ASR result.
func (r *run) waitFinal(ctx context.Context, finals <-chan streamEnd, heard <-chan heardText, sceneDone *<-chan struct{}, onReadiness func()) (streamEnd, error) {
	var observations []shadowObservation
	observationsCapped := false
	var observationsDropped uint64
	var headStartSequence uint64
	finish := func(end streamEnd, err error) (streamEnd, error) {
		if err != nil && end.finalAvailableAt.IsZero() {
			end.finalAvailableAt = time.Now()
		}
		r.shadowMu.Lock()
		r.shadowObservations = append([]shadowObservation(nil), observations...)
		r.shadowObservationCap = observationsCapped
		r.shadowObservationDropped = observationsDropped
		final := end
		r.shadowFinal = &final
		r.shadowMu.Unlock()
		return end, err
	}
	for {
		if end, err, ready := r.recognitionReady(ctx, finals); ready {
			return finish(end, err)
		}
		select {
		case end := <-finals:
			return finish(end, nil)
		case <-ctx.Done():
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return finish(end, err)
			}
			return finish(streamEnd{}, r.ctxErr(ctx))
		case candidate := <-r.shadowCandidates:
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				observationsDropped++
				return finish(end, err)
			}
			if len(observations) < 128 {
				observations = append(observations, r.observeShadowCandidate(candidate))
			} else if !observationsCapped {
				observationsDropped++
				r.logShadowEvent(map[string]any{
					"event": "voxt_final_candidate_observation_capped", "run_id": r.id,
					"session_id": r.shadowSessionID, "limit": 128,
				})
				observationsCapped = true
			} else {
				observationsDropped++
			}
		case update := <-heard:
			headStartSequence++
			processingAt := time.Now()
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				state := r.head.lateGateState()
				r.logHeadStartDecision(update, headStartSequence, headPlan{},
					"authoritative_final_available_before_candidate_processing",
					state, state, processingAt, processingAt, time.Time{}, time.Time{}, nil, false)
				return finish(end, err)
			}
			plan, ok, reason, before, after := r.head.prepareLateObserved(update.text)
			planReadyAt := time.Now()
			var dispatchAt time.Time
			var dispatchReadyAt time.Time
			var requestIDs []uint64
			dispatched := false
			if ok {
				dispatch := r.dispatchLate(ctx, finals, plan)
				dispatchReadyAt, dispatchAt, requestIDs = dispatch.dispatchReadyAt, dispatch.dispatchedAt, dispatch.requestIDs
				dispatched = dispatch.dispatched
				after = r.head.lateGateState()
				if dispatch.terminal {
					reason = "authoritative_final_or_cancellation_during_preparation"
				} else if dispatch.dispatched {
				} else {
					reason = "late_dispatch_rejected_by_current_gates"
				}
				if dispatch.terminal {
					r.logHeadStartDecision(update, headStartSequence, plan, reason,
						before, after, processingAt, planReadyAt, dispatchReadyAt, dispatchAt, requestIDs, dispatched)
					return finish(dispatch.end, dispatch.err)
				}
			}
			r.logHeadStartDecision(update, headStartSequence, plan, reason,
				before, after, processingAt, planReadyAt, dispatchReadyAt, dispatchAt, requestIDs, dispatched)
		case result := <-r.init.llmDone:
			r.init.llmDone = nil
			if !r.useLLMReadiness(result) {
				return finish(streamEnd{}, r.ctxErr(ctx))
			}
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return finish(end, err)
			}
			onReadiness()
		case terms := <-r.init.termsDone:
			r.init.termsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return finish(streamEnd{}, r.ctxErr(ctx))
			}
			r.useTerms(terms)
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return finish(end, err)
			}
			onReadiness()
		case snippets := <-r.init.snippetsDone:
			r.init.snippetsDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return finish(streamEnd{}, r.ctxErr(ctx))
			}
			r.useSnippets(snippets)
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return finish(end, err)
			}
			onReadiness()
		case <-*sceneDone:
			*sceneDone = nil
			if r.cancelled.Load() || r.ctx.Err() != nil {
				return finish(streamEnd{}, r.ctxErr(ctx))
			}
			r.markSceneReady()
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return finish(end, err)
			}
			onReadiness()
		}
	}
}

// note (chenye): dispatchLate returns an observable terminal result before it
// starts prepared speculative requests.
func (r *run) dispatchLate(ctx context.Context, finals <-chan streamEnd, plan headPlan) lateDispatchResult {
	if end, err, ready := r.recognitionReady(ctx, finals); ready {
		return lateDispatchResult{end: end, err: err, terminal: true, dispatchReadyAt: time.Now()}
	}
	submission := r.head.submitLate(plan)
	return lateDispatchResult{
		terminal: false, dispatched: submission.dispatched, requestIDs: submission.requestIDs,
		dispatchReadyAt: submission.dispatchReadyAt, dispatchedAt: submission.dispatchedAt,
	}
}

func (r *run) logHeadStartDecision(update heardText, sequence uint64, plan headPlan, reason string,
	before, after lateGateState, processingAt, planReadyAt, dispatchReadyAt, dispatchAt time.Time,
	requestIDs []uint64, dispatched bool) {
	if !r.head.enabled.Load() {
		return
	}
	textHash := sha256.Sum256([]byte(update.text))
	stopAt := r.monotonicStoppedAt()
	toStopMS := func(at time.Time) float64 {
		if stopAt.IsZero() || at.IsZero() {
			return -1
		}
		return float64(at.Sub(stopAt).Microseconds()) / 1000
	}
	planHash := ""
	partHashes := make([]string, 0, len(plan.parts))
	if len(plan.parts) > 0 {
		planHash = r.head.planHash(plan)
		for _, part := range plan.parts {
			partHashes = append(partHashes, r.head.partPlanHash(part))
		}
	}
	planUsable := reason == "" || plan.hasFresh()
	eligibleWithoutBudget := planUsable && before.LLMReady && before.CleanupEligible
	source := "websocket_preview"
	if update.candidate {
		source = "websocket_candidate"
	}
	r.logShadowEvent(map[string]any{
		"event": "voxt_headstart_decision", "run_id": r.id,
		"session_id": r.shadowSessionID,
		"source":     source, "sequence": sequence,
		"candidate_sha256":                  hex.EncodeToString(textHash[:]),
		"candidate_arrived_ms":              toStopMS(update.receivedAt),
		"candidate_processing_started_ms":   toStopMS(processingAt),
		"candidate_plan_ready_ms":           toStopMS(planReadyAt),
		"dispatch_ready_ms":                 toStopMS(dispatchReadyAt),
		"dispatch_at_ms":                    toStopMS(dispatchAt),
		"request_ids":                       requestIDs,
		"candidate_processing_ms":           float64(planReadyAt.Sub(processingAt).Microseconds()) / 1000,
		"preparation_succeeded":             planUsable,
		"candidate_eligible_without_budget": eligibleWithoutBudget,
		"candidate_eligible":                dispatched,
		"dispatched":                        dispatched,
		"rejection_reason":                  reason,
		"late_generation_budget_before":     before.BudgetAvailable,
		"late_generation_budget_after":      after.BudgetAvailable && !dispatched,
		"late_already_dispatched_before":    before.LateAlreadyDispatched,
		"recording_headstart_generations":   before.RecordingGenerations,
		"llm_readiness_open":                before.LLMReady,
		"cleanup_eligible":                  before.CleanupEligible,
		"request_plan_sha256":               planHash,
		"part_request_sha256":               partHashes,
	})
}

func (r *run) recognitionReady(ctx context.Context, finals <-chan streamEnd) (streamEnd, error, bool) {
	select {
	case end := <-finals:
		return end, nil, true
	default:
	}
	select {
	case <-ctx.Done():
		return streamEnd{}, r.ctxErr(ctx), true
	default:
		return streamEnd{}, nil, false
	}
}
