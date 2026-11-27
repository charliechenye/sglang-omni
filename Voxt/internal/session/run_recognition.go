package session

import "context"

// note (chenye): waitFinal consumes whole-text updates while the session owner
// waits for the authoritative ASR result.
func (r *run) waitFinal(ctx context.Context, finals <-chan streamEnd, heard <-chan heardText, sceneDone *<-chan struct{}, onReadiness func()) (streamEnd, error) {
	var observations []shadowObservation
	observationsCapped := false
	finish := func(end streamEnd, err error) (streamEnd, error) {
		r.recordShadowOutcomes(end, observations, observationsCapped)
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
			if len(observations) < 128 {
				observations = append(observations, r.observeShadowCandidate(candidate))
			} else if !observationsCapped {
				r.logf("info", "{\"event\":\"voxt_final_candidate_observation_capped\",\"limit\":128}")
				observationsCapped = true
			} else {
			}
		case update := <-heard:
			if end, err, ready := r.recognitionReady(ctx, finals); ready {
				return finish(end, err)
			}
			plan, ok := r.head.prepareLate(update.text)
			if !ok {
				continue
			}
			if end, err, ready := r.dispatchLate(ctx, finals, plan); ready {
				return finish(end, err)
			}
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
func (r *run) dispatchLate(ctx context.Context, finals <-chan streamEnd, plan headPlan) (streamEnd, error, bool) {
	if end, err, ready := r.recognitionReady(ctx, finals); ready {
		return end, err, true
	}
	r.head.submitLate(plan)
	return streamEnd{}, nil, false
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
