package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type benchPairResult struct {
	TranscriptEqual         bool
	ResultEqual             bool
	ParityValid             bool
	Successful              bool
	SteadyState             bool
	ColdStart               bool
	FinalValid              bool
	RequestStartValid       bool
	FirstTokenValid         bool
	FirstVisibleTextValid   bool
	CleanupValid            bool
	DeliveryRequestValid    bool
	DeliveryCompleteValid   bool
	FinalDeltaMS            float64
	RequestStartMS          float64
	FirstTokenMS            float64
	FirstVisibleTextMS      float64
	CleanupDeltaMS          float64
	DeliveryRequestDeltaMS  float64
	DeliveryCompleteDeltaMS float64
	RequestDelta            int
}

func emitBenchPair(t *testing.T, scenario benchScenario, round int, treatment string, off, shadow *benchRun) benchPairResult {
	t.Helper()
	offData, shadowData := off.snapshot(), shadow.snapshot()
	coldStart := offData.ColdStart || shadowData.ColdStart
	offFinal := offsetMS(offData.StoppedAt, offData.FinalAt)
	shadowFinal := offsetMS(shadowData.StoppedAt, shadowData.FinalAt)
	offRequestStart, offRequestStartValid := firstLLMRequestOffset(offData.StoppedAt, offData.LLMCalls)
	shadowRequestStart, shadowRequestStartValid := firstLLMRequestOffset(shadowData.StoppedAt, shadowData.LLMCalls)
	offFirstToken, offFirstTokenValid := firstProviderTokenOffset(offData.StoppedAt, offData.LLMCalls)
	shadowFirstToken, shadowFirstTokenValid := firstProviderTokenOffset(shadowData.StoppedAt, shadowData.LLMCalls)
	offFirstVisibleText := offsetMS(offData.StoppedAt, offData.FirstDraftAt)
	shadowFirstVisibleText := offsetMS(shadowData.StoppedAt, shadowData.FirstDraftAt)
	offDeliveryRequest := offsetMS(offData.StoppedAt, offData.DeliveryRequestedAt)
	shadowDeliveryRequest := offsetMS(shadowData.StoppedAt, shadowData.DeliveryRequestedAt)
	offDeliveryComplete := offsetMS(offData.StoppedAt, offData.DeliveryCompletedAt)
	shadowDeliveryComplete := offsetMS(shadowData.StoppedAt, shadowData.DeliveryCompletedAt)
	offCleanup := offsetMS(offData.StoppedAt, offData.CleanupDoneAt)
	shadowCleanup := offsetMS(shadowData.StoppedAt, shadowData.CleanupDoneAt)
	result := benchPairResult{
		TranscriptEqual:         offData.Snapshot.Raw != "" && shadowData.Snapshot.Raw != "" && textHash(offData.Snapshot.Raw) == textHash(shadowData.Snapshot.Raw),
		ResultEqual:             offData.Snapshot.Result != "" && shadowData.Snapshot.Result != "" && textHash(offData.Snapshot.Result) == textHash(shadowData.Snapshot.Result),
		ParityValid:             offData.Snapshot.Raw != "" && shadowData.Snapshot.Raw != "" && offData.Snapshot.Result != "" && shadowData.Snapshot.Result != "",
		Successful:              benchRunSuccessful(offData) && benchRunSuccessful(shadowData),
		SteadyState:             !offData.Warmup && !shadowData.Warmup && !coldStart,
		ColdStart:               coldStart,
		FinalValid:              offFinal >= 0 && shadowFinal >= 0,
		RequestStartValid:       offRequestStartValid && shadowRequestStartValid,
		FirstTokenValid:         offFirstTokenValid && shadowFirstTokenValid,
		FirstVisibleTextValid:   !offData.FirstDraftAt.IsZero() && !shadowData.FirstDraftAt.IsZero(),
		CleanupValid:            offCleanup >= 0 && shadowCleanup >= 0,
		DeliveryRequestValid:    offDeliveryRequest >= 0 && shadowDeliveryRequest >= 0,
		DeliveryCompleteValid:   offDeliveryComplete >= 0 && shadowDeliveryComplete >= 0,
		FinalDeltaMS:            subtractIfValid(shadowFinal, offFinal),
		RequestStartMS:          subtractWhenValid(shadowRequestStart, offRequestStart, shadowRequestStartValid, offRequestStartValid),
		FirstTokenMS:            subtractWhenValid(shadowFirstToken, offFirstToken, shadowFirstTokenValid, offFirstTokenValid),
		FirstVisibleTextMS:      subtractWhenValid(shadowFirstVisibleText, offFirstVisibleText, !shadowData.FirstDraftAt.IsZero(), !offData.FirstDraftAt.IsZero()),
		CleanupDeltaMS:          subtractIfValid(shadowCleanup, offCleanup),
		DeliveryRequestDeltaMS:  subtractIfValid(shadowDeliveryRequest, offDeliveryRequest),
		DeliveryCompleteDeltaMS: subtractIfValid(shadowDeliveryComplete, offDeliveryComplete),
		RequestDelta:            len(shadowData.LLMCalls) - len(offData.LLMCalls),
	}
	row := map[string]any{
		"event": "final_candidate_benchmark_pair", "scenario": scenario.Name, "treatment_arm": treatment, "round": round,
		"transcript_equal":               result.TranscriptEqual,
		"result_equal":                   result.ResultEqual,
		"off_stop_to_final_available_ms": offFinal, "shadow_stop_to_final_available_ms": shadowFinal,
		"final_available_delta_shadow_minus_off_ms": result.FinalDeltaMS,
		"off_stop_to_first_llm_request_ms":          offRequestStart, "shadow_stop_to_first_llm_request_ms": shadowRequestStart,
		"first_llm_request_delta_shadow_minus_off_ms": result.RequestStartMS,
		"first_llm_request_delta_valid":               result.RequestStartValid,
		"off_stop_to_first_provider_token_ms":         offFirstToken, "shadow_stop_to_first_provider_token_ms": shadowFirstToken,
		"first_provider_token_delta_shadow_minus_off_ms": result.FirstTokenMS,
		"first_provider_token_delta_valid":               result.FirstTokenValid,
		"off_stop_to_first_visible_cleanup_text_ms":      offFirstVisibleText, "shadow_stop_to_first_visible_cleanup_text_ms": shadowFirstVisibleText,
		"first_visible_cleanup_text_delta_shadow_minus_off_ms": result.FirstVisibleTextMS,
		"first_visible_cleanup_text_delta_valid":               result.FirstVisibleTextValid,
		"off_stop_to_cleanup_complete_ms":                      offCleanup, "shadow_stop_to_cleanup_complete_ms": shadowCleanup,
		"cleanup_complete_delta_shadow_minus_off_ms": result.CleanupDeltaMS,
		"paired_successful_sample":                   result.Successful,
		"steady_state_sample":                        result.SteadyState,
		"cold_start_pair":                            result.ColdStart,
		"parity_sample_valid":                        result.ParityValid,
		"off_stop_to_delivery_requested_ms":          offDeliveryRequest, "shadow_stop_to_delivery_requested_ms": shadowDeliveryRequest,
		"delivery_requested_delta_shadow_minus_off_ms": result.DeliveryRequestDeltaMS,
		"off_stop_to_simulated_delivery_completed_ms":  offDeliveryComplete, "shadow_stop_to_simulated_delivery_completed_ms": shadowDeliveryComplete,
		"simulated_delivery_completed_delta_shadow_minus_off_ms": result.DeliveryCompleteDeltaMS,
		"off_llm_request_count":                                  len(offData.LLMCalls), "shadow_llm_request_count": len(shadowData.LLMCalls),
		"llm_request_delta": result.RequestDelta,
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
	return result
}

func benchRunSuccessful(data benchRunSnapshot) bool {
	return data.Snapshot.State == session.Done && data.Snapshot.Problem.Code == "" &&
		!data.Snapshot.Problem.Degraded && data.Snapshot.Raw != "" && data.Snapshot.Result != ""
}

func emitBenchSummary(t *testing.T, scenario benchScenario, treatment string, pairs []benchPairResult) {
	t.Helper()
	finalDeltas, requestStartDeltas, firstTokenDeltas, visibleTextDeltas := []float64{}, []float64{}, []float64{}, []float64{}
	cleanupDeltas, deliveryRequestDeltas, deliveryCompleteDeltas := []float64{}, []float64{}, []float64{}
	transcriptMatches, resultMatches, requestDelta, successfulPairs, failedPairs, parityPairs, coldStartPairs := 0, 0, 0, 0, 0, 0, 0
	for _, pair := range pairs {
		if pair.ColdStart {
			coldStartPairs++
		}
		if !pair.SteadyState {
			continue
		}
		if !pair.Successful {
			failedPairs++
			continue
		}
		successfulPairs++
		if pair.ParityValid {
			parityPairs++
			if pair.TranscriptEqual {
				transcriptMatches++
			}
			if pair.ResultEqual {
				resultMatches++
			}
		}
		requestDelta += pair.RequestDelta
		if pair.FinalValid {
			finalDeltas = append(finalDeltas, pair.FinalDeltaMS)
		}
		if pair.RequestStartValid {
			requestStartDeltas = append(requestStartDeltas, pair.RequestStartMS)
		}
		if pair.FirstTokenValid {
			firstTokenDeltas = append(firstTokenDeltas, pair.FirstTokenMS)
		}
		if pair.FirstVisibleTextValid {
			visibleTextDeltas = append(visibleTextDeltas, pair.FirstVisibleTextMS)
		}
		if pair.CleanupValid {
			cleanupDeltas = append(cleanupDeltas, pair.CleanupDeltaMS)
		}
		if pair.DeliveryRequestValid {
			deliveryRequestDeltas = append(deliveryRequestDeltas, pair.DeliveryRequestDeltaMS)
		}
		if pair.DeliveryCompleteValid {
			deliveryCompleteDeltas = append(deliveryCompleteDeltas, pair.DeliveryCompleteDeltaMS)
		}
	}
	row := map[string]any{
		"event": "final_candidate_benchmark_summary", "scenario": scenario.Name, "treatment_arm": treatment,
		"paired_sample_count": len(pairs), "paired_successful_sample_count": successfulPairs,
		"failed_or_degraded_pair_count": failedPairs, "cold_start_pair_count": coldStartPairs,
		"steady_state_sample_count": successfulPairs + failedPairs, "parity_sample_count": parityPairs,
		"transcript_match_count":     transcriptMatches,
		"cleanup_result_match_count": resultMatches, "llm_request_delta_total": requestDelta,
		"final_available_delta_ms":              distribution(finalDeltas),
		"first_llm_request_delta_ms":            distribution(requestStartDeltas),
		"first_provider_token_delta_ms":         distribution(firstTokenDeltas),
		"first_visible_cleanup_text_delta_ms":   distribution(visibleTextDeltas),
		"cleanup_complete_delta_ms":             distribution(cleanupDeltas),
		"delivery_requested_delta_ms":           distribution(deliveryRequestDeltas),
		"simulated_delivery_completed_delta_ms": distribution(deliveryCompleteDeltas),
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
}

func distribution(values []float64) map[string]any {
	if len(values) == 0 {
		return map[string]any{"sample_count": 0, "p50": -1, "p95": -1}
	}
	sort.Float64s(values)
	percentile := func(fraction float64) float64 {
		index := int(float64(len(values))*fraction+0.999999) - 1
		return values[max(0, min(index, len(values)-1))]
	}
	return map[string]any{"sample_count": len(values), "p50": percentile(0.50), "p95": percentile(0.95)}
}
