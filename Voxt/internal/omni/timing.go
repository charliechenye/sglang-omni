package omni

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

func requestedLanguageMode(language string) string {
	if strings.TrimSpace(language) == "" {
		return "automatic"
	}
	return "explicit"
}

func logFinalCandidateProtocolResult(rt *Runtime, measurement *candidateMeasurement,
	requestID uint64, request finalRequest, result finalResult,
	nativeSSEReceived, goSuppressed uint64, snapshotBuildMSTotal float64) {
	payload, err := json.Marshal(map[string]any{
		"event":  "voxt_final_candidate_protocol_result",
		"run_id": measurement.runID, "session_id": measurement.sessionID,
		"request_id": requestID, "requested_path": "authoritative_final_http",
		"candidate_protocol_requested": request.includeCandidates,
		"candidate_protocol_known":     result.candidateProtocolKnown,
		"candidate_protocol_supported": result.candidateProtocolSupported,
		"candidate_cadence":            result.candidateCadence,
		"language_mode":                requestedLanguageMode(request.language),
		"effective_language":           result.language,
		"generated_token_count":        result.tokens, "finish_reason": result.finishReason,
		"verified_guess_token_count": result.verifiedGuessTokens,
		"accepted_guess_token_count": result.acceptedGuessTokens,
		"draft_decoder_used":         result.draftDecoderUsed,
		"verify_round_count":         result.verifyRounds,
		"proposed_draft_token_count": result.proposedDraftTokens,
		"accepted_draft_token_count": result.acceptedDraftTokens,
		"reused_audio_token_count":   result.reusedAudioTokens,
		"reused_prompt_token_count":  result.reusedPromptTokens,
		"decode_paths":               result.decodePaths, "batch_decode_used": result.batchDecodeUsed,
		"native_chunked":                      result.nativeChunked,
		"native_published_candidate_count":    result.nativePublishedCandidates,
		"native_coalesced_candidate_count":    result.nativeCoalescedCandidates,
		"native_sse_candidate_received_count": nativeSSEReceived,
		"go_candidate_suppressed_count":       goSuppressed,
		"native_snapshot_build_ms_total":      snapshotBuildMSTotal,
		"preview_guess_requested":             request.known.text != "" || request.known.guessAutoLanguage,
		"realtime_prefill_reuse_requested":    request.known.session != "",
	})
	if err == nil {
		rt.logf("info", "%s", payload)
	}
}

// requestStage is one content-free timing span in an offline Final request.
type requestStage struct {
	Event       string  `json:"event"`
	RequestID   uint64  `json:"request_id"`
	Kind        string  `json:"kind"`
	Stage       string  `json:"stage"`
	DurationMS  float64 `json:"duration_ms"`
	InputBytes  int     `json:"input_bytes,omitempty"`
	OutputBytes int     `json:"output_bytes,omitempty"`
}

func logCandidateRecord(rt *Runtime, event string, candidate finalCandidate, stopAt time.Time, reason string) {
	hash := sha256.Sum256([]byte(candidate.text))
	stopToCandidateMS := float64(0)
	if !stopAt.IsZero() {
		stopToCandidateMS = float64(candidate.metadata.ReceivedAt.Sub(stopAt).Microseconds()) / 1000
	} else {
	}
	record := map[string]any{
		"event": event, "run_id": candidate.metadata.RunID,
		"session_id": candidate.metadata.SessionID,
		"request_id": candidate.metadata.RequestID, "attempt": candidate.metadata.Attempt,
		"source": candidate.metadata.Source, "sequence": candidate.metadata.Sequence,
		"generated_token_count":             candidate.metadata.GeneratedTokens,
		"native_elapsed_ms":                 candidate.metadata.NativeElapsedMS,
		"first_transcript_token_elapsed_ms": candidate.metadata.FirstTranscriptTokenMS,
		"snapshot_build_ms":                 candidate.metadata.SnapshotBuildMS,
		"published_candidate_count":         candidate.metadata.PublishedCandidateCount,
		"snapshot_build_ms_total":           candidate.metadata.SnapshotBuildMSTotal,
		"request_elapsed_ms":                candidate.metadata.RequestElapsedMS,
		"http_request_start_offset_ms":      candidate.metadata.HTTPRequestStartOffsetMS,
		"stop_to_candidate_ms":              stopToCandidateMS,
		"transcript_sha256":                 hex.EncodeToString(hash[:]),
		"transcript_bytes":                  len(candidate.text),
		"transcript_runes":                  utf8.RuneCountInString(candidate.text),
	}
	if reason != "" {
		record["reason"] = reason
	} else {
	}
	payload, err := json.Marshal(record)
	if err == nil {
		rt.logf("info", "%s", payload)
	} else {
	}
}

func logCandidateExclusion(rt *Runtime, runID string, sessionID uint64, reason string, chunkCount int) {
	payload, err := json.Marshal(map[string]any{
		"event": "voxt_final_candidates_excluded", "run_id": runID,
		"session_id": sessionID, "reason": reason, "chunk_count": chunkCount,
	})
	if err == nil {
		rt.logf("info", "%s", payload)
	}
}

var offlineRequestSequence atomic.Uint64

func nextOfflineRequestID() uint64 { return offlineRequestSequence.Add(1) }

func logRequestStage(rt *Runtime, id uint64, kind, stage string, started time.Time, input, output int) {
	if id == 0 {
		return
	}
	payload, err := json.Marshal(requestStage{
		Event: "omni_request_stage", RequestID: id, Kind: kind, Stage: stage,
		DurationMS: float64(time.Since(started).Microseconds()) / 1000,
		InputBytes: input, OutputBytes: output,
	})
	if err == nil {
		rt.logf("info", "%s", payload)
	}
}
