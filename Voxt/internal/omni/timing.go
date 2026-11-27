package omni

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

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
		"event": event, "session_id": candidate.metadata.SessionID,
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
