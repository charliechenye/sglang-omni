package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"
)

type benchCandidateCoverage struct {
	ProtocolReported           bool           `json:"protocol_reported"`
	ProtocolEnabled            bool           `json:"protocol_enabled"`
	Cadence                    string         `json:"cadence,omitempty"`
	DecodePath                 string         `json:"decode_path,omitempty"`
	ExclusionReasons           []string       `json:"exclusion_reasons,omitempty"`
	HTTPObservedCount          int            `json:"http_observed_candidate_count"`
	WebSocketObservedCount     int            `json:"websocket_observed_candidate_count"`
	SuppressedCount            int            `json:"suppressed_candidate_count"`
	SuppressionReasons         map[string]int `json:"suppression_reasons,omitempty"`
	EligibilityCount           int            `json:"eligibility_observation_count"`
	ObservationsCapped         bool           `json:"observations_capped"`
	ObservationsDropped        uint64         `json:"observations_dropped"`
	ChannelCoalesced           uint64         `json:"candidate_channel_coalesced"`
	ChannelPending             int            `json:"candidate_channel_pending"`
	CandidateProcessingMSTotal float64        `json:"candidate_processing_ms_total"`
	CandidateGateMSTotal       float64        `json:"candidate_gate_ms_total"`
	processingSamples          []float64
	gateSamples                []float64
}

func candidateNativeStats(events []json.RawMessage) (int, uint64, float64, float64, float64) {
	receivedCount := 0
	publishedCount := uint64(0)
	buildMSTotal := float64(-1)
	buildMSReceived := float64(0)
	firstTokenMS := float64(-1)
	for _, raw := range events {
		var event struct {
			Event          string  `json:"event"`
			Source         string  `json:"source"`
			FirstToken     float64 `json:"first_transcript_token_elapsed_ms"`
			SnapshotBuild  float64 `json:"snapshot_build_ms"`
			PublishedCount uint64  `json:"published_candidate_count"`
			BuildMSTotal   float64 `json:"snapshot_build_ms_total"`
		}
		if json.Unmarshal(raw, &event) != nil || event.Event != "voxt_final_candidate_observed" || event.Source != "http_final" {
			continue
		}
		receivedCount++
		buildMSReceived += event.SnapshotBuild
		if event.PublishedCount > publishedCount {
			publishedCount = event.PublishedCount
		}
		if event.BuildMSTotal > buildMSTotal {
			buildMSTotal = event.BuildMSTotal
		}
		if firstTokenMS < 0 && event.FirstToken > 0 {
			firstTokenMS = event.FirstToken
		}
	}
	return receivedCount, publishedCount, buildMSTotal, buildMSReceived, firstTokenMS
}

func candidateProtocolCoverage(events []json.RawMessage) benchCandidateCoverage {
	coverage := benchCandidateCoverage{SuppressionReasons: make(map[string]int)}
	exclusionReasons := make(map[string]bool)
	for _, raw := range events {
		var event struct {
			Event               string  `json:"event"`
			Source              string  `json:"source"`
			Reason              string  `json:"reason"`
			Cadence             string  `json:"cadence"`
			DecodePath          string  `json:"decode_path"`
			Enabled             bool    `json:"enabled"`
			ProcessingMS        float64 `json:"candidate_processing_ms"`
			GateMS              float64 `json:"candidate_gate_ms"`
			ObservationsCapped  bool    `json:"observations_capped"`
			ObservationsDropped uint64  `json:"observations_dropped"`
			ChannelCoalesced    uint64  `json:"candidate_channel_coalesced"`
			ChannelPending      int     `json:"candidate_channel_pending"`
		}
		if json.Unmarshal(raw, &event) != nil {
			continue
		}
		switch event.Event {
		case "voxt_final_candidate_protocol":
			coverage.ProtocolReported = true
			coverage.ProtocolEnabled = event.Enabled
			coverage.Cadence, coverage.DecodePath = event.Cadence, event.DecodePath
			if event.Reason != "" {
				exclusionReasons[event.Reason] = true
			}
		case "voxt_final_candidates_excluded":
			if event.Reason != "" {
				exclusionReasons[event.Reason] = true
			}
		case "voxt_final_candidate_observed":
			switch event.Source {
			case "http_final":
				coverage.HTTPObservedCount++
			case "websocket_preview", "websocket_candidate":
				coverage.WebSocketObservedCount++
			}
		case "voxt_final_candidate_suppressed":
			coverage.SuppressedCount++
			reason := event.Reason
			if reason == "" {
				reason = "unspecified"
			}
			coverage.SuppressionReasons[reason]++
		case "voxt_final_candidate_observation_capped":
			coverage.ObservationsCapped = true
		case "voxt_final_candidate_eligibility":
			coverage.EligibilityCount++
			if event.ProcessingMS >= 0 {
				coverage.CandidateProcessingMSTotal += event.ProcessingMS
				coverage.processingSamples = append(coverage.processingSamples, event.ProcessingMS)
			}
			if event.GateMS >= 0 {
				coverage.CandidateGateMSTotal += event.GateMS
				coverage.gateSamples = append(coverage.gateSamples, event.GateMS)
			}
		case "voxt_final_candidate_summary":
			coverage.ObservationsCapped = coverage.ObservationsCapped || event.ObservationsCapped
			coverage.ObservationsDropped += event.ObservationsDropped
			coverage.ChannelCoalesced += event.ChannelCoalesced
			coverage.ChannelPending += event.ChannelPending
		}
	}
	for reason := range exclusionReasons {
		coverage.ExclusionReasons = append(coverage.ExclusionReasons, reason)
	}
	sort.Strings(coverage.ExclusionReasons)
	return coverage
}

func emitCandidateCoverageSummary(t *testing.T, runs []*benchRun) {
	t.Helper()
	type coverageKey struct {
		arm         string
		language    string
		decodePath  string
		sampleClass string
	}
	type coverageSummary struct {
		sessions             int
		protocolReported     int
		protocolEnabled      int
		candidateSessions    int
		noCandidateSessions  int
		httpCandidates       int
		websocketCandidates  int
		suppressedCandidates int
		observationsDropped  uint64
		channelCoalesced     uint64
		exclusionReasons     map[string]int
		suppressionReasons   map[string]int
		processingSamples    []float64
		gateSamples          []float64
	}
	groups := make(map[coverageKey]*coverageSummary)
	for _, run := range runs {
		data := run.snapshot()
		coverage := candidateProtocolCoverage(data.ExperimentLogs)
		decodePath := coverage.DecodePath
		if decodePath == "" {
			decodePath = "unreported"
		}
		language := "explicit"
		if strings.TrimSpace(data.Scenario.Language) == "" {
			language = "automatic"
		}
		sampleClass := "steady_state"
		if data.ColdStart {
			sampleClass = "cold_start"
		} else if data.Warmup {
			sampleClass = "warmup"
		}
		key := coverageKey{arm: data.Arm, language: language, decodePath: decodePath, sampleClass: sampleClass}
		group := groups[key]
		if group == nil {
			group = &coverageSummary{exclusionReasons: make(map[string]int), suppressionReasons: make(map[string]int)}
			groups[key] = group
		}
		group.sessions++
		if coverage.ProtocolReported {
			group.protocolReported++
		}
		if coverage.ProtocolEnabled {
			group.protocolEnabled++
			if coverage.HTTPObservedCount == 0 {
				group.noCandidateSessions++
			}
		}
		if coverage.HTTPObservedCount > 0 {
			group.candidateSessions++
		}
		group.httpCandidates += coverage.HTTPObservedCount
		group.websocketCandidates += coverage.WebSocketObservedCount
		group.suppressedCandidates += coverage.SuppressedCount
		group.observationsDropped += coverage.ObservationsDropped
		group.channelCoalesced += coverage.ChannelCoalesced
		group.processingSamples = append(group.processingSamples, coverage.processingSamples...)
		group.gateSamples = append(group.gateSamples, coverage.gateSamples...)
		for _, reason := range coverage.ExclusionReasons {
			group.exclusionReasons[reason]++
		}
		for reason, count := range coverage.SuppressionReasons {
			group.suppressionReasons[reason] += count
		}
	}
	keys := make([]coverageKey, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(left, right int) bool {
		if keys[left].arm != keys[right].arm {
			return keys[left].arm < keys[right].arm
		}
		if keys[left].language != keys[right].language {
			return keys[left].language < keys[right].language
		}
		if keys[left].decodePath != keys[right].decodePath {
			return keys[left].decodePath < keys[right].decodePath
		}
		return keys[left].sampleClass < keys[right].sampleClass
	})
	for _, key := range keys {
		group := groups[key]
		row := map[string]any{
			"event": "final_candidate_benchmark_coverage_summary",
			"arm":   key.arm, "language_mode": key.language, "decode_path": key.decodePath,
			"sample_class":   key.sampleClass,
			"total_sessions": group.sessions, "protocol_reported_sessions": group.protocolReported,
			"protocol_enabled_sessions":     group.protocolEnabled,
			"sessions_with_candidates":      group.candidateSessions,
			"sessions_without_candidates":   group.noCandidateSessions,
			"http_candidates_observed":      group.httpCandidates,
			"websocket_candidates_observed": group.websocketCandidates,
			"suppressed_candidates":         group.suppressedCandidates,
			"suppression_reasons":           group.suppressionReasons,
			"protocol_exclusion_reasons":    group.exclusionReasons,
			"observations_dropped":          group.observationsDropped,
			"candidate_channel_coalesced":   group.channelCoalesced,
			"candidate_processing_ms":       distribution(group.processingSamples),
			"candidate_gate_ms":             distribution(group.gateSamples),
		}
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("FINAL_CANDIDATE_BENCH %s\n", encoded)
	}
}
