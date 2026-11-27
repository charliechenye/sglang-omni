package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

type logEvent struct {
	Event                       string  "json:\"event\""
	SessionID                   uint64  "json:\"session_id\""
	RequestID                   uint64  "json:\"request_id\""
	Attempt                     int     "json:\"attempt\""
	Source                      string  "json:\"source\""
	Sequence                    uint64  "json:\"sequence\""
	GeneratedTokenCount         int     "json:\"generated_token_count\""
	StopToCandidateMS           float64 "json:\"stop_to_candidate_ms\""
	CandidateEligible           bool    "json:\"candidate_eligible\""
	LatestPreviewMatch          bool    "json:\"latest_preview_match\""
	ExtendsLatestHypothesis     bool    "json:\"extends_latest_hypothesis\""
	MeaningfulTextBoundary      bool    "json:\"meaningful_text_boundary\""
	CandidateLeadMS             float64 "json:\"candidate_lead_ms\""
	OracleCompatible            bool    "json:\"oracle_compatible\""
	OracleIncrementalCompatible bool    "json:\"oracle_incremental_compatible\""
	ExactTranscriptMatch        bool    "json:\"exact_transcript_match\""
	ExactCleanupPlanMatch       bool    "json:\"exact_cleanup_plan_match\""
	NearCleanupPlanCompatible   bool    "json:\"near_cleanup_plan_compatible\""
	CleanupRepairEligible       bool    "json:\"cleanup_repair_eligible\""
	CandidateUsableBeforeFinal  bool    "json:\"candidate_usable_before_final\""
}

type candidateKey struct {
	sessionID uint64
	requestID uint64
	attempt   int
	source    string
	sequence  uint64
}

type candidate struct {
	eligibility logEvent
	outcome     logEvent
	hasOutcome  bool
}

type policy struct {
	name      string
	threshold float64
	matches   func(logEvent) bool
}

type policyResult struct {
	Name                           string  "json:\"name\""
	Threshold                      float64 "json:\"threshold,omitempty\""
	EligibleCandidateCount         int     "json:\"eligible_candidate_count\""
	CandidateSessionCount          int     "json:\"candidate_session_count\""
	SelectedSessionCount           int     "json:\"selected_session_count\""
	ResolvedSelectionCount         int     "json:\"resolved_selection_count\""
	OracleCompatibleSelectionCount int     "json:\"oracle_compatible_selection_count\""
	ExactTranscriptMatchCount      int     "json:\"exact_transcript_match_count\""
	ExactCleanupPlanMatchCount     int     "json:\"exact_cleanup_plan_match_count\""
	NearCleanupRepairCount         int     "json:\"near_cleanup_repair_count\""
	WastedIfDispatchedCount        int     "json:\"wasted_if_dispatched_count\""
	LeadSampleCount                int     "json:\"lead_sample_count\""
	LeadP50MS                      float64 "json:\"lead_p50_ms\""
	LeadP95MS                      float64 "json:\"lead_p95_ms\""
}

type replayReport struct {
	FormatVersion                   int            "json:\"format_version\""
	SessionCount                    int            "json:\"session_count\""
	ReceivedCandidateCount          int            "json:\"received_candidate_count\""
	SessionEligibilityCount         int            "json:\"session_eligibility_count\""
	ObservedWithoutEligibilityCount int            "json:\"observed_without_eligibility_count\""
	OracleCompatibleCandidateCount  int            "json:\"oracle_compatible_candidate_count\""
	OracleIncrementalCandidateCount int            "json:\"oracle_incremental_candidate_count\""
	Policies                        []policyResult "json:\"policies\""
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}

func run(arguments []string, output io.Writer) error {
	flags := flag.NewFlagSet("final-candidate-replay", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	inputPath := flags.String("input", "-", "Voxt JSON log or benchmark JSONL file, or - for stdin")
	tokenCadences := flags.String("token-cadences", "8,16,32", "minimum committed tokens before an HTTP candidate can trigger")
	timeCadences := flags.String("time-cadences-ms", "250,500,1000", "minimum milliseconds after Stop before an HTTP candidate can trigger")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	tokenThresholds, err := parseThresholds(*tokenCadences)
	if err != nil {
		return fmt.Errorf("token cadences: %w", err)
	}
	timeThresholds, err := parseThresholds(*timeCadences)
	if err != nil {
		return fmt.Errorf("time cadences: %w", err)
	}
	reader, closeReader, err := openInput(*inputPath)
	if err != nil {
		return err
	}
	defer closeReader()
	report, err := replay(reader, tokenThresholds, timeThresholds)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}

func openInput(path string) (io.Reader, func(), error) {
	if path == "-" {
		return os.Stdin, func() {}, nil
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, func() {}, err
	}
	return file, func() { _ = file.Close() }, nil
}

func parseThresholds(raw string) ([]int, error) {
	var thresholds []int
	for _, part := range strings.Split(raw, ",") {
		threshold, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || threshold <= 0 {
			return nil, fmt.Errorf("expected positive comma-separated integers, got %q", raw)
		}
		thresholds = append(thresholds, threshold)
	}
	sort.Ints(thresholds)
	return thresholds, nil
}

func replay(reader io.Reader, tokenThresholds, timeThresholds []int) (replayReport, error) {
	sessions := make(map[uint64][]candidate)
	outcomes := make(map[candidateKey]logEvent)
	seenEligibility := make(map[candidateKey]bool)
	sessionIDs := make(map[uint64]bool)
	receivedCandidates := 0
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	for scanner.Scan() {
		line := scanner.Text()
		start := strings.IndexByte(line, '{')
		if start < 0 {
			continue
		}
		var event logEvent
		if err := json.Unmarshal([]byte(line[start:]), &event); err != nil {
			continue
		}
		var events []logEvent
		if event.Event == "final_candidate_benchmark_run" {
			var benchmark struct {
				ExperimentEvents []json.RawMessage `json:"experiment_events"`
			}
			if json.Unmarshal([]byte(line[start:]), &benchmark) == nil {
				for _, rawEvent := range benchmark.ExperimentEvents {
					var nested logEvent
					if json.Unmarshal(rawEvent, &nested) == nil {
						events = append(events, nested)
					}
				}
			}
		} else {
			events = append(events, event)
		}
		for _, event := range events {
			key := candidateKey{
				sessionID: event.SessionID, requestID: event.RequestID,
				attempt: event.Attempt, source: event.Source, sequence: event.Sequence,
			}
			switch event.Event {
			case "voxt_final_candidate_observed":
				receivedCandidates++
				sessionIDs[event.SessionID] = true
			case "voxt_final_candidate_eligibility":
				candidate := candidate{eligibility: event}
				sessions[event.SessionID] = append(sessions[event.SessionID], candidate)
				seenEligibility[key] = true
				sessionIDs[event.SessionID] = true
			case "voxt_final_candidate_outcome":
				outcomes[key] = event
			case "voxt_final_candidate_summary":
				sessionIDs[event.SessionID] = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return replayReport{}, err
	}
	if len(sessionIDs) == 0 && receivedCandidates == 0 {
		return replayReport{}, errors.New("no final-candidate events found in input")
	}
	for sessionID, candidates := range sessions {
		for index := range candidates {
			key := candidateKey{
				sessionID: candidates[index].eligibility.SessionID,
				requestID: candidates[index].eligibility.RequestID,
				attempt:   candidates[index].eligibility.Attempt,
				source:    candidates[index].eligibility.Source,
				sequence:  candidates[index].eligibility.Sequence,
			}
			if outcome, ok := outcomes[key]; ok {
				candidates[index].outcome = outcome
				candidates[index].hasOutcome = true
			}
		}
		sort.SliceStable(candidates, func(left, right int) bool {
			if candidates[left].eligibility.StopToCandidateMS == candidates[right].eligibility.StopToCandidateMS {
				return candidates[left].eligibility.Sequence < candidates[right].eligibility.Sequence
			}
			return candidates[left].eligibility.StopToCandidateMS < candidates[right].eligibility.StopToCandidateMS
		})
		sessions[sessionID] = candidates
	}
	policies := []policy{
		{name: "first_eligible_http", matches: func(event logEvent) bool { return event.Source == "http_final" }},
		{name: "latest_preview_match", matches: func(event logEvent) bool {
			return event.Source == "http_final" && event.LatestPreviewMatch
		}},
		{name: "meaningful_boundary", matches: func(event logEvent) bool {
			return event.Source == "http_final" && event.MeaningfulTextBoundary
		}},
		{name: "extends_latest_hypothesis", matches: func(event logEvent) bool {
			return event.Source == "http_final" && event.ExtendsLatestHypothesis
		}},
		{name: "websocket_control", matches: func(event logEvent) bool {
			return event.Source == "websocket_preview"
		}},
	}
	for _, threshold := range tokenThresholds {
		policies = append(policies, policy{
			name: "token_cadence", threshold: float64(threshold),
			matches: func(event logEvent) bool {
				return event.Source == "http_final" && event.GeneratedTokenCount >= threshold
			},
		})
	}
	for _, threshold := range timeThresholds {
		policies = append(policies, policy{
			name: "time_cadence_ms", threshold: float64(threshold),
			matches: func(event logEvent) bool {
				return event.Source == "http_final" && event.StopToCandidateMS >= float64(threshold)
			},
		})
	}
	report := replayReport{
		FormatVersion: 1, SessionCount: len(sessionIDs),
		ReceivedCandidateCount:          receivedCandidates,
		SessionEligibilityCount:         len(seenEligibility),
		ObservedWithoutEligibilityCount: max(0, receivedCandidates-len(seenEligibility)),
	}
	for _, outcome := range outcomes {
		if outcome.OracleCompatible {
			report.OracleCompatibleCandidateCount++
		}
		if outcome.OracleIncrementalCompatible {
			report.OracleIncrementalCandidateCount++
		}
	}
	for _, current := range policies {
		report.Policies = append(report.Policies, evaluatePolicy(current, sessions))
	}
	return report, nil
}

func evaluatePolicy(current policy, sessions map[uint64][]candidate) policyResult {
	result := policyResult{Name: current.name, Threshold: current.threshold}
	var leads []float64
	for _, candidates := range sessions {
		hasCandidate := false
		for _, currentCandidate := range candidates {
			event := currentCandidate.eligibility
			if currentCandidate.hasOutcome &&
				currentCandidate.outcome.CandidateUsableBeforeFinal &&
				event.CandidateEligible && current.matches(event) {
				result.EligibleCandidateCount++
				hasCandidate = true
			}
		}
		if !hasCandidate {
			continue
		}
		result.CandidateSessionCount++
		for _, currentCandidate := range candidates {
			event := currentCandidate.eligibility
			if !currentCandidate.hasOutcome ||
				!currentCandidate.outcome.CandidateUsableBeforeFinal ||
				!event.CandidateEligible || !current.matches(event) {
				continue
			}
			result.SelectedSessionCount++
			if currentCandidate.hasOutcome {
				result.ResolvedSelectionCount++
				outcome := currentCandidate.outcome
				if outcome.OracleCompatible {
					result.OracleCompatibleSelectionCount++
				}
				if outcome.ExactTranscriptMatch {
					result.ExactTranscriptMatchCount++
				}
				if outcome.ExactCleanupPlanMatch {
					result.ExactCleanupPlanMatchCount++
				}
				if outcome.NearCleanupPlanCompatible && outcome.CleanupRepairEligible {
					result.NearCleanupRepairCount++
				}
				if !outcome.OracleIncrementalCompatible {
					result.WastedIfDispatchedCount++
				}
				if outcome.CandidateLeadMS > 0 {
					leads = append(leads, outcome.CandidateLeadMS)
				}
			}
			break
		}
	}
	result.LeadSampleCount = len(leads)
	result.LeadP50MS = percentile(leads, 0.50)
	result.LeadP95MS = percentile(leads, 0.95)
	return result
}

func percentile(values []float64, fraction float64) float64 {
	if len(values) == 0 {
		return -1
	}
	sort.Float64s(values)
	index := int(float64(len(values))*fraction + 0.999999)
	return values[min(len(values)-1, max(0, index-1))]
}
