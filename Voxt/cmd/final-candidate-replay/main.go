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
	sessions := make(map[string]*sessionTrace)
	seenCandidates := 0
	seenEligibility := 0
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	for scanner.Scan() {
		line := scanner.Text()
		start := strings.IndexByte(line, '{')
		if start < 0 {
			continue
		}
		var envelope map[string]json.RawMessage
		if json.Unmarshal([]byte(line[start:]), &envelope) != nil {
			continue
		}
		var eventType string
		_ = json.Unmarshal(envelope["event"], &eventType)
		var events []logEvent
		if eventType == "final_candidate_benchmark_run" {
			var benchmark struct {
				Warmup           bool              `json:"warmup"`
				ColdStart        bool              `json:"cold_start"`
				ExperimentEvents []json.RawMessage `json:"experiment_events"`
			}
			if json.Unmarshal([]byte(line[start:]), &benchmark) == nil {
				if benchmark.Warmup || benchmark.ColdStart {
					continue
				}
				for _, raw := range benchmark.ExperimentEvents {
					var nested logEvent
					if json.Unmarshal(raw, &nested) == nil {
						events = append(events, nested)
					}
				}
			}
		} else {
			var event logEvent
			if json.Unmarshal([]byte(line[start:]), &event) == nil {
				events = append(events, event)
			}
		}
		for _, event := range events {
			if !replayEvent(event.Event) {
				continue
			}
			key := eventSessionKey(event)
			if key == "" {
				continue
			}
			trace := sessions[key]
			if trace == nil {
				trace = &sessionTrace{
					outcomes:            make(map[candidateKey]logEvent),
					outcomesByIdentity:  make(map[candidateIdentity]logEvent),
					adoptions:           make(map[uint64]logEvent),
					adoptionsByPlanHash: make(map[string]logEvent),
					requestStarts:       make(map[string]logEvent),
					requestStartsByID:   make(map[uint64]logEvent),
					requestResults:      make(map[string]logEvent),
					requestResultsByID:  make(map[uint64]logEvent),
					finalAvailableMS:    -1, cancelledAtMS: -1,
				}
				sessions[key] = trace
			}
			switch event.Event {
			case "voxt_final_candidate_observed":
				seenCandidates++
				trace.observedCandidateOnly++
			case "voxt_final_candidate_eligibility":
				seenEligibility++
				trace.candidates = append(trace.candidates, candidate{eligibility: event})
				if event.RecordingGenerations > trace.recordingGenerations || !trace.recordingGensKnown {
					trace.recordingGenerations = event.RecordingGenerations
					trace.recordingGensKnown = true
				}
			case "voxt_headstart_decision":
				trace.candidates = append(trace.candidates, candidate{eligibility: event, production: true})
				if event.RecordingGenerations > trace.recordingGenerations || !trace.recordingGensKnown {
					trace.recordingGenerations = event.RecordingGenerations
					trace.recordingGensKnown = true
				}
			case "voxt_final_candidate_outcome":
				trace.outcomes[eventKey(key, event)] = event
				if event.CandidatePlanHash != "" {
					identity := candidateIdentity{
						session: key, planSHA: event.CandidatePlanHash,
						candidateSHA: event.TranscriptSHA256,
					}
					trace.outcomesByIdentity[identity] = event
				}
			case "voxt_headstart_adoption":
				trace.adoptions[event.RequestID] = event
				trace.adoptionsByPlanHash[event.RequestPlanHash] = event
			case "voxt_headstart_request_start":
				if previous, ok := trace.requestStarts[event.RequestPlanHash]; !ok || event.RequestStartMS < previous.RequestStartMS {
					trace.requestStarts[event.RequestPlanHash] = event
				}
				trace.requestStartsByID[event.RequestID] = event
			case "voxt_headstart_request_done":
				trace.requestResults[event.RequestPlanHash] = event
				trace.requestResultsByID[event.RequestID] = event
			case "voxt_final_candidate_summary":
				if event.FinalStatus != "cancelled" && event.FinalStatus != "failed" && event.FinalAvailableMS >= 0 {
					trace.finalAvailableMS = event.FinalAvailableMS
					trace.finalAvailableKnown = true
				}
				if event.FinalStatus == "cancelled" {
					trace.cancelledAtMS = event.CancellationMS
					trace.cancellationKnown = event.CancellationMS >= 0
				}
			case "voxt_session_cancelled":
				trace.cancelledAtMS = event.CancellationMS
				trace.cancellationKnown = event.CancellationMS >= 0
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return replayReport{}, err
	}
	if len(sessions) == 0 {
		return replayReport{}, errors.New("no final-candidate events found in input")
	}
	for sessionKey, trace := range sessions {
		for index := range trace.candidates {
			current := &trace.candidates[index]
			if current.production {
				identity := candidateIdentity{
					session: sessionKey, planSHA: current.eligibility.RequestPlanHash,
					candidateSHA: current.eligibility.CandidateSHA256,
				}
				if outcome, ok := trace.outcomesByIdentity[identity]; ok {
					current.outcome = outcome
					current.hasOutcome = true
				}
				resolveActualWebSocketOutcome(trace, current)
			} else {
				key := eventKeyForCandidate(sessionKey, current)
				if outcome, ok := trace.outcomes[key]; ok {
					current.outcome = outcome
					current.hasOutcome = true
				}
			}
		}
		sort.SliceStable(trace.candidates, func(left, right int) bool {
			leftAt := candidateReadyMS(trace.candidates[left])
			rightAt := candidateReadyMS(trace.candidates[right])
			if leftAt == rightAt {
				return trace.candidates[left].eligibility.Sequence < trace.candidates[right].eligibility.Sequence
			}
			return leftAt < rightAt
		})
	}
	policies := buildPolicies(tokenThresholds, timeThresholds)
	results := make(map[string]policyResult, len(policies))
	for _, current := range policies {
		result := evaluatePolicy(current, sessions)
		results[current.name+":"+formatThreshold(current.threshold)] = result
	}
	baseline := results["current_websocket_baseline:0"]
	for index := range policies {
		current := policies[index]
		if !current.combined {
			continue
		}
		key := current.name + ":" + formatThreshold(current.threshold)
		result := results[key]
		compareToBaseline(&result, current, sessions)
		results[key] = result
	}
	_ = baseline
	report := replayReport{FormatVersion: 2, SessionCount: len(sessions), ReceivedCandidateCount: seenCandidates, SessionEligibilityCount: seenEligibility}
	if seenCandidates > seenEligibility {
		report.ObservedWithoutEligibilityCount = seenCandidates - seenEligibility
	}
	for _, trace := range sessions {
		for _, event := range trace.outcomes {
			if event.OracleCompatible {
				report.OracleCompatibleCandidateCount++
			}
			if event.OracleIncrementalCompatible {
				report.OracleIncrementalCandidateCount++
			}
		}
		for _, current := range trace.candidates {
			if !current.hasOutcome {
				report.UnresolvedOutcomeCount++
			}
		}
	}
	for _, current := range policies {
		report.Policies = append(report.Policies, results[current.name+":"+formatThreshold(current.threshold)])
	}
	return report, nil
}

func eventSessionKey(event logEvent) string {
	if event.SessionID != 0 {
		return fmt.Sprintf("session:%d", event.SessionID)
	}
	if event.RunID != "" {
		return "run:" + event.RunID
	}
	return ""
}

func replayEvent(event string) bool {
	switch event {
	case "voxt_final_candidate_observed", "voxt_final_candidate_eligibility",
		"voxt_final_candidate_outcome", "voxt_final_candidate_summary",
		"voxt_headstart_decision", "voxt_headstart_adoption",
		"voxt_headstart_request_start", "voxt_headstart_request_done", "voxt_session_cancelled":
		return true
	default:
		return false
	}
}
