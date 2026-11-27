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
	armFilter := flags.String("arm", "", "benchmark arm to report: off, shadow, or shadow_dense")
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
	report, err := replayForArm(reader, tokenThresholds, timeThresholds, *armFilter)
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
	return replayForArm(reader, tokenThresholds, timeThresholds, "")
}

func replayForArm(reader io.Reader, tokenThresholds, timeThresholds []int, armFilter string) (replayReport, error) {
	sessions := make(map[string]*sessionTrace)
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
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
		var benchmarkKey string
		if eventType == "final_candidate_benchmark_run" {
			var benchmark struct {
				RunID             string            `json:"run_id"`
				SessionID         uint64            `json:"session_id"`
				Arm               string            `json:"arm"`
				Scenario          string            `json:"scenario"`
				Round             int               `json:"round"`
				AudioSHA256       string            `json:"audio_sha256"`
				LanguageHint      string            `json:"language_hint"`
				LanguageMode      string            `json:"language_mode"`
				EffectiveLanguage string            `json:"effective_language"`
				Draft             bool              `json:"draft"`
				Warmup            bool              `json:"warmup"`
				ColdStart         bool              `json:"cold_start"`
				Status            string            `json:"status"`
				Problem           string            `json:"problem"`
				ExperimentEvents  []json.RawMessage `json:"experiment_events"`
			}
			if json.Unmarshal([]byte(line[start:]), &benchmark) == nil {
				if armFilter != "" && benchmark.Arm != armFilter {
					continue
				}
				runID := benchmark.RunID
				if runID == "" {
					runID = fmt.Sprintf("line-%d", lineNumber)
				}
				arm := benchmark.Arm
				if arm == "" {
					arm = "unclassified"
				}
				benchmarkKey = fmt.Sprintf("benchmark:%s:%s:%s", arm, benchmark.Scenario, runID)
				trace := newSessionTrace()
				trace.arm, trace.scenario, trace.round = arm, benchmark.Scenario, benchmark.Round
				trace.runID, trace.audioSHA256 = runID, benchmark.AudioSHA256
				trace.languageHint, trace.languageMode = benchmark.LanguageHint, benchmark.LanguageMode
				trace.effectiveLanguage, trace.draft = benchmark.EffectiveLanguage, benchmark.Draft
				trace.warmup, trace.coldStart = benchmark.Warmup, benchmark.ColdStart
				trace.runStatus, trace.runProblem, trace.benchmark = benchmark.Status, benchmark.Problem, true
				sessions[benchmarkKey] = trace
				for _, raw := range benchmark.ExperimentEvents {
					var nested logEvent
					if json.Unmarshal(raw, &nested) == nil {
						nested.RunID = benchmarkKey
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
			key := benchmarkKey
			if key == "" {
				key = eventSessionKey(event)
			}
			if key == "" {
				continue
			}
			trace := sessions[key]
			if trace == nil {
				trace = newSessionTrace()
				trace.runID = event.RunID
				trace.arm = "unclassified"
				sessions[key] = trace
			}
			applyReplayEvent(trace, key, event)
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
	if len(sessions) == 0 {
		return replayReport{}, errors.New("no final-candidate sessions found in input")
	}
	return buildReplayReport(sessions, tokenThresholds, timeThresholds, armFilter), nil
}

func newSessionTrace() *sessionTrace {
	return &sessionTrace{
		outcomes:           make(map[candidateKey]logEvent),
		outcomesByIdentity: make(map[candidateIdentity]logEvent),
		adoptions:          make(map[uint64]logEvent),
		requestStartsByID:  make(map[uint64]logEvent),
		requestResultsByID: make(map[uint64]logEvent),
		finalAvailableMS:   -1, cancelledAtMS: -1,
	}
}

func applyReplayEvent(trace *sessionTrace, sessionKey string, event logEvent) {
	switch event.Event {
	case "voxt_final_candidate_observed":
		trace.observedCandidateOnly++
	case "voxt_final_candidate_eligibility":
		trace.candidates = append(trace.candidates, candidate{
			eligibility: event, counterfactualStateUnresolved: event.CounterfactualStateUnresolved,
		})
	case "voxt_headstart_decision":
		trace.candidates = append(trace.candidates, candidate{eligibility: event, production: true})
	case "voxt_final_candidate_outcome":
		trace.outcomes[eventKey(sessionKey, event)] = event
		if event.CandidatePlanHash != "" {
			trace.outcomesByIdentity[candidateIdentity{
				session: sessionKey, planSHA: event.CandidatePlanHash,
				candidateSHA: event.TranscriptSHA256,
			}] = event
		}
	case "voxt_headstart_decision_outcome":
		if event.CandidatePlanHash != "" {
			trace.outcomesByIdentity[candidateIdentity{
				session: sessionKey, planSHA: event.CandidatePlanHash,
				candidateSHA: event.CandidateSHA256,
			}] = event
		}
	case "voxt_headstart_initial_budget":
		trace.initialBudgetKnown = true
		trace.recordingGenerations = event.InitialRecordingGenerations
		trace.lateAllowance = event.InitialLateAllowance
		trace.lateAlreadyConsumed = event.InitialLateConsumed
		trace.generationLimit = event.GenerationLimit
		trace.multipartLimit = event.MultipartLimit
	case "voxt_headstart_adoption":
		if event.RequestID != 0 {
			trace.adoptions[event.RequestID] = event
		}
	case "voxt_headstart_request_start":
		if event.RequestID != 0 {
			if previous, ok := trace.requestStartsByID[event.RequestID]; !ok || event.RequestStartMS < previous.RequestStartMS {
				trace.requestStartsByID[event.RequestID] = event
			}
		}
	case "voxt_headstart_request_done":
		if event.RequestID != 0 {
			trace.requestResultsByID[event.RequestID] = event
		}
	case "voxt_cleanup_fallback_started":
		trace.fallbackAttemptCount++
	case "voxt_cleanup_fallback_outcome":
		switch event.FallbackOutcome {
		case "succeeded":
			trace.fallbackSuccessCount++
		case "failed":
			trace.fallbackFailureCount++
		}
	case "voxt_final_candidate_summary":
		if event.InitialBudgetKnown != nil && *event.InitialBudgetKnown {
			trace.initialBudgetKnown = true
			trace.recordingGenerations = event.InitialRecordingGenerationCount
			trace.lateAllowance = event.InitialLateAllowance
			trace.lateAlreadyConsumed = event.InitialLateConsumedSummary
			trace.generationLimit = event.GenerationLimit
			trace.multipartLimit = event.MultipartLimit
		}
		if event.EffectiveLanguage != "" {
			trace.effectiveLanguage = event.EffectiveLanguage
		}
		if event.FinalStatus != "cancelled" && event.FinalStatus != "asr_failure" && event.FinalAvailableMS >= 0 {
			trace.finalAvailableMS = event.FinalAvailableMS
			trace.finalAvailableKnown = true
		}
		if event.FinalStatus == "cancelled" {
			trace.cancelledAtMS = event.CancellationMS
			trace.cancellationKnown = event.CancellationMS >= 0
		}
	case "voxt_final_candidate_protocol_result":
		if event.EffectiveLanguage != "" {
			trace.effectiveLanguage = event.EffectiveLanguage
		}
	case "voxt_session_cancelled":
		trace.cancelledAtMS = event.CancellationMS
		trace.cancellationKnown = event.CancellationMS >= 0
	}
}

func buildReplayReport(sessions map[string]*sessionTrace, tokenThresholds, timeThresholds []int, armFilter string) replayReport {
	byArm := make(map[string]map[string]*sessionTrace)
	for key, trace := range sessions {
		arm := trace.arm
		if arm == "" {
			arm = "unclassified"
		}
		if byArm[arm] == nil {
			byArm[arm] = make(map[string]*sessionTrace)
		}
		byArm[arm][key] = trace
	}
	armNames := make([]string, 0, len(byArm))
	for arm := range byArm {
		armNames = append(armNames, arm)
	}
	sort.Strings(armNames)
	report := replayReport{FormatVersion: 3}
	for _, arm := range armNames {
		current := buildArmReport(arm, byArm[arm], tokenThresholds, timeThresholds)
		report.SessionCount += current.SessionCount
		addCoverageCounts(&report, byArm[arm])
		report.Arms = append(report.Arms, current)
	}
	if armFilter != "" {
		for _, current := range report.Arms {
			if current.Arm == armFilter {
				report.SelectedArm = current.Arm
				report.SessionCount = current.SessionCount
				report.Policies = current.Policies
				report.ReceivedCandidateCount = 0
				report.SessionEligibilityCount = 0
				report.ObservedWithoutEligibilityCount = 0
				report.OracleCompatibleCandidateCount = 0
				report.OracleIncrementalCandidateCount = 0
				report.UnresolvedOutcomeCount = 0
				addCoverageCounts(&report, byArm[current.Arm])
				return report
			}
		}
	}
	if len(report.Arms) == 1 && report.Arms[0].Arm == "unclassified" {
		report.Policies = report.Arms[0].Policies
	}
	return report
}

func addCoverageCounts(report *replayReport, traces map[string]*sessionTrace) {
	for _, trace := range traces {
		report.ReceivedCandidateCount += trace.observedCandidateOnly
		report.SessionEligibilityCount += len(trace.candidates)
		for _, event := range trace.outcomes {
			if event.OracleCompatible {
				report.OracleCompatibleCandidateCount++
			}
			if event.OracleIncrementalCompatible {
				report.OracleIncrementalCandidateCount++
			}
		}
		for _, current := range trace.candidates {
			if !current.hasOutcome || current.outcome.OutcomeUnresolved {
				report.UnresolvedOutcomeCount++
			}
		}
	}
	if report.ReceivedCandidateCount > report.SessionEligibilityCount {
		report.ObservedWithoutEligibilityCount = report.ReceivedCandidateCount - report.SessionEligibilityCount
	}
}

func buildArmReport(arm string, traces map[string]*sessionTrace, tokenThresholds, timeThresholds []int) armReport {
	steady := make(map[string]*sessionTrace)
	scenarios := make(map[string]map[string]*sessionTrace)
	report := armReport{Arm: arm, SessionCount: len(traces)}
	for key, trace := range traces {
		if trace.warmup {
			report.WarmupSessionCount++
		} else if trace.coldStart {
			report.ColdStartCount++
		} else {
			report.SteadySessionCount++
			steady[key] = trace
		}
		scenario := trace.scenario
		if scenario == "" {
			scenario = "unclassified"
		}
		if scenarios[scenario] == nil {
			scenarios[scenario] = make(map[string]*sessionTrace)
		}
		scenarios[scenario][key] = trace
		report.Runs = append(report.Runs, summarizeSession(trace))
	}
	report.Policies = evaluatePolicies(steady, tokenThresholds, timeThresholds)
	scenarioNames := make([]string, 0, len(scenarios))
	for name := range scenarios {
		scenarioNames = append(scenarioNames, name)
	}
	sort.Strings(scenarioNames)
	for _, name := range scenarioNames {
		all := scenarios[name]
		steadyScenario := make(map[string]*sessionTrace)
		current := scenarioReport{Scenario: name, SessionCount: len(all)}
		for key, trace := range all {
			if trace.warmup {
				current.WarmupSessionCount++
			} else if trace.coldStart {
				current.ColdStartCount++
			} else {
				current.SteadySessionCount++
				steadyScenario[key] = trace
			}
		}
		current.Policies = evaluatePolicies(steadyScenario, tokenThresholds, timeThresholds)
		report.Scenarios = append(report.Scenarios, current)
	}
	sort.Slice(report.Runs, func(i, j int) bool {
		if report.Runs[i].Scenario != report.Runs[j].Scenario {
			return report.Runs[i].Scenario < report.Runs[j].Scenario
		}
		if report.Runs[i].Round != report.Runs[j].Round {
			return report.Runs[i].Round < report.Runs[j].Round
		}
		return report.Runs[i].RunID < report.Runs[j].RunID
	})
	return report
}

func evaluatePolicies(sessions map[string]*sessionTrace, tokenThresholds, timeThresholds []int) []policyResult {
	policies := buildPolicies(tokenThresholds, timeThresholds)
	results := make(map[string]policyResult, len(policies))
	for _, current := range policies {
		results[current.name+":"+formatThreshold(current.threshold)] = evaluatePolicy(current, sessions)
	}
	for _, current := range policies {
		if !current.combined {
			continue
		}
		key := current.name + ":" + formatThreshold(current.threshold)
		result := results[key]
		compareToBaseline(&result, current, sessions)
		results[key] = result
	}
	ordered := make([]policyResult, 0, len(policies))
	for _, current := range policies {
		ordered = append(ordered, results[current.name+":"+formatThreshold(current.threshold)])
	}
	return ordered
}

func summarizeSession(trace *sessionTrace) sessionSummary {
	unresolved := 0
	for _, current := range trace.candidates {
		if !current.hasOutcome || current.outcome.OutcomeUnresolved {
			unresolved++
		}
	}
	return sessionSummary{
		RunID: trace.runID, Arm: trace.arm, Scenario: trace.scenario, Round: trace.round,
		AudioSHA256: trace.audioSHA256, LanguageHint: trace.languageHint, LanguageMode: trace.languageMode,
		EffectiveLanguage: trace.effectiveLanguage, Draft: trace.draft,
		Warmup: trace.warmup, ColdStart: trace.coldStart, Status: trace.runStatus,
		Problem: trace.runProblem, CandidateCount: len(trace.candidates) + trace.observedCandidateOnly,
		UnresolvedCount: unresolved, InitialBudgetKnown: trace.initialBudgetKnown,
		InitialRecordingGenerations: trace.recordingGenerations, InitialLateAllowance: trace.lateAllowance,
		InitialLateConsumed: trace.lateAlreadyConsumed, GenerationLimit: trace.generationLimit,
		MultipartLimit: trace.multipartLimit,
	}
}

func eventSessionKey(event logEvent) string {
	if event.RunID != "" {
		return "run:" + event.RunID
	}
	if event.SessionID != 0 {
		return fmt.Sprintf("session:%d", event.SessionID)
	}
	return ""
}

func replayEvent(event string) bool {
	switch event {
	case "voxt_final_candidate_observed", "voxt_final_candidate_eligibility",
		"voxt_final_candidate_outcome", "voxt_final_candidate_summary",
		"voxt_headstart_initial_budget", "voxt_headstart_decision_outcome",
		"voxt_headstart_decision", "voxt_headstart_adoption",
		"voxt_headstart_request_start", "voxt_headstart_request_done",
		"voxt_cleanup_fallback_started", "voxt_cleanup_fallback_outcome",
		"voxt_session_cancelled":
		return true
	default:
		return false
	}
}
