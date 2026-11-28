package main

type logEvent struct {
	Event                           string   `json:"event"`
	RunID                           string   `json:"run_id"`
	SessionID                       uint64   `json:"session_id"`
	RequestID                       uint64   `json:"request_id"`
	Attempt                         int      `json:"attempt"`
	Source                          string   `json:"source"`
	Sequence                        uint64   `json:"sequence"`
	GeneratedTokenCount             int      `json:"generated_token_count"`
	StopToCandidateMS               float64  `json:"stop_to_candidate_ms"`
	StopToEligibilityMS             float64  `json:"stop_to_eligibility_ms"`
	DispatchReadyMS                 float64  `json:"dispatch_ready_ms"`
	PlanReadyMS                     float64  `json:"candidate_plan_ready_ms"`
	FinalAvailableMS                float64  `json:"stop_to_final_available_ms"`
	CandidateEligible               bool     `json:"candidate_eligible"`
	EligibleWithoutBudget           *bool    `json:"candidate_eligible_without_budget"`
	PreparationSucceeded            bool     `json:"preparation_succeeded"`
	LLMReady                        bool     `json:"llm_readiness_open"`
	CleanupEligible                 bool     `json:"cleanup_eligible"`
	Dispatched                      bool     `json:"dispatched"`
	RequestIDs                      []uint64 `json:"request_ids"`
	FreshRequestIDs                 []uint64 `json:"fresh_request_ids"`
	ReusedRequestIDs                []uint64 `json:"reused_request_ids"`
	GenerationID                    uint64   `json:"generation_id"`
	RequestOrigin                   string   `json:"request_origin"`
	Cancelled                       bool     `json:"cancelled"`
	TerminalStatus                  string   `json:"terminal_status"`
	TerminalTraceComplete           bool     `json:"terminal_trace_complete"`
	RecordingGenerations            int      `json:"recording_headstart_generations"`
	LateBudgetBefore                bool     `json:"late_generation_budget_before"`
	LatestPreviewMatch              bool     `json:"latest_preview_match"`
	ExtendsLatestHypothesis         bool     `json:"extends_latest_hypothesis"`
	MeaningfulTextBoundary          bool     `json:"meaningful_text_boundary"`
	CandidateLeadMS                 float64  `json:"candidate_lead_ms"`
	OracleCompatible                bool     `json:"oracle_compatible"`
	OracleIncrementalCompatible     bool     `json:"oracle_incremental_compatible"`
	ExactTranscriptMatch            bool     `json:"exact_transcript_match"`
	ExactCleanupPlanMatch           bool     `json:"exact_cleanup_plan_match"`
	NearCleanupPlanCompatible       bool     `json:"near_cleanup_plan_compatible"`
	CleanupRepairEligible           bool     `json:"cleanup_repair_eligible"`
	CandidateUsableBeforeFinal      bool     `json:"candidate_usable_before_final"`
	OutcomeUnresolved               bool     `json:"outcome_unresolved"`
	FinalStatus                     string   `json:"final_status"`
	RejectionReason                 string   `json:"rejection_reason"`
	Outcome                         string   `json:"outcome"`
	Adopted                         bool     `json:"adopted"`
	Succeeded                       bool     `json:"succeeded"`
	PlanReason                      string   `json:"plan_reason"`
	PlanHash                        string   `json:"plan_sha256"`
	CandidatePlanHash               string   `json:"candidate_plan_sha256"`
	RequestPlanHash                 string   `json:"request_plan_sha256"`
	PartPlanHashes                  []string `json:"part_request_sha256"`
	TranscriptSHA256                string   `json:"transcript_sha256"`
	CandidateSHA256                 string   `json:"candidate_sha256"`
	RequestStartMS                  float64  `json:"stop_to_request_start_ms"`
	RequestStartValid               bool     `json:"request_start_valid"`
	CandidateArrivalMS              float64  `json:"candidate_arrived_ms"`
	CandidateArrivalValid           bool     `json:"candidate_arrival_valid"`
	ArrivalLeadMS                   float64  `json:"arrival_lead_ms"`
	CancellationMS                  float64  `json:"cancellation_at_ms"`
	TerminalMS                      float64  `json:"stop_to_terminal_ms"`
	InitialBudgetKnown              *bool    `json:"initial_headstart_budget_known"`
	InitialRecordingGenerations     int      `json:"recording_generation_count"`
	InitialRecordingGenerationCount int      `json:"initial_recording_generation_count"`
	InitialLateAllowance            int      `json:"initial_late_generation_allowance"`
	InitialLateConsumed             bool     `json:"late_generation_already_consumed"`
	InitialLateConsumedSummary      bool     `json:"initial_late_generation_already_consumed"`
	GenerationLimit                 int      `json:"effective_generation_limit"`
	MultipartLimit                  int      `json:"effective_multipart_limit"`
	FreshRequestPartCount           int      `json:"fresh_request_part_count"`
	FreshCompatiblePartCount        int      `json:"fresh_compatible_cleanup_parts"`
	ExistingEquivalentRequests      int      `json:"existing_equivalent_requests"`
	CounterfactualStateUnresolved   bool     `json:"counterfactual_state_unresolved"`
	DispatchReadyBeforeFinal        bool     `json:"dispatch_ready_before_final"`
	TerminalState                   string   `json:"terminal_state"`
	Problem                         string   `json:"problem"`
	FallbackOutcome                 string   `json:"fallback_outcome"`
	Arm                             string   `json:"arm"`
	Scenario                        string   `json:"scenario"`
	Round                           int      `json:"round"`
	RunStatus                       string   `json:"status"`
	AudioSHA256                     string   `json:"audio_sha256"`
	LanguageHint                    string   `json:"language_hint"`
	LanguageMode                    string   `json:"language_mode"`
	EffectiveLanguage               string   `json:"effective_language"`
	Draft                           bool     `json:"draft"`
	Warmup                          bool     `json:"warmup"`
	ColdStart                       bool     `json:"cold_start"`
}

type candidateKey struct {
	session  string
	request  uint64
	attempt  int
	source   string
	sequence uint64
}

type candidate struct {
	eligibility                   logEvent
	outcome                       logEvent
	hasOutcome                    bool
	production                    bool
	actualAdoptionCount           int
	hasActualOutcome              bool
	exactAdoptionCount            int
	nearRepairAdoptionCount       int
	failedRequestCount            int
	cancelledRequestCount         int
	reusedRecordingAdoptionCount  int
	unadoptedFreshRequestCount    int
	freshRequestCount             int
	partiallyResolved             bool
	actualOutcomeStatus           string
	partialAdoption               bool
	actualRequestStartMS          float64
	actualRequestStartKnown       bool
	counterfactualStateUnresolved bool
}

type sessionTrace struct {
	candidates               []candidate
	outcomes                 map[candidateKey]logEvent
	adoptions                map[uint64]logEvent
	requestStartsByID        map[uint64]logEvent
	requestResultsByID       map[uint64]logEvent
	fallbackAttemptCount     int
	fallbackSuccessCount     int
	fallbackFailureCount     int
	finalAvailableMS         float64
	finalAvailableKnown      bool
	cancelledAtMS            float64
	cancellationKnown        bool
	arm                      string
	scenario                 string
	round                    int
	runID                    string
	audioSHA256              string
	languageHint             string
	languageMode             string
	effectiveLanguage        string
	draft                    bool
	warmup                   bool
	coldStart                bool
	runStatus                string
	runProblem               string
	benchmark                bool
	initialBudgetKnown       bool
	recordingGenerations     int
	lateAllowance            int
	lateAlreadyConsumed      bool
	generationLimit          int
	multipartLimit           int
	observedCandidateOnly    int
	terminalDecisionComplete bool
}

type policy struct {
	name      string
	threshold float64
	combined  bool
	match     func(candidate) bool
}

type policyResult struct {
	Name                                   string  `json:"name"`
	Threshold                              float64 `json:"threshold,omitempty"`
	SessionCount                           int     `json:"session_count"`
	SuccessfulSessionCount                 int     `json:"successful_session_count"`
	FailedSessionCount                     int     `json:"failed_session_count"`
	CancelledSessionCount                  int     `json:"cancelled_session_count"`
	NoCandidateSessionCount                int     `json:"no_candidate_session_count"`
	UnresolvedPairCount                    int     `json:"unresolved_pair_count"`
	EligibleCandidateCount                 int     `json:"eligible_candidate_count"`
	CandidateSessionCount                  int     `json:"candidate_session_count"`
	SelectedSessionCount                   int     `json:"selected_session_count"`
	ResolvedSelectionCount                 int     `json:"resolved_selection_count"`
	OracleCompatibleSelectionCount         int     `json:"oracle_compatible_selection_count"`
	ExactTranscriptMatchCount              int     `json:"exact_transcript_match_count"`
	ExactCleanupPlanMatchCount             int     `json:"exact_cleanup_plan_match_count"`
	NearCleanupRepairCount                 int     `json:"near_cleanup_repair_count"`
	WastedIfDispatchedCount                int     `json:"wasted_if_dispatched_count"`
	UnresolvedSelectionCount               int     `json:"unresolved_selection_count"`
	UnresolvedCounterfactualCount          int     `json:"unresolved_counterfactual_count"`
	ActualAdoptionCount                    int     `json:"actual_adoption_count"`
	ActualHeadStartDispatchCount           int     `json:"actual_headstart_dispatch_count"`
	ResolvedHeadStartDispatchCount         int     `json:"resolved_actual_headstart_dispatch_count"`
	ActualHeadStartMissCount               int     `json:"actual_headstart_miss_count"`
	UnresolvedHeadStartDispatches          int     `json:"unresolved_actual_headstart_dispatch_count"`
	RecordingTimeAdoptionCount             int     `json:"recording_time_adoption_count"`
	FreshLateGenerationDispatchCount       int     `json:"fresh_late_generation_dispatch_count"`
	FreshLateGenerationMissCount           int     `json:"fresh_late_generation_miss_count"`
	FreshLateRequestCount                  int     `json:"fresh_late_request_count"`
	ResolvedLateGenerationCount            int     `json:"resolved_late_generation_count"`
	ExactAdoptionCount                     int     `json:"exact_adoption_count"`
	NearRepairAdoptionCount                int     `json:"near_repair_adoption_count"`
	PartialMultipartAdoptionCount          int     `json:"partial_multipart_adoption_count"`
	FailedSpeculativeRequestCount          int     `json:"failed_speculative_request_count"`
	ReusedRecordingRequestAdoptionCount    int     `json:"reused_recording_request_adoption_count"`
	FreshLateRequestAdoptionCount          int     `json:"fresh_late_request_adoption_count"`
	FreshLateGenerationContributionCount   int     `json:"fresh_late_generation_incremental_contribution_count"`
	FreshLateRequestNotAdoptedCount        int     `json:"fresh_late_request_not_adopted_count"`
	FreshLateRequestFailureCount           int     `json:"fresh_late_request_failure_count"`
	FreshLateRequestCancelledCount         int     `json:"fresh_late_request_cancelled_count"`
	PartiallyResolvedLateGenerationCount   int     `json:"partially_resolved_late_generation_count"`
	FreshLatePartialMultipartAdoptionCount int     `json:"fresh_late_partial_multipart_adoption_count"`
	UnresolvedLateGenerationCount          int     `json:"unresolved_late_generation_count"`
	FallbackAttemptCount                   int     `json:"fallback_attempt_count"`
	FallbackSuccessCount                   int     `json:"fallback_success_count"`
	FallbackFailureCount                   int     `json:"fallback_failure_count"`
	FallbackUnresolvedCount                int     `json:"fallback_unresolved_count"`
	ActualRequestStartSampleCount          int     `json:"actual_request_start_sample_count"`
	ActualRequestStartP50MS                float64 `json:"actual_request_start_p50_ms"`
	ActualRequestStartP95MS                float64 `json:"actual_request_start_p95_ms"`
	LeadSampleCount                        int     `json:"lead_sample_count"`
	ArrivalLeadP50MS                       float64 `json:"arrival_lead_p50_ms"`
	ArrivalLeadP95MS                       float64 `json:"arrival_lead_p95_ms"`
	DispatchReadyLeadP50MS                 float64 `json:"dispatch_ready_lead_p50_ms"`
	DispatchReadyLeadP95MS                 float64 `json:"dispatch_ready_lead_p95_ms"`
	IncrementalCompatibleSelections        int     `json:"potential_incremental_compatible_selections"`
	PreservedCompatibleSelections          int     `json:"preserved_compatible_selections"`
	DisplacedCompatibleSelections          int     `json:"displaced_compatible_selections"`
	CompatibleFreshRequestParts            int     `json:"compatible_fresh_request_parts"`
	ExistingRecordingCoverageParts         int     `json:"existing_recording_coverage_parts"`
	NewWastedGenerations                   int     `json:"new_wasted_generations"`
	AvoidedWastedGenerations               int     `json:"avoided_wasted_generations"`
	IncrementalDispatchReadyLeadMS         float64 `json:"incremental_dispatch_ready_lead_ms"`
	EarlierCompatibleSelections            int     `json:"earlier_compatible_selections"`
	LaterCompatibleSelections              int     `json:"later_compatible_selections"`
	PairedLeadSampleCount                  int     `json:"paired_lead_sample_count"`
	PairedLeadP50MS                        float64 `json:"paired_lead_p50_ms"`
	PairedLeadP95MS                        float64 `json:"paired_lead_p95_ms"`
	counterfactualSelectionUnknown         bool    `json:"-"`
}

type replayReport struct {
	FormatVersion                   int            `json:"format_version"`
	SelectedArm                     string         `json:"selected_arm,omitempty"`
	Arms                            []armReport    `json:"arms,omitempty"`
	SessionCount                    int            `json:"session_count"`
	ReceivedCandidateCount          int            `json:"received_candidate_count"`
	SessionEligibilityCount         int            `json:"session_eligibility_count"`
	ObservedWithoutEligibilityCount int            `json:"observed_without_eligibility_count"`
	OracleCompatibleCandidateCount  int            `json:"oracle_compatible_candidate_count"`
	OracleIncrementalCandidateCount int            `json:"oracle_incremental_candidate_count"`
	UnresolvedOutcomeCount          int            `json:"unresolved_outcome_count"`
	Policies                        []policyResult `json:"policies"`
}

type armReport struct {
	Arm                string           `json:"arm"`
	SessionCount       int              `json:"session_count"`
	SteadySessionCount int              `json:"steady_session_count"`
	WarmupSessionCount int              `json:"warmup_session_count"`
	ColdStartCount     int              `json:"cold_start_session_count"`
	Runs               []sessionSummary `json:"runs"`
	Policies           []policyResult   `json:"policies"`
	Scenarios          []scenarioReport `json:"scenarios"`
}

type scenarioReport struct {
	Scenario           string         `json:"scenario"`
	SessionCount       int            `json:"session_count"`
	SteadySessionCount int            `json:"steady_session_count"`
	WarmupSessionCount int            `json:"warmup_session_count"`
	ColdStartCount     int            `json:"cold_start_session_count"`
	Policies           []policyResult `json:"policies"`
}

type sessionSummary struct {
	RunID                       string `json:"run_id"`
	Arm                         string `json:"arm"`
	Scenario                    string `json:"scenario"`
	Round                       int    `json:"round"`
	AudioSHA256                 string `json:"audio_sha256"`
	LanguageHint                string `json:"language_hint"`
	LanguageMode                string `json:"language_mode"`
	EffectiveLanguage           string `json:"effective_language,omitempty"`
	Draft                       bool   `json:"draft"`
	Warmup                      bool   `json:"warmup"`
	ColdStart                   bool   `json:"cold_start"`
	Status                      string `json:"status"`
	Problem                     string `json:"problem,omitempty"`
	CandidateCount              int    `json:"candidate_count"`
	UnresolvedCount             int    `json:"unresolved_candidate_count"`
	InitialBudgetKnown          bool   `json:"initial_headstart_budget_known"`
	InitialRecordingGenerations int    `json:"initial_recording_generation_count"`
	InitialLateAllowance        int    `json:"initial_late_generation_allowance"`
	InitialLateConsumed         bool   `json:"initial_late_generation_already_consumed"`
	GenerationLimit             int    `json:"effective_generation_limit"`
	MultipartLimit              int    `json:"effective_multipart_limit"`
}
