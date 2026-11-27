package main

type logEvent struct {
	Event                       string   `json:"event"`
	RunID                       string   `json:"run_id"`
	SessionID                   uint64   `json:"session_id"`
	RequestID                   uint64   `json:"request_id"`
	Attempt                     int      `json:"attempt"`
	Source                      string   `json:"source"`
	Sequence                    uint64   `json:"sequence"`
	GeneratedTokenCount         int      `json:"generated_token_count"`
	StopToCandidateMS           float64  `json:"stop_to_candidate_ms"`
	StopToEligibilityMS         float64  `json:"stop_to_eligibility_ms"`
	DispatchReadyMS             float64  `json:"dispatch_ready_ms"`
	PlanReadyMS                 float64  `json:"candidate_plan_ready_ms"`
	FinalAvailableMS            float64  `json:"stop_to_final_available_ms"`
	CandidateEligible           bool     `json:"candidate_eligible"`
	EligibleWithoutBudget       *bool    `json:"candidate_eligible_without_budget"`
	PreparationSucceeded        bool     `json:"preparation_succeeded"`
	LLMReady                    bool     `json:"llm_readiness_open"`
	CleanupEligible             bool     `json:"cleanup_eligible"`
	Dispatched                  bool     `json:"dispatched"`
	RequestIDs                  []uint64 `json:"request_ids"`
	RecordingGenerations        int      `json:"recording_headstart_generations"`
	LateBudgetBefore            bool     `json:"late_generation_budget_before"`
	LatestPreviewMatch          bool     `json:"latest_preview_match"`
	ExtendsLatestHypothesis     bool     `json:"extends_latest_hypothesis"`
	MeaningfulTextBoundary      bool     `json:"meaningful_text_boundary"`
	CandidateLeadMS             float64  `json:"candidate_lead_ms"`
	OracleCompatible            bool     `json:"oracle_compatible"`
	OracleIncrementalCompatible bool     `json:"oracle_incremental_compatible"`
	ExactTranscriptMatch        bool     `json:"exact_transcript_match"`
	ExactCleanupPlanMatch       bool     `json:"exact_cleanup_plan_match"`
	NearCleanupPlanCompatible   bool     `json:"near_cleanup_plan_compatible"`
	CleanupRepairEligible       bool     `json:"cleanup_repair_eligible"`
	CandidateUsableBeforeFinal  bool     `json:"candidate_usable_before_final"`
	OutcomeUnresolved           bool     `json:"outcome_unresolved"`
	FinalStatus                 string   `json:"final_status"`
	RejectionReason             string   `json:"rejection_reason"`
	Outcome                     string   `json:"outcome"`
	Adopted                     bool     `json:"adopted"`
	Succeeded                   bool     `json:"succeeded"`
	PlanReason                  string   `json:"plan_reason"`
	PlanHash                    string   `json:"plan_sha256"`
	CandidatePlanHash           string   `json:"candidate_plan_sha256"`
	RequestPlanHash             string   `json:"request_plan_sha256"`
	PartPlanHashes              []string `json:"part_request_sha256"`
	TranscriptSHA256            string   `json:"transcript_sha256"`
	CandidateSHA256             string   `json:"candidate_sha256"`
	RequestStartMS              float64  `json:"stop_to_request_start_ms"`
	RequestStartValid           bool     `json:"request_start_valid"`
	CandidateArrivalMS          float64  `json:"candidate_arrived_ms"`
	CandidateArrivalValid       bool     `json:"candidate_arrival_valid"`
	ArrivalLeadMS               float64  `json:"arrival_lead_ms"`
	CancellationMS              float64  `json:"cancellation_at_ms"`
	TerminalMS                  float64  `json:"stop_to_terminal_ms"`
}

type candidateKey struct {
	session  string
	request  uint64
	attempt  int
	source   string
	sequence uint64
}

type candidateIdentity struct {
	session      string
	planSHA      string
	candidateSHA string
}

type candidate struct {
	eligibility             logEvent
	outcome                 logEvent
	hasOutcome              bool
	production              bool
	actualAdoptionCount     int
	hasActualOutcome        bool
	exactAdoptionCount      int
	nearRepairAdoptionCount int
	failedRequestCount      int
	partialAdoption         bool
	actualRequestStartMS    float64
	actualRequestStartKnown bool
}

type sessionTrace struct {
	candidates            []candidate
	outcomes              map[candidateKey]logEvent
	outcomesByIdentity    map[candidateIdentity]logEvent
	adoptions             map[uint64]logEvent
	adoptionsByPlanHash   map[string]logEvent
	requestStarts         map[string]logEvent
	requestStartsByID     map[uint64]logEvent
	requestResults        map[string]logEvent
	requestResultsByID    map[uint64]logEvent
	finalAvailableMS      float64
	finalAvailableKnown   bool
	cancelledAtMS         float64
	cancellationKnown     bool
	recordingGenerations  int
	recordingGensKnown    bool
	observedCandidateOnly int
}

type policy struct {
	name      string
	threshold float64
	combined  bool
	match     func(candidate) bool
}

type policyResult struct {
	Name                            string  `json:"name"`
	Threshold                       float64 `json:"threshold,omitempty"`
	EligibleCandidateCount          int     `json:"eligible_candidate_count"`
	CandidateSessionCount           int     `json:"candidate_session_count"`
	SelectedSessionCount            int     `json:"selected_session_count"`
	ResolvedSelectionCount          int     `json:"resolved_selection_count"`
	OracleCompatibleSelectionCount  int     `json:"oracle_compatible_selection_count"`
	ExactTranscriptMatchCount       int     `json:"exact_transcript_match_count"`
	ExactCleanupPlanMatchCount      int     `json:"exact_cleanup_plan_match_count"`
	NearCleanupRepairCount          int     `json:"near_cleanup_repair_count"`
	WastedIfDispatchedCount         int     `json:"wasted_if_dispatched_count"`
	UnresolvedSelectionCount        int     `json:"unresolved_selection_count"`
	UnresolvedCounterfactualCount   int     `json:"unresolved_counterfactual_count"`
	ActualAdoptionCount             int     `json:"actual_adoption_count"`
	ExactAdoptionCount              int     `json:"exact_adoption_count"`
	NearRepairAdoptionCount         int     `json:"near_repair_adoption_count"`
	PartialMultipartAdoptionCount   int     `json:"partial_multipart_adoption_count"`
	FailedSpeculativeRequestCount   int     `json:"failed_speculative_request_count"`
	ActualRequestStartSampleCount   int     `json:"actual_request_start_sample_count"`
	ActualRequestStartP50MS         float64 `json:"actual_request_start_p50_ms"`
	ActualRequestStartP95MS         float64 `json:"actual_request_start_p95_ms"`
	LeadSampleCount                 int     `json:"lead_sample_count"`
	ArrivalLeadP50MS                float64 `json:"arrival_lead_p50_ms"`
	ArrivalLeadP95MS                float64 `json:"arrival_lead_p95_ms"`
	DispatchReadyLeadP50MS          float64 `json:"dispatch_ready_lead_p50_ms"`
	DispatchReadyLeadP95MS          float64 `json:"dispatch_ready_lead_p95_ms"`
	IncrementalCompatibleSelections int     `json:"incremental_compatible_selections"`
	PreservedCompatibleSelections   int     `json:"preserved_compatible_selections"`
	DisplacedCompatibleSelections   int     `json:"displaced_compatible_selections"`
	NewWastedGenerations            int     `json:"new_wasted_generations"`
	AvoidedWastedGenerations        int     `json:"avoided_wasted_generations"`
	IncrementalDispatchReadyLeadMS  float64 `json:"incremental_dispatch_ready_lead_ms"`
}

type replayReport struct {
	FormatVersion                   int            `json:"format_version"`
	SessionCount                    int            `json:"session_count"`
	ReceivedCandidateCount          int            `json:"received_candidate_count"`
	SessionEligibilityCount         int            `json:"session_eligibility_count"`
	ObservedWithoutEligibilityCount int            `json:"observed_without_eligibility_count"`
	OracleCompatibleCandidateCount  int            `json:"oracle_compatible_candidate_count"`
	OracleIncrementalCandidateCount int            `json:"oracle_incremental_candidate_count"`
	UnresolvedOutcomeCount          int            `json:"unresolved_outcome_count"`
	Policies                        []policyResult `json:"policies"`
}
