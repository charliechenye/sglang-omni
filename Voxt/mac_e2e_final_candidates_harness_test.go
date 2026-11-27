package main

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type benchScenario struct {
	Name            string   `json:"name"`
	Audio           string   `json:"audio"`
	Language        string   `json:"language"`
	HeadStartCase   string   `json:"headstart_case"`
	ProtocolSmoke   bool     `json:"protocol_smoke"`
	DictionaryTerms []string `json:"dictionary_terms"`
	Draft           bool     `json:"draft"`
	PiecesAfterMS   int      `json:"pieces_after_ms"`
	PieceMinMS      int      `json:"piece_min_ms"`
	PieceTargetMS   int      `json:"piece_target_ms"`
	PieceMaxMS      int      `json:"piece_max_ms"`
}

type benchCall struct {
	StartAt, FirstTokenAt, DoneAt time.Time
	RequestID                     uint64
	InputChars                    int
	Kind                          string
	Usage                         llm.Usage
	Succeeded                     bool
}

type benchRun struct {
	mu                  sync.Mutex
	shadowReportOnce    sync.Once
	shadowReportDone    chan struct{}
	experimentEvents    chan struct{}
	Scenario            benchScenario
	RunID               string
	ExperimentSessionID uint64
	Arm                 string
	Warmup              bool
	Round               int
	ColdStart           bool
	ProtocolSmokePassed bool
	StartedAt           time.Time
	StoppedAt           time.Time
	Snapshot            session.Snapshot
	RecordingAt         time.Time
	FinalAt             time.Time
	FirstDraftAt        time.Time
	CleanupDoneAt       time.Time
	ResultReadyAt       time.Time
	DeliveryRequestedAt time.Time
	DeliveryCompletedAt time.Time
	SessionDoneAt       time.Time
	LLMCalls            []benchCall
	ExperimentLogs      []json.RawMessage
	PlaybackFactor      float64
}

type benchLog struct {
	mu      sync.Mutex
	runs    map[string]*benchRun
	pending map[string][]json.RawMessage
}

func (l *benchLog) registerRun(run *benchRun, id string) {
	l.mu.Lock()
	if l.runs == nil {
		l.runs = make(map[string]*benchRun)
	}
	l.runs[id] = run
	pending := l.pending[id]
	delete(l.pending, id)
	l.mu.Unlock()
	run.mu.Lock()
	run.RunID = id
	for _, raw := range pending {
		var event struct {
			SessionID uint64 `json:"session_id"`
			Event     string `json:"event"`
		}
		if json.Unmarshal(raw, &event) == nil {
			if event.SessionID != 0 {
				run.ExperimentSessionID = event.SessionID
			}
			run.ExperimentLogs = append(run.ExperimentLogs, append(json.RawMessage(nil), raw...))
			run.notifyExperimentEvent()
			if event.Event == "voxt_final_candidate_summary" && run.shadowReportDone != nil {
				run.shadowReportOnce.Do(func() { close(run.shadowReportDone) })
			}
		}
	}
	run.mu.Unlock()
}

func (l *benchLog) add(_ string, message string) {
	start := strings.IndexByte(message, '{')
	if start < 0 {
		return
	}
	var event struct {
		Event     string `json:"event"`
		RunID     string `json:"run_id"`
		SessionID uint64 `json:"session_id"`
	}
	raw := json.RawMessage(message[start:])
	if json.Unmarshal(raw, &event) != nil || !benchTraceEvent(event.Event) {
		return
	}
	l.mu.Lock()
	owner := l.runs[event.RunID]
	if owner == nil && event.RunID != "" {
		if l.pending == nil {
			l.pending = make(map[string][]json.RawMessage)
		}
		l.pending[event.RunID] = append(l.pending[event.RunID], append(json.RawMessage(nil), raw...))
	}
	if owner != nil {
		owner.mu.Lock()
		if event.SessionID != 0 {
			owner.ExperimentSessionID = event.SessionID
		}
		owner.ExperimentLogs = append(owner.ExperimentLogs, append(json.RawMessage(nil), raw...))
		owner.notifyExperimentEvent()
		if event.Event == "voxt_final_candidate_summary" && owner.shadowReportDone != nil {
			owner.shadowReportOnce.Do(func() { close(owner.shadowReportDone) })
		}
		owner.mu.Unlock()
	}
	l.mu.Unlock()
}

func benchTraceEvent(event string) bool {
	return strings.HasPrefix(event, "voxt_final_candidate_") ||
		strings.HasPrefix(event, "voxt_headstart_") ||
		strings.HasPrefix(event, "voxt_cleanup_request_plan")
}

func runBenchSession(t *testing.T, manager *session.Manager, logs *benchLog, audio *benchAudio, pcm []byte, scenario benchScenario, arm string, warmup, round int, cold bool, done <-chan session.Snapshot, activeMu *sync.Mutex, activeRun **benchRun) *benchRun {
	t.Helper()
	mode := ""
	if arm == "shadow" || arm == "shadow_dense" {
		mode = arm
		if err := os.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", mode); err != nil {
			t.Fatal(err)
		}
	} else {
		if err := os.Setenv("VOXT_EXPERIMENT_FINAL_CANDIDATES", "instrumented"); err != nil {
			t.Fatal(err)
		}
	}
	run := &benchRun{
		Scenario: scenario, Arm: arm, Warmup: warmup >= 0, Round: round,
		ColdStart: cold, StartedAt: time.Now(), shadowReportDone: make(chan struct{}),
		experimentEvents: make(chan struct{}, 1),
	}
	run.PlaybackFactor = audio.frameFactor
	ready := audio.prepare(pcm)
	select {
	case <-done:
	default:
	}
	activeMu.Lock()
	*activeRun = run
	activeMu.Unlock()
	id, err := manager.Start(session.StartRequest{Kind: store.KindTranscribe})
	if err != nil {
		t.Fatalf("start %s session: %v", arm, err)
	}
	logs.registerRun(run, id)
	select {
	case <-ready:
	case <-time.After(20 * time.Minute):
		t.Fatalf("scenario %q audio did not finish playback", scenario.Name)
	}
	stopAt := time.Now()
	manager.Stop(id)
	run.mu.Lock()
	run.StoppedAt = stopAt
	run.mu.Unlock()
	select {
	case snapshot := <-done:
		run.Snapshot = snapshot
	case <-time.After(20 * time.Minute):
		t.Fatalf("scenario %q %s session did not finish", scenario.Name, arm)
	}
	select {
	case <-run.shadowReportDone:
	case <-time.After(30 * time.Second):
		t.Fatalf("scenario %q %s terminal experiment report did not finish", scenario.Name, arm)
	}
	waitHeadStartRequestEvents(run, 5*time.Second)
	if scenario.ProtocolSmoke && arm != "off" {
		coverage := candidateProtocolCoverage(run.snapshot().ExperimentLogs)
		run.mu.Lock()
		run.ProtocolSmokePassed = coverage.HTTPObservedCount > 0
		run.mu.Unlock()
	}
	activeMu.Lock()
	*activeRun = nil
	activeMu.Unlock()
	return run
}

func waitHeadStartRequestEvents(run *benchRun, timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	complete := false
	var pending []uint64
	for {
		run.mu.Lock()
		logs := append([]json.RawMessage(nil), run.ExperimentLogs...)
		run.mu.Unlock()
		pending = pendingHeadStartRequestIDs(logs)
		if len(pending) == 0 {
			complete = true
			break
		}
		if run.experimentEvents == nil {
			break
		}
		timedOut := false
		select {
		case <-run.experimentEvents:
		case <-timer.C:
			timedOut = true
		}
		if timedOut {
			break
		}
	}
	run.mu.Lock()
	logs := append([]json.RawMessage(nil), run.ExperimentLogs...)
	run.mu.Unlock()
	pending = pendingHeadStartRequestIDs(logs)
	complete = len(pending) == 0
	record, _ := json.Marshal(map[string]any{
		"event": "voxt_headstart_request_collection", "complete": complete,
		"unresolved_request_ids": pending, "timeout_ms": timeout.Milliseconds(),
	})
	run.mu.Lock()
	run.ExperimentLogs = append(run.ExperimentLogs, record)
	run.mu.Unlock()
}

func (r *benchRun) notifyExperimentEvent() {
	if r.experimentEvents == nil {
		return
	}
	select {
	case r.experimentEvents <- struct{}{}:
	default:
	}
}

func pendingHeadStartRequestIDs(logs []json.RawMessage) []uint64 {
	started, done := make(map[uint64]bool), make(map[uint64]bool)
	for _, raw := range logs {
		var event struct {
			Event      string   `json:"event"`
			RequestID  uint64   `json:"request_id"`
			RequestIDs []uint64 `json:"request_ids"`
		}
		if json.Unmarshal(raw, &event) != nil {
			continue
		}
		switch event.Event {
		case "voxt_headstart_decision":
			for _, requestID := range event.RequestIDs {
				if requestID != 0 {
					started[requestID] = true
				}
			}
		case "voxt_headstart_request_start":
			if event.RequestID != 0 {
				started[event.RequestID] = true
			}
		case "voxt_headstart_request_done":
			if event.RequestID != 0 {
				done[event.RequestID] = true
			}
		}
	}
	pending := make([]uint64, 0)
	for requestID := range started {
		if !done[requestID] {
			pending = append(pending, requestID)
		}
	}
	sort.Slice(pending, func(i, j int) bool { return pending[i] < pending[j] })
	return pending
}

func (r *benchRun) observe(snapshot session.Snapshot) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Snapshot = snapshot
	now := time.Now()
	if snapshot.State == session.Recording && r.RecordingAt.IsZero() {
		r.RecordingAt = now
	}
	if snapshot.Raw != "" && r.FinalAt.IsZero() {
		r.FinalAt = now
	}
	if snapshot.Draft != "" && r.FirstDraftAt.IsZero() {
		r.FirstDraftAt = now
	}
	if snapshot.State == session.Delivering && r.CleanupDoneAt.IsZero() {
		r.CleanupDoneAt = now
		r.ResultReadyAt = now
	}
	if snapshot.Result != "" && r.ResultReadyAt.IsZero() {
		r.ResultReadyAt = now
	}
	if snapshot.State.Ended() && r.SessionDoneAt.IsZero() {
		r.SessionDoneAt = now
	}
}
