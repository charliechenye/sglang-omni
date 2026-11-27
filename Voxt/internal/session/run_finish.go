package session

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type finishOpts struct {
	card    deliver.Reason
	replace bool
}

// finish delivers result (unless it is a try or empty; a note's is
// saved), keeps the history entry and ends the session.
func (r *run) finish(st State, problem *Problem, result string, o finishOpts) {
	cfg := r.m.cfg
	var dres deliver.Result
	noted := false
	switch {
	case result == "" || r.req.Try:
	case r.req.Note:
		if err := r.saveNote(result); err != nil {
			r.logf("error", "note not saved, the card shows it: %v", err)
			problem = &Problem{Code: ProblemInternal, Err: err, Degraded: true}
			break
		}
		noted = true
	case o.card != "":
		dres.Reason = o.card
		r.m.update(r, func(s *Snapshot) { s.Result, s.Card = result, true })
	default:
		if r.req.Kind == store.KindTranscribe {
			if !r.m.beginTranscribeDelivery(r, result) {
				// Cancellation or loss of the current session slot must not
				// insert this result into another session's target.
				r.endCancelled(nil)
				return
			}
		} else {
			r.m.update(r, func(s *Snapshot) { s.State, s.Result = Delivering, result })
		}
		press := r.press(o.replace) // app enhancement's auto key
		ctx, cancel := context.WithTimeout(context.Background(), cfg.DeliverTime)
		dres = deliver.Insert(ctx, cfg.Text, deliver.Request{Text: result, Target: r.req.Target, Press: press, Self: cfg.Self, Activate: cfg.Activate,
			Replace: o.replace, VerifyMS: deliver.DefaultVerifyMS, Stop: r.ctx, KeepClipboard: r.settings.General.CopyResult})
		cancel()
		if r.cancelled.Load() && errors.Is(dres.Err, context.Canceled) {
			// Cancelled before the paste was sent: nothing went in.
			r.endCancelled(nil)
			return
		}
		if dres.Err != nil {
			r.logf("warning", "insert: %v", dres.Err)
		}
		if stopAt := r.stoppedAt(); !stopAt.IsZero() {
			r.logf("info", "stop_to_delivery=%v", r.now().Sub(stopAt).Round(time.Millisecond))
		}
		r.logf("info", "delivery: %s", dres.Diag())
		switch {
		case dres.Reason == deliver.ReasonUnconfirmed || dres.Reason == deliver.ReasonSecureInput:
			r.logf("warning", "pasted into %s, but the text did not show there (%s): the card shows it, the clipboard holds it", dres.To.BundleID, dres.Reason)
		case dres.Inserted && dres.Moved(r.req.Target):
			r.logf("info", "inserted into %s, not %s (the user switched apps; no auto key) in %v", dres.To.BundleID, r.req.Target.BundleID, dres.Elapsed.Round(time.Millisecond))
		case dres.Inserted:
			r.logf("info", "inserted in %v", dres.Elapsed.Round(time.Millisecond))
		default:
			r.logf("info", "not inserted (%s; %s in front) in %v", dres.Reason, dres.Front.BundleID, dres.Elapsed.Round(time.Millisecond))
		}
	}
	if r.asrDegraded != nil && (problem == nil || problem.Degraded) {
		problem = r.asrDegraded
	}
	snap := r.m.snapshot(r)
	total := r.now().Sub(snap.StartedAt)
	var highlights []dict.Span
	if r.req.Kind == store.KindTranscribe && r.settings.Dictionary.Enabled {
		highlights = dict.HighlightSpans(result, r.terms)
	}
	r.m.end(r, st, func(s *Snapshot) {
		s.Result, s.Problem = result, problem
		s.Highlights = highlights
		s.Recording = r.keepsRecording(store.StatusDone)
		s.Inserted = dres.Inserted
		if l, ok := dres.Landing(); ok {
			s.Landing = &l
		}
		// A text that did not go in shows in the card only when it must (a
		// forced card, an unsaved note, Rewrite's follow-ups) or the
		// feature's card is on; else the app copies it and says so.
		s.Card = result != "" && !r.req.Try && !dres.Inserted && !noted &&
			(o.card != "" || r.req.Note || r.req.Kind == store.KindRewrite || r.settings.Features.ShowsCard(string(r.req.Kind)))
		s.Delivery = dres.Reason
		s.Timings.Deliver, s.Timings.Total = dres.Elapsed, total
	}, func(s Snapshot) { r.keep(s, store.StatusDone) })
}

// fail ends the session failed before it recorded anything worth keeping.
func (r *run) fail(p *Problem) {
	r.cancel()
	r.logf("warning", "failed: %s: %v", p.Code, p.Err)
	r.m.end(r, Failed, func(s *Snapshot) { s.Problem = p }, nil)
}

// endFailed ends the session failed after it recorded: the entry keeps
// the recording when the settings say so.
func (r *run) endFailed(p *Problem, stream asr.Stream, more ...func(*Snapshot)) {
	if !r.stopStreamReader(stream) {
		p = &Problem{Code: ProblemInternal, Err: errors.New("the speech stream reader did not stop")}
	}
	r.cancel()
	r.logf("warning", "failed: %s: %v", p.Code, p.Err)
	total := r.now().Sub(r.m.snapshot(r).StartedAt)
	audio := asr.Duration(r.pcm.bytes(), asr.SampleRate)
	r.m.end(r, Failed, func(s *Snapshot) {
		s.Problem = p
		s.Timings.Total = total
		s.Timings.Audio = audio
		s.Recording = audio >= r.m.cfg.MinRecording && r.keepsRecording(store.StatusFailed)
		for _, fn := range more {
			fn(s)
		}
	}, func(s Snapshot) {
		if audio >= r.m.cfg.MinRecording {
			r.keep(s, store.StatusFailed)
		}
	})
}

func (r *run) endCancelled(stream asr.Stream) {
	readerStopped := r.stopStreamReader(stream)
	r.cancel()
	if !readerStopped {
		r.logf("warning", "the speech stream reader did not stop after cancellation")
	}
	r.logf("info", "cancelled")
	total := r.now().Sub(r.m.snapshot(r).StartedAt)
	audio := asr.Duration(r.pcm.bytes(), asr.SampleRate)
	r.m.end(r, Cancelled, func(s *Snapshot) {
		s.Timings.Total = total
		s.Timings.Audio = audio
		s.Recording = audio >= r.m.cfg.MinRecording && r.keepsRecording(store.StatusCancelled)
	}, func(s Snapshot) {
		if audio >= r.m.cfg.MinRecording {
			r.keep(s, store.StatusCancelled)
		}
	})
}

// asrProblem classifies a speech model's error.
func (r *run) asrProblem(err error) *Problem {
	if r.cancelled.Load() {
		return &Problem{Code: ProblemInternal, Err: context.Canceled}
	}
	return &Problem{Code: ProblemASR, Err: err}
}

func (r *run) ctxErr(ctx context.Context) error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return &asr.Error{Kind: asr.KindTimeout, Detail: fmt.Sprintf("no result within %v (%v of audio)", r.asrLimit, asr.Duration(r.pcm.bytes(), asr.SampleRate).Round(time.Second)), Retryable: true}
	}
	return &asr.Error{Kind: asr.KindCancelled, Detail: "cancelled"}
}

func (r *run) timeoutProblem(ctx context.Context) *Problem {
	return &Problem{Code: ProblemASR, Err: r.ctxErr(ctx)}
}

// saveNote saves a note session's result (Config.Notes).
func (r *run) saveNote(text string) error {
	if r.m.cfg.Notes == nil {
		return errors.New("no notes store")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return r.m.cfg.Notes.Put(ctx, store.Note{ID: r.id, Text: text, CreatedAt: r.m.snapshot(r).StartedAt, AppName: r.req.Target.Name})
}

// keep writes the history entry of an ended session (and its recording),
// when inHistory says so.
func (r *run) keep(s Snapshot, status store.Status) {
	cfg := r.m.cfg
	if !r.inHistory(status) {
		return
	}
	e := store.Entry{
		ID: r.id, Kind: r.req.Kind, CreatedAt: s.StartedAt, Status: status, Conversation: cmp.Or(r.req.Conversation, r.id), Replaces: r.req.Replaces,
		AppBundle: s.Target.BundleID, AppName: s.Target.Name,
		RawText: s.Raw, Result: s.Result, Title: s.Title, SourceText: s.Selection, TypedText: s.Typed, TargetLang: s.TargetLanguage,
		ASRProvider: s.ASR.ID, ASRModel: s.ASR.Model, LLMProvider: s.LLM.ID, LLMModel: s.LLM.Model,
		AudioMS: s.Timings.Audio.Milliseconds(), SentMS: s.Timings.Sent.Milliseconds(), ASRMS: s.Timings.ASR.Milliseconds(), LLMMS: s.Timings.LLM.Milliseconds(),
		TotalMS: s.Timings.Total.Milliseconds(), DSP: s.DSP, GainDB: s.GainDB,
	}
	if r.req.Kind == store.KindRewrite {
		e.PromptText, e.RawText = s.Raw, "" // the spoken instruction
	}
	if s.Timings.LLM == 0 {
		e.LLMProvider, e.LLMModel = "", ""
	}
	if r.req.Text != nil || s.Timings.Audio == 0 {
		e.ASRProvider, e.ASRModel = "", "" // nothing was recognised
	}
	if m, ok := r.knownScene(); ok {
		e.URLHost = m.Host
		if m.Matched() {
			e.RuleID, e.RuleName, e.RuleSource = m.Group.ID, m.Group.Name, string(m.Source)
		}
	}
	switch {
	case s.Inserted:
		e.Delivery = "insert"
		if to := s.Landing.App; to.PID != 0 && to.PID != s.Target.PID {
			e.InsertedBundle, e.InsertedName = to.BundleID, to.Name // where it went; the app and the rule stay the start app's
		}
	case s.Card:
		e.Delivery = "card"
	}
	if s.Problem != nil && s.Problem.Err != nil {
		e.Error = s.Problem.Err.Error()
	}
	if s.Recording {
		name := r.id + ".wav"
		dir := cfg.History.AudioDir()
		if err := os.MkdirAll(dir, 0o755); err == nil {
			if err := os.WriteFile(filepath.Join(dir, name), asr.WAV(r.pcm.bytes(), asr.SampleRate), 0o644); err == nil {
				e.Audio = name
				if !r.settings.Privacy.KeepAudio {
					e.AudioUntil = r.now().Add(cfg.AudioRetention)
					r.logf("info", "the recording is kept until %s, for Retry", e.AudioUntil.Format(time.DateTime))
				}
			} else {
				r.logf("warning", "keeping the recording: %v", err)
			}
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := cfg.History.Put(ctx, e); err != nil {
		r.logf("error", "history: %v", err)
	}
}

// inHistory reports whether the session's history entry is written: not
// for a try, nor for a note saved from its recognition (the Notes page
// keeps it). A note whose recognition failed, or that was cancelled,
// keeps its entry and recording, as a dictation does.
func (r *run) inHistory(status store.Status) bool {
	if r.req.Try || r.m.cfg.History == nil || !r.settings.Privacy.KeepHistory {
		return false
	}
	return !(r.req.Note && status == store.StatusDone && !r.recognitionFailed)
}

// keepsRecording reports whether the session's history entry keeps the
// recording: the settings keep recordings, recognition failed, or the
// user cancelled (for a while: Config.AudioRetention), so Retry can
// recover it.
func (r *run) keepsRecording(status store.Status) bool {
	if !r.inHistory(status) || r.req.Text != nil || len(r.pcm.bytes()) == 0 {
		return false
	}
	return r.settings.Privacy.KeepAudio || r.recognitionFailed || r.cancelled.Load()
}

func isHelperErr(err error) bool { return errors.Is(err, core.ErrHelper) }
