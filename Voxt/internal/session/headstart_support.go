package session

import (
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

const (
	headStartPauseEnvironment = "VOXT_EXPERIMENT_HEADSTART_PAUSE_MS"
	// headHangover is how long, in milliseconds, the user is silent before
	// the text known so far is asked about. The answer then has a start on
	// Stop, and changes of the live text while they stay silent are asked
	// about at once.
	headHangover = 500
	// rateRest is how long a service that limited the rate rests.
	rateRest = 10 * time.Minute
)

func headStartPauseMilliseconds(value string, present bool) int {
	if !present {
		return headHangover
	}
	milliseconds, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil {
		return headHangover
	}
	switch milliseconds {
	case 100, 150, 300, headHangover:
		return milliseconds
	default:
		return headHangover
	}
}

func configuredHeadStartPauseMilliseconds() int {
	value, present := os.LookupEnv(headStartPauseEnvironment)
	return headStartPauseMilliseconds(value, present)
}

// pauseWatch watches the recording for pauses: it tells r.silent when the
// speech ends (true) and goes on (false); the newest wins.
type pauseWatch struct {
	r          *run
	detector   *vad.Detector
	frameBytes int
	speaking   bool
	quiet      atomic.Bool
}

func (r *run) watchPauses() *pauseWatch {
	if r.silent == nil {
		return nil
	}
	milliseconds := r.headStartPauseMilliseconds
	if milliseconds == 0 {
		milliseconds = headHangover
	}
	frameMilliseconds := vad.Defaults().FrameMS
	return &pauseWatch{
		r: r, detector: vad.NewDetector(asr.SampleRate, vad.Params{FrameMS: frameMilliseconds, HangoverMS: milliseconds}),
		frameBytes: asr.SampleRate * frameMilliseconds / 1000 * 2,
	}
}

// hear takes the recording's next PCM.
func (w *pauseWatch) hear(pcm []byte) {
	if w == nil {
		return
	}
	transitioned := false
	for len(pcm) > 0 {
		frameBytes := min(len(pcm), w.frameBytes)
		w.detector.Write(pcm[:frameBytes])
		if speaking := w.detector.Speaking(); speaking != w.speaking {
			w.speaking = speaking
			transitioned = true
		}
		pcm = pcm[frameBytes:]
	}
	if transitioned {
		w.quiet.Store(!w.speaking)
		select {
		case <-w.r.silent:
		default:
		}
		w.r.silent <- !w.speaking
	}
}

// noteRate notes that a service limited the rate: it refused the request,
// or an attempt of it, for that.
func (m *Manager) noteRate(service string, resp llm.Response, err error) {
	if llm.Code(err) != llm.CodeRateLimited && !resp.RateLimited {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.restUntil == nil {
		m.restUntil = map[string]time.Time{}
	}
	m.restUntil[service] = time.Now().Add(rateRest)
}

// resting says whether a service limited the rate lately (noteRate): it
// is asked one part at a time, with no head start.
func (m *Manager) resting(service string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return time.Now().Before(m.restUntil[service])
}
