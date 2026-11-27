package omni

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"
)

func TestFinalCandidateEventsRemainSeparateFromTheAuthoritativeResult(t *testing.T) {
	payload := `{"type":"transcript.text.candidate","text":"working text","sequence":3,"generated_token_count":24,"elapsed_ms":12.5,"language":"English","published_candidate_count":4,"snapshot_build_ms_total":1.5}`
	result, candidate, err := parseEvent(payload)
	if err != nil || result != nil || candidate == nil {
		t.Fatalf("candidate event parsed as result=%+v candidate=%+v error=%v", result, candidate, err)
	}
	if candidate.text != "working text" || candidate.metadata.Sequence != 3 || candidate.metadata.GeneratedTokens != 24 || candidate.metadata.Language != "English" || candidate.metadata.PublishedCandidateCount != 4 || candidate.metadata.SnapshotBuildMSTotal != 1.5 {
		t.Fatalf("candidate %+v", candidate)
	}
	result, candidate, err = parseEvent(`{"type":"transcript.text.done","text":"authoritative text","generation_metadata":{"generated_token_count":24,"finish_reason":"stop"}}`)
	if err != nil || candidate != nil || result == nil || result.text != "authoritative text" {
		t.Fatalf("done event parsed as result=%+v candidate=%+v error=%v", result, candidate, err)
	}
	result, candidate, err = parseEvent(`{"type":"transcript.text.candidate","text":"incomplete","sequence":0}`)
	if err != nil || result != nil || candidate != nil {
		t.Fatalf("malformed advisory event parsed as result=%+v candidate=%+v error=%v", result, candidate, err)
	}
}

func TestFinalCandidateBeforeErrorRemainsAdvisory(t *testing.T) {
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintln(w, `data: {"type":"transcript.text.candidate","text":"before failure","sequence":1,"generated_token_count":8,"elapsed_ms":20}`)
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, `data: {"type":"error","error":{"type":"server_error","code":"transcription_failed"}}`)
		_, _ = fmt.Fprintln(w)
	}))
	t.Cleanup(httpServer.Close)
	parsed, err := url.Parse(httpServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	host, portText, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	oldClient := loopback
	loopback = httpServer.Client()
	t.Cleanup(func() { loopback = oldClient })
	runtime := &Runtime{cfg: Config{Log: func(string, string) {}}}
	leased := &lease{s: &server{rt: runtime}, p: &process{ep: endpoint{host: host, port: port, model: "qwen3"}}}
	candidateCount := 0
	_, err = transcribeTimed(context.Background(), leased, finalRequest{
		pcm: []byte{0, 0}, includeCandidates: true,
		onCandidate: func(finalCandidate) { candidateCount++ },
	}, time.Second, 0)
	if err == nil || candidateCount != 1 {
		t.Fatalf("error=%v candidate count=%d", err, candidateCount)
	}
}

func TestFinalCandidatesAfterDoneAreIgnored(t *testing.T) {
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintln(w, `data: {"type":"transcript.text.candidate","text":"before final","sequence":1,"generated_token_count":8,"elapsed_ms":20}`)
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, `data: {"type":"transcript.text.done","text":"authoritative","generation_metadata":{"generated_token_count":8,"finish_reason":"stop"}}`)
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, `data: {"type":"transcript.text.candidate","text":"after final","sequence":2,"generated_token_count":16,"elapsed_ms":40}`)
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, `data: [DONE]`)
		_, _ = fmt.Fprintln(w)
	}))
	t.Cleanup(httpServer.Close)
	parsed, err := url.Parse(httpServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	host, portText, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	oldClient := loopback
	loopback = httpServer.Client()
	t.Cleanup(func() { loopback = oldClient })
	runtime := &Runtime{cfg: Config{Log: func(string, string) {}}}
	leased := &lease{s: &server{rt: runtime}, p: &process{ep: endpoint{host: host, port: port, model: "qwen3"}}}
	candidateCount := 0
	result, err := transcribeTimed(context.Background(), leased, finalRequest{
		pcm: []byte{0, 0}, includeCandidates: true,
		onCandidate: func(finalCandidate) { candidateCount++ },
	}, time.Second, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result.text != "authoritative" || candidateCount != 1 {
		t.Fatalf("result=%q candidate count=%d", result.text, candidateCount)
	}
}
