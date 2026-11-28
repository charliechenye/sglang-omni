// SPDX-License-Identifier: Apache-2.0
// Model-independent transcription results, their JSON form, and long audio
// decoded one 30 s window at a time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/generation.h"
#include "nlohmann/json.hpp"

namespace omni_common {

struct SpeakerSegment {
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  // Empty when the output carried no speaker segments.
  std::string speaker;
  std::string text;
  // Note (Dayuxiaoshui): SenseVoice's language, emotion and audio event of
  // the segment's audio; none for other models.
  std::optional<std::string> language;
  std::optional<std::string> emotion;
  std::optional<std::string> event;
};

struct TranscriptionResult {
  std::string text;
  std::optional<std::string> language;
  int generated_token_count = 0;
  FinishReason finish_reason = FinishReason::kLength;
  // Timestamped speaker segments, for models that produce them.
  std::vector<SpeakerSegment> segments;
  // Rounds of draft verification, and the draft proposals the target checked
  // and accepted; all 0 when decoding without a draft.
  int verify_round_count = 0;
  int proposed_draft_token_count = 0;
  int accepted_draft_token_count = 0;
  // Ordered text of the windows of a request cut into several.
  std::vector<std::string> window_texts;
  // Audio tokens a live session had already encoded.
  int reused_audio_token_count = 0;
  // Prompt tokens a live session had already computed.
  int reused_prompt_token_count = 0;
  // note (chenye): Supplied guess tokens included in initial target
  // verification.
  int verified_guess_token_count = 0;
  // note (chenye): Prefix of those guesses accepted during initial
  // verification, including a token that ends decoding by EOS, budget, or loop
  // detection.
  int accepted_guess_token_count = 0;
  // Actual decoder paths, distinct from requested draft or guess settings.
  std::vector<std::string> decode_paths;
  bool draft_decoder_used = false;
  bool batch_decode_used = false;
  bool native_chunked = false;
  // Time from Qwen3-ASR request start to its first committed transcript token
  // when candidate observation is enabled; zero otherwise.
  double first_transcript_token_elapsed_ms = 0.0;
  // Qwen3-ASR model completion on the same request-relative clock as candidates.
  std::optional<double> authoritative_completion_elapsed_ms;
};

struct TranscriptCandidate {
  std::string text;
  uint64_t sequence = 0;
  int generated_token_count = 0;
  std::optional<std::string> language;
  double elapsed_ms = 0.0;
  double first_transcript_token_elapsed_ms = 0.0;
  double snapshot_build_ms = 0.0;
  uint64_t published_candidate_count = 0;
  double snapshot_build_ms_total = 0.0;
};

using TranscriptCandidatePublisher =
    std::function<void(TranscriptCandidate)>;

// Segments as the HTTP and realtime APIs send them: start, end, speaker, text,
// and language, emotion and event when set.
nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments);

// Sample ranges of at most 30 s that cover samples, each cut at the quietest
// 0.1 s in the 5 s before its limit, the last at least 1 s long; one range
// for shorter audio.
std::vector<std::pair<size_t, size_t>>
ThirtySecondWindows(const std::vector<float> &samples);

// samples transcribed one window at a time and the texts joined with
// JoinTranscriptParts, token and draft counts summed. max_new_tokens gives a
// window's budget from the tokens the windows before it generated; below 1,
// the request stops there. Truncation by either ends it with kLength, and
// window_texts lists every window's text when there are several.
TranscriptionResult TranscribeEachWindow(
    const std::vector<float> &samples,
    const std::vector<std::pair<size_t, size_t>> &windows,
    const std::function<int(const std::pair<size_t, size_t> &window,
                            int generated_token_count)> &max_new_tokens,
    const std::function<TranscriptionResult(
        const std::vector<float> &window_samples, int max_new_tokens)>
        &transcribe_window);

} // namespace omni_common
