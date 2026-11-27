// SPDX-License-Identifier: Apache-2.0
// Candidate eligibility and text extraction from committed ASR tokens.
#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace qwen3_asr {

using CandidateDecoder =
    std::function<std::string(const std::vector<int> &, bool)>;

std::optional<std::size_t>
FindCandidateMarker(const std::vector<int> &token_ids,
                    const std::vector<int> &marker_ids);

bool HasCandidateTranscriptToken(const std::vector<int> &token_ids,
                                 const std::vector<int> &marker_ids,
                                 bool explicit_language);

bool ShouldPublishCandidateSnapshot(bool stop_token, bool token_loop,
                                    bool token_budget_reached);

// token_ids must contain only tokens already committed by the decoder.
std::optional<std::string>
ExtractCandidateText(const std::vector<int> &token_ids,
                     const std::vector<int> &marker_ids, bool explicit_language,
                     const std::string &prefix, const CandidateDecoder &decode);

} // namespace qwen3_asr
