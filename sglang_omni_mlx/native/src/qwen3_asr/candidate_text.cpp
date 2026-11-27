// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/candidate_text.h"

#include <algorithm>
#include <cstdint>

namespace qwen3_asr {
namespace {

bool IsValidCandidateText(const std::string &text) {
  for (std::size_t i = 0; i < text.size();) {
    const unsigned char first = static_cast<unsigned char>(text[i]);
    if (first < 0x80) {
      if (first < 0x20 || first == 0x7f) {
        return false;
      }
      ++i;
      continue;
    }
    std::size_t width = 0;
    uint32_t codepoint = 0;
    if ((first & 0xe0) == 0xc0) {
      width = 2;
      codepoint = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
      width = 3;
      codepoint = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
      width = 4;
      codepoint = first & 0x07;
    } else {
      return false;
    }
    if (i + width > text.size()) {
      return false;
    }
    for (std::size_t j = 1; j < width; ++j) {
      const unsigned char continuation =
          static_cast<unsigned char>(text[i + j]);
      if ((continuation & 0xc0) != 0x80) {
        return false;
      }
      codepoint = (codepoint << 6) | (continuation & 0x3f);
    }
    if ((width == 2 && codepoint < 0x80) || (width == 3 && codepoint < 0x800) ||
        (width == 4 && codepoint < 0x10000) || codepoint > 0x10ffff ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff) ||
        (codepoint >= 0x80 && codepoint <= 0x9f) || codepoint == 0xfffd) {
      return false;
    }
    i += width;
  }
  return true;
}

bool EndsWithPartialMarker(const std::vector<int> &token_ids,
                           const std::vector<int> &marker_ids) {
  if (marker_ids.size() < 2) {
    return false;
  }
  const std::size_t maximum = std::min(token_ids.size(), marker_ids.size() - 1);
  for (std::size_t length = maximum; length > 0; --length) {
    if (std::equal(token_ids.end() - length, token_ids.end(),
                   marker_ids.begin())) {
      return true;
    }
  }
  return false;
}

} // namespace

std::optional<std::size_t>
FindCandidateMarker(const std::vector<int> &token_ids,
                    const std::vector<int> &marker_ids) {
  if (marker_ids.empty() || marker_ids.size() > token_ids.size()) {
    return std::nullopt;
  }
  for (std::size_t start = 0; start + marker_ids.size() <= token_ids.size();
       ++start) {
    if (std::equal(marker_ids.begin(), marker_ids.end(),
                   token_ids.begin() + start)) {
      return start;
    }
  }
  return std::nullopt;
}

bool HasCandidateTranscriptToken(const std::vector<int> &token_ids,
                                 const std::vector<int> &marker_ids,
                                 bool explicit_language) {
  if (explicit_language) {
    return !token_ids.empty();
  }
  const auto marker = FindCandidateMarker(token_ids, marker_ids);
  return marker.has_value() && token_ids.size() > *marker + marker_ids.size();
}

bool ShouldPublishCandidateSnapshot(bool stop_token, bool token_loop,
                                    bool token_budget_reached) {
  return !stop_token && !token_loop && !token_budget_reached;
}

std::optional<std::string>
ExtractCandidateText(const std::vector<int> &token_ids,
                     const std::vector<int> &marker_ids, bool explicit_language,
                     const std::string &prefix,
                     const CandidateDecoder &decode) {
  if (!decode ||
      !HasCandidateTranscriptToken(token_ids, marker_ids, explicit_language)) {
    return std::nullopt;
  }
  const auto marker = FindCandidateMarker(token_ids, marker_ids);
  if (!marker.has_value() && EndsWithPartialMarker(token_ids, marker_ids)) {
    return std::nullopt;
  }
  const std::size_t transcript_start =
      marker.has_value() ? *marker + marker_ids.size() : 0;
  const std::vector<int> transcript_ids(token_ids.begin() + transcript_start,
                                        token_ids.end());
  std::string text = prefix + decode(transcript_ids, true);
  if (text.empty() || text.find("<|") != std::string::npos ||
      !IsValidCandidateText(text)) {
    return std::nullopt;
  }
  return text;
}

} // namespace qwen3_asr
