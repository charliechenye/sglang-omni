// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen3_asr/candidate_text.h"

namespace {

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string Decode(const std::vector<int> &ids, bool skip_special_tokens) {
  std::string out;
  for (int id : ids) {
    if (skip_special_tokens && id == 99) {
      continue;
    }
    switch (id) {
    case 1:
      out += "hello";
      break;
    case 2:
      out += "你好";
      break;
    case 3:
      out += "\xe4";
      break;
    case 4:
      out += "\xbd\xa0";
      break;
    case 5:
      out += "\x01";
      break;
    case 6:
      out += "<|unfinished|>";
      break;
    case 99:
      out += "<|eos|>";
      break;
    default:
      out += "?";
      break;
    }
  }
  return out;
}

void TestExplicitLanguageWithoutGeneratedMarker() {
  const std::vector<int> marker{8, 9};
  auto english = qwen3_asr::ExtractCandidateText({1}, marker, true, "", Decode);
  auto chinese = qwen3_asr::ExtractCandidateText({2}, marker, true, "", Decode);
  Require(english == "hello", "explicit English should not need an ASR marker");
  Require(chinese == "你好", "explicit Chinese should not need an ASR marker");
}

void TestAutomaticLanguageMarkerBoundary() {
  const std::vector<int> marker{8, 9};
  Require(!qwen3_asr::HasCandidateTranscriptToken({7, 8}, marker, false),
          "an incomplete marker must not make a candidate eligible");
  Require(!qwen3_asr::ExtractCandidateText({7, 8}, marker, false, "", Decode),
          "automatic output before the marker must be rejected");
  Require(
      !qwen3_asr::ExtractCandidateText({1, 8}, marker, true, "", Decode),
      "an incomplete marker suffix must not publish as explicit-language text");
  Require(!qwen3_asr::HasCandidateTranscriptToken({8, 9}, marker, false),
          "a marker without transcript tokens must be rejected");
  Require(qwen3_asr::ExtractCandidateText({7, 8, 9, 1}, marker, false, "",
                                          Decode) == "hello",
          "automatic output after the complete marker should publish");
}

void TestMultilingualTokenBoundariesAndUnsafeText() {
  const std::vector<int> marker{8, 9};
  Require(
      !qwen3_asr::ExtractCandidateText({8, 9, 3}, marker, false, "", Decode),
      "a partial UTF-8 code point must not publish");
  Require(qwen3_asr::ExtractCandidateText({8, 9, 3, 4}, marker, false, "",
                                          Decode) == "你",
          "a complete multilingual character may publish");
  Require(!qwen3_asr::ExtractCandidateText({1, 5}, marker, true, "", Decode),
          "control characters must not publish");
  Require(!qwen3_asr::ExtractCandidateText({1, 6}, marker, true, "", Decode),
          "special-token spellings must not publish");
  Require(!qwen3_asr::ExtractCandidateText({99}, marker, true, "", Decode),
          "EOS alone must not publish text");
}

void TestOnlyCommittedTokensAndTerminalSnapshots() {
  const std::vector<int> marker{8, 9};
  const std::vector<int> target_verified_tokens{8, 9, 1};
  const std::vector<int> rejected_draft_tokens{2};
  const auto candidate = qwen3_asr::ExtractCandidateText(
      target_verified_tokens, marker, false, "", Decode);
  Require(candidate == "hello",
          "candidate extraction must use the target-verified snapshot");
  Require(candidate != Decode(rejected_draft_tokens, true),
          "rejected draft tokens must not leak into the candidate");
  Require(!qwen3_asr::ShouldPublishCandidateSnapshot(true, false, false),
          "EOS termination must not emit a candidate snapshot");
  Require(!qwen3_asr::ShouldPublishCandidateSnapshot(false, true, false),
          "token-loop termination must not emit a candidate snapshot");
  Require(!qwen3_asr::ShouldPublishCandidateSnapshot(false, false, true),
          "the final token-budget snapshot must not publish");
  Require(qwen3_asr::ShouldPublishCandidateSnapshot(false, false, false),
          "a nonterminal committed snapshot may publish");
}

} // namespace

int main() {
  try {
    TestExplicitLanguageWithoutGeneratedMarker();
    TestAutomaticLanguageMarkerBoundary();
    TestMultilingualTokenBoundariesAndUnsafeText();
    TestOnlyCommittedTokensAndTerminalSnapshots();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
