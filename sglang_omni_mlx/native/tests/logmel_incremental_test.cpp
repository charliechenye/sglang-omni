// SPDX-License-Identifier: Apache-2.0
#include "common/audio.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace mx = mlx::core;

float maximum_absolute_error_seen = 0.0f;
int unequal_element_count_seen = 0;

int StftFrameCount(size_t sample_count) {
  const size_t padded_sample_count =
      std::max(sample_count, static_cast<size_t>(omni_common::kFftSize));
  return 1 + static_cast<int>(padded_sample_count / omni_common::kHopLength);
}

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

float SpeechLikeSample(size_t sample_index, int sample_rate, float amplitude) {
  const float time_seconds =
      static_cast<float>(sample_index) / static_cast<float>(sample_rate);
  const float syllable_envelope =
      0.2f + 0.8f * std::fabs(std::sin(2.0f * static_cast<float>(M_PI) * 3.7f *
                                       time_seconds));
  return amplitude * syllable_envelope *
         (0.55f * std::sin(2.0f * static_cast<float>(M_PI) * 173.0f *
                           time_seconds) +
          0.3f * std::sin(2.0f * static_cast<float>(M_PI) * 419.0f *
                          time_seconds) +
          0.15f * std::sin(2.0f * static_cast<float>(M_PI) * 911.0f *
                           time_seconds));
}

std::vector<float> MakeSpeechLikeAudio(size_t sample_count,
                                       float amplitude = 0.2f) {
  std::vector<float> samples(sample_count);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    samples[sample_index] =
        SpeechLikeSample(sample_index, omni_common::kSampleRate, amplitude);
  }
  return samples;
}

mx::array CheckFeatures(const std::vector<float> &samples,
                        omni_common::FinalStftFrame final_frame,
                        const omni_common::IncrementalLogMelCache *previous,
                        omni_common::IncrementalLogMelCache &next,
                        const std::string &case_name,
                        int expected_computed_frame_count = -1) {
  const mx::array expected = omni_common::LogMel(samples, final_frame);
  const omni_common::IncrementalLogMelResult incremental =
      omni_common::IncrementalLogMel(samples, final_frame, previous);
  mx::eval(expected);
  mx::eval(incremental.features);
  mx::synchronize();
  Require(expected.shape() == incremental.features.shape(),
          case_name + ": tensor shape changed");
  Require(expected.dtype() == incremental.features.dtype(),
          case_name + ": tensor dtype changed");
  const int unequal_element_count =
      mx::sum(
          mx::astype(mx::not_equal(expected, incremental.features), mx::int32))
          .item<int>();
  const float maximum_absolute_error =
      mx::max(mx::abs(mx::subtract(expected, incremental.features)))
          .item<float>();
  maximum_absolute_error_seen =
      std::max(maximum_absolute_error_seen, maximum_absolute_error);
  unequal_element_count_seen += unequal_element_count;
  if (unequal_element_count > 0) {
    const float *expected_data = expected.data<float>();
    const float *incremental_data = incremental.features.data<float>();
    std::cerr << case_name << ": shape=";
    for (const auto dimension : expected.shape()) {
      std::cerr << dimension << "x";
    }
    std::cerr << " time_frames=" << expected.shape(1)
              << " computed_frames=" << incremental.computed_frame_count
              << '\n';
    for (size_t index = 0; index < expected.size(); ++index) {
      if (expected_data[index] != incremental_data[index]) {
        std::cerr << "  element " << index << " expected=" << std::hexfloat
                  << expected_data[index]
                  << " incremental=" << incremental_data[index]
                  << std::defaultfloat << '\n';
      }
    }
  }
  Require(
      unequal_element_count == 0,
      case_name +
          ": unequal elements=" + std::to_string(unequal_element_count) +
          " maximum absolute error=" + std::to_string(maximum_absolute_error));
  if (expected_computed_frame_count >= 0) {
    Require(incremental.computed_frame_count == expected_computed_frame_count,
            case_name + ": computed an unexpected number of STFT frames");
  } else {
  }
  next = incremental.cache;
  return incremental.features;
}

void CheckGrowingAudio(omni_common::FinalStftFrame final_frame) {
  const std::vector<float> recording_samples = MakeSpeechLikeAudio(5 * 16000);
  omni_common::IncrementalLogMelCache cache;
  std::mt19937 append_generator(6841);
  std::uniform_int_distribution<int> append_size(1, 3200);
  size_t sample_count = 0;
  while (sample_count < recording_samples.size()) {
    const std::array<size_t, 16> boundary_counts = {
        0,   1,   159, 160,  161,  399,  400,  401,
        799, 800, 801, 1599, 1600, 1601, 3199, 3200};
    size_t next_sample_count = sample_count;
    if (sample_count < boundary_counts.size() && sample_count == 0) {
      next_sample_count = boundary_counts[1];
    } else if (sample_count == 1) {
      next_sample_count = boundary_counts[2];
    } else if (sample_count == 159) {
      next_sample_count = boundary_counts[3];
    } else if (sample_count == 160) {
      next_sample_count = boundary_counts[4];
    } else if (sample_count == 161) {
      next_sample_count = boundary_counts[5];
    } else if (sample_count == 399) {
      next_sample_count = boundary_counts[6];
    } else if (sample_count == 400) {
      next_sample_count = boundary_counts[7];
    } else if (sample_count == 401) {
      next_sample_count = boundary_counts[8];
    } else if (sample_count == 799) {
      next_sample_count = boundary_counts[9];
    } else if (sample_count == 800) {
      next_sample_count = boundary_counts[10];
    } else if (sample_count == 801) {
      next_sample_count = boundary_counts[11];
    } else if (sample_count == 1599) {
      next_sample_count = boundary_counts[12];
    } else if (sample_count == 1600) {
      next_sample_count = boundary_counts[13];
    } else if (sample_count == 1601) {
      next_sample_count = boundary_counts[14];
    } else if (sample_count == 3199) {
      next_sample_count = boundary_counts[15];
    } else {
      next_sample_count = std::min(
          recording_samples.size(),
          sample_count + static_cast<size_t>(append_size(append_generator)));
    }
    const std::vector<float> snapshot_samples(recording_samples.begin(),
                                              recording_samples.begin() +
                                                  next_sample_count);
    CheckFeatures(snapshot_samples, final_frame,
                  cache.pcm_samples != nullptr ? &cache : nullptr, cache,
                  "growing audio length " + std::to_string(next_sample_count));
    sample_count = next_sample_count;
  }
}

void CheckRepeatedSnapshot(omni_common::FinalStftFrame final_frame) {
  const std::vector<float> samples = MakeSpeechLikeAudio(32173);
  omni_common::IncrementalLogMelCache cache;
  CheckFeatures(samples, final_frame, nullptr, cache, "initial snapshot");
  CheckFeatures(samples, final_frame, &cache, cache, "repeated snapshot", 0);
}

void CheckShortInputTransition(omni_common::FinalStftFrame final_frame) {
  const std::vector<float> recording_samples = MakeSpeechLikeAudio(401);
  omni_common::IncrementalLogMelCache cache;
  for (const size_t sample_count : {size_t{399}, size_t{400}, size_t{401}}) {
    const std::vector<float> snapshot_samples(
        recording_samples.begin(), recording_samples.begin() + sample_count);
    CheckFeatures(snapshot_samples, final_frame,
                  cache.pcm_samples != nullptr ? &cache : nullptr, cache,
                  "short input transition " + std::to_string(sample_count));
  }
}

void CheckNormalizationChanges(omni_common::FinalStftFrame final_frame) {
  std::vector<float> samples = MakeSpeechLikeAudio(32000, 1e-5f);
  omni_common::IncrementalLogMelCache cache;
  const mx::array quiet_features =
      CheckFeatures(samples, final_frame, nullptr, cache, "quiet prefix");
  const int previous_frame_count = quiet_features.shape(1);
  const std::vector<float> loud_samples = MakeSpeechLikeAudio(16000, 0.5f);
  samples.insert(samples.end(), loud_samples.begin(), loud_samples.end());
  const mx::array combined_features =
      CheckFeatures(samples, final_frame, &cache, cache, "quiet then loud");
  const mx::array previous_features =
      mx::slice(combined_features, {0, 0},
                {combined_features.shape(0), previous_frame_count});
  Require(!mx::array_equal(quiet_features, previous_features).item<bool>(),
          "a later loud frame must be able to change normalized history");
}

void CheckSilenceThenSpeech(omni_common::FinalStftFrame final_frame) {
  std::vector<float> samples(16000, 0.0f);
  omni_common::IncrementalLogMelCache cache;
  CheckFeatures(samples, final_frame, nullptr, cache, "silence prefix");
  const std::vector<float> speech_samples = MakeSpeechLikeAudio(16000, 0.2f);
  samples.insert(samples.end(), speech_samples.begin(), speech_samples.end());
  CheckFeatures(samples, final_frame, &cache, cache, "silence then speech");
}

void CheckCacheInvalidation(omni_common::FinalStftFrame final_frame) {
  const std::vector<float> original_samples = MakeSpeechLikeAudio(24001);
  omni_common::IncrementalLogMelCache cache;
  CheckFeatures(original_samples, final_frame, nullptr, cache,
                "cache invalidation baseline");
  const omni_common::IncrementalLogMelCache earlier_snapshot = cache;

  std::vector<float> changed_prefix = original_samples;
  changed_prefix[3] += 0.25f;
  CheckFeatures(changed_prefix, final_frame, &cache, cache,
                "changed PCM prefix", StftFrameCount(changed_prefix.size()));

  const std::vector<float> shorter_samples(original_samples.begin(),
                                           original_samples.begin() + 5000);
  CheckFeatures(shorter_samples, final_frame, &cache, cache, "shorter audio",
                StftFrameCount(shorter_samples.size()));

  std::vector<float> shifted_samples = original_samples;
  std::rotate(shifted_samples.begin(), shifted_samples.begin() + 1,
              shifted_samples.end());
  CheckFeatures(shifted_samples, final_frame, &cache, cache, "shifted audio",
                StftFrameCount(shifted_samples.size()));

  const omni_common::FinalStftFrame other_layout =
      final_frame == omni_common::FinalStftFrame::kDropped
          ? omni_common::FinalStftFrame::kKept
          : omni_common::FinalStftFrame::kDropped;
  CheckFeatures(shifted_samples, other_layout, &cache, cache, "changed layout",
                StftFrameCount(shifted_samples.size()));

  omni_common::IncrementalLogMelCache new_session_cache;
  CheckFeatures(
      shifted_samples, other_layout,
      new_session_cache.pcm_samples != nullptr ? &new_session_cache : nullptr,
      new_session_cache, "new session", StftFrameCount(shifted_samples.size()));

  CheckFeatures(original_samples, final_frame, &earlier_snapshot, cache,
                "copied cache snapshot", 0);
}

void CheckEightSecondBoundaries(omni_common::FinalStftFrame final_frame) {
  const size_t eight_second_samples = 8 * omni_common::kSampleRate;
  const std::vector<float> recording_samples =
      MakeSpeechLikeAudio(eight_second_samples + 2 * omni_common::kHopLength);
  omni_common::IncrementalLogMelCache cache;
  for (const size_t sample_count :
       {eight_second_samples - 1, eight_second_samples,
        eight_second_samples + 1,
        eight_second_samples + omni_common::kHopLength - 1,
        eight_second_samples + omni_common::kHopLength,
        eight_second_samples + omni_common::kHopLength + 1}) {
    const std::vector<float> snapshot_samples(
        recording_samples.begin(), recording_samples.begin() + sample_count);
    CheckFeatures(snapshot_samples, final_frame,
                  cache.pcm_samples != nullptr ? &cache : nullptr, cache,
                  "eight-second encoder window boundary " +
                      std::to_string(sample_count));
  }
}

} // namespace

int main() {
  try {
    for (const omni_common::FinalStftFrame final_frame :
         {omni_common::FinalStftFrame::kDropped,
          omni_common::FinalStftFrame::kKept}) {
      CheckGrowingAudio(final_frame);
      CheckRepeatedSnapshot(final_frame);
      CheckShortInputTransition(final_frame);
      CheckNormalizationChanges(final_frame);
      CheckSilenceThenSpeech(final_frame);
      CheckCacheInvalidation(final_frame);
      CheckEightSecondBoundaries(final_frame);
    }
    std::cout
        << "incremental LogMel parity passed for both layouts; max_abs_error="
        << maximum_absolute_error_seen
        << " unequal_elements=" << unequal_element_count_seen << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
