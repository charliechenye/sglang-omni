// SPDX-License-Identifier: Apache-2.0
#include "common/audio.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mlx/backend/metal/metal.h"

namespace {

enum class AudioKind { kSpeechLike, kQuiet, kChangingAmplitude, kFixture };

struct Measurement {
  std::string audio_kind;
  int duration_seconds;
  int append_interval_ms;
  double full_logmel_ms;
  double baseline_total_ms;
  double candidate_total_ms;
  double baseline_final_ms;
  double candidate_final_ms;
  double baseline_p50_ms;
  double candidate_p50_ms;
  double baseline_p95_ms;
  double candidate_p95_ms;
  double candidate_host_p50_ms;
  double candidate_eval_sync_p50_ms;
  double candidate_repeated_ms;
  double candidate_prefix_compare_ms;
  uint64_t baseline_stft_frames;
  uint64_t candidate_stft_frames;
  uint64_t cache_bytes;
};

constexpr std::array<int, 6> kDurationsSeconds = {1, 5, 10, 20, 30, 60};
constexpr std::array<int, 3> kAppendIntervalsMs = {100, 500, 1000};
constexpr int kWarmupRepeats = 2;
constexpr int kMeasurementRepeats = 3;
using Clock = std::chrono::steady_clock;

std::vector<float> ReadFixture(const std::string &fixture_path) {
  std::ifstream fixture(fixture_path, std::ios::binary);
  if (!fixture) {
    throw std::runtime_error("could not open the supplied WAV fixture");
  } else {
  }
  const std::string wav_bytes((std::istreambuf_iterator<char>(fixture)),
                              std::istreambuf_iterator<char>());
  return omni_common::DecodeWav(wav_bytes);
}

float SyntheticSample(AudioKind audio_kind, size_t sample_index) {
  const float time_seconds = static_cast<float>(sample_index) /
                             static_cast<float>(omni_common::kSampleRate);
  const float syllable_envelope =
      0.5f + 0.5f * std::sin(2.0f * static_cast<float>(M_PI) *
                             3.7f * time_seconds);
  const float speech_like =
      (0.16f * std::sin(2.0f * static_cast<float>(M_PI) * 173.0f *
                        time_seconds) +
       0.08f * std::sin(2.0f * static_cast<float>(M_PI) * 419.0f *
                        time_seconds) +
       0.03f * std::sin(2.0f * static_cast<float>(M_PI) * 911.0f *
                        time_seconds)) *
      syllable_envelope;
  if (audio_kind == AudioKind::kSpeechLike) {
    return speech_like;
  } else if (audio_kind == AudioKind::kQuiet) {
    return speech_like * 1e-4f;
  } else if (audio_kind == AudioKind::kChangingAmplitude) {
    return speech_like * (time_seconds < 5.0f ? 1e-4f : 1.0f);
  } else {
    throw std::logic_error("fixture samples must come from the WAV input");
  }
}

std::vector<float> MakeAudioSamples(AudioKind audio_kind, int duration_seconds,
                                    const std::vector<float> &fixture_samples) {
  const size_t sample_count = static_cast<size_t>(duration_seconds) *
                              omni_common::kSampleRate;
  std::vector<float> audio_samples(sample_count);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    if (audio_kind == AudioKind::kFixture) {
      audio_samples[sample_index] =
          fixture_samples[sample_index % fixture_samples.size()];
    } else {
      audio_samples[sample_index] = SyntheticSample(audio_kind, sample_index);
    }
  }
  return audio_samples;
}

int StftFrameCount(size_t sample_count) {
  const size_t padded_sample_count =
      std::max(sample_count, static_cast<size_t>(omni_common::kFftSize));
  return 1 + static_cast<int>(padded_sample_count / omni_common::kHopLength);
}

double Percentile(std::vector<double> values_ms, double quantile) {
  std::sort(values_ms.begin(), values_ms.end());
  const size_t index = static_cast<size_t>(
      std::ceil(quantile * static_cast<double>(values_ms.size())) - 1.0);
  return values_ms.at(index);
}

double TimeFullLogMel(const std::vector<float> &audio_samples) {
  const auto start = Clock::now();
  const mlx::core::array features = omni_common::LogMel(
      audio_samples, omni_common::FinalStftFrame::kDropped);
  mlx::core::eval(features);
  mlx::core::synchronize();
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double TimeIncremental(
    const std::vector<float> &audio_samples,
    omni_common::IncrementalLogMelCache &cache, uint64_t &frames_processed,
    double &host_prepare_ms) {
  const auto start = Clock::now();
  const omni_common::IncrementalLogMelResult result =
      omni_common::IncrementalLogMel(
          audio_samples, omni_common::FinalStftFrame::kDropped,
          cache.pcm_samples != nullptr ? &cache : nullptr);
  const auto prepared = Clock::now();
  host_prepare_ms =
      std::chrono::duration<double, std::milli>(prepared - start).count();
  mlx::core::eval(result.features);
  mlx::core::synchronize();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  frames_processed += static_cast<uint64_t>(result.computed_frame_count);
  cache = result.cache;
  return elapsed_ms;
}

double TimePrefixComparison(const omni_common::IncrementalLogMelCache &cache,
                            const std::vector<float> &audio_samples) {
  const auto start = Clock::now();
  const bool matches =
      cache.pcm_samples != nullptr &&
      cache.pcm_samples->size() <= audio_samples.size() &&
      std::equal(cache.pcm_samples->begin(), cache.pcm_samples->end(),
                 audio_samples.begin());
  if (!matches) {
    throw std::runtime_error("benchmark PCM prefix unexpectedly changed");
  } else {
  }
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

Measurement Measure(AudioKind audio_kind, int duration_seconds,
                    int append_interval_ms,
                    const std::vector<float> &fixture_samples) {
  const std::vector<float> recording_samples = MakeAudioSamples(
      audio_kind, duration_seconds, fixture_samples);
  for (int warmup_index = 0; warmup_index < kWarmupRepeats; ++warmup_index) {
    static_cast<void>(TimeFullLogMel(recording_samples));
    omni_common::IncrementalLogMelCache cache;
    uint64_t ignored_frames = 0;
    double ignored_prepare_ms = 0.0;
    static_cast<void>(TimeIncremental(recording_samples, cache, ignored_frames,
                                      ignored_prepare_ms));
  }
  std::vector<double> full_audio_times_ms;
  for (int repeat_index = 0; repeat_index < kMeasurementRepeats;
       ++repeat_index) {
    full_audio_times_ms.push_back(TimeFullLogMel(recording_samples));
  }
  const double full_logmel_ms = Percentile(full_audio_times_ms, 0.5);

  const size_t append_samples = static_cast<size_t>(
      append_interval_ms * omni_common::kSampleRate / 1000);
  std::vector<double> baseline_refresh_times_ms;
  std::vector<double> candidate_refresh_times_ms;
  std::vector<double> candidate_host_prepare_times_ms;
  std::vector<double> candidate_eval_sync_times_ms;
  const auto measure_refreshes = [&] {
    double baseline_total_ms = 0.0;
    uint64_t baseline_frames = 0;
    for (size_t snapshot_size = append_samples;
         snapshot_size < recording_samples.size(); snapshot_size += append_samples) {
      const std::vector<float> snapshot(recording_samples.begin(),
                                        recording_samples.begin() + snapshot_size);
      const double elapsed_ms = TimeFullLogMel(snapshot);
      baseline_refresh_times_ms.push_back(elapsed_ms);
      baseline_total_ms += elapsed_ms;
      baseline_frames += static_cast<uint64_t>(StftFrameCount(snapshot_size));
    }
    const double baseline_final_ms = TimeFullLogMel(recording_samples);
    baseline_refresh_times_ms.push_back(baseline_final_ms);
    baseline_total_ms += baseline_final_ms;
    baseline_frames += static_cast<uint64_t>(StftFrameCount(recording_samples.size()));

    double candidate_total_ms = 0.0;
    uint64_t candidate_frames = 0;
    omni_common::IncrementalLogMelCache cache;
    for (size_t snapshot_size = append_samples;
         snapshot_size < recording_samples.size(); snapshot_size += append_samples) {
      const std::vector<float> snapshot(recording_samples.begin(),
                                        recording_samples.begin() + snapshot_size);
      double host_prepare_ms = 0.0;
      const double elapsed_ms =
          TimeIncremental(snapshot, cache, candidate_frames, host_prepare_ms);
      candidate_refresh_times_ms.push_back(elapsed_ms);
      candidate_host_prepare_times_ms.push_back(host_prepare_ms);
      candidate_eval_sync_times_ms.push_back(elapsed_ms - host_prepare_ms);
      candidate_total_ms += elapsed_ms;
    }
    double candidate_final_host_prepare_ms = 0.0;
    const double candidate_final_ms =
        TimeIncremental(recording_samples, cache, candidate_frames,
                        candidate_final_host_prepare_ms);
    candidate_refresh_times_ms.push_back(candidate_final_ms);
    candidate_host_prepare_times_ms.push_back(
        candidate_final_host_prepare_ms);
    candidate_eval_sync_times_ms.push_back(candidate_final_ms -
                                           candidate_final_host_prepare_ms);
    candidate_total_ms += candidate_final_ms;
    uint64_t repeated_frames = 0;
    double repeated_host_prepare_ms = 0.0;
    const double repeated_ms =
        TimeIncremental(recording_samples, cache, repeated_frames,
                        repeated_host_prepare_ms);
    if (repeated_frames != 0) {
      throw std::runtime_error("repeated snapshot computed new STFT frames");
    } else {
    }
    const double prefix_compare_ms =
        TimePrefixComparison(cache, recording_samples);
    const uint64_t cache_bytes =
        static_cast<uint64_t>(cache.pcm_samples->size() * sizeof(float)) +
        static_cast<uint64_t>(cache.power_spectrum->nbytes()) +
        static_cast<uint64_t>(cache.normalized_features->nbytes());
    return Measurement{
        audio_kind == AudioKind::kSpeechLike
            ? "speech_like"
            : audio_kind == AudioKind::kQuiet
                  ? "quiet"
                  : audio_kind == AudioKind::kChangingAmplitude ? "changing"
                                                                : "fixture",
        duration_seconds,
        append_interval_ms,
        full_logmel_ms,
        baseline_total_ms,
        candidate_total_ms,
        baseline_final_ms,
        candidate_final_ms,
        Percentile(baseline_refresh_times_ms, 0.5),
        Percentile(candidate_refresh_times_ms, 0.5),
        Percentile(baseline_refresh_times_ms, 0.95),
        Percentile(candidate_refresh_times_ms, 0.95),
        Percentile(candidate_host_prepare_times_ms, 0.5),
        Percentile(candidate_eval_sync_times_ms, 0.5),
        repeated_ms,
        prefix_compare_ms,
        baseline_frames,
        candidate_frames,
        cache_bytes};
  };

  std::vector<double> baseline_totals_ms;
  std::vector<double> candidate_totals_ms;
  std::vector<double> baseline_final_times_ms;
  std::vector<double> candidate_final_times_ms;
  std::vector<double> candidate_repeated_times_ms;
  std::vector<double> candidate_prefix_comparison_times_ms;
  Measurement measurement{};
  for (int repeat_index = 0; repeat_index < kMeasurementRepeats;
       ++repeat_index) {
    measurement = measure_refreshes();
    baseline_totals_ms.push_back(measurement.baseline_total_ms);
    candidate_totals_ms.push_back(measurement.candidate_total_ms);
    baseline_final_times_ms.push_back(measurement.baseline_final_ms);
    candidate_final_times_ms.push_back(measurement.candidate_final_ms);
    candidate_repeated_times_ms.push_back(measurement.candidate_repeated_ms);
    candidate_prefix_comparison_times_ms.push_back(
        measurement.candidate_prefix_compare_ms);
  }
  measurement.baseline_total_ms = Percentile(baseline_totals_ms, 0.5);
  measurement.candidate_total_ms = Percentile(candidate_totals_ms, 0.5);
  measurement.baseline_final_ms = Percentile(baseline_final_times_ms, 0.5);
  measurement.candidate_final_ms = Percentile(candidate_final_times_ms, 0.5);
  measurement.candidate_repeated_ms =
      Percentile(candidate_repeated_times_ms, 0.5);
  measurement.candidate_prefix_compare_ms =
      Percentile(candidate_prefix_comparison_times_ms, 0.5);
  return measurement;
}

void PrintMeasurement(const Measurement &measurement) {
  std::cout << measurement.audio_kind << ',' << measurement.duration_seconds
            << ',' << measurement.append_interval_ms << ','
            << measurement.full_logmel_ms << ','
            << measurement.baseline_total_ms << ','
            << measurement.candidate_total_ms << ','
            << measurement.baseline_final_ms << ','
            << measurement.candidate_final_ms << ','
            << measurement.baseline_p50_ms << ','
            << measurement.candidate_p50_ms << ','
            << measurement.baseline_p95_ms << ','
            << measurement.candidate_p95_ms << ','
            << measurement.candidate_host_p50_ms << ','
            << measurement.candidate_eval_sync_p50_ms << ','
            << measurement.candidate_repeated_ms << ','
            << measurement.candidate_prefix_compare_ms << ','
            << measurement.baseline_stft_frames << ','
            << measurement.candidate_stft_frames << ','
            << measurement.cache_bytes << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const char *device_name = std::getenv("MLX_DEVICE");
    if (device_name != nullptr && std::string_view(device_name) == "cpu") {
      mlx::core::set_default_device(mlx::core::Device::cpu);
    } else {
    }
    const char *metallib_path = std::getenv("MLX_METALLIB_PATH");
    if (metallib_path != nullptr) {
      mlx::core::metal::set_metallib_path(metallib_path);
    } else {
    }
    std::vector<float> fixture_samples;
    if (argc == 2) {
      fixture_samples = ReadFixture(argv[1]);
      if (fixture_samples.empty()) {
        throw std::invalid_argument("the supplied WAV fixture is empty");
      } else {
      }
    } else if (argc == 1) {
    } else {
      throw std::invalid_argument(
          "usage: logmel_incremental_benchmark [fixture.wav]");
    }
    std::cout << "audio_kind,duration_seconds,append_interval_ms,full_logmel_ms,"
                 "baseline_total_refresh_ms,candidate_total_refresh_ms,"
                 "baseline_final_refresh_ms,candidate_final_refresh_ms,"
                 "baseline_p50_ms,candidate_p50_ms,baseline_p95_ms,"
                 "candidate_p95_ms,candidate_host_prepare_p50_ms,"
                 "candidate_eval_sync_p50_ms,candidate_repeated_ms,"
                 "candidate_prefix_compare_ms,baseline_stft_frames,"
                 "candidate_stft_frames,candidate_retained_cache_bytes\n";
    for (const AudioKind audio_kind : {AudioKind::kSpeechLike,
                                       AudioKind::kQuiet,
                                       AudioKind::kChangingAmplitude}) {
      for (const int duration_seconds : kDurationsSeconds) {
        for (const int append_interval_ms : kAppendIntervalsMs) {
          PrintMeasurement(Measure(audio_kind, duration_seconds,
                                   append_interval_ms, {}));
        }
      }
    }
    if (!fixture_samples.empty()) {
      for (const int duration_seconds : kDurationsSeconds) {
        for (const int append_interval_ms : kAppendIntervalsMs) {
          PrintMeasurement(Measure(AudioKind::kFixture, duration_seconds,
                                   append_interval_ms, fixture_samples));
        }
      }
    } else {
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
