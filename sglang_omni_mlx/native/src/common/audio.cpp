// SPDX-License-Identifier: Apache-2.0
#include "common/audio.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr float kPcm16FullScale = 32768.0f;
// The CUDA server's encode_pcm scales by 32767, not 32768.
constexpr float kPcm16OutputScale = 32767.0f;
constexpr uint16_t kWavFormatPcm = 1;
constexpr uint16_t kWavFormatFloat = 3;

uint32_t ReadU32(std::string_view bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint16_t ReadU16(std::string_view bytes, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

float HertzToMel(float frequency_hz, float linear_step_hz, float min_log_hz,
                 float min_log_mel, float log_step) {
  if (frequency_hz < min_log_hz) {
    return frequency_hz / linear_step_hz;
  } else {
    return min_log_mel + std::log(frequency_hz / min_log_hz) / log_step;
  }
}

float MelToHertz(float mel, float linear_step_hz, float min_log_hz,
                 float min_log_mel, float log_step) {
  if (mel < min_log_mel) {
    return linear_step_hz * mel;
  } else {
    return min_log_hz * std::exp(log_step * (mel - min_log_mel));
  }
}

std::vector<float> BuildMelFilterBank(int mel_bin_count, int fft_size) {
  const int frequency_bin_count = fft_size / 2 + 1;
  const float linear_step_hz = 200.0f / 3.0f;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / linear_step_hz;
  const float log_step = std::log(6.4f) / 27.0f;

  std::vector<float> bin_frequencies_hz(frequency_bin_count);
  for (int i = 0; i < frequency_bin_count; ++i) {
    bin_frequencies_hz[i] = static_cast<float>(i) *
                            static_cast<float>(kSampleRate) /
                            static_cast<float>(fft_size);
  }
  const float mel_max =
      HertzToMel(static_cast<float>(kSampleRate) / 2.0f, linear_step_hz,
                 min_log_hz, min_log_mel, log_step);
  std::vector<float> edges_hz(mel_bin_count + 2);
  for (int i = 0; i < mel_bin_count + 2; ++i) {
    edges_hz[i] = MelToHertz(static_cast<float>(i) * mel_max /
                                 static_cast<float>(mel_bin_count + 1),
                             linear_step_hz, min_log_hz, min_log_mel, log_step);
  }
  std::vector<float> filters(
      static_cast<size_t>(frequency_bin_count) * mel_bin_count, 0.0f);
  for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
    const float low = edges_hz[mel_bin];
    const float center = edges_hz[mel_bin + 1];
    const float high = edges_hz[mel_bin + 2];
    const float normalization = 2.0f / (high - low);
    for (int frequency_bin = 0; frequency_bin < frequency_bin_count;
         ++frequency_bin) {
      const float frequency_hz = bin_frequencies_hz[frequency_bin];
      float weight = 0.0f;
      if (low <= frequency_hz && frequency_hz < center) {
        weight = (frequency_hz - low) / (center - low);
      } else if (center <= frequency_hz && frequency_hz <= high) {
        weight = (high - frequency_hz) / (high - center);
      } else {
        weight = 0.0f;
      }
      filters[static_cast<size_t>(frequency_bin) * mel_bin_count + mel_bin] =
          weight * normalization;
    }
  }
  return filters;
}

std::vector<float> BuildPeriodicHannWindow() {
  std::vector<float> window(kFftSize);
  const float denominator = static_cast<float>(kFftSize);
  // Note (Jiaxin Deng): Swift's Float.pi, one float step below (float)M_PI.
  const float pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  for (int n = 0; n < kFftSize; ++n) {
    window[n] =
        0.5f *
        (1.0f - std::cos(2.0f * pi * static_cast<float>(n) / denominator));
  }
  return window;
}

int StftFrameCount(size_t sample_count) {
  const size_t padded_sample_count =
      std::max(sample_count, static_cast<size_t>(kFftSize));
  return 1 + static_cast<int>(padded_sample_count / kHopLength);
}

int LogMelFrameCount(size_t sample_count, FinalStftFrame final_frame) {
  const int stft_frame_count = StftFrameCount(sample_count);
  return final_frame == FinalStftFrame::kDropped ? stft_frame_count - 1
                                                 : stft_frame_count;
}

int StableLogMelFrameCount(size_t sample_count, FinalStftFrame final_frame) {
  if (sample_count < static_cast<size_t>(kFftSize)) {
    return 0;
  } else {
  }
  // Centered frames are stable once their right half lies within the PCM.
  const int direct_frame_count =
      1 + static_cast<int>((sample_count - kFftSize / 2) / kHopLength);
  return std::min(LogMelFrameCount(sample_count, final_frame),
                  direct_frame_count);
}

mx::array ComputePowerSpectrum(const std::vector<float> &samples,
                               int first_frame, int frame_count) {
  std::vector<float> padded_samples = samples;
  if (padded_samples.size() < static_cast<size_t>(kFftSize)) {
    padded_samples.resize(kFftSize, 0.0f);
  } else {
  }
  const int sample_count = static_cast<int>(padded_samples.size());
  const mx::array audio(padded_samples.data(), {sample_count}, mx::float32);
  constexpr int padding = kFftSize / 2;
  const mx::array head = mx::slice(audio, {padding}, {0}, {-1});
  const mx::array tail =
      mx::slice(audio, {sample_count - 2}, {sample_count - padding - 2}, {-1});
  const mx::array padded = mx::concatenate({head, audio, tail});
  const int all_frame_count = 1 + (padded.shape(0) - kFftSize) / kHopLength;
  assert(first_frame >= 0 && frame_count >= 0 &&
         first_frame + frame_count <= all_frame_count);
  mx::array frames =
      mx::as_strided(padded, {all_frame_count, kFftSize}, {kHopLength, 1}, 0);
  if (first_frame != 0 || frame_count != all_frame_count) {
    frames = mx::slice(frames, {first_frame, 0},
                       {first_frame + frame_count, kFftSize});
  } else {
  }
  const mx::array window(PeriodicHannWindow().data(), {kFftSize}, mx::float32);
  return mx::square(mx::abs(mx::fft::rfft(mx::multiply(frames, window), 1)));
}

mx::array ProjectLogMelSpectrum(const mx::array &power_spectrum) {
  const mx::array filters(MelFilterBank().data(),
                          {kFftSize / 2 + 1, kMelBinCount}, mx::float32);
  return mx::log10(mx::maximum(mx::matmul(power_spectrum, filters),
                               mx::array(kLogMelFloor)));
}

mx::array NormalizeLogMelSpectrum(const mx::array &raw_log_spectrum) {
  const mx::array clipped_log_spectrum = mx::maximum(
      raw_log_spectrum,
      mx::subtract(mx::max(raw_log_spectrum), mx::array(kLogMelDynamicRange)));
  return mx::transpose(mx::divide(
      mx::add(clipped_log_spectrum, mx::array(4.0f)), mx::array(4.0f)));
}

} // namespace

std::vector<float> DecodeWav(std::string_view wav_bytes, int sample_rate) {
  if (wav_bytes.size() < 12 || wav_bytes.substr(0, 4) != "RIFF" ||
      wav_bytes.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("audio must be a RIFF/WAVE file");
  } else {
  }
  size_t offset = 12;
  bool have_format = false;
  uint16_t audio_format = 0;
  uint16_t channel_count = 0;
  uint32_t file_sample_rate = 0;
  uint16_t bits_per_sample = 0;
  while (offset + 8 <= wav_bytes.size()) {
    const std::string_view chunk_id = wav_bytes.substr(offset, 4);
    const uint32_t chunk_size = ReadU32(wav_bytes, offset + 4);
    const size_t body_offset = offset + 8;
    const size_t body_size =
        std::min<size_t>(chunk_size, wav_bytes.size() - body_offset);
    if (chunk_id == "fmt ") {
      if (body_size < 16) {
        throw std::invalid_argument("WAV fmt chunk is truncated");
      } else {
      }
      audio_format = ReadU16(wav_bytes, body_offset);
      channel_count = ReadU16(wav_bytes, body_offset + 2);
      file_sample_rate = ReadU32(wav_bytes, body_offset + 4);
      bits_per_sample = ReadU16(wav_bytes, body_offset + 14);
      have_format = true;
    } else if (chunk_id == "data") {
      if (!have_format || channel_count != 1 ||
          file_sample_rate != static_cast<uint32_t>(sample_rate)) {
        const std::string kilohertz =
            sample_rate % 1000 == 0
                ? std::to_string(sample_rate / 1000)
                : std::to_string(sample_rate / 1000) + "." +
                      std::to_string(sample_rate % 1000 / 100);
        throw std::invalid_argument("audio must be " + kilohertz + " kHz mono");
      } else if (audio_format == kWavFormatPcm && bits_per_sample == 16) {
        std::vector<float> samples(body_size / 2);
        for (size_t i = 0; i < samples.size(); ++i) {
          int16_t value = 0;
          std::memcpy(&value, wav_bytes.data() + body_offset + 2 * i,
                      sizeof(value));
          samples[i] = static_cast<float>(value) / kPcm16FullScale;
        }
        return samples;
      } else if (audio_format == kWavFormatFloat && bits_per_sample == 32) {
        std::vector<float> samples(body_size / 4);
        std::memcpy(samples.data(), wav_bytes.data() + body_offset,
                    samples.size() * 4);
        return samples;
      } else {
        throw std::invalid_argument("audio must be PCM16 or float32 WAV");
      }
    } else {
    }
    offset = body_offset + chunk_size + (chunk_size & 1u);
  }
  throw std::invalid_argument("WAV file has no data chunk");
}

std::string Pcm16Bytes(const std::vector<float> &samples) {
  std::string bytes;
  bytes.reserve(samples.size() * 2);
  for (const float sample : samples) {
    // Note (Minyi Tang): NaN becomes silence, since std::clamp passes it
    // through and casting NaN to an integer is undefined.
    const float bounded =
        std::isnan(sample) ? 0.0f : std::clamp(sample, -1.0f, 1.0f);
    const auto value = static_cast<int16_t>(bounded * kPcm16OutputScale);
    bytes.push_back(static_cast<char>(value & 0xff));
    bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  }
  return bytes;
}

std::string WavBytes(const std::string &pcm, int sample_rate,
                     int channel_count) {
  std::string bytes;
  const auto append = [&](uint32_t value, int size) {
    for (int index = 0; index < size; ++index) {
      bytes.push_back(static_cast<char>((value >> (8 * index)) & 0xff));
    }
  };
  const auto data_size = static_cast<uint32_t>(pcm.size());
  bytes += "RIFF";
  append(36 + data_size, 4);
  bytes += "WAVEfmt ";
  append(16, 4);                              // fmt chunk size
  append(1, 2);                               // PCM
  append(channel_count, 2);                   // channels
  append(sample_rate, 4);                     // frames per second
  append(sample_rate * 2 * channel_count, 4); // bytes per second
  append(2 * channel_count, 2);               // bytes per frame
  append(16, 2);                              // bits per sample
  bytes += "data";
  append(data_size, 4);
  return bytes + pcm;
}

const std::vector<float> &MelFilterBank(int mel_bin_count, int fft_size) {
  static std::mutex mutex;
  static std::map<std::pair<int, int>, std::vector<float>> filters_by_shape;
  const std::lock_guard<std::mutex> lock(mutex);
  const std::pair<int, int> shape = {mel_bin_count, fft_size};
  auto found = filters_by_shape.find(shape);
  if (found == filters_by_shape.end()) {
    found = filters_by_shape
                .emplace(shape, BuildMelFilterBank(mel_bin_count, fft_size))
                .first;
  } else {
  }
  return found->second;
}

const std::vector<float> &PeriodicHannWindow() {
  static const std::vector<float> window = BuildPeriodicHannWindow();
  return window;
}

mx::array LogMel(const std::vector<float> &samples,
                 FinalStftFrame final_frame) {
  const int stft_frame_count = StftFrameCount(samples.size());
  mx::array power_spectrum = ComputePowerSpectrum(samples, 0, stft_frame_count);
  if (final_frame == FinalStftFrame::kDropped) {
    power_spectrum =
        mx::slice(power_spectrum, {0, 0},
                  {power_spectrum.shape(0) - 1, power_spectrum.shape(1)});
  } else {
  }
  return NormalizeLogMelSpectrum(ProjectLogMelSpectrum(power_spectrum));
}

IncrementalLogMelResult
IncrementalLogMel(const std::vector<float> &samples, FinalStftFrame final_frame,
                  const IncrementalLogMelCache *previous) {
  const int output_frame_count = LogMelFrameCount(samples.size(), final_frame);
  const int all_stft_frame_count = StftFrameCount(samples.size());
  bool has_compatible_prefix =
      previous != nullptr && previous->pcm_samples != nullptr &&
      previous->power_spectrum.has_value() &&
      previous->final_frame == final_frame &&
      previous->pcm_samples->size() <= samples.size() &&
      previous->power_spectrum->shape(0) ==
          LogMelFrameCount(previous->pcm_samples->size(), final_frame) &&
      previous->normalized_features.has_value() &&
      previous->normalized_features->shape(0) == kMelBinCount &&
      previous->normalized_features->shape(1) ==
          previous->power_spectrum->shape(0) &&
      previous->stable_frame_count >= 0 &&
      previous->stable_frame_count <= previous->power_spectrum->shape(0) &&
      previous->stable_frame_count <= output_frame_count &&
      previous->power_spectrum->shape(1) == kFftSize / 2 + 1;
  if (has_compatible_prefix) {
    has_compatible_prefix =
        previous->power_spectrum->dtype() == mx::float32 &&
        previous->normalized_features->dtype() == mx::float32;
  } else {
  }
  if (has_compatible_prefix) {
    has_compatible_prefix =
        std::equal(previous->pcm_samples->begin(), previous->pcm_samples->end(),
                   samples.begin());
  } else {
  }
  const bool is_same_snapshot =
      has_compatible_prefix && previous->pcm_samples->size() == samples.size();
  int computed_frame_count = 0;
  mx::array power_spectrum(0.0f);
  std::shared_ptr<const std::vector<float>> pcm_snapshot;
  if (is_same_snapshot) {
    power_spectrum = *previous->power_spectrum;
    pcm_snapshot = previous->pcm_samples;
  } else if (has_compatible_prefix) {
    const int first_new_frame = previous->stable_frame_count;
    const int new_frame_count = output_frame_count - first_new_frame;
    if (new_frame_count == 0) {
      power_spectrum = *previous->power_spectrum;
    } else {
      const mx::array new_power_spectrum =
          ComputePowerSpectrum(samples, first_new_frame, new_frame_count);
      power_spectrum =
          first_new_frame > 0
              ? mx::concatenate({mx::slice(*previous->power_spectrum, {0, 0},
                                           {first_new_frame, kFftSize / 2 + 1}),
                                 new_power_spectrum},
                                0)
              : new_power_spectrum;
      computed_frame_count = new_frame_count;
    }
    pcm_snapshot = std::make_shared<const std::vector<float>>(samples);
  } else {
    power_spectrum = ComputePowerSpectrum(samples, 0, all_stft_frame_count);
    if (final_frame == FinalStftFrame::kDropped) {
      power_spectrum =
          mx::slice(power_spectrum, {0, 0},
                    {power_spectrum.shape(0) - 1, power_spectrum.shape(1)});
    } else {
    }
    computed_frame_count = all_stft_frame_count;
    pcm_snapshot = std::make_shared<const std::vector<float>>(samples);
  }
  IncrementalLogMelCache cache{
      pcm_snapshot, power_spectrum, std::nullopt, final_frame,
      StableLogMelFrameCount(samples.size(), final_frame)};
  mx::array features =
      is_same_snapshot
          ? *previous->normalized_features
          : NormalizeLogMelSpectrum(ProjectLogMelSpectrum(power_spectrum));
  cache.normalized_features = features;
  return {std::move(features), std::move(cache), computed_frame_count};
}

mx::array WhisperWindowFeatures(const float *samples, int sample_count,
                                int mel_bin_count) {
  // Note (Dayuxiaoshui): the reference pads before the floor, so the floor
  // covers the padded tail too.
  std::vector<float> window_samples(kWhisperWindowSampleCount, 0.0f);
  std::copy(samples,
            samples + std::min(sample_count, kWhisperWindowSampleCount),
            window_samples.begin());
  const mx::array audio(window_samples.data(), {kWhisperWindowSampleCount},
                        mx::float32);
  constexpr int padding = kFftSize / 2;
  const mx::array head = mx::slice(audio, {padding}, {0}, {-1});
  const mx::array tail =
      mx::slice(audio, {kWhisperWindowSampleCount - 2},
                {kWhisperWindowSampleCount - padding - 2}, {-1});
  const mx::array padded = mx::concatenate({head, audio, tail});
  const int frame_count = 1 + (padded.shape(0) - kFftSize) / kHopLength;
  const mx::array frames =
      mx::as_strided(padded, {frame_count, kFftSize}, {kHopLength, 1}, 0);
  // Note (Dayuxiaoshui): Swift's Float.pi is one step below (float)M_PI, and
  // the window must match Voxt's bit for bit.
  const float pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  const mx::array window = mx::multiply(
      mx::array(0.5f),
      mx::subtract(
          mx::array(1.0f),
          mx::cos(mx::divide(mx::multiply(mx::array(2.0f * pi),
                                          mx::arange(kFftSize, mx::float32)),
                             mx::array(static_cast<float>(kFftSize))))));
  mx::array power = mx::square(mx::abs(
      mx::fft::rfft(mx::multiply(frames, mx::expand_dims(window, 0)), 1)));
  power = mx::slice(power, {0, 0}, {power.shape(0) - 1, power.shape(1)});
  const mx::array filters(MelFilterBank(mel_bin_count).data(),
                          {kFftSize / 2 + 1, mel_bin_count}, mx::float32);
  mx::array log_spectrum = mx::log10(
      mx::maximum(mx::matmul(mx::transpose(filters), mx::transpose(power)),
                  mx::array(kLogMelFloor)));
  log_spectrum =
      mx::maximum(log_spectrum, mx::subtract(mx::max(log_spectrum),
                                             mx::array(kLogMelDynamicRange)));
  return mx::transpose(
      mx::divide(mx::add(log_spectrum, mx::array(4.0f)), mx::array(4.0f)));
}

bool PeakIsSilent(const float *samples, size_t sample_count,
                  float peak_threshold) {
  for (size_t i = 0; i < sample_count; ++i) {
    if (std::fabs(samples[i]) >= peak_threshold) {
      return false;
    } else {
    }
  }
  return true;
}

} // namespace omni_common
