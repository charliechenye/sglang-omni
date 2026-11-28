// SPDX-License-Identifier: Apache-2.0
#include "common/asr_service.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <semaphore>
#include <stdexcept>
#include <typeinfo>

#include "civetweb.h"
#include "common/audio.h"
#include "common/generation.h"
#include "common/http.h"
#include "common/service.h"
#include "common/transcription.h"
#include "common/worker.h"
#include "nlohmann/json.hpp"

namespace asr_service {

namespace {

namespace mx = mlx::core;
using omni_common::TranscriptionResult;
using omni_server::BadRequest;
using omni_server::CancelFlag;
using omni_server::Json;
using omni_server::ReadBody;
using omni_server::TranscriptionWorker;
using omni_server::WriteJson;

constexpr auto kHeartbeatInterval = std::chrono::milliseconds(250);
constexpr auto kPreviewHeartbeatInterval = std::chrono::milliseconds(50);

class CandidateMailbox {
public:
  ~CandidateMailbox() { delete latest_.exchange(nullptr); }

  void Publish(omni_common::TranscriptCandidate candidate) {
    if (closed_.load(std::memory_order_acquire)) {
      return;
    } else {
    }
    candidate.sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto *snapshot = new omni_common::TranscriptCandidate(std::move(candidate));
    if (closed_.load(std::memory_order_acquire)) {
      delete snapshot;
      return;
    } else {
    }
    snapshot->published_candidate_count = ++published_candidate_count_;
    snapshot->snapshot_build_ms_total =
        snapshot_build_ms_total_ += snapshot->snapshot_build_ms;
    auto *replaced = latest_.exchange(snapshot, std::memory_order_acq_rel);
    if (replaced != nullptr) {
      ++coalesced_candidate_count_;
    } else {
    }
    delete replaced;
    Signal();
  }

  std::unique_ptr<omni_common::TranscriptCandidate> Take() {
    signaled_.store(false, std::memory_order_release);
    return std::unique_ptr<omni_common::TranscriptCandidate>(
        latest_.exchange(nullptr, std::memory_order_acq_rel));
  }

  bool WaitFor(std::chrono::milliseconds duration) {
    return ready_.try_acquire_for(duration);
  }

  uint64_t PublishedCount() const { return published_candidate_count_; }
  uint64_t CoalescedCount() const { return coalesced_candidate_count_; }

  void Complete() {
    closed_.store(true, std::memory_order_release);
    Signal();
  }

private:
  void Signal() {
    if (!signaled_.exchange(true, std::memory_order_acq_rel)) {
      ready_.release();
    } else {
    }
  }

  std::atomic<omni_common::TranscriptCandidate *> latest_{nullptr};
  std::binary_semaphore ready_{0};
  std::atomic<uint64_t> sequence_{0};
  // One model-worker producer updates these before publishing the snapshot;
  // the HTTP writer only reads the copied fields after latest_.exchange.
  uint64_t published_candidate_count_ = 0;
  uint64_t coalesced_candidate_count_ = 0;
  double snapshot_build_ms_total_ = 0.0;
  std::atomic<bool> signaled_{false};
  std::atomic<bool> closed_{false};
};

struct ServerState {
  TranscriptionWorker *worker = nullptr;
  const ServedModel *model = nullptr;
  std::string model_name;
};

// Speaker segments, for models that produce them.
void AddSegments(Json &body, const TranscriptionResult &result) {
  if (!result.segments.empty()) {
    body["segments"] = omni_common::SpeakerSegmentsJson(result.segments);
  } else {
  }
}

Json DoneEvent(const TranscriptionResult &result,
               bool include_generation_metadata) {
  Json event = {{"type", "transcript.text.done"}, {"text", result.text}};
  AddSegments(event, result);
  if (include_generation_metadata) {
    event["generation_metadata"] = {
        {"generated_token_count", result.generated_token_count},
        {"language",
         result.language.has_value() ? Json(*result.language) : Json()},
        {"finish_reason", omni_common::FinishReasonName(result.finish_reason)}};
    if (result.authoritative_completion_elapsed_ms.has_value()) {
      event["generation_metadata"]["authoritative_completion_elapsed_ms"] =
          *result.authoritative_completion_elapsed_ms;
    }
    if (!result.window_texts.empty()) {
      event["generation_metadata"]["window_texts"] = result.window_texts;
    }
    if (result.reused_audio_token_count > 0) {
      event["generation_metadata"]["reused_audio_token_count"] =
          result.reused_audio_token_count;
    } else {
    }
    if (result.reused_prompt_token_count > 0) {
      event["generation_metadata"]["reused_prompt_token_count"] =
          result.reused_prompt_token_count;
    } else {
    }
    if (result.verify_round_count > 0) {
      event["generation_metadata"]["verify_round_count"] =
          result.verify_round_count;
      event["generation_metadata"]["proposed_draft_token_count"] =
          result.proposed_draft_token_count;
      event["generation_metadata"]["accepted_draft_token_count"] =
          result.accepted_draft_token_count;
    } else {
    }
    event["generation_metadata"]["verified_guess_token_count"] =
        result.verified_guess_token_count;
    event["generation_metadata"]["accepted_guess_token_count"] =
        result.accepted_guess_token_count;
    event["generation_metadata"]["decode_paths"] = result.decode_paths;
    event["generation_metadata"]["draft_decoder_used"] =
        result.draft_decoder_used;
    event["generation_metadata"]["batch_decode_used"] =
        result.batch_decode_used;
    event["generation_metadata"]["native_chunked"] = result.native_chunked;
  } else {
  }
  return event;
}

Json CandidateProtocolEvent(const std::string &cadence) {
  return {{"type", "transcript.text.candidate_protocol"},
          {"supported", true},
          {"protocol_version", 1},
          {"candidate_cadence", cadence}};
}

Json CandidateSummaryEvent(const CandidateMailbox &mailbox) {
  return {{"type", "transcript.text.candidate_summary"},
          {"published_candidate_count", mailbox.PublishedCount()},
          {"coalesced_candidate_count", mailbox.CoalescedCount()}};
}

const Json &FailureEvent() {
  static const Json event = {{"type", "error"},
                             {"error",
                              {{"type", "server_error"},
                               {"code", "transcription_failed"},
                               {"message", "Transcription failed."}}}};
  return event;
}

bool WriteSse(mg_connection *connection, const std::string &payload) {
  const std::string line = "data: " + payload + "\n\n";
  return mg_write(connection, line.data(), line.size()) > 0;
}

Json CandidateEvent(const omni_common::TranscriptCandidate &candidate,
                    bool delivery_during_terminal_drain) {
  Json event = {{"type", "transcript.text.candidate"},
                {"text", candidate.text},
                {"sequence", candidate.sequence},
                {"generated_token_count", candidate.generated_token_count},
                {"elapsed_ms", candidate.elapsed_ms},
                {"first_transcript_token_elapsed_ms",
                 candidate.first_transcript_token_elapsed_ms},
                {"snapshot_build_ms", candidate.snapshot_build_ms},
                {"published_candidate_count",
                 candidate.published_candidate_count},
                {"snapshot_build_ms_total", candidate.snapshot_build_ms_total}};
  event["delivery_during_terminal_drain"] = delivery_during_terminal_drain;
  if (candidate.language.has_value()) {
    event["language"] = *candidate.language;
  } else {
  }
  return event;
}

int HandleTranscriptions(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  if (!omni_server::IsPost(connection)) {
    return WriteJson(connection, 405, {{"detail", "Method Not Allowed"}});
  } else {
  }
  const char *content_type = mg_get_header(connection, "Content-Type");
  const auto form = omni_common::ParseMultipartForm(
      content_type ? content_type : "", ReadBody(connection));
  if (!form.has_value() || form->count("file") == 0 ||
      !form->at("file").filename.has_value()) {
    return BadRequest(connection, "file is required");
  } else {
  }
  const bool stream = omni_common::FormFlag(TextField(*form, "stream"));
  const bool include_generation_metadata =
      omni_common::FormFlag(TextField(*form, "include_generation_metadata"));
  const bool include_transcript_candidates =
      omni_common::FormFlag(TextField(*form, "include_transcript_candidates"));
  if (include_generation_metadata && !stream) {
    return BadRequest(connection,
                      "include_generation_metadata requires stream=true");
  } else if (include_transcript_candidates && !stream) {
    return BadRequest(connection,
                      "include_transcript_candidates requires stream=true");
  } else {
  }
  const std::shared_ptr<CandidateMailbox> candidate_mailbox =
      include_transcript_candidates ? std::make_shared<CandidateMailbox>()
                                    : nullptr;
  const std::string candidate_cadence =
      TextField(*form, "candidate_cadence").value_or("bounded");
  Transcription transcription;
  try {
    std::vector<float> samples =
        omni_common::DecodeWav(form->at("file").value);
    if (candidate_mailbox != nullptr) {
      transcription = state->model->PrepareWithCandidates(
          std::move(samples), *form,
          [candidate_mailbox](omni_common::TranscriptCandidate candidate) {
            candidate_mailbox->Publish(std::move(candidate));
          });
    } else {
      transcription = state->model->Prepare(std::move(samples), *form);
    }
  } catch (const std::invalid_argument &error) {
    return BadRequest(connection, error.what());
  } catch (const std::out_of_range &) {
    return BadRequest(connection, "a numeric field is out of range");
  }
  const CancelFlag cancel = omni_server::NewCancelFlag();
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  state->worker->Submit(std::move(transcription), cancel,
                        [promise, candidate_mailbox](
                            std::optional<TranscriptionResult> result,
                            std::exception_ptr error) {
                          if (error) {
                            promise->set_exception(error);
                          } else {
                            promise->set_value(std::move(*result));
                          }
                          if (candidate_mailbox != nullptr) {
                            candidate_mailbox->Complete();
                          } else {
                          }
                        });
  if (!stream) {
    try {
      const TranscriptionResult result = future.get();
      Json body = {{"text", result.text}};
      AddSegments(body, result);
      return WriteJson(connection, 200, body);
    } catch (...) {
      return WriteJson(connection, 500, {{"detail", "Transcription failed."}});
    }
  } else {
  }
  mg_printf(connection, "HTTP/1.1 200 OK\r\nContent-Type: "
                        "text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Connection: close\r\n\r\n");
  if (candidate_mailbox != nullptr) {
    WriteSse(connection, CandidateProtocolEvent(candidate_cadence).dump());
  } else {
  }
  // Note (Jiaxin Deng): SSE comment heartbeats fail to write once the peer is
  // gone, which cancels the decode it was waiting for.
  const auto heartbeat_interval =
      omni_common::FormFlag(TextField(*form, "preview"))
          ? kPreviewHeartbeatInterval
          : kHeartbeatInterval;
  if (candidate_mailbox == nullptr) {
    while (future.wait_for(heartbeat_interval) != std::future_status::ready) {
      if (mg_write(connection, ":\n\n", 3) <= 0) {
        cancel->store(true);
      } else {
      }
    }
  } else {
    while (future.wait_for(std::chrono::milliseconds(0)) !=
           std::future_status::ready) {
      if (candidate_mailbox->WaitFor(heartbeat_interval)) {
        std::unique_ptr<omni_common::TranscriptCandidate> candidate =
            candidate_mailbox->Take();
        if (candidate != nullptr &&
            !WriteSse(connection, CandidateEvent(*candidate, false).dump())) {
          cancel->store(true);
        } else {
        }
      } else if (mg_write(connection, ":\n\n", 3) <= 0) {
        cancel->store(true);
      } else {
      }
    }
  }
  try {
    const TranscriptionResult result = future.get();
    WriteSse(connection, DoneEvent(result, include_generation_metadata).dump());
  } catch (const omni_common::GenerationCancelled &) {
    return 200;
  } catch (const std::exception &error) {
    std::cerr << "transcription failed: " << typeid(error).name() << "\n";
    WriteSse(connection, FailureEvent().dump());
  }
  if (candidate_mailbox != nullptr) {
    std::unique_ptr<omni_common::TranscriptCandidate> candidate =
        candidate_mailbox->Take();
    if (candidate != nullptr) {
      WriteSse(connection, CandidateEvent(*candidate, true).dump());
    } else {
    }
    WriteSse(connection, CandidateSummaryEvent(*candidate_mailbox).dump());
  } else {
  }
  WriteSse(connection, "[DONE]");
  return 200;
}

struct SocketState {
  std::shared_ptr<omni_server::RealtimeConnection> session;
  std::string fragments;
};

void SocketReady(mg_connection *connection, void *data) {
  const auto *factory = static_cast<const RealtimeFactory *>(data);
  auto *socket = new SocketState();
  socket->session = (*factory)([connection](const std::string &text) {
    return omni_server::SendWebSocketText(connection, text);
  });
  mg_set_user_connection_data(connection, socket);
}

int SocketData(mg_connection *connection, int bits, char *data, size_t length,
               void *) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  const int opcode = bits & 0x0F;
  if (socket == nullptr || opcode == MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE) {
    return 0;
  } else if (opcode == MG_WEBSOCKET_OPCODE_PING) {
    mg_lock_connection(connection);
    mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_PONG, data, length);
    mg_unlock_connection(connection);
    return 1;
  } else if (opcode == MG_WEBSOCKET_OPCODE_PONG) {
    return 1;
  } else {
  }
  socket->fragments.append(data, length);
  if ((bits & 0x80) == 0) {
    return 1;
  } else {
  }
  const std::string message = std::move(socket->fragments);
  socket->fragments.clear();
  nlohmann::json event;
  try {
    event = nlohmann::json::parse(message);
  } catch (const nlohmann::json::exception &) {
    socket->session->SendError("invalid_request_error", "invalid_json",
                               "Events must be JSON.");
    return 1;
  }
  if (!event.is_object()) {
    return 0;
  } else {
  }
  try {
    return socket->session->Handle(event) ? 1 : 0;
  } catch (const std::exception &error) {
    // Note (Jiaxin Deng): log the type alone, never audio or text.
    std::cerr << "realtime session failed: " << typeid(error).name() << "\n";
    return 0;
  }
}

void SocketClosed(const mg_connection *connection, void *) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  if (socket != nullptr) {
    socket->session->Close();
    delete socket;
  } else {
  }
}

class AsrService : public omni_server::Service {
public:
  AsrService(const std::filesystem::path &directory, const ModelLoader &load,
             int max_batch,
             const std::optional<omni_server::BatchRunner> &batch_runner) {
    worker_ = std::make_unique<TranscriptionWorker>(
        [&]() { model_ = load(directory); },
        [this]() { model_->ReleaseIdleMemory(); }, max_batch, batch_runner);
  }

  void SetModelName(const std::string &model_name,
                    bool has_chosen_model_name) override {
    model_->set_model_name(model_name, has_chosen_model_name);
  }

  void Register(mg_context *context, const std::string &model_name) override {
    state_ = {worker_.get(), model_.get(), model_name};
    if (model_->Transcribes()) {
      mg_set_request_handler(context, "/v1/audio/transcriptions$",
                             HandleTranscriptions, &state_);
    } else {
    }
    model_->AddHandlers(context, *worker_);
  }

  std::map<std::string, int> RequestStates() const override {
    return model_->RequestStates(*worker_);
  }

  void CancelAll() override { worker_->CancelAll(); }

private:
  std::unique_ptr<ServedModel> model_;
  std::unique_ptr<TranscriptionWorker> worker_;
  ServerState state_;
};

} // namespace

void ServedModel::ReleaseIdleMemory() {
  // Note (khazic): a decode keeps MLX's buffer cache, since freeing every
  // step's buffers slows each token; an idle server returns it, so it
  // holds only the model. The synchronize lets the step a decode queued
  // before it stopped return its buffers first.
  mx::synchronize();
  mx::clear_cache();
}

std::optional<std::string> TextField(const FormFields &form,
                                     const std::string &name) {
  return omni_server::Field(form, name);
}

std::optional<int> IntegerField(const FormFields &form,
                                const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size()) {
    throw std::invalid_argument(name + " must be an integer");
  } else {
  }
  return value;
}

std::optional<float> NumberField(const FormFields &form,
                                 const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  float value = 0;
  try {
    value = std::stof(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size() || !std::isfinite(value)) {
    throw std::invalid_argument(name + " must be a finite number");
  } else {
  }
  return value;
}

void AddRealtimeHandler(mg_context *context, const RealtimeFactory &factory) {
  mg_set_websocket_handler(context, "/v1/realtime", nullptr, SocketReady,
                           SocketData, SocketClosed,
                           const_cast<RealtimeFactory *>(&factory));
}

omni_server::ServedKind CreateServedKind(const ServedKind &kind) {
  return {
      kind.model_kind,
      [load = kind.load, max_batch = kind.max_batch,
       batch_runner =
           kind.batch_runner](const std::filesystem::path &directory) {
        return std::make_unique<AsrService>(
            directory, load, max_batch ? max_batch() : 1, batch_runner);
      },
      kind.flags,
      kind.check_flags,
  };
}

int Serve(int argc, char **argv, const std::vector<ServedKind> &kinds) {
  std::vector<omni_server::ServedKind> services;
  for (const auto &kind : kinds) {
    services.push_back(CreateServedKind(kind));
  }
  return omni_server::Serve(argc, argv, services);
}

} // namespace asr_service
