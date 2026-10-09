# SPDX-License-Identifier: Apache-2.0
"""Session-resident streaming perception for MiniCPM-o native duplex."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Literal, Protocol, TypedDict

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
from sglang_omni.models.minicpm_o.components.whisper_encoder import AudioEncoderState
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SAMPLE_RATE = 16000
UNIT_MS = 1000
FIRST_CHUNK_MS = 1035
IMAGE_TOKENS = 64
MAX_FRAME_PIXELS = 4096 * 4096
IMAGE_PREFETCH_MODALITY = "image_prefetch"
IMAGE_PREFETCH_CLEAR_MODALITY = "image_prefetch_clear"


class ImageEncoder(Protocol):
    def __call__(
        self, *, pixel_values: list[torch.Tensor], tgt_sizes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pass


class ImageFeatureBatch(TypedDict):
    pixel_values: list[list[torch.Tensor]]
    tgt_sizes: list[torch.Tensor]


class StreamingConfig(TypedDict):
    effective_first_chunk_ms: float


class ProcessorAudioFeatures(TypedDict):
    audio_features: torch.Tensor
    audio_feature_lens: list[torch.Tensor]


class StreamingAudioProcessor(Protocol):
    """The checkpoint processor's streaming surface used by one session."""

    def set_streaming_mode(
        self,
        *,
        mode: str,
        chunk_ms: int,
        first_chunk_ms: int,
        cnn_redundancy_ms: int,
        enable_sliding_window: bool,
        slide_trigger_seconds: float,
        slide_stride_seconds: float,
    ) -> None:
        pass

    def process_image(
        self, images: list[Image.Image], *, max_slice_nums: int
    ) -> ImageFeatureBatch:
        pass

    def get_streaming_chunk_size(self) -> int:
        pass

    def get_streaming_config(self) -> StreamingConfig:
        pass

    def process_audio(
        self, audio: np.ndarray, *, sampling_rate: int
    ) -> ProcessorAudioFeatures:
        pass

    def process_audio_streaming(
        self, audio: np.ndarray, *, reset: bool, return_batch_feature: bool
    ) -> ProcessorAudioFeatures:
        pass


class ProcessorFactory(Protocol):
    def __call__(self) -> StreamingAudioProcessor:
        pass


class EmbeddingSpanPlan(TypedDict):
    modality: Literal["audio", "image"]
    token_start: int
    token_end: int
    embedding_start: int
    embedding_end: int


class PerceptionStepPlan(TypedDict):
    token_ids: list[int]
    input_embeds: torch.Tensor
    embedding_spans: list[EmbeddingSpanPlan]


@dataclass(kw_only=True)
class AudioFeatureBatch:
    audio_features: torch.Tensor
    audio_feature_lens: torch.Tensor


@dataclass(kw_only=True)
class PreparedImageFeatures:
    pixel_values: list[torch.Tensor]
    tgt_sizes: torch.Tensor


def audio_feature_batch(processor_output: ProcessorAudioFeatures) -> AudioFeatureBatch:
    return AudioFeatureBatch(
        audio_features=processor_output["audio_features"],
        audio_feature_lens=torch.cat(
            [length.reshape(-1) for length in processor_output["audio_feature_lens"]]
        ),
    )


@dataclass(kw_only=True)
class MiniCPMOPerceptionState:
    """All mutable checkpoint perception state owned by one session."""

    tokenizer: PreTrainedTokenizerBase
    processor: StreamingAudioProcessor
    audio_encoder: MiniCPMOAudioEncoder
    max_slice_nums: int
    image_encoder: ImageEncoder
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    audio_chunk_index: int = 0
    audio_encoder_state: AudioEncoderState | None = None
    prefix_token_ids: list[int] = field(default_factory=list)
    prefix_embeds: torch.Tensor | None = None
    prefix_schema: list[tuple[Literal["token", "audio"], int]] = field(
        default_factory=list
    )
    prefetched_images: dict[int, list[tuple[bytes, PreparedImageFeatures]]] = field(
        default_factory=dict
    )
    next_image_unit_index: int = 0
    is_open: bool = True

    @classmethod
    def open(
        cls,
        *,
        tokenizer: PreTrainedTokenizerBase,
        processor: StreamingAudioProcessor,
        audio_encoder: MiniCPMOAudioEncoder,
        prompt: str,
        reference_audio: bytes,
        image_encoder: ImageEncoder,
        max_slice_nums: int,
    ) -> MiniCPMOPerceptionState:
        processor.set_streaming_mode(
            mode="exact",
            chunk_ms=UNIT_MS,
            first_chunk_ms=FIRST_CHUNK_MS,
            cnn_redundancy_ms=20,
            enable_sliding_window=True,
            slide_trigger_seconds=30.0,
            slide_stride_seconds=10.0,
        )
        state = cls(
            tokenizer=tokenizer,
            processor=processor,
            audio_encoder=audio_encoder,
            image_encoder=image_encoder,
            max_slice_nums=max_slice_nums,
        )
        prompt_ids = list(
            tokenizer.encode(
                f"<|im_start|>system\n{prompt}\n", add_special_tokens=False
            )
        )
        im_end_ids = list(tokenizer.encode("<|im_end|>", add_special_tokens=False))
        state.prefix_token_ids = list(prompt_ids)
        state.prefix_token_ids.append(
            tokenizer.convert_tokens_to_ids("<|audio_start|>")
        )
        reference_waveform, _ = AudioMediaIO(target_sr=SAMPLE_RATE).load_bytes(
            reference_audio
        )
        waveform = np.asarray(reference_waveform, dtype=np.float32).reshape(-1)
        batch = audio_feature_batch(
            processor.process_audio(waveform, sampling_rate=SAMPLE_RATE)
        )
        state.prefix_embeds = audio_encoder(
            audio_features=batch.audio_features,
            audio_feature_lens=batch.audio_feature_lens,
        )["audio_embeds"]
        count = int(state.prefix_embeds.shape[0])
        state.prefix_token_ids.extend([tokenizer.unk_token_id] * count)
        state.prefix_token_ids.append(tokenizer.convert_tokens_to_ids("<|audio_end|>"))
        state.prefix_token_ids.extend(im_end_ids)
        state.prefix_schema = [
            ("token", len(prompt_ids) + 1),
            ("audio", count),
            ("token", 1 + len(im_end_ids)),
        ]
        return state

    def close(self) -> None:
        self.is_open = False
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.audio_encoder_state = None
        self.prefix_embeds = None
        self.prefetched_images.clear()

    def take_prefetched_image(
        self, unit_index: int, encoded_image: bytes
    ) -> PreparedImageFeatures | None:
        prefetched_images = self.prefetched_images.get(unit_index, [])
        for image_index, (cached_image, prepared_image) in enumerate(prefetched_images):
            if cached_image == encoded_image:
                prefetched_images.pop(image_index)
                if prefetched_images:
                    pass
                else:
                    self.prefetched_images.pop(unit_index, None)
                return prepared_image
            else:
                pass
        return None

    def finish_image_unit(self, unit_index: int) -> None:
        self.next_image_unit_index = max(self.next_image_unit_index, unit_index + 1)
        self.prefetched_images.pop(unit_index, None)

    def prepared_image_nbytes(
        self, prepared_image_features: PreparedImageFeatures
    ) -> int:
        return sum(
            pixel_value.numel() * pixel_value.element_size()
            for pixel_value in prepared_image_features.pixel_values
        ) + (
            prepared_image_features.tgt_sizes.numel()
            * prepared_image_features.tgt_sizes.element_size()
        )

    def held(self) -> ResourceUsage:
        if not self.is_open:
            return ResourceUsage()
        else:
            size = (
                int(self.audio_buffer.nbytes)
                + (
                    self.audio_encoder_state.nbytes
                    if self.audio_encoder_state is not None
                    else 0
                )
                + estimate_cache_bytes(self.prefix_embeds)
                + sum(
                    self.prepared_image_nbytes(prepared_image_features)
                    for prefetched_images in self.prefetched_images.values()
                    for _, prepared_image_features in prefetched_images
                )
            )
            return ResourceUsage(slots={"perception": 1}, bytes=max(size, 1))

    def encode_audio(self, waveform: np.ndarray) -> torch.Tensor:
        need_samples = self.processor.get_streaming_chunk_size()
        # note (Junnan Li): The checkpoint front-pads the first chunk to 1035 ms so the encoder's CNN context is full.
        if self.audio_chunk_index == 0:
            first_chunk_samples = FIRST_CHUNK_MS * SAMPLE_RATE // 1000
            padding = max(
                first_chunk_samples - self.audio_buffer.size - waveform.size, 0
            )
        else:
            padding = 0
        self.audio_buffer = np.concatenate(
            [np.zeros(padding, dtype=np.float32), self.audio_buffer, waveform]
        )
        assert self.audio_buffer.size >= need_samples, (
            self.audio_buffer.size,
            need_samples,
        )
        batch = audio_feature_batch(
            self.processor.process_audio_streaming(
                self.audio_buffer[:need_samples].copy(),
                reset=False,
                return_batch_feature=True,
            )
        )
        audio_embeds, self.audio_encoder_state = self.audio_encoder.forward_streaming(
            audio_features=batch.audio_features,
            audio_feature_lens=batch.audio_feature_lens,
            state=self.audio_encoder_state,
            prefix_extra_frames=0 if self.audio_chunk_index == 0 else 2,
            suffix_extra_frames=2,
        )
        if self.audio_chunk_index == 0:
            consumed_ms = int(
                self.processor.get_streaming_config()["effective_first_chunk_ms"]
            )
            consumed_samples = consumed_ms * SAMPLE_RATE // 1000
        else:
            consumed_samples = need_samples
        self.audio_buffer = self.audio_buffer[consumed_samples:].copy()
        self.audio_chunk_index += 1
        return audio_embeds

    def prepare_image(self, encoded_image: bytes) -> PreparedImageFeatures:
        with Image.open(BytesIO(encoded_image)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("unit image must be JPEG or PNG")
            elif image.width * image.height > MAX_FRAME_PIXELS:
                raise ValueError("unit image exceeds pixel limit")
            else:
                frame = image.convert("RGB")
        processed_image = self.processor.process_image(
            [frame], max_slice_nums=self.max_slice_nums
        )
        return PreparedImageFeatures(
            pixel_values=processed_image["pixel_values"][0],
            tgt_sizes=processed_image["tgt_sizes"][0],
        )

    def encode_images(
        self, prepared_image_features: tuple[PreparedImageFeatures, ...]
    ) -> tuple[torch.Tensor, ...]:
        if not prepared_image_features:
            return ()
        else:
            pass

        pixel_values = [
            slice_pixel_values
            for prepared_image in prepared_image_features
            for slice_pixel_values in prepared_image.pixel_values
        ]
        tgt_sizes = torch.cat(
            [prepared_image.tgt_sizes for prepared_image in prepared_image_features],
            dim=0,
        )
        image_embeds = self.image_encoder(
            pixel_values=pixel_values,
            tgt_sizes=tgt_sizes,
        )["image_embeds"]
        row_counts = [
            len(prepared_image.pixel_values) * IMAGE_TOKENS
            for prepared_image in prepared_image_features
        ]
        assert image_embeds.ndim == 2
        assert image_embeds.shape[0] == sum(row_counts)
        return image_embeds.split(row_counts, dim=0)

    def build_step_plan(
        self, audio_embeds: torch.Tensor, image_embeds: tuple[torch.Tensor, ...] = ()
    ) -> PerceptionStepPlan:
        token_ids: list[int] = []
        embedding_blocks: list[torch.Tensor] = []
        spans: list[EmbeddingSpanPlan] = []

        def add_embeds(
            embeddings: torch.Tensor, modality: Literal["audio", "image"] = "audio"
        ) -> None:
            token_start = len(token_ids)
            row_count = int(embeddings.shape[0])
            embedding_start = sum(int(block.shape[0]) for block in embedding_blocks)
            token_ids.extend([self.tokenizer.unk_token_id] * row_count)
            embedding_blocks.append(embeddings)
            spans.append(
                EmbeddingSpanPlan(
                    modality=modality,
                    token_start=token_start,
                    token_end=token_start + row_count,
                    embedding_start=embedding_start,
                    embedding_end=embedding_start + row_count,
                )
            )

        if self.audio_chunk_index == 1 and self.prefix_token_ids:
            token_cursor = 0
            embedding_cursor = 0
            for segment_kind, segment_length in self.prefix_schema:
                if segment_kind == "token":
                    token_ids.extend(
                        self.prefix_token_ids[
                            token_cursor : token_cursor + segment_length
                        ]
                    )
                else:
                    assert self.prefix_embeds is not None
                    add_embeds(
                        self.prefix_embeds[
                            embedding_cursor : embedding_cursor + segment_length
                        ]
                    )
                    embedding_cursor += segment_length
                token_cursor += segment_length
        else:
            pass

        token_ids.append(self.tokenizer.convert_tokens_to_ids("<unit>"))
        for frame_embeds in image_embeds:
            assert (
                frame_embeds.ndim == 2
                and frame_embeds.shape[1] == audio_embeds.shape[1]
            )
            assert (
                frame_embeds.shape[0] > 0 and frame_embeds.shape[0] % IMAGE_TOKENS == 0
            )
            for slice_index, slice_embeds in enumerate(
                frame_embeds.split(IMAGE_TOKENS)
            ):
                marker = "image" if slice_index == 0 else "slice"
                token_ids.append(self.tokenizer.convert_tokens_to_ids(f"<{marker}>"))
                add_embeds(slice_embeds, "image")
                token_ids.append(self.tokenizer.convert_tokens_to_ids(f"</{marker}>"))
        add_embeds(audio_embeds)
        return PerceptionStepPlan(
            token_ids=token_ids,
            input_embeds=torch.cat(embedding_blocks, dim=0),
            embedding_spans=spans,
        )
