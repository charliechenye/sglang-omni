# SPDX-License-Identifier: Apache-2.0
"""Golden parity tests for Qwen3OmniThinkerModelRunner._inject_multimodal_embeds.

Every test builds its expected embeddings directly from the constructed inputs,
so the oracle stays independent of how the scatter is implemented. The
test_no_host_syncs_* tests additionally assert that the merge calls none of
``Tensor.item`` / ``Tensor.any`` / ``Tensor.nonzero`` / ``torch.where``.
"""
from __future__ import annotations

import types

import torch

from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)

VOCAB = 100
HIDDEN = 8
IMAGE_ID = 91
VIDEO_ID = 92
AUDIO_ID = 93
TEXT = 7


def _runner() -> Qwen3OmniThinkerModelRunner:
    torch.manual_seed(0)
    r = object.__new__(Qwen3OmniThinkerModelRunner)
    r._embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
    r._image_token_id = IMAGE_ID
    r._video_token_id = VIDEO_ID
    r._audio_token_id = AUDIO_ID
    return r


def _req(input_ids, omni_model_inputs, *, is_chunked=0, consumed=None, positions=True):
    req = types.SimpleNamespace(
        origin_input_ids=list(input_ids),
        omni_model_inputs=omni_model_inputs,
        _omni_consumed=consumed,
        is_chunked=is_chunked,
    )
    if positions:
        # note (chenrui): build_sglang_thinker_request records these for every
        # request, so omitting them pins a fallback the served path never takes.
        pad_values = (omni_model_inputs or {}).get("pad_values", {})
        ids = torch.tensor(input_ids, dtype=torch.long)
        req._omni_mm_positions = {
            modality: (ids == pad_values.get(modality, token_id)).nonzero(
                as_tuple=True
            )[0]
            for modality, token_id in (
                ("image", IMAGE_ID),
                ("video", VIDEO_ID),
                ("audio", AUDIO_ID),
            )
        }
    return req


def _batches(reqs, chunk_ids=None, prefix_lens=None):
    """Build (forward_batch, schedule_batch) stand-ins.

    chunk_ids: per-request token ids for THIS extend step (defaults to the
    full prompt). prefix_lens: tokens already prefilled per request.
    """
    if chunk_ids is None:
        chunk_ids = [r.origin_input_ids for r in reqs]
    if prefix_lens is None:
        prefix_lens = [0] * len(reqs)
    flat = [t for ids in chunk_ids for t in ids]
    forward_batch = types.SimpleNamespace(
        input_ids=torch.tensor(flat, dtype=torch.long),
        extend_seq_lens_cpu=[len(ids) for ids in chunk_ids],
        extend_prefix_lens_cpu=list(prefix_lens),
    )
    schedule_batch = types.SimpleNamespace(reqs=list(reqs))
    return forward_batch, schedule_batch


def _base_embeds(runner, forward_batch):
    ids = forward_batch.input_ids.clamp(0, VOCAB - 1)
    return runner._embed_tokens(ids).detach().clone()


def _rand(n):
    return torch.randn(n, HIDDEN)


class _NativeAudioItem:
    modality = types.SimpleNamespace(name="AUDIO")
    format = types.SimpleNamespace(name="PRECOMPUTED_EMBEDDING")

    def __init__(self, positions, embeds):
        self.precomputed_embeddings = embeds
        self.model_specific_data = {
            "positions_cpu": torch.tensor(positions, dtype=torch.long)
        }

    def is_precomputed_embedding(self):
        return True


def _native_audio_req(input_ids, positions, embeds, *, is_chunked=0):
    req = _req(input_ids, None, is_chunked=is_chunked)
    req.multimodal_inputs = types.SimpleNamespace(
        mm_items=[_NativeAudioItem(positions, embeds)]
    )
    return req


class _ExtendMode:
    def is_extend(self):
        return True


def _custom_forward_batch(fb):
    fb.forward_mode = _ExtendMode()
    fb.positions = torch.arange(fb.input_ids.numel(), dtype=torch.long)
    fb.mrope_positions = None
    return fb


def test_text_only_batch_returns_none():
    runner = _runner()
    req = _req([TEXT, TEXT, TEXT], None)
    fb, sb = _batches([req])
    assert runner._inject_multimodal_embeds(fb, sb) is None


def test_mixed_native_audio_and_legacy_audio_merge():
    runner = _runner()
    native_ids = [TEXT, AUDIO_ID, AUDIO_ID, TEXT]
    legacy_ids = [TEXT, AUDIO_ID, TEXT]
    native_audio = _rand(2)
    legacy_audio = _rand(1)
    reqs = [
        _native_audio_req(native_ids, [1, 2], native_audio),
        _req(legacy_ids, {"audio_embeds": legacy_audio}),
    ]
    fb, sb = _batches(reqs)

    out, ds, masks = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = native_audio
    expected[5:6] = legacy_audio
    torch.testing.assert_close(out, expected)
    assert ds is None and masks is None


def test_mixed_native_audio_and_legacy_visual_deepstack_merge():
    runner = _runner()
    native_audio = _rand(2)
    image_audio = _rand(2)
    deepstack = [_rand(2), _rand(2)]
    reqs = [
        _native_audio_req([TEXT, AUDIO_ID, TEXT, AUDIO_ID], [1, 3], native_audio),
        _req(
            [TEXT, IMAGE_ID, IMAGE_ID, TEXT],
            {
                "image_embeds": image_audio,
                "image_deepstack_visual_embeds": deepstack,
            },
        ),
    ]
    fb, sb = _batches(reqs)

    out, ds, masks = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1] = native_audio[0]
    expected[3] = native_audio[1]
    expected[5:7] = image_audio
    torch.testing.assert_close(out, expected)
    assert masks.tolist() == [False, False, False, False, False, True, True, False]
    for actual, expected_layer in zip(ds, deepstack):
        torch.testing.assert_close(actual, expected_layer)


def test_native_audio_and_text_only_requests_stay_on_standard_forward():
    runner = _runner()
    reqs = [
        _native_audio_req([TEXT, AUDIO_ID, TEXT], [1], _rand(1)),
        _req([TEXT, TEXT], None),
    ]
    fb, sb = _batches(reqs)
    fb = _custom_forward_batch(fb)
    sb.forward_mode = _ExtendMode()

    assert runner.custom_prefill_forward(fb, sb, reqs) is None


def test_two_native_audio_requests_and_one_legacy_visual_merge():
    runner = _runner()
    native0 = _rand(1)
    native1 = _rand(2)
    image = _rand(1)
    reqs = [
        _native_audio_req([AUDIO_ID, TEXT], [0], native0),
        _native_audio_req([TEXT, AUDIO_ID, TEXT, AUDIO_ID], [1, 3], native1),
        _req([TEXT, IMAGE_ID], {"image_embeds": image}),
    ]
    fb, sb = _batches(reqs)

    out, ds, masks = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[0] = native0[0]
    expected[3] = native1[0]
    expected[5] = native1[1]
    expected[7] = image[0]
    torch.testing.assert_close(out, expected)
    assert ds is None and masks is None


def test_chunked_native_audio_and_legacy_request_merge_current_chunk():
    runner = _runner()
    native_prompt = [TEXT, AUDIO_ID, AUDIO_ID, TEXT, AUDIO_ID, TEXT]
    native_audio = _rand(3)
    native = _native_audio_req(
        native_prompt, [1, 2, 4], native_audio, is_chunked=1
    )
    legacy = _req(
        [TEXT, IMAGE_ID, TEXT, IMAGE_ID],
        {"image_embeds": _rand(2)},
        is_chunked=1,
    )
    fb, sb = _batches(
        [native, legacy],
        chunk_ids=[native_prompt[:4], legacy.origin_input_ids[:3]],
        prefix_lens=[0, 0],
    )

    out, ds, masks = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = native_audio[:2]
    expected[5] = legacy.omni_model_inputs["image_embeds"][0]
    torch.testing.assert_close(out, expected)
    assert ds is None and masks is None


def test_mixed_custom_prefill_is_eager_and_uses_one_scatter(monkeypatch):
    runner = _runner()
    native_audio = _rand(2)
    legacy_audio = _rand(1)
    reqs = [
        _native_audio_req([TEXT, AUDIO_ID, AUDIO_ID], [1, 2], native_audio),
        _req([TEXT, AUDIO_ID], {"audio_embeds": legacy_audio}),
    ]
    fb, sb = _batches(reqs)
    fb = _custom_forward_batch(fb)
    sb.forward_mode = _ExtendMode()

    clone_calls = 0
    index_copy_calls = 0
    original_clone = torch.Tensor.clone
    original_index_copy = torch.Tensor.index_copy_

    def counted_clone(self, *args, **kwargs):
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(self, *args, **kwargs)

    def counted_index_copy(self, *args, **kwargs):
        nonlocal index_copy_calls
        index_copy_calls += 1
        return original_index_copy(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counted_clone)
    monkeypatch.setattr(torch.Tensor, "index_copy_", counted_index_copy)
    runner._forward_with_omni_embeds = lambda *args: types.SimpleNamespace(
        can_run_cuda_graph=False
    )

    result = runner.custom_prefill_forward(fb, sb, reqs)

    assert result.can_run_cuda_graph is False
    assert index_copy_calls == 1
    assert clone_calls == 0


def test_single_request_image_merge():
    runner = _runner()
    ids = [TEXT, IMAGE_ID, IMAGE_ID, IMAGE_ID, TEXT]
    image_embeds = _rand(3)
    req = _req(ids, {"image_embeds": image_embeds})
    fb, sb = _batches([req])

    out, ds, masks = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:4] = image_embeds
    torch.testing.assert_close(out, expected)
    assert ds is None and masks is None
    assert req.omni_model_inputs is None


def test_mixed_batch_image_audio_and_audio_only_and_text():
    runner = _runner()
    ids0 = [TEXT, IMAGE_ID, IMAGE_ID, TEXT, AUDIO_ID, AUDIO_ID, AUDIO_ID]
    ids1 = [AUDIO_ID, AUDIO_ID, TEXT]
    ids2 = [TEXT, TEXT]
    img0, aud0, aud1 = _rand(2), _rand(3), _rand(2)
    reqs = [
        _req(ids0, {"image_embeds": img0, "audio_embeds": aud0}),
        _req(ids1, {"audio_embeds": aud1}),
        _req(ids2, None),
    ]
    fb, sb = _batches(reqs)

    out, ds, masks = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = img0
    expected[4:7] = aud0
    expected[7:9] = aud1
    torch.testing.assert_close(out, expected)
    assert ds is None and masks is None


def test_pad_values_replace_hashed_token_ids():
    # note (chenrui): the media-cache path substitutes hashed ids that sit far
    # beyond the vocab, so the embedding lookup has to clamp before indexing.
    runner = _runner()
    pad_img = VOCAB + 12345
    ids = [TEXT, pad_img, pad_img, TEXT]
    image_embeds = _rand(2)
    req = _req(ids, {"image_embeds": image_embeds, "pad_values": {"image": pad_img}})
    fb, sb = _batches([req])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = image_embeds
    torch.testing.assert_close(out, expected)


def test_modality_with_embeds_but_no_tokens_in_chunk_is_skipped():
    runner = _runner()
    ids = [TEXT, AUDIO_ID, TEXT]
    audio_embeds = _rand(1)
    req = _req(ids, {"image_embeds": _rand(2), "audio_embeds": audio_embeds})
    fb, sb = _batches([req])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:2] = audio_embeds
    torch.testing.assert_close(out, expected)


def test_chunked_prefill_advances_consumed_offsets():
    runner = _runner()
    prompt = [TEXT, IMAGE_ID, IMAGE_ID, IMAGE_ID, TEXT, IMAGE_ID, TEXT]
    image_embeds = _rand(4)
    inputs = {"image_embeds": image_embeds}

    req = _req(prompt, inputs, is_chunked=1)
    fb1, sb1 = _batches([req], chunk_ids=[prompt[:4]], prefix_lens=[0])
    out1, _, _ = runner._inject_multimodal_embeds(fb1, sb1)

    expected1 = _base_embeds(runner, fb1)
    expected1[1:4] = image_embeds[0:3]
    torch.testing.assert_close(out1, expected1)
    assert req._omni_consumed == {"image": 3}
    assert req.omni_model_inputs is inputs

    req.is_chunked = 0
    fb2, sb2 = _batches([req], chunk_ids=[prompt[4:]], prefix_lens=[4])
    out2, _, _ = runner._inject_multimodal_embeds(fb2, sb2)

    expected2 = _base_embeds(runner, fb2)
    expected2[1:2] = image_embeds[3:4]
    torch.testing.assert_close(out2, expected2)
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None


def test_image_deepstack_slice_and_mask():
    runner = _runner()
    ids = [TEXT, IMAGE_ID, IMAGE_ID, TEXT]
    image_embeds = _rand(2)
    ds_layers = [_rand(2), _rand(2)]
    req = _req(
        ids,
        {"image_embeds": image_embeds, "image_deepstack_visual_embeds": ds_layers},
    )
    fb, sb = _batches([req])

    out, ds, mask = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = image_embeds
    torch.testing.assert_close(out, expected)
    assert mask.tolist() == [False, True, True, False]
    assert len(ds) == 2
    for layer_out, layer_in in zip(ds, ds_layers):
        torch.testing.assert_close(layer_out, layer_in)


def test_merged_image_video_deepstack_interleave():
    # note (chenrui): image and video rows share one joint tensor ordered by
    # prompt position, where landing under the wrong modality stays silent.
    runner = _runner()
    ids = [VIDEO_ID, IMAGE_ID, TEXT, VIDEO_ID, IMAGE_ID]
    img_e, vid_e = _rand(2), _rand(2)
    img_ds = [_rand(2)]
    vid_ds = [_rand(2)]
    req = _req(
        ids,
        {
            "image_embeds": img_e,
            "video_embeds": vid_e,
            "image_deepstack_visual_embeds": img_ds,
            "video_deepstack_visual_embeds": vid_ds,
        },
    )
    fb, sb = _batches([req])

    out, ds, mask = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[0] = vid_e[0]
    expected[1] = img_e[0]
    expected[3] = vid_e[1]
    expected[4] = img_e[1]
    torch.testing.assert_close(out, expected)

    assert mask.tolist() == [True, True, False, True, True]
    joint = ds[0]
    torch.testing.assert_close(joint[0], vid_ds[0][0])
    torch.testing.assert_close(joint[1], img_ds[0][0])
    torch.testing.assert_close(joint[2], vid_ds[0][1])
    torch.testing.assert_close(joint[3], img_ds[0][1])


def test_precombined_deepstack_uses_visual_offset():
    runner = _runner()
    ids = [IMAGE_ID, VIDEO_ID, TEXT]
    img_e, vid_e = _rand(1), _rand(1)
    ds_layers = [_rand(2)]
    req = _req(
        ids,
        {
            "image_embeds": img_e,
            "video_embeds": vid_e,
            "deepstack_visual_embeds": ds_layers,
        },
    )
    fb, sb = _batches([req])

    out, ds, mask = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[0] = img_e[0]
    expected[1] = vid_e[0]
    torch.testing.assert_close(out, expected)
    assert mask.tolist() == [True, True, False]
    torch.testing.assert_close(ds[0], ds_layers[0])


def test_multi_request_deepstack_concat_and_combined_mask():
    runner = _runner()
    ids0 = [IMAGE_ID, TEXT]
    ids1 = [TEXT, VIDEO_ID]
    img_e, vid_e = _rand(1), _rand(1)
    img_ds = [_rand(1), _rand(1)]
    vid_ds = [_rand(1), _rand(1)]
    reqs = [
        _req(ids0, {"image_embeds": img_e, "image_deepstack_visual_embeds": img_ds}),
        _req(ids1, {"video_embeds": vid_e, "video_deepstack_visual_embeds": vid_ds}),
    ]
    fb, sb = _batches(reqs)

    out, ds, mask = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[0] = img_e[0]
    expected[3] = vid_e[0]
    torch.testing.assert_close(out, expected)

    assert mask.tolist() == [True, False, False, True]
    assert len(ds) == 2
    for layer_idx in range(2):
        torch.testing.assert_close(ds[layer_idx][0], img_ds[layer_idx][0])
        torch.testing.assert_close(ds[layer_idx][1], vid_ds[layer_idx][0])


def test_build_time_positions_take_precedence_over_prompt_scan():
    runner = _runner()
    ids = [TEXT, IMAGE_ID, IMAGE_ID, TEXT]
    image_embeds = _rand(2)
    req = _req(ids, {"image_embeds": image_embeds})
    req._omni_mm_positions = {
        "image": torch.tensor([1, 2]),
        "video": torch.empty(0, dtype=torch.long),
        "audio": torch.empty(0, dtype=torch.long),
    }
    # note (chenrui): an unreadable prompt makes a rescan fail loudly instead of
    # silently agreeing with the recorded positions.
    req.origin_input_ids = None
    fb, sb = _batches([req], chunk_ids=[ids])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = image_embeds
    torch.testing.assert_close(out, expected)


def test_prompt_scan_fallback_when_positions_missing():
    runner = _runner()
    ids = [TEXT, IMAGE_ID, TEXT, AUDIO_ID, AUDIO_ID]
    image_embeds, audio_embeds = _rand(1), _rand(2)
    req = _req(
        ids,
        {"image_embeds": image_embeds, "audio_embeds": audio_embeds},
        positions=False,
    )
    assert not hasattr(req, "_omni_mm_positions")
    fb, sb = _batches([req])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:2] = image_embeds
    expected[3:5] = audio_embeds
    torch.testing.assert_close(out, expected)


def test_prefix_lens_as_cpu_tensor():
    runner = _runner()
    prompt = [TEXT, IMAGE_ID, IMAGE_ID, TEXT]
    image_embeds = _rand(2)
    req = _req(
        prompt, {"image_embeds": image_embeds}, is_chunked=0, consumed={"image": 1}
    )
    fb, sb = _batches([req], chunk_ids=[prompt[2:]], prefix_lens=[2])
    fb.extend_prefix_lens_cpu = torch.tensor([2], dtype=torch.int64)

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[0:1] = image_embeds[1:2]
    torch.testing.assert_close(out, expected)


NO_SYNCS = {"item": 0, "any": 0, "nonzero": 0, "where": 0}


def _count_sync_ops(monkeypatch, runner, fb, sb):
    # note (chenrui): nonzero joins the obvious three because a device-resident
    # bool mask routed through it syncs while the other counts stay at zero.
    calls = {name: 0 for name in NO_SYNCS}
    originals = {
        "item": torch.Tensor.item,
        "any": torch.Tensor.any,
        "nonzero": torch.Tensor.nonzero,
        "where": torch.where,
    }

    def counting(name):
        original = originals[name]

        def wrapper(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in ("item", "any", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, counting(name))
    monkeypatch.setattr(torch, "where", counting("where"))
    runner._inject_multimodal_embeds(fb, sb)
    return calls


def _mixed_mm_batch():
    reqs = [
        _req(
            [TEXT, IMAGE_ID, IMAGE_ID, AUDIO_ID, TEXT],
            {"image_embeds": _rand(2), "audio_embeds": _rand(1)},
        )
        for _ in range(4)
    ]
    return _batches(reqs)


def _deepstack_mm_batch():
    reqs = [
        _req(
            [VIDEO_ID, IMAGE_ID, TEXT, IMAGE_ID, VIDEO_ID],
            {
                "image_embeds": _rand(2),
                "video_embeds": _rand(2),
                "image_deepstack_visual_embeds": [_rand(2), _rand(2)],
                "video_deepstack_visual_embeds": [_rand(2), _rand(2)],
            },
        )
        for _ in range(4)
    ]
    return _batches(reqs)


def test_no_host_syncs_on_hot_path(monkeypatch):
    runner = _runner()
    fb, sb = _mixed_mm_batch()
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == NO_SYNCS


def test_no_host_syncs_on_deepstack_path(monkeypatch):
    runner = _runner()
    fb, sb = _deepstack_mm_batch()
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == NO_SYNCS


def test_no_host_syncs_with_cpu_tensor_extend_lens(monkeypatch):
    runner = _runner()
    fb, sb = _mixed_mm_batch()
    fb.extend_seq_lens_cpu = torch.tensor(fb.extend_seq_lens_cpu, dtype=torch.int64)
    fb.extend_prefix_lens_cpu = torch.tensor(
        fb.extend_prefix_lens_cpu, dtype=torch.int64
    )
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == NO_SYNCS
