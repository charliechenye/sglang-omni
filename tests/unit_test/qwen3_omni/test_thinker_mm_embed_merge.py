# SPDX-License-Identifier: Apache-2.0
"""Golden parity tests for ThinkerModelRunner._inject_multimodal_embeds.

The merge is exercised on a bare instance (``object.__new__``, same pattern as
test_thinker_lookahead_eligible) with stand-in forward/schedule batches. Every
test builds its expected embeddings directly from the constructed inputs — an
oracle independent of the implementation — so these tests pin the current
behavior and must stay green across the batch-scatter rewrite.

test_no_host_syncs_on_hot_path additionally asserts the rewrite's contract:
no ``Tensor.item`` / ``Tensor.any`` / ``torch.where`` on the merge hot path.
"""
from __future__ import annotations

import types

import pytest
import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

VOCAB = 100
HIDDEN = 8
IMAGE_ID = 91
VIDEO_ID = 92
AUDIO_ID = 93
TEXT = 7  # arbitrary text token


def _runner() -> ThinkerModelRunner:
    torch.manual_seed(0)
    r = object.__new__(ThinkerModelRunner)
    r._embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
    r._image_token_id = IMAGE_ID
    r._video_token_id = VIDEO_ID
    r._audio_token_id = AUDIO_ID
    return r


def _req(input_ids, omni_model_inputs, *, is_chunked=0, consumed=None):
    return types.SimpleNamespace(
        origin_input_ids=list(input_ids),
        omni_model_inputs=omni_model_inputs,
        _omni_consumed=consumed,
        is_chunked=is_chunked,
    )


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


# ---------------------------------------------------------------------------
# Plain merges
# ---------------------------------------------------------------------------


def test_text_only_batch_returns_none():
    runner = _runner()
    req = _req([TEXT, TEXT, TEXT], None)
    fb, sb = _batches([req])
    assert runner._inject_multimodal_embeds(fb, sb) is None


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
    assert req.omni_model_inputs is None  # non-chunked request is cleared


def test_mixed_batch_image_audio_and_audio_only_and_text():
    # the regression spec from issue #1150: image+audio in one request,
    # audio-only in another, text-only in a third.
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
    # media-cache path: prompt carries hashed ids far beyond the vocab, matched
    # via pad_values; the embedding lookup must clamp them without crashing.
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
    # image embeds present but no image tokens in the ids: only audio merges.
    runner = _runner()
    ids = [TEXT, AUDIO_ID, TEXT]
    audio_embeds = _rand(1)
    req = _req(ids, {"image_embeds": _rand(2), "audio_embeds": audio_embeds})
    fb, sb = _batches([req])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:2] = audio_embeds
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------


def test_chunked_prefill_advances_consumed_offsets():
    runner = _runner()
    prompt = [TEXT, IMAGE_ID, IMAGE_ID, IMAGE_ID, TEXT, IMAGE_ID, TEXT]
    image_embeds = _rand(4)
    inputs = {"image_embeds": image_embeds}

    # chunk 1: first 4 tokens (contains image tokens 0..2)
    req = _req(prompt, inputs, is_chunked=1)
    fb1, sb1 = _batches([req], chunk_ids=[prompt[:4]], prefix_lens=[0])
    out1, _, _ = runner._inject_multimodal_embeds(fb1, sb1)

    expected1 = _base_embeds(runner, fb1)
    expected1[1:4] = image_embeds[0:3]
    torch.testing.assert_close(out1, expected1)
    assert req._omni_consumed == {"image": 3}
    assert req.omni_model_inputs is inputs  # kept while chunked

    # chunk 2: remaining 3 tokens (contains image token 3), final chunk
    req.is_chunked = 0
    fb2, sb2 = _batches([req], chunk_ids=[prompt[4:]], prefix_lens=[4])
    out2, _, _ = runner._inject_multimodal_embeds(fb2, sb2)

    expected2 = _base_embeds(runner, fb2)
    expected2[1:2] = image_embeds[3:4]
    torch.testing.assert_close(out2, expected2)
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None


# ---------------------------------------------------------------------------
# Deepstack
# ---------------------------------------------------------------------------


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
    # both image and video deepstack present: per layer, rows must land at
    # their own modality's positions inside the joint visual mask.
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
    # joint layer: visual positions in order 0,1,3,4 -> vid0, img0, vid1, img1
    joint = ds[0]
    torch.testing.assert_close(joint[0], vid_ds[0][0])
    torch.testing.assert_close(joint[1], img_ds[0][0])
    torch.testing.assert_close(joint[2], vid_ds[0][1])
    torch.testing.assert_close(joint[3], img_ds[0][1])


def test_precombined_deepstack_uses_visual_offset():
    # generic deepstack_visual_embeds branch: rows are sliced by the running
    # _visual consumed offset when both modalities are present.
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
    # when request build recorded placeholder positions, the runner must use
    # them as-is (origin_input_ids should not even be consulted).
    runner = _runner()
    ids = [TEXT, IMAGE_ID, IMAGE_ID, TEXT]
    image_embeds = _rand(2)
    req = _req(ids, {"image_embeds": image_embeds})
    req._omni_mm_positions = {
        "image": torch.tensor([1, 2]),
        "video": torch.empty(0, dtype=torch.long),
        "audio": torch.empty(0, dtype=torch.long),
    }
    req.origin_input_ids = None  # would crash the fallback scan
    fb, sb = _batches([req], chunk_ids=[ids])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = image_embeds
    torch.testing.assert_close(out, expected)


def test_prefix_lens_as_cpu_tensor():
    # ForwardBatch.extend_prefix_lens_cpu can be a CPU tensor rather than a
    # list on some sglang paths; chunk resolution must handle both.
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


# ---------------------------------------------------------------------------
# Host-sync contract (RED until the batch-scatter rewrite lands)
# ---------------------------------------------------------------------------


def _count_sync_ops(monkeypatch, runner, fb, sb):
    calls = {"item": 0, "any": 0, "where": 0}
    orig_item, orig_any, orig_where = torch.Tensor.item, torch.Tensor.any, torch.where

    def counting_item(self, *a, **k):
        calls["item"] += 1
        return orig_item(self, *a, **k)

    def counting_any(self, *a, **k):
        calls["any"] += 1
        return orig_any(self, *a, **k)

    def counting_where(*a, **k):
        calls["where"] += 1
        return orig_where(*a, **k)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    monkeypatch.setattr(torch.Tensor, "any", counting_any)
    monkeypatch.setattr(torch, "where", counting_where)
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


def test_no_host_syncs_on_hot_path(monkeypatch):
    # the rewrite's contract: placement comes from CPU-side metadata, so the
    # merge never calls Tensor.item / Tensor.any / torch.where — the ops that
    # force a device->host sync per request on the current implementation.
    runner = _runner()
    fb, sb = _mixed_mm_batch()
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == {
        "item": 0,
        "any": 0,
        "where": 0,
    }


def test_no_host_syncs_with_cpu_tensor_extend_lens(monkeypatch):
    # ForwardBatch.extend_seq_lens_cpu / extend_prefix_lens_cpu can be CPU
    # tensors (not lists) on some sglang paths; int(tensor[i]) would call
    # Tensor.item() per request. The merge must normalize them up front so the
    # no-sync contract holds for the production input type too.
    runner = _runner()
    fb, sb = _mixed_mm_batch()
    fb.extend_seq_lens_cpu = torch.tensor(fb.extend_seq_lens_cpu, dtype=torch.int64)
    fb.extend_prefix_lens_cpu = torch.tensor(
        fb.extend_prefix_lens_cpu, dtype=torch.int64
    )
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == {
        "item": 0,
        "any": 0,
        "where": 0,
    }
