# SPDX-License-Identifier: Apache-2.0
"""Golden parity tests for ThinkerModelRunner._inject_multimodal_embeds.

The merge is exercised on a bare instance (``object.__new__``, same pattern as
test_thinker_lookahead_eligible) with stand-in forward/schedule batches. Every
test builds its expected embeddings directly from the constructed inputs — an
oracle independent of the implementation — so they pin the merge's behavior
regardless of how the scatter is built.

The tests under "Host-sync contract" additionally assert that the merge calls
none of ``Tensor.item`` / ``Tensor.any`` / ``Tensor.nonzero`` / ``torch.where``.
"""
from __future__ import annotations

import types

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


def _req(input_ids, omni_model_inputs, *, is_chunked=0, consumed=None, positions=True):
    req = types.SimpleNamespace(
        origin_input_ids=list(input_ids),
        omni_model_inputs=omni_model_inputs,
        _omni_consumed=consumed,
        is_chunked=is_chunked,
    )
    if positions:
        # note (chenrui): build_sglang_thinker_request records these for every
        # request, so a fixture that omits them pins a fallback the served path
        # never reaches. Pass positions=False to reach it deliberately.
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
    assert req.omni_model_inputs is None


def test_mixed_batch_image_audio_and_audio_only_and_text():
    # note (chenrui): the regression spec from issue #1150 -- image+audio in one
    # request, audio-only in another, text-only in a third.
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
    # note (chenrui): on the media-cache path the prompt carries hashed ids far
    # beyond the vocab, so the embedding lookup must clamp instead of crashing.
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
    # note (chenrui): a modality can carry embeds while none of its placeholders
    # fall in this chunk, and consuming from it there would corrupt the offsets.
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
    # note (chenrui): embeds are consumed across forward passes, so each chunk
    # must resume where the previous one stopped and only the last may clear.
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
    # note (chenrui): image and video deepstack rows share one joint tensor
    # ordered by prompt position, so a row landing under the wrong modality is
    # silently wrong rather than a shape error.
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
    # note (chenrui): visual positions 0,1,3,4 -> vid0, img0, vid1, img1.
    joint = ds[0]
    torch.testing.assert_close(joint[0], vid_ds[0][0])
    torch.testing.assert_close(joint[1], img_ds[0][0])
    torch.testing.assert_close(joint[2], vid_ds[0][1])
    torch.testing.assert_close(joint[3], img_ds[0][1])


def test_precombined_deepstack_uses_visual_offset():
    # note (chenrui): when the encoder already combined the two modalities there
    # is nothing to interleave, and the rows are sliced by a running visual
    # offset that is separate from the per-modality ones.
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


# ---------------------------------------------------------------------------
# Placeholder placement metadata
# ---------------------------------------------------------------------------


def test_build_time_positions_take_precedence_over_prompt_scan():
    # note (chenrui): recorded positions are authoritative -- the prompt here is
    # unreadable, so any rescan of it fails loudly instead of silently agreeing.
    runner = _runner()
    ids = [TEXT, IMAGE_ID, IMAGE_ID, TEXT]
    image_embeds = _rand(2)
    req = _req(ids, {"image_embeds": image_embeds})
    req._omni_mm_positions = {
        "image": torch.tensor([1, 2]),
        "video": torch.empty(0, dtype=torch.long),
        "audio": torch.empty(0, dtype=torch.long),
    }
    req.origin_input_ids = None
    fb, sb = _batches([req], chunk_ids=[ids])

    out, _, _ = runner._inject_multimodal_embeds(fb, sb)

    expected = _base_embeds(runner, fb)
    expected[1:3] = image_embeds
    torch.testing.assert_close(out, expected)


def test_prompt_scan_fallback_when_positions_missing():
    # note (chenrui): requests that predate the build-time recording still have
    # to merge, so the scan of origin_input_ids must reach the same placement.
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
    # note (chenrui): ForwardBatch.extend_prefix_lens_cpu is a CPU tensor rather
    # than a list on some sglang paths, and chunk resolution must handle both.
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
# Host-sync contract: the merge must never read from device back to host
# ---------------------------------------------------------------------------

NO_SYNCS = {"item": 0, "any": 0, "nonzero": 0, "where": 0}


def _count_sync_ops(monkeypatch, runner, fb, sb):
    # note (chenrui): nonzero is counted alongside the obvious three because a
    # device-resident bool mask reintroduced through it would sync while item,
    # any and where all still read zero.
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
    # note (chenrui): the interleaved image+video branch rebuilds placement from
    # a sort permutation, which the plain merge never exercises.
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
    # note (chenrui): placement comes from CPU-side metadata, so none of the ops
    # that force a device->host sync should appear.
    runner = _runner()
    fb, sb = _mixed_mm_batch()
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == NO_SYNCS


def test_no_host_syncs_on_deepstack_path(monkeypatch):
    runner = _runner()
    fb, sb = _deepstack_mm_batch()
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == NO_SYNCS


def test_no_host_syncs_with_cpu_tensor_extend_lens(monkeypatch):
    # note (chenrui): these arrive as CPU tensors on some sglang paths, where
    # int(tensor[i]) would put a .item() back per request.
    runner = _runner()
    fb, sb = _mixed_mm_batch()
    fb.extend_seq_lens_cpu = torch.tensor(fb.extend_seq_lens_cpu, dtype=torch.int64)
    fb.extend_prefix_lens_cpu = torch.tensor(
        fb.extend_prefix_lens_cpu, dtype=torch.int64
    )
    assert _count_sync_ops(monkeypatch, runner, fb, sb) == NO_SYNCS
