# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the batch-carried prefill inputs payload."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)

_REAL_SHAPE = object()


def _forward_batch(
    *,
    num_tokens: int = 4,
    batch_size: int = 2,
    input_embeds=None,
    mm_inputs=_REAL_SHAPE,
) -> SimpleNamespace:
    if mm_inputs is _REAL_SHAPE:
        mm_inputs = [None] * batch_size
    return SimpleNamespace(
        input_embeds=input_embeds,
        mm_inputs=mm_inputs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        batch_size=batch_size,
    )


def _payload(*, rows: int = 4, rids: tuple[str, ...] = ("a", "b")) -> OmniPrefillInputs:
    return OmniPrefillInputs(
        input_embeds=torch.zeros(rows, 8),
        rids=rids,
    )


def test_attach_replaces_prepare_for_extend_placeholder() -> None:
    forward_batch = _forward_batch()
    payload = _payload()

    attach_omni_prefill_inputs(forward_batch, payload)

    assert forward_batch.input_embeds is None
    assert get_omni_prefill_inputs(forward_batch) is payload


def test_attach_accepts_batches_without_mm_inputs() -> None:
    forward_batch = _forward_batch(mm_inputs=None)
    payload = _payload()

    attach_omni_prefill_inputs(forward_batch, payload)

    assert get_omni_prefill_inputs(forward_batch) is payload


def test_attach_rejects_batch_carried_embeds() -> None:
    forward_batch = _forward_batch(input_embeds=torch.zeros(4, 8))

    with pytest.raises(RuntimeError, match="input_embeds to stay"):
        attach_omni_prefill_inputs(forward_batch, _payload())


def test_attach_rejects_double_attach() -> None:
    forward_batch = _forward_batch()
    attach_omni_prefill_inputs(forward_batch, _payload())

    with pytest.raises(RuntimeError, match="refusing to attach twice"):
        attach_omni_prefill_inputs(forward_batch, _payload())


def test_attach_rejects_real_multimodal_inputs() -> None:
    for mm_inputs in ([object()], [None, object()]):
        forward_batch = _forward_batch(mm_inputs=mm_inputs)

        with pytest.raises(RuntimeError, match="SGLang multimodal inputs"):
            attach_omni_prefill_inputs(forward_batch, _payload())


def test_attach_rejects_token_row_mismatch() -> None:
    forward_batch = _forward_batch(num_tokens=5)

    with pytest.raises(RuntimeError, match="extend-window tokens"):
        attach_omni_prefill_inputs(forward_batch, _payload(rows=4))


def test_attach_rejects_rid_count_mismatch() -> None:
    forward_batch = _forward_batch(batch_size=3)

    with pytest.raises(RuntimeError, match="rids must cover the batch"):
        attach_omni_prefill_inputs(forward_batch, _payload(rids=("only-one",)))


def test_get_ignores_foreign_mm_inputs() -> None:
    assert get_omni_prefill_inputs(_forward_batch()) is None
    assert get_omni_prefill_inputs(_forward_batch(mm_inputs=[object()])) is None
    assert get_omni_prefill_inputs(_forward_batch(mm_inputs=None)) is None
