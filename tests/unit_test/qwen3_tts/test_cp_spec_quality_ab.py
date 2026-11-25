# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for multi-prompt C1 quality corpus and WER/CER scorer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.qwen3_tts_cp_spec_k2.quality_ab import (
    DEFAULT_CORPUS,
    edit_distance,
    load_corpus,
    normalize_tokens,
    score_transcript,
)


def test_corpus_has_distinct_en_zh_seeded_inputs() -> None:
    rows = load_corpus(None)
    assert len(rows) == 4
    assert {row["language"] for row in rows} == {"en", "zh"}
    assert len({r["seed"] for r in rows}) == 4
    assert {r["id"] for r in rows} == {r["id"] for r in DEFAULT_CORPUS}


def test_manifest_rejects_duplicate_ids_and_invalid_languages(tmp_path: Path) -> None:
    p = tmp_path / "prompts.jsonl"
    first = DEFAULT_CORPUS[0]
    p.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(p)
    p.write_text(json.dumps({**first, "language": "jp"}) + "\n")
    with pytest.raises(ValueError, match="only en / zh"):
        load_corpus(p)
    p.write_text(json.dumps({**first, "id": "../hello"}) + "\n")
    with pytest.raises(ValueError, match="case id"):
        load_corpus(p)


def test_english_wer_tokenization_and_edit_distance() -> None:
    assert normalize_tokens("It's a TEST, with 3 samples!", "en") == [
        "it's", "a", "test", "with", "3", "samples"
    ]
    assert edit_distance(["one", "two", "three"], ["one", "four", "three"]) == 1
    result = score_transcript("We can test this today.", "We can test tomorrow.", "en")
    assert result["metric"] == "WER"
    assert result["errors"] == 2
    assert result["reference_units"] == 5
    assert result["error_rate"] == pytest.approx(.4)


def test_chinese_cer_removes_punctuation_preserves_characters() -> None:
    assert normalize_tokens("今天，你好！", "zh") == ["今", "天", "你", "好"]
    score = score_transcript("今天，你好！", "今天好", "zh")
    assert score["metric"] == "CER"
    assert score["errors"] == 1
    assert score["reference_units"] == 4
    assert score["error_rate"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="empty"):
        score_transcript("...", "今天", "zh")
