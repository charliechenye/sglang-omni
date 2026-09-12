# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys

from sglang_omni.models.fun_cosyvoice3 import stages


def test_modelscope_guard_restores_changed_root_handler_levels(
    monkeypatch, tmp_path
) -> None:
    root = logging.getLogger()
    demoted = logging.StreamHandler()
    demoted.setLevel(logging.INFO)
    untouched = logging.FileHandler(tmp_path / "log.txt")
    untouched.setLevel(logging.WARNING)
    root.addHandler(demoted)
    root.addHandler(untouched)
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)

    def _import_like_modelscope(name: str):
        assert name == "modelscope"
        demoted.setLevel(logging.ERROR)
        raise ImportError("side effects happen before the failure too")

    monkeypatch.setattr(stages.importlib, "import_module", _import_like_modelscope)
    try:
        stages.import_modelscope_preserving_root_handlers()
        assert demoted.level == logging.INFO
        assert untouched.level == logging.WARNING
    finally:
        root.removeHandler(demoted)
        root.removeHandler(untouched)
        untouched.close()
