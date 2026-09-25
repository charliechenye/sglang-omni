# SPDX-License-Identifier: Apache-2.0
"""Nominal scheduler lifecycle contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod


class StartupFinalizable(ABC):
    """Scheduler contract for synchronous work before process startup."""

    @abstractmethod
    def finalize_startup(self) -> None:
        """Finish startup-only work before scheduler threads start."""
        raise NotImplementedError
