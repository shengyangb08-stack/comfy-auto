from __future__ import annotations

import abc

import numpy as np

from contentcheck.results import ModelResult


class BaseChecker(abc.ABC):
    """Base class for all anatomy-anomaly checkers."""

    name: str = "base"

    @abc.abstractmethod
    def check(self, frame: np.ndarray) -> ModelResult:
        """Analyse *frame* (BGR uint8) and return a `ModelResult`."""

    def cleanup(self) -> None:
        """Release resources. Override if needed."""
