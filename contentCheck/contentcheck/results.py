from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class ModelResult:
    model_name: str
    score: float  # 0.0 = normal, 1.0 = definitely abnormal
    anomalies: list[str]
    details: str


@dataclasses.dataclass
class FrameReport:
    frame_index: int
    timestamp_sec: float
    frame: np.ndarray
    model_results: list[ModelResult]

    @property
    def max_score(self) -> float:
        if not self.model_results:
            return 0.0
        return max(r.score for r in self.model_results)

    @property
    def is_flagged(self) -> bool:
        return self.max_score > 0.0

    def scores_by_model(self) -> dict[str, float]:
        return {r.model_name: r.score for r in self.model_results}


@dataclasses.dataclass
class VideoReport:
    video_path: str
    total_frames_extracted: int
    frame_reports: list[FrameReport]
    threshold: float

    @property
    def flagged_reports(self) -> list[FrameReport]:
        return [r for r in self.frame_reports if r.max_score >= self.threshold]
