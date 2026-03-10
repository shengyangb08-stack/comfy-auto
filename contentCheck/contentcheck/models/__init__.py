from __future__ import annotations

from contentcheck.models.base import BaseChecker
from contentcheck.models.mediapipe_checker import MediaPipeChecker
from contentcheck.models.yolo_pose import YoloPoseChecker
from contentcheck.models.llm_checker import GeminiChecker, GrokChecker

MODEL_REGISTRY: dict[str, type[BaseChecker]] = {
    "mediapipe": MediaPipeChecker,
    "yolo": YoloPoseChecker,
    "llm-gemini": GeminiChecker,
    "llm-grok": GrokChecker,
}

__all__ = [
    "MODEL_REGISTRY",
    "BaseChecker",
    "MediaPipeChecker",
    "YoloPoseChecker",
    "GeminiChecker",
    "GrokChecker",
]
