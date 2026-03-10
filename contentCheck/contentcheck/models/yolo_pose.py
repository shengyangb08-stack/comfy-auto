from __future__ import annotations

import math

import numpy as np

from contentcheck.models.base import BaseChecker
from contentcheck.results import ModelResult

# COCO keypoint indices used by YOLO-Pose (17 keypoints).
_KP = {
    "nose": 0,
    "l_eye": 1,  "r_eye": 2,
    "l_ear": 3,  "r_ear": 4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow": 7,     "r_elbow": 8,
    "l_wrist": 9,     "r_wrist": 10,
    "l_hip": 11,      "r_hip": 12,
    "l_knee": 13,     "r_knee": 14,
    "l_ankle": 15,    "r_ankle": 16,
}

_CONF_THRESHOLD = 0.3
_MIN_LIMB_SYMMETRY = 0.35
_MAX_LIMB_SYMMETRY = 2.8
_MIN_JOINT_ANGLE = 15.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


class YoloPoseChecker(BaseChecker):
    name = "yolo"

    def __init__(self, **_kwargs: object) -> None:
        from ultralytics import YOLO
        self._model = YOLO("yolo11n-pose.pt")

    def check(self, frame: np.ndarray) -> ModelResult:
        results = self._model(frame, verbose=False)
        anomalies: list[str] = []
        sub_scores: list[float] = []

        for det_idx, result in enumerate(results):
            if result.keypoints is None:
                continue
            kps_batch = result.keypoints.data.cpu().numpy()
            for person_idx, kps in enumerate(kps_batch):
                if kps.shape[0] < 17:
                    continue
                coords = kps[:, :2]
                confs = kps[:, 2] if kps.shape[1] >= 3 else np.ones(17)
                tag = f"Person {person_idx}"
                self._check_symmetry(coords, confs, tag, anomalies, sub_scores)
                self._check_angles(coords, confs, tag, anomalies, sub_scores)
                self._check_proportions(coords, confs, tag, anomalies, sub_scores)

        score = max(sub_scores) if sub_scores else 0.0
        details = "; ".join(anomalies) if anomalies else "No anomalies detected"
        return ModelResult(
            model_name=self.name, score=score, anomalies=anomalies, details=details
        )

    def _check_symmetry(
        self, kps: np.ndarray, confs: np.ndarray, tag: str,
        anomalies: list[str], scores: list[float],
    ) -> None:
        pairs = [
            ("l_shoulder", "l_elbow", "r_shoulder", "r_elbow", "upper arm"),
            ("l_elbow", "l_wrist", "r_elbow", "r_wrist", "forearm"),
            ("l_hip", "l_knee", "r_hip", "r_knee", "thigh"),
            ("l_knee", "l_ankle", "r_knee", "r_ankle", "shin"),
        ]
        for la, lb, ra, rb, label in pairs:
            idxs = [_KP[la], _KP[lb], _KP[ra], _KP[rb]]
            if min(confs[i] for i in idxs) < _CONF_THRESHOLD:
                continue
            left = _dist(kps[idxs[0]], kps[idxs[1]])
            right = _dist(kps[idxs[2]], kps[idxs[3]])
            if left < 1e-4 or right < 1e-4:
                continue
            ratio = left / right
            if ratio < _MIN_LIMB_SYMMETRY or ratio > _MAX_LIMB_SYMMETRY:
                anomalies.append(f"{tag}: {label} L/R ratio {ratio:.2f}")
                scores.append(min(1.0, abs(ratio - 1.0) * 0.5))

    def _check_angles(
        self, kps: np.ndarray, confs: np.ndarray, tag: str,
        anomalies: list[str], scores: list[float],
    ) -> None:
        joints = [
            ("l_shoulder", "l_elbow", "l_wrist", "left elbow"),
            ("r_shoulder", "r_elbow", "r_wrist", "right elbow"),
            ("l_hip", "l_knee", "l_ankle", "left knee"),
            ("r_hip", "r_knee", "r_ankle", "right knee"),
        ]
        for a, b, c, label in joints:
            ia, ib, ic = _KP[a], _KP[b], _KP[c]
            if min(confs[ia], confs[ib], confs[ic]) < _CONF_THRESHOLD:
                continue
            ang = _angle(kps[ia], kps[ib], kps[ic])
            if ang < _MIN_JOINT_ANGLE:
                anomalies.append(f"{tag}: {label} extreme angle {ang:.1f}°")
                scores.append(0.6)

    def _check_proportions(
        self, kps: np.ndarray, confs: np.ndarray, tag: str,
        anomalies: list[str], scores: list[float],
    ) -> None:
        """Check that upper-body to lower-body ratio is plausible."""
        torso_pts = [_KP["l_shoulder"], _KP["r_shoulder"], _KP["l_hip"], _KP["r_hip"]]
        if min(confs[i] for i in torso_pts) < _CONF_THRESHOLD:
            return
        shoulder_mid = (kps[_KP["l_shoulder"]] + kps[_KP["r_shoulder"]]) / 2
        hip_mid = (kps[_KP["l_hip"]] + kps[_KP["r_hip"]]) / 2
        torso = _dist(shoulder_mid, hip_mid)
        if torso < 1e-4:
            return

        shoulder_w = _dist(kps[_KP["l_shoulder"]], kps[_KP["r_shoulder"]])
        ratio = shoulder_w / torso
        if ratio < 0.15 or ratio > 3.0:
            anomalies.append(f"{tag}: shoulder-width/torso ratio {ratio:.2f}")
            scores.append(min(1.0, abs(ratio - 1.0) * 0.4))
