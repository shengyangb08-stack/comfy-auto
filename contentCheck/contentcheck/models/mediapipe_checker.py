from __future__ import annotations

import logging
import math
import os

import cv2
import mediapipe as mp
import numpy as np

from contentcheck.models.base import BaseChecker
from contentcheck.results import ModelResult

log = logging.getLogger("contentcheck.mediapipe")

_FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

_POSE = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13,    "r_elbow": 14,
    "l_wrist": 15,    "r_wrist": 16,
    "l_hip": 23,      "r_hip": 24,
    "l_knee": 25,     "r_knee": 26,
    "l_ankle": 27,    "r_ankle": 28,
}

_MIN_SEGMENT_RATIO = 0.3
_MAX_SEGMENT_RATIO = 3.5
_MIN_JOINT_ANGLE = 15.0
_MIN_LIMB_SYMMETRY = 0.35
_MAX_LIMB_SYMMETRY = 2.8

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def _lm_to_px(landmark, h: int, w: int) -> np.ndarray:
    return np.array([landmark.x * w, landmark.y * h])


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex *b* formed by segments b->a and b->c."""
    v1 = a - b
    v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def _seg_len(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class MediaPipeChecker(BaseChecker):
    name = "mediapipe"

    def __init__(self, **_kwargs: object) -> None:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        hand_model = os.path.join(_MODELS_DIR, "hand_landmarker.task")
        pose_model = os.path.join(_MODELS_DIR, "pose_landmarker.task")

        if not os.path.isfile(hand_model):
            raise FileNotFoundError(
                f"Hand model not found at {hand_model}. "
                "Download it: curl -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        if not os.path.isfile(pose_model):
            raise FileNotFoundError(
                f"Pose model not found at {pose_model}. "
                "Download it: curl -o models/pose_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )

        log.debug("Loading hand landmarker from %s", hand_model)
        self._hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=4,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )
        )
        log.debug("Loading pose landmarker from %s", pose_model)
        self._pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
            )
        )
        log.debug("MediaPipe models loaded")

    def check(self, frame: np.ndarray) -> ModelResult:
        h, w = frame.shape[:2]
        log.info("Checking frame %dx%d", w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        anomalies: list[str] = []
        sub_scores: list[float] = []

        self._check_hands(mp_image, frame, h, w, anomalies, sub_scores)
        self._check_pose(mp_image, h, w, anomalies, sub_scores)

        score = max(sub_scores) if sub_scores else 0.0
        details = "; ".join(anomalies) if anomalies else "No anomalies detected"
        log.info("Final score: %.2f  anomalies: %d", score, len(anomalies))
        return ModelResult(
            model_name=self.name, score=score, anomalies=anomalies, details=details
        )

    # ---- hand analysis --------------------------------------------------

    def _check_hands(
        self, mp_image: mp.Image, frame_bgr: np.ndarray, h: int, w: int,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("Running hand landmarker ...")
        result = self._hand_landmarker.detect(mp_image)

        n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
        log.info("Hand detection: %d hand(s) found", n_hands)
        if not result.hand_landmarks:
            return

        if n_hands > 2:
            msg = f"Detected {n_hands} hands (expected <=2)"
            log.warning("ANOMALY hand count: %s", msg)
            anomalies.append(msg)
            sub_scores.append(0.8)

        for hand_idx, hand_lms in enumerate(result.hand_landmarks):
            handedness = ""
            if result.handedness and hand_idx < len(result.handedness):
                cats = result.handedness[hand_idx]
                if cats:
                    handedness = f" ({cats[0].category_name}, conf={cats[0].score:.2f})"
            log.info("  Hand %d%s — analysing 21 landmarks", hand_idx, handedness)
            pts = [_lm_to_px(lm, h, w) for lm in hand_lms]
            self._check_finger_proportions(pts, hand_idx, anomalies, sub_scores)
            self._check_finger_angles(pts, hand_idx, anomalies, sub_scores)
            self._check_finger_order(pts, hand_idx, anomalies, sub_scores)
            self._check_finger_count(frame_bgr, pts, hand_idx, anomalies, sub_scores)

    def _check_finger_proportions(
        self, pts: list[np.ndarray], hand_idx: int,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("  Hand %d: checking finger segment proportions", hand_idx)
        for fname, idxs in _FINGER_LANDMARKS.items():
            if fname == "thumb":
                continue
            segs = [_seg_len(pts[idxs[i]], pts[idxs[i + 1]]) for i in range(3)]
            if segs[0] < 1e-4:
                log.debug("    %s: proximal segment ~0 px, skipping", fname)
                continue
            r1 = segs[1] / (segs[0] + 1e-9)
            r2 = segs[2] / (segs[1] + 1e-9)
            log.debug(
                "    %s segments: prox=%.1f mid=%.1f dist=%.1f px  "
                "ratios mid/prox=%.2f dist/mid=%.2f",
                fname, segs[0], segs[1], segs[2], r1, r2,
            )
            for ratio, label in [(r1, "mid/prox"), (r2, "dist/mid")]:
                if ratio < _MIN_SEGMENT_RATIO or ratio > _MAX_SEGMENT_RATIO:
                    s = min(1.0, abs(ratio - 1.0) * 0.5)
                    msg = f"Hand {hand_idx} {fname}: unusual {label} ratio {ratio:.2f}"
                    log.warning("    ANOMALY proportion: %s (score %.2f)", msg, s)
                    anomalies.append(msg)
                    sub_scores.append(s)
                else:
                    log.debug("    %s %s ratio %.2f — OK", fname, label, ratio)

    def _check_finger_angles(
        self, pts: list[np.ndarray], hand_idx: int,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("  Hand %d: checking finger joint angles", hand_idx)
        for fname, idxs in _FINGER_LANDMARKS.items():
            if fname == "thumb":
                continue
            for j in range(1, 3):
                angle = _angle_at(pts[idxs[j - 1]], pts[idxs[j]], pts[idxs[j + 1]])
                joint_name = "PIP" if j == 1 else "DIP"
                log.debug("    %s %s (joint %d): %.1f°", fname, joint_name, j, angle)
                if angle < _MIN_JOINT_ANGLE:
                    msg = f"Hand {hand_idx} {fname} joint {j}: extreme angle {angle:.1f}"
                    log.warning("    ANOMALY angle: %s (score 0.60)", msg)
                    anomalies.append(msg)
                    sub_scores.append(0.6)

    def _check_finger_order(
        self, pts: list[np.ndarray], hand_idx: int,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("  Hand %d: checking MCP joint ordering", hand_idx)
        mcp_idxs = [5, 9, 13, 17]
        mcps = [pts[i] for i in mcp_idxs]
        wrist = pts[0]
        angles = [math.atan2(m[1] - wrist[1], m[0] - wrist[0]) for m in mcps]
        names = ["index", "middle", "ring", "pinky"]
        for name, a in zip(names, angles):
            log.debug("    %s MCP angle from wrist: %.1f°", name, math.degrees(a))

        inversions = sum(
            1 for i in range(len(angles) - 1) if abs(angles[i + 1] - angles[i]) > math.pi * 0.8
        )
        log.debug("    Angular inversions: %d", inversions)
        if inversions >= 2:
            msg = f"Hand {hand_idx}: MCP joint order scrambled"
            log.warning("    ANOMALY order: %s (score 0.70)", msg)
            anomalies.append(msg)
            sub_scores.append(0.7)

    # ---- contour-based finger counting -----------------------------------

    def _segment_hand(
        self, frame_bgr: np.ndarray, pts: list[np.ndarray],
        x1: int, y1: int, x2: int, y2: int,
    ) -> np.ndarray | None:
        """Segment the hand from the background using Otsu thresholding.

        Uses landmark positions to determine which side of the threshold
        is the hand (foreground).
        """
        hand_crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Landmarks should fall in the foreground (white). Check and invert if needed.
        test_idxs = [0, 8, 12, 16]
        in_white = 0
        for idx in test_idxs:
            py = int(round(pts[idx][1])) - y1
            px = int(round(pts[idx][0])) - x1
            if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                if mask[py, px] > 0:
                    in_white += 1
        if in_white < len(test_idxs) // 2:
            mask = 255 - mask
            log.debug("    Otsu: inverted mask (landmarks were in dark region)")

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
        return mask

    def _check_finger_count(
        self, frame_bgr: np.ndarray, pts: list[np.ndarray], hand_idx: int,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        """Count fingers via convexity-defect analysis on the hand contour."""
        log.debug("  Hand %d: contour-based finger counting", hand_idx)
        h, w = frame_bgr.shape[:2]

        pts_arr = np.array(pts, dtype=np.float32)
        xs, ys = pts_arr[:, 0], pts_arr[:, 1]
        bbox_w = float(xs.max() - xs.min())
        bbox_h = float(ys.max() - ys.min())
        pad = int(max(bbox_w, bbox_h) * 0.35)

        x1 = max(0, int(xs.min() - pad))
        y1 = max(0, int(ys.min() - pad))
        x2 = min(w, int(xs.max() + pad))
        y2 = min(h, int(ys.max() + pad))

        if x2 - x1 < 20 or y2 - y1 < 20:
            log.debug("    Hand region too small (%dx%d), skipping", x2 - x1, y2 - y1)
            return

        mask = self._segment_hand(frame_bgr, pts, x1, y1, x2, y2)
        if mask is None:
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            log.debug("    No contours found after segmentation")
            return

        # Pick the contour that contains the most landmark points
        best_contour = None
        best_hits = -1
        for cnt in contours:
            hits = sum(
                1 for p in pts
                if cv2.pointPolygonTest(cnt, (float(p[0]) - x1, float(p[1]) - y1), False) >= 0
            )
            if hits > best_hits:
                best_hits = hits
                best_contour = cnt

        contour = best_contour
        contour_area = cv2.contourArea(contour)
        crop_area = (x2 - x1) * (y2 - y1)
        log.debug("    Best contour: %.0f px² (%.0f%% of crop, contains %d/21 landmarks)",
                   contour_area, 100 * contour_area / crop_area, best_hits)

        if len(contour) < 10:
            log.debug("    Contour too few points (%d), skipping", len(contour))
            return

        hull_idx = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_idx)
        if defects is None:
            log.debug("    No convexity defects found")
            return

        hand_diag = math.sqrt(bbox_w ** 2 + bbox_h ** 2)
        min_depth = hand_diag * 0.06
        max_angle = 110.0
        significant = 0
        log.debug("    Convexity defects (min_depth=%.1f, max_angle=%.0f°):", min_depth, max_angle)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            if depth < min_depth:
                continue
            start = contour[s][0].astype(float)
            end = contour[e][0].astype(float)
            far = contour[f][0].astype(float)
            ba = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            bb = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            bc = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            cos_a = (bb ** 2 + bc ** 2 - ba ** 2) / (2 * bb * bc + 1e-9)
            angle_deg = math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))
            counted = angle_deg < max_angle
            log.debug("      defect: depth=%.1f  angle=%.1f°  %s",
                       depth, angle_deg, "COUNTED" if counted else "skipped")
            if counted:
                significant += 1

        finger_count = significant + 1
        log.info("  Hand %d: contour finger count = %d  (%d defects)", hand_idx, finger_count, significant)

        if finger_count > 5:
            msg = f"Hand {hand_idx}: contour analysis detected {finger_count} fingers (expected 5)"
            log.warning("    ANOMALY finger count: %s (score 0.85)", msg)
            anomalies.append(msg)
            sub_scores.append(0.85)
        elif finger_count < 4 and finger_count > 0:
            log.debug("    Low finger count (%d) — may be a fist or partial view, not flagging", finger_count)

    # ---- pose analysis --------------------------------------------------

    def _check_pose(
        self, mp_image: mp.Image, h: int, w: int,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("Running pose landmarker ...")
        result = self._pose_landmarker.detect(mp_image)

        n_poses = len(result.pose_landmarks) if result.pose_landmarks else 0
        log.info("Pose detection: %d person(s) found", n_poses)
        if not result.pose_landmarks:
            return

        for person_idx, pose_lms in enumerate(result.pose_landmarks):
            log.info("  Person %d — analysing pose landmarks", person_idx)
            pts = {name: _lm_to_px(pose_lms[idx], h, w) for name, idx in _POSE.items()}
            vis = {name: pose_lms[idx].visibility for name, idx in _POSE.items()}
            log.debug("    Landmark visibility: %s",
                       {k: f"{v:.2f}" for k, v in vis.items()})
            self._check_limb_symmetry(pts, vis, anomalies, sub_scores)
            self._check_joint_angles_pose(pts, vis, anomalies, sub_scores)

    def _check_limb_symmetry(
        self, pts: dict, vis: dict,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("    Checking limb L/R symmetry")
        pairs = [
            ("l_shoulder", "l_elbow", "r_shoulder", "r_elbow", "upper arm"),
            ("l_elbow", "l_wrist", "r_elbow", "r_wrist", "forearm"),
            ("l_hip", "l_knee", "r_hip", "r_knee", "thigh"),
            ("l_knee", "l_ankle", "r_knee", "r_ankle", "shin"),
        ]
        for la, lb, ra, rb, label in pairs:
            min_vis = min(vis[la], vis[lb], vis[ra], vis[rb])
            if min_vis < 0.5:
                log.debug("      %s: skipped (min visibility %.2f < 0.5)", label, min_vis)
                continue
            left_len = _seg_len(pts[la], pts[lb])
            right_len = _seg_len(pts[ra], pts[rb])
            if left_len < 1e-4 or right_len < 1e-4:
                log.debug("      %s: skipped (near-zero length L=%.1f R=%.1f)", label, left_len, right_len)
                continue
            ratio = left_len / right_len
            log.debug("      %s: L=%.1f px  R=%.1f px  ratio=%.2f", label, left_len, right_len, ratio)
            if ratio < _MIN_LIMB_SYMMETRY or ratio > _MAX_LIMB_SYMMETRY:
                s = min(1.0, abs(ratio - 1.0) * 0.5)
                msg = f"Pose: {label} L/R ratio {ratio:.2f}"
                log.warning("      ANOMALY symmetry: %s (score %.2f)", msg, s)
                anomalies.append(msg)
                sub_scores.append(s)
            else:
                log.debug("      %s ratio %.2f — OK", label, ratio)

    def _check_joint_angles_pose(
        self, pts: dict, vis: dict,
        anomalies: list[str], sub_scores: list[float],
    ) -> None:
        log.debug("    Checking pose joint angles")
        joints = [
            ("l_shoulder", "l_elbow", "l_wrist", "left elbow"),
            ("r_shoulder", "r_elbow", "r_wrist", "right elbow"),
            ("l_hip", "l_knee", "l_ankle", "left knee"),
            ("r_hip", "r_knee", "r_ankle", "right knee"),
        ]
        for a, b, c, label in joints:
            min_vis = min(vis[a], vis[b], vis[c])
            if min_vis < 0.5:
                log.debug("      %s: skipped (min visibility %.2f < 0.5)", label, min_vis)
                continue
            angle = _angle_at(pts[a], pts[b], pts[c])
            log.debug("      %s: %.1f°", label, angle)
            if angle < _MIN_JOINT_ANGLE:
                msg = f"Pose: {label} extreme angle {angle:.1f}"
                log.warning("      ANOMALY angle: %s (score 0.60)", msg)
                anomalies.append(msg)
                sub_scores.append(0.6)
            else:
                log.debug("      %s angle %.1f° — OK", label, angle)

    def cleanup(self) -> None:
        self._hand_landmarker.close()
        self._pose_landmarker.close()
