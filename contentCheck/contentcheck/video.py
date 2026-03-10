from __future__ import annotations

import cv2
import numpy as np


def extract_frames(video_path: str, fps: int = 1) -> list[tuple[int, float, np.ndarray]]:
    """Extract frames from a video at the given sample rate.

    Returns a list of (frame_index, timestamp_seconds, frame_bgr) tuples.
    With fps=1, one frame is extracted per second of video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    interval = int(round(video_fps / fps))
    if interval < 1:
        interval = 1

    frames: list[tuple[int, float, np.ndarray]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            timestamp = idx / video_fps
            frames.append((len(frames), timestamp, frame))
        idx += 1

    cap.release()
    return frames
