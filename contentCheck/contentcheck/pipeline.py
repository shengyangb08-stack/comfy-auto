from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from contentcheck.models.base import BaseChecker
from contentcheck.results import FrameReport, ModelResult, VideoReport
from contentcheck.video import extract_frames

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _print_header(input_path: str, model_names: list[str], threshold: float) -> None:
    print()
    print("=" * 70)
    print("  contentcheck — Abnormal Anatomy Detector")
    print("=" * 70)
    print(f"  Input     : {input_path}")
    print(f"  Models    : {', '.join(model_names)}")
    print(f"  Threshold : {threshold}")
    print("=" * 70)
    print()


def _print_frame_row(
    fr: FrameReport, model_names: list[str], threshold: float,
) -> None:
    scores = fr.scores_by_model()
    flag = "🚩" if fr.max_score >= threshold else "  "
    cols = f"  {flag}  Frame {fr.frame_index:<3}  t={fr.timestamp_sec:5.1f}s  │"
    for mn in model_names:
        s = scores.get(mn, -1)
        if s < 0:
            cols += "   n/a  │"
        else:
            cols += f"  {s:.2f}  │"
    print(cols)


def _print_anomaly_details(fr: FrameReport) -> None:
    for mr in fr.model_results:
        if mr.anomalies:
            for a in mr.anomalies:
                print(f"        [{mr.model_name}] {a}")


def run_image(
    image_path: str,
    checkers: list[BaseChecker],
    threshold: float = 0.5,
    output_dir: str = "output",
) -> VideoReport:
    model_names = [c.name for c in checkers]
    _print_header(image_path, model_names, threshold)

    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        print(f"Cannot read image: {image_path}")
        sys.exit(1)

    print(f"  Image loaded: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}\n")

    hdr = f"        {'Image':<16}│"
    for mn in model_names:
        hdr += f"  {mn:^6}│"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    model_results: list[ModelResult] = []
    for checker in checkers:
        try:
            result = checker.check(frame_bgr)
        except Exception as exc:
            result = ModelResult(
                model_name=checker.name, score=0.0,
                anomalies=[f"Error: {exc}"], details=str(exc),
            )
        model_results.append(result)

    fr = FrameReport(
        frame_index=0, timestamp_sec=0.0,
        frame=frame_bgr, model_results=model_results,
    )

    scores = fr.scores_by_model()
    flag = "🚩" if fr.max_score >= threshold else "  "
    cols = f"  {flag}  {os.path.basename(image_path):<16}│"
    for mn in model_names:
        s = scores.get(mn, -1)
        cols += f"  {s:.2f}  │" if s >= 0 else "   n/a  │"
    print(cols)
    if fr.max_score >= threshold:
        _print_anomaly_details(fr)

    report = VideoReport(
        video_path=image_path,
        total_frames_extracted=1,
        frame_reports=[fr],
        threshold=threshold,
    )

    print()
    print("─" * 70)
    if fr.max_score >= threshold:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"flagged_{os.path.basename(image_path)}"
        path = os.path.join(output_dir, fname)
        cv2.imwrite(path, frame_bgr)
        scores_str = ", ".join(f"{mr.model_name}={mr.score:.2f}" for mr in model_results)
        print(f"\n  Image flagged (score >= {threshold}):\n")
        print(f"    Saved: {path}  ({scores_str})")
        for mr in model_results:
            for a in mr.anomalies:
                print(f"      - [{mr.model_name}] {a}")
        print()
    else:
        print(f"\n  Score below threshold ({threshold}). Image looks clean.\n")

    for c in checkers:
        c.cleanup()

    return report


def run(
    video_path: str,
    checkers: list[BaseChecker],
    threshold: float = 0.5,
    output_dir: str = "output",
    save_frames_dir: str | None = None,
) -> VideoReport:
    model_names = [c.name for c in checkers]
    _print_header(video_path, model_names, threshold)

    print("Extracting frames (1 per second) ...")
    frames = extract_frames(video_path, fps=1)
    print(f"  → {len(frames)} frames extracted")

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        for idx, ts, f in frames:
            p = os.path.join(save_frames_dir, f"frame_{idx:03d}_t{ts:.1f}s.jpg")
            cv2.imwrite(p, f)
        print(f"  → Saved extracted frames to {save_frames_dir}")
    print()

    if not frames:
        print("No frames extracted. Is the video file valid?")
        sys.exit(1)

    # Table header
    hdr = f"        {'Frame':<8} {'Time':>6}  │"
    for mn in model_names:
        hdr += f"  {mn:^6}│"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    frame_reports: list[FrameReport] = []

    for frame_idx, timestamp, frame_bgr in frames:
        model_results: list[ModelResult] = []
        for checker in checkers:
            t0 = time.time()
            try:
                result = checker.check(frame_bgr)
            except Exception as exc:
                result = ModelResult(
                    model_name=checker.name,
                    score=0.0,
                    anomalies=[f"Error: {exc}"],
                    details=str(exc),
                )
            elapsed = time.time() - t0
            model_results.append(result)

        fr = FrameReport(
            frame_index=frame_idx,
            timestamp_sec=timestamp,
            frame=frame_bgr,
            model_results=model_results,
        )
        frame_reports.append(fr)
        _print_frame_row(fr, model_names, threshold)
        if fr.max_score >= threshold:
            _print_anomaly_details(fr)

    report = VideoReport(
        video_path=video_path,
        total_frames_extracted=len(frames),
        frame_reports=frame_reports,
        threshold=threshold,
    )

    # Save flagged frames
    flagged = report.flagged_reports
    print()
    print("─" * 70)
    if flagged:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n  {len(flagged)} frame(s) flagged (score >= {threshold}):\n")
        for fr in flagged:
            fname = f"frame_{fr.frame_index:03d}_t{fr.timestamp_sec:.1f}s.jpg"
            path = os.path.join(output_dir, fname)
            cv2.imwrite(path, fr.frame)
            scores_str = ", ".join(
                f"{mr.model_name}={mr.score:.2f}" for mr in fr.model_results
            )
            print(f"    Saved: {path}  ({scores_str})")
            for mr in fr.model_results:
                if mr.anomalies:
                    for a in mr.anomalies:
                        print(f"      - [{mr.model_name}] {a}")
        print()
    else:
        print(f"\n  No frames exceeded the threshold ({threshold}). Video looks clean.\n")

    # Cleanup
    for c in checkers:
        c.cleanup()

    return report
