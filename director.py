"""
Director — orchestrate 6 x 5-second segments into a 30-second video.

Usage:
    python director.py <input_image> [--prompt TEXT] [--steps N] [--cfg F]
                                     [--provider gemini|grok]
                                     [--max-retries 10] [--threshold 0.9]

Requires a running ComfyUI server (default http://127.0.0.1:8188).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_COMFYUI_INPUT = os.path.join(_PROJECT_ROOT, "ComfyUI", "input")
_COMFYUI_OUTPUT = os.path.join(_PROJECT_ROOT, "ComfyUI", "output")
_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "output")

NUM_SEGMENTS = 6
SEGMENT_DURATION = 5

# ── Relaxed content-check prompt (NSFW-tolerant) ─────────────────────

RELAXED_CHECK_PROMPT = """\
You are an expert at detecting **structural body deformations** in \
AI-generated images/video frames. Be very conservative: only flag when \
you have clear, unambiguous evidence. Avoid false positives.

ONLY flag these severe issues (and only when you are certain):
- Extra or missing fingers (more or fewer than 5 per hand) — visible and countable
- Extra or missing hands or limbs (3+ arms, 3+ legs, 1 arm, etc.)
- Limbs growing from the wrong body part
- A face with too many or too few eyes/mouths
- Completely impossible geometry (e.g. torso twisted 360 degrees)

Do NOT flag any of the following — they are EXPECTED or can look like deformations:
- Nudity, sexual poses, or explicit content
- Unusual or extreme body positions / flexibility
- Mild asymmetry or slight proportion differences
- Skin texture imperfections or lighting artefacts
- Body parts partially occluded or hidden behind other body parts (e.g. hand behind leg, arm behind torso)
- Perspective effects: limbs that appear to overlap or "merge" due to camera angle
- Fingers or hands that look odd but are just bent, angled, or partially visible
- Anything ambiguous — when in doubt, score low (0.0–0.3) and do NOT flag

Respond ONLY with valid JSON (no markdown fences). Use this schema:
{
  "score": <float 0.0-1.0, 0 = no deformation, 1 = severe deformation>,
  "anomalies": ["<short description>", ...],
  "explanation": "<one-sentence summary>"
}
If no structural deformations are found, return \
{"score": 0.0, "anomalies": [], "explanation": "No structural deformations detected."}.
"""


# ── Pacing (excitement + stableness per segment) ───────────────────────

def _get_pacing(seg_idx: int) -> tuple[int, int]:
    """Return (excitement 0-10, stableness 1-5) for orchestrated flow.

    Arc: intro (calm) -> establish -> build -> peak -> sustain -> wind down.
    Avoids making the whole movie too plain or too crazy.
    """
    # seg 1: intro, seg 2: calm, seg 3: build, seg 4: peak, seg 5: sustain, seg 6: wind down
    pacing = [
        (4, 3),   # 1: intro — moderate calm
        (3, 2),   # 2: establish — calm, minimal movement
        (5, 4),   # 3: build — more energy
        (7, 5),   # 4: peak — high energy, full movement
        (5, 4),   # 5: sustain — moderate
        (3, 3),   # 6: wind down — calm
    ]
    idx = min(seg_idx - 1, len(pacing) - 1)
    return pacing[idx]


# ── Helpers ───────────────────────────────────────────────────────────

def _make_session_dir() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(ts.encode()).hexdigest()[:6]
    path = os.path.join(_OUTPUT_ROOT, f"{ts}_{short_hash}")
    os.makedirs(path, exist_ok=True)
    return path


def _random_seed() -> int:
    return random.randint(0, 1125899906842624)


def _copy_image_to_comfyui_input(image_path: str, dest_name: str) -> str:
    """Copy an image into ComfyUI/input/ and return the basename."""
    dest = os.path.join(_COMFYUI_INPUT, dest_name)
    shutil.copy2(image_path, dest)
    return dest_name


def _find_output_video(history: dict) -> str | None:
    """Find the first .mp4 output file from a ComfyUI history record."""
    for _nid, node_out in history.get("outputs", {}).items():
        for _key, items in node_out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                fname = item.get("filename", "")
                if fname.endswith(".mp4"):
                    subfolder = item.get("subfolder", "")
                    if subfolder:
                        return os.path.join(_COMFYUI_OUTPUT, subfolder, fname)
                    return os.path.join(_COMFYUI_OUTPUT, fname)
    return None


def _find_output_latent(history: dict) -> str | None:
    """Find the saved .latent file from a ComfyUI history record."""
    from run_itv_pass1 import extract_saved_latent
    return extract_saved_latent(history)


def extract_frames_at_1fps(video_path: str) -> list[np.ndarray]:
    """Return one BGR frame per second from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []

    for sec in range(SEGMENT_DURATION + 1):
        frame_idx = int(sec * fps)
        if frame_idx >= total:
            frame_idx = total - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()
    return frames


def extract_last_frame(video_path: str) -> np.ndarray:
    """Return the very last frame of a video as BGR ndarray.

    Uses frame-by-frame read instead of seeking, because CAP_PROP_FRAME_COUNT
    often returns 0 for H.264 MP4, which would incorrectly yield frame 0.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame
    cap.release()
    if last_frame is None:
        raise RuntimeError(f"Failed to read any frame from {video_path}")
    return last_frame


def _get_ffmpeg_path() -> str:
    """Get ffmpeg binary path from imageio_ffmpeg (bundled with ComfyUI)."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    raise RuntimeError(
        "ffmpeg not found. Install imageio-ffmpeg or add ffmpeg to PATH."
    )


def concatenate_videos(video_paths: list[str], output_path: str) -> None:
    """Concatenate multiple mp4 files using ffmpeg concat demuxer."""
    ffmpeg = _get_ffmpeg_path()
    list_file = output_path + ".concat.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for vp in video_paths:
                safe = vp.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  Concatenated {len(video_paths)} segments -> {output_path}")
    finally:
        if os.path.isfile(list_file):
            os.remove(list_file)


# ── Content check ────────────────────────────────────────────────────

def _build_checker(api_keys: dict):
    """Build a GeminiChecker with the relaxed NSFW-tolerant prompt."""
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, "contentCheck"))
    from contentcheck.models.llm_checker import GeminiChecker

    key = api_keys.get("gemini") or os.environ.get("GEMINI_API_KEY", "")
    return GeminiChecker(api_key=key, prompt=RELAXED_CHECK_PROMPT)


def check_frames(frames: list[np.ndarray], checker, threshold: float
                  ) -> tuple[bool, list[dict]]:
    """Check frames for deformations. Returns (passed, per-frame results)."""
    results: list[dict] = []
    flagged = False
    for i, frame in enumerate(frames):
        try:
            mr = checker.check(frame)
            entry = {
                "frame_sec": i,
                "score": mr.score,
                "anomalies": mr.anomalies,
                "explanation": mr.details,
            }
        except Exception as exc:
            entry = {
                "frame_sec": i,
                "score": 0.0,
                "anomalies": [f"Check error: {exc}"],
                "explanation": str(exc),
            }
        results.append(entry)
        if entry["score"] >= threshold:
            flagged = True
            print(f"    Frame t={i}s FLAGGED (score={entry['score']:.2f}): "
                  f"{entry['anomalies']}")
    return not flagged, results


# ── Load API keys ────────────────────────────────────────────────────

def _load_api_keys() -> dict:
    keys_path = os.path.join(_SCRIPT_DIR, "api_keys.json")
    if os.path.isfile(keys_path):
        with open(keys_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Main pipeline ────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="director",
        description="Generate a 30-second video from 6 x 5-second segments.",
    )
    p.add_argument("image", help="Path to the input image.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Initial positive prompt for segment 1.")
    p.add_argument("--steps", type=int, default=None,
                   help="Sampling steps per segment.")
    p.add_argument("--cfg", type=float, default=None,
                   help="CFG guidance scale.")
    p.add_argument("--provider", choices=["gemini", "grok"], default="gemini",
                   help="LLM provider for autoprompt (default: gemini).")
    p.add_argument("--max-retries", type=int, default=10,
                   help="Max retries per segment on content-check failure.")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="Content-check score threshold (default: 0.9).")
    p.add_argument("--skip-check", action="store_true",
                   help="Skip content checking entirely (faster, no QA).")
    return p


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    from run_itv_pass1 import run_workflow, check_server, COMFYUI_URL
    from autoprompt import generate_prompt

    if not check_server():
        print(f"ERROR: ComfyUI server not reachable at {COMFYUI_URL}")
        print("Start it with:  python ComfyUI\\main.py --disable-auto-launch")
        sys.exit(1)

    api_keys = _load_api_keys()
    checker = None
    if not args.skip_check:
        print("Initialising content checker (relaxed mode)...")
        checker = _build_checker(api_keys)

    session_dir = _make_session_dir()
    print(f"\nSession directory: {session_dir}")
    print(f"Segments: {NUM_SEGMENTS} x {SEGMENT_DURATION}s = "
          f"{NUM_SEGMENTS * SEGMENT_DURATION}s total\n")

    manifest: dict = {
        "input_image": os.path.abspath(args.image),
        "started_at": datetime.datetime.now().isoformat(),
        "settings": {
            "steps": args.steps,
            "cfg": args.cfg,
            "provider": args.provider,
            "threshold": args.threshold,
            "max_retries": args.max_retries,
            "skip_check": args.skip_check,
        },
        "segments": [],
        "final_video": None,
    }

    current_image_path = os.path.abspath(args.image)
    current_prompt = args.prompt
    segment_videos: list[str] = []
    prev_latent_path: str | None = None  # for segments 2+

    for seg_idx in range(1, NUM_SEGMENTS + 1):
        seg_label = f"seg_{seg_idx:02d}"
        print("=" * 70)
        print(f"  SEGMENT {seg_idx}/{NUM_SEGMENTS}")
        print("=" * 70)

        # For segments 2+, generate prompt from the last frame with pacing
        excitement, stableness = _get_pacing(seg_idx) if seg_idx > 1 else (None, None)
        if seg_idx > 1:
            print(f"  Generating autoprompt (excitement={excitement}, stableness={stableness})...")
            current_prompt = generate_prompt(
                current_image_path,
                duration=SEGMENT_DURATION,
                provider=args.provider,
                excitement=excitement,
                stableness=stableness,
            )
            prompt_preview = current_prompt[:120].replace("\n", " ")
            print(f"  Autoprompt: {prompt_preview}...")

        # Copy current image into ComfyUI/input/
        img_basename = f"director_{seg_label}_{os.path.basename(current_image_path)}"
        print(f"  Input image: {os.path.basename(current_image_path)}")
        _copy_image_to_comfyui_input(current_image_path, img_basename)

        seg_record: dict = {
            "segment": seg_idx,
            "prompt": current_prompt,
            "excitement": excitement,
            "stableness": stableness,
            "attempts": [],
            "final_seed": None,
            "video_file": None,
            "content_check": None,
        }

        succeeded = False
        for attempt in range(1, args.max_retries + 1):
            seed = _random_seed()
            print(f"\n  Attempt {attempt}/{args.max_retries}  seed={seed}")

            attempt_record: dict = {
                "attempt": attempt,
                "seed": seed,
                "status": "pending",
                "content_check_results": None,
            }

            try:
                history = run_workflow(
                    img_basename,
                    steps=args.steps,
                    cfg=args.cfg,
                    prompt=current_prompt,
                    seed=seed,
                    filename_prefix=f"director_{seg_label}",
                    prev_latent_path=prev_latent_path,
                    latent_filename_prefix=f"latents/director_{seg_label}",
                )
            except RuntimeError as exc:
                print(f"  Workflow error: {exc}")
                attempt_record["status"] = "workflow_error"
                attempt_record["error"] = str(exc)
                seg_record["attempts"].append(attempt_record)
                continue

            status_str = history.get("status", {}).get("status_str", "")
            if status_str == "error":
                print(f"  ComfyUI execution error.")
                attempt_record["status"] = "comfyui_error"
                seg_record["attempts"].append(attempt_record)
                continue

            video_src = _find_output_video(history)
            if not video_src or not os.path.isfile(video_src):
                print(f"  Could not find output video file.")
                attempt_record["status"] = "no_video"
                seg_record["attempts"].append(attempt_record)
                continue

            # Content check
            if checker is not None:
                print(f"  Running content check on {SEGMENT_DURATION + 1} frames...")
                frames = extract_frames_at_1fps(video_src)
                passed, check_results = check_frames(
                    frames, checker, args.threshold,
                )
                attempt_record["content_check_results"] = check_results

                if not passed:
                    print(f"  FAILED content check — retrying with new seed.")
                    attempt_record["status"] = "content_flagged"
                    seg_record["attempts"].append(attempt_record)
                    continue

                print(f"  Content check PASSED.")
            else:
                check_results = []

            # Success — copy video to session dir
            seg_video_path = os.path.join(session_dir, f"{seg_label}.mp4")
            shutil.copy2(video_src, seg_video_path)
            segment_videos.append(seg_video_path)

            attempt_record["status"] = "success"
            seg_record["attempts"].append(attempt_record)
            seg_record["final_seed"] = seed
            seg_record["video_file"] = f"{seg_label}.mp4"
            seg_record["content_check"] = check_results
            succeeded = True

            # Extract last frame for next segment
            last_frame = extract_last_frame(video_src)
            last_frame_name = f"{seg_label}_lastframe.png"
            last_frame_path = os.path.join(session_dir, last_frame_name)
            cv2.imwrite(last_frame_path, last_frame)
            current_image_path = last_frame_path
            print(f"  Saved last frame: {last_frame_name}")

            # Copy saved latent to ComfyUI/input for next segment (latent continuity)
            latent_src = _find_output_latent(history)
            if latent_src and os.path.isfile(latent_src):
                next_seg = seg_idx + 1
                if next_seg <= NUM_SEGMENTS:
                    prev_latent_name = f"director_seg_{next_seg:02d}_prev.latent"
                    prev_latent_dest = os.path.join(_COMFYUI_INPUT, prev_latent_name)
                    shutil.copy2(latent_src, prev_latent_dest)
                    prev_latent_path = prev_latent_dest
                    print(f"  Saved prev latent for seg {next_seg}: {prev_latent_name}")
            break

        if not succeeded:
            print(f"\n  SEGMENT {seg_idx} FAILED after {args.max_retries} attempts.")
            print("  Aborting pipeline.")
            seg_record["video_file"] = None
            manifest["segments"].append(seg_record)
            break

        manifest["segments"].append(seg_record)
        print(f"\n  Segment {seg_idx} complete.\n")

    # ── Concatenation ─────────────────────────────────────────────────

    if len(segment_videos) == NUM_SEGMENTS:
        print("=" * 70)
        print("  CONCATENATING")
        print("=" * 70)
        final_path = os.path.join(session_dir, "final_30s.mp4")
        try:
            concatenate_videos(segment_videos, final_path)
            manifest["final_video"] = "final_30s.mp4"
            print(f"\n  Final video: {final_path}")
        except Exception as exc:
            print(f"  Concatenation failed: {exc}")
            manifest["final_video"] = None
    else:
        print(f"\n  Only {len(segment_videos)}/{NUM_SEGMENTS} segments completed."
              f" Skipping concatenation.")
        manifest["final_video"] = None

    # ── Write manifest ────────────────────────────────────────────────

    manifest["finished_at"] = datetime.datetime.now().isoformat()
    manifest_path = os.path.join(session_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest: {manifest_path}")

    print("\n" + "=" * 70)
    if manifest["final_video"]:
        print(f"  DONE — {os.path.join(session_dir, manifest['final_video'])}")
    else:
        print("  DONE — pipeline did not produce a final video.")
    print("=" * 70)


if __name__ == "__main__":
    main()
