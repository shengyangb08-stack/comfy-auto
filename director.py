"""
Director — orchestrate N x 5-second segments into a video.

Usage:
  With script (剧本) — full control per segment:
    python director.py <input_image> <script.json> [--provider gemini|grok] ...

  Without script — auto-generated prompts:
    python director.py <input_image> [--segments N] [--duration SECONDS]  (default: 2 segments = 10s)
                                     [--prompt TEXT] [--steps N] [--cfg F] ...

Script format (script.json):
  {"segments": [{"segment": 1, "high_level_prompt": "...", "excitement": 4, "stableness": 3}, ...]}
  excitement/stableness optional (defaults: 5, 3). See script/script_example.json, script/script_two_segment_climax.json,
  script/script_five_segment_arc.json, SCRIPT_README.md.

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
import sys
import time

import cv2
import numpy as np

from project_paths import get_director_sessions_root

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_COMFYUI_INPUT = os.path.join(_PROJECT_ROOT, "ComfyUI", "input")
_COMFYUI_OUTPUT = os.path.join(_PROJECT_ROOT, "ComfyUI", "output")
_OUTPUT_ROOT = get_director_sessions_root()

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


# ── Pacing (excitement + stableness) + narrative arc for autoprompt ──

def _get_pacing(seg_idx: int, num_segments: int) -> tuple[int, int]:
    """Return (excitement 0-10, stableness 1-5) for motion → climax arc.

    Default intent (no script):
    - 2 segments: segment 1 = more motion; segment 2 = climax (squirt, face, release).
    - 1 segment: combined motion + climax in one 5s block.
    - 5 segments: stable → motion → more motion → climax (seg 4) → stable (seg 5).
    - 3–4 segments: segment 1 motion, segment 2 designated climax; later = aftermath/cool-down.
    """
    if num_segments == 1:
        return (8, 4)
    if num_segments == 2:
        if seg_idx == 1:
            return (8, 2)   # more motion, wider second-by-second detail
        return (9, 4)     # climax — strong + enough stableness for full lines

    # Five segments: stable → motion → more motion → climax → stable
    if num_segments == 5:
        pacing5 = {
            1: (4, 4),   # stable
            2: (6, 3),   # motion
            3: (8, 2),   # more motion
            4: (9, 4),   # climax
            5: (4, 4),   # back to stable
        }
        return pacing5[seg_idx]

    # 3–4 segments (legacy): motion → climax on seg 2 → taper
    if seg_idx == 1:
        return (8, 2)
    if seg_idx == 2:
        return (9, 4)
    return (5, 4)


def _segment_arc_instruction(seg_idx: int, num_segments: int) -> str:
    """Extra LLM instructions: timeline density + explicit climax when required."""
    if num_segments == 1:
        return (
            "NARRATIVE ARC (single 5s clip): Build energy then reach a clear CLIMAX in this segment. "
            "Explicitly describe orgasm, squirt if applicable, and facial expression (eyes, mouth, tension release) "
            "as the peak approaches. Within the 5 seconds, at least 3 seconds must be fully detailed timestamp lines."
        )
    if num_segments == 2:
        if seg_idx == 1:
            return (
                "NARRATIVE ARC (segment 1 of 2 — MOTION): Prioritize MORE movement and body dynamics. "
                "Do NOT climax here — reserve orgasm/squirt and peak facial expression for segment 2. "
                "Across this 5s block, at least 3 seconds (e.g. 0–2 or 0–3) must be full detailed timestamp lines; "
                "use any remaining lines to transition toward rising tension."
            )
        return (
            "NARRATIVE ARC (segment 2 of 2 — CLIMAX): The character MUST reach CLIMAX in this block. "
            "Explicitly describe: orgasm, squirt if applicable, and facial expression (eyes closing, mouth opening, "
            "release, afterglow). Peak intensity should land around seconds 2–4 when possible. "
            "At least 3 seconds must still be fully detailed timestamp lines."
        )

    # Five segments: stable → motion → more motion → climax → stable
    if num_segments == 5:
        arc5 = {
            1: (
                "NARRATIVE ARC (segment 1 of 5 — STABLE): Calm, minimal movement; subtle breathing, small shifts only. "
                "No climax. At least 3 seconds fully detailed timestamp lines."
            ),
            2: (
                "NARRATIVE ARC (segment 2 of 5 — MOTION): Clear increase in movement and energy vs segment 1. "
                "Still no climax. At least 3 detailed seconds."
            ),
            3: (
                "NARRATIVE ARC (segment 3 of 5 — MORE MOTION): Strongest movement before the peak; build tension. "
                "Do not climax yet. At least 3 detailed seconds."
            ),
            4: (
                "NARRATIVE ARC (segment 4 of 5 — CLIMAX): The character MUST reach CLIMAX here. "
                "Explicit orgasm, squirt if applicable, facial expression (eyes, mouth, release). "
                "Peak ~seconds 2–4. At least 3 detailed seconds."
            ),
            5: (
                "NARRATIVE ARC (segment 5 of 5 — STABLE AGAIN): Afterglow, calm, slower movement; return to stability. "
                "No new climax. At least 3 detailed seconds."
            ),
        }
        return arc5[seg_idx]

    # 3–4 segments: climax on segment 2, then aftermath
    if seg_idx == 1:
        return (
            "NARRATIVE ARC (segment 1): More motion; at least 3 detailed seconds; do not climax yet."
        )
    if seg_idx == 2:
        return (
            "NARRATIVE ARC (segment 2 — CLIMAX): Designated climax segment. At least one full run must reach climax "
            "across the video — this is it. Explicit orgasm, squirt if applicable, facial expression; peak ~seconds 2–4."
        )
    return (
        "NARRATIVE ARC (later segment): Aftermath, cool-down, or continuation after climax; "
        "keep timestamp format; at least 3 seconds detailed if possible."
    )


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
    from run_itv_director import find_output_video
    return find_output_video(history, _COMFYUI_OUTPUT)


def _find_output_latent(history: dict) -> str | None:
    """Find the saved .latent file from a ComfyUI history record."""
    from run_itv_director import extract_saved_latent
    return extract_saved_latent(history, _COMFYUI_OUTPUT)


def _find_output_saved_images(history: dict, save_image_id: str = "773") -> list[str]:
    """Find all saved images from SaveImage node (for next segment's extend input)."""
    from run_itv_director import extract_saved_images
    return extract_saved_images(history, save_image_id, _COMFYUI_OUTPUT)


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
        description="Generate a video from N x 5-second segments.",
    )
    p.add_argument("image", help="Path to the input image.")
    p.add_argument("script", nargs="?", default=None,
                   help="Path to script JSON (剧本). If provided, segments/prompts come from script.")
    p.add_argument("--segments", type=int, default=2,
                   help="Number of 5-second segments when not using script (default: 2 = 10s).")
    p.add_argument("--duration", type=int, default=None,
                   help="Total duration in seconds when not using script; overrides --segments.")
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
    p.add_argument("--lightning-combo", choices=["1", "2", "3"], default="2",
                   help="Lightning LoRA: 1=more motion, 2=less degradation (default), 3=balanced.")
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

    script_entries: list[dict] | None = None
    if args.script:
        if not os.path.isfile(args.script):
            print(f"ERROR: Script not found: {args.script}", file=sys.stderr)
            sys.exit(1)
        with open(args.script, encoding="utf-8") as f:
            script_data = json.load(f)
        script_entries = script_data.get("segments", [])
        if not script_entries:
            print("ERROR: Script has no 'segments' array.", file=sys.stderr)
            sys.exit(1)
        num_segments = len(script_entries)
        print(f"  Loaded script: {num_segments} segments")
    else:
        # Resolve segment count: --duration overrides --segments
        if args.duration is not None:
            num_segments = max(1, args.duration // SEGMENT_DURATION)
            if args.duration % SEGMENT_DURATION:
                print(f"  Note: {args.duration}s rounds to {num_segments} x {SEGMENT_DURATION}s = "
                      f"{num_segments * SEGMENT_DURATION}s")
        else:
            num_segments = max(1, args.segments)

    from run_itv_director import (
        run_first5,
        run_extend5,
        check_server,
        COMFYUI_URL,
        get_image_size,
    )
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
    total_seconds = num_segments * SEGMENT_DURATION
    print(f"\nSession directory: {session_dir}")
    print(f"Segments: {num_segments} x {SEGMENT_DURATION}s = {total_seconds}s total\n")

    manifest: dict = {
        "input_image": os.path.abspath(args.image),
        "started_at": datetime.datetime.now().isoformat(),
        "settings": {
            "segments": num_segments,
            "segment_duration": SEGMENT_DURATION,
            "total_seconds": total_seconds,
            "script": args.script,
            "steps": args.steps,
            "cfg": args.cfg,
            "provider": args.provider,
            "threshold": args.threshold,
            "max_retries": args.max_retries,
            "skip_check": args.skip_check,
            "lightning_combo": args.lightning_combo,
        },
        "segments": [],
        "final_video": None,
    }

    # Anchor image: original input, used for all segments (character reference)
    anchor_image_path = os.path.abspath(args.image)
    anchor_basename = f"director_anchor_{os.path.basename(anchor_image_path)}"
    _copy_image_to_comfyui_input(anchor_image_path, anchor_basename)
    print(f"  Anchor image (for run): {anchor_basename}")

    current_image_path = anchor_image_path  # for seg 1; seg 2+ = last saved image (for autoprompt)
    current_prompt = args.prompt  # used only for seg 1 when no script; else from script
    prev_images_folder: str | None = None  # session_dir/seg_XX/ for extend (LoadImagesFromFolderKJ)
    prev_latent_basename: str | None = None  # filename in ComfyUI/input/ for extend
    prev_width: int | None = None  # resolution from previous segment (for extend)
    prev_height: int | None = None

    seg_pad = len(str(num_segments))
    for seg_idx in range(1, num_segments + 1):
        seg_label = f"seg_{seg_idx:0{seg_pad}d}"
        seg_images_dir = os.path.join(session_dir, seg_label)  # where we store this segment's images
        print("=" * 70)
        print(f"  SEGMENT {seg_idx}/{num_segments}")
        print("=" * 70)

        # Resolve prompt for this segment
        excitement, stableness = None, None
        if script_entries:
            entry = script_entries[seg_idx - 1]
            high_level = entry.get("high_level_prompt", "").strip()
            if not high_level:
                print(f"  ERROR: Segment {seg_idx} has no high_level_prompt in script.")
                break
            excitement = entry.get("excitement", 5)
            stableness = entry.get("stableness", 3)
            print(f"  Script: \"{high_level}\" (ex={excitement}, st={stableness})")
            print(f"  Generating autoprompt (excitement={excitement}, stableness={stableness})...")
            arc = _segment_arc_instruction(seg_idx, num_segments)
            llm_prompt = generate_prompt(
                current_image_path,
                duration=SEGMENT_DURATION,
                provider=args.provider,
                excitement=excitement,
                stableness=stableness,
                segment_arc=arc,
            )
            current_prompt = high_level + "\n" + llm_prompt
            print(f"  Prompt (high_level + LLM):\n{current_prompt}")
        elif seg_idx == 1:
            if not current_prompt:
                print(f"  Generating autoprompt for segment 1 (no --prompt, no script)...")
                excitement, stableness = _get_pacing(1, num_segments)
                arc = _segment_arc_instruction(1, num_segments)
                current_prompt = generate_prompt(
                    current_image_path,
                    duration=SEGMENT_DURATION,
                    provider=args.provider,
                    excitement=excitement,
                    stableness=stableness,
                    segment_arc=arc,
                )
                print(f"  Autoprompt:\n{current_prompt}")
            else:
                print(f"  Using --prompt for segment 1")
                print(f"  Prompt:\n{current_prompt}")
        else:
            excitement, stableness = _get_pacing(seg_idx, num_segments)
            arc = _segment_arc_instruction(seg_idx, num_segments)
            print(f"  Generating autoprompt (excitement={excitement}, stableness={stableness})...")
            current_prompt = generate_prompt(
                current_image_path,
                duration=SEGMENT_DURATION,
                provider=args.provider,
                excitement=excitement,
                stableness=stableness,
                segment_arc=arc,
            )
            print(f"  Autoprompt:\n{current_prompt}")

        # Seg 1: first5 uses source image. Seg 2+: extend5 uses prev_images_folder + prev_latent + anchor
        if seg_idx == 1:
            img_basename = f"director_{seg_label}_source_{os.path.basename(current_image_path)}"
            print(f"  Source: single image {os.path.basename(current_image_path)}")
            _copy_image_to_comfyui_input(current_image_path, img_basename)
        else:
            print(f"  Extend: prev images from {prev_images_folder}, latent={prev_latent_basename}")

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
        if script_entries and seg_idx <= len(script_entries):
            seg_record["high_level_prompt"] = script_entries[seg_idx - 1].get("high_level_prompt", "")

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
                if seg_idx == 1:
                    history, prompt_used = run_first5(
                        img_basename,
                        prompt=current_prompt,
                        seed=seed,
                        steps=args.steps,
                        cfg=args.cfg,
                        filename_prefix=f"director_{seg_label}",
                        latent_filename_prefix=f"director_{seg_label}",
                        lightning_combo=args.lightning_combo,
                    )
                else:
                    if prev_width is None or prev_height is None:
                        raise RuntimeError(
                            "Extend requires prev_width/prev_height from previous segment. "
                            "Ensure seg 1 completed and saved images."
                        )
                    history, prompt_used = run_extend5(
                        anchor_basename,
                        prev_images_folder,
                        prev_latent_basename,
                        prompt=current_prompt,
                        seed=seed,
                        width=prev_width,
                        height=prev_height,
                        steps=args.steps,
                        cfg=args.cfg,
                        filename_prefix=f"director_{seg_label}",
                        latent_filename_prefix=f"director_{seg_label}",
                        lightning_combo=args.lightning_combo,
                    )
                seg_record["prompt"] = prompt_used  # actual prompt sent (with LoRA trigger)
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

            # Success — copy video and images to session dir (all in run folder)
            seg_video_path = os.path.join(session_dir, f"{seg_label}.mp4")
            shutil.copy2(video_src, seg_video_path)

            attempt_record["status"] = "success"
            seg_record["attempts"].append(attempt_record)
            seg_record["final_seed"] = seed
            seg_record["video_file"] = f"{seg_label}.mp4"
            seg_record["content_check"] = check_results
            succeeded = True

            # Copy saved images to session_dir/seg_XX/ for next segment's extend
            saved_images = _find_output_saved_images(history)
            next_seg = seg_idx + 1
            if saved_images and next_seg <= num_segments:
                os.makedirs(seg_images_dir, exist_ok=True)
                for i, src in enumerate(saved_images):
                    if os.path.isfile(src):
                        ext = os.path.splitext(src)[1]
                        dest = os.path.join(seg_images_dir, f"frame_{i:05d}{ext}")
                        shutil.copy2(src, dest)
                prev_images_folder = seg_images_dir
                current_image_path = saved_images[-1]
                # Get resolution from first image for next extend (must match prev latent)
                sz = get_image_size(saved_images[0])
                if sz:
                    prev_width, prev_height = sz
                    print(f"  Saved {len(saved_images)} images to {seg_label}/ for seg {next_seg} (res {prev_width}x{prev_height})")
                else:
                    print(f"  Saved {len(saved_images)} images to {seg_label}/ for seg {next_seg}")
            elif next_seg <= num_segments:
                # Fallback: extract last frame from video
                last_frame = extract_last_frame(video_src)
                fallback_img = os.path.join(seg_images_dir, "frame_00000.png")
                os.makedirs(seg_images_dir, exist_ok=True)
                cv2.imwrite(fallback_img, last_frame)
                prev_images_folder = seg_images_dir
                current_image_path = fallback_img
                sz = get_image_size(fallback_img)
                if sz:
                    prev_width, prev_height = sz
                print(f"  Saved fallback frame for seg {next_seg} from video")

            # Full-timeline frames for post_edit: last segment's Save batch = full extended video frames
            if seg_idx == num_segments and saved_images:
                seg_all_dir = os.path.join(session_dir, "seg_all")
                os.makedirs(seg_all_dir, exist_ok=True)
                for i, src in enumerate(saved_images):
                    if os.path.isfile(src):
                        ext = os.path.splitext(src)[1]
                        dest = os.path.join(seg_all_dir, f"frame_{i:05d}{ext}")
                        shutil.copy2(src, dest)
                print(
                    f"  Saved {len(saved_images)} frames to seg_all/ for post-processing "
                    f"(full timeline; post_editor uses this by default)."
                )

            # Copy latent to session dir and ComfyUI/input for next segment
            latent_src = _find_output_latent(history)
            if latent_src and os.path.isfile(latent_src):
                session_latent = os.path.join(session_dir, f"{seg_label}.latent")
                shutil.copy2(latent_src, session_latent)
                if next_seg <= num_segments:
                    prev_latent_basename = f"director_seg_{next_seg:0{seg_pad}d}_prev.latent"
                    prev_latent_dest = os.path.join(_COMFYUI_INPUT, prev_latent_basename)
                    shutil.copy2(latent_src, prev_latent_dest)
                    print(f"  Saved latent for seg {next_seg}: {prev_latent_basename}")
            break

        if not succeeded:
            print(f"\n  SEGMENT {seg_idx} FAILED after {args.max_retries} attempts.")
            print("  Aborting pipeline.")
            seg_record["video_file"] = None
            manifest["segments"].append(seg_record)
            break

        manifest["segments"].append(seg_record)
        print(f"\n  Segment {seg_idx} complete.\n")

    # ── Final video ───────────────────────────────────────────────────
    # Extend workflow combines video internally; last segment = full video

    completed = sum(1 for s in manifest["segments"] if s.get("video_file"))
    last_seg_label = f"seg_{num_segments:0{seg_pad}d}"
    final_seg_video = os.path.join(session_dir, f"{last_seg_label}.mp4")
    final_video_name = f"final_{total_seconds}s.mp4"
    final_path = os.path.join(session_dir, final_video_name)
    if completed == num_segments and os.path.isfile(final_seg_video):
        shutil.copy2(final_seg_video, final_path)
        manifest["final_video"] = final_video_name
        print(f"\n  Final video: {final_path}")
    else:
        if completed < num_segments:
            print(f"\n  Only {completed}/{num_segments} segments completed.")
        manifest["final_video"] = None

    # ── Write manifest ────────────────────────────────────────────────

    manifest["finished_at"] = datetime.datetime.now().isoformat()
    seg_all_dir = os.path.join(session_dir, "seg_all")
    if os.path.isdir(seg_all_dir):
        for name in os.listdir(seg_all_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                manifest["post_edit_frames"] = "seg_all"
                break

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
