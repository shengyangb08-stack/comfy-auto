"""
Post-editor: after director.py, upscale frame folders, RIFE interpolate (2x), then encode MP4.

Pipeline (ComfyUI API, server must be running):
  1) Scale_Up_API.json     — LoadImagesFromFolderKJ → 4× model → ImageScaleBy (auto ≤4K long edge) → SaveImage
  2) Fill_frame_API.json   — FL_RIFE multiplier 2 (batches of N frames)
  3) Final_Combine_API.json — VHS_VideoCombine from filled frames folder
     (LoadImages crop size capped to ~QHD+20%% by default to limit RAM; upscale step unchanged)

Typical usage (director session with seg_XX/ or seg_all/ images):

  python post_editor.py path/to/director_session_YYYYMMDD_hash

Or a folder that already has sequential frame_00000.png, …:

  python post_editor.py --frames-dir C:\\path\\to\\frames

See POST_EDITOR_README.md for options.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Any

# Reuse Comfy client + paths from run_itv_director
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from run_itv_director import (  # noqa: E402
    COMFYUI_OUTPUT_DIR,
    check_server,
    extract_saved_images,
    get_image_size,
    queue_prompt,
    wait_for_completion,
)

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

WORKFLOW_SCALE = os.path.join(_SCRIPT_DIR, "api_workflow", "Scale_Up_API.json")
WORKFLOW_FILL = os.path.join(_SCRIPT_DIR, "api_workflow", "Fill_frame_API.json")
WORKFLOW_COMBINE = os.path.join(_SCRIPT_DIR, "api_workflow", "Final_Combine_API.json")

# Node IDs match the checked-in API JSONs
NODE_SCALE_IMAGESCALE = "1"  # ImageScaleBy (after 4× model)
NODE_SCALE_LOAD = "4"
NODE_SCALE_SAVE = "5"
NODE_FILL_LOAD = "2"
NODE_FILL_SAVE = "3"
NODE_COMBINE_LOAD = "2"
NODE_COMBINE_VHS = "1"

DEFAULT_BATCH = 16

# Default cap: UHD long edge (4K). ImageScaleBy reduces only when 4× model would exceed this.
DEFAULT_MAX_OUTPUT_EDGE = 3840

# Final combine only: max width×height for LoadImages (QHD 2560×1440 + 20%). Scale/RIFE still use full-res files.
QHD_TOTAL_PIXELS = 2560 * 1440
DEFAULT_COMBINE_MAX_PIXELS = int(QHD_TOTAL_PIXELS * 1.2)  # 4_423_680


def combine_loader_dims_for_pixel_cap(
    width: int,
    height: int,
    *,
    max_total_pixels: int,
) -> tuple[int, int]:
    """
    Uniform downscale so ``width * height <= max_total_pixels`` (aspect ratio preserved).

    Used for the VHS combine step only; does not change on-disk images from scale/RIFE.
    """
    if max_total_pixels <= 0:
        return width, height
    if width < 1 or height < 1:
        return max(1, width), max(1, height)
    total = float(width) * float(height)
    if total <= float(max_total_pixels):
        return width, height
    scale = math.sqrt(float(max_total_pixels) / total)
    w_new = max(1, int(round(width * scale)))
    h_new = max(1, int(round(height * scale)))
    while w_new * h_new > max_total_pixels:
        if w_new >= h_new and w_new > 1:
            w_new -= 1
        elif h_new > 1:
            h_new -= 1
        else:
            break
    return w_new, h_new


def compute_image_scale_by_cap_max_edge(
    width: int,
    height: int,
    *,
    upscale_model_factor: float = 4.0,
    max_long_edge: int = DEFAULT_MAX_OUTPUT_EDGE,
) -> float:
    """
    Workflow: Load → ImageUpscaleWithModel (~4×) → ImageScaleBy(scale_by).

    Choose ``scale_by`` ≤ 1 so the larger side after both steps is ≤ ``max_long_edge``.
    """
    if width < 1 or height < 1 or upscale_model_factor <= 0:
        return 1.0
    after_model = max(
        float(width) * upscale_model_factor,
        float(height) * upscale_model_factor,
    )
    if after_model <= float(max_long_edge):
        return 1.0
    s = float(max_long_edge) / after_model
    return max(0.01, min(1.0, round(s, 4)))


# Flat folder: any of these (non-recursive) — not only frame_00000.png (Comfy Save uses other names).
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_image_paths_in_folder(folder: str, *, recursive: bool = False) -> list[str]:
    """All image files under ``folder``, natural sort by basename. Default: top-level only."""
    if not os.path.isdir(folder):
        return []
    paths: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(_IMAGE_SUFFIXES):
                    p = os.path.join(root, name)
                    if os.path.isfile(p):
                        paths.append(p)
    else:
        for name in os.listdir(folder):
            if not name.lower().endswith(_IMAGE_SUFFIXES):
                continue
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                paths.append(p)
    return _natural_sort_frame_paths(paths)


def clear_folder_images(folder: str) -> None:
    """Remove all image files in a flat folder (pipeline cleanup between runs)."""
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if not name.lower().endswith(_IMAGE_SUFFIXES):
            continue
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _natural_sort_frame_paths(paths: list[str]) -> list[str]:
    def key(p: str) -> tuple[Any, ...]:
        base = os.path.basename(p)
        parts = re.split(r"(\d+)", base)
        out: list[Any] = []
        for x in parts:
            if x.isdigit():
                out.append(int(x))
            else:
                out.append(x.lower())
        return tuple(out)

    return sorted(paths, key=key)


def _dir_has_frames(d: str) -> bool:
    return len(list_image_paths_in_folder(d)) > 0


def gather_frames_from_single_dir(d: str) -> list[str]:
    return list_image_paths_in_folder(d)


def list_numeric_seg_dirs(session_dir: str) -> list[str]:
    """seg_1, seg_2, … (numeric only) — excludes seg_all."""
    dirs: list[str] = []
    for path in glob.glob(os.path.join(session_dir, "seg_*")):
        if not os.path.isdir(path):
            continue
        base = os.path.basename(path)
        if base.lower() == "seg_all":
            continue
        if re.match(r"^seg_\d+$", base, re.IGNORECASE):
            dirs.append(path)

    def seg_num(p: str) -> int:
        m = re.search(r"seg_(\d+)", os.path.basename(p), re.IGNORECASE)
        return int(m.group(1)) if m else 0

    return sorted(dirs, key=seg_num)


def gather_frames_from_session(session_dir: str, source: str) -> list[str]:
    """
    Collect frame image paths in timeline order.

    - **auto** (default): use ``seg_all/`` if present (full timeline from last segment);
      else use the highest numeric ``seg_N/`` only (not a concat of all seg_*).
    - **seg_all**: only ``seg_all/``.
    - **last**: only highest ``seg_N/`` (excludes seg_all).
    - **all**: concatenate frames from every numeric ``seg_1`` … ``seg_N`` in order
      (can duplicate frames if each folder is cumulative — avoid unless you know you need it).
    """
    seg_all = os.path.join(session_dir, "seg_all")

    if source == "auto":
        if _dir_has_frames(seg_all):
            print("  Frames source: seg_all/ (full timeline from director).")
            out = gather_frames_from_single_dir(seg_all)
        else:
            dirs = list_numeric_seg_dirs(session_dir)
            if not dirs:
                raise FileNotFoundError(
                    f"No seg_N/ or seg_all/ frames under {session_dir}. "
                    "Re-run director with a recent build, or use --frames-dir."
                )
            last_d = dirs[-1]
            print(
                f"  Frames source: {os.path.basename(last_d)}/ only (no seg_all/ — old session or "
                "missing Save batch on last segment)."
            )
            out = gather_frames_from_single_dir(last_d)
        if not out:
            raise FileNotFoundError(f"No frame_* images in chosen folder under {session_dir}")
        return out

    if source == "seg_all":
        if not _dir_has_frames(seg_all):
            raise FileNotFoundError(
                f"seg_all/ missing or empty under {session_dir}. "
                "Re-run director, or use --frames-source last|auto."
            )
        out = gather_frames_from_single_dir(seg_all)
        if not out:
            raise FileNotFoundError(f"No frame_* in seg_all/ under {session_dir}")
        return out

    if source == "last":
        dirs = list_numeric_seg_dirs(session_dir)
        if not dirs:
            raise FileNotFoundError(f"No seg_N/ folders under {session_dir}")
        out = gather_frames_from_single_dir(dirs[-1])
        if not out:
            raise FileNotFoundError(f"No frame_* in {dirs[-1]}")
        return out

    if source == "all":
        dirs = list_numeric_seg_dirs(session_dir)
        if not dirs:
            raise FileNotFoundError(f"No seg_N/ folders under {session_dir}")
        out: list[str] = []
        for d in dirs:
            out.extend(gather_frames_from_single_dir(d))
        out = _natural_sort_frame_paths(out)
        if not out:
            raise FileNotFoundError(f"No frame_* images under seg_* in {session_dir}")
        print(
            "  Frames source: all numeric seg_* concatenated (may duplicate if folders are cumulative)."
        )
        return out

    raise ValueError(f"Unknown frames source: {source}")


def materialize_sequential_folder(frame_paths: list[str], dest_dir: str) -> str:
    """Copy frames into dest_dir as frame_00000.png, … for LoadImagesFromFolderKJ."""
    os.makedirs(dest_dir, exist_ok=True)
    for i, src in enumerate(frame_paths):
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        dst = os.path.join(dest_dir, f"frame_{i:05d}{ext}")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
    return dest_dir


def find_first_mp4_in_history(history: dict, output_dir: str) -> str | None:
    """Walk Comfy history outputs for the first .mp4 path on disk."""
    found: list[str] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            fn = obj.get("filename", "")
            if isinstance(fn, str) and fn.endswith(".mp4"):
                sub = obj.get("subfolder", "") or ""
                p = os.path.join(output_dir, sub, fn) if sub else os.path.join(output_dir, fn)
                found.append(p)
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for x in obj:
                collect(x)

    collect(history.get("outputs", {}))
    for p in found:
        if os.path.isfile(p):
            return p
    return found[0] if found else None


def _load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_folder_loader(
    workflow: dict,
    node_id: str,
    folder: str,
    width: int,
    height: int,
    start_index: int,
    image_load_cap: int,
) -> None:
    node = workflow[node_id]
    if node.get("class_type") != "LoadImagesFromFolderKJ":
        raise ValueError(f"Node {node_id} is not LoadImagesFromFolderKJ")
    inp = node.setdefault("inputs", {})
    inp["folder"] = os.path.abspath(folder)
    inp["width"] = width
    inp["height"] = height
    inp["start_index"] = int(start_index)
    inp["image_load_cap"] = int(image_load_cap)
    inp.setdefault("keep_aspect_ratio", "crop")
    inp.setdefault("include_subfolders", False)


def _set_save_prefix(workflow: dict, node_id: str, prefix: str) -> None:
    node = workflow[node_id]
    inp = node.setdefault("inputs", {})
    inp["filename_prefix"] = prefix


def _set_scale_workflow_image_scale_by(workflow: dict, scale_by: float) -> None:
    node = workflow.get(NODE_SCALE_IMAGESCALE)
    if not node or node.get("class_type") != "ImageScaleBy":
        raise ValueError("Scale_Up workflow: node 1 must be ImageScaleBy")
    node.setdefault("inputs", {})["scale_by"] = float(scale_by)


def _set_vhs_combine(
    workflow: dict,
    node_id: str,
    *,
    folder: str,
    width: int,
    height: int,
    frame_rate: float,
    filename_prefix: str,
) -> None:
    _set_folder_loader(workflow, node_id, folder, width, height, 0, 0)
    vhs = workflow["1"]
    if vhs.get("class_type") != "VHS_VideoCombine":
        raise ValueError("Final combine: expected node 1 = VHS_VideoCombine")
    inp = vhs.setdefault("inputs", {})
    inp["frame_rate"] = float(frame_rate)
    inp["filename_prefix"] = filename_prefix


def run_scale_batch(
    workflow_template: dict,
    *,
    folder: str,
    width: int,
    height: int,
    start_index: int,
    batch_size: int,
    save_prefix: str,
    scale_by: float = 1.0,
) -> tuple[dict, list[str]]:
    wf = json.loads(json.dumps(workflow_template))
    _set_folder_loader(wf, NODE_SCALE_LOAD, folder, width, height, start_index, batch_size)
    _set_scale_workflow_image_scale_by(wf, scale_by)
    _set_save_prefix(wf, NODE_SCALE_SAVE, save_prefix)
    pid = queue_prompt(wf).get("prompt_id")
    if not pid:
        raise RuntimeError("queue_prompt returned no prompt_id")
    hist = wait_for_completion(pid)
    paths = extract_saved_images(hist, save_image_id=NODE_SCALE_SAVE, output_dir=COMFYUI_OUTPUT_DIR)
    if not paths:
        paths = extract_saved_images(hist, save_image_id="5", output_dir=COMFYUI_OUTPUT_DIR)
    if not paths:
        # scan any save
        paths = extract_saved_images(hist, save_image_id="773", output_dir=COMFYUI_OUTPUT_DIR)
    paths = _natural_sort_frame_paths(paths)
    return hist, paths


def run_fill_batch(
    workflow_template: dict,
    *,
    folder: str,
    width: int,
    height: int,
    start_index: int,
    batch_size: int,
    save_prefix: str,
) -> tuple[dict, list[str]]:
    wf = json.loads(json.dumps(workflow_template))
    _set_folder_loader(wf, NODE_FILL_LOAD, folder, width, height, start_index, batch_size)
    _set_save_prefix(wf, NODE_FILL_SAVE, save_prefix)
    pid = queue_prompt(wf).get("prompt_id")
    if not pid:
        raise RuntimeError("queue_prompt returned no prompt_id")
    hist = wait_for_completion(pid)
    paths = extract_saved_images(hist, save_image_id=NODE_FILL_SAVE, output_dir=COMFYUI_OUTPUT_DIR)
    if not paths:
        paths = extract_saved_images(hist, save_image_id="3", output_dir=COMFYUI_OUTPUT_DIR)
    paths = _natural_sort_frame_paths(paths)
    return hist, paths


def run_final_combine(
    workflow_template: dict,
    *,
    folder: str,
    width: int,
    height: int,
    frame_rate: float,
    filename_prefix: str,
) -> tuple[dict, str | None]:
    wf = json.loads(json.dumps(workflow_template))
    _set_vhs_combine(
        wf,
        NODE_COMBINE_LOAD,
        folder=folder,
        width=width,
        height=height,
        frame_rate=frame_rate,
        filename_prefix=filename_prefix,
    )
    pid = queue_prompt(wf).get("prompt_id")
    if not pid:
        raise RuntimeError("queue_prompt returned no prompt_id")
    hist = wait_for_completion(pid)
    vid = find_first_mp4_in_history(hist, COMFYUI_OUTPUT_DIR)
    return hist, vid


def append_frames_to_dir(image_paths: list[str], dest_dir: str, start_index: int) -> int:
    """Copy images to dest_dir as frame_{start_index+i:05d}.{ext} preserving order."""
    os.makedirs(dest_dir, exist_ok=True)
    for i, src in enumerate(image_paths):
        ext = os.path.splitext(src)[1].lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        dst = os.path.join(dest_dir, f"frame_{start_index + i:05d}{ext}")
        shutil.copy2(src, dst)
    return start_index + len(image_paths)


def main() -> None:
    p = argparse.ArgumentParser(description="Post-editor: upscale → RIFE 2x → MP4 (batched).")
    p.add_argument(
        "session",
        nargs="?",
        default=None,
        help="Director session directory (contains seg_XX/ and manifest.json)",
    )
    p.add_argument(
        "--frames-dir",
        default=None,
        help="Skip session gather; folder of images (.png/.jpg/…), any filenames, natural sort",
    )
    p.add_argument(
        "--frames-source",
        choices=("auto", "seg_all", "last", "all"),
        default="auto",
        help="Session frames: auto=seg_all if present else highest seg_N only; "
        "seg_all=only seg_all/; last=highest seg_N; all=concat all seg_N (can duplicate)",
    )
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Frames per Comfy batch (default: 16)")
    p.add_argument(
        "--source-fps",
        type=float,
        default=16.0,
        help="FPS of the *input* frame sequence (before RIFE). Output defaults to source_fps*2.",
    )
    p.add_argument(
        "--combine-fps",
        type=float,
        default=None,
        help="VHS combine frame_rate. Default: source_fps * 2 (keeps real-time duration after 2x frames).",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Working folder for postedit/ (default: <session>/postedit or parent of --frames-dir)",
    )
    p.add_argument(
        "--skip-scale",
        action="store_true",
        help="Skip upscale; use flattened source frames as RIFE input",
    )
    p.add_argument("--skip-fill", action="store_true", help="Skip RIFE; combine from scaled only")
    p.add_argument(
        "--max-output-edge",
        type=int,
        default=DEFAULT_MAX_OUTPUT_EDGE,
        help=f"Max image side (px) after 4× upscale + ImageScaleBy (default {DEFAULT_MAX_OUTPUT_EDGE} ≈ 4K UHD).",
    )
    p.add_argument(
        "--upscale-model-factor",
        type=float,
        default=4.0,
        help="Assumed upscale from ImageUpscaleWithModel (default 4 for 4x_*.pth). Used to compute ImageScaleBy.",
    )
    p.add_argument(
        "--scale-by",
        type=float,
        default=None,
        metavar="S",
        help="ImageScaleBy factor after the 4× model (0–1]. Overrides --max-output-edge auto cap when set.",
    )
    p.add_argument(
        "--combine-max-pixels",
        type=int,
        default=DEFAULT_COMBINE_MAX_PIXELS,
        metavar="N",
        help="Final VHS step: max width×height for LoadImagesFromFolderKJ (0 = no cap). "
        f"Default {DEFAULT_COMBINE_MAX_PIXELS} (QHD 2560×1440 + 20%%). Reduces RAM; does not re-encode scale/RIFE.",
    )
    args = p.parse_args()

    if args.skip_scale and args.scale_by is not None:
        print(
            "WARNING: --scale-by is ignored when --skip-scale (no upscale/ImageScaleBy step).",
            file=sys.stderr,
        )

    if not args.frames_dir and not args.session:
        p.error("Provide session directory or --frames-dir")

    if not check_server():
        print(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}", file=sys.stderr)
        sys.exit(1)

    for wf_path in (WORKFLOW_SCALE, WORKFLOW_FILL, WORKFLOW_COMBINE):
        if not os.path.isfile(wf_path):
            print(f"ERROR: Missing workflow {wf_path}", file=sys.stderr)
            sys.exit(1)

    # Resolve frame list + work dirs
    if args.frames_dir:
        session_home = os.path.abspath(os.path.join(args.frames_dir, ".."))
        if args.work_dir:
            work_root = os.path.abspath(args.work_dir)
        else:
            work_root = os.path.join(session_home, "postedit_direct_frames")
        raw_frames = list_image_paths_in_folder(args.frames_dir)
        if not raw_frames:
            print(f"ERROR: No image files (.png/.jpg/…) in {args.frames_dir}", file=sys.stderr)
            sys.exit(1)
        material_dir = os.path.join(work_root, "00_source_flat")
        os.makedirs(work_root, exist_ok=True)
        materialize_sequential_folder(raw_frames, material_dir)
    else:
        session_dir = os.path.abspath(args.session)
        if not os.path.isdir(session_dir):
            print(f"ERROR: Not a directory: {session_dir}", file=sys.stderr)
            sys.exit(1)
        work_root = args.work_dir or os.path.join(session_dir, "postedit")
        os.makedirs(work_root, exist_ok=True)
        frame_paths = gather_frames_from_session(session_dir, args.frames_source)
        print(f"Collected {len(frame_paths)} frames (source={args.frames_source}).")
        material_dir = os.path.join(work_root, "00_source_flat")
        materialize_sequential_folder(frame_paths, material_dir)

    material_frames = list_image_paths_in_folder(material_dir)
    if not material_frames:
        print("ERROR: No image files after materialize.", file=sys.stderr)
        sys.exit(1)
    first = material_frames[0]
    wh = get_image_size(first)
    if not wh:
        print("ERROR: Could not read image size from first frame (cv2 needed).", file=sys.stderr)
        sys.exit(1)
    width, height = wh
    material_w, material_h = width, height
    print(f"Frame size: {width}x{height}")

    scale_by_factor = 1.0
    scale_by_manual = False
    if not args.skip_scale:
        if args.scale_by is not None:
            scale_by_factor = max(0.01, min(1.0, float(args.scale_by)))
            scale_by_manual = True
            print(
                f"ImageScaleBy factor: {scale_by_factor} "
                f"(manual --scale-by; auto cap from --max-output-edge is not used)"
            )
        else:
            scale_by_factor = compute_image_scale_by_cap_max_edge(
                width,
                height,
                upscale_model_factor=args.upscale_model_factor,
                max_long_edge=args.max_output_edge,
            )
            after_4x = max(width * args.upscale_model_factor, height * args.upscale_model_factor)
            print(
                f"ImageScaleBy factor: {scale_by_factor} "
                f"(max edge ≤ {args.max_output_edge}px; ~{args.upscale_model_factor}× model → "
                f"~{after_4x:.0f}px long edge before extra scale)"
            )

    total = len(material_frames)
    batch = max(1, args.batch)
    combine_fps = args.combine_fps if args.combine_fps is not None else (args.source_fps * 2.0)

    scale_wf = _load_workflow(WORKFLOW_SCALE)
    fill_wf = _load_workflow(WORKFLOW_FILL)
    combine_wf = _load_workflow(WORKFLOW_COMBINE)

    scaled_dir = os.path.join(work_root, "01_scaled")
    filled_dir = os.path.join(work_root, "02_rife_filled")

    # ── Scale (batched) ─────────────────────────────────────────────
    if args.skip_scale:
        print("Skipping scale — using 00_source_flat as scaled input.")
        scaled_dir = material_dir
    else:
        os.makedirs(scaled_dir, exist_ok=True)
        clear_folder_images(scaled_dir)
        idx_out = 0
        for start in range(0, total, batch):
            n = min(batch, total - start)
            print(f"\n[Scale] batch start_index={start} count={n}")
            prefix = f"postedit_scale/{time.strftime('%H%M%S')}_{start:05d}"
            _, paths = run_scale_batch(
                scale_wf,
                folder=material_dir,
                width=width,
                height=height,
                start_index=start,
                batch_size=n,
                save_prefix=prefix,
                scale_by=scale_by_factor,
            )
            if len(paths) != n:
                print(f"  WARNING: expected {n} images, got {len(paths)}")
            idx_out = append_frames_to_dir(paths, scaled_dir, idx_out)
        print(f"Scaled frames: {idx_out} → {scaled_dir}")

    # Re-read dimensions after scale (1.5 * ImageScaleBy in workflow)
    first_scaled = list_image_paths_in_folder(scaled_dir)
    if first_scaled:
        wh2 = get_image_size(first_scaled[0])
        if wh2:
            width, height = wh2
            print(f"After scale, frame size: {width}x{height}")

    scaled_total = len(list_image_paths_in_folder(scaled_dir))
    if scaled_total == 0:
        print("ERROR: No scaled frames produced.", file=sys.stderr)
        sys.exit(1)

    # ── RIFE fill (batched) ─────────────────────────────────────────
    if args.skip_fill:
        print("Skipping RIFE — combining from scaled folder.")
        filled_dir = scaled_dir
    else:
        os.makedirs(filled_dir, exist_ok=True)
        clear_folder_images(filled_dir)
        idx_out = 0
        for start in range(0, scaled_total, batch):
            n = min(batch, scaled_total - start)
            print(f"\n[Fill / RIFE] batch start_index={start} count={n}")
            prefix = f"postedit_fill/{time.strftime('%H%M%S')}_{start:05d}"
            _, paths = run_fill_batch(
                fill_wf,
                folder=scaled_dir,
                width=width,
                height=height,
                start_index=start,
                batch_size=n,
                save_prefix=prefix,
            )
            idx_out = append_frames_to_dir(paths, filled_dir, idx_out)
        print(f"Filled frames: {idx_out} → {filled_dir}")

    filled_count = len(list_image_paths_in_folder(filled_dir))
    if filled_count == 0:
        print("ERROR: No frames to combine.", file=sys.stderr)
        sys.exit(1)

    # ── Final video (optional downscale for loader only — limits RAM in LoadImagesFromFolderKJ) ──
    filled_paths = list_image_paths_in_folder(filled_dir)
    wh_fill = get_image_size(filled_paths[0]) if filled_paths else None
    if not wh_fill:
        print("ERROR: Could not read frame size from filled folder.", file=sys.stderr)
        sys.exit(1)
    combine_src_w, combine_src_h = wh_fill
    cap = int(args.combine_max_pixels)
    if cap > 0:
        combine_w, combine_h = combine_loader_dims_for_pixel_cap(
            combine_src_w,
            combine_src_h,
            max_total_pixels=cap,
        )
        if (combine_w, combine_h) != (combine_src_w, combine_src_h):
            print(
                f"Combine loader capped to ≤{cap} px² (~QHD+20%): {combine_w}x{combine_h} "
                f"(frames on disk {combine_src_w}x{combine_src_h}; scale/RIFE unchanged)"
            )
        else:
            print(f"Combine loader: {combine_w}x{combine_h} (within pixel cap {cap})")
    else:
        combine_w, combine_h = combine_src_w, combine_src_h
        print(f"Combine loader (--combine-max-pixels 0): {combine_w}x{combine_h} (no cap)")

    # ── Final video ───────────────────────────────────────────────────
    prefix = f"postedit_final/{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"\n[Combine] folder={filled_dir} frame_rate={combine_fps} prefix={prefix}")
    _, mp4_out = run_final_combine(
        combine_wf,
        folder=filled_dir,
        width=combine_w,
        height=combine_h,
        frame_rate=combine_fps,
        filename_prefix=prefix,
    )
    if not mp4_out or not os.path.isfile(mp4_out):
        print("ERROR: Could not find output MP4 in ComfyUI output.", file=sys.stderr)
        sys.exit(1)

    dest_name = "postedit_final.mp4"
    dest = os.path.join(work_root, dest_name)
    shutil.copy2(mp4_out, dest)
    print(f"\nDone. Video: {dest}")
    print(f"Also under ComfyUI output: {mp4_out}")

    report = {
        "work_root": os.path.abspath(work_root),
        "material_dir": os.path.abspath(material_dir),
        "scaled_dir": os.path.abspath(scaled_dir),
        "filled_dir": os.path.abspath(filled_dir),
        "output_video": os.path.abspath(dest),
        "comfy_output_mp4": os.path.abspath(mp4_out),
        "source_fps": args.source_fps,
        "combine_fps": combine_fps,
        "batch": batch,
        "frames_source": getattr(args, "frames_source", None),
        "scale_image_scale_by": scale_by_factor,
        "scale_by_manual": scale_by_manual,
        "max_output_edge": args.max_output_edge,
        "upscale_model_factor": args.upscale_model_factor,
        "material_input_frame_size": [material_w, material_h],
        "combine_source_frame_size": [combine_src_w, combine_src_h],
        "combine_loader_size": [combine_w, combine_h],
        "combine_max_pixels": cap if cap > 0 else 0,
        "input_frame_size": [combine_src_w, combine_src_h],
    }
    report_path = os.path.join(work_root, "postedit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")

    # Optionally merge into session manifest
    if args.session and not args.frames_dir:
        man_path = os.path.join(os.path.abspath(args.session), "manifest.json")
        if os.path.isfile(man_path):
            try:
                with open(man_path, "r", encoding="utf-8") as f:
                    man = json.load(f)
                man["post_editor"] = report
                with open(man_path, "w", encoding="utf-8") as f:
                    json.dump(man, f, indent=2, ensure_ascii=False)
                print(f"Updated manifest: {man_path}")
            except (OSError, json.JSONDecodeError) as e:
                print(f"(Could not update manifest: {e})")


if __name__ == "__main__":
    main()
