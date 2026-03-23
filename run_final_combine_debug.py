"""
Debug helper: run only the Final_Combine (VHS Video Combine) step with verbose logging.

Use when post_editor seems to hang on the combine step — logs frame counts, queue state,
and elapsed time while waiting for ComfyUI.

  python run_final_combine_debug.py --folder "D:\\...\\postedit\\02_rife_filled"
  python run_final_combine_debug.py --folder "..." --scale-by 0.5

Optional --scale-by multiplies loader W×H. By default --combine-max-pixels caps total pixels (~QHD+20%)
to limit RAM (LoadImages concatenates all frames). Use --combine-max-pixels 0 to disable.

Requires ComfyUI running (same as director). Reads comfy_auto/api_workflow/Final_Combine_API.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from post_editor import (  # noqa: E402
    DEFAULT_COMBINE_MAX_PIXELS,
    NODE_COMBINE_LOAD,
    WORKFLOW_COMBINE,
    _load_workflow,
    _set_vhs_combine,
    combine_loader_dims_for_pixel_cap,
    find_first_mp4_in_history,
    list_image_paths_in_folder,
)
from run_itv_director import (  # noqa: E402
    COMFYUI_OUTPUT_DIR,
    COMFYUI_URL,
    check_server,
    get_history,
    get_image_size,
    queue_prompt,
)


def _fetch_queue() -> dict | None:
    try:
        req = urllib.request.urlopen(f"{COMFYUI_URL}/queue", timeout=15)
        return json.loads(req.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return {"_error": str(exc)}


def wait_for_completion_debug(
    prompt_id: str,
    *,
    poll_interval: float,
    log_every: float,
) -> dict:
    """Poll history; log queue + elapsed on each log_every seconds."""
    print(f"Polling history for prompt_id={prompt_id} (interval={poll_interval}s, log every {log_every}s)")
    t0 = time.monotonic()
    last_log = -log_every  # force first log at 0
    while True:
        now = time.monotonic()
        elapsed = now - t0
        if elapsed - last_log >= log_every:
            last_log = elapsed
            q = _fetch_queue()
            if q and "_error" not in q:
                run_n = len(q.get("queue_running", []) or [])
                pen_n = len(q.get("queue_pending", []) or [])
                print(
                    f"[{elapsed:9.1f}s] /queue: running={run_n} pending={pen_n}"
                )
            elif q and "_error" in q:
                print(f"[{elapsed:9.1f}s] /queue: unavailable ({q['_error']})")

        hist_entry = get_history(prompt_id)
        if prompt_id in hist_entry:
            rec = hist_entry[prompt_id]
            status = rec.get("status", {})
            if status.get("completed", False) or status.get("status_str") == "success":
                print(f"[{time.monotonic() - t0:9.1f}s] DONE status_str=success")
                return rec
            if status.get("status_str") == "error":
                print(f"[{time.monotonic() - t0:9.1f}s] FAILED")
                for m in status.get("messages", []) or []:
                    print(f"  {m}")
                return rec

        time.sleep(poll_interval)


def main() -> None:
    p = argparse.ArgumentParser(description="Final combine only (VHS) with debug logging.")
    p.add_argument(
        "--folder",
        required=True,
        help="Folder with images (.png/.jpg/… any names; natural sort). E.g. .../postedit/02_rife_filled",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subfolders (e.g. Comfy nested output paths)",
    )
    p.add_argument("--frame-rate", type=float, default=32.0, help="VHS frame_rate (default: 32)")
    p.add_argument("--width", type=int, default=None, help="Loader width (default: from first frame)")
    p.add_argument("--height", type=int, default=None, help="Loader height (default: from first frame)")
    p.add_argument(
        "--scale-by",
        type=float,
        default=None,
        metavar="S",
        help="Multiply loader W×H after --width/--height (e.g. 0.5 for half resolution). "
        "Final_Combine has no ImageScaleBy; this only scales LoadImagesFromFolderKJ crop size.",
    )
    p.add_argument(
        "--combine-max-pixels",
        type=int,
        default=DEFAULT_COMBINE_MAX_PIXELS,
        metavar="N",
        help="Max width×height for LoadImages (0 = no cap). Default: QHD+20%% to limit RAM.",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="filename_prefix for VHS output under ComfyUI output",
    )
    p.add_argument("--poll", type=float, default=2.0, help="History poll interval seconds (default: 2)")
    p.add_argument(
        "--log-every",
        type=float,
        default=10.0,
        help="Print queue + elapsed every N seconds (default: 10)",
    )
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    paths = list_image_paths_in_folder(folder, recursive=args.recursive)
    total_bytes = sum(os.path.getsize(p) for p in paths)
    n_frames = len(paths)
    print(f"COMFYUI_URL={COMFYUI_URL}")
    print(f"Folder: {folder}  (recursive={args.recursive})")
    print(f"Images: {n_frames}  (~{total_bytes / (1024 * 1024):.1f} MiB total)")
    if n_frames == 0:
        print(
            "ERROR: no .png/.jpg/.jpeg/.webp/.bmp in folder (use --recursive if files are in subfolders).",
            file=sys.stderr,
        )
        sys.exit(1)

    first = paths[0]
    wh = get_image_size(first)
    if not wh:
        print("ERROR: could not read first frame size (cv2).", file=sys.stderr)
        sys.exit(1)
    w, h = wh
    if args.width is not None:
        w = args.width
    if args.height is not None:
        h = args.height
    print(f"Loader size: {w}x{h} (from {os.path.basename(first)})")
    if args.scale_by is not None:
        s = float(args.scale_by)
        if s <= 0:
            print("ERROR: --scale-by must be > 0", file=sys.stderr)
            sys.exit(1)
        w = max(1, int(round(w * s)))
        h = max(1, int(round(h * s)))
        print(f"After --scale-by {s}: loader {w}x{h}")

    cap = int(args.combine_max_pixels)
    if cap > 0:
        w0, h0 = w, h
        w, h = combine_loader_dims_for_pixel_cap(w, h, max_total_pixels=cap)
        if (w, h) != (w0, h0):
            print(f"After combine pixel cap (≤{cap} px²): loader {w}x{h} (was {w0}x{h0})")

    if not check_server():
        print(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}", file=sys.stderr)
        sys.exit(1)

    wf = _load_workflow(WORKFLOW_COMBINE)
    prefix = args.prefix or f"debug_combine/{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Workflow: {WORKFLOW_COMBINE}")
    print(f"VHS filename_prefix: {prefix}")
    print(f"Node {NODE_COMBINE_LOAD} LoadImagesFromFolderKJ + node 1 VHS")

    t_queue = time.monotonic()
    _set_vhs_combine(
        wf,
        NODE_COMBINE_LOAD,
        folder=folder,
        width=w,
        height=h,
        frame_rate=float(args.frame_rate),
        filename_prefix=prefix,
    )
    if args.recursive:
        wf[NODE_COMBINE_LOAD]["inputs"]["include_subfolders"] = True
        print("LoadImagesFromFolderKJ: include_subfolders=True (matches --recursive)")

    print(f"Queueing prompt… ({time.monotonic() - t_queue:.2f}s prep)")
    res = queue_prompt(wf)
    pid = res.get("prompt_id")
    if not pid:
        print(f"ERROR: no prompt_id: {res}", file=sys.stderr)
        sys.exit(1)
    print(f"prompt_id={pid}")

    hist = wait_for_completion_debug(pid, poll_interval=args.poll, log_every=args.log_every)
    mp4 = find_first_mp4_in_history(hist, COMFYUI_OUTPUT_DIR)
    print(f"Output MP4 (from history): {mp4}")
    if mp4 and os.path.isfile(mp4):
        print(f"File size: {os.path.getsize(mp4) / (1024 * 1024):.2f} MiB")
    else:
        print("WARNING: mp4 path missing or not on disk yet — check ComfyUI output folder.")


if __name__ == "__main__":
    main()
