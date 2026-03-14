"""
Trigger ITV_single workflow via ComfyUI API.

Supports optional prev_latent for temporal continuity between segments.
Segment 1: no prev latent. Segments 2+: use prev latent from previous run.

Usage:
    python run_itv_pass1.py <image_filename> [--steps N] [--cfg F] [--prompt TEXT] [--seed S]
    python run_itv_pass1.py <image_filename> [--prev-latent FILENAME]  # for segments 2+

The image must already exist in ComfyUI's input folder:
    ComfyUI/input/<image_filename>

For --prev-latent: the .latent file must exist in ComfyUI/input/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

COMFYUI_URL = "http://127.0.0.1:8188"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
WORKFLOW_PATH = os.path.join(_PROJECT_ROOT, "ComfyUI", "user", "default", "workflows", "ITV_single_API.json")
COMFYUI_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "ComfyUI", "output")
COMFYUI_INPUT_DIR = os.path.join(_PROJECT_ROOT, "ComfyUI", "input")

NODE_IDS = {
    "image":           "62",       # LoadImage
    "total_steps":     "195",      # INTConstant — total step (3-ksampler path)
    "cfg":             "314",      # FloatConstant — cfg for first ksampler (no lightning)
    "pos_prompt":      "562:552",  # CLIPTextEncode — positive prompt
    "seed":            "562:550",  # Seed (rgthree)
    "filename_prefix": "512",      # VHS_VideoCombine
    "save_latent_prefix": "771",   # SaveLatent
    "load_latent":     "773",      # LoadLatent
    "wan_prev_samples": "562:555",  # WanImageToVideoSVIPro (has prev_samples)
}


def check_server():
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(COMFYUI_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8188
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        sock.connect((host, port))
        sock.close()
    except (OSError, socket.timeout):
        return False
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
        return True
    except Exception:
        return False


def queue_prompt(prompt):
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ComfyUI rejected prompt (HTTP {exc.code}): {body[:500]}"
        ) from exc


def get_history(prompt_id):
    try:
        resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}")
        return json.loads(resp.read())
    except Exception:
        return {}


def wait_for_completion(prompt_id, poll_interval=3.0):
    print(f"Waiting for prompt {prompt_id} to complete...")
    elapsed = 0
    while True:
        history = get_history(prompt_id)
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("completed", False) or status.get("status_str") == "success":
                print(f"Completed in ~{elapsed:.0f}s")
                return history[prompt_id]
            if status.get("status_str") == "error":
                print(f"FAILED after ~{elapsed:.0f}s")
                msgs = status.get("messages", [])
                for m in msgs:
                    print(f"  {m}")
                return history[prompt_id]
        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed % 15 < poll_interval:
            print(f"  ...still running ({elapsed:.0f}s)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_itv_pass1",
        description="Trigger ITV_single workflow via ComfyUI API.",
    )
    p.add_argument("image", help="Image filename (must exist in ComfyUI/input/).")
    p.add_argument("--steps", type=int, default=None,
                   help="Total sampling steps (default: workflow value, currently 7).")
    p.add_argument("--cfg", type=float, default=None,
                   help="CFG guidance scale for the first ksampler (default: workflow value, currently 4.0).")
    p.add_argument("--prompt", type=str, default=None,
                   help="Positive prompt text (default: workflow value).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for generation (default: workflow value).")
    p.add_argument("--prev-latent", type=str, default=None,
                   help="Previous latent filename in ComfyUI/input/ (for segments 2+).")
    return p


def apply_overrides(workflow: dict, *, image: str, steps: int | None = None,
                    cfg: float | None = None, prompt: str | None = None,
                    seed: int | None = None,
                    filename_prefix: str | None = None,
                    prev_latent_path: str | None = None,
                    latent_filename_prefix: str | None = None) -> list[str]:
    """Override workflow parameters in-memory. Returns a log of changes."""
    changes: list[str] = []

    old_image = workflow[NODE_IDS["image"]]["inputs"]["image"]
    workflow[NODE_IDS["image"]]["inputs"]["image"] = image
    changes.append(f"  image: {old_image} -> {image}")

    if steps is not None:
        old = workflow[NODE_IDS["total_steps"]]["inputs"]["value"]
        workflow[NODE_IDS["total_steps"]]["inputs"]["value"] = steps
        changes.append(f"  steps: {old} -> {steps}")

    if cfg is not None:
        old = workflow[NODE_IDS["cfg"]]["inputs"]["value"]
        workflow[NODE_IDS["cfg"]]["inputs"]["value"] = cfg
        changes.append(f"  cfg:   {old} -> {cfg}")

    if prompt is not None:
        old = workflow[NODE_IDS["pos_prompt"]]["inputs"]["text"]
        workflow[NODE_IDS["pos_prompt"]]["inputs"]["text"] = prompt
        preview = (prompt[:60] + "...") if len(prompt) > 60 else prompt
        changes.append(f"  prompt: \"{preview}\"")

    if seed is not None:
        old = workflow[NODE_IDS["seed"]]["inputs"]["seed"]
        workflow[NODE_IDS["seed"]]["inputs"]["seed"] = seed
        changes.append(f"  seed:  {old} -> {seed}")

    if filename_prefix is not None:
        old = workflow[NODE_IDS["filename_prefix"]]["inputs"]["filename_prefix"]
        workflow[NODE_IDS["filename_prefix"]]["inputs"]["filename_prefix"] = filename_prefix
        changes.append(f"  filename_prefix: {old} -> {filename_prefix}")

    if latent_filename_prefix is not None:
        workflow[NODE_IDS["save_latent_prefix"]]["inputs"]["filename_prefix"] = latent_filename_prefix
        changes.append(f"  latent_prefix: {latent_filename_prefix}")

    # prev_samples: segment 1 = none, segment 2+ = load from file
    wan_node = workflow[NODE_IDS["wan_prev_samples"]]
    if prev_latent_path is None:
        # Segment 1: remove prev_samples so WanImageToVideoSVIPro runs without it
        wan_node["inputs"].pop("prev_samples", None)
        changes.append("  prev_samples: removed (segment 1)")
    else:
        # Segment 2+: set LoadLatent to load the prev latent file
        latent_basename = os.path.basename(prev_latent_path)
        workflow[NODE_IDS["load_latent"]]["inputs"]["latent"] = latent_basename
        changes.append(f"  prev_latent: {latent_basename}")

    return changes


def extract_output_files(history: dict) -> list[dict]:
    """Pull every output file entry from a completed history record."""
    files: list[dict] = []
    for _node_id, node_out in history.get("outputs", {}).items():
        for _key, items in node_out.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "filename" in item:
                        files.append(item)
    return files


def extract_saved_latent(history: dict) -> str | None:
    """Return the full path to the saved .latent file, or None if not found."""
    for _node_id, node_out in history.get("outputs", {}).items():
        # SaveLatent returns {"ui": {"latents": [{"filename", "subfolder", "type"}]}}
        ui = node_out.get("ui", {}) if isinstance(node_out, dict) else {}
        latents = ui.get("latents", node_out.get("latents", []))
        if isinstance(latents, list):
            for item in latents:
                if isinstance(item, dict) and "filename" in item:
                    fname = item["filename"]
                    subfolder = item.get("subfolder", "")
                    if subfolder:
                        return os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname)
                    return os.path.join(COMFYUI_OUTPUT_DIR, fname)
    return None


def run_workflow(
    image: str,
    *,
    steps: int | None = None,
    cfg: float | None = None,
    prompt: str | None = None,
    seed: int | None = None,
    filename_prefix: str | None = None,
    prev_latent_path: str | None = None,
    latent_filename_prefix: str | None = None,
) -> dict:
    """Load, override, queue, and wait. Returns the history dict for this prompt.

    Raises RuntimeError on server or queue failures.
    """
    if not check_server():
        raise RuntimeError(f"ComfyUI server not reachable at {COMFYUI_URL}")

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    changes = apply_overrides(
        workflow, image=image, steps=steps, cfg=cfg,
        prompt=prompt, seed=seed, filename_prefix=filename_prefix,
        prev_latent_path=prev_latent_path,
        latent_filename_prefix=latent_filename_prefix,
    )
    print("Overrides:")
    for c in changes:
        print(c)

    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {result}")

    print(f"\nQueued prompt_id: {prompt_id}")
    history = wait_for_completion(prompt_id)
    return history


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        history = run_workflow(
            args.image,
            steps=args.steps,
            cfg=args.cfg,
            prompt=args.prompt,
            seed=args.seed,
            prev_latent_path=args.prev_latent,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    for f in extract_output_files(history):
        print(f"  Output: {f['filename']}")
    latent_path = extract_saved_latent(history)
    if latent_path:
        print(f"  Latent: {latent_path}")


if __name__ == "__main__":
    main()
