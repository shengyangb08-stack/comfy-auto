"""
Run ITV workflows for Director: first 5s and extend 5s.

Uses:
- ITV_single_first5_API.json: generate the very first 5 seconds (source image + prompt)
- ITV_single_extend5_API.json: extend previous video by 5 seconds (latent + prev images + anchor)

Outputs are saved to the session folder. The extend workflow combines video internally,
so no concatenation is needed — the final segment's output is the full 30s video.

Usage:
    Called by director.py. Can also run standalone for testing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error

COMFYUI_URL = "http://127.0.0.1:8188"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
COMFYUI_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "ComfyUI", "output")
COMFYUI_INPUT_DIR = os.path.join(_PROJECT_ROOT, "ComfyUI", "input")

WORKFLOW_FIRST5 = os.path.join(_SCRIPT_DIR, "api_workflow", "ITV_single_first5_API.json")
WORKFLOW_EXTEND5 = os.path.join(_SCRIPT_DIR, "api_workflow", "ITV_single_extend5_API.json")

# First5: single source image, no prev_samples
NODE_FIRST5 = {
    "source_image": "774",      # LoadImage — source = anchor for first 5s
    "save_image": "773",
    "save_latent": "771",
    "video_combine": "512",
    "total_steps": "195",
    "cfg": "314",
    "pos_prompt": "562:552",
    "seed": "562:550",
}

# Extend5: prev latent + prev images folder + anchor image
NODE_EXTEND5 = {
    "anchor_image": "774",
    "resize_anchor": "775",  # ImageScale - exact dimensions to match prev latent
    "load_latent": "792",
    "load_images_folder": "793",
    "save_image": "773",
    "save_latent": "771",
    "video_combine": "512",
    "total_steps": "195",
    "cfg": "314",
    "pos_prompt": "788:575",
    "prompt_string": "759",     # string_a for prompt
    "seed": "788:583",
}


def check_server() -> bool:
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


def queue_prompt(prompt: dict) -> dict:
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


def get_history(prompt_id: str) -> dict:
    try:
        resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}")
        return json.loads(resp.read())
    except Exception:
        return {}


def wait_for_completion(prompt_id: str, poll_interval: float = 3.0) -> dict:
    print(f"Waiting for prompt {prompt_id} to complete...")
    elapsed = 0.0
    while True:
        history = get_history(prompt_id)
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("completed", False) or status.get("status_str") == "success":
                print(f"Completed in ~{elapsed:.0f}s")
                return history[prompt_id]
            if status.get("status_str") == "error":
                print(f"FAILED after ~{elapsed:.0f}s")
                for m in status.get("messages", []):
                    print(f"  {m}")
                return history[prompt_id]
        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed % 15 < poll_interval:
            print(f"  ...still running ({elapsed:.0f}s)")


def _path_from_image_item(item: dict, output_dir: str) -> str | None:
    if isinstance(item, dict) and "filename" in item:
        fname = item["filename"]
        subfolder = item.get("subfolder", "")
        if subfolder:
            return os.path.join(output_dir, subfolder, fname)
        return os.path.join(output_dir, fname)
    return None


def extract_saved_latent(history: dict, output_dir: str | None = None) -> str | None:
    """Return the full path to the saved .latent file, or None if not found."""
    out = output_dir or COMFYUI_OUTPUT_DIR
    for _node_id, node_out in history.get("outputs", {}).items():
        ui = node_out.get("ui", {}) if isinstance(node_out, dict) else {}
        latents = ui.get("latents", node_out.get("latents", []))
        if isinstance(latents, list):
            for item in latents:
                if isinstance(item, dict) and "filename" in item:
                    fname = item["filename"]
                    subfolder = item.get("subfolder", "")
                    if subfolder:
                        return os.path.join(out, subfolder, fname)
                    return os.path.join(out, fname)
    return None


def extract_saved_images(history: dict, save_image_id: str = "773",
                        output_dir: str | None = None) -> list[str]:
    """Return all saved image paths from SaveImage node."""
    out = output_dir or COMFYUI_OUTPUT_DIR
    paths: list[str] = []
    node_out = history.get("outputs", {}).get(save_image_id)
    if isinstance(node_out, dict):
        ui = node_out.get("ui", {})
        images = ui.get("images", [])
        if isinstance(images, list):
            for item in images:
                p = _path_from_image_item(item, out)
                if p:
                    paths.append(p)
            if paths:
                return paths
    for _node_id, node_out in history.get("outputs", {}).items():
        ui = node_out.get("ui", {}) if isinstance(node_out, dict) else {}
        images = ui.get("images", node_out.get("images", []))
        if isinstance(images, list):
            for item in images:
                p = _path_from_image_item(item, out)
                if p:
                    paths.append(p)
            if paths:
                return paths
    return paths


def get_image_size(image_path: str) -> tuple[int, int] | None:
    """Return (width, height) of an image file. Use exact dimensions to match prev latent."""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    except Exception:
        pass
    return None


def find_output_video(history: dict, output_dir: str | None = None) -> str | None:
    """Find the first .mp4 output file from a ComfyUI history record."""
    out = output_dir or COMFYUI_OUTPUT_DIR
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
                        return os.path.join(out, subfolder, fname)
                    return os.path.join(out, fname)
    return None


def run_first5(
    source_image: str,
    *,
    prompt: str,
    seed: int,
    steps: int | None = None,
    cfg: float | None = None,
    filename_prefix: str | None = None,
    latent_filename_prefix: str | None = None,
) -> dict:
    """Run ITV_single_first5 workflow. Source image = anchor for first 5s.

    Returns the history dict. Use extract_saved_latent, extract_saved_images,
    find_output_video to get output paths.
    """
    if not check_server():
        raise RuntimeError(f"ComfyUI server not reachable at {COMFYUI_URL}")

    with open(WORKFLOW_FIRST5, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    n = NODE_FIRST5
    workflow[n["source_image"]]["inputs"]["image"] = source_image
    workflow[n["pos_prompt"]]["inputs"]["text"] = prompt
    workflow[n["seed"]]["inputs"]["seed"] = seed

    if steps is not None:
        workflow[n["total_steps"]]["inputs"]["value"] = steps
    if cfg is not None:
        workflow[n["cfg"]]["inputs"]["value"] = cfg
    if filename_prefix is not None:
        workflow[n["video_combine"]]["inputs"]["filename_prefix"] = filename_prefix
    if latent_filename_prefix is not None:
        workflow[n["save_latent"]]["inputs"]["filename_prefix"] = latent_filename_prefix

    print("First5 overrides: source_image=%s, seed=%s" % (source_image, seed))
    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {result}")
    return wait_for_completion(prompt_id)


def run_extend5(
    anchor_image: str,
    prev_images_folder: str,
    prev_latent_basename: str,
    *,
    prompt: str,
    seed: int,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    filename_prefix: str | None = None,
    latent_filename_prefix: str | None = None,
) -> dict:
    """Run ITV_single_extend5 workflow.

    - anchor_image: original source image of the run (for anchor latent)
    - prev_images_folder: absolute path to folder with previous segment's images
    - prev_latent_basename: filename of .latent in ComfyUI/input/ (copy there before calling)
    - width, height: resolution for LoadImagesFromFolderKJ and FindPerfectResolution.
      Must match the previous segment's output. Pass from get_image_size(prev_first_image).
    """
    if not check_server():
        raise RuntimeError(f"ComfyUI server not reachable at {COMFYUI_URL}")

    with open(WORKFLOW_EXTEND5, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    n = NODE_EXTEND5
    workflow[n["anchor_image"]]["inputs"]["image"] = anchor_image
    workflow[n["load_latent"]]["inputs"]["latent"] = prev_latent_basename
    workflow[n["load_images_folder"]]["inputs"]["folder"] = os.path.abspath(prev_images_folder)

    if width is not None and height is not None:
        workflow[n["resize_anchor"]]["inputs"]["width"] = width
        workflow[n["resize_anchor"]]["inputs"]["height"] = height
        workflow[n["load_images_folder"]]["inputs"]["width"] = width
        workflow[n["load_images_folder"]]["inputs"]["height"] = height

    workflow[n["pos_prompt"]]["inputs"]["text"] = prompt
    workflow[n["prompt_string"]]["inputs"]["string_a"] = prompt
    workflow[n["seed"]]["inputs"]["seed"] = seed

    if steps is not None:
        workflow[n["total_steps"]]["inputs"]["value"] = steps
    if cfg is not None:
        workflow[n["cfg"]]["inputs"]["value"] = cfg
    if filename_prefix is not None:
        workflow[n["video_combine"]]["inputs"]["filename_prefix"] = filename_prefix
    if latent_filename_prefix is not None:
        workflow[n["save_latent"]]["inputs"]["filename_prefix"] = latent_filename_prefix

    res_str = f" {width}x{height}" if (width is not None and height is not None) else ""
    print("Extend5 overrides: anchor=%s, prev_folder=%s, prev_latent=%s,%s seed=%s" % (
        anchor_image, prev_images_folder, prev_latent_basename, res_str, seed))
    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {result}")
    return wait_for_completion(prompt_id)
