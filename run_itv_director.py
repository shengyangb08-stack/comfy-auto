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

# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ Lightning LoRA Combos (applied to both first5 and extend5 workflows)
# ═══════════════════════════════════════════════════════════════════════════════
# Combo 1: More Motion (⚠️ Rapid Video Degradation)
#   High: wan22\lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors, weight 4
#   Low:  wan22\wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors, weight 1.4
#
# Combo 2: Less Degradation (✨ Cleaner Image) — DEFAULT
#   High: wan22\Wan_2_2_I2V_A14B_HIGH_lightx2v_MoE_distill_lora_rank_64_bf16.safetensors, weight 1
#   Low:  wan22\wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors, weight 1
#
# Combo 3: Balanced (⚖️ Moderate Motion & Degradation)
#   High: wan22\lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors, weight 3
#   Low:  wan22\lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors, weight 1.5
#
# Workflow uses 3ksampler path by default.
# ═══════════════════════════════════════════════════════════════════════════════

LORA_PREFIX = "wan22\\"

LIGHTNING_COMBOS = {
    "1": {  # More Motion
        "high_lora": f"{LORA_PREFIX}lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        "high_weight": 4.0,
        "low_lora": f"{LORA_PREFIX}wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "low_weight": 1.4,
    },
    "2": {  # Less Degradation (default)
        "high_lora": f"{LORA_PREFIX}Wan_2_2_I2V_A14B_HIGH_lightx2v_MoE_distill_lora_rank_64_bf16.safetensors",
        "high_weight": 1.0,
        "low_lora": f"{LORA_PREFIX}wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "low_weight": 1.0,
    },
    "3": {  # Balanced
        "high_lora": f"{LORA_PREFIX}lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        "high_weight": 3.0,
        "low_lora": f"{LORA_PREFIX}lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        "low_weight": 1.5,
    },
}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
COMFYUI_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "ComfyUI", "output")
COMFYUI_INPUT_DIR = os.path.join(_PROJECT_ROOT, "ComfyUI", "input")

WORKFLOW_FIRST5 = os.path.join(_SCRIPT_DIR, "api_workflow", "ITV_single_first5_API.json")
WORKFLOW_EXTEND5 = os.path.join(_SCRIPT_DIR, "api_workflow", "ITV_single_extend5_API.json")
LORA_METADATA = os.path.join(_PROJECT_ROOT, "ComfyUI", "models", "loras", "wan22", "lora_metadata.json")

# Lightning LoRA nodes (same IDs in both workflows)
NODE_LIGHTNING_HIGH = "368:359"
NODE_LIGHTNING_LOW = "368:365"

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


def _load_lora_metadata() -> list[dict]:
    """Load LoRA metadata from wan22/lora_metadata.json. Returns list of lora entries."""
    if not os.path.isfile(LORA_METADATA):
        return []
    try:
        with open(LORA_METADATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("loras", [])
    except (json.JSONDecodeError, OSError):
        return []


# TODO: Change how we decide what content LoRAs to load (e.g. by prompt keywords, script, or CLI arg).
def _get_default_content_loras() -> list[dict]:
    """Return all content LoRAs from metadata to load."""
    loras = _load_lora_metadata()
    result: list[dict] = []
    for entry in loras:
        file_name = entry.get("file", "")
        if not file_name:
            continue
        if not file_name.startswith("wan22"):
            file_name = f"{LORA_PREFIX}{file_name}"
        high = entry.get("high_weight")
        strength = 1.0
        if isinstance(high, (list, tuple)) and len(high) >= 2:
            strength = (high[0] + high[1]) / 2
        elif isinstance(high, (int, float)):
            strength = float(high)
        result.append({"file": file_name, "strength": strength})
    return result


def _apply_content_lora(workflow: dict, lora_entries: list[dict]) -> None:
    """Apply content LoRAs to all Power Lora Loader nodes. Same LoRAs on lightning high/low + no-lightning."""
    if not lora_entries:
        return
    count = 0
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "Power Lora Loader (rgthree)":
            inputs = node.setdefault("inputs", {})
            for i, entry in enumerate(lora_entries):
                inputs[f"lora_{i + 1}"] = {
                    "on": True,
                    "lora": entry["file"],
                    "strength": entry.get("strength", 1.0),
                    "strengthTwo": None,
                }
            count += 1
    if count:
        files = [e["file"] for e in lora_entries]
        print("Content LoRA(s) applied to %d Power Lora Loader(s): %s" % (count, files))


# TODO: Remove this default insert - make trigger word insertion conditional or configurable.
def _prepend_lora_trigger_to_prompt(prompt: str) -> str:
    """Prepend trigger words of all loaded content LoRAs at the very front of the prompt. By default always applied."""
    loras = _load_lora_metadata()
    if not loras:
        return prompt
    parts: list[str] = []
    for entry in loras:
        trigger = entry.get("trigger_word")
        if not trigger:
            continue
        if isinstance(trigger, list):
            part = next((str(t).strip() for t in trigger if t), "")
        else:
            part = str(trigger).strip()
        if part:
            parts.append(part)
    if parts:
        prefix = " ".join(parts)
        print("LoRA trigger prepended (%d chars): %s" % (
            len(prefix), prefix[:150] + "..." if len(prefix) > 150 else prefix))
        return prefix + " " + prompt
    return prompt


def _print_modified_prompt(label: str, prompt: str, max_len: int = 400) -> None:
    """Print the modified prompt for debugging. Truncate if very long."""
    if len(prompt) <= max_len:
        print("%s prompt: %s" % (label, prompt))
    else:
        print("%s prompt (%d chars): %s..." % (label, len(prompt), prompt[:max_len]))


def _apply_lightning_combo(workflow: dict, combo: str) -> None:
    """Apply lightning combo to workflow. combo: '1'|'2'|'3'. Uses 3ksampler path."""
    cfg = LIGHTNING_COMBOS.get(combo, LIGHTNING_COMBOS["2"])
    workflow[NODE_LIGHTNING_HIGH]["inputs"]["lora_name"] = cfg["high_lora"]
    workflow[NODE_LIGHTNING_HIGH]["inputs"]["strength_model"] = cfg["high_weight"]
    workflow[NODE_LIGHTNING_LOW]["inputs"]["lora_name"] = cfg["low_lora"]
    workflow[NODE_LIGHTNING_LOW]["inputs"]["strength_model"] = cfg["low_weight"]


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
    lightning_combo: str = "2",
) -> dict:
    """Run ITV_single_first5 workflow. Source image = anchor for first 5s.

    lightning_combo: '1'=more motion, '2'=less degradation (default), '3'=balanced.
    """
    if not check_server():
        raise RuntimeError(f"ComfyUI server not reachable at {COMFYUI_URL}")

    with open(WORKFLOW_FIRST5, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    _apply_lightning_combo(workflow, lightning_combo)
    _apply_content_lora(workflow, _get_default_content_loras())

    prompt = _prepend_lora_trigger_to_prompt(prompt)
    _print_modified_prompt("First5", prompt)
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

    print("First5 overrides: source_image=%s, seed=%s, lightning=%s" % (source_image, seed, lightning_combo))
    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {result}")
    history = wait_for_completion(prompt_id)
    return (history, prompt)


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
    lightning_combo: str = "2",
) -> dict:
    """Run ITV_single_extend5 workflow.

    - anchor_image: original source image of the run (for anchor latent)
    - prev_images_folder: absolute path to folder with previous segment's images
    - prev_latent_basename: filename of .latent in ComfyUI/input/ (copy there before calling)
    - width, height: resolution for LoadImagesFromFolderKJ and FindPerfectResolution.
      Must match the previous segment's output. Pass from get_image_size(prev_first_image).
    - lightning_combo: '1'|'2'|'3'.
    """
    if not check_server():
        raise RuntimeError(f"ComfyUI server not reachable at {COMFYUI_URL}")

    with open(WORKFLOW_EXTEND5, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    _apply_lightning_combo(workflow, lightning_combo)
    _apply_content_lora(workflow, _get_default_content_loras())

    prompt = _prepend_lora_trigger_to_prompt(prompt)
    _print_modified_prompt("Extend5", prompt)
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
    print("Extend5 overrides: anchor=%s, prev_folder=%s, prev_latent=%s,%s seed=%s, lightning=%s" % (
        anchor_image, prev_images_folder, prev_latent_basename, res_str, seed, lightning_combo))
    result = queue_prompt(workflow)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {result}")
    history = wait_for_completion(prompt_id)
    return (history, prompt)
