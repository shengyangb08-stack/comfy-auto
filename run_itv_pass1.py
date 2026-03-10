"""
Trigger ITV_pass1 workflow via ComfyUI API with a custom input image.

Usage:
    python run_itv_pass1.py <image_filename>

The image must already exist in ComfyUI's input folder:
    ComfyUI/input/<image_filename>

Example:
    python run_itv_pass1.py my_photo.jpg
"""

import json
import sys
import time
import urllib.request
import urllib.error

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = r"ComfyUI\user\default\workflows\ITV_pass1_API.json"
LOAD_IMAGE_NODE_ID = "62"


def check_server():
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
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_filename = sys.argv[1]

    if not check_server():
        print(f"ERROR: ComfyUI server not reachable at {COMFYUI_URL}")
        print("Start it with:  python ComfyUI\\main.py --disable-auto-launch")
        sys.exit(1)

    print(f"ComfyUI server is running at {COMFYUI_URL}")

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        prompt = json.load(f)

    old_image = prompt[LOAD_IMAGE_NODE_ID]["inputs"]["image"]
    prompt[LOAD_IMAGE_NODE_ID]["inputs"]["image"] = image_filename
    print(f"Input image: {old_image} -> {image_filename}")

    result = queue_prompt(prompt)
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"ERROR: Failed to queue prompt: {result}")
        sys.exit(1)

    print(f"Queued prompt_id: {prompt_id}")
    history = wait_for_completion(prompt_id)

    outputs = history.get("outputs", {})
    if outputs:
        print("\nOutputs:")
        for node_id, node_out in outputs.items():
            for key, items in node_out.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "filename" in item:
                            print(f"  Node {node_id}: {item['filename']}")


if __name__ == "__main__":
    main()
