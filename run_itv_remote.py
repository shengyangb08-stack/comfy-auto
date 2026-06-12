"""
Run ITV_single_first5 on a remote RunPod ComfyUI instance (smoke test).

See comfy_auto/REMOTE_README.md for full pipeline docs.

Flow:
  1. Create session under runpod_test_sessions/
  2. SCP input image → remote ComfyUI/input/
  3. POST first5 workflow to /prompt
  4. Poll /history until done
  5. SCP all outputs back to session/outputs/

For multi-segment runs use director_remote.py instead.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from datetime import datetime
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import run_itv_director as ritv
from project_paths import get_runpod_test_sessions_root
from runpod_client import (
    comfyui_url,
    download_outputs,
    load_secrets,
    upload_input_basename,
)


def _make_session_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    session = os.path.join(get_runpod_test_sessions_root(), f"{stamp}_{suffix}")
    os.makedirs(session, exist_ok=True)
    for sub in ("inputs", "outputs", "logs"):
        os.makedirs(os.path.join(session, sub), exist_ok=True)
    return session


def run_remote_first5(
    *,
    image_path: str,
    prompt: str,
    seed: int,
    lightning_combo: str = "2",
    steps: int | None = None,
    cfg: float | None = None,
    secrets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    secrets = secrets or load_secrets()
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    session_dir = _make_session_dir()
    session_id = os.path.basename(session_dir)
    output_prefix = f"runpod_test_{session_id}"
    remote_image_name = f"runpod_test_{session_id}_{os.path.basename(image_path)}"

    manifest: dict[str, Any] = {
        "session_id": session_id,
        "session_dir": session_dir,
        "started_at": datetime.now().isoformat(),
        "image_local": os.path.abspath(image_path),
        "image_remote": remote_image_name,
        "prompt": prompt,
        "seed": seed,
        "lightning_combo": lightning_combo,
        "output_prefix": output_prefix,
    }
    with open(os.path.join(session_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Session: {session_dir}")

    with comfyui_url(secrets) as comfy_url:
        ritv.COMFYUI_URL = comfy_url
        manifest["comfyui_url"] = comfy_url

        # Step 2–3: upload image, queue first5 on pod
        upload_input_basename(
            image_path,
            remote_image_name,
            secrets=secrets,
            session_dir=session_dir,
        )

        history, final_prompt = ritv.run_first5(
            remote_image_name,
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            filename_prefix=f"{output_prefix}/final",
            latent_filename_prefix=f"{output_prefix}/latent",
            lightning_combo=lightning_combo,
        )

        with open(os.path.join(session_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Step 4–5: download all matching outputs (legacy: full SCP, not pod-only staging)
        local_out = os.path.join(session_dir, "outputs")
        downloaded = download_outputs(
            history,
            secrets=secrets,
            local_out_dir=local_out,
            output_prefix=output_prefix,
        )

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["downloaded_files"] = downloaded
    manifest["video_local"] = ritv.find_output_video(history, local_out)
    with open(os.path.join(session_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Session: {session_dir}")
    if downloaded:
        print("Downloaded:")
        for p in downloaded:
            print(f"  {p}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ITV first5 on remote RunPod ComfyUI")
    parser.add_argument("--image", required=True, help="Local source image path")
    parser.add_argument("--prompt", required=True, help="Positive prompt")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--lightning", choices=["1", "2", "3"], default="2", help="Lightning LoRA combo")
    parser.add_argument("--steps", type=int, default=None, help="Total steps override")
    parser.add_argument("--cfg", type=float, default=None, help="CFG override (no-lightning ksampler)")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**53 - 1)

    try:
        run_remote_first5(
            image_path=args.image,
            prompt=args.prompt,
            seed=seed,
            lightning_combo=args.lightning,
            steps=args.steps,
            cfg=args.cfg,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
