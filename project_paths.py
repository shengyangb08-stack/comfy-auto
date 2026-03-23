"""
Only `director.py` session folders are redirected off C: by default.

- ComfyUI `input` / `output` (renders, temp files from workflows) stay under the repo on **C:**
  — whatever your launcher uses (绘世, default `ComfyUI\\input` / `ComfyUI\\output`).
- Director runs write sessions under **COMFY_PROJECTS_ROOT** (default **D:\\ComfyProjects**).

Override the drive/root:
  set COMFY_PROJECTS_ROOT=E:\\MyData
"""

from __future__ import annotations

import os

_DEFAULT_PROJECTS_ROOT = r"D:\ComfyProjects"


def get_comfy_projects_root() -> str:
    root = os.environ.get("COMFY_PROJECTS_ROOT", _DEFAULT_PROJECTS_ROOT)
    root = os.path.abspath(os.path.expanduser(os.path.expandvars(root)))
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create COMFY_PROJECTS_ROOT: {root}\n"
            f"Set COMFY_PROJECTS_ROOT to a writable path, or create the folder.\n"
            f"Original error: {exc}"
        ) from exc
    return root


def get_director_sessions_root() -> str:
    """Where director.py creates `YYYYMMDD_HHMMSS_xxxxxx` session folders."""
    p = os.path.join(get_comfy_projects_root(), "comfy_auto", "director_sessions")
    os.makedirs(p, exist_ok=True)
    return p
