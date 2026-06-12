"""
SSH/SCP helpers and ComfyUI remote URL management for RunPod.

Used by director_remote.py, post_editor_remote.py, and run_itv_remote.py.

Key concepts:
  - Session id: same string locally (folder name) and on pod (director_<id> prefix).
  - ComfyUI input/output: standard folders; LoadImage paths are relative to input/.
  - HTTP vs SCP: workflows queue over HTTPS; binary assets use SCP or ssh cp on pod.

See comfy_auto/REMOTE_README.md for full layout and troubleshooting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import run_itv_director as ritv

SECRETS_PATH = os.path.join(_SCRIPT_DIR, "secrets.json")
DEFAULT_TUNNEL_LOCAL_PORT = 18188
REMOTE_COMFY_PORT = 8188


def load_secrets() -> dict[str, Any]:
    if not os.path.isfile(SECRETS_PATH):
        raise FileNotFoundError(
            f"Missing {SECRETS_PATH}. Copy secrets.example.json and fill in runpod + comfyui."
        )
    with open(SECRETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def expand_user(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def ssh_base_cmd(runpod: dict[str, Any]) -> list[str]:
    key = expand_user(runpod.get("ssh_key_path") or "~/.ssh/id_ed25519")
    port = str(runpod.get("ssh_port") or 22)
    user = runpod.get("ssh_user") or "root"
    host = runpod["ssh_host"]
    return [
        "ssh",
        "-p", port,
        "-i", key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
        f"{user}@{host}",
    ]


def scp_base_cmd(runpod: dict[str, Any]) -> list[str]:
    key = expand_user(runpod.get("ssh_key_path") or "~/.ssh/id_ed25519")
    port = str(runpod.get("ssh_port") or 22)
    return ["scp", "-P", port, "-i", key, "-o", "StrictHostKeyChecking=accept-new"]


def run_cmd(cmd: list[str], *, label: str = "", check: bool = True) -> subprocess.CompletedProcess:
    if label:
        print(label)
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def remote_comfy_paths(secrets: dict[str, Any]) -> tuple[str, str, str]:
    """Return (comfy_root, input_dir, output_dir) on the pod."""
    comfy = secrets.get("comfyui") or {}
    root = (comfy.get("remote_comfyui_path") or "/workspace/runpod-slim/ComfyUI").rstrip("/")
    return root, f"{root}/input", f"{root}/output"


# ── Session path helpers (director_<session_id> namespace) ───────────────────


def remote_session_tag(session_id: str) -> str:
    """Shared remote namespace for a director run (same id as local session folder)."""
    return f"director_{session_id}"


def remote_session_layout(secrets: dict[str, Any], session_id: str) -> dict[str, str]:
    """Remote input/output roots for one director session."""
    _root, remote_input, remote_output = remote_comfy_paths(secrets)
    tag = remote_session_tag(session_id)
    return {
        "tag": tag,
        "input_dir": f"{remote_input}/{tag}",
        "output_dir": f"{remote_output}/{tag}",
    }


def remote_input_session_dir(secrets: dict[str, Any], session_id: str, *parts: str) -> str:
    """POSIX path on the pod under ComfyUI/input/director_<session_id>/..."""
    layout = remote_session_layout(secrets, session_id)
    bits = [layout["input_dir"], *parts]
    return "/".join(bits)


def remote_output_session_dir(secrets: dict[str, Any], session_id: str, *parts: str) -> str:
    """POSIX path on the pod under ComfyUI/output/director_<session_id>/..."""
    layout = remote_session_layout(secrets, session_id)
    bits = [layout["output_dir"], *parts]
    return "/".join(bits)


def remote_input_basename(session_id: str, name: str) -> str:
    """Path relative to ComfyUI/input/ for LoadImage / LoadLatent nodes."""
    return f"{remote_session_tag(session_id)}/{name}"


def remote_segment_output_prefix(session_id: str, seg_label: str) -> str:
    """Workflow filename_prefix under ComfyUI/output/."""
    return f"{remote_session_tag(session_id)}/{seg_label}"


def remote_prepare_session(secrets: dict[str, Any], session_id: str) -> dict[str, str]:
    """Create remote input/output session folders on the pod."""
    layout = remote_session_layout(secrets, session_id)
    ssh_mkdir(layout["input_dir"], secrets=secrets)
    ssh_mkdir(layout["output_dir"], secrets=secrets)
    print(f"  Remote session: {layout['tag']}")
    print(f"    input:  {layout['input_dir']}")
    print(f"    output: {layout['output_dir']}")
    return layout


def ssh_run(
    secrets: dict[str, Any],
    remote_cmd: str,
    *,
    label: str = "",
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    runpod = secrets["runpod"]
    cmd = ssh_base_cmd(runpod) + [remote_cmd]
    if label:
        print(label)
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=capture_output, text=capture_output)


def remote_cp(secrets: dict[str, Any], src: str, dst: str, *, label: str = "") -> None:
    ssh_run(secrets, f"cp '{src}' '{dst}'", label=label or f"Remote cp {src} -> {dst}")


def download_remote_file(
    remote_path: str,
    local_path: str,
    *,
    secrets: dict[str, Any],
    label: str = "",
) -> None:
    runpod = secrets["runpod"]
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    remote_src = f"{runpod['ssh_user'] or 'root'}@{runpod['ssh_host']}:{remote_path}"
    run_cmd(
        scp_base_cmd(runpod) + [remote_src, local_path],
        label=label or f"Downloading {os.path.basename(remote_path)}",
    )


def ref_to_remote_output_path(secrets: dict[str, Any], ref: dict[str, str]) -> str:
    _root, _remote_input, remote_output = remote_comfy_paths(secrets)
    sub = ref.get("subfolder") or ""
    fname = ref["filename"]
    if sub:
        return f"{remote_output}/{sub}/{fname}"
    return f"{remote_output}/{fname}"


def parse_segment_history(history: dict[str, Any], secrets: dict[str, Any]) -> dict[str, Any]:
    """
    Map ComfyUI /history outputs to absolute POSIX paths on the pod.

    Used after first5/extend5 so director_remote can stage extend inputs
    without downloading frame folders locally.
    """
    refs = history_output_refs(history)
    video_ref: dict[str, str] | None = None
    latent_ref: dict[str, str] | None = None
    image_refs: list[dict[str, str]] = []

    for ref in refs:
        fname = ref["filename"]
        if fname.endswith(".mp4"):
            video_ref = ref
        elif fname.endswith(".latent"):
            latent_ref = ref
        elif fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_refs.append(ref)

    _root, _remote_input, remote_output = remote_comfy_paths(secrets)
    pre_image_folder_remote: str | None = None
    last_image_remote: str | None = None
    if image_refs:
        last_ref = image_refs[-1]
        last_image_remote = ref_to_remote_output_path(secrets, last_ref)
        sub = last_ref.get("subfolder") or ""
        if sub:
            pre_image_folder_remote = f"{remote_output}/{sub}"
        else:
            pre_image_folder_remote = remote_output

    return {
        "video_ref": video_ref,
        "latent_ref": latent_ref,
        "image_refs": image_refs,
        "video_remote": ref_to_remote_output_path(secrets, video_ref) if video_ref else None,
        "latent_remote": ref_to_remote_output_path(secrets, latent_ref) if latent_ref else None,
        "pre_image_folder_remote": pre_image_folder_remote,
        "last_image_remote": last_image_remote,
    }


def remote_list_dir_count(secrets: dict[str, Any], remote_dir: str, pattern: str = "*.png") -> int:
    """Count files matching pattern in a remote directory (non-recursive)."""
    result = ssh_run(
        secrets,
        f"ls -1 '{remote_dir}'/{pattern} 2>/dev/null | wc -l",
        capture_output=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def remote_materialize_sequential_folder(
    secrets: dict[str, Any],
    source_dir: str,
    dest_dir: str,
    *,
    label: str = "",
) -> int:
    """
    On the pod: copy source_dir images → dest_dir/frame_00000.png, … (sort -V).

    LoadImagesFromFolderKJ expects sequential frame_* names; director pre_image
    uses pre_00001_.png etc., so post-editor materializes a flat folder first.
    """
    cmd = (
        f"mkdir -p '{dest_dir}' && rm -f '{dest_dir}'/*.png '{dest_dir}'/*.jpg && "
        f"i=0; for f in $(ls -1 '{source_dir}'/*.* 2>/dev/null | sort -V); do "
        f"case \"$f\" in *.png|*.jpg|*.jpeg|*.webp) "
        f"cp \"$f\" '{dest_dir}/frame_'$(printf '%05d' $i)'.png'; "
        f"i=$((i+1));; esac; done; echo $i"
    )
    result = ssh_run(
        secrets,
        cmd,
        label=label or f"Materialize frames -> {dest_dir}",
        capture_output=True,
    )
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return remote_list_dir_count(secrets, dest_dir)


def remote_clear_images(secrets: dict[str, Any], remote_dir: str) -> None:
    ssh_run(
        secrets,
        f"mkdir -p '{remote_dir}' && rm -f '{remote_dir}'/*.png '{remote_dir}'/*.jpg "
        f"'{remote_dir}'/*.jpeg '{remote_dir}'/*.webp",
        label="",
    )


def _natural_sort_key(basename: str) -> tuple:
    import re
    parts = re.split(r"(\d+)", basename)
    out: list = []
    for x in parts:
        if x.isdigit():
            out.append(int(x))
        else:
            out.append(x.lower())
    return tuple(out)


def remote_collect_history_images(
    history: dict[str, Any],
    secrets: dict[str, Any],
    dest_dir: str,
    start_index: int,
    *,
    label: str = "",
) -> int:
    """
    After a Scale_Up or RIFE batch: cp ComfyUI output PNGs into dest_dir/frame_NNNNN.png.

    Runs on pod via ssh cp (no download to local machine).
    """
    refs = history_output_refs(history)
    image_refs = [
        r for r in refs
        if r["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    image_refs.sort(key=lambda r: _natural_sort_key(r["filename"]))
    ssh_mkdir(dest_dir, secrets=secrets)
    for i, ref in enumerate(image_refs):
        src = ref_to_remote_output_path(secrets, ref)
        dst = f"{dest_dir}/frame_{start_index + i:05d}.png"
        remote_cp(secrets, src, dst, label=label if i == 0 else "")
    return start_index + len(image_refs)


def remote_video_from_history(history: dict[str, Any], secrets: dict[str, Any]) -> str | None:
    for ref in history_output_refs(history):
        if ref["filename"].endswith(".mp4"):
            return ref_to_remote_output_path(secrets, ref)
    return None


def upload_session_input(
    local_path: str,
    dest_name: str,
    *,
    secrets: dict[str, Any],
    session_id: str,
    session_dir: str | None = None,
    label: str = "",
) -> str:
    """Upload into input/director_<session_id>/<dest_name>. Returns LoadImage-relative path."""
    layout = remote_session_layout(secrets, session_id)
    ssh_mkdir(layout["input_dir"], secrets=secrets)
    if session_dir:
        inputs_dir = os.path.join(session_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        import shutil
        local_copy = os.path.join(inputs_dir, dest_name.replace("/", "_"))
        shutil.copy2(local_path, local_copy)
        upload_path = local_copy
    else:
        upload_path = local_path
    remote_path = f"{layout['input_dir']}/{dest_name}"
    upload_file(upload_path, remote_path, secrets=secrets, label=label)
    return remote_input_basename(session_id, dest_name)


@contextmanager
def comfyui_url(secrets: dict[str, Any]):
    """Yield ComfyUI base URL; start SSH tunnel when remote_url is not set."""
    comfy = secrets.get("comfyui") or {}
    remote_url = (comfy.get("remote_url") or "").strip().rstrip("/")
    runpod = secrets.get("runpod") or {}
    tunnel_proc: subprocess.Popen | None = None

    if remote_url:
        ritv.COMFYUI_URL = remote_url
        if not ritv.check_server():
            raise RuntimeError(f"ComfyUI not reachable at {remote_url}")
        print(f"ComfyUI reachable at {remote_url}")
        yield remote_url
        return

    if not runpod.get("ssh_host"):
        raise RuntimeError("Set comfyui.remote_url or runpod.ssh_host in secrets.json")

    local_port = DEFAULT_TUNNEL_LOCAL_PORT
    key = expand_user(runpod.get("ssh_key_path") or "~/.ssh/id_ed25519")
    port = str(runpod.get("ssh_port") or 22)
    user = runpod.get("ssh_user") or "root"
    host = runpod["ssh_host"]

    tunnel_cmd = [
        "ssh", "-N",
        "-L", f"{local_port}:127.0.0.1:{REMOTE_COMFY_PORT}",
        "-p", port, "-i", key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=yes",
        f"{user}@{host}",
    ]
    print(f"Starting SSH tunnel localhost:{local_port} -> pod:{REMOTE_COMFY_PORT}")
    print("  $", " ".join(tunnel_cmd))
    tunnel_proc = subprocess.Popen(tunnel_cmd)
    try:
        url = f"http://127.0.0.1:{local_port}"
        for _ in range(30):
            ritv.COMFYUI_URL = url
            if ritv.check_server():
                print(f"ComfyUI reachable at {url}")
                yield url
                return
            time.sleep(1)
        raise RuntimeError(
            f"ComfyUI not reachable via SSH tunnel at {url}. "
            "Is ComfyUI running on the pod? Or set comfyui.remote_url in secrets.json."
        )
    finally:
        ritv.COMFYUI_URL = "http://127.0.0.1:8188"
        if tunnel_proc is not None:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()


def ssh_mkdir(remote_dir: str, *, secrets: dict[str, Any]) -> None:
    runpod = secrets["runpod"]
    run_cmd(
        ssh_base_cmd(runpod) + [f"mkdir -p '{remote_dir}'"],
        label=f"mkdir {remote_dir}",
    )


def upload_file(
    local_path: str,
    remote_path: str,
    *,
    secrets: dict[str, Any],
    label: str = "",
) -> None:
    runpod = secrets["runpod"]
    remote_parent = remote_path.rsplit("/", 1)[0]
    if remote_parent:
        ssh_mkdir(remote_parent, secrets=secrets)
    remote_dest = f"{runpod['ssh_user'] or 'root'}@{runpod['ssh_host']}:{remote_path}"
    run_cmd(
        scp_base_cmd(runpod) + [local_path, remote_dest],
        label=label or f"Upload {os.path.basename(local_path)} -> {remote_path}",
    )


def upload_input_basename(
    local_path: str,
    remote_basename: str,
    *,
    secrets: dict[str, Any],
    session_dir: str | None = None,
    label: str = "",
) -> str:
    """Upload to remote ComfyUI/input/<basename>. Returns basename for LoadImage."""
    _root, remote_input, _ = remote_comfy_paths(secrets)
    if session_dir:
        inputs_dir = os.path.join(session_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        import shutil
        local_copy = os.path.join(inputs_dir, remote_basename)
        shutil.copy2(local_path, local_copy)
        upload_path = local_copy
    else:
        upload_path = local_path
    remote_path = f"{remote_input}/{remote_basename}"
    upload_file(upload_path, remote_path, secrets=secrets, label=label)
    return remote_basename


def upload_folder(
    local_dir: str,
    remote_dir: str,
    *,
    secrets: dict[str, Any],
    label: str = "",
) -> None:
    """Recursively upload a local folder to an absolute path on the pod."""
    runpod = secrets["runpod"]
    ssh_mkdir(remote_dir, secrets=secrets)
    remote_parent = remote_dir.rsplit("/", 1)[0]
    folder_name = remote_dir.rsplit("/", 1)[-1]
    remote_dest = f"{runpod['ssh_user'] or 'root'}@{runpod['ssh_host']}:{remote_parent}/"
    run_cmd(
        scp_base_cmd(runpod) + ["-r", local_dir, remote_dest],
        label=label or f"Upload folder -> {remote_dir}",
    )
    # scp -r local_dir remote_parent/ creates remote_parent/<basename(local_dir)>
    # If names differ, rename on remote
    local_name = os.path.basename(local_dir.rstrip("/\\"))
    if local_name != folder_name:
        run_cmd(
            ssh_base_cmd(runpod) + [
                f"rm -rf '{remote_dir}' && mv '{remote_parent}/{local_name}' '{remote_dir}'"
            ],
            label=f"Rename remote folder to {folder_name}",
        )


def history_output_refs(history: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(fname: str, subfolder: str = "", kind: str = "file") -> None:
        key = (fname, subfolder)
        if not fname or key in seen:
            return
        seen.add(key)
        refs.append({"filename": fname, "subfolder": subfolder or "", "type": kind})

    for node_out in history.get("outputs", {}).values():
        if not isinstance(node_out, dict):
            continue
        ui = node_out.get("ui") or {}
        for item in ui.get("images") or []:
            if isinstance(item, dict):
                add(item.get("filename", ""), item.get("subfolder", ""), "image")
        for item in ui.get("latents") or []:
            if isinstance(item, dict):
                add(item.get("filename", ""), item.get("subfolder", ""), "latent")
        for item in ui.get("gifs") or []:
            if isinstance(item, dict):
                add(item.get("filename", ""), item.get("subfolder", ""), "video")
        for _key, items in node_out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                fname = item.get("filename", "")
                if fname.endswith((".mp4", ".webm", ".png", ".jpg", ".jpeg", ".latent")):
                    add(fname, item.get("subfolder", ""), "output")

    return refs


def download_outputs(
    history: dict[str, Any],
    *,
    secrets: dict[str, Any],
    local_out_dir: str,
    output_prefix: str,
) -> list[str]:
    """SCP workflow outputs from remote ComfyUI/output into local_out_dir."""
    _remote_root, _remote_input, remote_output = remote_comfy_paths(secrets)
    runpod = secrets["runpod"]
    scp = scp_base_cmd(runpod)
    ssh = ssh_base_cmd(runpod)
    os.makedirs(local_out_dir, exist_ok=True)
    downloaded: list[str] = []

    refs = history_output_refs(history)
    prefix_token = output_prefix.split("/")[0] if "/" in output_prefix else output_prefix

    def _matches_prefix(rel: str) -> bool:
        # ComfyUI may write either subfolder/prefix/... or flat prefix_NNNNN.ext
        return (
            rel == prefix_token
            or rel.startswith(f"{prefix_token}/")
            or rel.startswith(f"{prefix_token}_")
        )

    if not refs:
        print("No output refs in history; listing remote output prefix via SSH")
        list_cmd = ssh + [f"ls -la {remote_output}/{prefix_token} 2>/dev/null || ls -la {remote_output}"]
        subprocess.run(list_cmd, check=False)

    for ref in refs:
        fname = ref["filename"]
        sub = ref["subfolder"]
        rel = f"{sub}/{fname}" if sub else fname
        if prefix_token and not _matches_prefix(rel):
            continue
        remote_src = f"{runpod['ssh_user'] or 'root'}@{runpod['ssh_host']}:{remote_output}/{rel}"
        local_rel_dir = os.path.join(local_out_dir, sub) if sub else local_out_dir
        os.makedirs(local_rel_dir, exist_ok=True)
        local_path = os.path.join(local_rel_dir, fname)
        try:
            run_cmd(scp + [remote_src, local_path], label=f"Downloading {rel}")
            downloaded.append(local_path)
        except subprocess.CalledProcessError:
            print(f"  WARN: could not download {rel}")

    prefix_dir = prefix_token
    remote_glob = f"{remote_output}/{prefix_dir}"
    remote_src = f"{runpod['ssh_user'] or 'root'}@{runpod['ssh_host']}:{remote_glob}"
    prefix_local = os.path.join(local_out_dir, prefix_dir)
    os.makedirs(prefix_local, exist_ok=True)
    try:
        run_cmd(
            scp + ["-r", remote_src, prefix_local],
            label=f"Downloading output folder {prefix_dir}/",
        )
        for root, _dirs, files in os.walk(prefix_local):
            for f in files:
                p = os.path.join(root, f)
                if p not in downloaded:
                    downloaded.append(p)
    except subprocess.CalledProcessError:
        print(f"  WARN: could not download folder {prefix_dir}/")

    if not downloaded:
        list_cmd = ssh + [
            f"ls -1 {remote_output}/{prefix_token}_* 2>/dev/null || true"
        ]
        result = subprocess.run(list_cmd, capture_output=True, text=True, check=False)
        for line in result.stdout.strip().splitlines():
            fname = os.path.basename(line.strip())
            if not fname:
                continue
            remote_file = f"{runpod['ssh_user'] or 'root'}@{runpod['ssh_host']}:{remote_output}/{fname}"
            local_path = os.path.join(local_out_dir, fname)
            try:
                run_cmd(scp + [remote_file, local_path], label=f"Downloading {fname}")
                downloaded.append(local_path)
            except subprocess.CalledProcessError:
                print(f"  WARN: could not download {fname}")

    return downloaded
