"""
Remote post-editor: upscale → RIFE 2× → final MP4 on RunPod ComfyUI.

See comfy_auto/REMOTE_README.md for parameters and folder layout.

Intermediates stay on the pod under output/director_<session_id>/postedit/.
Only the polished final MP4 is downloaded locally.

Usage (after director_remote.py):
    python post_editor_remote.py path/to/director_session_YYYYMMDD_hash

    python post_editor_remote.py path/to/session --skip-scale --skip-fill   # combine test

Integrated from director_remote.py:
    python director_remote.py image.png --segments 2 --post-edit ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import run_itv_director as ritv
from post_editor import (
    DEFAULT_BATCH,
    DEFAULT_COMBINE_MAX_PIXELS,
    DEFAULT_MAX_OUTPUT_EDGE,
    NODE_COMBINE_LOAD,
    NODE_COMBINE_VHS,
    NODE_FILL_LOAD,
    NODE_FILL_SAVE,
    NODE_SCALE_IMAGESCALE,
    NODE_SCALE_LOAD,
    NODE_SCALE_SAVE,
    WORKFLOW_COMBINE,
    WORKFLOW_FILL,
    WORKFLOW_SCALE,
    _load_workflow,
    _set_save_prefix,
    _set_scale_workflow_image_scale_by,
    combine_loader_dims_for_pixel_cap,
    compute_image_scale_by_cap_max_edge,
    get_image_size,
)
from runpod_client import (
    comfyui_url,
    download_remote_file,
    load_secrets,
    remote_clear_images,
    remote_collect_history_images,
    remote_comfy_paths,
    remote_list_dir_count,
    remote_materialize_sequential_folder,
    remote_output_session_dir,
    remote_prepare_session,
    remote_session_layout,
    remote_session_tag,
    remote_video_from_history,
)


def _set_folder_loader_remote(
    workflow: dict,
    node_id: str,
    folder: str,
    width: int,
    height: int,
    start_index: int,
    image_load_cap: int,
) -> None:
    """Patch LoadImagesFromFolderKJ with POSIX paths (required on Linux pod)."""
    node = workflow[node_id]
    if node.get("class_type") != "LoadImagesFromFolderKJ":
        raise ValueError(f"Node {node_id} is not LoadImagesFromFolderKJ")
    inp = node.setdefault("inputs", {})
    inp["folder"] = folder.replace("\\", "/")
    inp["width"] = width
    inp["height"] = height
    inp["start_index"] = int(start_index)
    inp["image_load_cap"] = int(image_load_cap)
    inp.setdefault("keep_aspect_ratio", "crop")
    inp.setdefault("include_subfolders", False)


def _set_vhs_combine_remote(
    workflow: dict,
    node_id: str,
    *,
    folder: str,
    width: int,
    height: int,
    frame_rate: float,
    filename_prefix: str,
) -> None:
    _set_folder_loader_remote(workflow, node_id, folder, width, height, 0, 0)
    vhs = workflow[NODE_COMBINE_VHS]
    if vhs.get("class_type") != "VHS_VideoCombine":
        raise ValueError("Final combine: expected VHS_VideoCombine")
    inp = vhs.setdefault("inputs", {})
    inp["frame_rate"] = float(frame_rate)
    inp["filename_prefix"] = filename_prefix


def resolve_remote_source_pre_image(manifest: dict[str, Any], secrets: dict[str, Any]) -> str:
    """POSIX path to last segment pre_image folder on the pod."""
    segments = manifest.get("segments") or []
    if not segments:
        raise FileNotFoundError("Manifest has no segments")
    last = segments[-1]
    prefix = last.get("remote_output_prefix")
    if not prefix:
        session_id = manifest.get("session_id")
        if not session_id:
            raise ValueError("Manifest missing session_id and remote_output_prefix")
        seg_idx = last.get("segment", len(segments))
        seg_pad = len(str(manifest.get("settings", {}).get("segments", seg_idx)))
        prefix = f"{remote_session_tag(session_id)}/seg_{seg_idx:0{seg_pad}d}"
    _root, _rin, rout = remote_comfy_paths(secrets)
    return f"{rout}/{prefix}/pre_image"


@dataclass
class RemotePostEditorConfig:
    """Inputs for one post-edit run. session_id must match director_remote session."""

    session_id: str
    source_pre_image_remote: str  # POSIX: .../output/director_<id>/seg_N/pre_image
    local_session_dir: str | None = None  # where to save postedit_final.mp4
    batch: int = DEFAULT_BATCH  # frames per Scale_Up / RIFE API call
    source_fps: float = 16.0  # Wan pre_image frame rate before RIFE
    combine_fps: float | None = None  # default source_fps * 2 after RIFE doubling
    skip_scale: bool = False
    skip_fill: bool = False
    max_output_edge: int = DEFAULT_MAX_OUTPUT_EDGE
    upscale_model_factor: float = 4.0
    scale_by: float | None = None  # manual ImageScaleBy; None = auto from max_output_edge
    combine_max_pixels: int = DEFAULT_COMBINE_MAX_PIXELS
    local_output_name: str = "postedit_final.mp4"


@dataclass
class RemotePostEditorResult:
    remote_work_root: str
    remote_material_dir: str
    remote_scaled_dir: str
    remote_filled_dir: str
    remote_video: str
    local_video: str | None = None
    report: dict[str, Any] = field(default_factory=dict)


class RemotePostEditor:
    """Run Scale_Up → Fill_frame → Final_Combine on RunPod; download final MP4 only."""

    def __init__(self, secrets: dict[str, Any], comfy_url: str) -> None:
        self.secrets = secrets
        self.comfy_url = comfy_url
        ritv.COMFYUI_URL = comfy_url

    def run(self, config: RemotePostEditorConfig) -> RemotePostEditorResult:
        """Execute Scale_Up → Fill_frame → Final_Combine entirely on the pod."""
        if not ritv.check_server():
            raise RuntimeError(f"ComfyUI not reachable at {self.comfy_url}")

        for wf_path in (WORKFLOW_SCALE, WORKFLOW_FILL, WORKFLOW_COMBINE):
            if not os.path.isfile(wf_path):
                raise FileNotFoundError(f"Missing workflow {wf_path}")

        layout = remote_session_layout(self.secrets, config.session_id)
        remote_prepare_session(self.secrets, config.session_id)

        # Working directories — all under output/director_<session_id>/postedit/
        work_root = remote_output_session_dir(self.secrets, config.session_id, "postedit")
        material_dir = f"{work_root}/00_source_flat"
        scaled_dir = f"{work_root}/01_scaled"
        filled_dir = f"{work_root}/02_rife_filled"

        # ── Step 1: Copy pre_*.png → frame_00000.png … for LoadImagesFromFolderKJ ─
        print(f"\nPost-edit source (pod): {config.source_pre_image_remote}")
        n_frames = remote_materialize_sequential_folder(
            self.secrets,
            config.source_pre_image_remote,
            material_dir,
            label=f"Materialize source frames -> {material_dir}",
        )
        if n_frames == 0:
            raise RuntimeError(f"No frames materialized from {config.source_pre_image_remote}")
        print(f"  Materialized {n_frames} frames -> {material_dir}")

        # Probe one frame locally to get width/height for workflow nodes
        width, height = self._probe_remote_frame_size(material_dir, config)
        material_w, material_h = width, height
        print(f"Frame size: {width}x{height}")

        scale_by_factor = 1.0
        scale_by_manual = False
        if not config.skip_scale:
            if config.scale_by is not None:
                scale_by_factor = max(0.01, min(1.0, float(config.scale_by)))
                scale_by_manual = True
                print(f"ImageScaleBy factor: {scale_by_factor} (manual)")
            else:
                scale_by_factor = compute_image_scale_by_cap_max_edge(
                    width,
                    height,
                    upscale_model_factor=config.upscale_model_factor,
                    max_long_edge=config.max_output_edge,
                )
                print(f"ImageScaleBy factor: {scale_by_factor} (auto cap ≤ {config.max_output_edge}px)")

        combine_fps = config.combine_fps if config.combine_fps is not None else (config.source_fps * 2.0)
        batch = max(1, config.batch)
        tag = layout["tag"]

        scale_wf = _load_workflow(WORKFLOW_SCALE)
        fill_wf = _load_workflow(WORKFLOW_FILL)
        combine_wf = _load_workflow(WORKFLOW_COMBINE)

        # ── Step 2: Scale_Up (4× model + ImageScaleBy), batched via API ─────────
        if config.skip_scale:
            print("Skipping scale — using 00_source_flat as scaled input.")
            scaled_dir = material_dir
        else:
            remote_clear_images(self.secrets, scaled_dir)
            idx_out = 0
            for start in range(0, n_frames, batch):
                n = min(batch, n_frames - start)
                print(f"\n[Scale] batch start_index={start} count={n}")
                prefix = f"{tag}/postedit/scale/{time.strftime('%H%M%S')}_{start:05d}"
                history = self._run_scale_batch(
                    scale_wf,
                    folder=material_dir,
                    width=width,
                    height=height,
                    start_index=start,
                    batch_size=n,
                    save_prefix=prefix,
                    scale_by=scale_by_factor,
                )
                idx_out = remote_collect_history_images(
                    history,
                    self.secrets,
                    scaled_dir,
                    idx_out,
                    label="Collect scaled frames on pod",
                )
                # Batch outputs land in ComfyUI/output/<prefix>/; cp → 01_scaled/
            scaled_count = remote_list_dir_count(self.secrets, scaled_dir)
            print(f"Scaled frames on pod: {scaled_count} -> {scaled_dir}")
            if scaled_count == 0:
                raise RuntimeError("No scaled frames produced on pod")
            if not config.skip_scale:
                width, height = self._probe_remote_frame_size(scaled_dir, config)
                print(f"After scale, frame size: {width}x{height}")

        scaled_total = remote_list_dir_count(self.secrets, scaled_dir)
        if scaled_total == 0:
            raise RuntimeError("No scaled frames on pod")

        # ── Step 3: Fill_frame (RIFE 2×), batched via API ───────────────────────
        if config.skip_fill:
            print("Skipping RIFE — combining from scaled folder.")
            filled_dir = scaled_dir
        else:
            remote_clear_images(self.secrets, filled_dir)
            idx_out = 0
            for start in range(0, scaled_total, batch):
                n = min(batch, scaled_total - start)
                print(f"\n[Fill / RIFE] batch start_index={start} count={n}")
                prefix = f"{tag}/postedit/fill/{time.strftime('%H%M%S')}_{start:05d}"
                history = self._run_fill_batch(
                    fill_wf,
                    folder=scaled_dir,
                    width=width,
                    height=height,
                    start_index=start,
                    batch_size=n,
                    save_prefix=prefix,
                )
                idx_out = remote_collect_history_images(
                    history,
                    self.secrets,
                    filled_dir,
                    idx_out,
                )
            filled_count = remote_list_dir_count(self.secrets, filled_dir)
            print(f"Filled frames on pod: {filled_count} -> {filled_dir}")
            if filled_count == 0:
                raise RuntimeError("No filled frames on pod")

        filled_total = remote_list_dir_count(self.secrets, filled_dir)
        if filled_total == 0:
            raise RuntimeError("No frames to combine on pod")

        # Cap loader dims for VHS combine (reduces RAM; does not resize on-disk frames)
        combine_src_w, combine_src_h = self._probe_remote_frame_size(filled_dir, config)
        cap = int(config.combine_max_pixels)
        if cap > 0:
            combine_w, combine_h = combine_loader_dims_for_pixel_cap(
                combine_src_w,
                combine_src_h,
                max_total_pixels=cap,
            )
            if (combine_w, combine_h) != (combine_src_w, combine_src_h):
                print(
                    f"Combine loader capped: {combine_w}x{combine_h} "
                    f"(on-disk {combine_src_w}x{combine_src_h})"
                )
            else:
                print(f"Combine loader: {combine_w}x{combine_h}")
        else:
            combine_w, combine_h = combine_src_w, combine_src_h

        # ── Step 4: Final_Combine (VHS MP4) ─────────────────────────────────────
        combine_prefix = f"{tag}/postedit/final/{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[Combine] folder={filled_dir} frame_rate={combine_fps}")
        history = self._run_final_combine(
            combine_wf,
            folder=filled_dir,
            width=combine_w,
            height=combine_h,
            frame_rate=combine_fps,
            filename_prefix=combine_prefix,
        )
        video_remote = remote_video_from_history(history, self.secrets)
        if not video_remote:
            raise RuntimeError("No MP4 in ComfyUI history after final combine")

        # ── Step 5: Download only the final MP4 to local session dir ───────────
        local_video: str | None = None
        if config.local_session_dir:
            os.makedirs(config.local_session_dir, exist_ok=True)
            local_video = os.path.join(config.local_session_dir, config.local_output_name)
            download_remote_file(
                video_remote,
                local_video,
                secrets=self.secrets,
                label=f"Download postedit final -> {config.local_output_name}",
            )
            print(f"\nPost-edit video (local): {local_video}")
        print(f"Post-edit video (pod): {video_remote}")

        report = {
            "remote_work_root": work_root,
            "remote_material_dir": material_dir,
            "remote_scaled_dir": scaled_dir,
            "remote_filled_dir": filled_dir,
            "remote_video": video_remote,
            "local_video": local_video,
            "source_pre_image_remote": config.source_pre_image_remote,
            "source_fps": config.source_fps,
            "combine_fps": combine_fps,
            "batch": batch,
            "scale_image_scale_by": scale_by_factor,
            "scale_by_manual": scale_by_manual,
            "material_input_frame_size": [material_w, material_h],
            "combine_source_frame_size": [combine_src_w, combine_src_h],
            "combine_loader_size": [combine_w, combine_h],
            "skip_scale": config.skip_scale,
            "skip_fill": config.skip_fill,
        }

        return RemotePostEditorResult(
            remote_work_root=work_root,
            remote_material_dir=material_dir,
            remote_scaled_dir=scaled_dir,
            remote_filled_dir=filled_dir,
            remote_video=video_remote,
            local_video=local_video,
            report=report,
        )

    def _probe_remote_frame_size(
        self,
        remote_dir: str,
        config: RemotePostEditorConfig,
    ) -> tuple[int, int]:
        """Download one frame briefly to read dimensions."""
        tmp_dir = os.path.join(config.local_session_dir or os.getcwd(), "_postedit_probe")
        os.makedirs(tmp_dir, exist_ok=True)
        local_probe = os.path.join(tmp_dir, "probe_frame.png")
        remote_probe = f"{remote_dir}/frame_00000.png"
        download_remote_file(
            remote_probe,
            local_probe,
            secrets=self.secrets,
            label="",
        )
        wh = get_image_size(local_probe)
        if not wh:
            raise RuntimeError(f"Could not read frame size from {remote_probe}")
        return wh

    def _run_scale_batch(
        self,
        workflow_template: dict,
        *,
        folder: str,
        width: int,
        height: int,
        start_index: int,
        batch_size: int,
        save_prefix: str,
        scale_by: float,
    ) -> dict:
        wf = json.loads(json.dumps(workflow_template))
        _set_folder_loader_remote(
            wf, NODE_SCALE_LOAD, folder, width, height, start_index, batch_size
        )
        _set_scale_workflow_image_scale_by(wf, scale_by)
        _set_save_prefix(wf, NODE_SCALE_SAVE, save_prefix)
        return self._queue_and_wait(wf)

    def _run_fill_batch(
        self,
        workflow_template: dict,
        *,
        folder: str,
        width: int,
        height: int,
        start_index: int,
        batch_size: int,
        save_prefix: str,
    ) -> dict:
        wf = json.loads(json.dumps(workflow_template))
        _set_folder_loader_remote(
            wf, NODE_FILL_LOAD, folder, width, height, start_index, batch_size
        )
        _set_save_prefix(wf, NODE_FILL_SAVE, save_prefix)
        return self._queue_and_wait(wf)

    def _run_final_combine(
        self,
        workflow_template: dict,
        *,
        folder: str,
        width: int,
        height: int,
        frame_rate: float,
        filename_prefix: str,
    ) -> dict:
        wf = json.loads(json.dumps(workflow_template))
        _set_vhs_combine_remote(
            wf,
            NODE_COMBINE_LOAD,
            folder=folder,
            width=width,
            height=height,
            frame_rate=frame_rate,
            filename_prefix=filename_prefix,
        )
        return self._queue_and_wait(wf)

    def _queue_and_wait(self, workflow: dict) -> dict:
        result = ritv.queue_prompt(workflow)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Failed to queue prompt: {result}")
        history = ritv.wait_for_completion(prompt_id)
        status_str = history.get("status", {}).get("status_str", "")
        if status_str == "error":
            raise RuntimeError(f"ComfyUI execution error: {history.get('status', {})}")
        return history


def run_post_editor_remote(
    config: RemotePostEditorConfig,
    *,
    secrets: dict[str, Any] | None = None,
) -> RemotePostEditorResult:
    secrets = secrets or load_secrets()
    with comfyui_url(secrets) as url:
        editor = RemotePostEditor(secrets, url)
        return editor.run(config)


def run_from_manifest(
    session_dir: str,
    *,
    secrets: dict[str, Any] | None = None,
    **kwargs: Any,
) -> RemotePostEditorResult:
    """CLI entry: read manifest.json, resolve last segment pre_image on pod, run pipeline."""
    session_dir = os.path.abspath(session_dir)
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No manifest.json in {session_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("remote"):
        raise ValueError("Manifest is not from director_remote (remote=false)")
    session_id = manifest.get("session_id") or os.path.basename(session_dir)
    secrets = secrets or load_secrets()
    source = resolve_remote_source_pre_image(manifest, secrets)
    config = RemotePostEditorConfig(
        session_id=session_id,
        source_pre_image_remote=source,
        local_session_dir=session_dir,
        **kwargs,
    )
    result = run_post_editor_remote(config, secrets=secrets)
    manifest["post_editor"] = result.report
    manifest["postedit_final_video"] = (
        os.path.basename(result.local_video) if result.local_video else None
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Remote post-editor: upscale → RIFE → MP4 on RunPod (download final only)."
    )
    p.add_argument(
        "session",
        help="Director remote session directory (contains manifest.json)",
    )
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--source-fps", type=float, default=16.0)
    p.add_argument("--combine-fps", type=float, default=None)
    p.add_argument("--skip-scale", action="store_true")
    p.add_argument("--skip-fill", action="store_true")
    p.add_argument("--max-output-edge", type=int, default=DEFAULT_MAX_OUTPUT_EDGE)
    p.add_argument("--scale-by", type=float, default=None)
    p.add_argument("--combine-max-pixels", type=int, default=DEFAULT_COMBINE_MAX_PIXELS)
    p.add_argument(
        "--output-name",
        default="postedit_final.mp4",
        help="Local filename for downloaded final video",
    )
    return p


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    if not os.path.isdir(args.session):
        print(f"ERROR: Not a directory: {args.session}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_from_manifest(
            args.session,
            batch=args.batch,
            source_fps=args.source_fps,
            combine_fps=args.combine_fps,
            skip_scale=args.skip_scale,
            skip_fill=args.skip_fill,
            max_output_edge=args.max_output_edge,
            scale_by=args.scale_by,
            combine_max_pixels=args.combine_max_pixels,
            local_output_name=args.output_name,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    if result.local_video:
        print(f"  DONE — {result.local_video}")
    else:
        print(f"  DONE — remote {result.remote_video}")
    print("=" * 70)


if __name__ == "__main__":
    main()
