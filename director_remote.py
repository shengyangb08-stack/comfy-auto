"""
Remote Director — same pipeline as director.py, workflows run on RunPod ComfyUI.

See comfy_auto/REMOTE_README.md for setup, parameters, and pod/local folder layout.

Local machine: LLM autoprompt, content check, session manifest.
Remote pod: ComfyUI first5 / extend5 via HTTP API; intermediates stay on pod.

Intermediate segment frames/latents remain under output/input/director_<session_id>/
on the pod. Only the final video (and small thumbs/check files when needed) are
downloaded locally.

Usage (same CLI as director.py):
    python director_remote.py <input_image> [script.json] [--segments N] ...
    python director_remote.py image.png --prompt "..." --segments 2 --post-edit

Requires comfy_auto/secrets.json with comfyui.remote_url and runpod SSH (TCP/SCP).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from director import (
    SEGMENT_DURATION,
    build_parser,
    check_frames,
    extract_frames_at_1fps,
    _build_checker,
    _get_pacing,
    _load_api_keys,
    _merge_prompt_tips_for_director,
    _random_seed,
    _segment_arc_instruction,
)
from project_paths import get_director_sessions_root
from runpod_client import (
    comfyui_url,
    download_remote_file,
    load_secrets,
    parse_segment_history,
    ref_to_remote_output_path,
    remote_cp,
    remote_input_basename,
    remote_prepare_session,
    remote_segment_output_prefix,
    remote_session_layout,
    upload_session_input,
)

import run_itv_director as ritv


def _make_session_dir() -> str:
    """Create local session folder under D:\\ComfyProjects\\...\\director_sessions."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(ts.encode()).hexdigest()[:6]
    path = os.path.join(get_director_sessions_root(), f"{ts}_{short_hash}")
    os.makedirs(path, exist_ok=True)
    return path


def _stage_latent_for_extend(
    parsed: dict,
    *,
    secrets: dict,
    session_id: str,
    next_seg_label: str,
) -> str:
    """Copy segment latent from remote output to remote input for extend5 LoadLatent."""
    latent_remote = parsed.get("latent_remote")
    if not latent_remote:
        raise RuntimeError("No latent in ComfyUI history for extend staging")

    layout = remote_session_layout(secrets, session_id)
    latent_name = f"{next_seg_label}_prev.latent"
    latent_input_remote = f"{layout['input_dir']}/{latent_name}"
    remote_cp(
        secrets,
        latent_remote,
        latent_input_remote,
        label=f"Stage latent on pod -> {latent_input_remote}",
    )
    return remote_input_basename(session_id, latent_name)


def _download_thumb_for_llm(
    parsed: dict,
    *,
    secrets: dict,
    session_dir: str,
    seg_label: str,
) -> str | None:
    """Download one pre-image frame for LLM autoprompt (last frame, else first)."""
    image_refs = parsed.get("image_refs") or []
    if not image_refs:
        return None
    pick = image_refs[-1]
    thumb_dir = os.path.join(session_dir, "_thumb")
    os.makedirs(thumb_dir, exist_ok=True)
    for idx, ref in enumerate((pick, image_refs[0])):
        remote_path = ref_to_remote_output_path(secrets, ref)
        ext = os.path.splitext(remote_path)[1] or ".png"
        suffix = "last" if idx == 0 else "first"
        local_thumb = os.path.join(thumb_dir, f"{seg_label}_{suffix}{ext}")
        try:
            download_remote_file(
                remote_path,
                local_thumb,
                secrets=secrets,
                label=f"Download thumb for LLM ({os.path.basename(remote_path)})",
            )
            return local_thumb
        except Exception:
            continue
    return None


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    parser.add_argument(
        "--post-edit",
        action="store_true",
        help="After director completes, run remote post-editor (upscale → RIFE → final MP4).",
    )
    parser.add_argument(
        "--post-edit-skip-scale",
        action="store_true",
        help="Post-edit: skip Scale_Up (combine test / debug).",
    )
    parser.add_argument(
        "--post-edit-skip-fill",
        action="store_true",
        help="Post-edit: skip RIFE Fill_frame.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    secrets = load_secrets()

    # ── Parse segment count: script JSON, --duration, or --segments ──────────
    script_entries: list[dict] | None = None
    script_data: dict | None = None
    script_dir: str | None = None
    if args.script:
        if not os.path.isfile(args.script):
            print(f"ERROR: Script not found: {args.script}", file=sys.stderr)
            sys.exit(1)
        script_dir = os.path.dirname(os.path.abspath(args.script))
        with open(args.script, encoding="utf-8") as f:
            script_data = json.load(f)
        script_entries = script_data.get("segments", [])
        if not script_entries:
            print("ERROR: Script has no 'segments' array.", file=sys.stderr)
            sys.exit(1)
        num_segments = len(script_entries)
        print(f"  Loaded script: {num_segments} segments")
    else:
        if args.duration is not None:
            num_segments = max(1, args.duration // SEGMENT_DURATION)
            if args.duration % SEGMENT_DURATION:
                print(
                    f"  Note: {args.duration}s rounds to {num_segments} x {SEGMENT_DURATION}s = "
                    f"{num_segments * SEGMENT_DURATION}s"
                )
        else:
            num_segments = max(1, args.segments)

    from autoprompt import generate_prompt

    prompt_tips_text, prompt_tips_labels = _merge_prompt_tips_for_director(
        script_data=script_data,
        script_dir=script_dir,
        extra_file=args.llm_tips,
        use_default_file=not args.no_default_llm_tips,
    )
    if prompt_tips_text:
        print(f"  LLM prompt tips: {', '.join(prompt_tips_labels)}")

    with comfyui_url(secrets) as comfy_url:
        ritv.COMFYUI_URL = comfy_url

        # ── Optional content checker (downloads segment MP4 only when enabled) ─
        api_keys = _load_api_keys()
        checker = None
        if not args.skip_check:
            print("Initialising content checker (relaxed mode)...")
            checker = _build_checker(api_keys)

        session_dir = _make_session_dir()
        session_id = os.path.basename(session_dir)
        total_seconds = num_segments * SEGMENT_DURATION
        print(f"\nSession directory: {session_dir}")
        print(f"Session id: {session_id}")
        print(f"Remote ComfyUI: {comfy_url}")
        print(f"Segments: {num_segments} x {SEGMENT_DURATION}s = {total_seconds}s total\n")

        # ── Step 1: Create matching input/output dirs on the pod ───────────────
        remote_layout = remote_prepare_session(secrets, session_id)

        manifest: dict = {
            "input_image": os.path.abspath(args.image),
            "started_at": datetime.datetime.now().isoformat(),
            "remote": True,
            "comfyui_url": comfy_url,
            "session_id": session_id,
            "remote_session": remote_layout,
            "settings": {
                "segments": num_segments,
                "segment_duration": SEGMENT_DURATION,
                "total_seconds": total_seconds,
                "script": args.script,
                "steps": args.steps,
                "cfg": args.cfg,
                "provider": args.provider,
                "threshold": args.threshold,
                "max_retries": args.max_retries,
                "skip_check": args.skip_check,
                "lightning_combo": args.lightning_combo,
                "llm_tips_sources": prompt_tips_labels,
                "llm_tips_extra_file": args.llm_tips,
                "no_default_llm_tips": args.no_default_llm_tips,
            },
            "segments": [],
            "final_video": None,
        }

        anchor_image_path = os.path.abspath(args.image)
        anchor_name = f"anchor_{os.path.basename(anchor_image_path)}"
        # ── Step 2: Upload anchor once (extend5 LoadImage reads this every seg) ─
        anchor_basename = upload_session_input(
            anchor_image_path,
            anchor_name,
            secrets=secrets,
            session_id=session_id,
            session_dir=session_dir,
            label=f"Upload anchor -> {remote_layout['input_dir']}/{anchor_name}",
        )
        print(f"  Anchor image (remote): {anchor_basename}")

        current_image_path = anchor_image_path
        current_prompt = args.prompt
        prev_images_folder_remote: str | None = None
        prev_latent_basename: str | None = None
        prev_width: int | None = None
        prev_height: int | None = None
        last_video_remote: str | None = None
        last_pre_image_folder_remote: str | None = None

        seg_pad = len(str(num_segments))

        for seg_idx in range(1, num_segments + 1):
            seg_label = f"seg_{seg_idx:0{seg_pad}d}"
            # ComfyUI SaveImage/SaveLatent/VHS prefix: output/director_<id>/seg_N/
            output_prefix = remote_segment_output_prefix(session_id, seg_label)

            print("=" * 70)
            print(f"  SEGMENT {seg_idx}/{num_segments}")
            print("=" * 70)

            # ── Step 3a: Build prompt (script / --prompt / LLM autoprompt) ─────
            excitement, stableness = None, None
            if script_entries:
                entry = script_entries[seg_idx - 1]
                high_level = entry.get("high_level_prompt", "").strip()
                if not high_level:
                    print(f"  ERROR: Segment {seg_idx} has no high_level_prompt in script.")
                    break
                excitement = entry.get("excitement", 5)
                stableness = entry.get("stableness", 3)
                print(f"  Script: \"{high_level}\" (ex={excitement}, st={stableness})")
                print(f"  Generating autoprompt (excitement={excitement}, stableness={stableness})...")
                arc = _segment_arc_instruction(seg_idx, num_segments)
                llm_prompt = generate_prompt(
                    current_image_path,
                    duration=SEGMENT_DURATION,
                    provider=args.provider,
                    excitement=excitement,
                    stableness=stableness,
                    segment_arc=arc,
                    prompt_tips=prompt_tips_text,
                )
                current_prompt = high_level + "\n" + llm_prompt
                print(f"  Prompt (high_level + LLM):\n{current_prompt}")
            elif seg_idx == 1:
                if not current_prompt:
                    print("  Generating autoprompt for segment 1 (no --prompt, no script)...")
                    excitement, stableness = _get_pacing(1, num_segments)
                    arc = _segment_arc_instruction(1, num_segments)
                    current_prompt = generate_prompt(
                        current_image_path,
                        duration=SEGMENT_DURATION,
                        provider=args.provider,
                        excitement=excitement,
                        stableness=stableness,
                        segment_arc=arc,
                        prompt_tips=prompt_tips_text,
                    )
                    print(f"  Autoprompt:\n{current_prompt}")
                else:
                    print("  Using --prompt for segment 1")
                    print(f"  Prompt:\n{current_prompt}")
            else:
                excitement, stableness = _get_pacing(seg_idx, num_segments)
                arc = _segment_arc_instruction(seg_idx, num_segments)
                print(f"  Generating autoprompt (excitement={excitement}, stableness={stableness})...")
                current_prompt = generate_prompt(
                    current_image_path,
                    duration=SEGMENT_DURATION,
                    provider=args.provider,
                    excitement=excitement,
                    stableness=stableness,
                    segment_arc=arc,
                    prompt_tips=prompt_tips_text,
                )
                print(f"  Autoprompt:\n{current_prompt}")

            source_basename: str | None = None
            if seg_idx == 1:
                # Seg 1 only: upload source image; extend reuses pod-side pre_image/
                source_name = f"{seg_label}_source_{os.path.basename(current_image_path)}"
                print(f"  Source: {os.path.basename(current_image_path)}")
                source_basename = upload_session_input(
                    current_image_path,
                    source_name,
                    secrets=secrets,
                    session_id=session_id,
                    session_dir=session_dir,
                    label=f"Upload source -> {remote_layout['input_dir']}/{source_name}",
                )
            else:
                # Seg 2+: extend5 reads prev pre_image/ + latent already on pod
                if not prev_images_folder_remote or not prev_latent_basename:
                    raise RuntimeError(
                        "Extend requires prev images folder and latent staged on pod from previous segment"
                    )
                print(
                    f"  Extend (on pod): frames={prev_images_folder_remote}, "
                    f"latent={prev_latent_basename}"
                )

            seg_record: dict = {
                "segment": seg_idx,
                "prompt": current_prompt,
                "excitement": excitement,
                "stableness": stableness,
                "remote_output_prefix": output_prefix,
                "attempts": [],
                "final_seed": None,
                "video_file": None,
                "remote_video": None,
                "content_check": None,
            }
            if script_entries and seg_idx <= len(script_entries):
                seg_record["high_level_prompt"] = script_entries[seg_idx - 1].get(
                    "high_level_prompt", ""
                )

            succeeded = False
            for attempt in range(1, args.max_retries + 1):
                seed = _random_seed()
                print(f"\n  Attempt {attempt}/{args.max_retries}  seed={seed}")

                attempt_record: dict = {
                    "attempt": attempt,
                    "seed": seed,
                    "status": "pending",
                    "content_check_results": None,
                }

                try:
                    # ── Step 3b: Queue workflow on pod (HTTP POST, no bulk SCP) ───
                    if seg_idx == 1:
                        history, prompt_used = ritv.run_first5(
                            source_basename,
                            prompt=current_prompt,
                            seed=seed,
                            steps=args.steps,
                            cfg=args.cfg,
                            filename_prefix=output_prefix,
                            latent_filename_prefix=output_prefix,
                            lightning_combo=args.lightning_combo,
                        )
                    else:
                        if prev_width is None or prev_height is None:
                            raise RuntimeError(
                                "Extend requires prev_width/prev_height from previous segment."
                            )
                        history, prompt_used = ritv.run_extend5(
                            anchor_basename,
                            prev_images_folder_remote,
                            prev_latent_basename,
                            prompt=current_prompt,
                            seed=seed,
                            width=prev_width,
                            height=prev_height,
                            steps=args.steps,
                            cfg=args.cfg,
                            filename_prefix=output_prefix,
                            latent_filename_prefix=output_prefix,
                            lightning_combo=args.lightning_combo,
                        )
                    seg_record["prompt"] = prompt_used
                except RuntimeError as exc:
                    print(f"  Workflow error: {exc}")
                    attempt_record["status"] = "workflow_error"
                    attempt_record["error"] = str(exc)
                    seg_record["attempts"].append(attempt_record)
                    continue

                status_str = history.get("status", {}).get("status_str", "")
                if status_str == "error":
                    print("  ComfyUI execution error.")
                    attempt_record["status"] = "comfyui_error"
                    seg_record["attempts"].append(attempt_record)
                    continue

                # ── Step 3c: Parse ComfyUI history → remote paths (no download) ─
                parsed = parse_segment_history(history, secrets)
                video_remote = parsed.get("video_remote")
                if not video_remote:
                    print("  Could not find output video in ComfyUI history.")
                    attempt_record["status"] = "no_video"
                    seg_record["attempts"].append(attempt_record)
                    continue

                check_results: list = []
                if checker is not None:
                    # Content check needs one segment MP4 locally (temporary)
                    check_dir = os.path.join(session_dir, "_check")
                    os.makedirs(check_dir, exist_ok=True)
                    check_video = os.path.join(check_dir, f"{seg_label}.mp4")
                    download_remote_file(
                        video_remote,
                        check_video,
                        secrets=secrets,
                        label=f"Download segment video for content check",
                    )
                    print(f"  Running content check on {SEGMENT_DURATION + 1} frames...")
                    frames = extract_frames_at_1fps(check_video)
                    passed, check_results = check_frames(frames, checker, args.threshold)
                    attempt_record["content_check_results"] = check_results
                    if not passed:
                        print("  FAILED content check — retrying with new seed.")
                        attempt_record["status"] = "content_flagged"
                        seg_record["attempts"].append(attempt_record)
                        continue
                    print("  Content check PASSED.")

                attempt_record["status"] = "success"
                seg_record["attempts"].append(attempt_record)
                seg_record["final_seed"] = seed
                seg_record["remote_video"] = video_remote
                last_video_remote = video_remote
                pre_folder = parsed.get("pre_image_folder_remote")
                if pre_folder:
                    last_pre_image_folder_remote = pre_folder
                succeeded = True

                next_seg = seg_idx + 1
                if next_seg <= num_segments:
                    # ── Step 3d: Stage next segment on pod (cp latent, keep frames) ─
                    if not pre_folder:
                        print("  ERROR: No pre-image folder on pod for extend.")
                        succeeded = False
                        break
                    prev_images_folder_remote = pre_folder

                    next_seg_label = f"seg_{next_seg:0{seg_pad}d}"
                    try:
                        prev_latent_basename = _stage_latent_for_extend(
                            parsed,
                            secrets=secrets,
                            session_id=session_id,
                            next_seg_label=next_seg_label,
                        )
                    except RuntimeError as exc:
                        print(f"  ERROR: {exc}")
                        succeeded = False
                        break

                    thumb = _download_thumb_for_llm(
                        parsed,
                        secrets=secrets,
                        session_dir=session_dir,
                        seg_label=seg_label,
                    )
                    # One PNG for LLM vision on next segment's autoprompt
                    if thumb and os.path.isfile(thumb):
                        current_image_path = thumb
                        sz = ritv.get_image_size(thumb)
                        if sz:
                            prev_width, prev_height = sz
                            print(
                                f"  Staged on pod for seg {next_seg}: "
                                f"frames={prev_images_folder_remote}, latent={prev_latent_basename} "
                                f"(res {prev_width}x{prev_height})"
                            )
                        else:
                            print(
                                f"  Staged on pod for seg {next_seg}: "
                                f"frames={prev_images_folder_remote}, latent={prev_latent_basename}"
                            )
                    else:
                        print("  WARN: Could not download thumb; reusing prior image for LLM.")
                break

            if not succeeded:
                print(f"\n  SEGMENT {seg_idx} FAILED after {args.max_retries} attempts.")
                print("  Aborting pipeline.")
                seg_record["video_file"] = None
                manifest["segments"].append(seg_record)
                break

            manifest["segments"].append(seg_record)
            print(f"\n  Segment {seg_idx} complete (outputs on pod).\n")

        completed = sum(1 for s in manifest["segments"] if s.get("remote_video"))
        final_video_name = f"final_{total_seconds}s.mp4"
        final_path = os.path.join(session_dir, final_video_name)
        # ── Step 4: Download raw Wan MP4 (last segment = full timeline) ───────
        if completed == num_segments and last_video_remote:
            download_remote_file(
                last_video_remote,
                final_path,
                secrets=secrets,
                label=f"Download final video -> {final_video_name}",
            )
            manifest["final_video"] = final_video_name
            manifest["remote_final_video"] = last_video_remote
            print(f"\n  Final video: {final_path}")
        else:
            if completed < num_segments:
                print(f"\n  Only {completed}/{num_segments} segments completed.")
            manifest["final_video"] = None

        # ── Step 5 (optional): Post-edit upscale → RIFE → combine on pod ───────
        if args.post_edit and completed == num_segments and last_pre_image_folder_remote:
            print("\n" + "=" * 70)
            print("  POST-EDIT (remote)")
            print("=" * 70)
            from post_editor_remote import RemotePostEditorConfig, run_post_editor_remote

            try:
                pe_result = run_post_editor_remote(
                    RemotePostEditorConfig(
                        session_id=session_id,
                        source_pre_image_remote=last_pre_image_folder_remote,
                        local_session_dir=session_dir,
                        skip_scale=args.post_edit_skip_scale,
                        skip_fill=args.post_edit_skip_fill,
                    ),
                    secrets=secrets,
                )
                manifest["post_editor"] = pe_result.report
                manifest["postedit_final_video"] = "postedit_final.mp4"
                manifest["remote_postedit_video"] = pe_result.remote_video
                print(f"\n  Post-edit final: {pe_result.local_video}")
            except Exception as exc:
                print(f"  Post-edit FAILED: {exc}")
                manifest["post_editor_error"] = str(exc)
        elif args.post_edit and completed < num_segments:
            print("\n  Post-edit skipped (director did not complete all segments).")

        manifest["remote_pre_image_source"] = last_pre_image_folder_remote
        manifest["finished_at"] = datetime.datetime.now().isoformat()
        manifest_path = os.path.join(session_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n  Manifest: {manifest_path}")

        print("\n" + "=" * 70)
        if manifest.get("postedit_final_video"):
            print(f"  DONE — {os.path.join(session_dir, manifest['postedit_final_video'])}")
        elif manifest["final_video"]:
            print(f"  DONE — {final_path}")
        else:
            print("  DONE — pipeline did not produce a final video.")
        print("=" * 70)


if __name__ == "__main__":
    main()
