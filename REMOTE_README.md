# Remote RunPod pipeline (comfy_auto)

Run ITV director and post-processing on a **RunPod ComfyUI** pod from your Windows machine. Workflows are sent over **HTTPS** (`/prompt`); images and videos move via **SCP/SSH**.

Local Python: always use the bundled interpreter:

```powershell
& ".\python\python.exe" comfy_auto/director_remote.py ...
```

---

## Prerequisites

### `comfy_auto/secrets.json`

Copy from `secrets.example.json`. Required keys:

| Key | Purpose |
|-----|---------|
| `comfyui.remote_url` | Pod ComfyUI URL, e.g. `https://<pod>-8188.proxy.runpod.net` |
| `comfyui.remote_comfyui_path` | ComfyUI root on pod, e.g. `/workspace/runpod-slim/ComfyUI` |
| `runpod.ssh_host` | Pod TCP IP for SCP |
| `runpod.ssh_port` | SSH port (RunPod exposes a custom port) |
| `runpod.ssh_user` | Usually `root` |
| `runpod.ssh_key_path` | Path to your SSH private key |

Optional: omit `remote_url` and the client will open an SSH tunnel to port 8188 (slower to set up).

### Pod models & nodes

Same as local ITV + post-editor. Workflows use `sage_attention: disabled` in API JSON (pod may lack `sageattention`).

---

## Scripts overview

| Script | Role |
|--------|------|
| **`runpod_client.py`** | Shared SSH/SCP, session paths, remote file ops (not run directly) |
| **`run_itv_remote.py`** | Smoke test: single **first5** segment only |
| **`director_remote.py`** | Full **N × 5s** director on pod (first5 + extend5) |
| **`post_editor_remote.py`** | Upscale → RIFE → final MP4 on pod (after director) |
| **`run_itv_director.py`** | Workflow API helpers (used by all of the above) |

---

## Session layout

Each director run gets one **session id** (same name locally and on the pod), e.g. `20260611_213140_16d67e`.

### Local (`D:\ComfyProjects\comfy_auto\director_sessions\<session_id>\`)

| Path | Contents |
|------|----------|
| `manifest.json` | Run metadata, remote paths, segment prompts |
| `inputs/` | Copies of files uploaded to the pod |
| `final_10s.mp4` | Raw Wan output (last segment video, downloaded) |
| `postedit_final.mp4` | Polished output (if post-edit ran) |
| `_thumb/` | One PNG per segment for LLM autoprompt only |
| `_check/` | Temp segment MP4s when content check is enabled |
| `_postedit_probe/` | Tiny probe downloads for frame dimensions |

### Pod (`/workspace/.../ComfyUI/`)

```
input/director_<session_id>/
  anchor_*.png              # extend5 anchor (LoadImage)
  seg_1_source_*.png        # first5 source
  seg_2_prev.latent         # staged latent for extend (cp from output)

output/director_<session_id>/
  seg_1/                    # first5 outputs
    pre_image/pre_*.png     # frames for extend + post-edit source
    seg_1_00001.mp4         # ~5s raw segment video
    seg_1_00001_.latent
  seg_2/                    # extend5 outputs (full timeline pre_images)
    pre_image/pre_*.png
    seg_2_00002.mp4         # ~10s raw video (used as final_10s download)
  postedit/                 # post-editor working dirs (all on pod)
    00_source_flat/         # frame_00000.png … (materialized from seg_N/pre_image)
    01_scaled/
    02_rife_filled/
    final/*.mp4
```

**Design:** Segment intermediates stay on the pod. Between segments we `cp` latent output → input and point extend5 at the previous `pre_image/` folder — no bulk re-upload of frames.

---

## Workflows (API JSON)

| Workflow | When |
|----------|------|
| `ITV_single_first5_API.json` | Segment 1 |
| `ITV_single_extend5_API.json` | Segments 2+ |
| `Scale_Up_API.json` | Post-edit step 1 |
| `Fill_frame_API.json` | Post-edit step 2 (RIFE 2×) |
| `Final_Combine_API.json` | Post-edit step 3 |

Raw segment MP4s come from **VHS inside first5/extend5**. Post-editor **Final_Combine** is a separate, higher-quality encode after upscale + RIFE.

---

## 1. `director_remote.py`

Same CLI as local `director.py`, plus remote-only flags.

### Usage

```powershell
# 2 segments, fixed prompt for seg 1, skip content check
& ".\python\python.exe" comfy_auto/director_remote.py `
  "ComfyUI\input\your_image.png" `
  --prompt "..." `
  --segments 2 `
  --skip-check `
  --lightning-combo 2

# With script JSON
& ".\python\python.exe" comfy_auto/director_remote.py image.png script.json

# Director + post-edit in one command
& ".\python\python.exe" comfy_auto/director_remote.py image.png `
  --segments 2 --skip-check --post-edit
```

### Arguments (inherited from `director.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `image` | (required) | Local source image for segment 1 |
| `script` | optional | JSON with `segments[]` (high_level_prompt, excitement, stableness) |
| `--segments` | `2` | Number of 5-second segments (ignored if script provided) |
| `--duration` | — | Total seconds; overrides `--segments` |
| `--prompt` | — | Prompt for segment 1 (or autoprompt if omitted) |
| `--steps` | — | Override sampling steps |
| `--cfg` | — | Override CFG |
| `--provider` | `gemini` | LLM for autoprompt: `gemini` or `grok` |
| `--max-retries` | `10` | Retries per segment on content-check failure |
| `--threshold` | `0.9` | Content-check score threshold |
| `--skip-check` | off | Skip deformation content check (faster) |
| `--lightning-combo` | `2` | `1`=more motion, `2`=less degradation, `3`=balanced |
| `--llm-tips` | — | Extra `.md`/`.txt` merged into LLM prompt |
| `--no-default-llm-tips` | off | Don't load `prompt_llm_tips.md` |

### Remote-only arguments

| Argument | Description |
|----------|-------------|
| `--post-edit` | After all segments succeed, run `post_editor_remote` on last segment `pre_image/` |
| `--post-edit-skip-scale` | Post-edit: skip Scale_Up (debug combine) |
| `--post-edit-skip-fill` | Post-edit: skip RIFE |

### Pipeline steps (what the script does)

1. Load secrets; connect to ComfyUI (HTTPS or SSH tunnel).
2. Create local session dir + remote `input/director_<id>` and `output/director_<id>`.
3. Upload anchor image once (used by all extend5 runs).
4. **For each segment:**
   - Build prompt (script / `--prompt` / LLM autoprompt).
   - Seg 1: upload source image → queue **first5** via API.
   - Seg 2+: extend reads prev `pre_image/` + staged latent on pod → queue **extend5**.
   - On success: `cp` latent to input for next segment; download one thumb PNG for LLM.
   - Optional: download segment MP4 for content check.
5. Download last segment MP4 → `final_<N>s.mp4`.
6. Optional `--post-edit`: upscale → RIFE → combine → `postedit_final.mp4`.
7. Write `manifest.json`.

---

## 2. `post_editor_remote.py`

Standalone post-processing on an existing remote director session.

### Usage

```powershell
# Full pipeline (scale → RIFE → combine)
& ".\python\python.exe" comfy_auto/post_editor_remote.py `
  "D:\ComfyProjects\comfy_auto\director_sessions\20260611_213140_16d67e"

# Debug: combine only (skip upscale + RIFE)
& ".\python\python.exe" comfy_auto/post_editor_remote.py `
  "...\session_dir" --skip-scale --skip-fill
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `session` | (required) | Local director session dir (must contain `manifest.json` with `"remote": true`) |
| `--batch` | `16` | Frames per ComfyUI batch (scale and RIFE) |
| `--source-fps` | `16.0` | Input frame rate before RIFE |
| `--combine-fps` | `source_fps × 2` | Final VHS encode FPS (default 32 after 2× RIFE) |
| `--skip-scale` | off | Use materialized source frames as “scaled” input |
| `--skip-fill` | off | Skip RIFE; combine from scaled folder |
| `--max-output-edge` | `3840` | Max long edge (px) after 4× upscale + ImageScaleBy |
| `--scale-by` | auto | Manual ImageScaleBy factor (0–1); overrides auto cap |
| `--combine-max-pixels` | QHD×1.2 | Cap loader resolution for Final_Combine (RAM); 0 = no cap |
| `--output-name` | `postedit_final.mp4` | Local filename for downloaded result |

### Pipeline steps

1. Read `manifest.json`; locate last segment `pre_image/` on pod.
2. **Materialize** `pre_*.png` → `postedit/00_source_flat/frame_00000.png` … (on pod via SSH).
3. **Scale_Up** (batched): 4× model + ImageScaleBy → `01_scaled/`.
4. **Fill_frame** (batched): RIFE 2× → `02_rife_filled/`.
5. **Final_Combine**: VHS MP4 at `combine_fps`.
6. Download **only** the final MP4 locally; update manifest.

---

## 3. `run_itv_remote.py`

Minimal test: one **first5** run, downloads all outputs to `runpod_test_sessions/`.

### Usage

```powershell
& ".\python\python.exe" comfy_auto/run_itv_remote.py `
  --image "ComfyUI\input\test.png" `
  --prompt "your prompt" `
  --seed 42 `
  --lightning 2
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Local source image |
| `--prompt` | (required) | Positive prompt |
| `--seed` | random | Seed (`-1` = random) |
| `--lightning` | `2` | Lightning LoRA combo `1`/`2`/`3` |
| `--steps` | — | Steps override |
| `--cfg` | — | CFG override |

Outputs go to `D:\ComfyProjects\comfy_auto\runpod_test_sessions\<timestamp>_<id>\`.

---

## Typical end-to-end command

```powershell
cd C:\Users\sheng\Documents\ComfyUI-aki-v3

& ".\python\python.exe" comfy_auto/director_remote.py `
  "ComfyUI\input\00010-1741776828-50.png" `
  --prompt "..." `
  --segments 2 `
  --skip-check `
  --lightning-combo 2 `
  --post-edit
```

**Deliverables in session folder:**

- `final_10s.mp4` — raw Wan generation (~360×512, 16 fps)
- `postedit_final.mp4` — upscaled + RIFE + re-encoded (~1440×2048 loader, 32 fps)

---

## Troubleshooting

| Issue | Check |
|-------|--------|
| 403 on `/prompt` | RunPod proxy needs `User-Agent` header (handled in `run_itv_director.py`) |
| `sageattention` missing | API workflows use `sage_attention: disabled` |
| Extend fails “no prev frames” | Last segment must write `pre_image/`; check `remote_output_prefix` in manifest |
| Post-edit slow | Batches collect frames via per-file `ssh cp` on pod; future optimization possible |
| SCP fails | Verify `runpod.ssh_port` / key; use TCP SSH host, not `ssh.runpod.io` proxy |

---

## Related local docs

- `PROJECT_PATHS.md` — where sessions are stored on `D:\ComfyProjects`
- `post_editor.py` / local post-editor — same three workflows, all local paths
- `secrets.example.json` — credential template
