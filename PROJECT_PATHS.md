# Where data goes

## ComfyUI **input / output** — stay on **C:** (repo)

`run_itv_director.py` / `director.py` read Comfy’s history from:

- `<repo>\ComfyUI\output`
- Images/latents use `<repo>\ComfyUI\input`

Use your normal launcher (e.g. 绘世). **No** need to move Comfy I/O to D: for director to work.

## Director **sessions only** — default **D:**

Each `director.py` run creates a folder like:

- `COMFY_PROJECTS_ROOT\comfy_auto\director_sessions\YYYYMMDD_HHMMSS_xxxxxx\`

Default `COMFY_PROJECTS_ROOT` is **`D:\ComfyProjects`**.

Override:

```powershell
$env:COMFY_PROJECTS_ROOT = "E:\MyData"
```

## Why this works

Comfy writes each render to **C:** → director **copies** video/latents/frames into the session folder on **D:**. Only the session tree (videos, manifest, `seg_*` frames) needs to be large and off C:.

## `seg_all/` and `post_editor.py`

After a full director run, the **last segment’s** saved frame batch is copied to **`seg_all/`** (full timeline for the final video). **`post_editor`** defaults to **`--frames-source auto`**, which uses **`seg_all/`** when present.

Older sessions without `seg_all/` fall back to **only the highest `seg_N/`** (not a concat of every `seg_*`, which would duplicate frames when each folder is cumulative).

Manifest may include `"post_edit_frames": "seg_all"`.
