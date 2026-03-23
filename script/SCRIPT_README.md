# Using a script (剧本) with `director.py`

Session folders (`manifest.json`, `seg_*`, final mp4) are created under **`COMFY_PROJECTS_ROOT\comfy_auto\director_sessions\`** (default **`D:\ComfyProjects`**) — see **`PROJECT_PATHS.md`**. ComfyUI’s own `input`/`output` stay in the repo on C:.

## Command

```bash
python director.py <image.png> <script.json> [--provider gemini|grok] [--lightning-combo 1|2|3] ...
```

Example JSON scripts ship under **`script/`** (e.g. `script/script_two_segment_climax.json`).

- **Segment count** = number of objects in `script.json` → `"segments"` (not `--segments`).
- Each segment is **5 seconds** of video (same as no-script mode).

## Script JSON shape

```json
{
  "segments": [
    {
      "segment": 1,
      "high_level_prompt": "Short creative direction for this 5s block.",
      "excitement": 8,
      "stableness": 2
    },
    {
      "segment": 2,
      "high_level_prompt": "Continue or escalate; this block is where climax lands.",
      "excitement": 9,
      "stableness": 4
    }
  ]
}
```

- **`high_level_prompt`** (required): Your scene intent for that segment. The LLM still expands it into second-by-second lines and **prepends** this text to the generated prompt.
- **`excitement`** / **`stableness`** (optional): Per-segment overrides for the autoprompt LLM (defaults `5` / `3` if omitted). Use higher **excitement** on the climax segment if you want more intensity.

See `script/script_example.json` for a longer multi-segment example.

---

## How “climax” is enforced (not magic — layered rules)

The pipeline does **three** things so climax tends to land where you want:

### 1. Automatic **narrative arc** text (`segment_arc`)

For every segment, the code passes extra instructions into `generate_prompt()` **based only on**  
`segment index` + **total number of segments** (not read from JSON).

Roughly:

| Segments | Arc |
|----------|-----|
| **1** | Build and **reach climax** in this single 5s clip. |
| **2** | Seg 1: more motion, no climax. Seg 2: **must** depict climax. |
| **5** | Stable → motion → more motion → **climax (seg 4)** → stable again. See `script/script_five_segment_arc.json`. |
| **3–4** | Seg 1: motion, no climax. Seg 2: **designated climax**. Later: aftermath / cool-down. |

So with **2 segments**, climax is **aimed at segment 2**; with **5 segments**, climax is **aimed at segment 4**.

### 2. Your **`high_level_prompt`**

You should **match** the arc in words, e.g.:

- Segment 1: foreplay / movement / tension — avoid asking for orgasm here.
- Segment 2: explicitly ask for climax, squirt, facial expression, release.

That aligns the **user text** with the automatic arc instructions.

### 3. **`excitement` / `stableness`** in the script

Optional but useful: e.g. segment 2 `excitement: 9`, `stableness: 4` so the LLM writes **full** second-by-second detail for the peak.

---

## What the script does **not** do

- It does **not** contain a separate “climax: true” flag — arc is **positional** (segment index).
- It does **not** guarantee the video model will always obey; it **steers** the **text prompt** strongly.

---

## Minimal 2-segment example (`script/script_two_segment_climax.json`)

```json
{
  "segments": [
    {
      "segment": 1,
      "high_level_prompt": "Build tension and movement; hands, hips, breathing; no orgasm yet.",
      "excitement": 8,
      "stableness": 2
    },
    {
      "segment": 2,
      "high_level_prompt": "She reaches orgasm: visible release, squirt if appropriate, eyes and mouth showing climax.",
      "excitement": 9,
      "stableness": 4
    }
  ]
}
```

Run:

```bash
python director.py your.png script/script_two_segment_climax.json
```

---

## No script (default)

```bash
python director.py your.png
```

Default is **2 segments** (10s): same arc as above (motion → climax) using `_get_pacing()` only.
