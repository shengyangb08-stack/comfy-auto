# Optional LLM prompt tips (director autoprompt)

Edit this file to steer **wording** the vision LLM uses when expanding scenes into second-by-second lines.
Loaded automatically by `director.py` unless you pass `--no-default-llm-tips`.

You can also set per-script tips in JSON (`llm_tips` / `llm_tips_file`) or add another file with `--llm-tips path`. Multiple sources are merged with `---` separators.

Examples (customize freely):

- Prefer concrete visual descriptions over vague or model-flagged terms where it helps the video model.
- For fluid / release scenes, describe **visible water-like flow from the vulva** rather than slang that may trigger safety filters.

## Prompting style: prefer **Direct Action (Clinical / Mechanical)**

Video models (e.g. Wan2.2) track **pixel displacement**. They work best from **dry, mechanical** lines: who moves what, where things are in space, and how motion **reciprocates** or repeats. Favor short, physics-like phrasing over mood or story.

**Good (direct / mechanical)** — e.g. “Subject A thrusts into Subject B,” “Reciprocating motion,” “Insert and withdraw.”

**Avoid when possible**

- **Type A — Descriptive & narrative (“storytelling”)** — e.g. “The camera follows her passion…”, “She is experiencing…” — tends to produce weak or static motion; the model chases atmosphere instead of clear subject motion.
- **Type B — Explicit slang / emotional wording** — may add motion but often skews art style (rougher, cartoonish) and is **less precise** for physics than neutral anatomical / mechanical wording.

When expanding `high_level_prompt` into second-by-second lines, **bias toward direct-action phrasing** unless the user explicitly asks for mood-only or poetic narration.

