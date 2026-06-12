import json
from pathlib import Path

wf_path = Path(r"c:\Users\sheng\Documents\ComfyUI-aki-v3\ComfyUI\user\default\workflows\ITV_single_first5.json")
models_root = Path(r"c:\Users\sheng\Documents\ComfyUI-aki-v3\ComfyUI\models")

with wf_path.open(encoding="utf-8") as f:
    wf = json.load(f)

LOADER_MAP = {
    "CLIPLoader": ("clip_name", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
    "UNETLoader": ("unet_name", "diffusion_models"),
    "UnetLoaderGGUF": ("unet_name", "unet_gguf"),
    "LoraLoaderModelOnly": ("lora_name", "loras"),
    "UpscaleModelLoader": ("model_name", "upscale_models"),
}

entries = []


def walk_nodes(nodes, prefix=""):
    for n in nodes:
        t = n.get("type", "")
        wv = n.get("widgets_values", [])
        title = n.get("title", t)
        if t in LOADER_MAP:
            folder = LOADER_MAP[t][1]
            val = wv[0] if wv else None
            entries.append((t, title, val, folder, prefix))
        elif t == "Power Lora Loader (rgthree)":
            for item in wv:
                if isinstance(item, dict) and item.get("lora"):
                    entries.append((t, title, item["lora"], "loras", prefix))
        elif t == "FL_RIFE" and wv:
            entries.append((t, title, wv[0], "rife", prefix))
        elif t == "LoadImage" and wv:
            entries.append((t, "LoadImage", wv[0], "input", prefix))


walk_nodes(wf.get("nodes", []))
for sg in wf.get("definitions", {}).get("subgraphs", []):
    walk_nodes(sg.get("nodes", []), f"subgraph:{sg.get('name')}")

# Resolve local paths
FOLDER_SEARCH = {
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "diffusion_models": ["diffusion_models", "unet"],
    "unet_gguf": ["unet", "diffusion_models"],
    "loras": ["loras"],
    "upscale_models": ["upscale_models"],
    "rife": ["rife", "custom_nodes"],
    "input": [],
}


def find_local(rel_path, folder_hint):
    if not rel_path:
        return None
    rel = Path(str(rel_path).replace("\\", "/"))
    if folder_hint == "input":
        p = Path(r"c:\Users\sheng\Documents\ComfyUI-aki-v3\ComfyUI\input") / rel
        return p if p.exists() else None
    for sub in FOLDER_SEARCH.get(folder_hint, [folder_hint]):
        p = models_root / sub / rel
        if p.exists():
            return p
    # basename search
    name = rel.name
    for sub in FOLDER_SEARCH.get(folder_hint, [folder_hint]):
        base = models_root / sub
        if not base.exists():
            continue
        for hit in base.rglob(name):
            if hit.is_file():
                return hit
    return None


print("=== WORKFLOW MODEL INVENTORY ===\n")
to_copy = []
for t, title, val, folder, prefix in entries:
    skip = folder == "unet_gguf" or (t == "UNETLoader")
    local = find_local(val, folder) if val else None
    status = "SKIP (gguf/unet - user handles)" if skip else ("FOUND" if local else "MISSING")
    print(f"[{status}] {t} | {title}")
    print(f"         ref: {val}")
    print(f"         dest folder: models/{folder}/")
    if local:
        print(f"         local: {local}")
        if not skip:
            to_copy.append((local, folder, val))
    print()

print("=== COPY LIST (non-gguf) ===")
for local, folder, val in to_copy:
    rel = str(val).replace("\\", "/")
    print(f"{local} -> models/{folder}/{rel}")
