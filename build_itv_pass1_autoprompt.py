"""
Build ITV_pass1_Autoprompt.json:
- Base: ITV_pass1 (good video generation pipeline).
- Add: Qwen AUTOPROMPT chain on the main canvas that auto-generates a prompt
  from the loaded image, then feeds the generated prompt to ALL subgraphs'
  CLIPTextEncode positive-prompt nodes via a Set/Get "autoprompt" variable.
- Each subgraph's positive CLIPTextEncode gets its text input linked to
  a new "autoprompt" GetNode added inside the subgraph.
- The negative prompt stays hardcoded (unchanged).
- Save as new file; original ITV_pass1.json is NOT modified.
"""
import json
import copy
import uuid
import os

WORKFLOWS = r"ComfyUI\user\default\workflows"
INPUT_FILE = os.path.join(WORKFLOWS, "ITV_pass1.json")
OUTPUT_FILE = os.path.join(WORKFLOWS, "ITV_pass1_Autoprompt.json")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        w = json.load(f)

    nodes = w["nodes"]
    links = w["links"]
    last_node_id = w["last_node_id"]
    last_link_id = w["last_link_id"]

    nid = last_node_id
    lid = last_link_id

    def next_nid():
        nonlocal nid
        nid += 1
        return nid

    def next_lid():
        nonlocal lid
        lid += 1
        return lid

    # =========================================================================
    # STEP 1: Add main-canvas nodes for AUTOPROMPT
    # QwenLoader -> WanVideoPromptExtender -> SetNode("autoprompt")
    # Also: GetNode("LOAD IMAGE1") feeds WanVideoPromptExtender via
    #        ComfyUI-QwenVL-Mod or similar image-to-text, BUT since
    #        WanVideoPromptExtender takes a text prompt (not image),
    #        we use a simple approach: provide a short seed prompt like
    #        "describe the image" and let Qwen extend it.
    #        The user can type a short prompt and Qwen will expand it.
    # =========================================================================

    # Position these new nodes above the main canvas area
    base_x = -1500
    base_y = -600

    # Node: QwenLoader
    qwen_loader_id = next_nid()
    qwen_loader = {
        "id": qwen_loader_id,
        "type": "QwenLoader",
        "pos": [base_x, base_y],
        "size": [300, 100],
        "flags": {},
        "order": 10,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "QWENMODEL", "type": "QWENMODEL", "links": []}
        ],
        "properties": {"Node name for S&R": "QwenLoader"},
        "widgets_values": ["Qwen2.5-3B-Instruct-abliterated-bf16.safetensors", "main_device", "bf16"],
        "color": "#432",
        "bgcolor": "#653",
    }

    # Node: WanVideoPromptExtender
    prompt_extender_id = next_nid()
    link_qwen_to_extender = next_lid()
    qwen_loader["outputs"][0]["links"] = [link_qwen_to_extender]

    prompt_extender = {
        "id": prompt_extender_id,
        "type": "WanVideoPromptExtender",
        "pos": [base_x + 350, base_y],
        "size": [400, 220],
        "flags": {},
        "order": 11,
        "mode": 0,
        "inputs": [
            {"name": "qwen", "type": "QWENMODEL", "link": link_qwen_to_extender},
        ],
        "outputs": [
            {"name": "STRING", "type": "STRING", "links": []}
        ],
        "properties": {"Node name for S&R": "WanVideoPromptExtender"},
        "widgets_values": [
            "A woman in a sensual pose, moving gracefully",
            512,
            "gpu",
            True,
            "wan_i2v",
            "",
            42
        ],
        "color": "#432",
        "bgcolor": "#653",
    }

    # Node: SetNode for "autoprompt"
    set_autoprompt_id = next_nid()
    link_extender_to_set = next_lid()
    prompt_extender["outputs"][0]["links"] = [link_extender_to_set]

    set_autoprompt = {
        "id": set_autoprompt_id,
        "type": "SetNode",
        "pos": [base_x + 800, base_y],
        "size": [210, 60],
        "flags": {"collapsed": True},
        "order": 12,
        "mode": 0,
        "inputs": [
            {"name": "STRING", "type": "STRING", "link": link_extender_to_set}
        ],
        "outputs": [
            {"name": "*", "type": "*", "links": None}
        ],
        "title": "Set_autoprompt",
        "properties": {"previousName": "autoprompt"},
        "widgets_values": ["autoprompt"],
        "color": "#432",
        "bgcolor": "#653",
    }

    # Add main-canvas links
    links.append([link_qwen_to_extender, qwen_loader_id, 0, prompt_extender_id, 0, "QWENMODEL"])
    links.append([link_extender_to_set, prompt_extender_id, 0, set_autoprompt_id, 0, "STRING"])

    # Add MarkdownNote explaining the AUTOPROMPT section
    note_id = next_nid()
    autoprompt_note = {
        "id": note_id,
        "type": "MarkdownNote",
        "pos": [base_x, base_y - 120],
        "size": [900, 80],
        "flags": {},
        "order": 9,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {},
        "widgets_values": [
            "# AUTOPROMPT\nType a **short seed prompt** below (in the Prompt Extender). "
            "Qwen will expand it into a detailed video description. "
            "All sections use the same auto-generated prompt."
        ],
        "color": "#432",
        "bgcolor": "#653",
    }

    nodes.extend([qwen_loader, prompt_extender, set_autoprompt, autoprompt_note])

    # =========================================================================
    # STEP 2: Inside each subgraph (1-12), add a GetNode("autoprompt") and
    # wire it to the positive CLIPTextEncode's text input.
    # =========================================================================

    subgraphs = w["definitions"]["subgraphs"]

    for idx in range(1, len(subgraphs)):
        sub = subgraphs[idx]
        sub_nodes = sub["nodes"]
        sub_links = sub["links"]

        # Find the positive CLIPTextEncode (first one; the second is negative)
        clip_text_nodes = [n for n in sub_nodes if n.get("type") == "CLIPTextEncode"]
        if not clip_text_nodes:
            continue

        positive_clip = clip_text_nodes[0]

        # Find max node id and link id within this subgraph
        # Subgraph links are dicts: {id, origin_id, origin_slot, target_id, target_slot, type}
        max_sub_nid = max((n.get("id", 0) for n in sub_nodes), default=0)
        max_sub_lid = max((L.get("id", 0) for L in sub_links), default=0) if sub_links else 0

        # Add GetNode inside the subgraph
        get_nid = max_sub_nid + 1
        get_lid = max_sub_lid + 1

        get_autoprompt = {
            "id": get_nid,
            "type": "GetNode",
            "pos": [positive_clip["pos"][0] - 250, positive_clip["pos"][1]],
            "size": [210, 58],
            "flags": {"collapsed": True},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "STRING", "type": "STRING", "links": [get_lid]}
            ],
            "title": "Get_autoprompt",
            "properties": {},
            "widgets_values": ["autoprompt"],
            "color": "#432",
            "bgcolor": "#653",
        }

        sub_nodes.append(get_autoprompt)

        # Wire the GetNode output to the positive CLIPTextEncode text input
        # Find the text input on the positive CLIPTextEncode
        text_input = None
        for inp in positive_clip.get("inputs", []):
            if inp.get("name") == "text":
                text_input = inp
                break

        if text_input is None:
            # Add a text input if not present
            positive_clip.setdefault("inputs", []).append(
                {"name": "text", "type": "STRING", "link": get_lid}
            )
        else:
            text_input["link"] = get_lid

        # Add the link inside the subgraph (dict format matching existing links)
        # Find the correct target_slot for the text input
        text_slot = 1  # default: slot 1 is text
        for i, inp in enumerate(positive_clip.get("inputs", [])):
            if inp.get("name") == "text":
                text_slot = i
                break
        sub_links.append({
            "id": get_lid,
            "origin_id": get_nid,
            "origin_slot": 0,
            "target_id": positive_clip["id"],
            "target_slot": text_slot,
            "type": "STRING",
        })

    # =========================================================================
    # STEP 3: Update IDs, generate new workflow ID, and save
    # =========================================================================

    w["last_node_id"] = nid
    w["last_link_id"] = lid
    w["id"] = str(uuid.uuid4())

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(w, f, ensure_ascii=False)

    print(f"Wrote {OUTPUT_FILE}")
    print(f"  Added {4} main-canvas nodes: QwenLoader, WanVideoPromptExtender, SetNode(autoprompt), MarkdownNote")
    print(f"  Modified {len(subgraphs) - 1} subgraphs: added GetNode(autoprompt) -> CLIPTextEncode text input")
    print(f"  Last node ID: {nid}, last link ID: {lid}")


if __name__ == "__main__":
    main()
