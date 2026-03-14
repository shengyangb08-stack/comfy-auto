"""
Generate a second-by-second video prompt from an image using Gemini or Grok vision API.

Usage:
    python autoprompt.py <image_path>
    python autoprompt.py <image_path> --duration 10
    python autoprompt.py <image_path> --provider grok

Reads API keys from api_keys.json in the same directory.
"""

import argparse
import base64
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_KEYS_FILE = os.path.join(_SCRIPT_DIR, "api_keys.json")

_STABLENESS_INSTRUCTIONS = {
    1: "Only describe the first second (At 0 seconds: ...) in detail. For seconds 1-5, write brief lines like (At 1 seconds: Scene holds, no significant change.) (At 2 seconds: Same, static.) — the whole 5 seconds should feel mostly static with minimal movement.",
    2: "Describe seconds 0 and 1 in detail. For seconds 2-5, write that the scene holds or has minimal change.",
    3: "Describe seconds 0, 1, 2 in detail. For seconds 3-5, minimal change or scene holds.",
    4: "Describe seconds 0, 1, 2, 3 in detail. For seconds 4-5, minimal or brief wrap-up.",
    5: "Describe all seconds 0-5 with full detail and movement.",
}

_EXCITEMENT_GUIDANCE = {
    0: "Very calm, minimal movement, subtle, mostly static. Avoid dramatic or vivid descriptions.",
    1: "Very calm, minimal movement, subtle, mostly static. Avoid dramatic or vivid descriptions.",
    2: "Very calm, minimal movement, subtle, mostly static. Avoid dramatic or vivid descriptions.",
    3: "Low energy, gentle, subtle movements only.",
    4: "Low energy, gentle, subtle movements only.",
    5: "Moderate energy, natural movement, balanced.",
    6: "Moderate energy, natural movement, balanced.",
    7: "High energy, dynamic, vivid, noticeable movement.",
    8: "High energy, dynamic, vivid, noticeable movement.",
    9: "Very high energy, dramatic, intense, vivid movement.",
    10: "Very high energy, dramatic, intense, vivid movement.",
}


def _get_excitement_guidance(excitement: int) -> str:
    return _EXCITEMENT_GUIDANCE.get(max(0, min(10, excitement)), _EXCITEMENT_GUIDANCE[5])


def _get_stableness_instruction(stableness: int) -> str:
    return _STABLENESS_INSTRUCTIONS.get(max(1, min(5, stableness)), _STABLENESS_INSTRUCTIONS[3])


_SYSTEM_PROMPT = """\
You are a professional AI video prompt engineer. You specialize in writing \
second-by-second descriptions for image-to-video generation models (like Wan2.2 / Kling / Runway).

Your job: given a still image, write a {duration}-second video description that brings the scene to life.

CONTROL PARAMETERS (follow these strictly):
- Excitement level: {excitement_guidance}
- Stableness / movement scope: {stableness_instruction}

CAMERA — CRITICAL: Keep the camera completely stable. Never change the angle, zoom in, \
zoom out, pan, or move the camera. Describe the scene as if viewed from a single, \
fixed camera position throughout. Only the subjects and environment may move — not \
the camera. This is essential for quality when chaining segments.

FORMAT — you MUST follow this exact structure:
- One line per second, starting from 0 up to {duration}.
- Each line MUST begin with "(At N second:" or "(At N seconds:" and end with ")".
- Inside each line, describe: camera angle/movement, subject actions, body dynamics, \
  environmental details, lighting, and any subtle motion (hair, fabric, skin, etc.).
- Be explicit and anatomically precise for NSFW content — the model needs \
  exact descriptions of body positions, movements, and interactions.
- Write in present tense, cinematic style.
- Output ONLY the timestamped lines — no preamble, no explanation, no markdown, \
  no extra text before or after.

Here are two examples:

--- Example 1 (SFW, 5 seconds) ---
(At 0 seconds: Wide shot showing a slightly overweight man casually walking down a city street, camera fixed in front, urban environment with buildings and cars.)
(At 1 second: Suddenly, a massive shark bursts from the pavement ahead, looking terrifying at first, pavement cracking, dust and debris flying, camera from side angle.)
(At 2 seconds: Medium shot from the side, the man stumbles backward in shock, while the shark dramatically slows down and strikes a comically exaggerated sexy pose, revealing large, exaggerated shark breasts, covered by a colorful bikini.)
(At 3 seconds: Close-up on the man's face, eyes wide in disbelief, as he turns to look at the shark, small cartoon-style hearts floating above his head to emphasize his amazement, camera slightly low-angle.)
(At 4 seconds: Dynamic travelling shot showing the man frozen in the street, the shark maintaining its sexy pose, water splashes and debris still moving realistically, urban chaos around.)
(At 5 seconds: Wide cinematic shot pulling back, showing the man standing in the street, staring at the bikini-wearing shark with hearts above his head, epic perspective highlighting absurdity and humor.)

--- Example 2 (NSFW, 5 seconds) ---
(At 0 seconds: The couple in a cozy bedroom, anime style, soft lighting highlighting their intimate embrace, her back arched slightly as he positions himself.)
(At 1 second: The man's hips moving rhythmically, the head of his penis sliding effortlessly into her vagina, her body responding with a gentle, fluid motion, anime-style motion lines emphasizing the smooth penetration.)
(At 2 seconds: Her back arching deeply against him to intensify the pleasure, hips swaying with each thrust, breasts bouncing subtly, small hearts floating around them to capture the erotic energy.)
(At 3 seconds: Her face, eyes closed in bliss, a soft moan escaping, hands resting behind her head, anime-style blush on her cheeks, the air filled with a seductive aura.)
(At 4 seconds: The man penetrating her deeply, her body moving in sync with his, the bed sheets slightly rumpled, the room's warm lighting enhancing the intimate, lustful atmosphere.)
(At 5 seconds: The couple locked in a passionate embrace, the scene exuding vibrant, seductive energy, anime style with smooth lines and soft shadows.)
"""


def load_api_keys() -> dict[str, str]:
    if os.path.isfile(_KEYS_FILE):
        with open(_KEYS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def image_to_base64(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".gif": "image/gif", ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _build_system_prompt(duration: int, excitement: int, stableness: int) -> str:
    return _SYSTEM_PROMPT.format(
        duration=duration,
        excitement_guidance=_get_excitement_guidance(excitement),
        stableness_instruction=_get_stableness_instruction(stableness),
    )


def generate_prompt_grok(
    image_path: str,
    duration: int,
    api_key: str,
    excitement: int = 5,
    stableness: int = 3,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    b64, mime = image_to_base64(image_path)
    system = _build_system_prompt(duration, excitement, stableness)

    response = client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Generate a {duration}-second video prompt from this image (excitement={excitement}, stableness={stableness})."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


def generate_prompt_gemini(
    image_path: str,
    duration: int,
    api_key: str,
    excitement: int = 5,
    stableness: int = 3,
) -> str:
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=api_key)
    image = Image.open(image_path)
    system = _build_system_prompt(duration, excitement, stableness)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            system,
            f"Generate a {duration}-second video prompt from this image (excitement={excitement}, stableness={stableness}).",
            image,
        ],
    )
    return response.text.strip()


def generate_prompt(
    image_path: str,
    duration: int = 5,
    provider: str = "gemini",
    excitement: int = 5,
    stableness: int = 3,
) -> str:
    keys = load_api_keys()

    if provider == "grok":
        key = keys.get("grok") or os.environ.get("GROK_API_KEY", "")
        if not key:
            print("ERROR: Grok API key not found.", file=sys.stderr)
            sys.exit(1)
        return generate_prompt_grok(image_path, duration, key, excitement, stableness)
    else:
        key = keys.get("gemini") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("ERROR: Gemini API key not found.", file=sys.stderr)
            sys.exit(1)
        return generate_prompt_gemini(image_path, duration, key, excitement, stableness)


def main():
    parser = argparse.ArgumentParser(
        prog="autoprompt",
        description="Generate a second-by-second video prompt from an image.",
    )
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--duration", type=int, default=5,
                        help="Video duration in seconds (default: 5).")
    parser.add_argument("--provider", choices=["grok", "gemini"], default="gemini",
                        help="LLM provider (default: gemini).")
    parser.add_argument("--excitement", type=int, default=5, metavar="0-10",
                        help="Excitement level 0-10, higher = more dynamic (default: 5).")
    parser.add_argument("--stableness", type=int, default=3, metavar="1-5",
                        help="Stableness 1-5, 1=only first second described (default: 3).")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Image:      {args.image}")
    print(f"Duration:   {args.duration}s")
    print(f"Provider:   {args.provider}")
    print(f"Excitement: {args.excitement}/10")
    print(f"Stableness: {args.stableness}/5")
    print("Generating prompt...\n")

    prompt = generate_prompt(
        args.image, args.duration, args.provider,
        excitement=args.excitement, stableness=args.stableness,
    )
    print(prompt)


if __name__ == "__main__":
    main()
