from __future__ import annotations

import base64
import io
import json
import os
import re

import cv2
import numpy as np
from PIL import Image

from contentcheck.models.base import BaseChecker
from contentcheck.results import ModelResult

_PROMPT = """\
You are an expert at detecting anatomical abnormalities in images, \
especially those produced by AI image/video generators.

Carefully inspect every person visible in this image. Look for:
- Extra or missing fingers / toes
- Fingers merging, splitting, or bending unnaturally
- Extra or missing limbs
- Impossible joint angles or hyper-extended joints
- Severe left-right asymmetry in limb length or thickness
- Disproportionate body parts (e.g. tiny hands on a large body)
- Unnatural skin texture seams or blending artefacts around anatomy

Respond ONLY with valid JSON (no markdown fences). Use this schema:
{
  "score": <float 0.0–1.0, 0 = perfectly normal, 1 = clearly abnormal>,
  "anomalies": ["<short description>", ...],
  "explanation": "<one-sentence summary>"
}
If no people are visible, return {"score": 0.0, "anomalies": [], "explanation": "No people detected."}.
"""


def _frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0.5, "anomalies": ["LLM response could not be parsed"], "explanation": text[:200]}


# ──────────────────────────────────────────────────────────────────────
# Gemini
# ──────────────────────────────────────────────────────────────────────

class GeminiChecker(BaseChecker):
    name = "llm-gemini"

    def __init__(self, *, api_key: str | None = None, **_kwargs: object) -> None:
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "Gemini API key required. Pass --gemini-api-key or set GEMINI_API_KEY."
            )
        self._client = genai.Client(api_key=key)
        self._model = "gemini-3-flash-preview"

    def check(self, frame: np.ndarray) -> ModelResult:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        response = self._client.models.generate_content(
            model=self._model,
            contents=[_PROMPT, image],
        )
        parsed = _parse_llm_json(response.text)
        score = float(parsed.get("score", 0.5))
        anomalies = parsed.get("anomalies", [])
        explanation = parsed.get("explanation", "")
        return ModelResult(
            model_name=self.name,
            score=score,
            anomalies=anomalies,
            details=explanation,
        )


# ──────────────────────────────────────────────────────────────────────
# Grok (xAI — OpenAI-compatible API)
# ──────────────────────────────────────────────────────────────────────

class GrokChecker(BaseChecker):
    name = "llm-grok"

    def __init__(self, *, api_key: str | None = None, **_kwargs: object) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("GROK_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "Grok API key required. Pass --grok-api-key or set GROK_API_KEY."
            )
        self._client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    def check(self, frame: np.ndarray) -> ModelResult:
        b64 = base64.b64encode(_frame_to_jpeg_bytes(frame)).decode()
        response = self._client.chat.completions.create(
            model="grok-2-vision-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        text = response.choices[0].message.content or ""
        parsed = _parse_llm_json(text)
        score = float(parsed.get("score", 0.5))
        anomalies = parsed.get("anomalies", [])
        explanation = parsed.get("explanation", "")
        return ModelResult(
            model_name=self.name,
            score=score,
            anomalies=anomalies,
            details=explanation,
        )
