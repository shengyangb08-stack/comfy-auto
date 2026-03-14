"""CLI entry-point: python -m contentcheck <input> [options]"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from contentcheck.models import MODEL_REGISTRY
from contentcheck.models.base import BaseChecker
from contentcheck.pipeline import is_image, run, run_image

_KEYS_FILENAME = "api_keys.json"


def _load_api_keys() -> dict[str, str]:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.getcwd(),
        os.path.dirname(_this_dir),                # contentCheck/
        os.path.dirname(os.path.dirname(_this_dir)),  # comfy_auto/
    ]
    for d in search_dirs:
        path = os.path.join(d, _KEYS_FILENAME)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="contentcheck",
        description="Detect abnormal human anatomy in images or video frames.",
    )
    p.add_argument("input", help="Path to input video or image file.")
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=["llm-gemini"],
        help="Which detection models to run (default: llm-gemini). "
             f"Available: {', '.join(MODEL_REGISTRY.keys())}",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for flagging a frame (default: 0.5).",
    )
    p.add_argument(
        "--output",
        default="output",
        help="Directory to save flagged frames (default: output/).",
    )
    p.add_argument(
        "--save-frames",
        default=None,
        metavar="DIR",
        help="Save all extracted frames to this directory (for inspection).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
    )
    p.add_argument("--gemini-api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var).")
    p.add_argument("--grok-api-key", default=None, help="Grok API key (or set GROK_API_KEY env var).")
    return p


def _instantiate_checkers(args: argparse.Namespace) -> list[BaseChecker]:
    keys = _load_api_keys()
    checkers: list[BaseChecker] = []
    for name in args.models:
        cls = MODEL_REGISTRY[name]
        kwargs: dict = {}
        if name == "llm-gemini":
            kwargs["api_key"] = args.gemini_api_key or keys.get("gemini") or None
        elif name == "llm-grok":
            kwargs["api_key"] = args.grok_api_key or keys.get("grok") or None
        try:
            checkers.append(cls(**kwargs))
        except Exception as exc:
            print(f"Error initialising model '{name}': {exc}", file=sys.stderr)
            sys.exit(1)
    return checkers


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="  %(levelname)-7s [%(name)s] %(message)s",
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    checkers = _instantiate_checkers(args)
    if is_image(args.input):
        run_image(
            image_path=args.input,
            checkers=checkers,
            threshold=args.threshold,
            output_dir=args.output,
        )
    else:
        run(
            video_path=args.input,
            checkers=checkers,
            threshold=args.threshold,
            output_dir=args.output,
            save_frames_dir=args.save_frames,
        )


if __name__ == "__main__":
    main()
