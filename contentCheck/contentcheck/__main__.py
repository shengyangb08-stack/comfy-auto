"""CLI entry-point: python -m contentcheck <input> [options]"""
from __future__ import annotations

import argparse
import logging
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from contentcheck.models import MODEL_REGISTRY
from contentcheck.models.base import BaseChecker
from contentcheck.pipeline import is_image, run, run_image


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
        default=["mediapipe"],
        help="Which detection models to run (default: mediapipe). "
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
    checkers: list[BaseChecker] = []
    for name in args.models:
        cls = MODEL_REGISTRY[name]
        kwargs: dict = {}
        if name == "llm-gemini" and args.gemini_api_key:
            kwargs["api_key"] = args.gemini_api_key
        elif name == "llm-grok" and args.grok_api_key:
            kwargs["api_key"] = args.grok_api_key
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
