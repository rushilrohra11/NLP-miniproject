"""Standalone runner for the Clinical NLP pipeline.

Flow:
1. Sanitize text
2. Convert speech to text if audio is provided
3. Summarize
4. Generate SOAP note

The script prints clean JSON to stdout and optionally saves the result to a
file under outputs/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from backend.services.pipeline_service import process_audio_pipeline, process_text_pipeline
from utils.file_utils import ensure_directory


LOGGER = logging.getLogger("clinical_nlp_pipeline")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Clinical NLP pipeline on either raw text or an audio file.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Clinical text input to process.")
    group.add_argument("--audio", type=Path, help="Path to a wav/mp3 audio file.")

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON result. Defaults to stdout only.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of pretty-formatted JSON.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _validate_audio_path(audio_path: Path) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")
    if audio_path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
        raise ValueError("Unsupported audio format. Use wav, mp3, m4a, flac, or ogg.")
    return audio_path


def _run_pipeline(text: Optional[str] = None, audio: Optional[Path] = None) -> Dict[str, Any]:
    if text is not None:
        LOGGER.info("Running text pipeline")
        result = process_text_pipeline(text)
        result["input_type"] = "text"
        result["input_path"] = None
        return result

    if audio is not None:
        validated_audio = _validate_audio_path(audio)
        LOGGER.info("Running audio pipeline for %s", validated_audio)
        result = process_audio_pipeline(str(validated_audio))
        result["input_type"] = "audio"
        result["input_path"] = str(validated_audio)
        return result

    raise ValueError("Either text or audio input must be provided.")


def _write_output(output_path: Path, payload: Dict[str, Any]) -> None:
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = _run_pipeline(text=args.text, audio=args.audio)
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1

    json_kwargs = {"ensure_ascii": False}
    if not args.compact:
        json_kwargs["indent"] = 2

    rendered = json.dumps(result, **json_kwargs)
    print(rendered)

    if args.output:
        try:
            _write_output(args.output, result)
            LOGGER.info("Saved output to %s", args.output)
        except Exception as exc:
            LOGGER.error("Failed to write output file: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
