"""Clinical NLP service functions used by the FastAPI routes."""

import re
from typing import Any, Dict

from utils.privacy import sanitize_text
from utils.soap_formatter import generate_soap
from utils.summarizer import summarize_text, summarize_transcript_third_person


DIALOGUE_LINE_PATTERN = re.compile(
    r"^\s*(Doctor|Patient|Nurse|Accompanier|Support Staff|Other):\s*(.+)$",
    re.IGNORECASE,
)


def _sanitize_dialogue_preserve_labels(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    sanitized_lines = []

    for line in lines:
        match = DIALOGUE_LINE_PATTERN.match(line)
        if not match:
            cleaned = sanitize_text(line)
            if cleaned:
                sanitized_lines.append(f"Other: {cleaned}")
            continue

        speaker = match.group(1).strip().title()
        utterance = match.group(2).strip()
        cleaned_utterance = sanitize_text(utterance)
        if cleaned_utterance:
            sanitized_lines.append(f"{speaker}: {cleaned_utterance}")

    return "\n".join(sanitized_lines).strip()


def _looks_like_dialogue(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    labeled = sum(1 for line in lines if DIALOGUE_LINE_PATTERN.match(line))
    return labeled >= 2


def summarize_clinical_text(input_text: str) -> str:
    """Summarize clinical text using HuggingFace Transformers."""
    raw_text = input_text.strip()
    if not raw_text:
        return ""

    if _looks_like_dialogue(raw_text):
        sanitized_dialogue = _sanitize_dialogue_preserve_labels(raw_text)
        if not sanitized_dialogue:
            return ""
        return summarize_transcript_third_person(sanitized_dialogue)

    cleaned_text = sanitize_text(raw_text)
    if not cleaned_text:
        return ""

    return summarize_text(cleaned_text)


def generate_medical_note(input_text: str) -> Dict[str, Any]:
    """Generate a SOAP note from summarized or raw clinical text."""
    cleaned_text = sanitize_text(input_text)
    if not cleaned_text:
        return {"SOAP": {}, "ExtractedEntities": {}}

    summary = summarize_clinical_text(cleaned_text)
    source_text = summary or cleaned_text
    soap = generate_soap(source_text)

    return soap
