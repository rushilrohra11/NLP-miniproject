"""End-to-end clinical NLP pipeline helpers.

This module keeps the flow in one place:
Audio -> Speech-to-Text -> Summarization -> SOAP Note
"""

from __future__ import annotations

from typing import Any, Dict

from backend.services.nlp_service import summarize_clinical_text
from backend.services.speech_service import speech_to_text
from utils.privacy import sanitize_text
from utils.soap_formatter import generate_soap


def process_text_pipeline(input_text: str) -> Dict[str, Any]:
    """Run the clinical NLP pipeline on raw text input."""
    sanitized_text = sanitize_text(input_text)
    summary = summarize_clinical_text(sanitized_text)
    soap_note = generate_soap(summary or sanitized_text)

    return {
        "sanitized_text": sanitized_text,
        "transcription": None,
        "summary": summary,
        "soap_note": soap_note,
    }


def process_audio_pipeline(file_path: str) -> Dict[str, Any]:
    """Run the full clinical NLP pipeline on an audio file."""
    transcription = speech_to_text(file_path)
    sanitized_transcription = sanitize_text(transcription)
    summary = summarize_clinical_text(sanitized_transcription)
    soap_note = generate_soap(summary or sanitized_transcription)

    return {
        "sanitized_text": sanitized_transcription,
        "transcription": transcription,
        "summary": summary,
        "soap_note": soap_note,
    }
