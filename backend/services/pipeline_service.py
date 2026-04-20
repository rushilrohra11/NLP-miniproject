"""End-to-end clinical NLP pipeline helpers.

This module keeps the flow in one place:
Audio -> Speech-to-Text -> Summarization -> SOAP Note
"""

from __future__ import annotations

from typing import Any, Dict

from backend.services.nlp_service import summarize_clinical_text
from backend.services.rag_service import build_rag_payload
from backend.services.speech_service import speech_to_text
from utils.privacy import sanitize_text
from utils.soap_formatter import generate_soap
from utils.speaker_ner import extract_speaker_entities, normalize_transcription_dialogue


def process_text_pipeline(input_text: str) -> Dict[str, Any]:
    """Run the clinical NLP pipeline on raw text input."""
    sanitized_text = sanitize_text(input_text)
    rag_payload = build_rag_payload(sanitized_text)
    summary = summarize_clinical_text(sanitized_text)
    soap_note = generate_soap(summary or sanitized_text)

    return {
        "sanitized_text": sanitized_text,
        "transcription": None,
        "transcription_dialogue": None,
        "summary": summary,
        "soap_note": soap_note,
        "rag_context": rag_payload["prompt_context"],
        "rag_citations": rag_payload["citations"],
    }


def process_audio_pipeline(file_path: str) -> Dict[str, Any]:
    """Run the full clinical NLP pipeline on an audio file."""
    transcription = speech_to_text(file_path)
    transcription_dialogue = normalize_transcription_dialogue(transcription)
    transcription_entities = extract_speaker_entities(transcription)
    sanitized_transcription = sanitize_text(transcription)
    rag_payload = build_rag_payload(sanitized_transcription)
    summary_input = transcription_dialogue or transcription
    summary = summarize_clinical_text(summary_input)
    soap_note = generate_soap(summary or sanitized_transcription)

    return {
        "sanitized_text": sanitized_transcription,
        "transcription": transcription,
        "transcription_dialogue": transcription_dialogue,
        "transcription_entities": transcription_entities,
        "summary": summary,
        "soap_note": soap_note,
        "rag_context": rag_payload["prompt_context"],
        "rag_citations": rag_payload["citations"],
    }
