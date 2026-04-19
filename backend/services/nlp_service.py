"""Clinical NLP service functions used by the FastAPI routes."""

from typing import Any, Dict

from utils.privacy import sanitize_text
from utils.soap_formatter import generate_soap
from utils.summarizer import summarize_text


def summarize_clinical_text(input_text: str) -> str:
    """Summarize clinical text using HuggingFace Transformers."""
    cleaned_text = sanitize_text(input_text)
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
