"""Privacy helpers for masking patient-identifiable information.

The `sanitize_text(text)` function uses a small NLP layer plus regex rules to
remove patient names and mask sensitive data such as phone numbers and IDs.
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy


NAME_REPLACEMENT = "[REDACTED_NAME]"
PHONE_REPLACEMENT = "[REDACTED_PHONE]"
ID_REPLACEMENT = "[REDACTED_ID]"


PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
ID_PATTERN = re.compile(
    r"\b(?:mrn|id|patient id|medical record number|member id)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
    re.IGNORECASE,
)
GENERIC_ID_PATTERN = re.compile(r"\b\d{6,}\b")
NAME_HINT_PATTERN = re.compile(
    r"\b(?:patient|name|mr\.?|mrs\.?|ms\.?|miss|dr\.?)\s*[:#-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_nlp():
    """Load a spaCy model when available, otherwise use a blank pipeline.

    The blank pipeline keeps the function usable even when the small English
    model is not installed.
    """
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


def _mask_regex_patterns(text: str) -> str:
    """Mask phone numbers and ID-like values using regex rules."""
    text = PHONE_PATTERN.sub(PHONE_REPLACEMENT, text)
    text = ID_PATTERN.sub(lambda match: re.sub(r"[A-Z0-9-]{4,}$", ID_REPLACEMENT, match.group(0)), text)
    text = GENERIC_ID_PATTERN.sub(ID_REPLACEMENT, text)
    return text


def _mask_names_with_nlp(text: str) -> str:
    """Use spaCy named entity recognition to remove PERSON names.

    When the full English model is present, PERSON entities are masked. The
    regex fallback also catches common label patterns like `Patient: John Doe`.
    """
    nlp = _load_nlp()
    doc = nlp(text)

    masked_text = text
    for ent in sorted(doc.ents, key=lambda item: item.start_char, reverse=True):
        if ent.label_ == "PERSON":
            masked_text = masked_text[: ent.start_char] + NAME_REPLACEMENT + masked_text[ent.end_char :]

    masked_text = NAME_HINT_PATTERN.sub(lambda match: match.group(0).replace(match.group(1), NAME_REPLACEMENT), masked_text)
    return masked_text


def sanitize_text(text: str) -> str:
    """Remove patient names and mask sensitive information.

    Args:
        text: Raw clinical text.

    Returns:
        Sanitized text with private data masked.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return ""

    cleaned_text = _mask_regex_patterns(cleaned_text)
    cleaned_text = _mask_names_with_nlp(cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text
