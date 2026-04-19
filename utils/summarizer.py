"""Text summarization helpers built on HuggingFace Transformers.

This module uses the `facebook/bart-large-cnn` model and splits long input
into smaller chunks so each part can be summarized safely.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List

from transformers import pipeline


logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-cnn"


@lru_cache(maxsize=1)
def _get_summarizer():
    """Load and cache the summarization pipeline."""
    try:
        logger.info("Loading summarization model...")
        return pipeline("summarization", model=MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load summarization model: {e}")
        raise


# Common clinical terms that should survive cleanup even if they look generic.
MEDICAL_KEYWORDS = {
    "bp",
    "cbc",
    "chest",
    "diagnosis",
    "dyspnea",
    "fever",
    "headache",
    "heart",
    "hypertension",
    "hyperglycemia",
    "icd",
    "infection",
    "labs",
    "medication",
    "medications",
    "nausea",
    "pain",
    "patient",
    "plan",
    "pulse",
    "respiratory",
    "sats",
    "shortness",
    "symptom",
    "symptoms",
    "temperature",
    "treatment",
    "vitals",
    "wbc",
}


NOISE_WORDS = {
    "actually",
    "basically",
    "kind",
    "like",
    "maybe",
    "note",
    "okay",
    "please",
    "sort",
    "stuff",
    "thanks",
    "thank",
    "um",
    "uh",
    "you know",
}


def _clean_medical_text(text: str) -> str:
    """Remove obvious noise while keeping clinically useful wording.

    The goal is not to rewrite the note. It is to strip filler words,
    timestamps, transcription artifacts, and extra punctuation so the model
    receives cleaner clinical input.
    """
    normalized = text.strip().lower()
    if not normalized:
        return ""

    # Remove common transcription and formatting noise.
    normalized = re.sub(r"\b(?:uh|um|erm|hmm|yeah|okay|ok)\b", " ", normalized)
    normalized = re.sub(r"\b(?:audio|transcript|transcription|dictated|recording)\b", " ", normalized)
    normalized = re.sub(r"\b(?:today|yesterday|tonight|morning|afternoon)\b", " ", normalized)
    normalized = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", normalized)
    normalized = re.sub(r"\[(?:noise|inaudible|crosstalk|silence)\]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s/%.,:+-]", " ", normalized)

    words = normalized.split()
    cleaned_words: List[str] = []

    for word in words:
        stripped_word = word.strip(".,:;+-/")
        if not stripped_word:
            continue

        if stripped_word in NOISE_WORDS:
            continue

        # Keep important medical terms even when they are short or generic.
        if stripped_word in MEDICAL_KEYWORDS:
            cleaned_words.append(stripped_word)
            continue

        # Drop repeated filler fragments and very short non-medical noise.
        if len(stripped_word) <= 1:
            continue

        cleaned_words.append(stripped_word)

    cleaned_text = " ".join(cleaned_words)
    cleaned_text = re.sub(r"\s+([.,:;])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def _split_text_into_chunks(text: str, max_words: int = 450) -> List[str]:
    """Split text into word-based chunks.

    BART has a token limit, so this keeps each chunk small enough to process
    reliably while staying simple and easy to follow.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start : start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _clean_summary(summary: str) -> str:
    """Normalize whitespace in generated summaries."""
    cleaned = " ".join(summary.split()).strip()
    cleaned = re.sub(r"\s+([.,:;])", r"\1", cleaned)
    return cleaned


def summarize_text(text: str) -> str:
    """Summarize text using `facebook/bart-large-cnn`.

    If the input is long, the text is chunked and each chunk is summarized
    separately. The final summary is a clean combination of all chunk summaries.
    """
    cleaned_text = _clean_medical_text(text)
    if not cleaned_text:
        return ""

    try:
        summarizer = _get_summarizer()
        chunks = _split_text_into_chunks(cleaned_text)

        if not chunks:
            return ""

        summaries: List[str] = []
        for chunk in chunks:
            result = summarizer(
                chunk,
                max_length=110,
                min_length=25,
                do_sample=False,
            )
            if result and "summary_text" in result[0]:
                summaries.append(_clean_summary(result[0]["summary_text"]))

        final_summary = _clean_summary(" ".join(summaries))

        # Keep the final output concise by trimming excess repeated whitespace and
        # returning the first meaningful summary string.
        return final_summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return cleaned_text
