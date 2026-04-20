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

SPEAKER_LINE_PATTERN = re.compile(
    r"^\s*(Doctor|Patient|Nurse|Accompanier|Support Staff|Other):\s*(.+)$",
    re.IGNORECASE,
)


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


def _third_personize(text: str) -> str:
    """Convert common first-person patient phrasing into third-person narrative."""
    converted = f" {text.strip()} "
    replacements = [
        (r"\bI am\b", "the patient is"),
        (r"\bI'm\b", "the patient is"),
        (r"\bI have\b", "the patient has"),
        (r"\bI've had\b", "the patient has had"),
        (r"\bI had\b", "the patient had"),
        (r"\bI feel\b", "the patient feels"),
        (r"\bI felt\b", "the patient felt"),
        (r"\bI took\b", "the patient took"),
        (r"\bI was\b", "the patient was"),
        (r"\bmy\b", "the patient's"),
        (r"\bme\b", "the patient"),
        (r"\bI\b", "the patient"),
    ]
    for pattern, replacement in replacements:
        converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
    return _clean_summary(converted.strip())


def _clip_text(text: str, max_words: int = 18) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def summarize_transcript_third_person(transcript_dialogue: str) -> str:
    """Create a concise third-person summary from speaker-labeled dialogue."""
    lines = [line.strip() for line in transcript_dialogue.splitlines() if line.strip()]
    if not lines:
        return ""

    turns = []
    for line in lines:
        match = SPEAKER_LINE_PATTERN.match(line)
        if not match:
            continue
        speaker = match.group(1).lower()
        utterance = match.group(2).strip()
        if utterance:
            turns.append((speaker, utterance))

    if not turns:
        return _extractive_fallback_summary(transcript_dialogue)

    patient_turns = [u for s, u in turns if s == "patient"]
    doctor_turns = [u for s, u in turns if s == "doctor"]
    nurse_turns = [u for s, u in turns if s == "nurse"]
    accompanier_turns = [u for s, u in turns if s == "accompanier"]
    support_turns = [u for s, u in turns if s == "support staff"]

    summary_lines: List[str] = []

    if patient_turns:
        patient_text = _third_personize(" ".join(patient_turns[:2]))
        summary_lines.append(f"The patient reported that {_clip_text(patient_text)}.")

    if doctor_turns:
        doctor_text = _clean_summary(" ".join(doctor_turns[:2]))
        summary_lines.append(f"The doctor discussed evaluation and management, including {_clip_text(doctor_text)}.")

    if nurse_turns:
        nurse_text = _clean_summary(" ".join(nurse_turns[:1]))
        summary_lines.append(f"The nurse contributed clinical observations: {_clip_text(nurse_text, max_words=14)}.")

    if accompanier_turns:
        acc_text = _third_personize(" ".join(accompanier_turns[:1]))
        summary_lines.append(f"An accompanier added collateral information: {_clip_text(acc_text, max_words=14)}.")

    if support_turns:
        support_text = _clean_summary(" ".join(support_turns[:1]))
        summary_lines.append(f"Support staff assisted with logistics: {_clip_text(support_text, max_words=14)}.")

    if not summary_lines:
        return _extractive_fallback_summary(transcript_dialogue)

    return _clean_summary(" ".join(summary_lines))


def _extractive_fallback_summary(text: str, max_sentences: int = 2, max_words: int = 45) -> str:
    """Return a concise fallback summary when model inference is unavailable.

    This prevents returning the full input/transcription as a pseudo-summary.
    """
    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

    if not sentence_parts:
        words = text.split()
        dynamic_limit = min(max_words, max(12, int(len(words) * 0.6)))
        clipped = " ".join(words[:dynamic_limit]).strip()
        return _clean_summary(clipped)

    selected = " ".join(sentence_parts[:max_sentences]).strip()
    words = selected.split()
    dynamic_limit = min(max_words, max(12, int(len(text.split()) * 0.6)))
    if len(words) > dynamic_limit:
        selected = " ".join(words[:dynamic_limit]).strip()

    return _clean_summary(selected)


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

        # Guard against degenerate outputs where model returns near-input text.
        if not final_summary:
            return _extractive_fallback_summary(cleaned_text)

        input_words = cleaned_text.split()
        summary_words = final_summary.split()
        if len(input_words) > 0 and len(summary_words) >= int(0.9 * len(input_words)):
            return _extractive_fallback_summary(cleaned_text)

        # Keep the final output concise by trimming excess repeated whitespace and
        # returning the first meaningful summary string.
        return final_summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return _extractive_fallback_summary(cleaned_text)
