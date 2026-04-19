"""SOAP note generation from summarized medical text.

The `generate_soap(summary_text)` function uses a simple rule-based approach
plus spaCy entity extraction to map symptoms, medicines, and conditions into
SOAP sections. It returns JSON-ready structured output.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List

import spacy
from spacy.language import Language

from utils.text_utils import normalize_text


SUBJECTIVE_HINTS = {
    "complaint",
    "complaints",
    "denies",
    "history",
    "hx",
    "mild",
    "pain",
    "patient reports",
    "presented",
    "reports",
    "says",
    "symptom",
    "symptoms",
    "worsening",
}

OBJECTIVE_HINTS = {
    "bp",
    "cbc",
    "exam",
    "findings",
    "labs",
    "measurement",
    "observed",
    "oxygen",
    "pulse",
    "temp",
    "temperature",
    "vital",
    "vitals",
    "wbc",
}

ASSESSMENT_HINTS = {
    "assessment",
    "diagnosis",
    "likely",
    "possible",
    "suspect",
    "consistent with",
    "impression",
    "concern for",
}

PLAN_HINTS = {
    "admit",
    "follow up",
    "follow-up",
    "medication",
    "monitor",
    "plan",
    "prescribe",
    "repeat",
    "start",
    "treatment",
    "therapy",
    "titrate",
}


KEY_MEDICAL_TERMS = {
    "pain",
    "fever",
    "cough",
    "dyspnea",
    "shortness of breath",
    "chest pain",
    "hypertension",
    "diabetes",
    "infection",
    "nausea",
    "vomiting",
    "headache",
    "labs",
    "vitals",
    "bp",
    "hr",
    "rr",
    "spo2",
    "wbc",
    "cbc",
}


SYMPTOM_KEYWORDS = {
    "pain",
    "fever",
    "cough",
    "dyspnea",
    "shortness of breath",
    "nausea",
    "vomiting",
    "headache",
    "fatigue",
    "dizziness",
    "chest pain",
}

MEDICINE_KEYWORDS = {
    "acetaminophen",
    "advil",
    "amoxicillin",
    "aspirin",
    "ibuprofen",
    "insulin",
    "metformin",
    "prednisone",
    "statin",
    "tylenol",
}

CONDITION_KEYWORDS = {
    "asthma",
    "copd",
    "diabetes",
    "fever",
    "hypertension",
    "infection",
    "pneumonia",
    "sepsis",
}


def _load_nlp() -> Language:
    """Load spaCy if the small English model is available.

    If the model is missing, fall back to a blank English pipeline with a
    sentence segmenter so the function still works without extra downloads.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    _add_medical_entity_ruler(nlp)
    return nlp


def _add_medical_entity_ruler(nlp: Language) -> None:
    """Add a small entity ruler so medical terms get explicit labels.

    This keeps the NLP logic easy to explain and avoids depending on a large
    medical model for a beginner-friendly project.
    """
    if "medical_entity_ruler" in nlp.pipe_names:
        return

    if "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner", name="medical_entity_ruler")
    else:
        ruler = nlp.add_pipe("entity_ruler", name="medical_entity_ruler")

    patterns = []
    for term in sorted(SYMPTOM_KEYWORDS):
        patterns.append({"label": "SYMPTOM", "pattern": term})
    for term in sorted(MEDICINE_KEYWORDS):
        patterns.append({"label": "MEDICINE", "pattern": term})
    for term in sorted(CONDITION_KEYWORDS):
        patterns.append({"label": "CONDITION", "pattern": term})

    ruler.add_patterns(patterns)


def _split_sentences(text: str) -> List[str]:
    nlp = _load_nlp()
    doc = nlp(text)
    sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text.strip() else []


def _is_subjective(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(hint in lowered for hint in SUBJECTIVE_HINTS)


def _is_objective(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(hint in lowered for hint in OBJECTIVE_HINTS) or bool(
        re.search(r"\b\d+(?:\.\d+)?\s?(?:mg|ml|mmhg|bpm|c|f|%)\b", lowered)
    )


def _is_assessment(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(hint in lowered for hint in ASSESSMENT_HINTS)


def _is_plan(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(hint in lowered for hint in PLAN_HINTS)


def _extract_relevant_sentences(sentences: List[str], predicate) -> List[str]:
    return [sentence for sentence in sentences if predicate(sentence)]


def _extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Extract symptoms, medicines, and conditions with spaCy entities.

    The entity ruler adds explicit medical labels, which keeps the logic simple
    and explainable for a beginner-friendly project.
    """
    nlp = _load_nlp()
    doc = nlp(text)

    extracted: Dict[str, List[str]] = {
        "Symptoms": [],
        "Medicines": [],
        "Conditions": [],
    }

    for ent in doc.ents:
        value = _dedupe_and_clean(ent.text.lower())
        if not value:
            continue

        if ent.label_ == "SYMPTOM":
            extracted["Symptoms"].append(value)
        elif ent.label_ == "MEDICINE":
            extracted["Medicines"].append(value)
        elif ent.label_ == "CONDITION":
            extracted["Conditions"].append(value)

    # Keep the lists unique while preserving order.
    for key, values in extracted.items():
        extracted[key] = list(OrderedDict.fromkeys(values))

    return extracted


def _fallback_section(sentences: List[str], section_name: str) -> str:
    """Use a simple keyword fallback when no rule matched a section.

    This keeps the output meaningful even for short summaries that do not have
    obvious SOAP cues.
    """
    lowered_sentences = [sentence.lower() for sentence in sentences]

    if section_name == "Subjective":
        for sentence in sentences:
            if any(term in sentence.lower() for term in KEY_MEDICAL_TERMS):
                return sentence
    elif section_name == "Objective":
        for sentence in sentences:
            if re.search(r"\b\d+(?:\.\d+)?\b", sentence):
                return sentence
    elif section_name == "Assessment":
        for sentence in sentences:
            if any(word in sentence.lower() for word in {"diagnosis", "likely", "possible", "suggests"}):
                return sentence
    elif section_name == "Plan":
        for sentence in sentences:
            if any(word in sentence.lower() for word in {"plan", "follow", "start", "monitor", "treat"}):
                return sentence

    if lowered_sentences:
        return sentences[0]
    return ""


def _join_items(items: List[str]) -> str:
    if not items:
        return ""
    return ", ".join(items)


def _sentence_for_entities(sentences: List[str], entities: List[str]) -> str:
    """Pick a concise sentence that mentions extracted entities.

    This helps keep the SOAP sections readable while still preserving the
    most important clinical terms.
    """
    if not entities:
        return ""

    for sentence in sentences:
        lowered = sentence.lower()
        if any(entity in lowered for entity in entities):
            return sentence
    return ""


def _dedupe_and_clean(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = cleaned.strip(" ,.;:")
    return cleaned


def _concise_summary(text: str, max_sentences: int = 2) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    compact = " ".join(sentences[:max_sentences])
    return _dedupe_and_clean(compact)


def _format_section(sentences: List[str], section_name: str, predicate) -> str:
    extracted = _extract_relevant_sentences(sentences, predicate)
    if not extracted:
        extracted = [_fallback_section(sentences, section_name)]

    filtered = [_dedupe_and_clean(sentence) for sentence in extracted if sentence.strip()]
    filtered = [sentence for sentence in filtered if sentence]
    if not filtered:
        return ""

    return _concise_summary(" ".join(filtered), max_sentences=2)


def _map_to_soap(sentences: List[str], entities: Dict[str, List[str]]) -> Dict[str, str]:
    """Map extracted entities into SOAP sections using simple rules."""
    symptom_sentence = _sentence_for_entities(sentences, entities["Symptoms"])
    medicine_sentence = _sentence_for_entities(sentences, entities["Medicines"])
    condition_sentence = _sentence_for_entities(sentences, entities["Conditions"])

    subjective_parts = [
        _format_section(sentences, "Subjective", _is_subjective),
        _join_items(entities["Symptoms"]),
        symptom_sentence,
    ]
    objective_parts = [
        _format_section(sentences, "Objective", _is_objective),
        _join_items(entities["Medicines"]),
    ]
    assessment_parts = [
        _format_section(sentences, "Assessment", _is_assessment),
        _join_items(entities["Conditions"]),
        condition_sentence,
    ]
    plan_parts = [
        _format_section(sentences, "Plan", _is_plan),
        medicine_sentence,
        _join_items(entities["Medicines"]),
    ]

    def combine(parts: List[str]) -> str:
        filtered = [_dedupe_and_clean(part) for part in parts if part and _dedupe_and_clean(part)]
        if not filtered:
            return ""
        return _concise_summary(". ".join(filtered), max_sentences=2)

    return {
        "Subjective": combine(subjective_parts),
        "Objective": combine(objective_parts),
        "Assessment": combine(assessment_parts),
        "Plan": combine(plan_parts),
    }


def generate_soap(summary_text: str) -> Dict[str, str]:
    """Convert summarized medical text into SOAP format.

    Args:
        summary_text: A summarized clinical note or patient summary.

    Returns:
        JSON-ready dictionary with SOAP sections.
    """
    cleaned_text = _dedupe_and_clean(summary_text)
    if not cleaned_text:
        return {
            "Subjective": "",
            "Objective": "",
            "Assessment": "",
            "Plan": "",
        }

    sentences = _split_sentences(cleaned_text)
    if not sentences:
        sentences = [cleaned_text]

    extracted_entities = _extract_medical_entities(cleaned_text)
    soap = _map_to_soap(sentences, extracted_entities)

    # If a section is still empty, use a simple fallback so the output stays
    # usable even for short or noisy summaries.
    if not soap["Subjective"]:
        soap["Subjective"] = _fallback_section(sentences, "Subjective")
    if not soap["Objective"]:
        soap["Objective"] = _fallback_section(sentences, "Objective")
    if not soap["Assessment"]:
        soap["Assessment"] = _fallback_section(sentences, "Assessment")
    if not soap["Plan"]:
        soap["Plan"] = _fallback_section(sentences, "Plan")

    return {
        "SOAP": {
            "Subjective": _concise_summary(soap["Subjective"], max_sentences=2),
            "Objective": _concise_summary(soap["Objective"], max_sentences=2),
            "Assessment": _concise_summary(soap["Assessment"], max_sentences=2),
            "Plan": _concise_summary(soap["Plan"], max_sentences=2),
        },
        "ExtractedEntities": extracted_entities,
    }
