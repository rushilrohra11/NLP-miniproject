"""KSOAP note generation from summarized clinical text.

This module builds a clinically structured note using a rule-based KSOAP
formatter and returns extracted entities for symptoms, medicines, and
conditions.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Tuple

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


SYMPTOM_TERMS = {
    "abdominal pain",
    "back pain",
    "body ache",
    "chest pain",
    "chills",
    "cough",
    "diarrhea",
    "dizziness",
    "dyspnea",
    "fatigue",
    "fever",
    "headache",
    "joint pain",
    "loss of appetite",
    "nausea",
    "palpitations",
    "rash",
    "runny nose",
    "shortness of breath",
    "sore throat",
    "vomiting",
    "wheezing",
}

CONDITION_TERMS = {
    "allergic rhinitis",
    "anemia",
    "asthma",
    "bronchitis",
    "copd",
    "dehydration",
    "diabetes",
    "fever",
    "gastritis",
    "hypertension",
    "infection",
    "migraine",
    "pneumonia",
    "sepsis",
    "sinusitis",
    "upper respiratory infection",
    "urinary tract infection",
    "viral illness",
}

MEDICATION_TERMS = {
    "acetaminophen",
    "advil",
    "albuterol",
    "amlodipine",
    "amoxicillin",
    "aspirin",
    "atorvastatin",
    "azithromycin",
    "cetirizine",
    "doxycycline",
    "ibuprofen",
    "insulin",
    "lisinopril",
    "losartan",
    "metformin",
    "naproxen",
    "omeprazole",
    "ondansetron",
    "paracetamol",
    "prednisone",
    "salbutamol",
    "tylenol",
}

NEGATION_CUES = {
    "denies",
    "denied",
    "no",
    "without",
    "negative for",
}

DOSAGE_PATTERN = re.compile(
    r"\b([a-z][a-z0-9\-]{2,})\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|iu)\b",
    re.IGNORECASE,
)

STOP_MEDICATION_TOKENS = {
    "blood",
    "heart",
    "level",
    "monitor",
    "patient",
    "plan",
    "reports",
    "symptoms",
    "taking",
    "treatment",
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
    for term in sorted(SYMPTOM_TERMS):
        patterns.append({"label": "SYMPTOM", "pattern": term})
    for term in sorted(MEDICATION_TERMS):
        patterns.append({"label": "MEDICINE", "pattern": term})
    for term in sorted(CONDITION_TERMS):
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


def _contains_negation(sentence: str, term: str) -> bool:
    lowered = sentence.lower()
    term_index = lowered.find(term)
    if term_index < 0:
        return False

    window_start = max(0, term_index - 30)
    context_window = lowered[window_start:term_index]
    return any(cue in context_window for cue in NEGATION_CUES)


def _collect_terms_in_order(text: str, terms: set[str]) -> List[str]:
    matches: List[Tuple[int, str]] = []
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            matches.append((m.start(), term.lower()))
    matches.sort(key=lambda item: item[0])
    return [term for _, term in matches]


def _extract_symptoms(text: str, sentences: List[str]) -> Tuple[List[str], List[str]]:
    positive: List[str] = []
    negated: List[str] = []

    for term in _collect_terms_in_order(text, SYMPTOM_TERMS):
        detected_negated = False
        for sentence in sentences:
            lowered = sentence.lower()
            if term in lowered and _contains_negation(sentence, term):
                detected_negated = True
                break
        if detected_negated:
            negated.append(term)
        else:
            positive.append(term)

    return list(OrderedDict.fromkeys(positive)), list(OrderedDict.fromkeys(negated))


def _extract_medications(text: str) -> List[str]:
    found = _collect_terms_in_order(text, MEDICATION_TERMS)

    for match in DOSAGE_PATTERN.finditer(text):
        token = match.group(1).strip().lower()
        if token in STOP_MEDICATION_TOKENS:
            continue
        found.append(token)

    return list(OrderedDict.fromkeys(found))


def _extract_conditions(text: str) -> List[str]:
    found = _collect_terms_in_order(text, CONDITION_TERMS)
    return list(OrderedDict.fromkeys(found))


def _extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Extract symptoms, medicines, and conditions with clinical rules.

    The extractor combines lexicon-based phrase matching, dosage-aware
    medication capture, and simple negation handling for symptom quality.
    """
    cleaned_text = normalize_text(text)
    sentences = _split_sentences(cleaned_text)

    symptoms, negated_symptoms = _extract_symptoms(cleaned_text, sentences)
    medicines = _extract_medications(cleaned_text)
    conditions = _extract_conditions(cleaned_text)

    return {
        "Symptoms": symptoms,
        "NegatedSymptoms": negated_symptoms,
        "Medicines": medicines,
        "Conditions": conditions,
    }


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


def _build_key_data(entities: Dict[str, List[str]]) -> str:
    symptom_text = _join_items(entities.get("Symptoms", [])) or "not clearly documented"
    medicine_text = _join_items(entities.get("Medicines", [])) or "none documented"
    condition_text = _join_items(entities.get("Conditions", [])) or "working diagnosis pending"
    negated = entities.get("NegatedSymptoms", [])
    negated_text = _join_items(negated)

    segments = [
        f"Key symptoms: {symptom_text}",
        f"Current medications: {medicine_text}",
        f"Clinical conditions: {condition_text}",
    ]
    if negated_text:
        segments.append(f"Symptoms denied: {negated_text}")
    return "; ".join(segments)


def _ensure_section_defaults(soap: Dict[str, str], entities: Dict[str, List[str]]) -> Dict[str, str]:
    subjective_default = "Patient-reported symptoms are limited in the available transcript."
    objective_default = "Objective vitals or laboratory data were not explicitly documented."
    assessment_default = "Clinical impression remains preliminary based on available details."
    plan_default = "Continue monitoring and follow up with clinician-directed treatment plan."

    if not soap["Subjective"]:
        symptoms = _join_items(entities.get("Symptoms", []))
        soap["Subjective"] = (
            f"Patient-reported symptoms include {symptoms}." if symptoms else subjective_default
        )
    if not soap["Objective"]:
        medicines = _join_items(entities.get("Medicines", []))
        soap["Objective"] = f"Current documented medications: {medicines}." if medicines else objective_default
    if not soap["Assessment"]:
        conditions = _join_items(entities.get("Conditions", []))
        soap["Assessment"] = (
            f"Assessment is most consistent with {conditions}." if conditions else assessment_default
        )
    if not soap["Plan"]:
        plan_items = _join_items(entities.get("Medicines", []))
        soap["Plan"] = f"Continue/consider treatment with {plan_items} and monitor response." if plan_items else plan_default

    return soap


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
            "KSOAP": {
                "KeyData": "",
                "Subjective": "",
                "Objective": "",
                "Assessment": "",
                "Plan": "",
            },
            "SOAP": {
                "Subjective": "",
                "Objective": "",
                "Assessment": "",
                "Plan": "",
            },
            "ExtractedEntities": {
                "Symptoms": [],
                "NegatedSymptoms": [],
                "Medicines": [],
                "Conditions": [],
            },
        }

    sentences = _split_sentences(cleaned_text)
    if not sentences:
        sentences = [cleaned_text]

    extracted_entities = _extract_medical_entities(cleaned_text)
    soap = _map_to_soap(sentences, extracted_entities)
    soap = _ensure_section_defaults(soap, extracted_entities)

    normalized_soap = {
        "Subjective": _concise_summary(soap["Subjective"], max_sentences=2),
        "Objective": _concise_summary(soap["Objective"], max_sentences=2),
        "Assessment": _concise_summary(soap["Assessment"], max_sentences=2),
        "Plan": _concise_summary(soap["Plan"], max_sentences=2),
    }

    ksoap = {
        "KeyData": _build_key_data(extracted_entities),
        "Subjective": normalized_soap["Subjective"],
        "Objective": normalized_soap["Objective"],
        "Assessment": normalized_soap["Assessment"],
        "Plan": normalized_soap["Plan"],
    }

    return {
        "KSOAP": ksoap,
        "SOAP": normalized_soap,
        "ExtractedEntities": extracted_entities,
    }
