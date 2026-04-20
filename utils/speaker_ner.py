"""Speaker role extraction helpers for transcript text.

This module identifies common clinical conversation roles and returns both a
normalized speaker-labeled dialogue and role/entity metadata for UI display.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple


ROLE_ALIASES: Dict[str, str] = {
    "doctor": "doctor",
    "dr": "doctor",
    "dr.": "doctor",
    "patient": "patient",
    "pt": "patient",
    "pt.": "patient",
    "nurse": "nurse",
    "rn": "nurse",
    "accompanier": "accompanier",
    "caregiver": "accompanier",
    "family member": "accompanier",
    "relative": "accompanier",
    "support staff": "support_staff",
    "staff": "support_staff",
}

ROLE_DISPLAY: Dict[str, str] = {
    "doctor": "Doctor",
    "patient": "Patient",
    "nurse": "Nurse",
    "accompanier": "Accompanier",
    "support_staff": "Support Staff",
    "other": "Other",
}

ROLE_LINE_PATTERN = re.compile(
    r"^\s*(doctor|dr\.?|patient|pt\.?|nurse|rn|accompanier|caregiver|family member|relative|support staff|staff)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

DOCTOR_NAME_PATTERN = re.compile(r"\b(?:[Dd][Rr]\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b")
PATIENT_NAME_PATTERN = re.compile(r"\b(?:[Pp]atient|[Pp]t\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b")
NURSE_NAME_PATTERN = re.compile(r"\b(?:[Nn]urse|[Rr][Nn])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b")
ACCOMPANIER_NAME_PATTERN = re.compile(
    r"\b(?:[Aa]ccompanier|[Cc]aregiver|[Ff]amily member|[Rr]elative)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b",
)
SUPPORT_STAFF_NAME_PATTERN = re.compile(r"\b(?:[Ss]upport staff|[Ss]taff)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b")


def _dedupe(items: List[str]) -> List[str]:
    return list(OrderedDict.fromkeys(items))


def _normalize_role(raw_role: str) -> str:
    return ROLE_ALIASES.get(raw_role.strip().lower(), "other")


def _split_dialogue_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    return [text.strip()] if text.strip() else []


def _extract_turns(lines: List[str]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    for line in lines:
        match = ROLE_LINE_PATTERN.match(line)
        if match:
            role = _normalize_role(match.group(1))
            utterance = match.group(2).strip()
            if utterance:
                turns.append((role, utterance))
        else:
            turns.append(("other", line))
    return turns


def normalize_transcription_dialogue(transcription: str) -> str:
    """Normalize transcript into speaker-style dialogue lines.

    If speaker labels are already present, they are normalized. Otherwise text
    is retained under `Other` so the transcription box always shows dialogue.
    """
    text = transcription.strip()
    if not text:
        return ""

    turns = _extract_turns(_split_dialogue_lines(text))
    normalized_lines = [f"{ROLE_DISPLAY.get(role, 'Other')}: {utterance}" for role, utterance in turns]
    return "\n".join(normalized_lines)


def extract_speaker_entities(transcription: str) -> Dict[str, Any]:
    """Extract role cues and mentions from transcript text.

    This is transcript-level role extraction, not audio diarization.
    """
    text = transcription.strip()
    if not text:
        return {
            "roles_detected": [],
            "doctor_mentions": [],
            "patient_mentions": [],
            "nurse_mentions": [],
            "accompanier_mentions": [],
            "support_staff_mentions": [],
            "turn_count": {
                "doctor": 0,
                "patient": 0,
                "nurse": 0,
                "accompanier": 0,
                "support_staff": 0,
                "other": 0,
            },
            "dialogue_lines": [],
        }

    turns = _extract_turns(_split_dialogue_lines(text))
    turn_count = {
        "doctor": 0,
        "patient": 0,
        "nurse": 0,
        "accompanier": 0,
        "support_staff": 0,
        "other": 0,
    }
    for role, _ in turns:
        turn_count[role] = turn_count.get(role, 0) + 1

    doctor_mentions = [name.strip() for name in DOCTOR_NAME_PATTERN.findall(text)]
    patient_mentions = [name.strip() for name in PATIENT_NAME_PATTERN.findall(text)]
    nurse_mentions = [name.strip() for name in NURSE_NAME_PATTERN.findall(text)]
    accompanier_mentions = [name.strip() for name in ACCOMPANIER_NAME_PATTERN.findall(text)]
    support_staff_mentions = [name.strip() for name in SUPPORT_STAFF_NAME_PATTERN.findall(text)]

    roles_detected = [role for role, count in turn_count.items() if count > 0 and role != "other"]

    # Add mention-only roles even when no explicit turn labels are present.
    if doctor_mentions and "doctor" not in roles_detected:
        roles_detected.append("doctor")
    if patient_mentions and "patient" not in roles_detected:
        roles_detected.append("patient")
    if nurse_mentions and "nurse" not in roles_detected:
        roles_detected.append("nurse")
    if accompanier_mentions and "accompanier" not in roles_detected:
        roles_detected.append("accompanier")
    if support_staff_mentions and "support_staff" not in roles_detected:
        roles_detected.append("support_staff")

    dialogue_lines = [f"{ROLE_DISPLAY.get(role, 'Other')}: {utterance}" for role, utterance in turns]

    return {
        "roles_detected": roles_detected,
        "doctor_mentions": _dedupe(doctor_mentions),
        "patient_mentions": _dedupe(patient_mentions),
        "nurse_mentions": _dedupe(nurse_mentions),
        "accompanier_mentions": _dedupe(accompanier_mentions),
        "support_staff_mentions": _dedupe(support_staff_mentions),
        "turn_count": turn_count,
        "dialogue_lines": dialogue_lines,
    }
