from __future__ import annotations

from utils.speaker_ner import extract_speaker_entities, normalize_transcription_dialogue


def test_extract_speaker_entities_detects_doctor_and_patient_turns() -> None:
    transcript = """
    Doctor: Hello, what brings you in today?
    Patient: I have fever and cough for two days.
    Doctor: Any chest pain?
    Patient: No chest pain.
    """.strip()

    entities = extract_speaker_entities(transcript)

    assert entities["turn_count"]["doctor"] == 2
    assert entities["turn_count"]["patient"] == 2
    assert "doctor" in entities["roles_detected"]
    assert "patient" in entities["roles_detected"]


def test_extract_speaker_entities_detects_names_from_mentions() -> None:
    transcript = (
        "Dr. Adams discussed treatment. "
        "Patient John Smith agreed to follow up. "
        "Nurse Emily checked blood pressure. "
        "Accompanier Rahul asked about medications. "
        "Support staff Maria scheduled the next appointment."
    )

    entities = extract_speaker_entities(transcript)

    assert "Adams" in entities["doctor_mentions"]
    assert "John Smith" in entities["patient_mentions"]
    assert "Emily" in entities["nurse_mentions"]
    assert "Rahul" in entities["accompanier_mentions"]
    assert "Maria" in entities["support_staff_mentions"]


def test_normalize_transcription_dialogue_formats_role_lines() -> None:
    transcript = "Doctor: Hello there\nPatient: I have fever\nNurse: Vitals are stable"

    dialogue = normalize_transcription_dialogue(transcript)

    assert "Doctor: Hello there" in dialogue
    assert "Patient: I have fever" in dialogue
    assert "Nurse: Vitals are stable" in dialogue
