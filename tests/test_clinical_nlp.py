from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import pipeline_service
from backend.services.speech_service import speech_to_text
from utils.soap_formatter import generate_soap


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _create_sample_wav(file_path: Path, duration_seconds: float = 0.25) -> None:
    """Create a tiny silent WAV file for audio-upload tests."""
    sample_rate = 16000
    total_frames = int(sample_rate * duration_seconds)

    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * total_frames)


def test_generate_soap_from_sample_summary() -> None:
    sample_summary = (FIXTURES_DIR / "expected_summary.txt").read_text(encoding="utf-8").strip()
    expected_soap = json.loads((FIXTURES_DIR / "expected_soap.json").read_text(encoding="utf-8"))

    soap = generate_soap(sample_summary)

    assert set(soap.keys()) == set(expected_soap.keys())
    assert set(soap["SOAP"].keys()) == set(expected_soap["SOAP"].keys())
    assert set(soap["ExtractedEntities"].keys()) == set(expected_soap["ExtractedEntities"].keys())
    assert "headache" in soap["SOAP"]["Subjective"].lower()
    assert "fever" in soap["SOAP"]["Subjective"].lower()
    assert soap["ExtractedEntities"]["Symptoms"] == expected_soap["ExtractedEntities"]["Symptoms"]
    assert soap["ExtractedEntities"]["Medicines"] == expected_soap["ExtractedEntities"]["Medicines"]
    assert soap["ExtractedEntities"]["Conditions"] == expected_soap["ExtractedEntities"]["Conditions"]


def test_speech_to_text_with_sample_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _create_sample_wav(audio_path)

    class FakeResponse:
        text = "Patient reports headache and fever. Taking ibuprofen."

    class FakeModels:
        def generate_content(self, model: str, contents: list[object]) -> FakeResponse:
            assert model == "gemini-2.5-flash"
            assert contents
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"
            self.models = FakeModels()

    monkeypatch.setattr("backend.services.speech_service.Client", FakeClient)
    monkeypatch.setattr("backend.services.speech_service.settings.gemini_api_key", "test-key")

    transcript = speech_to_text(str(audio_path))

    assert transcript == "Patient reports headache and fever. Taking ibuprofen."


def test_process_endpoint_returns_transcription_summary_and_soap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "sample.wav"
    _create_sample_wav(audio_path)

    sample_transcript = (FIXTURES_DIR / "sample_doctor_patient_conversation.txt").read_text(encoding="utf-8").strip()
    expected_summary = (FIXTURES_DIR / "expected_summary.txt").read_text(encoding="utf-8").strip()

    monkeypatch.setattr(pipeline_service, "speech_to_text", lambda file_path: sample_transcript)
    monkeypatch.setattr(pipeline_service, "summarize_clinical_text", lambda text: expected_summary)

    client = TestClient(app)

    with audio_path.open("rb") as audio_file:
        response = client.post("/api/process", files={"file": ("sample.wav", audio_file, "audio/wav")})

    assert response.status_code == 200
    payload = response.json()

    assert payload["transcription"] == sample_transcript
    assert payload["transcription_dialogue"]
    assert payload["summary"] == expected_summary
    assert payload["soap_note"]["SOAP"]["Subjective"]
    assert payload["soap_note"]["ExtractedEntities"]["Symptoms"]
    assert "transcription_entities" in payload
    assert "turn_count" in payload["transcription_entities"]
