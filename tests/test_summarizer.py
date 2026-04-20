from __future__ import annotations

import pytest

from backend.services.nlp_service import summarize_clinical_text
from utils.summarizer import summarize_text


def test_summarize_text_uses_concise_fallback_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_summarizer():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("utils.summarizer._get_summarizer", fake_get_summarizer)

    input_text = (
        "Patient reports fever for three days. "
        "Patient also reports cough and fatigue. "
        "No chest pain was reported. "
        "Hydration is reduced and appetite is low."
    )

    summary = summarize_text(input_text)

    assert summary
    assert summary != input_text.lower()
    assert len(summary.split()) < len(input_text.split())


def test_summarize_clinical_text_dialogue_returns_third_person_summary() -> None:
    dialogue = (
        "Doctor: What brings you in today?\n"
        "Patient: I have had fever and cough for three days.\n"
        "Doctor: We should monitor symptoms and continue hydration."
    )

    summary = summarize_clinical_text(dialogue)

    assert summary
    assert "The patient reported" in summary
    assert "The doctor discussed" in summary
    assert " I " not in f" {summary} "
