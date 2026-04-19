from __future__ import annotations

from utils.evaluation import evaluate_summarization, explain_evaluation_results


def test_evaluate_summarization_returns_scores_and_explanation() -> None:
    reference = "The patient has fever, cough, and chest pain. Treatment includes rest and ibuprofen."
    generated = "Patient has fever and cough. Treatment includes ibuprofen and rest."

    results = evaluate_summarization(reference, generated)

    assert set(results.keys()) == {"rouge1", "rouge2", "rougeL", "bleu", "interpretation"}
    assert 0.0 <= results["rouge1"] <= 1.0
    assert 0.0 <= results["rouge2"] <= 1.0
    assert 0.0 <= results["rougeL"] <= 1.0
    assert 0.0 <= results["bleu"] <= 1.0
    assert isinstance(results["interpretation"], str)

    explanation = explain_evaluation_results(results)
    assert "ROUGE-1" in explanation
    assert "BLEU" in explanation
