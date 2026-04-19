"""Summarization evaluation helpers.

This module computes ROUGE and BLEU scores for a generated summary against a
reference summary, then adds a short plain-English interpretation so the
results are easy to explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from utils.text_utils import normalize_text


@dataclass(frozen=True)
class SummarizationEvaluation:
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    interpretation: str


def _tokenize(text: str) -> List[str]:
    """Tokenize text with simple whitespace splitting after normalization."""
    normalized = normalize_text(text).lower()
    return normalized.split() if normalized else []


def _interpret_scores(rouge1: float, rouge2: float, rougeL: float, bleu: float) -> str:
    """Convert raw scores into a short human-readable explanation."""
    average_overlap = (rouge1 + rouge2 + rougeL) / 3.0

    if average_overlap >= 0.6 and bleu >= 0.4:
        return "Strong similarity: the generated summary closely matches the reference summary."
    if average_overlap >= 0.35 and bleu >= 0.2:
        return "Moderate similarity: the summary captures the main ideas, but wording differs."
    if average_overlap >= 0.2 or bleu >= 0.1:
        return "Partial similarity: some important content is present, but coverage is limited."
    return "Low similarity: the generated summary differs significantly from the reference."


def evaluate_summarization(reference_summary: str, generated_summary: str) -> Dict[str, Any]:
    """Evaluate a generated summary against a reference summary.

    Returns a JSON-friendly dictionary with ROUGE-1, ROUGE-2, ROUGE-L,
    BLEU, and a plain-English interpretation of the result.
    """
    reference_text = normalize_text(reference_summary)
    generated_text = normalize_text(generated_summary)

    if not reference_text or not generated_text:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "interpretation": "One or both summaries are empty, so evaluation cannot be meaningfully computed.",
        }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, generated_text)

    reference_tokens = [_tokenize(reference_text)]
    generated_tokens = _tokenize(generated_text)

    bleu_score = sentence_bleu(
        reference_tokens,
        generated_tokens,
        smoothing_function=SmoothingFunction().method1,
    )

    result = SummarizationEvaluation(
        rouge1=round(rouge_scores["rouge1"].fmeasure, 4),
        rouge2=round(rouge_scores["rouge2"].fmeasure, 4),
        rougeL=round(rouge_scores["rougeL"].fmeasure, 4),
        bleu=round(bleu_score, 4),
        interpretation="",
    )

    interpretation = _interpret_scores(result.rouge1, result.rouge2, result.rougeL, result.bleu)

    return {
        "rouge1": result.rouge1,
        "rouge2": result.rouge2,
        "rougeL": result.rougeL,
        "bleu": result.bleu,
        "interpretation": interpretation,
    }


def explain_evaluation_results(results: Dict[str, Any]) -> str:
    """Create a short readable explanation for CLI output or logs."""
    rouge1 = results.get("rouge1", 0.0)
    rouge2 = results.get("rouge2", 0.0)
    rougeL = results.get("rougeL", 0.0)
    bleu = results.get("bleu", 0.0)
    interpretation = results.get("interpretation", "")

    return (
        f"ROUGE-1: {rouge1:.4f}\n"
        f"ROUGE-2: {rouge2:.4f}\n"
        f"ROUGE-L: {rougeL:.4f}\n"
        f"BLEU: {bleu:.4f}\n"
        f"Result: {interpretation}"
    )
