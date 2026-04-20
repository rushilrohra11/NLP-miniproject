from __future__ import annotations

from backend.services.rag_service import build_rag_payload


def test_build_rag_payload_returns_context_and_citations() -> None:
    query = "Patient has fever and cough and needs follow-up guidance"

    payload = build_rag_payload(query, top_k=2)

    assert set(payload.keys()) == {"prompt_context", "citations"}
    assert isinstance(payload["prompt_context"], str)
    assert isinstance(payload["citations"], list)
    assert len(payload["citations"]) >= 1

    first = payload["citations"][0]
    assert "source" in first
    assert "chunk_id" in first
    assert "score" in first
    assert "text" in first
