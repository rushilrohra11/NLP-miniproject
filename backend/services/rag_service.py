"""RAG orchestration helpers for retrieval-augmented generation."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.rag_retriever import retrieve_clinical_context


def build_rag_payload(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Retrieve context and format it for downstream generation and API output."""
    retrieved = retrieve_clinical_context(query, top_k=top_k)

    context_lines: List[str] = []
    citations: List[Dict[str, Any]] = []

    for item in retrieved:
        context_lines.append(item.text)
        citations.append(
            {
                "source": item.source,
                "chunk_id": item.chunk_id,
                "score": round(item.score, 4),
                "text": item.text,
            }
        )

    prompt_context = "\n\n".join(context_lines)

    return {
        "prompt_context": prompt_context,
        "citations": citations,
    }
