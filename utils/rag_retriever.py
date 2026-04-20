"""Lightweight retrieval helpers for grounding generation with local knowledge files."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


DEFAULT_KNOWLEDGE_SNIPPETS = [
    (
        "built_in://fever_cough",
        "For fever and cough, include duration, associated symptoms, hydration status, and clear follow-up instructions.",
    ),
    (
        "built_in://hypertension_followup",
        "For hypertension follow-up, document adherence, BP trends, lifestyle counseling, and monitoring plan.",
    ),
    (
        "built_in://soap_structure",
        "SOAP notes should separate patient-reported subjective details from objective findings and include an actionable plan.",
    ),
]

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    chunk_id: int
    text: str
    score: float


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _chunk_text(text: str, max_words: int = 120) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
            continue

        for start in range(0, len(words), max_words):
            piece = " ".join(words[start : start + max_words]).strip()
            if piece:
                chunks.append(piece)
    return chunks


def _load_knowledge_documents(knowledge_dir: Path) -> List[tuple[str, str]]:
    documents: List[tuple[str, str]] = []

    if knowledge_dir.exists() and knowledge_dir.is_dir():
        for path in sorted(knowledge_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if text.strip():
                documents.append((str(path), text))

    if not documents:
        documents.extend(DEFAULT_KNOWLEDGE_SNIPPETS)

    return documents


@lru_cache(maxsize=1)
def _build_index() -> Dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    knowledge_dir = project_root / "data" / "knowledge"

    documents = _load_knowledge_documents(knowledge_dir)

    chunks: List[tuple[str, int, str]] = []
    for source, raw_text in documents:
        for chunk_id, chunk in enumerate(_chunk_text(raw_text), start=1):
            chunks.append((source, chunk_id, chunk))

    doc_freq: Dict[str, int] = {}
    chunk_tokens: List[List[str]] = []
    for _, _, text in chunks:
        tokens = _tokenize(text)
        chunk_tokens.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    total_chunks = max(len(chunks), 1)
    idf: Dict[str, float] = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log((1.0 + total_chunks) / (1.0 + freq)) + 1.0

    vectors: List[Dict[str, float]] = []
    norms: List[float] = []
    for tokens in chunk_tokens:
        tf: Dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0
        for token in list(tf.keys()):
            tf[token] = tf[token] * idf.get(token, 1.0)
        norm = math.sqrt(sum(weight * weight for weight in tf.values())) or 1.0
        vectors.append(tf)
        norms.append(norm)

    return {
        "chunks": chunks,
        "idf": idf,
        "vectors": vectors,
        "norms": norms,
    }


def retrieve_clinical_context(query: str, top_k: int = 3) -> List[RetrievedChunk]:
    """Retrieve top-k relevant knowledge chunks for grounding generation."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    index = _build_index()
    chunks: List[tuple[str, int, str]] = index["chunks"]  # type: ignore[assignment]
    idf: Dict[str, float] = index["idf"]  # type: ignore[assignment]
    vectors: List[Dict[str, float]] = index["vectors"]  # type: ignore[assignment]
    norms: List[float] = index["norms"]  # type: ignore[assignment]

    query_tf: Dict[str, float] = {}
    for token in _tokenize(cleaned_query):
        query_tf[token] = query_tf.get(token, 0.0) + 1.0
    if not query_tf:
        return []

    for token in list(query_tf.keys()):
        query_tf[token] = query_tf[token] * idf.get(token, 1.0)

    query_norm = math.sqrt(sum(weight * weight for weight in query_tf.values())) or 1.0

    scored: List[RetrievedChunk] = []
    for idx, (source, chunk_id, text) in enumerate(chunks):
        vec = vectors[idx]
        dot = 0.0
        for token, weight in query_tf.items():
            dot += weight * vec.get(token, 0.0)
        score = dot / (query_norm * norms[idx])
        if score > 0:
            scored.append(RetrievedChunk(source=source, chunk_id=chunk_id, text=text, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
