"""Small retrieval comparison; draft evidence labels are not a validated benchmark."""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from .config import ChunkingConfig, DEFAULT_CHUNKING_CONFIG
from .documents import Document
from .embeddings import EmbeddingProvider
from .retrieval import search
from .vector_store import build_vector_index


def _resolve_evidence(questions: list[dict], documents: list[Document]) -> None:
    by_path = {document.relative_path: document for document in documents}
    for item in questions:
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("评测问题必须为非空文本")
        if not isinstance(item.get("evidence"), list):
            raise ValueError("每道题必须包含 evidence 列表（无答案题为空列表）")
        for evidence in item["evidence"]:
            document = by_path.get(evidence["path"])
            quote = evidence["quote"]
            if document is None or not quote or document.content.count(quote) != 1:
                raise ValueError("评测证据必须唯一对应当前资料中的原文，请核对问题文件。")
            start = document.content.index(quote)
            evidence["char_start"] = start
            evidence["char_end"] = start + len(quote)


def evidence_hit(hits, evidence: list[dict]) -> bool:
    """One returned chunk must contain the whole annotated evidence span."""
    return any(
        hit.chunk.relative_path == item["path"]
        and hit.chunk.char_start <= item["char_start"]
        and hit.chunk.char_end >= item["char_end"]
        for hit in hits for item in evidence
    )


def _hit_rows(hits) -> list[dict]:
    return [
        {
            "chunk_id": hit.chunk.chunk_id,
            "path": hit.chunk.relative_path,
            "char_start": hit.chunk.char_start,
            "char_end": hit.chunk.char_end,
            "score": hit.score,
        }
        for hit in hits
    ]


def compare_retrieval(
    documents: list[Document],
    provider: EmbeddingProvider,
    questions_path: Path,
    top_k: int = 3,
    chunking_config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k 必须 >= 1")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("问题文件需要包含非空列表")
    _resolve_evidence(questions, documents)
    index, stats = build_vector_index(documents, provider, chunking_config)
    rows = []
    for item in questions:
        started = perf_counter()
        keyword_hits = search(index.chunks, item["question"], top_k=top_k)
        keyword_ms = (perf_counter() - started) * 1000
        vector_result = index.search(item["question"], provider, top_k=top_k)
        evidence = item["evidence"]
        rows.append({
            "question": item["question"],
            "category": item.get("category", ""),
            "review_status": item.get("review_status", "pending_human_review"),
            "evidence": evidence,
            "keyword": {
                "hits": _hit_rows(keyword_hits),
                "evidence_hit": evidence_hit(keyword_hits, evidence) if evidence else None,
                "elapsed_ms": keyword_ms,
            },
            "vector": {
                "hits": _hit_rows(vector_result.hits),
                "evidence_hit": evidence_hit(vector_result.hits, evidence) if evidence else None,
                "elapsed_ms": vector_result.elapsed_ms,
            },
        })
    summary = {}
    for method in ("keyword", "vector"):
        summary[method] = {
            "evidence_hits": sum(row[method]["evidence_hit"] is True for row in rows),
            "answerable_count": sum(bool(row["evidence"]) for row in rows),
            "unanswerable_empty": sum(
                not row[method]["hits"] for row in rows if not row["evidence"]
            ),
            "unanswerable_count": sum(not row["evidence"] for row in rows),
            "query_total_ms": sum(row[method]["elapsed_ms"] for row in rows),
        }
    return {
        "mode": "offline_test_double",
        "real_semantic_quality": "not_measured",
        "top_k": top_k,
        "metric": "One Top-k chunk contains a complete annotated evidence span in its source file.",
        "index_metadata": asdict(index.metadata),
        "build_ms": stats.elapsed_ms,
        "summary": summary,
        "questions": rows,
    }
