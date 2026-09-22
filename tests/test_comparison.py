from pathlib import Path
import json

from rag_assistant.chunking import Chunk
from rag_assistant.comparison import compare_retrieval, evidence_hit
from rag_assistant.documents import load_documents
from rag_assistant.embeddings import OfflineHashEmbedding
from rag_assistant.vector_store import VectorHit


def test_same_document_wrong_chunk_is_not_evidence():
    wrong = Chunk("d#0", "d", "title", "wrong", 0, 5, "guide.md")
    right = Chunk("d#1", "d", "title", "right", 10, 15, "guide.md")
    evidence = [{"path": "guide.md", "char_start": 10, "char_end": 15}]
    assert not evidence_hit([VectorHit(wrong, 0.9)], evidence)
    assert evidence_hit([VectorHit(right, 0.1)], evidence)


def test_ten_sample_questions_have_traceable_evidence_and_separate_timings():
    root = Path(__file__).resolve().parents[1]
    report = compare_retrieval(
        load_documents(root / "data/sample"), OfflineHashEmbedding(),
        root / "data/eval/retrieval_questions.json",
    )
    rows = report["questions"]
    assert len(rows) == 10
    assert {"同义表达", "精确编号", "精确术语"} <= {row["category"] for row in rows}
    assert all(row["review_status"] == "pending_human_review" for row in rows)
    assert report["real_semantic_quality"] == "not_measured"
    for method in ("keyword", "vector"):
        assert report["summary"][method]["answerable_count"] == 9
        assert report["summary"][method]["unanswerable_count"] == 1
        assert report["summary"][method]["query_total_ms"] == sum(row[method]["elapsed_ms"] for row in rows)
        assert rows[-1][method]["evidence_hit"] is None
    json.dumps(report, ensure_ascii=False)
