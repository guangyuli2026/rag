"""检索测试：中文关键词、Top-k 排序、同义词局限。"""

from rag_assistant.chunking import Chunk, chunk_documents
from rag_assistant.documents import Document, load_documents
from rag_assistant.retrieval import search, split_keywords


def _doc(doc_id: str, title: str, content: str) -> Document:
    return Document(doc_id=doc_id, title=title, relative_path=f"{doc_id}.md", content=content)


def test_split_keywords_keeps_chinese_sequence():
    # 中文连续文本整体保留，不被空格规则误切
    assert split_keywords("植物 浇水") == ["植物", "浇水"]
    assert split_keywords("绿萝怎么养") == ["绿萝怎么养"]


def test_chinese_keyword_finds_matching_chunk():
    chunks = chunk_documents(
        [
            _doc("doc_001", "植物", "绿萝喜欢散射光，一周浇水一次。"),
            _doc("doc_002", "徒步", "登山杖能减轻膝盖压力。"),
        ]
    )
    hits = search(chunks, "浇水", top_k=3)

    assert len(hits) == 1
    assert hits[0].chunk.doc_id == "doc_001"
    assert hits[0].score == 1


def test_top_k_returns_most_relevant_first():
    chunks = chunk_documents(
        [
            _doc("doc_001", "植物", "浇水。"),
            _doc("doc_002", "植物", "浇水浇水浇水。"),
            _doc("doc_003", "徒步", "爬山。"),
        ]
    )
    hits = search(chunks, "浇水", top_k=2)

    assert [h.chunk.doc_id for h in hits] == ["doc_002", "doc_001"]
    assert [h.score for h in hits] == [3, 1]


def test_no_hit_returns_empty():
    chunks = chunk_documents([_doc("doc_001", "植物", "绿萝需要散射光。")])
    assert search(chunks, "滑雪") == []


def test_synonym_is_not_found_limitation():
    """同义词查不到：'检索' 匹配不到 '搜索'，这是纯字面匹配的已知局限。"""
    chunks = chunk_documents([_doc("doc_001", "说明", "本工具支持全文搜索功能。")])
    hits = search(chunks, "检索")
    assert hits == []


def test_end_to_end_with_real_sample_files():
    """用 data/sample 真实示例做一次端到端验证（不依赖模型）。"""
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "data" / "sample"
    docs = load_documents(sample)
    assert len(docs) >= 3  # 三份示例资料

    chunks = chunk_documents(docs)
    hits = search(chunks, "徒步")

    assert hits
    assert hits[0].chunk.title == "周末徒步计划"
    # 命中片段必须能定位回原文
    doc = next(d for d in docs if d.doc_id == hits[0].chunk.doc_id)
    assert doc.content[hits[0].chunk.char_start : hits[0].chunk.char_end] == hits[0].chunk.text
