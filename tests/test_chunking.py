"""切分测试：空文本、中文、超长段落、来源定位、重叠。"""

from rag_assistant.chunking import Chunk, chunk_document, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_document("doc_001", "标题", "") == []


def test_short_paragraph_stays_one_chunk():
    chunks = chunk_text("只有一句话。", chunk_size=500, overlap=0)
    assert len(chunks) == 1
    text, start, end = chunks[0]
    assert text == "只有一句话。"
    assert (start, end) == (0, 6)


def test_paragraphs_merged_up_to_chunk_size():
    text = "第一段。\n\n第二段。\n\n第三段。"
    # chunk_size 足够大时三段合并为一块
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0][0] == text


def test_paragraphs_split_when_exceeding_chunk_size():
    text = "甲段落内容。\n\n乙段落内容。\n\n丙段落内容。"
    # chunk_size=8：一段约 6 字符，两块以上
    chunks = chunk_text(text, chunk_size=8, overlap=0)
    assert len(chunks) >= 2
    # 每个块都能从原文区间切回
    for block, start, end in chunks:
        assert text[start:end] == block
    # 无重叠时块区间不相交，且覆盖原文的绝大部分（只丢弃空行分隔符）
    for i in range(len(chunks) - 1):
        assert chunks[i][2] <= chunks[i + 1][1]
    covered = sum(end - start for _, start, end in chunks)
    assert covered >= len(text) - 4


def test_source_location_maps_back_to_original(tmp_path):
    """每个块的 text 必须能用 [char_start, char_end) 从原文切回来。"""
    from rag_assistant.documents import load_documents

    (tmp_path / "src.md").write_text(
        "# 测试\n\n第一段内容。\n\n第二段内容，稍微长一点点。", encoding="utf-8"
    )
    doc = load_documents(tmp_path)[0]
    chunks = chunk_document(doc.doc_id, doc.title, doc.content, chunk_size=20, overlap=5)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert doc.content[chunk.char_start : chunk.char_end] == chunk.text


def test_long_paragraph_is_hard_split():
    # 单个超长段落（无空行）按 chunk_size 硬切
    text = "很" * 30
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) == 3
    assert all(end - start <= 10 for _, start, end in chunks)
    assert "".join(t for t, _, _ in chunks) == text


def test_overlap_extends_next_chunk_start():
    text = "段落甲" * 10 + "\n\n" + "段落乙" * 10
    chunks = chunk_text(text, chunk_size=15, overlap=5)
    assert len(chunks) >= 2
    # 第 2 块起点应早于第 1 块结束（有重叠）
    _, s1, e1 = chunks[0]
    _, s2, e2 = chunks[1]
    assert s2 < e1 and s2 >= s1 + 1
    assert e2 > e1


def test_chunk_ids_are_traceable():
    # "一段。" 4 字符 + 分隔符 2 + "二段。" 4 = 10，chunk_size=6 放不下两段
    chunks = chunk_document("doc_002", "标题", "一段。\n\n二段。", chunk_size=6, overlap=0)
    assert [c.chunk_id for c in chunks] == ["doc_002#0", "doc_002#1"]
    assert all(c.doc_id == "doc_002" for c in chunks)
