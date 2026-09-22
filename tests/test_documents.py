"""文档加载测试：空文档、重复导入、来源路径保留。"""

from pathlib import Path

from rag_assistant.documents import load_documents


def _make_sample(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_skips_empty_document(tmp_path):
    _make_sample(tmp_path, "empty.md", "")
    _make_sample(tmp_path, "normal.txt", "hello world")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0].title == "normal"


def test_load_keeps_title_path_and_content(tmp_path):
    path = _make_sample(tmp_path, "guide.md", "# 使用指南\n\n第一段正文。")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "使用指南"  # 取 Markdown 首个 # 标题
    assert doc.relative_path == "guide.md"
    assert doc.content == "# 使用指南\n\n第一段正文。"


def test_duplicate_import_does_not_duplicate_documents(tmp_path):
    _make_sample(tmp_path, "note.md", "内容 A")

    # 同一目录重复加载两次，应仍只有一份文档（按相对路径去重）
    first = load_documents(tmp_path)
    second = load_documents(tmp_path)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].relative_path == second[0].relative_path


def test_doc_ids_are_stable_in_sorted_order(tmp_path):
    _make_sample(tmp_path, "b.md", "内容 B")
    _make_sample(tmp_path, "a.md", "内容 A")

    docs = load_documents(tmp_path)

    # 按文件名排序，doc_001 对应 a.md，doc_002 对应 b.md
    assert [d.relative_path for d in docs] == ["a.md", "b.md"]
    assert docs[0].doc_id == "doc_001"
    assert docs[1].doc_id == "doc_002"


def test_raises_when_root_missing(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "not_exist")
