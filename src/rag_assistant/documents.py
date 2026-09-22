"""文档加载：读取 UTF-8 Markdown/TXT，保留可追溯的来源信息。

本模块只负责"把文件读进来"，不负责切分和检索。
输出是 Document 对象列表，每个对象带有稳定的文档标识和相对路径，
方便以后把检索结果映射回原始文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """一份已加载的原文文档。

    doc_id:         稳定标识（doc_001、doc_002……），用于检索结果追溯
    title:          文档标题（Markdown 首个 # 标题，否则用文件名）
    relative_path:  相对资料根目录的路径，定位原始文件用
    content:        原文全文（UTF-8）
    """

    doc_id: str
    title: str
    relative_path: str
    content: str


def _extract_title(path: Path, content: str) -> str:
    """从 Markdown 首个 `# ` 标题取标题；没有标题时用文件名。"""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem


def load_documents(root: Path) -> list[Document]:
    """读取 root 下所有 UTF-8 Markdown/TXT，按文件名排序，返回 Document 列表。

    重复导入同一个文件时不会产生重复文档：这里用"相对路径 -> 文档"的字典
    去重，同一路径只加载一次（后续版本可加上"检测内容变化"的能力）。
    """
    if not root.is_dir():
        raise FileNotFoundError(f"资料目录不存在：{root}")

    documents: dict[str, Document] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        if content == "":
            continue  # 空文档不产生 Document，避免后续切分空文本
        rel = path.relative_to(root).as_posix()
        doc_id = f"doc_{len(documents) + 1:03d}"
        documents[rel] = Document(
            doc_id=doc_id,
            title=_extract_title(path, content),
            relative_path=rel,
            content=content,
        )
    return list(documents.values())
