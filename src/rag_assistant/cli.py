"""Command-line entry point: offline demo plus document retrieval commands.

用法示例：
  rag-demo                    原演示（模型未接入提示）
  rag-demo docs               列出已加载的示例文档
  rag-demo chunks             查看切分结果（前 5 个片段）
  rag-demo chunks --doc doc_001 --limit 10   只看某份文档的片段
  rag-demo search "浇水"       关键词检索，返回 Top-3 片段和来源
  rag-demo search "植物 阳光" -k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
from .chunking import chunk_documents
from .documents import load_documents
from .retrieval import search

# 项目根目录 = src/rag_assistant/ 向上两级；示例资料固定放在 data/sample
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "sample"


def _load_pipeline(data_dir: Path):
    """加载文档并切分，返回 (documents, chunks)。两条命令共用的公共步骤。"""
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents)
    return documents, chunks


def _cmd_docs(args: argparse.Namespace) -> int:
    documents, _ = _load_pipeline(args.data)
    if not documents:
        print("（没有找到可加载的文档）")
        return 0
    print(f"共 {len(documents)} 份文档：")
    for doc in documents:
        paragraphs = doc.content.count("\n\n") + 1
        print(f"  {doc.doc_id}  {doc.title}  [{doc.relative_path}]  {len(doc.content)} 字符")
    return 0


def _cmd_chunks(args: argparse.Namespace) -> int:
    documents, chunks = _load_pipeline(args.data)
    if not chunks:
        print("（没有可显示的片段，先确认 data/sample 下有 .md/.txt 资料）")
        return 0
    print(f"共 {len(chunks)} 个片段（chunk_size=500, overlap=50）")
    shown = [c for c in chunks if c.doc_id == args.doc] if args.doc else chunks
    shown = shown[: args.limit]
    for chunk in shown:
        print(f"\n[{chunk.chunk_id}] 来源: {chunk.title}（{chunk.doc_id}）")
        print(f"  原文位置: chars {chunk.char_start}–{chunk.char_end}")
        preview = chunk.text if len(chunk.text) <= 80 else chunk.text[:77] + "..."
        print(f"  内容: {preview}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    _, chunks = _load_pipeline(args.data)
    hits = search(chunks, args.query, top_k=args.top_k)
    if not hits:
        print(f"没有命中：'{args.query}'（纯字面匹配，同义词查不到）")
        return 0
    print(f"查询: {args.query}    返回 Top-{len(hits)}")
    for hit in hits:
        chunk = hit.chunk
        print(f"\n[得分 {hit.score}] {chunk.chunk_id}  来源: {chunk.title}（{chunk.doc_id}）")
        print(f"  原文位置: chars {chunk.char_start}–{chunk.char_end}")
        print(f"  {chunk.text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the parser separately so the command can be tested without a process."""
    parser = argparse.ArgumentParser(
        prog="rag-demo",
        description="RAG 学习项目：离线演示 + 文档检索基线",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rag-learning-assistant {__version__}",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help="资料目录（默认 data/sample）",
    )
    parser.add_argument(
        "--question",
        help="可选的演示问题；当前版本只回显问题，不调用模型",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="命令")

    subparsers.add_parser("docs", help="列出已加载的文档")

    chunks_parser = subparsers.add_parser("chunks", help="查看切分结果")
    chunks_parser.add_argument("--doc", help="只显示指定 doc_id 的片段")
    chunks_parser.add_argument("--limit", type=int, default=5, help="最多显示几个片段")

    search_parser = subparsers.add_parser("search", help="关键词检索")
    search_parser.add_argument("query", help="查询内容")
    search_parser.add_argument("-k", "--top-k", type=int, default=3, help="返回前几个片段")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a shell-friendly exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "docs":
        return _cmd_docs(args)
    if args.command == "chunks":
        return _cmd_chunks(args)
    if args.command == "search":
        return _cmd_search(args)

    # 无子命令时保留原演示行为
    print("RAG 学习项目演示")
    print("状态：模型服务尚未接入")
    if args.question:
        print(f"问题：{args.question}")
        print("回答：当前演示不生成答案；本地资料检索基线已可用（rag-demo search \"关键词\"）。")
        return 0
    print("可用命令：rag-demo docs / chunks / search \"问题\"")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
