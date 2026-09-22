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
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from . import __version__
from .chunking import chunk_documents
from .comparison import compare_retrieval
from .documents import load_documents
from .embeddings import EmbeddingConfigurationError, provider_from_environment
from .retrieval import search
from .config import DEFAULT_CHUNKING_CONFIG
from .vector_store import IndexCompatibilityError, VectorIndex, build_vector_index

# 项目根目录 = src/rag_assistant/ 向上两级；示例资料固定放在 data/sample
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "sample"
_DEFAULT_INDEX = _PROJECT_ROOT / "data" / "index"
_DEFAULT_QUESTIONS = _PROJECT_ROOT / "data" / "eval" / "retrieval_questions.json"
_DEFAULT_REPORT = _PROJECT_ROOT / "reports" / "retrieval-comparison.json"


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
    print(f"共 {len(chunks)} 个片段（chunk_size={DEFAULT_CHUNKING_CONFIG.chunk_size}, overlap={DEFAULT_CHUNKING_CONFIG.overlap}）")
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


def _cmd_vector_index(args: argparse.Namespace) -> int:
    """Build and save the offline vector index."""
    try:
        provider = provider_from_environment()
    except EmbeddingConfigurationError as exc:
        print(f"配置错误：{exc}")
        return 2

    started = perf_counter()
    documents = load_documents(args.data)
    index, stats = build_vector_index(
        documents, provider, chunking_config=DEFAULT_CHUNKING_CONFIG
    )
    index.save(args.index_dir)
    print(f"向量索引已保存：{args.index_dir}")
    print(
        f"模型={provider.model_id} 维度={provider.dimension} "
        f"文档={stats.document_count} 片段={stats.chunk_count} "
        f"构建耗时={stats.elapsed_ms:.2f} ms"
    )
    print(f"含读取和保存总耗时={(perf_counter() - started) * 1000:.2f} ms")
    print("当前使用离线测试替身；这些分数不能代表真实语义检索效果。")
    return 0


def _cmd_vector_search(args: argparse.Namespace) -> int:
    """Load a compatible index and run cosine-similarity search."""
    started = perf_counter()
    try:
        provider = provider_from_environment()
        documents = load_documents(args.data)
        index = VectorIndex.load(
            args.index_dir,
            provider,
            documents=documents,
            chunking_config=DEFAULT_CHUNKING_CONFIG,
        )
    except (EmbeddingConfigurationError, IndexCompatibilityError, FileNotFoundError) as exc:
        print(f"无法使用向量索引：{exc}")
        return 2

    result = index.search(args.query, provider, top_k=args.top_k)
    print("当前使用离线测试替身；真实语义效果未验证。")
    print(f"含索引加载和校验总耗时={(perf_counter() - started) * 1000:.2f} ms")
    if not result.hits:
        print(f"没有向量命中：'{args.query}'")
        print(f"检索耗时：{result.elapsed_ms:.2f} ms")
        return 0
    print(
        f"向量查询: {args.query}    返回 Top-{len(result.hits)} "
        f"检索耗时={result.elapsed_ms:.2f} ms"
    )
    for hit in result.hits:
        chunk = hit.chunk
        print(
            f"\n[余弦相似度 {hit.score:.4f}] {chunk.chunk_id} "
            f"来源: {chunk.title}（{chunk.relative_path}）"
        )
        print(f"  原文位置: chars {chunk.char_start}–{chunk.char_end}")
        print(f"  {chunk.text}")
    print("相似度是排序信号，不是答案正确概率。")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare keyword and offline-vector retrieval on fixed labeled questions."""
    try:
        provider = provider_from_environment()
    except EmbeddingConfigurationError as exc:
        print(f"配置错误：{exc}")
        return 2

    documents = load_documents(args.data)
    report = compare_retrieval(
        documents, provider, args.questions, top_k=args.top_k
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("离线替身对比；默认题目为待人工核对的草稿，不代表真实语义效果。")
    for number, row in enumerate(report["questions"], 1):
        print(f"{number}. [{row['category']}] {row['question']}")
        for method, label in (("keyword", "关键词"), ("vector", "离线向量")):
            value = row[method]
            status = str(value["evidence_hit"]) if row["evidence"] else "无答案题（单独统计）"
            sources = ", ".join(
                f"{hit['path']}[{hit['char_start']}:{hit['char_end']}]" for hit in value["hits"]
            ) or "空"
            print(f"  {label}: 证据命中={status}，{value['elapsed_ms']:.3f} ms，{sources}")
    for method, label in (("keyword", "关键词"), ("vector", "离线向量")):
        stats = report["summary"][method]
        print(
            f"{label} Top-{args.top_k} 草稿证据命中={stats['evidence_hits']}/{stats['answerable_count']}；"
            f"无答案题返回空={stats['unanswerable_empty']}/{stats['unanswerable_count']}；"
            f"查询总耗时={stats['query_total_ms']:.3f} ms"
        )
    print(f"索引构建耗时={report['build_ms']:.3f} ms；逐题结果已保存：{args.report}")
    return 0


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是 >= 1 的整数")
    return number

#构造parser,add_argument后，有了这个属性
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
    #数据路径
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
    #子命令
    subparsers = parser.add_subparsers(dest="command", metavar="命令")

    subparsers.add_parser("docs", help="列出已加载的文档")
    #chunks子命令，专属--doc和--limit参数
    chunks_parser = subparsers.add_parser("chunks", help="查看切分结果")
    chunks_parser.add_argument("--doc", help="只显示指定 doc_id 的片段")
    chunks_parser.add_argument("--limit", type=int, default=5, help="最多显示几个片段")

    search_parser = subparsers.add_parser("search", help="关键词检索")
    search_parser.add_argument("query", help="查询内容")
    search_parser.add_argument("-k", "--top-k", type=int, default=3, help="返回前几个片段")

    vector_index_parser = subparsers.add_parser("vector-index", help="建立 NumPy 向量索引")
    vector_index_parser.add_argument(
        "--index-dir", type=Path, default=_DEFAULT_INDEX, help="本地索引目录"
    )

    vector_search_parser = subparsers.add_parser("vector-search", help="向量相似度检索")
    vector_search_parser.add_argument("query", help="查询内容")
    vector_search_parser.add_argument("-k", "--top-k", type=_positive_int, default=3, help="返回前几个片段")
    vector_search_parser.add_argument(
        "--index-dir", type=Path, default=_DEFAULT_INDEX, help="本地索引目录"
    )

    compare_parser = subparsers.add_parser("compare", help="比较关键词和向量检索")
    compare_parser.add_argument(
        "--questions", type=Path, default=_DEFAULT_QUESTIONS, help="评测问题 JSON 文件"
    )
    compare_parser.add_argument("-k", "--top-k", type=_positive_int, default=3, help="每种检索返回前几个片段")
    compare_parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT, help="逐题结果 JSON 文件")

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
    vector_commands = {
        "vector-index": _cmd_vector_index,
        "vector-search": _cmd_vector_search,
        "compare": _cmd_compare,
    }
    if args.command in vector_commands:
        try:
            return vector_commands[args.command](args)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Do not echo arbitrary parser/adapter exceptions that might contain local values.
            print(f"操作失败（{type(exc).__name__}）；请检查资料路径、问题格式或数值配置。")
            return 2

    # 无子命令时保留原演示行为
    print("RAG 学习项目演示")
    print("状态：模型服务尚未接入")
    if args.question:
        print(f"问题：{args.question}")
        print("回答：当前演示不生成答案；本地资料检索基线已可用（rag-demo search \"关键词\"）。")
        return 0
    print("可用命令：rag-demo docs / chunks / search \"问题\" / vector-index / vector-search")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
