from pathlib import Path

from rag_assistant.chunking import chunk_documents
from rag_assistant.cli import main
from rag_assistant.config import DEFAULT_CHUNKING_CONFIG
from rag_assistant.documents import load_documents

_DATA = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_demo_without_question(capsys):
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "RAG 学习项目演示" in output
    assert "模型服务尚未接入" in output


def test_demo_echoes_question(capsys):
    assert main(["--question", "项目现在能做什么？"]) == 0

    output = capsys.readouterr().out
    assert "问题：项目现在能做什么？" in output
    assert "本地资料检索" in output

def test_cli_chunks_shows_config_actually_used(capsys):
    """CLI 显示的切分参数必须等于配置，且与实际切分结果一致。"""
    assert main(["chunks", "--limit", "5"]) == 0
    out = capsys.readouterr().out

    cfg = DEFAULT_CHUNKING_CONFIG
    # 1) 打印的值 == 配置的值（证明不是手写 500/50）
    assert f"chunk_size={cfg.chunk_size}" in out
    assert f"overlap={cfg.overlap}" in out

    # 2) 打印的片段数 == 用同一份配置实际切出来的数量
    chunks = chunk_documents(
        load_documents(_DATA),
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
    )
    assert f"共 {len(chunks)} 个片段" in out

