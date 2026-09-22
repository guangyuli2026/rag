import json

import pytest

from rag_assistant import cli
from rag_assistant.embeddings import OfflineHashEmbedding


@pytest.fixture(autouse=True)
def offline_provider(monkeypatch):
    monkeypatch.setattr(cli, "provider_from_environment", lambda: OfflineHashEmbedding())


def test_cli_build_search_then_reject_stale_and_corrupt_index(tmp_path, capsys):
    data = tmp_path / "docs"
    data.mkdir()
    source = data / "guide.txt"
    source.write_text("example for retrieval", encoding="utf-8")
    index_dir = tmp_path / "index"
    common = ["--data", str(data)]
    build = common + ["vector-index", "--index-dir", str(index_dir)]
    query = common + ["vector-search", "example for retrieval", "--index-dir", str(index_dir)]

    assert cli.main(query) == 2
    assert "vector-index" in capsys.readouterr().out
    assert cli.main(build) == 0
    output = capsys.readouterr().out
    assert "文档=1 片段=1" in output and "耗时=" in output
    assert "example for retrieval" not in output  # Build logs do not dump documents.
    assert cli.main(query) == 0
    output = capsys.readouterr().out
    assert "余弦相似度 1.0000" in output
    assert "guide.txt" in output and "离线测试替身" in output

    source.write_text("changed source", encoding="utf-8")
    assert cli.main(query) == 2
    assert "资料内容已变化" in capsys.readouterr().out
    assert cli.main(build) == 0
    assert cli.main(query) == 0
    (index_dir / "index.npz").write_bytes(b"broken")
    assert cli.main(query) == 2
    assert "重新建立索引" in capsys.readouterr().out


def test_compare_saves_ten_rows(tmp_path, capsys):
    report_path = tmp_path / "result.json"
    assert cli.main(["compare", "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["questions"]) == 10
    assert "草稿" in capsys.readouterr().out


@pytest.mark.parametrize("command", [["vector-search", "query"], ["compare"]])
def test_negative_top_k_is_rejected(command):
    with pytest.raises(SystemExit) as error:
        cli.main(command + ["-k", "-1"])
    assert error.value.code == 2
