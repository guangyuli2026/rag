from rag_assistant.cli import main


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
