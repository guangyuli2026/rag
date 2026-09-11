"""Command-line entry point for the first, offline project demo."""

from __future__ import annotations#有些新语法 / 特性本来属于未来版本，想在旧版本 Python 里提前启用，就从 `__future__` 导入

import argparse #**命令行参数解析器**。用来读取你在终端输入的参数
from typing import Sequence

from . import __version__#`.` 代表**当前包**，从同包下读取版本号变量 `__version__`


def build_parser() -> argparse.ArgumentParser:
    """Build the parser separately so the command can be tested without a process."""
    parser = argparse.ArgumentParser(description="RAG 学习项目的离线演示")
    parser.add_argument(
        "--question",
        help="可选的演示问题；当前版本只回显问题，不调用模型",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rag-learning-assistant {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic demo and return a shell-friendly exit code."""
    args = build_parser().parse_args(argv)#parser对象自带的实例方法
    print("RAG 学习项目演示")
    print("状态：模型服务尚未接入")
    if args.question:
        print(f"问题：{args.question}")
        print("回答：当前演示不生成答案；下一步将加入本地资料检索。")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
