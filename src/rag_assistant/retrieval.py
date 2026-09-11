"""简单关键词检索：返回与查询最相关的 Top-k 片段。

当前是一个刻意保持简单的基线，方便解释和对比：
  - 查询先被切成"关键词"：按空白和常见标点切分；
  - 中文连续文本整体作为一个关键词（中文没有空格分词，不能只按空格切）；
  - 片段得分 = 各关键词在片段文本中出现的次数之和（大小写不敏感）；
  - 按得分降序取前 k 个，得分为 0（没有命中）的片段不返回。

局限（后续步骤会逐条改进）：
  - 纯字面匹配："检索"匹配不到"搜索"这类同义表达；
  - 没有词形归一：英文 "RAG" 与 "rag" 已用 lower() 统一，但复数/时态仍不处理；
  - 没有权重：出现 1 次的常见词和关键术语得分一样；
  - 线性扫描全部片段，数据量大了会慢（步骤 7 再谈倒排索引/BM25）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk

# 关键词分隔符：空白 + 常见中英文标点。
# 用 re.escape 逐字转义后拼接，避免手写字符类时 `[]`、`-` 等字符的歧义解析。
_SEPARATORS = "，。！？、；：\"\"''（）《》〈〉、,.!?;:()[]{}<>/|\\—…·-"
_KEYWORD_SPLIT = re.compile(r"[\s" + re.escape(_SEPARATORS) + r"]+")


@dataclass(frozen=True)
class Hit:
    """一次检索命中：片段 + 得分。得分只代表字面匹配强度，不是答案概率。"""

    chunk: Chunk
    score: int


def split_keywords(query: str) -> list[str]:
    """把查询切成关键词列表。中文连续文本整体保留为一个关键词。"""
    return [word for word in _KEYWORD_SPLIT.split(query) if word]


def score_chunk(text: str, keywords: list[str]) -> int:
    """片段得分 = 每个关键词在该片段中出现的次数之和（大小写不敏感）。"""
    lowered = text.lower()
    return sum(lowered.count(word.lower()) for word in keywords)


def search(chunks: list[Chunk], query: str, top_k: int = 3) -> list[Hit]:
    """在全部片段中做关键词检索，返回得分最高的 Top-k 个命中。"""
    keywords = split_keywords(query)
    if not keywords:
        return []

    hits = [Hit(chunk, score_chunk(chunk.text, keywords)) for chunk in chunks]
    hits = [hit for hit in hits if hit.score > 0]
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]
