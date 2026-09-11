"""文本切分：把原文切成带可追溯偏移的片段（Chunk）。

本阶段用一个简单、可解释的策略：
  1. 按空行把原文切成段落；
  2. 贪心合并相邻段落，直到接近 chunk_size（字符数）；
  3. 单个段落超过 chunk_size 时，在段落内部按 chunk_size 硬切；
  4. 相邻片段之间按 overlap 字符数重叠（让被切在边界的语义尽量完整）。

每个 Chunk 都记录它在原文中的 [char_start, char_end) 字符区间，
凭这个区间可以随时从原文切片回来验证，也可以作为后续评测的定位证据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 连续两个以上换行（中间允许空格/制表符）视为段落分隔
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class Chunk:
    """一个可追溯的文本片段。

    chunk_id:     f"{doc_id}#{序号}"，如 doc_001#3，检索结果里直接引用
    doc_id:       来源文档标识
    title:        来源文档标题
    text:         片段文本
    char_start:   片段在原文中的起始偏移（含）
    char_end:     片段在原文中的结束偏移（不含），text == content[char_start:char_end]
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str
    char_start: int
    char_end: int


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    """按空行切分，返回 [(段落起始偏移, 段落文本)]。"""
    paragraphs: list[tuple[int, str]] = []
    start = 0
    for match in _PARAGRAPH_SPLIT.finditer(text):
        paragraphs.append((start, text[start : match.start()]))
        start = match.end()
    paragraphs.append((start, text[start:]))
    return paragraphs


def _basic_blocks(text: str, chunk_size: int) -> list[tuple[str, int, int]]:
    """不含重叠的基础切分：合并段落至 chunk_size，超长段落内部硬切。"""
    paragraphs = _split_paragraphs(text)
    blocks: list[tuple[str, int, int]] = []

    index = 0
    while index < len(paragraphs):
        start, para = paragraphs[index]
        if not para:
            index += 1
            continue

        # 超长段落：内部按 chunk_size 硬切（后续统一补 overlap）
        if len(para) > chunk_size:
            pos = start
            end_of_para = start + len(para)
            while pos < end_of_para:
                end = min(pos + chunk_size, end_of_para)
                blocks.append((text[pos:end], pos, end))
                pos = end
            index += 1
            continue

        # 普通段落：贪心向后合并，直到加入下一段会超过 chunk_size
        # 注意：块直接取原文区间 text[block_start:block_end]，保留段落间的
        # 原始分隔符，保证 text == content[char_start:char_end] 恒成立。
        block_start = start
        block_end = start + len(para)
        next_index = index + 1
        while next_index < len(paragraphs):
            next_start, next_para = paragraphs[next_index]
            if not next_para:
                next_index += 1
                continue
            if len(next_para) > chunk_size:
                break
            # 精确计算：块长度 = 原文跨度（含原始段落分隔符）+ 下一段长度
            if next_start - block_start + len(next_para) > chunk_size:
                break
            block_end = next_start + len(next_para)
            next_index += 1

        block_text = text[block_start:block_end]
        blocks.append((block_text, block_start, block_end))
        index = next_index

    return blocks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[tuple[str, int, int]]:
    """把一段原文切成带偏移的块，返回 [(文本, char_start, char_end)]。

    参数：
      chunk_size  块的最大字符数（不是 token 数，见下文说明）
      overlap     相邻块之间重叠的字符数；0 表示不重叠
    """
    if not text:
        return []
    if chunk_size < 1:
        raise ValueError("chunk_size 必须 >= 1")

    blocks = _basic_blocks(text, chunk_size)

    # 统一补重叠：从第 2 块起，把起点向前回退 overlap 字符
    if overlap > 0:
        for i in range(1, len(blocks)):
            prev_text, prev_start, prev_end = blocks[i - 1]
            cur_text, cur_start, cur_end = blocks[i]
            new_start = max(prev_start + 1, cur_start - overlap)
            if new_start < cur_start:
                blocks[i] = (text[new_start:cur_end], new_start, cur_end)

    return blocks


def chunk_document(doc_id: str, title: str, content: str, **kwargs) -> list[Chunk]:
    """把一份文档切成 Chunk 列表，编号从 0 开始。"""
    return [
        Chunk(
            chunk_id=f"{doc_id}#{index}",
            doc_id=doc_id,
            title=title,
            text=text,
            char_start=start,
            char_end=end,
        )
        for index, (text, start, end) in enumerate(chunk_text(content, **kwargs))
    ]


def chunk_documents(
    documents: list,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """把多份文档全部切成片段。重复文档已在加载层去重，这里直接顺序切分。"""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                doc.doc_id,
                doc.title,
                doc.content,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks
