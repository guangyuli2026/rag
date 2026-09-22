from dataclasses import dataclass
@dataclass(frozen=True)
class ChunkingConfig:
    """文本切分配置。

    chunk_size:   基础块的最大字符数；补重叠后的长度可能更大
    overlap:      从第二块开始向前扩展的字符数
    """

    chunk_size: int = 500
    overlap: int = 50
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
