"""Small NumPy-backed vector index with traceable chunk mappings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from .chunking import Chunk, chunk_documents
from .config import ChunkingConfig, DEFAULT_CHUNKING_CONFIG
from .documents import Document
from .embeddings import EmbeddingProvider

SCHEMA_VERSION = 2
CHUNKER_VERSION = "paragraph-backward-overlap-v1"


class IndexCompatibilityError(RuntimeError):
    """Raised when an index cannot be used with the current configuration."""


@dataclass(frozen=True)
class IndexMetadata:
    schema_version: int
    model_id: str
    dimension: int
    embedding_config: str
    chunker_version: str
    chunk_size: int
    overlap: int
    content_version: str
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class BuildStats:
    document_count: int
    chunk_count: int
    elapsed_ms: float


@dataclass(frozen=True)
class VectorHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class VectorSearchResult:
    hits: list[VectorHit]
    elapsed_ms: float


def _unique_documents(documents: Sequence[Document]) -> list[Document]:
    """Deduplicate repeated imports by relative path, then sort deterministically."""
    by_path: dict[str, Document] = {}
    for document in documents:
        key = document.relative_path or document.doc_id
        if key in by_path and by_path[key] != document:
            raise ValueError("同一路径出现不同版本的文档，请提供单一资料快照。")
        by_path[key] = document
    return [by_path[key] for key in sorted(by_path)]


def content_version(documents: Sequence[Document]) -> str:
    """Hash document paths and contents without logging or storing the raw input."""
    snapshot = [asdict(document) for document in _unique_documents(documents)]
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _check_vectors(vectors: np.ndarray, row_count: int, dimension: int) -> None:
    if vectors.ndim != 2 or vectors.shape != (row_count, dimension):
        raise ValueError(
            f"向量形状不匹配：得到 {vectors.shape}，期望 {(row_count, dimension)}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("向量包含非有限值")


@dataclass
class VectorIndex:
    vectors: np.ndarray
    chunks: list[Chunk]
    metadata: IndexMetadata

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        _check_vectors(self.vectors, len(self.chunks), self.metadata.dimension)
        if self.metadata.chunk_count != len(self.chunks):
            raise ValueError("索引片段数不一致")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("索引存在重复片段 ID")
        for chunk in self.chunks:
            if chunk.char_start < 0 or chunk.char_end - chunk.char_start != len(chunk.text):
                raise ValueError("片段原文范围与文本长度不一致")

    def _check_provider(self, provider: EmbeddingProvider) -> None:
        if provider.model_id != self.metadata.model_id:
            raise IndexCompatibilityError(
                "Embedding 模型标识不一致；请用当前模型重新建立索引。"
            )
        if provider.dimension != self.metadata.dimension:
            raise IndexCompatibilityError(
                "Embedding 维度不一致；请用当前配置重新建立索引。"
            )
        if provider.config_signature != self.metadata.embedding_config:
            raise IndexCompatibilityError("Embedding 输入或参数配置不一致；请重新建立索引。")

    def search(
        self,
        query: str,
        provider: EmbeddingProvider,
        top_k: int = 3,
    ) -> VectorSearchResult:
        """Embed one query and return cosine-similarity Top-k hits."""
        started = time.perf_counter()
        self._check_provider(provider)
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        if not query.strip() or not self.chunks:
            return VectorSearchResult([], (time.perf_counter() - started) * 1000)

        query_vector = np.asarray(provider.embed_query(query), dtype=np.float64)
        _check_vectors(query_vector, 1, self.metadata.dimension)
        query_norm = float(np.linalg.norm(query_vector[0]))
        if query_norm == 0:
            return VectorSearchResult([], (time.perf_counter() - started) * 1000)

        vectors = self.vectors.astype(np.float64)
        row_norms = np.linalg.norm(vectors, axis=1)
        denominators = row_norms * query_norm
        scores = np.divide(
            vectors @ query_vector[0],
            denominators,
            out=np.zeros(len(self.chunks), dtype=np.float64),
            where=denominators != 0,
        )
        # Zero vectors have no cosine direction; exclude them even if other scores are negative.
        candidates = np.flatnonzero(row_norms > 0)
        scores = np.clip(scores, -1.0, 1.0)
        order = candidates[np.argsort(-scores[candidates], kind="stable")[:top_k]]
        hits = [VectorHit(self.chunks[int(i)], float(scores[int(i)])) for i in order]
        return VectorSearchResult(hits, (time.perf_counter() - started) * 1000)

    def save(self, index_dir: Path) -> None:
        """Persist vectors and their source mapping under a local index directory."""
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": asdict(self.metadata),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        # One atomic replacement keeps vectors and row mappings from different builds apart.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=index_dir, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream,
                    vectors=self.vectors,
                    metadata_json=np.array(json.dumps(payload, ensure_ascii=False)),
                )
            os.replace(temporary, index_dir / "index.npz")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        provider: EmbeddingProvider,
        documents: Sequence[Document] | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> "VectorIndex":
        archive_path = index_dir / "index.npz"
        if not archive_path.exists():
            raise FileNotFoundError(
                f"索引不完整：{index_dir}。请先运行 vector-index 建立索引。"
            )
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                payload = json.loads(archive["metadata_json"].item())
                if payload["metadata"]["schema_version"] != SCHEMA_VERSION:
                    raise IndexCompatibilityError("索引格式版本不兼容；请重新建立索引。")
                metadata = IndexMetadata(**payload["metadata"])
                index = cls(
                    vectors=archive["vectors"],
                    chunks=[Chunk(**item) for item in payload["chunks"]],
                    metadata=metadata,
                )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, BadZipFile) as exc:
            raise IndexCompatibilityError("索引文件损坏或格式无效；请重新建立索引。") from exc
        if metadata.chunker_version != CHUNKER_VERSION:
            raise IndexCompatibilityError("切分算法版本不一致；请重新建立索引。")
        index._check_provider(provider)

        if chunking_config is not None and (
            metadata.chunk_size != chunking_config.chunk_size
            or metadata.overlap != chunking_config.overlap
        ):
            raise IndexCompatibilityError("切分参数不一致；请重新建立索引。")
        if documents is not None:
            if metadata.content_version != content_version(documents):
                raise IndexCompatibilityError("资料内容已变化（新增、修改或删除）；请重新建立索引。")
            expected_chunks = chunk_documents(
                _unique_documents(documents),
                chunk_size=metadata.chunk_size,
                overlap=metadata.overlap,
            )
            if index.chunks != expected_chunks:
                raise IndexCompatibilityError("索引来源映射与原文不一致；请重新建立索引。")
        return index


def build_vector_index(
    documents: Sequence[Document],
    provider: EmbeddingProvider,
    chunking_config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> tuple[VectorIndex, BuildStats]:
    """Build an index from the current document snapshot."""
    started = time.perf_counter()
    if chunking_config.chunk_size < 1 or not 0 <= chunking_config.overlap < chunking_config.chunk_size:
        raise ValueError("切分配置需要 chunk_size >= 1 且 0 <= overlap < chunk_size")
    unique_docs = _unique_documents(documents)
    chunks = chunk_documents(
        unique_docs,
        chunk_size=chunking_config.chunk_size,
        overlap=chunking_config.overlap,
    )
    vectors = (
        np.asarray(provider.embed_documents([chunk.text for chunk in chunks]), dtype=np.float32)
        if chunks else np.empty((0, provider.dimension), dtype=np.float32)
    )
    _check_vectors(vectors, len(chunks), provider.dimension)
    metadata = IndexMetadata(
        schema_version=SCHEMA_VERSION,
        model_id=provider.model_id,
        dimension=provider.dimension,
        embedding_config=provider.config_signature,
        chunker_version=CHUNKER_VERSION,
        chunk_size=chunking_config.chunk_size,
        overlap=chunking_config.overlap,
        content_version=content_version(unique_docs),
        document_count=len(unique_docs),
        chunk_count=len(chunks),
    )
    index = VectorIndex(vectors=vectors, chunks=chunks, metadata=metadata)
    stats = BuildStats(
        document_count=len(unique_docs),
        chunk_count=len(chunks),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    return index, stats
