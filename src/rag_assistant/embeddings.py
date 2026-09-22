"""Embedding providers used by the vector-retrieval baseline.

The project has no selected model service yet.  ``OfflineHashEmbedding`` is a
deterministic test double for validating the index pipeline; its scores are not
evidence of real semantic-search quality.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from dotenv import dotenv_values


class EmbeddingConfigurationError(RuntimeError):
    """Raised when a requested provider has not been configured."""


class EmbeddingProvider(Protocol):
    """Small interface shared by offline and future real providers."""

    model_id: str
    dimension: int

    @property
    def config_signature(self) -> str:
        """Public configuration fingerprint; must never contain credentials."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dimension) matrix using the provider's document format."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (1, dimension) matrix using the compatible query format."""


@dataclass(frozen=True)
class OfflineHashEmbedding:
    """Deterministic local substitute based on hashed character n-grams.

    Character n-grams make the substitute useful for testing batching,
    persistence and ranking while keeping its limitations explicit: it does
    not understand meaning, synonyms or world knowledge.
    """

    dimension: int = 128
    ngram_min: int = 2
    ngram_max: int = 3
    model_id: str = "offline-hash-char-ngram-v2"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension 必须 >= 1")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("ngram 范围无效")

    def _bucket(self, feature: str) -> int:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dimension

    @property
    def config_signature(self) -> str:
        settings = {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "preprocessing": "lower-whitespace-v1",
            "weighting": "ngram-length",
            "input_format": "same-plain-text-for-query-and-document",
        }
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = " ".join(text.lower().split())
            for size in range(self.ngram_min, self.ngram_max + 1):
                for start in range(max(0, len(normalized) - size + 1)):
                    bucket = self._bucket(normalized[start : start + size])
                    # v2 固定使用长度加权计数；哈希碰撞仍会影响排序。
                    matrix[row, bucket] += float(size)

            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


def provider_from_environment(
    environ: dict[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> EmbeddingProvider:
    """Create the configured provider without printing secrets.

    Only the offline provider is implemented in this step.  A real provider
    must be selected from its official documentation before its adapter is
    added; this function deliberately does not guess a model name.
    """
    # Explicit mappings isolate tests; environment variables override local .env.
    if environ is None:
        path = dotenv_path or Path(__file__).resolve().parents[2] / ".env"
        values = {**dotenv_values(path, interpolate=False, encoding="utf-8-sig"), **os.environ}
    else:
        values = environ
    provider_name = (values.get("RAG_EMBEDDING_PROVIDER") or "offline").strip().lower()
    if provider_name in {"", "offline", "hash"}:
        if any(values.get(name) for name in (
            "RAG_EMBEDDING_MODEL", "RAG_EMBEDDING_API_KEY", "RAG_EMBEDDING_BASE_URL"
        )):
            raise EmbeddingConfigurationError(
                "当前为离线替身，但配置了真实模型参数。请先选择服务商并实现适配器。"
            )
        try:
            return OfflineHashEmbedding(
                dimension=int(values.get("RAG_OFFLINE_DIMENSION", "128")),
                ngram_min=int(values.get("RAG_OFFLINE_NGRAM_MIN", "2")),
                ngram_max=int(values.get("RAG_OFFLINE_NGRAM_MAX", "3")),
            )
        except (ValueError, TypeError) as exc:
            raise EmbeddingConfigurationError("离线维度和 n-gram 参数必须为有效的正整数。") from exc
    raise EmbeddingConfigurationError(
        "尚未实现所选 Embedding 服务商。请依据官方文档添加适配器；本次没有调用 API。"
    )
