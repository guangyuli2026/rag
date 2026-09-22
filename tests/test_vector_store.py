from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pytest

from rag_assistant.config import ChunkingConfig
from rag_assistant.documents import Document, load_documents
from rag_assistant.embeddings import OfflineHashEmbedding
from rag_assistant.vector_store import (
    IndexCompatibilityError,
    VectorIndex,
    build_vector_index,
)


def _doc(path: str, content: str, doc_id: str = "doc_001") -> Document:
    return Document(
        doc_id=doc_id,
        title=Path(path).stem,
        relative_path=path,
        content=content,
    )


def test_repeated_import_does_not_duplicate_chunks():
    provider = OfflineHashEmbedding(dimension=32)
    config = ChunkingConfig(chunk_size=100, overlap=0)
    document = _doc("guide.md", "绿萝喜欢散射光。")

    index, stats = build_vector_index([document, document], provider, config)

    assert stats.document_count == 1
    assert stats.chunk_count == 1
    assert len(index.chunks) == 1


def test_save_load_preserves_source_mapping_and_search(tmp_path):
    provider = OfflineHashEmbedding(dimension=32)
    config = ChunkingConfig(chunk_size=100, overlap=0)
    documents = [
        _doc("plants.md", "绿萝喜欢散射光。"),
        _doc("hiking.md", "登山杖能减轻膝盖压力。", doc_id="doc_002"),
    ]
    index, _ = build_vector_index(documents, provider, config)
    index.save(tmp_path / "index")

    loaded = VectorIndex.load(
        tmp_path / "index", provider, documents=documents, chunking_config=config
    )
    result = loaded.search("登山杖能减轻膝盖压力。", provider, top_k=1)

    assert result.hits[0].chunk.relative_path == "hiking.md"
    assert result.hits[0].chunk.text == "登山杖能减轻膝盖压力。"
    # Identical text has identical nonzero vectors, so its cosine is 1.
    assert result.hits[0].score == pytest.approx(1.0)
    np.testing.assert_array_equal(loaded.vectors, index.vectors)
    assert loaded.chunks == index.chunks
    by_path = {doc.relative_path: doc for doc in documents}
    for chunk in loaded.chunks:
        assert by_path[chunk.relative_path].content[chunk.char_start:chunk.char_end] == chunk.text
    assert result.elapsed_ms >= 0


def test_changed_or_deleted_document_requires_rebuild(tmp_path):
    provider = OfflineHashEmbedding(dimension=32)
    config = ChunkingConfig(chunk_size=100, overlap=0)
    original = [_doc("guide.md", "绿萝喜欢散射光。")]
    index, _ = build_vector_index(original, provider, config)
    index.save(tmp_path / "index")

    with pytest.raises(IndexCompatibilityError, match="资料内容已变化"):
        VectorIndex.load(
            tmp_path / "index",
            provider,
            documents=[_doc("guide.md", "绿萝需要明亮环境。")],
            chunking_config=config,
        )

    with pytest.raises(IndexCompatibilityError, match="资料内容已变化"):
        VectorIndex.load(
            tmp_path / "index",
            provider,
            documents=[],
            chunking_config=config,
        )


def test_incompatible_model_dimension_and_chunking_require_rebuild(tmp_path):
    provider = OfflineHashEmbedding(dimension=32)
    config = ChunkingConfig(chunk_size=100, overlap=0)
    index, _ = build_vector_index([_doc("guide.md", "散射光")], provider, config)
    index.save(tmp_path / "index")

    with pytest.raises(IndexCompatibilityError, match="维度不一致"):
        VectorIndex.load(
            tmp_path / "index",
            OfflineHashEmbedding(dimension=64),
            chunking_config=config,
        )

    with pytest.raises(IndexCompatibilityError, match="切分参数不一致"):
        VectorIndex.load(
            tmp_path / "index",
            provider,
            chunking_config=ChunkingConfig(chunk_size=50, overlap=0),
        )

    with pytest.raises(IndexCompatibilityError, match="模型标识不一致"):
        VectorIndex.load(tmp_path / "index", replace(provider, model_id="other-model"))

    with pytest.raises(IndexCompatibilityError, match="输入或参数配置不一致"):
        VectorIndex.load(tmp_path / "index", replace(provider, ngram_max=4))


class FixedVectors:
    """Known geometry verifies cosine math independently from hashing quality."""

    model_id = "test-geometry"
    dimension = 2
    config_signature = "test-geometry-plain-input-v1"

    def embed_documents(self, texts):
        assert list(texts) == ["A", "B", "C", "D"]
        return np.array([[3, 4], [0, 2], [-2, 0], [0, 0]], dtype=np.float32)

    def embed_query(self, text):
        assert text == "query"
        return np.array([[3, 0]], dtype=np.float32)


def test_cosine_uses_lengths_and_excludes_zero_document_vectors():
    provider = FixedVectors()
    documents = [_doc(f"{letter}.txt", letter, letter) for letter in "ABCD"]
    index, _ = build_vector_index(documents, provider)
    result = index.search("query", provider, top_k=10)

    assert [hit.chunk.doc_id for hit in result.hits] == list("ABC")
    assert [hit.score for hit in result.hits] == pytest.approx([0.6, 0.0, -1.0])
    assert len(index.search("query", provider, top_k=1).hits) == 1
    assert index.search("   ", provider).hits == []
    with pytest.raises(ValueError, match="top_k"):
        index.search("query", provider, top_k=-1)


def test_empty_corpus_and_zero_query_vector(tmp_path):
    provider = OfflineHashEmbedding()
    index, _ = build_vector_index([], provider)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, provider, documents=[])
    assert loaded.vectors.shape == (0, provider.dimension)
    assert loaded.search("query", provider).hits == []
    populated, _ = build_vector_index([_doc("a.txt", "test")], provider)
    assert populated.search("a", provider).hits == []  # No 2/3-character n-gram.


def test_snapshot_rebuild_applies_add_modify_delete_and_is_idempotent(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "b.txt"
    source.write_text("old information", encoding="utf-8")
    provider = OfflineHashEmbedding()
    index_dir = tmp_path / "index"
    original, _ = build_vector_index(load_documents(data), provider)
    original.save(index_dir)

    (data / "a.txt").write_text("new document", encoding="utf-8")
    with pytest.raises(IndexCompatibilityError, match="资料内容已变化"):
        VectorIndex.load(index_dir, provider, documents=load_documents(data))
    source.write_text("updated information", encoding="utf-8")
    updated, _ = build_vector_index(load_documents(data), provider)
    updated.save(index_dir)
    assert {chunk.relative_path for chunk in updated.chunks} == {"a.txt", "b.txt"}
    assert all("old information" not in chunk.text for chunk in updated.chunks)

    source.unlink()
    with pytest.raises(IndexCompatibilityError, match="资料内容已变化"):
        VectorIndex.load(index_dir, provider, documents=load_documents(data))
    remaining = load_documents(data)
    rebuilt, _ = build_vector_index(remaining, provider)
    rebuilt.save(index_dir)
    repeated, _ = build_vector_index(remaining + remaining, provider)
    repeated.save(index_dir)
    loaded = VectorIndex.load(index_dir, provider, documents=remaining)
    assert [chunk.relative_path for chunk in loaded.chunks] == ["a.txt"]
    assert loaded.metadata == rebuilt.metadata
    np.testing.assert_array_equal(loaded.vectors, rebuilt.vectors)


@pytest.mark.parametrize("failure", ["schema", "shape", "nan", "source", "count", "chunker"])
def test_invalid_archive_requires_rebuild(tmp_path, failure):
    provider = OfflineHashEmbedding(dimension=32)
    documents = [_doc("a.txt", "example text")]
    index, _ = build_vector_index(documents, provider)
    index.save(tmp_path)
    path = tmp_path / "index.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = json.loads(archive["metadata_json"].item())
        vectors = archive["vectors"].copy()
    if failure == "schema":
        payload["metadata"]["schema_version"] = 999
    elif failure == "shape":
        vectors = vectors[:, :-1]
    elif failure == "nan":
        vectors[0, 0] = np.nan
    elif failure == "source":
        payload["chunks"][0]["relative_path"] = "wrong.txt"
    elif failure == "count":
        payload["metadata"]["chunk_count"] = 999
    else:
        payload["metadata"]["chunker_version"] = "other"
    np.savez_compressed(path, vectors=vectors, metadata_json=np.array(json.dumps(payload)))
    with pytest.raises(IndexCompatibilityError, match="重新建立索引"):
        VectorIndex.load(tmp_path, provider, documents=documents)


def test_interrupted_save_preserves_previous_snapshot(tmp_path, monkeypatch):
    provider = OfflineHashEmbedding()
    original, _ = build_vector_index([_doc("a.txt", "old")], provider)
    original.save(tmp_path)
    previous_bytes = (tmp_path / "index.npz").read_bytes()
    changed, _ = build_vector_index([_doc("a.txt", "new")], provider)

    def fail_replace(*args):
        raise OSError("simulated write failure")

    monkeypatch.setattr("rag_assistant.vector_store.os.replace", fail_replace)
    with pytest.raises(OSError):
        changed.save(tmp_path)
    assert (tmp_path / "index.npz").read_bytes() == previous_bytes
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("bad", [np.zeros((2, 5)), np.full((1, 32), np.nan)])
def test_bad_provider_output_is_rejected(monkeypatch, bad):
    provider = OfflineHashEmbedding(dimension=32)
    monkeypatch.setattr(OfflineHashEmbedding, "embed_documents", lambda self, texts: bad)
    with pytest.raises(ValueError):
        build_vector_index([_doc("a.txt", "example")], provider)
