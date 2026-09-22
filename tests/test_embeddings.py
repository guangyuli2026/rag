import numpy as np
import pytest

from rag_assistant.embeddings import (
    EmbeddingConfigurationError, OfflineHashEmbedding, provider_from_environment,
)


def test_offline_embedding_is_deterministic_and_normalized():
    provider = OfflineHashEmbedding(dimension=32)
    first = provider.embed(["绿萝需要散射光", "绿萝需要散射光"])
    second = provider.embed(["绿萝需要散射光", "完全不同的文本"])

    assert first.shape == (2, 32)
    assert np.array_equal(first[0], second[0])
    assert np.isclose(np.linalg.norm(first[0]), 1.0)
    assert np.isfinite(first).all()


def test_provider_defaults_to_offline_without_real_configuration():
    provider = provider_from_environment({})

    assert provider.model_id == "offline-hash-char-ngram-v2"


def test_local_dotenv_is_read_but_environment_takes_precedence(tmp_path, monkeypatch):
    for key in (
        "RAG_EMBEDDING_PROVIDER", "RAG_EMBEDDING_MODEL", "RAG_EMBEDDING_API_KEY",
        "RAG_EMBEDDING_BASE_URL", "RAG_OFFLINE_DIMENSION", "RAG_OFFLINE_NGRAM_MIN",
        "RAG_OFFLINE_NGRAM_MAX",
    ):
        monkeypatch.delenv(key, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text('RAG_EMBEDDING_PROVIDER="offline"\nRAG_OFFLINE_DIMENSION=64\n', encoding="utf-8")
    assert provider_from_environment(dotenv_path=dotenv).dimension == 64
    monkeypatch.setenv("RAG_OFFLINE_DIMENSION", "256")
    assert provider_from_environment(dotenv_path=dotenv).dimension == 256


@pytest.mark.parametrize("settings", [
    {"RAG_EMBEDDING_PROVIDER": "not-implemented"},
    {"RAG_EMBEDDING_MODEL": "not-selected"},
    {"RAG_OFFLINE_DIMENSION": "0"},
    {"RAG_OFFLINE_NGRAM_MIN": "4", "RAG_OFFLINE_NGRAM_MAX": "2"},
    {"RAG_OFFLINE_DIMENSION": "not-a-number"},
])
def test_invalid_configuration_is_explicit(settings):
    with pytest.raises(EmbeddingConfigurationError):
        provider_from_environment(settings)


def test_configuration_errors_never_echo_credentials():
    value = "test-placeholder-not-a-real-credential"
    with pytest.raises(EmbeddingConfigurationError) as error:
        provider_from_environment({"RAG_EMBEDDING_API_KEY": value})
    assert value not in str(error.value)


def test_document_query_format_is_compatible_and_config_changes_are_detectable():
    provider = OfflineHashEmbedding()
    np.testing.assert_array_equal(provider.embed_documents(["same text"]), provider.embed_query("same text"))
    assert provider.config_signature != OfflineHashEmbedding(ngram_max=4).config_signature
