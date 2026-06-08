from __future__ import annotations

import pytest

from backend.retrieval.embedding_factory import LocalHashEmbedding, create_llamaindex_embedding


def test_embedding_factory_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "local")

    embedding = create_llamaindex_embedding()

    assert isinstance(embedding, LocalHashEmbedding)


def test_embedding_factory_accepts_local_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "hashing")

    embedding = create_llamaindex_embedding()

    assert isinstance(embedding, LocalHashEmbedding)


def test_embedding_factory_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_AGENT_EMBEDDING_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported IP_AGENT_EMBEDDING_PROVIDER"):
        create_llamaindex_embedding()
