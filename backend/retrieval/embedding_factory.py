from __future__ import annotations

import os
from typing import List

from backend.retrieval.embeddings import HashingEmbeddingModel

try:
    from llama_index.core.embeddings import BaseEmbedding
except ImportError:  # pragma: no cover - optional dependency guard.
    BaseEmbedding = object


class LocalHashEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter for the deterministic local fallback."""

    dim: int = 384

    def _model(self) -> HashingEmbeddingModel:
        return HashingEmbeddingModel(dim=self.dim)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model().embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model().embed(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def create_llamaindex_embedding() -> BaseEmbedding:
    provider = os.getenv("IP_AGENT_EMBEDDING_PROVIDER", "bge_m3").strip().lower()

    if provider in {"local", "hash", "hashing"}:
        return LocalHashEmbedding()

    if provider == "openai":
        return _create_openai_embedding()

    if provider in {"bge_m3", "bge-m3", "huggingface", "hf"}:
        return _create_huggingface_embedding(provider)

    raise ValueError(
        "Unsupported IP_AGENT_EMBEDDING_PROVIDER. "
        "Use one of: local, openai, bge_m3, huggingface."
    )


def _create_openai_embedding() -> BaseEmbedding:
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "OpenAI embeddings require `llama-index-embeddings-openai`. "
            "Install it with `pip install llama-index-embeddings-openai`."
        ) from exc

    api_key = (
        os.getenv("IP_AGENT_EMBEDDING_API_KEY")
        or os.getenv("IP_AGENT_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "OpenAI embeddings require IP_AGENT_EMBEDDING_API_KEY, "
            "IP_AGENT_LLM_API_KEY, or OPENAI_API_KEY."
        )

    kwargs = {
        "model": os.getenv("IP_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": api_key,
    }
    api_base = os.getenv("IP_AGENT_EMBEDDING_BASE_URL") or os.getenv("IP_AGENT_LLM_BASE_URL")
    if api_base:
        kwargs["api_base"] = api_base.rstrip("/")

    return OpenAIEmbedding(**kwargs)


def _create_huggingface_embedding(provider: str) -> BaseEmbedding:
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "HuggingFace embeddings require `llama-index-embeddings-huggingface`. "
            "Install it with `pip install llama-index-embeddings-huggingface`."
        ) from exc

    default_model = "BAAI/bge-m3" if provider in {"bge_m3", "bge-m3"} else "BAAI/bge-small-en-v1.5"
    model_name = os.getenv("IP_AGENT_EMBEDDING_MODEL", default_model)
    return HuggingFaceEmbedding(model_name=model_name)
