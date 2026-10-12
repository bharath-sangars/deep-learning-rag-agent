"""
config.py
=========
Centralised configuration and LLM provider factory.

All settings are loaded from environment variables via pydantic-settings.
PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    """Supported LLM backend providers."""

    GROQ = "groq"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class EmbeddingProvider(str, Enum):
    """Supported embedding backend providers."""

    LOCAL = "local"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider
    llm_provider: LLMProvider = LLMProvider.GROQ
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        alias="GROQ_MODEL",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL",
    )

    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        alias="LMSTUDIO_BASE_URL",
    )
    lmstudio_model: str = Field(
        default="local-model",
        alias="LMSTUDIO_MODEL",
    )

    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # Vector store
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_DB_PATH",
    )
    chroma_collection_name: str = Field(
        default="deep_learning_corpus",
        alias="CHROMA_COLLECTION_NAME",
    )

    # Retrieval
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")
    similarity_threshold: float = Field(
        default=0.3,
        alias="SIMILARITY_THRESHOLD",
    )
    max_context_tokens: int = Field(
        default=3000,
        alias="MAX_CONTEXT_TOKENS",
    )

    # Application
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_title: str = Field(
        default="Deep Learning Interview Prep Agent",
        alias="APP_TITLE",
    )
    corpus_dir: str = Field(
        default="./data/corpus",
        alias="CORPUS_DIR",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings instance.

    The environment and .env file are parsed when the instance is created.
    """
    return Settings()


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

class LLMFactory:
    """Create a LangChain chat model for the selected provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self) -> BaseChatModel:
        """Instantiate the configured chat model."""

        provider = self._settings.llm_provider

        if provider == LLMProvider.GROQ:
            return self._create_groq()

        if provider == LLMProvider.OLLAMA:
            return self._create_ollama()

        if provider == LLMProvider.LMSTUDIO:
            return self._create_lmstudio()

        raise ValueError(f"Unsupported LLM provider: {provider}")

    def _create_groq(self) -> BaseChatModel:
        """Create a Groq chat model. Requires GROQ_API_KEY."""

        from langchain_groq import ChatGroq

        api_key = self._settings.groq_api_key.strip()

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is missing. Add your API key to the .env file."
            )

        return ChatGroq(
            model=self._settings.groq_model,
            api_key=api_key,
            temperature=0,
        )

    def _create_ollama(self) -> BaseChatModel:
        """Create a locally hosted Ollama chat model."""

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self._settings.ollama_model,
            base_url=self._settings.ollama_base_url,
            temperature=0,
        )

    def _create_lmstudio(self) -> BaseChatModel:
        """Create a chat model using LM Studio's OpenAI-compatible API."""

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self._settings.lmstudio_model,
            base_url=self._settings.lmstudio_base_url,
            api_key="lm-studio",
            temperature=0,
        )


# ---------------------------------------------------------------------------
# Embedding Factory
# ---------------------------------------------------------------------------

class EmbeddingFactory:
    """Create a LangChain-compatible embedding model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self):
        """Instantiate the configured embedding model."""

        provider = self._settings.embedding_provider

        if provider == EmbeddingProvider.LOCAL:
            return self._create_local()

        if provider == EmbeddingProvider.OPENAI:
            return self._create_openai()

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    def _create_local(self):
        """Create a local Hugging Face sentence-transformers model."""

        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _create_openai(self):
        """Create OpenAI embeddings. Requires OPENAI_API_KEY."""

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
        )