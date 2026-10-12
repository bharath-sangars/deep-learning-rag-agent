"""
store.py
Persistent ChromaDB vector-store manager for the Deep Learning RAG Agent.

File location:
src/rag_agent/vectorstore/store.py
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """Manage persistent storage and semantic retrieval using ChromaDB."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client: Any = None
        self._collection: Any = None
        self._initialise()

    def _initialise(self) -> None:
        """Initialize the persistent ChromaDB collection."""
        db_path = Path(self._settings.chroma_db_path).expanduser()
        db_path.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=str(db_path)
            )
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            raise RuntimeError(
                f"Could not initialize ChromaDB: {exc}"
            ) from exc

    @staticmethod
    def generate_chunk_id(
        source: str, chunk_text: str
    ) -> str:
        """Generate a stable ID for a source and chunk."""
        raw = f"{source}::{chunk_text}"
        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:32]

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        """Convert a model or mapping to a dictionary."""
        if value is None:
            return {}

        if isinstance(value, dict):
            return dict(value)

        for method_name in ("to_dict", "model_dump", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    result = method()
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

        return {}

    @classmethod
    def _metadata_to_dict(cls, metadata: Any) -> dict[str, Any]:
        """Convert metadata into ChromaDB-compatible scalar values."""
        raw = cls._to_dict(metadata)
        cleaned: dict[str, Any] = {}

        for key, value in raw.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                cleaned[str(key)] = value
            elif isinstance(value, (list, tuple)):
                cleaned[str(key)] = ", ".join(
                    str(item) for item in value
                )
            else:
                cleaned[str(key)] = str(value)

        return cleaned

    @staticmethod
    def _metadata_from_dict(metadata: dict[str, Any]) -> Any:
        """Build ChunkMetadata with Pydantic v1/v2 compatibility."""
        factory = getattr(ChunkMetadata, "from_dict", None)

        if callable(factory):
            try:
                return factory(metadata)
            except Exception:
                pass

        try:
            return ChunkMetadata(**metadata)
        except Exception:
            fields = getattr(
                ChunkMetadata, "model_fields", None
            )
            if fields is None:
                fields = getattr(
                    ChunkMetadata, "__fields__", {}
                )

            filtered = {
                key: value
                for key, value in metadata.items()
                if key in fields
            }
            return ChunkMetadata(**filtered)

    def check_duplicate(self, chunk_id: str) -> bool:
        """Return True if the given chunk ID already exists."""
        response = self._collection.get(ids=[str(chunk_id)])
        return bool(response.get("ids"))

    def ingest(
        self, chunks: list[DocumentChunk]
    ) -> IngestionResult:
        """Embed and store chunks, tracking inserted and skipped items."""
        result = IngestionResult()

        if not chunks:
            return result

        batch_size = 64

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            new_chunks: list[DocumentChunk] = []

            for chunk in batch:
                try:
                    if self.check_duplicate(str(chunk.chunk_id)):
                        result.skipped += 1
                    else:
                        new_chunks.append(chunk)
                except Exception as exc:
                    message = (
                        f"Duplicate check failed for "
                        f"{chunk.chunk_id}: {exc}"
                    )
                    result.errors.append(message)
                    logger.warning(message)

            if not new_chunks:
                continue

            try:
                texts = [
                    str(chunk.chunk_text)
                    for chunk in new_chunks
                ]
                ids = [
                    str(chunk.chunk_id)
                    for chunk in new_chunks
                ]
                embeddings = self._embeddings.embed_documents(
                    texts
                )
                metadatas = [
                    self._metadata_to_dict(
                        getattr(chunk, "metadata", None)
                    )
                    for chunk in new_chunks
                ]

                for metadata in metadatas:
                    metadata.setdefault("source", "unknown")

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                result.ingested += len(new_chunks)

            except Exception as exc:
                message = (
                    f"Failed to store batch starting at "
                    f"{start}: {exc}"
                )
                result.errors.append(message)
                logger.exception(message)

        return result

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks with optional metadata filters."""
        if not query_text or not query_text.strip():
            return []

        total = int(self._collection.count())
        if total == 0:
            return []

        conditions: list[dict[str, Any]] = []

        if topic_filter:
            conditions.append({"topic": str(topic_filter)})

        if difficulty_filter:
            conditions.append(
                {"difficulty": str(difficulty_filter)}
            )

        where = None

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        default_k = getattr(self._settings, "retrieval_k", 5)
        requested_k = max(
            1, min(int(k if k is not None else default_k), total)
        )

        query_embedding = self._embeddings.embed_query(
            query_text
        )

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": requested_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where is not None:
            kwargs["where"] = where

        response = self._collection.query(**kwargs)

        ids = (response.get("ids") or [[]])[0]
        texts = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        threshold = float(
            getattr(self._settings, "similarity_threshold", 0.0)
        )

        for chunk_id, text, metadata, distance in zip(
            ids, texts, metadatas, distances
        ):
            score = max(
                0.0, min(1.0, 1.0 - float(distance))
            )

            if score < threshold:
                continue

            try:
                chunk_metadata = self._metadata_from_dict(
                    metadata or {}
                )

                results.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_id),
                        chunk_text=str(text or ""),
                        metadata=chunk_metadata,
                        score=score,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not create RetrievedChunk {}: {}",
                    chunk_id,
                    exc,
                )

        return sorted(
            results,
            key=lambda item: float(item.score),
            reverse=True,
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """List sources and the number of chunks stored for each."""
        response = self._collection.get(
            include=["metadatas"]
        )
        grouped: dict[str, dict[str, Any]] = {}

        for metadata in response.get("metadatas") or []:
            metadata = metadata or {}
            source = str(metadata.get("source") or "unknown")

            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "topic": metadata.get("topic", "Unknown"),
                    "chunk_count": 0,
                }

            grouped[source]["chunk_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda item: item["source"].casefold(),
        )

    def get_document_chunks(
        self, source: str
    ) -> list[DocumentChunk]:
        """Return all chunks belonging to a source document."""
        response = self._collection.get(
            where={"source": str(source)},
            include=["documents", "metadatas"],
        )

        ids = response.get("ids") or []
        texts = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        chunks: list[DocumentChunk] = []

        for chunk_id, text, metadata in zip(
            ids, texts, metadatas
        ):
            try:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(chunk_id),
                        chunk_text=str(text or ""),
                        metadata=self._metadata_from_dict(
                            metadata or {}
                        ),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not create DocumentChunk {}: {}",
                    chunk_id,
                    exc,
                )

        def chunk_order(chunk: DocumentChunk) -> int:
            metadata = self._metadata_to_dict(
                getattr(chunk, "metadata", None)
            )
            try:
                return int(
                    metadata.get(
                        "chunk_index",
                        metadata.get("chunk_number", 0),
                    )
                )
            except (TypeError, ValueError):
                return 0

        return sorted(chunks, key=chunk_order)

    def get_collection_stats(self) -> dict[str, Any]:
        """Return chunk count, topics, sources, and bonus-topic status."""
        documents = self.list_documents()
        response = self._collection.get(
            include=["metadatas"]
        )
        metadatas = response.get("metadatas") or []

        topics = sorted({
            str(doc["topic"])
            for doc in documents
            if doc.get("topic")
            and doc.get("topic") != "Unknown"
        })

        bonus_present = any(
            str((metadata or {}).get("is_bonus", "false"))
            .strip()
            .lower() in {"true", "1", "yes"}
            for metadata in metadatas
        )

        return {
            "total_chunks": int(self._collection.count()),
            "topics": topics,
            "sources": [
                doc["source"] for doc in documents
            ],
            "bonus_topics_present": bonus_present,
        }

    def delete_document(self, source: str) -> int:
        """Delete all chunks from a source and return the count deleted."""
        response = self._collection.get(
            where={"source": str(source)}
        )
        ids = response.get("ids") or []

        if ids:
            self._collection.delete(ids=ids)

        logger.info(
            "Deleted {} chunks for source '{}'",
            len(ids),
            source,
        )
        return len(ids)

    def delete_source(self, source: str) -> int:
        """Compatibility alias for delete_document."""
        return self.delete_document(source)