"""
chunker.py
==========
Document loading and chunking pipeline.

Loads PDF and Markdown files into structured DocumentChunk objects
ready for embedding and vector-store storage.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """Load raw documents and split them into DocumentChunk objects."""

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """Load, split, and prepare a file's chunks with metadata and IDs."""
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if file_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{file_path.suffix}'. "
                f"Supported types: PDF, Markdown (.pdf, .md, .markdown)."
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        if file_path.suffix.lower() == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        else:
            raw_chunks = self._chunk_markdown(
                file_path, chunk_size, chunk_overlap
            )

        base_metadata = self._infer_metadata(file_path, metadata_overrides)
        chunks = []

        for index, raw_chunk in enumerate(raw_chunks):
            metadata_dict = base_metadata.to_dict() if hasattr(base_metadata, "to_dict") else dict(base_metadata)
            metadata_dict["chunk_index"] = index

            if raw_chunk.get("page") is not None:
                metadata_dict["page"] = raw_chunk["page"]
            if raw_chunk.get("header"):
                metadata_dict["header"] = raw_chunk["header"]

            metadata = ChunkMetadata.from_dict(metadata_dict) if hasattr(ChunkMetadata, "from_dict") else ChunkMetadata(**metadata_dict)
            text = raw_chunk["text"].strip()
            if not text:
                continue

            chunk_id = VectorStoreManager.generate_chunk_id(
                str(file_path.name), text
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info("Chunked '{}' into {} chunks", file_path.name, len(chunks))
        return chunks

    def _chunk_markdown(
        self, file_path: Path, chunk_size: int, chunk_overlap: int
    ) -> list[dict[str, Any]]:
        """Split a Markdown file using header splitting and recursive chunking."""
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        header_docs = header_splitter.split_text(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        final_chunks = []
        for doc in header_docs:
            splits = text_splitter.split_text(doc.page_content)
            header_path = " > ".join(
                [
                    val
                    for key, val in doc.metadata.items()
                    if key.startswith("header")
                ]
            )
            for split in splits:
                final_chunks.append(
                    {
                        "text": split,
                        "header": header_path or None,
                        "page": None,
                    }
                )

        if not final_chunks and text.strip():
            splits = text_splitter.split_text(text)
            for split in splits:
                final_chunks.append(
                    {"text": split, "header": None, "page": None}
                )

        return final_chunks

    def _chunk_pdf(
        self, file_path: Path, chunk_size: int, chunk_overlap: int
    ) -> list[dict[str, Any]]:
        """Load and split a PDF file page by page."""
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        final_chunks = []
        for page_idx, page in enumerate(pages):
            splits = text_splitter.split_text(page.page_content)
            for split in splits:
                final_chunks.append(
                    {
                        "text": split,
                        "header": None,
                        "page": page_idx + 1,
                    }
                )

        return final_chunks

    def _infer_metadata(
        self, file_path: Path, overrides: dict | None = None
    ) -> ChunkMetadata:
        """Infer topic, difficulty, and source from the file name or overrides."""
        overrides = overrides or {}
        filename = file_path.stem.lower()

        topic = overrides.get("topic")
        if not topic:
            if "lstm" in filename:
                topic = "LSTM"
            elif "cnn" in filename:
                topic = "CNN"
            elif "ann" in filename:
                topic = "ANN"
            elif "transformer" in filename:
                topic = "Transformer"
            else:
                topic = filename.replace("_", " ").title()

        difficulty = overrides.get("difficulty")
        if not difficulty:
            if "beginner" in filename:
                difficulty = "beginner"
            elif "intermediate" in filename:
                difficulty = "intermediate"
            elif "advanced" in filename:
                difficulty = "advanced"
            else:
                difficulty = "intermediate"

        is_bonus = overrides.get("is_bonus", "bonus" in filename)
        doc_type = overrides.get("type", "note")

        try:
            return ChunkMetadata(
                source=str(file_path.name),
                topic=str(topic),
                difficulty=str(difficulty),
                is_bonus=bool(is_bonus),
                type=str(doc_type),
            )
        except TypeError:
            return ChunkMetadata(
                source=str(file_path.name),
                topic=str(topic),
                difficulty=str(difficulty),
                is_bonus=bool(is_bonus),
            )