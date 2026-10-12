"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Run from the project root:
    streamlit run src/rag_agent/ui/app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return one cached vector-store manager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return one cached document chunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


def initialise_session_state() -> None:
    """Initialize state values used across Streamlit reruns."""
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert common Pydantic/dataclass/dict metadata objects to a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            result = value.to_dict()
            return result if isinstance(result, dict) else {}
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:
            pass
    return {}


def _metadata_for_chunk(chunk: Any) -> dict[str, Any]:
    """Read chunk metadata whether it is a model or a plain dictionary."""
    metadata = getattr(chunk, "metadata", None)
    return _as_dict(metadata)


def _refresh_documents(store: VectorStoreManager) -> None:
    """Refresh the document list and keep the selected source valid."""
    docs = store.list_documents() or []
    normalized: list[dict[str, Any]] = []
    for doc in docs:
        if isinstance(doc, dict):
            item = dict(doc)
        else:
            item = _as_dict(doc)

        if item.get("source"):
            normalized.append(item)

    st.session_state["ingested_documents"] = normalized
    available = {doc["source"] for doc in normalized}
    selected = st.session_state.get("selected_document")

    if selected not in available:
        st.session_state["selected_document"] = (
            normalized[0]["source"] if normalized else None
        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Deep Learning RAG Interview Prep",
        page_icon="🤖",
        layout="wide",
    )
    
    initialise_session_state()
    
    try:
        store = get_vector_store()
        chunker = get_chunker()
        graph = get_graph()
    except Exception as e:
        st.error(f"Failed to initialize application dependencies: {e}")
        return

    _refresh_documents(store)

    # --- Sidebar UI ---
    with st.sidebar:
        st.title("📚 Document Management")
        
        uploaded_file = st.file_uploader("Upload a Markdown Note", type=["md", "markdown"])
        if uploaded_file is not None:
            if st.button("Ingest Document"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)
                
                try:
                    chunks = chunker.chunk_file(tmp_path)
                    store.ingest(chunks)
                    st.success(f"Successfully ingested {uploaded_file.name}!")
                    _refresh_documents(store)
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

        st.divider()
        st.subheader("Ingested Documents")
        docs = st.session_state.get("ingested_documents", [])
        if docs:
            for doc in docs:
                st.text(f"• {doc['source']}")
        else:
            st.info("No documents ingested yet.")

    # --- Main Chat Interface ---
    st.title("🧠 Deep Learning RAG Agent")
    st.write("Ask questions based on your study notes and documentation.")

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = graph.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
                    )
                    messages = response.get("messages", [])
                    answer = messages[-1].content if messages else "No response generated."
                    
                    st.markdown(answer)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    import traceback
                    error_msg = f"Error generating response: {e}\n```python\n{traceback.format_exc()}\n```"
                    st.error(error_msg)
                    st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()