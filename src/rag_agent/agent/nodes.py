"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # Extract original query from the latest HumanMessage
    original_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
        elif hasattr(msg, "type") and msg.type == "human":
            original_query = getattr(msg, "content", "")
            break

    if not original_query and state.messages:
        # Fallback to whatever string representation the last message has
        original_query = str(state.messages[-1].content)

    if not original_query:
        original_query = "deep learning interview question"

    rewrite_prompt = (
        "Rewrite the following user question into a concise keyword-focused query "
        "optimized for vector similarity search over deep learning technical notes.\n\n"
        f"User Question: {original_query}\n\n"
        "Rewritten Query (output only the rewritten query text without explanation):"
    )

    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten = response.content.strip()
        if not rewritten:
            rewritten = original_query
    except Exception:
        rewritten = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten,
    }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.
    """
    query_text = state.rewritten_query or state.original_query or "deep learning"
    
    try:
        store = VectorStoreManager()
        raw_results = store.query(
            query_text=query_text,
            topic_filter=state.topic_filter,
            difficulty_filter=state.difficulty_filter,
            k=4,
        )
    except Exception:
        raw_results = []

    if not raw_results:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    chunks = []
    for item in raw_results:
        if isinstance(item, dict):
            chunk_id = item.get("chunk_id", "")
            chunk_text = item.get("chunk_text", "")
            score = float(item.get("score", 0.0))
            metadata = item.get("metadata", {})
        else:
            chunk_id = getattr(item, "chunk_id", "")
            chunk_text = getattr(item, "chunk_text", "")
            score = float(getattr(item, "score", 0.0))
            metadata = getattr(item, "metadata", {})

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                score=score,
                metadata=metadata,
            )
        )

    return {
        "retrieved_chunks": chunks,
        "no_context_found": len(chunks) == 0,
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.no_context_found or not state.retrieved_chunks:
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.rewritten_query,
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    context_lines = []
    sources_set = set()
    total_score = 0.0

    for chunk in state.retrieved_chunks:
        meta = chunk.metadata
        source_name = getattr(meta, "source", "unknown")
        topic = getattr(meta, "topic", "General")
        sources_set.add(source_name)
        total_score += chunk.score

        context_lines.append(
            f"\n{chunk.chunk_text}\n"
        )

    context_str = "\n".join(context_lines)
    avg_confidence = total_score / len(state.retrieved_chunks) if state.retrieved_chunks else 0.0

    # Prepare messages
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    context_msg = SystemMessage(content=f"Here is the retrieved documentation context:\n\n{context_str}")
    
    trimmed_history = trim_messages(
        state.messages[:-1] if state.messages else [],
        max_tokens=getattr(settings, "max_context_tokens", 2000),
        strategy="last",
        token_counter=len,
        include_system=False,
    )

    query_msg = HumanMessage(content=state.original_query or "Please answer based on the context.")

    prompt_messages = [system_msg, context_msg] + list(trimmed_history) + [query_msg]

    try:
        ai_response = llm.invoke(prompt_messages)
        answer_text = ai_response.content
    except Exception as e:
        answer_text = f"Error generating answer via LLM: {e}"

    response = AgentResponse(
        answer=answer_text,
        sources=list(sources_set),
        confidence=float(avg_confidence),
        no_context_found=False,
        rewritten_query=state.rewritten_query,
    )

    return {
        "final_response": response,
        "messages": [AIMessage(content=answer_text)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.
    """
    if state.no_context_found:
        return "end"
    return "generate"