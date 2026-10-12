"""
graph.py
========
LangGraph agent graph definition and compilation.

Assembles the RAG nodes into a directed state graph and compiles it
with an in-memory checkpointer for conversation persistence.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    generation_node,
    query_rewrite_node,
    retrieval_node,
    should_retry_retrieval,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    """Build and compile the LangGraph RAG workflow."""

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        """Assemble the nodes and edges and compile the graph."""
        graph = StateGraph(AgentState)

        graph.add_node("query_rewrite", query_rewrite_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("generation", generation_node)

        graph.add_edge(START, "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")

        graph.add_conditional_edges(
            "retrieval",
            should_retry_retrieval,
            {
                "generate": "generation",
                "end": END,
            },
        )

        graph.add_edge("generation", END)

        return graph.compile(checkpointer=self._checkpointer)


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Return the singleton compiled graph."""
    return AgentGraphBuilder().build()