from langgraph.graph import StateGraph, END

from eval.models import EvalState
from eval.config import EvalConfig
from eval.pipeline import route, build_packet, deduce
from eval.retrieve import retrieve
from eval.adapters.base import BaseModelAdapter
from ingest.qdrant_indexer import QdrantIndexer


def build_graph(config: EvalConfig, adapter: BaseModelAdapter, indexer: QdrantIndexer):
    graph = StateGraph(EvalState)

    def router_node(state: EvalState) -> dict:
        plan, gen = route(state.question, adapter)
        cost = adapter.get_cost(gen.input_tokens, gen.output_tokens)
        return {
            "router_plan": plan,
            "total_tokens": state.total_tokens + gen.input_tokens + gen.output_tokens,
            "total_cost": state.total_cost + cost,
        }

    def retrieve_node(state: EvalState) -> dict:
        assert state.router_plan is not None
        chunks = retrieve(state.router_plan.expanded_queries, config.top_k, indexer)
        return {"retrieved_chunks": chunks}

    def packet_node(state: EvalState) -> dict:
        packet = build_packet(state.retrieved_chunks)
        return {"evidence_packet": packet}

    def deduce_node(state: EvalState) -> dict:
        assert state.evidence_packet is not None
        answer, gen = deduce(state.question, state.evidence_packet, adapter)
        cost = adapter.get_cost(gen.input_tokens, gen.output_tokens)
        return {
            "final_answer": answer,
            "total_tokens": state.total_tokens + gen.input_tokens + gen.output_tokens,
            "total_cost": state.total_cost + cost,
        }

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("packet", packet_node)
    graph.add_node("deduce", deduce_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "packet")
    graph.add_edge("packet", "deduce")
    graph.add_edge("deduce", END)

    return graph.compile()
