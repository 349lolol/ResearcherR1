from langgraph.graph import StateGraph, END

from eval.models import EvalState
from eval.config import EvalConfig
from eval.pipeline import route, build_packet, deduce
from eval.retrieve import retrieve
from eval.verify import verify
from eval.repair import repair
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
        total_tokens = state.total_tokens + gen.input_tokens + gen.output_tokens
        total_cost = state.total_cost + cost

        # Naive: skip verification
        if config.ablation == "Naive":
            return {"final_answer": answer, "total_tokens": total_tokens, "total_cost": total_cost}

        # EFR / EFR+Verify: verify
        verification, verify_gen = verify(answer, state.retrieved_chunks, adapter)
        verify_cost = adapter.get_cost(verify_gen.input_tokens, verify_gen.output_tokens)
        total_tokens += verify_gen.input_tokens + verify_gen.output_tokens
        total_cost += verify_cost

        # All claims supported: done
        if verification.support_precision >= 1.0:
            return {"final_answer": answer, "verification": verification, "total_tokens": total_tokens, "total_cost": total_cost}

        # EFR: fail without repair
        if config.ablation == "EFR":
            return {"final_answer": answer, "verification": verification, "errors": state.errors + ["Verification failed"], "total_tokens": total_tokens, "total_cost": total_cost}

        # EFR+Verify: repair
        repaired, repair_gen = repair(answer, verification, state.evidence_packet, adapter)
        repair_cost = adapter.get_cost(repair_gen.input_tokens, repair_gen.output_tokens)
        total_tokens += repair_gen.input_tokens + repair_gen.output_tokens
        total_cost += repair_cost

        return {"final_answer": repaired, "verification": verification, "repair_count": 1, "total_tokens": total_tokens, "total_cost": total_cost}

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
