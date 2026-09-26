import argparse
import json
import operator
import time
from datetime import date
from typing import Annotated, Literal, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from neo4j import Session
from qdrant_client import QdrantClient

from inforce import graph, graphrag, vector
from inforce.classifier import classify
from inforce.config import Settings

Route = Literal["vector", "hybrid", "graph"]

TOP_K = {"low": 5, "medium": 10, "high": 10}
FALLBACK_ROUTE: Route = "hybrid"
ESCALATES_TO = {"vector": "hybrid", "hybrid": "graph"}
REQUIRED_PASSAGES = {"single-hop": 1, "multi-hop": 2}
DETERMINATE_STATES = {
    "in_force",
    "ended",
    "not_yet_in_force",
    "not_yet_issued",
    "changed_during_period",
}


class RouterState(TypedDict, total=False):
    query: str
    today: date | None
    force_route: Route | None
    max_escalations: int
    decision: dict | None
    classification: dict | None
    route_source: str
    route: Route
    top_k: int
    escalations: int
    evidence: list | dict
    retrieval_ms: int
    attempts: Annotated[list[dict], operator.add]


def assess_passages(hits: list[dict], depth: str | None, threshold: float) -> dict:
    relevant = [hit for hit in hits if hit["relevance"] >= threshold]
    required = REQUIRED_PASSAGES.get(depth, 1)
    reasons = []
    if not relevant:
        reasons.append(f"no passage reached relevance {threshold}")
    elif len(relevant) < required:
        reasons.append(
            f"{depth} question supported by {len(relevant)} relevant passage(s); {required} required"
        )
    return {
        "level": "weak" if reasons else "strong",
        "reasons": reasons,
        "relevant_passages": len(relevant),
        "relevant_documents": len({hit["doc_id"] for hit in relevant}),
        "top_relevance": max((hit["relevance"] for hit in hits), default=None),
    }


def assess_graph(evidence: dict) -> dict:
    reasons = []
    if not evidence["anchors"]:
        reasons.append("no regulatory document matched the question")
    if not (evidence["relationships"] or evidence["paths"] or evidence["provisions"]):
        reasons.append("no relationships, paths or provisions retrieved")
    if evidence["as_of"] and not any(
        t["state"] in DETERMINATE_STATES for t in evidence["temporal"]
    ):
        reasons.append("validity at the requested date could not be established")
    if evidence["conflicts"]:
        reasons.append("replacing and replaced documents both appear in force")
    return {
        "level": "weak" if reasons else "strong",
        "reasons": reasons,
        "anchors": len(evidence["anchors"]),
        "relationships": len(evidence["relationships"]),
        "provisions": len(evidence["provisions"]),
    }


def build_router(settings: Settings, qdrant: QdrantClient, graph_session: Session):
    def classify_query(state: RouterState) -> dict:
        if state.get("force_route"):
            return {
                "route": state["force_route"],
                "route_source": "forced",
                "decision": None,
                "classification": None,
                "top_k": TOP_K["medium"],
                "escalations": 0,
            }
        try:
            decision, usage = classify(state["query"], settings)
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "route": FALLBACK_ROUTE,
                "route_source": "fallback",
                "decision": None,
                "classification": {"error": f"{type(exc).__name__}: {exc}"},
                "top_k": TOP_K["medium"],
                "escalations": 0,
            }
        return {
            "route": decision.route,
            "route_source": "classifier",
            "decision": decision.model_dump(),
            "classification": usage,
            "top_k": TOP_K[decision.budget],
            "escalations": 0,
        }

    def retrieve(state: RouterState) -> dict:
        started = time.perf_counter()
        route, query = state["route"], state["query"]
        if route == "vector":
            evidence = vector.search(qdrant, settings, query, state["top_k"])
        elif route == "hybrid":
            evidence = vector.hybrid_search(qdrant, settings, query, state["top_k"])
        else:
            evidence = graphrag.retrieve(graph_session, query, state.get("today"))
        return {
            "evidence": evidence,
            "retrieval_ms": round((time.perf_counter() - started) * 1000),
        }

    def assess(state: RouterState) -> dict:
        started = time.perf_counter()
        evidence, route = state["evidence"], state["route"]
        if route == "graph":
            confidence = assess_graph(evidence)
        else:
            if route == "vector":
                model = vector.reranker(
                    settings.reranker_model, settings.embedding_cache_dir
                )
                scores = model.rerank(
                    state["query"], [vector.embedding_text(hit) for hit in evidence]
                )
                evidence = [
                    {**hit, "relevance": float(score)}
                    for hit, score in zip(evidence, scores, strict=True)
                ]
            else:
                evidence = [{**hit, "relevance": hit["score"]} for hit in evidence]
            depth = (state.get("decision") or {}).get("depth")
            confidence = assess_passages(evidence, depth, settings.relevance_threshold)
        attempt = {
            "route": route,
            "escalated": state["escalations"] > 0,
            "top_k": None if route == "graph" else state["top_k"],
            "evidence": evidence,
            "confidence": confidence,
            "latency_ms": {
                "retrieval": state["retrieval_ms"],
                "assessment": round((time.perf_counter() - started) * 1000),
            },
        }
        return {"attempts": [attempt]}

    def next_step(state: RouterState) -> str:
        weak = state["attempts"][-1]["confidence"]["level"] == "weak"
        can_escalate = (
            state["escalations"] < state["max_escalations"]
            and state["route"] in ESCALATES_TO
        )
        return "escalate" if weak and can_escalate else END

    def escalate(state: RouterState) -> dict:
        return {
            "route": ESCALATES_TO[state["route"]],
            "escalations": state["escalations"] + 1,
        }

    workflow = StateGraph(RouterState)
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("assess", assess)
    workflow.add_node("escalate", escalate)
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "assess")
    workflow.add_conditional_edges("assess", next_step, ["escalate", END])
    workflow.add_edge("escalate", "retrieve")
    return workflow.compile()


def route_query(
    query: str,
    settings: Settings,
    qdrant: QdrantClient,
    graph_session: Session,
    force_route: Route | None = None,
    today: date | None = None,
    max_escalations: int | None = None,
) -> dict:
    started = time.perf_counter()
    state = build_router(settings, qdrant, graph_session).invoke(
        {
            "query": query,
            "today": today,
            "force_route": force_route,
            "max_escalations": settings.max_escalations
            if max_escalations is None
            else max_escalations,
            "attempts": [],
        }
    )
    attempts = state["attempts"]
    return {
        "query": query,
        "route": attempts[0]["route"],
        "final_route": attempts[-1]["route"],
        "route_source": state["route_source"],
        "decision": state["decision"],
        "classification": state["classification"],
        "escalated": len(attempts) > 1,
        "confidence": attempts[-1]["confidence"],
        "attempts": attempts,
        "latency_ms": {
            "classification": (state["classification"] or {}).get("latency_ms", 0),
            "retrieval": sum(a["latency_ms"]["retrieval"] for a in attempts),
            "assessment": sum(a["latency_ms"]["assessment"] for a in attempts),
            "total": round((time.perf_counter() - started) * 1000),
        },
    }


def node_name(nodes: dict, node_id: str) -> str:
    node = nodes.get(node_id) or {}
    return node.get("number") or node.get("title") or node_id


def summarize(result: dict) -> str:
    lines = [
        f"route: {result['route']} ({result['route_source']}) decision={result['decision']}",
        (
            f"escalated: {result['escalated']} -> final route {result['final_route']}; "
            f"confidence {result['confidence']['level']} {result['confidence']['reasons']}"
        ),
        f"latency_ms: {result['latency_ms']}",
    ]
    if result["classification"] and "error" in result["classification"]:
        lines.append(f"classification error: {result['classification']['error']}")
    for attempt in result["attempts"]:
        evidence = attempt["evidence"]
        lines.append(f"[{attempt['route']}] confidence={attempt['confidence']}")
        if attempt["route"] == "graph":
            nodes = evidence["nodes"]
            lines.append(
                f"  anchors: {[node_name(nodes, a['id']) for a in evidence['anchors']]}"
            )
            for rel in evidence["relationships"][:8]:
                lines.append(
                    f"  {node_name(nodes, rel['start'])} -{rel['type']}-> {node_name(nodes, rel['end'])} "
                    f"[{rel['source_doc']} p.{rel['page_start']}]"
                )
            for state in evidence["temporal"][:6]:
                lines.append(
                    f"  temporal {state['state']}: {node_name(nodes, state['id'])}"
                )
        else:
            for rank, hit in enumerate(evidence, start=1):
                lines.append(
                    f"  {rank}. relevance={hit['relevance']:.2f} {hit['doc_id']} "
                    f"p.{hit['page_start']}-{hit['page_end']} [{(hit['section'] or '')[:55]}]"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route a question to vector, hybrid or graph retrieval."
    )
    parser.add_argument("query")
    parser.add_argument("--route", choices=["vector", "hybrid", "graph"])
    parser.add_argument("--no-escalation", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    qdrant = vector.connect(settings)
    with (
        graph.connect(settings) as driver,
        driver.session(database=settings.neo4j_database) as session,
    ):
        result = route_query(
            args.query,
            settings,
            qdrant,
            session,
            force_route=args.route,
            max_escalations=0 if args.no_escalation else None,
        )
    qdrant.close()
    print(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
        if args.json
        else summarize(result)
    )


if __name__ == "__main__":
    main()
