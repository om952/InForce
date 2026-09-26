import os
import time

import httpx
import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import ValidationError

from inforce import graph, router, vector
from inforce.classifier import RouteDecision
from inforce.config import Settings
from inforce.router import assess_graph, assess_passages, route_query

SETTINGS = Settings(relevance_threshold=2.0, max_escalations=1)


def decision(route, budget="medium", depth=None):
    depth = depth or ("single-hop" if route == "vector" else "multi-hop")
    return RouteDecision(depth=depth, temporal=False, budget=budget, route=route)


def passages(*scores, doc="doc-a"):
    return [
        {
            "chunk_id": f"{doc}-{i}",
            "doc_id": doc,
            "text": "t",
            "section": None,
            "score": score,
        }
        for i, score in enumerate(scores)
    ]


def graph_evidence(anchors=1, relationships=1, as_of=None, temporal=(), conflicts=()):
    return {
        "anchors": [{"id": f"a{i}"} for i in range(anchors)],
        "relationships": [{"type": "SUPERSEDES"}] * relationships,
        "paths": [],
        "provisions": [],
        "as_of": as_of,
        "temporal": list(temporal),
        "conflicts": list(conflicts),
    }


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, texts):
        return self.scores[: len(texts)]


@pytest.fixture
def stack(monkeypatch):
    calls, outputs = [], {}

    def fake(route):
        def run(*args):
            calls.append(route)
            return outputs[route]

        return run

    monkeypatch.setattr(router.vector, "search", fake("vector"))
    monkeypatch.setattr(router.vector, "hybrid_search", fake("hybrid"))
    monkeypatch.setattr(router.graphrag, "retrieve", fake("graph"))
    monkeypatch.setattr(router.vector, "embedding_text", lambda hit: hit["text"])

    def configure(classified, vector_relevance=(), **route_outputs):
        outputs.update(route_outputs)
        monkeypatch.setattr(
            router.vector, "reranker", lambda *a: FakeReranker(list(vector_relevance))
        )

        def fake_classify(query, settings):
            if isinstance(classified, Exception):
                raise classified
            return classified, {
                "input_tokens": 560,
                "output_tokens": 36,
                "latency_ms": 800,
            }

        monkeypatch.setattr(router, "classify", fake_classify)
        return calls

    return configure


def test_strong_vector_evidence_is_not_escalated(stack):
    calls = stack(
        decision("vector", "low"),
        vector_relevance=[6.0, 1.0],
        vector=passages(0.8, 0.7),
    )

    result = route_query("q", SETTINGS, None, None)

    assert calls == ["vector"]
    assert (
        result["escalated"],
        result["final_route"],
        result["confidence"]["level"],
    ) == (
        False,
        "vector",
        "strong",
    )
    assert [hit["relevance"] for hit in result["attempts"][0]["evidence"]] == [6.0, 1.0]


def test_weak_vector_evidence_escalates_once_to_hybrid_then_stops(stack):
    calls = stack(
        decision("vector", "low"),
        vector_relevance=[0.1, -3.0],
        vector=passages(0.7, 0.6),
        hybrid=passages(0.5, -1.0),
    )

    result = route_query("q", SETTINGS, None, None)

    assert calls == ["vector", "hybrid"]
    assert [a["route"] for a in result["attempts"]] == ["vector", "hybrid"]
    assert [a["escalated"] for a in result["attempts"]] == [False, True]
    assert result["confidence"] == result["attempts"][-1]["confidence"]
    assert result["confidence"]["level"] == "weak"


def test_multi_hop_question_with_one_relevant_passage_escalates_to_graph(stack):
    calls = stack(
        decision("hybrid"), hybrid=passages(2.3, -4.8, -5.0), graph=graph_evidence()
    )

    result = route_query("q", SETTINGS, None, None)

    assert calls == ["hybrid", "graph"]
    assert (
        "1 relevant passage(s); 2 required"
        in result["attempts"][0]["confidence"]["reasons"][0]
    )
    assert (result["final_route"], result["confidence"]["level"]) == ("graph", "strong")


def test_graph_is_the_top_of_the_ladder(stack):
    calls = stack(
        decision("graph", "high"), graph=graph_evidence(anchors=0, relationships=0)
    )

    result = route_query("q", SETTINGS, None, None)

    assert calls == ["graph"]
    assert (result["escalated"], result["confidence"]["level"]) == (False, "weak")


def test_escalation_can_be_disabled_for_baselines(stack):
    calls = stack(
        AssertionError("classifier must not run"),
        vector_relevance=[-5.0],
        vector=passages(0.6),
    )

    result = route_query(
        "q", SETTINGS, None, None, force_route="vector", max_escalations=0
    )

    assert calls == ["vector"]
    assert (
        result["route_source"],
        result["escalated"],
        result["confidence"]["level"],
    ) == (
        "forced",
        False,
        "weak",
    )


@pytest.mark.parametrize(
    "failure",
    [
        httpx.HTTPStatusError(
            "429",
            request=httpx.Request("POST", "https://x"),
            response=httpx.Response(429),
        ),
        httpx.ReadTimeout("timed out"),
        ValueError("GEMINI_API_KEY is not set"),
    ],
)
def test_classifier_failure_falls_back_to_hybrid(stack, failure):
    calls = stack(failure, hybrid=passages(5.0))

    result = route_query("q", SETTINGS, None, None)

    assert calls == ["hybrid"]
    assert (result["route"], result["route_source"], result["decision"]) == (
        "hybrid",
        "fallback",
        None,
    )
    assert type(failure).__name__ in result["classification"]["error"]


def test_budget_sets_evidence_size(stack, monkeypatch):
    sizes = []

    def dense(client, settings, query, top_k):
        sizes.append(top_k)
        return passages(1.0)

    stack(decision("vector", "low"), vector_relevance=[5.0])
    monkeypatch.setattr(router.vector, "search", dense)

    route_query("q", SETTINGS, None, None)

    assert sizes == [5]


@pytest.mark.parametrize(
    ("scores", "depth", "level"),
    [
        ([5.0, -1.0], "single-hop", "strong"),
        ([1.9, 0.5], "single-hop", "weak"),
        ([5.0, -1.0], "multi-hop", "weak"),
        ([5.0, 2.0], "multi-hop", "strong"),
        ([5.0], None, "strong"),
    ],
)
def test_assess_passages(scores, depth, level):
    hits = [{**hit, "relevance": hit["score"]} for hit in passages(*scores)]
    assert assess_passages(hits, depth, threshold=2.0)["level"] == level


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (graph_evidence(anchors=0, relationships=0), "no regulatory document matched"),
        (
            graph_evidence(
                as_of={"start": "2020-01-01"}, temporal=[{"state": "unknown_start"}]
            ),
            "could not be established",
        ),
        (
            graph_evidence(conflicts=[{"reason": "both in force"}]),
            "both appear in force",
        ),
    ],
)
def test_assess_graph_explains_weak_evidence(evidence, reason):
    confidence = assess_graph(evidence)
    assert confidence["level"] == "weak"
    assert any(reason in r for r in confidence["reasons"])


def test_more_than_one_escalation_is_rejected_by_configuration():
    with pytest.raises(ValidationError):
        Settings(max_escalations=2)


@pytest.fixture(scope="module")
def live_stack():
    if os.environ.get("INFORCE_LIVE_LLM") != "1":
        pytest.skip("calls the Gemini API; set INFORCE_LIVE_LLM=1 to run")
    settings = Settings()
    qdrant = vector.connect(settings)
    driver = graph.connect(settings)
    try:
        driver.verify_connectivity()
    except (ServiceUnavailable, Neo4jError):
        pytest.skip("Neo4j is not reachable; run `docker compose up -d`")
    with driver.session(database=settings.neo4j_database) as session:
        yield settings, qdrant, session
    driver.close()
    qdrant.close()


@pytest.mark.parametrize(
    ("query", "routes", "level"),
    [
        ("How many working days can an NFO remain open?", ["vector"], "strong"),
        (
            "Which circular replaced the Master Circular for Mutual Funds dated June 27, 2024?",
            ["graph"],
            "strong",
        ),
        (
            "How have skin-in-the-game requirements for AMC employees evolved since 2021?",
            ["hybrid", "graph"],
            "strong",
        ),
        ("What is the repo rate set by the RBI?", ["vector", "hybrid"], "weak"),
    ],
)
def test_live_routing_and_escalation(live_stack, query, routes, level):
    settings, qdrant, session = live_stack
    time.sleep(4.2)

    result = route_query(query, settings, qdrant, session)

    assert [a["route"] for a in result["attempts"]] == routes
    assert result["confidence"]["level"] == level
