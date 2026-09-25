import json
import os

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from inforce import classifier
from inforce.classifier import RouteDecision, classify
from inforce.config import Settings

SETTINGS = Settings(gemini_api_key=SecretStr("test-key"), classifier_model="test-model")
GRAPH = {"depth": "multi-hop", "temporal": False, "budget": "high", "route": "graph"}


def gemini_reply(text: str, status: int = 200, **extra) -> httpx.Response:
    body = {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 540, "candidatesTokenCount": 36},
        "modelVersion": "test-model",
        **extra,
    }
    return httpx.Response(status, json=body)


def client_returning(
    *responses: httpx.Response, requests: list | None = None
) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_valid_reply_is_parsed_and_request_uses_the_schema():
    requests = []
    client = client_returning(gemini_reply(json.dumps(GRAPH)), requests=requests)

    decision, usage = classify(
        "Which circular superseded Circular X?", SETTINGS, client
    )

    assert decision == RouteDecision(**GRAPH)
    assert usage == {
        "provider": "gemini",
        "model": "test-model",
        "input_tokens": 540,
        "output_tokens": 36,
        "thinking_tokens": 0,
        "latency_ms": usage["latency_ms"],
        "attempts": 1,
    }
    [request] = requests
    body = json.loads(request.content)
    assert request.url.path.endswith("/models/test-model:generateContent")
    assert request.headers["x-goog-api-key"] == "test-key"
    assert (
        body["generationConfig"]["responseJsonSchema"]
        == RouteDecision.model_json_schema()
    )
    assert (
        body["contents"][0]["parts"][0]["text"]
        == "Question: Which circular superseded Circular X?"
    )


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({**GRAPH, "route": "keyword"}),
        json.dumps({**GRAPH, "temporal": "false"}),
        json.dumps({k: v for k, v in GRAPH.items() if k != "budget"}),
        json.dumps({**GRAPH, "confidence": 0.9}),
        "The question needs graph retrieval.",
    ],
)
def test_replies_that_break_the_schema_are_rejected(text):
    with pytest.raises(ValidationError):
        classify(
            "Which circular superseded Circular X?",
            SETTINGS,
            client_returning(gemini_reply(text)),
        )


def test_blocked_reply_raises_with_reason():
    blocked = httpx.Response(200, json={"promptFeedback": {"blockReason": "OTHER"}})

    with pytest.raises(ValueError, match="blockReason"):
        classify("question", SETTINGS, client_returning(blocked))


def test_rate_limit_waits_for_the_advertised_retry_delay(monkeypatch):
    waits = []
    monkeypatch.setattr(classifier.time, "sleep", waits.append)
    quota = httpx.Response(
        429,
        json={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "13s",
                    }
                ]
            }
        },
    )

    decision, usage = classify(
        "question", SETTINGS, client_returning(quota, gemini_reply(json.dumps(GRAPH)))
    )

    assert decision.route == "graph"
    assert (waits, usage["attempts"]) == ([13.0], 2)


def test_persistent_errors_are_raised(monkeypatch):
    monkeypatch.setattr(classifier.time, "sleep", lambda _: None)
    unavailable = [httpx.Response(503, json={}) for _ in range(classifier.MAX_ATTEMPTS)]

    with pytest.raises(httpx.HTTPStatusError):
        classify("question", SETTINGS, client_returning(*unavailable))


def test_missing_api_key_is_reported():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        classify("question", Settings(gemini_api_key=None), client_returning())


LIVE_EXAMPLES = [
    ("What is the minimum period for which an NFO must stay open?", "vector"),
    (
        "Which circular replaced the Master Circular for Mutual Funds dated June 27, 2024?",
        "graph",
    ),
    ("Which regulations governed mutual funds on 1 March 2026?", "graph"),
    (
        "Summarize how the skin-in-the-game rules for AMC employees evolved since 2021.",
        "hybrid",
    ),
]


@pytest.mark.skipif(
    os.environ.get("INFORCE_LIVE_LLM") != "1",
    reason="calls the Gemini API; set INFORCE_LIVE_LLM=1 to run",
)
@pytest.mark.parametrize(("query", "route"), LIVE_EXAMPLES)
def test_live_classification(query, route):
    decision, _ = classify(query, Settings())
    assert decision.route == route
