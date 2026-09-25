import argparse
import json
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from inforce.config import Settings

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
RETRYABLE_STATUS = {429, 500, 503}
MAX_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 60


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    depth: Literal["single-hop", "multi-hop"]
    temporal: bool
    budget: Literal["low", "medium", "high"]
    route: Literal["vector", "hybrid", "graph"]


INSTRUCTIONS = """You route questions about SEBI mutual fund regulations to a retrieval strategy.
Decide only what evidence the question needs. Do not answer it.

depth
- single-hop: the answer is contained in one passage or provision.
- multi-hop: the answer needs connecting regulatory documents or provisions through relationships
  (supersedes, amends, rescinds, repeals, refers to), or combining/comparing several documents or versions.

temporal
- true: the answer depends on a date or period, on what is currently in force or still valid,
  or on how something changed over time.
- false: otherwise. Naming a dated circular, or asking which document replaced, amended or rescinded
  another, is not temporal unless the question also asks about a date, a period or current validity.

budget (the retrieval effort the evidence requires)
- low: look up one passage.
- medium: gather and synthesise many passages across documents.
- high: traverse relationships between documents or resolve which version was valid.

route
- vector: what a provision says - a definition, requirement, limit or procedure - even when a
  specific circular or regulation is named.
- hybrid: broad overviews, summaries, comparisons, or explanations of how a topic changed, needing
  many passages but not an exact answer about which document replaced or applied to which.
- graph: the answer is a relationship between regulatory documents (which one superseded, amended,
  rescinded, repealed, replaced or cited another), or which document or rule applied on a date,
  applies today, or is still in force.

Usually vector goes with single-hop and low, hybrid with medium, graph with multi-hop and high.

Examples
"What is an AIF?" -> {"depth": "single-hop", "temporal": false, "budget": "low", "route": "vector"}
"What does Circular X say about expense ratios?" -> {"depth": "single-hop", "temporal": false, "budget": "low", "route": "vector"}
"Which circular superseded Circular X?" -> {"depth": "multi-hop", "temporal": false, "budget": "high", "route": "graph"}
"What was applicable in 2023?" -> {"depth": "multi-hop", "temporal": true, "budget": "high", "route": "graph"}
"Summarize how this requirement changed over time." -> {"depth": "multi-hop", "temporal": true, "budget": "medium", "route": "hybrid"}"""


def retry_delay(response: httpx.Response, default: float) -> float:
    if "retry-after" in response.headers:
        return float(response.headers["retry-after"])
    try:
        details = response.json()["error"]["details"]
    except (ValueError, KeyError, TypeError):
        return default
    for detail in details:
        if detail.get("@type", "").endswith("RetryInfo") and "retryDelay" in detail:
            return float(detail["retryDelay"].rstrip("s"))
    return default


def classify(
    query: str, settings: Settings, client: httpx.Client | None = None
) -> tuple[RouteDecision, dict]:
    if settings.gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY is not set")
    body = {
        "systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
        "contents": [{"role": "user", "parts": [{"text": f"Question: {query}"}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": RouteDecision.model_json_schema(),
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    http = client or httpx.Client(timeout=settings.llm_timeout_seconds)
    started = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = http.post(
            GEMINI_URL.format(model=settings.classifier_model),
            headers={"x-goog-api-key": settings.gemini_api_key.get_secret_value()},
            json=body,
        )
        if response.status_code not in RETRYABLE_STATUS or attempt == MAX_ATTEMPTS:
            break
        time.sleep(
            min(retry_delay(response, default=2**attempt), MAX_RETRY_DELAY_SECONDS)
        )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()

    data = response.json()
    candidate = (data.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts") or []
    if not parts or "text" not in parts[0]:
        raise ValueError(
            f"no classification returned (finishReason={candidate.get('finishReason')}, "
            f"promptFeedback={data.get('promptFeedback')})"
        )
    decision = RouteDecision.model_validate_json(parts[0]["text"])
    usage = data.get("usageMetadata", {})
    return decision, {
        "provider": settings.llm_provider,
        "model": data.get("modelVersion", settings.classifier_model),
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "thinking_tokens": usage.get("thoughtsTokenCount", 0),
        "latency_ms": round(latency_ms),
        "attempts": attempt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify the retrieval strategy for a question."
    )
    parser.add_argument("query")
    args = parser.parse_args()
    decision, usage = classify(args.query, Settings())
    print(json.dumps({"decision": decision.model_dump(), "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
