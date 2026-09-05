import json

import httpx
import pytest

from nist_rag.config import AppError, Settings
from nist_rag.llm import GroqLLM
from nist_rag.models import QueryPlan


def completion(content='{"query":"NIST GOVERN function"}', finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish,
                }
            ]
        },
    )


def run(responses, **overrides):
    calls, waits = [], []

    def handler(request):
        calls.append(json.loads(request.content))
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    llm = GroqLLM(
        Settings(api_key="test-key", **overrides),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=waits.append,
    )
    return llm, calls, waits


def test_rate_limit_honors_retry_after_then_recovers():
    llm, calls, waits = run([httpx.Response(429, headers={"Retry-After": "3"}), completion()])
    try:
        assert (
            llm.complete(QueryPlan, "Rewrite", {"question": "Govern?"}).query
            == "NIST GOVERN function"
        )
        assert waits == [3.0]
        assert len(calls) == 2
        assert calls[0]["response_format"] == {"type": "json_object"}
    finally:
        llm.close()


def test_long_rate_limit_wait_is_reported_without_early_retry():
    llm, calls, waits = run([httpx.Response(429, headers={"Retry-After": "120"})])
    with pytest.raises(AppError, match="120 seconds"):
        llm.complete(QueryPlan, "Rewrite", {})
    assert waits == []
    assert len(calls) == 1
    llm.close()


@pytest.mark.parametrize(
    "bad", [completion("bad-json"), completion('{"query":42}'), completion(finish="length")]
)
def test_bad_or_truncated_json_can_recover(bad):
    llm, calls, _ = run([bad, completion()])
    assert llm.complete(QueryPlan, "Rewrite", {}).query
    assert len(calls) == 2
    llm.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_api_errors_are_not_retried(status):
    llm, calls, waits = run([httpx.Response(status)])
    with pytest.raises(AppError):
        llm.complete(QueryPlan, "Rewrite", {})
    assert len(calls) == 1
    assert waits == []
    llm.close()


def test_network_retries_are_bounded():
    llm, calls, waits = run([httpx.ConnectError("offline") for _ in range(3)])
    with pytest.raises(AppError, match="connection"):
        llm.complete(QueryPlan, "Rewrite", {})
    assert len(calls) == 3
    assert len(waits) == 2
    llm.close()


def test_missing_key_fails_before_http():
    with pytest.raises(AppError, match="GROQ_API_KEY"):
        GroqLLM(Settings())
