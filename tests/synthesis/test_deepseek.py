"""DeepSeek's chat completions: what a turn sends, and what a refusal says."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from research_rag.project.policy import ResearchError
from research_rag.synthesis.deepseek import DeepSeek

KEY = "sk-never-printed"
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _client(handler: Any) -> DeepSeek:
    return DeepSeek(
        base_url="https://api.deepseek.test",
        model="deepseek-flash",
        reasoning="max",
        key=KEY,
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
    )


async def _complete(handler: Any) -> Any:
    async with _client(handler) as client:
        return await client.complete([{"role": "user", "content": "Why?"}], TOOLS)


@pytest.mark.anyio
async def test_a_turn_sends_the_model_messages_and_tools_and_returns_the_message_whole() -> (
    None
):
    seen: list[httpx.Request] = []
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should search.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "why"}'},
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            },
        )

    turn = await _complete(handler)

    (request,) = seen
    assert request.url == "https://api.deepseek.test/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "deepseek-flash"
    assert body["reasoning_effort"] == "max"
    assert body["messages"] == [{"role": "user", "content": "Why?"}]
    assert body["tools"] == TOOLS
    assert turn.message == message
    assert turn.usage == {"prompt_tokens": 120, "completion_tokens": 30}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "body", "named"),
    [
        (
            401,
            {"error": {"message": "Authentication Fails"}},
            "research-rag api-key set",
        ),
        (402, {"error": {"message": "Insufficient Balance"}}, "balance"),
        (429, {"error": {"message": "Rate limit reached"}}, "Ask again"),
        (400, {"error": {"message": "messages: reasoning_content missing"}}, "missing"),
        (503, {"error": {"message": "Server overloaded"}}, "503"),
    ],
    ids=[
        "key refused",
        "no balance",
        "rate limited",
        "request refused",
        "server failed",
    ],
)
async def test_a_refusal_names_its_remedy_and_never_the_key(
    status: int, body: dict[str, Any], named: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    with pytest.raises(ResearchError, match=named) as refused:
        await _complete(handler)
    assert KEY not in str(refused.value)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "named"),
    [
        (httpx.ReadTimeout("slow"), "did not answer within 30 seconds"),
        (httpx.ConnectError("refused"), "could not be reached"),
    ],
    ids=["timeout", "unreachable"],
)
async def test_an_endpoint_that_does_not_answer_is_named(
    failure: Exception, named: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    with pytest.raises(ResearchError, match=named) as refused:
        await _complete(handler)
    assert "https://api.deepseek.test" in str(refused.value)
    assert KEY not in str(refused.value)


@pytest.mark.anyio
async def test_a_reply_without_a_message_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ResearchError, match="no message"):
        await _complete(handler)
