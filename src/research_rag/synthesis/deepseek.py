"""DeepSeek's chat completions, for the turns of one synthesis.

The key travels in the Authorization header and nowhere else, and no message
here repeats it. With tools, DeepSeek needs each assistant message sent back
whole, `reasoning_content` included, so a turn returns the message as it came.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from ..project.policy import ResearchError

PROVIDER = "deepseek"

# How much of DeepSeek's own refusal a message carries.
_DETAIL_CHARACTERS = 300


@dataclass(frozen=True, slots=True)
class Turn:
    message: dict[str, Any]
    usage: dict[str, int]


class DeepSeek:
    """One connection to `{base_url}/chat/completions`, opened for one synthesis."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        reasoning: str,
        key: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning = reasoning
        self._timeout = timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        body = {
            "model": self.model,
            "reasoning_effort": self.reasoning,
            "messages": messages,
            "tools": tools,
        }
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException:
            raise ResearchError(
                f"DeepSeek at {self.base_url} did not answer within "
                f"{self._timeout:g} seconds. Ask again, or raise "
                "synthesis.timeout_seconds."
            ) from None
        except httpx.HTTPError as exc:
            raise ResearchError(
                f"DeepSeek could not be reached at {self.base_url} "
                f"({type(exc).__name__}). Check this machine's network."
            ) from None
        if response.status_code != 200:
            raise _refusal(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ResearchError(
                f"DeepSeek at {self.base_url} answered with no message."
            ) from None
        usage = payload.get("usage") or {}
        return Turn(
            message=dict(message),
            usage={
                name: int(value)
                for name, value in usage.items()
                if isinstance(value, int) and not isinstance(value, bool)
            },
        )


def _refusal(response: httpx.Response) -> ResearchError:
    status = response.status_code
    try:
        detail = str(response.json()["error"]["message"])[:_DETAIL_CHARACTERS]
    except (ValueError, KeyError, TypeError):
        detail = response.text[:_DETAIL_CHARACTERS]
    if status == 401:
        remedy = "DeepSeek refused the key. Set another with: research-rag api-key set"
    elif status == 402:
        remedy = "The DeepSeek account has no balance left. Top it up, then ask again."
    elif status == 429:
        remedy = "DeepSeek is limiting requests from this key. Ask again in a moment."
    elif status >= 500:
        remedy = "DeepSeek failed on its side. Ask again later."
    else:
        remedy = "DeepSeek refused the request."
    return ResearchError(f"{remedy} (HTTP {status}: {detail})")
