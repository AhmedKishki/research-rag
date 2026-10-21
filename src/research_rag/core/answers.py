"""`AnswerWorkflow`: one white-box answer, and the account's key behind it.

An answer is refused before anything leaves the machine when this app is
offline, holds no key, has no generation to search, or is already answering.
Otherwise DeepSeek searches the selected generation through `search`, under the
reader's filters, and the result names what was sent. `answers/loop.py` holds
the loop; this module holds what the service lends it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..answers.deepseek import PROVIDER, DeepSeek
from ..answers.loop import Limits, run_answer
from ..project.credentials import clear_deepseek_key, deepseek_key, set_deepseek_key
from ..project.policy import ResearchError
from .admission import as_caller

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..project.config import ResearchConfig

#: The caller every search an answer makes is counted under.
ANSWER_CALLER = "answer"


class AnswerWorkflow:
    config: ResearchConfig
    _answer_activity: dict[str, Any] | None

    async def answer(self, question: str, **filters: Any) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ResearchError("question must not be empty")
        self.refresh_settings()  # type: ignore[attr-defined]
        settings = self.config.settings
        if settings.offline:
            raise ResearchError(
                "runtime.offline is on, and an answer sends the question and the "
                "passages it retrieves to DeepSeek. Turn it off with: "
                "research-rag config --write runtime.offline=false"
            )
        key, _origin = deepseek_key()
        if key is None:
            raise ResearchError(
                "No DeepSeek key is set, so nothing was sent. Set one with: "
                "research-rag api-key set"
            )
        if self._load_current_optional() is None:  # type: ignore[attr-defined]
            raise ResearchError("No knowledge base exists; call ingest first")
        if self._answer_activity is not None:
            raise ResearchError(
                "An answer is already running "
                f"({self._answer_activity['phase']}); ask again once it finishes."
            )
        started = time.monotonic()
        activity: dict[str, Any] = {"phase": "starting", "searches": [], "attempt": 0}
        self._answer_activity = activity

        def report(phase: str, detail: dict[str, Any]) -> None:
            activity.update(
                phase=phase,
                elapsed_seconds=round(time.monotonic() - started, 1),
                attempt=detail.get("attempt", activity["attempt"]),
            )
            if "query" in detail:
                activity["searches"].append(detail["query"])

        async def search(query: str, top_k: int) -> dict[str, Any]:
            return await self.search(  # type: ignore[attr-defined]
                query, top_k=top_k, rerank=True, **filters
            )

        async def read(chunk_id: str) -> dict[str, Any]:
            return await self.get_passage(chunk_id, context_chunks=1)  # type: ignore[attr-defined]

        try:
            async with DeepSeek(
                base_url=settings.answers_base_url,
                model=settings.answers_model,
                key=key,
                timeout_seconds=settings.answers_timeout_seconds,
            ) as client:
                with as_caller(ANSWER_CALLER):
                    result = await run_answer(
                        question,
                        search=search,
                        read=read,
                        complete=client.complete,
                        limits=Limits(
                            max_searches=settings.answers_max_searches,
                            max_attempts=settings.answers_max_attempts,
                            timeout_seconds=settings.answers_timeout_seconds,
                        ),
                        report=report,
                    )
        finally:
            self._answer_activity = None
        result["sent"] = {"endpoint": settings.answers_base_url, **result["sent"]}
        return {"provider": PROVIDER, "model": settings.answers_model, **result}

    def answer_activity(self) -> dict[str, Any] | None:
        """What a running answer is doing, for a reader waiting on it."""

        if self._answer_activity is None:
            return None
        return {
            **self._answer_activity,
            "searches": list(self._answer_activity["searches"]),
        }

    async def credentials_read(self) -> dict[str, Any]:
        key, origin = deepseek_key()
        return {"deepseek": {"set": key is not None, "origin": origin}}

    async def credentials_write(
        self, *, api_key: str | None = None, clear: bool = False
    ) -> dict[str, Any]:
        if clear == bool(api_key):
            raise ResearchError("Give an api_key to set, or clear, but not both.")
        if clear:
            clear_deepseek_key()
        else:
            set_deepseek_key(str(api_key))
        return await self.credentials_read()
