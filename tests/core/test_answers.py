"""`ResearchService.answer`: what it refuses before DeepSeek, and what it records.

DeepSeek is a scripted stand-in that reads the passages the app labelled and
copies one whole, so the answer renders from the corpus this test built.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest

import research_rag.core.answers as answers_module
from research_rag.answers.deepseek import Turn
from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.credentials import set_deepseek_key
from research_rag.project.policy import ResearchError
from tests.conftest import write_pdf
from tests.service_helpers import make_service


def _call(name: str, arguments: dict[str, Any]) -> Turn:
    return Turn(
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        usage={"prompt_tokens": 10},
    )


def _copy_the_first_passage(messages: list[dict[str, Any]]) -> Turn:
    found = json.loads(messages[-1]["content"])["passages"]
    if not found:
        return _call("no_answer", {})
    first = found[0]
    return _call(
        "submit_derivation",
        {
            "derivation": {
                "conclusion": "Cobalt is evidence about labour.",
                "sources": [first["label"]],
                "steps": [
                    {
                        "id": "A",
                        "claim": "c",
                        "supports": "Z",
                        "status": "part",
                        "why": "w",
                    }
                ],
                "sentences": [
                    {
                        "step": "A",
                        "pieces": [{"source": first["label"], "text": first["text"]}],
                    }
                ],
                "checks": [{"source": first["label"], "result": "Copied whole."}],
            }
        },
    )


class ScriptedDeepSeek:
    """Stands in for `answers.deepseek.DeepSeek`: search once, then copy."""

    made: ClassVar[list[dict[str, Any]]] = []
    gate: ClassVar[asyncio.Event | None] = None
    asked: ClassVar[asyncio.Event | None] = None

    def __init__(self, **settings: Any) -> None:
        ScriptedDeepSeek.made.append(settings)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def complete(self, messages: list[dict[str, Any]], tools: Any) -> Turn:
        if ScriptedDeepSeek.asked is not None:
            ScriptedDeepSeek.asked.set()
        if ScriptedDeepSeek.gate is not None:
            await ScriptedDeepSeek.gate.wait()
        if messages[-1]["role"] == "user":
            return _call("search", {"query": "cobalt labour"})
        return _copy_the_first_passage(messages)


@pytest.fixture
def deepseek(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedDeepSeek]:
    ScriptedDeepSeek.made = []
    ScriptedDeepSeek.gate = None
    ScriptedDeepSeek.asked = None
    monkeypatch.setattr(answers_module, "DeepSeek", ScriptedDeepSeek)
    set_deepseek_key("sk-core")
    return ScriptedDeepSeek


async def _built(project: Path) -> ResearchService:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence about labour."],
        title="Research Article",
    )
    service = make_service(project)
    await service.ingest(chunk_size=100, chunk_overlap=10)
    return service


def test_an_answer_renders_from_the_corpus_and_names_what_it_sent(
    project: Path, deepseek: type[ScriptedDeepSeek]
) -> None:
    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        service = await _built(project)
        result = await service.answer("What is cobalt evidence about?")
        return result, await service.search_history()

    result, history = asyncio.run(scenario())

    assert result["generated"] is True
    assert "Cobalt evidence about labour" in result["synthesis"]
    assert (result["provider"], result["model"]) == ("deepseek", "deepseek-flash")
    assert result["sent"]["endpoint"] == "https://api.deepseek.com"
    assert result["sent"]["chunk_ids"] == [result["sources"][0]["chunk_id"]]
    assert deepseek.made[0]["key"] == "sk-core"
    assert "sk-core" not in json.dumps(result)
    assert [entry["caller"] for entry in history["searches"]] == ["answer"]


def test_the_readers_filters_bound_every_search_an_answer_makes(
    project: Path, deepseek: type[ScriptedDeepSeek]
) -> None:
    async def scenario() -> dict[str, Any]:
        service = await _built(project)
        return await service.answer("What is cobalt?", titles_any=["Another Book"])

    result = asyncio.run(scenario())

    assert result["generated"] is False
    assert result["searches"] == [{"query": "cobalt labour", "returned": 0}]
    assert result["sent"]["chunk_ids"] == []


@pytest.mark.parametrize(
    ("condition", "named"),
    [
        ("offline", "runtime.offline"),
        ("no key", "research-rag api-key set"),
        ("no generation", "ingest"),
    ],
)
def test_an_answer_is_refused_before_anything_leaves_the_machine(
    project: Path,
    deepseek: type[ScriptedDeepSeek],
    condition: str,
    named: str,
) -> None:
    async def scenario() -> None:
        if condition == "no generation":
            service = make_service(project)
        else:
            service = await _built(project)
        if condition == "offline":
            # runtime.offline is read at start, so an offline app is a new one.
            service = make_service(
                resolve_config(project, offline=True, vanilla_executable=sys.executable)
            )
        if condition == "no key":
            from research_rag.project.credentials import clear_deepseek_key

            clear_deepseek_key()
        await service.answer("What is cobalt?")

    with pytest.raises(ResearchError, match=named):
        asyncio.run(scenario())
    assert deepseek.made == []


def test_a_second_answer_is_refused_while_one_runs_and_status_shows_its_phase(
    project: Path, deepseek: type[ScriptedDeepSeek]
) -> None:
    async def scenario() -> tuple[dict[str, Any], str, dict[str, Any]]:
        service = await _built(project)
        deepseek.gate = asyncio.Event()
        deepseek.asked = asyncio.Event()
        first = asyncio.create_task(service.answer("What is cobalt?"))
        await deepseek.asked.wait()
        status = await service.status()
        with pytest.raises(ResearchError) as refused:
            await service.answer("And labour?")
        deepseek.gate.set()
        return status, str(refused.value), await first

    status, refusal, first = asyncio.run(scenario())

    assert status["answer_activity"]["phase"] == "asking"
    assert "already running" in refusal
    assert first["generated"] is True
