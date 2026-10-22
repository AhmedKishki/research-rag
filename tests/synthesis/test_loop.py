"""One synthesis's tool loop, with a scripted model and an in-memory corpus.

The model is a list of turns, so each test states exactly what DeepSeek said
and asserts what the app did with it.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from typing import Any

import pytest

from research_rag.project.policy import ResearchError
from research_rag.synthesis.deepseek import Turn
from research_rag.synthesis.loop import Limits, run_synthesis

QUESTION = "Why did the cooperative dissolve?"
HITS = {
    "c-1": {
        "chunk_id": "c-1",
        "source_id": "src-mensah",
        "source_path": "mensah.pdf",
        "title": "Grain Cooperatives",
        "authors": ["Mensah"],
        "locator": {"page": 112},
        "text": "In 2009 the cooperative dissolved and its warehouse was sold.",
    },
    "c-2": {
        "chunk_id": "c-2",
        "source_id": "src-notes",
        "source_path": "field-notes.pdf",
        "title": "Field notes",
        "authors": [],
        "locator": {"section": "Interviews"},
        "text": "The cooperative dissolved because its members left.",
    },
}
LIMITS = Limits(max_searches=4, max_attempts=2, timeout_seconds=30)
PROSE = "THE MODEL'S OWN PROSE"

DERIVATION: dict[str, Any] = {
    "conclusion": "The cooperative dissolved in 2009 because its members left.",
    "sources": ["S1", "S2"],
    "steps": [
        {
            "id": "A",
            "claim": "members left",
            "supports": "B",
            "relation": "cause",
            "status": "stated",
            "why": "it is the only cause any source gives",
            "licence": {
                "source": "S2",
                "text": "The cooperative dissolved because its members left.",
            },
        },
        {
            "id": "B",
            "claim": "the cooperative dissolved in 2009",
            "supports": "Z",
            "status": "part",
            "why": "it is the event the question asks about",
        },
    ],
    "sentences": [
        {
            "step": ["A", "B"],
            "pieces": [
                {
                    "source": "S1",
                    "text": "In 2009 the cooperative dissolved and its warehouse was sold.",
                    "ops": [{"op": "DELETE", "text": " and its warehouse was sold."}],
                },
                {"source": "S2", "text": "because", "link": "A"},
                {"source": "S2", "text": "its members left."},
            ],
        }
    ],
    "excluded": [],
    "conflicts": [],
    "checks": [
        {"source": "S1", "result": "Keeps the date and the event."},
        {"source": "S2", "result": "The cause is S2's own relation."},
    ],
}


def _call(name: str, arguments: dict[str, Any], n: int = 1) -> Turn:
    return Turn(
        message={
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking",
            "tool_calls": [
                {
                    "id": f"call-{name}-{n}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        usage={"prompt_tokens": 100, "completion_tokens": 10},
    )


def _prose() -> Turn:
    return Turn(message={"role": "assistant", "content": PROSE}, usage={})


def _submit(derivation: dict[str, Any], n: int = 1) -> Turn:
    return _call("submit_derivation", {"derivation": derivation}, n)


class Corpus:
    def __init__(self, hits: dict[str, dict[str, Any]] | None = None) -> None:
        self.hits = hits or HITS
        self.queries: list[tuple[str, int]] = []
        self.reads: list[str] = []

    async def search(self, query: str, top_k: int) -> dict[str, Any]:
        self.queries.append((query, top_k))
        return {"generation_id": "g-1", "hits": list(self.hits.values())}

    async def read(self, chunk_id: str) -> dict[str, Any]:
        self.reads.append(chunk_id)
        return {
            "generation_id": "g-1",
            "context": [
                dict(self.hits[chunk_id]),
                {
                    "chunk_id": "c-excluded",
                    "title": "Grain Cooperatives",
                    "locator": {"page": 113},
                    "text": "A passage the reader excluded.",
                    "excluded_from_search": True,
                },
            ],
        }


class Model:
    """DeepSeek, as a list of turns; it records every request it was sent."""

    def __init__(self, *turns: Turn) -> None:
        self.turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        self.requests.append(copy.deepcopy(messages))
        return self.turns.pop(0)

    def tool_results(self) -> list[dict[str, Any]]:
        last = self.requests[-1]
        return [json.loads(m["content"]) for m in last if m["role"] == "tool"]


async def _synthesise(model: Model, corpus: Corpus | None = None, **limits: Any) -> Any:
    corpus = corpus or Corpus()
    reported: list[tuple[str, dict[str, Any]]] = []
    result = await run_synthesis(
        QUESTION,
        search=corpus.search,
        read=corpus.read,
        complete=model.complete,
        limits=replace(LIMITS, **limits),
        report=lambda phase, detail: reported.append((phase, detail)),
    )
    return result, reported


def test_a_derivation_from_retrieved_passages_renders_the_synthesis() -> None:
    model = Model(
        _call("search", {"query": "cooperative dissolved"}), _submit(DERIVATION)
    )

    result, reported = asyncio.run(_synthesise(model))

    assert result["generated"] is True
    assert result["synthesis"] == (
        "In 2009 the cooperative dissolved because its members left [S1; S2]."
    )
    assert "[S1] Mensah, *Grain Cooperatives*, p. 112." in result["references"]
    assert result["gaps"] == 0
    assert [(s["label"], s["chunk_id"], s["text"]) for s in result["sources"]] == [
        ("S1", "c-1", HITS["c-1"]["text"]),
        ("S2", "c-2", HITS["c-2"]["text"]),
    ]
    assert result["searches"] == [{"query": "cooperative dissolved", "returned": 2}]
    assert result["sent"] == {"chunk_ids": ["c-1", "c-2"]}
    assert result["generation_ids"] == ["g-1"]
    assert result["usage"] == {"prompt_tokens": 200, "completion_tokens": 20}
    assert ("searching", {"query": "cooperative dissolved"}) in reported
    # DeepSeek is sent each assistant message whole, its reasoning included.
    assert model.requests[1][2]["reasoning_content"] == "thinking"


def test_source_text_comes_from_the_retrieval_never_from_the_model() -> None:
    forged = copy.deepcopy(DERIVATION)
    forged["sources"] = {"S1": {"text": "The cooperative was thriving."}, "S2": {}}
    forged["sentences"] = [
        {
            "step": ["A", "B"],
            "pieces": [{"source": "S1", "text": "The cooperative was thriving."}],
        }
    ]
    model = Model(
        _call("search", {"query": "cooperative"}),
        _submit(forged),
        _submit(DERIVATION, 2),
    )

    result, _ = asyncio.run(_synthesise(model))

    (refusal,) = model.tool_results()[-1:]
    assert "not found" in refusal["refused"]
    assert refusal["attempts_left"] == 1
    assert result["generated"] is True
    # The refused attempt is named for a reader who asks; nothing else carries it.
    (refused,) = result["refusals"]
    assert refused.startswith("Attempt 1: ") and "not found" in refused
    assert "thriving" not in json.dumps(
        {k: v for k, v in result.items() if k != "refusals"}
    )


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda d: d.update(sources=["S1", "S9"]), "S9"),
        (
            lambda d: (
                d.update(ai_allowed="all"),
                d["steps"][0].update(
                    status="ai", text="So it ended.", missing="A → B: x"
                ),
            ),
            "not allowed here",
        ),
    ],
    ids=["a label never retrieved", "an AI-authored link"],
)
def test_a_derivation_outside_the_white_box_is_refused_to_the_model(
    change: Any, named: str
) -> None:
    bad = copy.deepcopy(DERIVATION)
    change(bad)
    model = Model(
        _call("search", {"query": "cooperative"}), _submit(bad), _submit(bad, 2)
    )

    result, _ = asyncio.run(_synthesise(model))

    assert named in model.tool_results()[-1]["refused"]
    assert result["generated"] is False


def test_attempts_exhausted_give_no_synthesis_and_no_model_text() -> None:
    bad = copy.deepcopy(DERIVATION)
    bad["sentences"][0]["pieces"][0]["text"] = PROSE
    model = Model(_call("search", {"query": "q"}), _submit(bad), _submit(bad, 2))

    result, _ = asyncio.run(_synthesise(model))

    assert result["generated"] is False
    assert "2 attempts" in result["reason"]
    assert "synthesis" not in result
    assert [r.split(":", 1)[0] for r in result["refusals"]] == [
        "Attempt 1",
        "Attempt 2",
    ]
    assert PROSE not in json.dumps({k: v for k, v in result.items() if k != "refusals"})
    assert result["sent"] == {"chunk_ids": ["c-1", "c-2"]}


def test_searches_beyond_the_bound_are_refused_to_the_model() -> None:
    corpus = Corpus()
    model = Model(
        _call("search", {"query": "first"}),
        _call("search", {"query": "second"}, 2),
        _call("no_synthesis", {}),
    )

    result, _ = asyncio.run(_synthesise(model, corpus, max_searches=1))

    assert corpus.queries == [("first", 8)]
    assert "searches" in model.tool_results()[-1]["refused"]
    assert result["generated"] is False


def test_no_synthesis_ends_without_one() -> None:
    model = Model(_call("search", {"query": "q"}), _call("no_synthesis", {}))

    result, _ = asyncio.run(_synthesise(model))

    assert result["generated"] is False
    assert (
        result["reason"] == "No retrieved passage supplies a step toward a synthesis."
    )


def test_reading_around_a_passage_labels_its_neighbours_but_not_an_excluded_one() -> (
    None
):
    corpus = Corpus()
    model = Model(
        _call("search", {"query": "q"}),
        _call("read_passage", {"label": "S1"}),
        _call("no_synthesis", {}),
    )

    result, _ = asyncio.run(_synthesise(model, corpus))

    assert corpus.reads == ["c-1"]
    read = model.tool_results()[-1]
    assert [p["label"] for p in read["passages"]] == ["S1"]
    assert "c-excluded" not in result["sent"]["chunk_ids"]


def test_prose_outside_a_tool_is_discarded_and_the_model_asked_again() -> None:
    model = Model(_call("search", {"query": "q"}), _prose(), _submit(DERIVATION))

    result, _ = asyncio.run(_synthesise(model))

    assert result["generated"] is True
    assert model.requests[-1][-1]["role"] == "user"
    assert PROSE not in json.dumps(
        {k: v for k, v in result.items() if k != "derivation"}
    )


def test_a_provider_failure_ends_the_answer_and_says_what_was_sent() -> None:
    class Failing(Model):
        async def complete(self, messages: Any, tools: Any) -> Turn:
            if self.requests:
                raise ResearchError("DeepSeek failed on its side. Ask again later.")
            return await super().complete(messages, tools)

    model = Failing(_call("search", {"query": "q"}))

    result, _ = asyncio.run(_synthesise(model))

    assert result["generated"] is False
    assert result["reason"].startswith("DeepSeek failed")
    assert result["sent"] == {"chunk_ids": ["c-1", "c-2"]}


def test_an_answer_that_runs_out_of_time_says_so() -> None:
    class Slow(Model):
        async def complete(self, messages: Any, tools: Any) -> Turn:
            await asyncio.sleep(5)
            raise AssertionError("unreachable")

    result, _ = asyncio.run(_synthesise(Slow(), timeout_seconds=0.05))

    assert result["generated"] is False
    assert "0.05 seconds" in result["reason"]


def test_a_citation_the_passage_carried_is_refused_until_deleted() -> None:
    cited = {
        **HITS,
        "c-2": {
            **HITS["c-2"],
            "text": "The cooperative dissolved because its members left (Mensah 2010: 4).",
        },
    }
    kept = copy.deepcopy(DERIVATION)
    kept["steps"][0]["licence"]["text"] = cited["c-2"]["text"]
    kept["sentences"][0]["pieces"][2]["text"] = "its members left (Mensah 2010: 4)."
    deleted = copy.deepcopy(kept)
    deleted["sentences"][0]["pieces"][2]["ops"] = [
        {"op": "DELETE", "text": " (Mensah 2010: 4)"}
    ]
    model = Model(
        _call("search", {"query": "cooperative"}), _submit(kept), _submit(deleted, 2)
    )

    result, _ = asyncio.run(_synthesise(model, Corpus(cited)))

    (refused,) = result["refusals"]
    assert "(Mensah 2010: 4)" in refused and "DELETE" in refused
    assert result["generated"] is True
    assert "Mensah 2010" not in result["synthesis"]
    assert 'DELETE " (Mensah 2010: 4)"' in result["provenance"]
