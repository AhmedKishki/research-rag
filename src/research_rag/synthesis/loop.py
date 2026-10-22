"""One synthesis: DeepSeek searches the corpus and submits a derivation.

The model is given four tools and answers only through them. A search or a read
labels each passage it returns once, `S1`, `S2`, …, and that labelled passage is
the only text a derivation may copy: the app builds the derivation's sources
from what this run retrieved, whatever the model put there. The renderer then
replays the derivation or refuses it, and a refusal goes back to the model.

Nothing the model writes reaches the reader as an answer except through the
renderer. A synthesis that fails says why in the app's own words, a verbose
result lists the rule each refused attempt broke, which may quote the wording
the model tried, and every result says which passages were sent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from ..project.policy import ResearchError
from .citations import source_citations
from .deepseek import Turn
from .ops import Rejected
from .render import render_parts

Search = Callable[[str, int], Awaitable[Mapping[str, Any]]]
Read = Callable[[str], Awaitable[Mapping[str, Any]]]
Complete = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[Turn]]
Report = Callable[[str, dict[str, Any]], None]

SYSTEM_PROMPT = files(__package__).joinpath("prompt.md").read_text(encoding="utf-8")

# The note each reference carries: the text is cleaned retrieval text, never
# checked against the original here.
RETRIEVAL_NOTE = "Retrieval result, wording not checked against the original"
DEFAULT_TOP_K = 8
MAXIMUM_TOP_K = 20
NO_SYNTHESIS = "No retrieved passage supplies a step toward a synthesis."
_LAST_ATTEMPT = (
    " This is the last attempt: keep every sentence that renders, drop or give a "
    "deviation to any that does not, and mark each link it leaves as a gap."
)
_NUDGE = (
    "Reply only by calling a tool: search, read_passage, submit_derivation, "
    "or no_synthesis."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the reader's corpus. Returns labelled passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAXIMUM_TOP_K,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_passage",
            "description": "Return the passages around a labelled passage.",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_derivation",
            "description": "Submit the white-box derivation that answers the question.",
            "parameters": {
                "type": "object",
                "properties": {"derivation": {"type": "object"}},
                "required": ["derivation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_synthesis",
            "description": "End the synthesis: no retrieved passage supplies any step.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass(frozen=True, slots=True)
class Limits:
    max_searches: int
    max_attempts: int
    timeout_seconds: float

    @property
    def max_turns(self) -> int:
        # Every retrieval and every attempt is a turn, plus room for a turn
        # that called no tool and the one that ends the synthesis.
        return self.max_searches + self.max_attempts + 4


@dataclass(slots=True)
class _Outcome:
    content: dict[str, Any]
    final: dict[str, Any] | None = None


class _Run:
    """What one synthesis has retrieved, sent, spent, and been refused."""

    def __init__(
        self, question: str, search: Search, read: Read, limits: Limits, report: Report
    ) -> None:
        self.question = question
        self._search = search
        self._read = read
        self.limits = limits
        self.report = report
        self.passages: dict[str, dict[str, Any]] = {}
        self._labels: dict[str, str] = {}
        self.searches: list[dict[str, Any]] = []
        self.retrievals = 0
        self.attempts = 0
        self.refusals: list[str] = []
        self.generation_ids: list[str] = []
        self.usage: dict[str, int] = {}

    async def call(self, name: str, arguments: Mapping[str, Any]) -> _Outcome:
        if name == "search":
            return await self._search_tool(arguments)
        if name == "read_passage":
            return await self._read_tool(arguments)
        if name == "submit_derivation":
            return self._submit_tool(arguments)
        if name == "no_synthesis":
            return _Outcome({"ended": True}, self.failed(NO_SYNTHESIS))
        return _Outcome({"refused": f"There is no tool named {name!r}."})

    async def _search_tool(self, arguments: Mapping[str, Any]) -> _Outcome:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return _Outcome({"refused": "search needs a query."})
        refusal = self._retrieval_refusal()
        if refusal:
            return refusal
        top_k = arguments.get("top_k")
        top_k = (
            min(max(top_k, 1), MAXIMUM_TOP_K)
            if isinstance(top_k, int) and not isinstance(top_k, bool)
            else DEFAULT_TOP_K
        )
        self.retrievals += 1
        self.report("searching", {"query": query})
        try:
            payload = await self._search(query, top_k)
        except ResearchError as exc:
            return _Outcome({"refused": str(exc)})
        passages = self._label(payload, payload.get("hits") or [])
        self.searches.append({"query": query, "returned": len(passages)})
        return _Outcome(
            {"passages": passages, "searches_left": self._retrievals_left()}
        )

    async def _read_tool(self, arguments: Mapping[str, Any]) -> _Outcome:
        label = str(arguments.get("label") or "")
        if label not in self.passages:
            return _Outcome(
                {"refused": f"{label!r} was not retrieved in this synthesis."}
            )
        refusal = self._retrieval_refusal()
        if refusal:
            return refusal
        self.retrievals += 1
        self.report("reading", {"label": label})
        try:
            payload = await self._read(self.passages[label]["chunk_id"])
        except ResearchError as exc:
            return _Outcome({"refused": str(exc)})
        # A neighbour the reader excluded is context for a reader, never
        # evidence, so it is neither labelled nor sent.
        context = [
            item
            for item in payload.get("context") or []
            if not item.get("excluded_from_search")
        ]
        passages = self._label(payload, context)
        return _Outcome(
            {"passages": passages, "searches_left": self._retrievals_left()}
        )

    def _submit_tool(self, arguments: Mapping[str, Any]) -> _Outcome:
        self.attempts += 1
        self.report("rendering", {"attempt": self.attempts})
        submitted = arguments.get("derivation", arguments)
        try:
            derivation = self._derivation(submitted)
            parts = render_parts(derivation)
            _hold_citations(parts)
        except Rejected as exc:
            self.refusals.append(f"Attempt {self.attempts}: {exc}")
            left = self.limits.max_attempts - self.attempts
            if left <= 0:
                return _Outcome(
                    {"refused": str(exc), "attempts_left": 0},
                    self.failed(
                        "No derivation passed the renderer in "
                        f"{self.attempts} attempts."
                    ),
                )
            refused = str(exc) + (_LAST_ATTEMPT if left == 1 else "")
            return _Outcome({"refused": refused, "attempts_left": left})
        labels = list(derivation["sources"])
        return _Outcome(
            {"rendered": True},
            {
                "generated": True,
                "question": self.question,
                **parts,
                "gaps": sum(
                    step.get("status") == "gap" for step in derivation["steps"]
                ),
                "sources": [self._source_record(label) for label in labels],
                "derivation": derivation,
                **self._disclosure(),
            },
        )

    def _derivation(self, submitted: Any) -> dict[str, Any]:
        if not isinstance(submitted, Mapping):
            raise Rejected("The derivation must be an object.")
        labels = submitted.get("sources")
        if isinstance(labels, Mapping):
            labels = list(labels)
        if not isinstance(labels, list) or not labels:
            raise Rejected("sources must list the labels the derivation uses.")
        labels = [str(label) for label in dict.fromkeys(map(str, labels))]
        unknown = [label for label in labels if label not in self.passages]
        if unknown:
            raise Rejected(
                f"{', '.join(unknown)} was not retrieved in this synthesis; "
                "name only labels a search or read returned."
            )
        derivation = {
            key: value
            for key, value in submitted.items()
            if key not in {"output_context"}
        }
        derivation.update(
            query=self.question,
            sources={label: self._source_entry(label) for label in labels},
            ai_allowed=[],
            normalisations=[],
        )
        return derivation

    def _retrieval_refusal(self) -> _Outcome | None:
        if self.retrievals < self.limits.max_searches:
            return None
        return _Outcome(
            {
                "refused": (
                    f"This synthesis has used its {self.limits.max_searches} searches "
                    "and reads. Submit a derivation from the passages retrieved, or "
                    "call no_synthesis."
                )
            }
        )

    def _retrievals_left(self) -> int:
        return self.limits.max_searches - self.retrievals

    def _label(
        self, payload: Mapping[str, Any], items: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        generation = payload.get("generation_id")
        if generation and generation not in self.generation_ids:
            self.generation_ids.append(str(generation))
        shown = []
        for item in items:
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or not str(item.get("text") or "").strip():
                continue
            label = self._labels.get(chunk_id)
            if label is None:
                label = f"S{len(self._labels) + 1}"
                self._labels[chunk_id] = label
                self.passages[label] = dict(item)
            passage = self.passages[label]
            shown.append(
                {
                    "label": label,
                    "title": passage.get("title"),
                    "authors": passage.get("authors") or [],
                    "locator": _locator(passage.get("locator")),
                    "text": passage["text"],
                }
            )
        return shown

    def _source_entry(self, label: str) -> dict[str, Any]:
        passage = self.passages[label]
        entry = {
            "text": passage["text"],
            "title": passage.get("title"),
            "author": "; ".join(passage.get("authors") or []) or None,
            "locator": _locator(passage.get("locator")),
            "note": RETRIEVAL_NOTE,
        }
        return {key: value for key, value in entry.items() if value}

    def _source_record(self, label: str) -> dict[str, Any]:
        passage = self.passages[label]
        record = {"label": label}
        for key in (
            "chunk_id",
            "source_id",
            "source_path",
            "title",
            "authors",
            "locator",
            "text",
        ):
            if passage.get(key) is not None:
                record[key] = passage[key]
        return record

    def add_usage(self, usage: Mapping[str, int]) -> None:
        for name, value in usage.items():
            self.usage[name] = self.usage.get(name, 0) + value

    def _disclosure(self) -> dict[str, Any]:
        return {
            "searches": self.searches,
            "refusals": self.refusals,
            "generation_ids": self.generation_ids,
            "sent": {
                "chunk_ids": [
                    self.passages[label]["chunk_id"] for label in self.passages
                ]
            },
            "usage": self.usage,
        }

    def failed(self, reason: str) -> dict[str, Any]:
        return {
            "generated": False,
            "question": self.question,
            "reason": reason,
            **self._disclosure(),
        }


def _hold_citations(parts: dict[str, Any]) -> None:
    """Refuse a citation a passage carried, unless its sentence states a deviation.

    A sentence's passage label is its citation. A citation kept under a deviation
    is recorded like any rule a deviation waives: in `deviations`, on the
    sentence's provenance line, and in the Reasoning.
    """

    refused = []
    deviations = {d["sentence"]: d for d in parts["deviations"]}
    for sentence in parts.pop("sentences"):
        kept = source_citations(sentence["text"])
        if not kept:
            continue
        if not sentence["deviation"]:
            refused += kept
            continue
        entry = deviations.setdefault(
            sentence["sentence"],
            {
                "sentence": sentence["sentence"],
                "reason": sentence["deviation"],
                "waived": [],
            },
        )
        entry["waived"] += [
            f"Keeps the citation {c!r} its passage carried." for c in kept
        ]
    if refused:
        raise Rejected(
            "The synthesis keeps citations its passages carried: "
            + ", ".join(repr(citation) for citation in refused)
            + ". DELETE each one, since the passage label is the citation, or give "
            "the sentence a deviation saying why."
        )
    parts["deviations"] = [deviations[n] for n in sorted(deviations)]
    _record_deviations(parts)


def _record_deviations(parts: dict[str, Any]) -> None:
    """Write `deviations` into the provenance lines and the Reasoning's list."""

    marked = {d["sentence"]: d["reason"] for d in parts["deviations"]}
    lines = parts["provenance"].split("\n")
    for i, line in enumerate(lines):
        number = line.split(". ", 1)[0]
        if number.isdigit() and int(number) in marked and "· DEVIATION:" not in line:
            lines[i] = f"{line} · DEVIATION: {marked[int(number)]}"
    parts["provenance"] = "\n".join(lines)
    lines = parts["reasoning"].split("\n")
    start = lines.index("- Deviations:") + 1
    end = start
    while end < len(lines) and lines[end].startswith("  - "):
        end += 1
    lines[start:end] = [
        f"  - Sentence {d['sentence']}: {d['reason']} "
        + (f"Waived: {' '.join(d['waived'])}" if d["waived"] else "No rule needed it.")
        for d in parts["deviations"]
    ] or ["  - None."]
    parts["reasoning"] = "\n".join(lines)


def _locator(locator: Any) -> str | None:
    """Where a passage sits, as a reference prints it: its page or its section."""

    if not isinstance(locator, Mapping):
        return None
    page = locator.get("page")
    if page is not None:
        label = locator.get("page_label")
        printed = (
            f" (printed {label})"
            if label is not None and str(label) != str(page)
            else ""
        )
        return f"p. {page}{printed}"
    for key in ("section", "section_title", "href", "section_index"):
        value = locator.get(key)
        if value is not None:
            return f"section {value}"
    return None


def _arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        arguments = None
    return name, arguments if isinstance(arguments, dict) else {}


async def run_synthesis(
    question: str,
    *,
    search: Search,
    read: Read,
    complete: Complete,
    limits: Limits,
    report: Report,
) -> dict[str, Any]:
    """Run one synthesis and return its result, generated or not."""

    run = _Run(question, search, read, limits, report)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    try:
        async with asyncio.timeout(limits.timeout_seconds):
            for _ in range(limits.max_turns):
                report("asking", {})
                turn = await complete(messages, TOOLS)
                run.add_usage(turn.usage)
                messages.append(turn.message)
                calls = turn.message.get("tool_calls") or []
                if not calls:
                    messages.append({"role": "user", "content": _NUDGE})
                    continue
                for call in calls:
                    name, arguments = _arguments(call)
                    outcome = await run.call(name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "content": json.dumps(outcome.content, ensure_ascii=False),
                        }
                    )
                    if outcome.final is not None:
                        return outcome.final
    except TimeoutError:
        return run.failed(
            f"The synthesis did not finish within {limits.timeout_seconds:g} seconds."
        )
    except ResearchError as exc:
        return run.failed(str(exc))
    return run.failed(
        f"DeepSeek did not submit a derivation within {limits.max_turns} turns."
    )
