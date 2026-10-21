"""The white-box renderer: the copy renders and refuses as the skill's does.

Each case mutates the skill's worked example, as the skill's own test does, so a
fix carried from `agent-skills` brings its case with it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from research_rag.answers.ops import Rejected
from research_rag.answers.render import render, render_parts

FIXTURES = Path(__file__).with_name("fixtures")
BASE = json.loads((FIXTURES / "mill.json").read_text(encoding="utf-8"))
COOP = json.loads((FIXTURES / "coop.json").read_text(encoding="utf-8"))
S3 = {
    "author": "Okafor",
    "title": "Town Notes",
    "locator": "p. 9",
    "text": "Incomes fell because the mill had closed.",
}


def _piece(d: dict[str, Any], n: int, k: int = 0) -> dict[str, Any]:
    return d["sentences"][n]["pieces"][k]


def _join(d: dict[str, Any], connective: str = "because", source: str = "S3") -> None:
    """Licensed join: B's link is stated by S3 and realised by its own connective."""

    d["sources"]["S3"] = S3
    d["checks"].append({"source": "S3", "result": "States the cause used for B."})
    d["steps"][1].update(status="stated", licence={"source": "S3", "text": S3["text"]})
    d["steps"][1].update(supports="A", relation="cause")
    d["steps"][0], d["steps"][1] = d["steps"][1], d["steps"][0]
    d["sentences"] = [
        {
            "step": ["B", "A"],
            "pieces": [
                {
                    "source": "S2",
                    "text": "Household incomes in the town fell by a third "
                    "between 1990 and 1992.",
                    "ops": [{"op": "DELETE", "text": "."}],
                },
                {"source": source, "text": connective, "link": "B"},
                {"source": "S1", "text": "the mill closed in 1990."},
            ],
        }
    ]


def _replace(d: dict[str, Any], other: dict[str, Any]) -> None:
    d.clear()
    d.update(copy.deepcopy(other))


CASES = [
    ("worked example renders", lambda d: None, None),
    ("documented join renders", lambda d: _replace(d, COOP), None),
    (
        "splice without connective",
        lambda d: d["sentences"][1]["pieces"].append(
            {"source": "S1", "text": "after the mill closed in 1990."}
        ),
        "without a licensed connective",
    ),
    ("licensed join renders", _join, None),
    (
        "connective from another source",
        lambda d: _join(d, "after", "S1"),
        "licence's source",
    ),
    ("connective outside licence", lambda d: _join(d, "sharply", "S3"), "not found"),
    (
        "tense change refused",
        lambda d: _piece(d, 0)["ops"].append(
            {"op": "INFLECT", "kind": "agreement", "from": "rose", "to": "rise"}
        ),
        "agreement list",
    ),
    (
        "number agreement allowed",
        lambda d: _piece(d, 1).update(
            ops=[{"op": "INFLECT", "kind": "agreement", "from": "fell", "to": "fells"}]
        ),
        None,
    ),
    (
        "punctuation outside list",
        lambda d: _piece(d, 1).update(
            ops=[
                {
                    "op": "NORMALISE",
                    "rule": "quotes",
                    "result": _piece(d, 1)["text"][:-1] + "?",
                }
            ]
        ),
        "quote glyphs",
    ),
    (
        "unknown normalise rule",
        lambda d: _piece(d, 1).update(ops=[{"op": "NORMALISE", "rule": "punctuation"}]),
        "closed list",
    ),
    (
        "undeclared pair",
        lambda d: _piece(d, 1).update(
            ops=[{"op": "NORMALISE", "rule": "pair", "from": "third", "to": "half"}]
        ),
        "not declared",
    ),
    (
        "delete splits a word",
        lambda d: _piece(d, 1).update(ops=[{"op": "DELETE", "text": "hold"}]),
        "splits a word",
    ),
    (
        "ai link not allowed",
        lambda d: d["steps"][1].update(status="ai", text="This followed."),
        "not allowed here",
    ),
    (
        "ai link with a date",
        lambda d: (
            d.update(ai_allowed=["B"]),
            d["steps"].append({"id": "Z", "claim": "c"}),
            d["steps"][1].update(supports="Z", status="ai", text="This began in 1990."),
        ),
        "figure, date",
    ),
    (
        "ai link to unstated Z",
        lambda d: (
            d.update(ai_allowed="all"),
            d["steps"][1].update(status="ai", text="Over the same years,"),
        ),
        "stated steps only",
    ),
    (
        "stated Z step renders",
        lambda d: (
            d["steps"].pop(),
            d["sentences"].pop(),
            d["steps"].insert(0, {"id": "Z", "claim": "prices rose"}),
            d["steps"].pop(),
            d["sentences"][0].update(step="Z"),
        ),
        None,
    ),
    (
        "step off the chain",
        lambda d: d["steps"][1].update(supports="Q"),
        "unknown step",
    ),
    (
        "loop",
        lambda d: (
            d["steps"][0].update(supports="B", status="gap", missing="A → B: x"),
            d["steps"][1].update(supports="A"),
        ),
        "loops",
    ),
    ("missing source check", lambda d: d["checks"].pop(), "contradiction check"),
    ("sentences out of order", lambda d: d["sentences"].reverse(), "chain order"),
    ("step with no sentence", lambda d: d["sentences"].pop(), "No source sentence"),
    (
        "copy from AI-authored span",
        lambda d: (
            d["sources"]["S2"].update(
                text=d["sources"]["S2"]["text"]
                + " ⟦AI-AUTHORED: The closure caused it.⟧"
            ),
            _piece(d, 1).update(text="The closure caused it."),
        ),
        "AI-authored or gap marker",
    ),
    (
        "antecedent from same source",
        lambda d: (
            d["sources"]["S2"].update(
                text="Household incomes fell. They fell by a third."
            ),
            _piece(d, 1).update(
                text="They fell by a third.",
                ops=[
                    {
                        "op": "INFLECT",
                        "kind": "antecedent",
                        "from": "They",
                        "to": "Household incomes",
                    }
                ],
            ),
        ),
        None,
    ),
    (
        "antecedent from another source",
        lambda d: (
            d["sources"]["S2"].update(text="They fell by a third."),
            _piece(d, 1).update(
                text="They fell by a third.",
                ops=[
                    {
                        "op": "INFLECT",
                        "kind": "antecedent",
                        "from": "They",
                        "to": "Prices",
                    }
                ],
            ),
        ),
        "same source",
    ),
    (
        "end-stop on a question",
        lambda d: (
            d["sources"]["S2"].update(text="Did incomes fall by a third?"),
            _piece(d, 1).update(
                text="incomes fall by a third",
                ops=[{"op": "NORMALISE", "rule": "end-stop"}],
            ),
        ),
        "full stop",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "refusal"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_a_derivation_renders_or_is_refused_as_the_skill_decides(
    mutate: Any, refusal: str | None
) -> None:
    d = copy.deepcopy(BASE)
    mutate(d)
    if refusal is None:
        render(d)
        return
    with pytest.raises(Rejected, match=refusal):
        render(d)


def test_the_worked_example_renders_the_four_parts_word_for_word() -> None:
    parts = render_parts(copy.deepcopy(BASE))

    assert parts["synthesis"] == (
        "Prices in the town rose sharply after the mill closed in 1990 [S1]. "
        "Household incomes in the town fell by a third between 1990 and 1992 [S2]. "
        "⟦GAP: income fall from 1990 to 1992 (B) → after the mill closed (Z): "
        "no source places the fall after the closure or links it to the mill⟧"
    )
    assert parts["references"] == (
        "- [S1] Lee, *Example Book*, p. 3.\n"
        "- [S2] Diaz, *Sample Study*, ch. 2. Retrieval result, wording not checked "
        "against the original."
    )
    assert parts["provenance"] == (
        '1. S1: DELETE "As noted earlier, " · NORMALISE case "p" → "P"\n'
        "2. S2: verbatim\n"
        "- After 2: GAP for B → Z"
    )
    assert parts["reasoning"].endswith(
        "- Wording check: passed by the white-box renderer. Meaning checks: best effort."
    )
    assert render(copy.deepcopy(BASE)).startswith("**Synthesis**\n\nPrices in the town")


@pytest.mark.parametrize(
    "malformed",
    [
        {"sources": {"S1": "text, not an object"}},
        {**BASE, "sentences": 5},
        [],
    ],
    ids=["source not an object", "sentences not a list", "not an object"],
)
def test_a_malformed_derivation_is_refused_not_raised(malformed: Any) -> None:
    with pytest.raises(Rejected):
        render(malformed)
