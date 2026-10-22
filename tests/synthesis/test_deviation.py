"""A stated deviation waives a form rule and is recorded; it never waives copying.

The cases are the skill's own `tests/test_deviation.py`, so a fix carried from
`agent-skills` brings its case with it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from research_rag.synthesis.ops import Rejected
from research_rag.synthesis.render import render, render_parts

BASE = json.loads(
    (Path(__file__).with_name("fixtures") / "mill.json").read_text(encoding="utf-8")
)
WHY = "The source's own wording reads better whole."


def _sentence(d: dict[str, Any], n: int, why: str = WHY) -> dict[str, Any]:
    d["sentences"][n]["deviation"] = why
    return d["sentences"][n]


def _piece(d: dict[str, Any], n: int) -> dict[str, Any]:
    return d["sentences"][n]["pieces"][0]


CASES = [
    (
        "a split word waived",
        lambda d: (
            _sentence(d, 1),
            _piece(d, 1).update(ops=[{"op": "DELETE", "text": "hold"}]),
        ),
        None,
    ),
    (
        "a splice waived",
        lambda d: _sentence(d, 1)["pieces"].append(
            {"source": "S1", "text": "after the mill closed in 1990."}
        ),
        None,
    ),
    (
        "a late sentence waived",
        lambda d: (d["sentences"].reverse(), _sentence(d, 1)),
        None,
    ),
    (
        "a lowercase start waived",
        lambda d: (_sentence(d, 0), _piece(d, 0)["ops"].pop()),
        None,
    ),
    (
        "a tense change still refused",
        lambda d: (
            _sentence(d, 0),
            _piece(d, 0)["ops"].append(
                {"op": "INFLECT", "kind": "agreement", "from": "rose", "to": "rise"}
            ),
        ),
        "agreement list",
    ),
    (
        "an invented span still refused",
        lambda d: (
            _sentence(d, 1),
            _piece(d, 1).update(text="Household incomes in the town rose.", ops=[]),
        ),
        "not found",
    ),
    (
        "a changed letter still refused",
        lambda d: (
            _sentence(d, 1),
            _piece(d, 1).update(
                ops=[
                    {
                        "op": "NORMALISE",
                        "rule": "quotes",
                        "result": _piece(d, 1)["text"].replace("fell", "felt"),
                    }
                ]
            ),
        ),
        "punctuation only",
    ),
    ("a deviation with nothing to waive recorded", lambda d: _sentence(d, 0), None),
    (
        "a missing full stop refused",
        lambda d: _piece(d, 1).update(text=_piece(d, 1)["text"][:-1]),
        "must end with a full stop",
    ),
    (
        "a missing full stop waived",
        lambda d: (
            _sentence(d, 1),
            _piece(d, 1).update(text=_piece(d, 1)["text"][:-1]),
        ),
        None,
    ),
    (
        "a deviation without a reason",
        lambda d: (
            _sentence(d, 1, ""),
            _piece(d, 1).update(ops=[{"op": "DELETE", "text": "hold"}]),
        ),
        "deviation saying why",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "refusal"), [case[1:] for case in CASES], ids=[case[0] for case in CASES]
)
def test_a_deviation_waives_a_form_rule_and_never_copying(
    mutate: Any, refusal: str | None
) -> None:
    d = copy.deepcopy(BASE)
    mutate(d)
    if refusal is not None:
        with pytest.raises(Rejected, match=refusal):
            render(d)
        return
    out = render(d)
    assert f"DEVIATION: {WHY}" in out
    assert "- Deviations:\n  - Sentence" in out


def test_the_parts_name_each_deviation_and_each_sentence() -> None:
    d = copy.deepcopy(BASE)
    _sentence(d, 1)
    _piece(d, 1).update(ops=[{"op": "DELETE", "text": "hold"}])

    parts = render_parts(d)

    (deviation,) = parts["deviations"]
    assert deviation["sentence"] == 2 and deviation["reason"] == WHY
    assert any("splits a word" in waived for waived in deviation["waived"])
    assert [s["deviation"] for s in parts["sentences"]] == [None, WHY]
    assert parts["sentences"][0]["text"].startswith("Prices in the town")


def test_a_deviation_that_waived_nothing_is_still_recorded() -> None:
    d = copy.deepcopy(BASE)
    _sentence(d, 0)

    parts = render_parts(d)

    assert parts["deviations"] == [{"sentence": 1, "reason": WHY, "waived": []}]
    assert (
        "1. S1: " in parts["provenance"] and f"DEVIATION: {WHY}" in parts["provenance"]
    )
    assert f"  - Sentence 1: {WHY} No rule needed it." in parts["reasoning"]


def test_an_end_stop_restores_the_full_stop_a_span_left_behind() -> None:
    d = copy.deepcopy(BASE)
    _piece(d, 1).update(
        text=_piece(d, 1)["text"][:-1], ops=[{"op": "NORMALISE", "rule": "end-stop"}]
    )

    assert "between 1990 and 1992 [S2]." in render_parts(d)["synthesis"]
