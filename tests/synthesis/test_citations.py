"""A passage's own citations: found in a rendered synthesis, never its labels."""

from __future__ import annotations

import pytest

from research_rag.synthesis.citations import source_citations


@pytest.mark.parametrize(
    ("sentence", "found"),
    [
        ("A relation between things” (p. 86) [S6].", ["(p. 86)"]),
        ("Use-value is fetishised (1981: 131) [S19].", ["(1981: 131)"]),
        ("As Marx argued (Marx 1976: 165) [S2].", ["(Marx 1976: 165)"]),
        ("Baudrillard disagrees (Baudrillard, 1981) [S2].", ["(Baudrillard, 1981)"]),
        (
            "Production [Capital, i, p. 80. mega, ii/6, p. 106] [S6].",
            ["[Capital, i, p. 80. mega, ii/6, p. 106]"],
        ),
        ("As shown in the part[158] on fetishism [S9].", ["[158]"]),
        ("The veil is ideological.12 It hides labour [S1].", ["12"]),
        ("The veil is ideological² [S1].", ["²"]),
    ],
    ids=[
        "page",
        "year and page",
        "author, year and page",
        "author and year",
        "bracketed reference",
        "bracketed footnote",
        "glued footnote",
        "superscript",
    ],
)
def test_a_citation_the_passage_carried_is_found(
    sentence: str, found: list[str]
) -> None:
    assert source_citations(sentence) == found


@pytest.mark.parametrize(
    "sentence",
    [
        "Prices rose sharply after the mill closed in 1990 [S1; S2].",
        "Incomes fell by a third (33 per cent) between 1990 and 1992 [S2].",
        "The mill closed (1990) and prices rose [S1].",
        "Incomes fell [S2]. ⟦GAP: fall from 1990 (B) → closure (Z): see p. 4 of S2 [S2]⟧",
        "The text is cited [sic] as written [S3].",
    ],
    ids=["labels", "a figure", "a bare year", "a gap marker", "an editorial mark"],
)
def test_a_label_a_gap_or_plain_content_is_not_a_citation(sentence: str) -> None:
    assert source_citations(sentence) == []
