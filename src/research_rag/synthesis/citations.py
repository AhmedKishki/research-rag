"""The citations a passage carried into a synthesis, which the model must delete.

A synthesis cites a passage by its label, such as [S1]. A passage often brings
its own references with it: a page, an author and year, a bracketed reference,
or a footnote number. Those point at works the reader is not shown, so the loop
refuses a synthesis that keeps one and the model deletes it, which provenance
records. A label and a gap marker are apparatus and are never read here.

The patterns are this app's, not the white-box-synthesis renderer's: a draft
built from a reader's own notes may want its citations, and a retrieval answer
does not.
"""

from __future__ import annotations

import re

_LABELS = re.compile(r"\[S\d+(?:; S\d+)*\]")
_MARKERS = re.compile(r"⟦[^⟧]*⟧")
_YEAR = r"(?:1[5-9]|20)\d{2}[a-z]?"
_NAME = r"[A-Z][\w'’-]+"
_CITATIONS = re.compile(
    "|".join(
        (
            # A bracketed reference: [12], [Capital, i, p. 80].
            r"\[[^\[\]]*\d[^\[\]]*\]",
            # A page: (p. 86), (pp. 12-14).
            r"\(pp?\.\s*[\dxivlc]+[^()]*\)",
            # A year and a page, with or without a name: (1981: 131), (Marx 1976: 165).
            rf"\((?:{_NAME}(?: et al\.)?,?\s)?{_YEAR}\s*[:,]\s*(?:pp?\.\s*)?\d[^()]*\)",
            # A name and a year: (Baudrillard, 1981), (Smith and Jones 2004).
            (
                rf"\((?:see |cf\. )?{_NAME}(?:(?:,| and| &) {_NAME})*(?: et al\.)?,? "
                rf"{_YEAR}(?:;[^()]*)?\)"
            ),
            # A raised footnote number.
            r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
            # A footnote number glued to a word's punctuation: ideological.12
            r"(?<=[a-z][.,;:”’\"')])\d{1,3}\b",
        )
    )
)


def source_citations(text: str) -> list[str]:
    """Each citation a passage carried that a rendered synthesis still holds."""

    plain = _MARKERS.sub(" ", _LABELS.sub(" ", text))
    found: list[str] = []
    for match in _CITATIONS.finditer(plain):
        if match.group() not in found:
            found.append(match.group())
    return found
