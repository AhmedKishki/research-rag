"""Local removal and cumulative PDF admission preserve readable neighbours."""

from pathlib import Path

import pytest

from research_rag.corpus.extraction import (
    ExtractionError,
    UncleanSourceError,
    _repeated_margin_signatures,
    _TextBlock,
    screen_source_units,
)
from research_rag.corpus.sources import SourceFile
from research_rag.corpus.text_quality import clean_damaged_pdf_passage


def _source() -> SourceFile:
    return SourceFile(
        Path("/unused/book.pdf"), "source", "book.pdf", "sources/book.pdf", ".pdf", 0, 0
    )


def _unit(text: str, index: int = 0) -> dict:
    return {
        "id": f"unit_{index}",
        "contents": text,
        "content_kind": "prose",
        "locator": {"type": "pdf_page", "page": index + 1},
        "quality_flags": [],
    }


def test_missing_letters_remove_the_whole_token_and_leave_a_gap() -> None:
    text, lost, spans = clean_damaged_pdf_passage("Alpha bro\ufffdken omega.")
    assert text == "Alpha [...] omega."
    assert lost == len("bro\ufffdken")
    assert spans[0]["start"] == 6 and spans[0]["end"] == 13
    assert "broken" not in text
    assert "bro" not in text


def test_clean_and_multilingual_text_is_unchanged() -> None:
    value = "A quotation 日本語 and café with a • bullet."
    assert clean_damaged_pdf_passage(value) == (value, 0, [])


def test_pdf_uses_document_loss_not_a_short_paragraph_share() -> None:
    record = {"physical_pages": 2}
    retained = screen_source_units(
        _source(),
        record,
        [_unit("A" * 98), _unit("\ufffd\ufffd", 1)],
        maximum_unclean_percent=2.0,
    )
    assert len(retained) == 1
    assert record["substantive_character_count"] == 100
    assert record["discarded_corrupt_character_count"] == 2
    assert record["unclean_character_rate"] == 0.02
    assert record["excluded_corrupt_passage_count"] == 1


def test_readable_prose_survives_local_token_loss() -> None:
    record = {"physical_pages": 1}
    text = "Valid " * 100 + "bad\ufffdword " + "remaining evidence."
    retained = screen_source_units(
        _source(), record, [_unit(text)], maximum_unclean_percent=2.0
    )
    assert "remaining evidence." in retained[0]["contents"]
    assert "badword" not in retained[0]["contents"]
    assert "[...]" in retained[0]["contents"]
    assert record["excluded_corrupt_passage_count"] == 0
    assert record["partially_cleaned_passage_count"] == 1
    assert record["cleaned_corrupt_span_count"] == 1
    assert record["discarded_corrupt_character_count"] == len("bad\ufffdword")
    assert "contents" not in record["cleaned_passages"][0]


def test_symbol_only_padding_does_not_dilute_substantive_loss() -> None:
    record = {"physical_pages": 3}
    with pytest.raises(UncleanSourceError, match="document budget"):
        screen_source_units(
            _source(),
            record,
            [_unit("A" * 98), _unit("\ufffd\ufffd\ufffd", 1), _unit("*" * 10000, 2)],
            maximum_unclean_percent=2.0,
        )
    assert record["substantive_character_count"] == 101
    assert record["discarded_corrupt_character_count"] == 3
    assert record["excluded_symbol_only_passage_count"] == 1


def test_completely_unreadable_pdf_is_not_admitted_even_at_100() -> None:
    with pytest.raises(ExtractionError, match="no readable"):
        screen_source_units(
            _source(),
            {"physical_pages": 1},
            [_unit("\ufffd\ufffd")],
            maximum_unclean_percent=100.0,
        )


def test_loss_budget_measures_removed_words_not_only_unknown_glyphs() -> None:
    record = {"physical_pages": 1}
    with pytest.raises(UncleanSourceError, match="Lost 10 of 60"):
        screen_source_units(
            _source(),
            record,
            [_unit("A" * 50 + " broken\ufffdxyz")],
            maximum_unclean_percent=2.0,
        )
    assert record["discarded_corrupt_character_count"] == 10


def test_chapter_local_damaged_folio_headers_are_confirmed_furniture() -> None:
    pages = []
    for index in range(30):
        text = (
            f"Introduction \ufffd {index + 1}"
            if index < 6
            else f"Unique heading {index}"
        )
        block = _TextBlock(0, (20, 20, 200, 35), (text,), text, 8)
        pages.append(([block], 400.0, 600.0))
    assert "introduction \ufffd #" in _repeated_margin_signatures(pages)


def test_sparse_damaged_margin_text_is_not_assumed_furniture() -> None:
    pages = []
    for index in range(30):
        text = (
            f"Introduction \ufffd {index + 1}"
            if index in (0, 15, 29)
            else f"Unique heading {index}"
        )
        block = _TextBlock(0, (20, 20, 200, 35), (text,), text, 8)
        pages.append(([block], 400.0, 600.0))
    assert "introduction \ufffd #" not in _repeated_margin_signatures(pages)
