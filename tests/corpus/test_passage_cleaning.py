"""Per-passage text repair, and the passage-level quality verdict.

`clean_unclean_passage` reverses a UTF-8 stream read with a single-byte codec.
`passage_health_reasons` judges one passage on its own evidence and its own
unreadable share. With `passage_level=True`, `source_health_reasons` refuses a
PDF whose cumulative discarded substantive text exceeds the accepted share;
`None` leaves the loss uncapped, and a file with nothing readable is always
refused. Every case here is a string, so no document or runtime is involved.
"""

from __future__ import annotations

from collections import Counter

import pytest

from research_rag.corpus.text_normalization import (
    clean_unclean_passage,
    is_known_formatting_glyph,
    recover_formatting_glyphs,
)
from research_rag.corpus.text_quality import (
    CHUNK_FLAG_CORRUPT_TEXT,
    CHUNK_FLAG_EXTRACTION_ARTIFACT,
    SOURCE_REASON_NO_TEXT,
    SOURCE_REASON_NO_TEXT_LAYER,
    SOURCE_REASON_UNCLEAN,
    chunk_health_flags,
    passage_health_reasons,
    source_health_reasons,
    text_corruption_reasons,
    text_health_reasons,
)


def _misdecoded_as_latin1(text: str) -> str:
    """The text as its UTF-8 bytes wrongly read as Latin-1."""

    return text.encode("utf-8").decode("latin-1")


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def test_bilingual_clean_text_is_returned_unchanged() -> None:
    text = "Résumé — the council kept 中文 records beside the Greek «λόγος»."

    assert clean_unclean_passage(text) == text
    assert passage_health_reasons(text) == []


def test_mojibake_accents_and_punctuation_are_repaired() -> None:
    assert clean_unclean_passage("The cafÃ© reopened.") == "The café reopened."
    assert clean_unclean_passage("itâ€™s the councilâ€™s minute") == (
        "it’s the council’s minute"
    )
    assert clean_unclean_passage("â€œquotedâ€\u009d") == "“quoted”"
    assert clean_unclean_passage("pages 1â€“3") == "pages 1–3"


def test_repair_reuses_the_existing_normalization() -> None:
    # The ligature arrives mojibake and is folded by the normalization the
    # repair returns through.
    assert clean_unclean_passage("ï¬\u0081nal draft") == "final draft"


def test_mixed_non_latin_text_repairs_only_the_damaged_run() -> None:
    clean = "中文 café and текст stay readable"

    damaged = _misdecoded_as_latin1(clean)

    assert damaged != clean
    assert clean_unclean_passage(damaged) == clean


def test_double_encoded_text_is_repaired_within_two_passes() -> None:
    assert clean_unclean_passage("cafÃƒÂ©") == "café"
    assert passage_health_reasons(clean_unclean_passage("cafÃƒÂ©")) == []


def test_irreparable_missing_glyphs_are_retained_for_withholding() -> None:
    repaired = clean_unclean_passage("cafÃ© text \ufffd\ufffd remains")

    assert "café" in repaired
    assert repaired.count("\ufffd") == 2
    assert "replacement_characters" in passage_health_reasons(repaired)


def test_a_replacement_character_inside_a_word_is_not_deleted() -> None:
    damaged = "the dam\ufffdaged quay record"

    assert clean_unclean_passage(damaged) == damaged
    assert "\ufffd" in clean_unclean_passage(damaged)


# ---------------------------------------------------------------------------
# A passage's own verdict
# ---------------------------------------------------------------------------


def test_without_a_share_only_the_mechanical_reasons_are_returned() -> None:
    damaged = "The cafÃ© text is a known damaged encoding sequence."

    assert passage_health_reasons(damaged) == text_health_reasons(damaged)
    assert passage_health_reasons(damaged) == ["known_mojibake"]


def test_a_passage_within_the_accepted_share_is_not_flagged_unclean() -> None:
    text = (
        "A readable sentence about the council and the quay. " * 40
    ) + "\ufffd\ufffd"

    assert passage_health_reasons(text, maximum_unclean_percent=1.0) == [
        "replacement_characters"
    ]


def test_a_passage_over_the_share_is_flagged_unclean_text() -> None:
    reasons = passage_health_reasons(
        "ok \ufffd\ufffd\ufffd\ufffd", maximum_unclean_percent=1.0
    )

    assert reasons == ["replacement_characters", SOURCE_REASON_UNCLEAN]


def test_the_share_never_waives_mechanical_evidence_even_at_one_hundred() -> None:
    reasons = passage_health_reasons(
        "ok \ufffd\ufffd\ufffd\ufffd", maximum_unclean_percent=100.0
    )

    assert "replacement_characters" in reasons
    assert SOURCE_REASON_UNCLEAN not in reasons


def test_bad_control_characters_count_toward_the_share() -> None:
    text = "a\x01b\x02c"

    assert passage_health_reasons(text, maximum_unclean_percent=1.0) == [
        SOURCE_REASON_UNCLEAN
    ]
    assert passage_health_reasons(text, maximum_unclean_percent=100.0) == []


def test_clean_text_under_a_share_adds_nothing() -> None:
    text = "A clean bilingual passage: café, 中文, текст."

    assert passage_health_reasons(text, maximum_unclean_percent=1.0) == []


# ---------------------------------------------------------------------------
# A source survives a readable minority
# ---------------------------------------------------------------------------


def test_a_source_with_no_retained_text_is_refused() -> None:
    assert source_health_reasons(
        unit_count=0,
        retained_count=0,
        withheld_reasons=Counter(),
    ) == [SOURCE_REASON_NO_TEXT]
    assert source_health_reasons(
        unit_count=3,
        retained_count=0,
        withheld_reasons=Counter({"symbol_only": 3}),
    ) == [SOURCE_REASON_NO_TEXT]


def test_a_fully_scanned_file_with_nothing_retained_names_the_missing_layer() -> None:
    assert source_health_reasons(
        unit_count=0,
        retained_count=0,
        withheld_reasons=Counter(),
        image_only_pages=8,
        physical_pages=8,
    ) == [SOURCE_REASON_NO_TEXT_LAYER]


def test_a_loss_over_the_document_budget_refuses_the_source() -> None:
    """A PDF's cumulative loss is the one share that can refuse it.

    A readable minority does not keep a source once the characters the cleaner had
    to discard exceed the accepted share. The same figures under no cap are kept,
    because `None` states no budget rather than a zero one.
    """

    figures = {
        "unit_count": 1000,
        "retained_count": 1,
        "withheld_reasons": Counter({"replacement_characters": 999}),
        "kept_characters": 200,
        "removed_characters": 90000,
        "letter_characters": 0,
        "withheld_characters": 5000,
        "passage_level": True,
    }

    assert source_health_reasons(**figures, maximum_unclean_percent=None) == []
    assert source_health_reasons(**figures, maximum_unclean_percent=0.1) == [
        SOURCE_REASON_UNCLEAN
    ]


def test_a_numeric_text_source_is_kept() -> None:
    assert (
        source_health_reasons(
            unit_count=3,
            retained_count=3,
            withheld_reasons=Counter(),
            kept_characters=400,
            letter_characters=0,
            passage_level=True,
        )
        == []
    )


def test_the_removed_source_vetoes_are_no_longer_reachable() -> None:
    from research_rag.corpus.text_quality import (
        SOURCE_REASON_ALMOST_ALL_WITHHELD,
        SOURCE_REASON_ARTIFACT_LADEN,
        SOURCE_REASON_NO_LETTER_TEXT,
        SOURCE_REASON_UNSAFE_TO_CLEAN,
    )

    reasons = source_health_reasons(
        unit_count=100,
        retained_count=1,
        withheld_reasons=Counter({"symbol_only": 99}),
        kept_characters=1,
        removed_characters=1_000_000,
        letter_characters=0,
        withheld_characters=1_000_000,
        maximum_unclean_percent=None,
        passage_level=True,
    )

    assert reasons == []
    assert SOURCE_REASON_ALMOST_ALL_WITHHELD not in reasons
    assert SOURCE_REASON_ARTIFACT_LADEN not in reasons
    assert SOURCE_REASON_UNSAFE_TO_CLEAN not in reasons
    assert SOURCE_REASON_NO_LETTER_TEXT not in reasons
    assert SOURCE_REASON_UNCLEAN not in reasons


# ---------------------------------------------------------------------------
# A formatting glyph a symbol face exposed as private use
# ---------------------------------------------------------------------------


def test_a_symbol_bullet_is_recovered_and_never_withheld() -> None:
    raw = "\uf0b7 electricity consumption in TWh"

    recovered = recover_formatting_glyphs(raw, "SymbolMT")

    assert recovered == "• electricity consumption in TWh"
    # A documented formatting glyph is not corruption whether or not its font
    # was recognised for the mapping, so it withholds no readable passage.
    assert text_corruption_reasons(raw) == []
    assert passage_health_reasons(raw, maximum_unclean_percent=1.0) == []
    assert passage_health_reasons(recovered, maximum_unclean_percent=1.0) == []
    assert chunk_health_flags(raw) == 0
    assert chunk_health_flags(recovered) == 0


@pytest.mark.parametrize(
    ("code", "character"),
    [
        ("\uf0b7", "•"),
        ("\uf0b4", "×"),
        ("\uf0b0", "°"),
        ("\uf0a3", "≤"),
        ("\uf0bc", "…"),
        ("\uf0c9", "⊃"),
    ],
)
def test_each_known_symbol_glyph_is_recovered(code: str, character: str) -> None:
    assert recover_formatting_glyphs(code, "SymbolMT") == character
    assert is_known_formatting_glyph(code)
    assert text_corruption_reasons(f"readable text {code}") == []


def test_adobe_symbol_extender_pieces_are_formatting_not_corruption() -> None:
    # CMEX10 exposes tall-delimiter pieces as Adobe's Corporate Use Subarea code
    # points. Unicode has no character for one, so none is invented: the passage
    # stays readable and a recognised face drops the pieces.
    for code in ("\uf8ed", "\uf8fb", "\uf8eb"):
        assert is_known_formatting_glyph(code)
        assert text_corruption_reasons(f"= σ2 {code}") == []

    assert recover_formatting_glyphs("σ2\uf8ed", "CMEX10") == "σ2 "


def test_an_icon_font_glyph_is_removed_without_guessing_text() -> None:
    assert recover_formatting_glyphs("\uf002Search", "FontAwesome") == " Search"
    assert recover_formatting_glyphs("\uf0daBook Series", "FontAwesome6Free-Solid") == (
        " Book Series"
    )


def test_ordinary_symbols_are_neither_mapped_nor_corrupt() -> None:
    for text in ("… — – • § † ‡", "∑ × ÷ ≈ → ∞", "🙂 🚀"):
        assert recover_formatting_glyphs(text, "SymbolMT") == text
        assert text_health_reasons(text) == ["symbol_only"]


def test_an_unknown_private_use_glyph_remains_corrupt() -> None:
    assert not is_known_formatting_glyph("\uf16d")
    assert not is_known_formatting_glyph("\ue000")
    # An unrecognised face keeps its glyphs; they are judged, never trusted.
    assert recover_formatting_glyphs("\uf16d", "university-press-fonts") == "\uf16d"
    assert text_corruption_reasons("prose \uf16d\uf09a more prose") == [
        "private_or_unassigned_characters"
    ]


def test_a_known_glyph_beside_replacement_corruption_is_still_corrupt() -> None:
    reasons = text_corruption_reasons("\uf0b7 readable � � text")

    assert "replacement_characters" in reasons
    assert "private_or_unassigned_characters" not in reasons


def test_a_punctuation_only_passage_is_non_evidence_not_corrupt() -> None:
    assert passage_health_reasons("… — ∑ ×", maximum_unclean_percent=1.0) == [
        "symbol_only"
    ]
    flags = chunk_health_flags("… — ∑ ×")
    assert flags & CHUNK_FLAG_CORRUPT_TEXT == 0
    assert flags & CHUNK_FLAG_EXTRACTION_ARTIFACT
    assert chunk_health_flags("\uf0b7") & CHUNK_FLAG_CORRUPT_TEXT == 0
