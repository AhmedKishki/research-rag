"""The extraction health gate: what is cleaned, what is withheld, what is refused.

Every case states its geometry or its markup and then checks one thing. The
synthetic sources are built here rather than shipped as fixtures so a reader can
see the placement, the size, and the class that decides each verdict.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf
import pytest
from ebooklib import epub

from research_rag.core.service import ResearchService
from research_rag.corpus.extraction import (
    REMOVAL_FIELDS,
    ExtractionError,
    empty_removal_counts,
    extract_sources,
    merge_removal_counts,
    screen_source_units,
    strip_non_argument_units,
)
from research_rag.corpus.sources import SourceFile, scan_sources
from research_rag.corpus.text_quality import (
    SOURCE_REASON_ALMOST_ALL_WITHHELD,
    SOURCE_REASON_ARTIFACT_LADEN,
    SOURCE_REASON_NO_LETTER_TEXT,
    SOURCE_REASON_NO_TEXT,
    SOURCE_REASON_NO_TEXT_LAYER,
    SOURCE_REASON_UNCLEAN,
    SOURCE_REASON_UNSAFE_TO_CLEAN,
    is_reference_entry,
    non_argument_removal_flags,
    source_health_reasons,
    strip_non_argument_sections,
)
from research_rag.project.config import resolve_config
from tests.conftest import write_epub, write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG

# A prose paragraph long enough that a page's own text, rather than a title line,
# establishes the body font size the footnote rule measures against.
BODY_PROSE = (
    "The extraction of a page is a layout decision before it is a reading one. "
    "A block that sits beside the body column is set narrow and set apart, and "
    "that geometry is what separates it from a paragraph without consulting a "
    "single word of what it says. A note apparatus is set below the body in a "
    "smaller size, and the same is true of both halves at once."
)
# A lead paragraph set larger than the prose it introduces. It is what a journal,
# a deck, and a standfirst all are, and a page that carries one above its body is
# the case where a median body size reads the body itself as a note.
LEAD_PARAGRAPH = (
    "This chapter follows the road from the archive to the market and argues "
    "that every account of the harvest is also an account of the labour that "
    "picked it, and that the council minutes recorded one of the two while the "
    "carts on the quay recorded the other for rather longer than that."
)
# A data table and a block quote are both set small and both sit low on a page,
# which is the shape of a note apparatus. Neither opens with a marker, so neither
# is one.
DATA_TABLE = "Year | Harvest | Wage | Strike 1919 | 4200 | 3.10 | 41 1920 | 5100 | 3.40"
BLOCK_QUOTE = (
    "Cobalt, they wrote, is the only metal that keeps a memory of the heat that "
    "made it, and the assay office in this town read that memory twice a year."
)
REFERENCES_HEADING = "References"
SYMBOLS_ONLY = "… — ∑ × 🙂"
CORRUPT_PAGE = "��ѪҶޜഝǄ䘉Ӌਁ corrupted ൠ؞༽൷㜭ᡀ࣏"
ENTRY_ONE = (
    "Smith, J. (2019). A survey of the field. Journal of Studies, 12(3), pp. 14-29."
)
ENTRY_TWO = "Jones, A. (2020). Another account. Review, 4(2), pp. 5-11."
# A running head that repeats on every page of a book, which is what the repeated
# margin rule is for.
RUNNING_HEAD = (
    "The Harvest Accounts of the Lower Town, volume four, edited from the "
    "council minute books and the customs returns of the assay office, printed "
    "for the historical society and sold at the shop beside the quay."
)
# Legitimate prose outside the Latin script, with the vowel points and matras each
# of these scripts writes with.
HEBREW = "הַאַרְכִיב יִסְדֹּ אֶת הַקִּיאָה וְהַדֶּרֶךְ הֵחִילָה לְפִי הַמַּחְקֵר אֶת שְׁנוֹת הַמִּלְחִימָה וְאֶת צִמְחֵי הָעִיר בְּעִיר."
ARABIC = (
    "وَأَشَارَ الْأَرْشِيفُ إِلَى أَنَّ الطَّرِيقَ هُوَ الَّذِي قَرَّرَ قِصَّةَ "
    "الْمَدِينَةِ فِي السَّنَوَاتِ الْأَخِيرَةِ وَأَنَّ الْعَمَّالَ كَتَبُوهَا."
)
DEVANAGARI = (
    "यह अध्याय विवेचन करता है कि स्मृति नगर के रास्तों में कैसे दर्ज होती है "
    "और वहाँ फिर से क्यों लौटती है, और यह प्रश्न पिछले तीन दशकों के शोध से अलग नहीं।"
)


def _paragraphs(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


def _removed(text: str) -> tuple[str, tuple[str, ...]]:
    """The paragraph-level answer, joined back the way one unit holds its text.

    `extraction.strip_non_argument_units` makes the same decision over a whole
    source's units and is what both build paths call; the tests below use the
    paragraph form so a case can state its own paragraphs, and one of them holds
    the two to the same answer.
    """

    kept, removed = strip_non_argument_sections(text.split("\n\n"))
    return "\n\n".join(kept), removed


def _pdf(
    path: Path,
    pages: list[list[tuple[tuple[float, ...], str, float]]],
    *,
    title: str = "Gate PDF",
) -> None:
    """A PDF from per-page placements, so a case can state exact geometry.

    Each page is a list of `(rect, text, fontsize)`, which is what a reader of the
    case sees: where a thing sits and how big it is set.
    """

    document = pymupdf.open()
    document.set_metadata({"title": title, "author": "Gate Author"})
    for blocks in pages:
        page = document.new_page()
        for rect, text, fontsize in blocks:
            page.insert_textbox(pymupdf.Rect(*rect), text, fontsize=fontsize)
    document.save(path)
    document.close()


def _epub(path: Path, sections: dict[str, str]) -> None:
    book = epub.EpubBook()
    book.set_identifier("gate-test")
    book.set_title("Gate Test")
    book.set_language("en")
    spine = ["nav"]
    for index, (href, content) in enumerate(sections.items(), 1):
        item = epub.EpubHtml(title=f"Section {index}", file_name=href, lang="en")
        item.content = content
        book.add_item(item)
        spine.append(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(str(path), book)


def _extracted(project: Path) -> tuple[list[dict], list[dict]]:
    config = resolve_config(project, vanilla_executable=sys.executable)
    return extract_sources(scan_sources(config).selected)


def _combined(units: list[dict]) -> str:
    return "\n".join(str(unit["contents"]) for unit in units)


def _service(project: Path) -> ResearchService:
    config = resolve_config(project, vanilla_executable=sys.executable)
    return ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )


def _named_source(relative_path: str) -> SourceFile:
    return SourceFile(
        path=Path("/tmp/sources") / relative_path,
        source_id=f"src_{relative_path.replace('.', '_')}",
        source_relative_path=relative_path,
        project_relative_path=f"sources/{relative_path}",
        extension=Path(relative_path).suffix,
        size=1,
        mtime_ns=0,
    )


def _unit(index: int, text: str, *, content_kind: str = "prose") -> dict:
    return {
        "id": f"doc:pdf-page:{index:06d}:region:001",
        "contents": text,
        "content_kind": content_kind,
        "locator": {"type": "pdf_page", "page": index, "page_label": str(index)},
    }


# ---------------------------------------------------------------------------
# Semantic cleaning: what comes out, and what must not
# ---------------------------------------------------------------------------


def test_reference_section_is_removed_and_main_text_citations_are_not() -> None:
    text = _paragraphs(
        "The earlier surveys disagreed about the definition (Smith 2019, 14).",
        "This paper keeps the narrower one and says so explicitly.",
        REFERENCES_HEADING,
        (
            "Smith, J. (2019). A survey of the field. Journal of Studies, "
            "12(3), pp. 14-29. https://doi.org/10.1000/example.1"
        ),
        (
            "Jones, A. (2020). Another account of the same material. "
            "Review, 4(2), pp. 5-11."
        ),
    )

    cleaned, removed = _removed(text)

    assert removed == ("bibliography",)
    assert "(Smith 2019, 14)" in cleaned
    assert "This paper keeps the narrower one" in cleaned
    assert "Smith, J. (2019)" not in cleaned
    assert "Jones, A. (2020)" not in cleaned


def test_marker_led_note_apparatus_is_removed_and_the_body_before_it_survives() -> None:
    text = _paragraphs(
        "The account below rests on three sources and says which.",
        "Notes",
        (
            "* The first note quotes a witness: the road was already cut, and "
            "the cutting decided what the road was for."
        ),
        "* The second note gives the archive reference and the access date.",
    )

    cleaned, removed = _removed(text)

    assert removed == ("footnotes",)
    assert cleaned == "The account below rests on three sources and says which."
    assert "archive reference" not in cleaned


def test_a_word_that_names_a_section_is_not_a_section_heading() -> None:
    text = _paragraphs(
        "References to the earlier work show a consistent gap in the record.",
        "The gap is the subject of this chapter.",
        "The next paragraph makes the case in full and must survive intact.",
    )

    assert _removed(text) == (text, ())


def test_a_reference_heading_above_ordinary_prose_is_not_a_reference_list() -> None:
    text = _paragraphs(
        REFERENCES_HEADING,
        (
            "The chapter collects what the archive holds and does not pretend the "
            "collection is complete."
        ),
        "Each account is dated so that a later reader can order them.",
    )

    assert _removed(text) == (text, ())


def test_a_sentence_ending_in_the_heading_word_is_body_text() -> None:
    text = _paragraphs(
        "The references.",
        "This is a sentence that names the word and then stops.",
    )

    assert _removed(text) == (text, ())


def test_a_heading_alone_is_not_a_section_and_the_text_after_it_is_kept() -> None:
    """A heading that opens nothing confirmed removes nothing, itself included."""

    text = _paragraphs(
        REFERENCES_HEADING,
        "The chapter collects what the archive holds, in date order.",
    )

    assert _removed(text) == (text, ())


def test_the_unit_level_removal_agrees_with_the_paragraph_level_one() -> None:
    """Both build paths call the unit form, so it has to reach the same answer.

    A section does not fit in a unit: a heading closes one page and its entries
    run over the next. The unit form is what `screen_source_units` calls, so it is
    the one that decides what an index holds.
    """

    units = [
        _unit(
            1,
            _paragraphs(
                "The chapter argues its case in full.",
                REFERENCES_HEADING,
                ENTRY_ONE,
            ),
        ),
        _unit(
            2,
            _paragraphs(
                ENTRY_TWO,
                "Appendix",
                "The appendix sets out the coding scheme the chapter used.",
            ),
        ),
    ]

    cleaned, counts = strip_non_argument_units(units)

    combined = _combined(cleaned)
    assert "Smith, J. (2019)" not in combined
    assert "Jones, A. (2020)" not in combined
    assert "The chapter argues its case in full." in combined
    assert "The appendix sets out the coding scheme" in combined
    assert counts["removed_reference_segments"] == 3
    assert counts["removed_non_argument_characters"] > 0


def test_a_section_stops_at_the_next_argument_heading() -> None:
    text = _paragraphs(
        "The chapter makes its case and cites the earlier work in passing.",
        REFERENCES_HEADING,
        ENTRY_ONE,
        ENTRY_TWO,
        "Appendix",
        "The appendix sets out the coding scheme the chapter used.",
    )

    cleaned, removed = _removed(text)

    assert removed == ("bibliography",)
    assert "cites the earlier work in passing" in cleaned
    assert ENTRY_ONE not in cleaned
    # The removal ran to the end of the section, not to the end of the unit.
    assert "Appendix" in cleaned
    assert "The appendix sets out the coding scheme" in cleaned


def test_a_section_opens_again_after_the_argument_it_stopped_at() -> None:
    text = _paragraphs(
        REFERENCES_HEADING,
        ENTRY_ONE,
        "Appendix",
        "The appendix sets out the coding scheme the chapter used.",
        REFERENCES_HEADING,
        ENTRY_TWO,
    )

    cleaned, removed = _removed(text)

    assert removed == ("bibliography",)
    assert ENTRY_ONE not in cleaned
    assert ENTRY_TWO not in cleaned
    assert "The appendix sets out the coding scheme" in cleaned


def test_an_unmarked_paragraph_ends_a_note_apparatus() -> None:
    text = _paragraphs(
        "The chapter states its evidence and where it came from.",
        "Notes",
        "* The first note names the archive box the copy came from.",
        "This paragraph carries no marker, so the apparatus ended at the note.",
    )

    cleaned, removed = _removed(text)

    assert removed == ("footnotes",)
    assert "no marker, so the apparatus ended" in cleaned


def test_a_full_name_entry_with_only_a_trailing_year_is_kept() -> None:
    """An entry shape with no marker in it is prose as far as this gate knows.

    A humanities list prints the author's given names out and ends the entry with a
    bare year, and there is nothing in that a running paragraph could not also
    print. Reading it as an entry would end the section at the first line of the
    argument a reader wanted, so it is kept. Under-removing a list leaves text in
    the index; over-removing one loses the document's own words.
    """

    text = _paragraphs(
        "The chapter cites the older scholarship in its own terms.",
        REFERENCES_HEADING,
        "Nussbaum, Martha. The Poetics of Grief. Ithaca, 1986.",
        "Tadiar, Hevia. Reading Rubber Economies. Durham, 2009.",
    )

    assert _removed(text) == (text, ())


def test_a_narrative_citation_is_not_a_reference_entry() -> None:
    """A citation in running prose carries a name and a year, and is not a list.

    Each of these is a paragraph of an argument that happens to open with a word
    and a comma or to carry a bracketed year. A rule that read either of those as
    an entry would remove the paragraph, and a paragraph inside a confirmed list
    is the paragraph that ends the section and saves the argument after it.
    """

    prose = (
        "However, the committee rejected the plan in March and asked for figures.",
        "Nevertheless, the assay book and the customs return agree about the wages.",
        "Later, the council asked whether the road or the market came first.",
        "See Smith (2019) for the survey, and Jones (2020) for the second count.",
        "Jones, A. (2020) argues that the road came first, and the record agrees.",
        "The earlier surveys disagreed about the definition (Smith 2019, 14).",
    )

    assert not [text for text in prose if is_reference_entry(text)]


def test_an_entry_is_recognised_by_its_author_or_by_its_locator() -> None:
    """The two openings a list actually prints, in both name orders."""

    entries = (
        ENTRY_ONE,
        ENTRY_TWO,
        "J. A. Smith (2019). A survey of the field. Journal of Studies, 12(3).",
        "Smith, J. A. A survey of the field. Journal of Studies, pp. 14-29.",
        "https://doi.org/10.1000/example.1",
    )

    assert not [text for text in entries if not is_reference_entry(text)]


def test_prose_between_two_entries_survives_and_ends_the_list() -> None:
    """An argument in the middle of a list is where a run of entries resumes.

    The paragraph is not an entry, so it ends the section: the heading, the entry
    above it, and nothing after it go. A rule that kept removing until the next
    heading took the argument and the rest of the list with it.
    """

    sequence = [
        "The chapter argues its case in the sections that follow.",
        REFERENCES_HEADING,
        ENTRY_ONE,
        "However, the committee rejected the plan in March and asked for figures.",
        ENTRY_TWO,
    ]

    flags = non_argument_removal_flags(sequence)

    assert flags == [None, "bibliography", "bibliography", None, None]
    kept, removed = strip_non_argument_sections(sequence)
    assert kept == [sequence[0], sequence[3], sequence[4]]
    assert removed == ("bibliography",)


def test_a_references_heading_above_prose_that_cites_removes_nothing() -> None:
    """A heading confirms nothing until a paragraph of the section's shape does.

    The prose after this heading cites, which is the one thing every argument
    paragraph can do, so it is not evidence of a list. Nothing is removed, and the
    heading with it, because a heading a document kept above its own prose is
    ordinary structure.
    """

    text = _paragraphs(
        REFERENCES_HEADING,
        (
            "The council resolved the question in 2019 (Smith 2019), and the "
            "customs return for that year agrees with the minute book."
        ),
        "The next paragraph says what the resolution cost the assay office.",
    )

    assert _removed(text) == (text, ())


def test_foreign_language_argument_survives_every_rule() -> None:
    text = _paragraphs(
        (
            "Die Untersuchung folgt der Frage, wie das Gedächtnis einer Stadt "
            "sich in ihren Straßen niederschlägt und dort wiederkehrt."
        ),
        "Обзор рассматривает материал как связь, а не как набор фактов.",
        "شريط الحاشية السفلي يشرح المصطلح المستخدم في النص الأصلي.",
    )

    assert _removed(text) == (text, ())


def test_formulas_and_numbered_argument_are_not_a_note_apparatus() -> None:
    text = _paragraphs(
        "The estimator satisfies E[X] = μ + σ²/μ for the lognormal case.",
        "1 The identity holds only where μ is strictly positive.",
        "2 A second step in the derivation follows the same substitution.",
        "3 The bound in the third line is what the appendix tightens.",
    )

    # No heading opens this block, so the marker-led paragraphs stay: a numbered
    # argument is body text, and only a heading makes a marked run an apparatus.
    assert _removed(text) == (text, ())


def test_a_quotation_within_body_text_is_not_read_as_an_apparatus() -> None:
    text = _paragraphs(
        "The witnesses agreed on the road and disagreed about its use.",
        "* Not every source is a note, and a marked paragraph is still a paragraph.",
        "* The second marked paragraph makes the same point in other words.",
    )

    assert _removed(text) == (text, ())


def test_cleaning_leaves_an_ordinary_page_unchanged() -> None:
    text = _paragraphs(
        "The first paragraph states the question the chapter answers.",
        "The second paragraph answers it in three sentences.",
        "The third paragraph says what would change the answer.",
    )

    assert _removed(text) == (text, ())


# ---------------------------------------------------------------------------
# The source verdict, from its own numbers
# ---------------------------------------------------------------------------


def test_no_readable_text_is_the_only_verdict_a_completely_dead_file_gets() -> None:
    assert source_health_reasons(
        unit_count=0,
        retained_count=0,
        withheld_reasons={},
    ) == [SOURCE_REASON_NO_TEXT]
    assert source_health_reasons(
        unit_count=3,
        retained_count=0,
        withheld_reasons={"replacement_characters": 3},
    ) == [SOURCE_REASON_NO_TEXT]


def test_one_bad_page_does_not_refuse_the_file() -> None:
    reasons = source_health_reasons(
        unit_count=20,
        retained_count=19,
        withheld_reasons={"replacement_characters": 1},
        kept_characters=20000,
    )

    assert reasons == []


def test_every_page_a_scan_is_refused_for_the_missing_text_layer() -> None:
    reasons = source_health_reasons(
        unit_count=12,
        retained_count=12,
        withheld_reasons={},
        image_only_pages=12,
        physical_pages=12,
    )

    assert reasons == [SOURCE_REASON_NO_TEXT_LAYER]


def test_a_fully_scanned_file_with_nothing_readable_names_the_missing_layer() -> None:
    assert source_health_reasons(
        unit_count=0,
        retained_count=0,
        withheld_reasons={},
        image_only_pages=8,
        physical_pages=8,
    ) == [SOURCE_REASON_NO_TEXT_LAYER]


def test_some_scanned_pages_do_not_refuse_a_file_that_can_be_read() -> None:
    reasons = source_health_reasons(
        unit_count=10,
        retained_count=10,
        withheld_reasons={},
        kept_characters=40000,
        image_only_pages=2,
        physical_pages=12,
    )

    assert reasons == []


def test_a_share_of_digits_is_not_evidence_of_a_scan() -> None:
    """The refusal must come from the pages, not from a character ratio.

    A page of tables, a concordance, and a formula set carries mostly digits and
    punctuation while being entirely readable, so a ratio that fired on that would
    refuse readable sources.
    """

    reasons = source_health_reasons(
        unit_count=40,
        retained_count=40,
        withheld_reasons={},
        kept_characters=20000,
        image_only_pages=0,
        physical_pages=40,
    )

    assert reasons == []


def test_a_mostly_unreadable_file_is_refused_for_what_it_lost() -> None:
    reasons = source_health_reasons(
        unit_count=20,
        retained_count=1,
        withheld_reasons={"symbol_only": 19},
        kept_characters=2000,
    )

    assert reasons == [
        SOURCE_REASON_ALMOST_ALL_WITHHELD,
        SOURCE_REASON_ARTIFACT_LADEN,
    ]


def test_a_file_whose_cleaning_removed_more_than_it_kept_is_refused() -> None:
    reasons = source_health_reasons(
        unit_count=4,
        retained_count=4,
        withheld_reasons={},
        kept_characters=900,
        removed_characters=1400,
    )

    assert reasons == [SOURCE_REASON_UNSAFE_TO_CLEAN]


def test_a_reference_heavy_work_is_kept_because_nothing_was_wrong_with_it() -> None:
    reasons = source_health_reasons(
        unit_count=4,
        retained_count=4,
        withheld_reasons={},
        kept_characters=1400,
        removed_characters=700,
    )

    assert reasons == []


def test_cleaning_too_little_removed_says_nothing() -> None:
    reasons = source_health_reasons(
        unit_count=40,
        retained_count=40,
        withheld_reasons={},
        kept_characters=400000,
        removed_characters=100,
    )

    assert reasons == []


def test_a_file_whose_retained_text_is_only_numbers_is_refused() -> None:
    """Digits, table rules, and layout marks are not a document.

    A glyph soup the extractor scraped off a drawing comes out this way: nothing to
    read and no reason in it that says which page it came from. Only a measured
    zero counts, so a caller that has not counted letters makes no claim here.
    """

    assert source_health_reasons(
        unit_count=3,
        retained_count=3,
        withheld_reasons={},
        kept_characters=400,
        letter_characters=0,
    ) == [SOURCE_REASON_NO_LETTER_TEXT]
    assert (
        source_health_reasons(
            unit_count=3,
            retained_count=3,
            withheld_reasons={},
            kept_characters=400,
        )
        == []
    )


def test_a_numeric_table_beside_a_labelled_body_is_not_refused() -> None:
    """The verdict is about the whole file, so one labelled page is enough.

    A statistical table set with digits is a document, and the heading that names
    it is the words a reader searches for. Refusing it would refuse the table of
    every paper that has one.
    """

    assert (
        source_health_reasons(
            unit_count=12,
            retained_count=12,
            withheld_reasons={},
            kept_characters=4000,
            letter_characters=240,
        )
        == []
    )


# ---------------------------------------------------------------------------
# The gate itself: what it records and what it says
# ---------------------------------------------------------------------------


def test_the_gate_records_every_exclusion_with_its_locator_and_reason() -> None:
    document: dict = {"title": "Recording"}
    units = [
        _unit(1, "A page that reads as ordinary argument about labour and time."),
        _unit(2, CORRUPT_PAGE),
        _unit(3, SYMBOLS_ONLY),
    ]

    retained = screen_source_units(_named_source("recording.pdf"), document, units)

    assert [unit["contents"] for unit in retained] == [
        "A page that reads as ordinary argument about labour and time."
    ]
    assert document["excluded_corrupt_unit_count"] == 2
    assert document["extracted_units"] == 1
    assert [
        entry["locator"]["page"] for entry in document["excluded_corrupt_units"]
    ] == [
        2,
        3,
    ]
    assert document["excluded_corrupt_units"][1]["reasons"] == ["symbol_only"]
    assert "corrupt_extraction_units_excluded" in document["metadata_warnings"]


def test_the_gate_keeps_a_source_it_can_read_and_records_no_warning() -> None:
    document: dict = {"title": "Readable"}
    units = [_unit(1, "An ordinary page of research evidence about labour.")]

    retained = screen_source_units(_named_source("readable.pdf"), document, units)

    assert len(retained) == 1
    assert document["excluded_corrupt_unit_count"] == 0
    assert document["excluded_corrupt_units"] == []
    assert "metadata_warnings" not in document


def test_the_gate_names_a_refused_file_its_own_locator_and_a_remedy() -> None:
    with pytest.raises(ExtractionError) as failure:
        screen_source_units(
            _named_source("scan.pdf"),
            {"title": "Scan"},
            [_unit(7, SYMBOLS_ONLY)],
        )

    message = str(failure.value)
    assert "no readable English-oriented text" in message
    assert "First excluded unit at page 7" in message
    assert "symbol_only=1" in message
    assert "research-rag exclude scan.pdf" in message
    assert SYMBOLS_ONLY not in message


def test_a_source_losing_more_than_the_accepted_share_is_refused_as_not_clean() -> None:
    """The share is of characters, measured before anything is chunked.

    One page of garbage in a long paper is a hole the unit rule already handles;
    a project that states how much loss it accepts refuses the file past that.
    The message says a loss is expected and names OCR, which is never run here.
    """

    ordinary = "A page that reads as ordinary argument about labour and time. " * 12
    units = [_unit(1, ordinary), _unit(2, ordinary), _unit(3, CORRUPT_PAGE)]

    with pytest.raises(ExtractionError) as failure:
        screen_source_units(
            _named_source("poor-layer.pdf"),
            {"title": "Poor layer"},
            units,
            maximum_unclean_percent=1.0,
        )

    message = str(failure.value)
    assert "Source is not clean (unclean_text)" in message
    assert "indexing it would lose that text" in message
    assert "research-rag ocr poor-layer.pdf" in message
    assert "never run automatically" in message
    assert "research-rag exclude poor-layer.pdf" in message
    assert CORRUPT_PAGE not in message


def test_a_source_within_the_accepted_share_is_kept_and_its_rate_is_recorded() -> None:
    ordinary = "A page that reads as ordinary argument about labour and time. " * 400
    document: dict = {"title": "Mostly clean"}
    units = [_unit(1, ordinary), _unit(2, CORRUPT_PAGE)]

    retained = screen_source_units(
        _named_source("mostly-clean.pdf"),
        document,
        units,
        maximum_unclean_percent=5.0,
    )
    unlimited = screen_source_units(
        _named_source("any.pdf"), {"title": "Any"}, units, maximum_unclean_percent=None
    )

    assert len(retained) == len(unlimited) == 1
    assert 0 < document["unclean_character_rate"] < 0.05


def test_the_source_gate_states_no_share_unless_asked() -> None:
    assert SOURCE_REASON_UNCLEAN not in source_health_reasons(
        unit_count=10,
        retained_count=9,
        withheld_reasons=Counter(),
        kept_characters=1000,
        withheld_characters=500,
    )
    assert SOURCE_REASON_UNCLEAN not in source_health_reasons(
        unit_count=10,
        retained_count=9,
        withheld_reasons=Counter(),
        kept_characters=1000,
        withheld_characters=10,
        maximum_unclean_percent=1.0,
    )
    assert SOURCE_REASON_UNCLEAN in source_health_reasons(
        unit_count=10,
        retained_count=9,
        withheld_reasons=Counter(),
        kept_characters=1000,
        withheld_characters=11,
        maximum_unclean_percent=1.0,
    )


def test_the_gate_names_an_epub_refusal_by_section_and_file() -> None:
    source = _named_source("broken.epub")
    unit = {
        "id": "doc:epub-section:000002:block:000001",
        "contents": CORRUPT_PAGE,
        "content_kind": "prose",
        "locator": {
            "type": "epub_section",
            "section_index": 2,
            "section_title": "Broken",
            "href": "broken.xhtml",
            "element_path": "/html[1]/body[1]/p[1]",
            "block_index": 1,
        },
    }

    with pytest.raises(ExtractionError) as failure:
        screen_source_units(source, {"title": "Broken"}, [unit])

    message = str(failure.value)
    assert "First excluded unit at section 2 (broken.xhtml)" in message
    assert "replacement_characters" in message
    assert CORRUPT_PAGE not in message


def test_the_gate_refuses_a_source_whose_cleaning_took_more_than_it_kept() -> None:
    with pytest.raises(ExtractionError, match="unsafe_to_clean"):
        screen_source_units(
            _named_source("mostly.pdf"),
            {"title": "Mostly"},
            [_unit(1, "A page of argument that survived the removal rules intact.")],
            removals={"removed_non_argument_characters": 4000},
        )


def test_the_gate_records_the_removal_counts_the_extractor_reported() -> None:
    document: dict = {"title": "Counted"}

    screen_source_units(
        _named_source("counted.pdf"),
        document,
        [_unit(1, "An ordinary page of research evidence about labour.")],
        removals={"removed_sidebar_blocks": 3, "removed_footnote_blocks": 2},
    )

    assert document["removed_sidebar_blocks"] == 3
    assert document["removed_footnote_blocks"] == 2


# ---------------------------------------------------------------------------
# Geometry: sidebars and footnotes, and the furniture they must not be confused with
# ---------------------------------------------------------------------------


def test_a_sidebar_beside_the_body_column_is_not_indexed(project: Path) -> None:
    _pdf(
        project / "sources" / "sidebar.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((24, 100, 58, 300), "A marginal note about the margin rule.", 8.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "A marginal note about the margin rule." not in _combined(units)
    assert "layout decision before it is a reading one" in _combined(units)
    assert documents[0]["removed_sidebar_blocks"] == 1


def test_a_sidebar_to_the_right_of_the_body_column_is_not_indexed(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "sidebar-right.pdf",
        [
            [
                ((72, 100, 480, 300), BODY_PROSE, 10.0),
                (
                    (494, 100, 528, 300),
                    "A marginal note to the right of the body.",
                    8.0,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "A marginal note to the right" not in _combined(units)
    assert documents[0]["removed_sidebar_blocks"] == 1


def test_a_footnote_set_in_the_foot_of_the_page_below_the_body_is_not_indexed(
    project: Path,
) -> None:
    """A note takes all four conditions: smaller, low, separated, and marked."""

    _pdf(
        project / "sources" / "footnote.pdf",
        [
            [
                ((72, 100, 540, 320), BODY_PROSE, 10.0),
                (
                    (72, 700, 540, 760),
                    "1 The note explains the definition used in the table above.",
                    7.0,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "The note explains the definition" not in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 1
    assert documents[0]["removed_non_argument_characters"] > 0


def test_an_unmarked_note_in_the_foot_of_the_page_is_kept(project: Path) -> None:
    """A note printed without its marker is left in, which is the safe direction.

    A note apparatus is numbered, and the marker is the only evidence a page
    carries that a small block at the foot is a note rather than evidence. A
    publisher that drops the numbers loses the removal, not the text.
    """

    _pdf(
        project / "sources" / "unmarked-note.pdf",
        [
            [
                ((72, 100, 540, 320), BODY_PROSE, 10.0),
                (
                    (72, 700, 540, 760),
                    "The archive copy of the minute book names the clerk who wrote it.",
                    7.0,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "The archive copy of the minute book" in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 0


def test_a_lead_paragraph_set_larger_than_the_body_is_not_a_note(
    project: Path,
) -> None:
    """A deck above the argument must not make the argument a note.

    A lead, a deck, and a standfirst are all set larger than the prose they
    introduce. A body size read as the middle of the page's sizes then finds every
    body paragraph below it, at the foot of the page or not, and the page's whole
    argument goes with its first paragraph.
    """

    _pdf(
        project / "sources" / "lead.pdf",
        [
            [
                ((72, 60, 540, 175), LEAD_PARAGRAPH, 13.0),
                ((72, 200, 540, 300), BODY_PROSE, 10.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert "the labour that picked it" in combined
    assert "layout decision before it is a reading one" in combined
    assert documents[0]["removed_footnote_blocks"] == 0
    assert documents[0]["removed_non_argument_characters"] == 0


def test_a_small_font_table_in_the_foot_of_the_page_is_not_a_note(
    project: Path,
) -> None:
    """A data table is set small and sits low, and is evidence either way."""

    _pdf(
        project / "sources" / "small-table.pdf",
        [
            [
                ((72, 100, 540, 320), BODY_PROSE, 10.0),
                ((90, 700, 520, 800), DATA_TABLE, 8.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "Year | Harvest | Wage" in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 0


def test_a_small_font_block_quote_in_the_foot_of_the_page_is_not_a_note(
    project: Path,
) -> None:
    """A block quote is set small, is indented, and sits low. None of that is a note."""

    _pdf(
        project / "sources" / "small-quote.pdf",
        [
            [
                ((72, 100, 540, 320), BODY_PROSE, 10.0),
                ((90, 700, 520, 800), BLOCK_QUOTE, 8.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "keeps a memory of the heat" in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 0


def test_body_text_in_the_lower_half_of_a_page_is_not_a_footnote(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "low-body.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                (
                    (72, 500, 540, 640),
                    (
                        "A second paragraph of the same body size that a page "
                        "carries low down because the argument continues there."
                    ),
                    10.0,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "the argument continues there" in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 0


def test_a_numbered_list_in_the_foot_of_the_page_is_not_a_note(
    project: Path,
) -> None:
    """A list carries the marker a note carries, and is set in the body size.

    The marker alone would take it. A page sets a numbered list in the body size
    and does not separate it from the body, so the other three conditions are what
    hold it.
    """

    _pdf(
        project / "sources" / "low-list.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((72, 690, 540, 730), "1. The first claim the chapter defends.", 10.0),
                ((72, 730, 540, 760), "2. The second claim follows from it.", 10.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert "The first claim the chapter defends" in combined
    assert "The second claim follows from it" in combined
    assert documents[0]["removed_footnote_blocks"] == 0


def _epub_spine(path: Path, sections: list[str], *, title: str = "Gate Book") -> None:
    """A multi-chapter EPUB, one file per chapter, in spine order.

    A book is the case that matters for section removal: its reference list is
    normally in the last chapter, split across several sections, and its chapters
    are separate spine items rather than one string.
    """

    book = epub.EpubBook()
    book.set_identifier("gate-book")
    book.set_title(title)
    book.set_language("en")
    spine = ["nav"]
    for index, content in enumerate(sections, 1):
        href = f"chapter{index}.xhtml"
        item = epub.EpubHtml(title=f"Chapter {index}", file_name=href, lang="en")
        item.content = content
        book.add_item(item)
        spine.append(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(str(path), book)


def test_an_epub_heading_and_its_separately_emitted_entries_are_removed(
    project: Path,
) -> None:
    """A publisher emits the heading and each entry as its own element.

    No single unit ever holds a heading and its entries together, so a rule that
    reads one unit finds nothing. This is the ordinary case, not an edge case.
    """

    _epub(
        project / "sources" / "split.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The chapter argues that the archive decides the reading.</p>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                f"<p>{ENTRY_ONE}</p>"
                f"<p>{ENTRY_TWO}</p>"
            )
        },
    )

    documents, units = _extracted(project)

    assert "Smith, J. (2019). A survey" not in _combined(units)
    assert "Jones, A. (2020). Another" not in _combined(units)
    assert "the archive decides the reading" in _combined(units)
    # The heading and both entries: three paragraphs went.
    assert documents[0]["removed_reference_segments"] == 3


def test_an_epub_reference_list_split_across_spine_items_is_removed(
    project: Path,
) -> None:
    _epub(
        project / "sources" / "split-spine.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The chapter states its argument and names its sources.</p>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                f"<p>{ENTRY_ONE}</p>"
            ),
            "back.xhtml": f"<p>{ENTRY_TWO}</p>",
        },
    )

    documents, units = _extracted(project)

    assert "Smith, J. (2019). A survey" not in _combined(units)
    assert "Jones, A. (2020). Another" not in _combined(units)
    assert "The chapter states its argument" in _combined(units)
    assert documents[0]["removed_reference_segments"] == 3


def test_a_multi_chapter_epub_keeps_every_chapter_and_removes_only_the_list(
    project: Path,
) -> None:
    _epub_spine(
        project / "sources" / "book.epub",
        [
            "<h1>One</h1><p>The first chapter makes its case in full.</p>",
            "<h1>Two</h1><p>The second chapter answers the objection it raised.</p>",
            (
                "<h1>Three</h1>"
                f"<p>The third chapter draws the two arguments together.</p>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                f"<p>{ENTRY_ONE}</p>"
                f"<p>{ENTRY_TWO}</p>"
            ),
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert "The first chapter makes its case" in combined
    assert "The second chapter answers the objection" in combined
    assert "The third chapter draws the two" in combined
    assert "Smith, J. (2019). A survey" not in combined
    assert documents[0]["removed_reference_segments"] == 3


def test_a_section_ends_at_the_next_chapter_of_a_book(project: Path) -> None:
    _epub_spine(
        project / "sources" / "resumes.epub",
        [
            (
                "<h1>One</h1>"
                "<p>The first chapter collects what the archive holds.</p>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                f"<p>{ENTRY_ONE}</p>"
                f"<p>{ENTRY_TWO}</p>"
            ),
            "<h1>Two</h1><p>The second chapter opens a new argument entirely.</p>",
        ],
    )

    _documents, units = _extracted(project)

    combined = _combined(units)
    assert "Smith, J. (2019). A survey" not in combined
    assert "The second chapter opens a new argument" in combined


def test_an_ambiguous_references_heading_above_prose_is_kept(project: Path) -> None:
    """A heading that opens no list is ordinary structure and must survive.

    The paragraphs after it are prose, so nothing confirms the section, and the
    heading has to stay too. A rule that removed the heading anyway would lose a
    chapter's own structure on a page that has no bibliography at all.
    """

    _epub(
        project / "sources" / "ambiguous.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                "<p>The chapter collects what the archive holds, in date order, "
                "and says plainly what the collection cannot settle.</p>"
            )
        },
    )

    _documents, units = _extracted(project)

    combined = _combined(units)
    assert REFERENCES_HEADING in combined
    assert "what the collection cannot settle" in combined


def test_a_pdf_reference_list_running_over_several_pages_is_removed(
    project: Path,
) -> None:
    """The heading closes the last page of text, the entries follow it.

    This is how a book actually sets a list: the heading is the last thing on the
    last page of the chapter, and the entries run on. Nothing after the heading
    carries argument text, so nothing closes the section early.
    """

    _pdf(
        project / "sources" / "book.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((72, 340, 540, 380), REFERENCES_HEADING, 11.0),
            ],
            [
                ((72, 100, 540, 200), ENTRY_ONE, 10.0),
                ((72, 330, 540, 400), ENTRY_TWO, 10.0),
            ],
            [
                ((72, 100, 540, 300), BODY_PROSE.replace("layout", "history"), 10.0),
                ((72, 340, 540, 440), "Appendix", 11.0),
                (
                    (72, 470, 540, 570),
                    "The appendix sets out the coding scheme the chapter used.",
                    10.0,
                ),
            ],
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert REFERENCES_HEADING not in combined
    assert "Smith, J. (2019). A survey" not in combined
    assert "Jones, A. (2020). Another" not in combined
    # The argument on the pages before and after the list is untouched.
    assert "layout decision before it is a reading one" in combined
    assert "history decision before it is a reading one" in combined
    assert "The appendix sets out the coding scheme" in combined
    assert documents[0]["removed_reference_segments"] == 3


def test_a_pdf_note_apparatus_running_onto_the_next_page_is_removed(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "notes.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((72, 340, 540, 380), "Notes", 11.0),
            ],
            [
                (
                    (72, 100, 540, 200),
                    "* The first note names the archive box the copy came from.",
                    10.0,
                ),
                (
                    (72, 230, 540, 330),
                    "* The second note records the access date of the same copy.",
                    10.0,
                ),
            ],
            [
                (
                    (72, 100, 540, 300),
                    (
                        "The next chapter opens a new argument entirely and must "
                        "be kept in full, notes or no notes."
                    ),
                    10.0,
                )
            ],
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert "The first note names the archive box" not in combined
    assert "The second note records the access date" not in combined
    assert "The next chapter opens a new argument" in combined
    # The heading and both notes.
    assert documents[0]["removed_note_segments"] == 3


def test_a_heading_followed_by_argument_closes_the_section_before_the_list(
    project: Path,
) -> None:
    """A heading with a page of prose after it removes nothing.

    A section is only removed once a paragraph of its own shape confirms it, so a
    heading that a chapter keeps above its own argument leaves both in place. This
    is the case a per-page rule gets wrong in the other direction: it would see the
    heading and assume a list.
    """

    _pdf(
        project / "sources" / "ambiguous-order.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((72, 340, 540, 380), REFERENCES_HEADING, 11.0),
            ],
            [
                (
                    (72, 100, 540, 300),
                    BODY_PROSE.replace("layout", "argument") + " " + ENTRY_ONE,
                    10.0,
                )
            ],
        ],
    )

    documents, units = _extracted(project)

    assert "Smith, J. (2019). A survey" in _combined(units)
    assert documents[0]["removed_reference_segments"] == 0


def test_a_pdf_page_of_only_entries_contributes_no_unit(project: Path) -> None:
    _pdf(
        project / "sources" / "entries-only.pdf",
        [
            [
                ((72, 100, 540, 200), BODY_PROSE, 10.0),
                ((72, 340, 540, 380), REFERENCES_HEADING, 11.0),
            ],
            [((72, 100, 540, 300), _paragraphs(ENTRY_ONE, ENTRY_TWO), 10.0)],
        ],
    )

    documents, units = _extracted(project)

    assert all("Smith, J. (2019)" not in str(unit["contents"]) for unit in units)
    assert all("Jones, A. (2020)" not in str(unit["contents"]) for unit in units)
    assert any("layout decision" in str(unit["contents"]) for unit in units)
    # The page held nothing but entries, so it yields no unit at all.
    assert documents[0]["removed_reference_segments"] == 3


def test_an_ambiguous_pdf_references_heading_above_prose_is_kept(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "ambiguous.pdf",
        [
            [
                ((72, 100, 540, 200), BODY_PROSE, 10.0),
                ((72, 340, 540, 380), REFERENCES_HEADING, 11.0),
                (
                    (72, 400, 540, 460),
                    (
                        "A paragraph of the chapter's own argument, set under the "
                        "heading and not at all shaped like a citation."
                    ),
                    10.0,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert REFERENCES_HEADING in combined
    assert "not at all shaped like a citation" in combined
    assert documents[0]["removed_reference_segments"] == 0


def test_a_pdf_figure_caption_in_the_foot_of_the_page_is_not_a_footnote(
    project: Path,
) -> None:
    """A caption is set small, sits low, and is evidence about the figure."""

    _pdf(
        project / "sources" / "caption.pdf",
        [
            [
                ((72, 100, 540, 320), BODY_PROSE, 10.0),
                (
                    (72, 700, 540, 760),
                    "Figure 3. The extractor before and after the cleanup rules.",
                    7.5,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "Figure 3." in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 0


def test_a_uniformly_small_page_keeps_every_block(project: Path) -> None:
    _pdf(
        project / "sources" / "uniform.pdf",
        [
            [
                ((72, 100, 540, 300), "A page set small throughout its own text.", 7.0),
                ((72, 340, 540, 420), "A second block of that same small size.", 7.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "A second block of that same small size." in _combined(units)
    assert documents[0]["removed_footnote_blocks"] == 0


def test_a_two_column_page_keeps_both_columns(project: Path) -> None:
    _pdf(
        project / "sources" / "columns.pdf",
        [
            [
                ((55, 120, 270, 240), "The left column of a two-column page.", 10.0),
                ((325, 120, 540, 240), "The right column of the same page.", 10.0),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "The left column" in _combined(units)
    assert "The right column" in _combined(units)
    assert documents[0]["removed_sidebar_blocks"] == 0


def test_a_pdf_reference_page_loses_its_entries_before_chunking(project: Path) -> None:
    _pdf(
        project / "sources" / "references.pdf",
        [
            [
                ((72, 90, 540, 200), BODY_PROSE, 10.0),
                (
                    (72, 230, 540, 330),
                    _paragraphs(
                        REFERENCES_HEADING,
                        ENTRY_ONE,
                        ENTRY_TWO,
                    ),
                    10.0,
                ),
            ]
        ],
    )

    documents, units = _extracted(project)

    assert "Smith, J. (2019)" not in _combined(units)
    assert "layout decision before it is a reading one" in _combined(units)
    assert documents[0]["removed_reference_segments"] == 3
    assert documents[0]["removed_non_argument_characters"] > 0


# ---------------------------------------------------------------------------
# EPUB furniture, named in markup rather than measured in geometry
# ---------------------------------------------------------------------------


def test_epub_removals_are_reported_even_when_nothing_was_removed(
    project: Path,
) -> None:
    """A source with no furniture reports zeros, not an empty record.

    An empty mapping cannot be told apart from a counter that was never written,
    so a real collection without sidebars must still show the category.
    """

    _epub(
        project / "sources" / "plain.epub",
        {"chapter.xhtml": "<h1>Chapter</h1><p>An ordinary page of argument.</p>"},
    )

    documents, _units = _extracted(project)

    assert documents[0]["removed_epub_furniture_elements"] == 0
    assert documents[0]["removed_sidebar_blocks"] == 0
    assert documents[0]["removed_reference_segments"] == 0


def test_epub_furniture_removed_is_counted_on_the_document(project: Path) -> None:
    _epub(
        project / "sources" / "furniture.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The chapter argues that the archive decides the reading.</p>"
                "<aside><p>A sidebar about the margin rule of the archive.</p></aside>"
                '<div class="sidenote"><p>A side note about the archive.</p></div>'
            )
        },
    )

    documents, units = _extracted(project)

    assert documents[0]["removed_epub_furniture_elements"] == 2
    assert "A sidebar about the margin rule" not in _combined(units)


def test_a_pdf_records_zero_removals_rather_than_an_empty_mapping(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "plain.pdf",
        [[((72, 100, 540, 300), BODY_PROSE, 10.0)]],
    )

    documents, _units = _extracted(project)

    for field in (
        "removed_repeated_margin_blocks",
        "removed_sidebar_blocks",
        "removed_footnote_blocks",
        "removed_reference_segments",
        "removed_note_segments",
        "removed_non_argument_characters",
        "image_only_pages",
    ):
        assert documents[0][field] == 0


def test_every_removal_category_the_record_carries_is_registered(
    project: Path,
) -> None:
    """A counter nothing reads is not a counter.

    `REMOVAL_FIELDS` is what both build paths filter a checkpoint through and what
    the gate merges into, so a category the extractor emits under a name the field
    list does not hold is dropped on the way to the document record: the count reads
    as zero and the refusal that should have seen it never does.
    """

    _pdf(
        project / "sources" / "counted.pdf",
        [
            [
                ((72, 20, 540, 92), RUNNING_HEAD, 8.0),
                ((72, 100, 540, 320), BODY_PROSE, 10.0),
                ((24, 100, 58, 320), "A marginal note about the margin rule.", 8.0),
                ((72, 700, 540, 760), "1 The note explains the definition used.", 7.0),
            ]
        ]
        * 3,
    )

    documents, _units = _extracted(project)

    assert set(empty_removal_counts()) == set(REMOVAL_FIELDS)
    recorded = {
        key for key in documents[0] if key.startswith(("removed_", "image_only_pages"))
    }
    assert recorded == set(REMOVAL_FIELDS)
    assert merge_removal_counts(None, {key: 1 for key in recorded}) == {
        key: 1 for key in recorded
    }
    assert documents[0]["removed_repeated_margin_blocks"] == 3
    assert documents[0]["removed_sidebar_blocks"] == 3
    assert documents[0]["removed_footnote_blocks"] == 3


def test_removed_furniture_counts_toward_the_unsafe_to_clean_verdict(
    project: Path,
) -> None:
    """The characters a geometry rule removed are the ones the verdict counts.

    A file whose text layer is mostly a running head has not been cleaned; it has
    been emptied, and indexing what is left would serve a document that is not
    there. The count has to include the furniture for `unsafe_to_clean` to see it,
    because a section rule alone never fires on such a file.
    """

    _pdf(
        project / "sources" / "head-heavy.pdf",
        [
            [
                ((72, 20, 540, 92), RUNNING_HEAD, 8.0),
                ((72, 300, 540, 340), "The council met again about the harvest.", 10.0),
            ]
        ]
        * 4,
    )

    with pytest.raises(ExtractionError, match="unsafe_to_clean") as failure:
        _extracted(project)

    assert "head-heavy.pdf" in str(failure.value)


def test_epub_sidebar_and_note_elements_are_not_indexed(project: Path) -> None:
    _epub(
        project / "sources" / "furniture.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The chapter argues that the archive decides the reading.</p>"
                "<aside><p>A sidebar about the margin rule of the archive.</p></aside>"
                '<div class="footnotes"><p>1 A note about the identifier.</p></div>'
                '<div class="sidenote"><p>A side note about the archive.</p></div>'
            )
        },
    )

    _documents, units = _extracted(project)

    assert "the archive decides the reading" in _combined(units)
    assert "A sidebar about the margin rule" not in _combined(units)
    assert "A note about the identifier" not in _combined(units)
    assert "A side note about the archive" not in _combined(units)


def test_epub_body_text_mentioning_footnotes_survives(project: Path) -> None:
    _epub(
        project / "sources" / "mentions.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The footnotes to this chapter were checked against the archive "
                "copy, and three of them name a source the text never cites.</p>"
            )
        },
    )

    _documents, units = _extracted(project)

    assert "checked against the archive copy" in _combined(units)


def test_an_epub_reference_section_is_removed_from_the_unit_text(project: Path) -> None:
    _epub(
        project / "sources" / "epub-references.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The chapter keeps its argument here and cites the earlier work "
                "in passing (Smith 2019).</p>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                f"<p>{ENTRY_ONE}</p>"
                f"<p>{ENTRY_TWO}</p>"
            )
        },
    )

    _documents, units = _extracted(project)

    assert "Smith, J. (2019). A survey" not in _combined(units)
    assert "cites the earlier work in passing" in _combined(units)


def test_an_epub_note_apparatus_is_removed_from_the_unit_text(project: Path) -> None:
    _epub(
        project / "sources" / "epub-notes.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>The chapter states its evidence in the body of the argument.</p>"
                "<h2>Notes</h2>"
                "<p>* The first note names the archive box the copy came from.</p>"
                "<p>* The second note records the access date of the same copy.</p>"
            )
        },
    )

    _documents, units = _extracted(project)

    assert "states its evidence in the body" in _combined(units)
    assert "names the archive box" not in _combined(units)


# ---------------------------------------------------------------------------
# Source-level refusals, end to end
# ---------------------------------------------------------------------------


def _scan_pdf(path: Path, *, pages: int = 3, text_pages: int = 0) -> None:
    """A PDF whose pages are images and carry no text layer at all.

    A scan is exactly this: a picture of a page, with nothing to select. It is
    written with an image rather than with sparse characters, because a page of
    digits is readable text and must not be mistaken for a scan.
    """

    document = pymupdf.open()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8))
    pixmap.set_rect(pixmap.irect, (255, 255, 255))
    for page_number in range(1, pages + 1):
        page = document.new_page()
        page.insert_image(page.rect, pixmap=pixmap)
        if page_number <= text_pages:
            page.insert_textbox(
                pymupdf.Rect(72, 100, 540, 400),
                "The page that opens the scan carries a real text layer, so the "
                "file is a mixed one rather than a scan of the whole document.",
                fontsize=11,
            )
    document.save(path)
    document.close()


def test_a_fully_scanned_pdf_is_refused_and_says_ocr_is_not_performed(
    project: Path,
) -> None:
    _scan_pdf(project / "sources" / "scanned.pdf", pages=3)

    with pytest.raises(ExtractionError) as failure:
        _extracted(project)

    message = str(failure.value)
    assert "no_text_layer" in message
    assert "No OCR is performed" in message
    assert "research-rag exclude scanned.pdf" in message


def test_a_mostly_scanned_pdf_with_one_readable_page_is_kept(project: Path) -> None:
    _scan_pdf(project / "sources" / "mixed-scan.pdf", pages=5, text_pages=1)

    documents, units = _extracted(project)

    assert any("real text layer" in str(unit["contents"]) for unit in units)
    assert documents[0]["image_only_pages"] == 4
    assert "image_only_pages_present" in documents[0]["metadata_warnings"]


def test_a_page_of_digits_beside_a_labelled_body_is_not_a_scan(
    project: Path,
) -> None:
    """Digits are text a reader can search, and must never be refused as a scan.

    A page of tables is entirely readable while carrying almost no letters, so
    whatever judges a page a scan has to look at the page's images and its text
    layer rather than at a share of its characters.
    """

    _pdf(
        project / "sources" / "figures.pdf",
        [
            [
                (
                    (72, 100, 540, 200),
                    "Table 3 sets out the counts the survey recorded each year.",
                    10.0,
                ),
                (
                    (80, 220, 520, 600),
                    "1  2  3\n4  5  6\n7  8  9\n0  1  2\n3  4  5\n6  7  8",
                    11.0,
                ),
            ]
        ]
        * 3,
    )

    documents, units = _extracted(project)

    assert documents[0]["image_only_pages"] == 0
    assert any("1 2 3" in str(unit["contents"]) for unit in units)


def test_a_file_of_nothing_but_numbers_is_refused(project: Path) -> None:
    """Numbers with nothing around them to say what they are.

    This is what a scraped drawing or a glyph soup comes out as. The units are not
    corrupt and not unreadable, so only the file-level verdict can catch it, and it
    is refused rather than indexed as a document of digits.
    """

    _pdf(
        project / "sources" / "numbers.pdf",
        [
            [
                (
                    (80, 120, 520, 600),
                    "1  2  3\n4  5  6\n7  8  9\n0  1  2\n3  4  5\n6  7  8",
                    11.0,
                )
            ]
        ]
        * 3,
    )

    with pytest.raises(ExtractionError, match=SOURCE_REASON_NO_LETTER_TEXT) as failure:
        _extracted(project)

    assert "numbers.pdf" in str(failure.value)


def test_a_non_latin_source_is_indexed_and_not_refused(project: Path) -> None:
    """Prose outside the Latin script is prose, marks and all.

    Every character of a combining script that is not a base letter is a mark, so a
    share of characters says nothing about whether the text is readable. A rule
    that read that share would refuse a Hebrew, Arabic, or Devanagari source while
    letting English through, and it would abort the whole ingest over it.
    """

    _epub_spine(
        project / "sources" / "scripts.epub",
        [
            f"<h1>Hebrew</h1><p>{HEBREW}</p>",
            f"<h1>Arabic</h1><p>{ARABIC}</p>",
            f"<h1>Devanagari</h1><p>{DEVANAGARI}</p>",
        ],
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert HEBREW in combined
    assert ARABIC in combined
    assert DEVANAGARI in combined
    assert documents[0]["removed_reference_segments"] == 0


def test_a_mixed_script_source_is_indexed_and_not_refused(project: Path) -> None:
    """A document that quotes a foreign source is not a damaged document.

    A translation, a quoted archive string, and a name in another script all appear
    in ordinary scholarship. Script mixing is recorded as a note and never withholds
    a unit, so the source survives with all of it.
    """

    _epub(
        project / "sources" / "mixed-scripts.epub",
        {
            "chapter.xhtml": (
                "<h1>Chapter</h1>"
                "<p>Die Untersuchung folgt der Frage, wie das Gedächtnis einer "
                "Stadt sich in ihren Straßen niederschlägt und dort wiederkehrt."
                "</p>"
                f"<p>{ARABIC}</p>"
                f"<p>{DEVANAGARI}</p>"
            )
        },
    )

    documents, units = _extracted(project)

    combined = _combined(units)
    assert "Gedächtnis einer" in combined
    assert ARABIC in combined
    assert DEVANAGARI in combined
    assert documents[0]["excluded_corrupt_unit_count"] == 0


def test_a_corrupt_source_is_refused_and_says_where_it_starts(project: Path) -> None:
    write_epub(project / "sources" / "broken.epub", CORRUPT_PAGE)

    with pytest.raises(ExtractionError) as failure:
        _extracted(project)

    message = str(failure.value)
    assert "First excluded unit at section" in message
    assert "replacement_characters" in message
    assert CORRUPT_PAGE not in message


def test_a_readable_source_with_one_bad_page_is_kept(project: Path) -> None:
    _epub(
        project / "sources" / "mixed.epub",
        {
            "clean.xhtml": "<h1>Clean</h1><p>Readable English research evidence.</p>",
            "broken.xhtml": f"<p>{CORRUPT_PAGE}</p>",
        },
    )

    documents, units = _extracted(project)

    assert len(units) == 1
    assert "Readable English research evidence." in str(units[0]["contents"])
    assert documents[0]["excluded_corrupt_unit_count"] == 1


# ---------------------------------------------------------------------------
# Failure atomicity through a staged build
# ---------------------------------------------------------------------------


def test_a_refused_source_aborts_the_build_and_preserves_the_current_generation(
    project: Path,
) -> None:
    write_pdf(project / "sources" / "good.pdf", ["Cobalt evidence about labour."])
    service = _service(project)
    config = service.config
    first = asyncio.run(service.ingest(chunk_size=50, chunk_overlap=10))
    assert config.current_path.exists()

    write_epub(project / "sources" / "corrupt.epub", CORRUPT_PAGE)
    with pytest.raises(ExtractionError):
        asyncio.run(service.ingest(chunk_size=50, chunk_overlap=10))

    # The selected generation is the one that worked, the failed build left the
    # pointer and the heavy staging data alone, and the record names the file
    # without quoting text it could not read.
    assert (
        json.loads(config.current_path.read_text(encoding="utf-8"))["generation_id"]
        == first["generation_id"]
    )
    assert not any(config.staging_root.iterdir())
    failures = list(config.failures_root.glob("*.json"))
    assert len(failures) == 1
    recorded = failures[0].read_text(encoding="utf-8")
    assert "corrupt.epub" in recorded
    assert CORRUPT_PAGE not in recorded


def test_a_refused_source_leaves_no_activation_behind(project: Path) -> None:
    _epub(
        project / "sources" / "only-corrupt.epub",
        {"corrupt.xhtml": f"<p>{CORRUPT_PAGE}</p>"},
    )
    service = _service(project)
    config = service.config

    with pytest.raises(ExtractionError):
        asyncio.run(service.ingest(chunk_size=50, chunk_overlap=10))

    assert not config.current_path.exists()
    assert not any(config.staging_root.iterdir())


def test_the_staged_path_applies_the_same_gate_as_a_direct_build(
    project: Path,
) -> None:
    """Both paths must reach the same units, in the same order, with the same ids.

    A staged build extracts in batches and a direct build in one pass, so the
    shared gate is the only place they can be held to one answer.
    """

    _pdf(
        project / "sources" / "mixed.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((24, 100, 58, 300), "A marginal note beside the body.", 8.0),
            ],
            [
                ((72, 90, 540, 200), SYMBOLS_ONLY, 10.0),
                (
                    (72, 230, 540, 330),
                    "A second readable page that carries the argument onward.",
                    10.0,
                ),
            ],
        ],
    )
    service = _service(project)
    direct_documents, direct_units = _extracted(project)

    result = asyncio.run(service.ingest(chunk_size=200, chunk_overlap=10))
    generation_root = Path(result["generation_root"])
    staged_units = [
        json.loads(line)
        for line in (generation_root / "corpus" / "extracted-units.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [unit["contents"] for unit in staged_units] == [
        str(unit["contents"]) for unit in direct_units
    ]
    assert [unit["id"] for unit in staged_units] == [
        str(unit["id"]) for unit in direct_units
    ]
    assert [unit["locator"] for unit in staged_units] == [
        unit["locator"] for unit in direct_units
    ]
    # The sidebar the direct path removed is absent from the staged build too, and
    # the symbol-only page was withheld by both.
    assert "A marginal note beside the body." not in _combined(staged_units)
    assert SYMBOLS_ONLY not in _combined(staged_units)
    assert (
        result["excluded_corrupt_unit_count"]
        == (direct_documents[0]["excluded_corrupt_unit_count"])
    )


def test_the_staged_path_removes_a_reference_list_the_direct_path_removes(
    project: Path,
) -> None:
    """The two paths must agree on a section that spans pages and spine items.

    The staged build writes each page batch before it knows the whole file, so this
    is the case where a per-batch rule would leave the list in and the direct
    build would take it out. The gate is what makes them one answer.
    """

    _epub_spine(
        project / "sources" / "book.epub",
        [
            "<h1>One</h1><p>The first chapter makes its case in full.</p>",
            (
                "<h1>Two</h1>"
                "<p>The second chapter draws the two arguments together.</p>"
                f"<h2>{REFERENCES_HEADING}</h2>"
                f"<p>{ENTRY_ONE}</p>"
            ),
            f"<p>{ENTRY_TWO}</p>",
        ],
    )
    service = _service(project)
    direct_documents, direct_units = _extracted(project)

    result = asyncio.run(service.ingest(chunk_size=200, chunk_overlap=10))
    generation_root = Path(result["generation_root"])
    staged_units = [
        json.loads(line)
        for line in (generation_root / "corpus" / "extracted-units.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [unit["contents"] for unit in staged_units] == [
        str(unit["contents"]) for unit in direct_units
    ]
    assert [unit["id"] for unit in staged_units] == [
        str(unit["id"]) for unit in direct_units
    ]
    assert "Smith, J. (2019). A survey" not in _combined(staged_units)
    assert "Jones, A. (2020). Another" not in _combined(staged_units)
    assert "The first chapter makes its case" in _combined(staged_units)
    assert direct_documents[0]["removed_reference_segments"] == 3


def test_the_staged_path_removes_a_pdf_reference_list_the_direct_path_removes(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "book.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((72, 340, 540, 380), REFERENCES_HEADING, 11.0),
            ],
            [((72, 100, 540, 300), _paragraphs(ENTRY_ONE, ENTRY_TWO), 10.0)],
        ],
    )
    service = _service(project)
    direct_documents, direct_units = _extracted(project)

    result = asyncio.run(service.ingest(chunk_size=200, chunk_overlap=10))
    generation_root = Path(result["generation_root"])
    staged_units = [
        json.loads(line)
        for line in (generation_root / "corpus" / "extracted-units.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [unit["contents"] for unit in staged_units] == [
        str(unit["contents"]) for unit in direct_units
    ]
    assert "Smith, J. (2019). A survey" not in _combined(staged_units)
    assert direct_documents[0]["removed_reference_segments"] == 3


def test_originals_are_untouched_by_a_build_that_removes_furniture(
    project: Path,
) -> None:
    _pdf(
        project / "sources" / "furniture.pdf",
        [
            [
                ((72, 100, 540, 300), BODY_PROSE, 10.0),
                ((24, 100, 58, 300), "A marginal note about the margin rule.", 8.0),
                (
                    (72, 340, 540, 400),
                    "1 The note explains the definition used in the table above.",
                    7.0,
                ),
            ]
        ],
    )
    path = project / "sources" / "furniture.pdf"
    before = path.read_bytes()

    service = _service(project)
    asyncio.run(service.ingest(chunk_size=200, chunk_overlap=10))

    assert path.read_bytes() == before
