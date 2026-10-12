from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest
from ebooklib import epub

import research_rag.corpus.extraction as extraction_module
import research_rag.project.config as config_module
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.corpus.extraction import (
    ExtractionError,
    _quality_flags,
    extract_sources,
)
from research_rag.corpus.sources import (
    SourcePolicyError,
    normalize_metadata,
    scan_sources,
    stable_source_id,
)
from research_rag.corpus.text_normalization import (
    normalize_inline_text,
    normalize_reading_text,
)
from research_rag.corpus.text_quality import (
    has_searchable_alphanumeric_content,
    text_corruption_reasons,
    text_health_reasons,
    text_script_notes,
)
from research_rag.project.config import (
    ConfigurationError,
    configured_source_directory,
    resolve_config,
)
from tests.conftest import write_epub, write_pdf

# The managed Docling environment is resolved by the caller; the shared test stub
# replaces the worker, so this stands in for the environment identity only.
_FAKE_ENV = DoclingEnv(
    root=Path("/nonexistent/docling-env"),
    spec_fingerprint="spec-test",
    versions={"docling": "2.135.0"},
)

CORRUPT_TEXT = (
    "��ѪҶޜഝǄ䘉Ӌਁ ⧠ᢃ⹤Ҷἅ ൠ؞༽൷㜭ᡀ࣏ "
    "䖜රѪտᆵǃ୶ъㅹ儈ԧ٬ъᘱⲴ⡷䶒ਉһǄ൘ᡰᴹᵳ╄ਈᯩ "
    "䶒ˈབྷཊᮠ൪ൠ൘䗷 ৫ഋॱᒤѝ࿻㓸؍ᤱ⿱ᴹᡆޜᴹ኎ᙗǄ "
    "ᵜ᮷䘈ᇎ䇱ਁ ⧠ˈ䜘࠶൪ൠᡰᴹᵳ⭡⊑ḃර⿱㩕Աъࡂ䖜㠣᭯ "
    "ᓌˈ⭡᭯ ᓌ᢯ᣵޘ䜘؞༽䍴䠁ˈᴰ㓸䙐ᡀ⊑ḃ⋫⨶ᡀᵜ⽮Պॆ "
    "ˈᒦᕅਁ ⧟ຳнޜǄᴹ∂ᓏ⢙൪ൠ 䳮�"
)


def _write_custom_epub(path: Path, content: str) -> None:
    book = epub.EpubBook()
    book.set_identifier("custom-locator-test")
    book.set_title("Locator Test")
    book.set_language("en")
    chapter = epub.EpubHtml(
        title="Locator Chapter",
        file_name="locator.xhtml",
        lang="en",
    )
    chapter.content = content
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


def test_only_pdf_and_epub_are_selected(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["A cobalt research passage."])
    write_epub(project / "sources" / "book.epub", "An amber research passage.")
    (project / "sources" / "notes.md").write_text("derived notes", encoding="utf-8")

    config = resolve_config(project, vanilla_executable=sys.executable)
    scan = scan_sources(config)

    assert [item.extension for item in scan.selected] == [".pdf", ".epub"]
    assert scan.ignored_extensions == {".md": 1}


def test_source_id_is_project_scoped_and_content_independent(project: Path) -> None:
    path = project / "sources" / "article.pdf"
    write_pdf(path, ["Initial source contents."])
    config = resolve_config(project, vanilla_executable=sys.executable)

    initial = scan_sources(config).selected[0]
    path.write_bytes(b"changed source bytes")
    changed = scan_sources(config).selected[0]

    assert initial.source_id == changed.source_id
    assert initial.source_id == stable_source_id(
        config.project_id,
        initial.source_relative_path,
    )
    assert initial.source_id != stable_source_id(
        "another-project",
        initial.source_relative_path,
    )
    assert initial.source_id != stable_source_id(config.project_id, "other.pdf")


def test_pdf_pages_and_epub_sections_preserve_locators(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["First page cobalt evidence.", "Second page amber evidence."],
        title="Located PDF",
    )
    write_epub(
        project / "sources" / "book.epub",
        "Section evidence about quartz.",
        title="Located EPUB",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)

    documents, units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )

    assert {item["format"] for item in documents} == {"pdf", "epub"}
    documents_by_id = {item["document_id"]: item for item in documents}
    assert all(
        item["source_id"] == documents_by_id[item["document_id"]]["source_id"]
        for item in units
    )
    assert all("source_id" not in item["locator"] for item in units)
    pdf_units = [item for item in units if item["locator"]["type"] == "pdf_page"]
    epub_units = [item for item in units if item["locator"]["type"] == "epub_section"]
    assert [item["locator"]["page"] for item in pdf_units] == [1, 2]
    assert any("quartz" in item["contents"] for item in epub_units)
    assert all("section_index" in item["locator"] for item in epub_units)
    assert all("element_path" in item["locator"] for item in epub_units)
    assert all("block_index" in item["locator"] for item in epub_units)


def test_epub_preserves_element_and_preceding_anchors_with_stable_fallbacks(
    project: Path,
) -> None:
    path = project / "sources" / "anchors.epub"
    _write_custom_epub(
        path,
        (
            "<h1>Navigation Chapter</h1>"
            '<p id="element-anchor">Element-anchored evidence.</p>'
            '<a id="standalone-anchor"></a>'
            "<p>Standalone-anchor evidence.</p>"
            "<p>Structural fallback evidence.</p>"
        ),
    )
    original = path.read_bytes()
    config = resolve_config(project, vanilla_executable=sys.executable)

    _documents, units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )
    _repeat_documents, repeated = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )

    assert path.read_bytes() == original
    assert [unit["id"] for unit in repeated] == [unit["id"] for unit in units]
    assert [unit["locator"] for unit in repeated] == [unit["locator"] for unit in units]
    assert [unit["locator"]["block_index"] for unit in units] == [1, 2, 3]

    element = units[0]
    assert element["contents"] == ("Navigation Chapter\n\nElement-anchored evidence.")
    assert element["locator"] == {
        "type": "epub_section",
        "section_index": 2,
        "section_title": "Navigation Chapter",
        "href": "locator.xhtml",
        "element_path": "/html[1]/body[1]/p[1]",
        "block_index": 1,
        "fragment": "element-anchor",
        "href_with_fragment": "locator.xhtml#element-anchor",
        "anchor_relation": "element",
        "anchor_source": "id",
    }

    preceding = units[1]["locator"]
    assert preceding["fragment"] == "standalone-anchor"
    assert preceding["href_with_fragment"] == "locator.xhtml#standalone-anchor"
    assert preceding["anchor_relation"] == "preceding"
    assert preceding["anchor_source"] == "id"

    fallback = units[2]["locator"]
    assert fallback["element_path"] == "/html[1]/body[1]/p[3]"
    assert not {
        "fragment",
        "href_with_fragment",
        "anchor_relation",
        "anchor_source",
    }.intersection(fallback)


def test_epub_mid_paragraph_anchor_splits_without_marker_leakage(
    project: Path,
) -> None:
    path = project / "sources" / "mid-paragraph.epub"
    _write_custom_epub(
        path,
        (
            "<p>Evidence before the internal marker. "
            '<a name="midpoint"></a>'
            "Evidence after the internal marker.</p>"
        ),
    )
    config = resolve_config(project, vanilla_executable=sys.executable)

    _documents, units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )

    assert [unit["contents"] for unit in units] == [
        "Evidence before the internal marker.",
        "Evidence after the internal marker.",
    ]
    assert [unit["locator"]["block_index"] for unit in units] == [1, 2]
    assert {unit["locator"]["element_path"] for unit in units} == {
        "/html[1]/body[1]/p[1]"
    }
    assert "fragment" not in units[0]["locator"]
    assert units[1]["locator"]["fragment"] == "midpoint"
    assert units[1]["locator"]["href_with_fragment"] == ("locator.xhtml#midpoint")
    assert units[1]["locator"]["anchor_relation"] == "nested"
    assert units[1]["locator"]["anchor_source"] == "name"
    assert "midpoint" not in " ".join(unit["contents"] for unit in units)


def test_epub_tables_keep_document_order_and_nested_blocks_are_not_duplicated(
    project: Path,
) -> None:
    path = project / "sources" / "ordered-blocks.epub"
    _write_custom_epub(
        path,
        (
            "<p>Prose before the table.</p>"
            '<table id="evidence-table">'
            "<tr><th>Term</th><th>Value</th></tr>"
            "<tr><td>Labour</td><td>Visible</td></tr>"
            "</table>"
            "<blockquote><p>Nested quoted evidence.</p></blockquote>"
        ),
    )
    config = resolve_config(project, vanilla_executable=sys.executable)

    _documents, units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )

    assert [unit["contents"] for unit in units] == [
        "Prose before the table.",
        "Term | Value\nLabour | Visible",
        "Nested quoted evidence.",
    ]
    assert [unit["content_kind"] for unit in units] == ["prose", "table", "prose"]
    assert units[1]["locator"]["fragment"] == "evidence-table"
    assert units[1]["locator"]["anchor_relation"] == "element"
    assert units[1]["locator"]["element_path"] == "/html[1]/body[1]/table[1]"
    assert units[2]["locator"]["element_path"] == ("/html[1]/body[1]/blockquote[1]")
    assert (
        "\n".join(unit["contents"] for unit in units).count("Nested quoted evidence.")
        == 1
    )


def test_formula_letters_and_ligatures_fold_for_matching() -> None:
    # Mathematical Alphanumeric Symbols come from formula fonts; without folding a
    # plain-text query can never match them.
    assert normalize_reading_text("𝑀𝑗𝑀𝑗𝑗𝑀") == "MjMjjM"
    assert normalize_inline_text("𝑀𝑗𝑗𝑇𝑗𝑗 and 𝑇𝑗𝑗𝑇𝑗𝑗") == "MjjTjj and TjjTjj"
    assert text_corruption_reasons("𝑀𝑗𝑀𝑗𝑗𝑀") == []
    assert text_script_notes("𝑀𝑗𝑗𝑇𝑗𝑗") == []

    assert (
        normalize_reading_text("The ﬁnal ﬂow aﬀects it.")
        == "The final flow affects it."
    )
    assert normalize_inline_text("ﬂour and ﬁbre") == "flour and fibre"


def test_folding_preserves_letters_and_symbols_that_carry_meaning() -> None:
    for text in (
        "Café workers’ organisations analyse political economy.",
        "Œuvre and æsthetic stay as printed.",
        "x¹ + y² = 3 in footnote ⁴",
        "W–(M–C–M′)–W′",
    ):
        assert normalize_inline_text(text) == text


def test_extracted_text_removes_layout_wrapping() -> None:
    raw = (
        "A sentence wraps in the\nmiddle of a line.\n\n"
        "A com-\nmodity remains readable.\r\n\r\n"
        "A PDF span says waste- value.\n\nFinal paragraph."
    )

    assert normalize_reading_text(raw) == (
        "A sentence wraps in the middle of a line.\n\n"
        "A commodity remains readable.\n\n"
        "A PDF span says waste-value.\n\nFinal paragraph."
    )
    assert normalize_inline_text(raw) == (
        "A sentence wraps in the middle of a line. "
        "A commodity remains readable. "
        "A PDF span says waste-value. Final paragraph."
    )


def test_corruption_policy_preserves_valid_symbols() -> None:
    assert text_corruption_reasons("W–(M–C–M′)–W′") == []
    assert (
        text_corruption_reasons(
            "Café workers’ organisations analyse political economy."
        )
        == []
    )
    assert text_corruption_reasons("https://example.org/10.1000/test") == []
    assert (
        text_corruption_reasons("Âme, Ãlvaro, and Zoë discuss political economy.") == []
    )
    assert text_corruption_reasons(CORRUPT_TEXT) == [
        "replacement_characters",
        "private_or_unassigned_characters",
    ]


def test_foreign_script_text_is_noted_but_never_withheld() -> None:
    chinese = "这是一个完整的中文段落，用于测试英语导向的过滤策略。这里有足够多的汉字。"
    assert text_corruption_reasons(chinese) == []
    assert text_script_notes(chinese) == ["non_latin_dominant"]

    quoted = "Latinlettersab αβγ БГД אבג enough"
    assert text_corruption_reasons(quoted) == []
    assert text_script_notes(quoted) == ["mixed_script_text"]


def test_foreign_language_quotation_stays_retrievable() -> None:
    # A Greek quotation carrying one unreadable glyph must remain evidence: a
    # single replacement character is withheld only with corroborating corruption
    # evidence, and script mixing never corroborates.
    quotation = "ὁ ἄργυρος κακὸν νόμισμ᾽ ἔβλαστε καὶ πόλεις πορθεῖ �"
    assert text_corruption_reasons(quotation) == []
    assert text_script_notes(quotation) == ["non_latin_dominant"]

    bibliography = (
        "Покровскій], Василій [Иванович]: [Review of:] «Капиталъ» Д. Рикардо "
        "въ связи съ позднѣйшими дополненіями и разъясненіями."
    )
    assert text_corruption_reasons(bibliography) == []


@pytest.mark.parametrize(
    "text",
    [
        "… — – • § † ‡",
        "∑ × ÷ ≈ → ∞",
        "🙂 🚀 🧠",
    ],
)
def test_symbol_only_text_has_an_inspectable_health_reason(text: str) -> None:
    assert not has_searchable_alphanumeric_content(text)
    assert text_health_reasons(text) == ["symbol_only"]


@pytest.mark.parametrize(
    "text",
    [
        "W–(M–C–M′)–W′",
        "https://example.org/10.1000/test",
        "Café workers’ organisations analyse political economy.",
        "x² + y₃ = 42",
    ],
)
def test_text_with_unicode_alphanumeric_content_is_searchable(text: str) -> None:
    assert has_searchable_alphanumeric_content(text)
    assert text_health_reasons(text) == []


def test_symbol_filter_preserves_formula_and_existing_page_number_guard() -> None:
    assert _quality_flags("W–(M–C–M′)–W′", "prose") == []
    assert _quality_flags("… — ∑ × 🙂", "figure") == ["extraction_artifact"]
    assert _quality_flags("[ 499 ]", "prose") == ["extraction_artifact"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Readable prose with two broken symbols: ��", "replacement_characters"),
        (
            "Readable prose with repeated private symbols: \ue000\ue001",
            "private_or_unassigned_characters",
        ),
        ("The cafÃ© text is a known damaged encoding sequence.", "known_mojibake"),
    ],
)
def test_each_corruption_signal_has_an_inspectable_reason(
    text: str,
    reason: str,
) -> None:
    assert reason in text_corruption_reasons(text)


def test_one_replacement_character_requires_another_corruption_signal() -> None:
    assert text_corruption_reasons("A single transcription marker � remains.") == []
    assert text_corruption_reasons("Damaged cafÃ© text � remains.") == [
        "replacement_characters",
        "known_mojibake",
    ]


def test_short_four_script_text_is_noted_without_letter_threshold() -> None:
    assert text_corruption_reasons("aαБא") == []
    assert text_script_notes("aαБא") == ["mixed_script_text"]


def test_corrupt_epub_units_are_excluded_with_locator_diagnostics(
    project: Path,
) -> None:
    path = project / "sources" / "mixed.epub"
    book = epub.EpubBook()
    book.set_identifier("mixed-corruption")
    book.set_title(CORRUPT_TEXT)
    book.set_language("en")
    clean = epub.EpubHtml(title="Clean", file_name="clean.xhtml", lang="en")
    clean.content = "<h1>Clean</h1><p>Readable English research evidence.</p>"
    corrupt = epub.EpubHtml(title="Broken", file_name="broken.xhtml", lang="en")
    corrupt.content = f"<h1>Broken</h1><p>{CORRUPT_TEXT}</p>"
    book.add_item(clean)
    book.add_item(corrupt)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", clean, corrupt]
    epub.write_epub(str(path), book)

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )

    assert len(units) == 1
    assert units[0]["contents"].startswith("Clean")
    assert documents[0]["title"] == "mixed"
    assert "corrupt_extracted_title" in documents[0]["metadata_warnings"]
    assert documents[0]["excluded_corrupt_unit_count"] == 1
    diagnostic = documents[0]["excluded_corrupt_units"][0]
    assert diagnostic["locator"]["type"] == "epub_section"
    assert diagnostic["reasons"] == [
        "replacement_characters",
        "private_or_unassigned_characters",
    ]
    assert text_script_notes(CORRUPT_TEXT) == [
        "non_latin_dominant",
        "mixed_script_text",
    ]
    assert CORRUPT_TEXT not in json.dumps(diagnostic, ensure_ascii=False)


def test_source_with_only_corrupt_text_fails_extraction(project: Path) -> None:
    write_epub(project / "sources" / "broken.epub", CORRUPT_TEXT)
    config = resolve_config(project, vanilla_executable=sys.executable)

    with pytest.raises(ExtractionError, match="no readable English-oriented text"):
        extract_sources(scan_sources(config).selected, docling_env=_FAKE_ENV)


def test_symbol_only_epub_unit_is_excluded_with_locator_diagnostics(
    project: Path,
) -> None:
    path = project / "sources" / "mixed-symbols.epub"
    book = epub.EpubBook()
    book.set_identifier("mixed-symbols")
    book.set_title("Mixed Symbols")
    book.set_language("en")
    clean = epub.EpubHtml(title="Clean", file_name="clean.xhtml", lang="en")
    clean.content = "<p>Readable English research evidence.</p>"
    symbols = epub.EpubHtml(title="Symbols", file_name="symbols.xhtml", lang="en")
    symbols.content = "<p>… — ∑ × 🙂</p>"
    book.add_item(clean)
    book.add_item(symbols)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", clean, symbols]
    epub.write_epub(str(path), book)

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )

    assert [unit["contents"] for unit in units] == [
        "Readable English research evidence."
    ]
    assert documents[0]["excluded_corrupt_unit_count"] == 1
    diagnostic = documents[0]["excluded_corrupt_units"][0]
    assert diagnostic["locator"] == {
        "type": "epub_section",
        "section_index": 3,
        "section_title": "",
        "href": "symbols.xhtml",
        "element_path": "/html[1]/body[1]/p[1]",
        "block_index": 1,
    }
    assert diagnostic["reasons"] == ["symbol_only"]
    assert "… — ∑ × 🙂" not in json.dumps(diagnostic, ensure_ascii=False)


def test_source_with_only_symbol_text_fails_extraction(project: Path) -> None:
    path = project / "sources" / "only-symbols.epub"
    book = epub.EpubBook()
    book.set_identifier("only-symbols")
    book.set_title("Only Symbols")
    book.set_language("en")
    symbols = epub.EpubHtml(title="Symbols", file_name="symbols.xhtml", lang="en")
    symbols.content = "<p>… — ∑ × 🙂</p>"
    book.add_item(symbols)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", symbols]
    epub.write_epub(str(path), book)

    config = resolve_config(project, vanilla_executable=sys.executable)

    with pytest.raises(ExtractionError, match="no readable English-oriented text"):
        extract_sources(scan_sources(config).selected, docling_env=_FAKE_ENV)


def test_doi_title_becomes_doi_and_filename_is_only_title_fallback(
    project: Path,
) -> None:
    path = project / "sources" / "fallback-title.pdf"
    document = pymupdf.open()
    document.set_metadata(
        {"title": "DOI: 10.5555/example.42", "author": "Microsoft Word"}
    )
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 100, 530, 700),
        "Ordinary body prose without a visible title or byline.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )
    source = documents[0]

    assert source["title"] == "fallback-title"
    assert source["authors"] == []
    assert source["doi"] == "10.5555/example.42"
    assert source["metadata_provenance"]["title"] == "filename"
    assert {"title_from_filename", "authors_missing"}.issubset(
        source["metadata_warnings"]
    )


def test_valid_pdf_embedded_metadata_is_used_independently(project: Path) -> None:
    path = project / "sources" / "well-named-article.pdf"
    document = pymupdf.open()
    document.set_metadata(
        {
            "title": "Embedded Article Title",
            "author": "Ada Lovelace; Grace Hopper",
        }
    )
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 100, 530, 700),
        "Ordinary body prose without a high-confidence visible title or byline.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )
    source = documents[0]

    assert source["title"] == "Embedded Article Title"
    assert source["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert source["metadata_provenance"]["title"] == "pdf_metadata"
    assert source["metadata_provenance"]["authors"] == "pdf_metadata"


def test_pdf_metadata_year_falls_back_to_catalog_dates() -> None:
    # Docling supplies the catalogue metadata; the shared identity helper reads
    # the year from a creation date when no visible front matter is available.
    automatic, provenance, _warnings = extraction_module._front_matter_identity(
        [],
        {"title": "Embedded Title", "creationDate": "D:20240310000000Z"},
        "stem",
    )

    assert automatic["year"] == 2024
    assert provenance["year"] == "pdf_metadata"


def test_epub_uses_opf_then_visible_metadata(project: Path) -> None:
    opf = project / "sources" / "opf.epub"
    write_epub(opf, "EPUB evidence.", title="Validated OPF Title")

    visible = project / "sources" / "visible.epub"
    book = epub.EpubBook()
    book.set_identifier("10.1000/visible")
    book.set_title("doi:10.1000/visible")
    book.set_language("en")
    book.add_author("Unknown")
    chapter = epub.EpubHtml(title="Visible", file_name="visible.xhtml", lang="en")
    chapter.content = (
        "<h1>The Visible Book Title</h1>"
        '<p class="byline">By Ada Lovelace and Grace Hopper</p>'
        "<p>Semantic evidence in the opening chapter.</p>"
    )
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(visible), book)

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(
        scan_sources(config).selected, docling_env=_FAKE_ENV
    )
    by_path = {item["source_relative_path"]: item for item in documents}

    assert by_path["opf.epub"]["title"] == "Validated OPF Title"
    assert by_path["opf.epub"]["metadata_provenance"]["title"] == "epub_opf"
    assert by_path["visible.epub"]["title"] == "The Visible Book Title"
    assert by_path["visible.epub"]["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert by_path["visible.epub"]["doi"] == "10.1000/visible"
    assert by_path["visible.epub"]["metadata_provenance"]["title"] == ("epub_visible")


def test_source_directory_cannot_escape_project(project: Path) -> None:
    with pytest.raises(ConfigurationError, match="escapes"):
        resolve_config(
            project,
            source_directory="../outside",
            vanilla_executable=sys.executable,
        )


def test_initialized_source_directory_can_be_reused(project: Path) -> None:
    (project / "library").mkdir()
    resolve_config(
        project,
        source_directory="library",
        vanilla_executable=sys.executable,
    )

    assert configured_source_directory(project) == "library"


def test_new_project_keeps_all_research_state_under_one_root(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)

    assert config.portable_root == project / ".research-rag"
    assert config.state_root == project / ".research-rag" / "runtime"
    assert not (project / ".ultrarag").exists()


def test_portable_project_identity_is_stable_and_legacy_review_state_migrates(
    project: Path,
) -> None:
    legacy = project / ".ultrarag" / "research"
    legacy.mkdir(parents=True)
    (legacy / "source-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {"article.pdf": {"title": "Reviewed title"}},
            }
        ),
        encoding="utf-8",
    )
    (legacy / "source-exclusions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "duplicate.pdf": {
                        "reason": "Reviewed duplicate",
                        "excluded_at": "2026-09-18T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (legacy / "current.json").write_text(
        json.dumps({"schema_version": 1, "generation_id": "legacy-generation"}),
        encoding="utf-8",
    )

    first = resolve_config(project, vanilla_executable=sys.executable)
    second = resolve_config(project, vanilla_executable=sys.executable)

    assert first.project_id == second.project_id
    descriptor = json.loads(first.project_config_path.read_text(encoding="utf-8"))
    assert descriptor == {
        "schema_version": 1,
        "project_id": first.project_id,
        "name": project.name,
        "source_directory": "sources",
    }
    assert json.loads(first.metadata_path.read_text(encoding="utf-8"))["sources"] == {
        "article.pdf": {"title": "Reviewed title"}
    }
    assert (
        "duplicate.pdf"
        in json.loads(first.source_exclusions_path.read_text(encoding="utf-8"))[
            "sources"
        ]
    )
    assert first.state_root == project / ".research-rag" / "runtime"
    assert (
        json.loads(first.current_path.read_text(encoding="utf-8"))["generation_id"]
        == "legacy-generation"
    )
    assert not legacy.exists()
    assert not (project / ".ultrarag").exists()


def test_conflicting_legacy_and_consolidated_runtime_is_refused(
    project: Path,
) -> None:
    legacy = project / ".ultrarag" / "research"
    consolidated = project / ".research-rag" / "runtime"
    legacy.mkdir(parents=True)
    consolidated.mkdir(parents=True)
    (legacy / "current.json").write_text("legacy", encoding="utf-8")
    (consolidated / "current.json").write_text("consolidated", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="both"):
        resolve_config(project, vanilla_executable=sys.executable)


def test_legacy_migration_preserves_other_ultrarag_state(project: Path) -> None:
    legacy = project / ".ultrarag" / "research"
    unrelated = project / ".ultrarag" / "vanilla-ui" / "settings.json"
    legacy.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    (legacy / "current.json").write_text("legacy", encoding="utf-8")
    unrelated.write_text("unrelated", encoding="utf-8")

    config = resolve_config(project, vanilla_executable=sys.executable)

    assert (config.state_root / "current.json").read_text(encoding="utf-8") == (
        "legacy"
    )
    assert unrelated.read_text(encoding="utf-8") == "unrelated"


def test_projects_share_only_the_configured_model_cache(tmp_path: Path) -> None:
    first_project = tmp_path / "first-project"
    second_project = tmp_path / "second-project"
    (first_project / "sources").mkdir(parents=True)
    (second_project / "sources").mkdir(parents=True)
    shared_models = tmp_path / "shared-models"

    first = resolve_config(
        first_project,
        vanilla_executable=sys.executable,
        model_cache_root=shared_models,
    )
    second = resolve_config(
        second_project,
        vanilla_executable=sys.executable,
        model_cache_root=shared_models,
    )

    assert first.model_cache_root == second.model_cache_root == shared_models
    assert first.state_root != second.state_root
    assert first.generations_root != second.generations_root
    assert not first.legacy_models_root.exists()
    assert not second.legacy_models_root.exists()


def test_relocated_runtime_root_is_claimed_then_reused(
    project: Path, tmp_path: Path
) -> None:
    runtime_root = tmp_path / "fast-local-runtime"
    marker = runtime_root / ".research-rag-runtime.json"

    relocated = resolve_config(
        project,
        vanilla_executable=sys.executable,
        runtime_root=runtime_root,
    )

    assert relocated.state_root == runtime_root.resolve()
    assert relocated.runtime_root == runtime_root.resolve()
    assert relocated.generations_root.is_dir()
    assert relocated.logs_root.is_dir()
    assert relocated.staging_root.is_dir()
    assert relocated.metadata_path.parent == project / ".research-rag"
    assert not (project / ".research-rag" / "runtime" / "generations").exists()
    marker_record = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_record["project_id"] == relocated.project_id
    assert marker_record["project_root"] == str(project.resolve())
    assert marker_record["schema_version"] == 1

    again = resolve_config(
        project,
        vanilla_executable=sys.executable,
        runtime_root=runtime_root / ".",
    )
    assert again.state_root == runtime_root.resolve()
    assert again.project_id == relocated.project_id


def test_relocated_runtime_root_refuses_another_project_and_foreign_data(
    project: Path,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "shared-runtime"
    resolve_config(
        project,
        vanilla_executable=sys.executable,
        runtime_root=runtime_root,
    )

    other_project = tmp_path / "other-project"
    (other_project / "sources").mkdir(parents=True)
    with pytest.raises(ConfigurationError, match="belongs to another project"):
        resolve_config(
            other_project,
            vanilla_executable=sys.executable,
            runtime_root=runtime_root,
        )

    with pytest.raises(ConfigurationError, match="belongs to another project"):
        resolve_config(
            other_project,
            vanilla_executable=sys.executable,
            runtime_root=tmp_path / "shared-runtime" / ".." / "shared-runtime",
        )
    assert (
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            runtime_root=runtime_root,
        ).state_root
        == runtime_root.resolve()
    )

    unrelated = tmp_path / "unrelated-data"
    (unrelated / "generations").mkdir(parents=True)
    (unrelated / "generations" / "keep.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="carries no project marker"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            runtime_root=unrelated,
        )
    assert (unrelated / "generations" / "keep.json").is_file()

    with pytest.raises(ConfigurationError, match="must be an absolute path"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            runtime_root="relative/runtime",
        )
    with pytest.raises(ConfigurationError, match="must not be the project root"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            runtime_root=project / ".research-rag",
        )

    occupied = tmp_path / "occupied.json"
    occupied.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a directory"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            runtime_root=occupied,
        )


def test_embedding_threads_option_is_validated(project: Path) -> None:
    default = resolve_config(project, vanilla_executable=sys.executable)
    assert default.embedding_threads is None

    # An environment variable arrives as a string, so it is coerced and checked.
    configured = resolve_config(
        project,
        vanilla_executable=sys.executable,
        embedding_threads="8",
    )
    assert configured.embedding_threads == 8
    assert (
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            embedding_threads="",
        ).embedding_threads
        is None
    )

    # 0 is how a file says "leave the thread count to the runtime".
    assert (
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            embedding_threads=0,
        ).embedding_threads
        is None
    )
    with pytest.raises(ConfigurationError, match="runtime.embedding_threads"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            embedding_threads="many",
        )


def test_default_runtime_root_needs_no_marker(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)

    assert config.runtime_root is None
    assert config.state_root == project / ".research-rag" / "runtime"
    assert not (config.state_root / ".research-rag-runtime.json").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_configuration_does_not_require_a_retired_gateway_executable(
    project: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.setattr(config_module, "GATEWAY_EXECUTABLE", "retired-gateway-absent")
    missing = (
        project / "retired-gateway-absent"
        if explicit
        else Path(sys.executable).parent / "retired-gateway-absent"
    )
    assert not missing.exists()

    config = resolve_config(project, vanilla_executable=missing if explicit else None)

    assert config.vanilla_executable == missing.absolute()
    assert config.project_root == project
    assert config.state_root == project / ".research-rag" / "runtime"
    assert config.portable_root == project / ".research-rag"
    assert not missing.exists()


def test_offline_mode_can_read_an_existing_legacy_model_cache(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_cache = project / ".ultrarag" / "research" / "models"
    old_cache.mkdir(parents=True)
    (old_cache / "cached-model.bin").write_bytes(b"cached")
    empty_shared = tmp_path / "empty-global-cache"
    monkeypatch.setattr(
        "research_rag.project.config.user_cache_path",
        lambda *_args, **_kwargs: empty_shared,
    )

    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        offline=True,
    )

    migrated_cache = project / ".research-rag" / "runtime" / "models"
    assert config.model_cache_root == migrated_cache
    assert (migrated_cache / "cached-model.bin").read_bytes() == b"cached"
    assert not old_cache.exists()


def test_pdf_symlink_is_rejected(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    write_pdf(outside, ["outside"])
    (project / "sources" / "linked.pdf").symlink_to(outside)
    config = resolve_config(project, vanilla_executable=sys.executable)

    with pytest.raises(SourcePolicyError, match="Symbolic-link"):
        scan_sources(config)


def test_language_metadata_takes_iso_codes_and_refuses_other_values() -> None:
    assert normalize_metadata({"language": ["EN", "de"]}) == {"language": ["en", "de"]}
    # BM25 cannot tokenize Arabic; the corpus-level setting is where that is refused.
    assert normalize_metadata({"language": ["ar"]}) == {"language": ["ar"]}
    with pytest.raises(SourcePolicyError, match="ISO 639"):
        normalize_metadata({"language": ["english"]})
    with pytest.raises(SourcePolicyError, match="list of strings"):
        normalize_metadata({"language": "de"})


def test_language_detection_is_conservative() -> None:
    """Detection answers from function words, or it answers nothing.

    The German sample is longer than the others on purpose: coverage of a large
    stopword list needs more text than coverage of a small one, and the floor is
    what keeps a short sample from being answered with a guess.
    """

    english = (
        "This chapter argues that the commodity is not a thing but a social "
        "relation between people that takes the form of a thing. The argument is "
        "that what appears to be a relation between things is in fact a relation "
        "between people, and that this is not a mistake of the observer but a "
        "property of the world that the theory has to explain."
    ) * 3
    german = (
        "Dieses Kapitel argumentiert, dass die Ware nicht eine Sache, sondern ein "
        "gesellschaftliches Verhaeltnis zwischen Menschen ist, das die Form einer "
        "Sache annimmt. Daraus folgt, dass der Wert einer Ware nicht durch die "
        "Arbeit bestimmt wird, die in sie eingegangen ist, sondern durch die "
        "gesellschaftlich notwendige Arbeit zu ihrer Reproduktion. Wir haben "
        "gesehen, dass die Analyse des Fetischs nicht von der Analyse der Form "
        "getrennt werden kann, und dass die Form das ist, worum es der Theorie "
        "geht."
    ) * 6
    arabic = (
        "يناقش هذا الفصل أن السلعة ليست شيئا بل علاقة اجتماعية بين الناس تتخذ شكل "
        "الشيء، ويترتب على ذلك أن قيمة السلعة لا يحددها العمل الذي دخل فيها."
    ) * 3

    assert extraction_module._detect_language(english) == "en"
    assert extraction_module._detect_language(german) == "de"
    # A language outside the candidate set covers no list at all, even with enough
    # text to judge a language the lists do cover.
    assert extraction_module._detect_language(arabic) == ""
    assert extraction_module._detect_language("yes no maybe") == ""
