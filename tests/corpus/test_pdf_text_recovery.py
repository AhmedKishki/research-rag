"""The bundled native PDF text recovery helper, and the parser port behind it.

Most tests build tiny PDFs with the standard library alone. One uses PyMuPDF to
produce a real file and its block boxes, which is the shape the caller passes.
A font with no ToUnicode mapping is read from the font's own encoding by the
bundled parser, where an inverse glyph map can fail.
"""

from __future__ import annotations

import pytest

from research_rag.corpus import pdf_text_recovery as module
from research_rag.corpus.pdf_native import content as pdfcontent
from research_rag.corpus.pdf_native import lines as pdflines
from research_rag.corpus.pdf_native.objects import Name, Stream

# ------------------------------------------------------------ PDF builders


def _pdf(objects: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    for number, body in enumerate(objects, start=1):
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    out += b"trailer\n<< /Size %d >>\n%%%%EOF\n" % (len(objects) + 1)
    return bytes(out)


def _stream(content: bytes) -> bytes:
    return b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"


def _font(body: bytes | None = None) -> bytes:
    if body is None:
        body = (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"
        )
    return body


def _page_pdf(
    content: bytes,
    *,
    font: bytes | None = None,
    rotate: int = 0,
    media: bytes = b"[0 0 400 600]",
) -> bytes:
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox "
        + media
        + b" /Rotate "
        + str(rotate).encode()
        + b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            _font(font),
            _stream(content),
        ]
    )


def _write(tmp_path, content: bytes, name: str = "page.pdf", **kwargs):
    path = tmp_path / name
    path.write_bytes(_page_pdf(content, **kwargs))
    return path


WHOLE_PAGE = {0: (0.0, 0.0, 400.0, 600.0)}


# ------------------------------------------------------------ recovery


def test_recovers_text_from_a_stdlib_pdf(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Recovered alpha text) Tj ET")

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0] == "Recovered alpha text"


def test_a_scan_cache_parses_each_pdf_once(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Recovered alpha text) Tj ET")
    opener = module.open_doc
    calls = []

    def counted_open(source):
        calls.append(source)
        return opener(source)

    monkeypatch.setattr(module, "open_doc", counted_open)
    cache = {}
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE, cache)[0]
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE, cache)[0]
    assert calls == [path]


def test_a_shared_cache_is_released_when_closed(tmp_path, monkeypatch) -> None:
    """A caller-owned cache is reusable, and closing it releases the parse."""

    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Recovered alpha text) Tj ET")
    opener = module.open_doc
    calls = []

    def counted_open(source):
        calls.append(source)
        return opener(source)

    monkeypatch.setattr(module, "open_doc", counted_open)
    cache = {}
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE, cache)[0]
    assert cache
    module.close_pdf_recovery_cache(cache)
    assert cache == {}
    # A released cache reopens on the next request rather than serving stale bytes.
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE, cache)[0]
    assert calls == [path, path]


def test_line_assembly_failure_is_nonfatal(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Recovered alpha text) Tj ET")

    def broken_lines(_chunks):
        raise ValueError("bad line geometry")

    monkeypatch.setattr(module, "lines_from_chunks", broken_lines)
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE) == ({}, module.REASON_FAILED)


def test_unknown_glyphs_are_not_silently_dropped(tmp_path) -> None:
    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding << /Differences [65 /not_a_known_glyph] >> >>"
    )
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (xAword) Tj ET", font=font)
    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)
    assert reason is None
    assert recovered[0] == "x\ufffdword"


def test_unknown_cids_remain_unreadable() -> None:
    class FakeDocument:
        def resolve(self, value):
            return value

    font = pdfcontent.Font(FakeDocument(), {"Subtype": "Type0"})
    assert font.decode(b"\x00A\x00B") == "\ufffd\ufffd"
    assert font.decode(b"\x00A\x00") == "\ufffd\ufffd"


def test_empty_tounicode_mapping_is_missing_text(tmp_path) -> None:
    content = b"BT /F1 24 Tf 50 500 Td (ab) Tj ET"
    cmap = b"2 beginbfchar <61> <0061> <62> <> endbfchar"
    path = tmp_path / "empty-map.pdf"
    path.write_bytes(
        _pdf(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                (
                    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] "
                    b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
                ),
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
                _stream(content),
                _stream(cmap),
            ]
        )
    )
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE)[0][0] == "a\ufffd"


@pytest.mark.parametrize(
    ("name", "character"),
    [("germandbls", "ß"), ("bullet", "•"), ("dagger", "†"), ("periodcentered", "·")],
)
def test_named_symbols_are_not_lossily_folded(name, character) -> None:
    assert pdfcontent.name_to_unicode(name) == character


def test_declared_winansi_encoding_keeps_ascii_quotes(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (it's exact) Tj ET")
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE)[0][0] == "it's exact"


def test_undefined_winansi_byte_remains_unreadable(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (a\x81b) Tj ET")
    assert module.recover_pdf_blocks(path, 1, WHOLE_PAGE)[0][0] == "a\ufffdb"


def test_a_filename_with_spaces_and_parentheses_is_read(tmp_path) -> None:
    path = _write(
        tmp_path,
        b"BT /F1 18 Tf 40 500 Td (Odd name works) Tj ET",
        name="a page (unusual).pdf",
    )

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0] == "Odd name works"


def test_no_tounicode_glyph_mapping_is_decoded(tmp_path) -> None:
    """A differing encoding with no ToUnicode is read from the glyph names.

    The inverse map a PDF reader needs is absent here; the bundled parser reads
    the encoding directly, which is the recovery this helper exists for.
    """

    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding << /Type /Encoding /Differences [48 /A /B /C] >> >>"
    )
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (012) Tj ET", font=font)

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0] == "ABC"


def test_line_breaks_are_kept(tmp_path) -> None:
    path = _write(
        tmp_path,
        b"BT /F1 12 Tf 50 500 Td (first line) Tj 0 -12 Td (second line) Tj ET",
    )

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0] == "first line\nsecond line"


def test_only_requested_blocks_are_returned(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Top block) Tj ET")

    recovered, reason = module.recover_pdf_blocks(
        path, 1, {0: (0.0, 0.0, 400.0, 300.0), 1: (0.0, 300.0, 400.0, 600.0)}
    )

    assert reason is None
    assert set(recovered) == {0}
    assert recovered[0] == "Top block"


def test_a_run_goes_to_the_smallest_box_only(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Only text) Tj ET")

    recovered, reason = module.recover_pdf_blocks(
        path, 1, {0: (40.0, 80.0, 300.0, 120.0), 1: WHOLE_PAGE[0]}
    )

    assert reason is None
    assert recovered[0] == "Only text"
    assert 1 not in recovered


def test_a_block_without_alphanumeric_text_is_omitted(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (####%%) Tj ET")

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered == {}


def test_a_blank_page_returns_no_text(tmp_path) -> None:
    path = _write(tmp_path, b"0 0 1 rg 0 0 100 100 re f")

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert recovered == {}
    assert reason is None


def test_an_empty_page_box_list_returns_nothing(tmp_path) -> None:
    path = tmp_path / "absent.pdf"

    assert module.recover_pdf_blocks(path, 1, {}) == ({}, None)


def test_unsupported_rotation_is_refused(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (Rotated) Tj ET", rotate=90)

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert recovered == {}
    assert reason == module.REASON_UNSUPPORTED_ROTATION


def test_a_missing_source_is_unavailable(tmp_path) -> None:
    recovered, reason = module.recover_pdf_blocks(
        tmp_path / "absent.pdf", 1, WHOLE_PAGE
    )

    assert recovered == {}
    assert reason == module.REASON_UNAVAILABLE


def test_a_page_beyond_the_document_is_failed(tmp_path) -> None:
    path = _write(tmp_path, b"BT /F1 24 Tf 50 500 Td (One page) Tj ET")

    recovered, reason = module.recover_pdf_blocks(path, 2, WHOLE_PAGE)

    assert recovered == {}
    assert reason == module.REASON_FAILED


def test_malformed_bytes_are_failed_not_fatal(tmp_path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a PDF at all")

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert recovered == {}
    assert reason == module.REASON_FAILED


@pytest.mark.parametrize("page_number", [0, -1, "1", 1.5, True, None])
def test_a_bad_page_number_fails_without_opening(tmp_path, page_number) -> None:
    recovered, reason = module.recover_pdf_blocks(
        tmp_path / "absent.pdf", page_number, WHOLE_PAGE
    )

    assert recovered == {}
    assert reason == module.REASON_FAILED


def test_non_finite_coordinates_fail_without_opening(tmp_path) -> None:
    recovered, reason = module.recover_pdf_blocks(
        tmp_path / "absent.pdf", 1, {0: (0.0, 0.0, float("inf"), 1.0)}
    )

    assert recovered == {}
    assert reason == module.REASON_FAILED


# ------------------------------------------------------------ real PDF


def test_pymupdf_block_boxes_are_matched(tmp_path) -> None:
    """The caller's boxes come from PyMuPDF, so one real file is checked."""

    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "real page.pdf"
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    page.insert_text((50, 100), "Alpha block first line", fontsize=12)
    page.insert_text((50, 120), "Alpha block second line", fontsize=12)
    page.insert_text((50, 300), "Beta block separate", fontsize=12)
    document.save(str(path))
    boxes = {
        index: tuple(block["bbox"])
        for index, block in enumerate(page.get_text("dict")["blocks"])
        if block.get("type") == 0
    }
    document.close()

    recovered, reason = module.recover_pdf_blocks(path, 1, boxes)

    assert reason is None
    assert recovered[0] == "Alpha block first line"
    assert recovered[1] == "Alpha block second line"
    assert recovered[2] == "Beta block separate"


def test_whole_page_recovery_reads_two_columns_in_column_order(tmp_path) -> None:
    """A whole-page fallback must not interleave the two columns.

    Each page here has no text layer PyMuPDF accepts, so the caller asks for the
    whole page. The bundled reader must return the left column top-down, then
    the right column top-down, not baseline by baseline across both.
    """

    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "two-column.pdf"
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    left = [f"Left column evidence {number}" for number in range(1, 9)]
    right = [f"Right column evidence {number}" for number in range(1, 9)]
    for number, text in enumerate(left):
        page.insert_text((40, 100 + number * 16), text, fontsize=11)
    for number, text in enumerate(right):
        # A few points lower, so the two columns do not share a baseline.
        page.insert_text((225, 106 + number * 16), text, fontsize=11)
    document.save(str(path))
    document.close()

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    lines = recovered[0].splitlines()
    assert lines == [*left, *right]


def test_whole_page_recovery_separates_columns_on_shared_baselines(tmp_path) -> None:
    """Two columns on identical baselines are not welded before separation.

    `group_lines` would merge each row's left and right runs into one line. The
    gutter between the columns must split the regions first, so the output is
    column-major, not row-major.
    """

    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "aligned-columns.pdf"
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    page.insert_text((40, 100), "Left first sentence.", fontsize=12)
    page.insert_text((225, 100), "Right first sentence.", fontsize=12)
    page.insert_text((40, 130), "Left second sentence.", fontsize=12)
    page.insert_text((225, 130), "Right second sentence.", fontsize=12)
    document.save(str(path))
    document.close()

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0].splitlines() == [
        "Left first sentence.",
        "Left second sentence.",
        "Right first sentence.",
        "Right second sentence.",
    ]


def test_whole_page_recovery_keeps_a_full_width_heading_whole(tmp_path) -> None:
    """A heading spanning both columns stays one line above the split columns."""

    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "heading-and-columns.pdf"
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    heading = "A full width heading spanning the entire page here"
    page.insert_text((40, 60), heading, fontsize=15)
    left = [f"Left body line {number}" for number in range(1, 4)]
    right = [f"Right body line {number}" for number in range(1, 4)]
    for number, text in enumerate(left):
        page.insert_text((40, 120 + number * 16), text, fontsize=11)
        page.insert_text((225, 120 + number * 16), right[number], fontsize=11)
    document.save(str(path))
    document.close()

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0].splitlines() == [heading, *left, *right]


def test_whole_page_recovery_keeps_single_column_order(tmp_path) -> None:
    """A single-column page keeps plain top-down baseline order, unsplit."""

    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "one-column.pdf"
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    lines = [f"Single column evidence {number}" for number in range(1, 11)]
    for number, text in enumerate(lines):
        page.insert_text((40, 80 + number * 16), text, fontsize=11)
    document.save(str(path))
    document.close()

    recovered, reason = module.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert reason is None
    assert recovered[0].splitlines() == lines


# ------------------------------------------------------------ parser port


class _FakeDocument:
    def __init__(self, content):
        self.content = content

    def page_content(self, page):
        return self.content

    def resolve(self, value):
        return value


def _extract(content):
    document = _FakeDocument(content)
    page = {"Resources": {"Font": {"F": {"BaseFont": Name("Helvetica")}}}}
    return pdfcontent.extract_page(document, page)


def test_nested_form_resources_and_matrix() -> None:
    document = _FakeDocument(b"q 1 0 0 1 100 200 cm /A Do Q")
    form = Stream(
        {"Subtype": Name("Form"), "Matrix": [1, 0, 0, 1, 20, 30]},
        b"BT /F 10 Tf 1 0 0 1 50 500 Tm (Form text) Tj ET",
        document,
    )
    page = {
        "Resources": {
            "Font": {"F": {"BaseFont": Name("Helvetica")}},
            "XObject": {"A": form},
        }
    }

    chunks = pdfcontent.extract_page(document, page)

    assert chunks[0].text == "Form text"
    assert (chunks[0].x, chunks[0].y) == (170, 730)


def test_recursive_form_guard() -> None:
    document = _FakeDocument(b"/A Do")
    resources = {"Font": {"F": {"BaseFont": Name("Helvetica")}}, "XObject": {}}
    form = Stream(
        {"Subtype": Name("Form"), "Resources": resources},
        b"BT /F 10 Tf (once) Tj ET /A Do",
        document,
    )
    resources["XObject"]["A"] = form

    chunks = pdfcontent.extract_page(document, {"Resources": resources})

    assert [chunk.text for chunk in chunks] == ["once"]


def test_punctuation_and_numbers() -> None:
    chunks = _extract(b"BT /F 10 Tf 1 0 0 1 50 500 Tm (Year 2022: 3.14!) Tj ET")

    assert chunks[0].text == "Year 2022: 3.14!"


def test_tj_preserves_line_matrix() -> None:
    chunks = _extract(
        b"BT /F 10 Tf 1 0 0 1 50 500 Tm [(First)] TJ 0 -12 Td [(Second)] TJ ET"
    )

    assert [chunk.x for chunk in chunks] == [50, 50]
    assert [chunk.y for chunk in chunks] == [500, 488]


def test_rise_is_not_rendering_mode() -> None:
    chunks = _extract(b"BT /F 10 Tf 1 0 0 1 50 500 Tm 3 Tr (ab) Tj 4 Ts (cd) Tj ET")

    assert chunks[0].y == 500
    assert chunks[1].y == 504


def test_small_type_is_not_superscript() -> None:
    chunks = _extract(
        b"BT /F 5 Tf 1 0 0 1 50 500 Tm "
        b"[(small)-400(type)] TJ 0 -6 Td (another line) Tj ET"
    )

    lines = pdflines.group_lines(chunks)

    assert len(lines) == 2
    assert pdflines.merge_line_marked(lines[0].chunks) == "small type"
    assert pdflines.SUP_MARK not in pdflines.merge_line_marked(lines[1].chunks)
