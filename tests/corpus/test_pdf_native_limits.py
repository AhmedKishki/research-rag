"""Resource limits in the bundled native PDF object parser.

Adversarial probes for the untrusted-input paths in
``research_rag.corpus.pdf_native.objects``: a declared row count, an encoded file
size, a filter chain, a stream decode, and a predictor never allocate from a
value the file chose. Constants are monkeypatched very low so no test builds a
large payload. Valid small streams keep decoding identically, and the recovery
helper turns a resource failure into a bounded reason instead of raising.
"""

from __future__ import annotations

import zlib

import pytest

from research_rag.corpus import pdf_text_recovery as recovery
from research_rag.corpus.pdf_native import objects
from research_rag.corpus.pdf_native.extract import open_doc
from research_rag.corpus.pdf_native.objects import Document, Name, Parser, Stream
from tests.corpus.test_pdf_text_recovery import WHOLE_PAGE, _page_pdf, _pdf

# ------------------------------------------------------------ interface


def test_the_resource_limits_are_positive_and_consistent() -> None:
    assert issubclass(objects.PdfResourceLimitError, ValueError)
    # Encoded-file, filter-chain, and xref caps are fixed safety boundaries.
    assert objects.MAX_NATIVE_DOCUMENT_BYTES == 256 * 1024 * 1024
    assert objects.MAX_FILTER_CHAIN == 16
    assert objects.MAX_XREF_ENTRIES == 1_000_000
    # Decode budgets are implementation defaults; only their safety ordering is
    # pinned here, so one stream can never be allowed more than the document.
    assert objects.MAX_DECODED_STREAM_BYTES > 0
    assert objects.MAX_DOCUMENT_DECODED_BYTES > 0
    assert objects.MAX_DECODED_STREAM_BYTES <= objects.MAX_DOCUMENT_DECODED_BYTES


# ------------------------------------------------------------ encoded file cap


def test_a_document_larger_than_the_encoded_cap_is_refused(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(objects, "MAX_NATIVE_DOCUMENT_BYTES", 64)
    path = tmp_path / "big.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"x" * 128)

    with pytest.raises(objects.PdfResourceLimitError):
        Document(str(path))


# ------------------------------------------------------------ filter chain


def test_a_filter_chain_longer_than_the_cap_is_refused() -> None:
    filters = [Name("FlateDecode")] * (objects.MAX_FILTER_CHAIN + 1)

    with pytest.raises(objects.PdfResourceLimitError):
        objects.apply_filters({"Filter": filters}, b"")


def test_a_filter_chain_at_the_cap_is_not_refused() -> None:
    filters = [Name("FlateDecode")] * objects.MAX_FILTER_CHAIN

    assert objects.apply_filters({"Filter": filters}, b"") == b""


# ------------------------------------------------------------ bounded decoders


class _BudgetDocument:
    def __init__(self) -> None:
        self._decoded_bytes = 0

    def resolve(self, value):
        return value


def test_a_flate_bomb_is_refused_within_the_stream_budget(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 1024)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 1024 * 1024)
    bomb = zlib.compress(b"\x00" * (1024 * 1024))

    assert objects.apply_filters({"Filter": Name("FlateDecode")}, bomb) == b""


def test_a_flate_stream_exactly_at_the_budget_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 1024)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 1024)
    payload = b"x" * 1024

    assert (
        objects.apply_filters({"Filter": Name("FlateDecode")}, zlib.compress(payload))
        == payload
    )


def test_raw_deflate_without_a_zlib_header_decodes() -> None:
    engine = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    payload = b"raw deflate text"
    raw = engine.compress(payload) + engine.flush()

    assert objects.apply_filters({"Filter": Name("FlateDecode")}, raw) == payload


def test_a_truncated_deflate_tail_keeps_the_bounded_prefix() -> None:
    payload = b"ABCDEFGHIJ" * 100
    compressed = zlib.compress(payload)
    truncated = compressed[: len(compressed) // 2]

    decoded = objects.apply_filters({"Filter": Name("FlateDecode")}, truncated)

    assert 0 < len(decoded) < len(payload)
    assert decoded == payload[: len(decoded)]


def test_an_ascii_hex_stream_decodes() -> None:
    assert (
        objects.apply_filters({"Filter": Name("ASCIIHexDecode")}, b"48656C6C6F>")
        == b"Hello"
    )


def _pack_lzw_codes(codes: list[int], width: int = 9) -> bytes:
    """Pack 9-bit LZW codes most-significant bit first, as the decoder reads."""

    bits = 0
    nbits = 0
    out = bytearray()
    for code in codes:
        bits = (bits << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((bits >> nbits) & 0xFF)
    if nbits:
        out.append((bits << (8 - nbits)) & 0xFF)
    return bytes(out)


def test_a_valid_lzw_stream_decodes() -> None:
    assert objects.lzw_decode(_pack_lzw_codes([65] * 5), early=1, limit=100) == b"AAAAA"


def test_an_lzw_bomb_is_refused_within_the_budget() -> None:
    stream = _pack_lzw_codes([65] * 20)

    assert objects.lzw_decode(stream, early=1, limit=8) == b""


def test_a_run_length_bomb_is_refused_within_the_budget() -> None:
    assert objects.rle_decode(b"\x81A" * 10, limit=64) == b""


def test_a_valid_run_length_stream_decodes() -> None:
    assert objects.rle_decode(b"\x02ABC") == b"ABC"


def test_an_ascii85_bomb_is_refused_within_the_budget() -> None:
    assert objects.a85_decode(b"z" * 10, limit=16) == b""


def test_a_valid_ascii85_stream_decodes() -> None:
    import base64

    assert objects.a85_decode(base64.a85encode(b"hello")) == b"hello"


# ------------------------------------------------------------ document work


def test_the_document_budget_spans_distinct_streams(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 64)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 12)
    doc = _BudgetDocument()
    first = Stream({"Filter": Name("FlateDecode")}, zlib.compress(b"A" * 4), doc)
    second = Stream({"Filter": Name("FlateDecode")}, zlib.compress(b"B" * 6), doc)

    assert first.data == b"A" * 4
    assert second.data == b"B" * 6
    assert doc._decoded_bytes == 10

    exhausted = Stream({"Filter": Name("FlateDecode")}, zlib.compress(b"C" * 4), doc)

    assert exhausted.data == b""
    assert doc._decoded_bytes == 10


def test_each_filter_stage_accounts_toward_document_work(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 1024 * 1024)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 1024 * 1024)
    doc = _BudgetDocument()
    payload = b"Hello world"
    compressed = zlib.compress(payload)
    stream = Stream(
        {
            "Filter": [Name("ASCIIHexDecode"), Name("FlateDecode")],
        },
        compressed.hex().encode("ascii"),
        doc,
    )

    assert stream.data == payload
    assert doc._decoded_bytes == len(compressed) + len(payload)


def test_cached_stream_data_is_not_charged_twice(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 1024 * 1024)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 1024 * 1024)
    doc = _BudgetDocument()
    stream = Stream({"Filter": Name("FlateDecode")}, zlib.compress(b"cached"), doc)

    first = stream.data
    charged = doc._decoded_bytes
    second = stream.data

    assert first == b"cached"
    assert second is first
    assert doc._decoded_bytes == charged


# ------------------------------------------------------------ predictor


@pytest.mark.parametrize(
    "parm",
    [
        {"Predictor": 12, "Columns": 10**9},
        {"Predictor": 12, "Columns": 1, "Colors": "1"},
        {"Predictor": 12, "Columns": 1, "BitsPerComponent": 3},
        {"Predictor": 12, "Columns": 0},
        {"Predictor": 12, "Columns": -5},
        {"Predictor": 12, "Columns": 1, "Colors": 10**6},
        {"Predictor": 12, "Columns": 1, "BitsPerComponent": True},
    ],
)
def test_invalid_predictor_dimensions_fail_bounded(parm) -> None:
    assert objects.apply_predictor(b"\x00\x01\x02", parm) == b""


def test_a_predictor_row_larger_than_the_budget_fails() -> None:
    parm = {"Predictor": 12, "Colors": 3, "BitsPerComponent": 8, "Columns": 100}

    assert objects.apply_predictor(b"\x00" * 10, parm, limit=32) == b""


def test_a_predictor_checks_row_completeness_before_allocating(monkeypatch) -> None:
    # A large but allowed row must not be sized from the file for one byte of
    # input: the whole-row check runs before the previous-row buffer exists.
    parm = {
        "Predictor": 12,
        "Colors": 1,
        "BitsPerComponent": 8,
        "Columns": 1 << 20,
    }

    def forbidden_allocation(*_args, **_kwargs):
        pytest.fail("Incomplete predictor input must not allocate a row buffer")

    monkeypatch.setattr(objects, "bytearray", forbidden_allocation, raising=False)

    assert objects.apply_predictor(b"", parm) == b""
    assert objects.apply_predictor(b"\x00", parm) == b""


def test_a_predictor_rejects_an_unsupported_png_selector() -> None:
    parm = {"Predictor": 12, "Colors": 1, "BitsPerComponent": 8, "Columns": 2}

    assert objects.apply_predictor(bytes([5, 1, 2]), parm) == b""


def test_a_predictor_rejects_a_partial_trailing_row() -> None:
    parm = {"Predictor": 12, "Colors": 1, "BitsPerComponent": 8, "Columns": 2}
    full = bytes([0, 1, 2])
    partial = full + bytes([0, 3])

    assert objects.apply_predictor(full, parm) == b"\x01\x02"
    assert objects.apply_predictor(partial, parm) == b""


def test_a_valid_png_up_predictor_decodes() -> None:
    parm = {"Predictor": 12, "Colors": 1, "BitsPerComponent": 8, "Columns": 1}

    assert objects.apply_predictor(bytes([2, 0x10, 2, 0x05]), parm) == b"\x10\x15"


def test_the_tiff_predictor_is_passthrough() -> None:
    assert (
        objects.apply_predictor(b"\x01\x02\x03\x04", {"Predictor": 2, "Columns": 4})
        == b"\x01\x02\x03\x04"
    )


# ------------------------------------------------------------ xref tables


def _classic_xref_pdf() -> bytes:
    header = b"%PDF-1.4\n"
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>",
    ]
    offsets = {}
    payload = b""
    for number, body in enumerate(bodies, start=1):
        offsets[number] = len(header) + len(payload)
        payload += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_offset = len(header) + len(payload)
    xref = b"xref\n0 %d\n" % (len(offsets) + 1)
    xref += b"0000000000 65535 f \n"
    for number in range(1, len(offsets) + 1):
        xref += b"%010d 00000 n \n" % offsets[number]
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(offsets) + 1,
        xref_offset,
    )
    return header + payload + xref + trailer


def _bare_document(buf: bytes) -> Document:
    doc = Document.__new__(Document)
    doc.buf = buf
    doc.xref = {}
    doc.compressed = {}
    doc.trailer = {}
    doc.cache = {}
    doc._objstm_cache = {}
    doc._extra = {}
    doc._objstm_indexed = False
    doc._loading = set()
    doc._decoded_bytes = 0
    return doc


def test_a_valid_classic_xref_is_loaded(tmp_path) -> None:
    path = tmp_path / "valid.pdf"
    path.write_bytes(_classic_xref_pdf())

    document = Document(str(path))

    assert document.xref.get(1) is not None
    assert len(document.pages()) == 1


def test_an_impossible_classic_xref_count_returns_false() -> None:
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"xref\n"
        b"0 100000000000000000000\n"
        b"0000000000 65535 f \n"
    )
    document = _bare_document(body)
    offset = body.index(b"xref")

    assert document._load_xref_at(offset) is False


def test_a_classic_xref_count_beyond_the_entry_cap_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_XREF_ENTRIES", 4)
    xref = b"xref\n0 10\n"
    for _ in range(10):
        xref += b"0000000000 65535 f \n"
    body = xref + b"trailer\n<< /Size 11 >>\n"
    document = _bare_document(body)

    assert document._load_xref_at(0) is False


def test_a_zero_width_stream_xref_returns_false() -> None:
    body = (
        b"1 0 obj\n"
        b"<< /Type /XRef /W [0 0 0] /Index [0 1] /Size 1 /Length 0 >>\n"
        b"stream\n\nendstream\nendobj\n"
    )
    document = _bare_document(body)

    assert document._load_xref_at(0) is False


def test_a_stream_xref_with_impossible_counts_returns_false() -> None:
    body = (
        b"1 0 obj\n"
        b"<< /Type /XRef /W [1 2 1] /Index [0 999999999] /Size 1 /Length 0 >>\n"
        b"stream\n\nendstream\nendobj\n"
    )
    document = _bare_document(body)

    assert document._load_xref_at(0) is False


# ------------------------------------------------------------ recovery


def _objstm_pdf(chain: int) -> bytes:
    filters = b" ".join([b"/FlateDecode"] * chain)
    body = (
        b"<< /Type /ObjStm /N 0 /First 0 /Length 4 /Filter ["
        + filters
        + b"] >>\nstream\nDATA\nendstream"
    )
    return _pdf([body])


def test_a_resource_failure_during_objstm_scanning_propagates(tmp_path) -> None:
    path = tmp_path / "objstm.pdf"
    path.write_bytes(_objstm_pdf(objects.MAX_FILTER_CHAIN + 1))

    with pytest.raises(objects.PdfResourceLimitError):
        open_doc(path)


def test_recovery_turns_a_resource_failure_into_a_bounded_reason(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "objstm.pdf"
    original = _objstm_pdf(objects.MAX_FILTER_CHAIN + 1)
    path.write_bytes(original)

    recovered, reason = recovery.recover_pdf_blocks(path, 1, WHOLE_PAGE)

    assert recovered == {}
    assert reason == recovery.REASON_FAILED
    assert path.read_bytes() == original


def test_recovery_catches_a_resource_failure_while_opening(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "page.pdf"
    path.write_bytes(_page_pdf(b"BT /F1 24 Tf 50 500 Td (Text) Tj ET"))

    def refuse(_source):
        raise objects.PdfResourceLimitError("cap")

    monkeypatch.setattr(recovery, "open_doc", refuse)

    assert recovery.recover_pdf_blocks(path, 1, WHOLE_PAGE) == (
        {},
        recovery.REASON_FAILED,
    )


# ------------------------------------------------------------ stream trimming


def test_a_missing_length_stream_trims_only_trailing_eol() -> None:
    content = b"LINE1\nLINE2"
    raw = b"stream\n" + content + b"\n\n\nendstream"
    parser = Parser(raw)
    parser.pos = len(b"stream")

    assert parser._read_stream({}) == content


def test_a_declared_length_stream_is_sliced_exactly() -> None:
    raw = b"stream\nHELLO\nendstream"
    parser = Parser(raw)
    parser.pos = len(b"stream")

    assert parser._read_stream({"Length": 5}) == b"HELLO"
