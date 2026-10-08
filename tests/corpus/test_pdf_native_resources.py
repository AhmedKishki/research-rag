"""Resource bounds in the bundled native PDF object parser.

Adversarial small probes for the untrusted-input paths in
`research_rag.corpus.pdf_native.objects`: a compressed payload never decodes
past a budget, a predictor never allocates from file-declared dimensions, a
classic xref row count never loops past EOF, and a stream padded with newlines
trims in one pass. Valid streams keep decoding identically.
"""

from __future__ import annotations

import base64
import zlib

import pytest

from research_rag.corpus.pdf_native import objects
from research_rag.corpus.pdf_native.objects import Document, Name, Parser, Stream

# ------------------------------------------------------------ decoding budget


def test_a_valid_flate_stream_is_unchanged() -> None:
    payload = b"hello flate world" * 3
    compressed = zlib.compress(payload)

    assert objects.apply_filters({"Filter": Name("FlateDecode")}, compressed) == payload


def test_a_flate_bomb_is_refused_within_budget(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 1024)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 1024)
    bomb = zlib.compress(b"\x00" * (1024 * 1024))

    assert objects.apply_filters({"Filter": Name("FlateDecode")}, bomb) == b""


def test_a_stream_exactly_at_budget_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 1024)
    payload = b"x" * 1024

    assert (
        objects.apply_filters({"Filter": Name("FlateDecode")}, zlib.compress(payload))
        == payload
    )


def test_raw_deflate_without_a_zlib_header_is_decoded() -> None:
    engine = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = engine.compress(b"raw deflate text") + engine.flush()

    assert (
        objects.apply_filters({"Filter": Name("FlateDecode")}, raw)
        == b"raw deflate text"
    )


class _BudgetDocument:
    def __init__(self) -> None:
        self._decoded_bytes = 0

    def resolve(self, value):
        return value


def test_the_document_decode_budget_is_cumulative(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 8)
    monkeypatch.setattr(objects, "MAX_DOCUMENT_DECODED_BYTES", 8)
    doc = _BudgetDocument()
    first = Stream({"Filter": Name("FlateDecode")}, zlib.compress(b"A" * 8), doc)

    assert first.data == b"A" * 8
    second = Stream({"Filter": Name("FlateDecode")}, zlib.compress(b"B" * 8), doc)
    assert second.data == b""


def test_run_length_bomb_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(objects, "MAX_DECODED_STREAM_BYTES", 64)

    assert objects.rle_decode(b"\x81A" * 10) == b""


def test_a_valid_run_length_stream_is_unchanged() -> None:
    assert objects.rle_decode(b"\x02ABC") == b"ABC"


def test_ascii85_bomb_is_bounded() -> None:
    assert objects.a85_decode(b"z" * 10, limit=16) == b""


def test_a_valid_ascii85_stream_is_unchanged() -> None:
    assert objects.a85_decode(base64.a85encode(b"hello")) == b"hello"


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
    ],
)
def test_invalid_predictor_dimensions_fail_bounded(parm) -> None:
    assert objects.apply_predictor(b"\x00\x01\x02", parm) == b""


def test_a_predictor_row_larger_than_budget_fails() -> None:
    parm = {"Predictor": 12, "Colors": 3, "BitsPerComponent": 8, "Columns": 100}

    assert objects.apply_predictor(b"\x00" * 10, parm, limit=32) == b""


def test_a_valid_png_up_predictor_decodes() -> None:
    data = bytes([2, 0x10, 2, 0x05])
    parm = {"Predictor": 12, "Colors": 1, "BitsPerComponent": 8, "Columns": 1}

    assert objects.apply_predictor(data, parm) == b"\x10\x15"


def test_the_tiff_predictor_is_passthrough() -> None:
    parm = {"Predictor": 2, "Columns": 4}

    assert objects.apply_predictor(b"\x01\x02\x03\x04", parm) == b"\x01\x02\x03\x04"


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


def test_a_valid_classic_xref_is_loaded(tmp_path) -> None:
    path = tmp_path / "valid.pdf"
    path.write_bytes(_classic_xref_pdf())

    document = Document(str(path))

    assert document.xref.get(1) is not None
    assert len(document.pages()) == 1


def test_a_xref_row_count_beyond_eof_is_bounded(tmp_path) -> None:
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"xref\n"
        b"0 100000000000000000000\n"
        b"0000000000 65535 f \n"
        b"trailer\n<< /Size 1 /Root 1 0 R >>\n"
        b"startxref\n"
    )
    offset = body.index(b"xref")
    path = tmp_path / "huge-count.pdf"
    path.write_bytes(body + b"%d\n%%%%EOF\n" % offset)

    document = Document(str(path))

    assert isinstance(document, Document)


# ------------------------------------------------------------ stream trimming


def test_stream_trailing_eol_trim_is_single_pass() -> None:
    raw = b"stream" + b"\n" + b"HELLO" + b"\n" * 200000 + b"endstream"
    parser = Parser(raw)
    parser.pos = len(b"stream")

    assert parser._read_stream({}) == b"HELLO"
