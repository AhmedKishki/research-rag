"""Documented font-glyph recovery for span cleaning and the native parser.

Two mechanisms are pinned here:

* `text_normalization.recover_formatting_glyphs` treats a control code or an
  undocumented private-use code from a recognised symbol, dingbat, icon, or
  mathematics face as a drawn glyph rather than corruption. CMEX10 exposes its
  large delimiters as codes 0x00 and 0x01, which a bad font map otherwise turns
  into replacement characters.
* `pdf_native.content.Font` reads a code from the font's own /Encoding
  /Differences when its ToUnicode map returns a control or private-use code
  point, but only where the encoding documents a known glyph.

The last group records a limit: the sampled private fonts name their glyphs
`UIback`, `UIforward`, `H17015`, `C211`, `C19`, and `C18`, none of which Adobe's
glyph list documents. No arbitrary font-name guess is made for them.
"""

from __future__ import annotations

import pytest

from research_rag.corpus import pdf_text_recovery as recovery
from research_rag.corpus.pdf_native import content as pdfcontent
from research_rag.corpus.text_normalization import (
    formatting_font_kind,
    is_known_formatting_glyph,
    recover_formatting_glyphs,
)

# ---------------------------------------------------------------- PDF builders


def _pdf(objects: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    for number, body in enumerate(objects, start=1):
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    out += b"trailer\n<< /Size %d >>\n%%%%EOF\n" % (len(objects) + 1)
    return bytes(out)


def _stream(content: bytes) -> bytes:
    return b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"


def _cmap(pairs: list[tuple[bytes, bytes]]) -> bytes:
    chars = b"".join(b"<%s> <%s>\n" % (src, dst) for src, dst in pairs)
    return (
        b"/CIDInit /ProcSet findresource begin 12 dict begin\n"
        b"begincmap\n1 beginbfchar\n" + chars + b"endbfchar\nendcmap\nend end"
    )


def _page_pdf(
    content: bytes,
    *,
    differences: bytes | None = None,
    cmap: bytes | None = None,
    base: bytes = b"/Helvetica",
) -> bytes:
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont " + base
    if differences is not None:
        font += b" /Encoding << /Type /Encoding /Differences " + differences + b" >>"
    if cmap is not None:
        font += b" /ToUnicode 6 0 R"
    font += b" >>"
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page,
        font,
        _stream(content),
    ]
    if cmap is not None:
        objects.append(_stream(cmap))
    return _pdf(objects)


WHOLE_PAGE = {0: (0.0, 0.0, 400.0, 600.0)}


def _write(tmp_path, content: bytes, **kwargs):
    path = tmp_path / "page.pdf"
    path.write_bytes(_page_pdf(content, **kwargs))
    return path


def _recover(tmp_path, content: bytes, **kwargs):
    recovered, reason = recovery.recover_pdf_blocks(
        _write(tmp_path, content, **kwargs), 1, WHOLE_PAGE
    )
    assert reason is None
    return recovered[0]


# ---------------------------------------------------------------- span cleaning


@pytest.mark.parametrize("font", ["CMEX10", "CMSY10", "CMMI10", "MSAM10", "MSBM10"])
def test_a_control_code_of_a_mathematics_face_is_formatting(font: str) -> None:
    # CMEX10 exposes code 0x00 and 0x01 where it drew a big parenthesis. The
    # span cleaner must remove such a code instead of leaving corruption.
    assert recover_formatting_glyphs("tr ΣT(θ)\x00\x01", font) == "tr ΣT(θ)  "


@pytest.mark.parametrize(
    "font",
    ["SymbolMT", "ZapfDingbats", "FontAwesome", "MaterialIcons", "CMEX10"],
)
def test_a_control_code_is_removed_from_every_recognised_face(font: str) -> None:
    assert recover_formatting_glyphs("a\x01b", font) == "a b"


def test_an_unknown_face_keeps_its_control_code() -> None:
    # The sampled UI and Pi fonts name glyphs Adobe does not document, so their
    # codes are preserved for the quality rules to judge, never trusted.
    assert recover_formatting_glyphs("\x01", "TT648O00") == "\x01"
    assert recover_formatting_glyphs("\x02", "AdvP4C4E59") == "\x02"


def test_layout_whitespace_controls_are_left_alone() -> None:
    # Tab, newline, carriage return, vertical tab and form feed carry line
    # structure, not a drawn glyph, even in a recognised face.
    assert recover_formatting_glyphs("a\tb\nc\rd\x0be\x0cf", "CMEX10") == (
        "a\tb\nc\rd\x0be\x0cf"
    )


def test_a_documented_symbol_glyph_is_still_mapped() -> None:
    assert recover_formatting_glyphs("\uf0b7 bullet", "SymbolMT") == "• bullet"
    assert recover_formatting_glyphs("\uf8ed", "CMEX10") == " "
    assert is_known_formatting_glyph("\uf0b7")


def test_an_unknown_private_use_glyph_is_still_kept() -> None:
    assert recover_formatting_glyphs("\uf16d", "university-press-fonts") == "\uf16d"
    assert recover_formatting_glyphs("\uf16d", "CMEX10") == " "


@pytest.mark.parametrize(
    ("font", "kind"),
    [
        ("SymbolMT", "symbol"),
        ("ZapfDingbats", "dingbat"),
        ("FontAwesome6Free-Solid", "icon"),
        ("OKJGIF+CMEX10", "math"),
        ("ABCXYZ+TT648O00", ""),
        ("Universal-NewswithCommPi", ""),
    ],
)
def test_the_recovery_family_comes_from_the_font_name(font: str, kind: str) -> None:
    assert formatting_font_kind(font) == kind


# ------------------------------------------------- documented glyph names


@pytest.mark.parametrize(
    ("name", "character"),
    [
        ("diamond", "\u2666"),
        ("copyright", "\u00a9"),
        ("fi", "fi"),
        ("acute", "\u00b4"),
        ("parenlefttp", "\uf8eb"),
        ("parenrighttp", "\uf8f6"),
        ("summation", "\u2211"),
        ("radical", "\u221a"),
        ("arrowright", "\u2192"),
        ("circleplus", "\u2295"),
        ("emptyset", "\u2205"),
        ("alpha", "\u03b1"),
        ("Omega", "\u2126"),
        ("infinity", "\u221e"),
    ],
)
def test_a_documented_glyph_name_decodes(name: str, character: str) -> None:
    assert pdfcontent.name_to_unicode(name) == character


def test_a_tex_only_glyph_name_decodes_to_nothing() -> None:
    # These names occur in TeX mathematics faces but not in Adobe's glyph list,
    # so they are left out rather than guessed at.
    for name in ("parenleftbig", "radicalbig", "summationdisplay", "hatwide"):
        assert pdfcontent.name_to_unicode(name) == ""


def test_an_undocumented_glyph_name_decodes_to_nothing() -> None:
    # Adobe's glyph list has no entry for any of the sampled private names.
    for name in ("UIback", "UIforward", "H17015", "C211", "C19", "C18"):
        assert pdfcontent.name_to_unicode(name) == ""


# ------------------------------------------------- native Font recovery


def test_differences_replace_a_control_code_from_a_broken_tounicode(tmp_path) -> None:
    # The ToUnicode map returns U+0001 for code 0x41; the encoding documents the
    # code as /diamond, so the drawn glyph is used.
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (AB) Tj ET",
            differences=b"[65 /diamond]",
            cmap=_cmap([(b"41", b"0001")]),
        )
        == "\u2666B"
    )


def test_differences_replace_a_private_use_code_from_a_broken_tounicode(
    tmp_path,
) -> None:
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (AB) Tj ET",
            differences=b"[65 /bullet]",
            cmap=_cmap([(b"41", b"F0B7")]),
        )
        == "\u2022B"
    )


def test_a_broken_tounicode_is_kept_without_an_encoding_name(tmp_path) -> None:
    # No /Differences documents the code, so the broken mapping is preserved
    # rather than replaced with a guessed character.
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (AB) Tj ET",
            cmap=_cmap([(b"41", b"0001")]),
        )
        == "\x01B"
    )


def test_an_undocumented_differences_name_stays_unreadable(tmp_path) -> None:
    # The encoding names the code, but the name is not documented, so the broken
    # ToUnicode mapping is kept rather than replaced with a guess.
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (AB) Tj ET",
            differences=b"[65 /not_a_known_glyph]",
            cmap=_cmap([(b"41", b"0001")]),
        )
        == "\x01B"
    )


def test_an_empty_tounicode_mapping_is_still_missing_text(tmp_path) -> None:
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (AB) Tj ET",
            differences=b"[65 /diamond]",
            cmap=_cmap([(b"41", b"")]),
        )
        == "\ufffdB"
    )


def test_cmex_delimiters_are_read_from_a_no_tounicode_encoding(tmp_path) -> None:
    # A CMEX-style face: no ToUnicode, and codes 0x00 and 0x01 named as the
    # delimiter pieces Adobe documents in its Corporate Use Subarea.
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (x\x00\x01) Tj ET",
            differences=b"[0 /parenlefttp /parenrighttp]",
        )
        == "x\uf8eb\uf8f6"
    )


def test_a_tex_only_glyph_is_blanked_in_a_recognised_face(tmp_path) -> None:
    # `hatwide` is not in Adobe's glyph list, but a mathematics face draws a
    # glyph there, so it is non-evidence rather than corruption.
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (x\x00y) Tj ET",
            differences=b"[0 /hatwide]",
            base=b"/XYZABC+CMEX10",
        )
        == "x y"
    )


def test_a_tex_only_glyph_stays_unreadable_in_an_unknown_face(tmp_path) -> None:
    # The same undocumented name in a font Adobe does not recognise is missing
    # text, not blanked, so a word is never manufactured.
    assert (
        _recover(
            tmp_path,
            b"BT /F1 24 Tf 50 500 Td (x\x00y) Tj ET",
            differences=b"[0 /hatwide]",
        )
        == "x\ufffdy"
    )
