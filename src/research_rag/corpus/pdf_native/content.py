"""Extract positioned text runs from PDF content streams.

The stream-level half of text extraction. A content stream is a program that
moves a cursor and shows strings; this module runs it and reports what was
shown and where:

    chunks = extract_page(doc, page)

Each `Chunk` is one shown run with its device-space origin, rendered size, the
advance along its own baseline, and the direction of that baseline. Grouping
runs into lines is `text.lines`; grouping pages into a document is
`text.extract`.

The operators are interpreted as PDF 32000-1 8-9 define them, including the
distinction that matters most here and is most often got wrong: showing text
advances the *text* matrix, while only `Tm`, `Td`, `TD` and `T*` move the
*text-line* matrix.

Adapted from the user's own AhmedKishki/pdf-tools project; the code is reused
here with the owner's permission. Image decoding, rendering, OCR and image
placement are omitted: this package recovers text only.
"""

import re

from .geometry import mat_mul
from .objects import Name, Stream, decode_name

# ---------------------------------------------------------------- ToUnicode


def _utf16be(hexs):
    try:
        b = bytes.fromhex(hexs.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not b:
        return ""
    if len(b) % 2:
        b += b"\x00"
    return b.decode("utf-16-be", "replace").replace("\x00", "")


def parse_cmap(data):
    """code (int) -> unicode string, from bfchar and bfrange sections."""
    m = {}
    for blk in re.finditer(rb"beginbfchar(.*?)endbfchar", data, re.DOTALL):
        for mm in re.finditer(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", blk.group(1)):
            d = _utf16be(mm.group(2))
            if d is not None:
                m[int(mm.group(1), 16)] = d
    for blk in re.finditer(rb"beginbfrange(.*?)endbfrange", data, re.DOTALL):
        body = blk.group(1)
        # array form: <lo> <hi> [ <d1> <d2> ... ]
        for mm in re.finditer(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]", body, re.DOTALL
        ):
            lo, hi = int(mm.group(1), 16), int(mm.group(2), 16)
            for i, it in enumerate(re.findall(rb"<([0-9A-Fa-f]*)>", mm.group(3))):
                if lo + i > hi:
                    break
                d = _utf16be(it)
                if d is not None:
                    m[lo + i] = d
        # consecutive form: <lo> <hi> <dststart>
        for mm in re.finditer(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", body
        ):
            lo, hi = int(mm.group(1), 16), int(mm.group(2), 16)
            base = mm.group(3)
            if len(base) < 4:
                continue
            tail = int(base[-4:], 16)
            for i in range(min(hi - lo + 1, 65536)):
                if tail + i <= 0x10FFFF:
                    m[lo + i] = chr(tail + i)
    return m


# ---------------------------------------------------------------- encodings

_CP1252 = {}


def _cp1252_map():
    if not _CP1252:
        for i in range(32, 256):
            try:
                _CP1252[i] = bytes([i]).decode("cp1252")
            except UnicodeDecodeError:
                _CP1252[i] = "\ufffd"
    return _CP1252


GLYPH_NAMES = {}
for _c, _n in [
    (32, "space"),
    (33, "exclam"),
    (34, "quotedbl"),
    (35, "numbersign"),
    (36, "dollar"),
    (37, "percent"),
    (38, "ampersand"),
    (39, "quoteright"),
    (40, "parenleft"),
    (41, "parenright"),
    (42, "asterisk"),
    (43, "plus"),
    (44, "comma"),
    (45, "hyphen"),
    (46, "period"),
    (47, "slash"),
    (48, "zero"),
    (49, "one"),
    (50, "two"),
    (51, "three"),
    (52, "four"),
    (53, "five"),
    (54, "six"),
    (55, "seven"),
    (56, "eight"),
    (57, "nine"),
    (58, "colon"),
    (59, "semicolon"),
    (60, "less"),
    (61, "equal"),
    (62, "greater"),
    (63, "question"),
    (64, "at"),
    (91, "bracketleft"),
    (92, "backslash"),
    (93, "bracketright"),
    (94, "asciicircum"),
    (95, "underscore"),
    (96, "quoteleft"),
    (123, "braceleft"),
    (124, "bar"),
    (125, "braceright"),
    (126, "asciitilde"),
    (160, "space"),
    (161, "exclamdown"),
    (162, "cent"),
    (163, "sterling"),
    (164, "currency"),
    (165, "yen"),
    (166, "brokenbar"),
    (167, "section"),
    (168, "dieresis"),
    (169, "copyright"),
    (170, "ordfeminine"),
    (171, "guillemotleft"),
    (172, "logicalnot"),
    (173, "hyphen"),
    (174, "registered"),
    (175, "macron"),
    (176, "degree"),
    (177, "plusminus"),
    (178, "twosuperior"),
    (179, "threesuperior"),
    (180, "acute"),
    (181, "mu"),
    (182, "paragraph"),
    (183, "periodcentered"),
    (184, "cedilla"),
    (185, "onesuperior"),
    (186, "ordmasculine"),
    (187, "guillemotright"),
    (188, "onequarter"),
    (189, "onehalf"),
    (190, "threequarters"),
    (191, "questiondown"),
    (192, "Agrave"),
    (193, "Aacute"),
    (194, "Acircumflex"),
    (195, "Atilde"),
    (196, "Adieresis"),
    (197, "Aring"),
    (198, "AE"),
    (199, "Ccedilla"),
    (200, "Egrave"),
    (201, "Eacute"),
    (202, "Ecircumflex"),
    (203, "Edieresis"),
    (204, "Igrave"),
    (205, "Iacute"),
    (206, "Icircumflex"),
    (207, "Idieresis"),
    (208, "Eth"),
    (209, "Ntilde"),
    (210, "Ograve"),
    (211, "Oacute"),
    (212, "Ocircumflex"),
    (213, "Otilde"),
    (214, "Odieresis"),
    (215, "multiply"),
    (216, "Oslash"),
    (217, "Ugrave"),
    (218, "Uacute"),
    (219, "Ucircumflex"),
    (220, "Udieresis"),
    (221, "Yacute"),
    (222, "Thorn"),
    (223, "germandbls"),
    (224, "agrave"),
    (225, "aacute"),
    (226, "acircumflex"),
    (227, "atilde"),
    (228, "adieresis"),
    (229, "aring"),
    (230, "ae"),
    (231, "ccedilla"),
    (232, "egrave"),
    (233, "eacute"),
    (234, "ecircumflex"),
    (235, "edieresis"),
    (236, "igrave"),
    (237, "iacute"),
    (238, "icircumflex"),
    (239, "idieresis"),
    (240, "eth"),
    (241, "ntilde"),
    (242, "ograve"),
    (243, "oacute"),
    (244, "ocircumflex"),
    (245, "otilde"),
    (246, "odieresis"),
    (247, "divide"),
    (248, "oslash"),
    (249, "ugrave"),
    (250, "uacute"),
    (251, "ucircumflex"),
    (252, "udieresis"),
    (253, "yacute"),
    (254, "thorn"),
    (255, "ydieresis"),
]:
    GLYPH_NAMES[_c] = _n

_LIG = {
    "fi": "fi",
    "fl": "fl",
    "ff": "ff",
    "ffi": "ffi",
    "ffl": "ffl",
    "quotesingle": "'",
    "quoteright": "’",
    "quoteleft": "‘",
    "quotedblleft": "“",
    "quotedblright": "”",
    "quotedbl": '"',
    "germandbls": "ß",
    "hyphen": "-",
    "minus": "−",
    "endash": "–",
    "emdash": "—",
    "bullet": "•",
    "dagger": "†",
    "periodcentered": "·",
}

# Adobe's standard Latin glyph names -> characters. Without this a simple font
# that carries WinAnsiEncoding with no /Differences loses every glyph whose
# name is not a ligature or an accent, which is to say digits, punctuation and
# brackets: `Frontier:` reads as `Frontier` and a whole line of figures
# disappears. Typographic symbols keep their Unicode identities; only letter
# ligatures expand to the same letters that normalization uses elsewhere.
_GLYPH_CHAR = {
    "exclam": "!",
    "quotedbl": '"',
    "numbersign": "#",
    "dollar": "$",
    "percent": "%",
    "ampersand": "&",
    "parenleft": "(",
    "parenright": ")",
    "asterisk": "*",
    "plus": "+",
    "comma": ",",
    "period": ".",
    "slash": "/",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "colon": ":",
    "semicolon": ";",
    "less": "<",
    "equal": "=",
    "greater": ">",
    "question": "?",
    "at": "@",
    "bracketleft": "[",
    "backslash": "\\",
    "bracketright": "]",
    "asciicircum": "^",
    "underscore": "_",
    "quoteleft": "`",
    "braceleft": "{",
    "bar": "|",
    "braceright": "}",
    "asciitilde": "~",
    "exclamdown": "\u00a1",
    "cent": "\u00a2",
    "sterling": "\u00a3",
    "currency": "\u00a4",
    "yen": "\u00a5",
    "brokenbar": "\u00a6",
    "section": "\u00a7",
    "dieresis": "\u00a8",
    "copyright": "\u00a9",
    "ordfeminine": "\u00aa",
    "guillemotleft": "\u00ab",
    "logicalnot": "\u00ac",
    "registered": "\u00ae",
    "macron": "\u00af",
    "degree": "\u00b0",
    "plusminus": "\u00b1",
    "twosuperior": "\u00b2",
    "threesuperior": "\u00b3",
    "acute": "\u00b4",
    "mu": "\u00b5",
    "paragraph": "\u00b6",
    "periodcentered": "\u00b7",
    "cedilla": "\u00b8",
    "onesuperior": "\u00b9",
    "ordmasculine": "\u00ba",
    "guillemotright": "\u00bb",
    "onequarter": "\u00bc",
    "onehalf": "\u00bd",
    "threequarters": "\u00be",
    "questiondown": "\u00bf",
}
for _c, _n in [
    (192, "Agrave"),
    (193, "Aacute"),
    (194, "Acircumflex"),
    (195, "Atilde"),
    (196, "Adieresis"),
    (197, "Aring"),
    (198, "AE"),
    (199, "Ccedilla"),
    (200, "Egrave"),
    (201, "Eacute"),
    (202, "Ecircumflex"),
    (203, "Edieresis"),
    (204, "Igrave"),
    (205, "Iacute"),
    (206, "Icircumflex"),
    (207, "Idieresis"),
    (208, "Eth"),
    (209, "Ntilde"),
    (210, "Ograve"),
    (211, "Oacute"),
    (212, "Ocircumflex"),
    (213, "Otilde"),
    (214, "Odieresis"),
    (215, "multiply"),
    (216, "Oslash"),
    (217, "Ugrave"),
    (218, "Uacute"),
    (219, "Ucircumflex"),
    (220, "Udieresis"),
    (221, "Yacute"),
    (222, "Thorn"),
    (223, "germandbls"),
    (224, "agrave"),
    (225, "aacute"),
    (226, "acircumflex"),
    (227, "atilde"),
    (228, "adieresis"),
    (229, "aring"),
    (230, "ae"),
    (231, "ccedilla"),
    (232, "egrave"),
    (233, "eacute"),
    (234, "ecircumflex"),
    (235, "edieresis"),
    (236, "igrave"),
    (237, "iacute"),
    (238, "icircumflex"),
    (239, "idieresis"),
    (240, "eth"),
    (241, "ntilde"),
    (242, "ograve"),
    (243, "oacute"),
    (244, "ocircumflex"),
    (245, "otilde"),
    (246, "odieresis"),
    (247, "divide"),
    (248, "oslash"),
    (249, "ugrave"),
    (250, "uacute"),
    (251, "ucircumflex"),
    (252, "udieresis"),
    (253, "yacute"),
    (254, "thorn"),
    (255, "ydieresis"),
]:
    _GLYPH_CHAR[_n] = chr(_c)
del _c, _n


def name_to_unicode(nm):
    nm = str(nm)
    m = re.fullmatch(r"u([0-9A-Fa-f]{4,6})", nm)
    if m:
        return chr(int(m.group(1), 16))
    if len(nm) == 1:
        return nm
    if nm == "space":
        return " "
    if nm == "nbspace":
        return "\u00a0"
    if nm in _LIG:
        return _LIG[nm]
    return _GLYPH_CHAR.get(nm, "")


class Font:
    """A font resource: decoding, widths and flags."""

    def __init__(self, doc, fdict):
        self.doc = doc
        self.d = fdict or {}
        self.subtype = str(self.d.get("Subtype", ""))
        base = self.d.get("BaseFont")
        self.base = str(base) if base is not None else ""
        self.name = re.sub(r"^[A-Z]{6}\+", "", self.base)
        self.two_byte = False
        self.tounicode = {}
        self.diff = {}
        self.base_encoding = "StandardEncoding"
        self.widths = {}
        self.default_width = 500
        self._load()

    def _load(self):
        d, doc = self.d, self.doc
        tu = doc.resolve(d.get("ToUnicode"))
        if isinstance(tu, Stream):
            try:
                self.tounicode = parse_cmap(tu.data)
            except Exception:  # noqa: BLE001 - a malformed cmap means no map.
                self.tounicode = {}

        enc = doc.resolve(d.get("Encoding"))
        if isinstance(enc, Name):
            self.base_encoding = str(enc)
            if str(enc) in ("Identity-H", "Identity-V"):
                self.two_byte = True
        elif isinstance(enc, dict):
            be = doc.resolve(enc.get("BaseEncoding"))
            if isinstance(be, Name):
                self.base_encoding = str(be)
            if isinstance(be, Name) and str(be) in ("Identity-H", "Identity-V"):
                self.two_byte = True
            diffs = doc.resolve(enc.get("Differences"))
            if isinstance(diffs, list):
                code = 0
                for it in diffs:
                    it = doc.resolve(it)
                    if isinstance(it, (int, float)):
                        code = int(it)
                    elif isinstance(it, Name):
                        self.diff[code] = str(it)
                        code += 1

        if self.subtype == "Type0":
            self.two_byte = True
            desc = doc.resolve(d.get("DescendantFonts"))
            df = doc.resolve(desc[0]) if isinstance(desc, list) and desc else desc
            if isinstance(df, dict):
                self.default_width = doc.resolve(df.get("DW")) or 1000
                dw = doc.resolve(df.get("W"))
                if isinstance(dw, list):
                    self._parse_w(dw)

        fc = doc.resolve(d.get("FirstChar"))
        ws = doc.resolve(d.get("Widths"))
        if isinstance(fc, (int, float)) and isinstance(ws, list):
            for i, w in enumerate(ws):
                w = doc.resolve(w)
                if isinstance(w, (int, float)):
                    self.widths[int(fc) + i] = float(w)
        fd = doc.resolve(d.get("FontDescriptor"))
        if isinstance(fd, dict):
            mw = doc.resolve(fd.get("MissingWidth"))
            if isinstance(mw, (int, float)):
                self.default_width = float(mw)
            if "Courier" in self.base:
                self.default_width = 600

    def _parse_w(self, dw):
        doc, i = self.doc, 0
        while i < len(dw):
            a = doc.resolve(dw[i])
            nxt = doc.resolve(dw[i + 1]) if i + 1 < len(dw) else None
            if isinstance(nxt, list):
                for j, w in enumerate(nxt):
                    w = doc.resolve(w)
                    if isinstance(w, (int, float)) and isinstance(a, (int, float)):
                        self.widths[int(a) + j] = float(w)
                i += 2
            elif isinstance(nxt, (int, float)) and i + 2 < len(dw):
                w = doc.resolve(dw[i + 2])
                if (
                    isinstance(a, (int, float))
                    and isinstance(w, (int, float))
                    and int(nxt) - int(a) <= 65535
                ):
                    for c in range(int(a), int(nxt) + 1):
                        self.widths[c] = float(w)
                i += 3
            else:
                i += 1

    def decode(self, bs):
        if self.two_byte:
            return "".join(
                self._unicode((bs[i] << 8) | bs[i + 1])
                for i in range(0, len(bs) - 1, 2)
            ) + ("\ufffd" if len(bs) % 2 else "")
        return "".join(self._unicode(b) for b in bs)

    def _unicode(self, code):
        if code in self.tounicode:
            return self.tounicode[code] or "\ufffd"
        if self.two_byte:
            # An undecodable CID is missing text, not a character we can omit
            # to manufacture a healthy-looking word beside it.
            return "\ufffd"
        if code in self.diff:
            return name_to_unicode(self.diff[code]) or "\ufffd"
        if self.base_encoding == "WinAnsiEncoding":
            return _cp1252_map().get(code, "\ufffd")
        if self.base_encoding == "MacRomanEncoding":
            return bytes([code]).decode("mac_roman") if code >= 32 else "\ufffd"
        if self.base_encoding != "StandardEncoding" or self.name in {
            "Symbol",
            "ZapfDingbats",
        }:
            return "\ufffd"
        if code in GLYPH_NAMES:
            return name_to_unicode(GLYPH_NAMES[code])
        if 32 <= code < 127:
            return chr(code)
        return _cp1252_map().get(code, "\ufffd")

    def width(self, code):
        return self.widths.get(code, self.default_width)


# ---------------------------------------------------------------- extraction


class Chunk:
    """One shown text run, positioned in device space.

    `size` is the rendered font size in points. `width` is the run's advance in
    text-space units, i.e. with `fsize*th` already applied, and `scale` is the
    factor that carries those units into device space -- so the drawn extent is
    `x .. x + width*scale`. `scale` is therefore roughly 1 for a `Tf`-sized
    run and roughly the `Tm` scale for a run whose size lives in the matrix; it
    is never the font size itself.
    """

    __slots__ = (
        "font",
        "order",
        "page",
        "scale",
        "size",
        "sub",
        "sup",
        "text",
        "ux",
        "uy",
        "width",
        "x",
        "y",
    )

    def __init__(self, text, x, y, size, font, width, scale=1.0, order=0):
        self.text = text
        self.x = x
        self.y = y
        self.size = size
        self.font = font
        self.width = width
        self.scale = scale
        self.page = 0
        # position in the content stream. Kerning can move a chunk *left* of
        # its predecessor (`[(V)111(erstehen)]`), so x alone is not a safe
        # ordering key within a line; stream order is.
        self.order = order
        # set by pdfline.mark_inline: this run is a materially smaller run
        # riding on a neighbour's baseline -- `sup` when raised above it, `sub`
        # when dropped below. False until a page has been classified, which is
        # what makes a chunk pdfline has not seen read as ordinary text.
        self.sup = False
        self.sub = False
        self.ux, self.uy = 1.0, 0.0


def _codes(font, bs):
    if font.two_byte:
        return [(bs[i] << 8) | bs[i + 1] for i in range(0, len(bs) - 1, 2)]
    return list(bs)


def _advance(font, bs, fsize, tc, tw, th):
    """Run advance in text-space units (widths are 1/1000 em).

    `tw` (word spacing) is added for each occurrence of the single-byte code
    32 only, not for every character after the first: in a simple font the two
    coincide often enough to hide the difference, but a CID font whose codes
    happen to include 32 would otherwise get word spacing on unrelated glyphs.
    """
    return sum(
        (
            (font.width(c) / 1000.0) * fsize
            + tc
            + (tw if (c == 32 and not font.two_byte) else 0.0)
        )
        * th
        for c in _codes(font, bs)
    )


def _mkchunk(font, bs, tm, ctm, fsize, tc, tw, th, trise, order=0):
    trm = mat_mul((fsize * th, 0, 0, fsize, 0, trise), mat_mul(tm, ctm))
    horiz = (trm[0] ** 2 + trm[1] ** 2) ** 0.5
    vert = (trm[2] ** 2 + trm[3] ** 2) ** 0.5
    size = horiz or vert
    # `_advance` has already multiplied by fsize*th, so the device-space
    # advance is `_advance * scale`, and `scale` must carry the *outer*
    # matrix only. Deriving it from the rendered size keeps the two in step:
    # for unrotated text size == fsize*th*|outer|, and for text rotated a
    # quarter turn the run advances along the y axis, where |vert| is the
    # same length.
    f = fsize * th
    scale = size / f if f else 1.0
    chunk = Chunk(
        font.decode(bs),
        trm[4],
        trm[5],
        size,
        font,
        _advance(font, bs, fsize, tc, tw, th),
        scale,
        order,
    )
    if horiz:
        chunk.ux, chunk.uy = trm[0] / horiz, trm[1] / horiz
    return chunk


def _do_tj(chunks, font, arr, tm, ctm, fsize, tc, tw, th, trise, tmstate):
    """Show a TJ array, returning the advanced *text* matrix.

    Only `Tm` moves. `Tlm`, the text-line matrix, is left alone: PDF 32000-1
    9.4.1 changes it under `Tm`, `Td`, `TD` and `T*` alone, so a `Td` after a
    `TJ` is measured from the line origin, not from where the last run ended.
    Writers that emit one `Tm` per text block and position every following run
    with `TD` depend on this; treating `TJ` as if it moved `Tlm` makes each
    line's x drift by the width of all the text before it.
    """
    tmm = tmstate
    for item in arr:
        if isinstance(item, bytes):
            if not item:
                continue
            ch = _mkchunk(font, item, tmm, ctm, fsize, tc, tw, th, trise, len(chunks))
            if ch.text:
                chunks.append(ch)
            tmm = mat_mul((1, 0, 0, 1, ch.width, 0), tmm)
        elif isinstance(item, (int, float)):
            tmm = mat_mul((1, 0, 0, 1, -float(item) / 1000.0 * fsize * th, 0), tmm)
    return tmm


def extract_page(  # noqa: C901 - ported content-stream interpreter.
    doc, page, _content=None, _ctm=None, _forms=(), _state=None
):
    content = doc.page_content(page) if _content is None else _content
    resources = doc.resolve(page.get("Resources")) or {}
    fonts_res = doc.resolve(resources.get("Font")) or {}
    xobj_res = doc.resolve(resources.get("XObject")) or {}
    cache = {}

    def get_font(nm):
        if nm not in cache:
            fd = doc.resolve(fonts_res.get(nm))
            cache[nm] = Font(doc, fd if isinstance(fd, dict) else {})
        return cache[nm]

    chunks = []
    ctm = _ctm or (1, 0, 0, 1, 0, 0)
    gs = []
    tm = tlm = (1, 0, 0, 1, 0, 0)
    font = None
    fsize = 0.0
    tc = tw = 0.0
    th = 1.0
    tl = 0.0
    trise = 0.0
    render = 0
    if _state is not None:
        font, fsize, tc, tw, th, tl, trise, render = _state
    p = 0
    n = len(content)
    stack = []

    while p < n:
        c = content[p]
        if c in b"\x00\t\n\x0c\r ":
            p += 1
            continue
        if c == 0x25:  # comment
            while p < n and content[p] not in b"\r\n":
                p += 1
            continue
        if c in b"(<>)":
            v, p = _read_operand(content, p)
            stack.append(v)
            continue
        if 0x30 <= c <= 0x39 or c in b"-+.":
            v, p = _read_number(content, p)
            stack.append(v)
            continue
        if c == 0x2F:
            nm, p = _read_name_token(content, p)
            stack.append(Name(nm))
            continue
        if c == 0x5B:  # [
            stack.append("[")
            p += 1
            continue
        if c == 0x5D:  # ]
            p += 1
            arr = []
            while stack and stack[-1] != "[":
                arr.append(stack.pop())
            if stack:
                stack.pop()
            arr.reverse()
            stack.append(arr)
            continue
        if c in b"{}":
            p += 1
            continue

        op = _read_op(content, p)
        if not op:
            p += 1
            continue
        p += len(op)
        op = op.decode("latin-1")

        if op == "q":
            gs.append((ctm, font, fsize, tc, tw, th, tl, trise, render))
        elif op == "Q":
            if gs:
                ctm, font, fsize, tc, tw, th, tl, trise, render = gs.pop()
        elif op == "cm" and len(stack) >= 6:
            ctm = mat_mul(tuple(float(x) for x in stack[-6:]), ctm)
        elif op in ("BT", "ET"):
            tm = tlm = (1, 0, 0, 1, 0, 0)
        elif op == "Tf" and len(stack) >= 2:
            if isinstance(stack[-2], Name):
                font = get_font(str(stack[-2]))
            try:
                fsize = float(stack[-1])
            except (TypeError, ValueError):
                fsize = 0.0
        elif op == "Td" and len(stack) >= 2:
            tlm = mat_mul((1, 0, 0, 1, float(stack[-2]), float(stack[-1])), tlm)
            tm = tlm
        elif op == "TD" and len(stack) >= 2:
            tl = -float(stack[-1])
            tlm = mat_mul((1, 0, 0, 1, float(stack[-2]), float(stack[-1])), tlm)
            tm = tlm
        elif op == "Tm" and len(stack) >= 6:
            tlm = tuple(float(x) for x in stack[-6:])
            tm = tlm
        elif op == "T*":
            tlm = mat_mul((1, 0, 0, 1, 0, -tl), tlm)
            tm = tlm
        elif op == "TL" and stack:
            tl = float(stack[-1])
        elif op == "Tc" and stack:
            tc = float(stack[-1])
        elif op == "Tw" and stack:
            tw = float(stack[-1])
        elif op == "Tz" and stack:
            th = float(stack[-1]) / 100.0
        elif op == "Ts" and stack:
            trise = float(stack[-1])
        elif op == "Tr" and stack:
            # Render mode, not text rise. It selects how glyphs are painted and
            # does not move them, so it is parsed and then ignored: reading it
            # as a rise lifts every following baseline by up to 7 pt and
            # shatters the lines around it. Mode 3 paints nothing but still
            # occupies space, and the runs in it are kept -- a producer that
            # hides text has usually put something there worth keeping.
            render = int(stack[-1])
        elif op == "Tj" and stack:
            if font is not None and isinstance(stack[-1], bytes):
                ch = _mkchunk(
                    font, stack[-1], tm, ctm, fsize, tc, tw, th, trise, len(chunks)
                )
                if ch.text:
                    chunks.append(ch)
                tm = mat_mul((1, 0, 0, 1, ch.width, 0), tm)
        elif op == "'" and stack:
            if font is not None and isinstance(stack[-1], bytes):
                tlm = mat_mul((1, 0, 0, 1, 0, -tl), tlm)
                tm = tlm
                ch = _mkchunk(
                    font, stack[-1], tm, ctm, fsize, tc, tw, th, trise, len(chunks)
                )
                if ch.text:
                    chunks.append(ch)
                tm = mat_mul((1, 0, 0, 1, ch.width, 0), tm)
        elif op == '"' and len(stack) >= 3:
            if font is not None and isinstance(stack[-1], bytes):
                tw = float(stack[-3])
                tc = float(stack[-2])
                tlm = mat_mul((1, 0, 0, 1, 0, -tl), tlm)
                tm = tlm
                ch = _mkchunk(
                    font, stack[-1], tm, ctm, fsize, tc, tw, th, trise, len(chunks)
                )
                if ch.text:
                    chunks.append(ch)
                tm = mat_mul((1, 0, 0, 1, ch.width, 0), tm)
        elif op == "TJ":
            arr = stack[-1] if stack and isinstance(stack[-1], list) else []
            if font is not None:
                tm = _do_tj(chunks, font, arr, tm, ctm, fsize, tc, tw, th, trise, tm)
        elif op == "Do" and stack and isinstance(stack[-1], Name):
            xo = doc.resolve(xobj_res.get(str(stack[-1])))
            subtype = str(xo.dict.get("Subtype")) if isinstance(xo, Stream) else ""
            if subtype == "Form" and id(xo) not in _forms and len(_forms) < 16:
                form_resources = dict(resources)
                form_resources.update(doc.resolve(xo.dict.get("Resources")) or {})
                matrix = doc.resolve(xo.dict.get("Matrix")) or (1, 0, 0, 1, 0, 0)
                matrix = tuple(float(doc.resolve(v)) for v in matrix)
                child_chunks = extract_page(
                    doc,
                    {"Resources": form_resources},
                    _content=xo.data,
                    _ctm=mat_mul(matrix, ctm),
                    _forms=_forms + (id(xo),),
                    _state=(font, fsize, tc, tw, th, tl, trise, render),
                )
                offset = len(chunks)
                for ch in child_chunks:
                    ch.order += offset
                chunks.extend(child_chunks)
        stack = []
    return chunks


# ---------------------------------------------------------------- tokenizer


def _read_number(buf, p):
    n = len(buf)
    start = p
    while p < n and buf[p] in b"0123456789+-.":
        p += 1
    tok = buf[start:p]
    try:
        return (float(tok) if b"." in tok else int(tok)), p
    except ValueError:
        try:
            return float(re.sub(rb"[^0-9.\-]", b"", tok) or b"0"), p
        except ValueError:
            return 0, p


def _read_name_token(buf, p):
    n = len(buf)
    start = p
    p += 1
    while p < n and buf[p] not in b"\x00\t\n\x0c\r ()<>[]{}/%":
        p += 1
    return decode_name(buf[start + 1 : p]), p


def _read_op(buf, p):
    n = len(buf)
    start = p
    while p < n and buf[p] not in b"\x00\t\n\x0c\r ()<>[]{}/%":
        p += 1
    return buf[start:p]


def _read_operand(buf, p):
    if buf[p] == 0x28:
        return _read_literal_str(buf, p)
    if buf[p] == 0x3C:
        return (None, p + 2) if buf[p : p + 2] == b"<<" else _read_hex_str(buf, p)
    if buf[p] == 0x3E:
        return None, p + (2 if buf[p : p + 2] == b">>" else 1)
    if buf[p] in (0x29, 0x7D):
        return None, p + 1
    start = p
    while p < len(buf) and buf[p] not in b"\x00\t\n\x0c\r ()<>[]{}/%":
        p += 1
    tok = buf[start:p]
    if re.fullmatch(rb"[+-]?(\d+\.?\d*|\.\d+)", tok):
        return _read_number(buf, start)[0], p
    return {"true": True, "false": False, "null": None}.get(tok), p


def _read_literal_str(buf, p):
    n = len(buf)
    i = p + 1
    depth = 1
    out = bytearray()
    while i < n and depth > 0:
        c = buf[i]
        if c == 0x5C:
            i += 1
            if i >= n:
                break
            e = buf[i]
            if e in b"nrtbf":
                out.append({0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}.get(e, e))
                i += 1
            elif e in b"01234567":
                o = bytearray()
                while i < n and len(o) < 3 and buf[i] in b"01234567":
                    o.append(buf[i])
                    i += 1
                out.append(int(o, 8) & 0xFF)
            elif e in (0x0A, 0x0D):
                i += 1
                if e == 0x0D and i < n and buf[i] == 0x0A:
                    i += 1
            else:
                out.append(e)
                i += 1
        elif c == 0x28:
            depth += 1
            out.append(c)
            i += 1
        elif c == 0x29:
            depth -= 1
            if depth:
                out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    return bytes(out), i


def _read_hex_str(buf, p):
    n = len(buf)
    i = p + 1
    hx = bytearray()
    while i < n and buf[i] != 0x3E:
        if buf[i] in b"0123456789abcdefABCDEF":
            hx.append(buf[i])
        i += 1
    i += 1
    if len(hx) % 2:
        hx.append(0x30)
    try:
        return bytes.fromhex(hx.decode("ascii")), i
    except ValueError:
        return b"", i
