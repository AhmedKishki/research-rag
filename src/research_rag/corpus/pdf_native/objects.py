"""Minimal pure-python PDF object model / parser (xref tables + object streams).

The bottom layer: it knows what a PDF file *is* -- names, references, strings,
dictionaries, streams, and how to find them by number -- and nothing about what
they mean. Fonts and images are decoded by the modules above it.

Standard library only, deliberately: a PDF should be readable on a machine with
nothing installed.

Adapted from the user's own AhmedKishki/pdf-tools project; the code is reused
here with the owner's permission.
"""

import contextlib
import os
import re
import zlib

# Resource budgets for untrusted PDF input. A declared row count, encoded file
# size, filter chain, compressed payload, or predictor dimension never sizes a
# decode before the bytes exist. A PDF that exceeds a budget raises
# PdfResourceLimitError, which the recovery helper turns into one bounded reason.
MAX_NATIVE_DOCUMENT_BYTES = 256 * 1024 * 1024
MAX_DECODED_STREAM_BYTES = 32 * 1024 * 1024
MAX_DOCUMENT_DECODED_BYTES = 128 * 1024 * 1024
MAX_FILTER_CHAIN = 16
MAX_XREF_ENTRIES = 1_000_000
MAX_PREDICTOR_COLUMNS = 1 << 20
MAX_PREDICTOR_ROW_BYTES = 16 * 1024 * 1024
MAX_PREDICTOR_COLORS = 64
MAX_XREF_STREAM_FIELD_BYTES = 8


class PdfResourceLimitError(ValueError):
    """A declared or decoded PDF resource exceeds a parser budget."""


class Name(str):
    __slots__ = ()

    def __repr__(self):
        return "/" + str.__str__(self)


class Ref:
    __slots__ = ("gen", "num")

    def __init__(self, num, gen):
        self.num, self.gen = num, gen

    def __repr__(self):
        return "%d %d R" % (self.num, self.gen)  # noqa: UP031 - ported form.

    def __eq__(self, o):
        return isinstance(o, Ref) and o.num == self.num and o.gen == self.gen

    def __hash__(self):
        return hash((self.num, self.gen))


class Str(bytes):
    """PDF literal string; bytes value."""

    __slots__ = ()


WS = b"\x00\t\n\x0c\r "
DELIM = b"()<>[]{}/%"
ENDERS = WS + DELIM


class Lexer:
    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def skip_ws(self):
        b, n = self.buf, len(self.buf)
        while self.pos < n:
            c = b[self.pos]
            if c in WS:
                self.pos += 1
            elif c == 0x25:
                while self.pos < n and b[self.pos] not in b"\r\n":
                    self.pos += 1
            else:
                return

    def read_token(self):
        self.skip_ws()
        b, n = self.buf, len(self.buf)
        if self.pos >= n:
            return b""
        start = self.pos
        c = b[self.pos]
        if c in b"[]{}":
            self.pos += 1
            return b[start : self.pos]
        if c == 0x3C:
            if b[self.pos : self.pos + 2] == b"<<":
                self.pos += 2
                return b"<<"
            self.pos += 1
            return b"<"
        if c == 0x3E:
            if b[self.pos : self.pos + 2] == b">>":
                self.pos += 2
                return b">>"
            self.pos += 1
            return b">"
        if c in b"()<>%":
            self.pos += 1
            return b[start : self.pos]
        if c == 0x2F:
            # '/' may be its own token or start a name
            self.pos += 1
            while self.pos < n and self.buf[self.pos] not in ENDERS:
                self.pos += 1
            return b[start : self.pos]
        while self.pos < n and b[self.pos] not in ENDERS:
            self.pos += 1
        return b[start : self.pos]


def is_regular(tok):
    return bool(tok) and tok[0] not in b"[]{}<>()/%" or tok in (b"<<", b">>")


class Parser(Lexer):
    def __init__(self, buf, pos=0, doc=None):
        Lexer.__init__(self, buf, pos)
        self.doc = doc
        self._pending_dict = None
        self._stream_dict = None
        self._stream_raw = None

    def parse(self, depth=0):
        if depth > 200:
            return None
        tok = self.read_token()
        return self._parse_from(tok, depth)

    def _parse_from(self, tok, depth=0):
        if tok == b"":
            return None
        if tok == b"<<":
            saved_pending = self._pending_dict
            self._pending_dict = {}
            self._parse_dict(depth)
            pd = self._pending_dict
            self._pending_dict = saved_pending
            if pd is None:
                return None
            self._maybe_stream(pd)
            return pd
        if tok == b"[":
            arr = []
            while True:
                t = self.read_token()
                if t == b"" or t == b"]":
                    break
                arr.append(self._parse_from(t, depth + 1))
            return arr
        if tok == b"<":
            return Str(self._read_hex())
        if tok == b"(":
            return Str(self._read_literal())
        if tok in (b">", b">>", b")", b"{"):
            return None
        if tok[0] == 0x2F:
            return Name(decode_name(tok[1:]))
        if tok == b"true":
            return True
        if tok == b"false":
            return False
        if tok == b"null":
            return None
        if re.match(rb"^[+-]?(\d+\.?\d*|\.\d+)$", tok):
            m = re.match(rb"^([+-]?\d+)$", tok)
            if m:
                save = self.pos
                t2 = self.read_token()
                if re.match(rb"^\d+$", t2 or b""):
                    t3 = self.read_token()
                    if t3 == b"R":
                        return Ref(int(tok), int(t2))
                    # Not a reference: the lookahead consumed tokens, so rewind
                    # to just past `tok`. Rewinding only to `save2` would keep
                    # t2 consumed and silently drop it -- which corrupts every
                    # value after it, e.g. an /ObjStm offset table where
                    # "83 675 84" must parse as 83 then 675.
                    self.pos = save
                    return int(tok)
                self.pos = save
                return int(tok)
            return float(tok)
        if is_regular(tok):
            return Str(tok)
        return None

    def _read_literal(self):
        b = self.buf
        n = len(b)
        i = self.pos
        depth = 1
        out = bytearray()
        while i < n and depth > 0:
            c = b[i]
            if c == 0x5C:
                i += 1
                if i >= n:
                    break
                e = b[i]
                if e in b"nrtbf":
                    out.append(
                        {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}.get(e, e)
                    )
                    i += 1
                elif e in b"01234567":
                    o = bytearray()
                    while i < n and len(o) < 3 and b[i] in b"01234567":
                        o.append(b[i])
                        i += 1
                    out.append(int(o, 8) & 0xFF)
                elif e in (0x0A, 0x0D):
                    i += 1
                    if e == 0x0D and i < n and b[i] == 0x0A:
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
                if depth > 0:
                    out.append(c)
                i += 1
            else:
                out.append(c)
                i += 1
        self.pos = i
        return bytes(out)

    def _read_hex(self):
        b = self.buf
        n = len(b)
        self.skip_ws()
        out = bytearray()
        i = self.pos
        while i < n and b[i] != 0x3E:
            if b[i] in b"0123456789abcdefABCDEF":
                out.append(b[i])
            i += 1
        self.pos = i + 1
        s = out.decode("ascii")
        if len(s) % 2:
            s += "0"
        try:
            return bytes.fromhex(s)
        except ValueError:
            return b""

    def _parse_dict(self, depth):
        d = self._pending_dict
        while True:
            t = self.read_token()
            if t == b"" or t == b">>":
                break
            if t[0] != 0x2F:
                continue
            key = decode_name(t[1:])
            self.skip_ws()
            save = self.pos
            t2 = self.read_token()
            if t2 == b"stream":
                self.pos = save
                d[key] = self._parse_from(self.read_token(), depth + 1)
                continue
            self.pos = save
            d[key] = self._parse_from(self.read_token(), depth + 1)

    def _maybe_stream(self, d):
        """A dict is followed by 'stream' when it is a stream object."""
        if self._stream_dict is not None:
            return
        save = self.pos
        self.skip_ws()
        if self.buf[self.pos : self.pos + 6] != b"stream":
            self.pos = save
            return
        self.pos += 6
        self._stream_dict = d
        self._stream_raw = self._read_stream(d)

    def _read_stream(self, d):
        b = self.buf
        n = len(b)
        if b[self.pos : self.pos + 2] == b"\r\n":
            self.pos += 2
        elif b[self.pos : self.pos + 1] in (b"\n", b"\r"):
            self.pos += 1
        start = self.pos
        ln = None
        lv = d.get("Length")
        if isinstance(lv, int):
            ln = lv
        elif isinstance(lv, Ref) and self.doc is not None:
            r = self.doc.resolve(lv)
            if isinstance(r, int):
                ln = r
        raw = None
        if (
            ln is not None
            and ln >= 0
            and start + ln <= n
            and re.compile(rb"[\r\n\s]endstream").search(b, start + ln, start + ln + 40)
        ):
            raw = b[start : start + ln]
        if raw is None:
            em = re.compile(rb"endstream").search(b, start)
            end = em.start() if em else n
            # Trim each trailing EOL in one backward pass; re-slicing per
            # character is quadratic on a stream padded with newlines.
            trim = end
            while trim > start and b[trim - 1] in b"\r\n":
                trim -= 1
            raw = b[start:trim]
        m = re.compile(rb"endstream").search(b, start + len(raw))
        self.pos = m.end() if m else start + len(raw)
        return raw


def decode_name(raw):
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == 0x23 and i + 2 < len(raw):
            try:
                out.append(int(raw[i + 1 : i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(raw[i])
        i += 1
    return bytes(out).decode("utf-8", "replace")


def _decode_budget(doc):
    """Bytes one stream may still decode to, within the document budget."""

    if doc is None:
        return MAX_DECODED_STREAM_BYTES
    remaining = MAX_DOCUMENT_DECODED_BYTES - getattr(doc, "_decoded_bytes", 0)
    return max(0, min(MAX_DECODED_STREAM_BYTES, remaining))


def _bounded_inflate(data, limit):
    """Inflate at most ``limit`` bytes, or return ``None`` when it exceeds it.

    ``decompress`` is given the budget, so a small payload cannot expand into a
    large allocation before the size check: the decoder stops at the budget and
    any unconsumed tail marks the stream as over budget.
    """

    if limit <= 0:
        return None
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            engine = zlib.decompressobj(wbits)
            out = engine.decompress(data, limit + 1)
            if len(out) > limit or engine.unconsumed_tail:
                continue
            # Tolerate a truncated tail; the output stays bounded either way.
            with contextlib.suppress(zlib.error):
                out += engine.flush()
        except zlib.error:
            continue
        if len(out) > limit:
            continue
        return out
    return None


def apply_filters(d, raw, doc=None):
    data = raw
    f = d.get("Filter")
    if f is None:
        return data
    if isinstance(f, Name):
        f = [f]
    if len(f) > MAX_FILTER_CHAIN:
        raise PdfResourceLimitError("PDF filter chain exceeds the parser budget")
    parms = d.get("DecodeParms") or d.get("DP")
    if isinstance(parms, dict) or parms is None:
        parms = [parms]
    while isinstance(parms, list) and len(parms) < len(f):
        parms.append(None)
    for i, filt in enumerate(f):
        budget = _decode_budget(doc)
        parm = parms[i] if i < len(parms) else None
        if isinstance(parm, Ref) and doc is not None:
            parm = doc.resolve(parm)
        filt = str(filt)
        if filt in ("FlateDecode", "Fl"):
            decoded = _bounded_inflate(data, budget)
            data = decoded if decoded is not None else b""
        elif filt in ("LZWDecode", "LZW"):
            data = lzw_decode(data, early=1, limit=budget)
        elif filt in ("ASCIIHexDecode", "AHx"):
            hx = re.sub(rb"[^0-9A-Fa-f]", b"", data.split(b">")[0])
            if len(hx) % 2:
                hx += b"0"
            data = bytes.fromhex(hx.decode("ascii"))
        elif filt in ("ASCII85Decode", "A85"):
            data = a85_decode(data, limit=budget)
        elif filt in ("RunLengthDecode", "RL"):
            data = rle_decode(data, limit=budget)
        else:
            break  # image codecs (DCT/JPX/CCITT) are handled by pdfimage
        if len(data) > budget:
            return b""
        # Charge each stage's output, so a chain of small expansions cannot
        # stay under the per-stream cap while working past the document budget.
        if doc is not None:
            doc._decoded_bytes = getattr(doc, "_decoded_bytes", 0) + len(data)
        if parm and isinstance(parm, dict):
            data = apply_predictor(data, parm, limit=_decode_budget(doc))
            if len(data) > _decode_budget(doc):
                return b""
    return data


_VALID_BITS_PER_COMPONENT = (1, 2, 4, 8, 16)


def _predictor_dimension(parm, key, default, cap, allowed=None):
    """One predictor dimension, or ``None`` when the file's value is unusable."""

    if key not in parm:
        return default
    value = parm.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > cap
        or (allowed is not None and value not in allowed)
    ):
        return None
    return value


def apply_predictor(data, parm, limit=None):
    pred = parm.get("Predictor", 1)
    if isinstance(pred, bool) or not isinstance(pred, int) or pred < 2:
        return data
    budget = MAX_DECODED_STREAM_BYTES
    if limit is not None:
        budget = min(limit, MAX_DECODED_STREAM_BYTES)
    colors = _predictor_dimension(parm, "Colors", 1, MAX_PREDICTOR_COLORS)
    bpc = _predictor_dimension(
        parm, "BitsPerComponent", 8, 32, _VALID_BITS_PER_COMPONENT
    )
    columns = _predictor_dimension(parm, "Columns", 1, MAX_PREDICTOR_COLUMNS)
    if colors is None or bpc is None or columns is None:
        return b""
    bpp = max(1, (colors * bpc + 7) // 8)
    rowlen = (columns * colors * bpc + 7) // 8
    # Never size a row allocation from the file: rowlen must fit the remaining
    # budget, and a zero-row stream is malformed rather than padded.
    if rowlen <= 0 or rowlen > MAX_PREDICTOR_ROW_BYTES or rowlen > budget:
        return b""
    if pred == 2:
        return data
    n = len(data)
    if not n or n % (rowlen + 1):
        return b""
    out = bytearray()
    prev = bytearray(rowlen)
    i = 0
    while i < n:
        ft = data[i]
        i += 1
        # A predictor stream is whole rows: a truncated final row and an
        # undefined PNG selector are both malformed, not padded.
        if i + rowlen > n:
            return b""
        row = bytearray(data[i : i + rowlen])
        i += rowlen
        if ft == 1:
            for j in range(bpp, rowlen):
                row[j] = (row[j] + row[j - bpp]) & 0xFF
        elif ft == 2:
            for j in range(rowlen):
                row[j] = (row[j] + prev[j]) & 0xFF
        elif ft == 3:
            for j in range(rowlen):
                left = row[j - bpp] if j >= bpp else 0
                row[j] = (row[j] + ((left + prev[j]) >> 1)) & 0xFF
        elif ft == 4:
            for j in range(rowlen):
                a = row[j - bpp] if j >= bpp else 0
                b = prev[j]
                c = prev[j - bpp] if j >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[j] = (row[j] + pr) & 0xFF
        elif ft != 0:
            return b""
        out.extend(row)
        if len(out) > budget:
            return b""
        prev = row
    return bytes(out)


def lzw_decode(data, early=1, limit=None):
    budget = MAX_DECODED_STREAM_BYTES
    if limit is not None:
        budget = min(limit, MAX_DECODED_STREAM_BYTES)
    if budget <= 0:
        return b""
    out = bytearray()
    prev = None
    bitpos = 0
    codelen = 9
    nbits = len(data) * 8

    def reset():
        return [bytes([i]) for i in range(256)] + [b"", b""]

    table = reset()
    while bitpos + codelen <= nbits:
        byte_i = bitpos >> 3
        chunk = data[byte_i : byte_i + 3]
        while len(chunk) < 3:
            chunk += b"\x00"
        val = (chunk[0] << 16) | (chunk[1] << 8) | chunk[2]
        code = (val >> (24 - (bitpos & 7) - codelen)) & ((1 << codelen) - 1)
        bitpos += codelen
        if code == 256:
            table = reset()
            codelen = 9
            prev = None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code]
        else:
            if code < len(table):
                entry = table[code]
                table.append(prev + entry[:1])
            else:
                entry = prev + prev[:1]
                table.append(entry)
        out.extend(entry)
        if len(out) > budget:
            return b""
        prev = entry
        if len(table) + early >= (1 << codelen) and codelen < 12:
            codelen += 1
    return bytes(out)


def a85_decode(data, limit=None):
    budget = MAX_DECODED_STREAM_BYTES
    if limit is not None:
        budget = min(limit, MAX_DECODED_STREAM_BYTES)
    if budget <= 0:
        return b""
    data = re.sub(rb"\s", b"", data)
    if data.startswith(b"<~"):
        data = data[2:]
    idx = data.find(b"~>")
    if idx >= 0:
        data = data[:idx]
    out = bytearray()
    i = 0
    while i < len(data):
        if data[i : i + 1] == b"z":
            out.extend(b"\x00\x00\x00\x00")
            i += 1
            if len(out) > budget:
                return b""
            continue
        chunk = data[i : i + 5]
        i += 5
        pad = 5 - len(chunk)
        chunk = chunk + b"u" * pad
        v = 0
        for c in chunk:
            v = v * 85 + (c - 33)
        b4 = (
            v.to_bytes(4, "big", signed=False) if v < (1 << 32) else b"\x00\x00\x00\x00"
        )
        out.extend(b4[: 4 - pad] if pad else b4)
        if len(out) > budget:
            return b""
    return bytes(out)


def rle_decode(data, limit=None):
    budget = MAX_DECODED_STREAM_BYTES
    if limit is not None:
        budget = min(limit, MAX_DECODED_STREAM_BYTES)
    if budget <= 0:
        return b""
    out = bytearray()
    i = 0
    while i < len(data):
        l = data[i]
        i += 1
        if l == 128:
            break
        if l < 128:
            out.extend(data[i : i + l + 1])
            i += l + 1
        else:
            if i < len(data):
                out.extend(bytes([data[i]]) * (257 - l))
                i += 1
        if len(out) > budget:
            return b""
    return bytes(out)


class Stream:
    __slots__ = ("_data", "dict", "doc", "raw")

    def __init__(self, d, raw, doc):
        self.dict, self.raw, self.doc, self._data = d, raw, doc, None

    def get(self, k, default=None):
        return self.dict.get(k, default)

    @property
    def data(self):
        if self._data is None:
            self._data = apply_filters(self.dict, self.raw, self.doc)
        return self._data


class Document:
    def __init__(self, path):
        size = os.path.getsize(path)
        if size > MAX_NATIVE_DOCUMENT_BYTES:
            raise PdfResourceLimitError("PDF exceeds the encoded size budget")
        with open(path, "rb") as f:
            self.buf = f.read()
        self.xref = {}
        self.compressed = {}
        self.trailer = {}
        self.cache = {}
        self._objstm_cache = {}
        self._extra = {}  # objects recovered from /ObjStm by scanning
        self._objstm_indexed = False
        self._loading = set()
        self._decoded_bytes = 0
        self._scan_all_xref()
        if not self.xref and not self.compressed:
            self._rebuild_xref()
        if not self.trailer.get("Root"):
            self._recover_catalog()

    def _scan_all_xref(self):
        tail = self.buf[-2048:]
        m = None
        for m in re.finditer(rb"startxref\s+(\d+)", tail):
            pass
        seen = set()
        starts = [int(m.group(1))] if m else []
        for off in starts:
            off2 = off
            for _ in range(64):
                if off2 in seen:
                    break
                seen.add(off2)
                if not self._load_xref_at(off2):
                    break
                nxt = self.trailer.get("Prev")
                if not isinstance(nxt, int):
                    break
                off2 = nxt

    def _xref_stream_fields(self, obj):
        """Validated ``(widths, index)`` for a cross-reference stream, or None.

        Field widths and the index run come from the file, so each is bounded
        before it can size a decode or a loop.
        """

        w = obj.dict.get("W")
        if isinstance(w, Ref):
            w = self.resolve(w)
        if not isinstance(w, list) or not w:
            return None
        widths = []
        for width in w:
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or width < 0
                or width > MAX_XREF_STREAM_FIELD_BYTES
            ):
                return None
            widths.append(width)
        if sum(widths) <= 0:
            return None
        index = obj.dict.get("Index") or [0, obj.dict.get("Size", 0)]
        if isinstance(index, Ref):
            index = self.resolve(index)
        if not isinstance(index, list) or not index or len(index) % 2:
            return None
        entries = []
        for value in index:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            entries.append(value)
        total = sum(entries[k + 1] for k in range(0, len(entries) - 1, 2))
        if total > MAX_XREF_ENTRIES:
            return None
        return widths, entries

    def _load_xref_at(self, off):
        if off < 0 or off >= len(self.buf):
            return False
        p = Parser(self.buf, off, self)
        p.skip_ws()
        if self.buf[p.pos : p.pos + 4] == b"xref":
            p.pos += 4
            while True:
                p.skip_ws()
                if self.buf[p.pos : p.pos + 7] == b"trailer":
                    p.pos += 7
                    tr = p.parse()
                    if isinstance(tr, dict):
                        self._merge_trailer(tr)
                    return True
                m = re.compile(rb"(\d+)\s+(\d+)").match(self.buf, p.pos)
                if not m:
                    return False
                first, count = int(m.group(1)), int(m.group(2))
                p.pos = m.end()
                p.skip_ws()
                end = len(self.buf)
                if count > MAX_XREF_ENTRIES:
                    return False
                # A declared row count cannot exceed the rows the file can
                # hold; never loop past EOF on an attacker-chosen count.
                count = min(count, max(0, (end - p.pos) // 20 + 1))
                for i in range(count):
                    if p.pos >= end:
                        break
                    ent = self.buf[p.pos : p.pos + 20]
                    em = re.match(rb"\s*(\d{1,10})\s+(\d{1,5})\s+([nf])", ent)
                    if not em:
                        p.pos += 20
                        continue
                    num = first + i
                    if em.group(3) == b"n" and num not in self.xref:
                        self.xref[num] = int(em.group(1))
                    p.pos += 20
            return False

        m = re.compile(rb"(\d+)\s+(\d+)\s+obj").match(self.buf, off)
        if not m:
            return False
        obj = self._parse_indirect_at(off)
        if not isinstance(obj, Stream):
            return False
        self._merge_trailer(obj.dict)
        fields = self._xref_stream_fields(obj)
        if fields is None:
            return False
        w, index = fields
        data = obj.data
        sz = sum(w)
        pos = 0
        for k in range(0, len(index) - 1, 2):
            first, count = index[k], index[k + 1]
            for i in range(count):
                if pos + sz > len(data):
                    break
                fields = []
                for width in w:
                    val = 0
                    for _ in range(width):
                        val = (val << 8) | data[pos]
                        pos += 1
                    fields.append(val)
                ftype = fields[0] if w[0] else 1
                num = first + i
                if ftype == 1 and num not in self.xref:
                    self.xref[num] = fields[1]
                elif ftype == 2 and num not in self.xref:
                    self.compressed[num] = (fields[1], fields[2])
        return True

    def _merge_trailer(self, tr):
        for k, v in tr.items():
            if k not in self.trailer:
                self.trailer[k] = v
        if "XRefStm" in tr and isinstance(tr["XRefStm"], int):
            self._load_xref_at(tr["XRefStm"])

    def _rebuild_xref(self):
        for m in re.finditer(rb"(?:^|[\r\n\s])(\d+)\s+(\d+)\s+obj\b", self.buf):
            self.xref[int(m.group(1))] = m.start(1)

    def _recover_catalog(self):
        """Last resort when the trailer does not name a root object.

        Cross-reference streams and hybrid files do not always leave a usable
        trailer behind, so fall back to indexing every object and picking the
        one typed as the document catalogue.
        """
        if not self.xref:
            self._rebuild_xref()
        # objects may live inside /ObjStm containers, which the top-level scan
        # cannot see
        self._index_objstms()
        for num in sorted(self.xref):
            obj = self.get_object(num)
            if isinstance(obj, dict) and obj.get("Type") == "Catalog":
                self.trailer["Root"] = Ref(num, 0)
                return
        for num, obj in sorted(self._extra.items()):
            if isinstance(obj, dict) and obj.get("Type") == "Catalog":
                self.trailer["Root"] = Ref(num, 0)
                return

    def resolve(self, obj, depth=0):
        while isinstance(obj, Ref) and depth < 64:
            obj = self.get_object(obj.num, depth + 1)
            depth += 1
        return obj

    def get_object(self, num, depth=0):
        if num in self.cache:
            return self.cache[num]
        if num in self._loading:
            return None
        self._loading.add(num)
        try:
            val = None
            if num in self.xref:
                val = self._parse_indirect_at(self.xref[num])
            if val is None and num in self.compressed:
                val = self._load_from_objstm(num)
            if val is None and not self._objstm_indexed:
                # A lookup miss may simply mean the cross-reference table did
                # not list an object that lives inside an /ObjStm. Index every
                # container once, then retry.
                self._index_objstms()
                val = self._extra.get(num)
            if val is None and num in self._extra:
                val = self._extra[num]
            self.cache[num] = val
            return val
        finally:
            self._loading.discard(num)

    def _parse_indirect_at(self, off):
        m = re.compile(rb"\s*(\d+)\s+(\d+)\s+obj").match(self.buf, off)
        if not m:
            m = re.compile(rb"\s*(\d+)\s+(\d+)\s+obj").search(self.buf, off)
            if not m:
                return None
        p = Parser(self.buf, m.end(), self)
        p._pending_dict = {}
        val = p.parse()
        if isinstance(val, dict) and p._stream_dict is not None:
            return Stream(p._stream_dict, p._stream_raw, self)
        return val

    def _load_from_objstm(self, num):
        cnum, _idx = self.compressed[num]
        entries = self._objstm_cache.get(cnum)
        if entries is None:
            entries = self._read_objstm(cnum)
            self._objstm_cache[cnum] = entries
        return entries.get(num)

    def _index_objstms(self):
        if self._objstm_indexed:
            return
        self._objstm_indexed = True
        for num in sorted(self.xref):
            try:
                obj = self._parse_indirect_at(self.xref[num])
            except PdfResourceLimitError:
                raise
            except Exception:  # noqa: BLE001, S112 - a broken object is skipped.
                continue
            if isinstance(obj, Stream) and obj.dict.get("Type") == "ObjStm":
                try:
                    for onum, oobj in self._read_objstm(num).items():
                        self._extra.setdefault(onum, oobj)
                except PdfResourceLimitError:
                    raise
                except Exception:  # noqa: BLE001, S112 - a broken stream is skipped.
                    continue

    def _read_objstm(self, cnum):
        st = self.resolve(Ref(cnum, 0))
        if not isinstance(st, Stream):
            return {}
        n = self.resolve(st.dict.get("N")) or 0
        first = self.resolve(st.dict.get("First")) or 0
        data = st.data
        hdr = Parser(data[:first] if first else data, 0, self)
        pairs = []
        for _ in range(int(n)):
            a = hdr.parse()
            b = hdr.parse()
            if not isinstance(a, int) or not isinstance(b, int):
                break
            pairs.append((a, b))
        return {onum: Parser(data, first + ooff, self).parse() for onum, ooff in pairs}

    INHERIT = ("Resources", "MediaBox", "CropBox", "Rotate")

    def pages(self):
        root = self.resolve(self.trailer.get("Root"))
        out = []
        if not isinstance(root, dict):
            return out
        self._walk(self.resolve(root.get("Pages")), out, {}, set())
        return out

    def _walk(self, node, out, inherited, seen, depth=0):
        if depth > 60 or not isinstance(node, dict):
            return
        if id(node) in seen:
            return
        seen.add(id(node))
        inh = dict(inherited)
        for k in self.INHERIT:
            if k in node:
                inh[k] = node[k]
        kids = self.resolve(node.get("Kids"))
        if node.get("Type") == "Page" or (kids is None and "Contents" in node):
            pg = dict(node)
            for k, v in inh.items():
                pg.setdefault(k, v)
            out.append(pg)
            return
        if isinstance(kids, list):
            for k in kids:
                self._walk(self.resolve(k), out, inh, seen, depth + 1)

    def page_content(self, page):
        c = self.resolve(page.get("Contents"))
        parts = []
        if isinstance(c, Stream):
            parts.append(c.data)
        elif isinstance(c, list):
            for x in c:
                r = self.resolve(x)
                if isinstance(r, Stream):
                    parts.append(r.data)
        return b"\n".join(parts)
