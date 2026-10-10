"""Original synthetic MOBI containers; no private or copyrighted book bytes."""

import struct
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

TEXT = (
    "<html><head><title>Synthetic Evidence</title></head><body>"
    '<h1 id="opening">Opening argument</h1>'
    "<p>Research evidence explains how cobalt institutions organise social "
    "relations through shared practices and material conditions. "
    "This paragraph preserves the author’s reasoning and café punctuation.</p>"
    '<h2 id="second">Second argument</h2>'
    "<p>Amber mechanisms connect collective action with durable outcomes. "
    "The evidence remains accessible in the original ebook.</p>"
    "<script>DO NOT INDEX SCRIPT</script></body></html>"
)


def mobi_bytes(html: str = TEXT, *, compression: int = 1, drm: int = 0) -> bytes:
    raw = html.encode("utf-8")
    title = b"Synthetic Evidence"
    exth_records = []
    for key, value in ((100, b"Example Author"), (106, b"2024-01-02")):
        exth_records.append(struct.pack(">II", key, 8 + len(value)) + value)
    body = b"".join(exth_records)
    exth = b"EXTH" + struct.pack(">II", 12 + len(body), len(exth_records)) + body
    exth += b"\0" * (-len(exth) % 4)
    header = bytearray(0xF4)
    title_offset = len(header) + len(exth)
    text_records = [raw[i : i + 4096] for i in range(0, len(raw), 4096)]
    struct.pack_into(
        ">HHIHHHH", header, 0, compression, 0, len(raw), len(text_records), 4096, drm, 0
    )
    header[16:20] = b"MOBI"
    struct.pack_into(">5I", header, 20, 0xE4, 2, 65001, 123, 6)
    struct.pack_into(">2I", header, 0x28, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into(">I", header, 0x50, len(text_records) + 1)
    struct.pack_into(">2I", header, 0x54, title_offset, len(title))
    struct.pack_into(">I", header, 0x5C, 9)  # English language code
    struct.pack_into(">I", header, 0x68, 6)
    struct.pack_into(">I", header, 0x6C, len(text_records) + 1)
    struct.pack_into(">I", header, 0x80, 0x40)
    header += exth + title
    if compression == 2:
        # Literal-run encoder, not a decoder: exercise the upstream PalmDOC
        # decoder on Unicode bytes without shipping somebody else's book.
        text_records = [
            b"".join(
                bytes((len(record[i : i + 8]),)) + record[i : i + 8]
                for i in range(0, len(record), 8)
            )
            for record in text_records
        ]
    return palm_container([bytes(header), *text_records])


def palm_container(records: list[bytes]) -> bytes:
    palm = bytearray(78)
    palm[:9] = b"Synthetic"
    palm[60:68] = b"BOOKMOBI"
    struct.pack_into(">H", palm, 76, len(records))
    offset = 78 + 8 * len(records) + 2
    entries = []
    for index, record in enumerate(records):
        entries.append(struct.pack(">II", offset, index + 1))
        offset += len(record)
    return bytes(palm) + b"".join(entries) + b"\0\0" + b"".join(records)


def write_mobi(path: Path, html: str = TEXT, **kwargs) -> None:
    path.write_bytes(mobi_bytes(html, **kwargs))


def kf8_bytes(html: str = TEXT) -> bytes:
    """A real KF8 container with one INDX skeleton and no stripped fragments."""
    legacy = mobi_bytes(html)
    header_offset = struct.unpack_from(">I", legacy, 78)[0]
    text_offset = struct.unpack_from(">I", legacy, 86)[0]
    old_header = legacy[header_offset:text_offset]
    header = bytearray(old_header[:0xF4] + b"\0" * 36 + old_header[0xF4:])
    struct.pack_into(">I", header, 20, 0x108)
    struct.pack_into(">I", header, 0x24, 8)
    struct.pack_into(">I", header, 0x54, struct.unpack_from(">I", header, 0x54)[0] + 36)
    struct.pack_into(">II", header, 0xC0, 0xFFFFFFFF, 1)
    struct.pack_into(">IIII", header, 0xF4, 0xFFFFFFFF, 0xFFFFFFFF, 2, 0)
    struct.pack_into(">I", header, 0x104, 0xFFFFFFFF)

    def varint(value):
        result = [value & 127 | 128]
        value >>= 7
        while value:
            result.insert(0, value & 127)
            value >>= 7
        return bytes(result)

    raw = html.encode("utf-8")
    main = bytearray(192)
    main[:4] = b"INDX"
    struct.pack_into(">I", main, 4, 192)
    struct.pack_into(">I", main, 24, 1)
    struct.pack_into(">I", main, 28, 65001)
    main += b"TAGX" + struct.pack(">II", 20, 1) + bytes((1, 1, 1, 0, 6, 2, 2, 0))
    entry = b"\x01a\x03" + varint(0) + varint(0) + varint(len(raw))
    extra = bytearray(192)
    extra[:4] = b"INDX"
    struct.pack_into(">I", extra, 4, 192)
    struct.pack_into(">II", extra, 20, 192 + len(entry), 1)
    extra += entry + b"IDXT" + struct.pack(">H", 192)
    return palm_container([bytes(header), raw, bytes(main), bytes(extra)])


def huff_mobi_bytes(
    html: str = TEXT, *, recursive: bool = False, cycle: bool = False
) -> bytes:
    """HUFF/CDIC terminal dictionary, optionally with a recursive phrase or cycle."""
    legacy = mobi_bytes(html)
    start = struct.unpack_from(">I", legacy, 78)[0]
    text = struct.unpack_from(">I", legacy, 86)[0]
    header = bytearray(legacy[start:text])
    struct.pack_into(">H", header, 0, 0x4448)
    struct.pack_into(">II", header, 0x70, 2, 2)
    huff = b"HUFF" + struct.pack(">IIIII", 24, 24, 1048, 0, 0)
    huff += struct.pack(">256I", *([255 << 8 | 128 | 8] * 256))
    huff += b"\0" * (64 * 4)
    encoded = html.encode("utf-8")
    entries = [struct.pack(">H", 0x8001) + bytes((255 - i,)) for i in range(256)]
    if recursive or cycle:
        encoded = encoded.replace(b"Research", b"\xff", 1)
        phrase = b"\xff" if cycle else b"Research"
        entries[0] = struct.pack(">H", len(phrase)) + phrase
    offsets = []
    offset = 512
    for entry in entries:
        offsets.append(offset)
        offset += len(entry)
    phrases = b"".join(entries)
    cdic = b"CDIC" + struct.pack(">III", 16, 256, 8)
    cdic += struct.pack(">256H", *offsets) + phrases
    return palm_container([bytes(header), encoded, huff, cdic])


def dual_mobi_bytes(*, second_drm: int = 0, second_compression: int = 1) -> bytes:
    legacy = mobi_bytes()
    start, text = (
        struct.unpack_from(">I", legacy, 78)[0],
        struct.unpack_from(">I", legacy, 86)[0],
    )
    header = bytearray(legacy[start:text])
    title_offset = struct.unpack_from(">I", header, 0x54)[0]
    boundary = struct.pack(">III", 121, 12, 2)
    header[title_offset:title_offset] = boundary
    struct.pack_into(">I", header, 0x54, title_offset + 12)
    exth_size, count = struct.unpack_from(">II", header, 0xF8)
    struct.pack_into(">II", header, 0xF8, exth_size + 12, count + 1)
    modern = kf8_bytes()
    offsets = [struct.unpack_from(">I", modern, 78 + i * 8)[0] for i in range(4)] + [
        len(modern)
    ]
    records = [modern[left:right] for left, right in pairwise(offsets)]
    modern_header = bytearray(records[0])
    struct.pack_into(">H", modern_header, 12, second_drm)
    struct.pack_into(">H", modern_header, 0, second_compression)
    return palm_container(
        [bytes(header), legacy[text:], b"BOUNDARY", bytes(modern_header), *records[1:]]
    )


def kf8_fragment_bytes(*, corrupt_length: bool = False) -> bytes:
    """A KF8 skeleton plus one actual body fragment, including its CTOC entry."""
    original = TEXT.encode("utf-8")
    start = original.index(b"<body>") + len(b"<body>")
    end = original.index(b"</body>")
    body = original[start:end]
    skeleton = original[:start] + original[end:]
    modern = kf8_bytes(skeleton.decode("utf-8"))
    header_start = struct.unpack_from(">I", modern, 78)[0]
    text_start = struct.unpack_from(">I", modern, 86)[0]
    header = bytearray(modern[header_start:text_start])
    struct.pack_into(">I", header, 4, len(skeleton) + len(body))
    struct.pack_into(">I", header, 0xF8, 4)  # fragment index

    def vwi(value):
        encoded = [value & 127 | 128]
        value >>= 7
        while value:
            encoded.insert(0, value & 127)
            value >>= 7
        return bytes(encoded)

    def index_pair(tags, text, values, *, ctoc_count=0):
        main = bytearray(192)
        main[:4] = b"INDX"
        struct.pack_into(">I", main, 4, 192)
        struct.pack_into(">I", main, 24, 1)
        struct.pack_into(">I", main, 28, 65001)
        struct.pack_into(">I", main, 52, ctoc_count)
        table = b"".join(bytes(tag) for tag in tags)
        main += b"TAGX" + struct.pack(">II", 12 + len(table), 1) + table
        entry = bytes((len(text),)) + text + bytes((sum(tag[2] for tag in tags),))
        entry += b"".join(vwi(value) for value in values)
        extra = bytearray(192)
        extra[:4] = b"INDX"
        struct.pack_into(">I", extra, 4, 192)
        struct.pack_into(">II", extra, 20, 192 + len(entry), 1)
        extra += entry + b"IDXT" + struct.pack(">H", 192)
        return bytes(main), bytes(extra)

    skel_main, skel_extra = index_pair(
        [(1, 1, 1, 0), (6, 2, 2, 0)], b"s", [1, 0, len(skeleton)]
    )
    frag_main, frag_extra = index_pair(
        [(2, 1, 1, 0), (3, 1, 2, 0), (4, 1, 4, 0), (6, 2, 8, 0)],
        str(start).encode(),
        [0, 0, 0, 0, len(body) + (1000 if corrupt_length else 0)],
        ctoc_count=1,
    )
    aid = b"aid='fragment'"
    ctoc = vwi(len(aid)) + aid
    return palm_container(
        [
            bytes(header),
            skeleton + body,
            skel_main,
            skel_extra,
            frag_main,
            frag_extra,
            ctoc,
        ]
    )


def install_small_pipe_worker(monkeypatch, script: str):
    """Use a real subprocess and a real 4 KiB pipe for deterministic IPC probes."""
    from research_rag.corpus import mobi_reader

    original = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        if "research_rag.corpus.mobi_reader" not in command:
            return original(command, **kwargs)
        process = original([sys.executable, "-c", script], pipesize=4096, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mobi_reader.subprocess, "Popen", launch)
    return processes
