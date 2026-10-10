"""Read unencrypted MOBI in memory through pinned KindleUnpack components.

No unpackBook/extract call, generated EPUB, resource extraction or parser-selected
output path is used. KF8 parts are reconstructed in memory with debug writes off.
Container/header bounds and decoded text length fail closed before HTML cleanup.
"""

from __future__ import annotations

import json
import os
import selectors
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from mobi.mobi_header import MobiHeader
from mobi.mobi_k8proc import K8Processor
from mobi.mobi_sectioner import Sectionizer

from ..project.config import child_process_environment

MAX_MOBI_BYTES = 128 * 1024 * 1024
MAX_MOBI_TEXT_BYTES = 64 * 1024 * 1024
MAX_MOBI_IPC_BYTES = 16 * 1024 * 1024
MAX_MOBI_HTML_TAGS = 50_000
MAX_MOBI_UNITS = 10_000
MOBI_TIMEOUT_SECONDS = 45


class MobiReadError(ValueError):
    """Unsupported, protected or structurally invalid MOBI evidence."""


@dataclass(frozen=True)
class MobiText:
    metadata: dict[str, list[str]]
    parts: tuple[str, ...]
    version: int
    compression: int


def _validate_header(sections: Sectionizer, start: int) -> None:
    header = sections.loadSection(start)
    if len(header) < 0x84 or header[16:20] != b"MOBI":
        raise MobiReadError("Not a MOBI header (PalmDOC-only files are unsupported)")
    if struct.unpack_from(">H", header, 12)[0]:
        raise MobiReadError("DRM/encrypted MOBI is unsupported")
    length, kind, codepage, _uid, version = struct.unpack_from(">5I", header, 20)
    if length < 0x74 or length + 16 > len(header):
        raise MobiReadError("Corrupt MOBI header length")
    if kind != 2 or version not in {4, 5, 6, 7, 8}:
        raise MobiReadError("Unsupported MOBI type/version (reflowable books only)")
    if codepage not in {1252, 65001}:
        raise MobiReadError("Unsupported MOBI text encoding")
    records = struct.unpack_from(">H", header, 8)[0]
    if struct.unpack_from(">H", header, 0)[0] not in {1, 2, 0x4448}:
        raise MobiReadError("Unsupported MOBI compression")
    if not 0 < struct.unpack_from(">I", header, 4)[0] <= MAX_MOBI_TEXT_BYTES:
        raise MobiReadError("MOBI text size exceeds the 64 MiB safety limit")
    if not records or start + records >= sections.num_sections:
        raise MobiReadError("Corrupt MOBI text record count")
    offset, size = struct.unpack_from(">2I", header, 0x54)
    if offset < 16 + length or offset + size > len(header):
        raise MobiReadError("Corrupt MOBI title bounds")
    if struct.unpack_from(">I", header, 0x80)[0] & 0x40:
        pos = 16 + length
        if header[pos : pos + 4] != b"EXTH" or pos + 12 > len(header):
            raise MobiReadError("Corrupt MOBI EXTH header")
        size, count = struct.unpack_from(">2I", header, pos + 4)
        end = pos + size
        if size < 12 or end > len(header):
            raise MobiReadError("Corrupt MOBI EXTH bounds")
        pos += 12
        for _ in range(count):
            if pos + 8 > end:
                raise MobiReadError("Corrupt MOBI metadata record")
            size = struct.unpack_from(">I", header, pos + 4)[0]
            if size < 8 or pos + size > end:
                raise MobiReadError("Corrupt MOBI metadata record bounds")
            pos += size


def _read_mobi_in_process(path: Path) -> MobiText:
    """Return one rendition: MOBI7 for dual books, reconstructed parts for KF8."""
    try:
        if path.is_symlink() or not path.is_file():
            raise MobiReadError("MOBI must be a regular, non-symlink source")
        if path.stat().st_size > MAX_MOBI_BYTES:
            raise MobiReadError("MOBI exceeds the 128 MiB safety limit")
        sections = Sectionizer(str(path))
        offsets = sections.sectionoffsets
        if (
            sections.ident != b"BOOKMOBI"
            or sections.num_sections < 2
            or offsets[0] < 78 + 8 * sections.num_sections
            or any(left >= right for left, right in pairwise(offsets))
        ):
            raise MobiReadError("Corrupt or unsupported MOBI container")
        _validate_header(sections, 0)
        header = MobiHeader(sections, 0)
        if header.isEncrypted():
            raise MobiReadError("DRM/encrypted MOBI is unsupported")
        if header.isPrintReplica() or header.isDictionary():
            raise MobiReadError("Print Replica and dictionary MOBI are unsupported")
        # The unused second rendition gets header/bounds/compression/DRM checks,
        # not decompression or KF8 reconstruction. Only selected text is evidence.
        boundary = header.getMetaData().get("Mobi8-Boundary-Section", [])
        if boundary and int(boundary[0]) != 0xFFFFFFFF:
            index = int(boundary[0])
            if index < 1 or index + 1 >= sections.num_sections:
                raise MobiReadError("Corrupt MOBI8 boundary")
            if sections.loadSection(index) != b"BOUNDARY":
                raise MobiReadError("Corrupt MOBI8 boundary marker")
            _validate_header(sections, index + 1)
        raw = header.getRawML()
        if len(raw) != struct.unpack_from(">I", header.header, 4)[0]:
            raise MobiReadError("Corrupt MOBI decompressed text length")
        if header.isK8():
            processor = K8Processor(header, sections, None, debug=False)
            if not processor.skeltbl:
                raise MobiReadError("KF8 has no reconstructable text skeletons")
            _validate_kf8(processor, raw)
            processor.buildParts(raw)
            parts = tuple(
                processor.getPart(i) for i in range(processor.getNumberOfParts())
            )
        else:
            parts = (raw,)
        if sum(len(part) for part in parts) > MAX_MOBI_TEXT_BYTES:
            raise MobiReadError(
                "Reconstructed MOBI text exceeds the 64 MiB safety limit"
            )
        decoded = tuple(part.decode(header.codec, errors="strict") for part in parts)
        if not any(part.strip() for part in decoded):
            raise MobiReadError("MOBI has no text")
        return MobiText(
            header.getMetaData(), decoded, header.version, header.compression
        )
    except MobiReadError:
        raise
    except Exception as exc:  # parser errors are source-local, never garbled fallback
        raise MobiReadError(f"Cannot parse MOBI: {type(exc).__name__}: {exc}") from exc


def _validate_kf8(processor: K8Processor, raw: bytes) -> None:
    """Check reconstruction slices rather than trusting Python's silent truncation."""
    flow_end = min(processor.fdsttbl[1], len(raw))
    if processor.fdsttbl[0] != 0 or flow_end <= 0:
        raise MobiReadError("Corrupt KF8 text flow bounds")
    if processor.fdst != 0xFFFFFFFF and (
        any(offset > len(raw) for offset in processor.fdsttbl)
        or any(left >= right for left, right in pairwise(processor.fdsttbl))
    ):
        raise MobiReadError("Corrupt KF8 flow table")
    fragment = 0
    for _number, _name, count, offset, size in processor.skeltbl:
        end = offset + size
        if offset < 0 or size <= 0 or end > flow_end:
            raise MobiReadError("Corrupt KF8 skeleton bounds")
        if count < 0 or fragment + count > len(processor.fragtbl):
            raise MobiReadError("Corrupt KF8 fragment count")
        assembled_size = size
        for insert, _aid, _file, _sequence, _start, length in processor.fragtbl[
            fragment : fragment + count
        ]:
            if (
                length < 0
                or end + length > flow_end
                or not offset <= insert <= offset + assembled_size
            ):
                raise MobiReadError("Corrupt KF8 fragment bounds")
            end += length
            assembled_size += length
        fragment += count
    if fragment != len(processor.fragtbl):
        raise MobiReadError("Unmatched KF8 fragments")


def read_mobi(path: Path) -> MobiText:
    """Bound parser memory/CPU in an offline, output-free child process."""
    payload = _worker_payload(path)
    try:
        return MobiText(
            payload["metadata"],
            tuple(payload["parts"]),
            payload["version"],
            payload["compression"],
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise MobiReadError("MOBI parser returned an invalid response") from exc


def extract_mobi_payload(path: Path, request: dict) -> tuple[dict, list[dict]]:
    """Return bounded serialized units, never HTML for the serving parent to parse."""
    payload = _worker_payload(path, request)
    document, units = payload.get("document"), payload.get("units")
    if (
        not isinstance(document, dict)
        or not isinstance(units, list)
        or len(units) > MAX_MOBI_UNITS
    ):
        raise MobiReadError("MOBI worker returned invalid extraction units")
    return document, units


def _worker_payload(path: Path, request: dict | None = None) -> dict:
    """Drain both pipes under byte/time caps before loading any JSON in the parent."""
    command = [sys.executable, "-m", "research_rag.corpus.mobi_reader", str(path)]
    if request is not None:
        command.append("--extract")
    environment = child_process_environment()
    # HTML worker imports the shared extraction helpers lazily. Do not let a
    # numerical library's idle thread stacks consume the worker's address budget.
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + MOBI_TIMEOUT_SECONDS
    encoded = json.dumps(request).encode("utf-8") if request is not None else b""
    if len(encoded) > 64 * 1024:
        raise MobiReadError("MOBI worker request exceeds its safety limit")
    with subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            assert process.stdin is not None
            sent = 0
            input_error = None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                if encoded:
                    os.set_blocking(process.stdin.fileno(), False)
                    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
                else:
                    process.stdin.close()
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise MobiReadError(
                            "MOBI parser exceeded its 45-second safety limit"
                        )
                    for key, _events in selector.select(remaining):
                        if key.data == "stdin":
                            try:
                                sent += os.write(key.fd, encoded[sent:])
                            except BlockingIOError:
                                continue
                            except OSError as exc:
                                # Keep draining diagnostics on early refusal;
                                # normalize the closed request pipe after reap.
                                input_error = exc
                            if input_error is not None or sent == len(encoded):
                                selector.unregister(key.fileobj)
                                process.stdin.close()
                            continue
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        target = buffers[key.data]
                        maximum = (
                            MAX_MOBI_IPC_BYTES if key.data == "stdout" else 64 * 1024
                        )
                        if len(target) + len(chunk) > maximum:
                            raise MobiReadError(
                                "MOBI worker output exceeds its IPC safety limit"
                            )
                        target.extend(chunk)
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if process.returncode or input_error is not None:
                detail = (
                    buffers["stderr"].decode("utf-8", errors="replace").strip()[-2000:]
                )
                raise MobiReadError(
                    detail
                    or (
                        "MOBI worker closed its request pipe before receiving the source"
                        if input_error is not None
                        else "MOBI parser exceeded a resource safety limit"
                    )
                )
        except subprocess.TimeoutExpired as exc:
            raise MobiReadError(
                "MOBI parser exceeded its 45-second safety limit"
            ) from exc
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    try:
        payload = json.loads(buffers["stdout"])
        if not isinstance(payload, dict):
            raise TypeError("Expected a JSON object")
        return payload
    except (ValueError, TypeError) as exc:
        raise MobiReadError("MOBI parser returned an invalid response") from exc


def _write_payload(payload: dict) -> None:
    """Stream JSON with an output budget; never build an unbounded JSON string."""
    total = 0
    for piece in json.JSONEncoder(ensure_ascii=False).iterencode(payload):
        encoded = piece.encode("utf-8")
        total += len(encoded)
        if total > MAX_MOBI_IPC_BYTES:
            raise MobiReadError(
                "MOBI worker output exceeds its 16 MiB IPC safety limit"
            )
        sys.stdout.buffer.write(encoded)


def _main() -> int:
    import contextlib
    import io
    from dataclasses import asdict

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        request = None
        if "--extract" in sys.argv[2:]:
            encoded_request = sys.stdin.buffer.read(64 * 1024 + 1)
            if len(encoded_request) > 64 * 1024:
                raise MobiReadError("MOBI worker request exceeds its safety limit")
            request = json.loads(encoded_request)
        # The dependency occasionally prints malformed-record diagnostics. They
        # must not contaminate the JSON handoff, and the failed source owns them.
        diagnostics = io.StringIO()
        with contextlib.redirect_stdout(diagnostics):
            parsed = _read_mobi_in_process(Path(sys.argv[1]))
            if request is not None:
                if sum(part.count("<") for part in parsed.parts) > MAX_MOBI_HTML_TAGS:
                    raise MobiReadError("MOBI HTML exceeds its 50,000-tag safety limit")
                # All source-driven HTML parsing, metadata inference and unit
                # construction execute inside the same memory/CPU limits.
                from .extraction import _mobi_units_from_text
                from .sources import SourceFile

                source = SourceFile(path=Path(sys.argv[1]), **request["source"])
                document, units = _mobi_units_from_text(
                    source, request["digest"], parsed
                )
                if len(units) > MAX_MOBI_UNITS:
                    raise MobiReadError("MOBI exceeds its 10,000-unit safety limit")
                payload = {"document": document, "units": units}
            else:
                payload = asdict(parsed)
        if "fixed corrupt fragment table" in diagnostics.getvalue():
            raise MobiReadError(
                "Corrupt KF8 fragment insertion table; repair is unsupported"
            )
        _write_payload(payload)
        return 0
    except (MobiReadError, ImportError, OSError, ValueError, MemoryError) as exc:
        print(f"MOBI refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
