"""Compact generation-local indexes for canonical JSONL artifacts.

The SQLite sidecar stores identifiers, ordinals, content digests, and byte
offsets only.  Passage and extraction text remain canonical in the JSONL files
and are read on demand, so the index does not duplicate corpus text.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .extraction import chunk_health_flags
from .storage import fsync_directory

# Key under which a loaded chunk record carries its stored rejection verdict.
LOOKUP_HEALTH_FLAGS_KEY = "_lookup_health_flags"
LOOKUP_SCHEMA_VERSION = 3
LOOKUP_RELATIVE_PATH = Path("indexes") / "artifact-lookup.sqlite3"
_SQL_BATCH_SIZE = 500


class ArtifactLookupError(RuntimeError):
    """Raised when a generation lookup cannot be built or queried safely."""


def _chunk_text(record: Mapping[str, Any]) -> str:
    return str(
        record.get("contents")
        or record.get("text")
        or record.get("embedding_text")
        or ""
    )


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _metadata_key(kind: str, field: str) -> str:
    return f"{kind}_{field}"


def _batched(values: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(values), _SQL_BATCH_SIZE):
        yield values[offset : offset + _SQL_BATCH_SIZE]


def _read_json_line(path: Path, line: bytes, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactLookupError(
            f"Invalid JSON in {path} at line {line_number}"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactLookupError(f"Expected an object in {path} at line {line_number}")
    return value


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE chunks (
            ordinal INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            document_chunk_index INTEGER NOT NULL,
            content_sha256 BLOB NOT NULL,
            byte_offset INTEGER NOT NULL,
            byte_length INTEGER NOT NULL,
            health_flags INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX chunks_document_position
            ON chunks(document_id, document_chunk_index);
        CREATE INDEX chunks_content ON chunks(content_sha256);
        CREATE TABLE units (
            ordinal INTEGER PRIMARY KEY,
            unit_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            byte_offset INTEGER NOT NULL,
            byte_length INTEGER NOT NULL
        );
        CREATE INDEX units_document ON units(document_id);
        """
    )
    connection.execute(f"PRAGMA user_version = {LOOKUP_SCHEMA_VERSION}")


def _index_chunks(
    connection: sqlite3.Connection,
    path: Path,
) -> tuple[int, str, dict[str, int]]:
    before = _file_identity(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            offset = handle.tell() - len(line)
            digest.update(line)
            if not line.strip():
                continue
            record = _read_json_line(path, line, line_number)
            chunk_id = record.get("chunk_id")
            document_id = record.get("document_id")
            document_chunk_index = record.get("document_chunk_index")
            contents = _chunk_text(record)
            if (
                not isinstance(chunk_id, str)
                or not chunk_id
                or not isinstance(document_id, str)
                or not document_id
                or isinstance(document_chunk_index, bool)
                or not isinstance(document_chunk_index, int)
                or document_chunk_index < 0
                or not contents
            ):
                raise ArtifactLookupError(
                    f"Invalid chunk lookup fields in {path} at line {line_number}"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO chunks(
                        ordinal, chunk_id, document_id,
                        document_chunk_index, content_sha256,
                        byte_offset, byte_length, health_flags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        count,
                        chunk_id,
                        document_id,
                        document_chunk_index,
                        hashlib.sha256(contents.encode("utf-8")).digest(),
                        offset,
                        len(line),
                        chunk_health_flags(
                            contents,
                            quality_flags=record.get("quality_flags"),
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ArtifactLookupError(
                    "Duplicate chunk identifier or document position in "
                    f"{path}: {chunk_id}"
                ) from exc
            count += 1
    after = _file_identity(path)
    if before != after:
        raise ArtifactLookupError(f"Artifact changed while it was indexed: {path}")
    return count, digest.hexdigest(), after


def _index_units(
    connection: sqlite3.Connection,
    path: Path,
) -> tuple[int, str, dict[str, int]]:
    before = _file_identity(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            offset = handle.tell() - len(line)
            digest.update(line)
            if not line.strip():
                continue
            record = _read_json_line(path, line, line_number)
            unit_id = record.get("id")
            document_id = record.get("document_id")
            if (
                not isinstance(unit_id, str)
                or not unit_id
                or not isinstance(document_id, str)
                or not document_id
            ):
                raise ArtifactLookupError(
                    f"Invalid extraction-unit lookup fields in {path} "
                    f"at line {line_number}"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO units(
                        ordinal, unit_id, document_id,
                        byte_offset, byte_length
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        count,
                        unit_id,
                        document_id,
                        offset,
                        len(line),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ArtifactLookupError(
                    f"Duplicate extraction-unit identifier in {path}: {unit_id}"
                ) from exc
            count += 1
    after = _file_identity(path)
    if before != after:
        raise ArtifactLookupError(f"Artifact changed while it was indexed: {path}")
    return count, digest.hexdigest(), after


def _write_metadata(
    connection: sqlite3.Connection,
    kind: str,
    count: int,
    digest: str,
    identity: Mapping[str, int],
) -> None:
    values = {"count": count, "sha256": digest, **identity}
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ((_metadata_key(kind, key), str(value)) for key, value in values.items()),
    )


def build_artifact_lookup(
    chunks_path: Path,
    units_path: Path,
    lookup_path: Path,
) -> None:
    """Atomically build a text-free offset index for two canonical JSONL files."""

    for artifact in (chunks_path, units_path):
        if (
            artifact.parent.is_symlink()
            or artifact.is_symlink()
            or not artifact.is_file()
        ):
            raise ArtifactLookupError(
                f"Canonical generation artifact is missing or unsafe: {artifact}"
            )
    if lookup_path.parent.is_symlink() or (
        lookup_path.parent.exists() and not lookup_path.parent.is_dir()
    ):
        raise ArtifactLookupError(
            f"Artifact lookup directory is unsafe: {lookup_path.parent}"
        )
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = lookup_path.with_name(f".{lookup_path.name}.{uuid.uuid4().hex}.tmp")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        _create_schema(connection)
        chunk_count, chunk_digest, chunk_identity = _index_chunks(
            connection,
            chunks_path,
        )
        unit_count, unit_digest, unit_identity = _index_units(
            connection,
            units_path,
        )
        _write_metadata(
            connection,
            "chunks",
            chunk_count,
            chunk_digest,
            chunk_identity,
        )
        _write_metadata(
            connection,
            "units",
            unit_count,
            unit_digest,
            unit_identity,
        )
        connection.commit()
        connection.close()
        connection = None
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, lookup_path)
        fsync_directory(lookup_path.parent)
    except (OSError, sqlite3.DatabaseError) as exc:
        raise ArtifactLookupError(f"Could not build artifact lookup: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata"))


def _stored_health_flags(row: sqlite3.Row) -> int | None:
    """Return the stored rejection verdict for a chunk row, if it has one.

    A lookup built before the verdict was stored, or a unit row, has no such
    column, and the caller then falls back to scanning the text at query time.
    """

    # `sqlite3.Row` iterates values, so membership has to be tested against its
    # column names explicitly.
    columns = row.keys()
    if "health_flags" not in columns:
        return None
    value = row["health_flags"]
    return None if value is None else int(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _lookup_is_current(
    chunks_path: Path,
    units_path: Path,
    lookup_path: Path,
) -> bool:
    if (
        chunks_path.is_symlink()
        or units_path.is_symlink()
        or not lookup_path.is_file()
        or lookup_path.is_symlink()
    ):
        return False
    try:
        with sqlite3.connect(lookup_path) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != LOOKUP_SCHEMA_VERSION:
                return False
            metadata = _read_metadata(connection)
            for kind, path in (("chunks", chunks_path), ("units", units_path)):
                identity = _file_identity(path)
                if any(
                    metadata.get(_metadata_key(kind, key)) != str(value)
                    for key, value in identity.items()
                ):
                    return False
                table_count = int(
                    connection.execute(f"SELECT COUNT(*) FROM {kind}").fetchone()[0]
                )
                if metadata.get(_metadata_key(kind, "count")) != str(table_count):
                    return False
            return True
    except (KeyError, OSError, sqlite3.DatabaseError, TypeError, ValueError):
        return False


def ensure_artifact_lookup(
    chunks_path: Path,
    units_path: Path,
    lookup_path: Path,
) -> ArtifactLookup:
    """Return a current lookup, rebuilding a missing or stale sidecar atomically."""

    if not _lookup_is_current(chunks_path, units_path, lookup_path):
        build_artifact_lookup(chunks_path, units_path, lookup_path)
    return ArtifactLookup(chunks_path, units_path, lookup_path)


class ArtifactLookup:
    """Read records from canonical artifacts through their compact offset index."""

    def __init__(self, chunks_path: Path, units_path: Path, lookup_path: Path) -> None:
        self.chunks_path = chunks_path
        self.units_path = units_path
        self.lookup_path = lookup_path

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                f"{self.lookup_path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.DatabaseError as exc:
            raise ArtifactLookupError(f"Could not open artifact lookup: {exc}") from exc

    @staticmethod
    def _load_rows(
        path: Path,
        rows: Sequence[sqlite3.Row],
    ) -> list[tuple[sqlite3.Row, dict[str, Any]]]:
        loaded: list[tuple[sqlite3.Row, dict[str, Any]]] = []
        try:
            with path.open("rb") as handle:
                for row in sorted(rows, key=lambda item: int(item["byte_offset"])):
                    handle.seek(int(row["byte_offset"]))
                    line = handle.read(int(row["byte_length"]))
                    record = _read_json_line(path, line, int(row["ordinal"]) + 1)
                    stored = _stored_health_flags(row)
                    if stored is not None:
                        # An internal key, read by the query path so it can reject
                        # a candidate without re-scanning its text. Public
                        # projections select their own fields and never expose it.
                        record[LOOKUP_HEALTH_FLAGS_KEY] = stored
                    loaded.append((row, record))
        except OSError as exc:
            raise ArtifactLookupError(
                f"Could not read generation artifact: {path}"
            ) from exc
        return loaded

    def chunk_count(self, document_ids: set[str] | None = None) -> int:
        with self._connect() as connection:
            if document_ids is None:
                return int(
                    connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                )
            if not document_ids:
                return 0
            values = sorted(document_ids)
            count = 0
            for batch in _batched(values):
                placeholders = ",".join("?" for _ in batch)
                count += int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM chunks "
                        f"WHERE document_id IN ({placeholders})",
                        tuple(batch),
                    ).fetchone()[0]
                )
            return count

    def chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(chunk_ids))
        if not unique:
            return {}
        rows: list[sqlite3.Row] = []
        with self._connect() as connection:
            for batch in _batched(unique):
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    connection.execute(
                        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                        tuple(batch),
                    ).fetchall()
                )
        return {
            str(row["chunk_id"]): record
            for row, record in self._load_rows(self.chunks_path, rows)
        }

    def chunks_by_contents(
        self,
        contents: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        requested = set(contents)
        result: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for _row, record in self._chunk_rows_by_contents(contents):
            text = _chunk_text(record)
            if text in requested:
                result[text].append(record)
        return dict(result)

    def _chunk_rows_by_contents(
        self,
        contents: Sequence[str],
    ) -> list[tuple[sqlite3.Row, dict[str, Any]]]:
        unique_contents = list(dict.fromkeys(contents))
        if not unique_contents:
            return []
        digests = {
            hashlib.sha256(value.encode("utf-8")).digest() for value in unique_contents
        }
        rows: list[sqlite3.Row] = []
        with self._connect() as connection:
            values = list(digests)
            for batch in _batched(values):
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    connection.execute(
                        f"SELECT * FROM chunks "
                        f"WHERE content_sha256 IN ({placeholders}) "
                        "ORDER BY ordinal",
                        tuple(batch),
                    ).fetchall()
                )
        loaded = self._load_rows(self.chunks_path, rows)
        return sorted(loaded, key=lambda item: int(item[0]["ordinal"]))

    def ordinals_by_contents(self, contents: Sequence[str]) -> dict[str, int]:
        """Return the first exact matching vector ordinal for each supplied text."""

        requested = set(contents)
        result: dict[str, int] = {}
        for row, record in self._chunk_rows_by_contents(contents):
            text = _chunk_text(record)
            if text in requested:
                result.setdefault(text, int(row["ordinal"]))
        return result

    def chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY document_chunk_index, ordinal
                """,
                (document_id,),
            ).fetchall()
        loaded = self._load_rows(self.chunks_path, rows)
        loaded.sort(
            key=lambda item: (
                int(item[0]["document_chunk_index"]),
                int(item[0]["ordinal"]),
            )
        )
        return [record for _row, record in loaded]

    def units_for_document(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM units
                WHERE document_id = ?
                ORDER BY ordinal
                """,
                (document_id,),
            ).fetchall()
        return [record for _row, record in self._load_rows(self.units_path, rows)]

    def document_ids(self, kind: str) -> list[str]:
        if kind not in {"chunks", "units"}:
            raise ValueError(f"Unsupported artifact kind: {kind}")
        with self._connect() as connection:
            return [
                str(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT document_id FROM {kind} ORDER BY document_id"
                )
            ]

    def has_document(self, kind: str, document_id: str) -> bool:
        if kind not in {"chunks", "units"}:
            raise ValueError(f"Unsupported artifact kind: {kind}")
        with self._connect() as connection:
            return (
                connection.execute(
                    f"SELECT 1 FROM {kind} WHERE document_id = ? LIMIT 1",
                    (document_id,),
                ).fetchone()
                is not None
            )

    def ordinal_for_contents(self, contents: str) -> int | None:
        return self.ordinals_by_contents([contents]).get(contents)


class DocumentArtifactMapping(Mapping[str, list[dict[str, Any]]]):
    """Lazy document-to-record mapping backed by an :class:`ArtifactLookup`."""

    def __init__(self, lookup: ArtifactLookup, kind: str) -> None:
        if kind not in {"chunks", "units"}:
            raise ValueError(f"Unsupported artifact kind: {kind}")
        self.lookup = lookup
        self.kind = kind

    def __getitem__(self, document_id: str) -> list[dict[str, Any]]:
        records = (
            self.lookup.chunks_for_document(document_id)
            if self.kind == "chunks"
            else self.lookup.units_for_document(document_id)
        )
        if not records:
            raise KeyError(document_id)
        return records

    def __iter__(self) -> Iterator[str]:
        return iter(self.lookup.document_ids(self.kind))

    def __len__(self) -> int:
        return len(self.lookup.document_ids(self.kind))

    def __contains__(self, document_id: object) -> bool:
        return isinstance(document_id, str) and self.lookup.has_document(
            self.kind,
            document_id,
        )


class VectorByContentsMapping(Mapping[str, Any]):
    """Lazy exact-text view over a memory-mapped embedding matrix."""

    def __init__(self, lookup: ArtifactLookup, vectors: Any) -> None:
        self.lookup = lookup
        self.vectors = vectors

    def __getitem__(self, contents: str) -> Any:
        ordinal = self.lookup.ordinal_for_contents(contents)
        if ordinal is None:
            raise KeyError(contents)
        return self.vectors[ordinal]

    def get_many(self, contents: Sequence[str]) -> dict[str, Any]:
        return {
            text: self.vectors[ordinal]
            for text, ordinal in self.lookup.ordinals_by_contents(contents).items()
        }

    def __iter__(self) -> Iterator[str]:
        with self.lookup._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks ORDER BY ordinal"
            ).fetchall()
        for _row, record in self.lookup._load_rows(self.lookup.chunks_path, rows):
            yield _chunk_text(record)

    def __len__(self) -> int:
        return self.lookup.chunk_count()


def validate_artifact_lookup(
    chunks_path: Path,
    units_path: Path,
    lookup_path: Path,
    *,
    expected_chunk_count: int,
    expected_unit_count: int,
) -> bool:
    """Fully validate sidecar integrity and every referenced canonical record."""

    if not _lookup_is_current(chunks_path, units_path, lookup_path):
        return False
    try:
        lookup = ArtifactLookup(chunks_path, units_path, lookup_path)
        with lookup._connect() as connection:
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                return False
            metadata = _read_metadata(connection)
            if (
                int(metadata[_metadata_key("chunks", "count")]) != expected_chunk_count
                or int(metadata[_metadata_key("units", "count")]) != expected_unit_count
            ):
                return False
            chunk_rows = connection.execute(
                "SELECT * FROM chunks ORDER BY ordinal"
            ).fetchall()
            unit_rows = connection.execute(
                "SELECT * FROM units ORDER BY ordinal"
            ).fetchall()
        if (
            len(chunk_rows) != expected_chunk_count
            or len(unit_rows) != expected_unit_count
        ):
            return False
        chunk_digest = _sha256_file(chunks_path)
        unit_digest = _sha256_file(units_path)
        if (
            metadata[_metadata_key("chunks", "sha256")] != chunk_digest
            or metadata[_metadata_key("units", "sha256")] != unit_digest
        ):
            return False
        for ordinal, (row, record) in enumerate(
            lookup._load_rows(chunks_path, chunk_rows)
        ):
            if (
                row["ordinal"] != ordinal
                or record.get("chunk_id") != row["chunk_id"]
                or record.get("document_id") != row["document_id"]
                or record.get("document_chunk_index") != row["document_chunk_index"]
                or hashlib.sha256(_chunk_text(record).encode("utf-8")).digest()
                != row["content_sha256"]
            ):
                return False
        for ordinal, (row, record) in enumerate(
            lookup._load_rows(units_path, unit_rows)
        ):
            if (
                row["ordinal"] != ordinal
                or record.get("id") != row["unit_id"]
                or record.get("document_id") != row["document_id"]
            ):
                return False
        return True
    except (
        ArtifactLookupError,
        KeyError,
        OSError,
        sqlite3.DatabaseError,
        TypeError,
        ValueError,
    ):
        return False
