"""PDF/EPUB source policy, discovery, hashing, and metadata validation."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ResearchConfig
from .settings import LANGUAGE_PATTERN

ALLOWED_SOURCE_EXTENSIONS = frozenset({".epub", ".pdf"})
METADATA_FIELDS = frozenset(
    {
        "title",
        "authors",
        "year",
        "doi",
        "language",
        "categories",
        "keywords",
        "project",
    }
)


class SourcePolicyError(ValueError):
    """Raised when a source violates the project research policy."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    source_id: str
    source_relative_path: str
    project_relative_path: str
    extension: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class SourceScan:
    selected: tuple[SourceFile, ...]
    ignored_extensions: dict[str, int]


def stable_source_id(project_id: str, source_relative_path: str) -> str:
    """Return a project-scoped identity that survives source-content changes."""

    relative = PurePosixPath(source_relative_path)
    if (
        not project_id
        or source_relative_path in {"", "."}
        or "\\" in source_relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != source_relative_path
    ):
        raise SourcePolicyError("Cannot create a source ID from an invalid identity")
    identity = f"{project_id}\0{source_relative_path}".encode()
    return f"src_{hashlib.sha256(identity).hexdigest()[:24]}"


def scan_sources(config: ResearchConfig) -> SourceScan:
    if not config.source_root.exists():
        return SourceScan(selected=(), ignored_extensions={})
    if not config.source_root.is_dir():
        raise SourcePolicyError(
            f"Configured sources path is not a directory: {config.source_root}"
        )

    selected: list[SourceFile] = []
    ignored: Counter[str] = Counter()
    for path in sorted(config.source_root.rglob("*")):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension not in ALLOWED_SOURCE_EXTENSIONS:
            ignored[extension or "<no extension>"] += 1
            continue
        if path.is_symlink():
            raise SourcePolicyError(f"Symbolic-link sources are not allowed: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(config.source_root)
        except ValueError as exc:
            raise SourcePolicyError(
                f"Source escapes the configured source directory: {path}"
            ) from exc
        stat = resolved.stat()
        source_relative_path = resolved.relative_to(config.source_root).as_posix()
        selected.append(
            SourceFile(
                path=resolved,
                source_id=stable_source_id(
                    config.project_id,
                    source_relative_path,
                ),
                source_relative_path=source_relative_path,
                project_relative_path=resolved.relative_to(
                    config.project_root
                ).as_posix(),
                extension=extension,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return SourceScan(
        selected=tuple(selected),
        ignored_extensions=dict(sorted(ignored.items())),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SourcePolicyError(f"Metadata field {field!r} must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = item.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


def _language_list(value: Any, field: str) -> list[str]:
    """Normalize a source's languages to lowercase ISO 639 codes.

    A source carries none, one, or several languages, because metadata records
    what the source is written in, which is not the same question as whether
    BM25 can tokenize it. A code BM25 has no stopword list for is accepted here
    and surfaced as a corpus-level warning, because refusing it would force the
    metadata to misdescribe a source that really is in that language.
    """

    languages: list[str] = []
    for item in _string_list(value, field):
        code = item.casefold()
        if not re.fullmatch(LANGUAGE_PATTERN, code):
            raise SourcePolicyError(
                f"Metadata field {field!r} must be ISO 639-1 codes: {item!r}"
            )
        if code not in languages:
            languages.append(code)
    return languages


def normalize_metadata(value: dict[str, Any]) -> dict[str, Any]:
    unknown = set(value) - METADATA_FIELDS
    if unknown:
        raise SourcePolicyError(
            f"Unsupported metadata fields: {', '.join(sorted(unknown))}"
        )

    normalized: dict[str, Any] = {}
    for field in ("title", "doi"):
        if field in value:
            item = value[field]
            if not isinstance(item, str):
                raise SourcePolicyError(f"Metadata field {field!r} must be a string")
            normalized[field] = item.strip()

    for field in ("authors", "categories", "keywords", "project"):
        if field in value:
            normalized[field] = _string_list(value[field], field)

    if "language" in value:
        normalized["language"] = _language_list(value["language"], "language")

    if "year" in value:
        year = value["year"]
        if year is not None and (
            isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999
        ):
            raise SourcePolicyError("Metadata field 'year' must be null or an integer")
        normalized["year"] = year
    return normalized
