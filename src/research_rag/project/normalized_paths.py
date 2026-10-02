"""One rule for a path a stored file is allowed to name.

Five readers used to carry their own copy of this check — the metadata, catalog,
and exclusion readers in `storage`, the artifact-path resolver in `generation`,
and the source-identity builder in `sources` — and the copies had already drifted
apart: the exclusion readers accepted the empty string that the metadata reader
refused, because the one line above them rejected it for a different reason.

A path is acceptable when it is relative, free of `..`, free of a backslash, and
written exactly as `PurePosixPath` would render it. Everything else is a hand-edited
file that has been edited wrongly, and each caller says so in its own words.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def normalized_relative_path(value: object) -> PurePosixPath | None:
    """Return the path as parts, or `None` when the value is not acceptable.

    The empty string and `.` are refused: a stored path that names nothing is a
    hand-edit mistake rather than a path. Callers that must tolerate one of them
    say so before calling.
    """

    if not isinstance(value, str) or value in {"", "."}:
        return None
    if "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    if relative.as_posix() != value:
        return None
    return relative
