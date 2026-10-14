"""The adapter contract between the workspace host and this app's service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A source file that an adapter has already authorized for browser access."""

    path: Path
    media_type: str
    filename: str
    content_disposition_type: str = "inline"


class UIRequestError(Exception):
    """A safe adapter error that may be returned to the local browser."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        if not 400 <= status_code <= 599:
            raise ValueError("status_code must be between 400 and 599")
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class UIAdapter(Protocol):
    """Normalize this app's service to the workspace operations."""

    async def health(self) -> Mapping[str, Any]: ...

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    async def source_file(self, source_path: str) -> SourceFile: ...

    async def list_clients(self) -> Sequence[Mapping[str, Any]]: ...

    async def disconnect_client(
        self, session_id: str, reason: str | None = None
    ) -> Mapping[str, Any]: ...
