"""The lexical backend contract the engine depends on.

The generation build and the search workflow reach exactly four operations:
token chunking, reading an existing BM25 index, building one, and searching it.
The protocol names them without naming a framework, so the engine depends on the
operations rather than on whichever implementation serves them.

Only the standard library is imported here. A caller that needs the protocol
does not install a retrieval or surface stack to get it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LexicalBackend(Protocol):
    """The chunking and BM25 operations a generation build and a search share."""

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None: ...

    async def initialize_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None: ...

    async def build_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None: ...

    async def search_bm25(self, query: str, top_k: int) -> list[str]: ...
