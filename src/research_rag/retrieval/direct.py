"""Direct execution of the pinned UltraRAG token/BM25 compatibility contract.

Dependencies stay lazy. A single serialized worker owns the mutable BM25 model;
cancellation drains that worker before another operation can touch its state.
No source documents or generation selection records are changed here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Self
from urllib.request import urlopen

from ..project.config import ResearchConfig
from ..project.policy import ResearchError

GPT2_CACHE_INPUTS = (
    ("vocab.bpe", "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    (
        "encoder.json",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
)
GPT2_CACHE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cache_directories(cache_root: Path | None) -> tuple[Path | None, list[Path]]:
    override = os.environ.get(
        "TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR")
    )
    if override is not None:
        return (Path(override), [Path(override)]) if override else (None, [])
    if cache_root is None:
        from platformdirs import user_cache_path

        cache_root = user_cache_path("research-rag", appauthor=False) / "models"
    primary = cache_root / "tiktoken"
    return primary, [primary, Path(tempfile.gettempdir()) / "data-gym-cache"]


def _cached_input(name: str, digest: str, roots: list[Path]) -> bytes | None:
    for root in roots:
        path = root / hashlib.sha1((GPT2_CACHE_URL + name).encode()).hexdigest()
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() == digest:
            return data
    return None


def tokenizer_cache_status(cache_root: Path | None = None) -> bool:
    """Read-only readiness check, without importing tokenizers or fetching."""
    _, roots = _cache_directories(cache_root)
    return all(
        _cached_input(name, digest, roots) is not None
        for name, digest in GPT2_CACHE_INPUTS
    )


@lru_cache(maxsize=1)
def _encoding(encoder: bytes) -> Any:
    import tiktoken
    from tiktoken_ext.openai_public import r50k_pat_str

    byte_order = [b for b in range(256) if chr(b).isprintable() and chr(b) != " "]
    mapping = {chr(b): b for b in byte_order}
    missing = [b for b in range(256) if b not in byte_order]
    mapping.update({chr(256 + i): b for i, b in enumerate(missing)})
    ranks = {
        bytes(mapping[c] for c in token): rank
        for token, rank in json.loads(encoder).items()
        if token != "<|endoftext|>"
    }
    return tiktoken.Encoding(
        name="gpt2",
        explicit_n_vocab=50257,
        pat_str=r50k_pat_str,
        mergeable_ranks=ranks,
        special_tokens={"<|endoftext|>": 50256},
    )


def load_gpt2_tokenizer(
    *, offline: bool = False, cache_root: Path | None = None
) -> Any:
    """Load verified GPT2 inputs; offline mode never downloads or writes.

    New assets enter the research-rag model cache. The historical temporary
    tiktoken cache is read-only fallback. Explicit environment overrides win.
    """
    primary, roots = _cache_directories(cache_root)
    inputs = {}
    for name, digest in GPT2_CACHE_INPUTS:
        data = _cached_input(name, digest, roots)
        if data is None:
            if offline:
                raise ResearchError(
                    f"Offline GPT2 tokenizer cache is missing or invalid: {primary}. "
                    "Prefetch the tokenizer before running offline."
                )
            with urlopen(GPT2_CACHE_URL + name, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"GPT2 tokenizer hash mismatch: {name}")
            if primary is not None:
                primary.mkdir(parents=True, exist_ok=True)
                path = (
                    primary / hashlib.sha1((GPT2_CACHE_URL + name).encode()).hexdigest()
                )
                fd, temporary = tempfile.mkstemp(dir=primary, prefix=".gpt2-")
                try:
                    with os.fdopen(fd, "wb") as target:
                        target.write(data)
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, path)
                    _sync_directory(primary)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        inputs[name] = data
    return _encoding(inputs["encoder.json"])


def count_gpt2_tokens(
    text: str, *, offline: bool = False, cache_root: Path | None = None
) -> int:
    """Count GPT2 tokens, retaining tokie's recognition of the end-of-text token."""
    return len(
        load_gpt2_tokenizer(offline=offline, cache_root=cache_root).encode(
            text, allowed_special="all"
        )
    )


def prefetch_gpt2_tokenizer(*, cache_root: Path | None = None) -> None:
    """Explicit online provisioning hook for doctor, never startup/status."""
    load_gpt2_tokenizer(cache_root=cache_root)


class DirectRetrieval:
    """Wrapper-compatible async chunking and one mutable BM25 retriever."""

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._contents: list[str] = []

    async def _run(self, operation: Callable[[], Any]) -> Any:
        async with self._lock:
            worker = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Repeated cancellation must not release the model while a thread
                # still mutates it. Observe worker errors, but retain cancellation.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 - cancellation retains precedence
                        break
                if not worker.cancelled():
                    worker.exception()
                raise

    async def start(self) -> None:
        """No process or dependency runtime is needed at startup."""

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def close(self) -> None:
        def release() -> None:
            self._model = self._tokenizer = None
            self._contents = []

        await self._run(release)

    async def aclose(self) -> None:
        await self.close()

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        def execute() -> None:
            try:
                from chonkie import TokenChunker
            except ImportError as exc:
                raise ResearchError(
                    "chonkie or tiktoken not installed. Please `pip install chonkie tiktoken`."
                ) from exc
            tokenizer = load_gpt2_tokenizer(
                offline=bool(self.config and self.config.offline),
                cache_root=self.config.model_cache_root if self.config else None,
            )
            overlap = (
                int(chunk_size / 4) if chunk_overlap >= chunk_size else chunk_overlap
            )
            chunker = TokenChunker(
                tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=overlap
            )
            rows = []
            with input_path.open(encoding="utf-8") as source:
                documents = [json.loads(line) for line in source if line.strip()]
            for doc in documents:
                doc_id = doc.get("id") or ""
                title = (doc.get("title") or "").strip()
                text = (doc.get("contents") or "").strip()
                if not text:
                    continue
                try:
                    chunks = chunker.chunk(text)
                except Exception as exc:
                    raise ResearchError(
                        f"fail chunked(doc_id={doc_id}): {exc}"
                    ) from exc
                for chunk in chunks:
                    rows.append(
                        {
                            "id": len(rows),
                            "doc_id": doc_id,
                            "title": title,
                            "contents": chunk.text.strip(),
                        }
                    )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="\n") as target:
                for row in rows:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                target.flush()
                os.fsync(target.fileno())
            _sync_directory(output_path.parent)

        await self._run(execute)

    def _initialize(self, chunks_path: Path, index_path: Path, language: str) -> None:
        import bm25s
        import orjson

        try:
            model = bm25s.BM25(backend="numba")
        except Exception:  # noqa: BLE001 - pinned upstream backend fallback
            model = bm25s.BM25(backend="numpy")
        try:
            tokenizer = bm25s.tokenization.Tokenizer(stopwords=language)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize BM25 tokenizer for language '{language}' "
                f"with tokenizer mode 'default': {exc}"
            ) from exc
        contents = []
        if chunks_path.exists():
            with chunks_path.open("rb") as source:
                for i, line in enumerate(source):
                    try:
                        item = orjson.loads(line)
                    except orjson.JSONDecodeError as exc:
                        raise ResearchError(f"Invalid JSON on line {i}: {exc}") from exc
                    if "contents" not in item:
                        raise ValueError(
                            f"Line {i}: missing key 'contents'. full item={item}"
                        )
                    contents.append(item["contents"])
        if index_path.exists():
            model = model.load(str(index_path), mmap=True, load_corpus=False)
            tokenizer.load_stopwords(str(index_path))
            tokenizer.load_vocab(str(index_path))
            model.corpus = contents
            model.backend = "numba"
        self._model, self._tokenizer = model, tokenizer
        self._contents = contents

    async def initialize_bm25(
        self, chunks_path: Path, index_path: Path, *, language: str = "en"
    ) -> None:
        await self._run(lambda: self._initialize(chunks_path, index_path, language))

    async def build_bm25(
        self, chunks_path: Path, index_path: Path, *, language: str = "en"
    ) -> None:
        def execute() -> None:
            previous = self._model, self._tokenizer, self._contents
            try:
                self._initialize(chunks_path, index_path, language)
                if not index_path.exists():
                    tokens = self._tokenizer.tokenize(self._contents, return_as="tuple")
                    self._model.index(tokens)
                    self._model.save(str(index_path), corpus=None)
                    self._tokenizer.save_stopwords(str(index_path))
                    self._tokenizer.save_vocab(str(index_path))
                    for path in index_path.iterdir():
                        if path.is_file():
                            with path.open("rb") as artifact:
                                os.fsync(artifact.fileno())
                    _sync_directory(index_path)
                    _sync_directory(index_path.parent)
                self._initialize(chunks_path, index_path, language)
            except BaseException:
                self._model, self._tokenizer, self._contents = previous
                raise

        await self._run(execute)

    async def search_bm25(self, query: str, top_k: int) -> list[str]:
        def execute() -> list[str]:
            if self._model is None:
                raise RuntimeError("BM25 retriever has not been initialized")
            tokens = self._tokenizer.tokenize(
                [query], return_as="tuple", update_vocab=False
            )
            results, _ = self._model.retrieve(tokens, k=top_k)
            batches = results.tolist() if hasattr(results, "tolist") else results
            if (
                not isinstance(batches, list)
                or not batches
                or not isinstance(batches[0], list)
            ):
                raise RuntimeError("UltraRAG BM25 returned an invalid passage result")
            return [str(item) for item in batches[0]]

        return await self._run(execute)
