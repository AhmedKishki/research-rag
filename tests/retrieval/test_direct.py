"""Pinned compatibility checks without runtime provisioning or downloads.

The chunk window and the BM25 artifacts are locked here against their historical
contract directly: the token window `direct.py` forwards to the chunker, the
normalization of a chunk row, and the bytes of a saved index reloaded by a fresh
backend. Nothing here reads the retired gateway package or a cached upstream
checkout, so no model, download, or skip stands in for the golden coverage.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chonkie
import pytest

from research_rag.project.policy import ResearchError
from research_rag.retrieval import direct


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture(autouse=True)
def clear_tokenizer_load_cache():
    direct._cached_gpt2_tokenizer.cache_clear()
    yield
    direct._cached_gpt2_tokenizer.cache_clear()


@pytest.fixture
def tokenizer_assets(tmp_path, monkeypatch):
    """Small hash-verified assets exercise cache policy without model downloads."""
    assets = {"vocab.bpe": b"test merges", "encoder.json": b'{"test":0}'}
    monkeypatch.setattr(
        direct,
        "GPT2_CACHE_INPUTS",
        tuple(
            (name, hashlib.sha256(data).hexdigest()) for name, data in assets.items()
        ),
    )
    monkeypatch.setattr(
        direct,
        "_encoding",
        lambda encoder: SimpleNamespace(encode=lambda text, **kwargs: list(text)),
    )
    monkeypatch.setattr(
        direct,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("tokenizer test attempted a download"),
    )
    root = tmp_path / "tokenizer-cache"
    root.mkdir()
    for name, data in assets.items():
        filename = hashlib.sha1((direct.GPT2_CACHE_URL + name).encode()).hexdigest()
        (root / filename).write_bytes(data)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(root))
    return root


def test_repeated_token_counts_reuse_verified_assets(tokenizer_assets, monkeypatch):
    reads = []
    read_bytes = Path.read_bytes

    def observe(path):
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observe)
    for _ in range(3):
        assert direct.count_gpt2_tokens("example", offline=True) == 7
    assert len(reads) == len(direct.GPT2_CACHE_INPUTS)
    # Health verifies the current bytes independently of a resident encoding.
    assert direct.tokenizer_cache_status()
    assert len(reads) == 2 * len(direct.GPT2_CACHE_INPUTS)


@pytest.mark.parametrize("change", ["remove", "replace", "edit"])
def test_tokenizer_cache_rechecks_changed_assets(tokenizer_assets, change):
    assert direct.count_gpt2_tokens("example", offline=True) == 7
    name, _digest = direct.GPT2_CACHE_INPUTS[-1]
    path = (
        tokenizer_assets
        / hashlib.sha1((direct.GPT2_CACHE_URL + name).encode()).hexdigest()
    )
    observed = path.stat()
    corrupt = b"x" * observed.st_size
    if change == "remove":
        path.unlink()
    elif change == "replace":
        replacement = tokenizer_assets / "replacement"
        replacement.write_bytes(corrupt)
        os.utime(replacement, ns=(observed.st_atime_ns, observed.st_mtime_ns))
        os.replace(replacement, path)
    else:
        path.write_bytes(corrupt)
        os.utime(path, ns=(observed.st_atime_ns, observed.st_mtime_ns))
    assert not direct.tokenizer_cache_status()
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens("example", offline=True)


def test_tokenizer_cache_follows_environment_selection(
    tokenizer_assets, tmp_path, monkeypatch
):
    assert direct.count_gpt2_tokens("example", offline=True) == 7
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "missing"))
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens("example", offline=True)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tokenizer_assets))
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens("example", offline=True)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tokenizer_assets))
    assert direct.count_gpt2_tokens("example", offline=True) == 7


def test_tokenizer_cache_follows_model_cache_root(
    tokenizer_assets, tmp_path, monkeypatch
):
    monkeypatch.setattr(direct, "_cache_directories", lambda root: (root, [root]))
    assert (
        direct.count_gpt2_tokens("example", offline=True, cache_root=tokenizer_assets)
        == 7
    )
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens(
            "example", offline=True, cache_root=tmp_path / "missing"
        )


def test_tokenizer_cache_follows_relative_override_directory(
    tokenizer_assets, tmp_path, monkeypatch
):
    monkeypatch.chdir(tokenizer_assets.parent)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", tokenizer_assets.name)
    assert direct.count_gpt2_tokens("example", offline=True) == 7
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens("example", offline=True)


def test_tokenizer_health_and_loading_keep_literal_tilde_overrides(
    tokenizer_assets, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home_cache = home / "cache"
    home_cache.mkdir(parents=True)
    for source in tokenizer_assets.iterdir():
        (home_cache / source.name).write_bytes(source.read_bytes())
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "~/cache")
    assert not direct.tokenizer_cache_status()
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens("example", offline=True)
    literal_cache = tmp_path / "~" / "cache"
    literal_cache.mkdir(parents=True)
    for source in tokenizer_assets.iterdir():
        (literal_cache / source.name).write_bytes(source.read_bytes())
    assert direct.tokenizer_cache_status()
    assert direct.count_gpt2_tokens("example", offline=True) == 7


def test_online_load_without_persistent_cache_cannot_satisfy_offline(
    tokenizer_assets, monkeypatch
):
    assets = {
        name: (
            tokenizer_assets
            / hashlib.sha1((direct.GPT2_CACHE_URL + name).encode()).hexdigest()
        ).read_bytes()
        for name, _digest in direct.GPT2_CACHE_INPUTS
    }
    downloads = []

    def fetch(url, **kwargs):
        downloads.append(url)
        return io.BytesIO(assets[url.rsplit("/", 1)[-1]])

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    monkeypatch.setattr(direct, "urlopen", fetch)
    assert direct.count_gpt2_tokens("example", offline=False) == 7
    assert len(downloads) == len(direct.GPT2_CACHE_INPUTS)
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.count_gpt2_tokens("example", offline=True)
    assert len(downloads) == len(direct.GPT2_CACHE_INPUTS)


@pytest.mark.anyio
async def test_token_floor_counting_runs_off_event_loop(tmp_path, monkeypatch):
    from research_rag.retrieval import search as search_module

    workflow = search_module.SearchWorkflow()
    workflow.config = SimpleNamespace(
        settings=SimpleNamespace(minimum_passage_words=0),
        offline=True,
        model_cache_root=tmp_path,
    )
    event_loop_thread = threading.get_ident()
    calls = []

    def count(text, tokenizer, **kwargs):
        calls.append((threading.get_ident(), text, tokenizer, kwargs))
        return 7

    monkeypatch.setattr(search_module, "passage_token_count", count)
    chunk = {"contents": "example"}
    assert not await workflow._passage_too_short_async(chunk)
    assert calls == []
    assert await workflow._passage_too_short_async(
        chunk, token_policy={"minimum_tokens": 10, "tokenizer": "gpt2"}
    )
    assert len(calls) == 1
    thread, text, tokenizer, options = calls[0]
    assert thread != event_loop_thread
    assert (text, tokenizer) == ("example", "gpt2")
    assert options == {"offline": True, "cache_root": tmp_path}


def test_synchronous_length_check_remains_available_to_audit_callers(
    tmp_path, monkeypatch
):
    from research_rag.retrieval import search as search_module

    workflow = search_module.SearchWorkflow()
    workflow.config = SimpleNamespace(
        settings=SimpleNamespace(minimum_passage_words=0),
        offline=True,
        model_cache_root=tmp_path,
    )
    monkeypatch.setattr(search_module, "passage_token_count", lambda *args, **kwargs: 7)
    assert (
        workflow._passage_too_short(
            {"contents": "example"},
            token_policy={"minimum_tokens": 10, "tokenizer": "gpt2"},
        )
        is True
    )


@pytest.fixture
def dependencies():
    for name in ("bm25s", "numba", "chonkie", "tiktoken", "orjson"):
        pytest.importorskip(name)


@pytest.fixture
def encoding(dependencies):
    try:
        return direct.load_gpt2_tokenizer(offline=True)
    except ResearchError as exc:
        pytest.skip(str(exc))


def _recording_token_chunker(windows: list[tuple[int, int]]) -> type:
    """A stand-in for Chonkie's chunker that records the token window it is asked for.

    The app-owned chunk contract is the window it forwards, not Chonkie's own
    windowing, so this keeps the assertion on configuration and row normalization
    and needs neither a tokenizer model nor a network fetch.
    """

    class RecordingTokenChunker:
        def __init__(
            self,
            *,
            tokenizer: Any,
            chunk_size: int,
            chunk_overlap: int,
        ) -> None:
            windows.append((chunk_size, chunk_overlap))

        def chunk(self, text: str) -> list[Any]:
            return [SimpleNamespace(text=text)]

    return RecordingTokenChunker


@pytest.mark.anyio
@pytest.mark.parametrize(
    "size,overlap,forwarded_overlap",
    [
        (12, 3, 3),  # an explicit overlap below the size is used unchanged
        (8, 8, 2),  # an overlap at or above the size becomes a quarter of it
        (24, 0, 0),
    ],
)
async def test_chunk_forwards_the_historical_token_window_and_normalizes_rows(
    tmp_path, monkeypatch, size, overlap, forwarded_overlap
):
    windows: list[tuple[int, int]] = []
    monkeypatch.setattr(chonkie, "TokenChunker", _recording_token_chunker(windows))
    monkeypatch.setattr(
        direct, "load_gpt2_tokenizer", lambda **kwargs: SimpleNamespace()
    )
    source, output = tmp_path / "source.jsonl", tmp_path / "chunks.jsonl"
    write_rows(
        source,
        [
            {
                "id": "unit-1",
                "title": " Heading ",
                "contents": "  The cobalt heron — Arbeit, café.  ",
            },
            {"id": "unit-empty", "contents": "  "},
            {"id": "unit-2", "contents": "  Second unit keeps punctuation!  "},
        ],
    )
    await direct.DirectRetrieval().chunk(
        source, output, chunk_size=size, chunk_overlap=overlap
    )
    assert windows == [(size, forwarded_overlap)]
    # Unit order is preserved, empty units are dropped, titles and contents are
    # stripped, and identifiers are reassigned in order.
    assert [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ] == [
        {
            "id": 0,
            "doc_id": "unit-1",
            "title": "Heading",
            "contents": "The cobalt heron — Arbeit, café.",
        },
        {
            "id": 1,
            "doc_id": "unit-2",
            "title": "",
            "contents": "Second unit keeps punctuation!",
        },
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("overlap", [1, 4])
async def test_real_chunker_preserves_the_historical_windows_without_model_assets(
    tmp_path, monkeypatch, overlap
):
    import tiktoken

    tokenizer = tiktoken.Encoding(
        name="historical-window-fixture",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
    monkeypatch.setattr(direct, "load_gpt2_tokenizer", lambda **kwargs: tokenizer)
    source, output = tmp_path / "source.jsonl", tmp_path / "chunks.jsonl"
    write_rows(
        source, [{"id": "unit-1", "title": " Heading ", "contents": " AaBbCcDd "}]
    )

    await direct.DirectRetrieval().chunk(
        source, output, chunk_size=4, chunk_overlap=overlap
    )

    assert [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ] == [
        {"id": index, "doc_id": "unit-1", "title": "Heading", "contents": text}
        for index, text in enumerate(["AaBb", "bCcD", "Dd"])
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("language", ["en", "de", "fr"])
async def test_bm25_build_is_deterministic_and_cross_loads(
    tmp_path, dependencies, language
):
    """Two builds of one corpus are byte-identical and a fresh backend reloads them.

    The correlation between a build and a load is the contract a search relies on,
    and the saved artifacts are compared byte for byte so a silent rewrite fails.
    """

    chunks = tmp_path / "chunks.jsonl"
    write_rows(
        chunks,
        [
            {"contents": text}
            for text in [
                "Labor capital and commodity exchange",
                "Arbeit Kapital und Warenproduktion",
                "le travail et la marchandise café",
                "Labor capital and commodity exchange",
                "Tokens punctuation: alpha-beta; 12345.",
            ]
        ],
    )
    first, second = tmp_path / "first", tmp_path / "second"
    builder = direct.DirectRetrieval()
    await builder.build_bm25(chunks, first, language=language)
    twin = direct.DirectRetrieval()
    await twin.build_bm25(chunks, second, language=language)
    assert {p.name for p in first.iterdir()} == {p.name for p in second.iterdir()}
    for name in (p.name for p in first.iterdir()):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    queries = ["capital", "Arbeit", "café", "absentword", "", "the and", "alpha-beta"]
    baseline = {query: await builder.search_bm25(query, 3) for query in queries}
    for query in queries:
        assert await twin.search_bm25(query, 3) == baseline[query]

    loader = direct.DirectRetrieval()
    await loader.initialize_bm25(chunks, first, language=language)
    for query in queries:
        assert await loader.search_bm25(query, 3) == baseline[query]

    # Rebuilding over an existing index rewrites nothing.
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    await builder.build_bm25(chunks, first, language=language)
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}
    await builder.close()
    await twin.close()
    await loader.close()


@pytest.mark.anyio
async def test_bm25_cross_load_round_trips_non_ascii_stopwords(tmp_path, dependencies):
    """A saved non-ASCII stopword list survives the trip to a fresh backend."""

    chunks = tmp_path / "chunks.jsonl"
    write_rows(
        chunks,
        [
            {
                "contents": (
                    "Arbeit und Übergang für größere Wähler während der Übung, "
                    "über die Warenproduktion"
                )
            }
        ],
    )
    index = tmp_path / "index"
    builder = direct.DirectRetrieval()
    await builder.build_bm25(chunks, index, language="de")
    saved = json.loads((index / "stopwords.tokenizer.json").read_text(encoding="utf-8"))
    non_ascii = [word for word in saved if any(ord(char) > 127 for char in word)]
    assert {"für", "über", "während", "können"} <= set(non_ascii)

    loader = direct.DirectRetrieval()
    await loader.initialize_bm25(chunks, index, language="de")
    assert list(loader._tokenizer.stopwords) == saved
    assert await loader.search_bm25("Übergang", 1) == await builder.search_bm25(
        "Übergang", 1
    )
    await builder.close()
    await loader.close()


@pytest.mark.anyio
async def test_cancellation_drains_worker_and_serializes_repeated_cancel():
    backend = direct.DirectRetrieval()
    entered, release = threading.Event(), threading.Event()
    order = []

    def first():
        entered.set()
        assert release.wait(5)
        order.append("first")

    task = asyncio.create_task(backend._run(first))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    second = asyncio.create_task(backend._run(lambda: order.append("second")))
    await asyncio.sleep(0)
    assert not task.done() and not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await second
    assert order == ["first", "second"]


def test_offline_cache_missing_never_fetches(tmp_path, monkeypatch, dependencies):
    import tiktoken
    from tiktoken.registry import ENCODINGS

    monkeypatch.delitem(ENCODINGS, "gpt2", raising=False)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        tiktoken,
        "get_encoding",
        lambda name: pytest.fail("network-capable loader reached"),
    )
    with pytest.raises(ResearchError, match="Offline GPT2"):
        direct.load_gpt2_tokenizer(offline=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_invalid_corpus_and_failed_init_preserve_state(tmp_path, dependencies):
    backend = direct.DirectRetrieval()
    source, index = tmp_path / "source.jsonl", tmp_path / "index"
    write_rows(
        source, [{"contents": "original passage"}, {"contents": "different passage"}]
    )
    await backend.build_bm25(source, index)
    model = backend._model
    source.write_text("{bad json}\n")
    with pytest.raises(ResearchError, match="Invalid JSON on line 0"):
        await backend.initialize_bm25(source, index)
    assert backend._model is model
    assert await backend.search_bm25("original", 1) == ["original passage"]
    write_rows(source, [{"id": "no contents"}])
    with pytest.raises(ValueError, match="missing key 'contents'"):
        await backend.initialize_bm25(source, index)
    await backend.close()
    with pytest.raises(RuntimeError, match="not been initialized"):
        await backend.search_bm25("original", 1)


@pytest.mark.anyio
async def test_chunk_deterministic_unit_order_and_empty_units(
    tmp_path, encoding, monkeypatch
):
    monkeypatch.setattr(direct, "load_gpt2_tokenizer", lambda **kwargs: encoding)
    source, output = tmp_path / "source.jsonl", tmp_path / "chunks.jsonl"
    write_rows(
        source,
        [
            {"id": "page-2", "title": " title ", "contents": " One short unit. "},
            {"id": "page-3", "contents": ""},
            {"id": "page-1", "contents": "Another unit."},
        ],
    )
    await direct.DirectRetrieval().chunk(
        source, output, chunk_size=256, chunk_overlap=64
    )
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": 0, "doc_id": "page-2", "title": "title", "contents": "One short unit."},
        {"id": 1, "doc_id": "page-1", "title": "", "contents": "Another unit."},
    ]


@pytest.mark.anyio
async def test_corrupt_index_preserves_previous_model_and_files(tmp_path, dependencies):
    backend = direct.DirectRetrieval()
    source, good, bad = (tmp_path / name for name in ("chunks.jsonl", "good", "bad"))
    write_rows(
        source, [{"contents": "one unique passage"}, {"contents": "two other passages"}]
    )
    await backend.build_bm25(source, good)
    model = backend._model
    bad.mkdir()
    (bad / "params.index.json").write_text("not JSON")
    with pytest.raises((ValueError, FileNotFoundError)):
        await backend.initialize_bm25(source, bad)
    assert backend._model is model
    assert (bad / "params.index.json").read_text() == "not JSON"
    assert await backend.search_bm25("unique", 1) == ["one unique passage"]
    with pytest.raises(ValueError):
        await backend.search_bm25("unique", 3)


def test_offline_verified_encoding_matches_resident(encoding, monkeypatch):
    import tiktoken
    from tiktoken.registry import ENCODINGS

    # This fixture already loaded a local cache; force the offline construction
    # branch and make all network-capable registry loads fatal.
    monkeypatch.delitem(ENCODINGS, "gpt2", raising=False)
    monkeypatch.setattr(
        tiktoken, "get_encoding", lambda name: pytest.fail("registry load")
    )
    local = direct.load_gpt2_tokenizer(offline=True)
    for text in (
        "English punctuation!",
        "Arbeit café 中文",
        "line\nnext",
        "<|endoftext|>",
    ):
        assert local.encode(text, disallowed_special=()) == encoding.encode(
            text, disallowed_special=()
        )


def test_gpt2_counts_match_cached_tokie_without_pretrained_fetch(encoding, tmp_path):
    tokie = pytest.importorskip("tokie")
    _, roots = direct._cache_directories(None)
    assets = {
        name: direct._cached_input(name, digest, roots)
        for name, digest in direct.GPT2_CACHE_INPUTS
    }
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [
                    {
                        "id": 50256,
                        "content": "<|endoftext|>",
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": True,
                        "special": True,
                    }
                ],
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True,
                },
                "post_processor": {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": False,
                    "use_regex": True,
                },
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": True,
                    "use_regex": True,
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": None,
                    "continuing_subword_prefix": "",
                    "end_of_word_suffix": "",
                    "fuse_unk": False,
                    "vocab": json.loads(assets["encoder.json"]),
                    "merges": [
                        line.split()
                        for line in assets["vocab.bpe"].decode().splitlines()[1:]
                        if line
                    ],
                },
            }
        )
    )
    tokenizer = tokie.Tokenizer.from_json(str(tokenizer_json))
    for text in (
        "",
        " ",
        "\n\t",
        " hello  world ",
        "Arbeit café 中文",
        "👋🏽 test",
        "a\r\nb\u00a0c",
        "naïve e\u0301",
        "don't 12345",
        "<|endoftext|>",
    ):
        assert direct.count_gpt2_tokens(text, offline=True) == tokenizer.count_tokens(
            text
        )


@pytest.mark.anyio
async def test_worker_error_during_cancellation_is_observed():
    backend = direct.DirectRetrieval()
    entered, release = threading.Event(), threading.Event()

    def fail():
        entered.set()
        assert release.wait(5)
        raise ValueError("worker failure")

    task = asyncio.create_task(backend._run(fail))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await backend._run(lambda: "usable") == "usable"


def test_prefetch_writes_only_verified_inputs_to_selected_cache(
    tmp_path, encoding, monkeypatch
):
    _, roots = direct._cache_directories(None)
    assets = {
        name: direct._cached_input(name, digest, roots)
        for name, digest in direct.GPT2_CACHE_INPUTS
    }
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "cache"))
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(assets[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(direct, "urlopen", fetch)
    assert not direct.tokenizer_cache_status(tmp_path)
    direct.prefetch_gpt2_tokenizer(cache_root=tmp_path)
    assert direct.tokenizer_cache_status(tmp_path)
    assert len(calls) == 2
    assert len(list((tmp_path / "cache").iterdir())) == 2
    direct.load_gpt2_tokenizer(offline=True, cache_root=tmp_path)
    assert len(calls) == 2


@pytest.mark.anyio
async def test_build_failure_restores_previous_retriever(
    tmp_path, dependencies, monkeypatch
):
    import bm25s

    backend = direct.DirectRetrieval()
    chunks, good, failed = (
        tmp_path / name for name in ("chunks.jsonl", "good", "failed")
    )
    write_rows(
        chunks, [{"contents": "unique first passage"}, {"contents": "second passage"}]
    )
    await backend.build_bm25(chunks, good)
    previous = backend._model

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bm25s.BM25, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        await backend.build_bm25(chunks, failed)
    assert backend._model is previous
    assert await backend.search_bm25("unique", 1) == ["unique first passage"]
