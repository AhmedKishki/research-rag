"""Pinned upstream differential checks without runtime provisioning or downloads."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import json
import logging
import os
import pwd
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from research_rag.gateway.manifest import BASELINE_COMMIT
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


@pytest.fixture
def upstream(dependencies):
    """Execute original pinned method ASTs, not a rewritten reference algorithm.

    Strip only registration decorators and defer annotations. Heavy unrelated
    imports/server startup never execute. The BM25 init/build/search bodies and
    corpus chunk/save/load bodies run unchanged from the cached pinned checkout.
    """
    import numpy as np
    import orjson
    from tqdm import tqdm

    root = (
        Path(pwd.getpwuid(os.getuid()).pw_dir)
        / ".cache/vanilla-ultra-rag-mcp/runtime"
        / f"UltraRAG-{BASELINE_COMMIT}"
    )
    if not root.is_dir():
        pytest.skip("Pinned UltraRAG checkout is not cached; no download permitted")
    namespace = {
        "app": SimpleNamespace(logger=logging.getLogger("direct-compat-test")),
        "os": os,
        "Path": Path,
        "json": json,
        "orjson": orjson,
        "np": np,
        "tqdm": tqdm,
        "ToolError": ToolError,
    }
    tokenizer_path = root / "servers/retriever/src/bm25_tokenizer.py"
    exec(compile(tokenizer_path.read_text(), str(tokenizer_path), "exec"), namespace)  # noqa: S102 - cached pinned upstream fixture
    selected = []
    for component, names in (
        ("corpus", {"chunk_documents", "_save_jsonl", "_load_jsonl"}),
        ("retriever", {"retriever_init", "bm25_index", "bm25_search"}),
    ):
        path = root / f"servers/{component}/src/{component}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            ):
                node.decorator_list = []
                selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - execute unchanged pinned upstream method bodies
        compile(ast.fix_missing_locations(module), "pinned-upstream-methods", "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.anyio
@pytest.mark.parametrize("size,overlap", [(12, 3), (8, 8), (24, 0)])
async def test_chunk_matches_upstream(
    tmp_path, upstream, encoding, monkeypatch, size, overlap
):
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)
    source, expected, actual = (
        tmp_path / name for name in ("source.jsonl", "upstream.jsonl", "direct.jsonl")
    )
    write_rows(
        source,
        [
            {
                "id": "unit-1",
                "title": " Heading ",
                "contents": "  " + "Labor and capital — Arbeit, café, 中文.\n" * 6,
            },
            {"id": "unit-empty", "contents": "  "},
            {"id": "unit-2", "contents": "Second unit retains punctuation! " * 4},
            {"id": 0, "contents": "last"},
        ],
    )
    await upstream.chunk_documents(
        str(source),
        {"token": {"chunk_overlap": overlap}},
        "token",
        "gpt2",
        size,
        str(expected),
        False,
    )
    await direct.DirectRetrieval().chunk(
        source, actual, chunk_size=size, chunk_overlap=overlap
    )
    assert actual.read_bytes() == expected.read_bytes()


@pytest.mark.anyio
@pytest.mark.parametrize("language", ["en", "de", "fr"])
async def test_bm25_differential_and_cross_load(tmp_path, upstream, language):
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
    old, new = tmp_path / "upstream", tmp_path / "direct"
    reference = SimpleNamespace()

    async def initialize(index):
        await upstream.retriever_init(
            reference,
            "",
            {
                "bm25": {
                    "lang": language,
                    "tokenizer": "default",
                    "save_path": str(index),
                }
            },
            32,
            str(chunks),
            backend="bm25",
        )

    await initialize(old)
    await upstream.bm25_index(reference, overwrite=False)
    await initialize(old)
    backend = direct.DirectRetrieval()
    await backend.build_bm25(chunks, new, language=language)
    assert {p.name for p in old.iterdir()} == {p.name for p in new.iterdir()}
    for name in (p.name for p in old.iterdir()):
        assert (old / name).read_bytes() == (new / name).read_bytes()
    queries = ["capital", "Arbeit", "café", "absentword", "", "the and", "alpha-beta"]
    expected = {}
    for query in queries:
        expected[query] = (await upstream.bm25_search(reference, [query], top_k=3))[
            "ret_psg"
        ][0]
        assert await backend.search_bm25(query, 3) == expected[query]
    await initialize(new)  # upstream reads directly produced artifacts
    await backend.initialize_bm25(chunks, old, language=language)  # and vice versa
    for query in queries:
        assert (await upstream.bm25_search(reference, [query], top_k=3))["ret_psg"][
            0
        ] == expected[query]
        assert await backend.search_bm25(query, 3) == expected[query]
    before = {p.name: p.read_bytes() for p in new.iterdir()}
    await backend.build_bm25(chunks, new, language=language)
    assert before == {p.name: p.read_bytes() for p in new.iterdir()}


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
