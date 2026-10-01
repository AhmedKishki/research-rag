"""Measure the per-unit write pattern on a real project build.

Harness behind the write-grouping decision in ``MEASUREMENTS.md``. One corpus is
built twice on the device ``--root`` names, through the real gateway, tokenizer
chunking, and embeddings, changing nothing but the write pattern:

* ``paired`` (before) fsyncs every atomic write's directory and the handoff file.
* ``grouped`` (after) is the server: a unit's artifacts defer that fsync, and the
  group commits once, before the checkpoint claiming the unit complete.

A pair's second run reads a warm page cache, so the default ``--order both``
prints both orders. Read the difference, not the absolute numbers.

    uv run python scripts/benchmark_write_pattern.py --root /mnt/data/rr-write-bench

The first run downloads the models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from fastmcp import Client

import research_rag.ingestion as ingestion_module
from research_rag.config import resolve_config
from research_rag.service import ResearchService
from research_rag.ultrarag import VanillaUltraRAG, create_vanilla_transport

WORDS = (
    "cobalt",
    "amber",
    "copper",
    "lithium",
    "silicon",
    "nickel",
    "graphite",
    "quartz",
    "manganese",
    "tungsten",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    wall_seconds: float
    chunk_count: int
    unit_count: int
    phase_seconds: dict[str, float]


def write_pdf(path: Path, pages: list[str], *, title: str) -> None:
    document = pymupdf.open()
    document.set_metadata({"title": title, "author": "Benchmark"})
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 760), text, fontsize=11)
    document.save(path)
    document.close()


def build_corpus(project: Path, *, sources: int, pages: int) -> None:
    """Write a corpus whose page lengths vary: uniform short pages hide the
    embedding padding cost, and pages that grow within a source punish large
    inference batches.
    """
    root = project / "sources"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(sources):
        word = WORDS[index % len(WORDS)]
        page_text = []
        for page in range(pages):
            sentence = (
                f"The {word} record {page} discusses labour, ecology, and value "
                f"across supply chains and the amber marsh, where the price of a "
                f"commodity hides the work that produced it. "
            )
            page_text.append(sentence * (4 + page * 5))
        write_pdf(
            root / f"record-{index:02d}-{word}.pdf",
            page_text,
            title=f"Record {index}",
        )


class durable_writes:
    """Force the per-file durable write pattern for a run's duration."""

    def __enter__(self) -> None:
        names = (
            "atomic_write_json",
            "atomic_write_jsonl",
            "write_handoff_jsonl",
            "fsync_directories",
        )
        self._originals: dict[str, Any] = {
            name: getattr(ingestion_module, name) for name in names
        }
        real_json = self._originals["atomic_write_json"]
        real_jsonl = self._originals["atomic_write_jsonl"]

        def durable_json(path: Path, value: Any, **kwargs: Any) -> None:
            real_json(path, value, fsync_parent=True)

        def durable_jsonl(path: Path, records: Any, **kwargs: Any) -> None:
            real_jsonl(path, records, fsync_parent=True)

        ingestion_module.atomic_write_json = durable_json
        ingestion_module.atomic_write_jsonl = durable_jsonl
        ingestion_module.write_handoff_jsonl = durable_jsonl
        ingestion_module.fsync_directories = lambda paths: None

    def __exit__(self, *exc_info: object) -> None:
        for name, original in self._originals.items():
            setattr(ingestion_module, name, original)


class chunk_batch:
    """Force a chunker batch size for the duration of a run.

    The size is a setting, so the environment variable the settings layer reads
    is what gets set, and a run resolving its configuration inside this context
    sees it.
    """

    VARIABLE = "RESEARCH_ULTRARAG_CHUNKING_BATCH_UNITS"

    def __init__(self, units: int) -> None:
        self.units = units

    def __enter__(self) -> None:
        self._original = os.environ.get(self.VARIABLE)
        os.environ[self.VARIABLE] = str(self.units)

    def __exit__(self, *exc_info: object) -> None:
        if self._original is None:
            os.environ.pop(self.VARIABLE, None)
        else:
            os.environ[self.VARIABLE] = self._original


class embedding_batch:
    """Force the ONNX inference batch size for a run's duration."""

    VARIABLE = "RESEARCH_ULTRARAG_EMBEDDING_INFERENCE_BATCH_SIZE"

    def __init__(self, size: int) -> None:
        self.size = size

    def __enter__(self) -> None:
        self._original = os.environ.get(self.VARIABLE)
        os.environ[self.VARIABLE] = str(self.size)

    def __exit__(self, *exc_info: object) -> None:
        if self._original is None:
            os.environ.pop(self.VARIABLE, None)
        else:
            os.environ[self.VARIABLE] = self._original


def patch_for(variant: str, side: str) -> AbstractContextManager[None]:
    if variant == "write-pattern":
        return durable_writes() if side == "before" else nullcontext()
    if variant == "chunk-batch":
        return chunk_batch(1 if side == "before" else 16)
    if variant == "embedding-batch":
        # 1 is current because FastEmbed pads a batch to its longest sequence.
        return embedding_batch(64 if side == "before" else 1)
    raise SystemExit(f"unsupported variant: {variant}")


async def run_once(
    project: Path,
    *,
    variant: str,
    side: str,
    sources: int,
    pages: int,
    chunk_size: int,
    chunk_overlap: int,
) -> RunResult:
    shutil.rmtree(project, ignore_errors=True)
    project.mkdir(parents=True)
    build_corpus(project, sources=sources, pages=pages)
    config = resolve_config(project)

    started = time.perf_counter()
    with patch_for(variant, side):
        async with Client(
            create_vanilla_transport(config),
            name="write-pattern-benchmark",
            timeout=1800,
            init_timeout=1800,
        ) as client:
            service = ResearchService(config, VanillaUltraRAG(client, config))
            result = await service.ingest(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            while result.get("status") == "in_progress":
                result = await service.ingest(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
    wall = time.perf_counter() - started

    manifest = json.loads(
        (Path(result["generation_root"]) / "manifest.json").read_text(encoding="utf-8")
    )
    return RunResult(
        wall_seconds=wall,
        chunk_count=int(manifest["chunk_count"]),
        unit_count=int(manifest["extraction_unit_count"]),
        phase_seconds={
            str(key): float(value)
            for key, value in manifest["build_metrics"]["phase_timings_seconds"].items()
        },
    )


def report(before: RunResult, after: RunResult) -> None:
    assert before.chunk_count == after.chunk_count, (before, after)
    print(f"corpus: {before.unit_count} units, {before.chunk_count} chunks")
    print(f"{'phase':<22}{'before':>10}{'after':>10}{'saved':>10}")
    for phase in sorted(set(before.phase_seconds) | set(after.phase_seconds)):
        left = before.phase_seconds.get(phase, 0.0)
        right = after.phase_seconds.get(phase, 0.0)
        print(f"{phase:<22}{left:>10.2f}{right:>10.2f}{left - right:>10.2f}")
    left_sum = sum(before.phase_seconds.values())
    right_sum = sum(after.phase_seconds.values())
    print(
        f"{'phase sum':<22}{left_sum:>10.2f}{right_sum:>10.2f}"
        f"{left_sum - right_sum:>10.2f}"
    )
    print(
        f"{'wall clock':<22}{before.wall_seconds:>10.2f}"
        f"{after.wall_seconds:>10.2f}"
        f"{before.wall_seconds - after.wall_seconds:>10.2f}"
    )


async def compare(
    root: Path,
    *,
    label: str,
    variant: str,
    reverse: bool,
    sources: int,
    pages: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[RunResult, RunResult]:
    async def run(name: str, side: str) -> RunResult:
        return await run_once(
            root / name,
            variant=variant,
            side=side,
            sources=sources,
            pages=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    order = ("after", "before") if reverse else ("before", "after")
    first = await run(f"{label}-{order[0]}", order[0])
    second = await run(f"{label}-{order[1]}", order[1])
    before, after = (second, first) if reverse else (first, second)
    return before, after


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B the per-unit write pattern on a real build."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Scratch directory on the device under test; it is replaced.",
    )
    parser.add_argument(
        "--order",
        choices=("both", "pair-then-group", "group-then-pair"),
        default="both",
        help="The second run of a pair sees a warm cache, so both orders matter.",
    )
    parser.add_argument(
        "--variant",
        choices=("write-pattern", "chunk-batch", "embedding-batch"),
        default="write-pattern",
        help=(
            "write-pattern compares the pre-Step-2a per-file durability against "
            "the grouped writes; chunk-batch compares one extraction unit per "
            "caller call against CHUNK_BATCH_UNITS; embedding-batch compares 64 "
            "sequences per ONNX inference against the current constant."
        ),
    )
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    arguments = parser.parse_args()

    root = arguments.root.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    settings = {
        "variant": arguments.variant,
        "sources": arguments.sources,
        "pages": arguments.pages,
        "chunk_size": arguments.chunk_size,
        "chunk_overlap": arguments.chunk_overlap,
    }
    if arguments.order in {"both", "pair-then-group"}:
        print(f"=== {arguments.variant}: before then after ===")
        report(*await compare(root, label="forward", reverse=False, **settings))
    if arguments.order in {"both", "group-then-pair"}:
        print(f"=== {arguments.variant}: after then before ===")
        report(*await compare(root, label="reverse", reverse=True, **settings))

    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
