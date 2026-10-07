"""CLI progress stays on stderr and never issues a second ingestion request."""

import asyncio
import io
import os
import threading
from typing import Any

import pytest

from research_rag.surfaces import cli


def _phase(completed=0, total=10, **changes):
    return {
        "build_id": "build-a",
        "phase": "embedding",
        "progress": {"completed": completed, "total": total, "unit": "chunks"},
        "eta_seconds": None,
        **changes,
    }


def _file_phase(completed=0, total=10, **changes):
    return _phase(
        completed,
        total,
        overall_progress={"completed": completed, "total": total, "unit": "sources"},
        **changes,
    )


def test_phase_eta_needs_two_advances_and_includes_stalled_time():
    estimator = cli._PhaseETA()
    assert estimator.observe(_phase(2), 100) is None
    assert estimator.observe(_phase(2), 102) is None
    assert estimator.observe(_phase(4), 104) is None
    assert estimator.observe(_phase(5), 108) == pytest.approx(40 / 3)
    assert estimator.observe(_phase(5), 112) == 20
    assert estimator.observe(_phase(10), 116) is None


def test_phase_eta_accepts_batch_jumps_and_ignores_source_change():
    estimator = cli._PhaseETA()
    assert estimator.observe(_file_phase(0, source="a.pdf"), 100) is None
    assert estimator.observe(_file_phase(3, source="d.pdf"), 106) is None
    assert estimator.observe(_file_phase(6, source="e.pdf"), 112) == 8
    assert estimator.observe(_phase(9), 118) == 2


@pytest.mark.parametrize(
    "changed",
    [
        _file_phase(6, build_id="build-b"),
        _file_phase(6, phase="dense_indexing"),
        _file_phase(1),
        _file_phase(6, total=20),
        _phase(6, progress={"completed": 6, "total": 10, "unit": "sources"}),
    ],
)
def test_phase_eta_resets(changed):
    estimator = cli._PhaseETA()
    for completed, now in [(0, 100), (2, 102), (4, 104)]:
        estimator.observe(_file_phase(completed), now)
    assert estimator.observe(changed, 106) is None


def test_phase_eta_rendering_uses_global_chunks_and_buckets(monkeypatch):
    clock = iter([100, 102, 104, 106, 108])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    estimator = cli._PhaseETA()
    progress = _file_phase(0)
    assert "phase ETA estimating" in cli._ingestion_progress_line(
        {"ingestion_progress": progress}, estimator
    )
    progress["progress"]["completed"] = 2
    cli._ingestion_progress_line({"ingestion_progress": progress}, estimator)
    progress["progress"]["completed"] = 3
    assert "phase ETA ~10s" in cli._ingestion_progress_line(
        {"ingestion_progress": progress}, estimator
    )
    assert progress["eta_seconds"] is None
    progress["eta_seconds"] = 42
    line = cli._ingestion_progress_line({"ingestion_progress": progress}, estimator)
    assert "phase ETA ~20s" in line
    progress.pop("overall_progress")
    progress["eta_seconds"] = None
    assert "phase ETA ~20s" in cli._ingestion_progress_line(
        {"ingestion_progress": progress}, estimator
    )


@pytest.mark.parametrize("phase", sorted(cli.PASSAGE_ETA_PHASES))
def test_passage_eta_supported_phases_and_resume(phase):
    estimator = cli._PhaseETA()
    assert estimator.observe(_phase(50, 100, phase=phase), 100) is None
    assert estimator.observe(_phase(60, 100, phase=phase), 110) is None
    assert estimator.observe(_phase(70, 100, phase=phase), 120) == 30


@pytest.mark.parametrize(
    "phase", ["source_hashing", "extraction", "chunking", "bm25_indexing", "finalizing"]
)
def test_non_passage_phases_never_display_numeric_eta(phase):
    status = {"ingestion_progress": _file_phase(5, phase=phase, eta_seconds=42)}
    line = cli._ingestion_progress_line(status, cli._PhaseETA())
    assert "unavailable" in line
    assert "~" not in line
    assert "ETA" not in cli._ingestion_progress_frame(status, line, 120)


def test_unknown_total_and_completed_passages_do_not_claim_deadline():
    unknown = {"ingestion_progress": _phase(0, 0, eta_seconds=42)}
    assert (
        "passage ETA unavailable until total is known"
        in cli._ingestion_progress_line(unknown)
    )
    complete = {"ingestion_progress": _phase(10, 10, eta_seconds=0)}
    line = cli._ingestion_progress_line(complete)
    assert "phase ETA finalizing" in line
    assert "~0s" not in line
    assert "finalizing" in cli._ingestion_progress_frame(complete, line, 120)


@pytest.mark.parametrize(
    "counts",
    [
        None,
        {},
        {"completed": True, "total": 10, "unit": "chunks"},
        {"completed": -1, "total": 10, "unit": "chunks"},
        {"completed": 11, "total": 10, "unit": "chunks"},
    ],
)
def test_invalid_passage_counts_do_not_estimate(counts):
    estimator = cli._PhaseETA()
    assert estimator.observe(_phase(progress=counts), 100) is None


def test_eta_minute_buckets_are_stable_across_nearby_polls(monkeypatch):
    clock = iter([100, 110, 120, 122])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    estimator = cli._PhaseETA()
    for completed in (0, 10):
        cli._ingestion_progress_line(
            {"ingestion_progress": _phase(completed, 100)}, estimator
        )
    status = {"ingestion_progress": _phase(20, 100)}
    assert "phase ETA ~2m 0s" in cli._ingestion_progress_line(status, estimator)
    assert "phase ETA ~2m 0s" in cli._ingestion_progress_line(status, estimator)


def test_ingest_progress_is_concurrent_and_stderr_only(monkeypatch, capsys):
    monkeypatch.setattr(cli, "INGEST_PROGRESS_INTERVAL", 0.001)

    class Operations:
        calls = 0
        reads = 0
        finished = False

        def __init__(self):
            self.read = asyncio.Event()

        async def ingest(self, *, force_recompute):
            self.calls += 1
            assert force_recompute
            await self.read.wait()
            self.finished = True
            return {"result": "unchanged"}

        async def status(self):
            assert not self.finished
            self.reads += 1
            if self.reads == 2:
                self.read.set()
            return {
                "ingestion_progress": {
                    "phase": "extraction",
                    "source": "book.pdf",
                    "source_progress": {"completed": 2, "total": 4},
                    "overall_percent": 25,
                    "eta_seconds": 30,
                }
            }

    operations = Operations()
    result = asyncio.run(cli._ingest_with_progress(operations, force_recompute=True))
    assert result == {"result": "unchanged"}
    assert operations.calls == 1
    assert operations.reads == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("Extraction") == 1
    assert "book.pdf" in output.err
    assert "2/4 items" not in output.err
    assert "passage ETA unavailable until total is known" in output.err
    assert "~30s" not in output.err
    assert "source_progress" not in output.err


def test_progress_line_suppresses_duplicate_counters_and_unsafe_source_controls():
    status = {
        "ingestion_progress": {
            "phase": "source_hashing",
            "source": "book\n\x1b[31m.pdf",
            "progress": {"completed": 96, "total": 127, "unit": "sources"},
            "overall_progress": {"completed": 96, "total": 127, "unit": "sources"},
            "source_progress": {"completed": 0, "total": 1, "unit": "sources"},
        }
    }
    line = cli._ingestion_progress_line(status)
    assert (
        line
        == "Hashing sources | 96/127 sources | book  [31m.pdf | phase ETA unavailable"
    )
    assert "\n" not in line and "\x1b" not in line


def test_tty_progress_replaces_one_line_and_finishes_with_newline(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr(cli.sys, "stderr", terminal)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr(
        cli.shutil, "get_terminal_size", lambda fallback: os.terminal_size((36, 24))
    )
    monkeypatch.setattr(cli, "INGEST_PROGRESS_INTERVAL", 0.001)

    async def run():
        class Operations:
            reads = 0

            async def ingest(self, **kwargs):
                await asyncio.sleep(0.008)
                return {"status": "unchanged"}

            async def status(self):
                self.reads += 1
                return {"ingestion_progress": _phase(self.reads, source="书.pdf")}

        return await cli._ingest_with_progress(Operations(), force_recompute=False)

    assert asyncio.run(run()) == {"status": "unchanged"}
    output = terminal.getvalue()
    assert output.count("\r\x1b[2K") >= 2
    assert "\r\x1b[1A" in output
    assert "% | phase ETA" in output
    assert output.endswith("\n")
    assert output.count("\n") >= 3
    assert all(
        len(line.rstrip("\n")) <= 35
        for line in output.split("\r\x1b[2K")[1:]
        if "\x1b" not in line
    )


def test_terminal_progress_clips_by_display_width():
    assert cli._terminal_progress_line("ab书cd", 5) == "ab书"


def test_terminal_frame_uses_phase_percentage_and_eta_without_batch_counts():
    status = {
        "ingestion_progress": _phase(
            20,
            40,
            phase="extraction",
            source="book.pdf",
            source_progress={"completed": 3, "total": 12, "unit": "extraction_units"},
            overall_progress={"completed": 3, "total": 117, "unit": "sources"},
        )
    }
    line = cli._ingestion_progress_line(status)
    frame = cli._ingestion_progress_frame(status, line, 80)
    assert frame.splitlines()[0] == "Extraction | 3/117 sources | book.pdf"
    assert "  3%" in frame.splitlines()[1]
    assert "ETA" not in frame
    assert "3/12" not in frame


@pytest.mark.parametrize("fail", [False, True])
def test_immediate_ingest_does_not_poll(fail):
    class Operations:
        async def ingest(self, **kwargs):
            if fail:
                raise ValueError("build failed")
            return {"ok": True}

        async def status(self):
            pytest.fail("status polled after ingestion returned")

    if fail:
        with pytest.raises(ValueError, match="build failed"):
            asyncio.run(cli._ingest_with_progress(Operations(), force_recompute=False))
    else:
        assert asyncio.run(
            cli._ingest_with_progress(Operations(), force_recompute=False)
        ) == {"ok": True}


def test_failed_progress_read_does_not_fail_ingestion(monkeypatch):
    monkeypatch.setattr(cli, "INGEST_PROGRESS_INTERVAL", 0.001)

    async def run():
        read = asyncio.Event()

        class Operations:
            async def ingest(self, **kwargs):
                await read.wait()
                return {"ok": True}

            async def status(self):
                read.set()
                raise RuntimeError("status unavailable")

        return await cli._ingest_with_progress(Operations(), force_recompute=False)

    assert asyncio.run(run()) == {"ok": True}


def test_remote_blocking_calls_leave_event_loop_free():
    async def run():
        started = threading.Event()
        release = threading.Event()
        loop_thread = threading.get_ident()

        class Control:
            def ingest(self, *, force_recompute) -> dict[str, Any]:
                assert threading.get_ident() != loop_thread
                started.set()
                assert release.wait(2)
                return {"ok": True}

            def status(self):
                assert threading.get_ident() != loop_thread
                assert started.wait(2)
                release.set()
                return {}

        remote = cli.Remote(Control())
        task = asyncio.create_task(remote.ingest(force_recompute=False))
        try:
            await remote.status()
            return await task
        finally:
            release.set()

    assert asyncio.run(run()) == {"ok": True}
