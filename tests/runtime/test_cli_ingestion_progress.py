"""CLI progress stays on stderr and never issues a second ingestion request."""

import asyncio
import io
import os
import threading
from typing import Any

import pytest

from research_rag.surfaces.cli import corpus as cli_corpus


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


@pytest.mark.parametrize(
    "progress, percentage",
    [
        (_file_phase(5, phase="extraction"), "50%"),
        (_phase(5, eta_seconds=42), "50%"),
        (_phase(0, 0), "--%"),
        (_phase(phase="finalizing", progress={"unit": "phase"}), "--%"),
    ],
    ids=["sources", "passages-with-backend-eta", "unknown-total", "uncounted-phase"],
)
def test_progress_displays_no_eta(progress, percentage):
    status = {"ingestion_progress": progress}
    line = cli_corpus._ingestion_progress_line(status)
    frame = cli_corpus._ingestion_progress_frame(status, line, 120)
    assert "ETA" not in line
    assert "ETA" not in frame
    assert percentage in frame
    if progress["progress"].get("total", 0):
        assert "5/10" in line


def test_completed_passages_display_finalizing():
    status = {"ingestion_progress": _phase(10, 10)}
    line = cli_corpus._ingestion_progress_line(status)
    frame = cli_corpus._ingestion_progress_frame(status, line, 120)
    assert "Finalizing" in line
    assert "Finalizing" in frame
    assert "100%" in frame


@pytest.mark.parametrize(
    "changes",
    [
        {"build_id": "build-b"},
        {"phase": "chunking"},
        {"source": "next.pdf"},
        {"progress": {"completed": 1, "total": 10, "unit": "chunks"}},
        {"overall_progress": {"completed": 1, "total": 10, "unit": "sources"}},
        {"source_progress": {"completed": 3, "total": 10, "unit": "extraction_units"}},
    ],
    ids=["build", "phase", "source", "global-count", "source-count", "within-source"],
)
def test_stall_is_visible_and_activity_resets_inactivity(monkeypatch, changes):
    now = 100.0
    monkeypatch.setattr(cli_corpus.time, "monotonic", lambda: now)
    activity = cli_corpus._IngestionActivity()
    progress = _file_phase(
        0,
        phase="extraction",
        source="book.pdf",
        source_progress={"completed": 2, "total": 10, "unit": "extraction_units"},
    )
    status = {"ingestion_progress": progress}
    cli_corpus._ingestion_progress_line(status, activity)
    now = 159
    assert "Stalled" not in cli_corpus._ingestion_progress_line(status, activity)
    now = 160
    line = cli_corpus._ingestion_progress_line(status, activity)
    assert "Stalled" in line
    assert "Stalled" in cli_corpus._ingestion_progress_frame(status, line, 120)
    assert "ready" not in line.lower()
    progress.update(changes)
    assert "Stalled" not in cli_corpus._ingestion_progress_line(status, activity)
    now = 219
    assert "Stalled" not in cli_corpus._ingestion_progress_line(status, activity)
    now = 220
    assert "Stalled" in cli_corpus._ingestion_progress_line(status, activity)


def test_missing_progress_clears_inactivity(monkeypatch):
    now = 100.0
    monkeypatch.setattr(cli_corpus.time, "monotonic", lambda: now)
    activity = cli_corpus._IngestionActivity()
    status = {"ingestion_progress": _phase()}
    cli_corpus._ingestion_progress_line(status, activity)
    now = 160
    assert "Stalled" in cli_corpus._ingestion_progress_line(status, activity)
    assert cli_corpus._ingestion_progress_line({}, activity) is None
    assert "Stalled" not in cli_corpus._ingestion_progress_line(status, activity)


def test_ingest_progress_is_concurrent_and_stderr_only(monkeypatch, capsys):
    monkeypatch.setattr(cli_corpus, "INGEST_PROGRESS_INTERVAL", 0.001)

    class Operations:
        async def call(self, name, **arguments):

            return await getattr(self, name)(**arguments)

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
    result = asyncio.run(
        cli_corpus._ingest_with_progress(operations, "ingest", force_recompute=True)
    )
    assert result == {"result": "unchanged"}
    assert operations.calls == 1
    assert operations.reads == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("Extraction") == 1
    assert "book.pdf" in output.err
    assert "2/4 items" not in output.err
    assert "ETA" not in output.err
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
    line = cli_corpus._ingestion_progress_line(status)
    assert line == "Hashing sources | 96/127 sources | book  [31m.pdf"
    assert "\n" not in line and "\x1b" not in line


def test_non_tty_heartbeat_reports_status_without_claiming_progress(
    monkeypatch, capsys
):
    monkeypatch.setattr(cli_corpus, "INGEST_PROGRESS_INTERVAL", 0.001)
    monkeypatch.setattr(cli_corpus, "INGEST_HEARTBEAT_INTERVAL", 0)
    monkeypatch.setattr(cli_corpus.sys.stderr, "isatty", lambda: False)

    async def run():
        finished = asyncio.Event()

        class Operations:
            reads = 0

            async def call(self, name, **arguments):
                if name == "ingest":
                    await finished.wait()
                    return {"ok": True}
                assert name == "status"
                self.reads += 1
                if self.reads == 3:
                    finished.set()
                return {"ingestion_progress": _phase(5, source="book.pdf")}

        return await cli_corpus._ingest_with_progress(Operations(), "ingest")

    assert asyncio.run(run()) == {"ok": True}
    output = capsys.readouterr()
    assert output.out == ""
    lines = output.err.splitlines()
    assert len(lines) == 2
    assert "status checked" not in lines[0]
    assert lines[1].startswith(lines[0] + " | status checked ")
    assert "Embedding | 5/10 chunks | book.pdf" in lines[0]
    assert "ETA" not in output.err


def test_tty_progress_replaces_one_line_and_finishes_with_newline(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr(cli_corpus.sys, "stderr", terminal)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr(
        cli_corpus.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((36, 24)),
    )
    monkeypatch.setattr(cli_corpus, "INGEST_PROGRESS_INTERVAL", 0.001)

    async def run():
        class Operations:
            async def call(self, name, **arguments):
                return await getattr(self, name)(**arguments)

            reads = 0

            async def ingest(self, **kwargs):
                await asyncio.sleep(0.008)
                return {"status": "unchanged"}

            async def status(self):
                self.reads += 1
                return {"ingestion_progress": _phase(self.reads, source="书.pdf")}

        return await cli_corpus._ingest_with_progress(
            Operations(), "ingest", force_recompute=False
        )

    assert asyncio.run(run()) == {"status": "unchanged"}
    output = terminal.getvalue()
    assert output.count("\r\x1b[2K") >= 2
    assert "\r\x1b[1A" in output
    assert "%" in output and "[#" in output
    assert "ETA" not in output
    assert output.endswith("\n")
    assert output.count("\n") >= 3
    assert all(
        len(line.rstrip("\n")) <= 35
        for line in output.split("\r\x1b[2K")[1:]
        if "\x1b" not in line
    )


def test_terminal_progress_clips_by_display_width():
    assert cli_corpus._terminal_progress_line("ab书cd", 5) == "ab书"


def test_terminal_frame_uses_phase_percentage_without_batch_counts():
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
    line = cli_corpus._ingestion_progress_line(status)
    frame = cli_corpus._ingestion_progress_frame(status, line, 80)
    assert frame.splitlines()[0] == "Extraction | 3/117 sources | book.pdf"
    assert "  3%" in frame.splitlines()[1]
    assert "[#" in frame
    assert "ETA" not in frame
    assert "3/12" not in frame


@pytest.mark.parametrize("fail", [False, True])
def test_immediate_ingest_does_not_poll(fail):
    class Operations:
        async def call(self, name, **arguments):
            return await getattr(self, name)(**arguments)

        async def ingest(self, **kwargs):
            if fail:
                raise ValueError("build failed")
            return {"ok": True}

        async def status(self):
            pytest.fail("status polled after ingestion returned")

    if fail:
        with pytest.raises(ValueError, match="build failed"):
            asyncio.run(
                cli_corpus._ingest_with_progress(
                    Operations(), "ingest", force_recompute=False
                )
            )
    else:
        assert asyncio.run(
            cli_corpus._ingest_with_progress(
                Operations(), "ingest", force_recompute=False
            )
        ) == {"ok": True}


def test_failed_progress_read_does_not_fail_ingestion(monkeypatch):
    monkeypatch.setattr(cli_corpus, "INGEST_PROGRESS_INTERVAL", 0.001)

    async def run():
        read = asyncio.Event()

        class Operations:
            async def call(self, name, **arguments):

                return await getattr(self, name)(**arguments)

            async def ingest(self, **kwargs):
                await read.wait()
                return {"ok": True}

            async def status(self):
                read.set()
                raise RuntimeError("status unavailable")

        return await cli_corpus._ingest_with_progress(
            Operations(), "ingest", force_recompute=False
        )

    assert asyncio.run(run()) == {"ok": True}


def test_remote_blocking_calls_leave_event_loop_free():
    async def run():
        started = threading.Event()
        release = threading.Event()
        loop_thread = threading.get_ident()

        class Control:
            def operation(self, name, arguments) -> dict[str, Any]:
                assert threading.get_ident() != loop_thread
                if name == "ingest":
                    started.set()
                    assert release.wait(2)
                    return {"ok": True}
                assert started.wait(2)
                release.set()
                return {}

        remote = cli_corpus.Remote(Control())
        task = asyncio.create_task(remote.call("ingest", force_recompute=False))
        try:
            await remote.call("status")
            return await task
        finally:
            release.set()

    assert asyncio.run(run()) == {"ok": True}
