"""CLI routing and local reads do not start a retrieval child process.

The routing tests hand `_operate` a recording stand-in. Read-only commands refuse
subprocess creation and normal service scopes hold the direct lexical backend.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

import research_rag.runtime.app as app_module
import research_rag.runtime.ownership as ownership_module
import research_rag.surfaces.cli as cli_module
from research_rag.core.operations import run
from research_rag.project import registry
from research_rag.project.config import GATEWAY_EXECUTABLE, ConfigurationError
from research_rag.project.policy import ResearchError
from research_rag.runtime import account
from research_rag.surfaces.cli import _run
from research_rag.surfaces.cli import corpus as cli_corpus
from research_rag.surfaces.cli import serving as cli_serving
from research_rag.surfaces.cli.corpus import _metadata_body, _operate
from research_rag.surfaces.cli.parser import _parser
from research_rag.surfaces.cli.projects import _init, _resolve
from research_rag.surfaces.cli.serving import (
    _service_processes,
    _start,
    _stop,
    _terminate,
)
from tests.conftest import write_pdf


def _args(*arguments: str) -> Any:
    return _parser().parse_args(list(arguments))


def _descriptor(project: Path) -> dict[str, Any]:
    path = project / ".research-rag" / "project.json"
    return json.loads(path.read_text(encoding="utf-8"))


class RecordingService:
    """A stand-in that records the one service call a command makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"operation": name, "arguments": arguments}

    async def status(self) -> dict[str, Any]:
        return self._record("status", {})

    async def ingest(self, *, force_recompute: bool = False) -> dict[str, Any]:
        return self._record("ingest", {"force_recompute": force_recompute})

    async def search(self, query: str, **options: Any) -> dict[str, Any]:
        return self._record("search", {"query": query, **options})

    async def list_sources(self, **filters: Any) -> dict[str, Any]:
        return self._record("list_sources", filters)

    async def synthesise(self, question: str, **filters: Any) -> dict[str, Any]:
        self.calls.append(("synthesise", {"question": question, **filters}))
        return {
            "generated": True,
            "synthesis": "It dissolved [S1].",
            "references": "- [S1] Mensah.",
            "provenance": "1. S1: verbatim",
            "reasoning": "- Z: it dissolved.",
            "gaps": 0,
            "sent": {"endpoint": "https://api.deepseek.com", "chunk_ids": ["c-1"]},
        }

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        return self._record("use_generation", {"generation_id": generation_id})

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        return self._record(
            "remove_generation", {"generation_id": generation_id, "confirm": confirm}
        )

    async def get_passage(
        self, chunk_id: str, *, context_chunks: int = 1
    ) -> dict[str, Any]:
        return self._record(
            "get_passage",
            {"chunk_id": chunk_id, "context_chunks": context_chunks},
        )

    async def set_source_inclusion(
        self,
        source_path: str | None = None,
        *,
        source_id: str | None = None,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "set_source_inclusion",
            {
                "source_path": source_path,
                "source_id": source_id,
                "included": included,
                "reason": reason,
            },
        )

    async def set_chunk_inclusion(
        self,
        chunk_id: str,
        *,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "set_chunk_inclusion",
            {"chunk_id": chunk_id, "included": included, "reason": reason},
        )

    async def set_source_metadata(
        self,
        metadata: dict[str, Any],
        *,
        source_path: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "set_source_metadata",
            {"metadata": metadata, "source_path": source_path, "source_id": source_id},
        )


class Operations:
    """The command line does not know or care whether an answer came from the app over
    loopback or from a service it opened itself, so the double runs the operation
    table in process against a recording service, as `Local` does.
    """

    def __init__(self) -> None:
        self.service = RecordingService()

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        if name == "list_generations":
            return {
                "generations": [{"generation_id": "20260930T191235Z-45608dc5"}],
                "retained_generation_count": 1,
                "retained_generation_bytes": 1024,
                "current_generation_id": "20260930T191235Z-45608dc5",
            }
        return await run(self.service, name, arguments, command_line=True)  # type: ignore[arg-type]


def test_init_creates_a_project_and_records_the_name_it_was_given(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fresh"

    report = _init(_args("--project-root", str(project), "init", "--name", "My Thesis"))

    assert report["project_name"] == "My Thesis"
    assert report["created"] == ["project_root", "source_root"]
    assert Path(report["source_root"]).is_dir()
    assert _descriptor(project) == {
        "name": "My Thesis",
        "project_id": report["project_id"],
        "schema_version": 1,
        "source_directory": "sources",
    }
    # A project has no launcher to write: the app is served from the terminal that
    # starts it, so there is nothing in the project that could outlive that terminal.
    assert not (project / ".research-rag" / "bin").exists()


def test_init_records_the_project_so_the_install_can_name_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "thesis"

    report = _init(_args("--project-root", str(project), "init"))

    # The record sits beside the settings this account already has, so one install
    # can name the project.
    assert report["registered"] == {
        "project_id": report["project_id"],
        "project_name": "thesis",
        "project_root": str(project),
        "registered_at": report["registered"]["registered_at"],
    }
    assert report["registry_path"].endswith("projects.json")
    assert [entry.project_id for entry in registry.load()] == [report["project_id"]]


def test_a_command_can_name_a_registered_project_instead_of_its_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "thesis"
    report = _init(_args("--project-root", str(project), "init"))

    assert _resolve(_args("--project", "thesis", "status")).project_root == project
    assert (
        _resolve(_args("--project", report["project_id"], "status")).project_root
        == project
    )
    assert _resolve(_args("status")).project_root == Path.cwd()
    assert (
        _resolve(_args("--project-root", str(project), "status")).project_root
        == project
    )


def test_a_call_that_names_two_projects_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "thesis"
    _init(_args("--project-root", str(project), "init"))

    with pytest.raises(ConfigurationError, match="name two projects"):
        _resolve(_args("--project", "thesis", "--project-root", str(project), "status"))
    with pytest.raises(ResearchError, match="No registered project"):
        _resolve(_args("--project", "no-such-project", "status"))


def test_the_projects_listing_names_every_registered_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))

    empty = asyncio.run(cli_module._run(_args("projects")))
    assert empty.payload is not None
    assert empty.payload["projects"] == []
    assert "No project is registered yet" in empty.payload["message"]

    first = tmp_path / "thesis"
    second = tmp_path / "layout-thesis"
    for project in (first, second):
        _init(_args("--project-root", str(project), "init"))

    listing = asyncio.run(cli_module._run(_args("projects")))
    assert listing.payload is not None
    assert listing.payload["project_count"] == 2
    assert [entry["project_name"] for entry in listing.payload["projects"]] == [
        "layout-thesis",
        "thesis",
    ]
    # Nothing was started to produce the listing: each project is only asked what is on disk.
    for entry in listing.payload["projects"]:
        assert entry["root_exists"] is True
        assert entry["app"] == {
            "running": False,
            "url": None,
            "port": None,
            "attached_to": None,
            "detached": False,
        }
        assert entry["attached_clients"] == 0
        assert "ready" not in entry


def test_the_projects_listing_reports_a_project_whose_directory_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "moved-away"
    _init(_args("--project-root", str(project), "init"))
    for path in sorted(project.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    project.rmdir()

    listing = asyncio.run(cli_module._run(_args("projects")))
    assert listing.payload is not None
    assert listing.payload["projects"][0]["root_exists"] is False
    assert "error" not in listing.payload["projects"][0]


def test_init_defaults_the_name_to_the_directory(tmp_path: Path) -> None:
    project = tmp_path / "plain-name"

    report = _init(_args("--project-root", str(project), "init"))

    assert report["project_name"] == "plain-name"
    assert _descriptor(project)["name"] == "plain-name"


def test_init_attaches_to_an_existing_directory_without_touching_its_files(
    tmp_path: Path,
) -> None:
    project = tmp_path / "existing"
    (project / "papers").mkdir(parents=True)
    (project / "papers" / "already-here.pdf").write_bytes(b"%PDF-1.4\n")
    (project / "README.md").write_text("my own work\n", encoding="utf-8")

    report = _init(_args("--project-root", str(project), "init", "--sources", "papers"))

    assert (project / "README.md").read_text(encoding="utf-8") == "my own work\n"
    assert (project / "papers" / "already-here.pdf").read_bytes() == b"%PDF-1.4\n"
    assert report["created"] == []
    assert report["source_root"] == str(project / "papers")
    assert _descriptor(project)["source_directory"] == "papers"


def test_init_is_idempotent_and_only_a_named_init_renames(tmp_path: Path) -> None:
    project = tmp_path / "stable"
    first = _init(_args("--project-root", str(project), "init", "--name", "First"))
    again = _init(_args("--project-root", str(project), "init"))
    renamed = _init(_args("--project-root", str(project), "init", "--name", "Second"))

    assert again["created"] == []
    assert again["project_name"] == "First"
    assert renamed["project_name"] == "Second"
    # A name is a label, so changing it must not move the identity a generation is
    # recorded against.
    assert first["project_id"] == again["project_id"] == renamed["project_id"]
    assert _descriptor(project)["name"] == "Second"


def test_init_refuses_to_move_an_initialized_project_to_another_source_directory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fixed-sources"
    _init(_args("--project-root", str(project), "init"))

    with pytest.raises(ConfigurationError, match="source directory differs"):
        _init(_args("--project-root", str(project), "init", "--sources", "docs"))


def test_init_refuses_a_name_that_cannot_be_recorded(tmp_path: Path) -> None:
    project = tmp_path / "bad-name"

    with pytest.raises(ConfigurationError, match="cannot be empty"):
        _init(_args("--project-root", str(project), "init", "--name", "   "))


def test_the_command_line_routes_each_command_to_its_service_operation() -> None:
    service = Operations()

    status = asyncio.run(_operate(_args("status"), service))
    # `message` is absent rather than null: the projection carries a sentence only
    # when it explains something, and "nothing has changed" needs none. Asserting
    # the absence is the contract, so a future `message: None` cannot creep back
    # in as a field every reader of `status` has to ignore.
    assert status == {
        "ready": False,
        "stale": False,
        "requires": ["ingest"],
    }
    assert "message" not in status
    verbose = asyncio.run(_operate(_args("status", "--verbose"), service))
    assert verbose["operation"] == "status"

    refresh = asyncio.run(_operate(_args("ingest", "--force-recompute"), service))
    assert refresh["arguments"] == {"force_recompute": True}

    sources = asyncio.run(_operate(_args("sources"), service))
    assert sources["operation"] == "list_sources"

    passage = asyncio.run(
        _operate(_args("passage", "abc123", "--context-chunks", "2"), service)
    )
    assert passage["arguments"] == {"chunk_id": "abc123", "context_chunks": 2}

    restored = asyncio.run(
        _operate(
            _args("include", "a.pdf"),
            service,
        )
    )
    assert restored["arguments"]["included"] is True
    assert restored["arguments"]["source_path"] == "a.pdf"


class _StatusStub:
    """A status answer with a diagnostic block a lean projection must drop."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        assert name == "status"
        return self._payload


def test_status_verbose_exposes_generation_diagnostics_that_lean_hides() -> None:
    diagnostics = {
        "policy_differences": ["layout_extraction", "retrieval_policy"],
        "recorded": {
            "extraction_policy_version": 2,
            "retrieval_policy_fingerprint": "a",
        },
        "current": {
            "extraction_policy_version": 3,
            "retrieval_policy_fingerprint": "b",
        },
    }
    payload = {
        "ready": True,
        "stale": False,
        "generation_id": "20260101T000000Z-abcdef",
        # A usable generation that serves only BM25: the missing dense half is a
        # limitation, and `generation_upgrade_required` is about serving, not drift.
        "hybrid_ready": False,
        "hybrid_upgrade_required": True,
        "generation_upgrade_required": False,
        "upgrade_reasons": [],
        "generation_diagnostics": diagnostics,
    }
    operations = _StatusStub(payload)

    # The normal projection is the bounded verdict: diagnostics stay behind
    # `--verbose`, and a usable generation asks for no build.
    lean = asyncio.run(_operate(_args("status"), operations))
    assert "generation_diagnostics" not in lean
    assert "requires" not in lean

    verbose = asyncio.run(_operate(_args("status", "--verbose"), operations))
    assert verbose["generation_diagnostics"] == diagnostics


class _LoggingSettings:
    def __init__(self, log_level: str) -> None:
        self.log_level = log_level


class _LoggingConfig:
    def __init__(self, log_level: str) -> None:
        self.settings = _LoggingSettings(log_level)


def test_cli_logging_follows_the_project_level_on_the_named_logger() -> None:
    """Only `research_rag` is configured; a dependency keeps its own level.

    The handler is installed once, so a command that resolves logging twice does
    not stack duplicates, and an unknown level leaves the logger as it was.
    """

    logger = logging.getLogger(cli_module._LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    try:
        cli_module._configure_logging(_LoggingConfig("debug"))
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert len(logger.handlers) == 1

        cli_module._configure_logging(_LoggingConfig("warn"))
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1

        cli_module._configure_logging(_LoggingConfig("not-a-level"))
        assert logger.level == logging.WARNING
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate


def test_search_carries_its_filters_and_reranks_by_default() -> None:
    service = Operations()

    payload = asyncio.run(
        _operate(
            _args(
                "search",
                "articulation",
                "--top-k",
                "12",
                "--category",
                "theory",
                "--author",
                "Crawford",
                "--title",
                "Atlas of AI",
                "--exclude-source-id",
                "sid-1",
            ),
            service,
        )
    )

    assert payload["arguments"] == {
        "query": "articulation",
        "top_k": 12,
        "categories_any": ["theory"],
        "projects_any": None,
        "keywords": None,
        "languages_any": None,
        "authors_any": ["Crawford"],
        "titles_any": ["Atlas of AI"],
        "source_ids": None,
        "exclude_source_ids": ["sid-1"],
        "rerank": True,
        "include_staleness": True,
    }


def test_excluding_a_source_records_the_reason_it_was_given() -> None:
    service = Operations()

    payload = asyncio.run(
        _operate(
            _args("exclude", "a.pdf", "--reason", "superseded by the reprint"),
            service,
        )
    )
    assert payload["arguments"] == {
        "source_path": "a.pdf",
        "source_id": None,
        "included": False,
        "reason": "superseded by the reprint",
    }


def test_excluding_a_passage_names_the_chunk_it_decides_about() -> None:
    """One command, two subjects: --chunk decides about a passage, a path about a file."""

    service = Operations()

    payload = asyncio.run(
        _operate(
            _args("exclude", "--chunk", "chk_1a2b", "--reason", "header repeated"),
            service,
        )
    )
    assert payload["arguments"] == {
        "chunk_id": "chk_1a2b",
        "included": False,
        "reason": "header repeated",
    }

    restored = asyncio.run(_operate(_args("include", "--chunk", "chk_1a2b"), service))
    assert restored["arguments"] == {
        "chunk_id": "chk_1a2b",
        "included": True,
        "reason": None,
    }
    assert not [
        call for call in service.service.calls if call[0] == "set_source_inclusion"
    ]


def test_a_decision_that_names_two_subjects_or_none_is_refused() -> None:
    """A command that dropped one of the two subjects would record a decision about a file
    when the reader asked about a passage, or the other way round, and neither reader
    would learn it from the answer.
    """

    service = RecordingService()

    with pytest.raises(ResearchError, match="Name the passage"):
        asyncio.run(_operate(_args("include"), service))

    with pytest.raises(ResearchError, match="Name the passage with --chunk"):
        asyncio.run(_operate(_args("exclude", "--reason", "nothing named"), service))

    with pytest.raises(ResearchError, match="about one passage"):
        asyncio.run(
            _operate(
                _args(
                    "exclude",
                    "a.pdf",
                    "--chunk",
                    "chk_1a2b",
                    "--reason",
                    "two subjects",
                ),
                service,
            )
        )

    with pytest.raises(ResearchError, match="about one passage"):
        asyncio.run(
            _operate(
                _args(
                    "include",
                    "--source-id",
                    "src_one",
                    "--chunk",
                    "chk_1a2b",
                ),
                service,
            )
        )

    assert service.calls == []


def test_metadata_clear_cannot_be_combined_with_a_field() -> None:
    with pytest.raises(ResearchError, match="cannot be combined"):
        _metadata_body(_args("metadata", "--clear", "--title", "X"))

    assert _metadata_body(_args("metadata", "--clear")) == {}
    assert _metadata_body(
        _args("metadata", "--title", "T", "--author", "A", "--author", "B")
    ) == {"title": "T", "authors": ["A", "B"]}


def test_a_reading_command_never_starts_a_child_process(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a command that only reads started a child process")

    monkeypatch.setattr(subprocess, "Popen", refuse)

    result = asyncio.run(_run(_args("--project-root", str(project), "sources")))

    assert result.payload is not None
    assert result.payload["source_count"] == 0


def test_normal_service_scope_uses_direct_backend(project: Path) -> None:
    from research_rag.retrieval.direct import DirectRetrieval
    from research_rag.surfaces.cli.corpus import _service

    config = _resolve(_args("--project-root", str(project), "status"))

    async def scenario() -> None:
        async with _service(config) as service:
            assert isinstance(service.lexical, DirectRetrieval)
            assert (await service.status())["ready"] is False

    asyncio.run(scenario())


def test_the_doctor_reports_without_starting_a_child_process(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every default dependency check reads; none starts a serving component."""

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the doctor started a child process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    write_pdf(project / "sources" / "evidence.pdf", ["The cobalt heron."])

    result = asyncio.run(_run(_args("--project-root", str(project), "doctor")))

    assert result.text is not None
    assert "project_identity" in result.text
    assert "direct_retrieval" in result.text
    assert result.exit_code == 1


def _fake_process(proc_root: Path, pid: int, arguments: list[str]) -> None:
    directory = proc_root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(("\0".join(arguments) + "\0").encode("utf-8"))


def test_the_stop_sweep_finds_only_processes_serving_this_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    proc_root = tmp_path / "proc"
    server = ["python", "-m", "research_rag", "--project-root", str(project)]
    _fake_process(proc_root, 10, server)
    # Another project's server, a shell and an editor naming this path, and a process of
    # ours with no project at all.
    _fake_process(proc_root, 11, [*server[:-1], str(tmp_path / "elsewhere")])
    _fake_process(proc_root, 12, ["/bin/bash", "-c", f"echo {project}"])
    _fake_process(proc_root, 13, ["vim", str(project)])
    _fake_process(proc_root, 14, ["python", "-m", "research_rag"])

    found = _service_processes(project, proc_root)

    assert [pid for pid, _ in found] == [10]
    assert found[0][1].startswith("python -m research_rag --project-root")


def test_the_stop_sweep_reads_a_path_as_a_path_not_as_a_program(tmp_path: Path) -> None:
    """A marker test that looks for the product name inside an argument reads a project
    directory as though it were the program being run, so the name has to be a whole
    argument or the sweep's second half is undone by its first.
    """

    from research_rag.runtime.ownership import invokes_this_app

    project = tmp_path / "research-rag"
    state = project / ".research-rag" / "runtime"

    assert not invokes_this_app(
        ["--project-root", str(project), "--runtime-root", str(state)], project
    )
    assert invokes_this_app(
        ["/srv/research-rag/.venv/bin/research-rag", "--project-root", str(project)],
        project,
    )
    assert invokes_this_app(["python", "-m", "research_rag"], project)
    assert not invokes_this_app(["python", "-m", "some_other_package"], project)


def test_the_stop_sweep_leaves_the_other_products_processes_alone(
    tmp_path: Path,
) -> None:
    """A hand-written wrapper and the vanilla gateway name this project, so a sweep
    that ended them would stop a process a reader started on purpose.
    """

    project = tmp_path / "ai-and-fetishism"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    elsewhere = "/srv/some-other-tool/.venv/bin"
    state = project / ".research-rag" / "runtime" / "ultrarag-runtime"
    _fake_process(
        proc_root,
        31,
        [
            f"{elsewhere}/python",
            f"{elsewhere}/another-rag-mcp",
            "--project-root",
            str(project),
        ],
    )
    _fake_process(
        proc_root,
        32,
        [
            f"{elsewhere}/python",
            f"{elsewhere}/{GATEWAY_EXECUTABLE}",
            "--workspace-root",
            str(state),
            "--log-level",
            "warn",
        ],
    )
    _fake_process(
        proc_root,
        33,
        [
            "/srv/research-rag/.venv/bin/research-rag",
            "--project-root",
            str(project),
            "ui",
        ],
    )

    found = [pid for pid, _command in _service_processes(project, proc_root)]

    assert found == [33]


def test_the_stop_sweep_accepts_the_equals_form_of_the_option(tmp_path: Path) -> None:
    project = tmp_path / "project"
    proc_root = tmp_path / "proc"
    _fake_process(
        proc_root,
        21,
        ["python", "-m", "research_rag", f"--project-root={project}"],
    )

    assert [pid for pid, _ in _service_processes(project, proc_root)] == [21]


def test_stopping_asks_first_and_kills_only_the_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep asks, waits, and insists only on what survived the asking.

    The sweep signals through the ownership proof rather than through a bare
    number, so this watches the proof's own view: a target it cannot prove is
    asked for twice and forced never.
    """

    asked: list[tuple[int, int]] = []
    still_running = {2}

    monkeypatch.setattr(cli_serving, "STOP_GRACE_SECONDS", 0)
    monkeypatch.setattr(
        cli_serving.ownership,
        "ask_to_stop",
        lambda pid, _root, number=signal.SIGTERM: (
            asked.append((pid, number)) or ownership_module.Outcome(True, "")
        ),
    )
    monkeypatch.setattr(cli_serving, "alive", lambda pid: pid in still_running)

    forced = _terminate(tmp_path / "project", [1, 2])

    assert asked == [(1, signal.SIGTERM), (2, signal.SIGTERM), (2, signal.SIGKILL)]
    assert forced == [2]


def test_a_pid_the_proof_refuses_is_never_signalled_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep's escalation is bounded by the same proof as its asking.

    A number the proof refuses would still be a candidate for the second, harder
    signal, so the refusal has to hold through both rounds rather than only the
    first.
    """

    attempted: list[tuple[int, int]] = []
    monkeypatch.setattr(cli_serving, "STOP_GRACE_SECONDS", 0)
    monkeypatch.setattr(cli_serving, "alive", lambda pid: True)
    monkeypatch.setattr(
        cli_serving.ownership,
        "ask_to_stop",
        lambda pid, _root, number=signal.SIGTERM: (
            attempted.append((pid, number))
            or ownership_module.Outcome(False, "not this project's app")
        ),
    )

    forced = _terminate(tmp_path / "project", [4242])

    assert attempted == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert forced == [4242]


def test_stop_without_servers_reports_the_app_and_starts_nothing(
    project: Path,
) -> None:
    """A stop is a question about the app, answered without signalling anything."""

    args = _args("--project-root", str(project), "stop")

    report = _stop(args, _resolve(args))

    assert report["project_root"] == str(project)
    assert report["running"] is False
    assert report["attached_to"] is None
    assert "servers" not in report
    assert "forced_pids" not in report


def test_start_serves_in_this_terminal_and_records_the_port_it_chose(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`start` runs the app here, so the port and the pid belong to this process."""

    served: list[object] = []

    async def _serving(config: Any, *, port: int | None, open_browser: bool):
        served.append((config, port, open_browser))
        return cli_serving.CommandResult()

    monkeypatch.setattr(cli_serving, "_serve_attached", _serving)
    args = _args("--project-root", str(project), "start", "--port", "5099")
    config = _resolve(args)

    asyncio.run(_start(args, config))

    assert served == [(config, 5099, False)]


def test_start_opens_a_browser_only_when_asked(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command line is where this app is worked from, so a browser is opt-in."""

    opened: list[str] = []
    served: list[object] = []

    async def _serving(config: Any, *, port: int | None, open_browser: bool):
        served.append((config, port, open_browser))
        return cli_serving.CommandResult()

    monkeypatch.setattr(cli_serving, "_open_browser", opened.append)
    monkeypatch.setattr(cli_serving, "_serve_attached", _serving)
    args = _args(
        "--project-root", str(project), "--start-ui", "start", "--port", "5099"
    )
    config = _resolve(args)

    asyncio.run(_start(args, config))

    assert served == [(config, 5099, True)]
    assert opened == []


def test_start_leaves_an_app_another_terminal_owns_alone(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reader deciding whether Ctrl-C here stops that app needs the difference."""

    served: list[object] = []

    async def _serving(*_: Any, **__: Any):
        served.append(True)
        return cli_serving.CommandResult()

    monkeypatch.setattr(cli_serving, "_serve_attached", _serving)
    # The serving process is declared attached, because this test is about an app
    # another terminal owns and not about the case that one owns no terminal.
    monkeypatch.setattr(account, "has_terminal", lambda _pid: True)
    args = _args("--project-root", str(project), "start")
    config = _resolve(args)
    # Both files are what the app writes, and a recorded port whose process is gone
    # reads as no app at all.
    (config.state_root / "research-rag-ui.port").write_text("5099\n", encoding="utf-8")
    (config.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )

    asyncio.run(_start(args, config))

    assert served == []
    assert "already served" in capsys.readouterr().out


def test_start_names_a_detached_app_and_its_remedy(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A process with no terminal holds the project, so no terminal holds it, and
    blaming one is a sentence about a terminal that does not exist."""

    served: list[object] = []

    async def _serving(*_: Any, **__: Any):
        served.append(True)
        return cli_serving.CommandResult()

    monkeypatch.setattr(cli_serving, "_serve_attached", _serving)
    monkeypatch.setattr(app_module, "has_terminal", lambda _pid: False)
    args = _args("--project-root", str(project), "start")
    config = _resolve(args)
    (config.state_root / "research-rag-ui.port").write_text("5099\n", encoding="utf-8")
    (config.state_root / "research-rag-ui.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )

    asyncio.run(_start(args, config))

    out = capsys.readouterr().out
    assert served == []
    assert "no terminal attached" in out
    assert "'research-rag stop'" in out
    assert "this terminal does not own" not in out


def test_generations_lists_and_can_roll_back() -> None:
    """`generations` is a listing, and `--use` is the one call that moves the pointer."""

    service = Operations()

    listed = asyncio.run(_operate(_args("generations"), service))

    assert listed["generations"] == [{"generation_id": "20260930T191235Z-45608dc5"}]
    assert listed["retained_generation_count"] == 1
    assert listed["current_generation_id"] == "20260930T191235Z-45608dc5"
    assert service.service.calls == []

    rolled = asyncio.run(
        _operate(_args("generations", "--use", "20260930T191235Z-45608dc5"), service)
    )

    assert rolled["operation"] == "use_generation"
    assert rolled["arguments"] == {"generation_id": "20260930T191235Z-45608dc5"}


def test_remove_generation_carries_the_repeated_id() -> None:
    """The command line passes the confirmation through rather than inventing one."""

    service = Operations()
    generation_id = "20260930T191235Z-45608dc5"

    removed = asyncio.run(
        _operate(
            _args("remove-generation", generation_id, "--confirm", generation_id),
            service,
        )
    )

    assert removed["operation"] == "remove_generation"
    assert removed["arguments"] == {
        "generation_id": generation_id,
        "confirm": generation_id,
    }


def test_remove_generation_refuses_without_a_confirmation() -> None:
    """`--confirm` is required, so a listing cannot become a deletion by accident."""

    with pytest.raises(SystemExit):
        _parser().parse_args(["remove-generation", "20260930T191235Z-45608dc5"])


def test_the_running_app_answers_in_a_thread_and_without_diagnostics() -> None:
    """`Control` is synchronous, so awaiting its answer raised a TypeError.

    The running app chooses its own retrieval, so the command line's diagnostic
    search flags are not sent to it.
    """

    class Answering:
        def operation(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"name": name, "arguments": arguments}

    async def exercise() -> list[dict[str, Any]]:
        remote = cli_corpus.Remote(Answering())  # type: ignore[arg-type]
        return [
            await remote.call("list_generations"),
            await remote.call("remove_generation", generation_id="g1", confirm="g1"),
            await remote.call(
                "search", query="q", rerank=False, retrieval_method="bm25"
            ),
        ]

    assert asyncio.run(exercise()) == [
        {"name": "list_generations", "arguments": {}},
        {
            "name": "remove_generation",
            "arguments": {"generation_id": "g1", "confirm": "g1"},
        },
        {"name": "search", "arguments": {"query": "q"}},
    ]


def test_config_writes_the_project_file_through_the_workspace_check(
    project: Path,
) -> None:
    """Every setting the workspace writes, the command line writes the same way.

    The values go through the service call the workspace makes, so they are coerced,
    refused, and costed alike, and a change that alters what the next build records
    waits for `--confirm`.
    """

    def config(*arguments: str) -> Any:
        return asyncio.run(
            _run(_args("--project-root", str(project), "config", *arguments))
        )

    written = config(
        "--write",
        "retrieval.exclude_apparatus=true",
        "--write",
        "retrieval.minimum_passage_words=12",
    )
    assert written.payload["changed"] == [
        "retrieval.exclude_apparatus",
        "retrieval.minimum_passage_words",
    ]
    stored = (project / ".research-rag" / "config.toml").read_text(encoding="utf-8")
    assert "exclude_apparatus = true" in stored
    settings = _resolve(_args("--project-root", str(project), "status")).settings
    assert settings.exclude_apparatus is True
    assert settings.minimum_passage_words == 12

    with pytest.raises(ResearchError, match="KEY=VALUE"):
        config("--write", "retrieval.exclude_apparatus")

    with pytest.raises(ResearchError, match=r"chunking\.size"):
        config("--write", "chunking.size=200")
    assert config("--write", "chunking.size=200", "--confirm").payload["changed"] == [
        "chunking.size"
    ]


def test_api_key_set_reads_the_key_from_stdin_and_never_prints_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from research_rag.project.credentials import deepseek_key

    monkeypatch.setattr("sys.stdin", io.StringIO("sk-from-stdin\n"))
    cli_module.main(["api-key", "set"])
    printed = capsys.readouterr().out

    assert deepseek_key() == ("sk-from-stdin", "account file")
    assert "sk-from-stdin" not in printed
    assert json.loads(printed)["deepseek"]["set"] is True


def test_api_key_status_and_clear_name_where_the_key_lives(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from research_rag.project.credentials import credentials_path, set_deepseek_key

    set_deepseek_key("sk-held")
    cli_module.main(["api-key", "status"])
    held = json.loads(capsys.readouterr().out)["deepseek"]
    cli_module.main(["api-key", "clear"])
    cleared = json.loads(capsys.readouterr().out)["deepseek"]
    monkeypatch.setenv("RESEARCH_RAG_DEEPSEEK_API_KEY", "sk-environment")
    cli_module.main(["api-key", "status"])
    environment = json.loads(capsys.readouterr().out)["deepseek"]

    assert held == {
        "set": True,
        "origin": "account file",
        "path": str(credentials_path()),
    }
    assert cleared["set"] is False
    assert environment["origin"] == "environment"


def test_api_key_takes_no_key_on_the_command_line() -> None:
    assert _args("api-key", "set").action == "set"
    with pytest.raises(SystemExit):
        _args("api-key", "set", "sk-in-shell-history")


@pytest.mark.parametrize("verbose", [False, True])
def test_synthesise_carries_its_filters_and_shows_the_rest_only_when_verbose(
    verbose: bool,
) -> None:
    operations = Operations()
    argv = [
        "synthesise",
        "Why did it dissolve?",
        "--author",
        "Mensah",
        "--reasoning",
        "low",
    ]

    payload = asyncio.run(
        _operate(_args(*argv, *(["--verbose"] if verbose else [])), operations)
    )

    (call,) = operations.service.calls
    assert call == (
        "synthesise",
        {
            "question": "Why did it dissolve?",
            "reasoning": "low",
            "categories_any": None,
            "projects_any": None,
            "keywords": None,
            "languages_any": None,
            "authors_any": ["Mensah"],
            "titles_any": None,
            "source_ids": None,
            "exclude_source_ids": None,
        },
    )
    assert payload["synthesis"] == "It dissolved [S1]."
    assert payload["references"] == "- [S1] Mensah."
    assert ("provenance" in payload, "reasoning" in payload) == (verbose, verbose)


def test_synthesise_offers_no_reasoning_below_low() -> None:
    assert _args("synthesise", "Why?", "--reasoning", "LOW").reasoning == "low"
    with pytest.raises(SystemExit):
        _args("synthesise", "Why?", "--reasoning", "none")
