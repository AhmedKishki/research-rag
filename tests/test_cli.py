"""No test here starts the vanilla gateway, and that is part of what is under test.

The routing tests hand `_operate` a recording stand-in and the laziness tests make the
transport refuse to be built.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest
from conftest import write_pdf

import research_rag.surfaces.cli as cli_module
import research_rag.ultrarag as ultrarag_module
from research_rag import launcher as launcher_module
from research_rag import registry
from research_rag.config import ConfigurationError
from research_rag.settings import SETTINGS
from research_rag.support import ResearchError
from research_rag.surfaces.cli import (
    _init,
    _metadata_body,
    _operate,
    _parser,
    _resolve,
    _run,
    _service_processes,
    _start,
    _stop,
    _terminate,
)
from research_rag.ultrarag import LazyGateway


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

    async def list_sources(self) -> dict[str, Any]:
        return self._record("list_sources", {})

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
    loopback or from a service it opened itself, so the double stands in for the
    interface, not for either implementation.
    """

    def __init__(self) -> None:
        self.service = RecordingService()

    async def status(self) -> dict[str, Any]:
        return await self.service.status()

    async def ingest(self, *, force_recompute: bool) -> dict[str, Any]:
        return await self.service.ingest(force_recompute=force_recompute)

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return await self.service.search(query, **arguments)

    async def sources(self) -> dict[str, Any]:
        return await self.service.list_sources()

    async def passage(self, chunk_id: str, *, context_chunks: int) -> dict[str, Any]:
        return await self.service.get_passage(chunk_id, context_chunks=context_chunks)

    async def set_source_inclusion(
        self,
        *,
        source_path: str | None,
        source_id: str | None,
        included: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return await self.service.set_source_inclusion(
            source_path, source_id=source_id, included=included, reason=reason
        )

    async def set_chunk_inclusion(
        self,
        *,
        chunk_id: str,
        included: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return await self.service.set_chunk_inclusion(
            chunk_id, included=included, reason=reason
        )

    async def set_source_metadata(
        self,
        *,
        source_path: str | None,
        source_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.service.set_source_metadata(
            metadata, source_path=source_path, source_id=source_id
        )

    async def generations(self) -> dict[str, Any]:
        return {
            "generations": [{"generation_id": "20260930T191235Z-45608dc5"}],
            "retained_generation_count": 1,
            "retained_generation_bytes": 1024,
            "current_generation_id": "20260930T191235Z-45608dc5",
        }

    async def use_generation(self, generation_id: str) -> dict[str, Any]:
        return await self.service.use_generation(generation_id)

    async def remove_generation(
        self, generation_id: str, *, confirm: str
    ) -> dict[str, Any]:
        return await self.service.remove_generation(generation_id, confirm=confirm)


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
    assert (project / ".research-rag" / "bin" / "open-research-rag-ui.sh").is_file()
    assert Path(report["launcher"]["link_path"]).is_symlink()


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
    assert status == {
        "ready": False,
        "stale": False,
        "requires": ["ingest"],
        "message": None,
    }
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


def test_search_carries_its_filters_and_reranks_by_default() -> None:
    service = RecordingService()

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
        "retrieval_method": "hybrid",
        "rerank": True,
    }


def test_excluding_a_source_records_the_reason_it_was_given() -> None:
    service = RecordingService()

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

    service = RecordingService()

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
    assert not [call for call in service.calls if call[0] == "set_source_inclusion"]


def test_a_decision_that_names_two_subjects_or_none_is_refused() -> None:
    """A command that dropped one of the two subjects would record a decision about a file
    when the reader asked about a passage, or the other way round, and neither reader
    would learn it from the answer.
    """

    service = RecordingService()

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


class _FakeGatewayClient:
    """A stand-in for the vanilla stdio client, counting how often it is opened."""

    opened: ClassVar[int] = 0
    calls: ClassVar[list[str]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).opened += 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def call_tool(self, name: str, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        type(self).calls.append(name)
        return {"tool": name}


def test_the_gateway_is_opened_by_the_first_call_and_then_reused(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeGatewayClient.opened = 0
    _FakeGatewayClient.calls = []
    monkeypatch.setattr(ultrarag_module, "Client", _FakeGatewayClient)
    monkeypatch.setattr(
        ultrarag_module, "create_vanilla_transport", lambda _config: object()
    )
    gateway = LazyGateway(_resolve(_args("--project-root", str(project), "status")))

    async def scenario() -> None:
        first = await gateway.call_tool("retriever_retriever_init", {})
        second = await gateway.call_tool("retriever_retriever_search", {})
        await gateway.aclose()

        assert first == {"tool": "retriever_retriever_init"}
        assert second == {"tool": "retriever_retriever_search"}

    asyncio.run(scenario())

    assert _FakeGatewayClient.opened == 1
    assert _FakeGatewayClient.calls == [
        "retriever_retriever_init",
        "retriever_retriever_search",
    ]


def test_a_reading_command_never_opens_the_gateway(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_config: Any) -> Any:
        raise AssertionError("a command that only reads opened the vanilla gateway")

    monkeypatch.setattr(ultrarag_module, "create_vanilla_transport", refuse)

    result = asyncio.run(_run(_args("--project-root", str(project), "sources")))

    assert result.payload is not None
    assert result.payload["source_count"] == 0


def test_the_doctor_reports_without_opening_the_gateway(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every dependency check reads, so the doctor never starts the gateway."""

    def refuse(_config: Any) -> Any:
        raise AssertionError("the doctor opened the vanilla gateway")

    monkeypatch.setattr(ultrarag_module, "create_vanilla_transport", refuse)
    write_pdf(project / "sources" / "evidence.pdf", ["The cobalt heron."])

    result = asyncio.run(_run(_args("--project-root", str(project), "doctor")))

    assert result.text is not None
    assert "project_identity" in result.text
    assert "vanilla_runtime" in result.text
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

    from research_rag.surfaces.cli import _invokes_this_app

    project = tmp_path / "research-rag"
    state = project / ".research-rag" / "runtime"

    assert not _invokes_this_app(
        ["--project-root", str(project), "--runtime-root", str(state)], project
    )
    assert _invokes_this_app(
        ["/srv/research-rag/.venv/bin/research-rag", "--project-root", str(project)],
        project,
    )
    assert _invokes_this_app(["python", "-m", "research_rag"], project)
    assert not _invokes_this_app(["python", "-m", "research_ultra_rag_mcp"], project)


def test_the_stop_sweep_leaves_the_other_products_processes_alone(
    tmp_path: Path,
) -> None:
    """The MCP server names this project root and its gateway names a directory under
    `.research-rag`, so a sweep that ended them would stop a server a reader started on
    purpose.
    """

    project = tmp_path / "ai-and-fetishism"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    server = "/srv/research-ultra-rag-mcp-server/.venv/bin"
    state = project / ".research-rag" / "runtime" / "ultrarag-runtime"
    _fake_process(
        proc_root,
        31,
        [
            "/srv/research-ultra-rag-mcp-server/.venv/bin/python",
            f"{server}/research-ultra-rag-mcp",
            "--project-root",
            str(project),
        ],
    )
    _fake_process(
        proc_root,
        32,
        [
            f"{server}/python",
            f"{server}/vanilla-ultra-rag-mcp",
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
            "serve",
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[int, int]] = []
    alive = {2}

    monkeypatch.setattr(cli_module, "STOP_GRACE_SECONDS", 0)
    monkeypatch.setattr(
        cli_module.os, "kill", lambda pid, number: sent.append((pid, number))
    )
    monkeypatch.setattr(cli_module, "_alive", lambda pid: pid in alive)

    forced = _terminate([1, 2])

    assert sent == [(1, signal.SIGTERM), (2, signal.SIGTERM), (2, signal.SIGKILL)]
    assert forced == [2]


def test_stop_without_servers_only_stops_the_app(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[list[str]] = []

    def record(command: list[str], **_options: Any) -> SimpleNamespace:
        recorded.append(command)
        return SimpleNamespace(
            returncode=0, stdout="No app is running for this project."
        )

    monkeypatch.setattr(cli_module, "subprocess", SimpleNamespace(run=record))
    args = _args("--project-root", str(project), "stop")

    report = _stop(args, _resolve(args))

    assert len(recorded) == 1
    assert recorded[0][0].endswith(".research-rag/bin/open-research-rag-ui.sh")
    assert recorded[0][1:] == ["--stop"]
    assert report["ui_launcher_status"] == 0
    assert report["ui_launcher_output"] == "No app is running for this project."
    assert "servers" not in report


def test_stop_with_servers_reports_what_it_found_and_stopped(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "subprocess",
        SimpleNamespace(
            run=lambda _command, **_options: SimpleNamespace(returncode=0, stdout="")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_service_processes",
        lambda _root, *_args: [(4242, "python -m research_rag")],
    )
    monkeypatch.setattr(cli_module, "_terminate", lambda _pids: [])
    args = _args("--project-root", str(project), "stop", "--servers")

    report = _stop(args, _resolve(args))

    assert report["servers"] == [{"pid": 4242, "command": "python -m research_rag"}]
    assert report["forced_pids"] == []
    assert len(report["notes"]) == 2


def test_set_overrides_reach_the_settings_by_resolving_in_this_process(
    project: Path,
) -> None:
    args = _args(
        "--project-root", str(project), "--set", "retrieval.rrf_k=30", "config"
    )

    config = _resolve(args)

    assert config.settings.rrf_k == 30
    assert config.settings_provenance["retrieval.rrf_k"] == "command line"


def test_config_prints_what_a_change_to_every_key_costs(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config` stays read-only, and it shows the cost before a file is edited."""

    args = _args("--project-root", str(project), "config")

    asyncio.run(cli_module._run(args))

    printed = capsys.readouterr().out
    assert "What a change to each key costs" in printed
    # A key the registry labels `identity` but no generation reads costs nothing.
    assert "retrieval.rerank_max_candidates" in printed
    assert "retrieval.rrf_k" in printed
    assert "dense.embedding_model" in printed
    assert not (project / ".research-rag" / "config.toml").exists()


def test_config_prints_the_description_of_every_key(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The terminal and the Config tab read one description, from the registry."""

    args = _args("--project-root", str(project), "config")

    asyncio.run(cli_module._run(args))

    # Wrapped to the terminal's width, so the whitespace is what a reader's eye drops.
    printed = " ".join(capsys.readouterr().out.split())
    for setting in SETTINGS:
        assert " ".join(setting.doc.split()) in printed


def _launched(project: Path, monkeypatch: pytest.MonkeyPatch, returncode: int = 0):
    recorded: list[list[str]] = []

    def record(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        recorded.append(command)
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(launcher_module.subprocess, "run", record)
    return recorded


def test_start_hands_over_to_the_project_launcher(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher owns the port choice, the lock, and the pid file."""

    recorded = _launched(project, monkeypatch)
    args = _args("--project-root", str(project), "start", "--port", "5099")
    config = _resolve(args)
    launcher_module.ensure_ui_launcher(
        project_root=config.project_root,
        portable_root=config.portable_root,
        state_root=config.state_root,
        project_name=config.project_name,
    )

    outcome = _start(args, config)

    assert outcome["running"] is True
    assert recorded[0][0].endswith(".research-rag/bin/open-research-rag-ui.sh")
    assert recorded[0][1:] == ["--port", "5099"]


def test_ui_starts_the_app_and_opens_a_browser(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _launched(project, monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr(cli_module, "_open_browser", opened.append)
    args = _args("--project-root", str(project), "ui")
    config = _resolve(args)
    launcher_module.ensure_ui_launcher(
        project_root=config.project_root,
        portable_root=config.portable_root,
        state_root=config.state_root,
        project_name=config.project_name,
    )

    _start(args, config)

    assert recorded[0][1:] == ["--open"]


def test_start_reports_a_launcher_that_fails(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start that failed names what failed and where the reason was written."""

    _launched(project, monkeypatch, returncode=3)
    args = _args("--project-root", str(project), "start")
    config = _resolve(args)
    launcher_module.ensure_ui_launcher(
        project_root=config.project_root,
        portable_root=config.portable_root,
        state_root=config.state_root,
        project_name=config.project_name,
    )

    with pytest.raises(ResearchError, match="did not start") as failure:
        _start(args, config)
    assert "research-rag-ui.log" in str(failure.value)


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
