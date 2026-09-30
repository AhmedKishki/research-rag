"""The browser workspace: the shared local interface over one research project.

The UI runs in this process and calls `ResearchService` directly. The MCP server
this app was seeded from served the same interface by starting a private stdio
copy of itself and forwarding each call across that boundary; the shared UI
already requested the complete payload, so nothing is lost by answering in
process and one process, one lock, and one generation replace a process tree.

`ResearchUIAdapter.call` is the whole boundary. The shared UI sends its full
optional argument set for every route, including fields a capability has turned
off, so each operation here names the arguments it accepts and drops the rest.
That is the same rule the tool-schema filter applied, and it keeps a control the
app does not serve from travelling as an argument the app ignores.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import uvicorn
from ui_ultra_rag_mcp import (
    AdapterFactory,
    SourceFile,
    UICapabilities,
    UIProfile,
    UIRequestError,
)
from ui_ultra_rag_mcp import create_ui_app as create_shared_ui_app

from .config import (
    ConfigurationError,
    ResearchConfig,
    resolve_source_reference,
)
from .service import ResearchService
from .sources import SourcePolicyError, scan_sources
from .ultrarag import LazyGateway, VanillaUltraRAG
from .version import version_label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from starlette.applications import Starlette

UI_NAME = "research-rag-ui"
MAX_ERROR_LENGTH = 1200
# Matches uvicorn's own default accept backlog, so a claimed socket is in the
# state uvicorn would have created for itself.
_CLAIM_BACKLOG = 2048
# The passed passages a browser request may ask for, and the neighbours it may
# read around a hit. The bounds are the search workflow's own: a browser is not
# a place where an unbounded budget belongs.
MAXIMUM_TOP_K = 50
MAXIMUM_CONTEXT_CHUNKS = 5

RESEARCH_UI_PROFILE = UIProfile(
    application_name="Research RAG",
    version_label=version_label(),
    project_label="Project",
    project_fallback_name="Research project",
    navigation_label="Research views",
    source_types_label="PDF + EPUB sources",
    ingest_intro=(
        "All included PDFs and EPUBs will be extracted and indexed. The current "
        "generation remains active unless the complete build succeeds."
    ),
    ingest_busy_message=(
        "Building BM25 and dense indexes. This can take several minutes…"
    ),
    # The shared UI's neutral labels apply here: the quote rule is stated once
    # in README.md, not on every passage a browser renders.
    capabilities=UICapabilities(
        metadata=True,
        force_recompute=True,
        source_selection=True,
        category_partitions=True,
        project_metadata=True,
        metadata_filters=True,
        bibliographic_filters=True,
        retrieval_modes=False,
        reranking=False,
        chunk_settings=False,
    ),
)

# The arguments each workspace operation accepts, by name. The shared UI sends
# its full optional set for every route, so an argument absent here is one this
# app does not serve and must not forward.
_SEARCH_ARGUMENTS = frozenset(
    {
        "query",
        "top_k",
        "categories_any",
        "projects_any",
        "keywords",
        "source_ids",
        "exclude_source_ids",
    }
)
_OPERATION_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "status": frozenset(),
    "ingest": frozenset({"force_recompute"}),
    "search": _SEARCH_ARGUMENTS,
    "list_sources": frozenset(),
    "get_passage": frozenset({"chunk_id", "context_chunks"}),
    "set_source_inclusion": frozenset({"source_path", "included", "reason"}),
    "set_source_metadata": frozenset({"source_path", "metadata"}),
}


def _string_list(value: Any) -> list[str] | None:
    """Return one optional list argument, or None when the request omitted it."""

    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise UIRequestError("A list argument must hold non-empty strings")
    return [item.strip() for item in value]


def _bounded_int(value: Any, *, default: int, maximum: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise UIRequestError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise UIRequestError(f"{name} must be between 1 and {maximum}")
    return value


class ResearchUIAdapter:
    """Answer the shared UI's workspace operations from one research service."""

    def __init__(self, config: ResearchConfig, service: ResearchService) -> None:
        self.config = config
        self.service = service

    async def health(self) -> Mapping[str, Any]:
        return {
            "project_root": str(self.config.project_root),
            "source_root": str(self.config.source_root),
        }

    def _arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return only the arguments this operation accepts."""

        accepted = _OPERATION_ARGUMENTS.get(operation)
        if accepted is None:
            raise UIRequestError(
                f"Research operation {operation!r} is not available",
                status_code=404,
            )
        return {key: value for key, value in arguments.items() if key in accepted}

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        accepted = self._arguments(operation, arguments)
        try:
            return await self._run(operation, accepted)
        except UIRequestError:
            raise
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise UIRequestError(message[:MAX_ERROR_LENGTH]) from exc

    async def _run(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if operation == "status":
            return await self.service.status()
        if operation == "list_sources":
            return await self.service.list_sources()
        if operation == "ingest":
            force_recompute = arguments.get("force_recompute", False)
            if not isinstance(force_recompute, bool):
                raise UIRequestError("force_recompute must be a boolean")
            return await self.service.ingest(force_recompute=force_recompute)
        if operation == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise UIRequestError("Search requires a non-empty query")
            return await self.service.search(
                query,
                top_k=_bounded_int(
                    arguments.get("top_k"),
                    default=10,
                    maximum=MAXIMUM_TOP_K,
                    name="top_k",
                ),
                categories_any=_string_list(arguments.get("categories_any")),
                projects_any=_string_list(arguments.get("projects_any")),
                keywords=_string_list(arguments.get("keywords")),
                source_ids=_string_list(arguments.get("source_ids")),
                exclude_source_ids=_string_list(arguments.get("exclude_source_ids")),
                # Hybrid retrieval with reranking is the only way this app
                # searches, and it is the way the measurements were taken.
                retrieval_method="hybrid",
                rerank=True,
                include_staleness=True,
            )
        if operation == "get_passage":
            chunk_id = arguments.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise UIRequestError("A passage request needs a chunk_id")
            return await self.service.get_passage(
                chunk_id,
                context_chunks=_bounded_int(
                    arguments.get("context_chunks"),
                    default=1,
                    maximum=MAXIMUM_CONTEXT_CHUNKS,
                    name="context_chunks",
                ),
            )
        if operation == "set_source_inclusion":
            included = arguments.get("included")
            if not isinstance(included, bool):
                raise UIRequestError("An inclusion request needs a boolean included")
            reason = arguments.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise UIRequestError("An exclusion reason must be a string")
            return await self.service.set_source_inclusion(
                source_path=_source_path(arguments),
                included=included,
                reason=reason,
            )
        if operation == "set_source_metadata":
            metadata = arguments.get("metadata")
            if not isinstance(metadata, Mapping):
                raise UIRequestError("A metadata request needs a metadata object")
            return await self.service.set_source_metadata(
                metadata=dict(metadata),
                source_path=_source_path(arguments),
            )
        raise UIRequestError(
            f"Research operation {operation!r} is not available",
            status_code=404,
        )

    async def source_file(self, source_path: str) -> SourceFile:
        try:
            target = resolve_source_reference(self.config, source_path)
            scan = scan_sources(self.config)
        except (ConfigurationError, SourcePolicyError, ValueError) as exc:
            raise UIRequestError(str(exc)) from exc
        selected = next((item for item in scan.selected if item.path == target), None)
        if selected is None:
            raise UIRequestError("Source was not found", status_code=404)
        media_type = (
            "application/pdf"
            if selected.extension == ".pdf"
            else "application/epub+zip"
        )
        disposition = "inline" if selected.extension == ".pdf" else "attachment"
        return SourceFile(
            path=selected.path,
            media_type=media_type,
            filename=selected.path.name,
            content_disposition_type=disposition,
        )


def _source_path(arguments: Mapping[str, Any]) -> str:
    value = arguments.get("source_path")
    if not isinstance(value, str) or not value.strip():
        raise UIRequestError("A source request needs a source_path")
    return value


def create_service(config: ResearchConfig) -> tuple[ResearchService, LazyGateway]:
    """Build the service and the gateway it opens on first use."""

    gateway = LazyGateway(config)
    return ResearchService(config, VanillaUltraRAG(gateway, config)), gateway


def _adapter_factory(config: ResearchConfig) -> AdapterFactory:
    @asynccontextmanager
    async def adapter_context() -> AsyncIterator[ResearchUIAdapter]:
        service, gateway = create_service(config)
        try:
            yield ResearchUIAdapter(config, service)
        finally:
            # The gateway is a child process this app started, and the workspace
            # is the only thing that knows when to stop it.
            await gateway.aclose()

    return adapter_context


def create_ui_app(
    config: ResearchConfig,
    *,
    service: ResearchService | None = None,
) -> Starlette:
    """Create the shared UI over one research project."""

    # More than one project can serve a workspace at the same time, so each one
    # names the project it serves instead of showing a generic label: a browser
    # window must be able to say which knowledge base it belongs to.
    profile = replace(
        RESEARCH_UI_PROFILE,
        project_fallback_name=config.project_name,
    )
    if service is not None:
        return create_shared_ui_app(
            profile=profile,
            adapter=ResearchUIAdapter(config, service),
        )
    return create_shared_ui_app(
        profile=profile,
        adapter_factory=_adapter_factory(config),
    )


UI_HOST = "127.0.0.1"


def _claim_loopback_port(host: str, port: int) -> socket.socket:
    """Bind and listen on the loopback port, and return the socket that holds it.

    The claim is the bind and the listen, not a probe followed by a bind, so two
    workspaces that start at the same time cannot both believe they hold the
    port: the loser gets an ``OSError`` here, before any server exists, and
    reports it. Listening is what makes the claim exclusive — a socket that is
    bound but not listening can still be bound again under ``SO_REUSEADDR``,
    which Linux uses to allow binding over a socket that is not accepting.
    ``SO_REUSEADDR`` is set anyway, matching what uvicorn sets for itself, so a
    port whose connections are still in ``TIME_WAIT`` after a stop is not
    mistaken for one another process holds.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(_CLAIM_BACKLOG)
    except OSError:
        sock.close()
        raise
    return sock


class Workspace:
    """The workspace served on one claimed loopback port.

    The host is fixed to loopback and no browser is opened here: the generated
    launcher owns choosing the port and opening a browser. This object exists so
    a caller embedding the workspace in its own process — a test, or a future
    desktop shell — can start and stop it without a launcher, and so a bind
    failure is reported with its reason instead of raised from a background task
    nobody is watching.

    The workspace shares this process's event loop, so every blocking operation
    it triggers must keep going through ``asyncio.to_thread`` the way the service
    already does; a synchronous call on the loop would stall the workspace it
    serves.

    The port is claimed by binding it before uvicorn is created, so a workspace
    that cannot claim its port says why instead of failing after a successful
    probe, and the claim is released with the socket.
    """

    def __init__(self, config: ResearchConfig, *, port: int) -> None:
        if not 1 <= port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        self.config = config
        self.host = UI_HOST
        self.port = port
        self.error: str | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ready(self) -> bool:
        """Whether the workspace is serving right now.

        uvicorn sets ``Server.started`` once and never clears it, so the live
        task is part of the test: after ``stop`` and after a bind failure the
        task is finished and the workspace is not serving.
        """

        return (
            self._server is not None
            and bool(self._server.started)
            and self._task is not None
            and not self._task.done()
        )

    async def start(self, *, service: ResearchService | None = None) -> None:
        """Serve the workspace for the life of this process."""

        try:
            claim = _claim_loopback_port(self.host, self.port)
        except OSError as exc:
            reason = exc.strerror or str(exc)
            self.error = (
                f"Port {self.port} is already in use on {self.host}, so the "
                f"workspace was not started; choose another --port ({reason})."
            )
            return
        self._socket = claim
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_ui_app(self.config, service=service),
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self._task = asyncio.create_task(self._server.serve(sockets=[claim]))
        self._task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        """Record a workspace that ended on its own, and release its claim.

        uvicorn can end the task by raising, which a probe-then-bind order hides
        behind an empty ``error``; the reason is kept here instead of showing a
        dead workspace with no explanation.
        """

        if not task.cancelled():
            failure = task.exception()
            if failure is not None:
                self.error = (
                    f"The workspace on {self.host}:{self.port} stopped: "
                    f"{failure.__class__.__name__}: {failure}"
                )
        self._release_claim()

    def _release_claim(self) -> None:
        """Close the claimed socket, which uvicorn also closes on shutdown."""

        claim, self._socket = self._socket, None
        if claim is not None:
            with contextlib.suppress(OSError):
                claim.close()

    async def wait(self) -> None:
        """Block until the serving task ends, whether it served or it failed."""

        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def stop(self) -> None:
        """Ask the workspace to stop and wait for its task to finish.

        A workspace that fails must not take this process down with it: whatever
        ended the task is recorded by ``_task_finished``, so it is not re-raised.
        """

        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            task, self._task = self._task, None
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._release_claim()
