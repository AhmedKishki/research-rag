"""Stable adapter contract between the shared UI and an MCP implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class UICapabilities:
    """Optional actions supported by one adapter, document and memory alike.

    ``documents`` stays on for an adapter that serves a corpus and turns off for
    one that serves something else — a memory-only server shows its own views
    instead of a search console it cannot answer.
    """

    documents: bool = True
    sources: bool = True
    passage_context: bool = True
    ingestion: bool = True
    metadata: bool = True
    source_inclusion: bool = True
    source_files: bool = True
    metadata_filters: bool = True
    bibliographic_filters: bool = False
    project_metadata: bool = False
    source_selection: bool = False
    category_partitions: bool = False
    retrieval_modes: bool = True
    reranking: bool = True
    chunk_settings: bool = True
    force_recompute: bool = False
    bundle_export: bool = False
    bundle_import: bool = False
    memory: bool = False
    memory_writes: bool = False
    # Whether this adapter can list and end the MCP clients attached to the
    # process serving it. Off by default: the workspace cannot know whether the
    # host is a server at all, and a memory-only server has no corpus clients.
    clients: bool = False
    # Whether this adapter can list the generations it retains and delete one.
    # The listing arrives in the status payload and is rendered for free; only
    # the removal needs a route, so the flag gates a panel that would otherwise
    # be a set of chips and a button that cannot act.
    generations: bool = False
    # Whether this adapter can run a statement against its own stored records.
    # Off by default: the workspace cannot know which store an adapter keeps,
    # and a panel that could change records must not appear beside one that only
    # reads.
    sql_console: bool = False
    # Whether this adapter can report its own settings and accept a change to
    # one. Off by default: the settings, their values, their origins, and the
    # cost of changing them are the server's facts, and a panel of invented
    # fields would ask a reader to edit something the server does not have.
    settings: bool = False
    # Whether this adapter can list the chunks it has excluded from retrieval
    # and exclude or restore one. Off by default, because a whole-source
    # exclusion already exists and a chunk-level control beside it is only
    # meaningful where the server can honour it.
    chunk_exclusion: bool = False
    # Whether this adapter can list the projects this installation serves and
    # whether an app is up for each. Off by default: the workspace cannot know
    # whether its host serves one project or several, and a selector with no
    # second project to switch to is a control that cannot act.
    projects: bool = False
    # Whether this adapter can produce this project's MCP client entry as text.
    # Off by default: the entry names the host's own command and its project,
    # and a generator here would be a second place for a client's configuration
    # to be wrong.
    agent_entry: bool = False

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UIProfile:
    """Visible labels and capabilities for one MCP adapter."""

    application_name: str
    project_label: str = "Project"
    project_fallback_name: str = "Knowledge base"
    # How this host starts one project's workspace, with `{project}` where the
    # project's name goes. It is the host's own command line, so it is the
    # host's to supply, and empty removes the block: a project with no app has
    # no address, and a command the workspace invented would be one the
    # reader could paste and fail on.
    project_start_command: str = ""
    navigation_label: str = "Knowledge base views"
    source_types_label: str = "document sources"
    ingest_intro: str = (
        "Included documents will be extracted and indexed. The current generation "
        "remains active unless the complete build succeeds."
    )
    ingest_busy_message: str = "Building the indexes. This can take several minutes…"
    footer_text: str = ""
    result_text_label: str = "Retrieved passage"
    copy_text_label: str = "Copy passage"
    bundle_import_intro: str = (
        "Place the archive in the server's project-local bundle directory, then "
        "enter its filename."
    )
    bundle_export_warning: str = (
        "Export this generation? The archive may contain complete original sources."
    )
    memory_label: str = "Memory"
    memory_standing_label: str = "Standing memory"
    memory_rounds_label: str = "Recorded rounds"
    memory_add_label: str = "Add a round"
    memory_note: str = ""
    capabilities: UICapabilities = UICapabilities()
    version_label: str = ""

    def __post_init__(self) -> None:
        if not self.application_name.strip():
            raise ValueError("application_name must not be empty")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capabilities"] = self.capabilities.as_dict()
        return value


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A source file that an adapter has already authorized for browser access."""

    path: Path
    media_type: str
    filename: str
    content_disposition_type: str = "inline"


class UIRequestError(Exception):
    """A safe adapter error that may be returned to the local browser."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        if not 400 <= status_code <= 599:
            raise ValueError("status_code must be between 400 and 599")
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class UIAdapter(Protocol):
    """Normalize one MCP server to the shared document-workspace operations."""

    async def health(self) -> Mapping[str, Any]: ...

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    async def source_file(self, source_path: str) -> SourceFile: ...


class ClientControl(Protocol):
    """The attached MCP clients of the process serving this workspace.

    Separate from `UIAdapter` because a host may serve a workspace and still not
    be a server: a stdio-only process has no other client to report. A route that
    needs this checks for it at run time rather than making every adapter carry
    two methods it cannot answer.
    """

    async def list_clients(self) -> Sequence[Mapping[str, Any]]: ...

    async def disconnect_client(
        self, session_id: str, reason: str | None = None
    ) -> Mapping[str, Any]: ...


class SqlConsole(Protocol):
    """Reading and editing an adapter's own stored records.

    Separate from `UIAdapter` because a statement is not a document operation:
    the workspace names a scope the adapter reported and forwards a statement
    unchanged, so a route that needs this checks for it at run time rather than
    making every adapter carry two methods it cannot answer. Nothing here names
    a store, resolves a path, or decides what a statement may say — the adapter
    runs the statement against its own storage and refuses whatever it will not
    do, and the refusal reaches the browser with its own status and reason.
    """

    async def sql_query(self, scope: str, statement: str) -> Mapping[str, Any]: ...

    async def sql_execute(self, scope: str, statement: str) -> Mapping[str, Any]: ...


AdapterContext: TypeAlias = AbstractAsyncContextManager[UIAdapter]
AdapterFactory: TypeAlias = Callable[[], AdapterContext]
