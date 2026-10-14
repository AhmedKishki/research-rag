"""The adapter contract between the workspace host and this app's service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class UICapabilities:
    """Optional actions supported by one adapter."""

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
    force_recompute: bool = False
    # Whether this adapter can list and end the MCP clients attached to the
    # process serving it. Off by default: the workspace cannot know whether the
    # host is a server at all.
    clients: bool = False
    # Whether this adapter can list the generations it retains and delete one.
    # The listing arrives in the status payload and is rendered for free; only
    # the removal needs a route, so the flag gates a panel that would otherwise
    # be a set of chips and a button that cannot act.
    generations: bool = False
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
    # Release checking only; installation and approval remain in the terminal.
    updates: bool = False
    # Whether this adapter counts what its searches returned and can report
    # those counts beside facts about its corpus. Off by default: the workspace
    # cannot know whether a host records anything about its searches.
    stats: bool = False

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
    # How a client of this host is given a name of its own, in the host's own
    # words. Empty removes the note: only a host that serves MCP clients over a
    # bridge has a variable for one to set, and a shared page cannot name it.
    client_naming_hint: str = ""
    source_types_label: str = "document sources"
    ingest_intro: str = (
        "Included documents will be extracted and indexed. The current generation "
        "remains active unless the complete build succeeds."
    )
    ingest_busy_message: str = "Building the indexes. This can take several minutes…"
    ingest_resume_note: str = ""
    footer_text: str = ""
    result_text_label: str = "Retrieved passage"
    copy_text_label: str = "Copy passage"
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
    """Normalize this app's service to the workspace operations."""

    async def health(self) -> Mapping[str, Any]: ...

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    async def source_file(self, source_path: str) -> SourceFile: ...


AdapterContext: TypeAlias = AbstractAsyncContextManager[UIAdapter]
AdapterFactory: TypeAlias = Callable[[], AdapterContext]
