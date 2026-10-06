"""The agent surface: eight tools and one resource over one project.

The tools reach the app's one `ResearchService` through a projection built for
this reader: every answer is lean, and the workspace and the command line want
the full one. The annotations and the descriptions below are the same contract
in the form a client reads before it calls anything, so they are kept to what a
caller decides on: which call, which identifier, which condition. Nothing here
starts a process, resolves a project, or parses a command line.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ..core.admission import as_caller
from ..core.blocked_answers import not_served_status, uninitialised_status
from ..core.review import DEFAULT_FIND_SOURCE_LIMIT
from ..core.tool_views import present_tool_response
from ..project.config import ResearchConfig
from ..project.instructions import AGENT_INSTRUCTIONS
from ..project.policy import ResearchError
from ..project.settings import LEAN_TOOL_DETAIL
from ..runtime.version import APP_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.service import ResearchService

SERVER_NAME = "research-rag"
# The path the app mounts this surface at, on the same loopback port as the
# browser workspace. It is a module constant because `app.py`, the stdio bridge,
# and a client configuration all have to name the same one.
MCP_PATH = "/mcp"
# The name the stdio bridge proxies this surface under, so a client that reads
# the server's own name sees the app rather than a transport in front of it.
AGENT_BRIDGE_NAME = "research-rag"
T = TypeVar("T")


SearchQuery: TypeAlias = Annotated[
    str,
    Field(
        description="Research question, exact phrase, name, or concept to retrieve.",
        min_length=1,
    ),
]
TopK: TypeAlias = Annotated[
    int,
    Field(
        description="Maximum number of ranked evidence passages to return (1-50).",
        ge=1,
        le=50,
    ),
]
KeywordFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive keywords; a result must contain every one. Omit or "
            "pass null for no keyword filter."
        )
    ),
]
SourceIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Stable source IDs to include; a result may match any one. Take them "
            "from find_source or from a search hit. An ID survives a change to the "
            "file's bytes and changes when the file is renamed or moved. Omit or "
            "pass null to search every source."
        )
    ),
]
ExcludeSourceIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Stable source IDs to exclude; a result may not match any one. Omit "
            "or pass null to exclude nothing. A reviewed exclusion always applies "
            "and cannot be undone here."
        )
    ),
]
CategoriesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive categories; a result's source must carry at least "
            "one. They come from reviewed source metadata, and a category no "
            "source carries returns no passage. Omit or pass null for no filter."
        )
    ),
]
ProjectsAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Reviewed project tags; a result's source must carry at least one. "
            "Omit or pass null for no filter."
        )
    ),
]
LanguagesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive ISO 639 codes; a result must be written in at least "
            "one. A source carries the language detected while extracting it, or "
            "the one a review set. Omit or pass null for no filter."
        )
    ),
]
AuthorsAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive authors; a result's source must carry at least one "
            "supplied name inside one of its author strings, so a surname finds "
            "its author. Reviewed authors win over extracted ones, and "
            "find_source reports the authors each source carries. Combine with "
            "`titles_any` to pin one work. Omit or pass null for no filter."
        )
    ),
]
TitlesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive title fragments; a result's source title must "
            "contain at least one, so a remembered fragment finds the work "
            "without its subtitle. Reviewed titles win over extracted ones, and "
            "find_source reports the title each source carries. A fragment no "
            "source carries returns no passages and says so in `applied_filters`. "
            "Omit or pass null for no filter."
        )
    ),
]
ForceRecompute: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Set true to bypass document, chunk, and vector reuse and rebuild all "
            "derived content. The previous generation remains selected on failure."
        )
    ),
]
ChunkId: TypeAlias = Annotated[
    str,
    Field(
        description=(
            "Exact chunk_id a search hit or get_passage answer returned for the "
            "current generation."
        ),
        min_length=1,
    ),
]
SourcePath: TypeAlias = Annotated[
    str,
    Field(
        description=(
            "PDF or EPUB path relative to the configured sources directory. Use "
            "the `source_relative_path` find_source reports for it; absolute and "
            "escaping paths are rejected."
        ),
        min_length=1,
    ),
]
SourceQuery: TypeAlias = Annotated[
    str,
    Field(
        description=(
            "A filename, a title, or an author to look up, matched "
            "case-insensitively against every source this project can name, "
            "including one whose file is gone or whose metadata was reviewed."
        ),
        min_length=1,
    ),
]
FindLimit: TypeAlias = Annotated[
    int,
    Field(
        description="Maximum number of matching sources to return (1-50).",
        ge=1,
        le=50,
    ),
]
InclusionFlag: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Set false to exclude the source from retrieval and future ingestion; "
            "set true to restore it. The original file is never changed."
        )
    ),
]
ChunkInclusionFlag: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Set false to exclude this passage from retrieval and every later one; "
            "set true to restore it. The original file is never changed."
        )
    ),
]
ExclusionReason: TypeAlias = Annotated[
    str | None,
    Field(
        description=(
            "Human-readable reason for the decision, such as identifying another "
            "file as the preferred copy. Required when included is false; omit or "
            "pass null when restoring."
        )
    ),
]
# A mapping rather than a model, because the review is the file a person edits and
# its fields are validated by the same code that reads that file. The schema
# carries the field names and their types, so a client can build the mapping
# without reading prose to know what a value may be.
MetadataReview: TypeAlias = Annotated[
    dict[str, Any],
    Field(
        description=(
            "The whole reviewed bibliography for this source: `title` and `doi` are "
            "strings, `year` is an integer or null, and `authors`, `categories`, "
            "`keywords`, `project`, and `language` are lists of strings, with "
            "`language` holding ISO 639-1 codes, one per language the source is "
            "written in. Any other field is refused. The call replaces this "
            "source's whole review, and an empty mapping clears it so automatic "
            "metadata applies again."
        )
    ),
]


class NotServed(ResearchError):
    """The app is not serving this project, with the answer `status` gives for it.

    Raised by the stand-in service the bridge uses while no app answers. Every
    tool reports it as the error it is, with the command that fixes it, and
    `status` returns the structured answer instead.
    """

    def __init__(self, reason: str, payload: dict[str, Any]) -> None:
        remedy = next(
            (
                str(entry["remedy"])
                for entry in payload.get("blocked_by", [])
                if entry.get("remedy")
            ),
            "",
        )
        super().__init__(
            f"{reason} Run `{remedy}` in a terminal, then call this tool again; "
            "this entry needs no change."
            if remedy
            else reason
        )
        self.payload = payload


async def _tool_call(operation: Callable[[], Awaitable[T]]) -> T:
    try:
        return await operation()
    except ResearchError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(f"Research workflow failed: {exc}") from exc


def _agent_app_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the part of the app's state an agent can act on.

    The app's state is the app's, and it grows with what the app chooses to
    record: how many clients are attached, which port it claimed, which URL its
    own agent surface is at. None of that is the project's, and none of it is
    something an agent can do anything with, so it is not merged into the
    project's verdict. The workspace's address and the condition that says the
    workspace is not being served are the two facts a client hands to a person,
    so they travel and the rest does not.

    A field added to the app's state is not added here by accident: an agent's
    answer carries what this function names.
    """

    bounded: dict[str, Any] = {}
    url = state.get("ui_url")
    if url:
        bounded["ui_url"] = url
    if state.get("ui_error"):
        bounded["ui_error"] = state["ui_error"]
    return bounded


def create_blocked_mcp(
    project_name: str, reason: str, *, config: ResearchConfig | None
) -> FastMCP[Any]:
    """The agent surface for a project this machine cannot serve right now.

    An agent's client entry names a project, not a directory, so the same entry
    can be written on one machine and used on another. It names a project this
    machine has never initialised, or one whose app is not running: in both cases
    there is no corpus to answer about, so this server answers and stops.

    Nothing is started to make it whole. An app belongs to the terminal that
    started it, so this surface's job is to say which command a reader runs in a
    terminal, and `status` is declared here rather than by the bridge so the tools
    an agent sees are declared in one file. The other seven operations are absent
    because there is no corpus behind them.
    """

    # A resolved config is the whole difference between the two conditions: it
    # means the project is on disk and only its app is missing.
    payload = (
        not_served_status(config, reason)
        if config is not None
        else uninitialised_status(project_name, reason)
    )
    condition = (
        "the project exists but no app is serving it"
        if config is not None
        else "no project has been initialised under that name on this machine"
    )
    app = FastMCP(
        name=SERVER_NAME,
        version=APP_VERSION,
        instructions=(
            f"This entry names the project {project_name!r}, and {condition}, so "
            f"this server offers `status` and nothing else. {reason} Nothing else "
            "here can succeed until that command has been run in a terminal."
        ),
    )

    def _present(operation: str, answer: dict[str, Any]) -> dict[str, Any]:
        return present_tool_response(operation, answer, detail=LEAN_TOOL_DETAIL)

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def status() -> dict[str, Any]:
        """Report why this project cannot be served, and the command that serves it.

        `blocked_by` carries the reason and the exact command. Run it and this
        entry serves the project it names, with no change to the entry.
        """

        return _present("status", payload)

    @app.resource(
        "research://status",
        name="current generation status",
        description=(
            "Why this project cannot be served, and the command that serves it."
        ),
        mime_type="application/json",
    )
    async def status_resource() -> str:
        return json.dumps(await status(), ensure_ascii=False, indent=2)

    return app


def create_unserved_mcp(config: ResearchConfig, reason: str) -> FastMCP[Any]:
    """The same eight tools, answering for a project no app is serving.

    The bridge uses it while the app is down, so an agent that connected before
    the app started still holds the whole tool list and finds each call working
    once the app is up. The tools are the ones `create_mcp` declares, so the two
    cannot differ: only the service behind them does.
    """

    payload = not_served_status(config, reason)

    async def connect() -> ResearchService:
        raise NotServed(reason, payload)

    return create_mcp(config, connect=connect, app_state=dict)


def create_mcp(
    config: ResearchConfig,
    *,
    connect: Callable[[], Awaitable[ResearchService]],
    app_state: Callable[[], Mapping[str, Any]],
    caller: Callable[[], str] | None = None,
) -> FastMCP[Any]:
    """Build the agent surface over the app's one service.

    `caller` names who the request in hand came from, so the service can take its
    waiting callers in rounds; it is empty when the surface cannot tell.

    `connect` opens the gateway on the first call, so a client that only lists
    tools never starts a process. A client waits for the handshake before it calls
    anything, so the handshake must not depend on unasked-for work.
    """

    app = FastMCP(
        name=SERVER_NAME,
        version=APP_VERSION,
        instructions=AGENT_INSTRUCTIONS,
    )

    async def service() -> ResearchService:
        return await connect()

    async def _service_call(
        operation: Callable[[ResearchService], Awaitable[T]],
    ) -> T:
        """Run one service operation, connecting the gateway if it is not up yet.

        Connection failures become workflow failures, so a gateway that cannot
        start is reported by the tool the user called.
        """

        async def run() -> T:
            return await operation(await service())

        with as_caller(caller() if caller is not None else ""):
            return await _tool_call(run)

    def _present(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return present_tool_response(
                operation,
                payload,
                detail=config.tool_detail,
            )
        except ResearchError as exc:
            raise ToolError(str(exc)) from exc

    async def _status_payload() -> dict[str, Any]:
        """Return the `status` answer, plus the app's own state, bounded.

        The workspace and the attached clients are the app's state rather than the
        project's, and a client about to open a browser needs to know where it is
        served and whether it is being served at all. Those two facts are the
        whole of it here: the app decides what else its state carries, so nothing
        is merged into an agent's answer wholesale.
        """

        try:
            payload = _present(
                "status", await _service_call(lambda instance: instance.status())
            )
        except ToolError as error:
            # No app is answering: the status of that is the answer, not an error.
            if isinstance(error.__cause__, NotServed):
                return _present("status", error.__cause__.payload)
            raise
        state = app_state()
        return {**payload, **_agent_app_state(state)}

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def status() -> dict[str, Any]:
        """Check readiness and freshness before searching.

        Inspect `requires`, `blocked_by`, and `degraded` for next steps.
        `ingest` rebuilds; `restart_app` requires a process restart, not a tool call.
        `ui_url` opens this project's browser workspace.
        """

        return await _status_payload()

    @app.resource(
        "research://status",
        name="current generation status",
        description=(
            "Readiness, freshness, and the selected generation for this project."
        ),
        mime_type="application/json",
    )
    async def status_resource() -> str:
        return json.dumps(await _status_payload(), ensure_ascii=False, indent=2)

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        }
    )
    async def ingest(
        force_recompute: ForceRecompute = False,
    ) -> dict[str, Any]:
        """Build or refresh the index, reusing compatible work.

        Obtain the user's agreement: this writes persistent state and may download
        models. Repeat identical calls while `status` is `in_progress`; finish on
        `ready` or `unchanged`. Report discarded or withheld evidence. Concurrent
        builds are refused with the resident build's phase.
        """

        return _present(
            "ingest",
            await _service_call(
                lambda instance: instance.ingest(force_recompute=force_recompute)
            ),
        )

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def search(
        query: SearchQuery,
        top_k: TopK = 10,
        categories_any: CategoriesAnyFilter = None,
        projects_any: ProjectsAnyFilter = None,
        keywords: KeywordFilter = None,
        languages_any: LanguagesAnyFilter = None,
        authors_any: AuthorsAnyFilter = None,
        titles_any: TitlesAnyFilter = None,
        source_ids: SourceIdFilter = None,
        exclude_source_ids: ExcludeSourceIdFilter = None,
    ) -> dict[str, Any]:
        """Find evidence for one research question, with follow-up IDs and source locators.

        Rephrase or raise `top_k` before concluding the corpus has no answer.
        Inspect `applied_filters`, `search_window_partial`, `relevance_limited`,
        `rerank_fallback`, `stale`, and `generation_upgrade_required` when present.
        Returned text is cleaned, not a verified quotation.
        """

        return _present(
            "search",
            await _service_call(
                lambda instance: instance.search(
                    query,
                    top_k=top_k,
                    categories_any=categories_any,
                    projects_any=projects_any,
                    keywords=keywords,
                    languages_any=languages_any,
                    authors_any=authors_any,
                    titles_any=titles_any,
                    source_ids=source_ids,
                    exclude_source_ids=exclude_source_ids,
                    retrieval_method="hybrid",
                    rerank=True,
                    include_staleness=True,
                )
            ),
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def find_source(
        query: SourceQuery,
        limit: FindLimit = DEFAULT_FIND_SOURCE_LIMIT,
    ) -> dict[str, Any]:
        """Locate a named work by filename, title, or author; never list the corpus.

        Use returned `source_id` to filter searches and `source_relative_path` for
        metadata or inclusion reviews. Check `indexed_in_current_generation`;
        unindexed sources need ingestion. Raise `limit` if truncated. This lookup
        may update the project-local source catalog.
        """
        return _present(
            "find_source",
            await _service_call(
                lambda instance: instance.find_source(query=query, limit=limit)
            ),
        )

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def get_passage(
        chunk_id: ChunkId,
    ) -> dict[str, Any]:
        """Read a current search hit with its immediate neighbours.

        Use a `chunk_id` from the selected generation. Excluded targets are refused;
        excluded neighbours carry `excluded_from_search: true` and are context,
        not eligible search evidence. Verify quotations in the original at the locator.
        """

        return _present(
            "get_passage",
            await _service_call(lambda instance: instance.get_passage(chunk_id)),
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_source_inclusion(
        included: InclusionFlag,
        source_path: SourcePath,
        reason: ExclusionReason = None,
    ) -> dict[str, Any]:
        """Exclude a source from retrieval, or restore one, without touching the file.

        Record a decision only after you or the user has reviewed the source, and
        give an exclusion its reason. The decision binds current retrieval at
        once and the next ingestion. Re-ingest to drop an excluded source
        physically.
        """

        return _present(
            "set_source_inclusion",
            await _service_call(
                lambda instance: instance.set_source_inclusion(
                    source_path=source_path,
                    included=included,
                    reason=reason,
                )
            ),
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_chunk_inclusion(
        included: ChunkInclusionFlag,
        chunk_id: ChunkId,
        reason: ExclusionReason = None,
    ) -> dict[str, Any]:
        """Exclude one passage from retrieval, or restore it, without touching the file.

        Record a decision only after you or the user has reviewed the passage,
        and give an exclusion its reason. The decision is enforced by the
        retrieval filter at once and in every later generation, so no ingestion
        is needed to keep it and a rebuild leaves the passage in the index. A
        `chunk_id` is derived from content rather than permanent, so the answer
        reports whether this generation still holds the passage the decision was
        made about.
        """

        return _present(
            "set_chunk_inclusion",
            await _service_call(
                lambda instance: instance.set_chunk_inclusion(
                    chunk_id=chunk_id,
                    included=included,
                    reason=reason,
                )
            ),
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_source_metadata(
        source_path: SourcePath,
        metadata: MetadataReview,
    ) -> dict[str, Any]:
        """Save reviewed bibliographic metadata for one source.

        The review binds retrieval at read time, without re-ingesting, and the
        person may edit the same review file by hand instead. Save only what a
        source or its original states: an invented title, author, year, or DOI
        misleads every later answer.
        """

        return _present(
            "set_source_metadata",
            await _service_call(
                lambda instance: instance.set_source_metadata(
                    metadata=metadata,
                    source_path=source_path,
                )
            ),
        )

    return app
