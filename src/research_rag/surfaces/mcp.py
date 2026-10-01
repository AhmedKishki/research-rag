"""The agent surface: eight tools and one resource over one project.

The tools reach the app's one `ResearchService` through a projection built for
this reader: every answer is lean, and the workspace and the command line want
the full one. Nothing here starts a process, resolves a project, or parses a
command line.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ..config import ResearchConfig
from ..instructions import AGENT_INSTRUCTIONS
from ..review import DEFAULT_FIND_SOURCE_LIMIT
from ..settings import LEAN_TOOL_DETAIL
from ..status import uninitialised_status
from ..tool_views import present_tool_response
from ..version import APP_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..service import ResearchService

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
            "Case-insensitive keyword filters; a result must contain every supplied "
            "keyword. Omit or pass null for no keyword filter."
        )
    ),
]
SourceIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Stable source IDs to include; a result may match any supplied ID. "
            "Obtain IDs from find_source. A source ID survives a change to the "
            "file's bytes and changes when the file is renamed or moved. Omit or "
            "pass null to search every source."
        )
    ),
]
ExcludeSourceIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Stable source IDs to exclude; a result may not match any supplied ID. "
            "Omit or pass null to exclude nothing. Reviewed source exclusions "
            "always apply and cannot be undone here."
        )
    ),
]
CategoriesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' category filters; a result must carry at "
            "least one supplied category. Categories come from reviewed source "
            "metadata, which set_source_metadata writes, and a category no source "
            "carries returns no passage. Omit or pass null for no filter."
        )
    ),
]
ProjectsAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Reviewed project tags to match with 'any of' semantics; a result must "
            "carry at least one supplied project. Omit or pass null for no filter."
        )
    ),
]
LanguagesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' language filters; a result must be written "
            "in at least one supplied ISO 639 code. A source carries the language "
            "detected while extracting it and the language a review set instead. "
            "Omit or pass null for no filter."
        )
    ),
]
AuthorsAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' author filters; a result's source must have "
            "at least one supplied name inside one of its author strings, so a "
            "surname finds its author. Reviewed authors win where a review exists "
            "and extracted ones apply otherwise; find_source reports the authors "
            "each source carries. Combine with `titles_any` to pin one work. Omit "
            "or pass null for no filter."
        )
    ),
]
TitlesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' title filters; a result's source title must "
            "contain at least one supplied phrase, so a remembered fragment finds "
            "the work without reproducing its subtitle. Reviewed titles win over "
            "extracted ones, and find_source reports the title each source carries. "
            "A filter that matches no source returns no passages and says so in "
            "`applied_filters` rather than searching everything. Omit or pass null "
            "for no filter."
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
        description="Exact chunk_id returned by search for the current generation.",
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
            "A filename, a title, or an author to look up. Matched "
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


async def _tool_call(operation: Callable[[], Awaitable[T]]) -> T:
    from ..support import ResearchError

    try:
        return await operation()
    except ResearchError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(f"Research workflow failed: {exc}") from exc


def create_uninitialised_mcp(project_name: str, reason: str) -> FastMCP[Any]:
    """The agent surface for a project this machine has not initialised.

    An agent's client entry names a project, not a directory, so the same entry
    can be written on one machine and used on another. That only works where a
    project exists first: where it does not, this server answers and stops.
    `status` is declared here rather than by the bridge, so the tools an agent
    sees are declared in one file and the other seven operations are absent
    because there is no corpus behind them.
    """

    payload = uninitialised_status(project_name, reason)
    app = FastMCP(
        name=SERVER_NAME,
        version=APP_VERSION,
        instructions=(
            f"This entry names the project {project_name!r}, and no project has "
            f"been initialised under that name on this machine, so this server "
            f"offers `status` and nothing else. {reason} Nothing else here can "
            "succeed until that command has been run."
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
        """Report that this project is not initialised on this machine, and the
        command that creates it.

        `blocked_by` carries the reason and the exact command. Run it and this
        entry serves the project it names, with no change to the entry.
        """

        return _present("status", payload)

    @app.resource(
        "research://status",
        name="current generation status",
        description=(
            "Why this project cannot be served, and the command that initialises it."
        ),
        mime_type="application/json",
    )
    async def status_resource() -> str:
        return json.dumps(await status(), ensure_ascii=False, indent=2)

    return app


def create_mcp(
    config: ResearchConfig,
    *,
    connect: Callable[[], Awaitable[ResearchService]],
    app_state: Callable[[], Mapping[str, Any]],
) -> FastMCP[Any]:
    """Build the agent surface over the app's one service.

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

        return await _tool_call(run)

    def _present(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        from ..support import ResearchError

        try:
            return present_tool_response(
                operation,
                payload,
                detail=config.tool_detail,
            )
        except ResearchError as exc:
            raise ToolError(str(exc)) from exc

    async def _status_payload() -> dict[str, Any]:
        """Return the `status` answer, plus the app's own state.

        The workspace and the attached clients are the app's state, not the
        project's, and a client about to open a browser needs to know that.
        """

        payload = _present(
            "status", await _service_call(lambda instance: instance.status())
        )
        return {**payload, **app_state()}

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def status() -> dict[str, Any]:
        """Report whether this project can be searched, and what must happen first.

        Call this before telling the user their corpus is up to date.
        `ready: false` or `stale: true` means the corpus is not what the user has.
        `requires` names the calls that close the gap: `ingest` rebuilds the
        generation, `restart_app` restarts this process so it runs the installed
        code. It is absent when nothing is required.

        `blocked_by` and `degraded` name what stands between this project and a
        search that answers, each with the reason and the command that fixes it.
        A tool error naming a gateway log holds the same information, and the log
        is where to read the rest. `ui_url` is where this project's browser
        workspace is served. `mcp_clients` counts the agents attached to this app.
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
        """Build or refresh the generation that search reads, reusing compatible work.

        Writes persistent state and may download a model, so get the user's
        agreement first. A long build answers status=in_progress: call it again
        until it returns ready or unchanged, then report what changed. One call
        covers at most `ingestion.work_budget_seconds` of work, so a larger build
        needs repeated identical calls, and the project's own config is where the
        budget is raised. A rejected call means another process holds the project,
        and its message names that build. Progress that goes backwards is reported
        as `superseded_build`: a changed corpus cannot resume the old one.
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
        """Retrieve evidence passages for a research question.

        Each passage gives its source filename, its authors, its position, and
        cleaned text. Quote only from the original at that locator. Ask for more
        passages before concluding the corpus has nothing: the reranker reorders
        about twice as many candidates as top_k, so a low top_k hides candidates
        from it. A question asked in other words is a different search.
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
        """Look up one source by filename, title, or author, and say if it is searchable.

        Call it before naming a source in set_source_inclusion or
        set_source_metadata, and when a user asks whether a work is in the
        project. It returns the matches with the `source_relative_path` and
        `source_id` those two operations take, and `indexed_in_current_generation`
        says whether a search can reach the source: a source that is only on disk
        or only reviewed needs `ingest` first.

        It is a lookup, not a listing: ask about a name rather than for the corpus.
        A source whose file is gone or whose metadata was reviewed is still
        answerable. A match count above the returned rows means `limit` hid some.
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
        """Return one passage with its immediate neighbors on each side.

        Use it to read around a hit. The text is still cleaned for retrieval, so
        quote from the original at the passage's locator. A passage a reader has
        excluded is refused here and still returned as a neighbour, because
        context is not evidence.
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

        Act only after the agent or user has reviewed the source, and give an
        exclusion its reason. The decision binds current retrieval at once and the
        next ingestion. Re-ingest to drop an excluded source physically.
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

        Act only after the agent or user has reviewed the passage, and give an
        exclusion its reason. The decision is enforced by the retrieval filter at
        once and in every later generation, so no ingestion is needed to keep it
        and a rebuild leaves the passage in the index. A `chunk_id` is derived
        from content rather than permanent, so the answer reports whether this
        generation still holds the passage the decision was made about.
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
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Save reviewed bibliographic metadata for one source.

        Fields: title, authors, year, doi, language, categories, keywords, and
        project. `language` takes ISO 639 codes, one per language the source is
        written in, and a code BM25 has no stopword list for is accepted. An empty
        review clears the entry, so automatic metadata applies again. The review
        binds retrieval at read time, without re-ingesting.
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
