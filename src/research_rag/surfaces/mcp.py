"""The agent surface: seven operations and two resources over one project.

This module is the whole of what an agent sees. It builds a `FastMCP` instance
whose tools and resources call the app's one `ResearchService`, and it owns the
lean/full answer projection, because that projection exists for this reader and
for no other: an agent's answer is 10.8 kB lean against 19.8 kB full on the
reference corpus, and the workspace and the command line both want the full one.

The app process owns the service, the gateway, and the port, and mounts this on
one loopback port beside the browser workspace. Nothing here starts a process,
resolves a project, or parses a command line.
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
        description=("Maximum number of ranked evidence passages to return (1-50)."),
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
            "Obtain IDs from list_sources. A source ID survives a change to the "
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
            "Case-insensitive 'any of' category filters; a result must contain at "
            "least one supplied category. Use it to search a set of corpus "
            "partitions in one call, and combine it with `categories` to require "
            "all of one set and any of another. Categories come from reviewed "
            "source metadata and `status.categories` lists the current inventory. "
            "Omit or pass null for no filter."
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
            "detected while extracting it and the language a review set instead, "
            "and `status.languages` lists the current inventory. Use it to search "
            "one language of a mixed corpus. Omit or pass null for no filter."
        )
    ),
]
AuthorsAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' author filters; a result's source must have "
            "at least one supplied name inside one of its author strings, so a "
            "surname finds its author without the bibliography's punctuation. Read "
            "reviewed authors where a review exists and extracted ones otherwise; "
            "`list_sources` reports the authors each source carries. Combine with "
            "`titles_any` to pin one work. Omit or pass null for no filter."
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
            "extracted ones, and `list_sources` reports the title each source "
            "carries. A filter that matches no source returns no passages and says "
            "so in `applied_filters` rather than searching everything. Omit or pass "
            "null for no filter."
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
            "list_sources.source_relative_path; absolute and escaping paths are "
            "rejected."
        ),
        min_length=1,
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
ExclusionReason: TypeAlias = Annotated[
    str | None,
    Field(
        description=(
            "Human-readable reason for the decision, such as identifying another "
            "file as the preferred copy. Required when included is false; omit or "
            "pass null when restoring a source."
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


def create_mcp(
    config: ResearchConfig,
    *,
    connect: Callable[[], Awaitable[ResearchService]],
    app_state: Callable[[], Mapping[str, Any]],
) -> FastMCP[Any]:
    """Build the agent surface over the app's one service.

    `connect` returns the running app's service, opening the gateway on the
    first call, so a client that only lists tools never starts a process. A
    client waits for the handshake before it will call anything, so the
    handshake must not depend on work no tool has asked for.
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

        Connection failures are translated like any other workflow failure, so a
        gateway that cannot start is reported by the tool the user called rather
        than by a server that never answered its handshake.
        """

        async def run() -> T:
            return await operation(await service())

        return await _tool_call(run)

    def _present(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a tool answer in this surface's configured detail mode."""

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

        The workspace and the attached clients are the app's state rather than
        the project's, and a client that is about to start a browser needs to
        know whether there is one to open.
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
        """Report readiness, the selected generation, and whether it is current.

        Call this first, and before telling the user their corpus is up to date.
        No generation means nothing can be searched yet; `stale` and
        `generation_upgrade_required` say what moved and whether to ingest again.

        `blocked_by` and `degraded` name what stands between this project and a
        search that answers, each with the reason and the command that fixes it.
        They are absent when there is nothing to act on. A tool error that names a
        gateway log holds the same information: read the log before retrying.
        `ui_url` is where the browser workspace for this project is served, and
        `mcp_clients` counts the agents attached to this app.
        """
        return await _status_payload()

    @app.resource(
        "research://status",
        name="current generation status",
        description=(
            "Readiness, freshness, and the selected generation for this project: "
            "the same answer the status tool gives."
        ),
        mime_type="application/json",
    )
    async def status_resource() -> str:
        return json.dumps(await _status_payload(), ensure_ascii=False, indent=2)

    @app.resource(
        "research://sources",
        name="source inventory",
        description=(
            "Every discovered source with its review and index state: the same "
            "answer the list_sources tool gives, and the inventory to read before "
            "naming a source in set_source_inclusion or set_source_metadata."
        ),
        mime_type="application/json",
    )
    async def sources_resource() -> str:
        payload = _present(
            "list_sources",
            await _service_call(lambda instance: instance.list_sources()),
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

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
        covers at most `ingestion.work_budget_seconds` of work, so a build larger
        than that budget needs repeated identical calls; a client that stops
        repeating them cannot finish it, and the project's own config is where the
        budget is raised. A rejected call means another process holds the project
        and its message names that build; progress that goes backwards is reported
        as `superseded_build`, because a changed corpus cannot resume the old one.
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
        passages before concluding that the corpus has nothing: the reranker
        reorders about twice as many candidates as top_k, so a low top_k hides
        candidates from it, and a question asked in different words is a
        different search rather than a narrower one.
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
    async def list_sources() -> dict[str, Any]:
        """List the project's sources: filenames, inclusion, and index state.

        The corpus inventory, and the answer also works before any ingestion.
        Use the filename it reports to name a source in set_source_inclusion.
        """

        return _present(
            "list_sources",
            await _service_call(lambda instance: instance.list_sources()),
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
        quote from the original at the passage's locator.
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
        exclusion its reason. The decision binds the current retrieval at once
        and the next ingestion; re-ingest to drop an excluded source physically.
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
        written in, and a code BM25 has no stopword list for is accepted because
        the metadata describes the source rather than the index. An empty review
        clears the entry so automatic metadata applies again. The review is
        authoritative at read time, so it binds current retrieval without
        re-ingesting, and the same JSON file may be edited by hand.
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
