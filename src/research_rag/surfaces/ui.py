"""Research workspace labels, capabilities, and operation argument boundaries."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .workspace import SourceFile, UICapabilities, UIProfile, UIRequestError
from .workspace import create_ui_app as create_shared_ui_app

if TYPE_CHECKING:  # pragma: no cover - typing only
    from starlette.applications import Starlette

from ..core.admission import as_caller
from ..core.service import ResearchService
from ..corpus.sources import SourcePolicyError, scan_sources
from ..project.config import (
    ConfigurationError,
    ResearchConfig,
    resolve_source_reference,
)
from ..project.registry import account_projects
from ..runtime.app import CLIENT_NAME_ENV, ClientError, ClientRegistry
from ..runtime.doctor import mcp_entry_block
from ..runtime.version import version_label

UI_NAME = "research-rag-ui"
MAX_ERROR_LENGTH = 1200
# The passed passages a browser request may ask for, and the neighbours it may
# read around a hit. The bounds are the search workflow's own: a browser is not
# a place for an unbounded budget.
MAXIMUM_TOP_K = 50
MAXIMUM_CONTEXT_CHUNKS = 5

RESEARCH_UI_PROFILE = UIProfile(
    application_name="Research RAG",
    version_label=version_label(),
    project_label="Project",
    project_fallback_name="Research project",
    navigation_label="Research views",
    source_types_label="PDF + EPUB sources",
    # A client entry carries a command and a project, so the app cannot tell one
    # agent from another and the reader is told how to name it instead. The
    # variable is the bridge's, so this sentence is written where it is declared.
    client_naming_hint=f"Set {CLIENT_NAME_ENV} in the client entry to name an agent.",
    ingest_intro=(
        "Build indexes from included PDFs and EPUBs. The selected generation "
        "is replaced only after a complete build succeeds."
    ),
    ingest_busy_message=("Building indexes. Large collections may take a while…"),
    ingest_resume_note="If a build runs out of time, run it again to resume from its checkpoint.",
    result_text_label="Cleaned passage",
    copy_text_label="Copy cleaned text",
    footer_text="Passages are cleaned text, not verified quotations. Open the original for exact wording.",
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
        # The app is a server, so a person in the workspace can see which agents
        # are attached to it and end one. A host that is not a server leaves this
        # off and the panel and its routes stay absent.
        clients=True,
        # A build leaves its predecessor on disk, so a reader in the workspace
        # can see what those builds cost and reclaim one. The listing is already
        # in the status payload. The removal asks for the id twice, which the
        # shared workspace insists on before it calls here.
        generations=True,
        # A reader tunes this project in the browser, and the cost of a change is
        # named before anything is written, because a generation records most of
        # these keys and a search would otherwise answer stale with no reason.
        settings=True,
        # A reader who objects to one passage rather than to a whole file records
        # that beside the source decision it is an alternative to, and the panel
        # lists what has been decided so the decision can be reversed.
        chunk_exclusion=True,
        # One installation serves several projects, so the workspace says which
        # one it is and can reach another. The listing is the account's own
        # record, read through the same code `research-rag projects` uses.
        projects=True,
        # A reader configuring an agent in the browser is handed the same entry
        # `research-rag doctor --mcp-entry` prints, rather than a second copy of
        # it written for a browser.
        agent_entry=True,
        updates=True,
        # Every search counts its first five ranks on this machine, without the
        # query, so the workspace can show which sources and passages answer.
        stats=True,
    ),
)

# The arguments each workspace operation accepts, by name. The shared UI sends
# its full optional set for every route, so an argument absent here is one this
# app does not serve and must not forward.
#
# All six filter layers the service supports are here. A layer the app serves
# and the workspace drops is a filter a reader can type into the command line
# and not into the browser, the disagreement this file exists to prevent. The
# set is the service's own; the profile below decides what the browser offers.
_SEARCH_ARGUMENTS = frozenset(
    {
        "query",
        "top_k",
        "categories_any",
        "projects_any",
        "keywords",
        "languages_any",
        "authors_any",
        "titles_any",
        "source_ids",
        "exclude_source_ids",
    }
)
# The filters the shared workspace's Sources view offers, which this app serves
# with the same predicate `search` uses.
_SOURCE_LISTING_FILTERS = frozenset(
    {"categories", "categories_any", "projects", "projects_any", "keywords"}
)

_OPERATION_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "status": frozenset(),
    "ingest": frozenset({"force_recompute"}),
    "search": _SEARCH_ARGUMENTS,
    "list_sources": _SOURCE_LISTING_FILTERS,
    "get_passage": frozenset({"chunk_id", "context_chunks"}),
    "set_source_inclusion": frozenset({"source_path", "included", "reason"}),
    "set_chunk_inclusion": frozenset({"chunk_id", "included", "reason"}),
    "list_chunk_exclusions": frozenset(),
    "set_source_metadata": frozenset({"source_path", "metadata"}),
    "remove_generation": frozenset({"generation_id", "confirm"}),
    "settings_read": frozenset(),
    "settings_write": frozenset({"values", "expected_revision", "confirm"}),
    "list_projects": frozenset(),
    "agent_entry": frozenset(),
    "check_updates": frozenset(),
    "search_stats": frozenset({"days", "top"}),
    "search_history": frozenset({"limit", "days"}),
    "clear_search_history": frozenset(),
    "source_chunks": frozenset({"source_id", "source_path", "page", "page_size"}),
    "open_source": frozenset({"source_path"}),
}


# Operations that take no argument and share the service method's name.
_PLAIN_READS = frozenset({"list_sources", "list_chunk_exclusions"})
# The operations about what searches returned and what a source holds.
_COUNT_OPERATIONS = frozenset(
    {
        "search_stats",
        "search_history",
        "clear_search_history",
        "source_chunks",
        "open_source",
    }
)


def _project_listing(config: ResearchConfig) -> dict[str, Any]:
    """The account's projects, projected as the workspace's selector reads them.

    The listing is the one `research-rag projects` prints, read through the same
    function, so the projects a browser names and the projects a terminal names
    cannot differ. Each entry carries the address its app is served on, or the
    absence of one: a project with no app has no URL, and a workspace that
    invented one would offer a reader a link that fails.

    `current` is this project's own recorded name, so the page marks the project
    it is serving rather than the first one in the list.
    """

    account = account_projects()
    entries = [
        {
            "project_name": entry.get("project_name"),
            "project_root": entry.get("project_root"),
            "running": bool((entry.get("app") or {}).get("running")),
            "url": (entry.get("app") or {}).get("url"),
            "attached_clients": entry.get("attached_clients", 0),
        }
        for entry in account.get("projects", [])
    ]
    return {
        "projects": entries,
        "current": config.project_name,
        "message": account.get("message") or "",
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


def _bounded_int(
    value: Any,
    *,
    default: int,
    maximum: int,
    name: str,
    minimum: int = 1,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise UIRequestError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise UIRequestError(f"{name} must be between {minimum} and {maximum}")
    return value


class ResearchUIAdapter:
    """Answer the shared UI's workspace operations from one research service.

    The service is the app's, so the workspace, the agent, and the command line
    are three readers of one state. The client methods read the same registry
    the agent's `status` counts and the command line lists: a person in the
    browser and a person in a terminal see the same agents and drop the same
    sessions.
    """

    def __init__(
        self,
        config: ResearchConfig,
        service: ResearchService,
        *,
        clients: ClientRegistry | None = None,
        app_state: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.service = service
        self.clients = clients
        self.app_state = app_state
        self._update_lock = asyncio.Lock()
        self._update_preview: dict[str, Any] | None = None
        self._update_checked_at = 0.0

    async def health(self) -> Mapping[str, Any]:
        return {
            "project_root": str(self.config.project_root),
            "source_root": str(self.config.source_root),
        }

    def _require_clients(self) -> ClientRegistry:
        """The registry, or a refusal when this adapter was built without one.

        Naming that beats reporting an empty list that reads as "no agents are
        attached".
        """

        if self.clients is None:
            raise UIRequestError(
                "This workspace is not served by a running app, so it has no "
                "clients to report.",
                status_code=501,
            )
        return self.clients

    async def list_clients(self) -> Sequence[Mapping[str, Any]]:
        return self._require_clients().report()

    async def disconnect_client(
        self, session_id: str, reason: str | None = None
    ) -> Mapping[str, Any]:
        registry = self._require_clients()
        try:
            client = registry.disconnect(
                session_id, reason or "Disconnected from the workspace."
            )
        except ClientError as exc:
            raise UIRequestError(str(exc)) from exc
        return client.report()

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
            with as_caller("workspace"):
                return await self._run(operation, accepted)
        except UIRequestError:
            raise
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise UIRequestError(message[:MAX_ERROR_LENGTH]) from exc

    async def _check_updates(self) -> Mapping[str, Any]:
        """Cache a read-only release preview; installation remains a CLI transaction."""

        from ..runtime import release, update

        async with self._update_lock:
            if (
                self._update_preview is not None
                and time.monotonic() - self._update_checked_at < 300
            ):
                return self._update_preview
            preview = await asyncio.to_thread(
                update.read_preview, offline=self.config.offline
            )
            installed = str(preview["install"]["version"])
            published = preview["published"]
            found = published.get("state") == release.FOUND
            available = bool(
                found
                and release.is_release_version(installed)
                and release.compare_versions(str(published["version"]), installed) > 0
            )
            public_release = (
                {
                    "version": published["version"],
                    "tag_name": published["tag"],
                    "name": published.get("name", published["tag"]),
                    "published_at": published["published"],
                    "html_url": published["url"],
                    "body": published.get("body", ""),
                }
                if found
                else None
            )
            self._update_preview = {
                "installed_version": installed,
                "update_available": available,
                "offline": self.config.offline,
                "release": public_release,
                "apply_command": "research-rag update --apply",
                "message": (
                    preview.get("blocked")
                    if found
                    else published.get("detail", "Release information is unavailable.")
                ),
            }
            self._update_checked_at = time.monotonic()
            return self._update_preview

    async def _run_counts(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """The operations about what searches returned and what a source holds."""

        if operation == "search_stats":
            return await self.service.search_stats(
                since_days=_days(arguments.get("days")),
                top=_bounded_int(
                    arguments.get("top"), default=20, maximum=100, name="top"
                ),
            )
        if operation == "search_history":
            return await self.service.search_history(
                limit=_bounded_int(
                    arguments.get("limit"), default=20, maximum=200, name="limit"
                ),
                since_days=_days(arguments.get("days")),
            )
        if operation == "clear_search_history":
            return await self.service.clear_search_history()
        if operation == "source_chunks":
            return await self.service.source_chunks(
                source_id=_optional_text(arguments.get("source_id")),
                source_path=_optional_text(arguments.get("source_path")),
                page=_bounded_int(
                    arguments.get("page"), default=1, maximum=1_000_000, name="page"
                ),
                page_size=_bounded_int(
                    arguments.get("page_size"), default=20, maximum=50, name="page_size"
                ),
            )
        if operation == "open_source":
            return await self._open_source(_source_path(arguments))
        raise UIRequestError(f"Research operation {operation!r} is not available", 404)

    async def _run(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if operation == "check_updates":
            return await self._check_updates()
        if operation in _COUNT_OPERATIONS:
            return await self._run_counts(operation, arguments)
        # The reads that take no argument, answered as the service words them.
        if operation in _PLAIN_READS:
            return await getattr(self.service, operation)()
        if operation == "status":
            payload = dict(await self.service.status())
            if self.app_state is not None:
                # The app's own state, which is what the control API's status and
                # the agent's resource already carry. The workspace reads the
                # same answer rather than a status that names the port beside it
                # differently.
                payload.update(self.app_state())
            return payload
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
                languages_any=_string_list(arguments.get("languages_any")),
                authors_any=_string_list(arguments.get("authors_any")),
                titles_any=_string_list(arguments.get("titles_any")),
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
                    minimum=0,
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
        if operation == "set_chunk_inclusion":
            return await self._set_chunk_inclusion(arguments)
        if operation == "set_source_metadata":
            metadata = arguments.get("metadata")
            if not isinstance(metadata, Mapping):
                raise UIRequestError("A metadata request needs a metadata object")
            return await self.service.set_source_metadata(
                metadata=dict(metadata),
                source_path=_source_path(arguments),
            )
        if operation == "remove_generation":
            generation_id = arguments.get("generation_id")
            if not isinstance(generation_id, str) or not generation_id:
                raise UIRequestError(
                    "A generation removal needs a generation_id string"
                )
            # The confirmation is carried, not defaulted. The shared workspace
            # already refuses a mismatch before it gets here. Forwarding anything
            # but the id the reader typed is how a browser would come to mean
            # "yes" on a click.
            confirm = arguments.get("confirm")
            if not isinstance(confirm, str):
                raise UIRequestError("A generation removal needs a confirm string")
            return await self.service.remove_generation(generation_id, confirm=confirm)
        if operation == "settings_read":
            return await self.service.settings_read()
        if operation == "settings_write":
            return await self._settings_write(arguments)
        if operation == "list_projects":
            # Probing each project's own app is blocking loopback I/O, and a
            # project whose app is slow to answer must not hold up the page that
            # asked about it.
            return await asyncio.to_thread(_project_listing, self.config)
        if operation == "agent_entry":
            # The generator `doctor --mcp-entry` prints, so a client's
            # configuration is the same text whichever surface produced it.
            return {"entry": mcp_entry_block(self.config)}
        raise UIRequestError(
            f"Research operation {operation!r} is not available",
            status_code=404,
        )

    async def _settings_write(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Carry one settings change to the service, with its three fields checked.

        The revision is required rather than defaulted: a write that named no
        revision could land on top of a change made since the page loaded, and the
        refusal for a revision that was not text belongs beside the field.
        """

        values = arguments.get("values")
        if not isinstance(values, Mapping) or not values:
            raise UIRequestError(
                "A settings write needs a values object naming what to change"
            )
        expected_revision = arguments.get("expected_revision")
        if not isinstance(expected_revision, str) or not expected_revision.strip():
            raise UIRequestError(
                "A settings write needs the expected_revision it read, so a "
                "change made since is refused instead of overwritten"
            )
        confirm = arguments.get("confirm", False)
        if not isinstance(confirm, bool):
            raise UIRequestError("confirm must be a boolean")
        return await self.service.settings_write(
            dict(values),
            expected_revision=expected_revision,
            confirm=confirm,
        )

    async def _set_chunk_inclusion(
        self, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Carry one passage decision to the service, with its three arguments checked.

        The reasons are validated here rather than in the service because the
        refusal belongs beside the field it is about.
        """

        included = arguments.get("included")
        if not isinstance(included, bool):
            raise UIRequestError("A chunk decision needs a boolean included")
        chunk_id = arguments.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise UIRequestError("A chunk decision needs a chunk_id")
        reason = arguments.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise UIRequestError("A chunk exclusion reason must be a string")
        return await self.service.set_chunk_inclusion(
            chunk_id,
            included=included,
            reason=reason,
        )

    async def _open_source(self, source_path: str) -> dict[str, Any]:
        """Ask the desktop to open an authorised source in its own viewer.

        The same check as serving the file decides which path is allowed, so the
        desktop is only ever handed a source this project selected. A machine
        with no desktop answers 501 and the page shows the file in the browser.
        """

        from ..runtime.viewer import ViewerUnavailable, open_in_default_viewer

        source = await self.source_file(source_path)
        try:
            viewer = await asyncio.to_thread(open_in_default_viewer, source.path)
        except ViewerUnavailable as exc:
            raise UIRequestError(str(exc), status_code=501) from exc
        return {"opened": True, "viewer": viewer, "filename": source.filename}

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


def _days(value: Any) -> float | None:
    """A window of days for a count, or None for every search."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise UIRequestError("days must be a positive number")
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise UIRequestError("A source is named by a string")
    return value.strip() or None


def _source_path(arguments: Mapping[str, Any]) -> str:
    value = arguments.get("source_path")
    if not isinstance(value, str) or not value.strip():
        raise UIRequestError("A source request needs a source_path")
    return value


def create_ui_app(
    config: ResearchConfig,
    *,
    service: ResearchService,
    clients: ClientRegistry | None = None,
    app_state: Callable[[], Mapping[str, Any]] | None = None,
) -> Starlette:
    """Create the shared workspace over the app's one service.

    The adapter is handed the service rather than a factory, because the app
    process owns the gateway and there is exactly one of it. The client registry
    is passed in for the same reason, and is absent when no process serves the
    workspace. `app_state` is absent the same way, and the workspace's status
    carries it beside the service's.
    """

    # More than one project can be served at the same time, so each one names the
    # project it serves instead of showing a generic label. A browser window must
    # be able to say which knowledge base it belongs to.
    profile = replace(RESEARCH_UI_PROFILE, project_fallback_name=config.project_name)
    return create_shared_ui_app(
        profile=profile,
        adapter=ResearchUIAdapter(
            config, service, clients=clients, app_state=app_state
        ),
    )
