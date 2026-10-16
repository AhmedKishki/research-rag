"""The workspace adapter: the shared operations, and the few only the workspace serves."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .workspace import SourceFile, UIRequestError
from .workspace import create_ui_app as create_workspace_app

if TYPE_CHECKING:  # pragma: no cover - typing only
    from starlette.applications import Starlette

from ..core.admission import as_caller
from ..core.operations import OPERATIONS, run
from ..core.service import ResearchService
from ..core.version import version_label
from ..corpus.sources import SourcePolicyError, scan_sources
from ..project.config import (
    ConfigurationError,
    ResearchConfig,
    resolve_source_reference,
)
from ..project.policy import ResearchError
from ..runtime.app import ClientError, ClientRegistry
from ..runtime.doctor import mcp_entry_block

LOGGER = logging.getLogger(__name__)
MAX_ERROR_LENGTH = 1200
# Fields the page sends with a search that the search does not read. They were
# always dropped here rather than refused, so a typed value changes nothing.
_PAGE_ONLY_SEARCH_FIELDS = ("categories", "projects", "document_ids")


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

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            with as_caller("workspace"):
                return await self._run(operation, dict(arguments))
        except UIRequestError:
            raise
        except (
            ResearchError,
            ClientError,
            ConfigurationError,
            SourcePolicyError,
        ) as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise UIRequestError(message[:MAX_ERROR_LENGTH]) from exc
        except Exception as exc:
            LOGGER.exception("research-rag workspace operation %s failed", operation)
            message = f"Unexpected app failure: {exc}. The serving terminal shows why."
            raise UIRequestError(message[:MAX_ERROR_LENGTH], status_code=500) from exc

    async def _run(
        self, operation: str, arguments: dict[str, Any]
    ) -> Mapping[str, Any]:
        if operation == "search":
            for field in _PAGE_ONLY_SEARCH_FIELDS:
                arguments.pop(field, None)
        if operation in OPERATIONS:
            payload = await run(self.service, operation, arguments)
            if operation == "status" and self.app_state is not None:
                # The app's own state, which the control API's status and the
                # agent's resource already carry.
                payload.update(self.app_state())
            return payload
        if operation == "check_updates":
            return await self._check_updates()
        if operation == "agent_entry":
            # The generator `doctor --mcp-entry` prints, so a client's
            # configuration is the same text whichever surface produced it.
            return {"entry": mcp_entry_block(self.config)}
        if operation == "open_source":
            source_path = arguments.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise UIRequestError("A source request needs a source_path")
            return await self._open_source(source_path)
        raise UIRequestError(
            f"Research operation {operation!r} is not available", status_code=404
        )

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
            scan = await asyncio.to_thread(scan_sources, self.config)
        except (ConfigurationError, SourcePolicyError, ValueError) as exc:
            raise UIRequestError(str(exc)) from exc
        selected = next((item for item in scan.selected if item.path == target), None)
        if selected is None:
            raise UIRequestError("Source was not found", status_code=404)
        media_type = {
            ".pdf": "application/pdf",
            ".epub": "application/epub+zip",
            ".mobi": "application/x-mobipocket-ebook",
        }[selected.extension]
        disposition = "inline" if selected.extension == ".pdf" else "attachment"
        return SourceFile(
            path=selected.path,
            media_type=media_type,
            filename=selected.path.name,
            content_disposition_type=disposition,
        )


def create_ui_app(
    config: ResearchConfig,
    *,
    service: ResearchService,
    clients: ClientRegistry | None = None,
    app_state: Callable[[], Mapping[str, Any]] | None = None,
    network_access: Any = None,
) -> Starlette:
    """Create the workspace over the app's one service.

    The adapter is handed the service rather than a factory, because the app
    process owns the retrieval backend and there is exactly one of it. The client
    registry is passed in for the same reason, and is absent when no process serves the
    workspace. `app_state` is absent the same way, and the workspace's status
    carries it beside the service's.
    """

    return create_workspace_app(
        adapter=ResearchUIAdapter(
            config, service, clients=clients, app_state=app_state
        ),
        version_label=version_label(),
        network_access=network_access,
    )
