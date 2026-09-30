"""`research-rag mcp`: a stdio front end to the running app.

An MCP client that speaks only stdio cannot open a socket, so this is the command
that connects one: it makes sure the app is up, then proxies stdio to the app's
agent endpoint on the app's own port. The proxy is a front end and nothing more —
the tools, the answer projection, the project, and the UltraRAG gateway are the
app's, so a stdio client and a browser cannot see two different states.

The bridge also names itself, which is what makes a disconnect legible: the app
lists clients by name, and dropping one ends its session, which ends the pipe and
therefore the client.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server import create_proxy

from .app import CLIENT_NAME_HEADER
from .config import ConfigurationError, ResearchConfig
from .control import ControlError, ensure_running
from .support import ResearchError
from .surfaces.mcp import AGENT_BRIDGE_NAME
from .version import APP_VERSION

# The env var a client configuration sets so a reader of the app's client list
# can tell which agent is which, rather than seeing "stdio-bridge" four times.
CLIENT_NAME_ENV = "RESEARCH_ULTRARAG_CLIENT_NAME"
DEFAULT_CLIENT_NAME = "stdio-bridge"


def client_name() -> str:
    """The name this bridge reports itself under, or a stated default."""

    return os.environ.get(CLIENT_NAME_ENV) or DEFAULT_CLIENT_NAME


def build_proxy(url: str, *, name: str) -> Any:
    """Return a stdio server that forwards everything to the app at `url`.

    Proxying rather than re-declaring the tools is the point: a second copy of the
    seven operations would be a second place for them to be wrong, and would give
    the agent a different answer than the workspace for the same question. The
    transport is built here rather than from a URL so the bridge can name itself,
    which is what makes it identifiable in the app's client list.
    """

    return create_proxy(
        StreamableHttpTransport(url, headers={CLIENT_NAME_HEADER: name}),
        name=AGENT_BRIDGE_NAME,
        version=APP_VERSION,
    )


def run(config: ResearchConfig, *, name: str | None = None) -> None:
    """Connect stdio to the app for this project, and stay up as long as it does."""

    try:
        with ensure_running(config) as control:
            url = f"{control.base_url}/mcp"
    except ControlError as exc:
        raise ResearchError(str(exc)) from exc
    build_proxy(url, name=name or client_name()).run(
        transport="stdio", show_banner=False
    )


def main(config: ResearchConfig, *, name: str | None = None) -> None:
    try:
        run(config, name=name)
    except (ConfigurationError, ResearchError) as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:  # pragma: no cover - a client closing the pipe
        pass
