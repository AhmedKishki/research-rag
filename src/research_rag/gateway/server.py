"""One stdio MCP gateway over UltraRAG's existing MCP servers."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.server.providers.proxy import ProxyClient, ProxyProvider

from .config import ConfigurationError, GatewayConfig, resolve_config
from .instructions import SERVER_INSTRUCTIONS
from .manifest import SERVER_SPECS, ServerSpec
from .runtime import RUNTIME_CACHE_ENV, RuntimeErrorBase, install_managed_runtime

SERVER_NAME = "vanilla-ultra-rag-mcp"
SERVER_VERSION = "0.1.3"


def _child_environment(config: GatewayConfig, namespace: str) -> dict[str, str]:
    env = dict(os.environ)
    source_root = str(config.ultrarag_root / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((source_root, existing_pythonpath))
        if existing_pythonpath
        else source_root
    )
    env["ULTRARAG_UI_STORAGE_ROOT"] = str(config.workspace_root / "ui-storage")
    env["ULTRARAG_LOG_TS"] = (
        f"vanilla_{namespace}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["log_level"] = config.log_level
    return env


def create_gateway(
    config: GatewayConfig,
    server_specs: tuple[ServerSpec, ...] = SERVER_SPECS,
) -> FastMCP[Any]:
    """Create the aggregate server without changing any upstream component."""
    transports: list[StdioTransport] = []

    @asynccontextmanager
    async def lifespan(_: FastMCP[Any]) -> AsyncIterator[dict[str, Any]]:
        try:
            yield {}
        finally:
            if transports:
                await asyncio.gather(
                    *(transport.close() for transport in transports),
                    return_exceptions=True,
                )

    gateway = FastMCP(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        lifespan=lifespan,
    )
    # Missing child components must fail discovery instead of silently exposing
    # an incomplete server.
    gateway.provider_error_strategy = "raise"

    for spec in server_specs:
        entrypoint = config.ultrarag_root / spec.relative_entrypoint
        transport = StdioTransport(
            command=str(config.python_executable),
            args=[str(entrypoint)],
            env=_child_environment(config, spec.namespace),
            cwd=str(config.workspace_root),
            keep_alive=True,
            log_file=config.workspace_root
            / "logs"
            / f"{spec.namespace}-child-stderr.log",
        )
        transports.append(transport)

        def client_factory(
            child_transport: StdioTransport = transport,
        ) -> ProxyClient:
            # Each short-lived proxy client shares one keep-alive transport, so
            # the upstream child process and its initialized state persist.
            return ProxyClient(child_transport)

        gateway.add_provider(
            ProxyProvider(client_factory),
            namespace=spec.namespace,
        )

    return gateway


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=SERVER_NAME,
        description=(
            "Expose an unmodified, pinned UltraRAG MCP surface through one "
            "namespaced stdio server."
        ),
    )
    parser.add_argument(
        "--namespace",
        action="append",
        choices=tuple(spec.namespace for spec in SERVER_SPECS),
        help=(
            "Expose only this UltraRAG namespace. Repeat for multiple namespaces. "
            "Omit to expose the complete vanilla surface."
        ),
    )
    parser.add_argument(
        "--ultrarag-root",
        default=os.environ.get("ULTRARAG_ROOT"),
        help=(
            "Optional pinned development checkout. If omitted, install or use "
            "the verified managed runtime snapshot."
        ),
    )
    parser.add_argument(
        "--runtime-cache-root",
        default=os.environ.get(RUNTIME_CACHE_ENV),
        help=f"Managed runtime cache root (default: ${RUNTIME_CACHE_ENV} or OS cache).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download a missing managed runtime.",
    )
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get("ULTRARAG_WORKSPACE_ROOT"),
        required=os.environ.get("ULTRARAG_WORKSPACE_ROOT") is None,
        help="External working directory for logs and vanilla runtime data.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python interpreter used for all upstream child servers.",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default="warn",
        help="UltraRAG child-process log level.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        ultrarag_root = args.ultrarag_root or install_managed_runtime(
            cache_root=args.runtime_cache_root,
            allow_download=not args.offline,
        )
        config = resolve_config(
            ultrarag_root=ultrarag_root,
            workspace_root=args.workspace_root,
            python_executable=args.python_executable,
            log_level=args.log_level,
        )
    except (ConfigurationError, RuntimeErrorBase) as exc:
        parser.error(str(exc))

    selected_namespaces = set(args.namespace or ())
    selected_specs = tuple(
        spec
        for spec in SERVER_SPECS
        if not selected_namespaces or spec.namespace in selected_namespaces
    )
    gateway = create_gateway(config, selected_specs)
    gateway.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
