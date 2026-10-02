"""Persistent MCP connection to the pinned vanilla UltraRAG gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from ..project.config import ResearchConfig, child_process_environment

# How long one vanilla call may take, and how much of a failed component's
# stderr is worth carrying back to the caller.
TRANSPORT_TIMEOUT_SECONDS = 1800
GATEWAY_LOG_TAIL_LINES = 20

# How long the handshake may take before a gateway is treated as unable to
# start. This is not the call budget: a handshake is the process coming up and
# answering `initialize`, so an ingestion-length timeout only ever applies to a
# gateway that has already failed. When a gateway exits mid-handshake the
# pinned client can leave the connection awaiting a response that will never
# arrive, and only this bound ends the wait — without it the failure a caller
# is told about in seconds is discovered half an hour later.
GATEWAY_INIT_TIMEOUT_SECONDS = 30


def create_vanilla_transport(config: ResearchConfig) -> StdioTransport:
    """Start the pinned gateway as a managed child of this server.

    The gateway inherits no top-level UI setting, so a variable exported for this
    server's own UI cannot travel down the process tree.
    """

    arguments = [
        "--workspace-root",
        str(config.ultrarag_workspace),
        "--log-level",
        config.log_level,
        "--namespace",
        "corpus",
        "--namespace",
        "retriever",
    ]
    if config.runtime_cache_root is not None:
        arguments.extend(["--runtime-cache-root", str(config.runtime_cache_root)])
    if config.offline:
        arguments.append("--offline")

    return StdioTransport(
        command=str(config.vanilla_executable),
        args=arguments,
        env=child_process_environment(),
        cwd=str(config.project_root),
        keep_alive=True,
        log_file=config.logs_root / "vanilla-gateway-stderr.log",
    )


def gateway_log_path(config: ResearchConfig) -> Path:
    return config.logs_root / "vanilla-gateway-stderr.log"


def child_logs_root(config: ResearchConfig) -> Path:
    return config.ultrarag_workspace / "logs"


def child_log_path(config: ResearchConfig, namespace: str) -> Path:
    return child_logs_root(config) / f"{namespace}-child-stderr.log"


def _log_tail(path: Path, *, lines: int = GATEWAY_LOG_TAIL_LINES) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line.rstrip() for line in content.splitlines() if line.strip()][-lines:]


def gateway_start_failure(config: ResearchConfig, exc: BaseException) -> ToolError:
    """Return the error a gateway that cannot start must answer with.

    A gateway that exits during its handshake leaves the reason in the log the
    transport already writes, and a caller who never opens that file has only the
    client's own "connection closed". This carries the reason, the last lines of
    every log the run produced, and the path of each one, so a tool answer names
    the cause rather than restating the symptom.
    """

    gateway_log = gateway_log_path(config)
    logs_root = child_logs_root(config)
    lines = [
        f"The UltraRAG gateway could not start: {exc}",
        f"Gateway stderr: {gateway_log}",
    ]
    tail = _log_tail(gateway_log)
    if tail:
        lines.append("Last lines:")
        lines.extend(f"  {line}" for line in tail)
    else:
        lines.append("That log is empty or absent.")
    written: Sequence[Path] = sorted(logs_root.glob("*-child-stderr.log"))
    lines.append(f"Component stderr: {logs_root}")
    for path in written:
        lines.append(f"  {path}")
        lines.extend(f"    {line}" for line in _log_tail(path))
    return ToolError("\n".join(lines))


def call_timeout_failure(
    config: ResearchConfig,
    name: str,
    exc: BaseException,
) -> ToolError:
    """Return the error a vanilla call that never answered must raise.

    The vanilla tools are namespaced, so the tool name says which component was
    busy, and that component writes its own stderr beside the gateway's. Naming
    both is enough to see what the component was doing. No process tree is
    inspected to work it out.
    """

    namespace = name.split("_", 1)[0]
    return ToolError(
        f"The {namespace} component did not answer {name} within "
        f"{TRANSPORT_TIMEOUT_SECONDS} seconds: {exc}. Its stderr is at "
        f"{child_log_path(config, namespace)} and the gateway's is at "
        f"{gateway_log_path(config)}."
    )


def _is_timeout(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "timed out" in message or "timeout" in message


@asynccontextmanager
async def vanilla_client(config: ResearchConfig) -> AsyncIterator[Client[Any]]:
    client: Client[Any] = Client(
        create_vanilla_transport(config),
        timeout=TRANSPORT_TIMEOUT_SECONDS,
        init_timeout=GATEWAY_INIT_TIMEOUT_SECONDS,
    )
    try:
        await client.__aenter__()
    except Exception as exc:
        raise gateway_start_failure(config, exc) from exc
    try:
        yield client
    finally:
        await client.__aexit__(None, None, None)


class LazyGateway:
    """The vanilla gateway a process opens when it first needs one, and not before.

    Which operations need the gateway is not a property of the surface: the
    BM25 index is initialized through it when a generation is loaded for
    querying, so a search needs one and a `status` usually does not. Predicting
    that would start a process nothing may call, so the connection opens on the
    first call and is closed with the process that owns this object.

    The workspace and the terminal both hold one of these for their whole run, so
    the gateway is a child of the process a user stopped rather than an orphan.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._client: Any = None

    async def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._client is None:
            stack = AsyncExitStack()
            self._client = await stack.enter_async_context(vanilla_client(self._config))
            self._stack = stack
        return await self._client.call_tool(name, *args, **kwargs)

    async def aclose(self) -> None:
        """Close the gateway if anything opened one, and never otherwise."""

        if self._stack is not None:
            await self._stack.aclose()


class VanillaUltraRAG:
    def __init__(
        self, client: Client[Any], config: ResearchConfig | None = None
    ) -> None:
        self.client = client
        self.config = config

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            result = await self.client.call_tool(
                name,
                arguments,
                timeout=TRANSPORT_TIMEOUT_SECONDS,
                raise_on_error=True,
            )
        except Exception as exc:
            if self.config is not None and _is_timeout(exc):
                raise call_timeout_failure(self.config, name, exc) from exc
            raise
        return result.data

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        await self.call(
            "corpus_chunk_documents",
            {
                "raw_chunk_path": str(input_path),
                "chunk_backend_configs": {"token": {"chunk_overlap": chunk_overlap}},
                "chunk_backend": "token",
                "tokenizer_or_token_counter": "gpt2",
                "chunk_size": chunk_size,
                "chunk_path": str(output_path),
                "use_title": False,
            },
        )

    async def initialize_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None:
        await self.call(
            "retriever_retriever_init",
            {
                "model_name_or_path": "",
                "backend_configs": {
                    "bm25": {
                        "lang": language,
                        "tokenizer": "default",
                        "save_path": str(index_path),
                    }
                },
                "batch_size": 32,
                "corpus_path": str(chunks_path),
                "gpu_ids": None,
                "is_multimodal": False,
                "backend": "bm25",
                # Present in the unified upstream schema but unused by BM25.
                "index_backend": "faiss",
                "index_backend_configs": {},
                "is_demo": False,
                "collection_name": "",
            },
        )

    async def build_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None:
        await self.initialize_bm25(chunks_path, index_path, language=language)
        await self.call("retriever_bm25_index", {"overwrite": False})
        # UltraRAG 0.3.0.2 must reload a newly built index to attach passages.
        # The reload names the same stopword language as the build: load_stopwords
        # corrects the list from the saved file, but the language is what the
        # retriever records, so omitting it records a corpus as English when it
        # was built as German.
        await self.initialize_bm25(chunks_path, index_path, language=language)

    async def search_bm25(self, query: str, top_k: int) -> list[str]:
        payload = await self.call(
            "retriever_bm25_search",
            {"query_list": [query], "top_k": top_k},
        )
        if not isinstance(payload, dict):
            raise TypeError("UltraRAG BM25 returned a non-object result")
        batches = payload.get("ret_psg")
        if (
            not isinstance(batches, list)
            or not batches
            or not isinstance(batches[0], list)
        ):
            raise RuntimeError("UltraRAG BM25 returned an invalid passage result")
        return [str(item) for item in batches[0]]
