"""`Connection closed` is what a client sees when the vanilla gateway exits during its
handshake, and the reason is in the log the transport already writes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from research_rag.project.config import ResearchConfig, resolve_config
from research_rag.retrieval.ultrarag import (
    GATEWAY_INIT_TIMEOUT_SECONDS,
    TRANSPORT_TIMEOUT_SECONDS,
    call_timeout_failure,
    child_log_path,
    create_vanilla_transport,
    gateway_log_path,
    gateway_start_failure,
    vanilla_client,
)


@pytest.fixture
def config(project: Path) -> ResearchConfig:
    return resolve_config(project)


def _write(config: ResearchConfig, relative: str, text: str) -> Path:
    path = config.state_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _failing_gateway(project: Path) -> Path:
    broken = project / "failing-gateway"
    broken.write_text(
        "#!/bin/sh\necho 'no runtime at /nowhere' >&2\nexit 3\n", encoding="utf-8"
    )
    broken.chmod(0o755)
    return broken


def test_a_failed_start_names_the_reason_and_both_logs(config: ResearchConfig) -> None:
    # A gateway's stderr is long, and only its end says why it stopped.
    noise = "".join(f"gateway line {index}\n" for index in range(40))
    gateway_log = _write(
        config,
        "logs/vanilla-gateway-stderr.log",
        f"{noise}RuntimeError: no runtime at /nowhere\nexit 3\n",
    )
    child_log = _write(
        config,
        "ultrarag-runtime/logs/retriever-child-stderr.log",
        "retriever: cannot reach the index\n",
    )

    message = str(gateway_start_failure(config, RuntimeError("Connection closed")))

    assert "The UltraRAG gateway could not start: Connection closed" in message
    assert str(gateway_log) in message
    assert "RuntimeError: no runtime at /nowhere" in message
    assert "exit 3" in message
    # The tail is carried, not the whole log.
    assert "gateway line 0" not in message
    assert "gateway line 39" in message
    assert str(child_log) in message
    assert "retriever: cannot reach the index" in message


def test_a_failed_start_without_any_log_is_still_usable(config: ResearchConfig) -> None:
    """A transport that wrote nothing must not leave the caller with nothing."""

    message = str(gateway_start_failure(config, RuntimeError("Connection closed")))

    assert "The UltraRAG gateway could not start: Connection closed" in message
    assert str(gateway_log_path(config)) in message
    assert "empty or absent" in message
    assert str(config.ultrarag_workspace / "logs") in message


def test_a_failed_start_with_an_empty_log_names_the_directory(
    config: ResearchConfig,
) -> None:
    _write(config, "logs/vanilla-gateway-stderr.log", "")

    message = str(gateway_start_failure(config, RuntimeError("Connection closed")))

    assert "empty or absent" in message
    assert str(config.ultrarag_workspace / "logs") in message


def test_a_call_that_never_answered_names_the_component(config: ResearchConfig) -> None:
    message = str(
        call_timeout_failure(
            config,
            "retriever_retriever_search",
            RuntimeError("Timed out while waiting for a response"),
        )
    )

    assert "retriever" in message
    assert "retriever_retriever_search" in message
    assert str(child_log_path(config, "retriever")) in message
    assert str(gateway_log_path(config)) in message
    assert str(TRANSPORT_TIMEOUT_SECONDS) in message


def test_a_gateway_that_exits_is_a_tool_error_naming_its_log(project: Path) -> None:
    broken = _failing_gateway(project)
    config = resolve_config(project, vanilla_executable=broken)

    async def connect() -> None:
        async with vanilla_client(config):
            pass

    with pytest.raises(ToolError) as raised:
        asyncio.run(connect())

    message = str(raised.value)
    assert "The UltraRAG gateway could not start" in message
    assert str(gateway_log_path(config)) in message
    assert str(config.ultrarag_workspace / "logs") in message


def test_the_transport_writes_the_log_the_message_names(
    config: ResearchConfig,
) -> None:
    """The failure message and the health report must name one file, not two."""

    transport = create_vanilla_transport(config)
    log_file = getattr(transport, "log_file", None)

    assert log_file is not None
    assert Path(str(log_file)) == gateway_log_path(config)
    assert child_log_path(config, "retriever").parent == (
        config.ultrarag_workspace / "logs"
    )


def test_the_handshake_is_bounded_by_its_own_timeout(config: ResearchConfig) -> None:
    """A gateway that exits mid-handshake leaves the client awaiting a response that never
    arrives, so the handshake needs its own bound.

    Reusing the call budget here made a failure the caller hears in seconds take the full
    30 minutes.
    """

    assert GATEWAY_INIT_TIMEOUT_SECONDS < TRANSPORT_TIMEOUT_SECONDS


def test_a_gateway_that_never_starts_is_reported_within_the_handshake_bound(
    config: ResearchConfig,
) -> None:
    """The reported failure must arrive, not merely be raised eventually."""

    broken = _failing_gateway(config.project_root)
    failing = resolve_config(config.project_root, vanilla_executable=broken)

    async def connect() -> None:
        async with vanilla_client(failing):
            pass

    started = time.monotonic()
    with pytest.raises(ToolError) as raised:
        asyncio.run(
            asyncio.wait_for(
                connect(),
                timeout=GATEWAY_INIT_TIMEOUT_SECONDS + 15,
            )
        )

    # What is asserted is that the reason is named and arrives long before the call
    # budget the handshake no longer shares.
    assert "The UltraRAG gateway could not start" in str(raised.value)
    assert time.monotonic() - started < TRANSPORT_TIMEOUT_SECONDS
