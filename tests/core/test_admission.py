"""Turns handed out in rounds, each wait bounded, nothing held past its block."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import research_rag.core.service as service_module
from research_rag.core.admission import Admission, AdmissionTimeout, as_caller
from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.policy import ResearchError
from tests.conftest import write_pdf
from tests.core.test_service import FakeDenseBackend, FakeUltraRAG


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def test_callers_are_served_in_rounds_not_in_arrival_order() -> None:
    """An agent that asks five times waits behind the others' first questions.

    A holds the one slot and queues four more; B and C each queue one. The order
    served is A's first, then one of each caller in turn, never A's five in a row.
    """

    async def exercise() -> list[str]:
        gate = Admission(1, wait_seconds=5, what="searches")
        served: list[str] = []
        release = asyncio.Event()

        async def holder() -> None:
            async with gate.slot("A"):
                served.append("A0")
                await release.wait()

        async def asker(caller: str, label: str) -> None:
            async with gate.slot(caller):
                served.append(label)

        first = asyncio.create_task(holder())
        await _settle()
        tasks = [asyncio.create_task(asker("A", f"A{n}")) for n in range(1, 5)]
        await _settle()
        tasks.append(asyncio.create_task(asker("B", "B1")))
        await _settle()
        tasks.append(asyncio.create_task(asker("C", "C1")))
        await _settle()
        release.set()
        await asyncio.gather(first, *tasks)
        return served

    assert asyncio.run(exercise()) == ["A0", "A1", "B1", "C1", "A2", "A3", "A4"]


def test_slots_run_together_up_to_their_number() -> None:
    async def exercise() -> int:
        gate = Admission(2, wait_seconds=5, what="searches")
        running = peak = 0

        async def work() -> None:
            nonlocal running, peak
            async with gate.slot("x"):
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.01)
                running -= 1

        await asyncio.gather(*(work() for _ in range(6)))
        return peak

    assert asyncio.run(exercise()) == 2


def test_a_wait_that_runs_out_says_how_many_were_ahead_and_frees_its_place() -> None:
    async def take(gate: Admission, caller: str) -> None:
        async with gate.slot(caller):
            pass

    async def exercise() -> tuple[int, int, int, int]:
        gate = Admission(1, wait_seconds=0.15, what="searches")
        async with gate.slot("A"):
            first = asyncio.create_task(take(gate, "B"))
            await asyncio.sleep(0.05)
            second = asyncio.create_task(take(gate, "C"))
            await asyncio.sleep(0.05)
            queued = gate.waiting
            with pytest.raises(AdmissionTimeout) as b:
                await first
            with pytest.raises(AdmissionTimeout) as c:
                await second
        return queued, b.value.ahead, c.value.ahead, gate.waiting + gate.free

    queued, ahead_of_b, ahead_of_c, settled = asyncio.run(exercise())
    assert queued == 2
    assert ahead_of_b == 0
    # B had left the line by the time C's wait ran out.
    assert ahead_of_c == 0
    assert settled == 1


def test_a_cancelled_wait_gives_nothing_up_and_blocks_no_one() -> None:
    async def exercise() -> list[str]:
        gate = Admission(1, wait_seconds=5, what="searches")
        served: list[str] = []
        release = asyncio.Event()

        async def hold() -> None:
            async with gate.slot("A"):
                await release.wait()

        async def take(caller: str) -> None:
            async with gate.slot(caller):
                served.append(caller)

        holder = asyncio.create_task(hold())
        await _settle()
        cancelled = asyncio.create_task(take("B"))
        await _settle()
        survivor = asyncio.create_task(take("C"))
        await _settle()
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        release.set()
        await asyncio.gather(holder, survivor)
        return served

    assert asyncio.run(exercise()) == ["C"]


def test_an_error_in_a_turn_still_returns_it() -> None:
    async def exercise() -> int:
        gate = Admission(1, wait_seconds=1, what="searches")
        with pytest.raises(RuntimeError):
            async with gate.slot("A"):
                raise RuntimeError("search failed")
        return gate.free

    assert asyncio.run(exercise()) == 1


def test_the_caller_is_whoever_the_surface_named() -> None:
    async def exercise() -> list[str]:
        gate = Admission(1, wait_seconds=1, what="searches")
        seen: list[str] = []
        with as_caller("agent 7"):
            async with gate.slot():
                seen.append("in")
        return seen

    assert asyncio.run(exercise()) == ["in"]


def test_a_search_behind_a_full_queue_is_told_where_it_stood(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search waits its turn and is then refused with the queue it stood in."""

    async def exercise() -> str:
        write_pdf(
            project / "sources" / "article.pdf",
            ["Cobalt evidence about labour."],
            title="Article",
        )
        config = resolve_config(project, vanilla_executable=sys.executable)
        monkeypatch.setattr(service_module, "SEARCH_QUEUE_SECONDS", 0.1)
        service = ResearchService(  # type: ignore[arg-type]
            config, FakeUltraRAG(), dense=FakeDenseBackend()
        )
        await service.ingest(chunk_size=100, chunk_overlap=10)
        slots = service.config.settings.search_concurrency
        async with asyncio.TaskGroup() as group:
            held = asyncio.Event()

            async def hold() -> None:
                async with service._searches.slot("busy"):
                    held.set()
                    await asyncio.sleep(0.5)

            for _ in range(slots):
                group.create_task(hold())
            await held.wait()
            with pytest.raises(ResearchError) as refused:
                await service.search("cobalt", top_k=1)
        return str(refused.value)

    message = asyncio.run(exercise())
    assert "ahead of this one" in message
    assert "call again" in message
