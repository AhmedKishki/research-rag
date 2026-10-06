"""A bounded number of turns, handed out fairly between the callers waiting.

Several agents, the workspace, and the command line share one app. One lock in
front of everything made a long build freeze them all, and no lock at all lets
eight agents run eight rerankers on the same cores. `Admission` sits between the
two: a fixed number of slots, a queue per caller, and the callers taken in
rounds, so an agent that asks twenty times waits behind the others' first
questions and not in front of them.

A wait is bounded. A caller that has not been given a turn in time is told how
many were ahead of it and to ask again, because an agent that is held silently
meets its own client's timeout and loses the answer anyway.

Nothing here knows what the turn is for. It never reads a request, and it never
holds a slot past the block that took it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ..project.policy import ResearchError

# The caller a request came from, as the surface that received it knows it: an
# agent by its bridge process, the workspace, or the command line.
_CALLER: ContextVar[str] = ContextVar("research_rag_caller", default="anonymous")


@contextmanager
def as_caller(caller: str) -> Iterator[None]:
    """Name who the work inside this block is done for."""

    token = _CALLER.set(caller or "anonymous")
    try:
        yield
    finally:
        _CALLER.reset(token)


def current_caller() -> str:
    return _CALLER.get()


class AdmissionTimeout(ResearchError):
    """No turn came in time. `ahead` is how many callers were still in front."""

    def __init__(self, message: str, ahead: int) -> None:
        super().__init__(message)
        self.ahead = ahead


@dataclass(eq=False)
class _Waiter:
    future: asyncio.Future[None] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class Admission:
    """`slots` turns at once, taken by caller in rounds, each wait bounded."""

    def __init__(self, slots: int, *, wait_seconds: float, what: str) -> None:
        if slots < 1:
            raise ValueError("an admission needs at least one slot")
        self._free = slots
        self._wait_seconds = wait_seconds
        self._what = what
        self._queues: dict[str, deque[_Waiter]] = {}
        self._rotation: deque[str] = deque()

    @property
    def waiting(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    @property
    def free(self) -> int:
        return self._free

    def _position(self, caller: str, waiter: _Waiter) -> int:
        """How many waiters the rounds would serve before this one."""

        queues = {key: list(queue) for key, queue in self._queues.items()}
        rotation = list(self._rotation)
        ahead = 0
        while rotation:
            key = rotation.pop(0)
            queue = queues[key]
            head = queue.pop(0)
            if head is waiter:
                return ahead
            ahead += 1
            if queue:
                rotation.append(key)
        return ahead

    def _grant(self) -> None:
        """Give each free slot to the next caller in the rounds."""

        while self._free > 0 and self._rotation:
            caller = self._rotation.popleft()
            queue = self._queues[caller]
            waiter = queue.popleft()
            if queue:
                self._rotation.append(caller)
            else:
                del self._queues[caller]
            if waiter.future.done():
                continue
            self._free -= 1
            waiter.future.set_result(None)

    def _forget(self, caller: str, waiter: _Waiter) -> None:
        queue = self._queues.get(caller)
        if queue is None or waiter not in queue:
            return
        queue.remove(waiter)
        if not queue:
            del self._queues[caller]
            self._rotation.remove(caller)

    @asynccontextmanager
    async def slot(self, caller: str | None = None) -> AsyncIterator[None]:
        """Hold one turn for the block, waiting for it in this caller's place."""

        who = caller or current_caller()
        if self._free > 0 and not self._rotation:
            self._free -= 1
        else:
            waiter = _Waiter()
            if who not in self._queues:
                self._queues[who] = deque()
                self._rotation.append(who)
            self._queues[who].append(waiter)
            try:
                await asyncio.wait_for(
                    asyncio.shield(waiter.future), timeout=self._wait_seconds
                )
            except TimeoutError as exc:
                ahead = self._position(who, waiter)
                self._forget(who, waiter)
                if waiter.future.done() and not waiter.future.cancelled():
                    # The turn arrived as the wait ran out, so it is given back.
                    self._release()
                else:
                    waiter.future.cancel()
                raise AdmissionTimeout(
                    f"{ahead} other {self._what} were ahead of this one and none "
                    f"finished within {self._wait_seconds:.0f} seconds. Nothing was "
                    "done: call again.",
                    ahead,
                ) from exc
            except asyncio.CancelledError:
                self._forget(who, waiter)
                if waiter.future.done() and not waiter.future.cancelled():
                    self._release()
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        self._free += 1
        self._grant()
