"""Pinned upstream surface used by the first vanilla compatibility release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASELINE_VERSION = "0.3.0.2"
BASELINE_COMMIT = "3a709a2aea3fbe46acca59c422621c94b6e86857"


@dataclass(frozen=True, slots=True)
class ServerSpec:
    namespace: str
    relative_entrypoint: Path


SERVER_SPECS: tuple[ServerSpec, ...] = tuple(
    ServerSpec(name, Path("servers") / name / "src" / f"{name}.py")
    for name in (
        "benchmark",
        "corpus",
        "custom",
        "evaluation",
        "generation",
        "memory",
        "prompt",
        "reranker",
        "retriever",
        "router",
        "sayhello",
    )
)

# These components keep initialized objects or mutable data in their child
# process. The gateway must therefore reuse their stdio sessions across calls.
STATEFUL_NAMESPACES = frozenset({"generation", "memory", "reranker", "retriever"})
