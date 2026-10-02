"""Aggregate MCP gateway for a pinned, unmodified UltraRAG release."""

from .manifest import (
    BASELINE_COMMIT,
    BASELINE_VERSION,
    SERVER_SPECS,
    STATEFUL_NAMESPACES,
)

__all__ = [
    "BASELINE_COMMIT",
    "BASELINE_VERSION",
    "SERVER_SPECS",
    "STATEFUL_NAMESPACES",
]
__version__ = "0.1.3"
