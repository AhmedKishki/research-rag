"""Compatibility exports for the workspace-owned request guards."""

from ..surfaces.workspace.write_guard import (
    LOOPBACK_NAMES,
    host_names_this_app,
    is_loopback_client,
    json_content_type,
    served_authority,
    write_refusal,
)

LOOPBACK_CLIENT_NAMES = LOOPBACK_NAMES

__all__ = [
    "LOOPBACK_CLIENT_NAMES",
    "LOOPBACK_NAMES",
    "host_names_this_app",
    "is_loopback_client",
    "json_content_type",
    "served_authority",
    "write_refusal",
]
