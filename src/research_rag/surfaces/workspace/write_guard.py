"""Shared loopback, Host, origin, and JSON checks for workspace and control writes."""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import SplitResult, urlsplit

if TYPE_CHECKING:
    from starlette.requests import Request

LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost"})


def _loopback_name(hostname: str | None) -> bool:
    """Whether this host component is a loopback name or a loopback address."""

    if hostname is None:
        return False
    if hostname.casefold() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _split(value: str) -> SplitResult | None:
    """Parse an authority or origin without raising on malformed headers."""

    try:
        parsed = urlsplit(value)
        if parsed.port is not None and not 0 < parsed.port <= 65535:
            return None
    except ValueError:
        return None
    return parsed


def host_names_this_app(host: str | None) -> bool:
    """Require a loopback Host, not just one that matches an attacker-controlled Origin."""

    if not host:
        return False
    parsed = _split(f"//{host}")
    return parsed is not None and _loopback_name(parsed.hostname)


def is_loopback_client(request: Request) -> bool:
    """Require a loopback TCP peer."""

    client = request.client
    if client is None:
        return False
    return _loopback_name(client.host)


def served_authority(request: Request) -> SplitResult | None:
    """Return the requested loopback authority, or None."""

    parsed = _split("//" + request.headers.get("host", ""))
    if parsed is None or not _loopback_name(parsed.hostname):
        return None
    return parsed


def write_refusal(request: Request) -> tuple[int, str] | None:
    """Return a refusal or None; local terminal requests may omit Origin."""

    if not is_loopback_client(request):
        return 403, "Writes are refused from another host"
    authority = served_authority(request)
    if authority is None:
        return 403, "Writes are refused for a host this app does not serve"
    if request.headers.get("sec-fetch-site", "").casefold() == "cross-site":
        return 403, "Cross-origin writes are blocked"
    origin = request.headers.get("origin")
    if origin:
        parsed = _split(origin)
        if parsed is None:
            return 403, "The Origin header could not be read"
        if parsed.scheme not in {"http", "https"}:
            return 403, "Cross-origin writes are blocked"
        if parsed.netloc != request.headers.get("host", ""):
            return 403, "Cross-origin writes are blocked"
        if not _loopback_name(parsed.hostname):
            return 403, "Writes are refused for a host this app does not serve"
    return None


def json_content_type(request: Request) -> bool:
    """Require the JSON media type, allowing charset parameters."""

    return request.headers.get("content-type", "").split(";", 1)[0].casefold() == (
        "application/json"
    )


__all__ = [
    "LOOPBACK_NAMES",
    "host_names_this_app",
    "is_loopback_client",
    "json_content_type",
    "served_authority",
    "write_refusal",
]
