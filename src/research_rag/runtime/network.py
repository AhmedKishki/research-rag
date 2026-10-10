"""Ephemeral HTTP home-LAN policy; never trust forwarded peer or Host claims."""

from __future__ import annotations

import ipaddress
import socket
import struct
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urlsplit

_PRIVATE = tuple(
    ipaddress.IPv4Network(net)
    for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
LOCAL_ONLY_PATHS = (
    "/control",
    "/mcp",
    "/api/projects",
    "/api/clients",
    "/api/agent-entry",
    "/api/updates",
    "/api/open-source",
    "/api/health",
    "/api/sql",
    "/api/memory",
    "/api/bundles",
)


def _private(address: str) -> bool:
    try:
        ip = ipaddress.IPv4Address(address)
    except ValueError:
        return False
    return any(ip in network for network in _PRIVATE)


def _loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def discover_lan_addresses() -> tuple[str, ...]:
    """Read assigned Linux interface IPv4 addresses, without DNS or network I/O.

    Fail closed on unsupported hosts or unavailable interfaces. A hostname lookup
    or UDP route probe is not proof that an address belongs to a local interface.
    """
    import fcntl

    addresses: set[str] = set()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
        for _, name in socket.if_nameindex():
            try:
                result = fcntl.ioctl(
                    handle.fileno(), 0x8915, struct.pack("256s", name.encode()[:15])
                )
            except OSError:
                continue
            address = socket.inet_ntoa(result[20:24])
            if _private(address):
                addresses.add(address)
    return tuple(sorted(addresses))


def _split(value: str) -> SplitResult | None:
    if (
        any(character.isspace() or ord(character) < 32 for character in value)
        or "\\" in value
    ):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            return None
        if parsed.path or parsed.query or parsed.fragment:
            return None
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return None
        return parsed
    except ValueError:
        return None


@dataclass
class NetworkAccess:
    """Validated interface snapshot and request gates, injectable into the workspace."""

    port: int
    lan: bool = False
    addresses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if any(not _private(address) for address in self.addresses):
            raise ValueError("LAN addresses must be RFC1918 IPv4 interface addresses")

    def served_authority(self, request: Any) -> SplitResult | None:
        hosts = request.headers.getlist("host")
        if len(hosts) != 1:
            return None
        parsed = _split("//" + hosts[0])
        if parsed is None or (parsed.port or 80) != self.port:
            return None
        hostname = parsed.hostname or ""
        if hostname == "localhost" or _loopback(hostname):
            return parsed
        if self.lan and hostname in self.addresses:
            return parsed
        return None

    def is_local(self, request: Any) -> bool:
        return request.client is not None and _loopback(request.client.host)

    def request_refusal(self, request: Any) -> tuple[int, str] | None:
        authority = self.served_authority(request)
        if authority is None:
            return 403, "Requests require a Host and port this app serves"
        local = self.is_local(request)
        if not local and (
            not self.lan or request.client is None or not _private(request.client.host)
        ):
            return 403, "Requests require a loopback or RFC1918 IPv4 peer"
        if not local:
            if authority.hostname not in self.addresses:
                return 403, "LAN requests require a local-interface Host"
            path = request.url.path
            if any(
                path == prefix or path.startswith(prefix + "/")
                for prefix in LOCAL_ONLY_PATHS
            ):
                return 403, "This API is available only on this machine"
        return None

    def write_refusal(self, request: Any) -> tuple[int, str] | None:
        refusal = self.request_refusal(request)
        if refusal:
            return refusal
        if request.headers.get("sec-fetch-site", "").casefold() == "cross-site":
            return 403, "Cross-origin writes are blocked"
        origins = request.headers.getlist("origin")
        if len(origins) > 1:
            return 403, "The Origin header could not be read"
        origin = origins[0] if origins else None
        if not origin and not self.is_local(request):
            return 403, "LAN browser writes require an Origin"
        if origin:
            parsed = _split(origin)
            if (
                parsed is None
                or parsed.scheme != "http"
                or parsed.netloc != request.headers.get("host")
            ):
                return 403, "Cross-origin writes are blocked"
        if (
            request.headers.get("content-type", "").split(";", 1)[0].casefold()
            != "application/json"
        ):
            return 415, "Writes require application/json"
        return None

    def report(self) -> dict[str, Any]:
        urls = (
            [f"http://{address}:{self.port}" for address in self.addresses]
            if self.lan
            else []
        )
        return {
            "enabled": self.lan,
            "urls": urls,
            "remote_page_urls": [url + "/remote" for url in urls],
            "workspace_urls": [url + "/next/" for url in urls],
            "startup_remedy": "Enable LAN access on this machine's Remote Access page, or run research-rag lan enable.",
        }
