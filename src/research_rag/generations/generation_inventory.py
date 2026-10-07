"""Every generation this project retains on disk.

This module owns a filesystem walk, not an answer about the project: it reads one
manifest per retained generation and measures the directory, so `status` can name
what a prune would consider. It may never validate a generation, select one, or
decide that a project can be served, because a retained generation is not a
served one and a damaged manifest is a reportable fact rather than a failure.

What moved here from `status.py`: `generation_inventory`.
"""

from __future__ import annotations

from typing import Any

from ..project.config import ResearchConfig
from ..storage.records import StorageError, directory_statistics, read_json


def generation_inventory(
    config: ResearchConfig,
    current_generation_id: str | None,
) -> dict[str, Any]:
    """Describe every retained generation on disk without validating it.

    These are exactly the directories a prune would consider, so `status`
    reports them with their size, file count, and recorded build configuration.
    Model and configuration facts come from that manifest, never today's settings.
    A missing, unreadable, or non-JSON manifest is reported with a ``manifest_error``
    instead of raising: the report must answer when one retained generation is
    damaged. This runs only on the read-only ``status`` surface, never on the
    search path, because it walks each generation's files.
    """

    root = config.generations_root
    records: list[dict[str, Any]] = []
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            record: dict[str, Any] = {
                "generation_id": entry.name,
                "is_current": entry.name == current_generation_id,
                "chunking": None,
                "retrieval": None,
                "retrieval_policy_fingerprint": None,
                "extraction_policy_version": None,
                "cleaning_policy_version": None,
                "artifact_policy_version": None,
            }
            try:
                manifest = read_json(entry / "manifest.json")
            except StorageError as exc:
                record["manifest_error"] = str(exc)
            else:
                if isinstance(manifest, dict):
                    record.update(
                        created_at=manifest.get("created_at"),
                        chunk_count=manifest.get("chunk_count"),
                        document_count=manifest.get("document_count"),
                        schema_version=manifest.get("schema_version"),
                        extraction_policy_version=manifest.get(
                            "extraction_policy_version"
                        ),
                        cleaning_policy_version=manifest.get("cleaning_policy_version"),
                        artifact_policy_version=manifest.get("artifact_policy_version"),
                        retrieval_policy_fingerprint=manifest.get(
                            "retrieval_policy_fingerprint"
                        ),
                        chunking=(
                            manifest["chunking"]
                            if isinstance(manifest.get("chunking"), dict)
                            else None
                        ),
                        retrieval=(
                            manifest["retrieval"]
                            if isinstance(manifest.get("retrieval"), dict)
                            else None
                        ),
                    )
                else:
                    record["manifest_error"] = "manifest.json is not a JSON object"
            file_count, size_bytes = directory_statistics(entry)
            record["file_count"] = file_count
            record["size_bytes"] = size_bytes
            records.append(record)
    records.sort(
        key=lambda item: (str(item.get("created_at") or ""), item["generation_id"]),
        reverse=True,
    )
    return {
        "generations": records,
        "retained_generation_count": len(records),
        "retained_generation_bytes": sum(int(item["size_bytes"]) for item in records),
    }
