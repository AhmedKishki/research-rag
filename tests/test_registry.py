"""The account's project record: one install, many projects.

The record is a pointer file, so these tests are about the file: what a caller
finds after `init`, what a selector resolves to, and what happens when the file
is absent, damaged, or written twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import research_rag.registry as registry_module
from research_rag import registry
from research_rag.support import ResearchError


@pytest.fixture
def account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the record at a throwaway account directory."""

    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    return home / "config" / "research-ultra-rag-mcp" / "projects.json"


def test_an_account_with_no_projects_says_so(account: Path) -> None:
    assert registry.registry_path() == account
    assert not account.exists()
    assert registry.load() == []
    assert registry.matches("anything") == []
    with pytest.raises(ResearchError, match="No registered project"):
        registry.resolve("anything")


def test_a_registered_project_is_found_by_name_and_by_id(account: Path) -> None:
    recorded = registry.register("pid-one", "Layout Thesis", "/tmp/layout-thesis")

    assert recorded.project_root == Path("/tmp/layout-thesis")
    assert registry.load() == [recorded]
    # One selector, two answers: the name a person knows and the id a script has.
    assert registry.resolve("layout thesis") == recorded
    assert registry.resolve("Layout Thesis") == recorded
    assert registry.resolve("pid-one") == recorded
    assert registry.resolve("LAYOUT THESIS") == recorded
    # A prefix is a guess, not a selector, so it resolves to nothing.
    assert registry.matches("layout") == []


def test_registering_again_follows_the_project_rather_than_duplicating_it(
    account: Path,
) -> None:
    first = registry.register("pid-one", "Layout Thesis", "/tmp/layout-thesis")
    moved = registry.register("pid-one", "Layout Thesis Renamed", "/tmp/thesis")

    assert registry.load() == [moved]
    assert registry.load() != [first]
    assert registry.resolve("layout thesis renamed") == moved
    with pytest.raises(ResearchError, match="No registered project"):
        registry.resolve("Layout Thesis")


def test_two_projects_may_share_a_name_and_are_told_apart(account: Path) -> None:
    first = registry.register("pid-one", "Notes", "/tmp/one")
    second = registry.register("pid-two", "Notes", "/tmp/two")

    assert registry.load() == [first, second]
    assert len(registry.matches("Notes")) == 2
    assert registry.resolve("pid-two") == second
    with pytest.raises(ResearchError, match="names more than one"):
        registry.resolve("Notes")


def test_forgetting_one_leaves_the_others(account: Path) -> None:
    registry.register("pid-one", "One", "/tmp/one")
    registry.register("pid-two", "Two", "/tmp/two")

    assert registry.forget("pid-one") is True
    assert registry.forget("pid-one") is False
    assert [project.project_id for project in registry.load()] == ["pid-two"]


def test_the_record_is_written_whole_or_not_at_all(account: Path) -> None:
    registry.register("pid-one", "One", "/tmp/one")

    # The replacement is a rename over the old file, so a reader either sees the
    # whole previous record or the whole new one, and no temporary file is left.
    assert (
        json.loads(account.read_text(encoding="utf-8"))["projects"][0]["project_id"]
        == "pid-one"
    )
    assert sorted(path.name for path in account.parent.iterdir()) == ["projects.json"]


def test_a_damaged_record_is_reported_rather_than_guessed(account: Path) -> None:
    account.parent.mkdir(parents=True)
    account.write_text("{not json", encoding="utf-8")

    with pytest.raises(ResearchError, match="could not be read"):
        registry.load()


def test_an_entry_that_is_not_a_pointer_is_skipped(account: Path) -> None:
    account.parent.mkdir(parents=True)
    account.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": [
                    {"project_id": "pid-one"},
                    "nonsense",
                    {
                        "project_id": "pid-two",
                        "project_name": "Two",
                        "project_root": "/tmp/two",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # One unreadable entry must not hide the projects a user still has.
    assert [project.project_id for project in registry.load()] == ["pid-two"]


def test_a_record_written_by_a_future_version_is_read_as_it_stands(
    account: Path,
) -> None:
    account.parent.mkdir(parents=True)
    account.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "projects": [
                    {
                        "project_id": "pid-one",
                        "project_name": "One",
                        "project_root": "/tmp/one",
                        "registered_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "projects_are_projects": True,
            }
        ),
        encoding="utf-8",
    )

    loaded = registry.load()
    assert [project.project_id for project in loaded] == ["pid-one"]
    # Registering again keeps whatever the file already carried.
    registry.register("pid-one", "One", "/tmp/one")
    document = json.loads(account.read_text(encoding="utf-8"))
    assert document["schema_version"] == registry_module.SCHEMA_VERSION
    assert document["projects_are_projects"] is True
