"""The account's project record is a pointer file, not a cache of a project's state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from filelock import FileLock

import research_rag.project.registry as registry_module
from research_rag.project import registry
from research_rag.project.policy import ResearchError


@pytest.fixture
def account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the record at a throwaway account directory."""

    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    return home / "config" / "research-rag" / "projects.json"


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
    # A prefix is a guess, not a selector.
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
    # whole previous record or the whole new one. The only other name in the
    # account directory is the lock the write holds, and no temporary file is left.
    assert (
        json.loads(account.read_text(encoding="utf-8"))["projects"][0]["project_id"]
        == "pid-one"
    )
    assert sorted(path.name for path in account.parent.iterdir()) == [
        "projects.json",
        "projects.lock",
    ]


@pytest.fixture
def throwaway_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home whose account directory exists, plus a scratch file beside it."""

    home = tmp_path / "home"
    (home / "config" / "research-rag").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("HOME", str(home))
    return home


_REGISTER_IN_A_CHILD = """
import sys

from research_rag.project import registry

registry.register(sys.argv[1], sys.argv[1], "/tmp/" + sys.argv[1])
print("registered", sys.argv[1], flush=True)
"""


def _child_env(home: Path) -> dict[str, str]:
    """A child that reads and writes the same throwaway account directory."""

    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }


@pytest.mark.integration
def test_a_write_waits_for_the_record_lock_held_by_another_process(
    throwaway_account: Path,
) -> None:
    """A writer does not read-modify-write while another writer holds the record.

    The property is not that the record survives a crash, which the rename already
    gives. It is that a command registering a project while another is mid-write
    waits instead of publishing a record built from a read taken before the other
    writer committed, which would drop the other's entry.
    """

    account = throwaway_account / "config" / "research-rag" / "projects.json"
    script = throwaway_account / "register_child.py"
    script.write_text(_REGISTER_IN_A_CHILD, encoding="utf-8")

    held = FileLock(registry.registry_lock_path(), timeout=1)
    held.acquire()
    try:
        child = subprocess.Popen(
            [sys.executable, str(script), "pid-child"],
            env=_child_env(throwaway_account),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Without the lock the child finishes here, on a read taken before this
            # process writes anything.
            time.sleep(1.5)
            assert child.poll() is None, "the child wrote without waiting for the lock"
        finally:
            child.kill()
            child.wait(timeout=30)
    finally:
        held.release()

    assert not account.exists()
    assert registry.load() == []


@pytest.mark.integration
def test_a_write_waits_and_then_publishes_what_it_read(
    throwaway_account: Path,
) -> None:
    """A writer that waited for the lock keeps the entry the holder committed.

    Waiting is only half the property. A writer that blocks on the lock and then
    publishes a record built from a read taken before it was granted would still
    drop the other writer's entry, so this asserts the surviving record holds
    both, in the order the lock was granted.
    """

    script = throwaway_account / "register_child.py"
    script.write_text(_REGISTER_IN_A_CHILD, encoding="utf-8")

    registry.register("pid-parent", "Parent", "/tmp/parent")

    held = FileLock(registry.registry_lock_path(), timeout=1)
    held.acquire()
    try:
        child = subprocess.Popen(
            [sys.executable, str(script), "pid-child"],
            env=_child_env(throwaway_account),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(1.5)
            assert child.poll() is None, "the child wrote without waiting for the lock"
        finally:
            held.release()
            assert child.wait(timeout=120) == 0, child.stderr.read()
    except BaseException:
        child.kill()
        child.wait(timeout=30)
        raise

    assert [project.project_id for project in registry.load()] == [
        "pid-parent",
        "pid-child",
    ]


@pytest.mark.integration
def test_records_written_by_many_processes_at_once_are_all_kept(
    throwaway_account: Path,
) -> None:
    """Every command that registers a project leaves its entry in the record."""

    script = throwaway_account / "register_child.py"
    script.write_text(_REGISTER_IN_A_CHILD, encoding="utf-8")
    children = [
        subprocess.Popen(
            [sys.executable, str(script), f"pid-{index}"],
            env=_child_env(throwaway_account),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(8)
    ]
    for child in children:
        assert child.wait(timeout=120) == 0, child.stderr.read()

    assert sorted(project.project_id for project in registry.load()) == sorted(
        f"pid-{index}" for index in range(8)
    )


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
