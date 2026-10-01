"""Nothing here reaches a network.

The tests that need real tags build a temporary repository and a local bare remote, so a
release is read through the same `git` calls the command makes. The rest hand the parser
the listing `git ls-remote --tags` would print.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from research_rag import release as release_module
from research_rag.process import CommandResult, subprocess_runner
from research_rag.release import (
    AMBIGUOUS,
    FOUND,
    NONE,
    UNREADABLE,
    parse_release_tag,
    release_consistency,
    releases_from_tags,
    tag_pairs,
)

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
COMMIT_C = "c" * 40


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "HOME": str(root),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    return completed.stdout.strip()


def _bare_remote(root: Path) -> Path:
    remote = root / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "--initial-branch=main")
    return remote


def _checkout(root: Path, remote: Path) -> Path:
    work = root / "work"
    work.mkdir()
    _git(work, "init", "--initial-branch=main")
    _git(work, "remote", "add", "origin", str(remote))
    return work


def _commit(work: Path, message: str, *, version: str | None = None) -> str:
    if version is not None:
        (work / "pyproject.toml").write_text(
            f'[project]\nname = "research-rag"\nversion = "{version}"\n',
            encoding="utf-8",
        )
    (work / "README.md").write_text(f"{message}\n", encoding="utf-8")
    _git(work, "add", "--all")
    _git(work, "commit", "--quiet", "-m", message)
    return _git(work, "rev-parse", "HEAD")


def _published(root: Path, *versions: str) -> tuple[Path, dict[str, str]]:
    work = _checkout(root, _bare_remote(root))
    commits: dict[str, str] = {}
    for version in versions:
        commits[version] = _commit(work, f"release {version}", version=version)
        _git(work, "tag", f"v{version}")
        _git(work, "push", "--quiet", "origin", "main", f"v{version}")
    return work, commits


def _tag(name: str, commit: str) -> tuple[str, str]:
    return name, commit


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.1.0", "0.1.0"),
        ("0.1.0", "0.1.0"),
        ("v1.2.3", "1.2.3"),
        ("v0.2.0-rc1", None),
        ("v0.2.0rc1", None),
        ("v1.0.0.dev3", None),
        ("v0.2.0+build.5", None),
        ("latest", None),
        ("release-0.1.0", None),
        ("v0.1", "0.1"),
    ],
)
def test_a_tag_naming_a_prerelease_or_build_metadata_is_not_a_release(
    tag: str, expected: str | None
) -> None:
    assert parse_release_tag(tag) == expected


def test_the_highest_release_is_chosen_by_version_and_not_lexically() -> None:
    """`0.10.0` is above `0.9.0`, which a string comparison gets backwards."""

    found = releases_from_tags([_tag("v0.9.0", COMMIT_A), _tag("v0.10.0", COMMIT_B)])

    assert found.state == FOUND
    assert found.version == "0.10.0"
    assert found.tag == "v0.10.0"


def test_a_prerelease_tag_is_left_out_of_the_release_answer() -> None:
    found = releases_from_tags(
        [
            _tag("v0.1.0", COMMIT_A),
            _tag("v0.2.0-rc1", COMMIT_B),
            _tag("v0.2.0+b1", COMMIT_C),
        ]
    )

    assert found.version == "0.1.0"


def test_two_tags_claiming_one_version_are_refused_rather_than_guessed() -> None:
    found = releases_from_tags([_tag("v0.1.0", COMMIT_A), _tag("0.1.0", COMMIT_B)])

    assert found.state == AMBIGUOUS
    assert found.version is None
    assert "0.1.0" in found.detail
    assert "v0.1.0" in found.detail
    assert "0.1.0" in found.detail


def test_a_latest_dist_tag_decides_which_release_is_current() -> None:
    found = releases_from_tags(
        [
            _tag("v0.1.0", COMMIT_A),
            _tag("v0.2.0", COMMIT_B),
            _tag("latest", COMMIT_A),
        ]
    )

    assert found.state == FOUND
    assert found.version == "0.1.0"
    assert found.dist_tag == "latest"
    assert "dist-tag" in found.detail


def test_a_dist_tag_pointing_at_no_release_tag_is_unreadable() -> None:
    found = releases_from_tags([_tag("v0.1.0", COMMIT_A), _tag("latest", COMMIT_C)])

    assert found.state == UNREADABLE
    assert found.version is None
    assert "latest" in found.detail


def test_a_remote_publishing_no_release_tag_says_so() -> None:
    found = releases_from_tags(
        [_tag("nightly", COMMIT_A), _tag("v0.2.0-rc1", COMMIT_B)]
    )

    assert found.state == NONE
    assert found.version is None
    assert "no tag naming a release version" in found.detail


def test_an_annotated_tag_is_read_at_the_commit_it_names() -> None:
    listing = "\n".join(
        (
            f"{COMMIT_A}\trefs/tags/v0.1.0",
            f"{COMMIT_B}\trefs/tags/v0.1.0^{{}}",
            f"{COMMIT_C}\trefs/tags/v0.2.0",
        )
    )

    assert tag_pairs(listing) == (
        ("v0.1.0", COMMIT_B),
        ("v0.2.0", COMMIT_C),
    )


def test_a_release_is_read_from_a_remote_without_a_token(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    work, commits = _published(root, "0.1.0", "0.2.0")

    found = release_module.remote_release(work, subprocess_runner)

    assert found.state == FOUND
    assert found.version == "0.2.0"
    assert found.tag == "v0.2.0"
    assert found.release is not None
    assert found.release.commit == commits["0.2.0"]


def test_a_repository_with_no_release_publishes_none(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    work, _ = _published(root)

    found = release_module.remote_release(work, subprocess_runner)

    assert found.state == NONE
    assert found.version is None


def test_a_remote_that_cannot_be_asked_is_unreadable_rather_than_a_guess(
    tmp_path: Path,
) -> None:
    work = tmp_path / "checkout"
    work.mkdir()

    def failing(argv: object, **_kwargs: object) -> CommandResult:
        return CommandResult(128, "", "could not resolve host")

    found = release_module.remote_release(work, failing)

    assert found.state == UNREADABLE
    assert "could not resolve host" in found.detail


def test_the_declared_version_is_read_from_pyproject_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "research-rag"\nversion = "1.4.2"\n', encoding="utf-8"
    )

    assert release_module.declared_version(tmp_path) == "1.4.2"
    assert release_module.declared_version(tmp_path / "absent") is None
    assert release_module.declared_version(None) is None


def test_a_declared_version_that_is_not_a_release_version_is_reported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    work = _checkout(root, _bare_remote(root))
    _commit(work, "first", version="0.2.0-rc1")

    problems = release_consistency(work, "0.2.0-rc1", subprocess_runner)

    assert any("not one a release tag may carry" in problem for problem in problems)


def test_a_tag_naming_the_declared_version_must_point_at_this_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    remote = _bare_remote(root)
    work = _checkout(root, remote)
    _commit(work, "release 0.1.0", version="0.1.0")
    _git(work, "tag", "v0.1.0")
    _git(work, "push", "--quiet", "origin", "v0.1.0")
    _commit(work, "work after the release", version="0.1.0")

    problems = release_consistency(work, "0.1.0", subprocess_runner)

    assert len(problems) == 1
    assert "v0.1.0" in problems[0]
    assert "this checkout is at" in problems[0]


def test_a_tag_naming_the_declared_version_at_this_commit_is_consistent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    remote = _bare_remote(root)
    work = _checkout(root, remote)
    _commit(work, "release 0.1.0", version="0.1.0")
    _git(work, "tag", "v0.1.0")

    assert release_consistency(work, "0.1.0", subprocess_runner) == ()


def test_an_untagged_repository_is_consistent_with_its_declared_version(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    work = _checkout(root, _bare_remote(root))
    _commit(work, "unreleased", version="0.1.0")

    assert release_consistency(work, "0.1.0", subprocess_runner) == ()


def test_this_repository_declares_a_version_a_release_tag_may_carry() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["project"]["version"]

    assert release_module.is_release_version(declared)
    assert release_module.declared_version(root) == declared
