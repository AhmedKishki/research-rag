"""What a remote publishes as a release, and what this checkout declares.

A published release is a git tag, so reading one needs no token and no API key:
`git ls-remote --tags` asks the remote directly and the local tag list answers
what a fetch brought. A tag naming a pre-release or build metadata is not a
release and is left out of the answer. Two tags claiming one version make the
current release unreadable rather than a guess, because either tag could be the
one a reader installed.

The package version has one home: the distribution metadata, which is what
`pyproject.toml` becomes when the package is built. `declared_version` reads the
file the release is cut from, and `release_consistency` is the check that it
names a release version and that a local tag carrying it points at the commit
this checkout is on.

Every external command arrives through an injected runner, so the whole module
is testable with no network: a temporary repository with real tags is a local
path, and nothing here reaches for a token or an API.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .process import Runner

# The four answers a release question can have. `unreadable` is the one a reader
# must act on, and it names the condition rather than picking a side.
FOUND = "found"
NONE = "none"
AMBIGUOUS = "ambiguous"
UNREADABLE = "unreadable"

# A tag that names a release: an optional `v`, then the numbers of a version.
# Anything else, a pre-release suffix such as `-rc1` or `.dev1`, or build
# metadata after `+`, is a candidate the reader did not publish as a release.
RELEASE_TAG = re.compile(r"^v?(?P<version>[0-9]+(?:\.[0-9]+)+)$")

# A remote may name one tag as the release it considers current, which says
# which release is latest without relying on the order of the numbers.
DIST_TAG = "latest"


@dataclass(frozen=True, slots=True)
class Release:
    version: str
    tag: str
    commit: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tag": self.tag,
            "commit": self.commit,
        }


@dataclass(frozen=True, slots=True)
class ReleaseSet:
    state: str = ""
    release: Release | None = None
    detail: str = ""
    dist_tag: str | None = None

    @property
    def version(self) -> str | None:
        return self.release.version if self.release is not None else None

    @property
    def tag(self) -> str | None:
        return self.release.tag if self.release is not None else None

    @property
    def found(self) -> bool:
        return self.state == FOUND and self.release is not None

    def as_dict(self) -> dict[str, object]:
        report: dict[str, object] = {"state": self.state}
        if self.release is not None:
            report.update(self.release.as_dict())
        if self.dist_tag:
            report["dist_tag"] = self.dist_tag
        if self.detail:
            report["detail"] = self.detail
        return report


def parse_release_tag(tag: str) -> str | None:
    found = RELEASE_TAG.match(tag.strip())
    return found.group("version") if found is not None else None


def is_release_version(version: str) -> bool:
    return parse_release_tag(version) == version.strip()


def version_key(version: str) -> tuple[int, ...]:
    """Return the comparable numbers of a version, most significant first."""

    return tuple(int(part) for part in version.split("."))


def compare_versions(left: str, right: str) -> int:
    """Compare two versions by value: negative, zero, or positive."""

    first, second = version_key(left), version_key(right)
    width = max(len(first), len(second))
    padded_first = first + (0,) * (width - len(first))
    padded_second = second + (0,) * (width - len(second))
    return (padded_first > padded_second) - (padded_first < padded_second)


def tag_pairs(listing: str) -> tuple[tuple[str, str | None], ...]:
    """Return each tag and the commit it points at, from `git ls-remote --tags`.

    An annotated tag is listed twice: once for the tag object and once for the
    commit it dereferences to. The peeled line is the commit; a tag listed only
    once is a lightweight tag, and its own line already is the commit.
    """

    names: dict[str, str | None] = {}
    order: list[str] = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[1].startswith("refs/tags/"):
            continue
        name = fields[1][len("refs/tags/") :]
        peeled = name.endswith("^{}")
        if peeled:
            name = name[:-3]
        if name not in names:
            names[name] = None
            order.append(name)
        if peeled or names[name] is None:
            names[name] = fields[0]
    return tuple((name, names[name]) for name in order)


def releases_from_tags(pairs: Sequence[tuple[str, str | None]]) -> ReleaseSet:
    """The release a remote's tags publish, or why there is not one to name.

    A dist-tag decides the answer when the remote publishes one, because a
    remote that names its current release has said so and the numbers are not a
    substitute for it.
    """

    claimed: dict[str, list[str]] = {}
    commits: dict[str, str | None] = {}
    for tag, commit in pairs:
        version = parse_release_tag(tag)
        if version is None:
            continue
        claimed.setdefault(version, []).append(tag)
        commits.setdefault(version, commit)
    if not claimed:
        return ReleaseSet(
            NONE,
            detail=(
                "the remote publishes no tag naming a release version, so it has "
                "published no release"
            ),
        )
    duplicated = sorted(version for version, tags in claimed.items() if len(tags) > 1)
    if duplicated:
        version = duplicated[0]
        tags = ", ".join(sorted(claimed[version]))
        return ReleaseSet(
            AMBIGUOUS,
            detail=(
                f"two tags claim version {version} ({tags}), so the current "
                "release cannot be read. Ask the maintainer which of them is the "
                "release, and delete the other tag on the remote."
            ),
        )
    dist_commit = dict(pairs).get(DIST_TAG)
    if DIST_TAG in {name for name, _ in pairs}:
        if dist_commit is None:
            return ReleaseSet(
                UNREADABLE,
                detail=(
                    f"the remote publishes a `{DIST_TAG}` tag that names no commit, "
                    "so the release it stands for cannot be read"
                ),
            )
        named = sorted(
            version for version, commit in commits.items() if commit == dist_commit
        )
        if not named:
            return ReleaseSet(
                UNREADABLE,
                detail=(
                    f"the remote's `{DIST_TAG}` tag points at commit "
                    f"{dist_commit[:12]}, which no release tag claims"
                ),
            )
        version = named[0]
        return ReleaseSet(
            FOUND,
            Release(version, min(claimed[version]), dist_commit),
            dist_tag=DIST_TAG,
            detail=(
                f"the remote's `{DIST_TAG}` dist-tag names release {version}, "
                "which is taken as the current one"
            ),
        )
    version = max(claimed, key=version_key)
    return ReleaseSet(
        FOUND,
        Release(version, min(claimed[version]), commits.get(version)),
        detail=(
            f"release {version} is the highest version the remote's tags name, of "
            f"{len(claimed)} release tag(s)"
        ),
    )


def remote_release(root: Path, run: Runner) -> ReleaseSet:
    """Ask a remote's tags directly, without a token or an API key."""

    listed = run(["git", "-C", str(root), "ls-remote", "--tags", "origin"])
    if not listed.ok:
        first = next(iter(listed.stderr.splitlines()), "") or listed.stdout.strip()
        return ReleaseSet(
            UNREADABLE,
            detail=f"`git ls-remote --tags origin` failed: {first}",
        )
    return releases_from_tags(tag_pairs(listed.stdout))


def declared_version(checkout: Path | None) -> str | None:
    """Return the version `pyproject.toml` declares, or None when it declares none."""

    if checkout is None:
        return None
    try:
        with (checkout / "pyproject.toml").open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, ValueError):
        return None
    project = document.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    return version if isinstance(version, str) and version else None


def local_release_commit(root: Path, tag: str, run: Runner) -> str | None:
    result = run(["git", "-C", str(root), "rev-parse", f"{tag}^{{commit}}"])
    if not result.ok:
        return None
    commit = result.stdout.strip()
    return commit or None


def local_release_tags(root: Path, run: Runner) -> tuple[tuple[str, str], ...]:
    listed = run(["git", "-C", str(root), "tag", "--list"])
    if not listed.ok:
        return ()
    found: list[tuple[str, str]] = []
    for tag in (line.strip() for line in listed.stdout.splitlines()):
        commit = local_release_commit(root, tag, run)
        if commit is not None:
            found.append((tag, commit))
    return tuple(found)


def release_consistency(checkout: Path, declared: str, run: Runner) -> tuple[str, ...]:
    """Every way this checkout's declared version fails to name its own release.

    Two things are checked and neither needs a network: the declared version is
    one a release tag may carry, and a local tag carrying it points at the
    commit this checkout is on. A repository with no tag at all is consistent,
    because an unreleased version is not an inconsistency.
    """

    problems: list[str] = []
    if not is_release_version(declared):
        problems.append(
            f"pyproject.toml declares version {declared}, which is not one a "
            "release tag may carry. Use MAJOR.MINOR.PATCH, with no pre-release "
            "suffix and no build metadata."
        )
    head_result = run(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    head = head_result.stdout.strip() if head_result.ok else ""
    for tag, commit in local_release_tags(checkout, run):
        if parse_release_tag(tag) != declared.strip():
            continue
        if head and commit != head:
            problems.append(
                f"tag {tag} names version {declared} and points at commit "
                f"{commit[:12]}, while this checkout is at {(head or '')[:12]}. "
                "Move the tag onto the commit that is the release, or declare the "
                "version that commit carries."
            )
    return tuple(problems)
