"""What a remote publishes as a release, and what this checkout declares.

A published release is a git tag, so reading one needs no token and no API key:
`git ls-remote --tags` asks the remote directly and the local tag list answers
what a fetch brought. A tag naming a pre-release or build metadata is not a
release and is left out of the answer. Two tags claiming one version make the
current release unreadable rather than a guess, because either tag could be the
one a reader installed.

A tag says which commit is a release. It says nothing about what the release
changed, so `published_release` reads the second half from where the release was
published: the latest stable release on GitHub, its body, its page, and its date.
That request needs no credential either, because the repository is public and no
token, key, or `Authorization` header is sent with it. A draft or a pre-release is
not a release a reader may install, and neither is answered here even if the API
names one. The body is the maintainer's published text rather than markup this
app trusts, so a surface that shows it sanitizes it.

`github_repository` reads a git remote conservatively: a URL that does not name
this repository on github.com yields None instead of a repository name, and a
release body is never read for a remote that names some other project.

The package version has one home: the distribution metadata, which is what
`pyproject.toml` becomes when the package is built. `declared_version` reads the
file the release is cut from, and `release_consistency` is the check that it
names a release version and that a local tag carrying it points at the commit
this checkout is on.

Every external command arrives through an injected runner, and the one network
call arrives through an injected `fetch`, so the whole module is testable with no
remote and no network: a temporary repository with real tags is a local path, and
`fetch_text` is replaced by a function that returns a recorded response.
"""

from __future__ import annotations

import json
import re
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .process import Runner
from .version import DISTRIBUTION_NAME

# The four answers a release question can have. `unreadable` is the one a reader
# must act on, and it names the condition rather than picking a side. `skipped`
# is the fifth and is the only one a reader asked for: nothing was asked of the
# network, because `--offline` was given.
FOUND = "found"
NONE = "none"
AMBIGUOUS = "ambiguous"
UNREADABLE = "unreadable"
SKIPPED = "skipped"

# A tag that names a release: an optional `v`, then the numbers of a version.
# Anything else, a pre-release suffix such as `-rc1` or `.dev1`, or build
# metadata after `+`, is a candidate the reader did not publish as a release.
RELEASE_TAG = re.compile(r"^v?(?P<version>[0-9]+(?:\.[0-9]+)+)$")

# A remote may name one tag as the release it considers current, which says
# which release is latest without relying on the order of the numbers.
DIST_TAG = "latest"

# The repository whose published releases describe this app, and the API that
# serves them. `releases/latest` is the endpoint that already excludes drafts and
# pre-releases, and the answer is checked against both flags as well, so a release
# a reader may not install is never previewed as one.
CANONICAL_REPOSITORY = "AhmedKishki/research-rag"
GITHUB_API_ROOT = "https://api.github.com"
# How long the one request may take and how large its answer may be. A changelog
# is prose, so an oversized response is a failure to read rather than a release.
FETCH_TIMEOUT_SECONDS = 10.0
RESPONSE_LIMIT_BYTES = 256 * 1024
# A page URL must be a page of the repository that was asked about, so a response
# naming another repository is refused rather than shown.
_GITHUB_PAGE_PREFIX = f"https://github.com/{CANONICAL_REPOSITORY}/releases/"
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_NAME_PART = re.compile(r"[A-Za-z0-9._-]+")
# Every form git accepts for GitHub, in one pattern: an optional scheme, an
# optional user, a host, then a path separated by `/` or `:`.
_REMOTE_URL = re.compile(
    r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*://)?(?:[^/@]+@)?(?P<host>[^/:]+)[/:](?P<path>.+)$"
)
# How one HTTP response is turned into text. Injected so a test answers from a
# recorded payload instead of reaching the network.
Fetch = Callable[[str, float], str]


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


@dataclass(frozen=True, slots=True)
class GitHubRelease:
    """One published release, as the repository published it.

    The body is plain text a maintainer wrote, not markup this app produced, and
    it is carried as text: a surface that renders it sanitizes it first.
    """

    version: str
    tag: str
    url: str
    published: str
    body: str = ""
    name: str = ""

    def as_dict(self) -> dict[str, object]:
        report: dict[str, object] = {
            "version": self.version,
            "tag": self.tag,
            "url": self.url,
            "published": self.published,
            "body": self.body,
        }
        if self.name:
            report["name"] = self.name
        return report


@dataclass(frozen=True, slots=True)
class PublishedRelease:
    """The published release a reader may install, or why there is not one."""

    state: str = ""
    release: GitHubRelease | None = None
    detail: str = ""
    repository: str = CANONICAL_REPOSITORY

    @property
    def found(self) -> bool:
        return self.state == FOUND and self.release is not None

    @property
    def version(self) -> str | None:
        return self.release.version if self.release is not None else None

    @property
    def tag(self) -> str | None:
        return self.release.tag if self.release is not None else None

    @property
    def url(self) -> str | None:
        return self.release.url if self.release is not None else None

    @property
    def published(self) -> str | None:
        return self.release.published if self.release is not None else None

    @property
    def body(self) -> str:
        return self.release.body if self.release is not None else ""

    def as_dict(self) -> dict[str, object]:
        report: dict[str, object] = {"state": self.state, "repository": self.repository}
        if self.release is not None:
            report.update(self.release.as_dict())
        if self.detail:
            report["detail"] = self.detail
        return report


def changelog_lines(release: GitHubRelease) -> tuple[str, ...]:
    """The published release as the lines a reader approves before installing it.

    What the release is, when it was published, where it is read in full, and then
    its own words. A release with an empty body is still a preview: the version
    and its page are what an approval is given against.
    """

    lines = [
        f"Release {release.version}, published {release.published}.",
        release.url,
        "",
    ]
    body = release.body.strip()
    if body:
        lines.extend(body.splitlines())
    return tuple(lines)


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


def github_repository(url: str | None) -> str | None:
    """The `owner/name` a git remote URL names, when it is a GitHub repository.

    Every form git accepts for GitHub is read and nothing else is. A remote on
    another host, or one naming a path rather than an owner and a repository,
    returns None rather than a guess: a release body read from the wrong place
    would be a changelog for a project this one is not.
    """

    if not url:
        return None
    found = _REMOTE_URL.match(url.strip())
    if found is None or found.group("host").lower() not in _GITHUB_HOSTS:
        return None
    path = found.group("path").removesuffix(".git")
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) != 2 or not all(_NAME_PART.fullmatch(part) for part in parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def fetch_text(url: str, timeout: float = FETCH_TIMEOUT_SECONDS) -> str:
    """One HTTPS request with the standard library, bounded in time and in size.

    No credential is sent: the repository is public, so the answer needs none, and
    a header that could carry one has no place here. Reading stops one byte past
    the limit rather than loading an answer of unknown length.
    """

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"{DISTRIBUTION_NAME}-update",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(RESPONSE_LIMIT_BYTES + 1)
    if len(payload) > RESPONSE_LIMIT_BYTES:
        raise ValueError(
            f"the answer is larger than {RESPONSE_LIMIT_BYTES} bytes, so it is not "
            "read as a changelog"
        )
    return payload.decode("utf-8", errors="replace")


def _published_date(value: object) -> str:
    """The calendar date an API timestamp carries, and nothing finer."""

    if not isinstance(value, str):
        raise TypeError("the release carries no publication date")
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError as exc:
        raise ValueError(f"the publication date {value!r} is not a timestamp") from exc


def _published_release_from(document: object, repository: str) -> GitHubRelease:
    """The release one API document describes, refused unless it checks out.

    Every field a preview shows is validated rather than assumed: a draft or a
    pre-release is not something a reader may install, a tag that names no
    release version cannot be compared with an installed version, and a page URL
    naming another repository is not this release's page. A field of the wrong
    type is a `TypeError` and a field with an impossible value is a `ValueError`;
    both reach the caller as one unreadable release.
    """

    if not isinstance(document, dict):
        raise TypeError("the answer is not a release document")
    if document.get("draft") or document.get("prerelease"):
        raise ValueError(
            "the release is a draft or a pre-release, so it is not installable"
        )
    tag = document.get("tag_name")
    if not isinstance(tag, str) or parse_release_tag(tag) is None:
        raise ValueError(f"the release names no release version in its tag {tag!r}")
    url = document.get("html_url")
    if not isinstance(url, str) or not url.startswith(_GITHUB_PAGE_PREFIX):
        raise ValueError(f"the release page {url!r} is not a page of {repository}")
    body = document.get("body")
    name = document.get("name")
    return GitHubRelease(
        version=parse_release_tag(tag) or "",
        tag=tag,
        url=url,
        published=_published_date(document.get("published_at")),
        body=body.strip() if isinstance(body, str) else "",
        name=name.strip() if isinstance(name, str) else "",
    )


def published_release(
    *,
    remote_url: str | None = None,
    repository: str = CANONICAL_REPOSITORY,
    fetch: Fetch | None = None,
) -> PublishedRelease:
    """The latest stable release published for this repository, or why not.

    `remote_url` is this installation's own remote, read so a release body is
    never taken from a repository that is not this one: a fork's release notes
    are not this app's changelog, and a remote that is not a GitHub repository at
    all is answered rather than guessed at. No remote at all is the installed
    distribution's ordinary case, where the release notes are the canonical
    ones.
    """

    named = github_repository(remote_url)
    if remote_url and named is None:
        return PublishedRelease(
            UNREADABLE,
            detail=(
                f"the remote {remote_url} is not a GitHub repository, so no "
                "published release is read for it"
            ),
        )
    if named is not None and named.casefold() != repository.casefold():
        return PublishedRelease(
            UNREADABLE,
            repository=named,
            detail=(
                f"the remote publishes {named}, which is not {repository}, so its "
                "release notes are not this app's changelog"
            ),
        )
    url = f"{GITHUB_API_ROOT}/repos/{repository}/releases/latest"
    request = fetch if fetch is not None else fetch_text
    try:
        text = request(url, FETCH_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return PublishedRelease(
                NONE,
                detail=(
                    f"{repository} publishes no stable release, so there is no "
                    "changelog to preview"
                ),
            )
        return PublishedRelease(
            UNREADABLE,
            detail=f"GitHub answered {exc.code} for the latest release of {repository}",
        )
    except (OSError, ValueError) as exc:
        return PublishedRelease(
            UNREADABLE,
            detail=f"the published release could not be read: {exc}",
        )
    try:
        release = _published_release_from(json.loads(text), repository)
    except (TypeError, ValueError) as exc:
        return PublishedRelease(
            UNREADABLE,
            detail=f"the published release could not be read: {exc}",
        )
    return PublishedRelease(
        FOUND,
        release,
        detail=(
            f"{repository} published release {release.version} as {release.tag} on "
            f"{release.published}"
        ),
    )


def skipped_published_release(detail: str) -> PublishedRelease:
    """The answer when the reader asked for no network at all."""

    return PublishedRelease(SKIPPED, detail=detail)


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
