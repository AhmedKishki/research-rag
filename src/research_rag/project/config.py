from __future__ import annotations

import contextlib
import filecmp
import json
import os
import shlex
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_path

from .settings import SETTINGS, EffectiveSettings, sources_for
from .settings_layers import SettingsError, resolve_settings

# The one console script this product installs, named because the health report
# prints commands an operator can paste.
CLI_COMMAND = "research-rag"

# The gateway this app spawns, installed by this package beside the command above.
# It is named for this app rather than for the project it proxies, because the
# reader pastes it and has to be able to see which product it belongs to.
GATEWAY_EXECUTABLE = "research-rag-gateway"

# Where a remedy points when the reader has to supply the directory themselves.
# It is a path-shaped token with no quoting, so the printed command stays
# pasteable after one edit.
PLACEHOLDER_PROJECT_ROOT = "/path/to/project"


class ConfigurationError(ValueError):
    pass


# The first loopback port an app claims. It is a default rather than a fixed choice,
# because an app is started from a terminal and a second project must not fail where
# a port was free.
DEFAULT_UI_PORT = 5051
_RUNTIME_MARKER = ".research-rag-runtime.json"
# Where this project recorded that its derived state was relocated to. Machine-local,
# and absent for a project that keeps its state inside its own root.
RUNTIME_ROOT_RECORD = "runtime-root"

# A server this project starts for itself is a managed child: it may not serve
# the browser UI, because only a server an operator started may. Every child
# transport marks the server it starts and drops the variables that describe a
# top-level invocation, so no exported setting can turn one server into a chain
# of them.
MANAGED_CHILD_ENV = "RESEARCH_RAG_MANAGED_CHILD"
# The process that started a child. A child orphaned before it can read its own
# parent, a client that dies between spawning and startup, still knows who owned
# it and can end itself with them.
OWNER_PID_ENV = "RESEARCH_RAG_OWNER_PID"
TOP_LEVEL_ONLY_ENV = ("RESEARCH_RAG_UI_PORT",)


def child_process_environment() -> dict[str, str]:
    """The environment for a server this process starts.

    The child inherits this process's environment except the top-level-only variables,
    plus the marker that makes it refuse to host a UI and the identity of its owner.
    An inherited owner id is overwritten.
    """

    environment = dict(os.environ)
    for name in TOP_LEVEL_ONLY_ENV:
        environment.pop(name, None)
    environment[MANAGED_CHILD_ENV] = "1"
    environment[OWNER_PID_ENV] = str(os.getpid())
    return environment


def declared_owner_pid() -> int | None:
    raw = os.environ.get(OWNER_PID_ENV)
    if raw is None or not raw.strip().isdigit():
        return None
    owner = int(raw)
    return owner if owner > 1 else None


def is_managed_child() -> bool:
    """Whether another server in this project started this process."""

    return os.environ.get(MANAGED_CHILD_ENV) == "1"


def apply_process_priority(nice: int) -> int | None:
    """Raise this process's niceness to ``nice``, and report what it became.

    One call covers the whole process tree, because children inherit the value. The
    increment is the difference from the current value, and a process at or above the
    target is left alone, making a second call in a child harmless.

    A refusal returns ``None`` rather than raising.
    """
    if nice <= 0:
        return None
    try:
        current = os.nice(0)
        if nice <= current:
            return current
        return os.nice(nice - current)
    except OSError:
        return None


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    project_root: Path
    source_root: Path
    state_root: Path
    portable_root: Path
    project_id: str
    project_name: str
    vanilla_executable: Path
    runtime_cache_root: Path | None
    model_cache_root: Path
    settings: EffectiveSettings
    settings_provenance: dict[str, str] = field(default_factory=dict)
    runtime_root: Path | None = None

    # The tunables live in one place — the merged settings — and these names
    # read through to them, so nothing has to be kept in step by hand.

    @property
    def offline(self) -> bool:
        return self.settings.offline

    @property
    def nice(self) -> int:
        return self.settings.nice

    @property
    def log_level(self) -> str:
        return self.settings.log_level

    @property
    def tool_detail(self) -> str:
        return self.settings.tool_detail

    @property
    def embedding_threads(self) -> int | None:
        return self.settings.embedding_threads

    @property
    def dense_backend(self) -> str:
        return self.settings.dense_backend

    @property
    def reranker_model(self) -> str:
        return self.settings.reranker_model

    @property
    def generations_root(self) -> Path:
        return self.state_root / "generations"

    @property
    def logs_root(self) -> Path:
        return self.state_root / "logs"

    @property
    def ultrarag_workspace(self) -> Path:
        return self.state_root / "ultrarag-runtime"

    @property
    def models_root(self) -> Path:
        """Compatibility name for the shared model-binary cache."""

        return self.model_cache_root

    @property
    def legacy_models_root(self) -> Path:
        return self.state_root / "models"

    @property
    def staging_root(self) -> Path:
        return self.state_root / "staging"

    @property
    def failures_root(self) -> Path:
        return self.state_root / "failures"

    @property
    def metadata_path(self) -> Path:
        return self.portable_root / "source-metadata.json"

    @property
    def source_exclusions_path(self) -> Path:
        return self.portable_root / "source-exclusions.json"

    @property
    def chunk_exclusions_path(self) -> Path:
        return self.portable_root / "chunk-exclusions.json"

    @property
    def source_catalog_path(self) -> Path:
        return self.portable_root / "source-catalog.json"

    @property
    def project_config_path(self) -> Path:
        return self.portable_root / "project.json"

    @property
    def current_path(self) -> Path:
        return self.state_root / "current.json"


def configured_source_directory(project_root: str | Path) -> str:
    descriptor_path = (
        Path(project_root).expanduser().resolve() / ".research-rag" / "project.json"
    )
    if not descriptor_path.exists():
        return "sources"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Invalid portable project descriptor: {descriptor_path}"
        ) from exc
    source_directory = (
        descriptor.get("source_directory")
        if isinstance(descriptor, dict) and descriptor.get("schema_version") == 1
        else None
    )
    if not isinstance(source_directory, str) or not source_directory:
        raise ConfigurationError(
            f"Portable project descriptor has no source_directory: {descriptor_path}"
        )
    return source_directory


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _contains_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _has_payload(path: Path) -> bool:
    return path.is_dir() and any(
        item.is_file() or item.is_symlink() for item in path.rglob("*")
    )


def _remove_empty_tree(path: Path) -> None:
    for child in sorted(
        path.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        child.rmdir()
    path.rmdir()


def _review_state_files(
    runtime_root: Path,
    project_state_root: Path,
) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for name in (
        "source-metadata.json",
        "source-exclusions.json",
        "source-catalog.json",
    ):
        legacy = runtime_root / name
        destination = project_state_root / name
        if not legacy.exists():
            continue
        if not legacy.is_file() or legacy.is_symlink():
            raise ConfigurationError(f"Legacy review state is not a file: {legacy}")
        if destination.is_symlink() or (
            destination.exists()
            and (
                not destination.is_file()
                or not filecmp.cmp(legacy, destination, shallow=False)
            )
        ):
            raise ConfigurationError(
                "Conflicting reviewed state exists in both the legacy and "
                f"consolidated locations: {name}"
            )
        result.append((legacy, destination))
    return result


def _move_review_state(runtime_root: Path, project_state_root: Path) -> None:
    files = _review_state_files(runtime_root, project_state_root)
    for legacy, destination in files:
        if destination.exists():
            legacy.unlink()
        else:
            os.replace(legacy, destination)


def _migrate_legacy_runtime(
    project: Path,
    project_state_root: Path,
    runtime_root: Path,
) -> None:
    """Move the former .ultrarag/research tree beneath .research-rag once."""

    legacy_root = project / ".ultrarag" / "research"
    if legacy_root.is_symlink():
        raise ConfigurationError(
            f"Legacy research state cannot be a symlink: {legacy_root}"
        )
    if legacy_root.exists() and not legacy_root.is_dir():
        raise ConfigurationError(
            f"Legacy research state is not a directory: {legacy_root}"
        )

    if legacy_root.is_dir():
        _review_state_files(legacy_root, project_state_root)
        if runtime_root.exists() and _has_payload(runtime_root):
            if _has_payload(legacy_root):
                raise ConfigurationError(
                    "Research runtime state exists in both .ultrarag/research and "
                    ".research-rag/runtime; resolve the duplicate state before "
                    "starting the server."
                )
            _remove_empty_tree(legacy_root)
        else:
            if runtime_root.exists():
                _remove_empty_tree(runtime_root)
            os.replace(legacy_root, runtime_root)

    if runtime_root.is_dir():
        _move_review_state(runtime_root, project_state_root)

    legacy_parent = project / ".ultrarag"
    if legacy_parent.is_dir():
        # A non-empty legacy parent is the expected case: it can hold files this
        # server never wrote, and leaving it is safe where removing it is not.
        with contextlib.suppress(OSError):
            legacy_parent.rmdir()


def _write_project_descriptor(
    descriptor_path: Path,
    *,
    project_id: str,
    name: str,
    source_directory: str,
) -> None:
    """Replace the descriptor atomically, so a reader never sees a partial one."""
    temporary = descriptor_path.with_name(
        f".{descriptor_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_id": project_id,
                "name": name,
                "source_directory": source_directory,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, descriptor_path)


def _normalize_project_name(name: str) -> str:
    """The name to record, refusing one that is not a usable project name."""

    normalized = name.strip()
    if not normalized:
        raise ConfigurationError("A project name cannot be empty")
    if any(character in normalized for character in "\r\n\t"):
        raise ConfigurationError("A project name cannot contain line breaks or tabs")
    return normalized


def _initialize_portable_project(
    project: Path,
    portable_root: Path,
    source_directory: str,
    name: str | None = None,
) -> tuple[str, str]:
    """Create or validate the portable project descriptor.

    A created project takes ``name``, falling back to the directory name; an existing
    one keeps its recorded name. The stable ``project_id`` is never rewritten, so
    naming cannot invalidate a generation.
    """

    portable_root.mkdir(parents=True, exist_ok=True)
    (portable_root / "bundles").mkdir(exist_ok=True)
    descriptor_path = portable_root / "project.json"
    requested_name = _normalize_project_name(name) if name is not None else None
    if descriptor_path.exists():
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Invalid portable project descriptor: {descriptor_path}"
            ) from exc
        if not isinstance(descriptor, dict) or descriptor.get("schema_version") != 1:
            raise ConfigurationError(
                f"Unsupported portable project descriptor: {descriptor_path}"
            )
        project_id = descriptor.get("project_id")
        project_name = descriptor.get("name")
        configured_sources = descriptor.get("source_directory")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ConfigurationError("Portable project descriptor has no project_id")
        if not isinstance(project_name, str) or not project_name.strip():
            raise ConfigurationError("Portable project descriptor has no project name")
        if configured_sources != source_directory:
            raise ConfigurationError(
                "Configured source directory differs from .research-rag/project.json: "
                f"{source_directory!r} != {configured_sources!r}"
            )
        if requested_name is not None and requested_name != project_name:
            _write_project_descriptor(
                descriptor_path,
                project_id=project_id,
                name=requested_name,
                source_directory=source_directory,
            )
            project_name = requested_name
    else:
        project_id = str(uuid.uuid4())
        project_name = requested_name or project.name
        _write_project_descriptor(
            descriptor_path,
            project_id=project_id,
            name=project_name,
            source_directory=source_directory,
        )

    return project_id, project_name


def project_command(project_root: str | Path, *arguments: str) -> str:
    """A copy-pasteable command naming one project, quoting what needs it."""

    parts = (CLI_COMMAND, "--project-root", str(project_root), *arguments)
    return " ".join(shlex.quote(part) for part in parts)


def project_selection_command(selector: str, *arguments: str) -> str:
    """A copy-pasteable command naming a project by its recorded name.

    Where `project_command` needs a directory this one needs a name, which is
    what a reader has when the project is recorded and its directory is a fact of
    the machine rather than something they typed.
    """

    parts = (CLI_COMMAND, "--project", selector, *arguments)
    return " ".join(shlex.quote(part) for part in parts)


def initialise_command(
    project_name: str,
    project_root: str | Path = PLACEHOLDER_PROJECT_ROOT,
) -> str:
    """A copy-pasteable command that creates a project under one recorded name.

    A name is all an agent's client entry carries, so this is the command that
    turns that entry into a served project on one machine. The directory is the
    one thing the entry cannot name, so an unset one is left to the reader.
    """

    return project_command(project_root, "init", "--name", project_name)


def runtime_root_claim_problem(candidate: Path, project_id: str) -> str | None:
    """Why this directory may not hold this project's derived state.

    A relocated root carries a marker naming its owning project, so two projects
    cannot silently share one set of generations and an unrelated directory is never
    adopted. The default in-project root needs no marker: the project owns it. This
    reads, so a health report can ask the same question.
    """

    if not candidate.exists():
        return None
    if not candidate.is_dir():
        return f"Runtime root is not a directory: {candidate}"
    marker_path = candidate / _RUNTIME_MARKER
    if not marker_path.is_file():
        try:
            has_payload = any(candidate.iterdir())
        except OSError:
            return f"Runtime root is not readable: {candidate}"
        if has_payload:
            return (
                "Runtime root is not empty and carries no project marker: "
                f"{candidate}. Point --runtime-root at an empty directory or at "
                "the directory this project already uses."
            )
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"Invalid runtime-root marker: {marker_path}"
    owner = marker.get("project_id") if isinstance(marker, dict) else None
    if owner != project_id:
        return (
            f"Runtime root belongs to another project ({owner!r}; this "
            f"project is {project_id!r}): {candidate}"
        )
    return None


def _prepare_runtime_root(
    candidate: Path,
    *,
    project: Path,
    project_id: str,
    marker_required: bool,
) -> Path:
    """Claim or validate the directory that holds disposable derived state.

    A relocated root is marked as `runtime_root_claim_problem` describes. The
    default in-project root needs no marker.
    """

    if not marker_required:
        # The default root is inside the project, so the project owns it by
        # construction and there is nothing to claim or validate.
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    problem = runtime_root_claim_problem(candidate, project_id)
    if problem is not None:
        raise ConfigurationError(problem)
    marker_path = candidate / _RUNTIME_MARKER
    candidate.mkdir(parents=True, exist_ok=True)
    if not marker_path.is_file():
        temporary = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "project_id": project_id,
                    "project_root": str(project),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, marker_path)
    return candidate


def resolve_config(
    project_root: str | Path,
    *,
    source_directory: str = "sources",
    project_name: str | None = None,
    vanilla_executable: str | Path | None = None,
    runtime_cache_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    model_cache_root: str | Path | None = None,
    offline: bool | None = None,
    log_level: str | None = None,
    dense_backend: str | None = None,
    embedding_threads: int | str | None = None,
    tool_detail: str | None = None,
    reranker_model: str | None = None,
    config_path: str | Path | None = None,
    settings_overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> ResearchConfig:
    """The project boundary, then every settings layer.

    Every keyword above is the *command-line* layer: passing one overrides the
    environment and the config files for this invocation.

    ``project_name`` is the exception: a project name is not a setting, and only
    `init` supplies one.
    """
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ConfigurationError(f"Project root is not a directory: {project}")

    source_argument = Path(source_directory)
    if source_argument.is_absolute():
        raise ConfigurationError("--source-directory must be project-relative")
    sources = (project / source_argument).resolve()
    if not _within(sources, project):
        raise ConfigurationError(
            f"Source directory escapes the project root: {source_directory}"
        )

    portable = (project / ".research-rag").resolve()
    if not _within(portable, project):
        raise ConfigurationError(f"Portable state escapes the project root: {portable}")
    default_state = (portable / "runtime").resolve()
    if not _within(default_state, portable):
        raise ConfigurationError(
            f"Research runtime escapes its project state: {default_state}"
        )
    relocated = runtime_root is not None
    custom_state = (
        Path(runtime_root).expanduser().resolve() if relocated else default_state
    )
    if relocated:
        if not Path(str(runtime_root)).expanduser().is_absolute():
            raise ConfigurationError("--runtime-root must be an absolute path")
        if custom_state in {project, portable}:
            raise ConfigurationError(
                "--runtime-root must not be the project root or its .research-rag"
            )

    executable = (
        Path(vanilla_executable or Path(sys.executable).parent / GATEWAY_EXECUTABLE)
        .expanduser()
        .absolute()
    )
    if not executable.is_file():
        raise ConfigurationError(
            f"{GATEWAY_EXECUTABLE} executable was not found: {executable}"
        )

    overrides = list(settings_overrides)
    for key, value in (
        ("runtime.model_cache_root", model_cache_root),
        ("runtime.log_level", log_level),
        ("runtime.tool_detail", tool_detail),
        ("runtime.embedding_threads", embedding_threads),
        ("dense.backend", dense_backend),
        ("dense.reranker_model", reranker_model),
    ):
        if value is not None and str(value).strip():
            overrides.append(f"{key}={value}")
    if offline:
        overrides.append("runtime.offline=true")

    try:
        values, provenance = resolve_settings(
            SETTINGS,
            sources_for(project),
            config_path=config_path,
            overrides=overrides,
            environ=environ,
        )
        settings = EffectiveSettings.from_values(values)
    except SettingsError as exc:
        # One error type for the caller: a settings layer problem is a
        # configuration problem, whether it came from a file, the environment, or
        # the command line.
        raise ConfigurationError(str(exc)) from exc

    portable.mkdir(parents=True, exist_ok=True)
    if not relocated:
        _migrate_legacy_runtime(project, portable, default_state)

    normalized_source_directory = source_argument.as_posix()
    # `project_name` is the caller's request, and the descriptor is authoritative
    # about the name that was actually recorded, so it is read back here.
    project_id, project_name = _initialize_portable_project(
        project,
        portable,
        normalized_source_directory,
        project_name,
    )
    state = _prepare_runtime_root(
        custom_state,
        project=project,
        project_id=project_id,
        marker_required=relocated,
    )

    cache = (
        Path(runtime_cache_root).expanduser().resolve()
        if runtime_cache_root is not None
        else None
    )
    # `research-rag` is the MCP server's directory name, kept so this
    # app reads the same embedding and reranker binaries that server
    # downloaded. A name matching this product would leave every existing cache
    # behind and re-download both models on the first build, silently, because the
    # first build is the only thing that needs them. `USER_CONFIG_DIRECTORY` and
    # `SETTINGS_ENVIRONMENT_PREFIX` are inherited for the same reason and cost the
    # user their settings; `settings.py` states that. `tests/test_data_roots.py`
    # fails if any of the three changes.
    configured_model_cache = (
        settings.model_cache_root
        if settings.model_cache_root is not None
        else user_cache_path("research-rag", appauthor=False) / "models"
    ).resolve()
    legacy_model_cache = state / "models"
    if (
        settings.offline
        and settings.model_cache_root is None
        and not _contains_files(configured_model_cache)
        and _contains_files(legacy_model_cache)
    ):
        configured_model_cache = legacy_model_cache

    state.mkdir(parents=True, exist_ok=True)
    (state / "generations").mkdir(exist_ok=True)
    (state / "logs").mkdir(exist_ok=True)
    (state / "staging").mkdir(exist_ok=True)
    (state / "failures").mkdir(exist_ok=True)
    (state / "ultrarag-runtime").mkdir(exist_ok=True)
    configured_model_cache.mkdir(parents=True, exist_ok=True)
    record_runtime_root(portable, state if relocated else None)

    return ResearchConfig(
        project_root=project,
        source_root=sources,
        state_root=state,
        portable_root=portable,
        project_id=project_id,
        project_name=project_name,
        vanilla_executable=executable,
        runtime_cache_root=cache,
        model_cache_root=configured_model_cache,
        settings=settings,
        settings_provenance=dict(provenance),
        runtime_root=custom_state if relocated else None,
    )


def resolve_source_reference(config: ResearchConfig, relative_path: str) -> Path:
    reference = Path(relative_path)
    if reference.is_absolute():
        raise ConfigurationError(
            "Source paths must be relative to the sources directory"
        )
    candidate = (config.source_root / reference).resolve()
    if not _within(candidate, config.source_root):
        raise ConfigurationError(
            f"Source path escapes the sources directory: {relative_path}"
        )
    return candidate


def record_runtime_root(portable_root: Path, runtime_root: Path | None) -> None:
    """Record where this project's derived state lives, or that it is not relocated.

    Written whenever the project's configuration is resolved, so a later command
    that has only the project's root to go on finds the state a relocated runtime
    put elsewhere rather than reporting a project that holds nothing.
    """

    record = portable_root / RUNTIME_ROOT_RECORD
    if runtime_root is None:
        record.unlink(missing_ok=True)
        return
    # Not suppressed: a project whose portable root cannot be written is one whose
    # relocated state a later command would report as missing, and a silent miss
    # there reads as an empty project rather than a broken one.
    record.write_text(f"{runtime_root}\n", encoding="utf-8")


def recorded_runtime_root(portable_root: Path) -> Path | None:
    """Where this project's derived state was relocated to, or None."""

    try:
        text = (portable_root / RUNTIME_ROOT_RECORD).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(text) if text else None
