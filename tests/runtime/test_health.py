"""A healthy project is a test only against a broken one.

The healthy case is asserted as "nothing blocked and nothing degraded" rather than as
the absence of a symptom.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import research_rag.runtime.health as health_module
import research_rag.runtime.version as version_module
from research_rag import gateway as vanilla_package
from research_rag.gateway import runtime as vanilla_runtime
from research_rag.project.config import ResearchConfig, resolve_config
from research_rag.retrieval.embeddings import (
    EMBEDDING_MODELS,
    EmbeddingModel,
    resolve_embedding_model,
)
from research_rag.retrieval.rerankers import (
    RERANKER_REQUIRED_FILES,
    resolve_reranker_model,
)
from research_rag.runtime.health import Check, HealthReport, health_report

# A check that disappears is as much a break as one that changes state.
EXPECTED_CHECKS = (
    "project_identity",
    "runtime_root",
    "direct_retrieval",
    "chunk_tokenizer",
    "embedding_model",
    "reranker_model",
    "lock",
    "generation",
    "capacity",
    "code_currency",
)

READY_STATUS: dict[str, Any] = {
    "ready": True,
    "stale": False,
    "generation_id": "20260101T000000Z-abcdef",
    "upgrade_reasons": [],
    "retained_generation_bytes": 1024,
    "message": "Ready.",
}

GIB = 1024**3


@pytest.fixture(autouse=True)
def _clear_the_runtime_cache():
    """Each test starts with no runtime answer cached from another test."""

    health_module._VANILLA_CACHE.clear()
    yield
    health_module._VANILLA_CACHE.clear()


def _cache_model(
    cache_root: Path, name: str, revision: str, required_files: tuple[str, ...]
) -> Path:
    snapshot = (
        cache_root / f"models--{name.replace('/', '--')}" / "snapshots" / revision
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename in required_files:
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")
    return snapshot


def _install_models(config: ResearchConfig) -> None:
    embedding = resolve_embedding_model(config.settings.embedding_model)
    _cache_model(
        config.model_cache_root,
        embedding.repository,
        embedding.revision,
        embedding.required_files,
    )
    reranker, revision = resolve_reranker_model(config.reranker_model)
    _cache_model(config.model_cache_root, reranker, revision, RERANKER_REQUIRED_FILES)


def _fake_runtime(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


def _report(
    config: ResearchConfig, status: dict[str, Any] | None = None
) -> HealthReport:
    return health_report(config, status or dict(READY_STATUS))


@pytest.fixture
def config(project: Path, tmp_path: Path) -> ResearchConfig:
    """A project whose caches are this test's own, not the machine's."""

    return _isolated(project, tmp_path)


def _isolated(project: Path, tmp_path: Path, **overrides: Any) -> ResearchConfig:
    return resolve_config(
        project_root=project,
        model_cache_root=tmp_path / "models",
        runtime_cache_root=tmp_path / "runtime-cache",
        **overrides,
    )


@pytest.fixture
def healthy(config: ResearchConfig, monkeypatch: pytest.MonkeyPatch) -> ResearchConfig:
    """A project whose dependencies are all in place."""

    _install_models(config)
    import research_rag.retrieval.direct as direct_module

    monkeypatch.setattr(direct_module, "tokenizer_cache_status", lambda _root: True)
    root = config.model_cache_root.parent / "runtime-cache" / "UltraRAG-test"
    _fake_runtime(root)
    (root / vanilla_runtime.MARKER_FILENAME).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _cache: root)
    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", lambda path: path)
    return config


def test_a_healthy_project_reports_nothing_to_act_on(healthy: ResearchConfig) -> None:
    report = _report(healthy)

    assert tuple(check.name for check in report.checks) == EXPECTED_CHECKS
    assert report.blocked_by == []
    assert report.degraded == []
    assert report.not_checked == []
    assert report.has_blocker is False


def test_normal_health_never_validates_optional_legacy_runtime(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_config: ResearchConfig) -> None:
        raise AssertionError("normal health reached the optional legacy runtime")

    monkeypatch.setattr(health_module, "_vanilla_runtime_check", refuse)
    assert _report(healthy).named("direct_retrieval").state == "ok"


def test_direct_dependency_pin_mismatch_is_actionable(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = health_module.version
    monkeypatch.setattr(
        health_module,
        "version",
        lambda name: "1.6.0" if name == "chonkie" else original(name),
    )
    check = _report(healthy).named("direct_retrieval")
    assert check.state == "blocked"
    assert "expected 1.7.0" in check.reason
    assert check.remedy_command == "uv sync --locked"


def test_missing_tokenizer_does_not_block_existing_indexes_offline(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import research_rag.retrieval.direct as direct_module

    monkeypatch.setattr(direct_module, "tokenizer_cache_status", lambda _root: False)
    report = _report(replace(healthy, settings=replace(healthy.settings, offline=True)))
    check = report.named("chunk_tokenizer")
    assert check.state == "warn"
    assert "Offline mode forbids fetching" in check.reason
    assert "--prefetch-models" in check.remedy_command
    assert report.blocked_by == []


def test_a_mismatched_runtime_is_blocked_and_names_the_file(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is named, not guessed."""

    class ValidationError(vanilla_runtime.RuntimeErrorBase):
        difference = SimpleNamespace(
            path="servers/memory/src/__pycache__/memory.cpython-311.pyc",
            kind="unexpected",
            mode="-rw-r--r--",
        )

    def refuse(path: Path) -> Path:
        raise ValidationError(
            "Managed runtime content hash mismatch: got dead, expected cafe. "
            "The tree differs at servers/memory/src/__pycache__/"
            "memory.cpython-311.pyc: unexpected in the installed tree, mode "
            "-rw-r--r--. Use a fresh cache location rather than modifying the "
            "snapshot."
        )

    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", refuse)
    health_module._VANILLA_CACHE.clear()
    report = health_module.HealthReport(
        (health_module._vanilla_runtime_check(healthy),)
    )

    blocked = report.named("vanilla_runtime")
    assert blocked.state == "blocked"
    assert "servers/memory/src/__pycache__/memory.cpython-311.pyc" in blocked.reason
    assert blocked.remedy_command == (
        f"research-rag --project-root {healthy.project_root} doctor --repair-runtime"
    )
    assert [entry["check"] for entry in report.blocked_by] == ["vanilla_runtime"]


def test_an_older_vanilla_still_blocks_and_says_it_cannot_name_the_file(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A release that cannot name a path is not a pass; it is a named limit."""

    class ValidationError(vanilla_runtime.RuntimeErrorBase):
        pass

    def refuse(path: Path) -> Path:
        raise ValidationError(
            "Managed runtime content hash mismatch: got dead, expected cafe. "
            "Use a fresh cache location rather than modifying the snapshot."
        )

    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", refuse)
    health_module._VANILLA_CACHE.clear()
    report = health_module.HealthReport(
        (health_module._vanilla_runtime_check(healthy),)
    )

    blocked = report.named("vanilla_runtime")
    assert blocked.state == "blocked"
    assert "does not name the differing file" in blocked.reason
    assert "research-rag-runtime --offline" in blocked.reason
    assert report.has_blocker is True


@pytest.mark.parametrize("release", ["older", "broken"])
def test_a_vanilla_release_that_cannot_be_used_is_unknown(
    config: ResearchConfig,
    monkeypatch: pytest.MonkeyPatch,
    release: str,
) -> None:
    """Neither an old release nor a broken install may read as healthy."""

    class Broken(SimpleNamespace):
        def __getattr__(self, name: str) -> Any:
            raise ImportError("the gateway runtime is not usable")

    monkeypatch.setattr(
        vanilla_package,
        "runtime",
        Broken() if release == "broken" else SimpleNamespace(),
    )
    health_module._VANILLA_CACHE.clear()

    report = health_module.HealthReport((health_module._vanilla_runtime_check(config),))

    checked = report.named("vanilla_runtime")
    assert checked.state == "unknown"
    assert report.not_checked == ["vanilla_runtime"]
    # Unchecked is its own state, and never appears among the conditions a caller may
    # dismiss as fine.
    disclosed = {entry["check"] for entry in report.blocked_by + report.degraded}
    assert "vanilla_runtime" not in disclosed
    assert checked.reason


def test_a_missing_runtime_blocks_offline_and_warns_online(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = config.model_cache_root.parent / "no-such-cache"
    monkeypatch.setattr(
        vanilla_runtime, "managed_runtime_path", lambda _cache: absent / "UltraRAG-x"
    )
    health_module._VANILLA_CACHE.clear()

    offline = resolve_config(config.project_root, offline=True)
    online = resolve_config(config.project_root)
    offline_check = health_module._vanilla_runtime_check(offline)
    online_check = health_module._vanilla_runtime_check(online)

    assert offline_check.state == "blocked"
    assert "offline mode forbids downloading it" in offline_check.reason
    assert online_check.state == "warn"
    assert "downloads it" in online_check.reason


def test_a_missing_embedding_model_blocks_and_a_missing_reranker_degrades(
    config: ResearchConfig,
) -> None:
    report = _report(
        _isolated(config.project_root, config.model_cache_root, offline=True)
    )

    embedding = report.named("embedding_model")
    assert embedding.state == "blocked"
    assert "no build can be dense" in embedding.reason
    # The reranker only removes the ranking, so it degrades instead of blocking.
    reranker = report.named("reranker_model")
    assert reranker.state == "warn"
    assert [entry["check"] for entry in report.degraded] == ["reranker_model"]
    blocked = {entry["check"] for entry in report.blocked_by}
    assert "embedding_model" in blocked
    assert "reranker_model" not in blocked
    assert reranker.remedy_command is not None
    assert "--prefetch-models" in reranker.remedy_command


def test_cached_models_are_not_reported(healthy: ResearchConfig) -> None:
    report = _report(healthy)

    assert report.named("embedding_model").state == "ok"
    assert report.named("reranker_model").state == "ok"
    assert "bge-small-en-v1.5" in report.named("embedding_model").reason


@pytest.mark.parametrize("facts", EMBEDDING_MODELS, ids=lambda model: model.name)
def test_health_checks_each_real_repository_and_its_required_files(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch, facts: EmbeddingModel
) -> None:
    monkeypatch.setattr(health_module, "resolve_embedding_model", lambda name: facts)
    snapshot = _cache_model(
        config.model_cache_root, facts.repository, facts.revision, facts.required_files
    )
    assert _report(config).named("embedding_model").state == "ok"
    # An empty directory or missing external ONNX data is not a cached model.
    (snapshot / facts.required_files[-1]).unlink()
    assert _report(config).named("embedding_model").state == "warn"


@pytest.mark.parametrize("offline,state", [(False, "warn"), (True, "blocked")])
@pytest.mark.parametrize("wrong", ["repository", "revision"])
def test_a_snapshot_from_another_repository_or_revision_cannot_satisfy_the_pin(
    config: ResearchConfig,
    offline: bool,
    state: str,
    wrong: str,
) -> None:
    config = resolve_config(
        config.project_root, model_cache_root=config.model_cache_root, offline=offline
    )
    facts = resolve_embedding_model(config.settings.embedding_model)
    _cache_model(
        config.model_cache_root,
        facts.name if wrong == "repository" else facts.repository,
        "52398278842ec682c6f32300af41344b1c0b0bb2"
        if wrong == "revision"
        else facts.revision,
        facts.required_files,
    )
    check = _report(config).named("embedding_model")
    assert check.state == state
    assert facts.repository in check.reason
    assert facts.revision in check.reason
    assert "--prefetch-models" in check.remedy_command


@pytest.mark.integration
def test_another_process_holding_the_project_is_a_degradation(
    healthy: ResearchConfig,
) -> None:
    held = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (healthy.state_root / "project.lock").write_text(
            f"{held.pid}\n", encoding="utf-8"
        )
        report = _report(healthy)
    finally:
        held.terminate()
        held.wait(timeout=30)

    check = report.named("lock")
    assert check.state == "warn"
    assert str(held.pid) in check.reason
    assert check.remedy_command == (
        f"research-rag --project-root {healthy.project_root} stop --servers"
    )
    assert [entry["check"] for entry in report.degraded] == ["lock"]


def test_a_stale_lock_left_by_a_dead_process_is_not_reported(
    healthy: ResearchConfig,
) -> None:
    (healthy.state_root / "project.lock").write_text("4194303\n", encoding="utf-8")

    report = _report(healthy)

    assert report.named("lock").state == "ok"
    assert report.degraded == []


@pytest.mark.parametrize(
    ("free", "state"),
    [(2 * GIB, "blocked"), (9 * GIB, "warn"), (64 * GIB, "ok")],
)
def test_free_space_is_compared_with_the_build_it_has_to_fit(
    healthy: ResearchConfig,
    monkeypatch: pytest.MonkeyPatch,
    free: int,
    state: str,
) -> None:
    monkeypatch.setattr(
        health_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=free, total=free * 2, used=free),
    )
    status = {**READY_STATUS, "retained_generation_bytes": 8 * GIB}

    report = _report(healthy, status)

    assert report.named("capacity").state == state
    assert str(free) in report.named("capacity").reason


def test_a_generation_nobody_ingested_blocks_and_says_what_to_run(
    healthy: ResearchConfig,
) -> None:
    report = _report(
        healthy,
        {
            "ready": False,
            "stale": True,
            "upgrade_reasons": [],
            "retained_generation_bytes": 0,
            "message": "No knowledge-base generation exists; call ingest.",
        },
    )

    check = report.named("generation")
    assert check.state == "blocked"
    assert check.reason == "No knowledge-base generation exists; call ingest."
    assert check.remedy_command == (
        f"research-rag --project-root {healthy.project_root} ingest"
    )


def test_a_generation_built_by_an_older_policy_warns(healthy: ResearchConfig) -> None:
    report = _report(
        healthy,
        {**READY_STATUS, "upgrade_reasons": ["retrieval_policy", "embedding_model"]},
    )

    check = report.named("generation")
    assert check.state == "warn"
    assert "retrieval_policy" in check.reason
    assert [entry["check"] for entry in report.degraded] == ["generation"]


def test_a_stale_generation_warns(healthy: ResearchConfig) -> None:
    report = _report(healthy, {**READY_STATUS, "stale": True})

    check = report.named("generation")
    assert check.state == "warn"
    assert "ingest indexes them" in check.reason


def test_a_state_root_claimed_by_another_project_blocks(
    project: Path, tmp_path: Path
) -> None:
    relocated = tmp_path / "relocated"
    config = resolve_config(
        project_root=project,
        runtime_root=relocated,
        model_cache_root=tmp_path / "models",
    )
    marker = relocated / ".research-rag-runtime.json"
    claimed = json.loads(marker.read_text(encoding="utf-8"))
    claimed["project_id"] = "00000000-0000-0000-0000-000000000000"
    marker.write_text(json.dumps(claimed), encoding="utf-8")

    report = _report(config)

    check = report.named("project_identity")
    assert check.state == "blocked"
    assert "belongs to another project" in check.reason
    assert report.has_blocker is True


def test_an_app_older_than_the_installed_code_warns(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(version_module, "APP_VERSION", "0.0.1")

    report = _report(healthy)

    check = report.named("code_currency")
    assert check.state == "warn"
    assert "0.0.1" in check.reason
    assert "Restart this app" in check.reason


def test_a_server_answering_from_another_checkout_warns(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two checkouts report one version, so the paths are the only evidence."""

    monkeypatch.setattr(
        version_module,
        "install_origin",
        lambda: Path("/home/somebody/else/research-rag"),
    )
    monkeypatch.setattr(
        version_module, "nearest_checkout", lambda: Path("/tmp/a-different-checkout")
    )

    report = _report(healthy)

    check = report.named("code_currency")
    assert check.state == "warn"
    assert "/home/somebody/else/research-rag" in check.reason
    assert "/tmp/a-different-checkout" in check.reason
    assert check.remedy_command


def test_one_checkout_is_not_reported_as_drift(
    healthy: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case must stay `ok`, or the warning stops being read."""

    single = Path("/srv/research-rag")
    monkeypatch.setattr(version_module, "install_origin", lambda: single)
    monkeypatch.setattr(version_module, "nearest_checkout", lambda: single)

    report = _report(healthy)

    assert report.named("code_currency").state == "ok"


def test_the_tree_is_read_once_per_marker(
    healthy: ResearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tree hash reads every byte of it, so a repeated status must not repeat it."""

    reads: list[Path] = []
    root = Path(healthy.model_cache_root).parent / "runtime-cache" / "UltraRAG-test"

    def counted(path: Path) -> Path:
        reads.append(path)
        return path

    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", counted)
    health_module._VANILLA_CACHE.clear()

    first = health_module._vanilla_runtime_check(healthy)
    second = health_module._vanilla_runtime_check(healthy)

    assert reads == [root]
    assert first is second

    # A reinstalled runtime is written with a new marker, and that re-reads the tree.
    (root / vanilla_runtime.MARKER_FILENAME).write_text(
        '{"repaired": true}\n', encoding="utf-8"
    )
    third = health_module._vanilla_runtime_check(healthy)

    assert reads == [root, root]
    assert third is not first


def test_every_check_carries_a_reason_and_a_remedy_when_it_blocks(
    healthy: ResearchConfig,
) -> None:
    report = _report(
        healthy,
        {
            "ready": False,
            "stale": True,
            "upgrade_reasons": [],
            "message": "call ingest",
        },
    )

    for check in report.checks:
        assert check.reason
        assert set(check.as_dict()) == {"check", "state", "reason", "remedy_command"}
        if check.blocked:
            assert check.remedy_command


def test_the_report_serializes_to_the_status_fields(healthy: ResearchConfig) -> None:
    report = _report(healthy)

    fields = report.as_status_fields()
    assert set(fields) == {"checks", "blocked_by", "degraded", "not_checked"}
    assert fields["checks"][0]["check"] == "project_identity"
    assert json.loads(json.dumps(fields)) == fields
    assert Check("x", "ok", "fine").as_disclosure() == {
        "check": "x",
        "reason": "fine",
        "remedy": None,
    }


def test_docling_backend_readiness_is_reported(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _isolated(
        project, tmp_path, settings_overrides=["ingestion.pdf_backend=docling"]
    )

    env = SimpleNamespace(root=tmp_path / "env", versions={"docling": "2.135.0"})

    # Environment ready, models absent, online: a lazy fetch is described.
    monkeypatch.setattr(health_module, "resolve_docling_env", lambda _config: env)
    monkeypatch.setattr(health_module, "docling_cache_has_models", lambda _root: False)
    check = _report(config).named("docling_backend")
    assert check is not None and check.state == "warn"
    assert "--prefetch-models" in (check.remedy_command or "")
    assert "lazily" in check.reason

    # Environment ready, models absent, offline: it cannot fetch.
    offline = _isolated(
        project,
        tmp_path,
        settings_overrides=["ingestion.pdf_backend=docling", "runtime.offline=true"],
    )
    check = _report(offline).named("docling_backend")
    assert check is not None and check.state == "warn"
    assert "cannot fetch" in check.reason

    # No environment, online: it is built lazily on first use.
    monkeypatch.setattr(health_module, "resolve_docling_env", lambda _config: None)
    check = _report(config).named("docling_backend")
    assert check is not None and check.state == "warn"
    assert "lazily" in check.reason
    assert _report(config).blocked_by == []

    # No environment, offline: it must be provisioned.
    check = _report(offline).named("docling_backend")
    assert check is not None and check.state == "warn"
    assert "runtime.offline forbids" in check.reason

    # Environment ready and models cached.
    monkeypatch.setattr(health_module, "resolve_docling_env", lambda _config: env)
    monkeypatch.setattr(health_module, "docling_cache_has_models", lambda _root: True)
    check = _report(config).named("docling_backend")
    assert check is not None and check.state == "ok"


def test_custom_backend_adds_no_docling_check(healthy: ResearchConfig) -> None:
    with pytest.raises(KeyError):
        _report(healthy).named("docling_backend")
