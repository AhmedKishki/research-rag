"""Model pins reach the hub and FastEmbed's explicit snapshot-path interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from huggingface_hub.errors import LocalEntryNotFoundError

import research_rag.retrieval.model_runtime as runtime
from research_rag.retrieval.embeddings import EMBEDDING_MODELS, EmbeddingModel
from research_rag.retrieval.model_cache import snapshot_path
from research_rag.retrieval.rerankers import RERANKER_MODELS, RERANKER_REQUIRED_FILES


def _cache(snapshot: Path, filenames: tuple[str, ...]) -> Path:
    for filename in filenames:
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")
    return snapshot


@pytest.mark.parametrize("facts", EMBEDDING_MODELS, ids=lambda model: model.name)
def test_embedding_repository_and_files_match_fastembed_registry(
    facts: EmbeddingModel,
) -> None:
    description = next(
        item
        for item in TextEmbedding.list_supported_models()
        if item["model"] == facts.name
    )
    assert description["sources"]["hf"] == facts.repository
    assert description["model_file"] == facts.model_file
    assert tuple(description["additional_files"]) == facts.additional_files
    assert description["dim"] == facts.dimension


@pytest.mark.parametrize("name", RERANKER_MODELS)
def test_reranker_repository_and_files_match_fastembed_registry(name: str) -> None:
    description = next(
        item
        for item in TextCrossEncoder.list_supported_models()
        if item["model"] == name
    )
    assert description["sources"]["hf"] == name
    assert description["model_file"] in RERANKER_REQUIRED_FILES
    assert set(description["additional_files"]) <= set(RERANKER_REQUIRED_FILES)


@pytest.mark.parametrize("facts", EMBEDDING_MODELS, ids=lambda model: model.name)
@pytest.mark.parametrize("offline", [False, True])
def test_embedder_downloads_exact_commit_and_loads_only_its_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    facts: EmbeddingModel,
    offline: bool,
) -> None:
    expected = snapshot_path(tmp_path, facts.repository, facts.revision)
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(_cache(expected, facts.required_files))

    monkeypatch.setattr(runtime, "snapshot_download", download)
    monkeypatch.setattr(runtime, "TextEmbedding", lambda **kwargs: kwargs)
    result = runtime._load_embedder(
        tmp_path, offline=offline, model=facts.name, threads=2
    )

    assert calls == [
        {
            "repo_id": facts.repository,
            "revision": facts.revision,
            "cache_dir": str(tmp_path),
            "allow_patterns": list(facts.required_files),
            "local_files_only": offline,
        }
    ]
    assert result == {
        "model_name": facts.name,
        "cache_dir": str(tmp_path),
        "cuda": False,
        "local_files_only": True,
        "specific_model_path": str(expected),
        "threads": 2,
    }


@pytest.mark.parametrize("name,revision", RERANKER_MODELS.items())
def test_reranker_downloads_exact_commit_and_loads_only_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, revision: str
) -> None:
    expected = snapshot_path(tmp_path, name, revision)
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(_cache(expected, RERANKER_REQUIRED_FILES))

    monkeypatch.setattr(runtime, "snapshot_download", download)
    monkeypatch.setattr(runtime, "TextCrossEncoder", lambda **kwargs: kwargs)
    result = runtime._load_cross_encoder(tmp_path, offline=False, model=name)

    assert calls[0]["repo_id"] == name
    assert calls[0]["revision"] == revision
    assert calls[0]["local_files_only"] is False
    assert calls[0]["allow_patterns"] == list(RERANKER_REQUIRED_FILES)
    assert result["specific_model_path"] == str(expected)
    assert result["local_files_only"] is True
    assert "revision" not in result


@pytest.mark.parametrize("offline", [False, True])
def test_a_complete_pin_needs_no_hub_ref_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    facts = EMBEDDING_MODELS[0]
    expected = _cache(
        snapshot_path(tmp_path, facts.repository, facts.revision), facts.required_files
    )
    monkeypatch.setattr(
        runtime, "snapshot_download", lambda **kwargs: pytest.fail("hub accessed")
    )
    monkeypatch.setattr(runtime, "TextEmbedding", lambda **kwargs: kwargs)
    assert runtime._load_embedder(tmp_path, offline=offline)[
        "specific_model_path"
    ] == str(expected)


@pytest.mark.parametrize(
    "problem", ["wrong_revision", "wrong_repository", "missing_file"]
)
def test_unavailable_pin_never_falls_back_to_another_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    facts = EMBEDDING_MODELS[0]
    repository = facts.name if problem == "wrong_repository" else facts.repository
    revision = "0" * 40 if problem == "wrong_revision" else facts.revision
    files = (
        facts.required_files[:-1] if problem == "missing_file" else facts.required_files
    )
    _cache(snapshot_path(tmp_path, repository, revision), files)

    def unavailable(**kwargs: Any) -> str:
        assert kwargs["repo_id"] == facts.repository
        assert kwargs["revision"] == facts.revision
        assert kwargs["local_files_only"] is True
        raise LocalEntryNotFoundError("Pinned snapshot unavailable")

    monkeypatch.setattr(runtime, "snapshot_download", unavailable)
    monkeypatch.setattr(
        runtime, "TextEmbedding", lambda **kwargs: pytest.fail("loaded a fallback")
    )
    with pytest.raises(LocalEntryNotFoundError, match="Pinned snapshot unavailable"):
        runtime._load_embedder(tmp_path, offline=True)


@pytest.mark.parametrize("wrong_path", [False, True])
def test_hub_result_must_be_the_complete_expected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_path: bool
) -> None:
    facts = EMBEDDING_MODELS[-1]
    downloaded = snapshot_path(
        tmp_path, facts.repository, "0" * 40 if wrong_path else facts.revision
    )
    _cache(
        downloaded, facts.required_files if wrong_path else facts.required_files[:-1]
    )
    monkeypatch.setattr(runtime, "snapshot_download", lambda **kwargs: str(downloaded))
    monkeypatch.setattr(
        runtime, "TextEmbedding", lambda **kwargs: pytest.fail("loaded invalid pin")
    )
    with pytest.raises(
        RuntimeError, match="hub returned" if wrong_path else "model.onnx_data"
    ):
        runtime._load_embedder(tmp_path, offline=False, model=facts.name)


def test_reranker_pin_failure_preserves_optional_fallback_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(**kwargs: Any) -> str:
        assert kwargs["local_files_only"] is True
        raise LocalEntryNotFoundError("Pinned snapshot unavailable")

    monkeypatch.setattr(runtime, "snapshot_download", unavailable)
    monkeypatch.setattr(
        runtime, "TextCrossEncoder", lambda **kwargs: pytest.fail("loaded a fallback")
    )
    with pytest.raises(
        runtime.RerankerUnavailable, match="Pinned snapshot unavailable"
    ):
        runtime._load_cross_encoder(tmp_path, offline=True)


@pytest.mark.parametrize(
    "facts,name", tuple(zip(EMBEDDING_MODELS, RERANKER_MODELS, strict=True))
)
def test_real_fastembed_constructors_honor_the_explicit_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facts: EmbeddingModel, name: str
) -> None:
    """Exercise the installed dependency, stopping only ONNX inference loading."""
    expected = _cache(
        snapshot_path(tmp_path, facts.repository, facts.revision), facts.required_files
    )
    revision = RERANKER_MODELS[name]
    reranker_snapshot = _cache(
        snapshot_path(tmp_path, name, revision), RERANKER_REQUIRED_FILES
    )
    monkeypatch.setattr(
        runtime, "snapshot_download", lambda **kwargs: pytest.fail("hub accessed")
    )
    monkeypatch.setattr(OnnxTextEmbedding, "load_onnx_model", lambda self: None)
    monkeypatch.setattr(OnnxTextCrossEncoder, "load_onnx_model", lambda self: None)

    embedder = runtime._load_embedder(tmp_path, offline=True, model=facts.name)
    reranker = runtime._load_cross_encoder(tmp_path, offline=True, model=name)
    assert embedder.model._model_dir == expected
    assert reranker.model._model_dir == reranker_snapshot
