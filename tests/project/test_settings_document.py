from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
from filelock import AsyncFileLock

from research_rag.core.service import ResearchService
from research_rag.project.config import resolve_config
from research_rag.project.policy import ResearchError
from research_rag.project.settings import (
    LOG_LEVELS,
    SETTINGS,
    SETTINGS_BY_KEY,
    default_config_file,
    packaged_defaults,
)
from research_rag.project.settings_document import (
    merged_document,
    render_project_document,
    setting_cost,
    write_project_document,
)
from research_rag.project.settings_layers import LAYER_PROJECT, SettingsError

pytestmark = pytest.mark.anyio

#: The keys the workspace's Config tab reads. The answer may carry more; a
#: renamed or removed one of these breaks a tab that is already shipped.
WORKSPACE_KEYS = frozenset(
    {"key", "label", "value", "kind", "layer", "origin", "writable", "cost"}
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _rows(answer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["key"]: row for section in answer["sections"] for row in section["settings"]
    }


def _config(project: Path, **kwargs: Any):
    return resolve_config(project, vanilla_executable=sys.executable, **kwargs)


def _service(project: Path, **kwargs: Any) -> ResearchService:
    """A settings answer reads no corpus and opens no gateway, so the retrieval stack
    stands in as two objects this test never calls.
    """

    config = _config(project, **kwargs)
    return ResearchService(
        config,
        cast(Any, object()),
        dense=cast(Any, object()),
    )


def _document(project: Path) -> dict[str, Any]:
    path = project / ".research-rag" / "config.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_a_value_naming_a_quote_a_backslash_a_newline_and_a_unicode_character_survives_a_round_trip() -> (
    None
):
    """The serialiser must not let a value's own characters end the line."""

    values = {
        "quote": 'he said "yes"',
        "backslash": "C:\\research\\rag",
        "newline": "first line\nsecond line",
        "unicode": "café — 日本語 ✓",
        "tab_and_return": "a\tb\rc",
        "control": "a\x07b",
    }
    document = {"runtime": {"model_cache_root": values["unicode"]}}
    for index, (_, value) in enumerate(values.items()):
        document["retrieval"] = document.get("retrieval", {})
        document["retrieval"][f"key_{index}"] = value

    parsed = tomllib.loads(render_project_document(document))

    written = document["runtime"]["model_cache_root"]
    assert parsed["runtime"]["model_cache_root"] == written
    for index, value in enumerate(values.values()):
        assert parsed["retrieval"][f"key_{index}"] == value


def test_a_float_keeps_its_type_through_a_round_trip(tmp_path: Path) -> None:
    """A whole-numbered float must not come back as an int and change the file."""

    text = render_project_document(
        {"retrieval": {"bm25_weight": 0.0, "dense_weight": 1.0}}
    )

    parsed = tomllib.loads(text)

    assert isinstance(parsed["retrieval"]["bm25_weight"], float)
    assert parsed["retrieval"]["dense_weight"] == 1.0
    assert isinstance(parsed["retrieval"]["dense_weight"], float)


def test_a_write_keeps_every_key_it_did_not_change(project: Path) -> None:
    """A browser write must not delete a sibling key it does not know about."""

    path = project / ".research-rag" / "config.toml"
    write_project_document(
        path,
        {
            "retrieval": {"rrf_k": 25, "prf": True},
            "chunking": {"size": 512, "overlap": 32},
            "some_future_section": {"kept": "yes"},
        },
    )

    write_project_document(
        path, merged_document(_document(project), {"retrieval.rrf_k": 40})
    )

    document = _document(project)
    assert document["retrieval"]["rrf_k"] == 40
    assert document["retrieval"]["prf"] is True
    assert document["chunking"]["size"] == 512
    assert document["chunking"]["overlap"] == 32
    assert document["some_future_section"]["kept"] == "yes"


async def test_a_written_value_becomes_the_effective_value_of_this_app(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()

    result = await service.settings_write(
        {"retrieval.rrf_k": 40},
        expected_revision=before["revision"],
        confirm=True,
    )

    after = await service.settings_read()
    values = {
        setting["key"]: setting["value"]
        for section in after["sections"]
        for setting in section["settings"]
    }
    assert values["retrieval.rrf_k"] == 40
    assert result["changed"] == ["retrieval.rrf_k"]
    assert result["revision"] == after["revision"]


async def test_the_settings_file_is_the_projects_own_layer(project: Path) -> None:
    service = _service(project)
    before = await service.settings_read()

    await service.settings_write(
        {"retrieval.bm25_weight": 0.5},
        expected_revision=before["revision"],
        confirm=True,
    )

    assert _document(project)["retrieval"]["bm25_weight"] == 0.5
    resolved = _config(project)
    assert resolved.settings.bm25_weight == 0.5
    assert not (project / ".research-rag" / "default.toml").exists()


async def test_an_unknown_key_is_refused_with_the_readers_own_message(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.no_such_knob": 3},
            expected_revision=before["revision"],
            confirm=True,
        )

    assert str(refusal.value) == (
        f"{LAYER_PROJECT} ({project / '.research-rag' / 'config.toml'}) sets an "
        "unknown setting: retrieval.no_such_knob"
    )
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_value_of_the_wrong_kind_is_refused_with_the_readers_own_message(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()
    source = f"{LAYER_PROJECT} ({project / '.research-rag' / 'config.toml'})"
    with pytest.raises(SettingsError) as from_the_reader:
        SETTINGS_BY_KEY["retrieval.rrf_k"].coerce("many", source=source)

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.rrf_k": "many"},
            expected_revision=before["revision"],
            confirm=True,
        )

    assert str(refusal.value) == str(from_the_reader.value)
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_value_outside_the_declared_range_is_refused(project: Path) -> None:
    service = _service(project)
    before = await service.settings_read()

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.rrf_k": 100_000},
            expected_revision=before["revision"],
            confirm=True,
        )

    assert "must be at most 1000" in str(refusal.value)
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_value_outside_the_declared_choices_is_refused(project: Path) -> None:
    service = _service(project)
    before = await service.settings_read()

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"dense.reranker_model": "no-such-reranker"},
            expected_revision=before["revision"],
            confirm=True,
        )

    assert "must be one of" in str(refusal.value)
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_symlinked_settings_file_is_refused_and_nothing_is_written(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()
    elsewhere = project / "elsewhere.toml"
    elsewhere.write_text("[retrieval]\nrrf_k = 5\n", encoding="utf-8")
    path = project / ".research-rag" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(elsewhere)

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.rrf_k": 40},
            expected_revision=before["revision"],
            confirm=True,
        )

    assert "must not be a symlink" in str(refusal.value)
    assert elsewhere.read_text(encoding="utf-8") == "[retrieval]\nrrf_k = 5\n"


async def test_a_write_against_a_revision_the_reader_never_held_is_refused(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.rrf_k": 40},
            expected_revision="not-the-revision-the-reader-held",
            confirm=True,
        )

    message = str(refusal.value)
    assert "Nothing was written" in message
    assert before["revision"] in message
    assert not (project / ".research-rag" / "config.toml").exists()


@pytest.mark.integration
async def test_a_write_is_refused_while_another_process_holds_the_project_lock(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()
    held = AsyncFileLock(service.config.state_root / "project.lock")
    await held.acquire()
    try:
        with pytest.raises(ResearchError) as refusal:
            await service.settings_write(
                {"runtime.tool_detail": "full"},
                expected_revision=before["revision"],
                confirm=True,
            )
    finally:
        await held.release()

    message = str(refusal.value)
    assert "Another research process is working on this project" in message
    assert "research-rag" in message
    assert "status" in message
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_costly_change_is_named_before_anything_is_written(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.rrf_k": 40},
            expected_revision=before["revision"],
        )

    message = str(refusal.value)
    assert "retrieval.rrf_k" in message
    assert "confirm" in message
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_key_that_moves_nothing_a_generation_records_is_written_without_a_warning(
    project: Path,
) -> None:
    service = _service(project)
    before = await service.settings_read()

    result = await service.settings_write(
        {"runtime.tool_detail": "full", "retrieval.rerank_max_candidates": 24},
        expected_revision=before["revision"],
    )

    assert result["requires_ingest"] is False
    assert result["changed"] == [
        "retrieval.rerank_max_candidates",
        "runtime.tool_detail",
    ]
    assert _document(project)["runtime"]["tool_detail"] == "full"


async def test_a_key_the_command_line_supplied_is_refused_a_project_write(
    project: Path,
) -> None:
    service = _service(project, settings_overrides=["retrieval.rrf_k=25"])
    before = await service.settings_read()
    rows = {
        setting["key"]: setting
        for section in before["sections"]
        for setting in section["settings"]
    }
    assert rows["retrieval.rrf_k"]["writable"] is False

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            {"retrieval.rrf_k": 40},
            expected_revision=before["revision"],
            confirm=True,
        )

    assert "cannot override" in str(refusal.value)
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_a_lone_surrogate_is_refused_by_the_writer_rather_than_the_encoder(
    project: Path,
) -> None:
    """A JSON body can carry a code point that is not a character.

    `\\ud800` names a code point rather than a character, and no UTF-8 encoding of
    it exists, so the file this writer produces could not hold it. The refusal
    names the key and the reason, where the alternative is an encoder failure
    part-way through a write -- which the writer's callers do not translate, so a
    reader would see a failed request with no reason in it.

    Every string setting the registry ships is validated before it reaches here,
    so this is the guard for the writer itself rather than for one shipped key.
    """

    with pytest.raises(SettingsError) as refusal:
        render_project_document({"runtime": {"model_cache_root": "\ud800"}})

    assert "runtime.model_cache_root" in str(refusal.value)
    assert "UTF-8" in str(refusal.value)


async def test_a_hostile_json_value_is_answered_with_a_refusal_naming_the_key(
    project: Path,
) -> None:
    """The value arrives the way a browser sends it, and the answer names the key.

    The body is encoded as `bytes` on purpose: a Python source literal cannot carry
    a lone surrogate through a UTF-8 encoder either, and JSON is what brings one
    to this app. The registry refuses both shipped string settings first, with a
    message of its own; what this pins is that the answer is that refusal rather
    than a failed request.
    """

    service = _service(project)
    before = await service.settings_read()
    body = json.dumps(
        {"values": {"language.bm25_stopwords": "\ud800"}, "expected_revision": ""}
    ).encode("utf-8", errors="surrogatepass")
    assert json.loads(body.decode("utf-8", errors="surrogatepass"))["values"]

    with pytest.raises(ResearchError) as refusal:
        await service.settings_write(
            json.loads(body)["values"],
            expected_revision=before["revision"],
        )

    assert "language.bm25_stopwords" in str(refusal.value)
    assert not (project / ".research-rag" / "config.toml").exists()


async def test_every_key_carries_the_description_the_registry_declares(
    project: Path,
) -> None:
    answer = await _service(project).settings_read()
    rows = _rows(answer)

    assert set(rows) == {setting.key for setting in SETTINGS}
    for setting in SETTINGS:
        assert rows[setting.key]["doc"] == setting.doc
        assert rows[setting.key]["doc"].strip()


async def test_every_key_carries_the_default_the_packaged_file_declares(
    project: Path,
) -> None:
    """A reader is told what a key starts from, and the file it starts from, once."""

    answer = await _service(project).settings_read()
    rows = _rows(answer)
    defaults = packaged_defaults()

    assert answer["default_file"] == str(default_config_file())
    assert answer["default_file"].endswith("default.toml")
    for setting in SETTINGS:
        assert rows[setting.key]["default"] == defaults[setting.key]
    # The value in force is the default for every key this project did not write,
    # and the row says so in one fact rather than by naming the file again. The
    # suite's own environment supplies the duplicate threshold.
    assert {row["key"] for row in rows.values() if row["defaulted"]} == set(
        defaults
    ) - {"retrieval.duplicate_cosine"}


async def test_a_bound_a_choice_or_a_variable_travels_only_where_one_is_declared(
    project: Path,
) -> None:
    answer = await _service(project).settings_read()
    rows = _rows(answer)

    for setting in SETTINGS:
        row = rows[setting.key]
        for name, declared in (
            ("minimum", setting.minimum is not None),
            ("maximum", setting.maximum is not None),
            ("choices", bool(setting.choices)),
            ("env", bool(setting.env)),
        ):
            assert (name in row) is declared, f"{setting.key}.{name}"
        assert row.get("minimum", setting.minimum) == setting.minimum
        assert row.get("maximum", setting.maximum) == setting.maximum
        assert row.get("choices", list(setting.choices)) == list(setting.choices)
        assert row.get("env", setting.env) == setting.env
    assert rows["runtime.nice"]["minimum"] == 0
    assert rows["runtime.nice"]["maximum"] == 19
    assert rows["runtime.log_level"]["choices"] == list(LOG_LEVELS)
    assert "minimum" not in rows["runtime.offline"]
    assert "choices" not in rows["retrieval.rrf_k"]


async def test_the_answer_adds_keys_the_workspace_ignores_and_removes_none(
    project: Path,
) -> None:
    answer = await _service(project).settings_read()

    assert json.loads(json.dumps(answer)) == answer
    for section in answer["sections"]:
        for row in section["settings"]:
            assert set(row) >= WORKSPACE_KEYS


async def test_the_settings_answer_names_a_layer_and_a_cost_for_every_key(
    project: Path,
) -> None:
    service = _service(project)

    answer = await service.settings_read()

    rows = [
        setting for section in answer["sections"] for setting in section["settings"]
    ]
    assert {row["key"] for row in rows} == {setting.key for setting in SETTINGS}
    for row in rows:
        # The suite's own environment supplies the duplicate threshold.
        assert row["writable"] is (row["key"] != "retrieval.duplicate_cosine")
        assert row["origin"]
        assert row["cost"]["level"] in {"none", "regeneration", "model"}
        assert row["cost"]["message"]
    assert answer["message"].endswith(f"{project / '.research-rag' / 'config.toml'}.")


async def test_the_read_answer_is_the_one_the_workspace_writes_through(
    project: Path,
) -> None:
    service = _service(project)

    first = await service.settings_read()
    second = await service.settings_read()

    assert first["revision"] == second["revision"]
    assert isinstance(first["revision"], str)


async def test_a_key_that_moves_the_retrieval_policy_costs_a_regeneration(
    project: Path,
) -> None:
    service = _service(project)

    cost = setting_cost(service.config.settings, SETTINGS_BY_KEY["retrieval.rrf_k"])

    assert cost["level"] == "regeneration"
    assert cost["moved"] == ["retrieval_policy"]


async def test_a_key_the_registry_calls_identity_but_no_generation_reads_costs_nothing(
    project: Path,
) -> None:
    service = _service(project)
    keys = (
        "retrieval.rerank_max_candidates",
        "retrieval.rerank_window_multiple",
        "retrieval.rerank_window_floor",
        "retrieval.prf",
        "retrieval.prf_documents",
        "retrieval.prf_terms",
        "retrieval.maximum_withheld_examples",
    )

    for key in keys:
        assert SETTINGS_BY_KEY[key].layer == "identity"
        assert setting_cost(service.config.settings, SETTINGS_BY_KEY[key])["level"] == (
            "none"
        )


async def test_a_key_that_changes_the_embedding_model_costs_a_model(
    project: Path,
) -> None:
    service = _service(project)

    cost = setting_cost(
        service.config.settings,
        SETTINGS_BY_KEY["dense.embedding_model"],
        value="intfloat/multilingual-e5-large",
    )

    assert cost["level"] == "model"
    assert cost["moved"] == ["embedding_model"]
    assert cost["requires_ingest"] is True


async def test_a_key_that_changes_the_reranker_keeps_the_stored_vectors(
    project: Path,
) -> None:
    service = _service(project)

    cost = setting_cost(
        service.config.settings,
        SETTINGS_BY_KEY["dense.reranker_model"],
        value="BAAI/bge-reranker-base",
    )

    assert cost["level"] == "model"
    assert cost["moved"] == ["reranker_model"]
    assert cost["requires_ingest"] is False


def test_a_value_the_registry_does_not_declare_is_refused_rather_than_dropped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"

    with pytest.raises(SettingsError) as refusal:
        write_project_document(path, {"retrieval": {"rrf_k": [1, 2, 3]}})

    assert "not a value this writer can write" in str(refusal.value)
    assert not path.exists()
