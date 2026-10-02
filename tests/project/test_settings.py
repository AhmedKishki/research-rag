from __future__ import annotations

import sys
from pathlib import Path

import pytest

import research_rag.project.config as config_module
from research_rag.core.service import retrieval_policy_fingerprint
from research_rag.project.config import (
    ConfigurationError,
    apply_process_priority,
    resolve_config,
)
from research_rag.project.settings import (
    MAXIMUM_WORK_BUDGET_SECONDS,
    SETTINGS,
    EffectiveSettings,
    sources_for,
)
from research_rag.project.settings_layers import (
    SettingsError,
    describe_settings,
    resolve_settings,
)

# A different value would tell every existing generation that its ranking policy
# changed and force a rebuild for nothing.
PUBLISHED_FINGERPRINT = (
    "b7285824c79d662c9abdba84238077abc1271d11a861c931ee0e590c3d4f380f"
)


def _resolve(project_root: Path, **kwargs) -> tuple[EffectiveSettings, dict[str, str]]:
    values, provenance = resolve_settings(SETTINGS, sources_for(project_root), **kwargs)
    return EffectiveSettings.from_values(values), provenance


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_packaged_default_file_supplies_every_setting(tmp_path: Path) -> None:
    settings, provenance = _resolve(tmp_path, environ={})

    for setting in SETTINGS:
        assert setting.key in provenance
    default_source = f"default ({sources_for(tmp_path).default_file})"
    assert provenance["chunking.size"] == default_source
    assert provenance["retrieval.rrf_k"] == default_source
    assert settings.chunk_size == 384
    assert settings.chunk_overlap == 64


def test_defaults_reproduce_the_published_fingerprint(tmp_path: Path) -> None:
    settings, _ = _resolve(tmp_path, environ={})

    assert retrieval_policy_fingerprint(settings) == PUBLISHED_FINGERPRINT


def test_a_query_time_ranking_knob_keeps_the_published_fingerprint(
    tmp_path: Path,
) -> None:
    """The source-diversity reordering applies to the candidates fusion and reranking already
    returned, so in the fingerprint it would flag every generation as needing an upgrade
    that reproduces the same index byte for byte.
    """

    spread, _ = _resolve(
        tmp_path,
        overrides=["retrieval.source_diversity_penalty=0.6"],
        environ={},
    )

    assert spread.source_diversity_penalty == 0.6
    assert retrieval_policy_fingerprint(spread) == PUBLISHED_FINGERPRINT


def test_a_layer_names_only_what_it_changes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(
        project / ".research-rag" / "config.toml",
        "[retrieval]\nrrf_k = 25\n\n[chunking]\noverlap = 32\n",
    )

    settings, provenance = _resolve(project, environ={})

    assert settings.rrf_k == 25
    assert settings.chunk_overlap == 32
    assert settings.chunk_size == 384
    assert provenance["retrieval.rrf_k"] != provenance["chunking.size"]


def test_precedence_runs_file_then_environment_then_command_line(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write(project / ".research-rag" / "config.toml", "[retrieval]\nrrf_k = 25\n")
    named = _write(tmp_path / "named.toml", "[retrieval]\nrrf_k = 40\n")

    from_environment, provenance = _resolve(
        project,
        config_path=named,
        environ={"RESEARCH_RAG_RETRIEVAL_RRF_K": "15"},
    )
    assert from_environment.rrf_k == 15
    assert provenance["retrieval.rrf_k"].startswith("environment")

    from_command_line, provenance = _resolve(
        project,
        config_path=named,
        overrides=["retrieval.rrf_k=5"],
        environ={"RESEARCH_RAG_RETRIEVAL_RRF_K": "15"},
    )
    assert from_command_line.rrf_k == 5
    assert provenance["retrieval.rrf_k"] == "command line"


def test_an_unknown_key_is_refused_in_a_file_and_on_the_command_line(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write(project / ".research-rag" / "config.toml", "[retrieval]\nrrf_kx = 25\n")

    with pytest.raises(SettingsError, match="unknown setting"):
        _resolve(project, environ={})

    with pytest.raises(SettingsError, match="unknown setting"):
        _resolve(tmp_path, overrides=["retrieval.rrf_kx=1"], environ={})


def test_out_of_bounds_and_cross_field_values_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="at most 384"):
        _resolve(tmp_path, overrides=["chunking.size=900"], environ={})

    with pytest.raises(SettingsError, match="must be below"):
        _resolve(
            tmp_path,
            overrides=["chunking.size=60", "chunking.overlap=60"],
            environ={},
        )

    with pytest.raises(SettingsError, match="must be one of"):
        _resolve(tmp_path, overrides=["dense.backend=lancedb"], environ={})

    with pytest.raises(SettingsError, match="must be true or false"):
        _resolve(tmp_path, overrides=["runtime.offline=maybe"], environ={})

    with pytest.raises(SettingsError, match="at most 1"):
        _resolve(
            tmp_path,
            overrides=["retrieval.source_diversity_penalty=1.5"],
            environ={},
        )


def test_the_ingest_budget_can_cover_a_whole_build(tmp_path: Path) -> None:
    """With a cap of five minutes, a long build needs one call per slice, and a client that
    stops repeating identical calls cannot finish it at all.
    """

    settings, _provenance = _resolve(
        tmp_path,
        overrides=[f"ingestion.work_budget_seconds={MAXIMUM_WORK_BUDGET_SECONDS}"],
        environ={},
    )
    assert settings.work_budget_seconds == MAXIMUM_WORK_BUDGET_SECONDS

    with pytest.raises(SettingsError, match="at most"):
        _resolve(
            tmp_path,
            overrides=[
                f"ingestion.work_budget_seconds={MAXIMUM_WORK_BUDGET_SECONDS + 1}"
            ],
            environ={},
        )


def test_a_symlinked_config_layer_is_refused(tmp_path: Path) -> None:
    real = _write(tmp_path / "real.toml", "[retrieval]\nrrf_k = 30\n")
    link = tmp_path / "link.toml"
    link.symlink_to(real)

    with pytest.raises(SettingsError, match="must not be a symlink"):
        _resolve(tmp_path, config_path=link, environ={})


def test_every_setting_says_what_its_key_does() -> None:
    """A key with no description reaches both readers as an unexplained row."""

    blank = [setting.key for setting in SETTINGS if not setting.doc.strip()]

    assert not blank


def test_the_report_names_every_setting_and_its_layer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(
        project / ".research-rag" / "config.toml",
        '[runtime]\nlog_level = "info"\n',
    )

    settings, provenance = _resolve(project, environ={})
    rendered = describe_settings(SETTINGS, settings.as_values(), provenance)

    assert "project config" in rendered
    for setting in SETTINGS:
        assert setting.key in rendered


def test_resolve_config_exposes_the_merged_settings(project: Path) -> None:
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=["retrieval.rrf_k=42", "chunking.size=200"],
    )

    assert config.settings.rrf_k == 42
    assert config.settings.chunk_size == 200
    assert config.offline is False
    assert config.reranker_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert config.settings_provenance["retrieval.rrf_k"] == "command line"

    with pytest.raises(ConfigurationError, match="must be one of"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            dense_backend="lancedb",
        )


def test_the_corpus_language_must_be_a_code(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="ISO 639-1"):
        _resolve(tmp_path, overrides=["language.corpus=german"], environ={})

    settings, _ = _resolve(tmp_path, overrides=["language.corpus= DE "], environ={})
    assert settings.language_corpus == "de"


def test_a_corpus_language_the_model_cannot_embed_is_reported(tmp_path: Path) -> None:
    mismatched, _ = _resolve(tmp_path, overrides=["language.corpus=de"], environ={})
    warning = mismatched.embedding_language_warning
    assert warning is not None and "covers en, not 'de'" in warning

    german, _ = _resolve(
        tmp_path,
        overrides=[
            "language.corpus=de",
            "dense.embedding_model=jinaai/jina-embeddings-v2-base-de",
        ],
        environ={},
    )
    assert german.embedding_language_warning is None
    assert german.embedding_dimension == 768


def test_the_corpus_language_must_have_bm25_stopwords(tmp_path: Path) -> None:
    """A language BM25 cannot tokenize must fail before a build, not during one."""
    with pytest.raises(SettingsError, match="no BM25 stopword list"):
        _resolve(tmp_path, overrides=["language.corpus=ja"], environ={})

    for supported in ("en", "de", "fr", "zh"):
        settings, _ = _resolve(
            tmp_path,
            overrides=[f"language.corpus={supported}"],
            environ={},
        )
        assert settings.language_corpus == supported


def test_the_language_is_part_of_the_ranking_policy(tmp_path: Path) -> None:
    english, _ = _resolve(tmp_path, environ={})
    german, _ = _resolve(tmp_path, overrides=["language.corpus=de"], environ={})

    assert retrieval_policy_fingerprint(german) != retrieval_policy_fingerprint(english)


def test_a_corpus_can_name_several_languages(tmp_path: Path) -> None:
    """A corpus written in more than one language says so, in one setting."""
    mixed, _ = _resolve(
        tmp_path,
        overrides=[
            "language.corpus=de,en",
            "dense.embedding_model=intfloat/multilingual-e5-large",
        ],
        environ={},
    )
    assert mixed.corpus_languages == ("de", "en")
    assert mixed.language_corpus == "de,en"
    assert mixed.bm25_stopwords_language == "de"
    assert mixed.embedding_language_warning is None


def test_the_gate_stopwords_stop_question_and_do_support_words(
    tmp_path: Path,
) -> None:
    """A query no topic is attached to has no content token to be admitted by."""
    english, _ = _resolve(tmp_path, environ={})
    assert {"how", "what", "when", "where", "which", "who", "why"} <= (
        english.gate_stopwords
    )
    assert {"do", "does", "did", "not", "will"} <= english.gate_stopwords

    german, _ = _resolve(tmp_path, overrides=["language.corpus=de"], environ={})
    assert {"wie", "und", "nicht"} <= german.gate_stopwords


def test_the_gate_stopwords_cover_every_language_of_a_mixed_corpus(
    tmp_path: Path,
) -> None:
    """One contentless query has to abstain in any language the corpus names."""
    mixed, _ = _resolve(tmp_path, overrides=["language.corpus=de,en"], environ={})
    assert {"wie", "und"} <= mixed.gate_stopwords
    assert {"how", "does"} <= mixed.gate_stopwords


def test_the_gate_stopwords_keep_their_own_words_without_bm25s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken bm25s environment must not turn the gate into no gate at all."""
    import research_rag.project.settings as settings_module

    monkeypatch.setattr(settings_module, "bm25_stopwords", lambda language: None)
    monkeypatch.setattr(settings_module, "_english_plus_stopwords", lambda: frozenset())
    resolved, _ = _resolve(tmp_path, environ={})
    assert {"how", "what", "why", "be", "with"} <= resolved.gate_stopwords


def test_a_mixed_corpus_filters_the_first_language_named(tmp_path: Path) -> None:
    """BM25 filters one language: the first named, unless another is chosen."""
    first, _ = _resolve(tmp_path, overrides=["language.corpus=de,en"], environ={})
    assert first.bm25_stopwords_language == "de"

    chosen, _ = _resolve(
        tmp_path,
        overrides=["language.corpus=de,en", "language.bm25_stopwords=en"],
        environ={},
    )
    assert chosen.bm25_stopwords_language == "en"

    # The default model covers English only, so the warning names the model that
    # would cover both languages.
    warning = chosen.embedding_language_warning
    assert warning is not None and "not 'de'" in warning
    assert "intfloat/multilingual-e5-large" in warning
    assert "not 'en'" not in warning


def test_a_language_list_is_kept_as_written(tmp_path: Path) -> None:
    """Order is meaningful and duplicates are not."""
    settings, _ = _resolve(
        tmp_path,
        overrides=[
            "language.corpus= DE , en , de ",
            "language.bm25_stopwords=DE",
        ],
        environ={},
    )
    assert settings.language_corpus == "de,en"
    assert settings.bm25_stopwords == "de"

    other, _ = _resolve(
        tmp_path,
        overrides=["language.corpus=en,de", "language.bm25_stopwords=en"],
        environ={},
    )
    assert other.language_corpus == "en,de"

    with pytest.raises(SettingsError, match="empty language"):
        _resolve(tmp_path, overrides=["language.corpus=de,"], environ={})
    with pytest.raises(SettingsError, match="no BM25 stopword list"):
        _resolve(tmp_path, overrides=["language.bm25_stopwords=ja"], environ={})


def test_the_bm25_language_is_what_the_policy_records(tmp_path: Path) -> None:
    english, _ = _resolve(tmp_path, environ={})
    german, _ = _resolve(tmp_path, overrides=["language.corpus=de"], environ={})
    german_first, _ = _resolve(
        tmp_path, overrides=["language.corpus=de,en"], environ={}
    )
    english_first, _ = _resolve(
        tmp_path, overrides=["language.corpus=en,de"], environ={}
    )

    # BM25 filters one list, and the order a corpus names its languages in writes
    # that choice.
    assert retrieval_policy_fingerprint(german_first) == retrieval_policy_fingerprint(
        german
    )
    assert retrieval_policy_fingerprint(english_first) == (
        retrieval_policy_fingerprint(english)
    )
    assert retrieval_policy_fingerprint(german) != retrieval_policy_fingerprint(english)


def test_the_stopword_language_roundtrips_through_the_installed_bm25s(
    tmp_path: Path,
) -> None:
    """The pinned bm25s must re-read a German list it writes itself."""
    german, _ = _resolve(tmp_path, overrides=["language.corpus=de"], environ={})
    assert german.bm25_stopwords_language == "de"


def test_a_stopword_list_that_cannot_roundtrip_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken serializer must fail at settings time, not after a long build."""
    from bm25s.utils import json_functions

    monkeypatch.setattr(json_functions, "dumps", lambda d, **kw: "not json")
    with pytest.raises(SettingsError, match="cannot re-read"):
        _resolve(tmp_path, overrides=["language.corpus=de"], environ={})


def test_the_nice_setting_leaves_priority_alone_unless_asked(project: Path) -> None:
    """0 is the shipped default: a process nobody asked to yield does not yield."""

    config = resolve_config(project, vanilla_executable=sys.executable)

    assert config.nice == 0


def test_apply_process_priority_raises_by_the_difference_and_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"current": 4}
    increments: list[int] = []

    def fake_nice(increment: int) -> int:
        increments.append(increment)
        state["current"] += increment
        return state["current"]

    monkeypatch.setattr(config_module.os, "nice", fake_nice)

    assert apply_process_priority(10) == 10
    assert increments == [0, 6]

    # Idempotent, so a child that inherited the value and applies it again is safe.
    assert apply_process_priority(10) == 10
    assert increments == [0, 6, 0]

    # 0 does not even read the current value.
    assert apply_process_priority(0) is None
    assert increments == [0, 6, 0]


def test_apply_process_priority_reports_a_refusal_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_increment: int) -> int:
        raise OSError("this platform has no niceness")

    monkeypatch.setattr(config_module.os, "nice", refuse)

    assert apply_process_priority(10) is None
