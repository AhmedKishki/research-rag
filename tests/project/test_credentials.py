"""The account's DeepSeek key: where it lives, who wins, and what is refused."""

from __future__ import annotations

import json
import stat

import pytest

from research_rag.project.credentials import (
    DEEPSEEK_KEY_ENVIRONMENT,
    clear_deepseek_key,
    credentials_path,
    deepseek_key,
    set_deepseek_key,
)
from research_rag.project.policy import ResearchError


def test_a_set_key_is_written_for_this_account_alone() -> None:
    set_deepseek_key("sk-account")

    path = credentials_path()
    assert path.parent.name == "research-rag"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "deepseek": {"api_key": "sk-account"},
    }
    assert deepseek_key() == ("sk-account", "account file")


def test_the_environment_key_wins_over_the_account_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_deepseek_key("sk-account")
    monkeypatch.setenv(DEEPSEEK_KEY_ENVIRONMENT, "sk-environment")

    assert deepseek_key() == ("sk-environment", "environment")


def test_no_key_anywhere_reads_as_none() -> None:
    assert deepseek_key() == (None, None)


def test_clearing_the_key_leaves_none_in_the_file() -> None:
    set_deepseek_key("sk-account")
    clear_deepseek_key()

    assert deepseek_key() == (None, None)
    assert "sk-account" not in credentials_path().read_text(encoding="utf-8")


@pytest.mark.parametrize("key", ["", "   ", "sk with space", "sk-\nnewline"])
def test_a_key_that_cannot_be_a_bearer_token_is_refused(key: str) -> None:
    with pytest.raises(ResearchError, match="API key"):
        set_deepseek_key(key)
    assert not credentials_path().exists()


@pytest.mark.parametrize(
    ("document", "named"),
    [
        ({"schema_version": 2, "deepseek": {}}, "schema_version"),
        ({"schema_version": 1, "openai": {}}, "openai"),
        ({"schema_version": 1, "deepseek": {"api_key": 5}}, "deepseek.api_key"),
        ({"schema_version": 1, "deepseek": {"token": "x"}}, "deepseek.token"),
    ],
    ids=["later version", "unknown provider", "wrong type", "unknown field"],
)
def test_a_credentials_file_this_app_cannot_read_is_refused_by_name(
    document: dict, named: str
) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ResearchError, match=named) as refused:
        deepseek_key()
    assert str(path) in str(refused.value)


def test_a_refusal_never_repeats_the_key() -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 1, "deepseek": {"api_key": "sk-secret", "x": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(ResearchError) as refused:
        deepseek_key()
    assert "sk-secret" not in str(refused.value)
