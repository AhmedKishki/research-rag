"""The layer stack, and the rules that keep a configuration honest.

A layer is a stack rather than a replacement: each one names the keys it
changes, the keys it leaves alone stay inherited, an undeclared key is an error
in every layer, and a value is read the same way whatever layer carried it. The
vocabulary of `layer` belongs to the caller, so the tests here use words that
belong to no server in this collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_rag.project.settings_layers import (
    LAYER_COMMAND_LINE,
    LAYER_DEFAULT,
    LAYER_ENVIRONMENT,
    LAYER_FILE,
    LAYER_PROJECT,
    LAYER_USER,
    Setting,
    SettingsError,
    SettingsSources,
    default_config_path,
    describe_settings,
    environment_settings,
    merge_settings,
    override_settings,
    project_config_path,
    read_config_document,
    resolve_settings,
    user_config_path,
)

# A caller's vocabulary, deliberately not either server's.
SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="window.tide",
        field="tide",
        kind=str,
        layer="shape",
        doc="How often the window closes.",
        env="EXAMPLE_TIDE",
    ),
    Setting(
        key="window.depth",
        field="depth",
        kind=int,
        layer="shape",
        doc="How deep the window goes.",
        minimum=1,
        maximum=64,
        env="EXAMPLE_DEPTH",
    ),
    Setting(
        key="window.glass",
        field="glass",
        kind=str,
        layer="surface",
        doc="Which glass the window is made of.",
        choices=("clear", "frosted"),
        normalize_case=True,
    ),
    Setting(
        key="window.latch",
        field="latch",
        kind=bool,
        layer="surface",
        doc="Whether the window latches.",
    ),
    Setting(
        key="model.name",
        field="model_name",
        kind=str,
        layer="shape",
        doc="The model, or empty for none.",
        empty="",
    ),
    Setting(
        key="limits.work",
        field="work",
        kind=float,
        layer="surface",
        doc="A budget.",
        minimum=0.0,
    ),
)

PACKAGED = """\
[window]
depth = 8
glass = "clear"
latch = true
tide = "Open"

[model]
name = "example/small"

[limits]
work = 2.5
"""

EMPTY = {"environ": {}}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sources(tmp_path: Path, **overrides: Any) -> SettingsSources:
    """Return sources whose packaged default holds every declared setting."""

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    defaults = {
        "default_file": _write(tmp_path / "default.toml", PACKAGED),
        "user_config": None,
        "project_root": project,
        "project_config": None,
    }
    return SettingsSources(**{**defaults, **overrides})


def test_the_layers_apply_lowest_first_and_per_key(tmp_path: Path) -> None:
    project_layer = _write(
        tmp_path / "project" / ".example-rag" / "config.toml",
        "[window]\ndepth = 12\nlatch = false\n",
    )
    named = _write(tmp_path / "named.toml", '[window]\ndepth = 20\nglass = "Frosted"\n')

    values, provenance = resolve_settings(
        SETTINGS,
        _sources(
            tmp_path,
            user_config=_write(
                tmp_path / "user.toml", '[window]\ndepth = 4\ntide = "Shut"\n'
            ),
            project_config=project_layer,
        ),
        config_path=named,
        environ={},
    )

    # --config beats the project, the project beats the account, and a key no
    # later layer named is still inherited rather than reset.
    assert values["depth"] == 20
    assert values["tide"] == "Shut"
    assert values["glass"] == "frosted", "a mode name is read in any spelling"
    assert values["work"] == 2.5
    assert LAYER_FILE in provenance["window.depth"]
    assert LAYER_USER in provenance["window.tide"]
    assert LAYER_PROJECT in provenance["window.latch"]
    assert provenance["limits.work"].startswith(LAYER_DEFAULT)


def test_a_file_layer_that_is_absent_is_not_a_layer(tmp_path: Path) -> None:
    values, provenance = resolve_settings(
        SETTINGS, _sources(tmp_path, user_config=tmp_path / "nowhere.toml"), **EMPTY
    )

    assert values["depth"] == 8
    assert provenance["window.depth"].startswith(LAYER_DEFAULT)


def test_the_environment_and_the_command_line_are_the_strongest(
    tmp_path: Path,
) -> None:
    values, provenance = resolve_settings(
        SETTINGS,
        _sources(tmp_path),
        overrides=["window.depth=30"],
        environ={"EXAMPLE_TIDE": "shut"},
    )

    assert values["depth"] == 30
    assert provenance["window.depth"] == LAYER_COMMAND_LINE
    assert values["tide"] == "shut"
    assert provenance["window.tide"].startswith(LAYER_ENVIRONMENT)


def test_a_setting_without_an_environment_name_has_no_environment_layer(
    tmp_path: Path,
) -> None:
    layers = environment_settings(
        {"EXAMPLE_TIDE": "shut", "EXAMPLE_GLASS": "frosted"}, SETTINGS
    )

    assert layers == {"window.tide": ("shut", "EXAMPLE_TIDE")}


def test_an_empty_variable_is_not_a_layer_saying_anything(tmp_path: Path) -> None:
    assert environment_settings({"EXAMPLE_TIDE": "   "}, SETTINGS) == {}


def test_a_string_from_a_layer_is_read_the_way_a_number_from_a_file_is(
    tmp_path: Path,
) -> None:
    from_file = resolve_settings(
        SETTINGS,
        _sources(tmp_path),
        config_path=_write(tmp_path / "a.toml", "[window]\ndepth = 30\n"),
        **EMPTY,
    )
    from_env = resolve_settings(
        SETTINGS, _sources(tmp_path), environ={"EXAMPLE_DEPTH": "30"}
    )
    from_set = resolve_settings(
        SETTINGS, _sources(tmp_path), overrides=["window.depth=30"]
    )

    assert from_file[0]["depth"] == from_env[0]["depth"] == from_set[0]["depth"] == 30


def test_a_boolean_written_as_a_string_is_read_as_a_boolean(tmp_path: Path) -> None:
    from_env, _ = resolve_settings(
        SETTINGS, _sources(tmp_path), environ={"EXAMPLE_LATCH": "TRUE"}
    )
    from_set, _ = resolve_settings(
        SETTINGS, _sources(tmp_path), overrides=["window.latch=false"]
    )

    assert from_env["latch"] is True
    assert from_set["latch"] is False


def test_an_empty_string_means_something_only_where_it_is_declared(
    tmp_path: Path,
) -> None:
    declared, _ = resolve_settings(
        SETTINGS, _sources(tmp_path), overrides=["model.name="]
    )
    string, _ = resolve_settings(
        SETTINGS, _sources(tmp_path), overrides=["window.tide="]
    )

    # Turning a model off is a value, a string setting reads "" as the empty
    # string, and a number has no reading for it at all.
    assert declared["model_name"] == ""
    assert string["tide"] == ""
    with pytest.raises(SettingsError, match="window.depth .*must be int"):
        resolve_settings(SETTINGS, _sources(tmp_path), overrides=["window.depth="])


def test_an_undeclared_key_is_refused_in_every_layer(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="unknown setting"):
        resolve_settings(
            SETTINGS,
            _sources(tmp_path),
            config_path=_write(tmp_path / "a.toml", "[window]\nnope = 1\n"),
            **EMPTY,
        )
    with pytest.raises(SettingsError, match="unknown setting"):
        resolve_settings(SETTINGS, _sources(tmp_path), overrides=["window.nope=1"])


def test_a_value_out_of_bounds_is_refused_by_name_and_layer(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match=r"window.depth .*at most 64"):
        resolve_settings(SETTINGS, _sources(tmp_path), overrides=["window.depth=900"])
    with pytest.raises(SettingsError, match=r"window.depth .*at least 1"):
        resolve_settings(SETTINGS, _sources(tmp_path), environ={"EXAMPLE_DEPTH": "0"})
    with pytest.raises(SettingsError, match="must be one of"):
        resolve_settings(
            SETTINGS, _sources(tmp_path), overrides=['window.glass="smoked"']
        )
    with pytest.raises(SettingsError, match="must be true or false"):
        resolve_settings(SETTINGS, _sources(tmp_path), overrides=["window.latch=maybe"])


def test_a_boolean_is_refused_where_a_number_is_expected(tmp_path: Path) -> None:
    """A TOML boolean under a numeric key is a wrong type, not a value of one."""

    layer = _write(tmp_path / "a.toml", "[window]\ndepth = true\n")

    with pytest.raises(SettingsError, match="not a boolean"):
        resolve_settings(SETTINGS, _sources(tmp_path), config_path=layer, **EMPTY)


def test_a_sibling_key_is_untouched_by_a_layer_that_omits_it(tmp_path: Path) -> None:
    layer = _write(tmp_path / "a.toml", "[window]\ndepth = 5\n")

    values, provenance = resolve_settings(
        SETTINGS, _sources(tmp_path), config_path=layer, **EMPTY
    )

    assert values["depth"] == 5
    assert values["glass"] == "clear"
    assert LAYER_FILE in provenance["window.depth"]
    assert provenance["window.glass"].startswith(LAYER_DEFAULT)


def test_a_table_where_one_setting_is_declared_is_refused(tmp_path: Path) -> None:
    layer = _write(tmp_path / "a.toml", "[window]\ndepth = { a = 1 }\n")

    with pytest.raises(SettingsError, match="single setting"):
        resolve_settings(SETTINGS, _sources(tmp_path), config_path=layer, **EMPTY)


def test_a_symlinked_missing_or_malformed_layer_is_refused_with_its_path(
    tmp_path: Path,
) -> None:
    link = tmp_path / "link.toml"
    link.symlink_to(_write(tmp_path / "real.toml", PACKAGED))
    broken = _write(tmp_path / "broken.toml", "this is = = not toml\n")

    with pytest.raises(SettingsError, match="must not be a symlink"):
        resolve_settings(SETTINGS, _sources(tmp_path), config_path=link, **EMPTY)
    with pytest.raises(SettingsError, match="not a readable file"):
        resolve_settings(
            SETTINGS,
            _sources(tmp_path),
            config_path=tmp_path / "gone.toml",
            **EMPTY,
        )
    with pytest.raises(SettingsError, match="not valid TOML") as refused:
        resolve_settings(SETTINGS, _sources(tmp_path), config_path=broken, **EMPTY)

    assert str(broken) in str(refused.value)


def test_a_layer_may_not_replace_the_packaged_defaults(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    broken = SettingsSources(
        default_file=tmp_path / "gone.toml",
        user_config=sources.user_config,
        project_root=sources.project_root,
        project_config=sources.project_config,
    )

    with pytest.raises(SettingsError, match="not a readable file"):
        resolve_settings(SETTINGS, broken, **EMPTY)


def test_a_setting_no_layer_supplied_is_refused(tmp_path: Path) -> None:
    partial = _write(tmp_path / "partial.toml", "[window]\ndepth = 3\n")

    with pytest.raises(SettingsError, match="No value for window.tide"):
        resolve_settings(
            SETTINGS,
            SettingsSources(
                default_file=partial,
                user_config=None,
                project_root=tmp_path,
                project_config=None,
            ),
            **EMPTY,
        )


def test_a_set_without_a_value_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="key=value"):
        resolve_settings(SETTINGS, _sources(tmp_path), overrides=["window.depth"])


def test_merge_reports_the_keys_a_layer_set() -> None:
    written = merge_settings({}, {"window": {"depth": 3}}, SETTINGS, source="test")

    assert written == {"window.depth"}


def test_override_reports_every_key_it_was_given() -> None:
    assert override_settings(["window.depth=3", "limits.work=0.5"], SETTINGS) == {
        "window.depth": "3",
        "limits.work": "0.5",
    }


def test_the_caller_owns_the_printed_order_and_the_section_names(
    tmp_path: Path,
) -> None:
    values, provenance = resolve_settings(
        SETTINGS,
        _sources(tmp_path),
        config_path=_write(tmp_path / "a.toml", '[window]\nglass = "frosted"\n'),
        **EMPTY,
    )

    printed = describe_settings(SETTINGS, values, provenance)

    for setting in SETTINGS:
        assert setting.key in printed
    assert printed.index("window.tide") < printed.index("model.name")
    assert "[limits]" in printed
    assert LAYER_FILE in printed
    assert "frosted" in printed


def test_a_registry_with_a_vocabulary_of_its_own_is_carried_untouched(
    tmp_path: Path,
) -> None:
    """`layer` is the caller's, so words this collection does not use are legal."""

    alien = (
        Setting(
            key="tide.state",
            field="tide",
            kind=str,
            layer="surf",
            doc="A vocabulary no server in this collection uses.",
        ),
    )
    defaults = _write(tmp_path / "alien.toml", '[tide]\nstate = "high"\n')

    values, provenance = resolve_settings(
        alien,
        SettingsSources(
            default_file=defaults,
            user_config=None,
            project_root=tmp_path,
            project_config=None,
        ),
        environ={},
    )

    assert values["tide"] == "high"
    assert {setting.layer for setting in alien} == {"surf"}
    assert "[tide]" in describe_settings(alien, values, provenance)


def test_the_path_helpers_take_the_callers_own_names(tmp_path: Path) -> None:
    default = default_config_path(Path(__file__))

    assert default.name == "default.toml"
    assert user_config_path("example-window").name == "config.toml"
    assert user_config_path("example-window") != user_config_path("other-window")
    assert project_config_path(tmp_path, Path(".example-rag") / "config.toml") == (
        tmp_path / ".example-rag" / "config.toml"
    )


def test_reading_a_document_returns_the_tables_it_holds(tmp_path: Path) -> None:
    document = read_config_document(
        _write(tmp_path / "a.toml", PACKAGED), source="test"
    )

    assert document["window"]["depth"] == 8
