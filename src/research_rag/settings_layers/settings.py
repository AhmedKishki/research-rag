"""The registry, the layer stack, and the coercion every layer shares.

The app declares its tunables as a sequence of `Setting` and passes that registry
to every function here. Nothing in this module holds a registry of its own, so
the vocabulary of what each setting costs is `settings.py`'s to choose.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The layer names, lowest precedence first. Provenance reports these strings, so
#: a caller printing its effective values can say where each one came from.
LAYER_DEFAULT = "default"
LAYER_USER = "user config"
LAYER_PROJECT = "project config"
LAYER_FILE = "--config"
LAYER_ENVIRONMENT = "environment"
LAYER_COMMAND_LINE = "command line"


class SettingsError(ValueError):
    """Raised when a settings layer is unreadable, unknown, or out of bounds."""


@dataclass(frozen=True, slots=True)
class Setting:
    """One tunable: where it lives, what it accepts, and how it is named.

    `key` is the dotted name a file uses, `field` the attribute it lands on, and
    `layer` this app's classification of what changing it costs.

    `env` is the variable name that sets the value, declared rather than derived,
    because a variable named here is the one the environment layer reads. A
    setting with no `env` has no environment layer.
    """

    key: str
    field: str
    kind: type
    layer: str
    doc: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    env: str = ""
    # What "" means, where it means something: a model turned off, a cache the
    # caller's own package holds. A setting without `empty` reads "" as the empty
    # string when it is a string, and refuses it otherwise.
    empty: str | None = None
    # A setting that names one of a fixed set of modes also accepts any spelling
    # of them. A setting that names a model, a path, or a revision does not:
    # those are case-sensitive identifiers, not mode words.
    normalize_case: bool = False

    def coerce(self, raw: Any, *, source: str) -> Any:
        """Return `raw` as this setting's type, or refuse it by name.

        One call serves every layer, so a number written in a file and the same
        number written in the environment or after `--set` are read one way and
        fail one way. `source` names the layer, and every message repeats it.
        """

        where = f"{self.key} ({source})"
        if isinstance(raw, bool) and self.kind is not bool:
            raise SettingsError(f"{where} must be {self.kind.__name__}, not a boolean")
        if self.kind is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.strip().casefold() in {"true", "false"}:
                return raw.strip().casefold() == "true"
            raise SettingsError(f"{where} must be true or false")
        if isinstance(raw, str) and not raw.strip():
            if self.empty is not None:
                return self.empty
            if self.kind is str:
                return ""
        try:
            value = self.kind(raw)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"{where} must be {self.kind.__name__}") from exc
        if self.kind is str:
            # Surrounding whitespace is never meaningful in a settings value.
            value = value.strip()
            if self.normalize_case:
                value = value.casefold()
        if self.choices and value not in self.choices:
            raise SettingsError(f"{where} must be one of: " + ", ".join(self.choices))
        if self.minimum is not None and value < self.minimum:
            raise SettingsError(f"{where} must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise SettingsError(f"{where} must be at most {self.maximum:g}")
        return value


@dataclass(frozen=True, slots=True)
class SettingsSources:
    """Where the file layers are, as this app names them.

    `default_file` is the app's own packaged defaults, `user_config` the
    account-wide overlay, and `project_config` the project's own. A file layer
    set to `None` is a layer this app does not have, and a path that does not
    exist is a layer nobody wrote, so both are read only when they are there.
    """

    default_file: Path
    user_config: Path | None
    project_root: Path
    project_config: Path | None


def read_config_document(path: Path, *, source: str) -> dict[str, Any]:
    """Read one TOML layer, refusing anything that is not a plain document."""

    if path.is_symlink():
        raise SettingsError(f"{source} must not be a symlink: {path}")
    if not path.is_file():
        raise SettingsError(f"{source} is not a readable file: {path}")
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"{source} is not valid TOML ({path}): {exc}") from exc
    except OSError as exc:
        raise SettingsError(f"{source} cannot be read ({path}): {exc}") from exc
    if not isinstance(document, dict):
        raise SettingsError(f"{source} must contain a TOML table: {path}")
    return document


def merge_settings(
    base: dict[str, Any],
    overlay: Mapping[str, Any],
    registry: Sequence[Setting],
    *,
    source: str,
    prefix: str = "",
) -> set[str]:
    """Merge one layer into `base` per key and return the keys it set.

    Tables merge recursively, so a layer names only what it changes. A scalar or
    an array the overlay supplies replaces whatever the layer below had. A key
    the registry does not declare is an error, which is what makes a typo loud
    instead of silent, and so is a table given where the registry declares a
    single setting.
    """

    known = {setting.key for setting in registry}
    written: set[str] = set()
    for key, value in overlay.items():
        if not isinstance(key, str):
            raise SettingsError(f"{source} has a non-string key: {key!r}")
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            if dotted in known:
                raise SettingsError(
                    f"{source} gives a table for the single setting {dotted}"
                )
            written |= merge_settings(
                base,
                value,
                registry,
                source=source,
                prefix=f"{dotted}.",
            )
            continue
        if dotted not in known:
            raise SettingsError(f"{source} sets an unknown setting: {dotted}")
        base[dotted] = value
        written.add(dotted)
    return written


def environment_settings(
    environ: Mapping[str, str],
    registry: Sequence[Setting],
) -> dict[str, tuple[str, str]]:
    """Return the environment layer as `key -> (raw value, variable name)`.

    A variable that is unset, or set to nothing, is not a layer saying anything,
    so a value that may be turned off is set in a file or after `--set` rather
    than by an empty variable.
    """

    values: dict[str, tuple[str, str]] = {}
    for setting in registry:
        if not setting.env:
            continue
        raw = environ.get(setting.env)
        if raw is None or not raw.strip():
            continue
        values[setting.key] = (raw.strip(), setting.env)
    return values


def override_settings(
    overrides: Sequence[str],
    registry: Sequence[Setting],
) -> dict[str, str]:
    """Return the command-line layer from repeated `key=value` pairs."""

    known = {setting.key for setting in registry}
    values: dict[str, str] = {}
    for item in overrides:
        key, separator, raw = str(item).partition("=")
        key = key.strip()
        if not separator or not key:
            raise SettingsError(f"--set expects key=value, got: {item!r}")
        if key not in known:
            raise SettingsError(
                f"--set names an unknown setting: {key}; the settings are "
                + ", ".join(sorted(known))
            )
        values[key] = raw.strip()
    return values


def resolve_settings(
    registry: Sequence[Setting],
    sources: SettingsSources,
    *,
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge every layer and return the values with the layer that supplied each.

    The values are keyed by `field`, so a caller builds its own settings object
    from them; the provenance is keyed by `key`. Nothing is read relative to the
    working directory: a client may start a server anywhere, so every path is
    placed by `sources` rather than by where the process happens to be.
    """

    environment = os.environ if environ is None else environ
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    layers: list[tuple[str, Path]] = [(LAYER_DEFAULT, Path(sources.default_file))]
    for name, candidate in (
        (LAYER_USER, sources.user_config),
        (LAYER_PROJECT, sources.project_config),
    ):
        if candidate is not None and candidate.exists():
            layers.append((name, candidate))
    if config_path is not None:
        layers.append((LAYER_FILE, Path(config_path).expanduser()))

    for name, layer_path in layers:
        document = read_config_document(layer_path, source=name)
        written = merge_settings(
            merged, document, registry, source=f"{name} ({layer_path})"
        )
        for key in written:
            provenance[key] = f"{name} ({layer_path})"

    for key, (raw, variable) in environment_settings(environment, registry).items():
        merged[key] = raw
        provenance[key] = f"{LAYER_ENVIRONMENT} ({variable})"

    for key, raw in override_settings(overrides, registry).items():
        merged[key] = raw
        provenance[key] = LAYER_COMMAND_LINE

    values: dict[str, Any] = {}
    for setting in registry:
        if setting.key not in merged:
            raise SettingsError(
                f"No value for {setting.key}: the packaged default must declare "
                "every setting"
            )
        values[setting.field] = setting.coerce(
            merged[setting.key],
            source=provenance.get(setting.key, LAYER_DEFAULT),
        )
        provenance.setdefault(setting.key, LAYER_DEFAULT)

    return values, provenance


def describe_settings(
    registry: Sequence[Setting],
    values: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> str:
    """Render the effective values, one line per key, with the layer that set it.

    Grouped by the section each key's name begins with, and in registry order
    inside a section, so the registry is what orders the output.
    """

    if not registry:
        raise SettingsError("The registry declares no settings")
    width = max(len(setting.key) for setting in registry)
    lines: list[str] = [
        "Effective settings, later layers overriding earlier ones:",
        "  default.toml < user config < project config < --config < environment < --set",
    ]
    for section in dict.fromkeys(setting.key.split(".")[0] for setting in registry):
        lines.append("")
        lines.append(f"[{section}]")
        for setting in registry:
            if not setting.key.startswith(f"{section}."):
                continue
            value = values[setting.field]
            rendered = '""' if value is None else repr(value)
            if isinstance(value, str):
                rendered = f'"{value}"'
            lines.append(
                f"  {setting.key:<{width}} = {rendered:<28} "
                f"# {provenance.get(setting.key, LAYER_DEFAULT)}"
            )
    return "\n".join(lines)
