"""The project's own settings document: what a write lands in, and what it costs.

The shared settings library reads a TOML layer and refuses a symlink, and it
ships no writer, so this module is where a settings change becomes a file. The
serialiser covers the scalar kinds `SETTINGS` declares and nothing else, and a
write merges rather than overwrites: the document is read with the shared reader,
only the keys being changed are set, and every sibling key is written back as it
was, so a hand edit this writer does not understand survives a write made in a
browser.

The cost of a change is computed and never declared: `Setting.layer` is the
registry's own classification and mislabels keys no generation reads, so the cost
comes from applying a value and recomputing what a build records, which is the
code the build itself runs. `AGENTS.md` states the rule.

A write adopts the settled values in this process rather than resolving every
layer again, because the layers this app was started with are not recoverable
from the files it reads. The settings the dense backends and the gateway were
constructed from are named separately, because a change to one of those is read
by what this process already loaded rather than by the next search.

The description of a key is `Setting.doc` in `settings.py`. This module publishes
it to both readers, so the workspace's Config tab and `research-rag config` show
one sentence per key rather than a description written twice.
"""

from __future__ import annotations

import math
import os
import textwrap
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..generations.generation import value_fingerprint
from ..storage.records import fsync_directory
from .config import project_command
from .settings import SETTINGS, SETTINGS_BY_KEY, EffectiveSettings, Setting
from .settings_layers import (
    LAYER_COMMAND_LINE,
    LAYER_DEFAULT,
    LAYER_ENVIRONMENT,
    LAYER_FILE,
    LAYER_PROJECT,
    SettingsError,
    read_config_document,
)
from .support import ResearchError, retrieval_policy_fingerprint

#: The three costs a change can carry.
COST_NONE = "none"
COST_REGENERATION = "regeneration"
COST_MODEL = "model"

#: The layers that outrank the project file, so a project write cannot change a
#: value one of them supplied.
ABOVE_PROJECT_LAYERS = frozenset({LAYER_COMMAND_LINE, LAYER_ENVIRONMENT, LAYER_FILE})

#: The settings read once, by this process or by the objects it constructed. A
#: change to one of them is used by what is already loaded, not by the next
#: search, so it takes effect at the next start.
RESTART_SETTINGS = frozenset(
    {
        "runtime.model_cache_root",
        "runtime.log_level",
        "runtime.nice",
        "runtime.offline",
        "runtime.embedding_threads",
        "dense.embedding_model",
        "dense.reranker_model",
        "dense.embedding_inference_batch_size",
    }
)

# Each cost carries what it means as one sentence, held without its full stop so
# a sentence can name the keys before it and a list can put one after another.
_NONE_CLAUSE = (
    "nothing a generation records moves, so the next search uses the new value"
)
_REGENERATION_CLAUSE = (
    "a generation records this, so search answers stale until ingest builds one with it"
)
_EMBEDDING_CLAUSE = (
    "a generation records this model, so ingest recomputes every vector with it and "
    "downloads what it needs"
)
_RERANKER_CLAUSE = (
    "the next search reranks with this model and downloads it, and the stored vectors "
    "stay valid"
)
_CAUSE_CLAUSES = (
    ("embedding_model", _EMBEDDING_CLAUSE),
    ("reranker_model", _RERANKER_CLAUSE),
)

# A value no setting declares, used to ask whether a generation reads a field at
# all. It is never written anywhere.
_PROBE_TEXT = "settings-cost-probe"

#: Distinguishes "the caller proposed no value" from a proposed `None`, which is
#: what `runtime.embedding_threads` cleared is.
_NOT_GIVEN = object()


def _toml_string(value: str) -> str:
    """Return one string as a TOML basic string.

    A basic string carries any character a reader may type, so a newline inside a
    value is escaped rather than ending the line, and a quote or a backslash is
    escaped rather than closing or continuing the string.
    """

    escapes = {
        "\\": "\\\\",
        '"': '\\"',
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
    }
    return (
        '"'
        + "".join(
            escapes[character]
            if character in escapes
            else f"\\u{ord(character):04X}"
            if character < " " or character == "\x7f"
            else character
            for character in value
        )
        + '"'
    )


def _toml_value(value: Any, key: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    raise SettingsError(f"{key} is not a value this writer can write: {value!r}")


def read_project_document(path: Path) -> dict[str, Any]:
    """Read the project's settings layer with the shared reader's own refusals.

    A file nobody wrote is an empty document, and a symlink is refused before it
    is followed: the reader refuses one, and a writer that followed it would put
    this project's settings wherever the link points.
    """

    if path.is_symlink() or path.exists():
        return read_config_document(path, source=LAYER_PROJECT)
    return {}


def render_project_document(document: Mapping[str, Any]) -> str:
    """Return a document as TOML, one table per section and keys in registry order.

    Every key the document holds is written back, so a sibling key this writer
    does not know about survives a write that changed something else.
    """

    scalars = {
        key: value for key, value in document.items() if not isinstance(value, Mapping)
    }
    tables = {
        key: value for key, value in document.items() if isinstance(value, Mapping)
    }
    lines = [
        f"{key} = {_toml_value(value, key)}" for key, value in sorted(scalars.items())
    ]
    declared = list(dict.fromkeys(setting.key.split(".")[0] for setting in SETTINGS))
    sections = [name for name in declared if name in tables]
    sections += sorted(set(tables) - set(sections))
    for section in sections:
        entries = tables[section]
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        names = [
            setting.key.split(".", 1)[1]
            for setting in SETTINGS
            if setting.key.startswith(f"{section}.")
        ]
        names += sorted(set(entries) - set(dict.fromkeys(names)))
        for name in dict.fromkeys(names):
            if name in entries:
                lines.append(
                    f"{name} = {_toml_value(entries[name], f'{section}.{name}')}"
                )
    return "\n".join(lines) + "\n" if lines else ""


def write_project_document(path: Path, document: Mapping[str, Any]) -> None:
    """Replace the project's settings layer atomically.

    A peer process may be resolving this project at this moment, so the new
    document replaces the old one whole or not at all, and the directory entry is
    persisted because the file is read by that peer.
    """

    if path.is_symlink():
        raise SettingsError(f"{LAYER_PROJECT} must not be a symlink: {path}")
    text = render_project_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def merged_document(
    document: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in document.items()
    }
    for key, value in values.items():
        section, separator, name = key.partition(".")
        if not separator or not name:
            raise SettingsError(f"A setting key must name a section: {key}")
        merged.setdefault(section, {})[name] = value
    return merged


def _public_value(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def layer_of(origin: str) -> str:
    """The layer a provenance string names, before its path or variable."""

    return origin.split(" (", 1)[0]


def is_writable(origin: str) -> bool:
    return layer_of(origin) not in ABOVE_PROJECT_LAYERS


def settings_revision(
    settings: EffectiveSettings,
    provenance: Mapping[str, str],
) -> str:
    """Return the comparison token for one settings answer.

    It covers the values the reader was shown and the layer each came from, so a
    write carrying a stale one is refused rather than overwriting what that reader
    never saw.
    """

    return value_fingerprint(
        {
            "values": {
                name: _public_value(value)
                for name, value in settings.as_values().items()
            },
            "provenance": dict(provenance),
        }
    )


def _recorded_identity(settings: EffectiveSettings) -> dict[str, Any]:
    facts = settings.embedding_facts
    return {
        "retrieval_policy": retrieval_policy_fingerprint(settings),
        "chunking": (
            settings.chunk_size,
            settings.chunk_overlap,
            settings.chunk_headers,
        ),
        "embedding_model": (facts.name, facts.revision, facts.dimension),
        "reranker_model": settings.reranker_model,
    }


def change_cost(before: EffectiveSettings, after: EffectiveSettings) -> dict[str, Any]:
    recorded_before = _recorded_identity(before)
    recorded_after = _recorded_identity(after)
    moved = [
        name
        for name in recorded_before
        if recorded_before[name] != recorded_after[name]
    ]
    if "embedding_model" in moved:
        level, clause, requires_ingest = COST_MODEL, _EMBEDDING_CLAUSE, True
    elif "reranker_model" in moved:
        level, clause, requires_ingest = COST_MODEL, _RERANKER_CLAUSE, False
    elif moved:
        level, clause, requires_ingest = COST_REGENERATION, _REGENERATION_CLAUSE, True
    else:
        level, clause, requires_ingest = COST_NONE, _NONE_CLAUSE, False
    return {
        "level": level,
        "message": f"{clause[0].upper()}{clause[1:]}.",
        "moved": moved,
        "requires_ingest": requires_ingest,
    }


def _probe_value(setting: Setting, current: Any) -> Any | None:
    """Return one value of this setting that differs from the one in force.

    A setting that declares its choices or its bounds names its own alternatives
    and a boolean names its opposite. A free-form string declares nothing, so a
    value that cannot be the one in force stands in: the question being asked is
    whether a generation reads the field at all, and any different value answers
    it.
    """

    if setting.choices:
        return next((choice for choice in setting.choices if choice != current), None)
    if setting.kind is bool:
        return not current
    for bound in (setting.minimum, setting.maximum):
        if bound is not None and bound != current:
            return setting.kind(bound)
    if setting.kind is str:
        return f"{current}{_PROBE_TEXT}"
    return None


def setting_cost(
    settings: EffectiveSettings,
    setting: Setting,
    *,
    value: Any = _NOT_GIVEN,
) -> dict[str, Any]:
    """Return what changing one setting costs.

    A proposed value is costed against it. Without one, the setting is changed to
    an alternative it accepts and that change's cost is the answer, which is why a
    key no generation reads costs nothing.
    """

    if value is _NOT_GIVEN:
        value = _probe_value(setting, settings.value(setting.key))
        if value is None:
            return _none_cost()
        # The probe answers whether a generation reads this field, so it is not
        # put through the registry's own checks: a free-form string has no declared
        # domain to probe with, and the question is about the field rather than
        # about the value.
        return change_cost(settings, replace(settings, **{setting.field: value}))
    proposed = EffectiveSettings.from_values(
        {**settings.as_values(), setting.field: value}
    )
    return change_cost(settings, proposed)


def _none_cost() -> dict[str, Any]:
    return {
        "level": COST_NONE,
        "message": f"{_NONE_CLAUSE[0].upper()}{_NONE_CLAUSE[1:]}.",
        "moved": [],
        "requires_ingest": False,
    }


def setting_costs(settings: EffectiveSettings) -> dict[str, dict[str, Any]]:
    return {setting.key: setting_cost(settings, setting) for setting in SETTINGS}


def key_costs(
    before: EffectiveSettings,
    after: EffectiveSettings,
    keys: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return what each changed key costs on its own.

    A key is costed against the change with that key undone, so a key that moves
    nothing of the proposed values is not charged for another key's move. A change
    that only stands together is charged whole, because undoing one key alone is
    not a set of values the registry accepts.
    """

    costs: dict[str, dict[str, Any]] = {}
    after_values = after.as_values()
    for key in keys:
        setting = SETTINGS_BY_KEY[key]
        try:
            without = EffectiveSettings.from_values(
                {**after_values, setting.field: before.value(key)}
            )
        except SettingsError:
            costs[key] = change_cost(before, after)
            continue
        costs[key] = change_cost(without, after)
    return costs


def describe_costs(settings: EffectiveSettings, *, width: int = 78) -> str:
    """Return the cost of changing every key, one group per level.

    The command line prints this beside the effective values, so a reader learns
    what a key costs before editing the file the workspace writes.
    """

    costs = setting_costs(settings)
    groups: dict[str, list[str]] = {}
    for setting in SETTINGS:
        groups.setdefault(costs[setting.key]["level"], []).append(setting.key)
    lines = ["", "What a change to each key costs, from what a generation records:"]
    for level in (COST_MODEL, COST_REGENERATION, COST_NONE):
        keys = groups.get(level)
        if not keys:
            continue
        lines.extend(
            textwrap.wrap(
                ", ".join(keys),
                width=width,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
        lines.extend(
            textwrap.wrap(
                costs[keys[0]]["message"],
                width=width,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
    return "\n".join(lines)


def describe_docs(*, width: int = 78) -> str:
    """Return what each key does, beside its key, grouped by section.

    The sentence is `Setting.doc`, printed from the registry rather than written
    here, so this answer and the workspace's Config tab read one description and
    cannot disagree.
    """

    lines = ["", "What each key does, from the registry that declares it:"]
    for section in dict.fromkeys(setting.key.split(".")[0] for setting in SETTINGS):
        lines.extend(("", f"[{section}]"))
        for setting in SETTINGS:
            if not setting.key.startswith(f"{section}."):
                continue
            indent = " " * (2 + len(setting.key) + 2)
            body = textwrap.wrap(
                setting.doc,
                width=max(24, width - len(indent)),
                break_on_hyphens=False,
            ) or [""]
            lines.append(f"  {setting.key}  {body[0]}")
            lines.extend(f"{indent}{extra}" for extra in body[1:])
    return "\n".join(lines)


def _label_for(setting: Setting) -> str:
    """A human name for a key, taken from the key itself."""

    leaf = setting.key.split(".")[-1].replace("_", " ")
    return leaf[:1].upper() + leaf[1:]


def declared_facts(setting: Setting) -> dict[str, Any]:
    """What this setting declares about its own domain, and nothing it does not.

    A bound, a fixed set of choices, and a variable name are present only where
    the registry declares one, so a reader cannot tell an unbounded setting from
    one whose bound was never loaded. The keys travel the way the registry
    spells them, so no surface retypes them.
    """

    facts: dict[str, Any] = {}
    if setting.minimum is not None:
        facts["minimum"] = setting.minimum
    if setting.maximum is not None:
        facts["maximum"] = setting.maximum
    if setting.choices:
        facts["choices"] = list(setting.choices)
    if setting.env:
        facts["env"] = setting.env
    return facts


def _refusal_for_layer(key: str, origin: str, path: Path) -> str:
    layer = layer_of(origin)
    if layer == LAYER_ENVIRONMENT:
        return (
            f"{key} is set by {origin}, which a project file cannot override. "
            f"Change the variable, or leave it unset and write the value in {path}."
        )
    if layer == LAYER_COMMAND_LINE:
        return (
            f"{key} is set by the command line this app was started with, which a "
            f"project file cannot override. Start the app without --set {key}=VALUE "
            f"and write the value in {path}."
        )
    return (
        f"{key} is set by {origin}, which outranks the project file. Drop that "
        f"layer and write the value in {path}."
    )


def _consequences(costs: Mapping[str, dict[str, Any]]) -> list[tuple[str, list[str]]]:
    causes: list[tuple[str, list[str]]] = []
    for cause, clause in _CAUSE_CLAUSES:
        keys = sorted(key for key, cost in costs.items() if cause in cost["moved"])
        if keys:
            causes.append((clause, keys))
    rest = sorted(
        key for key, cost in costs.items() if cost["level"] == COST_REGENERATION
    )
    if rest:
        causes.append((_REGENERATION_CLAUSE, rest))
    return causes


class SettingsWorkflow:
    """Read the settings this process resolved, and write the ones a project owns.

    Both are one service method each, so the workspace and the command line answer
    from the same values and refuse the same writes.
    """

    def project_settings_path(self) -> Path:
        """The one file a settings write lands in."""

        return self.config.portable_root / "config.toml"

    def section_rows(self) -> list[dict[str, Any]]:
        """Every setting, grouped by section, in registry order.

        Each row carries the sentence the registry declares for that key and the
        domain it declares, so a reader is told what the key does without a
        surface writing a description of its own.
        """

        costs = setting_costs(self.config.settings)
        rows: list[dict[str, Any]] = []
        for section in dict.fromkeys(setting.key.split(".")[0] for setting in SETTINGS):
            entries = []
            for setting in SETTINGS:
                if not setting.key.startswith(f"{section}."):
                    continue
                origin = self.config.settings_provenance.get(setting.key, LAYER_DEFAULT)
                cost = costs[setting.key]
                entries.append(
                    {
                        "key": setting.key,
                        "doc": setting.doc,
                        "label": _label_for(setting),
                        "value": _public_value(self.config.settings.value(setting.key)),
                        "kind": setting.kind.__name__,
                        "layer": setting.layer,
                        "origin": origin,
                        "writable": is_writable(origin),
                        "cost": {
                            "level": cost["level"],
                            "message": cost["message"],
                        },
                        **declared_facts(setting),
                    }
                )
            rows.append(
                {
                    "key": section,
                    "title": section[:1].upper() + section[1:],
                    "settings": entries,
                }
            )
        return rows

    async def settings_read(self) -> dict[str, Any]:
        """Return the settings this process resolved, and what each key costs.

        Nothing is resolved again, so a reader sees the values the retrieval stack
        and the next build actually use, each named with the layer that supplied it,
        the sentence that says what the key does, and the cost of changing it.
        """

        return {
            "revision": settings_revision(
                self.config.settings, self.config.settings_provenance
            ),
            "sections": self.section_rows(),
            "message": (
                f"{len(SETTINGS)} settings; a change to a writable key is written to "
                f"{self.project_settings_path()}."
            ),
        }

    async def settings_write(
        self,
        values: Mapping[str, Any],
        *,
        expected_revision: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Write this project's own settings, refusing what it must not change.

        The values are coerced by the same call every layer is coerced by, so a
        number typed in a browser and the same number in the file are refused the
        same way, and the reader is told which layer each value would come from.
        A change that moves what a generation records is refused until the caller
        confirms it, naming the keys, because that warning is what lets a reader
        choose to rebuild rather than meet it in an answer.
        """

        if not values:
            raise ResearchError("A settings write needs at least one value")
        path = self.project_settings_path()
        source = f"{LAYER_PROJECT} ({path})"
        async with self._operation(
            busy_command=project_command(self.config.project_root, "status")
        ):
            current = self.config.settings
            provenance = dict(self.config.settings_provenance)
            revision = settings_revision(current, provenance)
            if expected_revision != revision:
                raise ResearchError(
                    "The settings moved since they were read: this answer holds "
                    f"revision {expected_revision} and the app now holds {revision}. "
                    "Nothing was written. Read the settings again and reapply the "
                    "change."
                )
            try:
                document = read_project_document(path)
            except SettingsError as exc:
                raise ResearchError(str(exc)) from exc
            coerced: dict[str, Any] = {}
            try:
                for key, raw in values.items():
                    setting = SETTINGS_BY_KEY.get(key)
                    if setting is None:
                        raise ResearchError(f"{source} sets an unknown setting: {key}")
                    origin = provenance.get(key, LAYER_DEFAULT)
                    if not is_writable(origin):
                        raise ResearchError(_refusal_for_layer(key, origin, path))
                    coerced[setting.field] = setting.coerce(raw, source=source)
                proposed = EffectiveSettings.from_values(
                    {**current.as_values(), **coerced}
                )
            except SettingsError as exc:
                raise ResearchError(str(exc)) from exc

            changed = sorted(
                key for key in values if proposed.value(key) != current.value(key)
            )
            if not changed:
                return {
                    "revision": revision,
                    "changed": [],
                    "requires_ingest": False,
                    "message": f"No setting changed; {path} was not written.",
                }
            costs = key_costs(current, proposed, changed)
            if not confirm:
                warning = _cost_warning(costs, changed, self.config.project_root)
                if warning:
                    raise ResearchError(warning)
            try:
                write_project_document(
                    path,
                    merged_document(
                        document, {key: proposed.value(key) for key in changed}
                    ),
                )
            except SettingsError as exc:
                raise ResearchError(str(exc)) from exc

            settled = {
                **provenance,
                **{key: source for key in changed},
            }
            # The layers this app was started with are not in the files it reads,
            # so a write settles the values rather than resolving them again: each
            # key it wrote sits above the user file and below nothing that outranks
            # it.
            self.config = replace(
                self.config, settings=proposed, settings_provenance=settled
            )
            self.retrieval_policy_fingerprint = retrieval_policy_fingerprint(proposed)
            return {
                "revision": settings_revision(proposed, settled),
                "changed": changed,
                "requires_ingest": any(
                    cost["requires_ingest"] for cost in costs.values()
                ),
                "message": _saved_message(path, changed, costs),
            }


def _cost_warning(
    costs: Mapping[str, dict[str, Any]],
    keys: Iterable[str],
    project_root: Path,
) -> str:
    consequences = _consequences({key: costs[key] for key in keys})
    if not consequences:
        return ""
    named = " ".join(
        f"{', '.join(named_keys)}: {clause}." for clause, named_keys in consequences
    )
    return (
        f"These settings were not written. {named} Rebuild with "
        f"{project_command(project_root, 'ingest')} once they are saved, or call "
        "this again with confirm to write them now."
    )


def _saved_message(
    path: Path,
    changed: Iterable[str],
    costs: Mapping[str, dict[str, Any]],
) -> str:
    count = len(costs)
    parts = [f"Saved {count} {'value' if count == 1 else 'values'} to {path}."]
    parts.extend(
        f"{', '.join(named_keys)}: {clause}."
        for clause, named_keys in _consequences(costs)
    )
    restart = sorted(set(changed) & RESTART_SETTINGS)
    if restart:
        parts.append(
            "What this app already loaded keeps its own values until it restarts: "
            f"{', '.join(restart)}."
        )
    return " ".join(parts)
