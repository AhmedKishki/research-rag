"""The configuration layers behind this app's settings.

`settings.py` declares a registry of settings, `default.toml` ships the defaults
it needs, and this package resolves the layers that override them:

    default.toml  <  user config  <  project config  <  --config  <  environment  <  --set

Each later layer wins **per key** and names only what it changes, so a layer may
set one key and leave the rest inherited from the layer below. Every effective
value reports the layer that supplied it, so `research-rag` can print where a number
came from rather than only what it is.

Three rules follow from the registry `settings.py` passes in:

* a key the registry does not declare is an error in every layer, so a typo is
  refused rather than ignored;
* one coercion path serves every layer, so a number written in a file and the
  same number written in the environment or after `--set` are read the same way
  and fail the same way; and
* the vocabulary is the app's. `Setting.layer` is a plain string, the path
  helpers take the names this app declares, and the registry is `settings.py`'s.

This package holds the layering and nothing else: no registry of its own, no
default file, and no domain type.
"""

from .paths import default_config_path, project_config_path, user_config_path
from .settings import (
    LAYER_COMMAND_LINE,
    LAYER_DEFAULT,
    LAYER_ENVIRONMENT,
    LAYER_FILE,
    LAYER_PROJECT,
    LAYER_USER,
    Setting,
    SettingsError,
    SettingsSources,
    describe_settings,
    environment_settings,
    merge_settings,
    override_settings,
    read_config_document,
    resolve_settings,
)

__all__ = [
    "LAYER_COMMAND_LINE",
    "LAYER_DEFAULT",
    "LAYER_ENVIRONMENT",
    "LAYER_FILE",
    "LAYER_PROJECT",
    "LAYER_USER",
    "Setting",
    "SettingsError",
    "SettingsSources",
    "default_config_path",
    "describe_settings",
    "environment_settings",
    "merge_settings",
    "override_settings",
    "project_config_path",
    "read_config_document",
    "resolve_settings",
    "user_config_path",
]
