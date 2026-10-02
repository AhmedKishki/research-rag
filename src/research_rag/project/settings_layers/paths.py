"""Where each layer lives, named by the app.

The account directory, the project overlay, and the packaged default file are
three different places, and each is named by the module that owns it rather than
here. Every helper therefore takes the name it resolves, and this module knows
none of them.
"""

from __future__ import annotations

from pathlib import Path

import platformdirs

#: The file name every account and project overlay is called.
CONFIG_FILENAME = "config.toml"

#: The file name the packaged defaults are called, in the caller's package.
DEFAULT_CONFIG_FILENAME = "default.toml"


def user_config_path(app_name: str) -> Path:
    """Return the account-wide overlay path for this platform.

    `app_name` is this app's own account directory name, and it is `research-rag`
    because the settings directory is one of the names this app keeps.
    """

    return platformdirs.user_config_path(app_name) / CONFIG_FILENAME


def project_config_path(project_root: str | Path, relative: str | Path) -> Path:
    """Return the project overlay, placed by the caller's own state directory.

    `relative` is the path inside a project, such as `.research-rag/config.toml`,
    so the overlay travels with the project it configures.
    """

    return Path(project_root) / relative


def default_config_path(module_file: str | Path) -> Path:
    """Return the packaged default file beside the module that declares it.

    `module_file` is that module's `__file__`. The default file is this app's
    data and ships inside this app's package, never in this one.
    """

    return Path(module_file).with_name(DEFAULT_CONFIG_FILENAME)
