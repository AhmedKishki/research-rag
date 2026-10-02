"""The browser workspace this app serves.

`app.py` builds the Starlette application and starts it on a loopback address,
and `contracts.py` declares the profile, the capabilities, and the adapter this
app implements. The workspace is given results and never a path: it reaches no
store of its own, so a request cannot name a file for it to read.
"""

from .app import create_ui_app, run_ui
from .contracts import (
    AdapterFactory,
    SourceFile,
    UIAdapter,
    UICapabilities,
    UIProfile,
    UIRequestError,
)

__all__ = [
    "AdapterFactory",
    "SourceFile",
    "UIAdapter",
    "UICapabilities",
    "UIProfile",
    "UIRequestError",
    "create_ui_app",
    "run_ui",
]
