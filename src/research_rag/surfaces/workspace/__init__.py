"""The browser workspace this app serves.

`app.py` builds the Starlette application, and `contracts.py` declares the adapter this app implements. The workspace is
given results and never a path: it reaches no store of its own, so a request
cannot name a file for it to read.
"""

from .app import create_ui_app
from .contracts import SourceFile, UIAdapter, UIRequestError

__all__ = [
    "SourceFile",
    "UIAdapter",
    "UIRequestError",
    "create_ui_app",
]
