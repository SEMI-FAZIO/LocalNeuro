"""LocalNeuro web UI: a browser-based interface for chat, generation,
checkpoint management and training.

The whole UI is built on the Python standard library (``http.server``) plus a
hand-written HTML/CSS/JS frontend -- no web framework and no new dependencies.
Launch it with ``python scripts/webui.py``.
"""

from .server import serve

__all__ = ["serve"]
