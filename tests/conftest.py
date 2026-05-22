"""Pytest configuration: make the ``localneuro`` package importable.

This lets the test-suite run directly from a source checkout without first
installing the package (``pip install -e .``).
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
