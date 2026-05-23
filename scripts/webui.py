"""Launch the LocalNeuro web UI.

Starts a local server with a browser-based interface for chat, text
generation, checkpoint management and training. Built on the Python standard
library -- no extra dependencies.

Examples
--------
    python scripts/webui.py
    python scripts/webui.py --port 9000 --checkpoint checkpoints/run
    python scripts/webui.py --host 0.0.0.0          # reachable on your LAN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.utils import configure_console
from localneuro.webui.server import serve


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (use 0.0.0.0 to allow LAN access)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint directory to load on startup")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window automatically")
    args = parser.parse_args()

    serve(
        project_root=Path(__file__).resolve().parent.parent,
        host=args.host,
        port=args.port,
        device=args.device,
        open_browser=not args.no_browser,
        load_checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
