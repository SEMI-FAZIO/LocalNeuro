"""The LocalNeuro web UI HTTP server.

Built entirely on the Python standard library (``http.server``). It serves the
static frontend and a small JSON API. Streaming endpoints (chat, generation,
training metrics) use HTTP chunked transfer encoding to push newline-delimited
JSON events to the browser as they happen -- no web framework, no new
dependencies.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, Optional, Union
from urllib.parse import urlparse

from ..sampling import SamplingConfig
from .state import WebUIState
from .training_job import TrainingJob

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _sampling_from(data: Dict[str, object]) -> SamplingConfig:
    """Build a (validated) SamplingConfig from a request's ``sampling`` dict."""
    return SamplingConfig(
        temperature=float(data.get("temperature", 0.8)),
        top_k=int(data.get("top_k", 40)),
        top_p=float(data.get("top_p", 0.95)),
        repetition_penalty=float(data.get("repetition_penalty", 1.1)),
    )


class _Server(ThreadingHTTPServer):
    """Threaded HTTP server carrying the shared application objects."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, state: WebUIState, training: TrainingJob):
        super().__init__(address, handler)
        self.state = state
        self.training = training


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to the JSON API, the streaming API and static files."""

    protocol_version = "HTTP/1.1"
    server_version = "LocalNeuro"

    # Quiet by default -- the streaming endpoints would otherwise spam stderr.
    def log_message(self, *args) -> None:  # noqa: D102
        pass

    @property
    def state(self) -> WebUIState:
        return self.server.state  # type: ignore[attr-defined]

    @property
    def training(self) -> TrainingJob:
        return self.server.training  # type: ignore[attr-defined]

    # --------------------------------------------------------- routing
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_static("index.html")
        elif path.startswith("/static/"):
            self._send_static(path[len("/static/"):])
        elif path == "/favicon.ico":
            self._send_empty(204)
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/checkpoints":
            self._api_checkpoints()
        elif path == "/api/train/status":
            self._send_json(self.training.snapshot())
        elif path == "/api/train/events":
            self._api_train_events()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        self._body = self.rfile.read(length) if length > 0 else b""
        path = urlparse(self.path).path
        routes = {
            "/api/load": self._api_load,
            "/api/unload": self._api_unload,
            "/api/chat": self._api_chat,
            "/api/generate": self._api_generate,
            "/api/quantize": self._api_quantize,
            "/api/train/start": self._api_train_start,
            "/api/train/stop": self._api_train_stop,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_json({"error": "not found"}, 404)
        else:
            handler()

    # ------------------------------------------------------- responses
    def _send_json(self, obj: object, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_static(self, name: str) -> None:
        target = (_STATIC_DIR / name).resolve()
        # Path-traversal guard: the resolved path must stay inside static/.
        if _STATIC_DIR not in target.parents or not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            _CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _body_json(self) -> Dict[str, object]:
        if not self._body:
            return {}
        return json.loads(self._body.decode("utf-8"))

    # --------------------------------------------------------- streaming
    def _stream(self, events: Iterator[Dict[str, object]]) -> None:
        """Send an iterator of event dicts as a chunked NDJSON response."""
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for event in events:
                self._write_chunk(json.dumps(event, ensure_ascii=False) + "\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            pass  # the browser disconnected -- stop quietly
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()  # runs the generator's finally (releases locks)

    def _write_chunk(self, text: str) -> None:
        payload = text.encode("utf-8")
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    # --------------------------------------------------------- API: GET
    def _api_status(self) -> None:
        self._send_json({
            "model": self.state.model_info(),
            "device": str(self.state.device),
            "training": {
                "status": self.training.status,
                "running": self.training.is_running(),
            },
        })

    def _api_checkpoints(self) -> None:
        self._send_json({
            "checkpoints": self.state.list_checkpoints(),
            "configs": self.state.list_configs(),
        })

    def _api_train_events(self) -> None:
        """Stream training metric events until the run ends."""
        job = self.training
        channel = job.subscribe()

        def events() -> Iterator[Dict[str, object]]:
            try:
                while True:
                    try:
                        event = channel.get(timeout=15.0)
                    except queue.Empty:
                        if not job.is_running():
                            break
                        yield {"type": "ping"}
                        continue
                    yield event
                    if event.get("type") == "done":
                        break
            finally:
                job.unsubscribe(channel)

        self._stream(events())

    # -------------------------------------------------------- API: POST
    def _api_load(self) -> None:
        try:
            body = self._body_json()
            info = self.state.load(str(body["path"]))
            self._send_json({"ok": True, "model": info})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def _api_unload(self) -> None:
        self.state.unload()
        self._send_json({"ok": True})

    def _api_chat(self) -> None:
        try:
            body = self._body_json()
            sampling = _sampling_from(body.get("sampling", {}))
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        self._stream(self.state.chat_stream(
            message=str(body.get("message", "")),
            history=body.get("history", []),
            system=body.get("system"),
            sampling=sampling,
            max_new_tokens=int(body.get("max_new_tokens", 200)),
        ))

    def _api_generate(self) -> None:
        try:
            body = self._body_json()
            sampling = _sampling_from(body.get("sampling", {}))
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        seed = body.get("seed")
        self._stream(self.state.generate_stream(
            prompt=str(body.get("prompt", "")),
            sampling=sampling,
            max_new_tokens=int(body.get("max_new_tokens", 120)),
            seed=int(seed) if seed not in (None, "") else None,
        ))

    def _api_quantize(self) -> None:
        try:
            body = self._body_json()
            result = self.state.quantize(
                input_path=str(body["input"]),
                output_path=str(body["output"]),
                bits=int(body.get("bits", 8)),
            )
            self._send_json({"ok": True, **result})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def _api_train_start(self) -> None:
        try:
            self.training.start(self._body_json())
            self._send_json({"ok": True})
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, 409)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def _api_train_stop(self) -> None:
        self.training.stop()
        self._send_json({"ok": True})


def serve(
    project_root: Union[str, Path] = ".",
    host: str = "127.0.0.1",
    port: int = 8080,
    device: str = "auto",
    open_browser: bool = True,
    load_checkpoint: Optional[str] = None,
) -> None:
    """Start the LocalNeuro web UI and block until interrupted."""
    project_root = Path(project_root).resolve()
    state = WebUIState(project_root, device=device)
    training = TrainingJob(project_root)

    if load_checkpoint:
        try:
            state.load(load_checkpoint)
            print(f"loaded checkpoint: {load_checkpoint}")
        except Exception as exc:
            print(f"warning: could not load '{load_checkpoint}': {exc}")

    httpd = _Server((host, port), _Handler, state, training)
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/"
    print(f"LocalNeuro web UI running at {url}")
    print("press Ctrl+C to stop")

    if open_browser:
        def _open() -> None:
            time.sleep(0.7)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.shutdown()
        httpd.server_close()
