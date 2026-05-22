"""Web UI inference state: model loading, checkpoint discovery and streaming.

This object is the inference-side brain of the web UI. It owns the currently
loaded model, discovers checkpoints on disk, and exposes streaming chat and
text generation as plain Python generators of event dicts -- which the HTTP
server turns into a chunked NDJSON response.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from ..checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint
from ..config import ModelConfig
from ..inference.chat import ChatSession
from ..inference.engine import InferenceEngine
from ..quantization.quantize import quantize_model
from ..sampling import SamplingConfig
from ..utils import human_count, resolve_device

PathLike = Union[str, Path]


class WebUIState:
    """Holds the loaded model and serves inference requests for the web UI."""

    def __init__(self, project_root: PathLike, device: str = "auto") -> None:
        self.project_root = Path(project_root).resolve()
        self.device = resolve_device(device)
        self.engine: Optional[InferenceEngine] = None
        self.active_checkpoint: Optional[Path] = None
        # Generation is serialized -- the model is a single shared object.
        self._lock = threading.Lock()

    # ------------------------------------------------------------ discovery
    def _relname(self, path: Path) -> str:
        """A short display name: path relative to the project root if possible."""
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return str(path)

    def _describe_checkpoint(self, directory: Path) -> Dict[str, object]:
        config = ModelConfig.load(directory / "config.json")
        meta: Dict[str, object] = {}
        meta_path = directory / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        weights = directory / "weights.lnw"
        resolved = directory.resolve()
        return {
            "path": str(resolved),
            "name": self._relname(directory),
            "params": config.num_params(),
            "params_human": human_count(config.num_params()),
            "n_layers": config.n_layers,
            "d_model": config.d_model,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "step": meta.get("step"),
            "quantization": meta.get("quantization"),
            "size_bytes": weights.stat().st_size if weights.exists() else 0,
            "has_tokenizer": (directory / "tokenizer.json").exists(),
            "active": self.active_checkpoint is not None
            and resolved == self.active_checkpoint,
        }

    def list_checkpoints(self) -> List[Dict[str, object]]:
        """Find every checkpoint bundle under ``checkpoints/``."""
        found: List[Dict[str, object]] = []
        root = self.project_root / "checkpoints"
        if root.is_dir():
            for config_path in sorted(root.rglob("config.json")):
                directory = config_path.parent
                if not checkpoint_exists(directory):
                    continue
                try:
                    found.append(self._describe_checkpoint(directory))
                except Exception:
                    continue  # skip an unreadable / malformed checkpoint
        return found

    def list_configs(self) -> List[Dict[str, object]]:
        """List the model-size presets in ``configs/``."""
        presets: List[Dict[str, object]] = []
        root = self.project_root / "configs"
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                try:
                    config = ModelConfig.load(path)
                except Exception:
                    continue
                presets.append({
                    "name": path.stem,
                    "params_human": human_count(config.num_params()),
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "max_seq_len": config.max_seq_len,
                })
        return presets

    # ----------------------------------------------------------- model load
    def load(self, path: PathLike) -> Dict[str, object]:
        """Load a checkpoint and make it the active inference model."""
        directory = Path(path).resolve()
        if not checkpoint_exists(directory):
            raise ValueError(f"not a checkpoint directory: {directory}")
        bundle = load_checkpoint(
            directory, build_model=True,
            map_location=str(self.device), mmap=True,
        )
        if bundle.tokenizer is None:
            raise ValueError("checkpoint has no tokenizer.json")
        engine = InferenceEngine(bundle.model, bundle.tokenizer, device=self.device)
        with self._lock:
            self.engine = engine
            self.active_checkpoint = directory
        info = self.model_info()
        assert info is not None
        return info

    def unload(self) -> None:
        """Drop the active model."""
        with self._lock:
            self.engine = None
            self.active_checkpoint = None

    def model_info(self) -> Optional[Dict[str, object]]:
        """Describe the active model, or ``None`` if nothing is loaded."""
        engine = self.engine
        checkpoint = self.active_checkpoint
        if engine is None or checkpoint is None:
            return None
        model = engine._base
        config = model.config
        quantization = None
        meta_path = checkpoint / "meta.json"
        if meta_path.exists():
            try:
                quantization = json.loads(
                    meta_path.read_text(encoding="utf-8")
                ).get("quantization")
            except json.JSONDecodeError:
                pass
        return {
            "checkpoint": str(checkpoint),
            "name": self._relname(checkpoint),
            "params": model.num_parameters(),
            "params_human": human_count(model.num_parameters()),
            "n_layers": config.n_layers,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "quantization": quantization,
            "device": str(self.device),
        }

    # ------------------------------------------------------------ inference
    def generate_stream(
        self,
        prompt: str,
        sampling: SamplingConfig,
        max_new_tokens: int,
        seed: Optional[int] = None,
    ) -> Iterator[Dict[str, object]]:
        """Yield ``{"type": "token"|"done"|"error", ...}`` events for a completion."""
        if self.engine is None:
            yield {"type": "error", "message": "No model loaded."}
            return
        if not self._lock.acquire(blocking=False):
            yield {"type": "error", "message": "The model is busy with another request."}
            return
        try:
            for fragment in self.engine.stream_text(
                prompt, max_new_tokens=max_new_tokens, sampling=sampling, seed=seed,
            ):
                yield {"type": "token", "text": fragment}
            yield {"type": "done"}
        except Exception as exc:  # surface the failure to the browser
            yield {"type": "error", "message": str(exc)}
        finally:
            self._lock.release()

    def chat_stream(
        self,
        message: str,
        history: List[Dict[str, str]],
        system: Optional[str],
        sampling: SamplingConfig,
        max_new_tokens: int,
    ) -> Iterator[Dict[str, object]]:
        """Yield reply events for one chat turn."""
        if self.engine is None:
            yield {"type": "error", "message": "No model loaded."}
            return
        if not self._lock.acquire(blocking=False):
            yield {"type": "error", "message": "The model is busy with another request."}
            return
        try:
            session = ChatSession(
                self.engine, system_prompt=system or None,
                sampling=sampling, max_new_tokens=max_new_tokens,
            )
            session.history = [(turn["role"], turn["content"]) for turn in history]
            for fragment in session.stream(message):
                yield {"type": "token", "text": fragment}
            yield {"type": "done"}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
        finally:
            self._lock.release()

    # --------------------------------------------------------- quantization
    def quantize(
        self, input_path: PathLike, output_path: PathLike, bits: int
    ) -> Dict[str, object]:
        """Quantize a checkpoint to int8/int4 and save a new checkpoint."""
        source = Path(input_path).resolve()
        target = Path(output_path)
        if not target.is_absolute():
            target = (self.project_root / target)
        target = target.resolve()
        if not checkpoint_exists(source):
            raise ValueError(f"not a checkpoint directory: {source}")
        if target == source:
            raise ValueError("output directory must differ from the input")
        size_before = (source / "weights.lnw").stat().st_size
        bundle = load_checkpoint(source, build_model=True, map_location="cpu")
        layers = quantize_model(bundle.model, bits=bits)
        meta = dict(bundle.meta)
        meta["quantization"] = bits
        save_checkpoint(
            target, bundle.model, bundle.model_config,
            tokenizer=bundle.tokenizer, meta=meta,
        )
        size_after = (target / "weights.lnw").stat().st_size
        return {
            "layers": layers,
            "size_before": size_before,
            "size_after": size_after,
            "output": self._relname(target),
        }
