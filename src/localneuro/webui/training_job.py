"""Background training job for the web UI.

A :class:`TrainingJob` runs the full pipeline -- build a corpus, train a
tokenizer, tokenize, train a model -- on a daemon thread, so the HTTP server
stays responsive. Metric events are fan-ed out to any number of subscribers
(the live training dashboard) through per-subscriber queues, and the run can be
stopped cleanly at the next step boundary.
"""

from __future__ import annotations

import queue
import random
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ..checkpoint import checkpoint_exists, load_checkpoint
from ..config import ModelConfig, TrainConfig
from ..data.dataset import encode_corpus, write_token_bin
from ..data.preprocessing import read_corpus, split_train_val
from ..data.sft import build_sft_dataloader
from ..data.synthetic import generate_synthetic_corpus, generate_synthetic_instructions
from ..model import LocalNeuroLM
from ..tokenizer.bpe import BPETokenizer
from ..training.trainer import Trainer
from ..utils import Logger, set_seed

PathLike = Union[str, Path]

# Statuses that mean the job is occupying the training slot.
_ACTIVE = ("preparing", "running")


class TrainingJob:
    """Owns at most one training run and streams its progress."""

    def __init__(self, project_root: PathLike) -> None:
        self.project_root = Path(project_root).resolve()
        self._lock = threading.Lock()
        self.status = "idle"  # idle | preparing | running | finished | stopped | error
        self.error: Optional[str] = None
        self.params: Dict[str, object] = {}
        self.result: Optional[Dict[str, object]] = None
        # All "status" and "log" events emitted so far (never "done").
        self.history: List[Dict[str, object]] = []
        self._subscribers: Set["queue.Queue"] = set()
        self._thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    # ------------------------------------------------------------ state
    def is_running(self) -> bool:
        return self.status in _ACTIVE

    def snapshot(self) -> Dict[str, object]:
        """A JSON-serializable summary of the job for ``/api/train/status``."""
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "params": dict(self.params),
                "result": self.result,
                "history": list(self.history),
            }

    # ------------------------------------------------------- pub / sub
    def subscribe(self) -> "queue.Queue":
        """Return a queue pre-loaded with the full history.

        Done under the lock so no event can slip between the history copy and
        the subscription -- a subscriber sees every event exactly once.
        """
        channel: "queue.Queue" = queue.Queue()
        with self._lock:
            for event in self.history:
                channel.put(event)
            if self.status in _ACTIVE:
                self._subscribers.add(channel)
            else:
                channel.put({
                    "type": "done",
                    "status": self.status,
                    "error": self.error,
                    "result": self.result,
                })
        return channel

    def unsubscribe(self, channel: "queue.Queue") -> None:
        with self._lock:
            self._subscribers.discard(channel)

    def _emit(self, event: Dict[str, object]) -> None:
        """Record an event and push it to every live subscriber."""
        with self._lock:
            self.history.append(event)
            subscribers = list(self._subscribers)
        for channel in subscribers:
            channel.put(event)

    def _emit_status(self, status: str) -> None:
        with self._lock:
            self.status = status
        self._emit({"type": "status", "status": status})

    def _finish(
        self,
        status: str,
        error: Optional[str] = None,
        result: Optional[Dict[str, object]] = None,
    ) -> None:
        """Mark the job done and close out every subscriber."""
        with self._lock:
            self.status = status
            self.error = error
            self.result = result
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        done = {"type": "done", "status": status, "error": error, "result": result}
        for channel in subscribers:
            channel.put(done)

    # --------------------------------------------------------- control
    def start(self, params: Dict[str, object]) -> None:
        """Kick off a training run. Raises if one is already in progress."""
        with self._lock:
            if self.status in _ACTIVE:
                raise RuntimeError("a training run is already in progress")
            self.status = "preparing"
            self.error = None
            self.result = None
            self.params = dict(params)
            self.history = []
        self.stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(dict(params),), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Request the run to stop at the next step boundary."""
        self.stop_event.set()

    # --------------------------------------------------------- the run
    def _log(self, message: str) -> None:
        self._emit({"type": "log", "record": {"message": message}})

    def _on_trainer_record(self, record: Dict[str, object]) -> None:
        """Logger callback -- forwards every trainer record to subscribers."""
        self._emit({"type": "log", "record": record})

    def _build_model_config(
        self, params: Dict[str, object], vocab_size: int
    ) -> ModelConfig:
        preset = str(params.get("preset", "tiny-7m"))
        if preset == "custom":
            config = ModelConfig(
                vocab_size=vocab_size,
                d_model=int(params["d_model"]),
                n_layers=int(params["n_layers"]),
                n_heads=int(params["n_heads"]),
                d_ff=int(params["d_ff"]) if params.get("d_ff") else None,
                max_seq_len=int(params["max_seq_len"]),
            )
        else:
            config = ModelConfig.load(
                self.project_root / "configs" / f"{preset}.json"
            )
            config.vocab_size = vocab_size
        return config

    def _run(self, params: Dict[str, object]) -> None:
        try:
            seed = int(params.get("seed", 1337))
            set_seed(seed)
            steps = int(params.get("steps", 600))
            run_name = str(params.get("run_name") or "gui-run").strip() or "gui-run"
            out_dir = self.project_root / "checkpoints" / run_name
            self._emit_status("preparing")
            if str(params.get("mode", "pretrain")) == "finetune":
                self._run_finetune(params, seed, steps, run_name, out_dir)
            else:
                self._run_pretrain(params, seed, steps, run_name, out_dir)
        except Exception as exc:  # report any failure to the dashboard
            traceback.print_exc()
            self._finish("error", error=f"{type(exc).__name__}: {exc}")

    def _finish_run(self, run_name: str, out_dir: Path, step: int) -> None:
        final = "stopped" if self.stop_event.is_set() else "finished"
        self._finish(
            final,
            result={"checkpoint": run_name, "path": str(out_dir), "step": step},
        )

    def _run_pretrain(
        self, params: Dict[str, object], seed: int, steps: int,
        run_name: str, out_dir: Path,
    ) -> None:
        """Pretrain a new base model from scratch on synthetic + demo text."""
        # 1. Corpus.
        samples = int(params.get("synthetic_samples", 8000))
        self._log(f"generating {samples} synthetic samples")
        corpus = generate_synthetic_corpus(num_samples=samples, seed=seed)
        if params.get("use_demo_corpus", True):
            demo = self.project_root / "datasets" / "demo" / "tiny_corpus.txt"
            if demo.exists():
                corpus = corpus + "\n\n" + read_corpus(demo)
        self._log(f"corpus: {len(corpus):,} characters")

        # 2. Tokenizer.
        vocab_size = int(params.get("vocab_size", 2048))
        self._log(f"training BPE tokenizer (target vocab {vocab_size})")
        tokenizer = BPETokenizer.train(corpus, vocab_size=vocab_size)
        self._log(f"tokenizer: {tokenizer.vocab_size} tokens")

        # 3. Tokenize.
        train_text, val_text = split_train_val(corpus, val_fraction=0.1)
        train_ids = encode_corpus(train_text, tokenizer, doc_separator="\n")
        val_ids = encode_corpus(val_text, tokenizer, doc_separator="\n")
        write_token_bin(out_dir / "train.bin", train_ids, tokenizer.vocab_size)
        write_token_bin(out_dir / "val.bin", val_ids, tokenizer.vocab_size)
        self._log(
            f"tokenized: {len(train_ids):,} train / {len(val_ids):,} val tokens"
        )

        # 4. Model.
        model_config = self._build_model_config(params, tokenizer.vocab_size)
        model = LocalNeuroLM(model_config)
        self._log(
            f"model: {model.num_parameters():,} parameters, "
            f"{model_config.n_layers} layers"
        )

        # 5. Train.
        log_interval = max(1, steps // 60)
        train_config = TrainConfig(
            train_bin=str(out_dir / "train.bin"),
            val_bin=str(out_dir / "val.bin"),
            out_dir=str(out_dir),
            batch_size=int(params.get("batch_size", 16)),
            block_size=int(params.get("block_size", 128)),
            max_steps=steps,
            learning_rate=float(params.get("learning_rate", 6e-4)),
            warmup_steps=max(1, steps // 20),
            log_interval=log_interval,
            eval_interval=max(log_interval, steps // 4),
            eval_iters=20,
            checkpoint_interval=max(steps, 1),
            seed=seed,
        )
        logger = Logger(
            out_dir / "train_log.jsonl", quiet=True,
            on_record=self._on_trainer_record,
        )
        self._emit_status("running")
        trainer = Trainer(
            model, model_config, train_config,
            tokenizer=tokenizer, logger=logger, stop_event=self.stop_event,
        )
        trainer.train()
        self._finish_run(run_name, out_dir, trainer.step)

    def _run_finetune(
        self, params: Dict[str, object], seed: int, steps: int,
        run_name: str, out_dir: Path,
    ) -> None:
        """Instruction-fine-tune an existing base checkpoint (SFT)."""
        base = str(params.get("base_checkpoint") or "").strip()
        if not base or not checkpoint_exists(Path(base)):
            raise ValueError(f"base checkpoint not found: {base!r}")
        self._log(f"loading base checkpoint: {base}")
        bundle = load_checkpoint(base, build_model=True, map_location="cpu")
        if bundle.tokenizer is None:
            raise ValueError("base checkpoint has no tokenizer")
        if bundle.meta.get("quantization"):
            raise ValueError("cannot fine-tune a quantized checkpoint")
        tokenizer = bundle.tokenizer
        model_config = bundle.model_config

        # Instruction data.
        count = int(params.get("synthetic_instructions", 4000))
        self._log(f"generating {count} synthetic instructions")
        examples = generate_synthetic_instructions(count, seed=seed)
        random.Random(seed).shuffle(examples)
        block_size = min(
            int(params.get("block_size", 256)), model_config.max_seq_len
        )
        batch_size = int(params.get("batch_size", 16))
        val_count = max(8, int(len(examples) * 0.05))
        train_loader = build_sft_dataloader(
            examples[val_count:], tokenizer, block_size, batch_size, shuffle=True
        )
        val_loader = build_sft_dataloader(
            examples[:val_count], tokenizer, block_size, batch_size, shuffle=False
        )
        self._log(
            f"SFT dataset: {len(train_loader.dataset)} train / "
            f"{len(val_loader.dataset)} val examples"
        )

        # Fine-tune.
        log_interval = max(1, steps // 60)
        sentinel = str(out_dir / "_unused")  # no token-bin val fallback
        lr = float(params.get("learning_rate", 5e-5))
        train_config = TrainConfig(
            train_bin=sentinel,
            val_bin=sentinel,
            out_dir=str(out_dir),
            batch_size=batch_size,
            block_size=block_size,
            max_steps=steps,
            learning_rate=lr,
            min_learning_rate=lr * 0.1,
            warmup_steps=max(1, steps // 20),
            weight_decay=0.0,
            log_interval=log_interval,
            eval_interval=max(log_interval, steps // 4),
            eval_iters=20,
            checkpoint_interval=max(steps, 1),
            seed=seed,
        )
        logger = Logger(
            out_dir / "train_log.jsonl", quiet=True,
            on_record=self._on_trainer_record,
        )
        self._emit_status("running")
        trainer = Trainer(
            bundle.model, model_config, train_config,
            tokenizer=tokenizer, logger=logger, stop_event=self.stop_event,
            train_loader=train_loader, val_loader=val_loader,
            extra_meta={"stage": "sft", "base_checkpoint": base},
        )
        trainer.train()
        self._finish_run(run_name, out_dir, trainer.step)
