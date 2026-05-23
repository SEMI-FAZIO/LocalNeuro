"""The training loop.

The :class:`Trainer` ties together everything needed to train a LocalNeuro
model: data loading, mixed precision, gradient accumulation, gradient clipping,
the learning-rate schedule, periodic evaluation, logging and resumable
checkpointing.
"""

from __future__ import annotations

import math
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..checkpoint import checkpoint_exists, save_checkpoint
from ..config import ModelConfig, TrainConfig
from ..data.dataset import build_dataloader, infinite_loader
from ..model import LocalNeuroLM
from ..tokenizer.bpe import BPETokenizer
from ..utils import Logger, configure_threads, format_duration, resolve_device
from .evaluate import evaluate, perplexity
from .optim import (
    autocast_context,
    build_optimizer,
    cosine_lr,
    make_grad_scaler,
    resolve_precision,
    set_lr,
)


class Trainer:
    """Drives a single training run for a :class:`LocalNeuroLM`."""

    def __init__(
        self,
        model: LocalNeuroLM,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: Optional[BPETokenizer] = None,
        logger: Optional[Logger] = None,
        stop_event: Optional[threading.Event] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        extra_meta: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.cfg = train_config
        self.tokenizer = tokenizer
        # When set, the training loop exits cleanly at the next step boundary.
        self._stop_event = stop_event
        # Extra key/values merged into every saved checkpoint's meta.json
        # (used by fine-tuning to tag the checkpoint stage).
        self._extra_meta = dict(extra_meta or {})

        if train_config.block_size > model_config.max_seq_len:
            raise ValueError(
                f"block_size ({train_config.block_size}) exceeds the model's "
                f"max_seq_len ({model_config.max_seq_len})"
            )

        self.out_dir = Path(train_config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(self.out_dir / "train_log.jsonl")

        # --- device & precision ------------------------------------------
        configure_threads(train_config.cpu_threads)
        self.device = resolve_device(train_config.device)
        self.amp_dtype, use_scaler = resolve_precision(
            train_config.precision, self.device
        )
        self.scaler = make_grad_scaler(enabled=use_scaler)
        self.model.to(self.device)
        if train_config.grad_checkpoint:
            self.model.enable_gradient_checkpointing(True)
        if train_config.compile_model:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        # --- optimizer ----------------------------------------------------
        self.optimizer = build_optimizer(
            self.model,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(train_config.beta1, train_config.beta2),
        )

        # --- data ---------------------------------------------------------
        # Loaders may be injected (e.g. an SFTDataset loader for fine-tuning);
        # otherwise they are built from the token .bin files in train_config.
        if train_loader is not None:
            self.train_loader = train_loader
        else:
            self.train_loader = build_dataloader(
                train_config.train_bin,
                train_config.block_size,
                train_config.batch_size,
                shuffle=True,
                num_workers=train_config.num_workers,
            )
        if len(self.train_loader) == 0:
            raise ValueError(
                "the training data is too small to form a single batch; reduce "
                "batch_size / block_size or provide more data."
            )
        self._train_iter = infinite_loader(self.train_loader)

        if val_loader is not None:
            self.val_loader = val_loader
        elif Path(str(train_config.val_bin) + ".meta.json").exists():
            self.val_loader = build_dataloader(
                train_config.val_bin,
                train_config.block_size,
                train_config.batch_size,
                shuffle=False,
                num_workers=train_config.num_workers,
            )
        else:
            self.val_loader = None

        # --- run state ----------------------------------------------------
        self.step = 0
        self.best_val = float("inf")

    # ------------------------------------------------------------- resume
    def load_trainer_state(self, state: dict) -> None:
        """Restore optimizer / step / RNG state to resume a run."""
        self.step = int(state.get("step", 0))
        self.best_val = float(state.get("best_val", float("inf")))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        self.logger.info(f"resumed from step {self.step}")

    def _trainer_state(self) -> dict:
        return {
            "step": self.step,
            "best_val": self.best_val,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "train_config": self.cfg.to_dict(),
        }

    # -------------------------------------------------------------- saving
    def _save(self, directory: Path, val_loss: float, with_trainer_state: bool) -> None:
        # Keep meta.json strictly valid JSON: inf/nan -> null.
        meta = {
            "model": "LocalNeuroLM",
            "step": self.step,
            "val_loss": val_loss if math.isfinite(val_loss) else None,
            "best_val": self.best_val if math.isfinite(self.best_val) else None,
            "created": datetime.now().isoformat(timespec="seconds"),
            **self._extra_meta,
        }
        # Save the underlying model, not a torch.compile wrapper, so the
        # state-dict keys stay un-prefixed and load into a plain LocalNeuroLM.
        raw_model = getattr(self.model, "_orig_mod", self.model)
        save_checkpoint(
            directory,
            raw_model,
            self.model_config,
            tokenizer=self.tokenizer,
            trainer_state=self._trainer_state() if with_trainer_state else None,
            meta=meta,
        )

    # ------------------------------------------------------------- one step
    def _optimization_step(self) -> tuple[float, float]:
        """Run one optimizer step (with gradient accumulation). Returns
        ``(loss, grad_norm)``."""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(cfg.grad_accum_steps):
            inputs, targets = next(self._train_iter)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with autocast_context(self.device, self.amp_dtype):
                _, loss = self.model(inputs, targets=targets)
                loss = loss / cfg.grad_accum_steps
            self.scaler.scale(loss).backward()
            loss_value += float(loss.item())

        grad_norm = 0.0
        if cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_value, grad_norm

    # ----------------------------------------------------------------- loop
    def train(self) -> float:
        """Run the full training loop. Returns the best validation loss seen."""
        cfg = self.cfg
        self.logger.info(self.model_config_summary())
        self.logger.info(
            f"device={self.device}  precision={cfg.precision}  "
            f"steps={cfg.max_steps}  batch={cfg.batch_size}x{cfg.grad_accum_steps}  "
            f"block={cfg.block_size}"
        )
        start_time = time.time()
        self.model.train()

        while self.step < cfg.max_steps:
            if self._stop_event is not None and self._stop_event.is_set():
                self.logger.info("training stopped by request")
                break
            lr = cosine_lr(
                self.step,
                base_lr=cfg.learning_rate,
                min_lr=cfg.min_learning_rate,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
            )
            set_lr(self.optimizer, lr)

            loss_value, grad_norm = self._optimization_step()
            self.step += 1

            if self.step % cfg.log_interval == 0:
                self.logger.log(
                    {
                        "step": self.step,
                        "loss": loss_value,
                        "ppl": perplexity(loss_value),
                        "lr": lr,
                        "grad_norm": grad_norm,
                    }
                )

            if self.val_loader is not None and self.step % cfg.eval_interval == 0:
                val_loss = evaluate(
                    self.model,
                    self.val_loader,
                    max_batches=cfg.eval_iters,
                    device=self.device,
                    amp_dtype=self.amp_dtype,
                )
                self.logger.log(
                    {
                        "step": self.step,
                        "val_loss": val_loss,
                        "val_ppl": perplexity(val_loss),
                    }
                )
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save(self.out_dir / "best", val_loss, with_trainer_state=False)
                    self.logger.info(
                        f"  new best val_loss={val_loss:.4f} -> {self.out_dir / 'best'}"
                    )

            if self.step % cfg.checkpoint_interval == 0:
                self._save(self.out_dir, self.best_val, with_trainer_state=True)

        # Final checkpoint.
        self._save(self.out_dir, self.best_val, with_trainer_state=True)
        elapsed = format_duration(time.time() - start_time)
        self.logger.info(
            f"training finished: {self.step} steps in {elapsed}  "
            f"best_val={self.best_val:.4f}  checkpoint={self.out_dir}"
        )
        return self.best_val

    def model_config_summary(self) -> str:
        base = self.model
        # torch.compile wraps the model; reach through for the summary.
        if hasattr(base, "_orig_mod"):
            base = base._orig_mod  # type: ignore[assignment]
        return base.summary() if isinstance(base, LocalNeuroLM) else "LocalNeuroLM"
