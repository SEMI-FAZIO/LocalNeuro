"""End-to-end smoke test: a Trainer run actually drives the loss down.

The test trains a deliberately tiny model for a few dozen steps on the
synthetic instruction data through the SFTDataset path (so it also exercises
the injected-loader plumbing). It is not a quality benchmark -- it just
proves the whole stack (model + optimizer + dataloader + checkpoint save)
works together and learns *something* from a known-easy corpus.
"""

from __future__ import annotations

from localneuro.checkpoint import checkpoint_exists, load_checkpoint
from localneuro.config import ModelConfig, TrainConfig
from localneuro.data.sft import build_sft_dataloader
from localneuro.data.synthetic import generate_synthetic_instructions
from localneuro.model import LocalNeuroLM
from localneuro.tokenizer.bpe import BPETokenizer
from localneuro.training.trainer import Trainer


def test_trainer_drives_loss_down(tmp_path):
    # 1. Tiny tokenizer trained on the instruction prompts/responses so the
    #    test stays self-contained.
    examples = generate_synthetic_instructions(num_samples=120, seed=0)
    corpus = "\n".join(f"{ex.instruction}\n{ex.response}" for ex in examples)
    tokenizer = BPETokenizer.train(corpus, vocab_size=400)

    # 2. Tiny model -- a couple of small layers is enough to learn the
    #    synthetic patterns within a few dozen steps.
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=64,
    )
    model = LocalNeuroLM(config)

    # 3. SFT-style dataloader (exercises the injected-loader API).
    loader = build_sft_dataloader(examples, tokenizer, block_size=48, batch_size=8)

    out_dir = tmp_path / "run"
    train_cfg = TrainConfig(
        train_bin="UNUSED", val_bin="UNUSED",  # sentinel; loader is injected
        out_dir=str(out_dir),
        batch_size=8, block_size=48,
        max_steps=40,
        learning_rate=3e-3, min_learning_rate=3e-4, warmup_steps=5,
        weight_decay=0.0,
        log_interval=10, eval_interval=10**9,  # disable eval (no val loader)
        eval_iters=1,
        checkpoint_interval=40,
        seed=0,
    )

    trainer = Trainer(
        model, config, train_cfg,
        tokenizer=tokenizer,
        train_loader=loader,
        extra_meta={"stage": "smoke"},
    )
    starting_loss, _ = trainer._optimization_step()
    final_loss = trainer.train()  # returns best_val; with no val loader, +inf
    del final_loss  # unused -- the real signal is the loss reduction below

    # 4. After training, one more forward-only loss measurement should be
    #    visibly lower than the very first optimization step's loss.
    model.eval()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
    import torch
    with torch.no_grad():
        _, after_loss = model(inputs, targets=targets)
    assert float(after_loss) < starting_loss * 0.8, (
        f"trainer did not learn: start={starting_loss:.3f}, "
        f"after={float(after_loss):.3f}"
    )

    # 5. The final checkpoint exists and is loadable -- and the web UI flow
    #    (no trainer_state.pt deserialization) must work too.
    assert checkpoint_exists(out_dir)
    loaded = load_checkpoint(out_dir, build_model=True, load_trainer_state=False)
    assert loaded.trainer_state is None
    assert loaded.meta.get("stage") == "smoke"
