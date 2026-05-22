# Quickstart

This walkthrough trains a small model from scratch on the bundled demo data and
a synthetic corpus. Everything runs on a CPU.

Run all commands from the repository root.

## 0. Install

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## 1. One-command demo

The fastest way to see the whole pipeline work end to end:

```bash
python examples/end_to_end.py
```

This generates data, trains a tokenizer, trains a ~1M parameter model and prints
sample completions -- usually in two or three minutes.

## 2. Step by step with the CLI scripts

### Train a tokenizer

```bash
python scripts/train_tokenizer.py --synthetic-samples 8000 --vocab-size 4096
```

Writes `checkpoints/tokenizer.json`.

### Prepare the data

```bash
python scripts/prepare_data.py --synthetic-samples 8000
```

Writes `datasets/demo/train.bin` and `datasets/demo/val.bin`.

### Train the model

```bash
python scripts/train.py --config configs/tiny-7m.json --block-size 128 --steps 1500
```

Checkpoints land in `checkpoints/run/` (and the best validation checkpoint in
`checkpoints/run/best/`). Add `--resume` to continue an interrupted run.

### Generate text

```bash
python scripts/generate.py --checkpoint checkpoints/run --prompt "the model"
```

### Chat

```bash
python scripts/chat.py --checkpoint checkpoints/run
```

### Quantize for low-memory inference

```bash
python scripts/quantize.py --input checkpoints/run --output checkpoints/run-int8 --bits 8
python scripts/generate.py --checkpoint checkpoints/run-int8 --prompt "the model"
```

## 3. Using your own text

Point `--input` at your own UTF-8 text files:

```bash
python scripts/train_tokenizer.py --input mydata/*.txt --vocab-size 8000
python scripts/prepare_data.py --input mydata/*.txt --tokenizer checkpoints/tokenizer.json
python scripts/train.py --config configs/small-15m.json --steps 20000
```

## 4. Tuning for your hardware

| Limit            | What to change                                              |
|------------------|-------------------------------------------------------------|
| Low RAM          | smaller `--batch-size` / `--block-size`; `--grad-checkpoint` |
| Slow CPU         | smaller config (`configs/tiny-7m.json`), fewer `--steps`    |
| Have bf16 CPU    | add `--precision bf16`                                      |
| Have a GPU       | add `--device cuda` (and `--precision bf16` or `fp16`)      |
