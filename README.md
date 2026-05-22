# LocalNeuro

**An ultra-lightweight, fully self-contained tiny language model.**

LocalNeuro is a small language model you can train and run entirely on your own
computer -- no GPU, no internet, no remote API. Every part is built from
scratch in this repository: the transformer architecture, the byte-level BPE
tokenizer, the training loop, the checkpoint format and the CPU-first inference
engine. PyTorch is used **only** as a low-level tensor and autograd library.

* Trains and runs on a plain CPU (no CUDA, no AVX-512 required).
* Fits in **4-8 GB of RAM** -- the smallest preset uses well under 100 MB.
* Runs on Windows, Linux and macOS.
* 5M-40M parameter presets; the design scales up cleanly.
* int8 / int4 quantization, streaming generation, KV cache, gradient checkpointing.
* Readable, typed, modular code meant to be studied as well as used.

> LocalNeuro contains **no pretrained weights, no third-party model, and no
> third-party tokenizer vocabulary.** It is not a wrapper around an existing
> model. You train it yourself, from random initialization, on text you choose.

---

## Why "self-contained"?

| Concern | How LocalNeuro handles it |
|---|---|
| Architecture | Custom decoder-only transformer (`src/localneuro/model.py`). |
| Tokenizer | Custom byte-level BPE, trained on your data (`src/localneuro/tokenizer/`). |
| Weights | Always trained from scratch -- nothing is downloaded. |
| Checkpoint format | Custom pickle-free `.lnw` container (`src/localneuro/checkpoint.py`). |
| Inference | Custom KV-cached generation engine (`src/localneuro/inference/`). |
| Dependencies | `torch` (tensor math) and `numpy` only. ONNX is optional. |

---

## Project layout

```
LocalNeuro/
├── src/localneuro/         the core library
│   ├── config.py           model & training configuration (dataclasses)
│   ├── model.py            the decoder-only transformer
│   ├── modules.py          attention, RoPE, RMSNorm, SwiGLU
│   ├── kv_cache.py         key/value cache for fast decoding
│   ├── sampling.py         temperature / top-k / top-p / repetition penalty
│   ├── checkpoint.py       pickle-free, memory-mappable .lnw weight format
│   ├── tokenizer/          from-scratch byte-level BPE tokenizer
│   ├── data/               preprocessing, synthetic data, token datasets
│   ├── training/           optimizer, LR schedule, trainer, evaluation
│   ├── inference/          streaming generation engine + chat REPL
│   ├── quantization/       int8 / int4 weight quantization
│   └── webui/              browser-based UI (HTTP server + static frontend)
├── scripts/                command-line tools (train, generate, chat, ...)
├── configs/                model size presets (tiny / small / base)
├── datasets/demo/          a small bundled text corpus
├── examples/               one-command end-to-end demo + quickstart
├── tests/                  the test suite
└── docs/                   architecture, roadmap, benchmarks
```

The `tokenizer`, `training` and `inference` components requested as top-level
areas live as subpackages of `src/localneuro/` -- a standard Python `src`
layout that keeps imports clean.

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/<your-username>/LocalNeuro.git
cd LocalNeuro

python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate

pip install -r requirements.txt
pip install -e .            # optional: installs the `localneuro` package
```

---

## Quickstart

The fastest check that everything works -- generates data, trains a tokenizer,
trains a ~1M-parameter model and samples from it, in a couple of minutes:

```bash
python examples/end_to_end.py
```

See [examples/quickstart.md](examples/quickstart.md) for the step-by-step
version using the command-line tools.

---

## Web UI

LocalNeuro ships a browser-based interface -- chat, a generation playground, a
checkpoint manager and a live training dashboard -- built entirely on the
Python standard library (no extra dependencies):

```bash
python scripts/webui.py
```

This starts a local server at `http://127.0.0.1:8080/` and opens it in your
browser. The four tabs:

* **Chat** -- streaming, multi-turn conversation with the loaded model.
* **Playground** -- free-form text completion with live sampling sliders.
* **Models** -- browse and load checkpoints, and quantize them.
* **Training** -- configure and launch a run, watching the loss curve update live.

Options: `--port`, `--host 0.0.0.0` (expose on your LAN), `--device`,
`--checkpoint <dir>` (load a model on startup), `--no-browser`.

---

## Training

Training has three stages: train a tokenizer, tokenize the data, train the model.

```bash
# 1. learn a vocabulary
python scripts/train_tokenizer.py --synthetic-samples 8000 --vocab-size 4096

# 2. tokenize the corpus into binary token streams
python scripts/prepare_data.py --synthetic-samples 8000

# 3. train the model
python scripts/train.py --config configs/tiny-7m.json --block-size 128 --steps 1500
```

Checkpoints are written to `checkpoints/run/`, with the best-validation
checkpoint in `checkpoints/run/best/`. Training is **resumable** -- add
`--resume` to continue an interrupted run from the last checkpoint.

To train on your own text, pass `--input file1.txt file2.txt ...` to
`train_tokenizer.py` and `prepare_data.py`.

Key training options (`python scripts/train.py --help` for all of them):

| Option | Meaning |
|---|---|
| `--config` | model preset (`configs/tiny-7m.json`, `small-15m`, `base-40m`) |
| `--steps` | number of optimizer steps |
| `--batch-size`, `--block-size` | batch size and training sequence length |
| `--grad-accum` | gradient accumulation (large effective batch, low memory) |
| `--precision` | `fp32` (default), `bf16` (modern CPUs/GPUs), `fp16` (CUDA) |
| `--grad-checkpoint` | recompute activations to cut memory ~40% |
| `--device` | `auto`, `cpu`, `cuda`, `mps` |

---

## Running the model

Generate a continuation of a prompt:

```bash
python scripts/generate.py --checkpoint checkpoints/run --prompt "the model"
```

Interactive chat (streaming, multi-turn):

```bash
python scripts/chat.py --checkpoint checkpoints/run
```

Both support `--temperature`, `--top-k`, `--top-p` and `--repetition-penalty`.

Programmatic use:

```python
from localneuro.checkpoint import load_checkpoint
from localneuro.inference.engine import InferenceEngine
from localneuro.sampling import SamplingConfig

bundle = load_checkpoint("checkpoints/run")
engine = InferenceEngine(bundle.model, bundle.tokenizer, device="cpu")
text = engine.generate_text("the model", max_new_tokens=80,
                            sampling=SamplingConfig(temperature=0.7))
print(text)
```

---

## Quantization

Quantization stores each weight as a small integer instead of a 32-bit float,
shrinking the resident model so a larger network fits in a smaller RAM budget.

```bash
# int8 (~2-3x smaller) or int4 (--bits 4, ~3-5x smaller)
python scripts/quantize.py --input checkpoints/run --output checkpoints/run-int8 --bits 8

# generate / chat work unchanged on a quantized checkpoint
python scripts/generate.py --checkpoint checkpoints/run-int8 --prompt "the model"
```

See [docs/architecture.md](docs/architecture.md) for how quantization works.

---

## Model presets

| Preset | Params | Layers | d_model | Context | fp32 size |
|---|---|---|---|---|---|
| `configs/tiny-7m.json`  | ~7.2M  | 6  | 256 | 1024 | ~29 MB |
| `configs/small-15m.json`| ~15.5M | 7  | 384 | 1024 | ~62 MB |
| `configs/base-40m.json` | ~39.6M | 10 | 512 | 2048 | ~158 MB |

`base-40m` uses grouped-query attention (4 KV heads) to keep the KV cache
small. Memory and performance details are in
[docs/benchmarks.md](docs/benchmarks.md).

---

## Documentation

* [docs/architecture.md](docs/architecture.md) -- the architecture, attention,
  the tokenizer, quantization, the memory optimizations, and how to scale up.
* [docs/benchmarks.md](docs/benchmarks.md) -- model sizes, memory usage, CPU
  performance and how to reproduce the numbers.
* [docs/roadmap.md](docs/roadmap.md) -- planned improvements.
* [CONTRIBUTING.md](CONTRIBUTING.md) -- how to contribute and run the tests.

---

## Tests

```bash
pip install pytest
pytest
```

The tokenizer tests need only the standard library; the rest need `torch`.

---

## Publishing your own copy to GitHub

This project has no external service dependencies, so publishing is just:

```bash
git init
git add .
git commit -m "Initial commit: LocalNeuro tiny language model"
git branch -M main
git remote add origin https://github.com/<your-username>/LocalNeuro.git
git push -u origin main
```

`.gitignore` already excludes virtual environments, checkpoints and generated
token files, so only source code and the small demo corpus are committed.

---

## License

Released under the [MIT License](LICENSE). You may use, modify and redistribute
it freely, including commercially.
