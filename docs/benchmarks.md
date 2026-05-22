# Benchmarks

This page covers model size, memory footprint and CPU performance. The size
and memory figures are computed directly from the architecture and are exact.
Throughput depends heavily on your CPU, so this page gives a measurement recipe
rather than numbers that would not transfer to your machine.

## Parameter counts

Computed by `ModelConfig.num_params()` (see `src/localneuro/config.py`).

| Preset | Total params | Core (no embedding) | Embedding |
|---|---|---|---|
| `tiny-7m`  | 7,212,288  | 5,115,136  | 2,097,152 |
| `small-15m`| 15,537,792 | 12,392,064 | 3,145,728 |
| `base-40m` | 39,594,496 | 35,400,192 | 4,194,304 |

## Weight memory

The stored / resident weight size, by precision. `int8` and `int4` quantize the
linear layers only; the token embedding and the norms stay float32.

| Preset | float32 | int8 | int4 |
|---|---|---|---|
| `tiny-7m`  | ~28.8 MB | ~13.5 MB (2.1x) | ~11.0 MB (2.6x) |
| `small-15m`| ~62.2 MB | ~25.0 MB (2.5x) | ~18.8 MB (3.3x) |
| `base-40m` | ~158.4 MB| ~52.2 MB (3.0x) | ~34.5 MB (4.6x) |

The per-channel quantization scales add a negligible overhead (tens to a few
hundred KB). Quantizing the embedding too (see [roadmap.md](roadmap.md)) would
push the ratios closer to the theoretical 4x / 8x.

## KV cache memory

The KV cache size at inference is

```
n_layers × 2 × batch × n_kv_heads × max_seq_len × head_dim × bytes_per_element
```

At float32, batch size 1, and the full context window:

| Preset | KV heads | Context | KV cache |
|---|---|---|---|
| `tiny-7m`  | 8 | 1024 | ~12 MiB |
| `small-15m`| 6 | 1024 | ~21 MiB |
| `base-40m` | 4 | 2048 | ~40 MiB |

`base-40m` uses grouped-query attention: with full multi-head attention (8 KV
heads) its cache would be ~80 MiB instead of ~40 MiB.

## Inference memory budget

Approximate resident memory for generation on CPU (float32):

```
total ≈ weights + KV cache + a small working set (a few MB)
```

So `tiny-7m` runs comfortably in well under 100 MB, and `base-40m` in roughly
200 MB -- both far inside a 4-8 GB budget. Quantizing the weights lowers the
first term as shown above.

## Training memory budget

Training needs more than inference -- model, gradients, optimizer state and
activations all coexist:

```
parameters + gradients + Adam state ≈ params × 16 bytes
activations                         ≈ depends on batch × block × model size
logits buffer                       = batch × block × vocab × 4 bytes
```

The logits buffer is often the largest single tensor. Example: `tiny-7m` with
`--batch-size 16 --block-size 256` has a logits buffer of
`16 × 256 × 8192 × 4 B ≈ 134 MiB`.

Knobs to lower training memory, in order of impact:

1. `--batch-size` and `--block-size` -- both scale the activations and logits.
2. `--grad-accum` -- keep a large *effective* batch with a small real one.
3. `--grad-checkpoint` -- recompute activations in the backward pass (~40% less
   activation memory, ~30% more compute).
4. `--precision bf16` -- halves activation memory on a CPU that supports it.

## Measuring CPU performance

Throughput varies several-fold between CPUs, so measure on yours. A simple
generation benchmark:

```bash
python - <<'PY'
import time, sys
sys.path.insert(0, "src")
from localneuro.checkpoint import load_checkpoint
from localneuro.inference.engine import InferenceEngine
from localneuro.sampling import SamplingConfig

bundle = load_checkpoint("checkpoints/run")
engine = InferenceEngine(bundle.model, bundle.tokenizer, device="cpu")
n = 200
start = time.time()
engine.generate(engine.tokenizer.encode_ordinary("the model"),
                max_new_tokens=n, sampling=SamplingConfig(temperature=0.8))
dt = time.time() - start
print(f"{n/dt:.1f} tokens/sec  ({dt:.2f}s for {n} tokens)")
PY
```

Record your results here:

| Preset | Precision | CPU | tokens/sec (generation) |
|---|---|---|---|
| `tiny-7m`  | fp32 | _your CPU_ | _measure_ |
| `small-15m`| fp32 | _your CPU_ | _measure_ |
| `base-40m` | fp32 | _your CPU_ | _measure_ |

As a rough expectation on a modern multi-core desktop CPU, `tiny-7m` generates
on the order of tens of tokens per second and `base-40m` noticeably slower;
older laptops will be a few times slower again. Set `--cpu-threads` to tune the
thread count, and quantization lowers memory pressure (it does not speed up the
matmul -- see [architecture.md](architecture.md#9-quantization)).

## Reproducing the size figures

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
from localneuro.config import ModelConfig
for name in ("tiny-7m", "small-15m", "base-40m"):
    cfg = ModelConfig.load(f"configs/{name}.json")
    print(f"{name:11s} {cfg.num_params():>12,} params")
PY
```
