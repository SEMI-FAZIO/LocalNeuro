# Roadmap

LocalNeuro is intentionally small and complete: every feature it advertises is
fully implemented. This page lists directions for growth, not missing pieces.

## Model quality

- **Sliding-window context.** When the KV cache fills, drop the oldest entries
  and keep generating instead of stopping. Lets generation continue past
  `max_seq_len`.
- **Multi-turn instruction data.** Instruction fine-tuning (single-turn) is
  implemented; a natural next step is synthetic *multi-turn* conversations and
  loss masking across several assistant turns.
- **Dropout-free regularization options.** Weight averaging (EMA of weights)
  for steadier small-model training.

## Performance

- **Optional fused attention.** A `torch.nn.functional.scaled_dot_product_attention`
  fast path, selected by a config flag, keeping the current explicit
  implementation as the readable default.
- **True integer matmul.** Today quantization dequantizes before the matmul
  (a memory win, not a speed win). An integer GEMM path would make quantized
  inference faster, not just smaller.
- **Embedding quantization.** Quantize the token embedding / tied output
  projection as well, pushing the quantized model size toward the theoretical
  4x / 8x reduction.

## Training at larger scale

- **Sharded streaming dataset.** Train on corpora larger than RAM by streaming
  shards from disk instead of memory-mapping a single `.bin` file.
- **Distributed training.** Multi-GPU data-parallel training for models beyond
  ~100M parameters.
- **Learning-rate range test** and other diagnostics to make scaling up
  predictable.

## Tooling and ecosystem

- **KV-cache-aware ONNX export.** The current ONNX export covers the plain
  forward pass; exporting the incremental decode step would make ONNX runtimes
  as fast as the native engine.
- **Checkpoint inspection CLI.** A small tool to print a checkpoint's config,
  parameter count and quantization status.
- **TensorBoard logging.** An optional metrics backend alongside the JSONL log.

## Documentation

- A narrated notebook that builds a forward pass tensor by tensor.
- A short write-up on choosing model size and vocabulary for a given corpus.

Contributions toward any of these are welcome -- see
[CONTRIBUTING.md](../CONTRIBUTING.md).
