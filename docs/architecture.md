# Architecture

This document explains how LocalNeuro is built and *why* each decision was
made. It is meant to be readable start to finish.

## Contents

1. [Why this architecture](#1-why-this-architecture)
2. [The big picture](#2-the-big-picture)
3. [The tokenizer](#3-the-tokenizer)
4. [How attention works](#4-how-attention-works)
5. [Position information: RoPE](#5-position-information-rope)
6. [Normalization and the feed-forward block](#6-normalization-and-the-feed-forward-block)
7. [The KV cache](#7-the-kv-cache)
8. [Sampling](#8-sampling)
9. [Quantization](#9-quantization)
10. [Memory optimizations](#10-memory-optimizations)
11. [Scaling the model up](#11-scaling-the-model-up)

---

## 1. Why this architecture

LocalNeuro is a **decoder-only transformer** -- the same family as the model
behind most modern text generators, but shrunk to a size that trains on a
laptop.

Why a transformer at all, and not an RNN?

* **Parallel training.** A transformer sees a whole sequence at once, so every
  position is trained in parallel. An RNN must step through the sequence one
  token at a time, which is far slower to train.
* **Direct long-range access.** Attention lets any token look directly at any
  earlier token. An RNN has to carry information forward through every
  intermediate step, where it tends to fade.
* **It scales predictably.** The same code that trains a 7M-parameter model
  trains a 1B-parameter model -- only the numbers in the config change.

Why *decoder-only* (rather than encoder-decoder)? Because the task is simple
and uniform: **predict the next token**. A decoder-only stack with a causal
mask does exactly that, and nothing else is needed for a language model.

The specific building blocks -- rotary embeddings, RMSNorm, SwiGLU,
grouped-query attention -- are well-established techniques chosen because each
one buys real quality or efficiency per parameter, which matters most when the
parameter budget is tiny. They are general methods, not a model; every line
here is implemented from scratch.

---

## 2. The big picture

A forward pass, end to end:

```
token ids               [batch, seq]
  │
  ▼  token embedding     lookup table, vocab_size × d_model
hidden states           [batch, seq, d_model]
  │
  ▼  × N transformer blocks
  │     ├─ RMSNorm → causal self-attention → add to residual
  │     └─ RMSNorm → SwiGLU feed-forward   → add to residual
  │
  ▼  final RMSNorm
  ▼  output projection   (tied with the embedding by default)
logits                  [batch, seq, vocab_size]
```

Each block has two sub-layers, each wrapped in a **residual connection**
(`x = x + sublayer(norm(x))`). The residual path is a clean highway from input
to output; it is what lets gradients reach the early layers and makes deep
stacks trainable.

Code map: `model.py` assembles the stack, `modules.py` holds the blocks,
`config.py` holds the shape.

---

## 3. The tokenizer

A neural network only does arithmetic, so text must first become numbers. The
tokenizer is the bridge. LocalNeuro uses **byte-level Byte-Pair Encoding (BPE)**
(`src/localneuro/tokenizer/bpe.py`).

### Training the vocabulary

1. **Start from bytes.** Any text is encoded to UTF-8, so the base alphabet is
   the 256 possible byte values. This is why the tokenizer is multilingual for
   free and can never hit an out-of-vocabulary token: Latin, Cyrillic, CJK and
   emoji are all just byte sequences.
2. **Pre-tokenize.** Text is split into "words" along a regular expression so
   merges never cross obvious boundaries (e.g. word/punctuation).
3. **Merge greedily.** Count every adjacent pair of tokens; merge the most
   frequent pair into a single new token; repeat. Common sequences like `" the"`
   become one token after a few hundred merges.
4. **Stop** when the vocabulary reaches the target size.

The result is a vocabulary tuned to *your* corpus: frequent words are single
tokens, rare words are still spelled out in pieces, and nothing is ever
unrepresentable.

### Encoding and decoding

To encode, each pre-tokenized word's bytes are merged in the same rank order
that training discovered. Decoding simply concatenates the byte sequences of
each token id and interprets the result as UTF-8.

Because a multi-byte character can be split across two tokens, streaming output
uses a `StreamDecoder` that buffers raw bytes and only emits text once it forms
complete characters.

### Serialization

The tokenizer saves to a small JSON file containing just the ordered merge list
and the special tokens -- the byte alphabet and full vocabulary are rebuilt on
load. No external vocabulary file is ever used.

---

## 4. How attention works

Attention is the mechanism that lets a token gather context from the tokens
before it. The implementation is `CausalSelfAttention` in `modules.py`, written
out with an explicit softmax so every step is visible.

For each token the model computes three vectors:

* a **query** (`q`) -- "what am I looking for?"
* a **key** (`k`)   -- "what do I offer?"
* a **value** (`v`) -- "what will I contribute if chosen?"

These come from three linear projections of the token's hidden state.

The attention output for a token is then computed in four steps:

1. **Scores.** Compare this token's query with every key by a dot product:
   `scores = q · kᵀ`. A large score means "these two tokens are relevant to
   each other". The scores are divided by `√head_dim` to keep their magnitude
   stable as the head dimension grows.
2. **Causal mask.** A language model must not look at the future. Any score for
   a key *after* the current token is set to `-∞`, so it contributes nothing.
3. **Softmax.** The masked scores become a probability distribution -- the
   *attention weights*. Each weight says how much of each earlier token to mix
   in. The softmax is computed in float32 for numerical stability.
4. **Weighted sum.** Multiply the values by the weights and sum:
   `output = weights · v`. The token has now gathered a blend of the context it
   found relevant.

This happens in parallel across several **heads**. Each head has its own
queries, keys and values and can specialize -- one head might track syntax,
another might track which subject a pronoun refers to. The heads' outputs are
concatenated and passed through a final projection.

---

## 5. Position information: RoPE

Attention as described is **order-blind**: shuffle the tokens and the scores are
the same. The model needs to know token positions.

LocalNeuro uses **Rotary Position Embedding (RoPE)**. Instead of *adding* a
position vector, RoPE *rotates* each query and key vector by an angle
proportional to its absolute position. The key property: the dot product of two
rotated vectors depends only on the *difference* of their positions. So
attention automatically sees **relative distance**, which is what language
structure actually depends on, and it extrapolates to longer sequences far
better than a learned absolute-position table.

The rotation angles are pure functions of position, so the `cos`/`sin` tables
are precomputed once and never need to be saved to disk
(`RotaryEmbedding` in `modules.py`).

---

## 6. Normalization and the feed-forward block

**RMSNorm.** Before each sub-layer the hidden state is normalized with Root
Mean Square normalization. Compared with the classic LayerNorm it drops the
mean-subtraction and the bias term -- one fewer reduction, half the parameters,
and equally effective for language models. It is applied *inside* the residual
branch ("pre-norm"), which keeps the residual highway clean and training stable.

**SwiGLU feed-forward.** After attention, each token passes through a
position-wise feed-forward network. A plain MLP is `down(act(up(x)))`. SwiGLU
instead *gates* one projection with a SiLU-activated sibling:
`down(silu(gate(x)) · up(x))`. The multiplicative gate lets the network amplify
or suppress features channel by channel and reliably gives more quality per
parameter -- which is exactly the trade a tiny model wants.

**Grouped-query attention (GQA).** Normally each query head has its own key and
value heads. GQA lets several query heads *share* one key/value head. The model
keeps its expressive query heads, but the KV cache -- the dominant memory cost
at inference -- shrinks by the sharing factor. The `base-40m` preset uses 8
query heads and 4 KV heads, halving its cache.

---

## 7. The KV cache

When generating text, the model produces one token at a time, and each new
token attends to *all* previous ones. Recomputing the keys and values for the
whole prefix on every step would be O(n²) work.

The **KV cache** (`kv_cache.py`) fixes this. After a token's key and value are
computed once, they are stored. The next step computes the key/value for only
the single new token and reads the rest from the cache. Generation becomes
linear in sequence length instead of quadratic.

The cache is also the largest inference memory cost, which is why GQA (above)
matters: its size is
`n_layers × 2 × batch × n_kv_heads × max_seq_len × head_dim`.

---

## 8. Sampling

The model outputs a score (logit) for every possible next token. Turning those
scores into an actual token is *sampling* (`sampling.py`):

* **Temperature** scales the logits. Below 1.0 sharpens the distribution
  (safer, more repetitive); above 1.0 flattens it (more varied, more random).
  Temperature 0 is deterministic greedy decoding.
* **Top-k** keeps only the `k` highest-scoring tokens.
* **Top-p (nucleus)** keeps the smallest set of tokens whose probabilities sum
  to `p`, adapting how many options stay open to how confident the model is.
* **Repetition penalty** down-weights tokens that appeared recently, which
  curbs the looping that small models are prone to.

These compose into one pipeline: penalty → temperature → top-k → top-p → draw.

---

## 9. Quantization

A trained weight is a 32-bit float. Most of that precision is unnecessary for
inference. **Quantization** stores each weight as a small integer instead
(`src/localneuro/quantization/`).

LocalNeuro uses **symmetric, per-output-channel** quantization. For each row of
a weight matrix:

```
scale = max(|row|) / qmax          qmax = 127 for int8, 7 for int4
int_weight = round(row / scale)    clamped to ± qmax
row ≈ int_weight × scale           reconstruction
```

A separate scale per row means each output channel gets a range matched to its
own weights, which keeps accuracy high. int8 weights are **4× smaller** than
float32; int4 weights (packed two-per-byte) are **8× smaller**.

This is a *memory* optimization. The matmul itself still runs in floating
point -- the integer weight is dequantized on the fly, one layer at a time, so
the large float matrix is never all resident at once. The win is that the
*stored* model is small enough for a bigger network to fit in the same RAM.

A quantized checkpoint records its bit width in `meta.json`; loading rebuilds
the quantized layers automatically.

---

## 10. Memory optimizations

LocalNeuro's headline goal is **minimal resource use**. The techniques, and
where they live:

| Technique | Effect | Where |
|---|---|---|
| Weight tying | Embedding doubles as the output projection -- saves `vocab × d_model` params. | `model.py` |
| Grouped-query attention | Shrinks the KV cache by the query/KV sharing factor. | `modules.py` |
| KV cache | O(n) instead of O(n²) decoding -- avoids recompute. | `kv_cache.py` |
| Gradient checkpointing | Recompute activations in the backward pass; ~40% less activation memory while training. | `model.py` |
| Mixed precision (bf16) | Halves activation memory; bf16 needs no loss scaler. | `training/optim.py` |
| Gradient accumulation | A large effective batch with a small memory footprint. | `training/trainer.py` |
| Quantization | 4× / 8× smaller stored weights. | `quantization/` |
| Memory-mapped weights | The `.lnw` file is paged in on demand; load never needs more RAM than the model. | `checkpoint.py` |
| Generation fast-path | The output projection is computed only for the last token, not the whole prompt. | `model.py` |

The `.lnw` weight format deserves a note: it is a plain header (JSON) followed
by raw tensor bytes. No pickle -- so a downloaded model cannot run code -- and
it is memory-mappable, so loading a model never needs a second copy in RAM.

---

## 11. Scaling the model up

Nothing in the code is hard-coded to "tiny". To grow the model, change the
config (`config.py` / a JSON preset):

* **Wider** -- increase `d_model` (and `n_heads` so `head_dim` stays ~64).
  Capacity grows with `d_model²` per layer.
* **Deeper** -- increase `n_layers`. The scaled residual initialization already
  accounts for depth.
* **Longer context** -- increase `max_seq_len`. RoPE extrapolates well, so this
  is mostly a memory question (the KV cache grows linearly).
* **Bigger vocabulary** -- train the tokenizer with a larger `--vocab-size`.

For models past ~100M parameters the natural next steps are: a real GPU
training path (the code already supports `--device cuda` and fp16/bf16),
a sharded streaming dataset for corpora larger than RAM, and Flash-Attention as
an optional fast path for the attention kernel. These are tracked in
[roadmap.md](roadmap.md).

The point of LocalNeuro is that the *architecture* does not change as it grows
-- only the numbers do.
