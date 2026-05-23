"""Tests for instruction fine-tuning (SFT) data and the chat format."""

import torch

from localneuro.chat_format import format_chat, format_sft_example
from localneuro.data.sft import (
    InstructionExample,
    SFTDataset,
    build_sft_dataloader,
    build_sft_example,
)
from localneuro.data.synthetic import generate_synthetic_instructions
from localneuro.tokenizer.bpe import ASSISTANT, USER, BPETokenizer

_SAMPLE = (
    "the quick brown fox jumps over the lazy dog. " * 40
    + "hello world question answer count colour number"
)

_CACHED = None


def _tokenizer() -> BPETokenizer:
    global _CACHED
    if _CACHED is None:
        _CACHED = BPETokenizer.train(_SAMPLE, vocab_size=400)
    return _CACHED


def test_format_chat_structure():
    tok = _tokenizer()
    ids = format_chat(tok, [("user", "hi")], add_generation_prompt=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.special_id(ASSISTANT)  # trailing generation prompt
    assert tok.special_id(USER) in ids


def test_format_sft_example_split():
    tok = _tokenizer()
    token_ids, prompt_len = format_sft_example(tok, "what is two", "the answer is two")
    assert 0 < prompt_len < len(token_ids)
    # the prompt ends with the opening <|assistant|> token
    assert token_ids[prompt_len - 1] == tok.special_id(ASSISTANT)
    # the full sequence ends with <eos>
    assert token_ids[-1] == tok.eos_id


def test_build_sft_example_masks_the_prompt():
    tok = _tokenizer()
    built = build_sft_example(
        InstructionExample("count please", "one two three"), tok, max_len=256
    )
    assert built is not None
    input_ids, labels = built
    assert len(input_ids) == len(labels)
    # the loss must cover the response but not the prompt
    assert any(v == -1 for v in labels)
    assert any(v != -1 for v in labels)
    # masked positions form a contiguous prefix; the response is the tail
    first_kept = next(i for i, v in enumerate(labels) if v != -1)
    assert all(v == -1 for v in labels[:first_kept])
    assert all(v != -1 for v in labels[first_kept:])


def test_build_sft_example_skips_empty_response():
    tok = _tokenizer()
    assert build_sft_example(InstructionExample("hi", "   "), tok, max_len=256) is None


def test_build_sft_example_skips_too_long():
    tok = _tokenizer()
    long_example = InstructionExample("x", "word " * 500)
    assert build_sft_example(long_example, tok, max_len=64) is None


def test_sft_dataset_shapes():
    tok = _tokenizer()
    examples = [
        InstructionExample(f"question {i}", f"answer {i} here") for i in range(12)
    ]
    dataset = SFTDataset(examples, tok, block_size=64)
    assert len(dataset) == 12
    x, y = dataset[0]
    assert x.shape == (64,) and y.shape == (64,)
    assert x.dtype == torch.long and y.dtype == torch.long
    assert int((y == -1).sum()) > 0  # prompt + padding are masked


def test_sft_dataloader_batches():
    tok = _tokenizer()
    examples = [InstructionExample(f"q{i}", f"a{i} text here") for i in range(20)]
    loader = build_sft_dataloader(examples, tok, block_size=48, batch_size=4)
    x, y = next(iter(loader))
    assert x.shape == (4, 48) and y.shape == (4, 48)


def test_chat_format_matches_chatsession():
    """The SFT training layout must equal what ChatSession prompts with."""
    from localneuro.config import ModelConfig
    from localneuro.inference.chat import ChatSession
    from localneuro.inference.engine import InferenceEngine
    from localneuro.model import LocalNeuroLM

    tok = _tokenizer()
    config = ModelConfig(
        vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4,
        d_ff=64, max_seq_len=64,
    )
    engine = InferenceEngine(LocalNeuroLM(config), tok, device="cpu")
    session = ChatSession(engine)
    session.history = [("user", "hello"), ("assistant", "hi there")]
    expected = format_chat(tok, session.history, add_generation_prompt=True)
    assert session._build_prompt() == expected


def test_generate_synthetic_instructions_is_deterministic():
    examples = generate_synthetic_instructions(num_samples=50, seed=0)
    assert len(examples) == 50
    for example in examples:
        assert isinstance(example, InstructionExample)
        assert example.instruction.strip()
        assert example.response.strip()
    again = generate_synthetic_instructions(num_samples=50, seed=0)
    assert [e.instruction for e in examples] == [e.instruction for e in again]
