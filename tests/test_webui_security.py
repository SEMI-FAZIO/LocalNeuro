"""Tests for the web UI's path sandbox.

The web UI accepts paths from HTTP request bodies (load checkpoint, quantize,
fine-tune). A checkpoint bundle may contain ``trainer_state.pt``, which is a
pickle file -- ``torch.load`` on a crafted pickle is a real RCE vector. So
the web UI is layered:

1. ``resolve_within_root`` refuses any path outside ``project_root``.
2. The load / quantize / fine-tune flows pass ``load_trainer_state=False`` to
   :func:`load_checkpoint`, so even if a pickled file does slip in, it is
   never deserialized.

These tests cover both layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localneuro.checkpoint import load_checkpoint, save_checkpoint
from localneuro.config import ModelConfig
from localneuro.model import LocalNeuroLM
from localneuro.tokenizer.bpe import BPETokenizer
from localneuro.webui.state import WebUIState, resolve_within_root


def test_resolve_within_root_accepts_inside_paths(tmp_path):
    """Paths that resolve inside the project root are returned as absolute."""
    root = tmp_path.resolve()
    inside = root / "checkpoints" / "run"
    inside.mkdir(parents=True)
    assert resolve_within_root(root, inside) == inside.resolve()
    # Relative paths are joined to the root before validation.
    assert resolve_within_root(root, "checkpoints/run") == inside.resolve()


def test_resolve_within_root_rejects_absolute_outside(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside the project root"):
        resolve_within_root(root.resolve(), outside.resolve())


def test_resolve_within_root_rejects_traversal(tmp_path):
    """`../../etc/passwd` style traversal must be refused."""
    root = (tmp_path / "project").resolve()
    root.mkdir()
    with pytest.raises(ValueError, match="outside the project root"):
        resolve_within_root(root, root / ".." / ".." / "etc")


def test_resolve_within_root_allows_root_itself(tmp_path):
    root = tmp_path.resolve()
    assert resolve_within_root(root, root) == root


def _write_tiny_checkpoint(directory: Path) -> ModelConfig:
    """Helper: build a minimal checkpoint bundle under ``directory``.

    Includes a tokenizer so it is loadable via ``WebUIState.load`` (which
    requires one for inference).
    """
    tokenizer = BPETokenizer.train("the quick brown fox " * 40, vocab_size=320)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, d_model=32, n_layers=2,
        n_heads=4, d_ff=64, max_seq_len=32,
    )
    save_checkpoint(
        directory, LocalNeuroLM(config), config,
        tokenizer=tokenizer, meta={"step": 1},
    )
    return config


def test_webui_load_rejects_path_outside_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "stash" / "ckpt"
    outside.mkdir(parents=True)
    _write_tiny_checkpoint(outside)

    state = WebUIState(project, device="cpu")
    with pytest.raises(ValueError, match="outside the project root"):
        state.load(outside)


def test_webui_load_accepts_path_inside_root(tmp_path):
    project = tmp_path / "project"
    inside = project / "checkpoints" / "tiny"
    inside.mkdir(parents=True)
    _write_tiny_checkpoint(inside)

    state = WebUIState(project, device="cpu")
    info = state.load(inside)
    assert info is not None
    assert state.active_checkpoint == inside.resolve()


def test_webui_load_ignores_trainer_state(tmp_path):
    """Even if ``trainer_state.pt`` is present, the web UI flow must not
    deserialize it -- the file is a pickle and would be an RCE vector."""
    project = tmp_path / "project"
    inside = project / "checkpoints" / "tiny"
    inside.mkdir(parents=True)
    _write_tiny_checkpoint(inside)
    # A booby-trapped pickle that would raise on load.
    (inside / "trainer_state.pt").write_bytes(b"this is not a real pickle")

    state = WebUIState(project, device="cpu")
    # Must not raise: the web UI flow opts out of trainer_state loading.
    state.load(inside)


def test_load_checkpoint_skips_trainer_state_when_disabled(tmp_path):
    project = tmp_path / "ckpt"
    project.mkdir()
    _write_tiny_checkpoint(project)
    (project / "trainer_state.pt").write_bytes(b"not a pickle")

    # With load_trainer_state=False the corrupt pickle is left untouched.
    bundle = load_checkpoint(project, build_model=False, load_trainer_state=False)
    assert bundle.trainer_state is None
