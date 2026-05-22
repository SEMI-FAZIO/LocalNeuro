# Contributing to LocalNeuro

Contributions are welcome -- bug fixes, documentation, tests, new features and
ideas all help. This guide explains how to get set up and what the project
expects from a change.

## Development setup

```bash
git clone https://github.com/<your-username>/LocalNeuro.git
cd LocalNeuro

python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"     # editable install + pytest
```

## Running the tests

```bash
pytest
```

The tokenizer tests use only the standard library; the model, sampling,
quantization and checkpoint tests need `torch`. Please make sure the full suite
passes before opening a pull request, and add tests for any new behaviour.

A quick end-to-end smoke test is also useful:

```bash
python examples/end_to_end.py
```

## Code style

LocalNeuro values code that is easy to read and to learn from. Please keep to
the existing style:

- **Type hints** on public function signatures.
- **Docstrings** that explain *why*, not just *what* -- the existing modules
  are the reference.
- **Small, focused modules.** No catch-all files; no spaghetti.
- **Standard library and `torch`/`numpy` only** for core code. New runtime
  dependencies need a clear justification and should usually be optional.
- **No placeholders.** A merged change must be fully implemented -- no stubs,
  no `TODO` standing in for missing logic.
- Prefer clarity over cleverness. This is also a teaching codebase.

There is no enforced formatter; match the surrounding code (4-space indentation,
roughly 90-column lines).

## Project principles

Keep these in mind when proposing changes:

1. **Self-contained.** No pretrained weights, no third-party models, no
   third-party tokenizer vocabularies. PyTorch is a tensor library only.
2. **CPU-first.** Everything must work without a GPU and within a modest RAM
   budget. GPU support is a bonus, never a requirement.
3. **Small and scalable.** The architecture should stay simple enough to read
   in an afternoon, while scaling up by changing config numbers alone.

A change that conflicts with these principles is unlikely to be merged, however
useful in isolation.

## Submitting a change

1. Fork the repository and create a branch:
   `git checkout -b fix-some-bug` or `feature-some-thing`.
2. Make the change; add or update tests and documentation.
3. Run `pytest` and confirm it passes.
4. Commit with a clear message describing the *why* of the change.
5. Open a pull request that explains the motivation and the approach.

## Reporting bugs

Open a GitHub issue with: what you expected, what happened, the exact command
or code, and your OS / Python / torch versions. A minimal reproduction is the
single most helpful thing you can include.

Thank you for helping make LocalNeuro better.
