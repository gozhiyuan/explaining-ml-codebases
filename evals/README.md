# Public evaluation protocol

`cases.json` records pinned source and realistic requests. `examples/` contains
actual independent agent outputs from 2026-10-09 and a per-case `evaluation.json`.
Two agents used the core and companion instructions; the parent reviewed
selected source claims and implemented packaging/animation/site checks.

For a fresh behavioral evaluation, give an agent a case request, the skills
and a clean pinned checkout. Keep outputs outside source. Do not provide the
expected explanation. Evaluate source accuracy, audience fit, one consistent
example, evidence/uncertainty, purposeful visuals and artifact portability.
The expected invariants are for the reviewer, not the writer prompt.

For reproducible revalidation of the delivered examples:

```bash
git clone https://github.com/karpathy/nanoGPT /path/to/nanogpt
git -C /path/to/nanogpt checkout 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
git clone https://github.com/vllm-project/vllm /path/to/vllm
git -C /path/to/vllm checkout d29f7f5c9294be8e489dac34d45a939b95a06336
uv venv /path/to/eval-tools --python 3.12
uv pip install --python /path/to/eval-tools/bin/python 'torch==2.14.1' 'numpy==2.5.3'
python scripts/replay_evaluations.py --nanogpt /path/to/nanogpt --vllm /path/to/vllm \
  --python /path/to/eval-tools/bin/python --out /path/to/new-output
python -m unittest discover -s tests -v
```

The replay runs fixtures/checkers/renderers, without generating new prose or
testing an agent's decisions. Keep dependencies in a separate environment.
Do not download model weights. nanoGPT is deprecated upstream; the explicit
requested fixture still provides a useful small training example.

Actual execution boundaries:

- nanoGPT: original model and AST-extracted loader/LR functions, 13 grouped
  assertions; full training script, distributed execution and GPU tests not run.
- vLLM: selected original AST methods plus four scheduler arithmetic statements,
  eight grouped contracts with pool/coordinator/spec stand-ins. Upstream test
  collection failed at missing `tblib`; no upstream tests executed. Real hashing,
  free-queue operations, complete scheduler and model KV correctness remain gaps.
- Three original Manim templates rendered to H.264 MP4; computed attention
  probabilities tested independently. Frames spanning each sequence inspected.
  Presentation timing is illustrative. Browser playback was not verified.
- Article and media exports structurally checked after relocation. Browser tool
  rejected local file URLs, so controls, Mermaid runtime, responsive layout and
  HTML/video playback remain unverified. No protocol/tool workaround was used.
- Jekyll staging/local build checks are recorded in `results.json`. They do not
  imply publication. The user's existing site is used only in a scratch copy.

Symbol/excerpt checks and fixtures add evidence; they cannot establish that all
narrative claims or all production branches are correct. Record failures and
uncertainty rather than translating a limited check into end-to-end coverage.
