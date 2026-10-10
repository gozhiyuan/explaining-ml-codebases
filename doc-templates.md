# Doc templates

Two templates: `index.md` (always) and a sub-doc (one per subsystem in `deep`
mode). `check_explain_doc.py` checks the required headings and the per-section
labels listed here, so keep the exact words.

## Contents

- `index.md` template
- Sub-doc template
- Step section shape
- Tests table and gaps table
- Diagram conventions
- `simple` mode variant

---

## `index.md` template

Required headings: Mental model, At a glance, Big picture, Running example,
Doc map, Major components, End-to-end data flow, Test coverage, Gaps. Also
required in deep mode: Key sequences, Glossary. Required when the repo has
tensors: Shapes cheat sheet. Required when existing docs were found: Coverage
of existing docs. Write each heading exactly as in the template, with nothing
after it.

````markdown
# <Repo or system> explained

<2-4 sentences: what the system does, who this doc is for, what it covers.>

Commit: `<sha>` (<date>). Mode: `deep`. Audience: <audience>.
Scope: <golden path in one line>. Not covered: <list, or "see Out of scope">.

<Contents block. See "Contents" below.>

## Mental model

<What a reader must hold in their head after 60 seconds. 4-8 numbered
sentences, one per stage or process, in the order data moves. Each names the
thing, what it turns into, and the one fact that surprises people. No code
references here. Example for a training repo:

1. A one-time script turns text into a flat file of token ids.
2. Each training step cuts random windows from that file. There are no epochs.
3. The model predicts the next token at every position at once.
4. The loss is the mean cross-entropy over those positions.
5. Every N steps the loop measures val loss and saves only if it improved.>

## At a glance

| Knob | Value | Set in | What it changes |
|------|-------|--------|-----------------|
| Python / framework | ... | `pyproject.toml` | ... |
| Demo model / config | ... | `configs/x.yaml` | ... |

## Big picture

<Mermaid flowchart of the stages or processes. Then 3-6 sentences that walk
the diagram in order.>

## Running example

<The one concrete input used in every doc, with its values at each stage in a
short table: stage, what the example looks like after it.>

## Doc map

| Doc | Covers | Entry point | Read it when |
|-----|--------|-------------|--------------|
| [train.md](train.md) | data loading, model, loss, loop | `train.py:main` | you change the model or the loss |

## Major components

| Component | File | Role | Called by | Calls |
|-----------|------|------|-----------|-------|

<Mermaid `classDiagram` of the main classes: composition (`*--`), inheritance
(`<|--`), and calls (`..>`), with the key fields and methods. Use a
`flowchart` instead only when the code is mostly functions, not classes.>

## Key sequences

<One Mermaid `sequenceDiagram` per hot path: one training step (forward,
loss, backward, clip, optimizer step, and the collectives under DDP) and one
prediction request (entry, queue, batch, model, response). Skip a sequence the
repo does not have. Each sub-doc may hold a more detailed version.>

## End-to-end data flow

| Step | Input | Output | Code | Doc |
|------|-------|--------|------|-----|
| data.A1 | raw parquet `[rows, 12 cols]` | labels parquet | `build_labels` in `prep.py` | [data.md](data.md) |

## Test coverage

| Step | Test | Command | Status |
|------|------|---------|--------|

Summary: <n> steps, <p> with passing tests, <f> failing, <r> not run, <g> gaps.

## Gaps

| Step | Missing test or open question | Proposed test and assertion |
|------|-------------------------------|------------------------------|

## Glossary

| Term | Meaning in this repo | Code name |
|------|----------------------|-----------|

## Shapes cheat sheet

<One table, every tensor on the golden path in forward order: name, shape with
named dims, tiny-example value, real value, where it is made (function and file).
Readers keep this page open while reading the code.>

## Measured on this machine (optional)

<Timings, memory, or throughput you measured, with the hardware, versions,
config, and command. Never mix measured and estimated numbers in one table.>

## Coverage of existing docs

| Topic | Existing doc | Covered here | Note |
|-------|--------------|--------------|------|
| `<topic found in scope>` | `<existing doc link>` | `<doc link and step ID, or not covered>` | `<reason if excluded>` |

<State which existing doc remains authoritative for deployment or design
decisions, and which new doc traces source behavior and tests.>

## Out of scope

<Branches and subsystems not covered, and where they live in the code.>

## Refreshing this doc

<Commands used: scan, tests, check_explain_doc.py. The commit to diff against.>
````

---

## Sub-doc template

````markdown
# <Subsystem> (<entry> -> ... -> <exit>)

<2-3 sentences: what this subsystem does and when it runs.>

This doc is **not** <sibling topic>. See [<sibling>.md](<sibling>.md).

<Contents block. See "Contents" below.>

| Knob | Value | Set in | Notes |
|------|-------|--------|-------|

Running example used below: <short excerpt from index.md>.

## Overview

<ASCII call tree of the whole subsystem with step IDs, one line per call.>

Each node that produces something the next step uses ends with `-> output`:
the artifact, its type and shape, and where it lives. Nodes that only
dispatch get no tag. Keep each tag short, and put the detail in the
artifacts table.

```
main                                  -> best_model.pt (S3 model dir)              # train_entry.py  train.A1
  ├─ load_config                      -> cfg dict, 40 keys (memory)                # config.py       train.A1
  ├─ create_dataloaders               -> 2 DataLoaders, batch dict of [B,T,...]    # dataset.py      train.B1
  └─ Trainer.train                    -> best_model.pt, history list[dict]         # train.py        train.E1
```

Right under the tree, the artifacts table lists every output the reader will
meet in the doc:

| Step | Artifact | Type and shape | Lives in | Example |
|------|----------|----------------|----------|---------|
| train.A1 | `cfg` | `dict`, 40 keys | memory | `{"epochs": 50, "lr": 1e-4, ...}` |
| train.B1 | batch | `dict[str, Tensor]`, `categorical_ids [32, 240, C]` ... | GPU memory | see train.B2 |
| train.E1 | `best_model.pt` | `state_dict`, 3.4M params | `/opt/ml/model/` then S3 | written when val loss improves |

## <code>.A <Stage group name>

### <code>.A1 <Step name> (`path/to/file.py`)

<Step section shape, below.>

### <code>.A2 <Next step> ...

## Differences from <reference>

| Topic | Reference | This code | Example where it matters |
|-------|-----------|-----------|--------------------------|

## What this stack is not

| Path | Difference |
|------|------------|

## Verification checklist

1. <A check a reader can run on a real run or log, with the expected value.>
````

---

## Architecture card

One card per model architecture the doc set explains in depth. A small repo
with one model puts the card inside its model or train sub-doc as the
`<code>.C` group. A repo with many models gives each card its own file,
`arch-<family>.md`, with doc code `arch-<family>`.

````markdown
# <Family> architecture (<checkpoint or config used>)

<2-3 sentences: what the model is, where it sits in the repo, which config
this card uses.>

[TOC]

| Config key | Value | Source |
|------------|-------|--------|
| hidden size | 4096 | `hidden_size` in the checkpoint `config.json` |

## Architecture

<Mermaid flowchart TB with shapes on edges, as in playbook 7.>

<ASCII shape flow with the same shapes and real numbers.>

<Module class tree, with counts.>

## Parameters

<Formula for one block, the total, and the measured count if a tiny config was run.>

## <code>.A Inputs
### <code>.A1 Build the model inputs (`<processor or tokenizer file>`)
<Step section shape: how raw input becomes tensors, with the real tensors.>

## <code>.B Blocks
### <code>.B1 <Block name> (`<model file>`)
<Step section shape: excerpt, shapes, math with numbers.>

## <code>.C Outputs
### <code>.C1 <Head or sampler> (`<file>`)

## How this repo builds it

<Registry entry, config class, weight-loading key map, sharding across
devices, with a call tree from the model name to the loaded module.>

## Differences from <reference implementation or paper>
````

---

## Contents

Every doc longer than ~150 lines gets a contents block after its intro
paragraphs. Pick the form by where the repo is hosted
(`git remote get-url origin`):

| Host | Contents block | Why |
|------|----------------|-----|
| Bitbucket Cloud | `[TOC]` on its own line | Bitbucket renders it. Hand-written `#anchor` links break there, because Bitbucket prefixes heading ids with `markdown-header-` |
| GitLab | `[[_TOC_]]` on its own line | GitLab renders it |
| GitHub, local, or unknown | `## Contents` with a nested list of links to each `##` and step `###` heading | GitHub shows `[TOC]` as text. GitHub-style anchors (`#traind1-get_lr-warmup`) work there and in editor previews |

The HTML view (`render_html.py`) turns any of the three into working links,
and it has its own sidebar contents.

Links between docs point at the file (`[train.md](train.md)`) and name the
step ID in the text ("see train.D1"). Add `#anchor` only on GitHub-style hosts.

---

## Step section shape

**Headings carry the step ID.** Use exactly this form so readers, the test
matrix, and `check_explain_doc.py` can find each step:

| Level | Form | Example |
|-------|------|---------|
| Stage group | `## <code>.<Letter> <name>` | `## train.D Learning-rate schedule` |
| Step | `### <code>.<Letter><n> <name> (`file`)` | `### train.D1 get_lr warmup and cosine decay (`train.py`)` |
| Sub-step | a row in the step's sub-step table, ID `<code>.<Letter><n>.<m>` | `train.D1.2` |

`simple` mode in `index.md` may use `## 1. <name>` instead. Both forms are
checked.

A step reads like a short walkthrough: what it does, what goes in and out,
the real code, then what that code does to the running example. `Output`
always shows the object itself, not only its name: type, shape or schema,
where it lives, and a sample, in the formats from "Show the data, not only
the code" in [writing-modes.md](writing-modes.md). The checker warns when a
step has no example block or table. Use these
labels in this order. Omit `Math` or `Under the hood` when there is nothing
real to say. Never omit `Input`, `Output`, or `Tests`. The text under a label
can be several paragraphs. The labels are signposts, not a form to fill in.

This is a structural template, not a source excerpt or an executed test:

````markdown
### <stage>.<group><number> <operation> (`<source file>`)

**Idea.** <Purpose, then mechanism.>

**Input.** <Actual names, types, shapes/schemas, location and example values.>

**Output.** <The object itself, its location and a small sample. Label the sample
measured, from a named fixture, or illustrative.>

**Code.** <Call tree and a verbatim excerpt from the selected commit.>

**Example.** <Trace the shared running example through these lines.>

**Math.** <Formula, defined symbols and a computed example, when relevant.>

**Under the hood.** <Source-supported execution and memory behavior, when relevant.>

**Tests.**

| Step | Test | Command | Status | Asserts |
|------|------|---------|--------|---------|
| <step ID> | <existing test or none> | <real portable command or blank> | <actual result, not run with reason, or gap> | <verified assertion> |
````

For a filled example with retained evidence, see
[nanoGPT's embedding step](https://github.com/gozhiyuan/explaining-ml-codebases/blob/main/examples/nanogpt/model.md#modela1-embed-tokens-and-positions-modelpy).
Never copy a template's test name or status into a generated explanation.
Copy labeled excerpts from the named file; steps with logic, math or a
non-obvious condition need an excerpt.

Use a sub-step table only when a step has 4 or more sub-steps with different
inputs and outputs. Its `Code` column names the function, not a line.

---

## Tests table and gaps table

**Status values.** Use only these:

| Status | Meaning |
|--------|---------|
| `pass` | you ran it at the explained commit and it passed |
| `fail` | you ran it and it failed. Add the one-line error |
| `not run (needs GPU)` / `(needs network)` / `(needs AWS)` / `(needs deps: X)` / `(slow)` | it exists, you did not run it, and you say why |
| `gap` | no test exists. The row goes in the Gaps table |

**Commands.** Use a command the reader can paste from the repo root on any
machine, for example `pytest tests/test_x.py::test_name -q` or
`uv run pytest ...`. Never put a path from your machine in a command
(`/Users/<you>/...`, `/home/<you>/...`, a local venv). If the run needed a
special setup, describe it portably once ("`PYTHONPATH=src` from the model
folder"). For non-pytest repos, give the real command (`python -m unittest ...`,
`cargo test ...`).

**Gap rows** say what to test, not only that a test is missing:

| Step | Missing | Proposed test and assertion |
|------|---------|------------------------------|
| `<step ID>` | `<behavior without coverage>` | `<proposed test name, input and expected assertion justified by source>` |

---

## Diagram conventions

**ASCII** for call trees, shape flows, batch grids, KV block tables, and DAGs.
Use the `├─ └─` tree style. Put `# file.py  STEP_ID` comments at the right.
Give the file when the call moves to a new file. Leave it blank while the
calls stay in the same file. No line numbers.

**Mermaid** for the views ASCII draws badly. Pick the type by the question it
answers:

| Question | Diagram | Where |
|----------|---------|-------|
| What are the stages or processes, and what flows between them? | `flowchart` | index Big picture, each sub-doc Overview |
| How does data move through the model's layers, with what shapes? | `flowchart TB` with shapes on edges | every model architecture section and architecture card (required) |
| Which classes hold or extend which? | `classDiagram` | index Major components, model sub-doc |
| What happens, in order, in one training step or one request? | `sequenceDiagram` | index Key sequences, train and serve sub-docs |
| Which states does a request or job move through? | `stateDiagram-v2` | scheduler or job lifecycle sections |

A deep doc set has at least one of each of the first three that the repo
supports. Rules that avoid render errors:

- Start the block with the diagram type: `flowchart LR`, `sequenceDiagram`, `classDiagram`, `stateDiagram-v2`.
- Quote any label that has spaces, parentheses, brackets, or punctuation: `A["EngineCore (busy loop)"]`.
- Quote edge labels too, especially shapes: `-->|"[B, T, 256]"|`. Unquoted brackets in an edge label break the parse.
- Node IDs are plain words with no spaces: `engine_core`, not `engine core`.
- In `sequenceDiagram`, write aliases without quotes (`participant EC as EngineCore`). Quotes render literally there. Keep `;` and `#` out of message text.
- Mermaid keywords ignore case. A participant id such as `Opt`, `End`, `Loop`, `Alt`, `Par`, `Note`, `Box`, or `Rect` is read as the keyword, and the whole diagram fails. Use ids like `Optim`, `Done`, `Looper`. In a flowchart, never use a lowercase node id `end`.
- Keep each diagram under ~25 nodes. Split a bigger one.

For source-grounded examples, see the
[vLLM request and ownership diagrams](https://github.com/gozhiyuan/explaining-ml-codebases/blob/main/examples/vllm/index.md) and
[nanoGPT architecture and shape flow](https://github.com/gozhiyuan/explaining-ml-codebases/blob/main/examples/nanogpt/model.md).
Use the selected repository's actual actors, symbols and configuration;
these examples are not generic runtime contracts for other engines.

---

## `simple` mode variant

`index.md` keeps these headings: Mental model, At a glance, Big picture,
Running example, Doc map (only if sub-docs exist), Test coverage, Gaps. Major
components, data flow, and glossary become short lists. The Mental model is
the most important section in this mode. Write it for the audience, with the
analogy.

Each stage is a short section in `index.md`:

````markdown
## 2. Teaching the model (training)

**Idea.** <What, then an analogy.>

**Input.** <Plain words + one tiny example.> **Output.** <Same.>

**How it works.**
1. <3-6 short steps.>

**Why it matters.** <One sentence for this audience.>

**Code.** <Source files and symbols for this stage.>

**Tests.** <Existing test, portable command, actual status and assertion; or a gap with a proposed assertion.>
````
