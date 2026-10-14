---
name: explaining-ml-codebases
description: Explain, document, or refresh ML and AI codebases from source, including training, inference, serving, data pipelines, models, kernels, and agents. Produce simple or deep Markdown with code paths, tensor shapes, running examples, tests, and optional HTML. Choose source-grounded figures and selective state animations when they clarify the explanation. Use for repo walkthroughs and onboarding documents; answer isolated function questions directly. Blog narration and site-specific publishing are separate workflows.
---

# Explaining ML Codebases

## Overview

Turn a large ML/AI repo into markdown a reader can trust and check. Every
explained step states its idea, its input and output, a concrete example, the
code that does it, and the test that proves it.

**Core principle:** explain from the code, not from memory or upstream docs.
Show the key lines of real code and explain them. Point at code by file and
function, class, or config key, never by line number: lines drift on every
commit and readers do not jump to them. Every step points at a test, or names
the missing test as a gap.

Two depth modes:

| Mode | Like | Reader | Length |
|------|------|--------|--------|
| `simple` | eli5 | newcomer, PM, manager, non-specialist | short. One `index.md`, sub-docs only for very large repos |
| `deep` (default) | asd-ste100 | engineer who maintains or verifies the code | long. `index.md` plus one sub-doc per subsystem |

Both modes start simply. In deep mode, open the mental model, subsystem
overviews and new mechanisms with purpose, a few meaningful actions, and a
small input/output example; put implementation detail below. See "Give the
reader an easy entry" in [writing-modes.md](writing-modes.md).

Visuals are independent of depth: default to `auto`, with figures where useful
and selective state animation. Read [visual-guide.md](visual-guide.md) when
planning visuals. Static-only requests exclude players and videos. Markdown
is always produced; render HTML when requested. Blog narration and site
site formatting belong to `writing-ml-codebase-blogs`, the optional companion
in this repository, not another depth mode. Hand it Markdown plus evidence;
it can produce an article and apply a requested site profile.

Resolve `SKILL_DIR` to the folder containing this file (regardless of the
agent's installation location). Commands below use that path. The core works
without prerequisite skills; optional tools are loaded only for their output.
When subagents are unavailable, write the same drafts and facts sequentially.

For installation or optional tool setup, read [INSTALLATION.md](INSTALLATION.md).
Reuse available runtimes. An explicit setup request authorizes needed installs
within its scope and the host's execution policy; prefer isolated environments
and install only tools needed for the chosen output. If an optional backend
cannot be set up, deliver SVG/player output and report the unavailable video
or site preview. Missing test dependencies remain `not run` until installed
and actually executed; missing mandatory runtimes/assets block that helper.

## When to Use

- "Explain this repo / pipeline / model / serving stack."
- "Write a call-stack doc for training and prediction."
- "How does vLLM schedule a request?" at the level of a document, not one answer.
- "ELI5 how our training pipeline works for my manager."
- "Walk me through this FastAPI model service / this JAX trainer / this CUDA kernel / this recommender."
- "Improve / fix / refresh the explain docs in `docs/explain/`." Use **Revise mode**.
- "Explain the <model family> architecture in this repo." Write an architecture card (`arch-<family>.md`). If a doc set exists, add the card to it and link it from the doc map.

Not for: one-line questions about a single function (answer directly), or
non-ML repos (the archetype and stage playbooks will not fit).

## Inputs to settle first

Ask only for what you cannot infer. Use these defaults otherwise.

| Input | Default |
|-------|---------|
| Target | current repo. A public URL: shallow clone, record the commit SHA |
| Scope | whole repo for small repos. For large repos, the **golden path** (see below) |
| Mode | `deep`. Words like "ELI5", "simple", "for my manager" mean `simple` |
| Audience | `deep`: ML engineer new to this repo. `simple`: smart non-specialist |
| Demo model / config | the default config the README or CI uses. For serving engines, one popular dense model (for example a Llama-family model) |
| Visuals | `auto`: figures and selective animation. Honor `static`, requested styling, or specific media |
| Output dir | `<scope root>/docs/explain/`, where the scope root is the deepest folder that holds everything in scope (a single model folder when explaining one model). If the scope spans several top-level folders, use `<repo>/docs/explain/<topic>/`. User may override |

**Golden path.** Follow one representative path end to end: one model, one
config, one hardware setting, one request. List the alternative branches in a
table ("other attention backends", "other schedulers") instead of explaining
each one. A doc that covers every branch at equal depth explains none of them.

## Workflow

Copy this checklist into your todo tool and track it.

```
- [ ] 0. Settle inputs (target, scope, mode, audience, demo, output dir)
- [ ] 1. Recon: scan_ml_repo.py + README/docs/configs -> archetypes, must-cover list, doc plan
- [ ] 2. Choose the running example and storyboard useful visuals
- [ ] 3. Fan out: one subagent per subsystem writes a draft + facts
- [ ] 4. Tests: map every step ID to a test, run cheap tests, list gaps
- [ ] 5. Synthesize index.md, unify terms/example, render selected visuals
- [ ] 6. Verify docs and visuals; consolidate evidence.json before cleaning .work
- [ ] 7. (if requested) Render HTML and verify media playback/packaging
- [ ] 8. Report deliverables, checks, and limitations
```

### 1. Recon (you, not a subagent)

Run the scanner. It is stdlib-only and read-only.

```bash
python "$SKILL_DIR/scripts/scan_ml_repo.py" <repo> > /tmp/scan.md
python "$SKILL_DIR/scripts/scan_ml_repo.py" <repo> --json > /tmp/scan.json
```

It reports size tier, framework signals, archetype hints, entry points, key
classes (`nn.Module`, `Dataset`, schedulers, handlers), test files, and
configs. A second pass adds what Python analysis cannot see: notebooks (code
cells, and whether the repo is notebook-heavy), non-Python code (C/C++,
Rust, Go, Scala/Java entry points, types, tests, build files, language
bindings), SQL and dbt projects, and infra signals (Slurm, Kubernetes kinds,
launchers, Helm, Terraform). For non-Python code and notebooks, read those
sections of [framework-notes.md](framework-notes.md). Then read the README, top-level docs, the main config, and each entry
point's `main`.

Classify the repo with [ml-archetypes.md](ml-archetypes.md). Many repos mix
archetypes (for example, a Kafka/Flink feature pipeline feeding a PyTorch
model, or a distributed training job launched on Kubernetes). Each archetype
lists the components you must find. Use the relevant streaming, training,
serving and cluster sections of [stage-playbooks.md](stage-playbooks.md) for
these paths, even when recon misses a framework signal. If the
scanner reports TensorFlow, JAX, CUDA/Triton, or FastAPI, also read that
section of [framework-notes.md](framework-notes.md), and pass it to the
writers that cover that code. Note each data type the repo handles (tabular,
text, image, video, audio, sensor streams, robotics episodes, 3D, graphs,
multimodal), and pass the matching sections of
[modality-notes.md](modality-notes.md) to the data, loading, and serving
writers.

**Find the variant families.** Look for the places where a config key picks
one implementation from a family: losses, optimizers, schedulers, samplers,
augmentations, attention backends, metrics. Signs are a registry dict, a
`build_*` or `make_*` factory, Hydra `_target_`, or an `if cfg.x == ...`
chain. Record which value the golden path uses. Writers explain that variant
in depth and the others in one comparison table (playbook 14 in
[stage-playbooks.md](stage-playbooks.md)). The user can ask for another
variant later.

**Existing docs become a must-cover list.** Hand-written docs inside the
scope (call stacks, design docs, READMEs, runbooks, lessons-learned files)
hold knowledge the code does not show: deploy steps, past incidents, why a
choice was made. Skim their headings and tables. Write one topic per line to
`docs/explain/.work/must-cover.txt`, using a searchable name (an identifier,
a config key, or a short phrase), with the source after `#`:

```
<topic read in the repo> # <repo-relative doc path>:<location>
```

Keep it to the topics a maintainer would miss, usually 20-60 lines. Give
each writer the lines for its subsystem. Treat each topic as a claim to check
against the code, not a fact: old docs drift. A topic the doc set leaves out
on purpose goes in "Out of scope" with the reason.

**Independent mode.** If the user wants a version written from the code only
(for example to compare it with their own docs), skip this pass. Do not open
the docs they name. Give every writer and the fact checker a "Do not open"
list with those paths. Leave out the "Coverage of existing docs" table. Write
to a separate folder so nothing overwrites the old docs. After the run, the
coverage diff is a comparison tool, not a gate: build `must-cover.txt` from
the old docs and run `--must-cover` only if the user asks for the comparison.

Write the **doc plan** before any subagent starts:

```
docs/explain/
  index.md                     # always
  <subsystem>.md               # one per major subsystem (deep mode)
  arch-<family>.md             # one architecture card per model explained in depth (model zoos)
```

**Models get an architecture diagram.** Every doc that explains a model
draws it: a Mermaid flowchart with tensor shapes on the edges, plus an ASCII
shape flow (playbook 7 in [stage-playbooks.md](stage-playbooks.md)). For a
repo with many architectures (Hugging Face, vLLM, timm, diffusers), plan one
section on how a model plugs into the repo, one architecture card for the
golden model, and a table of how the other families differ. The user can ask
for more cards later ("explain the Qwen3 architecture"). Write each new card
as `arch-<family>.md` and add it to the doc map.

Splitting rules:

| Size tier (scanner) | `simple` | `deep` |
|---------------------|----------|--------|
| small (< 5k Python LOC) | `index.md` only | `index.md` + 1-3 sub-docs |
| medium (5k-50k) | `index.md` only | one sub-doc per stage group (data, train, eval, serve...) |
| large (> 50k) | `index.md` + up to 3 sub-docs | golden path only; one sub-doc per subsystem; split any sub-doc that passes ~1500 lines |

Assign each sub-doc a short code (`data`, `train`, `serve`, `kv`...). Step IDs
are `<code>.<Letter><n>` (`train.D6`), unique across the doc set, so the test
matrix in `index.md` can point at any step.

### 2. Running example

Pick **one** concrete input and carry it through every section. Every sub-doc
uses the same example so the reader can follow one thing end to end.

- Tabular or event pipeline: one entity (one event key, one user) with 2-3 rows.
- Training: one tiny batch with named dims, for example `B=2, T=4`, plus the real values (`B=32, T=240`).
- LLM serving: one prompt, its token IDs, one prefill step and two decode steps, the KV blocks it takes.
- RAG or agents: one query, the chunks it retrieves, the final prompt.

Record the example once in `index.md` under "Running example". Sub-docs quote
the part they need.

**Cover every input the golden path uses.** The example must exercise every
input channel and every branch that production runs, including the ones that
data turns on (an optional feature group, categorical ID or tag lists, a side image, a
conditioning input). An example with an empty channel leads writers to skip
that branch. If real data has a value there, give the example one.

**Fix the seed.** Any value that depends on random init, dropout, or sampling
(init loss, sampled tokens, attention weights) uses one seed and one config.
Use the repo's own seed if it has one. Give the seed to every writer.
Otherwise two docs report two different "measured" values for the same thing.

### 2a. Visual plan

Use the running example to identify difficult relationships, layouts, and
state changes. Read [visual-guide.md](visual-guide.md), choose useful figures
and optional players/videos, and save brief storyboards in `.work/visuals.json`.
Give each writer its visual briefs and shared values. Do not create a video
quota or change example values to fit a diagram. Prefer SVG for worked values,
Mermaid for structure, and the standalone trace player for evolving state.
Select optional tools from what is available; static figures remain a complete
fallback when video generation is unavailable.

In `auto`, create a video only for a specific teaching objective that benefits
from motion or interacting state changes. The bundled Manim scenes demonstrate
the renderer; they are not a default quota or a limit on what to explain.
Complex math, model transformations, scheduler/page ownership, request
lifecycles, concurrency and disaggregation may need original scenes built from
the selected source path. Do not substitute a generic template for that path.

### 3. Fan out to subagents

Use your subagent tool (Cursor: `Task`; Claude Code: `Agent`). Launch the
independent subsystems **in parallel**, within the available concurrency limit. Give each one the
**writer prompt** from [agent-prompts.md](agent-prompts.md), filled in with:
paths, mode, audience, step-ID prefix, running example, and output file
`docs/explain/.work/<code>.md` plus `.work/<code>.facts.json`.

Each writer must read [stage-playbooks.md](stage-playbooks.md) for its
stages, [writing-modes.md](writing-modes.md) for its mode, and
[doc-templates.md](doc-templates.md) for the section shape.

For a subsystem that is still too large for one agent, the writer splits it
and you launch child writers. This is the CodeWiki pattern: document the
leaves, then write the parent from the leaf docs.

Writers are read-only on source code. They may run cheap CPU snippets to
confirm a shape or a value (instantiate a tiny config, push a fake batch
through one module). They must never download large weights, use a GPU
without permission, or edit the repo.

**Run the real thing when it is small.** Real outputs beat reasoning about
them. A writer may run the golden path end to end (prepare data, a few
training steps, one prediction) in a **scratch copy outside the repo**
(`cp -r <repo> /tmp/explain-run-<code>`) when all of these hold:

- The data and weights it downloads total under ~200 MB, and network access is allowed.
- It runs on CPU (or the GPU the user allowed) in under ~10 minutes.
- It uses an existing environment or an isolated scratch environment whose
  setup is authorized by the request. Otherwise leave missing dependencies
  as `not run`; do not alter the source repo's environment or lockfiles.

Values from such a run are labeled measured, with the command, hardware, and
versions. The running example then uses the real values (for example the real
token ids) instead of UNVERIFIED ones. Delete the scratch copy afterward.

**Why, from history (optional, bounded).** Some choices look odd and the
code does not say why. For those few lines, a writer may look at history:
`git log -L <start>,<end>:<file> --oneline -n 3`, or `git blame -L <n>,<n>`
and then `git show --stat <hash>`. Cite the short hash. Keep it to about 5
lookups per sub-doc. Never read the whole log. Read a PR description or an
issue only when the user points to it.

### 4. Tests

Use the **test-mapper prompt** in [agent-prompts.md](agent-prompts.md), or do it yourself on small repos.

1. For each step ID, find tests that call its function or class.
2. Run the cheap ones with a timeout. Record the exact command and the result.
3. Mark tests that need a GPU, network, credentials, or cloud as `not run (needs X)`. Never mark them `pass`.
4. For each important step with no test, add a gap row: the step, a proposed test name, and the assertion it should make (input -> expected output).

The doc must show the user how to check each step themselves. A step with no
`Tests` entry is incomplete. A step with no test is fine **if** it is listed
as a gap.

### 5. Synthesize

You write `index.md` from the drafts, following the template in
[doc-templates.md](doc-templates.md). Write the **Mental model** last, after
you have read every draft: 4-8 sentences a reader can hold in their head,
including the facts that surprise people. Build the **Shapes cheat sheet**
from the writers' dims tables and shape flows. Then:

- Move each draft from `.work/` to its final name.
- Unify terms. One thing gets one name across every doc. Put it in the glossary.
- Unify symbols too. Read every sub-doc's dims table and formulas. One letter has one meaning across the doc set (writers often reuse `N` or `H` for different things).
- Check that every sub-doc uses the same running example values, including the seeded ones.
- Check that headings use the step-ID form (`### train.D1 ...`) and that each step ID has its own heading.
- Every function or key you name yourself in `index.md`, you open and read first.
- Read each doc once as a newcomer. If a paragraph is a chain of names and paths with no code and no example, replace it with an excerpt and a walkthrough.
- Check that overview and mechanism openings explain purpose, actions and the shared example before introducing implementation detail. Keep reference-only sections brief.
- Add cross-links. Each sub-doc says what it is **not** and links the sibling that covers it.
- If existing docs were found, add the "Coverage of existing docs" table to `index.md`: each must-cover topic, where this doc set covers it, or why it is out of scope. Link the existing docs. Say which one is the source of truth for what.
- Shorten paths. A sub-doc whose code lives under one folder declares it once, `<!-- explain-root: <actual scope path> -->`, plus a visible line naming that same path. Then it writes `src/train.py`, not the full path.
- Add a contents block to each doc over ~150 lines. The form depends on the host (`[TOC]` on Bitbucket). See "Contents" in [doc-templates.md](doc-templates.md).
- Render selected visuals from the shared example; retain editable SVG/scene/trace sources under `assets/`. Inspect them as described in [visual-guide.md](visual-guide.md).
- After verification and test mapping, consolidate facts, example provenance and visual briefs into `evidence.json` using [evidence-guide.md](evidence-guide.md). Link it from `index.md`. Only then delete `.work/`, unless the user wants to keep it.

### 6. Verify

```bash
python "$SKILL_DIR/scripts/check_explain_doc.py" docs/explain --repo <repo> --mode deep \
    --must-cover docs/explain/.work/must-cover.txt     # only when existing docs were found
```

With `--must-cover`, it warns for each topic that appears in no doc, so a
topic the old docs explained does not disappear without a reason in "Out of
scope".

It checks that:

- referenced files exist (in the repo or in installed packages such as `torch/...`)
- each "`symbol` in `file`" and `file::symbol` names something defined in that file, and each config key appears in its config file
- each call-tree line's call appears in the file its comment names
- each labeled code excerpt (`# file.py  Symbol` on its first line) still matches the file. It warns when the code changed, so the doc does not drift silently
- `pytest path::test` targets exist, and step sections have Input, Output, and Tests
- Mermaid blocks start with a valid type and avoid the known parse traps (keyword ids such as `Opt`, a lowercase `end` node), and `index.md` has the required sections
- no path from your machine (`/Users/...`, `/home/...`) appears anywhere

It also warns when a doc relies on line numbers. Fix every error and warning.

Then launch one **fresh** reviewer subagent with the **fact-check prompt** in
[agent-prompts.md](agent-prompts.md). It samples claims (shapes, defaults,
formulas, call order) and checks each against the code. Before you apply a
fix, open the cited line yourself. Reviewers make mistakes too.

`deep` mode: if the `asd-ste100` skill is installed, resolve `STE_SKILL_DIR`
to its actual installation folder and lint the prose (it skips
code blocks and tables):

```bash
python "$STE_SKILL_DIR/scripts/ste-lint.py" --disable passive-voice docs/explain/*.md
```

Inspect rendered visuals, first/middle/last animation states, controls and
static fallbacks. Keep measured/fixture/illustrative qualifications. Update
facts with actual test outcomes, then collect persistent evidence before
cleanup. The evidence collector validates structure, not source correctness.

### 8. Report

Tell the user: where the docs are, the doc map, the commit explained, test
results (passed / failed / not run), the top gaps, and what is out of scope.
Include visual assets, evidence.json, HTML asset folders to share, and any
rendering/playback checks that could not be completed.

### Revise mode (improve or refresh an existing doc set)

Use this when the user asks to improve, fix, update, or refresh docs this
skill already wrote, instead of starting over. Keep what passes. Change only
what is flagged.

1. **Build the worklist.** Run `check_explain_doc.py` on the folder and save the output. Every ERROR and WARN is an item: keyword ids in diagrams, line-number citations, steps with no example output, drifted excerpts, missing sections, machine paths.
2. **Find code changes.** Read the `Commit:` line in `index.md`. Run `git diff --stat <commit> -- <scope paths>`. Sections that name a changed file or function join the worklist, even if the checker passed them.
3. **Add the user's items.** For example "more output examples in data.md" or "explain the optional input path".
4. **Fan out by doc.** Launch one **reviser** per doc with work, using the reviser prompt in [agent-prompts.md](agent-prompts.md). Each one gets only its doc's items. Mechanical fixes (renaming a diagram id, removing a heading marker) you can make yourself.
5. **Verify as in step 6.** Run the checker until it reports 0 errors and no example-output warnings. Fact-check the changed sections only. Update the `Commit:` line and affected evidence/visuals while preserving unchanged records. Render the HTML again if one exists.

Report what changed per doc and what is still open.

### 7. HTML view (only when the user asks)

Markdown is the source of truth: it diffs in review, every host shows it, and
the checker validates it. Some hosts (Bitbucket) do not render Mermaid or
math. When the user asks for HTML, a web page, or a browsable view, render
the doc set into one page:

```bash
python "$SKILL_DIR/scripts/render_html.py" docs/explain   # writes docs/explain/explain.html
```

The page has a sidebar with every doc and its sections, search, rendered
Mermaid diagrams, KaTeX math (`\( \)`, `\[ \]`, `$$ $$`), highlighted code
with copy buttons, and links between docs that work inside the page. It loads
its libraries from a CDN, so the browser needs network access. Never edit the
HTML. Edit the markdown and render again. Follow the user's requested scope
for committing generated output.

The Markdown of every doc is embedded. Small images are embedded as data
URIs. In default `auto` media mode, videos, standalone iframe players, and
large images are copied into `explain.assets/` beside the HTML. Share both.
The renderer resolves paths from each Markdown file and handles multiline
HTML tags, posters, source elements, and media download links.

```bash
# Copy all media into the export's asset folder:
python "$SKILL_DIR/scripts/render_html.py" docs/explain -o dist/explain.html --media-mode bundle
# Embed all local media; fails rather than silently omitting assets over the limit:
python "$SKILL_DIR/scripts/render_html.py" docs/explain --media-mode inline --max-inline-mb 20
```

Iframe players must themselves be standalone; their dependencies are not
recursively packaged. See [visual-guide.md](visual-guide.md) for the included
trace player, static fallback, and optional MP4 export. CDN libraries still
require network access even in inline media mode; this is not an offline
export. Missing media or oversized inline assets produce a nonzero exit.

Links to repo files outside the doc set are rewritten to the Git host at the
explained commit, from `Commit:`. Pass `--link-base <url>` for another host.
Read the summary and fix unresolved links in Markdown. Host links work only
after that commit is pushed.

Check it before you report: open it (`open docs/explain/explain.html`, or a
browser tool) and visit every doc. A broken diagram shows its error in red in
place, and the page title starts with `[N diagram errors]`. With a browser
tool, read `window.explainDiagramErrors` on each doc. It must be 0. Fix the
markdown, render again, and recheck. Test video/iframe playback and media
loading after moving the HTML and its asset folder to a separate directory.

## Quick reference: what every step section must contain

| Element | `simple` | `deep` |
|---------|----------|--------|
| Idea (what + why) | 1-2 sentences + analogy | 1-3 sentences |
| Input / Output | plain words + one tiny example | the object itself: type, shape or schema change, where it lives (memory, disk, S3), and a sample from the running example (rows, array slice, JSON, file listing) |
| Code | key files and functions | call tree with step IDs and files, plus a verbatim excerpt (5-25 lines, annotated) for every step with logic or math |
| Example | the running example, small numbers | the running example, worked through each sub-step |
| Math | intuition, at most one formula | every formula the code computes, with a numeric example |
| Under the hood | only if it changes what the reader does | engine, parallelism, memory, hardware use, measured timings when you ran it |
| Diagram | a useful big-picture view | Call trees and shape flows for source detail; architecture, sequence or ownership figures chosen by the question. No diagram-type quota or repeated class-name chains |
| Tests | command + what it proves | table: step, test, command, status, what it asserts; plus gaps |

## Section format map: each stage is explained differently

Pick the sections from the stages the repo has (recon). Each stage has its
own main visual and its own main example. The full list is in
[stage-playbooks.md](stage-playbooks.md).

| Stage | Main visual | Main example to show | Playbook |
|-------|-------------|----------------------|----------|
| Trigger and pipeline | DAG with the artifact on each edge | the run parameters, deploy-time and run-time | 1 |
| Data preparation | data flow table, schema per step | 2-5 rows of the running example after each step | 2, modality notes |
| Distributed processing (Spark, Ray, Beam) | driver vs executor tree | what one task's input is, schema before and after | 3 |
| Features and encoding | feature table with formulas | one entity's features before and after | 4 |
| Data loading | batch grid with pad slots | one item, then one batch, with shapes and the mask | 6, modality notes |
| Model architecture | flowchart with shapes on edges, plus a class diagram | the running example through one block | 7 |
| Training loop | sequence diagram of one iteration | the loss on one item, the LR at a few steps | 8 |
| Tuning and calibration | before-and-after table | one prediction before and after | 9 |
| Evaluation | metric table | each metric computed by hand on a small example | 10 |
| Packaging and deployment | artifact tree, topology diagram | the artifact layout, the scaling knobs | 11, 12 |
| Request serving | request sequence diagram | one real request and response | 12 |
| Realtime streaming | session sequence diagram over time, latency budget table | frame and chunk sizes, partial and final outputs | 12 |
| Batch / offline scoring | flowchart from the input store through the workers to the output store | input rows, one batch through one worker, output rows and the commit marker | 12 |
| Async / queued inference | sequence diagram from submit to result | the stored payload, the queue message, the result object | 12 |
| Edge / on-device | export and runtime flow | memory and latency on the target device | 12, archetype K |
| Variant families | the switch call tree, a comparison table | the golden variant in depth | 14 |
| SQL / dbt pipeline | model DAG with grain and materialization | SQL excerpt and result rows per model | 2, archetype Q |
| Labeling and data quality | annotation flow | one item from raw input to adjudicated label | 2, archetype R |
| Privacy and governance | where sensitive fields enter and get protected | one record before and after masking, with fake values | 15 |
| Training cluster | topology flowchart, job lifecycle sequence | the launch script and the rank environment | 16 |
| Native code (C/C++, Rust, Go) | call tree across the language boundary | the data that crosses the boundary, typed on both sides | framework notes |

A repo type with its own checklist ([ml-archetypes.md](ml-archetypes.md))
adds to this map. A data type adds its layout rules
([modality-notes.md](modality-notes.md)).

## Red flags: stop and fix

- You describe a function you did not open.
- A shape has no named dims, or dims with no numbers.
- A formula has no worked numeric example (`deep`).
- A default value has no source (the config key, or the function or class that sets it).
- A step with logic or math has no code excerpt, or the excerpt is pseudo-code labeled as real code.
- A step names its output ("writes the features", "returns the batch") without showing it: no type, no shape or columns, no location, no sample.
- A call tree lists functions but not what each one hands to the next.
- A model input is missing from the architecture because its default is off, while the production data turns it on.
- A sentence carries several paths or line numbers. Name the function once, then explain.
- A section has no `Tests` entry and no gap row.
- A test is marked `pass` but you did not run it.
- The running example changes values between sections, or a "measured" value has no seed.
- Your brief said one thing, the code says another, and the doc follows the brief.
- You copied an upstream doc or blog instead of reading the code. Docs drift. When docs and code disagree, the code wins, and you record the mismatch.
- You explain every branch at the same depth instead of following the golden path.
- `simple` mode states something false to make it easier. A simplification is fine. Mark it "Simplified:" and keep it true.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| A wall of prose per stage | Lead with an Input / Output summary, then a call tree, then a sub-step table |
| An explanation of a serving engine with no request lifecycle | Trace the running request through every process and queue, in order |
| Model architecture without shapes | Write a shape flow diagram, one line per layer, `[B, T, D]` with real numbers |
| A Spark step explained as if it were pandas | Say what runs on the driver and what runs on executors, where the shuffle is, and when the lazy plan executes |
| Training and serving docs that silently disagree | Add a "train/serve parity" table. Name every place the two code paths differ |
| A deep doc so long nobody finishes it | Split by subsystem. Each sub-doc stays under ~1500 lines |

## Additional resources

- [visual-guide.md](visual-guide.md): automatic visual selection, design, trace player, optional video, and render checks.
- [evidence-guide.md](evidence-guide.md): preserve facts and running-example provenance for revision or companion workflows.
- `scripts/render_trace.py`: standalone HTML state player and SVG keyframes.
- `scripts/export_trace_video.mjs`: optional Playwright/FFmpeg MP4 export.
- `scripts/collect_evidence.py`: consolidate writer facts before cleanup.

- [ml-archetypes.md](ml-archetypes.md): 18 repo types (A-R), detection signals, components each must cover.
- [framework-notes.md](framework-notes.md): TensorFlow/Keras, JAX/Flax, CUDA and Triton kernels, FastAPI model apps, non-Python code (C/C++, Rust, Go, Scala/Java), notebooks.
- [modality-notes.md](modality-notes.md): tabular, text, image, video, audio, sensor streams, robotics episodes, 3D, graphs, multimodal, and dataset storage formats.
- [stage-playbooks.md](stage-playbooks.md): what to show for each ML stage, from data prep to serving.
- [doc-templates.md](doc-templates.md): `index.md` and sub-doc templates, test and gap tables.
- [writing-modes.md](writing-modes.md): `simple` and `deep` writing rules.
- [agent-prompts.md](agent-prompts.md): writer, test-mapper, and fact-check subagent prompts.
- `scripts/scan_ml_repo.py`: recon. Run it, do not read it.
- `scripts/check_explain_doc.py`: doc validator. Run it, do not read it.
- `scripts/render_html.py`: optional HTML reading view of the doc set. Run it, do not read it.
