# Detailed guide

[Quick start](README.md) · [Prompts and modes](USAGE.md) · [Installation](INSTALLATION.md)

Details of the explanation workflow, supported paths, validation and limits.

- [Why this exists](#the-problem)
- [Outputs and evidence](#what-this-skill-does)
- [Coverage and evaluations](#best-fit)
- [Workflow](#how-it-works)
- [Worked step](#what-a-step-looks-like)
- [Checklists](#what-it-covers)
- [Validator](#the-checker)
- [Standalone commands](#use-the-scripts-on-their-own)
- [Limits](#limits)
- [Repository files](#files)

## The problem

ML code is hard to review, and it is getting harder.

- **The important logic hides in a few lines.** A loss function, a mask, a
  `cumsum` over a flipped tensor. The function signature does not tell you
  the shapes, the units, or why the math is right.
- **A pipeline spans many systems.** A training run can pass through a
  Kafka consumer, a Flink or Ray processing job, object storage, and a
  distributed PyTorch or Megatron-LM job launched on Kubernetes. Serving adds
  a router, GPU workers, and model or KV-cache state. No single file shows
  how data, weights, and requests cross those boundaries.
- **The intermediate data is invisible.** Reviewers see a stream transform
  or `model(batch)`, but not an event's key and timestamp, an aggregation's
  state, the tensor shard held by one rank, or where each result is stored.
- **Training and serving drift apart.** Two copies of the feature code, two
  clocks, two window rules. A bug can hide in the difference.
- **More of the code is AI-generated.** Agents write code faster than people
  can read it. A PR can change a stream consumer, training loop, collective
  operation, and serving scheduler before anyone has traced the whole path.
- **Distributed behavior hides between workers.** A consumer restart can
  replay an event. A training rank can wait for a collective. A serving
  replica can route a request to workers that share model shards. Local code
  alone does not make ownership, ordering, or recovery obvious.
- **Docs go stale.** A design doc written last quarter cites line 812 of a
  file that moved. Nobody notices until someone trusts it.
- **Nobody knows which logic is tested.** A repo can have 300 passing tests
  and still no test for the mask, the label cutoff, or the threshold decode
  that decides the prediction.

## What this skill does

It traces implementation paths into worked explanations with source anchors,
example values and explicit verification status.

- **Real code, then what it does.** Each step pastes the 5-25 key lines,
  copied verbatim, with shape comments. Then it walks through what those
  lines do to one running example.
- **Real intermediate outputs.** Each step shows the object it produces: a
  stream event and its offsets, a window's state, a DataFrame's example rows,
  a tensor's dtype and shape per rank, or a checkpoint shard. It also says
  whether the object lives in a broker, process memory, GPU memory, local
  disk, or object storage.
- **One running example, end to end.** One entity, one prompt, or one batch
  moves through every doc, with a fixed seed, so the values agree everywhere.
- **Math with numbers.** Every formula the code computes, with each symbol
  defined and a worked numeric example.
- **Tests for every step.** For each step, it finds the tests, runs the
  cheap ones, and gives the command to paste. Where no test exists, it names
  the missing test and the assertion it should make. See "Every step is
  tied to code and a test" below.
- **Model architecture diagrams.** Every model gets a diagram from its input
  tensors to its output heads, with the tensor shape on every edge, the
  repeated stack, residuals, and conditioning inputs, plus a parameter count
  and the math per block. For model-zoo repos (Hugging Face, vLLM, timm,
  diffusers), it explains how a model plugs into the repo, writes an
  architecture card for one model, and compares the other families in a table.
  Ask later for more cards ("explain the Qwen3 architecture").
- **Config variants.** When a config key picks a loss, optimizer, scheduler,
  sampler, or attention backend, the doc shows the switch (key, default,
  factory), explains the variant your config uses in depth, and compares the
  others in one table with their formulas, files, users, and tests.
- **Diagrams.** Call trees show source detail. Mermaid or SVG figures explain
  a selected question about architecture, values or execution. Diagram types
  are optional; a repeated class-name chain adds no useful view.
- **Code by name, not by line number.** "`make_model` in `training.py`", not
  `training.py:212`. Names survive edits. Line numbers do not.
- **Plain language.** `deep` mode writes in a Simplified Technical English
  style: short active sentences, one name per concept, hedges kept. `simple`
  mode writes ELI5 style for managers and newcomers.

### Every step is tied to code and a test

Each explained step carries three anchors: the code that does it (function
and file, plus a verbatim excerpt), the data it produces, and the test that
proves it. The doc set is also a test-coverage map of the logic that matters.

- **Per step.** A tests table: test name, the exact command to paste, the
  status, and what the test asserts in input-to-output terms.
- **Honest status.** `pass` and `fail` only for tests that actually ran at
  the explained commit. Tests that need a GPU, the cloud, network, or a
  missing dependency are marked `not run (needs X)`, never `pass`.
- **Coverage matrix.** `index.md` lists every step ID with its test and
  status, and ends with a one-line summary: steps, passing, failing, not
  run, gaps.
- **Gaps as a test plan.** Every important step without a test gets a row
  with a proposed test name and the assertion it should make. A person or an
  agent can turn each row into a test directly.

See the recorded [nanoGPT coverage table](examples/nanogpt/index.md#test-coverage)
and [vLLM coverage table](examples/vllm/index.md#test-coverage) for worked
examples. Their tests are external validation fixtures shipped alongside the
docs; they are not upstream tests or production coverage. Commands, results,
and execution limits are retained in [evals/README.md](evals/README.md).

The checker keeps this honest: every `pytest path::test` it names must
exist, and every status must use the fixed words.

### Markdown for the record, HTML for reading

The **markdown is the source of truth.** It diffs in code review, every git
host shows it, and the bundled checker validates it. Ask for HTML and the
skill also builds `explain.html`: a sidebar with every doc
and section, search, rendered Mermaid diagrams and KaTeX math, highlighted
code with copy buttons, and links between docs. That helps on hosts like
Bitbucket that do not render Mermaid. Edit the markdown, then render again.

The HTML embeds Markdown and small images. Videos, standalone HTML players,
and large images are copied into `explain.assets/`; share that folder with the
page. Choose `--media-mode bundle` to copy all media or `--media-mode inline`
to embed media within a size limit. The page still needs internet access for
its rendering libraries. Links to source files point at your git host, pinned
to the commit the docs explain.

### Worked figures and selective animation

The agent chooses visuals by the question they answer: architecture maps for
ownership, SVG panels for concrete tokens/tensors and before/after states,
and animations for sequences or overlap. There is no figure or video quota.
Ask for static-only output to exclude animation. Visuals work in both depth
modes and do not require blog-style writing.

The included `render_trace.py` builds a standalone HTML stepper and SVG
keyframes from one shared JSON trace. It needs only Python's standard library.
Optional `export_trace_video.mjs` exports the same states to MP4 using Node,
Playwright/Chromium and FFmpeg. Its output shows discrete state changes;
continuous motion or mathematical transformations use the original optional
Manim templates in `render_manim.py`: tensor flow, cache reuse and causal
attention. Missing video tools do not block static figures or players.

See [visual-guide.md](visual-guide.md) for design, automatic selection,
embedding, optional tool setup, and visual checks. The sample
[cache trace](assets/prefix-cache-trace.json) illustrates the format; it is not
an authoritative explanation of a particular inference engine.

### Evidence that survives cleanup

`collect_evidence.py` consolidates writer facts, configuration, running-example
values, uncertainties and visual briefs into `evidence.json` before temporary
drafts are removed. A later documentation revision or companion article skill
can reuse this ledger and reopen source for new claims. See
[evidence-guide.md](evidence-guide.md). Blog narrative and site
formatting use the optional companion described below.

### Optional blog articles and GitHub Pages

This repository contains two independently installable skills:

| Skill | Output | Default reader |
|---|---|---|
| `explaining-ml-codebases` | Checked Markdown walkthroughs, optional docs HTML, evidence and visuals | Developers learning or maintaining a repo |
| `writing-ml-codebase-blogs` | Narrative `article.md`, claim manifest, illustrated article HTML, optional site package | Readers following one technical question |

The companion consumes explanation docs/evidence or reads source directly.
It builds an argument around one worked example rather than reproducing the
documentation table of contents. Figures and selective animations are chosen
by what they explain. A video is useful for a transformation or sequence;
static figures often explain ownership or comparison better. Static-only
requests exclude videos and interactive players.

For a narrative article or a series, use the separate companion. GitHub
Pages staging uses a configurable Jekyll profile; inspect the target site's
layout and asset conventions rather than assume a particular theme. See
[article and publishing prompts](USAGE.md#blog-articles-and-series).

See [the companion skill](skills/writing-ml-codebase-blogs/SKILL.md) and the
[tested examples](examples/README.md). The original Manim scenes use teaching
techniques familiar from mathematical explainers; they do not copy 3b1b's
scene code, art, narration or branding.

### For people now, for agents later

The same structure that helps a person helps an AI agent that debugs the
code later. Step IDs like `train.D1`, "function in file" citations, the
artifacts table, and the test matrix give the agent a map: where each value
is made, what it should look like, and which test to run to confirm a fix.

## Best fit

It works best on **Python ML and data repositories**, including code that
coordinates native runtimes, stream processors, and clusters. The depth of
automatic checks depends on the language; execution checks also depend on
available dependencies, data, GPUs, brokers, and cluster access.

| Support | Repos | What you get |
|---------|-------|--------------|
| **Python analysis** | PyTorch, Megatron-LM, TensorFlow, JAX, Ray Data, PyFlink applications, pipeline orchestration, Python serving code | recon parses entry points, classes, training loops, serving hooks and tests. Writers trace source and run feasible fixtures. pytest targets are mapped; execution status and missing dependencies are recorded |
| **Mixed-language analysis** | vLLM, SGLang, TensorRT-LLM, Text Generation Inference, notebook-heavy research repos, dbt and SQL pipelines | Python AST analysis plus notebook parsing and pattern-based native/SQL recon. Writers follow bindings and runtime boundaries; source symbols and excerpts are checked where supported |
| **Native and JVM analysis** | Kafka Streams or Flink Java applications, Scala data jobs, TensorRT/C++ runtimes, llama.cpp, ollama | recon finds entry points, types, tests, build files and bindings. Writers inspect source directly; runtime checks use the project's own environment and commands (`ctest`, `cargo test`, `go test`, `sbt test`) |

### Streaming and distributed systems

The workflow follows code and configuration across processes and ranks.
Recon supplies hints; the agent verifies the selected architecture against
source, launch scripts and manifests. These use cases have guidance in
[stage-playbooks.md](stage-playbooks.md) and [ml-archetypes.md](ml-archetypes.md):

| Workflow | Frameworks and platforms | What the explanation follows |
|---|---|---|
| Event and feature processing | Kafka, Kafka Streams, Flink, Beam, Ray Data, Spark, Feast | producers/consumers, partitions, offsets, event time, windows, state, backpressure, checkpoint/commit order and replay |
| Realtime model pipelines | PyTorch models connected to Kafka/Flink or SSE, WebSocket, gRPC and WebRTC services | event/frame to tensor, preprocessing, batching, model state, partial outputs, queue bounds and reconnect behavior |
| Distributed training | PyTorch DDP/FSDP, Megatron-LM/Core, DeepSpeed; Kubernetes, Slurm or Ray launchers | rank topology, data/model/optimizer shards, microbatches, actual collectives, tensor/pipeline/data parallelism, checkpoints and restart |
| Distributed inference and serving | TensorRT-LLM, vLLM, SGLang, Triton Inference Server; Kubernetes deployments | routing and replicas, TP/PP worker groups, weights or engine artifacts, prefill/decode, KV ownership, communication and streaming responses |

These systems occupy different layers: [Kafka](https://kafka.apache.org/intro/)
provides event streaming; PyTorch runs model computation; [Kubernetes](https://kubernetes.io/docs/concepts/workloads/controllers/job/) launches
and manages workloads. The docs follow the actual connections between them.
For NVIDIA stacks, distinguish TensorRT model optimization/runtime code from
[TensorRT-LLM's multi-GPU and multi-node inference](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html).
The training path similarly follows the selected
[Megatron parallelism configuration](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md).

For large systems, name the application or execution path to explain: one
event pipeline, one training configuration, or one model's serving lifecycle.
The skill can document these paths from source even when a full deployment
cannot run locally. Broker, multi-node and GPU behavior needs matching runtime
evidence; small local fixtures retain their stated verification boundary.

**Recorded evaluations.**

- nanoGPT's tiny CPU update, with source excerpts, measured tensors and an
  external validation fixture.
- vLLM's selected scheduler/KV contracts, with original methods tested using
  stand-ins. The complete engine and GPU execution were not run.

Both include developer docs and a companion article. See
[evals/README.md](evals/README.md) for pinned commits, checks and limits.

Kafka/Flink deployments, Megatron multi-node training, Kubernetes recovery and
TensorRT-LLM distributed inference are intended workflows supported by the
instructions; they have not received end-to-end evaluations in this checkout.

**Not a good fit.**

- Code that is not ML or data work. The checklists will not match.
- A question about one function. Ask the agent directly.
- A huge monorepo with no scope. Name the folders or the path to follow.
- Code the agent cannot read locally.

## How it works

The main agent plans and checks. Subagents can do deep reading in parallel;
an agent without subagents uses the same drafts and facts sequentially.

1. **Recon.** `scan_ml_repo.py` sizes the repo and scores 18 repo types
   (training code, LLM serving engine, data/feature pipeline, recommender, ...) from
   imports and class names. It also lists the entry points and the main
   classes. For vLLM it finds `AsyncLLM`, `EngineCore`, `Scheduler`,
   `KVCacheManager`, `GPUModelRunner`, and the attention backends.
2. **Plan.** It picks one golden path (one model, one config, one request),
   one running example with a seed, one doc per subsystem, and useful visual briefs.
3. **Write, in parallel.** One writer subagent per subsystem reads the code,
   runs small CPU snippets to get real shapes and values, and drafts its doc.
4. **Tests.** A test-mapper subagent links every step to tests, runs the cheap
   ones, and lists the gaps.
5. **Synthesize.** The main agent writes `index.md`, aligns terms and
   symbols, and adds a contents block in the form your git host renders.
6. **Verify.** `check_explain_doc.py` validates every doc. Then a fresh
   fact-checker subagent samples claims against the code. Rendered visuals are
   inspected, and facts/provenance are retained in `evidence.json`.
7. **HTML (on request).** `render_html.py` builds the reading view. The agent
   opens it and checks diagrams, player controls, media playback, and portable assets.

## What a step looks like

See [model.A1 in the nanoGPT example](examples/nanogpt/model.md#modela1-embed-tokens-and-positions-modelpy).
It follows actual token IDs through token and position embeddings, shows a
measured `[2,4,8]` output, quotes `GPT.forward`, and identifies the external
fixture that checks it. Its [measurement record](examples/nanogpt/assets/measurement.json)
retains the seed, configuration and results. The tiny configuration is for
explanation; it does not establish full training behavior.

## What it covers

The skill recognizes 18 repo types. Each one comes with a checklist of the
components the docs must explain.

| Type | Examples |
|------|----------|
| Tabular ML and MLOps pipelines | scikit-learn, XGBoost, AutoGluon, SageMaker Pipelines, Airflow |
| Deep learning and distributed training | PyTorch DDP/FSDP, Lightning, TensorFlow/Keras, JAX/Flax |
| LLM pretraining, fine-tuning, RL post-training | Megatron-LM/Core, TRL, DeepSpeed, PEFT, nanoGPT |
| LLM inference engines | vLLM, SGLang, TensorRT-LLM |
| Model serving apps and platforms | FastAPI, Triton Inference Server, KServe, SageMaker endpoints, Kubernetes deployments |
| Data and feature pipelines | Kafka/Kafka Streams, Flink, Beam, Ray Data, Spark, Feast |
| RAG and agents | LangChain, LlamaIndex, DSPy |
| Kernels and compilers | CUDA, Triton kernels, torch.compile |
| Recommenders and ranking | TorchRec, TFRS, two-tower retrieval |
| Computer vision and diffusion | torchvision, timm, detectron2, diffusers |
| Export, compilation, quantization | ONNX, TensorRT, TorchScript, bitsandbytes |
| Experiment tracking and MLOps platforms | MLflow, W&B, DVC |
| Eval harnesses | lm-evaluation-harness, HF evaluate, ragas |
| Time series and forecasting | statsmodels, Prophet, darts, GluonTS |
| Robotics and embodied learning | LeRobot, MuJoCo, Isaac, ROS, Gymnasium |
| Realtime and streaming AI services | PyTorch inference in event/media pipelines, SSE, WebSocket, gRPC, WebRTC (aiortc, LiveKit), voice agents |
| SQL, warehouse, and dbt pipelines | dbt, Snowflake, BigQuery, DuckDB, SQL in Airflow |
| Labeling, annotation, and data quality | Label Studio, Argilla, cleanlab, Snorkel, Great Expectations |

It reads more than Python. Notebooks are parsed cell by cell, and a
notebook-heavy repo is classified from its notebooks. C/C++, Rust, Go, and
Scala/Java code gets its entry points, main types, tests, build files, and
the bindings between languages (pybind11, PyO3, cgo, JNI), so engines like
llama.cpp, ollama, and the Rust router in Text Generation Inference are
recognized. Two cross-cutting playbooks cover privacy and data governance
(where sensitive fields enter and get masked) and training-cluster infra
(Slurm, Kubernetes training jobs, rendezvous, checkpoints, preemption).

Data types get their own rules too. For tabular, text, image, video, audio,
sensor streams, robotics episodes, point clouds, graphs, and multimodal
pairs, the docs show the raw format, the decode step, the tensor layout at
each step, how streams align in time, and the traps of that data type.

Each stage is explained in its own format. Data prep shows the engine and
the rows at each step. The model shows an architecture diagram with shapes.
Training shows one iteration and the loss math. Serving starts by naming the
mode, then shows what matters for that mode:

| Serving mode | What the docs show |
|--------------|--------------------|
| Online endpoint | one request through every process and queue, batching, contracts |
| Streaming session (WebSocket, WebRTC, token streaming) | the session over time, chunking, a latency budget, backpressure |
| Batch / offline (SageMaker Batch Transform, Spark or Ray scoring, offline LLM) | how work is split, where the model loads, one batch through one worker, output commit, reruns |
| Async / queued | payload storage, the queue, workers, result delivery, retries |
| Edge / on-device | export format, runtime, quantization cost, model updates |

Framework notes cover the TensorFlow, JAX, CUDA/Triton, and FastAPI details
that change where code runs and when.

## The checker

`check_explain_doc.py` makes the docs fail loudly instead of drifting
quietly. It checks that:

- every file, "function in file", `file::Class.method`, and config key a doc names exists
- every labeled code excerpt still matches its file, so a code change flags the doc
- every call-tree name appears in the file its comment names, and trees carry `-> output` tags
- every step has Input, Output, and Tests, and shows an example output
- every doc that explains a model has an architecture diagram with shapes on its edges
- every `pytest path::test` target exists, and test statuses use fixed words
- Mermaid blocks avoid the known parse traps (keyword ids such as `Opt`, a lowercase `end` node)
- no machine-specific path (`/Users/...`) appears, and long docs have a contents block
- optional: every topic on a must-cover list from older docs appears somewhere

## Use the scripts on their own

The scripts also work without an agent. Set `S` to this checkout or installed
skill's scripts directory:

```bash
S=/path/to/explaining-ml-codebases/scripts

# What kind of ML repo is this, and where do I start reading?
python "$S/scan_ml_repo.py" path/to/repo

# Validate a doc set (add --must-cover topics.txt to compare with older docs)
python "$S/check_explain_doc.py" docs/explain --repo .

# Build the HTML reading view
python "$S/render_html.py" docs/explain

# Package media into another directory
python "$S/render_html.py" docs/explain -o dist/explain.html --media-mode bundle

# Try the included illustrative trace
python "$S/render_trace.py" "$S/../assets/prefix-cache-trace.json" \
  -o /tmp/cache-player.html --frames-dir /tmp/cache-frames

# Optional: export a video using installed Playwright/Chromium and FFmpeg
node "$S/export_trace_video.mjs" /tmp/cache-player.html -o /tmp/cache-trace.mp4

# Optional original smooth mathematical scene, with editable source and poster
python "$S/render_manim.py" "$S/../assets/manim-attention.json" \
  -o /tmp/attention.mp4 --python /path/to/manim-env/bin/python
```

## Limits

- **It explains what the code does, not why it was designed that way.** It
  can look up a few commits for odd choices. It cannot recover a decision
  that lived only in a meeting. Keep your design docs and runbooks. The
  default mode reads them and links to them.
- **Large repos get one golden path, not every branch.** Other branches go
  in a table with file pointers.
- **Tests that need a GPU, the cloud, or a missing dependency are not run.**
  The doc marks them `not run (needs X)` and never `pass`.
- **The checker catches mechanical problems.** It cannot tell that an
  explanation is unclear. The fact-check pass samples claims. It does not
  prove every claim.
- **A full deep run on a large repo takes a while** and spawns several
  subagents. Scope it to the paths you care about.

## Files

```
README.md            quick start and output choices
USAGE.md             copyable prompts for docs, articles, series and site staging
INSTALLATION.md      agent-assisted setup, optional tools and fallback behavior
GUIDE.md             detailed human-facing reference
SKILL.md              workflow, modes, rules (what the agent reads first)
ml-archetypes.md      18 repo types and the components each must cover
modality-notes.md     tabular, text, image, video, audio, sensor, robotics, 3D, graph data
stage-playbooks.md    what to show for each ML stage, data prep to serving
framework-notes.md    TensorFlow, JAX, CUDA/Triton, FastAPI specifics
doc-templates.md      index and sub-doc templates, step shape, diagrams, contents
writing-modes.md      simple (ELI5) and deep (STE-style) writing rules
agent-prompts.md      writer, reviser, test-mapper, and fact-checker prompts
visual-guide.md       figure selection, styling, players, video, render checks
evidence-guide.md     persistent source facts and example provenance
assets/prefix-cache-trace.json  illustrative visual starter
scripts/scan_ml_repo.py       recon: repo type, entry points, key classes, tests
scripts/scan_extras.py        recon pass 2: notebooks, non-Python code, SQL/dbt, infra
scripts/check_explain_doc.py  validator for a doc set
scripts/render_html.py        HTML reading view with portable media packaging
scripts/render_trace.py       standalone state player and SVG keyframes
scripts/export_trace_video.mjs optional MP4 export
scripts/collect_evidence.py   persistent evidence consolidation
scripts/render_manim.py       original optional Manim scenes and numerical metadata
scripts/install_skills.py     install core/blog independently or package a release ZIP
skills/writing-ml-codebase-blogs/  narrative workflow and GitHub Pages adapter
evals/                       pinned public cases, protocol, recorded results
examples/                    source-grounded docs/articles and editable visuals
tests/test_visual_outputs.py portable export and evidence checks
```

## Acknowledgments

- [asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill) and [eli5](https://github.com/dreambigou/eli5) for the two writing modes.
- [CodeWiki](https://fsoft-ai4code.github.io/CodeWiki/) for documenting a large repo leaf-first, then writing the overview from the leaves.
- [DeepWiki](https://deepwiki.com/) for showing that repo-level docs with diagrams help readers.
