# Prompts and output choices

[Quick start](README.md) · [Detailed guide](GUIDE.md) · [Installation](INSTALLATION.md)

## Modes

| Mode | Use it when | Say |
|------|-------------|-----|
| `deep` (default) | an engineer must maintain or verify the code | "explain this repo", "write a call-stack doc" |
| `simple` | the reader is a manager, PM, or newcomer | "ELI5 how training works for my manager" |
| default existing-docs pass | you already have design docs or runbooks | nothing to add: the skill reads them, makes a must-cover list, and maps each old topic to the new docs |
| independent | you want a version from the code only, to compare with your own docs | "independent mode, don't open docs/design/" |
| revise | the docs exist and need fixes or a refresh after code changes | "revise mode on docs/explain/" |
| HTML view | you want a browsable page | "also render the HTML view" |

**Revise mode** keeps what passes and changes only what is flagged. It builds
a worklist from three sources: the checker's findings, the code that changed
since the `Commit:` line in `index.md` (`git diff`), and your requests. One
reviser subagent fixes each doc. Then the checker runs again until it is
clean. Run it after a big PR and the docs follow the code.

## Usage

Install the core or companion as described in [INSTALLATION.md](INSTALLATION.md),
then ask in plain language. Skill names make the intended workflow explicit.

**Explain a whole repo:**

```text
Explain this repo. Training and serving, deep mode.
```

**Explain one path in a large public repo:**

```text
Clone vllm-project/vllm and explain how one request is scheduled and served.
Golden path: one Llama model on 1 GPU. Write to docs/explain/.
```

**Explain one subsystem in a monorepo, with the HTML view:**

```text
Use the explaining-ml-codebases skill. Scope: the model in models/ranker/
and its scoring service in services/ranker_api/. Also render the HTML view.
```

**Explain a realtime event-to-model pipeline:**

```text
Explain one event from Kafka through our Flink feature job into the PyTorch
inference worker and its output topic. Include event-time rules, offset and
checkpoint ordering, model batching, backpressure, and replay after failure.
Write deep subsystem docs and the HTML view. Label unexecuted runtime checks.
```

**Explain distributed training or serving:**

```text
Explain our selected Megatron training configuration and Kubernetes launch.
Show what each rank owns, the microbatch schedule, collectives, effective
batch size, checkpoint shards, and how a failed job resumes.
```

```text
Explain one request through our TensorRT-LLM serving deployment: routing,
model/runtime initialization, multi-GPU workers, prefill/decode, KV cache,
and the streamed response. Pin the source and configuration being explained.
```

**Explain one more model architecture in a model-zoo repo:**

```text
In the vLLM explain docs, add an architecture card for the Qwen3 MoE model:
config, architecture diagram with shapes, routing math, and how vLLM loads
and shards it.
```

**Write for a non-technical reader:**

```text
ELI5 how our training pipeline works, for my manager.
```

**Compare with docs you wrote by hand:**

```text
Use the explaining-ml-codebases skill in independent mode. Don't open
docs/design/. Scope: models/forecaster/. Write to docs/explain-forecaster/.
```

**Fix or refresh existing explain docs:**

```text
Use the explaining-ml-codebases skill in revise mode on docs/explain/.
```

You do not need to ask for tests, the validator, the contents block, or a
diagram check. The skill does those every time.

## Blog articles and series

Blog writing uses `writing-ml-codebase-blogs`, the separate companion; it is
not another `simple`/`deep` mode of the documentation skill. Install both for a
docs-to-article workflow, or only the companion for an article from source.

**Turn checked docs into an illustrated article:**

```text
Use writing-ml-codebase-blogs to turn docs/explain/ and evidence.json into
an in-depth article about how this engine reuses a prompt prefix.
Audience: Python developers new to inference engines. Carry one concrete
example through the narrative, code and worked SVG figures. Preserve source
citations and uncertainty. Render article.md and article.html into docs/blog/.
Use animation only when it explains a sequence better than static panels.
```

**Write directly from source:**

```text
Use writing-ml-codebase-blogs to explain one nanoGPT training iteration
from source. Explain batching, embeddings, causal attention, loss, backward
and the optimizer update with shapes and numeric examples. Write a detailed
article and HTML; keep expensive or unavailable tests marked not run.
```

**Plan and write a multi-part series:**

```text
Use writing-ml-codebase-blogs to create a detailed series about this serving
repo. Plan the overview and separate articles for the request lifecycle,
scheduling, KV allocation and reuse, model execution, and distributed workers.
Inspect the selected source paths first; adapt the outline to what exists.
Give each article a distinct reader question, worked example and source links.
Deliver a series index, cross-linked article Markdown and HTML in docs/blog/.
Cover important branch conditions and failure paths. Do not compress the
whole repo into the length or scope of the bundled evaluation examples.
```

A prompt can ask for depth and scope; accurate articles still depend on source
reading and evidence. The bundled examples test narrow paths, not the quality
or completeness of a long generated series.

**Static figures or optional mathematical animation:**

```text
Use static-only visuals: worked SVG panels and architecture diagrams.
```

```text
Include an original Manim animation of the attention calculation if motion
helps. Set up only the tools needed, using an isolated tool environment.
If video rendering is unavailable, deliver SVG keyframes and a standalone
HTML player and report which video check could not run.
```

**Explain a complex mechanism with a custom visual:**

```text
Explain how this selected serving configuration schedules concurrent requests
and maps their tokens to KV pages. Read the actual scheduler, worker and backend
paths. Storyboard request IDs, token budgets, page ownership, inputs/outputs and
dependencies across the relevant CPU/GPU actors. Use an original animation only
if watching those interactions teaches more than a figure or stepper. Label
unmeasured timing illustrative and do not infer overlap from function-call order.
```

For disaggregated serving, scope the question to the implementation's prefill
producer, KV transfer, decode consumer and readiness conditions. For math or
model architecture, trace a concrete calculation or tensor transformation.
The bundled templates are starters; advanced scenes require source work and
custom code, not a longer version of the demonstration clips. Full videos can
run for several minutes. Give a time budget when useful, and ask for chapters
aligned with article sections for a long workflow or derivation.

**Stage for a GitHub Pages/Jekyll site:**

```text
Adapt docs/blog/article.md for the GitHub Pages/Jekyll site at /path/to/site.
Inspect its layout, front matter, math support and asset conventions. Create
a site profile using profiles/jekyll.json as a starting point. Stage the post
and assets in a new directory, then build a local preview if tools are available.
```

The site profile is optional and configurable. Markdown and article HTML work
without a site. The adapter currently targets Jekyll; other publishing systems
need their own formatting step. Local staging, a successful site build, browser
inspection and publication have separate statuses.
