# Explaining ML Codebases

**Turn a large ML repo into an explanation you can read, inspect and verify.**

An agent skill for **Codex, Claude Code, Cursor**, and other agents that support
`SKILL.md`. Navigate large ML, LLM and data engineering repos and review
AI-generated code with explanations you can check: source excerpts, tensor
shapes, concrete intermediate values, math, and tests with honest status.

Follow one example across a stream processor, training ranks or serving workers.
See what each step produces, who owns it, and where it goes next.

I built this from my experience reading ML codebases and writing technical
blog posts, with months of iteration alongside AI agents. It captures the
habits I find useful: understand the structure, trace a concrete example,
read the important code, check the tests, and make the mechanism visible.

## Why use it?

Reading a large ML repo means tracing behavior across files, configs, data
pipelines and GPU workers. A model's signature rarely tells you its tensor
shapes, a mask's meaning, or when shared state becomes ready. Reviewing
AI-generated changes adds more code to that tracing work.

The walkthrough puts the important source lines, their input/output values,
the explaining figure, and relevant tests together. You can follow a request
or training batch, check which logic has actually been tested, and find the
gaps without repeatedly jumping between implementation and test files.

## What you get

- **Code structure and walkthroughs:** clear component summaries and ordered
  steps, followed by real source excerpts, math and example inputs/outputs.
- **Tests alongside the explanation:** relevant test commands, actual results
  and uncovered behavior, so you can distinguish explained logic from tested logic.
- **Optional HTML:** searchable reading view with per-document TOCs, diagram zoom,
  math, code and embedded media.
- **High-quality diagrams:** editable SVG architecture maps, worked figures
  and interactive state players, designed for readable labels and clear boundaries.
- **Optional explanation videos:** original Manim scenes for math, model
  architecture and complex workflows, with **3Blue1Brown-inspired visual teaching**
  when requested. Full explanations need their own storyboard and scene work.
- **Optional blog articles:** a separate companion turns docs/evidence or source
  into narrative articles or a series, with Markdown and article HTML.

```text
docs/explain/                 docs/blog/  (optional companion)
  index.md                     article.md
  data.md / train.md / ...      article.json  (claim manifest)
  evidence.json                assets/       (figures and media)
  assets/                      article.html  (on request)
  explain.html  (on request)
```

## Quick start

Ask your agent with the GitHub URL:

```text
Install both skills from https://github.com/gozhiyuan/explaining-ml-codebases.
```

Or clone the repo and install directly with Python 3.9+:

```bash
git clone https://github.com/gozhiyuan/explaining-ml-codebases.git
cd explaining-ml-codebases

# Codex user skills; use your host's discovered skills path if it differs.
python3 scripts/install_skills.py --dest ~/.agents/skills --skill all

# Claude Code user skills:
python3 scripts/install_skills.py --dest ~/.claude/skills --skill all
```

Use `--skill core` for walkthroughs only or `--skill blog` for the companion
only. For Cursor, use the skills directory supported by your installation.
The installer copies the selected skills and their shared helpers, or creates
a release ZIP. It refuses to overwrite existing skills and does not install
optional dependencies.
See [installation and updates](INSTALLATION.md), including agent setup prompts,
legacy paths and optional dependencies. Codex locations follow the
[official skills documentation](https://developers.openai.com/codex/skills/).

## Choose an output

| Request | Skill | Result |
|---|---|---|
| Learn, review or maintain a codebase | `explaining-ml-codebases` | `deep` docs by default; `simple` for newcomers; optional docs HTML |
| Write a technical article or series | `writing-ml-codebase-blogs` | Narrative Markdown, claim manifest, figures; optional article HTML |
| Format for GitHub Pages/Jekyll | Blog companion + a site profile | Staged post and assets; local site preview when available |

Both skills can write Markdown and HTML and share visual tools. The core traces
code paths into a reference walkthrough with per-step inputs, outputs and tests.
The companion develops an article's argument, pacing, worked figures and series
structure. It can reuse the core's evidence or read source directly.

**Explain a repo:**

```text
Use explaining-ml-codebases to explain training and serving in this repo,
deep mode. Trace one model and configuration with shapes, math, intermediate
values and tests. Write docs/explain/ and render the HTML view.
```

**Create a blog article:**

```text
Use writing-ml-codebase-blogs to turn docs/explain/ and evidence.json into
an in-depth article about prefix caching. Carry one concrete example through
the code and worked figures. Preserve source citations and uncertainty.
Write article.md and render article.html. Animate only where motion helps.
```

**Prepare a site post:**

```text
Adapt that article for the GitHub Pages/Jekyll site at /path/to/site.
Inspect its conventions and create a matching profile from profiles/jekyll.json.
Stage the post and assets, then build a local preview if tools are available.
```

See [more prompts and modes](USAGE.md) for detailed series, realtime pipelines,
Megatron training, TensorRT-LLM serving, revision and static-only output.

## Coverage and examples

Use it for PyTorch and ML pipelines, LLM training and inference, Kafka/Flink
streams, Megatron/DeepSpeed distributed training, Kubernetes launches,
TensorRT-LLM/vLLM/SGLang serving, kernels, RAG, notebooks and mixed-language
repos. Python gets the deepest automatic analysis; native/JVM paths combine
pattern-based recon with source reading. Scope large repos to an execution path.

The agent uses recon hints from imports, entry points and configuration to
identify the repo's components, confirms them against source, and selects the
relevant explanation checklists. Visuals default to automatic selection:
architecture maps for ownership, worked figures for values and state, and
players or videos for sequences where motion helps. Ask for static-only output
to exclude animation. Framework detection is a starting point, not a guarantee.

[Recorded nanoGPT and vLLM examples](examples/README.md) include walkthroughs,
articles, editable figures and original Manim videos. They evaluate a tiny CPU
training update and selected scheduler/KV contracts. They are deliberately
narrow examples. Other workflows have source-reading guidance; cluster, broker
and GPU behavior still needs matching runtime evidence. See the
[coverage guide](GUIDE.md#best-fit) and [evaluation limits](evals/README.md).

The short video examples test rendering. Complex mathematical transformations,
scheduling, concurrent execution or disaggregated serving need their own
source-grounded storyboards and scenes. Full videos may run for several minutes
or use chapters; video is optional, not an article quota.

On GitHub, the example articles include static trace galleries, GIF previews
and MP4 links. Clone the repo and open their `article.html` files for interactive
players and video controls. SVG trace frames stay static on GitHub; GIF previews
can animate. HTML-only embeds are hidden in the repository Markdown view.

## Dependencies stay optional

Core helpers, SVG figures and standalone state players use **Python's standard
library**. No other skill, GPU or video stack is required. Docs/article HTML
uses CDN rendering libraries and needs internet access.

An agent can install needed tools when setup is authorized. MP4 export uses
Playwright/FFmpeg; smooth mathematical scenes use optional Manim and its
platform dependencies. Missing video tools fall back to SVG/player output;
missing repo test dependencies are reported as `not run`. A missing mandatory
runtime or asset is reported as a blocker for that output. See
[setup and fallback behavior](INSTALLATION.md#dependencies-and-fallbacks).

## Danger zone: personal taste

This skill reflects my taste in technical writing: concrete examples, clear
steps, source code beside the explanation, and visuals that show what changes.
Treat it as a starting template. Keep what helps, and adjust the voice, depth,
structure, diagram style and use of animation to suit your readers and your own
writing style. The source citations and honest test status are useful to retain
even when the presentation changes.

## Learn more

- [Detailed guide](GUIDE.md): workflow, coverage, worked step, checker and limits.
- [Prompts and modes](USAGE.md): docs, articles, series and generic publishing.
- [Installation](INSTALLATION.md): agent-assisted setup, updates and optional tools.
- [Visual guide](visual-guide.md): figure selection, animation and rendering checks.
- [Core instructions](SKILL.md) · [Blog companion](skills/writing-ml-codebase-blogs/SKILL.md).

MIT; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Writing guidance draws on [asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill)
and [eli5](https://github.com/dreambigou/eli5); original animations do not copy
3b1b's scenes or assets.
