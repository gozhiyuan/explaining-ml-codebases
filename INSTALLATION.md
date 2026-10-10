# Installation and optional setup

[Quick start](README.md) · [Prompts](USAGE.md) · [Detailed guide](GUIDE.md)

Skill installation copies instructions and helpers. It does not install an ML
repo, download models or install every possible rendering dependency.

## Ask your agent to install

Give your agent the repository URL:

```text
Install both skills from https://github.com/gozhiyuan/explaining-ml-codebases.
```

When the checkout is already open, `Install both skills from this repository.`
is enough.

For just the walkthrough skill, say `Install explaining-ml-codebases from this
repository.` For just the blog companion, name `writing-ml-codebase-blogs`.
The agent should discover its host's skills directory, use the packaging helper
below, and verify the installed files. Preserve an existing installation before
updating it. Optional rendering tools are set up when needed and authorized;
the installation prompt does not need to repeat these instructions.

The agent can clone the repository, read this guide, and run the installer.
The companion in the source tree uses shared top-level helpers: copying
only `skills/writing-ml-codebase-blogs/` is not sufficient for independent
rendering. Use this repository's installer or a packaged release.

### What the installer does

`scripts/install_skills.py` uses only Python's standard library. With `--dest`,
it copies `core`, `blog` or `all` into separate named skill folders, including
instructions, references, templates, helpers and license notices. The blog
package receives the shared renderers it needs to work independently. With
`--archive`, it creates the same folders inside a new ZIP instead.

It refuses to overwrite an installed skill or archive. It does not change agent
configuration, install packages, modify the repository being explained, or
bundle example repos, tests, model weights or tool environments.

## Install from a checkout

With Python 3.9+:

```bash
# Codex user skills:
python3 scripts/install_skills.py --dest ~/.agents/skills --skill all

# Codex skills for one target project:
python3 scripts/install_skills.py --dest /path/to/project/.agents/skills --skill all

# Claude Code user skills:
python3 scripts/install_skills.py --dest ~/.claude/skills --skill all

# Any supported host or existing skills location:
python3 scripts/install_skills.py --dest /path/to/agent/skills --skill core
```

`--skill core`, `blog` or `all` selects the output workflows independently.
The helper refuses to replace existing folders and does not modify agent
configuration. Keep a backup or choose a new destination before updating.

Current Codex locations are documented in
[Build skills](https://developers.openai.com/codex/skills/). Existing Codex
installations may expose `~/.codex/skills`; use the actual discovered path
for that host. Cursor users should use the skills location supported by their
editor version. Subagents are optional; the workflow also runs sequentially.

Invoke the skills by name in plain language. Codex CLI/IDE also supports
`$explaining-ml-codebases` and `$writing-ml-codebase-blogs`. If a newly installed
skill does not appear, restart the host or refresh its skills list.

## Release archive

```bash
python3 scripts/install_skills.py --archive dist/ml-codebase-skills.zip --skill all
```

Extract the two folders into your agent's skills directory. The blog renderer's
shared helpers are included. Tests, example repositories, evaluation outputs
and tool environments are not bundled. Releases do not require other skills;
STE or ELI5 helper skills are optional writing aids.

## Dependencies and fallbacks

| Requested output | Needed tools | If unavailable |
|---|---|---|
| Recon, docs checks, evidence, SVG figures, standalone HTML state player | Python 3.9+, standard library | Use another available compatible Python. Without one, report the helper/validation blocker; source reading may still proceed |
| Docs HTML or article HTML | Python; browser/network for CDN rendering libraries | Markdown, SVG and standalone players remain readable; do not claim a verified HTML render |
| Tests or measured examples from the source repo | That repo's dependencies, data and hardware | Retain source-based explanations and mark unexecuted tests `not run (needs X)` |
| Snapshot MP4 | Node.js, Playwright/browser, FFmpeg with libx264 | Deliver the standalone player and SVG keyframes; report the missing MP4 |
| Continuous/math MP4 | Manim Community, compatible Python, Cairo/Pango, FFmpeg; LaTeX/dvisvgm when needed | Use SVG panels/player output and report the unavailable animation backend |
| Jekyll post and asset staging | Python, a site profile | Staging works without Ruby/Jekyll; a missing profile must be created from site conventions |
| Local Jekyll preview | Target site's Ruby, Bundler, gems and build command | Deliver the staged post/assets and mark the site build unverified |

Core helpers do not require a pip requirements file. Installing the skills does
not install these optional tools. An explicit setup request authorizes the
agent to install needed packages within that request and its execution policy.
Prefer existing runtimes, then a separate tool venv or Node package directory;
keep source-repo test dependencies in a scratch repo environment. Do not change
the target repo's lockfiles or a shared Python environment to render figures.

System packages, browser downloads and large TeX installations depend on the
host's permissions, network and available package manager. If the request or
execution policy does not permit them, report that limit and use the fallback.
A permission prompt from the host may still apply. Avoid repeated installation
attempts with the same error; report the cause and finish the available output.

**Ask the agent to set up optional rendering:**

```text
Set up the tools needed to render this article's selected animations.
Reuse available runtimes, otherwise install only needed packages in an isolated
tool environment. Check platform prerequisites before choosing Manim or browser
export. Verify with a small render. If a dependency cannot be installed, keep
SVG figures and the standalone HTML player and report the missing video output.
```

For manual commands and upstream platform instructions, see
[visual-guide.md](visual-guide.md#optional-mp4-export) and the companion's
[animation guide](skills/writing-ml-codebase-blogs/references/animation.md).
Manim was tested with 0.22.0 (Python 3.11+). Scenes use original code; no GPU
or model weights are needed for visualization.

A missing optional tool should not fail the whole explanation. Invalid input,
a missing referenced asset or an unavailable mandatory runtime still fails the
relevant helper: repair the input or remove the unavailable media reference
and deliver a valid fallback. Never label a failed render or skipped test as
successful.
