---
name: writing-ml-codebase-blogs
description: Turn source-grounded ML or AI code explanations into illustrated technical articles or a coherent series. Use for blog drafts, narrative revisions, worked figures, and optional process or mathematical animation. Accept explanation Markdown and evidence, or inspect source directly. Produce platform-neutral Markdown and article HTML; apply a GitHub Pages profile only when requested.
---

# Illustrated ML codebase articles

Answer a reader's question through a concrete example, implementation decisions
and purposeful figures. Code at the recorded commit establishes behavior;
existing posts establish voice and site conventions. Neither an old article
nor an evidence ledger proves a new claim.

Infer audience and topic. Default to an engineer familiar with Python and
basic ML, new to this repo. A subsystem article answers one question; a series
gives each article a distinct job. Choose depth for that question instead of
transplanting every documentation section. For broad coverage, establish an
overview before opening narrower subsystems.

Resolve `SKILL_DIR` to this file's folder. Write to a separate article folder
unless editing an existing draft. Deliver `article.md`, `article.json`, and
editable visual sources under `assets/`; render `article.html` when requested.

Reuse available tools; core article helpers and SVG figures need only Python's
standard library. When setup is authorized, install only the needed optional
tools in an isolated environment, subject to the host's execution policy.
If video setup fails, deliver static figures or a standalone player and report
the missing clip. A missing Jekyll environment blocks its preview, not article
drafting or staging. Repair missing assets before rendering; report skipped
tests and unverified browser checks explicitly.

## Workflow

1. Read provided explanation docs and `evidence.json`. Check their commit,
   configuration, example and uncertainty against source. Without input docs,
   trace a representative source path and create a compact claim ledger.
   The explanation skill is useful but not a prerequisite.
2. Read [references/article-workflow.md](references/article-workflow.md). Record
   reader question, audience, source commit and scope. Plan the argument and
   running example. Introduce the problem, mental model and boundaries before
   detailed code. Move extensive test/reference tables into supporting docs.
3. Verify actual callers, ownership, branch conditions and downstream consumers.
   Reopen source for new or disputed claims. Keep uncertainty and test gaps.
4. Storyboard figures alongside the outline: question, takeaway, values,
   assumptions and source symbols. Use worked SVG panels for concrete values
   and Mermaid for structural maps. No visual quota is required.
5. Draft connected explanatory prose. Explain why the next section matters.
   Place short actual code excerpts at the decisions they explain. Label
   pseudocode explicitly. Preserve identifiers, units and qualification;
   STE sentence limits are advisory for article prose.
6. Render selected visuals from the same example. Read
   [references/animation.md](references/animation.md) when motion helps. Use
   controls, captions and static fallback frames. Do not imply CPU/GPU overlap
   or completed KV computation without source evidence. Durations are
   illustrative unless they encode measurements.
   In `auto`, justify each video by a learning objective and an advantage over
   static panels. Build it around actual transformations or interacting states;
   the bundled scenes are starters, not required article embellishments. Author
   a custom scene when the selected mechanism exceeds the templates.
7. Review source claims separately from readability. Run `scripts/check_article.py`,
   render with `scripts/render_article.py`, and inspect desktop/mobile layouts,
   figures, code, math, links and media. Symbol checks do not prove claims.
8. For GitHub Pages output, read [references/github-pages.md](references/github-pages.md).
   Stage adapted files and build a local preview. Publication is separate.

## Editorial constraints

- Cite files/symbols using commit-pinned source links. Distinguish source-reviewed,
  fixture, measured and illustrative information. Measured claims need actual
  commands/results. Preserve relevant uncertainty from explanation evidence.
- Carry one example through prose, objects/tensors, figures and animations.
  Identify abstractions affecting the conclusion. Hash registration and ready
  KV data can be different states; do not collapse them.
- Define unfamiliar terms where used. Keep names consistent without making
  every paragraph a chain of paths/classes. Avoid replies to the blog author.
- Captions state the view and takeaway. Place figures where they advance the
  argument. Keep series cross-links consistent with delivered posts.
- For GitHub repository Markdown, retain static trace frames, video posters
  and explicit Markdown MP4 links outside HTML media tags. Optional GIF previews
  can show motion; collapse looping previews and omit them for static-only
  requests. Iframe players run in local/hosted HTML, not GitHub's file view.
- Use original scenes and illustrations. Borrow teaching principles, not
  another author's artwork, narration, branding or restricted-license code.

## Helpers

`scripts/check_article.py ARTICLE.md --manifest article.json --repo SOURCE`
checks provenance, source references and local assets. Add `--evidence` when
using an explanation ledger. See the manifest contract in article-workflow.md.

`scripts/render_article.py ARTICLE.md -o article.html` uses this repository's
shared reading runtime. `install_skills.py` packages that runtime locally into
the companion, so rendering works independently after installation. Default
media mode embeds small images and copies players/videos beside HTML.
CDN libraries still need network access; this is a preview, not hosting.

Load a site profile only for that output. `profiles/jekyll.json` is a generic
starting point; inspect the target site's layout, front matter, math support
and asset conventions and adapt the profile before staging.
Report files, source commit, actual checks, visual inspection and uncertainty.
Distinguish article readiness, media rendering, local site build and publication.
