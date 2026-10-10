# Original mathematical and process scenes

Choose animation when watching a transformation advances the explanation.
The shared helpers offer a snapshot player and original Manim scenes. Installed
companions package helpers under `scripts/`; in this checkout use top-level
`scripts/`. No scenes, branding or narration from 3b1b are copied.

## Decide what the scene teaches

Default to figures. A selected video should explain a relationship that a
reader benefits from watching: a mathematical transformation, changing tensor
layout, competing requests, dependent state transitions or communication.
Record the learning objective and why motion improves it. Omit an optional
clip that only cycles labels or highlights an already clear block diagram.
An explicit video request still needs a purposeful storyboard.

Mathematical teaching techniques include keeping objects recognizable across
transformations, showing one contribution before combining many, and connecting
numbers to a geometric interpretation where appropriate. Manim supplies motion;
it does not supply that explanation or guarantee 3b1b-like teaching quality.

For repo-specific scenes, establish the selected commit, configuration and
backend, then build an ordered event ledger. Each beat identifies the actor,
request/tensor/block IDs, operation, input, output, owned state before/after,
dependency or wait, and supporting source symbol. Compute example values where
feasible; verify transitions and invariants independently of the animation.
Keep the same ledger in figures, prose and any player/video variants.

Useful serving questions include how a scheduler spends its budget across
requests; how logical token positions map to physical KV pages; how a request
moves from admission through prefill/decode and cleanup; where host submission,
GPU execution and result handling depend on one another; and when a remote
decode worker can use transferred KV. Inspect the actual implementation before
choosing lanes or order. Do not animate CPU/GPU overlap, a transport protocol,
KV readiness or a cancellation path from generic assumptions.

Use logical event ordering without a performance time scale unless a measured
trace establishes timing. Preserve request identity, page ownership, tensor
axes, units and branch conditions on screen. Show synchronization explicitly.
Separate long workflows into teaching segments rather than shrink every object
into an unreadable overview. Captions explain the consequence of each change.

## Duration and chapters

Production videos may be several minutes long; there is no fixed length limit.
Plan time for each calculation, transition and caption from the storyboard,
within the user's requested time budget. Prefer chapters or related clips for
long request lifecycles, concurrent execution or derivations. Keep stable visual
identities across chapters and link each segment beside the relevant article
section. Do not pad simple scenes to make them longer. A custom scene can have
as many teaching beats as needed; the short bundled demos are not length targets.

## Templates and custom scenes

The included scenes test rendering and small numerical examples. They are
starters, not a full mechanism library or a default requirement for articles.

Templates available through `render_manim.py`:

- `token-flow`: token cards transform through named tensor stages. This is
  conceptual shape/data flow, not literal movement of memory or unchanged IDs.
- `cache-reuse`: dependent block hashes, completed KV pages, then reuse links
  for another request. Qualify omitted source-specific allocation, registration,
  readiness and eviction bookkeeping; hashes can register before computation.
- `attention`: calculate tiny query/key scores, causal mask, row softmax and
  weighted values. Results are computed from the supplied illustrative inputs,
  not measured model activations unless their provenance establishes that.

Run `render_manim.py SPEC.json -o clip.mp4 --quality low --python TOOL_PYTHON`.
See bundled `assets/manim-*.json`. The wrapper uses a temporary media directory,
refuses to overwrite the clip, and writes a poster plus computed metadata.
It also writes an editable `.sources.zip` with the original scene, numerical
helper, renderer, specification and license. Link that archive when publishing
the clip so readers can reproduce it despite the site's renamed asset URLs.
It supports smooth transforms, reveals, arrows and mathematical objects.

For a mechanism beyond these templates, author an original Manim scene in the
article's assets and render it with Manim directly. `render_manim.py` accepts
only its three existing template schemas; renaming stages does not turn it
into a concurrency, scheduler or disaggregation simulator. For example:

```bash
/path/to/visual-tools/bin/python -m manim render assets/request_lifecycle.py \
  RequestLifecycle --renderer cairo --format mp4 --media_dir /path/to/new-media -ql
```

`RequestLifecycle` is the class the agent must author, not a bundled scene.
Retain the event ledger, original scene code, calculated values, source evidence,
poster and static explanatory frames alongside the clip. Inspect important
transitions for semantic correctness as well as legibility. A backend smoke
test does not establish the quality of a complete educational video.

Optional isolated setup:

```bash
uv venv /path/to/visual-tools --python 3.12
uv pip install --python /path/to/visual-tools/bin/python 'manim==0.22.0'
```

Manim's Cairo backend needs platform libraries from its installation guide.
The tested Manim0.22.0 requires Python 3.11 or newer.
The attention scene uses LaTeX; a GPU is not required. Use browser scenes or
static frames when the backend is unavailable and report the rendering limit.
Retain JSON, defining scene source, source provenance and assumptions. Inspect
representative frames and playback. Presentation timing is not a benchmark.

[Manim installation](https://docs.manim.community/en/stable/installation/uv.html)
and [3b1b scene licensing](https://github.com/3b1b/videos) are upstream references.
