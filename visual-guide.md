# Figures and state animations

Read this when planning a visual explanation. Visuals are part of ordinary
developer documentation; an article or blog is not required. Keep Markdown
and the shared running example authoritative.

## Decide automatically

Use `auto` unless the user requests static-only, animation, or a particular
tool/style. Identify the reader's difficult questions before drawing. There
is no quota for figures or videos.

| Reader needs to see | Choose |
|---|---|
| Components, boundaries, ownership | Mermaid architecture map or SVG swimlanes |
| Actual tokens, tensor layouts, memory pages, branches | Annotated SVG worked example |
| What an operation changes | Before/after panels with the changed values |
| Temporal order, overlap, recurring state transitions | HTML stepper; video when motion adds clarity or the user requests it |
| How changing an input changes a result | An interactive control backed by an explicit small model |
| A relationship already clear in prose | No additional visual |

Ask whether watching a transition teaches something that static panels would
make hard to follow. Animate queue ownership, allocation/reuse, or CPU/GPU
overlap when it helps. Do not make an MP4 just to highlight boxes. A stepper
with discrete snapshots is sufficient for many cache and scheduling examples.
Respect the requested output: Markdown-only readers get SVGs/keyframes and a
link to a player. An HTML request may include a player without a separate
request for animation. Explicit static-only requests exclude players/videos.

### Choose video for a mechanism

In `auto`, start with prose and figures. Select a video when following motion
or coupled state changes makes a specific difficult relationship easier to
understand. A repo being large, an article being long, or Manim being available
does not justify a clip. Naming a template is not an explanation plan.

Write the teaching objective and the advantage over a static figure before
rendering. For a math scene, make object correspondence visible: which entries
multiply, sum, get masked or normalize, and how the result relates to the next
operation. For an execution scene, show the actual changing inputs, outputs,
ownership and dependencies. Changing labels on a sequence of boxes is usually
better delivered as a figure or stepper.

Use complex workflows as candidate topics, not a mandatory video list:

| Topic | What a useful visual would expose |
|---|---|
| Attention, loss or another mathematical mechanism | One selected calculation, intermediate values, the geometric or numerical effect, and the shape/unit invariant |
| Model architecture | A concrete input through tensor layouts, branching/residual paths and parameter sharing; a static map often suffices unless the transformation itself is difficult |
| Paged KV and scheduling | Multiple request IDs, token budgets, logical-to-physical pages, allocation/refcounts, ready KV, admission and the next decision |
| A request from admission to completion | Input/output at tokenizer, queue, scheduler, worker, model and response boundaries; changes between prefill and decode; relevant cleanup/cancellation paths |
| Concurrent CPU/GPU work | Actor/stream lanes, multiple requests or iterations, host submission versus device completion, dependencies/waits and the source-supported overlap |
| Prefill/decode disaggregation | Selected producer/consumer processes, KV ownership, layout and transfer payload, send/receive/completion events, decode readiness and the implementation's failure path |

Read the selected backend and configuration before depicting concurrency or
disaggregation. A Python call returning does not prove device work completed;
a transfer submission does not prove the receiver can consume its data. Use
logical ordering without a performance scale when timings are not measured.
Make omitted branches explicit. A generic animation must not stand in for
explaining the chosen implementation.

The bundled token-flow, cache-reuse and attention scenes are rendering starters
with narrow examples. Use them only when they answer the article's question;
adapt the scene or author original code for other mechanisms. The three-template
helper does not automatically generate a complete scheduler, concurrent
execution or disaggregated-serving visualization. Advanced scene guidance is in
the companion's animation reference; a custom scene needs its own source and
semantic review, not just a successful MP4 export.

### Choose the duration

There is no demo-length cap. A complete explanation can run for several minutes
when the mechanism needs it. Choose duration from the teaching objective,
number of meaningful transitions and time needed to read labels or equations.
Respect any user time budget. Do not stretch a clip with repeated highlights.
For a long lifecycle or mathematical derivation, use chapters or separate clips
linked to the corresponding article sections. A short preview is optional and
does not replace the full explanation. The bundled scenes' duration is not a
recommended duration for production videos.

## Storyboard and evidence

Before drawing, write a brief for each selected visual in `.work/visuals.json`
(an array). Give it `id`, `question`, `doc`, `takeaway`, `kind`, `assets`,
`step_ids`, `source_symbols`, `assumptions`, and `evidence_status`. Use
`illustrative`, `fixture`, or `measured`; measured values need a command,
environment, and actual result. Save the final briefs in `evidence.json`.

For a video brief, also record `learning_objective`, `why_motion`, the scene's
ordered beats, and the meaningful input/output or state change at each beat.
Review the storyboard against a static alternative. If removing the motion
loses no useful relationship, omit the optional video or use a stepper.

State snapshots must share the prose's token IDs, budgets, tensor dimensions,
arrival ordering and configuration. Record before/after ownership and the
condition selecting each transition. Explain whether an arrow means data
transfer, a function call, a dependency, or time. Do not imply simultaneity
from side-by-side panels. Never confuse hash computation, block allocation,
cache-index registration, and completed KV computation. Check their order
and readiness rules in the selected repository version.

Animation durations are presentation choices. Label them illustrative unless
the visualization actually encodes measured timing. Represent synchronization
and CPU/GPU streams from source evidence, not from a generic engine model.
An example shipped with this skill is a design sample, not source evidence
about vLLM or any other repository.

## Design for reading

Prefer editable SVG for technical figures. Use Mermaid for structural maps.
Choose clean technical styling by default: white background, dark text,
consistent spacing, restrained accents, and short annotations. Preserve the
same role/color mapping across related figures. The included trace renderer
uses blue, green, amber, purple and gray; retain a text label for every role.
Use system/sans-serif fonts; use monospace for IDs and code. Package any
custom font and its license rather than using a machine-specific font path.

Design for the actual content width (typically 700-960 pixels). Aim for at
least 16-pixel body labels at displayed desktop size. Use short labels and
captions in the surrounding document instead of squeezing paragraphs into
boxes. Break dense scenes into panels or separate figures. Mobile figures
may use horizontal scrolling when preserving a meaningful token layout;
provide a nearby text explanation. Avoid full-architecture posters with
microscopic labels. Use numbers and labels as well as color.

Optional sketch styling can use Rough.js with restrained roughness and clean
text. Keep the diagram data/editable sources. Do not depend on ImageGen for
exact token IDs, formulas or source identifiers. Illustrations can be optional
decoration; they do not replace verified technical figures.

## Self-contained player and static figures

`scripts/render_trace.py` takes complete snapshots and generates a standalone
HTML player plus optional SVG frames. No Node, browser library, or Manim is
required for generation. The player has Previous/Next, a slider, Play/Pause,
and readable static states. It does not autoplay. JavaScript-free readers can
see the first state and expand all steps. Long sequences should be split.

Minimal trace contract (see `assets/prefix-cache-trace.json` for a full sample):

```json
{
  "title": "One request's cache lookup",
  "description": "Toy values; verify the target code before adapting.",
  "evidence_status": "illustrative",
  "frames": [
    {
      "title": "No reusable blocks yet",
      "caption": "Computing hashes does not compute KV tensors.",
      "duration": 3,
      "panels": [
        {"label": "Complete block", "tone": "amber", "tokens": [1, 2, 3, 4], "items": ["hash label: h1"]},
        {"label": "Cache lookup", "tone": "gray", "items": ["Ready cache: empty", "Reusable blocks: 0"]}
      ]
    }
  ]
}
```

Each frame needs a title, positive finite duration in seconds, and 1-3 panels.
Panels have a label, optional string `items`, optional short `tokens` (up to
four characters each), and a `tone`: blue, green, amber, purple, gray. Duration
controls presentation, not benchmark data. Adapt/extend SVG drawing code when
a tensor grid, dependency graph or true temporal motion better fits the topic.
The snapshot player is a starter, not a universal visualization engine.

```bash
python "$SKILL_DIR/scripts/render_trace.py" docs/explain/assets/cache-trace.json \
  -o docs/explain/assets/cache-player.html --frames-dir docs/explain/assets/cache-frames
```

Embed static frames and a link in Markdown. When the same source also produces
HTML, place raw HTML embeds inside an `explain-html-only` comment:

```markdown
![Cache state before KV computation](assets/cache-frames/frame-000.svg)

<!-- explain-html-only
<iframe src="assets/cache-player.html" title="Step through cache states"
        height="720" sandbox="allow-scripts" loading="lazy"></iframe>
-->

[Open the interactive explanation](assets/cache-player.html)
```

GitHub hides the comment; `render_html.py`, the article renderer and the Jekyll
adapter restore its HTML and package the referenced assets. Use pure HTML
inside the comment; keep Markdown captions and links outside it. Apply the same
wrapper to a video/figure embed in shared Markdown. Do not put a bare iframe
in repository Markdown: GitHub escapes it and displays the tag as text.

SVG keyframes are static on GitHub. The interactive SVG stepper runs in HTML.
For GitHub repository Markdown, include
enough static frames to explain the sequence, preferably in a collapsible
gallery for long traces. Keep a poster and an explicit Markdown MP4 link
outside `<video>` markup. Optional GIF previews show motion in repo Markdown;
label their illustrative timing and place looping previews in collapsed
details. Static-only requests exclude GIFs too. Opening an HTML file on GitHub
shows source; open it locally or publish a separately hosted page to run it.
See [GitHub's rendering restrictions](https://github.github.com/gfm/#disallowed-raw-html-extension-).

Iframe HTML must be standalone: inline its CSS, JavaScript, data and images.
The HTML renderer copies or embeds the file but does not recursively package
its dependencies. Use the provided generator for this contract. Scripts inside
ordinary Markdown are not initialized by `innerHTML`; use a standalone iframe
for custom players. Check the actual iframe height to avoid clipped controls.

## Optional MP4 export

For a finished video, `scripts/export_trace_video.mjs` renders the same states
at explicit timestamps with Playwright, then encodes an MP4 with FFmpeg.
It uses system fonts, fixed dimensions, no audio, and no autoplay. It refuses
to overwrite an existing export. It currently exports discrete state changes,
not interpolated movement; use a custom animation or Manim for continuous
motion. Generate a static poster/keyframes with the trace renderer.

Prerequisites: Node.js, Playwright with Chromium, and FFmpeg with libx264.
Prefer an available runtime. `NODE_PATH` can locate existing Node packages.
Pass `--browser-channel chrome` (or `msedge`) to reuse an installed browser
instead of downloading Playwright's Chromium. The browser runs headlessly
with a temporary profile, not the user's browsing session.
When installing optional dependencies is within the user's authorized scope,
use a separate tool environment rather than the target ML repository:

```bash
npm install --prefix /path/to/visual-tools playwright
/path/to/visual-tools/node_modules/.bin/playwright install chromium
NODE_PATH=/path/to/visual-tools/node_modules node "$SKILL_DIR/scripts/export_trace_video.mjs" \
  docs/explain/assets/cache-player.html -o docs/explain/assets/cache-trace.mp4
```

If tools are unavailable, finish with the HTML player and SVG keyframes and
report the video limitation. Do not download model weights or require a GPU
for visualization. Manim Community is an optional advanced backend; pin the
version and retain original scene sources. Its LaTeX-backed formulas need a
TeX environment. Do not copy scenes/assets from `3b1b/videos` as permissively
licensed templates: those contents use CC BY-NC-SA while Manim uses MIT.

For original continuous scenes, `scripts/render_manim.py` accepts bundled
`assets/manim-token-flow.json`, `manim-cache-reuse.json`, or
`manim-attention.json`. Use an isolated Manim Python environment:

```bash
uv venv /path/to/visual-tools --python 3.12
uv pip install --python /path/to/visual-tools/bin/python 'manim==0.22.0'
python "$SKILL_DIR/scripts/render_manim.py" "$SKILL_DIR/assets/manim-attention.json" \
  -o docs/explain/assets/attention.mp4 --python /path/to/visual-tools/bin/python
```

The attention template computes scores, causal masking, row softmax and
weighted values from tiny illustrative matrices. It does not measure a
model's hidden activations. Token-flow uses source-supported shapes; cache
reuse separates hash metadata, reserved pages, completed KV and subsequent
reuse. Adapt values and captions to the actual golden path. The renderer saves
a final poster, numerical metadata, JSON and editable scene code beside MP4.
Use `<video controls>` with the poster and a static fallback. In shared
Markdown, wrap this HTML block in the comment described above. Inspect keyframes
and actual playback when available; generation alone is not visual validation.

```html
<figure>
  <video controls playsinline preload="metadata" poster="assets/cache-frames/poster.svg">
    <source src="assets/cache-trace.mp4" type="video/mp4">
    <a href="assets/cache-trace.mp4">Open the cache trace video</a>
  </video>
  <figcaption>Illustrative states; durations are not measurements.</figcaption>
</figure>
```

## Render and inspect

Inspect SVGs at displayed desktop and mobile sizes. Fix overlapping labels,
small text, clipped arrows, ambiguous mappings, and missing units. Check the
player's first/middle/last states, Previous/Next, slider, replay, and static
fallback. Confirm every state agrees with the narrative and selected source.
For MP4, play it and inspect keyframes spanning the sequence; check its codec,
dimensions and duration with FFprobe when available.

Render the documentation HTML and verify actual media loads, including after
copying the export to another directory. Test iframe controls inside the docs
page. Missing assets are renderer errors. A file being generated does not
prove it displays correctly. If browser inspection is unavailable, state which
checks remain unverified and keep the usable static fallback.

Tool sources: [Rough.js](https://roughjs.com/),
[Playwright screenshots](https://playwright.dev/docs/screenshots),
[FFmpeg](https://ffmpeg.org/ffmpeg.html),
[Manim installation](https://docs.manim.community/en/stable/installation/uv.html),
[3b1b scene repository](https://github.com/3b1b/videos).
