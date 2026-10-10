# Source-grounded examples

These are actual skill evaluations, not generic templates.

| Repository | Walkthrough | Blog article | Scope |
|---|---|---|---|
| nanoGPT | [index](nanogpt/index.md) | [article](nanogpt/blog/article.md) | One tiny CPU update |
| vLLM | [index](vllm/index.md) | [article](vllm/blog/article.md) | Scheduler budget and prefix KV ownership |

Each folder retains evidence, editable SVG/JSON figures, a standalone player,
source checks and HTML. Blog assets include optional original Manim videos,
posters, numerical metadata and defining scenes. Static figures remain useful
without video tools. Titles, captions and provenance describe the simplified
fixture rather than imply real latency or model outputs.

The three short Manim clips are rendering demos for narrow examples. They do
not demonstrate a complete scheduler, CPU/GPU concurrency, disaggregated serving
or a polished long-form mathematical explanation. An actual article should use
a video only when its storyboard teaches the selected mechanism better than
figures or a stepper; it need not include these demo clips.

## View the animations

GitHub's repository Markdown view does not run the iframe players or the
articles' repository-relative HTML video embeds. Each article includes a
complete static trace gallery, video posters, animated GIF previews and direct
MP4 links. Expand a GIF preview to see motion in the Markdown page; use an MP4
for the full-resolution clip. GIF previews loop automatically when visible.

| Example | Generated video | Duration |
|---|---|---|
| nanoGPT | [Token/tensor flow MP4](nanogpt/blog/assets/token-flow.mp4?raw=1) · [GIF preview](nanogpt/blog/assets/token-flow.preview.gif) | 14.4 s |
| nanoGPT | [Illustrative attention calculation MP4](nanogpt/blog/assets/attention.mp4?raw=1) · [GIF preview](nanogpt/blog/assets/attention.preview.gif) | 15.9 s |
| vLLM | [Illustrative cache lifecycle MP4](vllm/blog/assets/cache-reuse.mp4?raw=1) · [GIF preview](vllm/blog/assets/cache-reuse.preview.gif) | 13.7 s |

For the interactive reading view, clone the repository and open
`examples/nanogpt/blog/article.html` or `examples/vllm/blog/article.html` in a
browser. Keep each page's adjacent `article.assets/` folder beside it. The
standalone `assets/update-player.html` and `assets/cache-player.html` also open
locally and do not need CDN libraries. A GitHub file link to HTML displays its
source; it does not host a running page. GitHub Pages hosting is a separate step.

MP4 encoding and GIF frames have been checked; interactive controls, responsive
layout and browser playback remain unverified in this evaluation environment.

See [evaluation instructions](../evals/README.md) for pinned commits, actual
commands and gaps. The HTML plus its adjacent `.assets/` folder is portable;
CDN rendering libraries require network access. Markdown source remains editable.
Generated asset URLs contain content hashes, so regenerate HTML after edits.

Upstream source excerpts retain the licenses in [UPSTREAM_LICENSES](UPSTREAM_LICENSES/).
Original scenes and skill code use the repository's MIT license.
