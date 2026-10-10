# Third-party references and evaluation inputs

The skill's instructions, helper scripts and Manim scenes are original and
licensed under MIT. Dependencies keep their own licenses and are not vendored.
Generated rendering pages load marked, Mermaid, KaTeX and highlight.js from
pinned CDN URLs. Manim Community, Playwright and FFmpeg are optional external
tools; consult their upstream distribution terms when bundling these tools.

Teaching inspiration includes [Aleksa Gordić's vLLM article](https://www.aleksagordic.com/blog/vllm)
and [3b1b/videos](https://github.com/3b1b/videos). Their figures, scene code,
narration and branding are not included. 3b1b/videos uses CC BY-NC-SA; this
repository's MIT license does not relicense that material. Manim Community is
a separate MIT-licensed project.

[asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill) informed the
writing guidelines. No official ASD-STE100 dictionary or standard is bundled.

Evaluation articles quote short source excerpts and link to pinned upstream
commits. Their accompanying source notices are in `examples/UPSTREAM_LICENSES/`:
[nanoGPT](https://github.com/karpathy/nanoGPT) (MIT, Andrej Karpathy) and
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, vLLM contributors).
Those upstream excerpts retain their original terms. Source checkouts and
model weights are not distributed with skill packages.
