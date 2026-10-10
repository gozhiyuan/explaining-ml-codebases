# Optional GitHub Pages adaptation

Keep `article.md` platform-neutral. Use a profile only when the user requests
site output. `profiles/jekyll.json` is a starting point for a Jekyll post layout,
not a universal theme configuration. Inspect the target site's layout, front
matter, title rendering, math support and asset directories; write a matching
JSON profile. The starting profile uses `layout: post`, `.post-content` for
media styling, and site-relative image/video/interactive/data directories.

```bash
python scripts/format_jekyll.py article.md --profile profiles/jekyll.json \
  --slug topic-slug --date YYYY-MM-DD --site /path/to/site --out /path/to/new-stage
```

Set `--title`, `--subtitle` and `--categories` for the article. The adapter reads
the target site only to check its layout. It writes a new staging directory
with `_posts/`, content-addressed assets and `asset-manifest.json`; it refuses
to overwrite existing staging output. Copy these staged files into a separate
preview checkout of the site, then run its local Jekyll build. The user has
authorized local creation; publishing requires authorization for publication.

Images, MP4s, poster frames, standalone HTML players and evidence/measurement
files receive site-relative Liquid `relative_url` paths, including `baseurl`.
The profile's `content_selector` scopes media CSS to the article: videos fit
its width, iframes fill it, and tall worked figures retain their full height
when overriding a theme's height limit. Inspect the resulting mobile layout separately.
Selecting `math: katex` adds pinned KaTeX CDN loading; network access is required
for math. Avoid adding it when the theme already renders math. Verify the
renderer's `.post-content`/`.post` target exists in the chosen theme.
MP4 uses normal HTML `<video controls playsinline preload="metadata" poster="…">`
and an adjacent caption/static figure. Do not autoplay. Players must be
self-contained; external local dependencies are rejected. Provide static
fallback figures for iframe content. Adapt local Markdown cross-links to
verified site URLs or `{% post_url … %}` before staging: the adapter rejects
unresolved local `.md` links rather than silently creating broken navigation.

Code fences are protected from Liquid evaluation. Inline code containing Liquid
markers is protected too. For Markdown display math in Jekyll, use a raw HTML
`<div class="math-block">$$…$$</div>` so Kramdown does not turn backslashes into
escapes; validate the rendered equation text. Mermaid isn't automatically
provided by this adapter: export important diagrams to SVG, or configure the
site's renderer deliberately.

Verify built post paths, all asset links (including `baseurl`), code blocks,
math, captions and mobile figures. Treat local build, browser inspection and
publication as separate checks. Preserve unrelated site changes.
