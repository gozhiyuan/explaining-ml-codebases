#!/usr/bin/env python3
"""Render an explaining-ml-codebases doc set (markdown) into one HTML page.

The markdown stays the source of truth. The HTML is a generated reading view:
sidebar navigation across docs, per-doc contents, search, Mermaid diagrams,
KaTeX math (\\( \\), \\[ \\], $$ $$), highlighted code, wide-table scrolling,
and links between docs that work inside the page.

The page loads marked, Mermaid, KaTeX, and highlight.js from a CDN, so the
browser needs network access the first time. Stdlib only, Python 3.9+.

Small local images are embedded by default. Videos, standalone HTML players,
and large images are copied into OUT_STEM.assets beside the output page.
Share that folder with the HTML. --media-mode inline embeds all media, up to
--max-inline-mb per asset; --media-mode bundle copies all media.
  - relative media paths are resolved from the originating markdown file
  - links to repo files outside the doc set point at the git host (Bitbucket,
    GitHub, GitLab) at the explained commit, so they work wherever the page is
    hosted. Use --link-base for other hosts.

Usage:
    render_html.py DOCS_DIR [-o OUT.html] [--title TITLE] [--link-base URL] [--no-embed-images]
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html as html_lib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urlsplit

CDN = {
    "marked": "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js",
    "mermaid": "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js",
    "katex_js": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
    "katex_css": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
    "hljs_js": "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js",
    "hljs_css": "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css",
}

LINK_RE = re.compile(r"\]\(([\w./-]+\.md)(?:#[^)]*)?\)")
H1_RE = re.compile(r"^#\s+(.+)$", re.M)
IMG_MD_RE = re.compile(r'!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(\s+"[^"]*")?\s*\)')
MEDIA_TAG_RE = re.compile(r'<(?:img|video|audio|source|iframe|a)\b[^>]*>', re.I | re.S)
CODE_SPAN_RE = re.compile(r'(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)')
MEDIA_SUFFIXES = {'.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif',
                  '.mp4', '.webm', '.ogg', '.mp3', '.wav', '.html', '.htm',
                  '.json', '.csv', '.txt', '.log', '.zip'}
LINK_MD_RE = re.compile(r'(?<!!)\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(\s+"[^"]*")?\s*\)')
FENCE_RE = re.compile(r"^\s*(```|~~~)")
COMMIT_RE = re.compile(r"Commit:\s*`([0-9a-f]{7,40})`")
REMOTE_RE = re.compile(
    r"^(?:git@|ssh://git@|https?://(?:[^@/]+@)?)([^/:]+)[:/](.+?)(?:\.git)?/?$"
)
EXTERNAL = ("http:", "https:", "mailto:", "data:", "#", "//", "blob:")
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def git(cwd: Path, *args: str) -> str:
    try:
        res = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def host_link_base(docs_dir: Path, commit: str) -> str:
    """URL prefix for a repo file at `commit`, built from the origin remote."""
    m = REMOTE_RE.match(git(docs_dir, "remote", "get-url", "origin"))
    if not m or not commit:
        return ""
    host, path = m.group(1).lower(), m.group(2)
    if "bitbucket" in host:
        return f"https://{host}/{path}/src/{commit}/"
    if "gitlab" in host:
        return f"https://{host}/{path}/-/blob/{commit}/"
    if "github" in host:
        return f"https://{host}/{path}/blob/{commit}/"
    return ""


def explained_commit(docs_dir: Path, index_text: str) -> str:
    """The commit the docs explain (index.md `Commit:` line), else HEAD."""
    m = COMMIT_RE.search(index_text)
    ref = m.group(1) if m else "HEAD"
    return git(docs_dir, "rev-parse", ref) or (m.group(1) if m else "")


class TagAttributes(HTMLParser):
    def handle_starttag(self, tag, attrs):
        self.tag, self.attrs = tag, dict(attrs)


def localize(md: str, doc_dir: Path, ctx: dict) -> str:
    """Package media and point out-of-set source links at the git host."""

    def package(src: str, kind: str):
        if src.startswith(EXTERNAL):
            return None
        parts = urlsplit(src)
        if parts.scheme:
            return None
        path = (doc_dir / unquote(parts.path)).resolve()
        if not path.is_file():
            ctx["missing_media"].append(src)
            return None
        size = path.stat().st_size
        mode = ctx["media_mode"]
        inline = mode == "inline" or (mode == "auto" and kind == "image"
                                       and ctx["embed_images"] and size <= MAX_IMAGE_BYTES)
        if inline and size > ctx["max_inline_bytes"]:
            ctx["oversized_media"].append(f"{src} ({size // 1024} KB)")
            return None
        key = (path, inline)
        if key in ctx["media_cache"]:
            return ctx["media_cache"][key] + ("#" + parts.fragment if parts.fragment else "")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if inline:
            uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
            ctx["embedded_media"] += 1
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            name = re.sub(r'[^\w.-]', '-', path.stem)[:60] + '-' + digest + path.suffix.lower()
            ctx["asset_dir"].mkdir(parents=True, exist_ok=True)
            dest = ctx["asset_dir"] / name
            if dest.resolve() != path:
                shutil.copyfile(path, dest)
            uri = quote(ctx["asset_dir"].name) + '/' + quote(name)
            ctx["copied_media"] += 1
        ctx["media_cache"][key] = uri
        return uri + ("#" + parts.fragment if parts.fragment else "")

    def img_md(m):
        uri = package(m.group(2), "image")
        return f"![{m.group(1)}]({uri}{m.group(3) or ''})" if uri else m.group(0)

    def media_html(m):
        tag = m.group(0)
        parser = TagAttributes()
        parser.feed(tag)
        if not hasattr(parser, 'attrs'):
            return tag
        names = ['href'] if parser.tag == 'a' else ['src']
        if parser.tag == 'video':
            names.append('poster')
        for attr in names:
            src = parser.attrs.get(attr)
            if not src:
                continue
            if (parser.tag == 'a' and Path(urlsplit(src).path).suffix.lower() not in MEDIA_SUFFIXES
                    and Path(urlsplit(src).path).name != 'evidence.json'):
                continue
            kind = 'image' if parser.tag == 'img' or attr == 'poster' else 'media'
            uri = package(src, kind)
            if uri:
                pattern = re.compile(r'((?<![\w:-])' + attr + r'\s*=\s*)(?:"[^"]*"|\x27[^\x27]*\x27|[^\s>]+)', re.I)
                tag = pattern.sub(lambda a: a.group(1) + '"' + html_lib.escape(uri, quote=True) + '"', tag, count=1)
        return tag

    def link(m):
        target = m.group(2)
        if target.startswith(EXTERNAL):
            return m.group(0)
        path_part, _, anchor = target.partition("#")
        resolved = (doc_dir / unquote(urlsplit(target).path)).resolve()
        artifact = resolved.is_relative_to(ctx['docs_dir']) and resolved.suffix.lower() in {'.py', '.yaml', '.yml', '.ipynb'}
        if Path(urlsplit(target).path).suffix.lower() in MEDIA_SUFFIXES or artifact:
            uri = package(target, "media")
            return f"[{m.group(1)}]({uri}{m.group(3) or ''})" if uri else m.group(0)
        resolved = (doc_dir / unquote(path_part)).resolve()
        if resolved.parent == ctx["docs_dir"] and resolved.name in ctx["doc_names"]:
            return m.group(0)  # another doc in this set: the page handles it
        if not resolved.exists():
            ctx["broken_links"].append(target)
            return m.group(0)
        root, base = ctx["repo_root"], ctx["link_base"]
        if not (root and base):
            ctx["local_links"].append(target)
            return m.group(0)
        try:
            rel = resolved.relative_to(root).as_posix()
        except ValueError:
            ctx["local_links"].append(target)
            return m.group(0)
        if anchor and rel.endswith(".md"):
            if "bitbucket" in base and not anchor.startswith("markdown-header-"):
                anchor = "markdown-header-" + anchor  # Bitbucket's heading id prefix
            url = f"{base}{rel}#{anchor}"
        else:
            url = base + rel
        ctx["host_links"] += 1
        return f"[{m.group(1)}]({url}{m.group(3) or ''})"

    out, prose, in_fence = [], [], False
    def flush():
        text = '\n'.join(prose)
        # Process whole prose blocks so multiline <video>/<iframe> tags work.
        def rewrite(chunk):
            chunk = IMG_MD_RE.sub(img_md, chunk)
            chunk = MEDIA_TAG_RE.sub(media_html, chunk)
            return LINK_MD_RE.sub(link, chunk)
        chunks, cursor = [], 0
        for span in CODE_SPAN_RE.finditer(text):
            chunks.extend([rewrite(text[cursor:span.start()]), span.group(0)])
            cursor = span.end()
        chunks.append(rewrite(text[cursor:]))
        text = ''.join(chunks)
        out.extend(text.split('\n') if prose else [])
        prose.clear()
    for line in md.split("\n"):
        if FENCE_RE.match(line):
            flush()
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            prose.append(line)
    flush()
    return "\n".join(out)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="__KATEX_CSS__">
<link rel="stylesheet" href="__HLJS_CSS__">
<style>
:root { --side: 300px; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --bg2: #f6f8fa; --accent: #0969da; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg); }
#side { position: fixed; top: 0; left: 0; bottom: 0; width: var(--side); overflow-y: auto; border-right: 1px solid var(--line); background: var(--bg2); padding: 14px 12px 40px; }
#side .brand { font-size: 16px; font-weight: 700; margin: 0 0 10px; }
#search { width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; margin-bottom: 10px; font-size: 14px; }
#nav a { display: block; color: var(--fg); text-decoration: none; padding: 2px 6px; border-radius: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#nav a:hover { background: #eaeef2; }
#nav a.doc { font-weight: 600; margin-top: 6px; }
#nav a.doc.active { background: #ddf4ff; color: var(--accent); }
#nav a.h2 { padding-left: 16px; font-size: 13px; }
#nav a.h3 { padding-left: 30px; font-size: 12.5px; color: var(--muted); }
#nav .hit small { color: var(--muted); }
#main { margin-left: var(--side); padding: 24px 48px 80px; max-width: 1180px; }
#content h1 { font-size: 28px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
#content h2 { font-size: 22px; border-bottom: 1px solid var(--line); padding-bottom: 4px; margin-top: 36px; }
#content h3 { font-size: 18px; margin-top: 28px; }
#content h1, #content h2, #content h3, #content h4 { scroll-margin-top: 12px; }
#content code { background: var(--bg2); padding: 1px 5px; border-radius: 4px; font-size: 85%; }
#content pre { position: relative; background: var(--bg2); padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
#content pre code { background: none; padding: 0; font-size: 13px; line-height: 1.45; }
#content pre .copy { position: absolute; top: 6px; right: 6px; font-size: 12px; padding: 2px 8px; border: 1px solid var(--line); background: #fff; border-radius: 4px; cursor: pointer; opacity: 0; }
#content pre:hover .copy { opacity: 1; }
.table-wrap { overflow-x: auto; margin: 12px 0; }
#content table { border-collapse: collapse; font-size: 13.5px; }
#content th, #content td { border: 1px solid var(--line); padding: 5px 9px; vertical-align: top; }
#content th { background: var(--bg2); }
#content blockquote { margin: 0; padding: 0 14px; color: var(--muted); border-left: 4px solid var(--line); }
#content img, #content video { display: block; max-width: 100%; height: auto; margin: 16px auto; }
#content figure { margin: 24px 0; }
#content figcaption { color: var(--muted); font-size: 14px; margin-top: 8px; }
#content iframe { width: 100%; border: 1px solid var(--line); border-radius: 8px; }
.mermaid { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 10px; text-align: center; overflow-x: auto; }
.mermaid-error { color: #cf222e; white-space: pre-wrap; font-family: monospace; font-size: 12px; }
.mermaid.mermaid-failed { border-color: #cf222e; text-align: left; }
.math-block { overflow-x: auto; margin: 12px 0; }
#pager { display: flex; justify-content: space-between; margin-top: 48px; border-top: 1px solid var(--line); padding-top: 14px; }
#pager a { color: var(--accent); text-decoration: none; }
#foot { color: var(--muted); font-size: 12px; margin-top: 24px; }
@media (max-width: 900px) { #side { position: static; width: auto; height: auto; } #main { margin-left: 0; padding: 16px; } }
@media print { #side, #pager { display: none; } #main { margin: 0; } }
body.article { font-size: 18px; line-height: 1.8; }
body.article #side, body.article #pager { display: none; }
body.article #main { margin: 0 auto; max-width: 900px; padding: 64px 40px 80px; }
body.article #content h1 { font-size: clamp(30px, 4vw, 44px); line-height: 1.2; border: 0; letter-spacing: -0.025em; }
body.article #content h2 { margin-top: 48px; font-size: 28px; border: 0; }
body.article #content h3 { font-size: 22px; }
body.article #content pre code { font-size: 14px; }
body.article #content figcaption { line-height: 1.6; }
@media(max-width:600px) { body.article #main { padding: 24px 16px 48px; } body.article { font-size: 17px; } }
</style>
</head>
<body __LAYOUT__>
<nav id="side">
  <div class="brand">__TITLE__</div>
  <input id="search" type="search" placeholder="Search headings and text">
  <div id="nav"></div>
</nav>
<main id="main">
  <article id="content"></article>
  <div id="pager"></div>
  <div id="foot">Generated __WHEN__ from the markdown in this folder. Edit the .md files, not this page.</div>
</main>
<script id="docs-data" type="application/json">__DOCS__</script>
<script src="__MARKED__"></script>
<script src="__KATEX_JS__"></script>
<script src="__HLJS_JS__"></script>
<script src="__MERMAID__"></script>
<script>
(function () {
  const DOCS = JSON.parse(document.getElementById("docs-data").textContent);
  const NAMES = DOCS.map(d => d.name);
  const esc = s => s.replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const plain = s => s.replace(/`/g, "").replace(/\*\*|__/g, "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");

  // GitHub-style slugs, so links written for GitHub/Cursor preview resolve here.
  function makeSlugger() {
    const seen = {};
    return function (text) {
      let s = plain(text).trim().toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu, "").replace(/ /g, "-");
      if (seen[s] !== undefined) { seen[s] += 1; s = s + "-" + seen[s]; } else { seen[s] = 0; }
      return s;
    };
  }

  function headingsOf(md) {
    const slug = makeSlugger(), out = [];
    for (const t of marked.lexer(md)) {
      if (t.type === "heading" && t.depth <= 3) out.push({depth: t.depth, text: plain(t.text), id: slug(t.text)});
    }
    return out;
  }
  DOCS.forEach(d => { d.headings = headingsOf(d.md); });

  function katexRender(tex, display) {
    try { return katex.renderToString(tex, {displayMode: display, throwOnError: false}); }
    catch (e) { return "<code>" + esc(tex) + "</code>"; }
  }
  const mathBlock = {
    name: "mathBlock", level: "block",
    start(src) { const m = src.match(/(^|\n)[ \t]*(\\\[|\$\$)/); return m ? m.index + m[1].length : undefined; },
    tokenizer(src) {
      const m = /^[ \t]*(?:\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$)[ \t]*(?:\n|$)/.exec(src);
      if (m) return {type: "mathBlock", raw: m[0], text: (m[1] || m[2]).trim()};
    },
    renderer(t) { return '<div class="math-block">' + katexRender(t.text, true) + "</div>"; }
  };
  const mathInline = {
    name: "mathInline", level: "inline",
    start(src) { const m = src.match(/\\\(|\\\[/); return m ? m.index : undefined; },
    tokenizer(src) {
      let m = /^\\\(([\s\S]+?)\\\)/.exec(src);
      if (m) return {type: "mathInline", raw: m[0], text: m[1], display: false};
      m = /^\\\[([\s\S]+?)\\\]/.exec(src);
      if (m) return {type: "mathInline", raw: m[0], text: m[1], display: true};
    },
    renderer(t) { return katexRender(t.text, t.display); }
  };

  let slug = makeSlugger();
  const renderer = {
    heading(text, level, raw) {
      const id = slug(raw);
      return "<h" + level + ' id="' + id + '">' + text + "</h" + level + ">\n";
    },
    code(code, info) {
      const lang = (info || "").trim().split(/\s+/)[0];
      if (lang === "mermaid") return '<div class="mermaid">' + esc(code) + "</div>";
      let html;
      if (lang && hljs.getLanguage(lang)) html = hljs.highlight(code, {language: lang}).value;
      else html = esc(code);
      return '<pre><button class="copy">copy</button><code class="hljs">' + html + "</code></pre>";
    },
    table(header, body) {
      return '<div class="table-wrap"><table><thead>' + header + "</thead><tbody>" + body + "</tbody></table></div>";
    },
    link(href, title, text) {
      const m = /^(?:\.\/)?([\w.\-\/]+\.md)(#.*)?$/.exec(href || "");
      if (m) {
        const name = m[1].split("/").pop();
        if (NAMES.includes(name)) href = "#/" + name + (m[2] || "");
      }
      const ext = /^https?:/.test(href || "") ? ' target="_blank" rel="noopener"' : "";
      return '<a href="' + esc(href || "") + '"' + (title ? ' title="' + esc(title) + '"' : "") + ext + ">" + text + "</a>";
    }
  };
  marked.use({gfm: true, extensions: [mathBlock, mathInline], renderer});
  mermaid.initialize({startOnLoad: false, securityLevel: "strict"});

  // One diagram at a time, so a broken diagram shows its own error and source
  // in place and the others still render. window.explainDiagramErrors and the
  // page title report the count, for a quick check after rendering.
  let renderSeq = 0;
  async function renderDiagrams(el, doc) {
    const nodes = [...el.querySelectorAll(".mermaid")];
    let errors = 0;
    for (const n of nodes) {
      const src = n.textContent, id = "mmd-" + (++renderSeq);
      try {
        const out = await mermaid.render(id, src);
        n.innerHTML = out.svg;
      } catch (err) {
        errors += 1;
        const stray = document.getElementById("d" + id) || document.getElementById(id);
        if (stray && !el.contains(stray)) stray.remove();
        n.classList.add("mermaid-failed");
        n.innerHTML = '<div class="mermaid-error">Diagram error: ' +
          esc(String(err && err.message ? err.message : err)) + "</div><pre>" + esc(src) + "</pre>";
      }
    }
    window.explainDiagramErrors = errors;
    document.title = (errors ? "[" + errors + " diagram error" + (errors > 1 ? "s" : "") + "] " : "") + doc.title;
  }

  function tocMarkdown(doc) {
    return doc.headings.filter(h => h.depth >= 2).map(h =>
      "  ".repeat(h.depth - 2) + "- [" + h.text.replace(/[\[\]]/g, "") + "](#/" + doc.name + "#" + h.id + ")").join("\n");
  }

  function render(name, anchor) {
    const doc = DOCS.find(d => d.name === name) || DOCS[0];
    let md = doc.md.replace(/^\s*(\[TOC\]|\[\[_TOC_\]\])\s*$/gm, () => tocMarkdown(doc));
    slug = makeSlugger();
    const el = document.getElementById("content");
    el.innerHTML = marked.parse(md);
    el.querySelectorAll("pre .copy").forEach(b => b.addEventListener("click", () => {
      navigator.clipboard.writeText(b.nextElementSibling.innerText);
      b.textContent = "copied"; setTimeout(() => { b.textContent = "copy"; }, 1200);
    }));
    renderDiagrams(el, doc);
    const i = DOCS.indexOf(doc), prev = DOCS[i - 1], next = DOCS[i + 1];
    document.getElementById("pager").innerHTML =
      (prev ? '<a href="#/' + prev.name + '">&larr; ' + esc(prev.title) + "</a>" : "<span></span>") +
      (next ? '<a href="#/' + next.name + '">' + esc(next.title) + " &rarr;</a>" : "<span></span>");
    if (!el.querySelector(".mermaid")) document.title = doc.title;
    buildNav(doc.name, "");
    if (anchor) { const t = document.getElementById(decodeURIComponent(anchor)); if (t) { t.scrollIntoView(); return; } }
    window.scrollTo(0, 0);
  }

  function buildNav(active, query) {
    const nav = document.getElementById("nav");
    const q = query.trim().toLowerCase();
    let html = "";
    if (q) {
      for (const d of DOCS) {
        for (const h of d.headings) if (h.text.toLowerCase().includes(q))
          html += '<a class="h2 hit" href="#/' + d.name + "#" + h.id + '">' + esc(h.text) + " <small>" + esc(d.name) + "</small></a>";
        const n = d.md.toLowerCase().split(q).length - 1;
        if (n) html += '<a class="doc" href="#/' + d.name + '">' + esc(d.title) + " <small>(" + n + " text matches)</small></a>";
      }
      nav.innerHTML = html || "<small>No matches</small>";
      return;
    }
    for (const d of DOCS) {
      html += '<a class="doc' + (d.name === active ? " active" : "") + '" href="#/' + d.name + '">' + esc(d.title) + "</a>";
      if (d.name === active)
        for (const h of d.headings) if (h.depth >= 2)
          html += '<a class="h' + h.depth + '" href="#/' + d.name + "#" + h.id + '" title="' + esc(h.text) + '">' + esc(h.text) + "</a>";
    }
    nav.innerHTML = html;
  }

  function route() {
    const h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
    const cut = h.indexOf("#");
    const name = cut < 0 ? h : h.slice(0, cut), anchor = cut < 0 ? "" : h.slice(cut + 1);
    render(name || DOCS[0].name, anchor);
  }
  document.getElementById("search").addEventListener("input", e => {
    const cur = (location.hash.replace(/^#\/?/, "").split("#")[0]) || DOCS[0].name;
    buildNav(cur, e.target.value);
  });
  window.addEventListener("hashchange", route);
  route();
})();
</script>
</body>
</html>
"""


def order_docs(docs_dir: Path):
    """index.md first, then the order of the Doc map table, then the rest."""
    files = sorted(p for p in docs_dir.glob("*.md") if not p.name.startswith("."))
    names = [p.name for p in files]
    order = []
    if "index.md" in names:
        order.append("index.md")
        text = (docs_dir / "index.md").read_text(errors="ignore")
        m = re.search(r"^##\s+Doc map.*?$(.*?)(?=^##\s)", text, re.M | re.S)
        for source in ([m.group(1)] if m else []) + [text]:
            for link in LINK_RE.finditer(source):
                n = link.group(1).split("/")[-1]
                if n in names and n not in order:
                    order.append(n)
    order += [n for n in names if n not in order]
    return [docs_dir / n for n in order]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("docs_dir")
    ap.add_argument("-o", "--out", help="output file (default: DOCS_DIR/explain.html)")
    ap.add_argument("--title", help="page title (default: the H1 of index.md)")
    ap.add_argument(
        "--link-base",
        help="URL prefix for repo files outside the doc set "
        "(default: built from the origin remote and the explained commit)",
    )
    ap.add_argument(
        "--no-embed-images", action="store_true", help="copy images alongside HTML in auto mode"
    )
    ap.add_argument('--media-mode', choices=['auto', 'bundle', 'inline'], default='auto',
                    help='auto: embed small images, copy other media; bundle: copy all; inline: embed all')
    ap.add_argument('--max-inline-mb', type=float, default=20,
                    help='maximum size per inline asset (default: 20 MiB)')
    ap.add_argument('--layout', choices=['docs', 'article'], default='docs')
    ap.add_argument('--entry', help='render only this Markdown filename in DOCS_DIR')
    args = ap.parse_args(argv)
    if args.max_inline_mb <= 0:
        ap.error('--max-inline-mb must be positive')
    if args.no_embed_images and args.media_mode == 'inline':
        ap.error('--no-embed-images conflicts with --media-mode inline')

    docs_dir = Path(args.docs_dir).resolve()
    paths = order_docs(docs_dir)
    if args.entry:
        if Path(args.entry).name != args.entry or not args.entry.endswith('.md'):
            ap.error('--entry must be a Markdown filename, without directory components')
        paths = [p for p in paths if p.name == args.entry]
    if not paths:
        print(f"no .md files in {docs_dir}", file=sys.stderr)
        return 2
    index_text = (
        (docs_dir / "index.md").read_text(errors="ignore")
        if (docs_dir / "index.md").is_file()
        else ""
    )
    root = git(docs_dir, "rev-parse", "--show-toplevel")
    commit = explained_commit(docs_dir, index_text) if root else ""
    out = (Path(args.out) if args.out else docs_dir / "explain.html").resolve()
    ctx = {
        "docs_dir": docs_dir,
        "doc_names": {p.name for p in paths},
        "repo_root": Path(root).resolve() if root else None,
        "link_base": args.link_base
        or (host_link_base(docs_dir, commit) if root else ""),
        "embed_images": not args.no_embed_images,
        "media_mode": args.media_mode,
        "max_inline_bytes": args.max_inline_mb * 1024 * 1024,
        "asset_dir": out.parent / (out.stem + '.assets'),
        "media_cache": {},
        "embedded_media": 0,
        "copied_media": 0,
        "host_links": 0,
        "missing_media": [],
        "oversized_media": [],
        "broken_links": [],
        "local_links": [],
    }
    if ctx["link_base"] and not ctx["link_base"].endswith("/"):
        ctx["link_base"] += "/"
    docs = []
    for p in paths:
        md = localize(p.read_text(errors="ignore"), p.parent, ctx)
        h1 = H1_RE.search(md)
        docs.append(
            {
                "name": p.name,
                "title": (h1.group(1).replace("`", "").strip() if h1 else p.stem),
                "md": md,
            }
        )
    title = args.title or docs[0]["title"]
    if ctx['missing_media'] or ctx['oversized_media']:
        for src in sorted(set(ctx['missing_media'])):
            print(f'ERROR media not found: {src}', file=sys.stderr)
        for src in sorted(set(ctx['oversized_media'])):
            print(f'ERROR media exceeds --max-inline-mb: {src}; use auto or bundle mode', file=sys.stderr)
        return 1

    payload = json.dumps(docs, ensure_ascii=False).replace("</", "<\\/")
    html = (
        TEMPLATE.replace("__TITLE__", title.replace("<", "&lt;"))
        .replace('__LAYOUT__', 'class="article"' if args.layout == 'article' else '')
        .replace("__WHEN__", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__MARKED__", CDN["marked"])
        .replace("__MERMAID__", CDN["mermaid"])
        .replace("__KATEX_JS__", CDN["katex_js"])
        .replace("__KATEX_CSS__", CDN["katex_css"])
        .replace("__HLJS_JS__", CDN["hljs_js"])
        .replace("__HLJS_CSS__", CDN["hljs_css"])
        .replace("__DOCS__", payload)  # last, so doc text is never substituted
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(docs)} docs: {', '.join(d['name'] for d in docs)})")
    print(f"media embedded: {ctx['embedded_media']}; copied: {ctx['copied_media']}")
    if ctx['copied_media']:
        print(f"share the HTML with: {ctx['asset_dir']}")
    if ctx["link_base"]:
        print(f"links to repo files -> {ctx['link_base']}: {ctx['host_links']}")
    for label, key in (
        ("link target not found", "broken_links"),
        (
            "link kept as a local path (no git host URL; breaks when hosted elsewhere)",
            "local_links",
        ),
    ):
        for item in sorted(set(ctx[key])):
            print(f"WARN {label}: {item}")
    if ctx["host_links"]:
        print("note: host links point at the explained commit. Push it, or they 404.")
    print(f"open it: open '{out.resolve()}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
