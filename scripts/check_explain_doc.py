#!/usr/bin/env python3
"""Validate explanation docs written with the explaining-ml-codebases skill.

Checks, against the explained repo:
  - code references (`path`, `path:line`, `path:start-end`, `path:symbol`) exist
    and line numbers are in range; call-tree comments in code blocks too
  - `pytest path::test` targets exist
  - relative markdown links resolve
  - numbered sections contain Input, Output, and Tests
  - index.md has the required headings for the mode
  - Tests-table status values use the allowed words
  - mermaid blocks start with a diagram type; unquoted labels with brackets
  - leftover template placeholders, TODO/TBD, unclosed code fences

Stdlib only, Python 3.9+. Exit 1 when any ERROR is found.

Usage:
    check_explain_doc.py DOCS_DIR_OR_FILES... --repo REPO [--mode deep|simple] [--json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "site-packages",
    ".eggs",
}
CODE_EXT = (
    "py|pyx|pyi|cu|cuh|cc|cpp|cxx|h|hpp|c|rs|go|java|scala|kt|ts|tsx|js|sh|"
    "yaml|yml|toml|json|cfg|ini|pbtxt|txt|ipynb|sql|proto|Dockerfile|mk"
)
# A missing file with one of these extensions is an error. Other extensions
# (json, yaml, txt...) are often runtime artifacts, so they only warn.
SOURCE_EXT = {
    "py",
    "pyx",
    "pyi",
    "cu",
    "cuh",
    "cc",
    "cpp",
    "cxx",
    "h",
    "hpp",
    "c",
    "rs",
    "go",
    "java",
    "scala",
    "kt",
    "ts",
    "tsx",
    "js",
    "sh",
    "proto",
    "sql",
    "ipynb",
}
REF_RE = re.compile(
    rf"^(?P<path>[\w.\-/]+\.(?:{CODE_EXT}))"
    r"(?::(?:(?P<l1>\d+)(?:-(?P<l2>\d+))?|:?(?P<sym>[A-Za-z_][\w.]*)))?$"
)
# "`make_model` in `training.py`", "`epochs` in `configs/_defaults.yaml`"
SYM_IN_FILE_RE = re.compile(
    r"`([A-Za-z_][\w.]*)(?:\(\))?`\s+(?:in|from|of)\s+`([\w./-]+\.[A-Za-z]+)`"
)
# First line of a labeled excerpt: "# model.py  GPT.forward"
EXCERPT_HEAD_RE = re.compile(
    r"^\s*(?:#|//|--)\s*([\w./-]+\.[A-Za-z]+)(?:\s+([A-Za-z_][\w.:]*|cell\s+\d+))?\s*$"
)
TREE_NAME_RE = re.compile(r"^[\s│├└─|`-]*([A-Za-z_][\w.]*)")
COMMENT_REF_RE = re.compile(
    rf"(?<![\w/.-])(?P<tok>[\w.\-/]+\.(?:{CODE_EXT})(?::\d+(?:-\d+)?)?)(?![\w/-])"
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
PYTEST_RE = re.compile(r"pytest\s+(?:-\S+\s+)*([\w./\-]+\.py(?:::[\w\[\]\-.]+)*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Step sections: "1. Name", "6.1 Name", or a step ID such as "data.A1 Name",
# "train.B Name" (a group), "model.B3.2. Name".
NUMBERED_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[a-z][\w-]*\.[A-Z]\d*(?:\.\d+)*\.?)\s")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([\w+-]*)")
PLACEHOLDER_RE = re.compile(r"<(?!/)([a-z][a-z _/|-]{1,30})>")
HTML_TAGS = {
    "br",
    "details",
    "summary",
    "sub",
    "sup",
    "b",
    "i",
    "em",
    "strong",
    "code",
    "pre",
    "p",
    "a",
    "img",
    "kbd",
    "div",
    "span",
    "table",
    "tr",
    "td",
    "th",
    "hr",
}
MERMAID_TYPES = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "gantt",
    "pie",
    "mindmap",
    "timeline",
    "journey",
    "quadrantChart",
    "block-beta",
    "sankey-beta",
    "xychart-beta",
    "gitGraph",
    "requirementDiagram",
    "C4Context",
    "C4Container",
    "C4Component",
    "architecture-beta",
    "packet-beta",
    "---",
    "%%",
)
STATUS_OK = re.compile(r"^(pass|fail\b.*|not run\b.*|gap|unknown)$", re.I)
INDEX_REQUIRED = {
    "deep": [
        "Mental model",
        "At a glance",
        "Big picture",
        "Running example",
        "Doc map",
        "Major components",
        "Key sequences",
        "End-to-end data flow",
        "Test coverage",
        "Gaps",
        "Glossary",
    ],
    "simple": [
        "Mental model",
        "At a glance",
        "Big picture",
        "Running example",
        "Test coverage",
        "Gaps",
    ],
}
EXPLAIN_ROOT_RE = re.compile(r"<!--\s*explain-root:\s*(\S+?)\s*-->")
SHAPE_RE = re.compile(r"\[\s*B\s*,\s*[A-Z]")
SECTION_LABELS = ("Input", "Output", "Tests")


class Report:
    def __init__(self):
        self.items = []
        self.line_refs = {}  # doc -> number of path:line citations

    def add(self, level, doc, line, msg):
        self.items.append({"level": level, "doc": str(doc), "line": line, "msg": msg})

    def errors(self):
        return [i for i in self.items if i["level"] == "ERROR"]


class RepoIndex:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files = []
        self.by_base = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                rel = (Path(dirpath) / name).relative_to(self.root).as_posix()
                self.files.append(rel)
                self.by_base.setdefault(name, []).append(rel)
        self._lines = {}
        self.doc_root = ""  # set per doc from <!-- explain-root: ... -->

    def resolve(self, ref: str, doc_dir: Path):
        """Return (list_of_matching_repo_relative_paths)."""
        ref = ref[2:] if ref.startswith("./") else ref
        if ref.startswith("/"):
            p = Path(ref)
            return [str(p)] if p.exists() else []
        if self.doc_root and (self.root / self.doc_root / ref).is_file():
            return [f"{self.doc_root}/{ref}"]
        if (self.root / ref).is_file():
            return [ref]
        cand = (doc_dir / ref).resolve()
        if cand.is_file():
            try:
                return [cand.relative_to(self.root).as_posix()]
            except ValueError:
                return [str(cand)]
        while ref.startswith("../"):
            ref = ref[3:]
        base = ref.rsplit("/", 1)[-1]
        hits = self.by_base.get(base, [])
        if "/" in ref:
            hits = [h for h in hits if h == ref or h.endswith("/" + ref)]
        return hits

    def has_dir(self, first_component: str) -> bool:
        return (self.root / first_component).is_dir()

    def resolve_library(self, ref: str):
        """Resolve `torch/nn/x.py` into an installed package without importing it.

        Returns an absolute path string, "" if the package is installed but the
        file is missing, or None if the package is not installed.
        """
        top = ref.split("/", 1)[0]
        if not top.isidentifier() or "/" not in ref:
            return None
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):
            return None
        if spec is None or not spec.submodule_search_locations:
            return None
        pkg_dir = Path(list(spec.submodule_search_locations)[0])
        cand = pkg_dir.parent / ref
        return str(cand) if cand.is_file() else ""

    def nlines(self, rel: str) -> int:
        if rel not in self._lines:
            p = Path(rel) if rel.startswith("/") else self.root / rel
            try:
                with p.open("rb") as fh:
                    self._lines[rel] = sum(1 for _ in fh)
            except OSError:
                self._lines[rel] = 0
        return self._lines[rel]

    def text(self, rel: str) -> str:
        """File text. For a notebook, the code of its code cells, not the JSON."""
        p = Path(rel) if rel.startswith("/") else self.root / rel
        if p.suffix == ".ipynb":
            try:
                nb = json.loads(p.read_text(errors="ignore"))
            except (OSError, ValueError):
                return ""
            cells = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = cell.get("source", "")
                    cells.append("".join(src) if isinstance(src, list) else str(src))
            return "\n".join(cells)
        try:
            return p.read_text(errors="ignore")
        except OSError:
            return ""


def skip_token(tok: str) -> bool:
    return (
        any(c in tok for c in "{}<>*$")
        or "..." in tok
        or tok.startswith(("http:", "https:", "s3:", "gs:", "~", "file:"))
    )


def check_ref(tok, doc, lineno, idx, doc_dir, rep):
    m = REF_RE.match(tok)
    if not m or skip_token(tok):
        return None
    path = m.group("path")
    hits = idx.resolve(path, doc_dir)
    if not hits:
        lib = idx.resolve_library(path)
        if lib:
            hits = [lib]
        elif lib == "":
            rep.add("ERROR", doc, lineno, f"file not in installed package: `{tok}`")
            return None
    if (
        not hits
        and idx.resolve_library(path) is None
        and path.split("/", 1)[0]
        in (
            "torch",
            "numpy",
            "transformers",
            "triton",
            "jax",
            "tensorflow",
            "pyspark",
        )
    ):
        rep.add("WARN", doc, lineno, f"library not installed, cannot check: `{tok}`")
        return None
    if not hits:
        ext = path.rsplit(".", 1)[-1]
        located = m.group("l1") or m.group("sym")
        if ext in SOURCE_EXT or located:
            rep.add("ERROR", doc, lineno, f"code reference not found in repo: `{tok}`")
        elif "/" in path and idx.has_dir(path.lstrip("./").split("/")[0]):
            rep.add(
                "WARN", doc, lineno, f"file not in repo (runtime artifact?): `{tok}`"
            )
        return None
    if len(hits) > 1:
        if m.group("l1") or m.group("sym"):
            rep.add(
                "WARN",
                doc,
                lineno,
                f"`{path}` matches {len(hits)} files; use a longer path so the "
                f"line/symbol can be checked (e.g. `{hits[0]}`)",
            )
        return None
    rel = hits[0]
    if m.group("l1"):
        rep.line_refs[str(doc)] = rep.line_refs.get(str(doc), 0) + 1
        n = idx.nlines(rel)
        l1 = int(m.group("l1"))
        l2 = int(m.group("l2") or l1)
        if l2 < l1:
            rep.add("ERROR", doc, lineno, f"line range reversed: `{tok}`")
        elif l1 < 1 or l2 > n:
            rep.add(
                "ERROR", doc, lineno, f"`{tok}` is past the end of {rel} ({n} lines)"
            )
        else:
            return rel, l1, l2
    elif m.group("sym"):
        src = idx.text(rel)
        is_py = rel.endswith((".py", ".pyi", ".ipynb"))
        for part in m.group("sym").split("."):
            if is_py:
                pat = re.compile(
                    rf"^\s*(?:async\s+def|def|class)\s+{re.escape(part)}\b"
                    rf"|^\s*{re.escape(part)}\s*[:=]",
                    re.M,
                )
            else:  # C++, Rust, Go, Scala, SQL...: the name must appear in the file
                pat = re.compile(rf"(?<![\w]){re.escape(part)}(?![\w])")
            if not pat.search(src):
                rep.add("ERROR", doc, lineno, f"`{part}` is not defined in {rel}")
                return None
    return rel, None, None


# Mermaid keywords are case-insensitive: a participant named `Opt` or `End`
# is read as the keyword and the whole diagram fails to parse.
SEQ_KEYWORDS = {
    "participant",
    "actor",
    "as",
    "loop",
    "alt",
    "else",
    "opt",
    "par",
    "and",
    "rect",
    "critical",
    "option",
    "break",
    "end",
    "note",
    "over",
    "activate",
    "deactivate",
    "autonumber",
    "title",
    "box",
    "create",
    "destroy",
    "links",
    "link",
    "properties",
    "details",
    "left",
    "right",
    "of",
}
SEQ_DECL_RE = re.compile(r"^\s*(?:participant|actor)\s+([A-Za-z_][\w-]*)")
SEQ_MSG_RE = re.compile(
    r"^\s*([A-Za-z_][\w-]*)\s*[-=]+[>x)]+[+-]?\s*([A-Za-z_][\w-]*)\s*:"
)
FLOW_END_RE = re.compile(r"(?:^|[\s;&>|-])end\s*(?:[\[\(\{>]|-->|---|$)")
MACHINE_PATH_RE = re.compile(r"(/Users/[^/\s`]+/|/home/[^/\s`]+/|[A-Za-z]:\\Users\\)")


def check_mermaid_line(kind, line, doc, lineno, rep):
    """Catch the Mermaid syntax traps that make a whole diagram fail to render."""
    if kind == "sequenceDiagram":
        ids = []
        m = SEQ_DECL_RE.match(line)
        if m:
            ids.append(m.group(1))
        m = SEQ_MSG_RE.match(line)
        if m:
            ids += [m.group(1), m.group(2)]
        for i in ids:
            if i.lower() in SEQ_KEYWORDS:
                rep.add(
                    "ERROR",
                    doc,
                    lineno,
                    f"sequenceDiagram id `{i}` is a Mermaid keyword (keywords ignore case). "
                    f"Rename it, for example `{i}X as {i}`",
                )
        if ";" in line or (re.search(r":.*#", line)):
            rep.add(
                "WARN",
                doc,
                lineno,
                "`;` or `#` in a sequenceDiagram message breaks parsing",
            )
    elif kind in ("flowchart", "graph"):
        if re.search(r"\|\s*\[", line):
            rep.add(
                "ERROR",
                doc,
                lineno,
                'unquoted edge label starting with "[" breaks the diagram. Write -->|"[B, T, D]"|',
            )
        if FLOW_END_RE.search(line.split("%%")[0]):
            rep.add(
                "ERROR",
                doc,
                lineno,
                "flowchart node id `end` (lowercase) breaks the diagram. Use `End` or `done`",
            )


STEP_ID_RE = re.compile(r"^[a-z][\w-]*\.[A-Z]\d+(?:\.\d+)*\.?\s")
FENCED_RE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)
TABLE_HEAD_RE = re.compile(r"^\|(.+)\|\s*\n\|[\s:|-]+\|\s*$", re.M)


def has_output_example(body: str) -> bool:
    """True if a step shows data: a non-code, non-tree block, or a non-test table."""
    body = body.split("**Tests.**", maxsplit=1)[0]
    for lang, block in FENCED_RE.findall(body):
        if lang == "mermaid" or "├─" in block or "└─" in block:
            continue
        if EXCERPT_HEAD_RE.match(
            block.lstrip("\n").splitlines()[0] if block.strip() else ""
        ):
            continue
        return True
    return any("Status" not in head for head in TABLE_HEAD_RE.findall(body))


def norm_code(line: str) -> str:
    """Code line without comments or extra whitespace, for excerpt matching."""
    line = re.sub(r"(^|\s)(#|//|--\s).*$", "", line)
    return " ".join(line.split())


def check_symbol_in_file(sym, file_tok, doc, lineno, idx, doc_dir, rep):
    """`sym` in `file`: the name must still appear in that file."""
    hits = idx.resolve(file_tok, doc_dir)
    if len(hits) != 1:
        return
    name = sym.split(".")[-1]
    if not re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", idx.text(hits[0])):
        rep.add("ERROR", doc, lineno, f"`{name}` does not appear in {hits[0]}")


def forward_args(src: str, qualname: str):
    """Argument names of Class.method in Python source, or None if not found."""
    import ast

    cls_name, _, meth = qualname.rpartition(".")
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == meth
                ):
                    a = item.args
                    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                    return [n for n in names if n not in ("self", "cls")]
    return None


def check_forward_inputs(text: str, doc, idx, doc_dir, rep):
    """Every argument of an excerpted `X.forward` should be named in the doc."""
    seen = set()
    for m in re.finditer(
        r"^\s*#\s*([\w./-]+\.py)\s+([\w.]+\.(?:forward|__call__))\s*$", text, re.M
    ):
        key = (m.group(1), m.group(2))
        if key in seen:
            continue
        seen.add(key)
        hits = idx.resolve(m.group(1), doc_dir)
        if len(hits) != 1:
            continue
        args = forward_args(idx.text(hits[0]), m.group(2))
        for a in args or []:
            if not re.search(rf"(?<![\w]){re.escape(a)}(?![\w])", text):
                rep.add(
                    "WARN",
                    doc,
                    0,
                    f"`{m.group(2)}` takes `{a}`, but the doc never mentions it. "
                    "Explain that input, or say why the golden path does not use it",
                )


def check_excerpt(exc, doc, idx, rep):
    """A labeled excerpt must still match the file it names."""
    src = {norm_code(x) for x in idx.text(exc["rel"]).splitlines()}
    src.discard("")
    body = [norm_code(x) for x in exc["lines"]]
    body = [b for b in body if b and b not in ("...", "…")]
    if len(body) < 2:
        return
    missing = [b for b in body if b not in src]
    if len(missing) > 0.2 * len(body):
        rep.add(
            "WARN",
            doc,
            exc["start"],
            f"excerpt from {exc['rel']} no longer matches the file "
            f"({len(body) - len(missing)} of {len(body)} lines found). "
            f"First missing: `{missing[0][:70]}`",
        )


TREE_STOPWORDS = {
    "self",
    "None",
    "True",
    "False",
    "return",
    "with",
    "else",
    "elif",
    "from",
    "import",
    "lambda",
    "only",
    "same",
    "each",
    "then",
    "when",
    "loop",
    "call",
    "calls",
    "line",
    "lines",
    "here",
    "this",
    "that",
    "step",
}
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]{3,})\s*\(")


def check_tree_citation(left, resolved, doc, lineno, idx, rep):
    """Warn when a call-tree line cites a line that does not mention its call,
    but a line within 3 of it does (the usual off-by-one).
    """
    rel, l1, l2 = resolved
    if l1 == 1:
        return  # whole-file citation
    names = [c for c in CALL_RE.findall(left) if c not in TREE_STOPWORDS]
    if not names:
        return
    src = idx.text(rel).splitlines()
    cited = "\n".join(src[l1 - 1 : l2])
    if any(n in cited for n in names):
        return
    if src[l1 - 1].lstrip().startswith("@") and l1 < len(src):
        if any(n in src[l1] for n in names):
            return  # decorator line of the cited def
    for delta in (1, -1, 2, -2, 3, -3):
        j = l1 + delta
        if 1 <= j <= len(src) and any(n in src[j - 1] for n in names):
            hit = next(n for n in names if n in src[j - 1])
            rep.add(
                "WARN",
                doc,
                lineno,
                f"{rel}:{l1} does not mention `{hit}`; line {j} does (off by {delta:+d}?)",
            )
            return


def check_pytest(target, doc, lineno, idx, doc_dir, rep):
    parts = target.split("::")
    hits = idx.resolve(parts[0], doc_dir)
    if not hits:
        rep.add("ERROR", doc, lineno, f"pytest file not found: `{parts[0]}`")
        return
    if len(parts) == 1 or len(hits) > 1:
        return
    src = idx.text(hits[0])
    for name in parts[1:]:
        name = name.split("[", 1)[0]
        if not re.search(
            rf"^\s*(?:async\s+def|def|class)\s+{re.escape(name)}\b", src, re.M
        ):
            rep.add("ERROR", doc, lineno, f"`{name}` not defined in {hits[0]}")


def slugify(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[`*_~]", "", s)
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def parse_doc(text):
    """Yield (lineno, line, in_fence, fence_lang) and collect fence problems."""
    in_fence, lang, fence_tok, start = False, "", "", 0
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        m = FENCE_RE.match(line)
        if m and (not in_fence or line.strip().startswith(fence_tok)):
            if not in_fence:
                in_fence, lang, fence_tok, start = True, m.group(2), m.group(1), i
                out.append((i, line, "open", lang))
                continue
            if line.strip().strip("`~") == "":
                in_fence = False
                out.append((i, line, "close", lang))
                continue
        out.append((i, line, in_fence, lang))
    return out, (start if in_fence else 0)


def check_doc(doc: Path, idx: RepoIndex, mode: str, n_docs: int, rep: Report):
    text = doc.read_text(errors="ignore")
    root_m = EXPLAIN_ROOT_RE.search(text)
    idx.doc_root = root_m.group(1).strip("/") if root_m else ""
    if idx.doc_root and not (idx.root / idx.doc_root).is_dir():
        rep.add(
            "ERROR",
            doc,
            0,
            f"explain-root `{idx.doc_root}` is not a folder in the repo",
        )
    rows, unclosed = parse_doc(text)
    if unclosed:
        rep.add("ERROR", doc, unclosed, "code fence opened here is never closed")
    doc_dir = doc.parent.resolve()

    headings = []
    mermaid_first = None
    status_col = None
    excerpt = None  # {"rel", "lines", "start"} while inside a labeled excerpt
    mermaid_kind = ""
    fence_lines, fence_start = [], 0
    first_in_fence = False
    for lineno, line, state, lang in rows:
        if state == "open":
            mermaid_first = lineno if lang == "mermaid" else None
            status_col = None
            excerpt, first_in_fence = None, lang != "mermaid"
            fence_lines, fence_start = [], lineno
            continue
        if state == "close":
            if mermaid_first is not None:
                rep.add("ERROR", doc, mermaid_first, "empty mermaid block")
            mermaid_first = None
            if excerpt:
                check_excerpt(excerpt, doc, idx, rep)
            tree = [x for x in fence_lines if "├─" in x or "└─" in x]
            path_like = sum(1 for x in tree if "/" in x.split("#")[0].split(" -> ")[0])
            if (
                len(tree) >= 3
                and path_like < len(tree) / 2
                and not any(" -> " in x for x in fence_lines)
            ):
                rep.add(
                    "WARN",
                    doc,
                    fence_start,
                    "call tree has no `-> output` tags. End each node that produces "
                    "something with `-> artifact, type/shape, location`",
                )
            excerpt, first_in_fence = None, False
            fence_lines = []
            continue
        if state is True:
            if lang != "mermaid":
                fence_lines.append(line)
            if excerpt is not None:
                excerpt["lines"].append(line)
            elif first_in_fence and line.strip():
                first_in_fence = False
                hm = EXCERPT_HEAD_RE.match(line)
                if hm:
                    hits = idx.resolve(hm.group(1), doc_dir)
                    if len(hits) == 1:
                        excerpt = {"rel": hits[0], "lines": [], "start": lineno}
            if lang == "mermaid":
                if mermaid_first is not None and line.strip():
                    if not line.strip().startswith(MERMAID_TYPES):
                        rep.add(
                            "ERROR",
                            doc,
                            lineno,
                            f"mermaid block must start with a diagram type, got `{line.strip()[:40]}`",
                        )
                    mermaid_kind = line.strip().split()[0]
                    mermaid_first = None
                else:
                    check_mermaid_line(mermaid_kind, line, doc, lineno, rep)
                for lab in re.findall(r"\b\w+\[([^\]\"]*)\]", line):
                    if any(c in lab for c in "(){}"):
                        rep.add(
                            "WARN", doc, lineno, f"quote this mermaid label: [{lab}]"
                        )
            elif "#" in line:
                left, comment = line.split("#", 1)
                left = left.split(" -> ")[0]  # drop the "-> output" tag of a call tree
                for m in COMMENT_REF_RE.finditer(comment):
                    resolved = check_ref(m.group("tok"), doc, lineno, idx, doc_dir, rep)
                    if not (resolved and left.strip()) or excerpt is not None:
                        continue
                    if resolved[1] is not None:
                        check_tree_citation(left, resolved, doc, lineno, idx, rep)
                    else:
                        nm = TREE_NAME_RE.match(left)
                        name = nm.group(1) if nm else ""
                        is_path = "/" in left  # a file-layout tree, not a call tree
                        codeish = not is_path and (
                            "_" in name or "." in name or re.search(r"[a-z][A-Z]", name)
                        )
                        last = name.split(".")[-1]
                        if codeish and not re.search(
                            rf"(?<![\w]){re.escape(last)}(?![\w])",
                            idx.text(resolved[0]),
                        ):
                            rep.add(
                                "WARN",
                                doc,
                                lineno,
                                f"call tree names `{last}`, but {resolved[0]} does not mention it",
                            )
            for m in PYTEST_RE.finditer(line):
                check_pytest(m.group(1), doc, lineno, idx, doc_dir, rep)
            continue

        hm = HEADING_RE.match(line)
        if hm:
            headings.append((lineno, len(hm.group(1)), hm.group(2)))
            if re.search(r"\((required|optional)\b", hm.group(2), re.I):
                rep.add(
                    "WARN",
                    doc,
                    lineno,
                    "template marker left in a heading. Remove '(required...)' or '(optional)'",
                )
            status_col = None
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if "Status" in cells:
                status_col = cells.index("Status")
            elif status_col is not None and not all(
                re.fullmatch(r":?-+:?", c) for c in cells
            ):
                if status_col < len(cells):
                    val = cells[status_col].strip("` ")
                    if val and not STATUS_OK.match(val):
                        rep.add(
                            "WARN",
                            doc,
                            lineno,
                            f"status `{val}` is not one of pass / fail / not run (needs X) / gap",
                        )
                    elif val.lower() == "unknown":
                        rep.add(
                            "WARN",
                            doc,
                            lineno,
                            "status still `unknown`; run the test mapper",
                        )
        else:
            status_col = None

        for sm in SYM_IN_FILE_RE.finditer(line):
            check_symbol_in_file(
                sm.group(1), sm.group(2), doc, lineno, idx, doc_dir, rep
            )
        for code in INLINE_CODE_RE.findall(line):
            code = code.strip()
            pm = PYTEST_RE.search(code)
            if pm:
                check_pytest(pm.group(1), doc, lineno, idx, doc_dir, rep)
            else:
                check_ref(code, doc, lineno, idx, doc_dir, rep)
        prose = INLINE_CODE_RE.sub("", line)
        for target in LINK_RE.findall(prose):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if path and not (doc_dir / path).exists():
                rep.add("ERROR", doc, lineno, f"broken link: {target}")
        for ph in PLACEHOLDER_RE.findall(prose):
            if ph.split()[0] not in HTML_TAGS:
                rep.add("WARN", doc, lineno, f"leftover placeholder <{ph}>")
        if re.search(r"\b(TODO|TBD|FIXME|XXX)\b", prose):
            rep.add("WARN", doc, lineno, "leftover TODO/TBD/FIXME")

    lines = text.splitlines()
    for i, (lineno, level, title) in enumerate(headings):
        if not NUMBERED_RE.match(title):
            continue
        end = len(lines)
        for nl, nlevel, _ in headings[i + 1 :]:
            if nlevel <= level:
                end = nl - 1
                break
        body = "\n".join(lines[lineno:end])
        missing = [lab for lab in SECTION_LABELS if not re.search(rf"\b{lab}\b", body)]
        if missing:
            rep.add(
                "ERROR" if mode == "deep" else "WARN",
                doc,
                lineno,
                f"section '{title[:50]}' is missing: {', '.join(missing)}",
            )
        if mode == "deep" and STEP_ID_RE.match(title) and not has_output_example(body):
            rep.add(
                "WARN",
                doc,
                lineno,
                f"step '{title.split()[0]}' shows no example output. Add the object itself: "
                "rows or schema table, array slice, trimmed JSON, or a file listing",
            )

    if doc.name == "index.md":
        required = list(INDEX_REQUIRED[mode])
        if mode == "simple" and n_docs > 1:
            required.append("Doc map")
        titles = [t.lower() for _, _, t in headings]
        for req in required:
            if not any(req.lower() in t for t in titles):
                rep.add("ERROR", doc, 1, f"index.md is missing the heading '{req}'")

    if len(lines) > 150 and not re.search(
        r"^\s*(\[TOC\]|\[\[_TOC_\]\]|#{1,3}\s+(Table of )?Contents\b)",
        text,
        re.M | re.I,
    ):
        rep.add(
            "WARN",
            doc,
            1,
            f"{len(lines)} lines and no contents block. Add [TOC] (Bitbucket), "
            "[[_TOC_]] (GitLab), or a '## Contents' list (GitHub)",
        )

    check_forward_inputs(text, doc, idx, doc_dir, rep)

    explains_model = doc.name != "index.md" and (
        re.search(r"^\s*#\s*[\w./-]+\.py\s+[\w.]*\b(forward|__call__)\b", text, re.M)
        or re.search(r"^#{1,3}\s.*\b(architecture|forward pass)\b", text, re.M | re.I)
        or len(re.findall(r"`[^`\n]*\.forward\b[^`\n]*`", text)) >= 2
    )
    if explains_model and SHAPE_RE.search(text):
        flows = [
            b
            for lang, b in FENCED_RE.findall(text)
            if lang == "mermaid" and b.lstrip().startswith(("flowchart", "graph"))
        ]
        if not any(re.search(r'\|"\s*\[', b) for b in flows):
            rep.add(
                "WARN",
                doc,
                1,
                "explains a model but has no architecture diagram: add a Mermaid "
                'flowchart TB with shapes on the edges (-->|"[B, T, D]"|)',
            )

    n_lines = rep.line_refs.get(str(doc), 0)
    if n_lines >= 5:
        rep.add(
            "WARN",
            doc,
            0,
            f"{n_lines} line-number citations. Lines drift on every commit. "
            "Name the function, class, or config key instead, and show an excerpt",
        )

    for n, raw in enumerate(lines, 1):
        pm = MACHINE_PATH_RE.search(raw)
        if pm:
            rep.add(
                "WARN",
                doc,
                n,
                f"machine-specific path `{pm.group(1)}...`. Readers clone the repo elsewhere. "
                "Use a path relative to the repo root and a portable command (`pytest ...`, `uv run pytest ...`)",
            )

    n_unverified = len(re.findall(r"UNVERIFIED", text))
    if n_unverified:
        rep.add("INFO", doc, 0, f"{n_unverified} UNVERIFIED claims")


def check_doc_set(docs, mode, rep):
    """Checks that span the whole doc set: diagram coverage, shapes cheat sheet."""
    texts = {d: d.read_text(errors="ignore") for d in docs}
    index = next((d for d in docs if d.name == "index.md"), None)
    if mode == "deep" and len(docs) > 1:
        joined = "\n".join(texts.values())
        for kind in ("classDiagram", "sequenceDiagram"):
            if not re.search(rf"```mermaid\s*\n\s*{kind}", joined):
                rep.add(
                    "WARN",
                    index or docs[0],
                    0,
                    f"no Mermaid {kind} anywhere in the doc set",
                )
    if index is not None and any(SHAPE_RE.search(t) for t in texts.values()):
        if not re.search(r"^#+\s+Shapes cheat sheet", texts[index], re.M | re.I):
            rep.add(
                "WARN",
                index,
                1,
                "the docs use tensor shapes but index.md has no 'Shapes cheat sheet'",
            )


def check_must_cover(path: Path, docs, rep):
    """Warn for each must-cover topic that no doc in the set mentions."""
    if not path.is_file():
        rep.add("ERROR", path, 0, "must-cover file not found")
        return
    joined = "\n".join(d.read_text(errors="ignore") for d in docs).lower()
    for n, raw in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        topic, _, source = raw.partition("#")
        topic = topic.strip()
        if topic and topic.lower() not in joined:
            where = f" (from {source.strip()})" if source.strip() else ""
            rep.add(
                "WARN",
                path,
                n,
                f"must-cover topic `{topic}` is in no doc{where}. "
                "Cover it, or list it under Out of scope with the reason",
            )


def collect_docs(paths):
    docs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            docs += sorted(x for x in p.glob("*.md"))
        elif p.suffix == ".md":
            docs.append(p)
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="docs dir or .md files")
    ap.add_argument("--repo", required=True, help="root of the explained repo")
    ap.add_argument("--mode", choices=("deep", "simple"), default="deep")
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--must-cover",
        help="file with one topic per line (text after # is the source); "
        "warns for each topic no doc mentions",
    )
    args = ap.parse_args(argv)

    docs = collect_docs(args.paths)
    if not docs:
        print("no .md files found", file=sys.stderr)
        return 2
    idx = RepoIndex(Path(args.repo))
    rep = Report()
    for d in docs:
        check_doc(d, idx, args.mode, len(docs), rep)
    check_doc_set(docs, args.mode, rep)
    if args.must_cover:
        check_must_cover(Path(args.must_cover), docs, rep)

    if args.json:
        print(json.dumps(rep.items, indent=2))
    else:
        order = {"ERROR": 0, "WARN": 1, "INFO": 2}
        for it in sorted(
            rep.items, key=lambda x: (order[x["level"]], x["doc"], x["line"])
        ):
            print(f"{it['level']:5} {it['doc']}:{it['line']}  {it['msg']}")
        n_err = len(rep.errors())
        n_warn = sum(1 for i in rep.items if i["level"] == "WARN")
        print(f"\n{len(docs)} docs checked: {n_err} errors, {n_warn} warnings")
    return 1 if rep.errors() else 0


if __name__ == "__main__":
    sys.exit(main())
