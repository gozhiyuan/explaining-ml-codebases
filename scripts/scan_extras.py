"""Extra recon for scan_ml_repo.py: notebooks, non-Python code, SQL/dbt, infra.

Runs as a second pass after the Python analysis and never changes it. Its
results go in their own report sections. They feed repo-type scores only
through their own evidence (build-file dependencies, dbt projects), and
notebook code counts toward scores only in notebook-heavy repos.
Stdlib only, Python 3.9+.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NON_PY = {
    ".c": "c/c++",
    ".cc": "c/c++",
    ".cpp": "c/c++",
    ".cxx": "c/c++",
    ".h": "c/c++",
    ".hpp": "c/c++",
    ".cu": "c/c++",
    ".cuh": "c/c++",
    ".rs": "rust",
    ".go": "go",
    ".scala": "scala/java",
    ".java": "scala/java",
    ".kt": "scala/java",
}
# Entry points, types, and tests per language family (regex heuristics).
LANG_RULES = {
    "c/c++": {
        "main": re.compile(r"^\s*int\s+main\s*\(", re.M),
        "type": re.compile(
            r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)\s*(?:final\s*)?[:{]", re.M
        ),
        "func": re.compile(
            r"^[A-Za-z_][\w:<>\s\*&]*?\b([a-z_][a-z0-9_]{3,})\s*\([^;]*\)\s*\{", re.M
        ),
        "test": re.compile(r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST)\s*\(", re.M),
    },
    "rust": {
        "main": re.compile(r"^\s*(?:async\s+)?fn\s+main\s*\(", re.M),
        "type": re.compile(
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Z]\w*)", re.M
        ),
        "func": re.compile(
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([a-z_]\w*)", re.M
        ),
        "test": re.compile(r"#\[(?:tokio::)?test\]"),
    },
    "go": {
        "main": re.compile(r"^func\s+main\s*\(", re.M),
        "type": re.compile(r"^type\s+([A-Z]\w*)\s+(?:struct|interface)", re.M),
        "func": re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(", re.M),
        "test": re.compile(r"^func\s+Test\w*\s*\(", re.M),
    },
    "scala/java": {
        "main": re.compile(
            r"(?:def\s+main\s*\(|public\s+static\s+void\s+main\s*\(|extends\s+App\b)"
        ),
        "type": re.compile(
            r"^\s*(?:public\s+|final\s+|abstract\s+|case\s+|sealed\s+)*(?:class|object|trait|interface)\s+([A-Z]\w*)",
            re.M,
        ),
        "func": re.compile(r"^\s*(?:override\s+)?def\s+([a-z]\w*)", re.M),
        "test": re.compile(
            r'(?:@Test\b|^\s*test\s*\(\s*"|"[^"]+"\s+should\b|\bit\s*\(\s*")', re.M
        ),
    },
}
BINDING_RULES = {
    "pybind11": re.compile(r"\bPYBIND11_MODULE\s*\("),
    "torch op registration": re.compile(r"\bTORCH_LIBRARY(?:_IMPL)?\s*\("),
    "PyO3": re.compile(r"#\[pymodule\]|#\[pyfunction\]"),
    "cgo": re.compile(r'^import\s+"C"', re.M),
    "JNI": re.compile(r"\bJNIEXPORT\b"),
    "ctypes/cffi (Python side)": re.compile(r"\b(ctypes\.CDLL|cffi\.FFI)\b"),
}
BUILD_FILES = {
    "CMakeLists.txt",
    "Cargo.toml",
    "go.mod",
    "build.sbt",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Makefile",
    "package.json",
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "meson.build",
    "setup.py",
    "pyproject.toml",
}
# Dependency name (or CMake package) -> framework label used by archetype scoring.
DEP_LABELS = {
    "ggml": "ggml",
    "llama": "ggml",
    "candle-core": "candle",
    "candle": "candle",
    "mistralrs": "candle",
    "tch": "pytorch",
    "ort": "onnx",
    "onnxruntime": "onnx",
    "tokio": "rust-async",
    "axum": "rust-web",
    "actix-web": "rust-web",
    "warp": "rust-web",
    "tonic": "grpc",
    "grpc": "grpc",
    "google.golang.org/grpc": "grpc",
    "github.com/gin-gonic/gin": "go-web",
    "github.com/labstack/echo": "go-web",
    "spark-core": "spark",
    "spark-sql": "spark",
    "spark-mllib": "spark",
    "flink-streaming-java": "flink",
    "flink-streaming-scala": "flink",
    "kafka-clients": "kafka",
    "kafka-streams": "kafka",
    "CUDA": "cuda",
    "CUDAToolkit": "cuda",
    "TensorRT": "tensorrt",
    "Torch": "pytorch",
}
SBATCH_RE = re.compile(r"^#SBATCH\b", re.M)
LAUNCH_RE = re.compile(
    r"\b(torchrun|deepspeed|accelerate launch|python -m torch\.distributed\.run|ray job submit|mpirun|srun)\b"
)
K8S_KIND_RE = re.compile(
    r"^kind:\s*(PyTorchJob|TFJob|MPIJob|RayJob|RayCluster|RayService|Job|CronJob|Deployment|StatefulSet|InferenceService|ServingRuntime|ScaledObject|HorizontalPodAutoscaler)\s*$",
    re.M,
)
DBT_REF_RE = re.compile(r"\{\{\s*ref\(\s*['\"]([\w.]+)['\"]")
DBT_SRC_RE = re.compile(r"\{\{\s*source\(")
DBT_TEST_RE = re.compile(
    r"^\s*-\s*(unique|not_null|relationships|accepted_values|dbt_utils\.\w+|dbt_expectations\.\w+)\b",
    re.M,
)
MAGIC_RE = re.compile(r"^\s*[%!].*$", re.M)


def _read(path: Path, limit: int = 2_000_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def notebook_code(path: Path) -> list:
    """Code cells of a notebook as strings (IPython magics and shell lines removed)."""
    try:
        nb = json.loads(_read(path, 20_000_000) or "{}")
    except (ValueError, TypeError):
        return []
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            src = "".join(src) if isinstance(src, list) else str(src)
            cells.append(MAGIC_RE.sub("", src))
    return cells


def _deps_from_build(path: Path, text: str) -> set:
    name, deps = path.name, set()
    if name == "Cargo.toml":
        for section in re.findall(
            r"^\[(?:[\w.-]*dependencies)\]\s*\n(.*?)(?=^\[|\Z)", text, re.S | re.M
        ):
            deps |= set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", section, re.M))
    elif name == "go.mod":
        deps |= set(re.findall(r"^\s*(?:require\s+)?([\w.\-/]+)\s+v[\d.]", text, re.M))
        deps |= {"/".join(d.split("/")[:3]) for d in list(deps)}
    elif name == "build.sbt":
        deps |= {b for _, b in re.findall(r'"([\w.-]+)"\s*%%?\s*"([\w.-]+)"', text)}
    elif name == "pom.xml" or name.startswith("build.gradle"):
        deps |= set(re.findall(r"<artifactId>([\w.-]+)</artifactId>", text))
        deps |= {b for _, b in re.findall(r"['\"]([\w.-]+):([\w.-]+):", text)}
    elif name == "CMakeLists.txt":
        deps |= set(re.findall(r"find_package\s*\(\s*(\w+)", text))
        if re.search(r"\bLANGUAGES\b[^)]*\bCUDA\b|enable_language\s*\(\s*CUDA", text):
            deps.add("CUDA")
        if re.search(r"\bggml\b", text):
            deps.add("ggml")
    deps = {re.sub(r"_2\.\d+$", "", d) for d in deps}  # scala suffixes: spark-core_2.12
    return deps


def scan_extras(
    root: Path,
    top: int,
    own: set,
    iter_files,
    is_noise,
    rank_paths,
    import_roots,
    frameworks: dict,
    train_re,
    py_loc: int,
) -> dict:
    nb_rows, nb_fw, nb_train = [], defaultdict(set), []
    np_files, np_loc = Counter(), Counter()
    np_mains, np_types, np_tests = defaultdict(list), defaultdict(list), Counter()
    np_test_files = Counter()
    type_refs = Counter()
    bindings = defaultdict(list)
    build_files, deps = [], Counter()
    sql_files, dbt_projects, dbt_models, dbt_refs = [], [], {}, Counter()
    dbt_sources = dbt_tests = 0
    sbatch, launchers, k8s_kinds, helm, terraform = [], Counter(), Counter(), [], []
    np_texts = []

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        ext, name = path.suffix.lower(), path.name
        noise = is_noise(rel)

        if ext == ".ipynb":
            cells = notebook_code(path)
            code = "\n".join(cells)
            loc = sum(1 for line in code.splitlines() if line.strip())
            defs = len(re.findall(r"^\s*(?:def|class)\s+\w+", code, re.M))
            fws = set()
            try:
                for mod in import_roots(ast.parse(code)):
                    r = mod.split(".")[0]
                    if r in frameworks and r not in own:
                        fws.add(frameworks[r])
            except (SyntaxError, ValueError, RecursionError):
                fws |= {
                    frameworks[m]
                    for m in re.findall(r"^\s*(?:import|from)\s+(\w+)", code, re.M)
                    if m in frameworks and m not in own
                }
            trains = bool(train_re.search(code))
            nb_rows.append(
                {
                    "path": rel,
                    "code_cells": len(cells),
                    "code_loc": loc,
                    "defs": defs,
                    "training": trains,
                    "frameworks": sorted(fws),
                }
            )
            for fw in fws:
                nb_fw[fw].add(rel)
            if trains and not noise:
                nb_train.append(rel)
            continue

        if name in BUILD_FILES and rel.count("/") <= 2:
            build_files.append(rel)
            for d in _deps_from_build(path, _read(path)):
                if d in DEP_LABELS:
                    deps[DEP_LABELS[d]] += 1
        if name == "dbt_project.yml":
            dbt_projects.append(rel)
        if name == "Chart.yaml":
            helm.append(rel)
        if ext == ".tf":
            terraform.append(rel)

        if ext in (".sh", ".slurm", ".sbatch", ".bash"):
            text = _read(path)
            if SBATCH_RE.search(text):
                sbatch.append(rel)
            for m in LAUNCH_RE.findall(text):
                launchers[m] += 1
        if ext in (".yaml", ".yml") and not noise:
            for kind in K8S_KIND_RE.findall(_read(path)):
                k8s_kinds[kind] += 1

        if ext == ".sql":
            sql_files.append(rel)
            text = _read(path)
            if "/models/" in f"/{rel}" or "{{" in text:
                model = Path(rel).stem
                refs = DBT_REF_RE.findall(text)
                dbt_models[model] = rel
                dbt_refs.update(refs)
                dbt_sources += len(DBT_SRC_RE.findall(text))
            continue
        if ext in (".yml", ".yaml") and ("/models/" in f"/{rel}" or "schema" in name):
            text = _read(path)
            if "models:" in text or "sources:" in text:
                dbt_tests += len(DBT_TEST_RE.findall(text))

        family = NON_PY.get(ext)
        if family:
            text = _read(path)
            np_files[family] += 1
            np_loc[family] += text.count("\n")
            rules = LANG_RULES[family]
            if rules["test"].search(text) or "/test" in f"/{rel}".lower():
                np_tests[family] += len(rules["test"].findall(text))
                np_test_files[family] += 1
                continue
            if rules["main"].search(text) and name != "build.rs":
                np_mains[family].append(rel)
            for t in rules["type"].findall(text):
                np_types[family].append((t, rel))
            for label, rx in BINDING_RULES.items():
                if label.startswith("ctypes"):
                    continue
                if rx.search(text):
                    bindings[label].append(rel)
            if not noise:
                np_texts.append(text)
        elif ext == ".py":
            text = _read(path)
            if BINDING_RULES["ctypes/cffi (Python side)"].search(text):
                bindings["ctypes/cffi (Python side)"].append(rel)

    # Rank non-Python types by how many files mention them.
    if np_types:
        names = {t for items in np_types.values() for t, _ in items}
        for text in np_texts:
            for word in set(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text)) & names:
                type_refs[word] += 1
    non_python = {}
    for fam in sorted(np_files, key=lambda f: -np_loc[f]):
        seen, types = set(), []
        for t, rel in sorted(
            np_types[fam], key=lambda x: (is_noise(x[1]), -type_refs[x[0]], x[1])
        ):
            if t not in seen and not is_noise(rel):
                seen.add(t)
                types.append({"name": t, "file": rel, "ref_files": type_refs[t]})
        non_python[fam] = {
            "files": np_files[fam],
            "loc": np_loc[fam],
            "entry_points": rank_paths(np_mains[fam])[:top],
            "key_types": types[:top],
            "test_cases": np_tests[fam],
            "test_files": np_test_files[fam],
        }

    # Distinct non-Python type names that look like serving-engine parts.
    engine_words = ("kv_cache", "kvcache", "sampler", "scheduler", "batch", "infer")
    engine_types = sorted(
        {
            t
            for items in np_types.values()
            for t, rel in items
            if not is_noise(rel) and any(w in t.lower() for w in engine_words)
        }
    )

    nb_code_loc = sum(r["code_loc"] for r in nb_rows)
    heavy = nb_code_loc > max(py_loc, 1)
    nb_rows.sort(key=lambda r: (is_noise(r["path"]), -(r["defs"] * 20 + r["code_loc"])))

    dbt = {}
    if dbt_projects or dbt_models:
        dbt = {
            "projects": dbt_projects[:top],
            "models": len(dbt_models),
            "most_referenced": [
                {"model": m, "refs": n, "file": dbt_models.get(m, "")}
                for m, n in dbt_refs.most_common(top)
            ],
            "ref_edges": sum(dbt_refs.values()),
            "source_calls": dbt_sources,
            "schema_tests": dbt_tests,
        }

    return {
        "notebooks": {
            "count": len(nb_rows),
            "code_loc": nb_code_loc,
            "heavy": heavy,
            "with_logic": [r for r in nb_rows if r["defs"] or r["training"]][:top],
            "top": nb_rows[:top],
        },
        "_nb_fw": nb_fw,
        "_nb_train": nb_train,
        "non_python": non_python,
        "engine_types": engine_types[: top * 2],
        "_engine_type_count": len(engine_types),
        "bindings": {k: rank_paths(v)[:top] for k, v in bindings.items()},
        "build_files": rank_paths(build_files)[: top * 2],
        "dep_labels": dict(deps),
        "sql": {"files": len(sql_files), "dbt": dbt},
        "infra": {
            "slurm_scripts": rank_paths(sbatch)[:top],
            "launchers": dict(launchers.most_common()),
            "k8s_kinds": dict(k8s_kinds.most_common()),
            "helm_charts": rank_paths(helm)[:top],
            "terraform_files": len(terraform),
        },
    }
