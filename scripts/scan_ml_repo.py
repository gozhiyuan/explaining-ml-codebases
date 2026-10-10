#!/usr/bin/env python3
"""Read-only recon of an ML/AI repository for the explaining-ml-codebases skill.

Reports size tier, archetype scores with evidence, canonical role classes,
entry points, key classes, tests, configs, and existing docs.
Stdlib only, Python 3.9+.

Usage:
    scan_ml_repo.py REPO [--json] [--top N]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_extras

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "site-packages",
    ".eggs",
    ".idea",
    ".vscode",
    "third_party",
    ".ipynb_checkpoints",
}
# Path components that rarely hold the golden path. Ranked last, and excluded
# from archetype evidence.
NOISE_DIRS = {
    "tests",
    "test",
    "benchmarks",
    "benchmark",
    "examples",
    "example",
    "docs",
    "doc",
    "tools",
    "ci",
    "archive",
    "deprecated",
    "legacy",
    "vendor",
    "vendored",
    "external",
    "extern",
    "3rdparty",
}
LANG_EXT = {
    ".py": "python",
    ".pyx": "cython",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cc": "c++",
    ".cpp": "c++",
    ".h": "c/c++ header",
    ".hpp": "c/c++ header",
    ".c": "c",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".scala": "scala",
    ".ts": "typescript",
    ".js": "javascript",
    ".sh": "shell",
    ".ipynb": "notebook",
}
CONFIG_EXT = {".yaml", ".yml", ".toml", ".json", ".pbtxt", ".cfg", ".ini"}

FRAMEWORKS = {
    "torch": "pytorch",
    "lightning": "lightning",
    "pytorch_lightning": "lightning",
    "jax": "jax",
    "flax": "jax",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "autogluon": "autogluon",
    "pyspark": "spark",
    "ray": "ray",
    "dask": "dask",
    "polars": "polars",
    "pandas": "pandas",
    "pyarrow": "arrow",
    "apache_beam": "beam",
    "pyflink": "flink",
    "kafka": "kafka",
    "confluent_kafka": "kafka",
    "feast": "feast",
    "transformers": "hf-transformers",
    "datasets": "hf-datasets",
    "tokenizers": "hf-tokenizers",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
    "deepspeed": "deepspeed",
    "megatron": "megatron",
    "openrlhf": "openrlhf",
    "verl": "verl",
    "vllm": "vllm",
    "sglang": "sglang",
    "tensorrt_llm": "tensorrt-llm",
    "lmdeploy": "lmdeploy",
    "triton": "triton-lang",
    "tritonclient": "triton-server",
    "onnx": "onnx",
    "onnxruntime": "onnx",
    "tensorrt": "tensorrt",
    "fastapi": "fastapi",
    "flask": "flask",
    "grpc": "grpc",
    "zmq": "zmq",
    "uvicorn": "uvicorn",
    "starlette": "starlette",
    "sagemaker": "sagemaker",
    "boto3": "aws",
    "mlflow": "mlflow",
    "wandb": "wandb",
    "hydra": "hydra",
    "omegaconf": "hydra",
    "langchain": "langchain",
    "llama_index": "llama-index",
    "dspy": "dspy",
    "faiss": "faiss",
    "chromadb": "chroma",
    "sentence_transformers": "sbert",
    "airflow": "airflow",
    "kfp": "kubeflow",
    "prefect": "prefect",
    "kserve": "kserve",
    "bentoml": "bentoml",
}

# Canonical role names per archetype. A class matches when its name equals the
# role or ends with it (GPUModelRunner matches ModelRunner).
ROLES = {
    "D. LLM inference/serving engine": [
        "AsyncLLM",
        "LLMEngine",
        "EngineCore",
        "EngineCoreProc",
        "Scheduler",
        "KVCacheManager",
        "BlockManager",
        "BlockPool",
        "RadixCache",
        "TokenizerManager",
        "DetokenizerManager",
        "InputProcessor",
        "OutputProcessor",
        "ModelRunner",
        "Worker",
        "Executor",
        "Sampler",
        "AttentionBackend",
    ],
    "B. Deep learning training": [
        "Trainer",
        "Dataset",
        "DataModule",
        "LightningModule",
        "Loss",
        "Optimizer",
        "LRScheduler",
        "Callback",
    ],
    "C. LLM pretrain/fine-tune/post-train": [
        "PPOTrainer",
        "GRPOTrainer",
        "DPOTrainer",
        "SFTTrainer",
        "RewardModel",
        "Critic",
        "Actor",
        "RolloutWorker",
        "ReferenceModel",
    ],
    "E. Model serving platform/endpoint": [
        "Handler",
        "Predictor",
        "TritonPythonModel",
        "InferenceService",
        "Router",
        "Server",
    ],
    "F. Data/feature pipeline": [
        "Pipeline",
        "Transformer",
        "FeatureStore",
        "Reader",
        "Writer",
        "Sink",
        "Source",
    ],
    "G. LLM app: RAG/agents": [
        "Retriever",
        "VectorStore",
        "Agent",
        "Tool",
        "Chain",
        "Reranker",
        "Embedder",
    ],
}
ARCHETYPE_FRAMEWORKS = {
    "A. Tabular/classic ML + MLOps": {
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "autogluon",
        "sagemaker",
        "mlflow",
        "airflow",
        "kubeflow",
        "prefect",
    },
    "B. Deep learning training": {"pytorch", "jax", "tensorflow", "lightning"},
    "C. LLM pretrain/fine-tune/post-train": {
        "megatron",
        "deepspeed",
        "trl",
        "peft",
        "openrlhf",
        "verl",
        "accelerate",
    },
    "D. LLM inference/serving engine": {"vllm", "sglang", "tensorrt-llm", "lmdeploy"},
    "E. Model serving platform/endpoint": {
        "fastapi",
        "flask",
        "triton-server",
        "kserve",
        "bentoml",
        "grpc",
    },
    "F. Data/feature pipeline": {"spark", "beam", "flink", "kafka", "dask", "feast"},
    "G. LLM app: RAG/agents": {"langchain", "llama-index", "dspy", "faiss", "chroma"},
    "H. Kernels/compilers": {"triton-lang", "tensorrt"},
}

# Broader coverage: more frameworks, and archetypes I-N.
FRAMEWORKS.update(
    {
        "optax": "jax",
        "orbax": "jax",
        "haiku": "jax",
        "equinox": "jax",
        "chex": "jax",
        "tensorflow_serving": "tf-serving",
        "tf_keras": "tensorflow",
        "gradio": "gradio",
        "streamlit": "streamlit",
        "celery": "celery",
        "redis": "redis",
        "rq": "rq",
        "cupy": "cupy",
        "numba": "numba",
        "pycuda": "pycuda",
        "cutlass": "cutlass",
        "torchrec": "torchrec",
        "tensorflow_recommenders": "tfrs",
        "merlin": "merlin",
        "implicit": "implicit",
        "lightfm": "lightfm",
        "recbole": "recbole",
        "torchvision": "torchvision",
        "timm": "timm",
        "detectron2": "detectron2",
        "mmdet": "mmdet",
        "mmcv": "mmcv",
        "ultralytics": "ultralytics",
        "segment_anything": "sam",
        "diffusers": "diffusers",
        "kornia": "kornia",
        "albumentations": "albumentations",
        "cv2": "opencv",
        "torch_tensorrt": "tensorrt",
        "openvino": "openvino",
        "coremltools": "coreml",
        "tvm": "tvm",
        "bitsandbytes": "bitsandbytes",
        "auto_gptq": "gptq",
        "awq": "awq",
        "optimum": "optimum",
        "dvc": "dvc",
        "comet_ml": "comet",
        "neptune": "neptune",
        "clearml": "clearml",
        "zenml": "zenml",
        "metaflow": "metaflow",
        "optuna": "optuna",
        "lm_eval": "lm-eval",
        "evaluate": "hf-evaluate",
        "ragas": "ragas",
        "deepeval": "deepeval",
        "opencompass": "opencompass",
        "statsmodels": "statsmodels",
        "prophet": "prophet",
        "sktime": "sktime",
        "darts": "darts",
        "gluonts": "gluonts",
        "neuralforecast": "neuralforecast",
        "pytorch_forecasting": "pytorch-forecasting",
        "tsfresh": "tsfresh",
    }
)
ARCHETYPE_FRAMEWORKS["E. Model serving platform/endpoint"] |= {
    "tf-serving",
    "gradio",
    "streamlit",
    "celery",
    "rq",
}
ARCHETYPE_FRAMEWORKS["H. Kernels/compilers"] |= {"cupy", "numba", "pycuda", "cutlass"}
ARCHETYPE_FRAMEWORKS.update(
    {
        "I. Recommenders/ranking": {
            "torchrec",
            "tfrs",
            "merlin",
            "implicit",
            "lightfm",
            "recbole",
        },
        "J. Computer vision/diffusion": {
            "torchvision",
            "timm",
            "detectron2",
            "mmdet",
            "mmcv",
            "ultralytics",
            "sam",
            "diffusers",
            "kornia",
            "albumentations",
            "opencv",
        },
        "K. Export/compile/quantize": {
            "onnx",
            "tensorrt",
            "openvino",
            "coreml",
            "tvm",
            "bitsandbytes",
            "gptq",
            "awq",
            "optimum",
        },
        "L. Experiment tracking/MLOps": {
            "mlflow",
            "wandb",
            "dvc",
            "comet",
            "neptune",
            "clearml",
            "zenml",
            "metaflow",
            "optuna",
            "feast",
        },
        "M. Eval harness/benchmarks": {
            "lm-eval",
            "hf-evaluate",
            "ragas",
            "deepeval",
            "opencompass",
        },
        "N. Time series/forecasting": {
            "statsmodels",
            "prophet",
            "sktime",
            "darts",
            "gluonts",
            "neuralforecast",
            "pytorch-forecasting",
            "tsfresh",
        },
    }
)
# Robotics, realtime streaming, and media libraries.
FRAMEWORKS.update(
    {
        "lerobot": "lerobot",
        "gymnasium": "gym",
        "gym": "gym",
        "mujoco": "mujoco",
        "isaacgym": "isaac",
        "isaaclab": "isaac",
        "omni": "isaac",
        "pybullet": "pybullet",
        "dm_control": "dm-control",
        "robosuite": "robosuite",
        "rclpy": "ros",
        "rospy": "ros",
        "mcap": "mcap",
        "rosbag": "ros",
        "rlds": "rlds",
        "zarr": "zarr",
        "aiortc": "webrtc",
        "livekit": "livekit",
        "pipecat": "pipecat",
        "websockets": "websockets",
        "av": "pyav",
        "ffmpeg": "ffmpeg",
        "gi": "gstreamer",
        "webrtcvad": "vad",
        "silero_vad": "vad",
        "torchaudio": "torchaudio",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "whisper": "whisper",
        "faster_whisper": "whisper",
        "decord": "decord",
        "webdataset": "webdataset",
    }
)
ARCHETYPE_FRAMEWORKS.update(
    {
        "O. Robotics/embodied learning": {
            "lerobot",
            "gym",
            "mujoco",
            "isaac",
            "pybullet",
            "dm-control",
            "robosuite",
            "ros",
            "mcap",
            "rlds",
        },
        "P. Realtime/streaming AI service": {
            "webrtc",
            "livekit",
            "pipecat",
            "websockets",
            "pyav",
            "ffmpeg",
            "gstreamer",
            "vad",
            "whisper",
        },
    }
)
ROLES["O. Robotics/embodied learning"] = [
    "Policy",
    "Env",
    "Robot",
    "Teleop",
    "Episode",
    "ReplayBuffer",
]
ROLES["P. Realtime/streaming AI service"] = [
    "Session",
    "Track",
    "AudioStream",
    "VideoStream",
    "VAD",
    "Transport",
]

# SQL/warehouse, labeling and data quality, privacy, cluster tooling, and
# labels that only build files (Cargo.toml, go.mod, CMake, sbt) provide.
FRAMEWORKS.update(
    {
        "dbt": "dbt",
        "snowflake": "snowflake",
        "sqlalchemy": "sqlalchemy",
        "duckdb": "duckdb",
        "sqlglot": "sqlglot",
        "pyathena": "athena",
        "psycopg2": "postgres",
        "psycopg": "postgres",
        "label_studio_sdk": "label-studio",
        "label_studio": "label-studio",
        "argilla": "argilla",
        "cleanlab": "cleanlab",
        "snorkel": "snorkel",
        "prodigy": "prodigy",
        "labelbox": "labelbox",
        "fiftyone": "fiftyone",
        "great_expectations": "great-expectations",
        "pandera": "pandera",
        "deepchecks": "deepchecks",
        "evidently": "evidently",
        "whylogs": "whylogs",
        "presidio_analyzer": "presidio",
        "presidio_anonymizer": "presidio",
        "scrubadub": "scrubadub",
        "opacus": "opacus",
        "submitit": "submitit",
        "sky": "skypilot",
        "kubernetes": "k8s-client",
    }
)
ARCHETYPE_FRAMEWORKS["D. LLM inference/serving engine"] |= {"ggml", "candle"}
ARCHETYPE_FRAMEWORKS["E. Model serving platform/endpoint"] |= {"rust-web", "go-web"}
ARCHETYPE_FRAMEWORKS["F. Data/feature pipeline"] |= {"flink", "kafka"}
ARCHETYPE_FRAMEWORKS["H. Kernels/compilers"] |= {"cuda"}
ARCHETYPE_FRAMEWORKS.update(
    {
        "Q. SQL/warehouse/dbt pipeline": {
            "dbt",
            "snowflake",
            "sqlalchemy",
            "duckdb",
            "sqlglot",
            "athena",
            "postgres",
        },
        "R. Labeling/annotation/data quality": {
            "label-studio",
            "argilla",
            "cleanlab",
            "snorkel",
            "prodigy",
            "labelbox",
            "fiftyone",
            "great-expectations",
            "pandera",
            "deepchecks",
            "evidently",
            "whylogs",
        },
    }
)
ROLES["R. Labeling/annotation/data quality"] = [
    "Annotation",
    "Annotator",
    "LabelingTask",
    "Expectation",
    "DataValidator",
]

ROLES["C. LLM pretrain/fine-tune/post-train"] += [
    "CausalSelfAttention",
    "ForCausalLM",
    "CausalLM",
    "DecoderLayer",
    "GPT",
]
ROLES.update(
    {
        "I. Recommenders/ranking": [
            "TwoTower",
            "Ranker",
            "Recommender",
            "CandidateGenerator",
            "EmbeddingBagCollection",
            "DLRM",
            "DeepFM",
            "DCN",
        ],
        "J. Computer vision/diffusion": [
            "Backbone",
            "Neck",
            "Detector",
            "UNet",
            "UNet2DConditionModel",
            "AutoencoderKL",
            "DiffusionPipeline",
            "VisionTransformer",
            "ResNet",
        ],
        "K. Export/compile/quantize": [
            "Exporter",
            "Quantizer",
            "Calibrator",
            "Compiler",
        ],
        "L. Experiment tracking/MLOps": [
            "Tracker",
            "Registry",
            "FeatureStore",
            "Experiment",
        ],
        "M. Eval harness/benchmarks": [
            "Task",
            "Evaluator",
            "Metric",
            "Benchmark",
            "Harness",
        ],
        "N. Time series/forecasting": [
            "Forecaster",
            "Forecasting",
            "Backtest",
            "TimeSeries",
        ],
    }
)

INTERESTING_BASES = {
    "Module": "nn.Module",
    "nn.Module": "nn.Module",
    "torch.nn.Module": "nn.Module",
    "Dataset": "Dataset",
    "IterableDataset": "Dataset",
    "LightningModule": "LightningModule",
    "LightningDataModule": "LightningDataModule",
    "PreTrainedModel": "PreTrainedModel",
}
SERVING_FUNCS = {"model_fn", "input_fn", "predict_fn", "output_fn", "transform_fn"}
CLI_MARKERS = re.compile(
    r"\b(argparse\.ArgumentParser|click\.command|typer\.Typer|fire\.Fire|"
    r"@hydra\.main|HfArgumentParser|jsonargparse)\b"
)
MAIN_GUARD = re.compile(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", re.M)
TRAIN_RE = re.compile(
    r"\.backward\(|optimizer\.step\(|\.zero_grad\(|accelerator\.backward"
    r"|GradientTape\(|apply_gradients\(|value_and_grad\(|jax\.grad\(|optax\.\w+\("
    r"|\.fit\([^)]*epochs\s*="
)
GPU_MARKERS = re.compile(r"(\bcuda\b|\bgpu\b|requires_gpu)", re.I)


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for name in filenames:
            yield Path(dirpath) / name


def is_noise(rel: str) -> bool:
    parts = rel.split("/")[:-1]
    return any(p.startswith(".") or p.lower() in NOISE_DIRS for p in parts)


def rank_paths(paths):
    return sorted(paths, key=lambda p: (is_noise(p), p.count("/"), p))


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def import_roots(tree: ast.AST):
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module)
    return roots


def base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{base_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    return ""


def is_script_style(tree: ast.Module) -> bool:
    """True for files that do real work at module level without a main guard."""
    work = 0
    for node in tree.body:
        if isinstance(node, (ast.For, ast.While, ast.With, ast.Try, ast.If)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ):
            work += 1
    return work >= 3


def is_test_file(rel: str) -> bool:
    name = os.path.basename(rel)
    return (
        (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "/tests/" in f"/{rel}"
    )


def own_packages(root: Path):
    names = set()
    for base in (root, root / "src"):
        if base.is_dir():
            for child in base.iterdir():
                if (
                    child.is_dir()
                    and (child / "__init__.py").exists()
                    and child.name.lower() not in NOISE_DIRS
                ):
                    names.add(child.name)
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        m = re.search(
            r'^name\s*=\s*"([^"]+)"', pyproject.read_text(errors="ignore"), re.M
        )
        if m:
            names.add(m.group(1).replace("-", "_"))
    return names


def entry_points_from_packaging(root: Path):
    found = []
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(errors="ignore")
        m = re.search(r"^\[project\.scripts\]\s*\n(.*?)(^\[|\Z)", text, re.S | re.M)
        if m:
            for line in m.group(1).splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    found.append(("pyproject [project.scripts]", line.strip()))
    for name in ("setup.py", "setup.cfg"):
        path = root / name
        if path.exists():
            text = path.read_text(errors="ignore")
            if "console_scripts" in text:
                for m in re.finditer(r"['\"]?([\w\-]+\s*=\s*[\w\.]+:\w+)['\"]?", text):
                    found.append((name, m.group(1)))
    return found


def docker_entry(path: Path):
    out = []
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if re.match(r"^\s*(ENTRYPOINT|CMD)\b", line):
                out.append(line.strip()[:160])
    except OSError:
        pass
    return out


def scan(root: Path, top: int):
    root = root.resolve()
    own = own_packages(root)
    lang_loc, dir_py_loc, ref_files = Counter(), Counter(), Counter()
    fw_all, fw_core = defaultdict(set), defaultdict(set)
    all_classes, serving_funcs, train_files = [], [], []
    main_scripts, script_files, cli_files = [], [], []
    tests, configs, dockerfiles, docs = [], [], [], []
    test_funcs = gpu_tests = notebooks = py_files = cuda_files = 0

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        ext, name = path.suffix.lower(), path.name
        if name.startswith("Dockerfile") or name.endswith(".Dockerfile"):
            dockerfiles.append((rel, docker_entry(path)))
        if ext in (".md", ".rst") and (
            rel.lower().startswith("docs/")
            or name.lower().startswith("readme")
            or "architecture" in name.lower()
            or "design" in rel.lower()
        ):
            docs.append(rel)
        python_config = ext == '.py' and (
            any(p.lower() in {'config', 'configs', 'conf'} for p in path.relative_to(root).parts[:-1])
            or name.lower() in {'config.py', 'configuration.py'}
        )
        if python_config or ext in CONFIG_EXT and (
            "config" in rel.lower()
            or "conf/" in rel.lower()
            or name in ("pyproject.toml", "setup.cfg", "pytest.ini")
            or name.endswith(".pbtxt")
            or "pipeline" in name.lower()
        ):
            configs.append(rel)
        lang = LANG_EXT.get(ext)
        if not lang:
            continue
        if lang == "notebook":
            notebooks += 1
            continue
        loc = count_lines(path)
        lang_loc[lang] += loc
        if lang == "cuda" and not is_noise(rel):
            cuda_files += 1
        if lang != "python":
            continue
        py_files += 1
        dir_py_loc[rel.split("/")[0] if "/" in rel else "."] += loc
        try:
            src = path.read_text(errors="ignore")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, OSError, RecursionError):
            continue
        noise = is_noise(rel)
        for mod in import_roots(tree):
            r = mod.split(".")[0]
            if r in FRAMEWORKS and r not in own:
                fw_all[FRAMEWORKS[r]].add(rel)
                if not noise:
                    fw_core[FRAMEWORKS[r]].add(rel)
        if is_test_file(rel):
            if name != "conftest.py":
                n = sum(
                    1
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")
                )
                test_funcs += n
                tests.append((rel, n))
                gpu_tests += bool(GPU_MARKERS.search(src))
            continue
        if not noise and TRAIN_RE.search(src):
            train_files.append(rel)
        if MAIN_GUARD.search(src):
            main_scripts.append(rel)
        elif is_script_style(tree):
            script_files.append(rel)
        if CLI_MARKERS.search(src):
            cli_files.append(rel)
        idents = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                idents.add(node.id)
            elif isinstance(node, ast.Attribute):
                idents.add(node.attr)
            elif isinstance(node, ast.ClassDef):
                all_classes.append(
                    (node.name, rel, node.lineno, [base_name(b) for b in node.bases])
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in SERVING_FUNCS and node.col_offset == 0:
                    serving_funcs.append((node.name, rel, node.lineno))
        ref_files.update(idents)

    # Base-class kinds, inherited through repo-local subclasses.
    kind_of = defaultdict(set)
    for cname, _, _, bases in all_classes:
        for b in bases:
            k = INTERESTING_BASES.get(b) or INTERESTING_BASES.get(b.split(".")[-1])
            if k:
                kind_of[cname].add(k)
    changed = True
    while changed:
        changed = False
        for cname, _, _, bases in all_classes:
            for b in bases:
                extra = kind_of.get(b.split(".")[-1], set()) - kind_of[cname]
                if extra:
                    kind_of[cname] |= extra
                    changed = True

    def class_key(c):
        return (is_noise(c[1]), -ref_files[c[0]], c[1].count("/"), c[1], c[2])

    key_classes = defaultdict(list)
    for c in all_classes:
        for k in kind_of.get(c[0], ()):
            key_classes[k].append(c)

    # Canonical roles: exact name first, then suffix match; core paths first.
    roles = {}
    role_hits = Counter()
    for arch, role_names in ROLES.items():
        found = {}
        for role in role_names:
            exact = [c for c in all_classes if c[0] == role]
            suffix = [c for c in all_classes if c[0] != role and c[0].endswith(role)]
            hits = sorted(exact, key=class_key) + sorted(suffix, key=class_key)
            hits = [c for c in hits if not is_noise(c[1])]
            if hits:
                found[role] = hits[:3]
                role_hits[arch] += 1 if exact else 0.5
        if found:
            roles[arch] = found

    # Second pass: notebooks, non-Python code, SQL/dbt, infra. It never edits
    # the Python results above. Notebook code joins the scores only when the
    # repo is notebook-heavy (more notebook code than .py code).
    extras = scan_extras.scan_extras(
        root,
        top,
        own,
        iter_files,
        is_noise,
        rank_paths,
        import_roots,
        FRAMEWORKS,
        TRAIN_RE,
        lang_loc.get("python", 0),
    )
    nb_fw, nb_train = extras.pop("_nb_fw"), extras.pop("_nb_train")
    if extras["notebooks"]["heavy"]:
        for fw, files in nb_fw.items():
            fw_core[fw] |= {f for f in files if not is_noise(f)}
            fw_all[fw] |= files
        train_files = train_files + nb_train
    dep_labels = extras["dep_labels"]
    engine_types = extras.pop("_engine_type_count")

    # Archetype scores with evidence (core files only).
    arch_scores = []
    # A repo that *is* a known framework (darts, diffusers, torchrec) imports
    # itself, which own-package filtering hides. Count that as strong evidence.
    own_fw = {FRAMEWORKS[p]: p for p in own if p in FRAMEWORKS}
    for arch, fws in ARCHETYPE_FRAMEWORKS.items():
        fw_ev = {fw: len(fw_core[fw]) for fw in fws if fw_core.get(fw)}
        # Generic role names (Tool, Source, Handler) only count with framework
        # evidence. LLM-shaped names (CausalSelfAttention, ForCausalLM) are
        # specific. Serving-engine names need several distinct hits, because a
        # diffusion repo also has noise "Scheduler" classes.
        engine_like = arch.startswith("D.") and role_hits[arch] >= 4
        specific = arch.startswith("C.")
        role_weight = role_hits[arch] if (fw_ev or engine_like or specific) else 0
        score = sum(1 + min(n, 50) / 10 for n in fw_ev.values()) + role_weight * 2
        ev = [
            f"{fw} in {n} files"
            for fw, n in sorted(fw_ev.items(), key=lambda x: (-x[1], x[0]))
        ]
        for fw in fws & set(own_fw):
            score += 10
            ev.insert(0, f"this repo is the `{own_fw[fw]}` package")
        if role_weight:
            ev.append(f"{len(roles.get(arch, {}))} canonical role classes")
        if arch.startswith(("B.", "C.")):
            if not train_files:
                score, ev = 0, ev + ["no training calls outside tests"]
            elif arch.startswith("C.") and not (fw_ev or role_hits[arch]):
                score, ev = (
                    0,
                    ev + ["training code, but no LLM framework or LLM-shaped classes"],
                )
            else:
                score += min(len(train_files), 20) / 2
                ev.append(f"training code in {len(train_files)} files")
        if arch.startswith("A.") and serving_funcs:
            score += 1
            ev.append("SageMaker-style serving hooks")
        if arch.startswith("E.") and serving_funcs:
            score += 3
            ev.append(f"{len(serving_funcs)} serving hooks (model_fn/...)")
        if arch.startswith("F.") and fw_core.get("ray"):
            ev.append("ray present: check for ray.data")
        if arch.startswith("H.") and cuda_files:
            score += min(cuda_files, 50) / 10
            ev.append(f"{cuda_files} .cu/.cuh files")
        gated = arch.startswith(("B.", "C.")) and score == 0  # no training code
        for fw in sorted(fws & set(dep_labels)):
            if gated:
                break
            score += 2
            ev.append(f"{fw} in build files")
        if arch.startswith("D.") and engine_types >= 3:
            score += 4
            ev.append(f"{engine_types} serving-engine types in non-Python code")
        if arch.startswith("Q."):
            dbt = extras["sql"]["dbt"]
            if dbt.get("projects"):
                score += 10
                ev.insert(0, f"dbt project, {dbt['models']} models")
            if extras["sql"]["files"] >= 10:
                score += min(extras["sql"]["files"], 50) / 10
                ev.append(f"{extras['sql']['files']} .sql files")
        if score >= 1:
            arch_scores.append((round(score, 1), arch, ev))
    arch_scores.sort(reverse=True)

    py_loc = lang_loc.get("python", 0)
    tier = "small" if py_loc < 5000 else "medium" if py_loc < 50000 else "large"

    def cls(c):
        return {
            "name": c[0],
            "file": c[1],
            "line": c[2],
            "bases": c[3],
            "ref_files": ref_files[c[0]],
        }

    return {
        "repo": str(root),
        "own_packages": sorted(own),
        "size": {
            "python_files": py_files,
            "python_loc": py_loc,
            "tier": tier,
            "loc_by_language": dict(lang_loc.most_common()),
            "notebooks": notebooks,
            "python_loc_by_top_dir": dict(dir_py_loc.most_common(top)),
        },
        "frameworks": {
            k: {"files": len(v), "core_files": len(fw_core.get(k, ()))}
            for k, v in sorted(fw_all.items(), key=lambda kv: -len(kv[1]))
        },
        "archetypes": [
            {
                "score": s,
                "archetype": a,
                "rank": "primary" if s >= 0.5 * arch_scores[0][0] else "secondary",
                "evidence": e,
            }
            for s, a, e in arch_scores
        ],
        "canonical_roles": {
            arch: {role: [cls(c) for c in hits] for role, hits in found.items()}
            for arch, found in roles.items()
        },
        "entry_points": {
            "packaging": entry_points_from_packaging(root),
            "main_guard_scripts": rank_paths(main_scripts)[:top],
            "script_style_files": rank_paths(script_files)[:top],
            "cli_parser_files": rank_paths(cli_files)[:top],
            "serving_functions": serving_funcs[:top],
            "dockerfiles": sorted(dockerfiles, key=lambda d: (is_noise(d[0]), d[0]))[
                :top
            ],
            "training_code_files": rank_paths(train_files)[:top],
        },
        "key_classes": {
            k: [cls(c) for c in sorted(v, key=class_key)[:top]]
            for k, v in sorted(key_classes.items())
        },
        "tests": {
            "files": len(tests),
            "test_functions": test_funcs,
            "files_with_gpu_markers": gpu_tests,
            "largest": sorted(tests, key=lambda t: -t[1])[:top],
            "pytest_config": [
                c
                for c in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
                if (root / c).exists()
                and "pytest" in (root / c).read_text(errors="ignore")
            ],
        },
        "configs": rank_paths(configs)[: top * 2],
        "docs": rank_paths(docs)[: top * 2],
        "extras": extras,
    }


def extras_markdown(x: dict) -> list:
    """Report sections for the second pass. Empty sections are left out."""
    out = []
    nb = x.get("notebooks", {})
    if nb.get("count"):
        out.append("## Notebooks")
        heavy = (
            " Notebook-heavy: notebook code counts toward the repo type."
            if nb["heavy"]
            else ""
        )
        out.append(
            f"{nb['count']} notebooks, {nb['code_loc']:,} lines of code in code cells.{heavy}"
        )
        rows = nb["with_logic"] or nb["top"]
        label = (
            "with functions, classes, or training code"
            if nb["with_logic"]
            else "largest"
        )
        out.append(f"Notebooks {label}:")
        for row in rows:
            flags = ", ".join(
                [f"{row['defs']} defs"] * bool(row["defs"])
                + ["training"] * row["training"]
                + [", ".join(row["frameworks"])] * bool(row["frameworks"])
            )
            out.append(
                f"- `{row['path']}`: {row['code_cells']} code cells, {row['code_loc']} lines ({flags or 'no defs'})"
            )
        out.append("")
    nps = x.get("non_python", {})
    if nps:
        out.append("## Non-Python code")
        for fam, d in nps.items():
            out.append(
                f"### {fam}: {d['files']} files, {d['loc']:,} lines, "
                f"{d['test_files']} test files ({d['test_cases']} test cases found by name)"
            )
            if d["entry_points"]:
                out.append(
                    "- entry points: " + ", ".join(f"`{p}`" for p in d["entry_points"])
                )
            for t in d["key_types"]:
                out.append(
                    f"- `{t['name']}` `{t['file']}` (referenced in {t['ref_files']} files)"
                )
        if x.get("bindings"):
            out.append("### Language boundaries (bindings)")
            for label, files in x["bindings"].items():
                out.append(f"- {label}: " + ", ".join(f"`{f}`" for f in files))
        if x.get("build_files"):
            out.append(
                "### Build files: " + ", ".join(f"`{f}`" for f in x["build_files"])
            )
        if x.get("dep_labels"):
            out.append(
                "### Dependencies from build files: "
                + ", ".join(f"{k} ({v})" for k, v in x["dep_labels"].items())
            )
        out.append("")
    sql = x.get("sql", {})
    if sql.get("files") or sql.get("dbt"):
        out.append("## SQL and dbt")
        out.append(f"{sql['files']} .sql files.")
        dbt = sql.get("dbt") or {}
        if dbt:
            out.append(
                f"dbt: {', '.join(f'`{p}`' for p in dbt['projects']) or 'models without dbt_project.yml'}, "
                f"{dbt['models']} models, {dbt['ref_edges']} ref() edges, "
                f"{dbt['source_calls']} source() calls, {dbt['schema_tests']} schema tests"
            )
            for m in dbt["most_referenced"]:
                out.append(
                    f"- `{m['model']}` referenced {m['refs']} times `{m['file']}`"
                )
        out.append("")
    infra = x.get("infra", {})
    if any(
        infra.get(k)
        for k in (
            "slurm_scripts",
            "launchers",
            "k8s_kinds",
            "helm_charts",
            "terraform_files",
        )
    ):
        out.append("## Infra signals")
        if infra["slurm_scripts"]:
            out.append(
                "- Slurm scripts: "
                + ", ".join(f"`{p}`" for p in infra["slurm_scripts"])
            )
        if infra["launchers"]:
            out.append(
                "- launch commands: "
                + ", ".join(f"{k} ({v})" for k, v in infra["launchers"].items())
            )
        if infra["k8s_kinds"]:
            out.append(
                "- Kubernetes kinds: "
                + ", ".join(f"{k} ({v})" for k, v in infra["k8s_kinds"].items())
            )
        if infra["helm_charts"]:
            out.append(
                "- Helm charts: " + ", ".join(f"`{p}`" for p in infra["helm_charts"])
            )
        if infra["terraform_files"]:
            out.append(f"- Terraform files: {infra['terraform_files']}")
        out.append("")
    return out


def to_markdown(r: dict) -> str:
    s = r["size"]
    out = [f"# ML repo scan: `{r['repo']}`", ""]
    out.append(
        f"**Size tier:** {s['tier']} ({s['python_loc']:,} Python LOC in "
        f"{s['python_files']} files, {s['notebooks']} notebooks)"
    )
    out.append("")
    out.append(
        "**LOC by language:** "
        + ", ".join(f"{k} {v:,}" for k, v in s["loc_by_language"].items())
    )
    out.append("")
    out.append(
        "**Python LOC by top-level dir:** "
        + ", ".join(f"`{k}` {v:,}" for k, v in s["python_loc_by_top_dir"].items())
    )
    out.append("")
    if r["own_packages"]:
        out.append(
            "**Own packages (excluded from framework counts):** "
            + ", ".join(f"`{p}`" for p in r["own_packages"])
        )
        out.append("")
    out.append("## Archetypes (score, evidence). Confirm by reading entry points")
    for a in r["archetypes"] or [
        {"score": 0, "archetype": "none detected", "rank": "", "evidence": []}
    ]:
        out.append(
            f"- **{a['archetype']}** ({a['score']}, {a['rank']}): "
            + "; ".join(a["evidence"])
        )
    out.append("")
    out.append(
        "**Frameworks (files / outside tests, examples, docs):** "
        + (
            ", ".join(
                f"{k} {v['files']}/{v['core_files']}"
                for k, v in r["frameworks"].items()
            )
            or "none"
        )
    )
    out.append("")
    out.append("## Canonical role classes (start reading here)")
    for arch, found in r["canonical_roles"].items():
        out.append(f"### {arch}")
        for role, hits in found.items():
            locs = ", ".join(f"`{c['name']}` `{c['file']}:{c['line']}`" for c in hits)
            out.append(f"- {role}: {locs}")
    out.append("")
    ep = r["entry_points"]
    out.append("## Entry points (core paths first)")
    for src, val in ep["packaging"]:
        out.append(f"- {src}: `{val}`")
    labels = [
        ("main_guard_scripts", "`__main__` scripts"),
        ("script_style_files", "script-style files (module-level work, no main guard)"),
        ("cli_parser_files", "CLI parsers"),
        ("training_code_files", "training code (backward/optimizer.step)"),
    ]
    for key, label in labels:
        if ep[key]:
            out.append(f"- {label}: " + ", ".join(f"`{p}`" for p in ep[key]))
    for n, f, ln in ep["serving_functions"]:
        out.append(f"- serving hook `{n}` at `{f}:{ln}`")
    for f, cmds in ep["dockerfiles"]:
        out.append(f"- Dockerfile `{f}`: " + ("; ".join(cmds) or "no ENTRYPOINT/CMD"))
    out.append("")
    out.append("## Key classes by base (most-referenced first)")
    for kind, items in r["key_classes"].items():
        out.append(f"### {kind}")
        for c in items:
            out.append(
                f"- `{c['name']}({', '.join(c['bases'])})` `{c['file']}:{c['line']}` "
                f"(referenced in {c['ref_files']} files)"
            )
    out.append("")
    t = r["tests"]
    out.append("## Tests")
    out.append(
        f"{t['files']} test files, {t['test_functions']} test functions, "
        f"{t['files_with_gpu_markers']} files mention cuda/gpu. "
        f"pytest config in: {', '.join(t['pytest_config']) or 'none'}"
    )
    for f, n in t["largest"]:
        out.append(f"- `{f}`: {n}")
    out.append("")
    out.append("## Configs")
    out.extend([f"- `{c}`" for c in r["configs"]] or ["none"])
    out.append("")
    out.extend(extras_markdown(r.get("extras") or {}))
    out.append("## Existing docs")
    out.extend([f"- `{d}`" for d in r["docs"]] or ["none"])
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--top", type=int, default=15, help="items per list")
    args = ap.parse_args(argv)
    root = Path(args.repo)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    result = scan(root, args.top)
    print(
        json.dumps(result, indent=2, default=list) if args.json else to_markdown(result)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
