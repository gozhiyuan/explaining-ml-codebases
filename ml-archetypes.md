# ML repo archetypes

Classify the repo before planning the docs. A repo can match several
archetypes. Each one tells you which components you must find and explain,
and which questions the doc must answer.

`scan_ml_repo.py` scores each archetype from import, class-name, and file
signals, and marks each one `primary` or `secondary`. Plan the doc set around
the primary ones. Mention a secondary one only where the golden path touches
it. Treat all of them as hints, and confirm each by reading the entry points.

## Contents

- A. Tabular / classic ML pipeline and MLOps
- B. Deep learning training codebase
- C. LLM pretraining, fine-tuning, post-training (RL)
- D. LLM inference and serving engine
- E. Model serving platform or endpoint wrapper
- F. Data and feature pipeline (batch or streaming)
- G. LLM application: RAG, agents, tools
- H. Kernels and compilers
- I. Recommenders and ranking
- J. Computer vision and diffusion models
- K. Model export, compilation, quantization
- L. Experiment tracking and MLOps platforms
- M. Evaluation harnesses and benchmarks
- N. Time series and forecasting
- O. Robotics and embodied learning
- P. Realtime and streaming AI services
- Q. SQL, warehouse, and dbt pipelines
- R. Labeling, annotation, and data quality

Framework-specific detail (TensorFlow/Keras, JAX/Flax, CUDA and Triton
kernels, FastAPI model apps) lives in [framework-notes.md](framework-notes.md).
Read the section for each framework the scanner reports.

---

## A. Tabular / classic ML pipeline and MLOps

**Signals.** `sklearn`, `xgboost`, `lightgbm`, `catboost`, `autogluon`;
`sagemaker` pipeline steps, Airflow DAGs, Kubeflow, Vertex, MLflow;
`build_pipeline.py`, `pipeline.yaml`, per-environment configs; Lambda or cron triggers.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Trigger | Lambda, EventBridge, cron, Airflow sensor. What starts a run and with which parameters |
| Pipeline DAG | step definitions, their order, artifacts passed between steps |
| Labels | how the target is built, the time cut-off, leakage guards |
| Features | transforms, encoders, missing-value handling, the feature list contract |
| Split | random vs time vs group split, the split key, overlap assertions |
| Train | estimator, hyperparameters, class weights |
| Tune / calibrate | HPO search space, calibration (Platt, isotonic, thresholds) |
| Evaluate + gate | metrics, slices, the pass/fail rule that blocks promotion |
| Register + deploy | registry, artifact layout (`model.tar.gz`), endpoint config |
| Score | batch or real-time path, the request/response contract |

**Questions the doc must answer.** Which clock does each timestamp use? What
is pinned (package versions, feature version)? Where can train and serve
disagree? What happens on partial failure? How do you roll back?

**Running example.** One entity (one event key, one customer) with 2-3 rows. Show
its label, its features after each transform, and its final score.

---

## B. Deep learning training codebase

**Signals.** `torch`, `lightning`, `jax`/`flax`, `tensorflow`/`keras`;
`nn.Module` subclasses; `Dataset`/`DataLoader`; `Trainer` classes; `train.py`.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Config | dataclass, YAML, Hydra, argparse. The resolved values for the golden-path run |
| Data | `Dataset.__getitem__`, `collate_fn`, samplers, padding and masks |
| Model | the class tree from the factory (`make_model`) down to leaf layers |
| Loss | every term, its weight, its mask |
| Optimizer + schedule | optimizer, LR schedule, warmup, weight decay groups, grad clipping |
| Loop | forward, backward, step, accumulation, eval cadence, early stopping |
| Distributed | DDP, FSDP, DeepSpeed ZeRO stage, TP/PP, `torchrun` launch |
| Precision + speed | AMP dtype, `torch.compile`, CUDA graphs, activation checkpointing |
| Checkpoints | what is saved, how resume works, what ships to inference |

**Questions.** What are the tensor shapes at every layer? Which mask applies
where? Where does the GPU sit idle (Python loops, host syncs, data loading)?
How many parameters, and how much memory per GPU?

The table above uses PyTorch names. For TensorFlow/Keras (`tf.data`,
`model.fit`, `GradientTape`, `tf.distribute`, SavedModel) and JAX/Flax (`jit`,
pytrees, `shard_map`, Optax, Orbax), read the matching section of
[framework-notes.md](framework-notes.md). The components are the same. Where
they live and when they run differ.

**Running example.** One tiny batch, `B=2, T=4` with named dims, plus the real
production dims.

---

## C. LLM pretraining, fine-tuning, post-training (RL)

**Signals.** `megatron`, `torchtitan`, `deepspeed`, `accelerate`, `peft`,
`trl`, `openrlhf`, `verl`, `axolotl`, `llamafactory`; tokenizer training;
`GRPO`/`PPO`/`DPO` classes; rollout, reward, and critic modules.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Tokenize + pack | tokenizer, sequence packing, attention masks across packed documents, data mixture weights |
| Model | the decoder block (attention, MLP or MoE, norm, RoPE), weight tying, init |
| Parallelism | TP / PP / DP / CP / EP degrees, how layers are sharded, communication ops |
| Objective | CE for pretraining or SFT. DPO, PPO, or GRPO losses for post-training |
| RL loop | rollout engine (often vLLM or SGLang), reward model or rule, advantage estimate, reference model, KL term, policy update |
| Adapters | LoRA rank and target modules, merge step |
| Checkpoint conversion | HF <-> framework format, sharded weights |

**Math to show.** Next-token CE. The RL loss in use (for example the PPO
clipped objective, or GRPO's group-normalized advantage), each with a numeric
example. Memory per GPU for the golden-path parallel config.

**Running example.** One prompt, its tokens, one rollout, its reward, its
advantage, and the loss value for that sample.

---

## D. LLM inference and serving engine

**Signals.** `vllm`, `sglang`, `text-generation-inference`, `tensorrt_llm`,
`lmdeploy`, `llama.cpp`; classes named `Scheduler`, `EngineCore`,
`ModelRunner`, `KVCacheManager`, `BlockManager`, `RadixCache`,
`TokenizerManager`, `Detokenizer`; ZMQ or Ray IPC; OpenAI-compatible API server.

**Components to find.** Trace them in request order.

| Component | vLLM V1 example | SGLang example |
|-----------|-----------------|----------------|
| API frontend | OpenAI server, `AsyncLLM` | HTTP server, `Engine` |
| Input processing | input processor, tokenizer | `TokenizerManager` |
| IPC | ZMQ to `EngineCoreProc` | ZMQ to the scheduler subprocess |
| Engine loop | `EngineCore.step`: schedule, execute, update. `step_with_batch_queue` when async scheduling is on | `Scheduler.event_loop_normal` / `event_loop_overlap` |
| Scheduler | `Scheduler`, or `AsyncScheduler` when `async_scheduling` resolves on (check the config default). Waiting/running queues, token budget, chunked prefill, preemption -> `SchedulerOutput` | waiting queue, running batch, policy (FCFS, LPM) |
| KV cache | `KVCacheManager`, paged blocks, prefix-cache block hashes | `RadixCache` tree, token-to-KV pool |
| Executor + workers | TP/PP/DP workers, one `ModelRunner` per GPU | `TpModelWorker`, `ModelRunner` |
| Model runner | builds input tensors and attention metadata, CUDA graphs | `ForwardBatch`, CUDA graphs |
| Model | model class for the demo family, attention backend | same, `RadixAttention` layer |
| Sampler | logits processors, temperature/top-p, structured output, speculative decoding | same |
| Output | detokenize, stream, stop conditions | `DetokenizerManager` |
| Model loading | weight loader, quantization, sharding across TP ranks | same |

Confirm these names in the code at the explained commit. Engines rename
classes often.

**Questions.** How many processes run, and what does each own? What crosses
each process boundary? How does one engine step choose tokens per request?
When is KV memory allocated and freed? What does a prefix-cache hit change?
How are requests spread across data-parallel ranks?

**Diagrams.** A process diagram (Mermaid flowchart). A sequence diagram of one
request. An ASCII table of the KV blocks the running request takes at each step.

**Running example.** One prompt on one demo model. Show token IDs, the prefill
step, two decode steps, the block table, and the streamed text.

---

## E. Model serving platform or endpoint wrapper

**Signals.** `tritonclient`, `config.pbtxt`, KServe `InferenceService`,
`ray.serve`, TorchServe handlers, BentoML, SageMaker `model_fn`/`input_fn`/
`predict_fn`/`output_fn`, FastAPI or Flask apps, multi-model endpoints, async
inference queues. Batch scoring signals: SageMaker `Transformer` (Batch
Transform) or Processing jobs that load a model, Spark `mapInPandas` or
pandas UDFs that call a model, Ray Data `map_batches` with a model actor,
offline `LLM.generate` calls, and scheduled scoring DAGs. Name every serving
mode the repo has (playbook 12 starts with a table of modes).

**Components to find.** Request contract (schema, content type), routing to a
model and version, the model repository layout, handler stages, dynamic
batching, autoscaling, async queue and result location, health checks, and
timeouts.

**Questions.** What is the exact request and response? Which code runs once
per container and which runs once per request? What fails open and what fails
closed? Where is the latency spent?

**App-style services** (FastAPI, Flask, Gradio, Streamlit, Celery workers)
also need: the app factory and lifespan hook that loads the model, the
dependency-injection chain per route, sync versus async handlers and where a
blocking model call runs (event loop, threadpool, worker process), background
task queues, and the number of workers times the model memory each one holds.
See [framework-notes.md](framework-notes.md), "FastAPI model apps".

**Running example.** One real-shaped request body and its response.

---

## F. Data and feature pipeline (batch or streaming)

**Signals.** `pyspark`, `ray.data`, `dask`, `polars`, `apache_beam`,
`pyflink`, `kafka`, Feast, Delta or Iceberg tables; `applyInPandas`, UDFs.

**Components to find.** Sources and their partition layout. The driver code
versus the executor code. Each transform with its schema before and after.
Joins, shuffles, broadcasts. The action that runs the lazy plan. Caching and
idempotency. For streaming: windows, watermarks, late data, exactly-once.

**Questions.** What runs where (driver, executor, Python worker)? How many
tasks, and what is one task's input? Where does the data shuffle? What is the
output layout? How does a rerun avoid duplicates?

**Running example.** 2-3 input rows. Show them after each transform.

---

## G. LLM application: RAG, agents, tools

**Signals.** `langchain`, `llama_index`, `dspy`, `haystack`, vector stores
(FAISS, Chroma, pgvector), tool-calling schemas, prompt templates.

**Components to find.** Ingestion, chunking, embedding, indexing, retrieval,
reranking, prompt assembly, the LLM call, tool calls and their loop, output
parsing, guardrails, and the eval harness.

**Running example.** One user query. Show the retrieved chunks, the final
prompt, any tool calls, and the answer.

---

## H. Kernels and compilers

**Signals.** `.cu`/`.cuh` files, `triton.jit`, CUTLASS, CuPy, Numba,
`torch.utils.cpp_extension`, `TORCH_LIBRARY` registrations, `torch.compile`
backends, TVM.

Explain kernels only where the golden path calls them. For each one:

| Component | Look for |
|-----------|----------|
| Dispatch | the Python call, the op registration, and the C++/CUDA entry it lands in |
| Math | what the kernel computes, written as the plain PyTorch it replaces |
| Launch | grid and block (or Triton program ids and `BLOCK_*` constants), one tile's work |
| Memory | global -> shared -> registers, what is reused, the arithmetic intensity |
| Fusion | which ops are fused and the memory traffic saved |
| Oracle | the reference implementation the tests compare against, and the tolerance |

Link the reference (pure PyTorch) version if the repo has one. It is the test
oracle. Details are in [framework-notes.md](framework-notes.md), "CUDA and
Triton kernels".

---

## I. Recommenders and ranking

**Signals.** `torchrec`, `tensorflow_recommenders`, NVIDIA Merlin, `implicit`,
`lightfm`, `recbole`; `EmbeddingBag`, sharded embedding tables; classes named
`TwoTower`, `Ranker`, `CandidateGenerator`; ANN indexes (FAISS, ScaNN).

**Components to find.**

| Component | Look for |
|-----------|----------|
| Events and labels | impressions, clicks, purchases; how negatives are sampled; the label window |
| Features | user, item, and context features; ID features and their hashing or vocab |
| Embedding tables | table sizes, dims, sharding across devices (row, column, table-wise) |
| Retrieval | two-tower or matrix factorization, the ANN index, how many candidates |
| Ranking | the ranking model (DLRM, DCN, DeepFM), feature crosses, the loss |
| Re-ranking | business rules, diversity, dedup |
| Online path | feature fetch, retrieval, ranking, and the latency budget for each |
| Offline eval | recall@k, NDCG, AUC, and how they relate to the online metric |

**Questions.** How large is each embedding table and where does it live? How
are negatives chosen, and how does that bias the scores? Do the online and
offline feature values match?

**Math to show.** The retrieval score (dot product of the two towers), the
sampled-softmax or BCE loss with its negative sampling correction, and NDCG on
a 5-item example.

**Running example.** One user, 3 candidate items, their features, the two-tower
scores, the ranker scores, and the final order.

---

## J. Computer vision and diffusion models

**Signals.** `torchvision`, `timm`, `detectron2`, `mmdet`/`mmcv`, `ultralytics`,
`segment_anything`, `diffusers`, `kornia`, `albumentations`; classes named
`Backbone`, `Neck`, `Detector`, `UNet`, `VAE`, noise schedulers.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Data | decode, augmentation pipeline (order matters), normalization constants, batch layout NCHW or NHWC |
| Backbone | the network, its stages, the stride and channels at each stage |
| Neck and head | FPN or similar, anchors or queries, the head outputs per level |
| Loss | classification, box regression, mask, or the diffusion objective |
| Post-processing | NMS, thresholds, box decoding, or the sampler loop |
| Diffusion | the noise schedule, the denoiser (UNet or DiT), conditioning (text encoder, cross-attention), the latent VAE, classifier-free guidance |

**Math to show.** For detection: IoU and the box encoding. For diffusion: the
forward noising \(x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\),
the training loss on \(\epsilon\), one sampler step, and the guidance formula,
each with numbers at one timestep.

**Running example.** One image (state its size). Show the tensor shape after
each backbone stage, then the final boxes or the latent at three sampler steps.

---

## K. Model export, compilation, quantization

**Signals.** `torch.onnx`, `onnxruntime`, `tensorrt`, `torch_tensorrt`,
`openvino`, `coremltools`, TFLite converters, TorchScript (`torch.jit`),
`torch.export`, `torch.compile`, XLA, `bitsandbytes`, GPTQ/AWQ code,
`torch.ao.quantization`.

**Components to find.** The export entry, the input signature and dynamic
axes, the graph captured (and what breaks capture), op coverage and fallbacks,
the optimization passes, the precision of each layer, the calibration data
and method, the runtime that loads the artifact, and the parity check.

**Questions.** Which inputs are dynamic? Which ops fall back to a slower path?
How far do outputs drift from the original model, and what tolerance do tests
allow? What is the size and latency before and after?

**Math to show.** The quantization map \(q = \mathrm{round}(x / s) + z\) with
the scale and zero point computed for one tensor, and the error on 3 values.

**Running example.** One input pushed through the original and the exported
model, with both outputs and the max absolute difference.

---

## L. Experiment tracking and MLOps platforms

**Signals.** `mlflow`, `wandb`, `dvc`, `comet_ml`, `neptune`, `clearml`,
`zenml`, `metaflow`, `feast`; `MLproject`, `dvc.yaml`, model registry calls.

**Components to find.** What is logged per run (params, metrics, artifacts,
code version, data version), where it is stored, how a run gets a model
registered, the registry stages (staging, production) and who promotes, how a
deployment picks a model version, how data is versioned, and how a feature
store serves the same feature offline and online.

**Questions.** Given a production model, can you find the exact code, data,
and config that made it? What triggers a promotion? What does rollback change?

**Running example.** One run: its id, logged params and metrics, the artifact
path, the registered version, and the deployment that loads it.

---

## M. Evaluation harnesses and benchmarks

**Signals.** `lm_eval`, Hugging Face `evaluate`, `ragas`, `deepeval`,
OpenCompass, HELM; task registries, YAML task configs, classes named `Task`,
`Metric`, `Evaluator`, `Benchmark`.

**Components to find.** The task registry and config format, how one sample
becomes a prompt (templates, few-shot selection, chat formatting), the model
adapter (local, API, vLLM backend), the scoring method (log-likelihood,
generation then match, LLM judge), metric aggregation and stderr, caching, and
contamination checks.

**Questions.** What exactly is compared for one sample? How sensitive is the
score to the prompt template and few-shot seed? Which settings must match to
compare two numbers?

**Running example.** One sample: the raw item, the final prompt, the model
output or log-likelihoods, the per-sample score, and the aggregate.

---

## N. Time series and forecasting

**Signals.** `statsmodels`, `prophet`, `sktime`, `darts`, `gluonts`,
`neuralforecast`, `pytorch_forecasting`, `tsfresh`; classes named `Forecaster`,
backtest or cross-validation helpers.

**Components to find.** The series index and frequency, how missing and
irregular timestamps are handled, the window (lookback, horizon, stride), lag
and calendar features, scaling fitted per series or globally, the model, the
backtest split (expanding or sliding), and how forecasts are reconciled across
a hierarchy.

**Questions.** Can any feature see the future at prediction time? Which clock
and timezone define a day? How is the backtest split, and does it match how
the model runs in production?

**Math to show.** The loss or likelihood, a quantile (pinball) loss example,
and MAPE or sMAPE on 3 points.

**Running example.** One series with 10 points, the windows cut from it, one
forecast, and its error.

---

## O. Robotics and embodied learning

**Signals.** `lerobot`, `gymnasium`/`gym`, `mujoco`, `isaacgym`/Isaac Lab,
`pybullet`, `dm_control`, `robosuite`, ROS (`rclpy`, `rospy`), `mcap`,
`rosbag`, RLDS, `zarr`; classes named `Policy`, `Env`, `Robot`, `Teleop`.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Data collection | teleoperation or scripted collection, the recorder, the episode format on disk |
| Episode dataset | streams, rates, time sync, normalization statistics (see "Robotics and embodied episodes" in [modality-notes.md](modality-notes.md)) |
| Policy | architecture (ACT, diffusion policy, VLA, RL actor-critic), observation history, action chunking |
| Training | imitation (behavior cloning) or RL (rollouts, rewards, replay buffer), the loss |
| Simulation | simulator, environment, domain randomization, the sim-to-real gap |
| Deployment loop | the control loop rate, observation read, policy inference, action send, safety limits, the latency budget per tick |

**Questions.** How are camera and joint streams aligned in time? What frame
is each action in? What is the control rate, and does inference fit inside
one tick? What stops the robot when the policy fails?

**Diagrams.** A `sequenceDiagram` of one control tick (read sensors, infer,
send action) with timings. The architecture diagram of the policy (section 7
of the playbooks).

**Running example.** One episode and one timestep: every observation
stream's value and timestamp, the policy input tensors, the action chunk out.

---

## P. Realtime and streaming AI services

**Signals.** `aiortc`, `livekit`, `pipecat`, `websockets`, `fastapi`
WebSocket routes, `grpc` streaming, `av` (PyAV), `ffmpeg`, `gstreamer`,
`webrtcvad`/`silero` VAD, `torchaudio`, `librosa`, `faster_whisper`,
`whisper`; classes named `Session`, `Track`, `Stream`, `Pipeline`, `VAD`.

**Components to find.** The transport and signaling, the session state, the
media decode and resample path, chunking and VAD, the streaming model calls
(partial and final results), the output encode, backpressure, and the
latency budget. Follow "Realtime and streaming serving" in
[stage-playbooks.md](stage-playbooks.md).

**Questions.** What is the unit on the wire, and how big is one chunk for
the model? Where does latency go between the user speaking and the first
response? What happens when the model falls behind the stream?

**Running example.** One 2-second utterance (or 10 video frames): frame
sizes on the wire, the chunks the model sees, partial and final outputs with
timestamps.

---

## Q. SQL, warehouse, and dbt pipelines

**Signals.** `dbt_project.yml`, `models/**/*.sql` with `{{ ref() }}` and
`{{ source() }}`, `schema.yml` tests, `snapshots/`, `seeds/`, `macros/`;
Python clients for Snowflake, BigQuery, Redshift, Athena, Postgres, DuckDB;
SQLAlchemy; Airflow or Dagster tasks that run SQL; plain `.sql` files.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Sources | raw tables, where they come from (ingestion, CDC, files), their freshness checks |
| Model DAG | the `ref()` graph from sources to marts. Name the layers (staging, intermediate, marts) |
| Grain | what one row of each model is, and its unique key |
| Materialization | view, table, incremental (the `unique_key`, the incremental filter, merge or append), ephemeral, snapshots for slowly changing dimensions |
| Transforms | joins and their keys, window functions, deduplication, late-arriving data |
| Tests | `unique`, `not_null`, `relationships`, `accepted_values`, custom data tests, and source freshness |
| Schedule | what runs `dbt build` or the SQL (Airflow, dbt Cloud, cron), with which selectors |
| Warehouse specifics | partitioning and clustering keys, cost or slot limits, permissions |
| ML link | which tables feed features or labels, and the as-of logic that prevents leakage |

**Diagrams.** A flowchart of the model DAG from sources to the ML-facing
tables, labeled with each model's grain and materialization.

**Output examples.** For each important model, the SQL excerpt (labeled
`-- models/marts/orders.sql  orders`) and 3-5 result rows of the running
example, with the columns the model adds.

**Tests to look for.** `dbt test --select <model>` per model, and
`dbt source freshness`. Give the selector command, and mark it not run when
it needs warehouse credentials.

**Running example.** One entity (one order, one customer) traced from the
source rows through each model to the feature or label table.

---

## R. Labeling, annotation, and data quality

**Signals.** Label Studio, Argilla, Prodigy, Labelbox, FiftyOne, CVAT;
cleanlab, Snorkel (weak supervision); Great Expectations, pandera,
Deepchecks, Evidently, whylogs; label schemas and annotation guidelines.

**Components to find.**

| Component | Look for |
|-----------|----------|
| Task and schema | what is labeled, the label set and its definitions, the annotation format (JSON, COCO, YOLO, spans) |
| Sampling | which items go to annotators: random, stratified, active learning (uncertainty), or error-driven |
| Annotation flow | the tool, the project config, pre-labels from a model, review and adjudication |
| Agreement | inter-annotator agreement (Cohen's kappa, Krippendorff's alpha), gold sets, annotator quality checks |
| Weak supervision | labeling functions and the label model that combines them |
| Label cleaning | noisy-label detection (confident learning), relabel loops |
| Versioning | how label versions and dataset versions are tracked, and which version trained which model |
| Data quality checks | schema, range, null, and distribution checks, where they run (ingest, training, serving), and what a failure does |

**Math to show.** The agreement metric on a small example. For weak
supervision, how the label model combines votes.

**Running example.** One item: the raw input, the task the annotator sees,
each annotation, the adjudicated label, and the version it lands in.
