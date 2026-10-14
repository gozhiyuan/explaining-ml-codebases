# Stage playbooks

Each ML stage needs a different explanation format. Read only the stages your
subsystem covers. Every stage also uses the common section shape in
[doc-templates.md](doc-templates.md): Idea, Input, Output, Code, Example,
Math, Under the hood, Tests.

Every stage shows its intermediate outputs as real objects (type, shape or
schema, location, sample). See "Show the data, not only the code" in
[writing-modes.md](writing-modes.md). The items below say which outputs
matter most for each stage. For the data stages (2-6) and for serving, also
read the section of [modality-notes.md](modality-notes.md) for each data type
the repo handles: tabular, text, image, video, audio, sensor streams,
robotics episodes, 3D, graphs, multimodal.

"Must show" items are required in `deep` mode. In `simple` mode, keep the
Idea, the Input/Output in plain words, one example, the key files, and the
test command.

## Contents

1. Trigger and orchestration
2. Data preparation and labels
3. Distributed or streaming data processing (engine under the hood)
4. Feature processing
5. Splitting
6. Data loading and batching
7. Model architecture
8. Training loop
9. Tuning, calibration, post-training
10. Evaluation and promotion gates
11. Packaging and deployment
12. Serving and prediction
13. Cross-cutting: config, versions, clocks, train/serve parity
14. Variant families (losses, optimizers, schedulers, and other plug-ins)
15. Privacy, security, and data governance
16. Training cluster and job infrastructure

---

## 1. Trigger and orchestration

**Must show.**
- How a run starts: the command, launcher, scheduler, or event, and what it passes in.
- The configuration surface (see 13). If some values are fixed when the code is deployed and others arrive with each run, split them into two tables. Skip the split when the repo has no deploy step.
- The pipeline DAG as ASCII, one line per step, with the artifact passed on each edge.
- Retry and idempotency rules. What happens on a second delivery of the same event?

**Diagram.** ASCII DAG. Add a Mermaid flowchart if there are branches.

**Tests to look for.** Event parsing, parameter resolution, idempotency guard.

## 2. Data preparation and labels

**Must show.**
- Each source: location, partition layout, schema (the columns the code uses), clock and timezone.
- The label definition as a formula, and the time cut-off that prevents leakage.
- Filters and dedup rules, with the row count effect where known.
- The running example rows before and after.

**Pitfalls to call out.** Label leakage, survivorship bias in the split key, timezone mixing.

When the data prep is SQL or dbt, follow archetype Q in
[ml-archetypes.md](ml-archetypes.md): the model DAG, each model's grain and
materialization, and SQL excerpts with result rows. When labels come from
annotators or labeling functions, follow archetype R.

**Tests to look for.** Label function on a hand-built row. Filter edge cases.

## 3. Distributed or streaming data processing (engine under the hood)

Use this for Kafka/Kafka Streams, Flink, Beam, Ray Data, Spark or Dask steps.
Identify the engine and execution mode from source and resolved configuration.
Follow the roles it actually uses; apply the relevant batch or streaming rules.

**Must show.**
- **What runs where.** Map source functions and owned state to actual processes: producers/brokers/consumers for Kafka, operators/tasks for a Flink job, actors/tasks for Ray, or driver/executors/Python workers for Spark. Show model workers and GPU boundaries when inference is embedded.
- **Partitioning.** Source topic or dataset, partition or grouping key, worker assignment, ordering scope, and one worker's input. For consumers, include group membership and reassignment; for operators, include keyed-state ownership.
- **Data flow table.** Step ID, operation, input/output schema, example values, and storage location. Retain event keys, timestamps and offsets or batch identifiers where they establish ordering or recovery.
- **Completion and recovery.** What makes an output durable, when progress is committed, what state is checkpointed, and what a retry or restart replays. Follow the actual commit/checkpoint order and external sink side effects before explaining delivery guarantees.
- **Batch execution, when present.** Lazy plan versus execution triggers, shuffles, joins, broadcasts, caching and reruns. For a UDF boundary, show the grouping key and actual input/output schema.
- **Stateful streaming, when present.** Event time versus processing time, windows, watermarks, late-event handling, state stores and checkpoint recovery. Trace one event through its state changes; show any duplicate/replayed event behavior.
- **Realtime model boundary, when present.** Event/frame to tensor, model version, batching, per-session or per-key model state, output publication, queue bounds and backpressure. Show how preprocessing agrees with training.

**Diagrams.** An event/process sequence for streaming or a task tree for batch
processing, with ownership and output/progress commits. Build the call tree
from actual source functions. At a UDF boundary, identify the API, grouping
key, frame schema and worker role. Do not assume one task per group or shared
training/serving transforms without checking the execution plan and code.

**Under the hood.** Follow the engine's relevant limits: consumer lag, queue
bounds, hot keys or skew, state growth, serialization, model batches, or task
memory. For Arrow-based UDFs, include Arrow batch size. Use source or measured
evidence for timing and memory claims.

**Tests to look for.** Pure transforms or preprocessing on fixture events/rows;
an out-of-order event, duplicate, restart/replay, sink retry and checkpoint
boundary when relevant; UDF schema checks for batch engines. State which tests
use a real broker/processor and which use stand-ins. A pure function test does
not establish broker delivery, distributed partitioning or recovery behavior.

## 4. Feature processing

**Must show.**
- A feature table: name, source column, transform, formula, missing-value rule, output dtype.
- Each formula, for example `z = (x - mean_train) / std_train`, and where the statistics are fitted (train split only).
- Encoders: vocab, reserved IDs (PAD, UNK), what happens to an unseen value.
- Missing indicators (`has_x`) and why NaN -> 0 alone is ambiguous.
- Train/serve parity: does serving call the **same** function? If not, list each difference.
- The running example's features before and after.

**Tests to look for.** Encoder round trip. Unseen-value handling. Parity test on one row through both paths.

## 5. Splitting

**Must show.** The split key and rule (random, time, group). The ratios. The
overlap check. Where split membership is saved. One example entity and the
split it lands in, with the reason.

For time series and recommenders, the split is by time. Show the cut-off
dates, the gap (if any) between train and test, and for backtests every fold's
window (archetypes I and N in [ml-archetypes.md](ml-archetypes.md)).

**Tests to look for.** A zero-overlap assertion. Determinism with a fixed seed.

## 6. Data loading and batching

**Must show.**
- `Dataset.__getitem__`: input (an index), output (a dict of tensors with shapes).
- `collate_fn`: padding, masks, and the batch shapes `[B, T, ...]`.
- Sampler, shuffle, `num_workers`, pinned memory, prefetch.
- Packing for LLMs: how documents share a sequence, and the attention mask that keeps them apart.
- Images: the augmentation list in order, which ones are random, the normalization mean and std, and the layout (NCHW or NHWC). Show one image's shape after each transform.
- TensorFlow `tf.data` or JAX input pipelines: see [framework-notes.md](framework-notes.md).
- The running example as one item, then as one padded batch. Show the mask values.

**Diagram.** A small grid of the batch, with pad slots marked.

**Tests to look for.** Item shapes. Collate padding and mask values.

## 7. Model architecture

**Must show.**
- **Inputs inventory (do this before the diagram).** List every argument of the model's `forward` and every input channel built in `__init__` (each embedding, projection, and side input). For each one, say whether the golden path uses it and what switches it: a config key, a value that the data produces (a vocab size above 0, an encoder file that exists, columns present in the split), or the checkpoint. A channel that production data turns on is on the golden path, even when its default is off and even when the running example has no value for it. In that case, give the running example a value (for example categorical input IDs), and explain the channel like any other. Mark only the channels that are really off, with the reason.
- **Architecture view (required).** After a plain purpose sentence, a few
  actions and a small input/output pair, show a Mermaid `flowchart TB` or an
  original SVG architecture map that
  draws how data moves through the model, from the input tensors to every
  output head. This is the picture readers look for first. Rules:
  - One node per block (embedding, each distinct layer type, norm, heads). Draw a repeated stack once and label it `× N`.
  - Put the tensor shape on every edge, quoted: `-->|"[B, T, 256]"|`. Use named dims with the real values.
  - Draw residual adds, concatenations, branches, gates, and masks where they happen.
  - Show which inputs enter where (positions, conditioning features, masks, KV cache).
  - Under the diagram, an ASCII shape flow with the same shapes, so the picture also reads on hosts that do not render Mermaid.

  A `classDiagram` shows which module contains which. It does not replace this diagram.

For a source-grounded example, see the
[nanoGPT architecture](https://github.com/gozhiyuan/explaining-ml-codebases/blob/main/examples/nanogpt/model.md#architecture). It includes
residual paths, named tensor dimensions and tied parameters from its selected
configuration. Use the current repository's architecture rather than copying
those dimensions or assuming extra input channels or output heads.

Layout tips: keep the main path one straight top-to-bottom chain. Group the
input tensors in one node with `<br/>` line breaks. Draw the repeated stack as
a `subgraph` with `direction TB`, and connect edges to nodes inside it, not to
the subgraph itself. Draw the heads from the last block node. Name outputs in
plain nodes, not circles.

- **Class tree** from the actual factory to the leaf modules, with file names
  and repetition counts read from the selected configuration.
- **Shape flow.** One line per operation, using the actual input/output names,
  named dimensions and values. Mark broadcasts, reshapes and concatenations.

- **Dims table.** Each named dim, its meaning, its value in the tiny example, and its real value.
- **Hyperparameters.** Hidden size, heads, layers, vocab, context length, with the source (config key, or the class that sets it).
- **Parameter count**, computed or measured. Show the formula for the biggest block.
- **Math.** Every non-trivial computation in the forward pass. For attention:
  `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M) V`. Then state what the mask `M` is in this code (causal, padding, document boundaries), with a 4x4 example of the mask.
- **Worked example.** Push the running example through one block with tiny numbers.
- **Hardware notes.** Python loops over layers or features, host-device syncs, memory peaks, fused kernels used.

**Diagrams.** Show the architecture view with shapes and a readable text shape
flow. Add a class/module diagram only when it explains an additional relationship,
such as weight sharing or meaningful repetition. Avoid a second picture that
only repeats the architecture or stacks module names in empty class boxes.

**Tests to look for.** Forward pass output shapes. Mask correctness. A known-value test on a tiny config.

### Repos with many architectures (model zoos)

Hugging Face `transformers`, vLLM, SGLang, timm, diffusers, torchvision,
and mmdetection each support dozens or hundreds of model families. Explain
the machinery once, explain one model in depth, and summarize the rest.

1. **The seam every model plugs into (one section).** How a name or config
   becomes a model: the config class, the registry or auto-class mapping
   (`AutoModelForCausalLM`, vLLM's model registry, timm's `create_model`), the
   base class or interface every model implements (forward signature,
   attention backend, KV cache, `load_weights`), how checkpoint keys map to
   modules, and how layers are sharded across devices (tensor and pipeline
   parallelism). Show this as a call tree with `-> output` tags, from the
   user's model name to a loaded, sharded module.
2. **One golden model, as an architecture card.** Pick the model the README
   or CI uses by default, the most-used family, or the one the user names.
   Write a full architecture card (see "Architecture card" in
   [doc-templates.md](doc-templates.md)): config values, the architecture
   diagram, shapes, parameter count, the math per block, inputs and outputs,
   and how this repo builds it (registry entry, weight loading, sharding).
3. **The other families, as a comparison table.** One row per family the
   user cares about (or the 5-8 most important): what differs from the
   golden model (norm, position encoding, attention variant such as GQA, MLA,
   or sliding window, MoE routing, activation, tied embeddings, extra
   inputs), with the model file. No full walkthrough.
4. **Room to grow.** `index.md` lists the cards that exist and says how to
   add one: "Explain the <family> architecture" writes a new
   `arch-<family>.md` card that follows the same template and links back to
   the seam section.

## 8. Training loop

**Must show.**
- The loop as a call tree: epoch, batch, forward, loss, backward, clip, step, schedule, eval, checkpoint.
- **Loss.** Every term, its weight, its mask, its formula, and a numeric example on one item.
- Optimizer and schedule: type, LR, warmup, decay, weight decay groups. Give the LR at a few steps.
- Distributed strategy: launcher, world size, what each rank holds (full model, a shard, a stage), and the collective ops per step.
- For PyTorch DDP/FSDP, Megatron-LM/Core or DeepSpeed, map the selected DP/TP/PP groups and any context/expert parallel groups to ranks. Show parameter, gradient and optimizer-state ownership and one tensor's local/global shape. Follow actual all-reduce, all-gather, reduce-scatter or point-to-point calls.
- For pipeline training, trace one microbatch's forward/backward order, stage transfers and synchronization. Read the actual schedule and accumulation boundary before explaining overlap or idle stages.
- Mixed precision, gradient accumulation, `torch.compile`, activation checkpointing.
- Early stopping and model selection: which metric, on which split.
- Checkpoint contents and resume.
- Effective batch size: `micro_batch * grad_accum * data_parallel_size`.
  Derive these factors from the selected configuration/schedule; TP and PP
  ranks partition model work and do not multiply the number of distinct samples.

**Diagram.** Mermaid `sequenceDiagram` of one iteration: data fetch, the
micro-step loop, backward, clip, optimizer step, LR update, and the
communication used by the selected strategy. DDP may use all-reduce; FSDP and
model-parallel paths require their actual collectives and stage transfers.
ASCII call tree of the loop.

**Under the hood.** GPU use: where it idles, how many syncs per step, data
loader stalls. If you ran a few steps, give the measured time per step with
the hardware, versions, and config.

**Tests to look for.** Loss on a hand-computed example. One training step reduces the loss on a fixed batch. Checkpoint save and load round trip.

## 9. Tuning, calibration, post-training

**Must show, for each that exists.**
- HPO: search space table, search algorithm, objective, the split it scores on.
- Calibration: method (temperature, Platt, isotonic, decision thresholds), the split it fits on, the formula, a worked example of one prediction before and after.
- Post-training: SFT, DPO, PPO, GRPO, distillation, quantization, pruning, LoRA merge. Give each its loss or transform and one example.
- A second training pass (for example re-weighting): what changes between pass 1 and pass 2.

**Tests to look for.** Threshold decode on a fixed probability vector. Quantize then dequantize error bound.

## 10. Evaluation and promotion gates

**Must show.**
- Each metric's formula, and a small example computed by hand.
- Splits and slices (production regimes, cohorts) evaluated separately.
- The gate: the exact rule that passes or blocks a model, and where it runs.
- Differences between offline eval and online behavior.
- Ranking metrics (recall@k, NDCG, MAP) on a 5-item example when the model ranks.
- If the repo is an eval harness, or calls one, follow archetype M in [ml-archetypes.md](ml-archetypes.md): one sample from raw item to prompt to score.

**Tests to look for.** Metric functions on known inputs. Gate pass and fail cases.

## 11. Packaging and deployment

**Must show.**
- Artifact layout as a tree (`model.tar.gz/`, `code/`, `config.pbtxt`).
- What is pinned inside the artifact versus read at runtime.
- The deploy command or pipeline, its inputs, and the target (endpoint, MME, Kubernetes).
- Container image and Python version.
- Rollback: what you change to go back, and what does not roll back.
- Export, compile, or quantize steps (ONNX, TensorRT, TorchScript, SavedModel, int8/int4): follow archetype K in [ml-archetypes.md](ml-archetypes.md). Show the parity check between the original and the exported model.
- The registry step (MLflow, W&B, SageMaker Model Registry): follow archetype L. Show how a deployment finds the version it loads.

**Tests to look for.** Artifact contents check. The load-from-artifact smoke test.

## 12. Serving and prediction

First name the serving mode, or modes, the repo uses. Each mode has its own
section below. Many repos have more than one, for example nightly batch
scoring plus an online endpoint for the same model.

| Mode | What it looks like | Section |
|------|--------------------|---------|
| Online request/response | HTTP or gRPC endpoint, one request in, one response out | Must show (below) |
| Streaming session | token streaming, live audio or video, WebSocket, WebRTC, control loops | Realtime and streaming serving |
| Batch / offline scoring | SageMaker Batch Transform or Processing job, Spark or Ray scoring job, scheduled scoring, offline LLM generation | Batch and offline inference |
| Async / queued | the request goes on a queue, the result goes to storage later (SageMaker async endpoints, SQS or Celery workers) | Async and queued inference |
| Embedded in a stream processor | the model runs inside a Flink, Kafka Streams, or Spark Structured Streaming job | Batch and offline inference, plus the streaming rules of playbook 3 |
| Edge / on-device | mobile, browser, embedded, or robot hardware | Edge and on-device inference |

**Must show (online request/response).**
- **Request lifecycle.** Trace the running request from the entry point to the response, through every process, thread, and queue, in order.
- **Process architecture.** Which processes and threads exist, what each owns, and how they talk (HTTP, gRPC, ZMQ, shared memory, queues).
- **Contracts.** The exact request and response schemas, with an example of each.
- **Load once versus per request.** Model load, warmup, caches.
- **Batching and scheduling.** Dynamic or continuous batching, the batch-size and token budget, queue policy, preemption.
- **Memory.** For LLMs: KV cache blocks, prefix cache, eviction. Show a block table for the running request.
- **Distribution.** Replicas, data-parallel ranks, routers, load balancing, TP/PP inside a replica.
- **Runtime path.** For TensorRT/TensorRT-LLM deployments, identify the configured backend and trace initialization, weight conversion or engine building/loading where used, rank mapping, native bindings and inter-worker communication. Keep replica routing separate from model-parallel execution; verify the deployed path rather than assume every backend builds a TensorRT engine.
- **Failure behavior.** Timeouts, retries, fail open or closed, partial results.
- **Latency budget.** Where time goes, if the code or logs show it.
- **Train/serve parity** for features and post-processing.

**Diagrams.** Mermaid `sequenceDiagram` for the request (required). Mermaid
flowchart for processes. `stateDiagram-v2` for the request states (waiting,
running, preempted, finished) when a scheduler exists. ASCII call tree for the
hot path.

**Tests to look for.** Handler on a sample payload. Schema validation. Batching edge cases (empty, one item, more than the max). Scheduler unit tests on fake requests.

### Realtime and streaming serving

Use this when the service streams in or out: token streaming, live speech
recognition or speech synthesis, video analytics, voice agents, robot
control loops. The request is a session, not one call.

**Must show.**
- **Transport.** Which one and why: HTTP with SSE (server-sent events), WebSocket, gRPC bidirectional streaming, WebRTC (media over UDP with SRTP, with signaling over another channel), MQTT, Kafka, or a ROS topic. For WebRTC, also show the signaling flow (offer, answer, ICE), the media server or library (LiveKit, Janus, mediasoup, aiortc, Pion), and the TURN/STUN setup.
- **Session lifecycle.** Connect, authenticate, negotiate, stream, reconnect, close. What state lives in the session (model cache, conversation history, KV cache, decoder state), where it lives, and when it is freed.
- **Framing and chunking.** The unit on the wire (audio frames of 20 ms, video frames, token deltas, sensor packets), its codec (Opus, PCM, H.264, VP8), and how frames become model inputs: buffering, resampling, chunk length and overlap, VAD (voice activity detection) or endpointing.
- **The streaming loop.** As a `sequenceDiagram` over time: frames in, partial results out, final results, interruption ("barge-in"), and end of stream. Show the running example as real frame and chunk sizes.
- **Latency budget.** A table, per stage: network, decode, buffering, model time per chunk, post-processing, encode, send. Show the target (for example time to first token, or end-of-speech to first audio) and what the code or logs measure.
- **Backpressure and overload.** What happens when the model is slower than the stream: queue bounds, dropped frames, adaptive chunk size, load shedding, per-session concurrency limits.
- **Ordering and clocks.** Sequence numbers, timestamps, jitter buffers, and how the service keeps outputs in order.
- **Failure behavior.** A dropped connection, a stalled model, a partial result at disconnect.

**Diagrams.** A `sequenceDiagram` of one session over time (required). A
flowchart of the media and model path. A latency budget table.

**Tests to look for.** A recorded stream replayed through the pipeline with
expected partial and final outputs. Chunk-boundary tests. Reconnect tests.
Load tests with N concurrent sessions.

### Batch and offline inference

Use this when a job scores many inputs at once: SageMaker Batch Transform or
a Processing job, a Spark or Ray scoring job, a scheduled scoring pipeline,
or offline LLM generation (vLLM `LLM.generate`, a provider batch API).

**Must show.**
- **Trigger and input selection.** What starts a run (schedule, event, pipeline step), and which inputs it scores: the partitions, the as-of time, the lookback window, the filters. The run id, and how a rerun of the same input behaves.
- **How the work is split.** The unit of work and how it reaches workers:
  - SageMaker Batch Transform: instance type and count, `max_payload`, `batch_strategy` (`SingleRecord` or `MultiRecord`), `split_type`, `content_type`, `assemble_with`, `join_source`, and `max_concurrent_transforms`.
  - Spark: the partitioning, the UDF type (`mapInPandas`, pandas UDF), the Arrow batch size, and rows per task.
  - Ray Data: `map_batches` batch size, actor pool size, GPUs per actor.
  - Offline LLM: prompts per call, the sampling settings, and how the engine batches them.
- **Model loading.** Where the model is loaded and how often: once per container, per executor, or per actor. Not once per row or per batch. How the artifact reaches each worker (S3 download, broadcast, a mounted volume).
- **Per-batch path.** One batch from input rows to output rows: decode, features, model call, post-processing. Show the running example's rows at each step.
- **Output and commit.** Output location and partitioning, the schema with example rows, how outputs join back to input keys, the model version column, and the commit marker (`_SUCCESS`, a manifest, a table commit) that downstream readers wait for.
- **Failure and reruns.** Partial failures, retries, poison records, idempotent overwrite or merge, and what a reader sees during a run.
- **Throughput and cost.** Rows per second per worker, worker count, GPU use, and the run time for a typical input size, measured or estimated (say which).
- **Train/serve parity.** The same features and post-processing as training and as any online endpoint. Name every difference.
- **Run checks.** Rows in versus rows out, null rates, the score distribution, and alerts.

**Diagrams.** A flowchart from the input store to the output store, through
the workers. A `sequenceDiagram` of one batch through one worker.

**Tests to look for.** The per-batch function on a fixture batch. Row count
in equals rows out. Rerun idempotency. Empty-input behavior.

### Async and queued inference

Use this when the caller does not wait: SageMaker async endpoints, a queue
(SQS, Pub/Sub, Kafka, Celery, RQ) in front of workers, or a long job behind
an API.

**Must show.** Where the request payload is stored (often S3) and what goes
on the queue. The worker that consumes it, its concurrency, and its
autoscaling, including scale to zero. Where the result goes, and how the
caller learns it is done (polling, a notification, a callback). Time-to-live
and retries. A dead-letter queue. Duplicate delivery and idempotency keys.
Show one request as a `sequenceDiagram` from submit to result.

### Edge and on-device inference

Use this when the model runs on a phone, in a browser, on embedded hardware,
or on a robot.

**Must show.** The export format and the runtime (TFLite, Core ML, ONNX
Runtime Mobile, TensorRT on Jetson, WebGPU/WebAssembly). Quantization, and
its accuracy cost. Pre- and post-processing on the device, and whether it
matches the training code. Memory and latency on the target hardware. How
model updates reach devices, and how a bad model is rolled back. See
archetype K in [ml-archetypes.md](ml-archetypes.md) for the export steps.

### Deployment topology and infra

Show this when the repo holds infra code (Kubernetes manifests, Helm
charts, Terraform or CDK, SageMaker or Vertex endpoint config, Docker
Compose, Ray Serve config).

- **Topology diagram.** A flowchart from the client to the model: load balancer or API gateway, service, pods or instances, GPU nodes, model store, cache, queue, database. Mark which parts scale.
- **Scaling knobs.** A table: replicas, autoscaling metric and thresholds (HPA, KEDA, SageMaker target tracking), GPU type and count per replica, GPU sharing (MIG, MPS, time slicing), concurrency per replica, scale to zero, and cold-start time.
- **Model rollout.** How a new model version reaches traffic: blue-green, canary, shadow traffic, the multi-model endpoint target, and the rollback path.
- **Configuration and secrets.** Where each setting and secret comes from at deploy time.
- **Observability.** Metrics, logs, traces, and the dashboards or alerts that exist.

## 13. Cross-cutting: config, versions, clocks, train/serve parity

Put these near the top of `index.md` or the relevant sub-doc.

- **Pinned table.** Knob, value for this run, where it is set, what it changes.
- **Configuration surface.** List only the layers the repo has: defaults in code, config files (YAML, Hydra, dataclasses), CLI flags, environment variables, deploy-time parameters (IaC, CI variables, endpoint config), and per-request fields. Give the precedence order (which layer wins) and one key traced through every layer it passes. A small repo may have one layer. Then say so in one line.
- **Clocks.** Each timestamp field, its clock (UTC or local), and conversion rules.
- **Parity table.** Each stage that both training and serving run, the function each calls, and every difference.
- **Differences from the reference.** R&D notebook, paper, or upstream repo versus this code. One row per difference, with an example where it changes a result.

## 14. Variant families (losses, optimizers, schedulers, and other plug-ins)

Large repos let a config key choose one implementation from a family: the
loss, the optimizer, the LR scheduler, the sampler, the augmentation set,
the attention backend, the tokenizer, metrics, callbacks. Model
architectures are the biggest family (section 7). The rest follow the same
pattern. Find these families in recon. A sign is a registry dict, a
`build_*`/`make_*` factory, a `getattr(module, name)` lookup, Hydra
`_target_`, or an `if cfg.x == "..."` chain.

For each family that the golden path touches:

1. **The switch.** The config key, its default, the value the golden path uses, and the factory or registry that turns the value into an object. Show the factory excerpt and a call tree from the key to the object, with `-> output` tags.
2. **The golden variant in depth.** The one the golden path runs, explained as a normal step: excerpt, math with a worked number, input and output, tests.
3. **The other variants in one table.** One row per variant. Columns: config value, class or function and file, what differs from the golden variant (the formula or the behavior), when the repo uses it (another config, an experiment, a sweep, unused), and its test.

Table schema (fill from the selected repo, not from this guide):

| Config value | Implementation and file | Difference from selected variant | Used by | Existing test or gap |
|---|---|---|---|---|
| `<actual value>` | `<symbol and path>` | `<source-supported formula or behavior>` | `<actual config or caller>` | `<verified test or uncovered behavior>` |

4. **Interactions.** Where a choice changes another part: a loss that needs class weights, a scheduler that needs `steps_per_epoch`, an optimizer with parameter groups, a variant that changes the output head's shape.

Do not explain every variant in depth. A user who needs another variant asks
for it ("explain the focal loss path"). That variant is then written as a
normal step in the same doc, or added in revise mode.

## 15. Privacy, security, and data governance

Show this whenever the pipeline handles personal, health, financial, or
other sensitive data. It often sits across stages, so give it one section
in `index.md` and point to the steps where it happens.

**Must show.**
- **Sensitive fields.** Which fields are personal or sensitive (names, ids, dates of birth, free text, images of people, locations), where they enter, and which steps read them.
- **Protection at each step.** Where data is removed, masked, hashed, tokenized, or pseudonymized (Presidio, scrubadub, custom rules), and where re-identification keys live. Show one record before and after, with fake values.
- **Free text and media.** How notes, transcripts, or images are scrubbed, and what the scrubber can miss.
- **Storage and access.** Encryption at rest and in transit, buckets and tables with their access roles, VPC or network isolation, and who can read raw data.
- **Logs, metrics, and errors.** Whether payloads or features reach logs, traces, error reports, or sample dumps. This is a common leak.
- **Models as data.** Whether training data can be extracted from the model or its outputs (memorization, embeddings), and any mitigation (differential privacy with Opacus, output filters).
- **Retention and deletion.** How long each dataset and model is kept, and how a deletion request reaches derived data.
- **Rules that apply.** HIPAA, GDPR, SOC 2, or internal policy, only when the repo or its docs name them. Do not guess compliance.

**Tests to look for.** Scrubber tests on known sensitive strings. Tests that
logs do not contain payloads. Access-policy tests.

## 16. Training cluster and job infrastructure

Show this when training runs on more than one machine, or through a
scheduler: Slurm, Kubernetes jobs (Kubeflow `PyTorchJob`, `TFJob`,
`MPIJob`), Ray jobs, SageMaker or Vertex training jobs, SkyPilot, submitit.

**Must show.**
- **Launch.** The script or manifest that starts a run (`sbatch` script with `#SBATCH` lines, a `PyTorchJob` YAML, `ray job submit`, an estimator call), and the launcher inside it (`torchrun`, `deepspeed`, `accelerate launch`, `srun`, `mpirun`).
- **Topology.** Nodes, GPUs per node, GPU type, the interconnect (NVLink, InfiniBand, EFA), and how ranks map to GPUs. Draw it as a flowchart.
- **Orchestration versus training.** Separate Kubernetes/Slurm job scheduling, pod or process replicas, and restart policy from the model framework's parallel groups. Follow how the manifest/launcher sets the rank topology; a replica count alone does not establish data or model parallelism.
- **Rendezvous and environment.** `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, and NCCL settings, and who sets each one.
- **Data staging.** Where data lives during training (shared filesystem, FSx or Lustre, local NVMe cache, streaming from S3), and the loader workers per GPU.
- **Checkpoints and preemption.** Where checkpoints go, how often, what a restart resumes, and how spot or preemptible instances are handled.
- **Sweeps and job arrays.** How hyperparameter sweeps launch (Slurm arrays, Ray Tune, W&B sweeps), and where their results land.
- **Logs and cost.** Where logs and metrics go, and the GPU-hours for a typical run, if known.

**Diagrams.** A topology flowchart. A `sequenceDiagram` of the job
lifecycle: submit, schedule, rendezvous, train, checkpoint, finish or
preempt and resume.

**Tests to look for.** A tiny multi-process run on CPU with the `gloo`
backend. A resume-from-checkpoint test. A manifest lint or dry run.
