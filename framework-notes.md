# Framework notes

Read the section for each framework the scanner reports. Each section lists
what the doc must show for that framework, on top of the stage playbooks. The
general rule stays the same: say where each thing runs, when it runs, and with
what shapes.

## Contents

- TensorFlow / Keras
- JAX / Flax
- CUDA and Triton kernels
- FastAPI model apps
- Non-Python code (C/C++, Rust, Go, Scala/Java)
- Notebooks

---

## TensorFlow / Keras

**Two execution modes.** Say which one each piece of code runs in. Eager code
runs line by line in Python. Code under `@tf.function` (and `model.fit` by
default) is traced once per input signature into a graph. Python side effects
(`print`, list appends, Python `if` on tensor values) run only at trace time.

| Component | Must show |
|-----------|-----------|
| Input pipeline | the `tf.data` chain in order: source, `map` (with `num_parallel_calls`), `shuffle(buffer)`, `batch`, `prefetch`. Element spec (shapes, dtypes) after each step. Where augmentation runs (CPU in the pipeline or GPU in the model) |
| Model | `Sequential`, functional, or subclassed. `model.summary()` shapes, with `None` for the batch dim. Where `build` creates the weights |
| Training | `compile(optimizer, loss, metrics)` and `fit`, or a custom loop with `GradientTape`. For a custom `train_step`, the full step: forward, loss, `tape.gradient`, `apply_gradients`, metric updates |
| Retracing | the input signature of each `tf.function`. Python scalars or changing shapes cause a retrace. Name any retrace the code can trigger |
| Distribution | `tf.distribute` strategy (`MirroredStrategy`, `MultiWorkerMirroredStrategy`, `TPUStrategy`), what the strategy scope wraps, the per-replica versus global batch size |
| Mixed precision | `mixed_precision.set_global_policy`, the loss scale optimizer, layers kept in float32 |
| Export and serving | SavedModel signatures (`serving_default`), the input and output names and shapes, and TF Serving's REST or gRPC request format |

**Tests to look for.** `tf.test.TestCase`, shape tests on one batch, a
SavedModel round trip that compares outputs.

## JAX / Flax

**Functional model.** Parameters are a pytree passed into pure functions, not
attributes on an object. Show the parameter tree (keys and shapes) for the
golden-path model, and the train state that bundles params, optimizer state,
and the step.

| Component | Must show |
|-----------|-----------|
| Model | Flax `nn.Module` (Linen) or `nnx`, Haiku, or Equinox. `init` with a dummy input creates params. `apply` runs the forward. Show the param tree with shapes |
| Randomness | explicit PRNG keys. Where keys are split, and which key feeds dropout versus init |
| Compilation | which functions are `jax.jit`ted, their static arguments, and what triggers a recompile (a new shape, a new static value). Python control flow on traced values fails or unrolls. Name the `lax.scan`, `lax.cond`, or `vmap` that replace loops |
| Train step | `value_and_grad`, the Optax update chain (for example `clip_by_global_norm` then `adamw`), `apply_updates`, and the returned new state. The step is pure: show what goes in and what comes out |
| Parallelism | `pmap` (older) or `jit` with `NamedSharding` and a `Mesh`, or `shard_map`. The mesh axes (data, model), the `PartitionSpec` of each large param, and the collectives (`psum`, `all_gather`) |
| Checkpoints | Orbax `CheckpointManager`, what is saved, and how a sharded checkpoint restores onto a different mesh |
| Host versus device | `jax.device_get` and `block_until_ready` calls. Async dispatch makes naive timing wrong: say how the code times a step |

**Tests to look for.** `chex` assertions, shape tests on `init` output, a
jitted versus non-jitted equality test.

## CUDA and Triton kernels

Explain a kernel only where the golden path calls it. For each kernel, follow
this order.

1. **The call.** The Python op the model calls, how it is registered (`TORCH_LIBRARY`, `torch.library.custom_op`, a pybind module, or `triton.jit`), and the file and line of the kernel entry.
2. **The reference.** The same computation in plain PyTorch or NumPy, 3-10 lines. This is what the kernel must match.
3. **The launch.** CUDA: grid and block dims and what one thread block computes. Triton: the program id axes, the `BLOCK_*` constants, and one program's tile. Give the numbers for the running example's shapes.
4. **The memory path.** What each tile loads from global memory, what stays in shared memory or registers, what is written back. One ASCII diagram of the tile loop.
5. **The math.** For fused or tiled algorithms (for example online softmax in flash attention), the running max and sum update with a numeric example on 2 tiles.
6. **Performance.** Arithmetic intensity (FLOPs per byte), whether the kernel is compute- or memory-bound, and the autotune configs if any.
7. **Correctness.** The test that compares the kernel to the reference, its dtypes, and its tolerance.

```
tile loop for one program (BLOCK_M=64 rows of Q, BLOCK_N=64 keys per step)
  load Q tile  [64, d]  global -> SRAM, once
  for each K/V tile j:
      load K_j, V_j  [64, d]  global -> SRAM
      S = Q K_j^T  [64, 64]   in registers
      update running max m, running sum l, rescale acc
      acc += softmax_part(S) V_j
  write acc / l  [64, d]  SRAM -> global
```

**Tests to look for.** `torch.testing.assert_close` against the reference
across shapes and dtypes. These usually need a GPU: mark them
`not run (needs GPU)` if none is available.

## FastAPI model apps

Many ML products are a FastAPI (or Flask) app around a model. The model math
may be small. The serving behavior is where bugs live.

| Component | Must show |
|-----------|-----------|
| App start | the app factory, `lifespan` or startup hook, where the model loads, and how long it takes. Is it loaded once per worker process? |
| Request contract | the Pydantic request and response models, field types and defaults, validation errors, and one example body of each |
| Route chain | for the main route: middleware, dependencies (`Depends`), the handler, pre-processing, the model call, post-processing, the response model |
| Concurrency | `async def` versus `def` handlers. A blocking model call inside `async def` stalls the event loop. Say where the call runs: event loop, the threadpool, a process pool, or a separate model server |
| Workers and memory | the server command (uvicorn or gunicorn), number of workers, and memory per worker times workers |
| Batching | whether requests are batched (a queue plus a batching loop) or each request calls the model alone |
| Background work | `BackgroundTasks`, Celery or RQ queues, and what happens to a task when the process restarts |
| State and caching | caches (in-process, Redis), their keys, and their invalidation |
| Errors and timeouts | exception handlers, the HTTP codes returned, client and server timeouts |
| Observability | logging, metrics endpoints, request ids |

**Diagram.** A `sequenceDiagram` of one request from client to response,
including the worker, the threadpool hop if any, and the model.

**Tests to look for.** `TestClient` tests on the main route with a sample
body, validation-error tests, and a test with the model mocked.

## Non-Python code (C/C++, Rust, Go, Scala/Java)

Many ML systems put the hot path in another language: llama.cpp and ggml in
C/C++, the Text Generation Inference router in Rust, ollama in Go, Spark
jobs in Scala. The scanner's "Non-Python code" section lists each language's
entry points, main types, test files, build files, and the bindings between
languages. Explain that code with the same rules as Python. Some parts need
more.

| Component | Must show |
|-----------|-----------|
| Build | the build system (CMake, Cargo, Go modules, sbt or Maven), the targets that matter for the golden path, feature flags or backends (CUDA, Metal, Vulkan), and one command that builds them |
| Entry points | the binary or service the golden path runs (`main`, `fn main`, `func main`, `object Main extends App`), and its flags |
| Language boundary | how Python and native code meet: pybind11, `TORCH_LIBRARY` ops, PyO3, cgo, JNI, ctypes or cffi, or a process boundary (gRPC, HTTP, a socket). Draw the call tree across the boundary, and show the data that crosses it with its type on both sides |
| Memory and ownership | who allocates and frees each large buffer (tensors, KV cache, batches), lifetimes and borrowing (Rust), and copies across the boundary |
| Concurrency | threads, async runtimes (tokio, Go routines and channels, Akka, futures), locks, and the queues between them |
| Data layout | structs and their fields for the main objects (`llama_batch`, `Request`), memory layout of tensors (row-major, quantized blocks) |
| Errors | error types and how they cross the boundary (exceptions, `Result`, error codes) |

Rules for these languages:

- Excerpts use the language's comment for the label line: `// src/llama-context.cpp  llama_decode`, `// router/src/infer/mod.rs  Infer::generate`, `-- models/orders.sql  orders`. The checker verifies them the same way.
- Name symbols the language's way: `Infer::generate`, `(*Scheduler).processPending`, `llama_decode`. "`llama_decode` in `src/llama-context.cpp`" is checked by name.
- Tests: `cargo test`, `go test ./...`, `ctest`, gtest binaries, `sbt test`. Give the exact command for the step's tests, and mark tests that need a GPU or a model file as not run, with the reason.

## Notebooks

In many research repos the logic lives in notebooks. The scanner parses code
cells, lists notebooks that define functions or classes or train a model,
and says when a repo is notebook-heavy.

- Treat a notebook as source code. Point at it by file and by the function, class, or heading it holds: "`Value` in `micrograd_lecture.ipynb`". When there is no name, use the cell number: `# lecture.ipynb  cell 7`. The checker matches excerpts against the notebook's code cells, not its JSON.
- State the execution order the notebook expects, and the hidden state between cells (globals defined in one cell and used later). Name any cell that must run first.
- Separate code that is logic (functions, the training loop) from exploration (plots, prints). Explain the logic. Mention the exploration briefly.
- Saved outputs are evidence: printed shapes, losses, and plots in the notebook can serve as output examples. Label them "from the saved notebook output" and give the cell.
- If the same logic exists in both a notebook and a `.py` file, say which one the pipeline runs, and note any drift between them.
- Tests: notebooks rarely have tests. Name the cells that act as checks (asserts, printed shapes), and propose tests for the functions in the gap table.
