# Training: stream windows to changed parameters

Paths are relative to the nanoGPT repository root. This document follows the loader and update logic. [model.md](model.md) explains the decoder and cross entropy.

## Contents

- [Overview](#overview)
- [train.A Data windows](#traina-data-windows)
- [train.B Parameter update](#trainb-parameter-update)
- [Other paths](#other-paths)

## Overview

```text
get_batch -> X,Y int64 [2,4], CPU                         # train.py train.A1
  └─ random offsets -> windows shifted by one
get_lr -> LR float 0.001                                 # train.py train.B1
GPT.configure_optimizers -> AdamW, two parameter groups  # model.py train.B1
GPT.forward -> logits [2,4,8], loss float scalar          # model.py model.A1-model.A3
  └─ loss.backward -> parameter gradients                # train.py train.B2
clip_grad_norm_ -> total norm and clipped gradients       # train.py train.B2
optimizer.step -> changed parameters                     # train.py train.B2
optimizer.zero_grad -> gradients None                    # train.py train.B2
```

| Artifact | Type / shape | Location | Sample |
|---|---|---|---|
| Stream | uint16 array, 12 tokens | Fixture file on local disk | `[0,1,2,3,4,5,6,7,0,1,2,3]` |
| Batch | two int64 tensors `[2,4]` | CPU memory | First X row `[7,0,1,2]` |
| Optimizer | AdamW with two groups | CPU process | decay=0.1 or 0.0 |
| Gradient | one float32 tensor per Parameter | CPU memory | Token weight gradient starts `[-0.096536,0.205138,-0.069415,-0.114340]` |
| Parameters | 888 distinct scalar parameters | CPU memory | Largest token-weight change 0.001004860 |

Running example: B=2, T=4, D=8, H=2, K=4, V=8, L=1, A=1, seed 1337. Dropout=0, bias=False, float32, CPU.

## train.A Data windows

### train.A1 Select shifted token windows (`train.py`)

**Idea.** `get_batch` chooses independent random starts in a flat file. Targets already contain the next token at each position.

**Input.** A uint16 file of 12 fixture token IDs, batch size two, and context length four.

**Output.** Measured CPU int64 tensors `[B,T]=[2,4]`:

```text
X = [[7,0,1,2], [5,6,7,0]]
Y = [[0,1,2,3], [6,7,0,1]]
```

**Code.**

```python
# train.py  get_batch
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
if device_type == 'cuda':
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
else:
    x, y = x.to(device), y.to(device)
```

`torch.randint` excludes its upper bound. A length-12 stream with T=4 permits starts 0 through 7. Start 7 gives X positions 7..10 and Y positions 8..11. Converting each slice to int64 allocates an array before stacking. CPU moves keep both tensors on CPU.

**Example.** The seed samples offsets 7 and 5. Every row has four inputs and four labels. No padding token, padding mask, document boundary mask, or Dataset object appears in this loader.

**Math.** `X[b,t]=stream[i_b+t]`, `Y[b,t]=stream[i_b+t+1]`. At b=0,t=0, X=stream[7]=7 and Y=stream[8]=0. Effective targets per update are BTA=2×4×1=8 on one process.

**Under the hood.** Character preparation sorts the unique characters and assigns contiguous integer IDs. It splits text 90%/10% before writing uint16 files. The `meta.pkl` dictionary stores `vocab_size`, `stoi`, and `itos`. The training process reads `vocab_size` for model construction. Our fixture bypasses character preparation and sets V=8 explicitly. Original preparation requires network data, so it was source-reviewed only.

**Tests.** External validation: `python tests/validate_step.py --repo "$NANO_REPO" --output assets/measurement.json` — **pass**. This script ships with these docs, outside the checkout. Set `NANO_REPO` to the original checkout. No upstream test suite exists at this commit.

Gap: sampler boundaries, the validation file branch, and actual text encoding need separate tests.

## train.B Parameter update

### train.B1 Set learning rate and optimizer groups (`train.py`, `model.py`)

**Idea.** The loop sets one learning rate for the iteration. AdamW separates matrix parameters from vectors for weight decay.

**Input.** Iteration zero, peak LR=0.004, warmup=3 updates, decay end=11, minimum LR=0.0004. The tiny model has 888 scalar parameters.

**Output.** Measured LR=0.001. AdamW receives six decayed tensors containing 864 scalars and three non-decayed tensors containing 24 scalars, all in CPU memory.

**Code.**

```python
# train.py  get_lr
# 1) linear warmup for warmup_iters steps
if it < warmup_iters:
    return learning_rate * (it + 1) / (warmup_iters + 1)
# 2) if it > lr_decay_iters, return min learning rate
if it > lr_decay_iters:
    return min_lr
```

```python
# model.py  GPT.configure_optimizers
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
```

```python
# model.py  GPT.configure_optimizers
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
print(f"using fused AdamW: {use_fused}")
```

The actual decay predicate is `p.dim() >= 2`. It includes token and position embeddings, not only matrix multiplication weights. The tied output weight appears once in named parameters. CPU chooses ordinary AdamW even when the optimizer exposes a fused option.

**Example.** Measured values from the source `get_lr` function:

| Iteration | LR |
|---|---|
| 0 | 0.001 |
| 2 | 0.003 |
| 3 | 0.004 |
| 7 | 0.0022 |
| 11 | 0.0004 |
| 12 | 0.0004 |

**Math.** Warmup returns `peak×(it+1)/(warmup+1)`. At it=0, `0.004×1/4=0.001`. Between warmup and decay end, r=(it−3)/(11−3), coefficient=(1+cos(πr))/2, LR=0.0004+coefficient×0.0036. At it=7, r=0.5 and LR=0.0022.

**Tests.** External validation: `python tests/validate_step.py --repo "$NANO_REPO" --output assets/measurement.json` — **pass**. This script ships with these docs, outside the checkout. Set `NANO_REPO` to the original checkout. No upstream test suite exists at this commit.

### train.B2 Backward, clip, update, clear (`train.py`)

**Idea.** Backward writes gradients without changing parameters. The optimizer uses clipped gradients to change parameters, then the loop releases gradient storage.

**Input.** Measured mean loss 2.056858 and the graph that produced it. A=1, gradient clip=1.0, LR=0.001, betas=(0.9,0.95), weight decay=0.1.

**Output.** Measured CPU parameter changes. The maximum absolute token-weight change is 0.001004860. All Parameter gradients are None after clearing. Re-evaluating this same batch gives loss 2.044125.

**Code.**

```python
# train.py
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
# clip the gradient
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
# step the optimizer and scaler if training in fp16
scaler.step(optimizer)
scaler.update()
# flush the gradients as soon as we can, no need for this memory anymore
optimizer.zero_grad(set_to_none=True)
```

The loss division averages accumulated micro-batches. With A=1 it leaves this loss unchanged. The source fetches new X,Y before backward. That reassignment does not change the tensors already retained by autograd. On this CPU fixture, the scaler is omitted because float32 uses no scaling.

**Example.** This update produces finite gradients for every parameter. The total norm changes from 1.218671 to 0.999999. Before clipping, the first four elements of token-row-zero gradient are `[-0.096536, 0.205138, -0.069415, -0.11434]`. None represents cleared storage, not a tensor filled with zeros.

```text
Before update: loss=2.056858, gradient norm=1.218671
After clipping: gradient norm=0.999999
After AdamW: maximum token-weight change=0.001004860
After clearing: every parameter.grad is None
Same-batch evaluation: loss=2.044125
```

**Math.** Cross entropy's logit gradient at one unignored position is `(p[v]−1[v=target])/(BT)`. First position, class zero: `(0.138861−1)/8≈−0.107642`. Clipping approximately multiplies all gradients by `min(1,1/1.218671)=0.820565`. PyTorch includes a small numerical stabilizer, so use the measured norm above for the exact result.

AdamW maintains first and second moments in the optimizer. They are not implemented by nanoGPT. With initially zero moments, bias correction makes the first step approximately `Δw=−LR×g/(|g|+ε)−LR×decay×w`, using the clipped gradient. For LR=0.001 and decay=0.1, decay alone multiplies a matrix weight by 0.9999. The measured maximum change includes both terms. Later steps depend on moment history.

**Under the hood.** The actual CPU context in `train.py` is `nullcontext()`. Setting `dtype='float32'` also disables its CUDA GradScaler. The full process includes evaluation at iteration zero before the first update. Our external script avoids those process side effects and records its narrower contract.

**Tests.** External validation: `python tests/validate_step.py --repo "$NANO_REPO" --output assets/measurement.json` — **pass**. This script ships with these docs, outside the checkout. Set `NANO_REPO` to the original checkout. No upstream test suite exists at this commit.

A lower loss on this batch verifies this tiny run. It is not evidence of generalization, stable long training, or a guarantee that every AdamW step decreases loss.

Gap: accumulation, a full `train.py` execution, evaluation cadence, checkpoint/resume, and mixed precision were not run.

## Other paths

| Switch | Golden path | Alternative | Validation |
|---|---|---|---|
| `init_from` | Random tiny initialization | resume or `gpt2*` loading | Not run |
| `decay_lr` | Source warmup/cosine | Fixed LR | Schedule sampled, fixed LR not tested |
| `gradient_accumulation_steps` | 1 | Multiple micro-batches | Not run |
| `RANK` environment | Single process | DDP synchronizes last accumulated backward | Not run |
| `device`, `dtype` | CPU float32 | CUDA AMP and scaler | Not run |
| `compile` | False | `torch.compile` before DDP wrapping | Not run |

The repository's `train.py` defaults differ from `GPTConfig`: the script sets `bias=False`, while the dataclass defaults to True. The fixture deliberately uses the script's value. [model.md](model.md) covers the attention API switch and inference head branch.
