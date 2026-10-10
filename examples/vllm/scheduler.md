# Scheduler: known tokens become scheduled positions

This walkthrough covers token counts and the model-execution boundary. [kv-cache.md](kv-cache.md) explains physical block bookkeeping. Paths below are relative to the source repository root.

## Overview

```text
EngineCore.step                         -> engine outputs in CPU memory # vllm/v1/engine/core.py
  ├─ Scheduler.schedule                 -> SchedulerOutput             # vllm/v1/core/sched/scheduler.py sched.A1
  │   ├─ KVCacheManager.allocate_slots   -> new block metadata          # vllm/v1/core/kv_cache_manager.py kv.A2
  │   └─ Scheduler._update_after_schedule -> advanced boundary          # vllm/v1/core/sched/scheduler.py sched.A2
  ├─ model_executor.execute_model       -> model output future         # vllm/v1/engine/core.py
  └─ Scheduler.update_from_output       -> appended output IDs          # vllm/v1/core/sched/scheduler.py sched.A2
```

| Step | Artifact | Type | Location | Illustrative sample |
|---|---|---|---|---|
| sched.A1 | Scheduled counts | `dict[str,int]` | CPU SchedulerOutput | `{"R":16}` |
| sched.A2 | Request boundaries | Integer counters and ID list | CPU Request | computed=35, known=36 |

## sched.A Scheduling

### sched.A1 Budget the uncomputed suffix (`vllm/v1/core/sched/scheduler.py`)

**Idea.** The scheduler tries to make the computed boundary catch up with known tokens. It serves running requests before considering waiting requests. The 35-token request needs three cold prefill chunks with a 16-token compute budget.

**Input.** One waiting request R with 35 constructed IDs; compute budget=16, input budget=16, draft slots=0. The prefix lookup returns either 0 or 32 already computed tokens.

**Output.** CPU `SchedulerOutput.num_scheduled_tokens`, illustrated by `{"R":16}` for the cold first chunk and `{"R":3}` for the warm first chunk. The per-step budget is remaining compute capacity, not a maximum total prompt length.

**Code.** Waiting requests use known token count rather than only prompt count, so resumed requests include their generated IDs:

```python
# vllm/v1/core/sched/scheduler.py  Scheduler.schedule
                    request_token_budget = min(token_budget, input_budget - draft_slots)
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
```

Later, the waiting branch applies the threshold, checks whether chunking is allowed, and caps by request budget. The running branch uses this arithmetic:

```python
# vllm/v1/core/sched/scheduler.py  Scheduler.schedule
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
```

`num_tokens_with_spec` equals known IDs plus draft IDs. In this example drafts and output placeholders are zero. The two budgets start at 16 each. After successful allocation, both lose the number scheduled; the input budget also loses per-request draft slots when drafting is active. A prefix hit of 32 spends neither budget on those 32 tokens.

**Example.** Alternative executions:

| Request state | Remaining known inputs | Scheduled | Budget left |
|---|---:|---:|---:|
| Cold waiting | 35−0=35 | 16 | 0 |
| Cold running | 35−16=19 | 16 | 0 |
| Cold running | 35−32=3 | 3 | 13 |
| Warm waiting | 35−32=3 | 3 | 13 |
| Decode after token 101 | 36−35=1 | 1 | 15 |
| Decode after token 102 | 37−36=1 | 1 | 15 |

**Math.** In the selected running branch, let K be known inputs, C the computed boundary, Q remaining compute budget, I remaining input budget, and L maximum model length. With no drafts/placeholders and the threshold disabled: scheduled=`min(K−C,Q,I,L−C−1)`. For the first decode, `min(36−35,16,16,128−35−1)=1`. The last limit reserves space for the sampled output position. The waiting branch has different code; do not apply this formula indiscriminately to all branches.

**Tests.** Original running arithmetic AST statements pass on the cold sequence 16,16,3,1,1. This does not test the waiting queue or the complete scheduler. Repository `test_schedule_concurrent_partial_requests` in `tests/v1/core/test_scheduler.py` source-reviews budget caps and subsequent one-token decodes; not run because upstream collection needs `tblib` and the vLLM environment. Gap: execute the complete toy lifecycle with the real scheduler.

### sched.A2 Advance counters before GPU completion (`vllm/v1/core/sched/scheduler.py`)

**Idea.** The scheduler must send the old computed boundary to the worker so the worker selects the correct input positions. It then advances its own accounting before model execution completes.

**Input.** R final prefill starts at computed=32 and schedules three positions. `SchedulerOutput` retains the starting boundary; the live Request is mutable CPU state.

**Output.** After scheduling, computed=35 and in-flight grows by 3. After the synchronous worker completes and outputs token 101, known token count becomes 36. The sampled token itself has no computed KV yet. These are illustrative lifecycle values from source review, not a full runtime measurement.

**Code.**

```python
# vllm/v1/core/sched/scheduler.py  Scheduler._update_after_schedule
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
```

The name `num_computed_tokens` therefore does not alone prove GPU readiness. `EngineCore.step` in `vllm/v1/engine/core.py` schedules, executes, waits for the future, optionally calls the separate sampler, and updates the scheduler from the output. `Request.append_output_token_ids` in `vllm/v1/request.py` extends both output IDs and all known IDs.

**Example.**

```text
before final prefill:       known=35, computed=32
schedule three positions:  known=35, computed=35, in_flight increases by 3
worker completes:          KV positions 0..34 are ready (source-reviewed boundary)
append output ID 101:      known=36, computed=35
decode input ID 101:       schedule one; computed=36
append output ID 102:      known=37, computed=36
```

This trace is sequential: the next schedule follows output processing. It does not prove readiness for overlapping async batches.

**Tests.** The upstream concurrent-partial-request test explicitly supplies empty sampled lists for incomplete chunks and one sampled ID for finished prefills. Status: not run (needs deps: tblib, vLLM runtime). The standalone test checks count arithmetic only. Gap: actual Request and worker lifecycle for this exact example.

## Other scheduler branches

| Variant | Current switch | Effect outside the example |
|---|---|---|
| No chunking | `enable_chunked_prefill=false` | Waiting branch can stop when the suffix exceeds the budget |
| Separate compute cap | `max_num_scheduled_tokens` | Can be below input budget; falls back to input cap when absent |
| Long prefill cap | `long_prefill_token_threshold>0` | Reduces per-request work before the global cap |
| Full input admission | `scheduler_reserve_full_isl` | Checks full sequence capacity before chunk admission |
| Encoder/drafts/Mamba | Input/spec/model configuration | Additional caps, reservations and alignment |

These are source-reviewed alternatives. Their branch combinations were not executed in the standalone contracts.
