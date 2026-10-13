# KV cache: a reused prefix still needs block ownership

This document covers full-attention block metadata for R. [scheduler.md](scheduler.md) covers the token budget. Paths below are relative to the source repository root.

## Contents

- [Overview](#overview)
- [kv.A Prefix and capacity](#kva-prefix-and-capacity)
- [Visual trace](#visual-trace)

## Overview

The KV manager gives a request ownership of the memory blocks its next step needs.

1. Find complete prefix blocks with matching context.
2. Check capacity, attach reused blocks, and reserve blocks for the new suffix.
3. Record cache metadata; the worker later writes new keys and values.

**Example (illustrative).** A 35-token prompt with block size 16 can reuse
blocks 1 and 2 for the first 32 tokens. It still needs block 3 for the tail.
A block ID or a hash entry alone does not prove that new KV values are ready.


```text
KVCacheManager.get_computed_blocks       -> grouped hits, token count, boundary # vllm/v1/core/kv_cache_manager.py kv.A1
  └─ UnitaryKVCacheCoordinator.find_longest_cache_hit -> one-group result        # vllm/v1/core/kv_cache_coordinator.py kv.A1
      └─ FullAttentionManager.find_longest_cache_hit -> prefix block metadata    # vllm/v1/core/single_type_kv_cache_manager.py kv.A1
KVCacheManager.allocate_slots            -> newly allocated blocks or None      # vllm/v1/core/kv_cache_manager.py kv.A2
  ├─ coordinator.get_num_blocks_to_allocate -> capacity demand                  # vllm/v1/core/kv_cache_manager.py kv.A2
  ├─ coordinator.allocate_new_computed_blocks -> prefix ownership               # vllm/v1/core/kv_cache_manager.py kv.A2
  ├─ coordinator.allocate_new_blocks     -> new suffix capacity                 # vllm/v1/core/kv_cache_manager.py kv.A2
  └─ coordinator.cache_blocks            -> full-block hash index metadata      # vllm/v1/core/kv_cache_manager.py kv.A3
```

| Step | Artifact | Type | Location | Illustrative sample |
|---|---|---|---|---|
| kv.A1 | Computed blocks | `KVCacheBlocks` and two integers | CPU metadata | `([1,2],),32,0` |
| kv.A2 | New suffix blocks | `KVCacheBlocks` or `None` | CPU metadata referring to GPU cache | `([3],)` |
| kv.A3 | Cache index | Hash-to-block metadata map | CPU BlockPool | h1 maps to block 1 |

## kv.A Prefix and capacity

A cache hit can save computation only when the request can own the required pages.

1. Look up matching complete prefix blocks.
2. Check capacity, attach the hits, and allocate the missing suffix blocks.
3. Register eligible hash metadata, keeping it distinct from completed KV.

**Input/output (illustrative).** A 35-token request reuses blocks 1 and 2 for
32 prefix tokens and reserves block 3 for the final three input tokens.


### kv.A1 Look up the prefix, with a last-token limit (`vllm/v1/core/kv_cache_manager.py`)

**Idea.** Cache hits skip initial model work. They require matching preceding context, not merely identical tokens in one block.

**Input.** R IDs `[1,...,35]`; two full 16-token blocks and three tail tokens. A completed prior request left block 1 for IDs 1–16 and block 2 for IDs 17–32.

**Output.** The current `get_computed_blocks` returns three values: grouped block metadata, hit-token count, and shared-prefix boundary. Warm output is `([1,2],),32,0`; cold output is `([],),0,0`. The scheduler's `_get_local_prefix_cache_hit` wrapper adds a fourth boolean for hybrid/connector logic. That boolean is false on this path.

**Code.**

```python
# vllm/v1/core/kv_cache_manager.py  KVCacheManager.get_computed_blocks
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # When kv_cache_report_mode is "full", emit BlockStored events
        # for the reused prefix cache blocks so that external consumers
```

The single-group coordinator passes physical block size 16 as the alignment. The full-attention lookup searches matching blocks from the start and stops on its first miss. Equal hash and physical granularity means no fine-grained partial hit in this example.

`hash_block_tokens` in `vllm/v1/core/kv_cache_utils.py` chains the parent hash, current IDs and extra keys:

```python
# vllm/v1/core/kv_cache_utils.py  hash_block_tokens
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


```

**Example.** Hash labels h1 and h2 are schematic, not measured digest bytes. h2 represents the prefix ending at position 32. Changing the first block invalidates that initial prefix. R's tail has only three IDs, so it contributes no full-block hash at granularity 16.

| Input length | Lookup limit | Maximum whole blocks | Warm hit |
|---|---:|---:|---:|
| 35 | 34 | floor(34/16)=2 | 32 tokens |
| 32 | 31 | floor(31/16)=1 | 16 tokens |

For an exactly 32-token prompt, both physical blocks might be cached, but this selected alignment replays the entire second block to obtain last-position logits. Do not generalize that whole-block result to the newer fine-grained hit branch.

**Tests.** Original full-attention lookup passes warm hit=32, changed-first-block miss=0, and exact-32 cap=16 with a fake cache map. Original manager method also passes normal and skipped-lookup branches. Fake hash resolution is identity because granularity is equal. `test_prefill` in `tests/v1/core/test_prefix_caching.py` checks miss/hit blocks and references with real managers, but was not run (needs deps: tblib, vLLM runtime). Hash construction here is source-reviewed, not executed by the standalone contracts.

### kv.A2 Attach cached blocks, allocate the suffix (`vllm/v1/core/kv_cache_manager.py`)

**Idea.** Saved compute does not remove the need to hold the prefix's KV. Attaching idle cached blocks removes them from free capacity.

**Input.** Warm hit=32, new computation=3, no lookahead or external tokens. R has no blocks attached yet. Pool has 11 total blocks; block 0 is null and 10 are usable. Two idle cached hits are in the free queue.

**Output.** Logical block table `([1,2,3],)` covers 48 slots. The allocation return contains only newly allocated block 3, `([3],)`. Full table lookup is separate. Idle blocks 1 and 2 become owned by R; each ref count moves from 0 to 1. Free capacity falls from 10 to 7. These physical IDs and ownership counts are illustrative; the standalone test validates the required count with stand-ins, not the real queue.

**Code.** The normal admission check runs before attachment and new allocation:

```python
# vllm/v1/core/kv_cache_manager.py  KVCacheManager.allocate_slots
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )
```

The source has extra gates for full-sequence admission, watermarks and in-flight reservations. The example disables full-sequence reservation, has no connector reservation, and admits no waiting request alongside running work, so no watermark applies. `SingleTypeKVCacheManager.get_num_blocks_to_allocate` counts both newly needed blocks and cached hits whose ref count is zero. `BlockPool.touch` in `vllm/v1/core/block_pool.py` removes idle hit blocks from the free queue and increments refs.

**Example.**

```text
logical positions    physical block    R owns after allocation    scheduled new work
0..15                1 (prefix hit)    yes                       none
16..31               2 (prefix hit)    yes                       none
32..34               3 (new suffix)    yes                       three tokens
35..47               3 (unused slots)  yes                       none
```

**Math.** Required logical blocks=`ceil((32+3)/16)=3`. New blocks=`3−2=1`. Free-capacity demand=`1 new + 2 idle hits=3`. With only two free blocks, allocation returns `None` before prefix attachment or new-block allocation. If the two hits were actively referenced by another request, touching would increase refs without removing them from the free queue; the required free count would differ.

For the cold path, three prefill chunks allocate one block each: after 16 inputs, `[1]`; after 32, `[1,2]`; after 35, `[1,2,3]`. Both decode inputs still fit block 3, so no fourth block is needed. Position 48 would require one.

**Tests.** Original count method returns demand=3 for two idle prefix hits plus one new block, zero for a running request extending within block 3, and one at 49 required positions. Original `allocate_slots` passes success and failure call-order contracts with a fake coordinator. The repository `test_prefill` covers actual `ref_cnt` and free-queue behavior; not run (needs deps: tblib, vLLM runtime).

### kv.A3 Separate hash registration from ready KV (`vllm/v1/core/single_type_kv_cache_manager.py`)

**Idea.** Hash computation, capacity allocation, index registration and completed KV computation are different events. The source registers full finalized blocks during allocation, before the executor runs.

**Input.** R has attached prefix blocks and capacity for 35 positions. Known token IDs are finalized; there are no rejected draft tokens. Two full blocks exist, with a three-token partial tail.

**Output.** The CPU hash index covers full blocks only. After scheduling a cold 16-token chunk, block 1 can already have hash metadata while its GPU KV is pending. After execution completes, those 16 positions become ready. This example's warm cache explicitly comes from completed work, avoiding an untested async readiness assumption.

**Code.**

```python
# vllm/v1/core/kv_cache_manager.py  KVCacheManager.allocate_slots
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

```

`SingleTypeKVCacheManager.cache_blocks` uses `num_full_blocks = num_tokens // self.block_size`, then delegates to `BlockPool.cache_full_blocks` in `vllm/v1/core/block_pool.py`. That method updates block hashes and the lookup index. Neither method performs model attention. On the selected engine path, GPU execution happens later in `EngineCore.step`.

**Example.**

| Event | Index metadata | Ready KV from this request |
|---|---|---|
| Cold chunk 1 allocated/scheduled | h1 can map to block 1 | none yet for this chunk |
| Cold chunk 1 execution completes | h1 remains registered | positions 0–15 |
| Cold chunk 2 allocated/scheduled | h2 can map to block 2 | only previous positions 0–15 before execution |
| Final prefill execution completes | h1 and h2 indexed; block 3 partial | positions 0–34 |
| Warm R admitted after prior producer completed | existing h1/h2, ref counts increase | prefix 0–31 already ready |

A freed hashed block can remain cached with ref count zero. Allocation may later evict it. `BlockPool.free_blocks` currently prepends unhashed blocks and appends hashed blocks, so its queue order is more specific than a blanket “append all freed blocks” description. That order is source-reviewed; this trace has no eviction or finishing R.

**Tests.** The original allocation contract proves the coordinator's cache registration is called after new allocation, within `allocate_slots`. It does not prove GPU readiness. Repository `test_cache_blocks` and `test_prefill` in `tests/v1/core/test_prefix_caching.py` inspect block metadata, not a full model run; not run (needs deps: tblib, vLLM runtime). Gap: completed-prefix replay compared with uncached logits on a real supported model.

## Visual trace

The states use the exact values above. Display durations are illustrative, not latency measurements. Green means ready reused KV; amber means newly scheduled pending work. The index label states registration separately from KV readiness.

![Warm allocation: two idle prefix hits plus one suffix block](assets/cache-frames/frame-005.svg)

<!-- explain-html-only
<iframe src="assets/cache-player.html" title="Step through scheduler and cache state" height="720" sandbox="allow-scripts" loading="lazy"></iframe>
-->

[Open the standalone trace](assets/cache-player.html). The [trace JSON](assets/cache-trace.json) and SVG frames remain editable. Without JavaScript, the player exposes static states.
