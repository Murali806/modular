# Phase 11: Paged KV Cache

`CPU-RUN` + `SOURCE` + `TEST` + `GPU-LAB`

## One Token's KV Payload

`SOURCE`

```text
token t
  |
  +-- layer 0  : K[3 heads x 64] + V[3 heads x 64]
  +-- layer 1  : K[3 heads x 64] + V[3 heads x 64]
  |      ...
  +-- layer 29 : K[3 heads x 64] + V[3 heads x 64]

FP32 bytes/token = 2 x 30 x 3 x 64 x 4 = 46,080 B
128-token page   = 46,080 x 128         = 5,898,240 B = 5.625 MiB
```

```mermaid
block-beta
  columns 5
  T["token t"] space L["30 layers"] space KV["K + V"]
  space space H["3 KV heads"] space D["64 x FP32"]
  T --> L
  L --> KV
  KV --> H
  H --> D
```

| Model A payload                         |    Result |
|-----------------------------------------|----------:|
| One token                               |  46,080 B |
| One 128-token page                      | 5.625 MiB |
| Whole pages in an ideal 1 GiB KV budget |       182 |
| Token slots in those pages              |    23,296 |

`ideal payload != deployable budget`

```text
device memory
  - weights
  - graph/runtime workspace
  - allocator reserve
  - communication buffers
  = memory left for KV pages
```

## Logical Tokens, Physical Pages

`SOURCE`

```text
request R logical positions
  0..127       128..255      256..383
     |             |             |
     v             v             v
page table R = [ block 7,      block 2,      block 11 ]
                    |             |              |
device pool      [page 2] ... [page 7] ... [page 11]
```

```mermaid
flowchart LR
    T0[Tokens 0..127] --> B7[Physical block 7]
    T1[Tokens 128..255] --> B2[Physical block 2]
    T2[Tokens 256..383] --> B11[Physical block 11]
    B7 --> KV7[K/V across every layer]
    B2 --> KV2[K/V across every layer]
    B11 --> KV11[K/V across every layer]
```

## Block Lifecycle

`CPU-RUN + SOURCE`

```text
 FREE
   | alloc: ref 0 -> 1
   v
 ACTIVE ---- full page + hash ----> COMMITTED + ACTIVE
   | release: ref -> 0                         |
   |                                          | release
   v                                          v
 FREE + CACHED + EVICTABLE <---- hit/touch ---- CACHED + ACTIVE
   |
   | LRU allocation under pressure
   v
 EVICT HASH -> ACTIVE FOR NEW OWNER
```

```mermaid
stateDiagram-v2
    [*] --> Free
    Free --> Active: alloc_block
    Active --> CommittedActive: full page + hash
    CommittedActive --> CachedFree: release / ref=0
    CachedFree --> CommittedActive: hit / touch
    CachedFree --> Active: LRU alloc / uncommit
    Active --> Free: release uncommitted
```

The two-block production-class probe observed:

| Event                            | Free LRU order | Prefix cache        |
|----------------------------------|----------------|---------------------|
| initial                          | `[0, 1]`       | `{}`                |
| commit A, release                | `[1, 0]`       | `A -> 0`            |
| hit A                            | `[1]`          | `A -> 0`            |
| release A, then commit/release B | `[0, 1]`       | `A -> 0, B -> 1`    |
| allocate LRU                     | `[1]`          | `B -> 1`; A evicted |

## Prefix Hash Chain

`CPU-RUN + SOURCE`

```text
33 prompt tokens, block size 8

[0..7] --H(parent,tokens)--> h0
 h0 + [8..15] ------------> h1
 h1 + [16..23] -----------> h2
 h2 + [24..31] -----------> h3
 token 32 remains active; a 100% hit is forbidden
```

```mermaid
sequenceDiagram
    actor C as Client request B
    participant X as TextContext
    participant M as BlockManager
    participant H as hash_request_tokens
    participant P as BlockPool
    participant K as Prefix cache
    C->>X: same leading token pages
    X->>M: reuse_blocks_from_prefix_cache
    M->>H: chained full-page hashes
    H-->>M: h0, h1, ...
    loop contiguous hashes
        M->>K: lookup hash
        K-->>M: physical block
        M->>P: touch block
    end
    M->>X: skip_processing(cached tokens)
    X-->>C: prefill only uncached suffix
```

Observed lookup invariants:

```text
cache = [h0, h1, h2, --]  -> hit = 3 pages
cache = [h0, --, h2, --]  -> hit = 1 page
                                  ^ stop at first gap

17-token request; two 8-token pages cached
  -> skipped 16
  -> processed 16
  -> active 1
  -> no transfer with NullConnector
```

```mermaid
flowchart LR
    H0[h0 hit] --> H1{h1?}
    H1 -->|hit| H2{h2?}
    H1 -->|miss| Stop1[Reusable prefix = 1]
    H2 -->|hit| H3{h3?}
    H2 -->|miss| Stop2[Reusable prefix = 2]
```

## Live HTTP Prefix Comparison

`CPU-RUN`; five seed/shared pairs per server mode.

```text
prefix caching enabled
  seed   157 tokens -> 157 misses -> retain first 128-token page
  shared 157 tokens -> 128 hits + 29 misses -> shorter prefill

prefix caching disabled
  seed/shared -> full prefill; cache counters remain inactive
```

```mermaid
sequenceDiagram
    actor C as External client
    participant A as MAX API
    participant K as Prefix cache
    participant G as CPU graph
    C->>A: seed prompt, 157 tokens
    A->>K: lookup
    K-->>A: 0 hit, 157 miss
    A->>G: full prefill
    G-->>K: retain full 128-token page
    C->>A: same prefix + new suffix
    A->>K: lookup
    K-->>A: 128 hit, 29 miss
    A->>G: prefill uncached suffix
```

| Mode     | G0 hit tokens | Miss tokens | Seed TTFT median | Shared TTFT median |
|----------|--------------:|------------:|-----------------:|-------------------:|
| enabled  |           640 |         930 |         57.11 ms |           24.69 ms |
| disabled |             0 |           0 |         41.16 ms |           41.36 ms |

Enabled totals cover five pairs: exactly `5 x 128 = 640` cached tokens. With
caching disabled, MAX does not publish hit/miss work for these requests.
Timings are a semantic CPU check, not a production speed claim.

[Machine-readable summary](11_cpu_prefix_cache_summary.json)

## Tier Walk

`SOURCE + BOUNDARY`

```text
hash chain
   |
   v
device prefix cache (G0)
   | first miss; remaining suffix only
   v
KVConnector
   +--> host tier --H2D--> device page
   +--> disk/dKV ----promotion----^

all leaves agree on one contiguous prefix
```

```mermaid
sequenceDiagram
    participant M as BlockManager
    participant G as Device BlockPool
    participant C as KVConnector
    participant H as Host/disk/dKV tier
    participant D as Device KV memory
    M->>G: walk hashes from root
    G-->>M: N contiguous device blocks
    M->>C: lookup remaining hashes
    C->>H: reconcile resident leaves
    H-->>C: longest joint prefix
    C-->>M: external hit count
    M->>D: allocate destination pages
    C->>D: load / copy cached KV
```

No host/device transfer was measured on this CPU-only machine.

## Pressure and Preemption

`SOURCE + TEST`

```text
new/chunked request
      |
      v
blocks needed <= free? -- yes --> allocate --> CE/TG batch
      |
      no
      v
select victim -> release pages -> rewind processing -> pending CE
      |                                      |
      +--------------- later ----------------+
                         re-prefill
```

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Allocating: scheduler admission
    Allocating --> Active: enough pages
    Allocating --> Preempted: insufficient pages
    Active --> Preempted: growth exceeds free pages
    Preempted --> Pending: release + rewind
    Active --> Complete: EOS / limit / cancel
    Complete --> CachedFree: release; committed pages retained
```

Phase 5 already proved KV-pressure preemption and re-prefill on CPU.

## Page to Mojo Kernel

`SOURCE + GPU-LAB`

```text
BlockManager page IDs + row offsets
             |
             v
PagedCacheValues / graph attention op
             |
             v
kv_cache_ragged.mojo
  - store K/V into page
  - read historical K/V
  - dispatch prefill/decode attention
             |
             v
CPU kernel [available] | GPU kernel [measure later]
```

```mermaid
flowchart TD
    BM[BlockManager metadata] --> PC[PagedCacheValues]
    PC --> OP[Attention graph/custom op]
    OP --> MJ[kv_cache_ragged.mojo]
    MJ --> CPU[CPU implementation]
    MJ --> GPU[GPU dispatch and launch]
    GPU --> LAB[GPU-LAB: bandwidth, occupancy, kernel time]
```

## Capacity Diagnosis

`GPU-LAB`

```text
                 saturation symptom
                        |
       +----------------+----------------+
       |                |                |
       v                v                v
 compute busy      HBM bytes/token   KV pages exhausted
 large CE GEMMs    decode GEMV/MHA   preemptions + queue
       |                |                |
   compute-bound   bandwidth-bound   capacity-bound
```

Measure before choosing a branch: TTFT, TPOT, batch shape, queue depth,
KV utilization, preemptions, accelerator utilization, and kernel mix.

## Reproduce

```bash
murali_docs/labs/kv_cache/run_kv_cache_trace.sh --compact \
  > /tmp/phase11_kv_cache_trace.json

# Run once against the default server, then restart it with
# --no-enable-prefix-caching and run the disabled mode.
python3 murali_docs/labs/kv_cache/http_prefix_cache_probe.py \
  --mode enabled --trials 5
python3 murali_docs/labs/kv_cache/http_prefix_cache_probe.py \
  --mode disabled --trials 5
```

## Source Pins

- `max/python/max/pipelines/kv_cache/paged_kv_cache/block_pool.py`: block
  allocation, reference counts, cache commits, touch, and LRU eviction.
- `max/python/max/pipelines/kv_cache/paged_kv_cache/block_manager.py`: chained
  hashes, contiguous lookup, reuse, tier loading, allocation, and release.
- `max/python/max/pipelines/kv_cache/prefix_hit.py`: multi-leaf longest-prefix
  reconciliation.
- `max/python/max/pipelines/kv_cache/kv_connector.py`: external tier boundary.
- `max/kernels/src/nn/kv_cache_ragged.mojo`: paged K/V stores and attention.
- `max/tests/integration/nn/kv_cache/test_prefix_cache_hit_counts.py`: gap,
  read-only, device, and external-tier invariants.
- `docs/max/serve/prefix-caching.mdx`: serving flags and supported scope.
