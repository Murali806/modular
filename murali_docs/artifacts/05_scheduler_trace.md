# Phase 5: Continuous Batching Scheduler

## Iteration Machine

`SOURCE + CPU-RUN`

```text
request ZMQ queue
       |
       v
retrieve pending -> CE/TG queues -> token + KV budget -> replica batch
                                              |              |
                                              | no fit       v
                                              +--------> preempt TG
                                                             |
                                                             v
pipeline.execute -> generated token -> advance/release -> next iteration
```

```mermaid
flowchart LR
    Z[ZMQ request queue] --> R[Retrieve pending]
    R --> Q[Replica CE / TG queues]
    Q --> B[Create token budgets]
    B --> P[Pick CE or TG priority]
    P --> C[Construct replica batch]
    C --> E[Pipeline execute]
    E --> A[Advance requests]
    A -->|more work| Q
    A -->|done| F[Free KV + respond]
    C -->|KV pressure| X[Preempt TG]
    X --> Q
```

## Mixed Prefill + Decode

`CPU-RUN`: batch size `3`, active-token budget `8`, in-flight batching on.

```text
iteration    1             2                 3             4
          +---------+  +-------------+  +-----------+  +-------+
A         | CE 4    |->| TG 1        |->| TG 1      |->| TG 1  | done
B         | CE 3    |->| TG 1        |->| TG 1 done |
C                       | CE 5 joins |->| TG 1      |->| TG 1  | done
          +---------+  +-------------+  +-----------+  +-------+
tokens        7             7                3            2
```

```mermaid
sequenceDiagram
    autonumber
    participant Q as Admission queue
    participant S as Scheduler
    participant B as TextBatchConstructor
    participant KV as Paged KV manager
    participant P as Fake pipeline

    Q->>S: A(prompt=4), B(prompt=3)
    S->>B: iteration 1 / budget=8
    B->>KV: reserve A + B
    B->>P: CE[A:4, B:3]
    P-->>B: first token for A + B
    Q->>S: C(prompt=5)
    S->>B: iteration 2 / budget=8
    B->>KV: reserve C
    B->>P: TG[A:1, B:1] + CE[C:5]
    P-->>B: next A/B + first C
    S->>B: iteration 3
    B->>P: TG[A:1, B:1, C:1]
    P-->>B: B done
    S->>B: iteration 4
    B->>P: TG[A:1, C:1]
    P-->>B: A + C done
```

| Iteration | Selected work                | Input tokens | KV pages after | Result                |
|----------:|------------------------------|-------------:|---------------:|-----------------------|
|         1 | `A CE:4`, `B CE:3`           |            7 |              7 | A/B enter TG          |
|         2 | `A TG:1`, `B TG:1`, `C CE:5` |            7 |             14 | C joins active decode |
|         3 | `A TG:1`, `B TG:1`, `C TG:1` |            3 |             12 | B completes           |
|         4 | `A TG:1`, `C TG:1`           |            2 |              0 | A/C complete          |

## Budget Passport

`SOURCE`

```text
active-token budget
  CE contributes its selected prompt chunk
  TG contributes one active token

total-context budget (when configured)
  rounds KV demand to page boundaries
  prevents a token-valid batch from exceeding KV capacity
```

```mermaid
flowchart TD
    I[Candidate request] --> T{Active-token room?}
    T -->|no| W[Wait]
    T -->|yes| K{KV/context room?}
    K -->|yes| A[Add to batch]
    K -->|CE partial fit| C[Chunk prompt]
    K -->|TG pressure| P[Preempt lower-priority TG]
    C --> A
    P --> W
```

## Chunked Prefill

`CPU-RUN`: prompt `11`, output limit `2`, budget `4`.

```text
NEW -> CE[0:4] -> CE[4:8] -> CE[8:11] + sample -> TG[1] -> DONE
          4             4             3                1
```

```mermaid
stateDiagram-v2
    [*] --> CE0
    CE0: active prompt 0..3
    CE0 --> CE1: advance chunk / processed=4
    CE1: active prompt 4..7
    CE1 --> CE2: advance chunk / processed=8
    CE2: active prompt 8..10
    CE2 --> TG: prompt complete + first token
    TG --> Done: final token
    Done --> [*]
```

| Iteration | Kind | Active | Processed after | Queue after |
|----------:|------|-------:|----------------:|-------------|
|         1 | CE   |      4 |               4 | CE head     |
|         2 | CE   |      4 |               8 | CE head     |
|         3 | CE   |      3 |              11 | TG          |
|         4 | TG   |      1 |              12 | complete    |

## KV-Pressure Preemption

`CPU-RUN`: two ten-token requests share only five pages of two tokens.

```text
pages used   i1  i2  i3  i4  i5  i6  i7  i8
             4   4   5   5   4   4   0   3

P            CE  TG  TG  TG  TG  TG  TG  DONE
Q            CE  TG  wait wait PREEMPT->CE wait wait CE(replay)
```

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant B as Batch constructor
    participant KV as 5-page KV cache
    participant P as Request P
    participant Q as Request Q

    S->>B: CE P + Q
    B->>KV: allocate 4 pages
    S->>B: TG P + Q
    B->>KV: page 5 becomes occupied
    S->>B: later TG step
    B->>KV: no growth room for both
    B->>Q: rewind + move TG to CE
    B->>KV: release Q pages
    S->>P: continue until done
    P->>KV: release all P pages
    S->>Q: re-admit CE
    Q->>KV: rebuild cache state
```

```mermaid
stateDiagram-v2
    [*] --> CE
    CE --> TG: prompt encoded
    TG --> Waiting: another TG gets scarce KV
    Waiting --> Preempted: selected as victim
    Preempted --> CE: processing rewound
    CE --> TG: cache rebuilt
    TG --> Done: output limit / EOS
    Done --> [*]
```

## Data-Parallel Placement

`CPU-RUN + SOURCE`

```text
arrival: D0 D1 D2 D3

least-loaded choice
  replica 0: D0, D2
  replica 1: D1, D3

global staged batch
  [D0 D2] [D1 D3]
```

```mermaid
flowchart TD
    Q[D0, D1, D2, D3] --> L{Least active replica}
    L --> R0[Replica 0: D0, D2]
    L --> R1[Replica 1: D1, D3]
    R0 --> G[Flattened global ragged batch]
    R1 --> G
    G --> DP[Per-replica execution]
```

Padding is `SOURCE` only in this probe: production can install
`DPBatchPadder`; the repository test helper does not.

## Reproduce

```bash
murali_docs/labs/scheduler/run_scheduler_trace.sh --compact \
  > /tmp/phase5_scheduler_trace.json
```

## Source Pins

| Decision                          | Source                                                                                                          |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| budgets + CE/TG selection         | [`text_batch_constructor.py`](../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py) |
| budget counters                   | [`token_budget.py`](../../max/python/max/serve/scheduler/batch_constructor/token_budget.py)                     |
| scheduler iteration               | [`text_generation_scheduler.py`](../../max/python/max/serve/scheduler/text_generation_scheduler.py)             |
| deterministic pipeline/KV doubles | [`common.py`](../../max/tests/tests/serve/scheduler/common.py)                                                  |
| expected scheduler traces         | [`test_paged_scheduler.py`](../../max/tests/tests/serve/scheduler/test_paged_scheduler.py)                      |
