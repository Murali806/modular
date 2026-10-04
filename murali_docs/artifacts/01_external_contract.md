# Phase 1: External Python -> MAX Serve

## Contract Surface

`CPU-RUN`

```text
external Python 3.10
       |
       | GET /health, GET /v1/models
       | POST /v1/chat/completions
       v
+-------------------- MAX Serve :18000 --------------------+
| JSON validation -> request id -> generation -> response  |
+----------------------------------------------------------+
       |
       +--> JSON response
       +--> SSE: data:{chunk}\n\n ... data:[DONE]\n\n
       +--> OpenAI-shaped 4xx error
```

```mermaid
flowchart LR
    C[External Python] -->|GET| H["/health"]
    C -->|GET| M["/v1/models"]
    C -->|POST JSON| R["/v1/chat/completions"]
    R -->|stream false| J[JSON completion]
    R -->|stream true| S[SSE chunks + DONE]
    R -->|invalid| E[OpenAI error envelope]
```

## Ingress + Admission Sequence

`SOURCE`, with endpoint behavior confirmed by `CPU-RUN`.

```text
Client -> Uvicorn -> request middleware -> OpenAI route -> Pydantic schema
                                                        |
                                                        v
                                              TextGenerationRequest
                                                        |
                                                        v
response generator -> TokenGeneratorPipeline -> tokenizer -> TextContext
                                                        |
                                                        v
                                             ZMQ worker proxy -> IPC
```

```mermaid
sequenceDiagram
    autonumber
    actor Client as External Python
    participant HTTP as Uvicorn
    participant MW as Request middleware
    participant Route as OpenAI chat route
    participant Schema as Pydantic schema
    participant Resp as Chat response generator
    participant API as TokenGeneratorPipeline
    participant Tok as TextTokenizer
    participant Proxy as ZMQ worker proxy
    participant IPC as Request IPC queue

    Client->>HTTP: POST /v1/chat/completions
    HTTP->>MW: ASGI request
    MW->>MW: create request_id + timer
    MW->>Route: request
    Route->>Schema: parse + validate JSON
    Schema-->>Route: CreateChatCompletionRequest
    Route->>Route: sampling params + TextGenerationRequest
    Route->>Resp: stream(request)
    Resp->>API: next_token_chunk(request)
    API->>Tok: new_context(request)
    Tok-->>API: TextContext + prompt token IDs
    API->>Proxy: stream(request_id, context)
    Proxy->>IPC: bounded put(context)
    IPC-->>Proxy: admitted
    Proxy-->>API: async response stream
    API-->>Resp: token generator
    Resp-->>Route: generator ready before HTTP 200
```

## Scheduler + Egress Sequence

`SOURCE`

```text
IPC -> worker interface -> scheduler -> batch constructor
                                      |
                                      v
                     TextGenerationPipeline.execute
                                      |
                         +------------+------------+
                         v                         v
                  compiled Llama graph          sampler
                         |                         |
                         +---------- CPU ----------+
                                      |
                                      v
response IPC -> detokenizer -> OpenAI chunk -> SSE -> client iterator
```

```mermaid
sequenceDiagram
    autonumber
    participant IPC as Request IPC queue
    participant WI as Worker interface
    participant Sched as Token scheduler
    participant Batch as TextBatchConstructor
    participant Pipe as TextGenerationPipeline
    participant Model as Llama3Model
    participant Exec as Compiled MAX model
    participant CPU as CPU kernels/runtime
    participant Samp as Sampler graph
    participant Out as Response IPC queue
    participant API as API pipeline + detokenizer
    participant SSE as OpenAI SSE generator
    actor Client as External Python

    IPC->>WI: TextContext
    WI->>Sched: pending request
    Sched->>Batch: construct_batch()
    Batch-->>Sched: CE/TG inputs + KV metadata
    Sched->>Pipe: execute(inputs)
    Pipe->>Model: execute(model buffers)
    Model->>Exec: compiled model.execute(...)
    Exec->>CPU: selected kernels
    CPU-->>Exec: logits
    Exec-->>Pipe: model outputs
    Pipe->>Samp: sample next token
    Samp->>CPU: sampler kernels
    CPU-->>Samp: token ID
    Pipe-->>Sched: per-request output
    Sched->>Out: publish(request_id, output)
    Out-->>API: token chunk
    API->>API: buffered detokenization
    API-->>SSE: decoded delta + status
    SSE-->>Client: data: {...}
    SSE-->>Client: data: [DONE]
```

## Request State Machine

```text
RECEIVED -> VALIDATED -> TOKENIZED -> ADMITTED -> PENDING
    |           |                         |           |
    |           +--> 400                  +--> 429    v
    |                                             PREFILL
    |                                                |
    |                                                v
    +--------------------------------------------- DECODE
                                                     |
                         +---------------------------+----------------+
                         v                           v                v
                       DONE                      CANCELLED          ERROR
                         |
                         v
                    SSE [DONE]
```

```mermaid
stateDiagram-v2
    [*] --> Received
    Received --> Validated
    Received --> HTTP4xx: malformed/invalid
    Validated --> Tokenized
    Tokenized --> Admitted
    Tokenized --> HTTP429: queue full
    Admitted --> Pending
    Pending --> Prefill
    Prefill --> Decode
    Decode --> Decode: next token
    Decode --> Done: EOS or limit
    Pending --> Cancelled: disconnect
    Decode --> Cancelled: disconnect
    Prefill --> Error
    Decode --> Error
    Done --> SSE_DONE
    HTTP4xx --> [*]
    HTTP429 --> [*]
    Cancelled --> [*]
    Error --> [*]
    SSE_DONE --> [*]
```

## Observed Responses

`CPU-RUN`, warmed server, 2026-10-04.

| Call | Result |
|---|---|
| `GET /health` | `200`, empty body |
| `GET /v1/models` | `200`, model ID + `max_model_len: 256` |
| valid non-stream chat | `200`, 16 prompt + 8 completion tokens |
| invalid `messages` type | `400`, `invalid_request_error` |
| stream | first frame `22.31 ms`, `[DONE]` `117.73 ms` |
| disconnect | client closed after 2 data frames; server stayed healthy |

Compact non-stream result:

```json
{
  "finish_reason": "length",
  "content": "Here is a sample response:\n\n",
  "prompt_tokens": 16,
  "completion_tokens": 8,
  "total_tokens": 24
}
```

Raw SSE shape:

```text
 22.31 ms | data: {... "delta":{"content":"Here","role":"assistant"} ...}
 33.69 ms | data: {... "delta":{"content":" is","role":"assistant"} ...}
117.69 ms | data: {... "finish_reason":"length" ...}
117.73 ms | data: [DONE]
```

## Metrics Cross-Check

`CPU-RUN`, snapshot after two completed chats plus one disconnected stream.

```text
HTTP middleware -> request_count{path,code}
scheduler       -> CE/TG batch counters
pipeline        -> input/output token counters
streaming layer -> TTFT / ITL / request time
```

| Signal | Value |
|---|---:|
| chat `200` / `400` | `3 / 1` |
| CE batches / TG batches | `3 / 16` |
| input / output tokens | `49 / 18` |
| mean TTFT | `29.01 ms` |
| mean recorded ITL | `13.72 ms` |
| KV prefix hits / misses | `0 / 49 tokens` |

## Header Commit Boundary

`SOURCE`

```text
parse / tokenize / worker handoff / fetch first item
                       |
             +---------+----------+
             |                    |
             v                    v
          failure              success
       HTTP 4xx/5xx       commit SSE HTTP 200
                                  |
                                  v
                         later error is in-stream
```

The route awaits submission and the first stream item before returning
`EventSourceResponse`.

## Reusable Client

```bash
python3 murali_docs/labs/client/max_serve_client.py --mode all

python3 murali_docs/labs/client/max_serve_client.py \
  --mode stream --max-tokens 64 --disconnect-after 2
```

Source pins:

- Route and schema conversion: [`openai_routes.py`](../../max/python/max/serve/router/openai_routes.py#L2217)
- `TextGenerationRequest`: [`openai_routes.py`](../../max/python/max/serve/router/openai_routes.py#L2482)
- SSE commit boundary: [`openai_routes.py`](../../max/python/max/serve/router/openai_routes.py#L2508)
- Tokenize + submit: [`llm.py`](../../max/python/max/serve/pipelines/llm.py#L294)
- Worker handoff: [`llm.py`](../../max/python/max/serve/pipelines/llm.py#L431)
- ZMQ proxy: [`zmq_interface.py`](../../max/python/max/serve/worker_interface/zmq_interface.py#L52)
- Scheduler iteration: [`text_generation_scheduler.py`](../../max/python/max/serve/scheduler/text_generation_scheduler.py#L205)
- Batch construction: [`text_batch_constructor.py`](../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1845)
- Model + sampling step: [`text_generation.py`](../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L515)
