# Streaming Versus Non-Streaming Response Branches

Sources:

- Branch in
  [`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L2508)
- `_start_stream(...)` in
  [`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L185)
- Chat response generator in
  [`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L687)
- Non-streaming `complete(...)` in
  [`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L1159)

## One-Line Purpose

This branch decides the HTTP response shape:

```text
stream=True   -> Server-Sent Events stream
stream=False  -> one final JSON response
```

It happens after `TextGenerationRequest(...)` is built, but before the client
receives the response body.

## Route Branch

```python
if completion_request.stream:
    error, token_stream = await _start_stream(
        await response_generator.stream(token_request)
    )
    if error is not None:
        return error
    return EventSourceResponse(token_stream, ping=100000, sep="\n")

response = await response_generator.complete([token_request])
return response
```

Visual:

```text
TextGenerationRequest
  |
  v
completion_request.stream?
  |
  +-- true
  |     |
  |     +-- response_generator.stream(token_request)
  |     +-- _start_stream(...)
  |     +-- EventSourceResponse(...)
  |     |
  |     v
  |   HTTP 200 text/event-stream
  |   token deltas arrive over time
  |
  +-- false
        |
        +-- response_generator.complete([token_request])
        |
        v
      HTTP 200 application/json
      full answer arrives at once
```

## Worked Example

Assume the route has already built:

```text
token_request = TextGenerationRequest(
  request_id=RequestID("req-123"),
  model_name="meta-llama/Llama-3.1-8B-Instruct",
  messages=[...],
)
```

Streaming request:

```text
INPUT: route state                      ACTION IN BRANCH                          OUTPUT TO CLIENT
------------------------------------    ----------------------------------------    ------------------------------------
completion_request.stream               choose branch                             streaming branch
  True                                ------------------------------->              response_generator.stream(...)

token_request                           submit request before SSE headers         async token generator
  TextGenerationRequest(...)          ------------------------------->              token_generator

token_generator                         _start_stream peeks first item            decision
  first item is token delta           ------------------------------->              ok: keep stream alive

EventSourceResponse                     wrap async generator                      HTTP response
  token_stream                        ------------------------------->              Content-Type: text/event-stream

model token chunk                       serialize delta frame                     SSE frame
  decoded_tokens="Hello"              ------------------------------->              data: {"delta":{"content":"Hello"}}

final model chunk                       serialize final frame                     SSE end
  finish_reason="stop"                ------------------------------->              data: [DONE]
```

Non-streaming request:

```text
INPUT: route state                      ACTION IN BRANCH                          OUTPUT TO CLIENT
------------------------------------    ----------------------------------------    ------------------------------------
completion_request.stream               choose branch                             non-streaming branch
  False                               ------------------------------->              response_generator.complete(...)

token_request                           run generation to completion              completed_outputs
  TextGenerationRequest(...)          ------------------------------->              [chunk1, chunk2, final_chunk]

completed_outputs                       concatenate decoded tokens                response text
  ["Hello", " world"]                 ------------------------------->              "Hello world"

response text + usage                   build OpenAI response object              HTTP response
  prompt_tokens=12                    ------------------------------->              CreateChatCompletionResponse(...)
  completion_tokens=2
```

## Streaming Path

```text
stream=True
  |
  v
response_generator.stream(token_request)
  |
  +-- pipeline.next_token_chunk(token_request)
  |     |
  |     +-- tokenizes request
  |     +-- submits context to model worker
  |     +-- returns async token generator
  |
  v
_start_stream(async_generator)
  |
  +-- read first item before SSE headers are committed
  |
  +-- first item is JSONResponse error?
  |     |
  |     +-- yes -> return HTTP error response
  |     |
  |     +-- no  -> put first item back in stream
  |
  v
EventSourceResponse(token_stream)
```

Why `_start_stream(...)` exists:

```text
Before SSE headers:
  route can still return HTTP 400/429/500 style error

After SSE headers:
  HTTP status is already 200
  errors must be sent inside the stream
```

## Streaming HTTP Shape

The client receives a stream of chunks.

```text
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"choices":[{"delta":{"content":"Hello"}}], ...}

data: {"choices":[{"delta":{"content":" world"}}], ...}

data: {"choices":[{"finish_reason":"stop"}], ...}

data: [DONE]
```

Conceptually:

```text
model produces token chunk
  -> decode chunk
  -> serialize OpenAI delta
  -> send SSE frame
  -> repeat until finished
```

## Non-Streaming Path

```text
stream=False
  |
  v
response_generator.complete([token_request])
  |
  +-- pipeline.all_tokens(token_request)
  |     |
  |     +-- tokenizes request
  |     +-- submits context to model worker
  |     +-- waits until generation is complete
  |
  +-- concatenate decoded token chunks
  |
  +-- parse tool calls, if enabled
  |
  +-- attach usage/logprobs/finish_reason
  |
  v
CreateChatCompletionResponse
```

## Non-Streaming HTTP Shape

The client receives one final JSON object.

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello world"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 2,
    "total_tokens": 14
  }
}
```

Conceptually:

```text
model produces all chunks
  -> collect chunks in memory
  -> concatenate text
  -> build final OpenAI response object
  -> return once
```

## Side-By-Side

| Question | Streaming | Non-streaming |
| --- | --- | --- |
| Request flag | `stream: true` | `stream: false` or omitted |
| Route method | `response_generator.stream(...)` | `response_generator.complete(...)` |
| HTTP wrapper | `EventSourceResponse` | `CreateChatCompletionResponse` JSON |
| Client sees | many delta events | one final object |
| First token latency | lower | hidden until full answer done |
| Error before headers | `_start_stream(...)` can still return error | normal exception/response path |
| Token handling | emit chunks as they arrive | collect all chunks, then format |

## Important Detail

Both branches use the same internal request:

```text
TextGenerationRequest
  |
  +-- streaming path
  |
  +-- non-streaming path
```

The branch changes the **HTTP response shape**, not the request object.

```text
Same generation work:
  tokenize
  schedule
  run model
  sample tokens

Different response packaging:
  stream deltas
  or final JSON
```

## Failure Shape

```text
Streaming path
  |
  +-- failure before first SSE event
  |     -> normal HTTP error response
  |
  +-- failure after SSE started
        -> error frame inside stream

Non-streaming path
  |
  +-- failure before final response
        -> normal HTTP error response
```

Short version:

```text
stream=True
  -> submit early
  -> peek first stream item
  -> return SSE stream

stream=False
  -> run to completion
  -> return one JSON response
```
