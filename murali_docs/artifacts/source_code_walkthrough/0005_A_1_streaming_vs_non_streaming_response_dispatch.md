# Complete Code Walk: Streaming Versus Non-Streaming Dispatch

Sources:
[`openai_routes.py` - branch](../../../max/python/max/serve/router/openai_routes.py#L2508),
[`openai_routes.py` - first-chunk gate](../../../max/python/max/serve/router/openai_routes.py#L185),
[`openai_routes.py` - streaming submit](../../../max/python/max/serve/router/openai_routes.py#L687),
[`openai_routes.py` - completion aggregation](../../../max/python/max/serve/router/openai_routes.py#L1159)

## Route Branch

```python
if completion_request.stream:
    # Resolve the submit and first chunk before the SSE headers go
    # out, so a failed handoff is an HTTP error.
    error, token_stream = await _start_stream(
        await response_generator.stream(token_request)
    )
    if error is not None:
        return error
    # We set a large timeout for ping otherwise benchmarking scripts
    # such as sglang will fail in parsing the ping message.
    return EventSourceResponse(token_stream, ping=100000, sep="\n")

response = await response_generator.complete([token_request])
return response
```

```text
completion_request.stream
          |
      +---+---+
      |       |
    True     False
      |       |
      |       +--> response_generator.complete([token_request])
      |                    |
      |                    v
      |          CreateChatCompletionResponse
      |
      +--> response_generator.stream(token_request)
                    |
                    v
              _start_stream(...)
                    |
              +-----+-----+
              |           |
         first item is   first token/event
         JSONResponse    or empty stream
              |           |
              v           v
        return error    EventSourceResponse
```

## Streaming Submission

```python
async def stream(
    self, request: TextGenerationRequest
) -> AsyncGenerator[str | JSONResponse, None]:
    # Submit the request before returning the response stream. Awaiting
    # next_token_chunk tokenizes and hands the request off to the model
    # worker, so a failed submission (e.g. a dead worker) raises here —
    # before the SSE 200 headers are sent — and the route maps it to an
    # HTTP error status.
    token_generator = await self.pipeline.next_token_chunk(request)
    return self._stream(request, token_generator)
```

```text
token_request
    |
    | await pipeline.next_token_chunk(request)
    |   - tokenize
    |   - create TextContext
    |   - submit to model worker
    v
token_generator: AsyncGenerator[TokenGeneratorOutput]
    |
    | self._stream(request, token_generator)
    v
SSE payload generator
```

## First-Chunk Gate

```python
async def _start_stream(
    stream: AsyncGenerator[_T, None],
) -> tuple[JSONResponse | None, AsyncGenerator[_T, None]]:
    # WALKTHROUGH COMMENT (not in source):
    # The source docstring explains that the first item is resolved before
    # SSE headers are committed, preserving a real HTTP error response.

    async def empty() -> AsyncGenerator[_T, None]:
        return
        yield  # unreachable; makes this an async generator

    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        return None, empty()

    if isinstance(first, JSONResponse):
        await stream.aclose()
        return first, empty()

    async def chained() -> AsyncGenerator[_T, None]:
        yield first
        async for item in stream:
            yield item

    return None, chained()
```

```text
stream.__anext__()
      |
      +-- StopAsyncIteration --> (None, empty generator)
      |
      +-- JSONResponse -------> close original stream
      |                          return (error, empty generator)
      |
      +-- normal first item ---> chained generator yields:
                                 first, then remaining stream items
```

## Non-Streaming Aggregation

```python
async def complete(
    self, requests: list[TextGenerationRequest]
) -> CreateChatCompletionResponse:
    if len(requests) != 1:
        raise NotImplementedError(
            "chat completions does not support multiple prompts"
        )
    request = requests[0]
    record_request_start()
    request_span = _tracer.start_span(
        "max.request",
        context=request_trace_ctx.get(),
        attributes={
            "gen_ai.request.model": request.model_name,
            "max.request_id": str(request.request_id),
        },
    )
    set_phase_parent(request_span)
    n_reasoning_tokens = 0
    n_tokens = 0
    n_prompt_tokens = 0
    n_cached_prompt_tokens = 0
    request_timer = StopWatch(start_ns=request.timestamp_ns)
    completed_outputs: list[TokenGeneratorOutput] = []

    try:
        completed_outputs = await self.pipeline.all_tokens(request)

        n_reasoning_tokens = sum(
            chunk.reasoning_token_count or 0 for chunk in completed_outputs
        )
        n_tokens = sum(chunk.token_count for chunk in completed_outputs)
        if len(completed_outputs) > 0:
            n_prompt_tokens = completed_outputs[0].prompt_token_count or 0
            if completed_outputs[0].cached_token_count is not None:
                n_cached_prompt_tokens = completed_outputs[
                    0
                ].cached_token_count

        response_message = "".join(
            chunk.decoded_tokens
            for chunk in completed_outputs
            if chunk.decoded_tokens is not None
        )
```

```text
pipeline.all_tokens(request)
          |
          v
completed_outputs: list[TokenGeneratorOutput]
          |
          +-- sum token counts
          +-- read prompt/cache counts
          +-- join decoded_tokens
          +-- process reasoning, logprobs, stop sequence, tools
          |
          v
single CreateChatCompletionResponse returned after generation finishes
```

## HTTP Timing Difference

```text
STREAMING                                         NON-STREAMING
----------------------------------------------    --------------------------------------
submit request                                    submit request
await first event/error                           await every generated output
commit SSE HTTP response                          build one response object
yield chunks over time                            return once at the end
```

## Complete Branch Input And Output Values

```python
token_request_snapshot = {
    "request_id": "req-chat-001",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": None,
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain KV cache in one sentence."},
    ],
    "sampling_params": {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 32,
        "stop": ["\nUser:"],
    },
}

streaming_values = {
    "completion_request.stream": True,
    "first_generator_item": (
        'data: {"id":"chatcmpl-req-chat-001",'
        '"object":"chat.completion.chunk",'
        '"model":"meta-llama/Llama-3.1-8B-Instruct",'
        '"choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"KV cache"},"finish_reason":null}]}\n\n'
    ),
    "_start_stream.error": None,
    "EventSourceResponse.ping": 100000,
    "EventSourceResponse.sep": "\n",
}

remaining_stream_items = [
    'data: {"id":"chatcmpl-req-chat-001","object":"chat.completion.chunk","model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":" stores."},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-req-chat-001","object":"chat.completion.chunk","model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    "data: [DONE]\n\n",
]

non_streaming_values = {
    "completion_request.stream": False,
    "completed_outputs": [
        {
            "decoded_tokens": "KV cache",
            "token_count": 2,
            "prompt_token_count": 28,
            "cached_token_count": 0,
            "status": "ACTIVE",
        },
        {
            "decoded_tokens": " stores.",
            "token_count": 3,
            "prompt_token_count": None,
            "cached_token_count": None,
            "status": "END_OF_SEQUENCE",
        },
    ],
    "response_message": "KV cache stores.",
    "response": {
        "id": "chatcmpl-req-chat-001",
        "object": "chat.completion",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "KV cache stores.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 28,
            "completion_tokens": 5,
            "total_tokens": 33,
        },
    },
}
```
