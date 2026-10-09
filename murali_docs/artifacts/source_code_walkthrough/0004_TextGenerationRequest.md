# `TextGenerationRequest(...)`

Sources:

- Construction in
  [`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L2482)
- Dataclass definition in
  [`text_generation.py`](../../../max/python/max/pipelines/modeling/types/pipeline_variants/text_generation.py#L336)

## One-Line Purpose

`TextGenerationRequest(...)` is MAX's internal request object for text
generation. It converts the already-validated OpenAI request into the shape the
tokenizer, worker, scheduler, and model pipeline understand.

```python
token_request = TextGenerationRequest(...)
```

## Big Picture

```text
CreateChatCompletionRequest
  |
  +-- OpenAI-facing fields
  |     model, messages, stream, tools, response_format, sampling params
  |
  v
openai_parse_chat_completion_request(...)
  |
  +-- request_messages
  +-- request_images
  +-- request_videos
  +-- request_decoded_images
  |
  v
TextGenerationRequest(...)
  |
  +-- internal MAX request object
  |
  v
pipeline.next_token_chunk(token_request)
```

## Worked Example

```text
INPUT: route data                       ACTION IN TextGenerationRequest(...)       OUTPUT: TextGenerationRequest field
------------------------------------    ----------------------------------------    ------------------------------------
request_id                              wrap as RequestID                         request_id
  "req-123"                          ------------------------------->               RequestID("req-123")

completion_request.model                copy model name                            model_name
  "meta-llama/Llama-3.1-8B-Instruct" ------------------------------->               "meta-llama/Llama-3.1-8B-Instruct"

completion_request.prompt_tokens        choose prompt/messages branch              prompt + messages
  None                                ------------------------------->               prompt=None
request_messages                                                                       messages=[TextGenerationRequestMessage(...)]
  [system msg, user msg]

request_images                          attach resolved media                      images
  []                                  ------------------------------->               []
request_decoded_images                                                                 decoded_images=[]
  []
request_videos                                                                         videos=[]
  []

tools                                   attach normalized tools                    tools
  None                                ------------------------------->               None

response_format                         attach output format                       response_format
  ResponseFormatText(type="text")     ------------------------------->               ResponseFormatText(type="text")

sampling_params                         attach sampling behavior                   sampling_params
  temperature=0.7                     ------------------------------->               SamplingParams(
  top_p=0.9                                                                           temperature=0.7,
  max_new_tokens=128                                                                  top_p=0.9,
  stop=["\n\nUser:"]                                                                  max_new_tokens=128,
                                                                                      stop=["\n\nUser:"],
                                                                                    )

completion_request.logprobs             convert logprobs request                   logprobs
  False                               ------------------------------->               0

request metadata                        attach observability/routing data          metadata fields
  request_timer.start_ns              ------------------------------->               timestamp_ns=...
  request.url.path                                                                     request_path="/v1/chat/completions"
  target_endpoint                                                                      target_endpoint=None
  cache_salt                                                                           cache_salt=None
```

Result:

```text
TextGenerationRequest
  |
  +-- request_id=RequestID("req-123")
  +-- model_name="meta-llama/Llama-3.1-8B-Instruct"
  +-- prompt=None
  +-- messages=[system msg, user msg]
  +-- sampling_params=SamplingParams(...)
  +-- response_format=ResponseFormatText(type="text")
  +-- images=[]
  +-- videos=[]
```

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `completion_request` | Typed OpenAI request from `_parse_openai_request_body(...)`. |
| Input | `request_messages` | Normalized `TextGenerationRequestMessage` list. |
| Input | `request_images` | Resolved image bytes. |
| Input | `request_videos` | Resolved video bytes. |
| Input | `request_decoded_images` | Decoded `PIL.Image` objects, or `None` where decode was skipped. |
| Input | `sampling_params` | MAX sampling configuration built from OpenAI fields. |
| Input | `tools` | Normalized tool definitions, if provided. |
| Input | `response_format` | Structured-output or text response format. |
| Output | `TextGenerationRequest` | Frozen dataclass passed into the token-generation pipeline. |

## Field Mapping In The Route

The route builds:

```python
token_request = TextGenerationRequest(
    request_id=RequestID(request_id),
    model_name=completion_request.model,
    prompt=prompt_token_ids or None,
    messages=[] if prompt_token_ids else request_messages,
    images=request_images,
    decoded_images=request_decoded_images,
    videos=request_videos,
    tools=tools,
    timestamp_ns=request.state.request_timer.start_ns,
    request_path=request.url.path,
    response_format=response_format,
    sampling_params=sampling_params,
    logprobs=logprobs_count,
    target_endpoint=_get_target_endpoint(...),
    dkv_cache_hint=completion_request.dkv_cache_hint,
    cache_salt=_get_cache_salt(...),
    chat_template_options=chat_template_options,
)
```

Visual:

```text
HTTP route state
  |
  +-- request_id -----------------------> request_id
  +-- request.state.request_timer ------> timestamp_ns
  +-- request.url.path -----------------> request_path
  |
CreateChatCompletionRequest
  |
  +-- model ----------------------------> model_name
  +-- prompt_tokens --------------------> prompt
  +-- resolved_chat_template_kwargs ----> chat_template_options
  +-- dkv_cache_hint -------------------> dkv_cache_hint
  +-- cache_salt -----------------------> cache_salt
  |
Parsed chat request
  |
  +-- messages -------------------------> messages
  +-- images ---------------------------> images
  +-- videos ---------------------------> videos
  +-- decoded_images -------------------> decoded_images
  |
Derived route values
  |
  +-- tools ----------------------------> tools
  +-- response_format ------------------> response_format
  +-- sampling_params ------------------> sampling_params
  +-- logprobs_count -------------------> logprobs
  +-- target_endpoint ------------------> target_endpoint
```

## Prompt Tokens Versus Messages

There is one important branch:

```python
prompt_token_ids = completion_request.prompt_tokens

prompt = prompt_token_ids or None
messages = [] if prompt_token_ids else request_messages
```

Meaning:

```text
prompt_tokens present?
  |
  +-- yes
  |     |
  |     +-- prompt = token IDs from request
  |     +-- messages = []
  |     +-- tokenizer skips normal chat-message tokenization
  |
  +-- no
        |
        +-- prompt = None
        +-- messages = normalized chat messages
        +-- tokenizer applies chat template and tokenizes messages
```

This keeps `prompt` and `messages` mutually exclusive.

## Object Shape

`TextGenerationRequest` is a frozen dataclass.

Key fields:

```text
TextGenerationRequest
  |
  +-- request_id
  +-- model_name
  +-- prompt
  +-- messages
  +-- images
  +-- videos
  +-- decoded_images
  +-- tools
  +-- response_format
  +-- sampling_params
  +-- timestamp_ns
  +-- request_path
  +-- logprobs
  +-- target_endpoint
  +-- dkv_cache_hint
  +-- cache_salt
  +-- chat_template_options
```

Example:

```text
TextGenerationRequest(
  request_id=RequestID("req-123"),
  model_name="meta-llama/Llama-3.1-8B-Instruct",
  prompt=None,
  messages=[
    TextGenerationRequestMessage(
      role="system",
      content="You are a concise assistant.",
    ),
    TextGenerationRequestMessage(
      role="user",
      content="Explain KV cache in one paragraph.",
    ),
  ],
  images=[],
  videos=[],
  decoded_images=[],
  tools=None,
  response_format=ResponseFormatText(type="text"),
  sampling_params=SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=128,
    stop=["\n\nUser:"],
  ),
  logprobs=0,
  request_path="/v1/chat/completions",
)
```

## Built-In Validation

`TextGenerationRequest.__post_init__` performs internal consistency checks.

```text
TextGenerationRequest(...)
  |
  +-- convert dict messages to TextGenerationRequestMessage
  |
  +-- prompt and messages both present?
  |     |
  |     +-- yes -> ValueError
  |
  +-- string prompt with images?
  |     |
  |     +-- yes -> ValueError
  |
  +-- string prompt with videos?
  |     |
  |     +-- yes -> ValueError
  |
  +-- number of image placeholders matches image byte list?
  |     |
  |     +-- no -> ValueError
  |
  +-- number of video placeholders matches video byte list?
        |
        +-- no -> ValueError
```

## Why This Object Exists

The OpenAI request is API-shaped. `TextGenerationRequest` is generation-shaped.

```text
OpenAI shape:
  user-facing API fields
  flexible JSON forms
  OpenAI-compatible names

MAX generation shape:
  request_id
  normalized messages
  resolved media bytes
  sampling params
  routing/cache hints
  tokenizer-ready structure
```

## Where It Goes Next

Streaming path:

```text
TextGenerationRequest
  |
  v
response_generator.stream(token_request)
  |
  v
pipeline.next_token_chunk(token_request)
  |
  v
tokenizer.new_context(token_request)
  |
  v
model_worker.stream(context.request_id, context)
```

Non-streaming path:

```text
TextGenerationRequest
  |
  v
response_generator.complete([token_request])
  |
  v
pipeline.next_token_chunk(token_request)
  |
  v
collect final response
```

Short version:

```text
OpenAI request fields + parsed media/messages
  -> TextGenerationRequest
  -> tokenizer.new_context(...)
  -> TextContext
  -> worker/scheduler/model
```
