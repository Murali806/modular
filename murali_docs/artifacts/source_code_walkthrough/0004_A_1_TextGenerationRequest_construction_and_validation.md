# Complete Code Walk: `TextGenerationRequest`

Sources:
[`openai_routes.py` - construction](../../../max/python/max/serve/router/openai_routes.py#L2482),
[`text_generation.py` - dataclass](../../../max/python/max/pipelines/modeling/types/pipeline_variants/text_generation.py#L336)

## Route Construction

```python
# WALKTHROUGH COMMENT (not in source):
# Pre-tokenized prompt IDs win over messages. This preserves the invariant in
# TextGenerationRequest.__post_init__ that prompt and messages are exclusive.
prompt_token_ids = completion_request.prompt_tokens
chat_template_options = completion_request.resolved_chat_template_kwargs
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
    target_endpoint=_get_target_endpoint(
        request, completion_request.target_endpoint
    ),
    dkv_cache_hint=completion_request.dkv_cache_hint,
    cache_salt=_get_cache_salt(
        request,
        completion_request.cache_salt,
        request.app.state.settings.use_client_cache_salt,
    ),
    chat_template_options=chat_template_options,
)
```

```text
VALIDATED OPENAI VALUES                    TextGenerationRequest FIELD
---------------------------------------    ----------------------------------
request_id                              -> request_id
completion_request.model               -> model_name
completion_request.prompt_tokens       -> prompt
request_messages                       -> messages, unless prompt_tokens exist
request_images/request_decoded_images  -> images/decoded_images
request_videos                         -> videos
tools/response_format                  -> tools/response_format
sampling_params/logprobs_count         -> sampling_params/logprobs
request timer/path                     -> timestamp_ns/request_path
routing/cache values                   -> target_endpoint/dkv_cache_hint/cache_salt
```

## Core Dataclass Fields

```python
@dataclass(frozen=True)
class TextGenerationRequest:
    """An immutable request for text token generation from a pipeline."""

    request_id: RequestID = field()
    model_name: str = field()
    prompt: str | Sequence[int] | None = None
    messages: list[TextGenerationRequestMessage] = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)
    videos: list[bytes] = field(default_factory=list)
    decoded_images: list[PILImage | None] = field(default_factory=list)
    tools: list[TextGenerationRequestTool] | None = None
    response_format: TextGenerationResponseFormat | None = None
    timestamp_ns: int = 0
    request_path: str = "/"
    logprobs: int = 0
    echo: bool = False
    chat_template_options: dict[str, Any] | None = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    target_endpoint: str | None = None
    dkv_cache_hint: dict[str, Any] | None = None
    cache_salt: str | None = None
```

```text
TextGenerationRequest
        |
        +-- identity: request_id, model_name
        +-- prompt source: prompt OR messages
        +-- media: images, decoded_images, videos
        +-- generation controls: sampling_params, logprobs, echo
        +-- structured output: tools, response_format
        +-- observability: timestamp_ns, request_path
        +-- routing/cache: target_endpoint, dkv_cache_hint, cache_salt
```

## Post-Construction Validation

```python
def __post_init__(self) -> None:
    """Validates mutual exclusivity, image-messaging constraints, and message-image consistency after object initialization."""
    # Convert dict messages to TextGenerationRequestMessage objects
    if self.messages is not None:
        converted_messages: list[TextGenerationRequestMessage] = []
        for msg in self.messages:
            if isinstance(msg, dict):
                converted_messages.append(
                    TextGenerationRequestMessage(**msg)
                )
            elif isinstance(msg, TextGenerationRequestMessage):
                converted_messages.append(msg)
            else:
                raise TypeError(f"Invalid message type: {type(msg)}")
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, "messages", converted_messages)

    if self.prompt and self.messages:
        raise ValueError(
            "both prompt and messages cannot be provided to TextGenerationRequest"
        )

    if self.images and isinstance(self.prompt, str):
        raise ValueError(
            "string prompts cannot be provided, when images are provided, use messages"
        )

    if self.videos and isinstance(self.prompt, str):
        raise ValueError(
            "string prompts cannot be provided, when videos are provided, use messages"
        )

    if self.images and self.number_of_images != len(self.images):
        raise ValueError(
            f"number of images provided in TextGenerationRequest do not match messages:\n{self.messages}"
        )

    if self.videos and self.number_of_videos != len(self.videos):
        raise ValueError(
            f"number of videos provided in TextGenerationRequest do not match messages:\n{self.messages}"
        )
```

```text
constructed fields
      |
      | convert dict messages
      v
messages: list[TextGenerationRequestMessage]
      |
      +-- prompt and messages both truthy? -------- yes --> ValueError
      |
      +-- string prompt plus images? -------------- yes --> ValueError
      |
      +-- string prompt plus videos? -------------- yes --> ValueError
      |
      +-- image placeholders != image bytes? ------ yes --> ValueError
      |
      +-- video placeholders != video bytes? ------ yes --> ValueError
      |
      v
valid immutable TextGenerationRequest
```

## Prompt-Tokens Branch

```text
completion_request.prompt_tokens
        |
    +---+---+
    |       |
  present  absent
    |       |
    v       v
prompt=IDs  prompt=None
messages=[] messages=request_messages
    |       |
    +---+---+
        |
        v
exactly one prompt source reaches tokenizer.new_context(request)
```

## Complete Construction Input And Output Values

```python
route_values = {
    "request_id": "req-chat-001",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt_tokens": None,
    "request_messages": [
        {
            "role": "system",
            "content": "You are a concise assistant.",
        },
        {
            "role": "user",
            "content": "Explain KV cache in one sentence.",
        },
    ],
    "request_images": [],
    "request_decoded_images": [],
    "request_videos": [],
    "tools": None,
    "response_format": {"type": "text"},
    "sampling_params": {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 32,
        "stop": ["\nUser:"],
    },
    "logprobs_count": 0,
    "timestamp_ns": 1760000000000000000,
    "request_path": "/v1/chat/completions",
    "target_endpoint": None,
    "dkv_cache_hint": None,
    "cache_salt": None,
    "chat_template_options": None,
}

token_request = TextGenerationRequest(
    request_id=RequestID("req-chat-001"),
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    prompt=None,
    messages=[
        TextGenerationRequestMessage(
            role="system",
            content="You are a concise assistant.",
        ),
        TextGenerationRequestMessage(
            role="user",
            content="Explain KV cache in one sentence.",
        ),
    ],
    images=[],
    videos=[],
    decoded_images=[],
    tools=None,
    response_format=TextGenerationResponseFormat(type="text"),
    timestamp_ns=1760000000000000000,
    request_path="/v1/chat/completions",
    logprobs=0,
    echo=False,
    chat_template_options=None,
    sampling_params=SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=32,
        stop=["\nUser:"],
    ),
    target_endpoint=None,
    dkv_cache_hint=None,
    cache_salt=None,
)

post_init_values = {
    "messages_converted_count": 2,
    "prompt_and_messages_conflict": False,
    "string_prompt_with_images": False,
    "string_prompt_with_videos": False,
    "number_of_images": 0,
    "len(images)": 0,
    "number_of_videos": 0,
    "len(videos)": 0,
    "validation_result": "success",
}
```
