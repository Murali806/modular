# Complete Code Walk: `CreateChatCompletionRequest` Schema Validation

Source:
[`openai.py`](../../../max/python/max/serve/schemas/openai.py#L236)

<strong><em><a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md"><span style="color:#0b63ce">Return to parser: 0001_A_1_parse_openai_request_body_http_ingress_and_parser.md</span></a></em></strong>.

<details>
<summary><strong>Q: Why is <code>model_validate(...)</code> not defined inside <code>CreateChatCompletionRequest</code>?</strong></summary>

`model_validate(...)` is inherited from Pydantic's `BaseModel`. It does not
need to be redefined in the `CreateChatCompletionRequest` class body.

```text
pydantic.BaseModel
|
+-- model_validate(...)
      ^
      |
_MaxRequestExtensions(BaseModel)

_ChatCompletionParamsBase
  created by create_model(...)
  also extends BaseModel
      ^
      |
CreateChatCompletionRequest
```

Both parent classes are Pydantic models:

- `_MaxRequestExtensions` explicitly extends `BaseModel`.
- `_ChatCompletionParamsBase` is built by `_model_from_typeddict(...)` using
  `create_model(...)`, which returns a dynamically created `BaseModel`
  subclass.

Therefore Python can resolve this inherited class method:

```python
validated_request = CreateChatCompletionRequest.model_validate(parsed)
```

```text
parsed: dict
    |
    | inherited BaseModel.model_validate(parsed)
    v
validate fields inherited from _ChatCompletionParamsBase
    |
    +-- validate MAX fields inherited from _MaxRequestExtensions
    +-- validate model and messages declared on CreateChatCompletionRequest
    +-- run before/after model validators
    |
    +-- invalid input --> ValidationError
    |
    +-- valid input ----> CreateChatCompletionRequest instance
```

Python searches the class inheritance hierarchy when a method is not found
directly on the class. The implementation being called here is
`pydantic.BaseModel.model_validate` with
`cls=CreateChatCompletionRequest`.

</details>

## 1. Strict Top-Level Field Policy

```python
# WALKTHROUGH COMMENT (not in source):
# Every request model built with this config rejects undeclared top-level
# fields unless `_parse_openai_request_body` removes them first under the
# `allow_extra_request_fields` runtime compatibility flag.

_FORBID_EXTRA = ConfigDict(
    extra="forbid",
    # OpenAI's TypedDict params reference further TypedDicts (e.g. for
    # message content parts and tool definitions); pydantic accepts them
    # as ``arbitrary_types_allowed``.
    arbitrary_types_allowed=True,
)
```

```text
parsed top-level key
        |
        | key in model_cls.model_fields?
        v
   +----+----+
   |         |
  yes        no
   |         |
   |         +-- allow_extra_request_fields=True --> parser drops key
   |         |
   |         +-- allow_extra_request_fields=False -> Pydantic extra="forbid"
   |                                                  -> ValidationError
   v
field validation
```

## 2. Nested Chat Message Shape

```python
# TypedDicts for tool-call objects inside chat messages. These match
# OpenAI's spec: ``function.name`` and ``function.arguments`` are both
# Required[str].
class _ToolCallFunction(TypedDict):
    name: str
    arguments: str


class _ToolCallParam(TypedDict):
    function: _ToolCallFunction
    id: NotRequired[str]
    type: NotRequired[str]


# Multi-modal content part; image_url/video_url are dicts to accept non-string
# vendor hints (e.g. max_long_side_pixel int, video fps float).
class _ContentPart(TypedDict):
    type: str
    text: NotRequired[str]
    image_url: NotRequired[dict[str, Any]]
    video_url: NotRequired[dict[str, Any]]


# MAX chat message schema. Vendor extensions like ``reasoning_content``
# are first-class fields so pydantic type-checks them at request
# validation time.
class ChatCompletionMessageParam(TypedDict):
    # ``root`` is a vendor role; parsed for all models but gated at the route to
    # those that declare it (``extra_chat_roles``), others get a 400.
    role: Literal[
        "developer", "system", "user", "assistant", "tool", "function", "root"
    ]
    content: NotRequired[str | list[_ContentPart] | None]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[_ToolCallParam]]
    function_call: NotRequired[_ToolCallFunction]
    refusal: NotRequired[str | None]
    audio: NotRequired[dict[str, str] | None]

    # MAX vendor extensions.
    reasoning_content: NotRequired[str | None]


ChatCompletionMessageParam.__pydantic_config__ = ConfigDict(extra="allow")  # type: ignore[attr-defined]
```

```text
messages: list[ChatCompletionMessageParam]
    |
    +-- message[0]
    |     +-- role: allowed Literal value
    |     +-- content: str | list[_ContentPart] | None
    |     +-- optional tool_calls
    |           +-- function.name: str
    |           +-- function.arguments: str
    |
    +-- message[1]
          +-- same nested validation
```

## 3. OpenAI TypedDict To Pydantic Model

```python
def _model_from_typeddict(name: str, td: type) -> type[BaseModel]:
    """Builds a pydantic ``BaseModel`` mirroring an OpenAI ``TypedDict``.

    All fields default to ``None`` because OpenAI marks only a few fields
    (e.g. ``model``, ``messages``, ``input``) as ``Required[...]``; we
    re-declare the truly required ones in the subclass below with no
    default.

    Field annotations are widened to ``Optional[T]`` so that clients which
    explicitly serialize unset fields as ``null`` (e.g. ``"tool_choice":
    null``) are accepted as equivalent to omission. OpenAI expresses this
    via ``NotRequired`` on the TypedDict, but the underlying type aliases
    (``ChatCompletionToolChoiceOptionParam`` and friends) are not
    ``Optional`` themselves.
    """
    fields: dict[str, Any] = {}
    # ``get_type_hints`` (without ``include_extras=True``) already strips
    # Required/NotRequired qualifiers, which is what we want here since the
    # top-level pydantic field is declared with a ``None`` default
    # regardless.
    for field_name, annotation in get_type_hints(td).items():
        # WALKTHROUGH COMMENT (not in source):
        # Convert each OpenAI SDK field T into Pydantic field `T | None`
        # with default None.
        fields[field_name] = (annotation | None, None)

    # WALKTHROUGH COMMENT (not in source):
    # Build a real Pydantic model class at module import time.
    return create_model(name, __config__=_FORBID_EXTRA, **fields)
```

```text
_OpenAIChatCompletionParams: TypedDict class
        |
        | get_type_hints(td)
        v
field name -> annotation
        |
        | fields[field_name] = (annotation | None, None)
        v
Pydantic field definitions
        |
        | create_model(..., __config__=_FORBID_EXTRA)
        v
_ChatCompletionParamsBase: type[BaseModel]
```

## 4. Generated Bases And MAX Extensions

```python
class _MaxRequestExtensions(BaseModel):
    """MAX-specific request fields shared by chat and text completions.

    These are NOT part of the OpenAI spec; OpenAI clients won't send them
    and OpenAI servers won't accept them. Other open-source inference
    servers (vLLM, sglang, ...) ship similar extensions.
    """

    model_config = _FORBID_EXTRA

    # Sampling parameters beyond the OpenAI spec.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    thinking_temperature: float | None = None

    # ``False`` on chat completions (MiniMax M3 only) folds reasoning into
    # ``content`` wrapped in ``<think>...</think>``; on completions it skips the
    # reasoning parser, so ``text`` and ``logprobs`` cover every generated
    # token. ``True`` (default) keeps reasoning out of ``content``/``text``.
    reasoning_split: bool = True

    # Generation control.
    min_tokens: int | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    # Returns the prompt and generated token ids on each response choice.
    return_token_ids: bool = False

    # Routing / cache hints used by disaggregated serving.
    target_endpoint: str | None = None
    dkv_cache_hint: dict[str, Any] | None = None
    # Per-request prefix-cache isolation for multi-tenant deployments.
    cache_salt: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "Per-request salt that isolates this prompt's prefix-cache "
            "entries from other requests. Combined with "
            "kv_cache_hash_seed via XOR. Works under any "
            "kv_cache_hash_algo: a cryptographic guarantee under "
            "sha256/sha256_64, best-effort under ahash64."
        ),
    )

    # OpenRouter reasoning object; mapped to enable_thinking in the route.
    reasoning: ReasoningConfig | None = None


# ---- Auto-generated request bases from OpenAI's TypedDict params ----------
#
# These pull in every OpenAI request field automatically; bumping the
# ``openai`` SDK version is enough to pick up new ones. Required fields
# from the underlying TypedDicts (``model``, ``messages``, ``input``) are
# re-declared on the subclasses below so they have no default.

_ChatCompletionParamsBase = _model_from_typeddict(
    "_ChatCompletionParamsBase", _OpenAIChatCompletionParams
)
_TextCompletionParamsBase = _model_from_typeddict(
    "_TextCompletionParamsBase", _OpenAITextCompletionParams
)
_EmbeddingParamsBase = _model_from_typeddict(
    "_EmbeddingParamsBase", _OpenAIEmbeddingParams
)
_SpeechParamsBase = _model_from_typeddict(
    "_SpeechParamsBase", _OpenAISpeechParams
)
```

```text
CreateChatCompletionRequest
        |
        +-- _ChatCompletionParamsBase
        |      OpenAI SDK fields such as temperature, top_p, tools, stop, ...
        |
        +-- _MaxRequestExtensions
               MAX fields such as top_k, min_p, cache_salt, target_endpoint, ...
```

## 5. Concrete Chat Request Schema And Validators

```python
class CreateChatCompletionRequest(
    _MaxRequestExtensions,
    _ChatCompletionParamsBase,  # type: ignore[misc,valid-type]
):
    """OpenAI chat completion request, extended with MAX fields.

    Inherits every OpenAI request field from
    ``CompletionCreateParamsBase``; adds MAX-only sampling/routing fields
    via ``_MaxRequestExtensions``.
    """

    # Required fields - re-declare so they have no default. Each message
    # is validated against :class:`ChatCompletionMessageParam`, our
    # explicit cross-section of the OpenAI message shapes plus MAX
    # vendor extensions (``reasoning_content``). Pydantic emits plain
    # dicts so the route reads fields via dict access.
    model: str
    messages: list[ChatCompletionMessageParam] = Field(min_length=1)

    max_tokens: int | None = None
    max_completion_tokens: int | None = None

    # Re-typed from the SDK union so the ``json_schema`` arm advertises a
    # boolean ``schema`` (a valid JSON Schema). Pydantic emits a plain dict;
    # the route reads it via dict access.
    response_format: ResponseFormat | None = None

    # ``stream`` lives on the OpenAI streaming/non-streaming subclasses, not
    # on ``CompletionCreateParamsBase`` - declare it explicitly here.
    stream: bool | None = False

    # Re-typed from the SDK literal to accept ``max``.
    reasoning_effort: ReasoningEffortLevel | None = None

    # MAX-only chat-template extension.
    chat_template_kwargs: dict[str, Any] | None = None

    # Pre-tokenized prompt injected by the orchestrator for KV cache-aware
    # routing. When set, the route uses these tokens directly instead of
    # tokenizing ``messages`` (the orchestrator must use the same
    # HuggingFace tokenizer as MAX so the IDs match the model vocabulary).
    # If both are provided, ``prompt_tokens`` takes precedence.
    prompt_tokens: list[int] | None = None

    @model_validator(mode="before")
    @classmethod
    def _translate_thinking_to_standard(cls, data: Any) -> Any:
        # A vendor ``thinking`` control ({"type": enabled|disabled|adaptive})
        # is translated to the standard ``enable_thinking``/``thinking``
        # chat-template booleans. ``adaptive`` leaves them unset: templates
        # default to adaptive when no reasoning flag is given, so the two
        # render identically. Client-set ``chat_template_kwargs`` win.
        if not isinstance(data, dict):
            return data
        thinking = data.get("thinking")
        if thinking is None:
            return data
        if not isinstance(thinking, dict) or set(thinking) - {"type"}:
            raise ValueError("`thinking` must be an object with a `type` field")
        mode = thinking.get("type")
        if mode not in ("enabled", "disabled", "adaptive"):
            raise ValueError(
                "`thinking.type` must be one of 'enabled', 'disabled', "
                f"'adaptive'; got {mode!r}"
            )
        data = dict(data)
        data.pop("thinking")
        if mode != "adaptive":
            enabled = mode == "enabled"
            kwargs = dict(data.get("chat_template_kwargs") or {})
            kwargs.setdefault("enable_thinking", enabled)
            kwargs.setdefault("thinking", enabled)
            data["chat_template_kwargs"] = kwargs
        return data

    @model_validator(mode="after")
    def _reconcile_max_completion_tokens(self) -> CreateChatCompletionRequest:
        # Accept both token-limit fields; ``max_completion_tokens`` wins.
        if (
            self.max_completion_tokens is not None
            and self.max_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            self.max_tokens = self.max_completion_tokens
        return self
```

```text
parsed: dict
    |
    | model_cls.model_validate(parsed)
    v
@model_validator(mode="before")
_translate_thinking_to_standard(data)
    |
    | optionally converts `thinking` into `chat_template_kwargs`
    v
Pydantic field validation
    |
    +-- model: required str
    +-- messages: required list, min_length=1
    |      +-- each item follows ChatCompletionMessageParam
    +-- stream: bool | None, default False
    +-- max_tokens: int | None
    +-- max_completion_tokens: int | None
    +-- inherited OpenAI and MAX fields
    +-- unknown top-level field -> forbidden
    |
    v
@model_validator(mode="after")
_reconcile_max_completion_tokens(self)
    |
    | if both token limits differ, max_completion_tokens wins
    v
CreateChatCompletionRequest instance
```

## 6. End-To-End Validation Handoff

```text
_parse_openai_request_body
        |
        | return model_cls.model_validate(parsed)
        v
CreateChatCompletionRequest.model_validate(parsed)
        |
        +-- success --> completion_request: CreateChatCompletionRequest
        |                    |
        |                    +-- completion_request.model
        |                    +-- completion_request.messages
        |                    +-- completion_request.stream
        |                    +-- completion_request.tools
        |
        +-- failure --> ValidationError
                             |
                             v
                       route returns HTTP 400
```

## 7. Complete Validation Input And Output Values

```python
parsed = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain KV cache in one sentence."},
    ],
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 32,
    "stream": True,
    "stop": ["\nUser:"],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup_doc",
                "description": "Look up a document.",
                "parameters": {},
            },
        }
    ],
    "tool_choice": "auto",
    "response_format": {"type": "text"},
}

before_validator_output = parsed

field_validation_results = {
    "model": "valid str",
    "messages": "valid list with 2 valid ChatCompletionMessageParam values",
    "temperature": "valid float 0.2",
    "top_p": "valid float 0.9",
    "max_tokens": "valid int 32",
    "stream": "valid bool True",
    "stop": "valid list[str] containing one string",
    "tools": "valid one-item tool list with parameters dict",
    "tool_choice": "valid literal 'auto'",
    "response_format": "valid {'type': 'text'} value",
    "unknown_top_level_fields": [],
}

after_validator_values = {
    "max_tokens": 32,
    "max_completion_tokens": None,
}

validated_model_dump_exclude_none_and_defaults = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain KV cache in one sentence."},
    ],
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 32,
    "stream": True,
    "stop": ["\nUser:"],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup_doc",
                "description": "Look up a document.",
                "parameters": {},
            },
        }
    ],
    "tool_choice": "auto",
    "response_format": {"type": "text"},
}
```

<strong><em><a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md"><span style="color:#0b63ce">Return to the entry code walk: 0001_A_1_parse_openai_request_body_http_ingress_and_parser.md</span></a></em></strong>.
