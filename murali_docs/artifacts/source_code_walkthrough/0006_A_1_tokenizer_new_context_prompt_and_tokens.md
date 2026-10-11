# Complete Code Walk: `tokenizer.new_context(...)` - Prompt And Tokens

Sources:
[`llm.py` - call site](../../../max/python/max/serve/pipelines/llm.py#L379),
[`tokenizer.py` - prompt selection](../../../max/python/max/pipelines/lib/tokenizer.py#L718),
[`tokenizer.py` - context construction](../../../max/python/max/pipelines/lib/tokenizer.py#L819)

<strong><em><a href="0006_A_2_tokenizer_new_context_metadata_and_state.md"><span style="color:#0b63ce">Continue to context state: 0006_A_2_tokenizer_new_context_metadata_and_state.md</span></a></em></strong>.

## Complete Prompt And Token Values

```python
request_prompt = None
request_messages = [
    TextGenerationRequestMessage(
        role="system",
        content="You are a concise assistant.",
    ),
    TextGenerationRequestMessage(
        role="user",
        content="Explain KV cache in one sentence.",
    ),
]
request_tools = None
chat_template_options = {}

rendered_prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a concise assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Explain KV cache in one sentence.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Illustrative Llama tokenizer output for the complete rendered prompt above.
token_ids = [
    128000,
    128006,
    9125,
    128007,
    271,
    2675,
    527,
    264,
    64694,
    18328,
    13,
    128009,
    128006,
    882,
    128007,
    271,
    849,
    21435,
    14736,
    304,
    832,
    11914,
    13,
    128009,
    128006,
    78191,
    128007,
    271,
]

generate_prompt_and_token_ids_output = {
    "prompt": rendered_prompt,
    "token_ids": token_ids,
}
```

The token IDs are illustrative and model-version dependent, but the list is
the complete value used throughout the remaining worked examples.

## Call Site

```python
with record_ms(METRICS.input_time):
    context = await self.tokenizer.new_context(request)
```

```text
request: TextGenerationRequest
        |
        | await tokenizer.new_context(request)
        v
context: TextContext
```

## Prompt Source Selection

```python
async def _generate_prompt_and_token_ids(
    self,
    prompt: Sequence[int] | str | None,
    messages: list[TextGenerationRequestMessage],
    tools: list[TextGenerationRequestTool] | None = None,
    **chat_template_options: Any,
) -> tuple[str | list[int], npt.NDArray[np.integer[Any]]]:
    if isinstance(prompt, str | list):
        return prompt, await self.encode(prompt, add_special_tokens=True)
    elif isinstance(messages, list):
        prompt = self.apply_chat_template(
            messages, tools, **chat_template_options
        )
        return prompt, await self._encode_chat_prompt(prompt)
    else:
        raise ValueError(
            "either prompt must be provided as a list[int] or str, or messages must be provided as a list[TextGenerationRequestMessage]"
        )
```

```text
request.prompt
     |
     +-- str ------------------> encode(prompt, add_special_tokens=True)
     |
     +-- list[int] ------------> encode(prompt, add_special_tokens=True)
     |
     +-- None
           |
           | request.messages is list
           v
     apply_chat_template(messages, tools, **options)
           |
           v
     rendered prompt: str
           |
           | _encode_chat_prompt(prompt)
           v
     token_ids: ndarray
```

## Chat-Prompt Encoder And Fallback

```python
async def _encode_chat_prompt(
    self, prompt: str
) -> npt.NDArray[np.integer[Any]]:
    """Encodes a rendered chat prompt, with the chat encoder if one is set.

    An encoder error sends the prompt to HuggingFace instead.
    """
    encoder = self._chat_encoder
    if encoder is None:
        return await self.encode(prompt, add_special_tokens=False)
    try:
        ids = await run_with_default_executor(_encode_with, encoder, prompt)
    except Exception:
        self._chat_encoder_outcomes["fallback"] += 1
        first = not self._chat_encoder_failed
        self._chat_encoder_failed = True
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "%s failed to encode a chat prompt, so HuggingFace encodes "
            "it%s",
            type(encoder).__name__,
            "; later failures log at DEBUG." if first else ".",
            exc_info=True,
        )
        return await self.encode(prompt, add_special_tokens=False)
    self._chat_encoder_outcomes["custom"] += 1
    if self.max_length and len(ids) > self.max_length:
        raise PromptTooLongError(len(ids), self.max_length)
    return ids
```

```text
rendered prompt
      |
      | custom chat encoder configured?
      v
  +---+---+
  |       |
 no      yes
  |       |
  |       | run custom encoder
  |       +-- success --> length check --> ids
  |       +-- failure --> record fallback --> HuggingFace encode
  |
  +--> HuggingFace encode(add_special_tokens=False)
```

## First Half Of `new_context`

```python
async def new_context(self, request: TextGenerationRequest) -> TextContext:
    """Create a new TextContext object, leveraging necessary information from TextGenerationRequest."""
    # Encode Prompt / Messages
    _prompt, token_ids = await self._generate_prompt_and_token_ids(
        prompt=request.prompt,
        messages=request.messages,
        tools=request.tools,
        **(request.chat_template_options or {}),
    )

    json_schema = (
        json.dumps(request.response_format.json_schema)
        if request.response_format
        and request.response_format.json_schema is not None
        else None
    )

    grammar = (
        request.response_format.grammar if request.response_format else None
    )

    grammar_state = GrammarEnforcementState.from_response_format(
        request.response_format
    )

    # Calculate Max Length
    max_new_tokens = None
    if request.sampling_params.max_new_tokens is not None:
        max_new_tokens = request.sampling_params.max_new_tokens

    max_gen_tokens = max_tokens_to_generate(
        len(token_ids), self.max_length, max_new_tokens
    )

    token_buffer = TokenBuffer(
        array=token_ids.astype(np.int64, copy=False),
    )
```

```text
TextGenerationRequest
        |
        +-- prompt/messages/tools/options --> token_ids
        +-- response_format.json_schema ---> json_schema: str | None
        +-- response_format.grammar -------> grammar: str | None
        +-- response_format ---------------> GrammarEnforcementState
        +-- sampling max_new_tokens -------> max_gen_tokens
        |
        +-- token_ids.astype(np.int64) ----> TokenBuffer
```

<strong><em><a href="0006_A_2_tokenizer_new_context_metadata_and_state.md"><span style="color:#0b63ce">Continue to context state: 0006_A_2_tokenizer_new_context_metadata_and_state.md</span></a></em></strong>.
