# Complete Code Walk: `PipelineTokenizer`

Source:
[`tokenizer.py`](../../../max/python/max/pipelines/modeling/types/tokenizer.py#L186)

<strong><em><a href="0002_A_1_get_pipeline_model_selection.md"><span style="color:#0b63ce">Return to get_pipeline: 0002_A_1_get_pipeline_model_selection.md</span></a></em></strong>.

## Complete Runtime-Checkable Protocol

```python
# WALKTHROUGH COMMENT (not in source):
# `@runtime_checkable` makes `isinstance(tokenizer, PipelineTokenizer)` legal.
# Python checks that the object exposes the protocol's required attributes;
# the class does not need to inherit from PipelineTokenizer explicitly.
@runtime_checkable
class PipelineTokenizer(
    Protocol[UnboundContextType, TokenizerEncoded, RequestType]
):
    """Interface for LLM tokenizers."""

    @property
    def eos_token_ids(self) -> set[int]:
        """The full set of token ids that end generation for this model.

        The tokenizer's declared EOS plus any additional terminators the
        model ends its turn with (for example, chat turn-end tokens from the
        model's generation config).
        """
        ...

    @property
    def expects_content_wrapping(self) -> bool:
        """If ``True``, this tokenizer expects messages to be wrapped as a dict.

        Text messages are formatted as:

        .. code-block:: json

            {
              "role": "user",
              "content": [{ "type": "text", "text": "text content" }]
            }

        instead of:

        .. code-block:: json

            { "role": "user", "content": "text_content" }

        NOTE: Multimodal messages omit the ``content`` property.
        Both :obj:`image_urls` and :obj:`image` content parts are converted to:

        .. code-block:: json

            { "type": "image" }

        Their content is provided as byte arrays through the top-level property
        on the request object, that is, :obj:`RequestType.images`.
        """
        ...

    async def new_context(self, request: RequestType) -> UnboundContextType:
        """Creates a new context from a request object.

        This is sent to the worker process once and then cached locally.

        Args:
            request: Incoming request.

        Returns:
            Initialized context.
        """
        ...

    async def encode(
        self, prompt: str, add_special_tokens: bool
    ) -> TokenizerEncoded:
        """Encodes text prompts as tokens.

        Args:
            prompt: Un-encoded prompt text.
            add_special_tokens: Whether to add special tokens (for example, BOS).

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.
        """
        ...

    async def decode(self, encoded: TokenIds, **kwargs) -> str:
        """Decodes response tokens to text.

        Args:
            encoded: Response token ids, as an array, a sequence, or a single
                id.
            **kwargs: Additional decoder options (for example, ``skip_special_tokens``).

        Returns:
            Un-encoded response text.
        """
        ...
```

```text
PipelineTokenizer contract
        |
        +-- eos_token_ids: set[int]
        |      consumed by EOS/stop tracking
        |
        +-- expects_content_wrapping: bool
        |      consumed while normalizing OpenAI message content
        |
        +-- new_context(request) -> Context
        |      consumed before worker admission
        |
        +-- encode(prompt, add_special_tokens) -> token IDs
        |      consumed for prompts and stop strings
        |
        +-- decode(encoded, **kwargs) -> str
               consumed by buffered detokenizers
```

## Structural Runtime Check

```text
pipeline.tokenizer object
        |
        | isinstance(..., PipelineTokenizer)
        v
required protocol members present?
        |
    +---+---+
    |       |
   no      yes
    |       |
    v       v
ValueError  get_pipeline may return pipeline

This checks object shape at runtime. It does not call encode, decode, or
new_context, and it does not prove their internal behavior is correct.
```

## Complete Protocol Input And Output Values

```python
tokenizer_values = {
    "type": "TextTokenizer",
    "eos_token_ids": {128001, 128009},
    "expects_content_wrapping": False,
    "new_context": "bound async method TextTokenizer.new_context",
    "encode": "bound async method TextTokenizer.encode",
    "decode": "bound async method TextTokenizer.decode",
}

isinstance_result = True

encode_input = {
    "prompt": "Hello",
    "add_special_tokens": True,
}
encode_output_illustrative = [128000, 9906]

decode_input_illustrative = [9906]
decode_kwargs = {"skip_special_tokens": True}
decode_output = "Hello"

new_context_input = {
    "request_id": "req-chat-001",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello",
    "messages": [],
}
new_context_output_snapshot = {
    "request_id": "req-chat-001",
    "tokens.prompt": [128000, 9906],
    "tokens.generated": [],
    "vocab_size": 128256,
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
}
```

Tokenizer IDs above are illustrative and model-specific; each list is the
complete value used by this worked example.
