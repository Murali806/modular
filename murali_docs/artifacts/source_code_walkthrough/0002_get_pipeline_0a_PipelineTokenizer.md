# `PipelineTokenizer`

Source:
[`tokenizer.py`](../../../max/python/max/pipelines/modeling/types/tokenizer.py#L186)

## One-Line Purpose

`PipelineTokenizer` is the protocol that says: "this tokenizer object has the
methods and properties MAX Serve needs for text generation requests."

```python
class PipelineTokenizer(Protocol):
    eos_token_ids: set[int]
    expects_content_wrapping: bool

    async def new_context(self, request) -> Context:
        ...

    async def encode(self, prompt: str, add_special_tokens: bool):
        ...

    async def decode(self, encoded, **kwargs) -> str:
        ...
```

## Visual Summary

```text
pipeline.tokenizer
  |
  +-- eos_token_ids
  |     token IDs that mean "generation should stop"
  |
  +-- expects_content_wrapping
  |     tells OpenAI route how chat message content should be shaped
  |
  +-- new_context(request)
  |     converts TextGenerationRequest into scheduler/model context
  |
  +-- encode(prompt, add_special_tokens)
  |     converts plain text prompt into token IDs
  |
  +-- decode(token_ids, **kwargs)
        converts generated token IDs back into text
```

## `eos_token_ids`

These are token IDs that terminate generation.

```text
Generated token id: 128009
  |
  v
Is 128009 in tokenizer.eos_token_ids?
  |
  +-- yes -> stop this request
  |
  +-- no  -> continue generation
```

It can include the model's normal EOS token plus chat-specific terminators such
as end-of-turn tokens.

## `expects_content_wrapping`

This tells the OpenAI route how to format chat message content before
tokenization.

If `False`, a simple text message can look like:

```json
{
  "role": "user",
  "content": "Hello"
}
```

If `True`, text is wrapped into content parts:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "Hello"
    }
  ]
}
```

Why this matters:

```text
OpenAI JSON messages
  -> openai_parse_chat_completion_request(...)
  -> uses tokenizer.expects_content_wrapping
  -> produces tokenizer-compatible message format
```

## `new_context(request)`

This is the bridge from API request to scheduler/model state.

Input:

```python
TextGenerationRequest
```

Output:

```python
TextContext
```

Conceptually:

```text
TextGenerationRequest
  |
  +-- messages / prompt
  +-- sampling params
  +-- tools / response format
  +-- images / videos, if any
  |
  v
tokenizer.new_context(request)
  |
  +-- applies chat template
  +-- tokenizes prompt
  +-- builds token state
  +-- attaches request metadata
  |
  v
TextContext
```

That `TextContext` is what gets sent to the worker and scheduler.

## `encode(prompt, add_special_tokens)`

Converts raw prompt text to token IDs.

```text
"Hello"
  -> encode(...)
  -> [9906]
```

`add_special_tokens=True` may add model-specific tokens like BOS/start tokens.

## `decode(token_ids, **kwargs)`

Converts generated token IDs back into text.

```text
[9906, 1917]
  -> decode(...)
  -> "Hello world"
```

The route and pipeline use this during streaming so model token IDs become
response text.

## Short Version

```text
PipelineTokenizer means:
  "This tokenizer can format inputs, create model contexts,
   encode text, decode generated tokens, and identify stop tokens."
```
