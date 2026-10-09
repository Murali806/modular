# `create_buffered_detokenizer(...)`

Sources:

- Call site in
  [`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L377)
- Factory function in
  [`incremental_detokenizer.py`](../../../max/python/max/serve/pipelines/incremental_detokenizer.py#L343)
- `BufferedDetokenizer` interface in
  [`incremental_detokenizer.py`](../../../max/python/max/serve/pipelines/incremental_detokenizer.py#L90)

## One-Line Purpose

`create_buffered_detokenizer(...)` creates a per-request decoder that converts
generated token IDs back into text without breaking multi-byte UTF-8 characters
during streaming.

```python
content_detokenizer = create_buffered_detokenizer(
    self.tokenizer,
    context.tokens.prompt,
    skip_special_tokens=True,
)
```

## Big Picture

```text
TextContext already exists
  |
  +-- context.tokens.prompt
  |     prompt token IDs from tokenizer.new_context(...)
  |
  v
create_buffered_detokenizer(...)
  |
  +-- inspect tokenizer implementation
  +-- choose safest incremental decoder
  +-- remember special-token skipping policy
  |
  v
BufferedDetokenizer
  |
  +-- decode([new generated token IDs])
  |
  v
user-visible text chunk
```

This function does **not** tokenize the prompt and does **not** run the model.
It prepares the object that will later decode generated token IDs into text.

## Why This Exists

Streaming receives tokens in chunks. Some visible characters may be split across
token boundaries. If each token is decoded alone, incomplete byte sequences can
turn into replacement characters.

```text
generated token chunks
  |
  +-- chunk 1 contains first bytes of a character
  +-- chunk 2 contains remaining bytes
  |
  v
bad per-token decode:
  chunk 1 -> "�"
  chunk 2 -> "�"

buffered decode:
  chunk 1 -> ""
  chunk 2 -> "😊"
```

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `tokenizer` | Pipeline tokenizer used to decode generated token IDs. |
| Input | `prompt_token_ids` | Prompt token IDs used to prime fast tokenizer decode-stream state. |
| Input | `skip_special_tokens` | Whether EOS/control tokens should be hidden from user-facing text. |
| Output | `BufferedDetokenizer` | Object with `decode(token_ids)` used while streaming chunks. |

## Worked Example

```text
INPUT                                  ACTION IN FACTORY                       OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
tokenizer                              get tokenizer.delegate                  hf_tokenizer
  TextTokenizer(delegate=...)        ------------------------------->          PreTrainedTokenizerFast(...)

hf_tokenizer                           check for fast tokenizer internals       branch decision
  has _tokenizer from tokenizers     ------------------------------->          use DecodeStreamDetokenizer

prompt_token_ids                       prime DecodeStream                      decode-stream state
  context.tokens.prompt              ------------------------------->          starts after prompt tokens
  [128000, 128006, ...]

skip_special_tokens                    configure output filtering              user-visible text policy
  True                              ------------------------------->           hide EOS/control tokens

factory return value                   caller stores it                        per-request decoder
  DecodeStreamDetokenizer(...)      ------------------------------->           content_detokenizer
```

Then, after the model starts producing tokens:

```text
INPUT                                  ACTION AT STREAM TIME                   OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
generated tokens                       content_detokenizer.decode(...)         text chunk
  [9906, 1917]                       ------------------------------->          "Hello world"

generated special token                special-token filtering                 text chunk
  [128009]                          ------------------------------->           ""

generated partial UTF-8 token          buffered decode                         text chunk
  [token_a]                         ------------------------------->           ""

next generated UTF-8 token             complete buffered character             text chunk
  [token_b]                         ------------------------------->           "😊"
```

Token IDs shown here are illustrative. Exact IDs are model/tokenizer-specific.

## Factory Decision Tree

```text
+------------------------------------------------------------------------------------------------------+
| create_buffered_detokenizer(tokenizer, prompt_token_ids, skip_special_tokens)                         |
|------------------------------------------------------------------------------------------------------|
| STEP 1: Get HuggingFace tokenizer                                                                    |
|                                                                                                      |
|   tokenizer                                                                                          |
|     |                                                                                                |
|     | if tokenizer has .delegate                                                                     |
|     v                                                                                                |
|   hf_tokenizer = tokenizer.delegate                                                                  |
|                                                                                                      |
| STEP 2: Choose implementation                                                                        |
|                                                                                                      |
|   Does hf_tokenizer exist AND does it expose tokenizers.Tokenizer through _tokenizer?                 |
|     |                                                                                                |
|     +-- YES                                                                                          |
|     |     |                                                                                          |
|     |     v                                                                                          |
|     |   DecodeStreamDetokenizer                                                                      |
|     |     - fastest path                                                                             |
|     |     - uses Rust tokenizers DecodeStream                                                        |
|     |     - gets prompt_token_ids so decoding starts after the prompt                                |
|     |                                                                                                |
|     +-- NO                                                                                           |
|           |                                                                                          |
|           v                                                                                          |
|         Does tokenizer have decode(...)?                                                             |
|           |                                                                                          |
|           +-- YES                                                                                    |
|           |     |                                                                                    |
|           |     v                                                                                    |
|           |   Utf8BufferingDetokenizer                                                               |
|           |     - decodes with tokenizer.decode(...)                                                 |
|           |     - buffers trailing broken UTF-8 pieces                                               |
|           |                                                                                          |
|           +-- NO                                                                                     |
|                 |                                                                                    |
|                 v                                                                                    |
|               ValueError                                                                             |
|               tokenizer cannot be used for streaming text decode                                      |
|------------------------------------------------------------------------------------------------------|
| END FACTORY                                                                                          |
+------------------------------------------------------------------------------------------------------+
```

Shorter view:

```text
tokenizer.delegate is HF fast tokenizer?
  |
  +-- yes -> DecodeStreamDetokenizer
  |
  +-- no
        |
        v
      tokenizer has decode(...)?
        |
        +-- yes -> Utf8BufferingDetokenizer
        |
        +-- no  -> ValueError
```

## Implementations

```text
BufferedDetokenizer
  |
  +-- DecodeStreamDetokenizer
  |     |
  |     +-- used for HuggingFace fast tokenizers
  |     +-- uses Rust tokenizers DecodeStream
  |     +-- internally handles incremental UTF-8 state
  |
  +-- Utf8BufferingDetokenizer
  |     |
  |     +-- used when tokenizer has decode(...) but not fast DecodeStream
  |     +-- detects trailing replacement characters
  |     +-- buffers up to _MAX_BUFFER_TOKENS
  |     +-- re-decodes buffered tokens with next chunk
  |
  +-- PassthroughDetokenizer
        |
        +-- direct decode helper class
        +-- create_buffered_detokenizer(...) does not currently select this path
        +-- factory raises ValueError if tokenizer has no decode(...)
```

## Where It Is Used

In `TokenGeneratorPipeline.next_token_chunk(...)`, two detokenizers are created:

```text
context = await tokenizer.new_context(request)
  |
  +-- content_detokenizer
  |     |
  |     +-- decodes normal assistant content tokens
  |
  +-- reasoning_detokenizer
        |
        +-- decodes reasoning tokens separately when reasoning parsing is active
```

Then during streaming:

```text
worker response token IDs
  |
  +-- content token IDs
  |     -> content_detokenizer.decode(...)
  |     -> decoded content text
  |
  +-- reasoning token IDs
        -> reasoning_detokenizer.decode(...)
        -> decoded reasoning text
```

## Sequence View

```text
TokenGeneratorPipeline.next_token_chunk(...)
  |
  v
self.tokenizer.new_context(request)
  |
  v
TextContext
  |
  +-- context.tokens.prompt
  |
  v
create_buffered_detokenizer(self.tokenizer, context.tokens.prompt, True)
  |
  v
content_detokenizer
  |
  v
model worker emits generated token IDs
  |
  v
content_detokenizer.decode(tokens)
  |
  v
decoded text in TokenGeneratorOutput
```

## Q&A

<details>
<summary><strong>Q1. Why does the detokenizer need <code>prompt_token_ids</code>?</strong></summary>

Fast tokenizer streaming decode can depend on prior token context. The prompt
tokens prime the decode stream so generated token pieces are decoded as if they
came after the original prompt.

```text
prompt tokens
  |
  +-- initialize DecodeStream
  |
  v
generated tokens decode with correct context
```

This is especially useful for tokenizers where spacing or word-boundary
behavior depends on the previous tokens.

</details>

<details>
<summary><strong>Q2. What does <code>skip_special_tokens=True</code> do?</strong></summary>

It prevents model control tokens from appearing in the user-visible response.

```text
generated token IDs
  |
  +-- normal text token
  |     -> emitted
  |
  +-- EOS / special token
        -> skipped
```

Example:

```text
tokens:
  [9906, 1917, 128009]

decoded with skip_special_tokens=True:
  "Hello world"

decoded with skip_special_tokens=False:
  "Hello world<|eot_id|>"
```

</details>

<details>
<summary><strong>Q3. What is the replacement character problem?</strong></summary>

`�` is the Unicode replacement character. It appears when bytes are incomplete
or invalid for the current decode operation.

```text
multi-byte character
  |
  +-- token chunk 1 has only part of the bytes
  +-- token chunk 2 has the remaining bytes
```

Without buffering:

```text
chunk 1 -> "�"
chunk 2 -> "�"
```

With buffering:

```text
chunk 1 -> buffer token, emit ""
chunk 2 -> combine buffered + new token, emit correct character
```

</details>

<details>
<summary><strong>Q4. What exactly is a HuggingFace fast tokenizer here?</strong></summary>

In this code path, "fast tokenizer" means the underlying HuggingFace tokenizer
has a `_tokenizer` object from the Rust `tokenizers` library.

```text
Pipeline tokenizer
  |
  +-- delegate
        |
        +-- HuggingFace tokenizer
              |
              +-- _tokenizer is tokenizers.Tokenizer
                    |
                    v
                  fast tokenizer path
```

Why it matters:

```text
fast tokenizer
  -> can use DecodeStream
  -> DecodeStream remembers incremental decode state
  -> safer and faster for streaming
```

</details>

<details>
<summary><strong>Q5. What happens in the non-fast tokenizer path?</strong></summary>

The factory uses `Utf8BufferingDetokenizer` if the tokenizer has a
`decode(...)` method.

```text
new token chunk
  |
  v
decode buffered_tokens + new_tokens
  |
  +-- output ends normally
  |     |
  |     v
  |   return decoded text
  |
  +-- output ends with replacement char
        |
        v
      keep the last few tokens in a buffer
        |
        v
      return only the safe prefix, or "" if nothing is safe yet
```

Example:

```text
chunk 1
  input tokens  -> [token_a]
  decode result -> "�"
  action        -> buffer [token_a]
  output        -> ""

chunk 2
  input tokens  -> [token_b]
  actual decode -> decode [token_a, token_b]
  output        -> "😊"
```

</details>

## Short Version

```text
create_buffered_detokenizer(...)
  -> chooses best incremental decoder for the tokenizer
  -> hides special tokens
  -> avoids broken UTF-8 during streaming
  -> returns BufferedDetokenizer.decode(token_ids)
```
