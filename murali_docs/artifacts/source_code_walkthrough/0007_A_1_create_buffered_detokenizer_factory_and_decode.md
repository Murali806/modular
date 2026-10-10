# Complete Code Walk: `create_buffered_detokenizer(...)`

Source:
[`incremental_detokenizer.py`](../../../max/python/max/serve/pipelines/incremental_detokenizer.py#L90)

## Interface

```python
class BufferedDetokenizer(abc.ABC):
    """Abstract base class for detokenizers with UTF-8 buffering.

    All implementations provide an async `decode()` method that handles
    multi-byte UTF-8 sequences that may span multiple tokens.
    """

    @abc.abstractmethod
    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs to text with proper UTF-8 handling.

        Args:
            token_ids: The token IDs to decode.

        Returns:
            The decoded text. May be empty if tokens represent incomplete
            UTF-8 sequences that are buffered for later completion.
        """
        ...
```

## Factory Selection

```python
def _is_fast_tokenizer(tokenizer: object) -> bool:
    """Checks if a tokenizer is a HuggingFace fast tokenizer."""
    return hasattr(tokenizer, "_tokenizer") and isinstance(
        getattr(tokenizer, "_tokenizer", None), Tokenizer
    )


def _get_hf_tokenizer(tokenizer: object) -> PreTrainedTokenizerBase | None:
    """Gets the underlying HuggingFace tokenizer, or None if not available."""
    if hasattr(tokenizer, "delegate"):
        return tokenizer.delegate
    return None


def create_buffered_detokenizer(
    tokenizer: object,
    prompt_token_ids: TokenIDSequence,
    skip_special_tokens: bool = True,
) -> BufferedDetokenizer:
    # WALKTHROUGH COMMENT (not in source):
    # The source docstring documents the selection order and arguments.
    hf_tokenizer = _get_hf_tokenizer(tokenizer)

    # Try DecodeStreamDetokenizer for fast tokenizers
    if hf_tokenizer is not None and _is_fast_tokenizer(hf_tokenizer):
        skipped_special_token_ids: set[int] | None = getattr(
            tokenizer, "skipped_special_token_ids", None
        )
        return DecodeStreamDetokenizer(
            tokenizer=hf_tokenizer,
            prompt_token_ids=prompt_token_ids,
            skip_special_tokens=skip_special_tokens,
            skipped_special_token_ids=skipped_special_token_ids,
        )

    # Fall back to Utf8BufferingDetokenizer for non-fast tokenizers
    if hasattr(tokenizer, "decode"):
        return Utf8BufferingDetokenizer(
            decode_func=tokenizer.decode,
            skip_special_tokens=skip_special_tokens,
        )

    # Last resort: PassthroughDetokenizer (shouldn't normally reach here)
    raise ValueError(
        f"Tokenizer {type(tokenizer).__name__} does not have a decode method."
    )
```

```text
tokenizer object
      |
      | has `.delegate`?
      v
HuggingFace delegate or None
      |
      | delegate has tokenizers.Tokenizer `_tokenizer`?
      v
  +---+---+
  |       |
 yes      no
  |       |
  |       | tokenizer has async decode?
  |       +-- yes --> Utf8BufferingDetokenizer
  |       +-- no  --> ValueError
  |
  +--> DecodeStreamDetokenizer
```

## Fast-Tokenizer Decode Stream

```python
class DecodeStreamDetokenizer(BufferedDetokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_token_ids: TokenIDSequence,
        skip_special_tokens: bool = True,
        skipped_special_token_ids: set[int] | None = None,
    ) -> None:
        self._skip_special_tokens = skip_special_tokens
        self._skipped_special_token_ids = skipped_special_token_ids
        self.skip_special_tokens = skip_special_tokens
        self.skipped_special_token_ids = skipped_special_token_ids

        if not _is_fast_tokenizer(tokenizer):
            raise TypeError(
                "DecodeStreamDetokenizer requires a HuggingFace fast tokenizer "
                "(PreTrainedTokenizerFast). The provided tokenizer does not "
                "have a _tokenizer attribute from the tokenizers library."
            )

        self._tokenizer: Tokenizer = tokenizer._tokenizer
        stream_skip_special = skip_special_tokens and (
            skipped_special_token_ids is None
        )
        self._stream = DecodeStream(
            ids=list(prompt_token_ids),
            skip_special_tokens=stream_skip_special,
        )

    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs using the native DecodeStream."""
        result_parts: list[str] = []
        excluded = self._skipped_special_token_ids

        for token_id in token_ids:
            if (
                self._skip_special_tokens
                and excluded is not None
                and token_id in excluded
            ):
                continue
            text = self._protected_step(token_id)
            if text:
                result_parts.append(text)

        return "".join(result_parts)

    def _protected_step(self, token_id: int) -> str:
        """Performs a single decode step with error recovery."""
        for _ in range(2):
            try:
                result: str | None = self._stream.step(  # type: ignore[attr-defined]
                    self._tokenizer, token_id
                )
                return result or ""
            except Exception as e:
                if _INVALID_PREFIX_ERR_MSG not in str(e):
                    raise
                logger.debug(
                    "Resetting decode stream due to invalid prefix error"
                )
                stream_skip_special = self._skip_special_tokens and (
                    self._skipped_special_token_ids is None
                )
                self._stream = DecodeStream(
                    skip_special_tokens=stream_skip_special
                )
        return ""
```

```text
prompt_token_ids prime DecodeStream
          |
generated token IDs arrive incrementally
          |
          +-- explicitly skipped special token --> no output
          |
          +-- normal token --> DecodeStream.step(...)
                                |
                                +-- complete text --> append
                                +-- incomplete bytes --> ""
                                +-- invalid prefix --> reset once and retry
```

## Generic UTF-8 Buffering

```python
class Utf8BufferingDetokenizer(BufferedDetokenizer):
    def __init__(
        self,
        decode_func: AsyncDecodeFunc,
        skip_special_tokens: bool = True,
    ) -> None:
        self._decode_func = decode_func
        self._skip_special_tokens = skip_special_tokens
        self._buffered_tokens: list[int] = []

    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs with UTF-8 buffering for incomplete sequences."""
        tokens = list(token_ids)

        # Prepend any buffered tokens from previous chunk
        if self._buffered_tokens:
            tokens = self._buffered_tokens + tokens
            self._buffered_tokens = []

        if not tokens:
            return ""

        decoded = await self._decode_func(
            np.array(tokens, dtype=np.int64),
            skip_special_tokens=self._skip_special_tokens,
        )

        if not decoded:
            return ""

        # Count trailing replacement characters
        trailing_replacements = 0
        for char in reversed(decoded):
            if char == _REPLACEMENT_CHAR:
                trailing_replacements += 1
            else:
                break

        if trailing_replacements == 0:
            return decoded

        tokens_to_buffer = min(len(tokens), _MAX_BUFFER_TOKENS)

        for num_buffer in range(1, tokens_to_buffer + 1):
            if num_buffer >= len(tokens):
                self._buffered_tokens = tokens
                return ""

            partial_tokens = tokens[:-num_buffer]
            partial_decoded = await self._decode_func(
                np.array(partial_tokens, dtype=np.int64),
                skip_special_tokens=self._skip_special_tokens,
            )

            if partial_decoded and not partial_decoded.endswith(
                _REPLACEMENT_CHAR
            ):
                self._buffered_tokens = tokens[-num_buffer:]
                return partial_decoded

        self._buffered_tokens = tokens[-tokens_to_buffer:]
        return (
            decoded[:-trailing_replacements]
            if trailing_replacements
            else decoded
        )
```

```text
new token IDs
    |
    | prepend tokens buffered from previous call
    v
decode combined token window
    |
    +-- no trailing U+FFFD --> return decoded text
    |
    +-- trailing U+FFFD
          |
          | progressively remove 1..8 trailing tokens and re-decode
          v
      prefix decodes cleanly?
          |
      +---+---+
      |       |
     yes      no
      |       |
      |       +--> buffer maximum trailing window
      |            return text without trailing replacement chars
      |
      +--> return clean prefix
           buffer removed trailing token IDs for next call
```

## Per-Request Instances

```python
content_detokenizer: BufferedDetokenizer = create_buffered_detokenizer(
    self.tokenizer,
    context.tokens.prompt,
    skip_special_tokens=skip_special_tokens,
)
reasoning_detokenizer: BufferedDetokenizer = create_buffered_detokenizer(
    self.tokenizer,
    context.tokens.prompt,
    skip_special_tokens=skip_special_tokens,
)
```

```text
same prompt token IDs
        |
        +--> content_detokenizer   keeps independent content decode state
        |
        +--> reasoning_detokenizer keeps independent reasoning decode state
```
