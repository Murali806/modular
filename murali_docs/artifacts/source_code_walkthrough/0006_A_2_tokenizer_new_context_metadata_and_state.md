# Complete Code Walk: `tokenizer.new_context(...)` - Context State

Sources:
[`tokenizer.py`](../../../max/python/max/pipelines/lib/tokenizer.py#L458),
[`tokenizer.py`](../../../max/python/max/pipelines/lib/tokenizer.py#L819),
[`context.py`](../../../max/python/max/pipelines/context/context.py#L248),
[`tokens.py`](../../../max/python/max/pipelines/context/tokens.py#L165)

<strong><em><a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md"><span style="color:#0b63ce">Return to prompt/token creation: 0006_A_1_tokenizer_new_context_prompt_and_tokens.md</span></a></em></strong>.

## EOS Tracker Construction

```python
async def build_eos_tracker_for_request(
    eos_token_ids: set[int],
    request: TextGenerationRequest,
    encode_fn: Callable[[str, bool], Awaitable[npt.NDArray[np.integer[Any]]]],
) -> EOSTracker:
    params = request.sampling_params
    resolved_eos_token_ids = set(eos_token_ids)
    eos_sequences: list[list[int]] = []
    if params.ignore_eos:
        resolved_eos_token_ids = set()
    else:
        if params.stop_token_ids:
            resolved_eos_token_ids.update(params.stop_token_ids)
        if params.stop:
            for stop_string in params.stop:
                tokenized = (await encode_fn(stop_string, False)).tolist()
                if tokenized:
                    eos_sequences.append(tokenized)
    return EOSTracker(
        eos_token_ids=resolved_eos_token_ids,
        eos_sequences=eos_sequences,
        eos_stop_strings=params.stop or [],
    )
```

```text
default eos_token_ids
       |
       +-- ignore_eos=True --> empty set
       |
       +-- stop_token_ids --> union with EOS set
       |
       +-- stop strings ----> encode each string -> eos_sequences
       v
EOSTracker(eos_token_ids, eos_sequences, eos_stop_strings)
```

## Grammar State Construction

```python
@classmethod
def from_response_format(
    cls, response_format: TextGenerationResponseFormat | None
) -> GrammarEnforcementState:
    """Creates a state from the given response format, or a default state."""
    if not response_format:
        return cls()
    return cls(
        grammar_enforced=response_format.grammar_enforced,
        tools_forced=response_format.tools_forced,
        requires_structured_output_flag=response_format.requires_structured_output_flag,
        has_json_schema=response_format.has_json_schema,
    )
```

```text
response_format
      |
      +-- None --> GrammarEnforcementState(default fields)
      |
      +-- present
            +-- grammar_enforced
            +-- tools_forced
            +-- requires_structured_output_flag
            +-- has_json_schema
```

## Final `TextContext` Construction

```python
context = TextContext(
    request_id=request.request_id,
    eos_tracker=await self.create_eos_tracker(request),
    max_length=len(token_ids) + max_gen_tokens
    if max_gen_tokens is not None
    else self.max_length,
    tokens=token_buffer,
    vocab_size=self.tokenizer_vocab_size,
    log_probabilities=request.logprobs,
    log_probabilities_echo=request.echo,
    json_schema=json_schema,
    grammar=grammar,
    grammar_state=grammar_state,
    sampling_params=request.sampling_params,
    model_name=request.model_name,
    target_endpoint=request.target_endpoint,
    dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
    cache_salt=request.cache_salt,
)

return context
```

```text
INPUT REQUEST FIELD                     OUTPUT CONTEXT FIELD
------------------------------------    ------------------------------------
request.request_id                   -> context.request_id
token_ids                            -> context.tokens.prompt
request sampling stop controls       -> context.eos_tracker
request.sampling_params              -> context.sampling_params
request.logprobs/echo                -> context.log_probabilities/..._echo
request.response_format              -> json_schema/grammar/grammar_state
tokenizer vocabulary size            -> context.vocab_size
request.model_name                   -> context.model_name
request.target_endpoint              -> context.target_endpoint
request.dkv_cache_hint               -> encoded context.dkv_cache_hint
request.cache_salt                   -> context.cache_salt
```

## Token Buffer State At Creation

```python
token_buffer = TokenBuffer(
    array=token_ids.astype(np.int64, copy=False),
)
```

```text
TokenBuffer backing array

+---------------- prompt ----------------+
| token_ids from prompt/messages/template |
+-----------------------------------------+
0                              prompt_length

generated_length = 0
processed_length = 0
active_length = prompt_length
pending length = 0

The batch constructor may later chunk this active prompt window.
```

## Final Object Boundary

```text
TextGenerationRequest                  TextContext
API/tokenizer-oriented                 scheduler/model-oriented
------------------------------         --------------------------------
prompt/messages/media metadata   --->  token buffer and max length
sampling/stop controls           --->  EOSTracker + SamplingParams
structured-output request        --->  grammar/json_schema/state
routing/cache objects            --->  worker-transferable metadata
```
