# `self.tokenizer.new_context(request)`

Sources:

- Call site in
  [`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L379)
- `PipelineTokenizer` protocol in
  [`tokenizer.py`](../../../max/python/max/pipelines/modeling/types/tokenizer.py#L232)
- Text tokenizer implementation in
  [`tokenizer.py`](../../../max/python/max/pipelines/lib/tokenizer.py#L819)
- `TextContext` definition in
  [`context.py`](../../../max/python/max/pipelines/context/context.py#L456)

<strong><em><a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md"><span style="color:#0b63ce">See complete code walkthrough: 0006_A_1_tokenizer_new_context_prompt_and_tokens.md</span></a></em></strong>.

## One-Line Purpose

`self.tokenizer.new_context(request)` converts a `TextGenerationRequest` into a
runtime `TextContext` that the worker and scheduler can execute.

```python
context = await self.tokenizer.new_context(request)
```

## Big Picture

```text
TextGenerationRequest
  |
  +-- normalized messages or prompt token IDs
  +-- sampling params
  +-- tools / response_format
  +-- images / videos, if any
  +-- routing/cache metadata
  |
  v
tokenizer.new_context(request)
  |
  +-- apply chat template, if needed
  +-- tokenize prompt/messages
  +-- build EOS tracker
  +-- build token buffer
  +-- attach sampling / grammar / routing metadata
  |
  v
TextContext
```

## Worked Example

Assume the OpenAI route already built this internal request:

```text
INPUT: TextGenerationRequest             ACTION IN new_context(...)              OUTPUT: TextContext
-------------------------------------    -----------------------------------     -------------------------------------
request_id                               copy request identity                   request_id
  RequestID("req-123")                ------------------------------->             RequestID("req-123")

model_name                               copy requested model name               model_name
  "meta-llama/Llama-3.1-8B-Instruct"  ------------------------------->             "meta-llama/Llama-3.1-8B-Instruct"

+------------------------------------------------------------------------------------------------------+
| PROMPT SOURCE IF / ELSE                                                                              |
|------------------------------------------------------------------------------------------------------|
| INPUT: request field                  ACTION IN new_context(...)              OUTPUT                  |
|------------------------------------------------------------------------------------------------------|
| IF prompt exists                                                                                     |
|                                                                                                      |
| prompt                                choose prompt source                    prompt source           |
|   "Explain KV cache directly."      ------ string prompt exists ------>       use prompt directly     |
|                                                                                                      |
| prompt string                         tokenize prompt                         produced token_ids       |
|   "Explain KV cache directly."      ------------------------------->          [128000, 849, ...]      |
|                                                                                                      |
| OR prompt is already token IDs                                                                       |
|                                                                                                      |
| prompt                                choose prompt source                    produced token_ids       |
|   [128000, 128006, 882, ...]        ------ token IDs exist ---------->         [128000, 128006, ...]   |
|                                                                                                      |
| ELSE prompt is None                                                                                  |
|                                                                                                      |
| prompt                                choose prompt source                    prompt source           |
|   None                              ------ prompt is None ----------->         use request.messages    |
|                                                                                                      |
| messages                              apply chat template                     rendered prompt string   |
|   [                                ---------------------------------->        <|begin_of_text|>       |
|     system: "You are a concise                                                    <|start_header_id|>system... |
|       assistant."                                                                  You are a concise assistant. |
|     user: "Explain KV cache                                                        <|start_header_id|>user...   |
|       in one paragraph."                                                           Explain KV cache...           |
|   ]                                                                                <|start_header_id|>assistant... |
|                                                                                                      |
| rendered prompt string                 tokenize prompt/messages              produced token_ids        |
|   <|begin_of_text|>...              ---------------------------------->      [128000, 128006, ...]    |
|------------------------------------------------------------------------------------------------------|
| END IF / ELSE: produced token_ids are now available for the shared TokenBuffer step below.            |
+------------------------------------------------------------------------------------------------------+

                                                                                  produced token_ids
                                                                                    [128000, 128006, 9125, ...]
                                                                                         |
                                                                                         | create TokenBuffer
                                                                                         |
                                                                                         v
                                                                                  context.tokens = TokenBuffer(
                                                                                    prompt=[128000, 128006, ...],
                                                                                    generated=[]
                                                                                  )

sampling_params                          attach sampling controls                sampling_params
  temperature=0.7                     ---------------------------------->          temperature=0.7
  top_p=0.9                                                                          top_p=0.9
  max_new_tokens=128                                                                  max_new_tokens=128

sampling_params.stop                     build EOS tracker                       eos_tracker
  stop=["\n\nUser:"]                  ---------------------------------->          EOS ids + stop-string tracker
tokenizer default EOS ids
  {128001, 128009}

response_format                          attach structured-output state          json_schema / grammar / grammar_state
  ResponseFormatText(type="text")     ---------------------------------->          json_schema=None
                                                                                    grammar=None
                                                                                    grammar_state=default

tools                                    apply during chat template              prompt text / grammar inputs
  None                                ---------------------------------->          no tool instructions added

images / videos                          multimodal preprocessing, if needed     context media state
  images=[]                           ---------------------------------->          no image state
  videos=[]                                                                       no video state

routing/cache metadata                   attach routing/cache metadata           routing/cache metadata
  target_endpoint=None                ---------------------------------->          target_endpoint=None
  dkv_cache_hint=None                                                               dkv_cache_hint=None
  cache_salt=None                                                                  cache_salt=None

prompt length + max_new_tokens           compute max allowed length              max_length
  len(token_ids) + 128                ---------------------------------->          prompt_length + 128,
                                                                                    capped by model max length
```

The rendered prompt string and token IDs are model-specific. The examples above
show the shape of the transformation, not universal token values.

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `request` | `TextGenerationRequest` built by the OpenAI route. |
| Input owner | `self.tokenizer` | Model-specific tokenizer implementing `PipelineTokenizer`. |
| Output | `TextContext` | Scheduler/model context with token IDs and generation state. |

## Sequence View

For the common text tokenizer path:

```text
new_context(request)
  |
  +-- _generate_prompt_and_token_ids(...)
  |     |
  |     +-- if request.prompt exists:
  |     |     use prompt / prompt token IDs
  |     |
  |     +-- else:
  |           apply chat template to request.messages + request.tools
  |
  +-- derive json_schema / grammar from request.response_format
  |
  +-- create GrammarEnforcementState
  |
  +-- compute max generated tokens
  |
  +-- create TokenBuffer(token_ids)
  |
  +-- create EOSTracker from EOS + stop params
  |
  +-- create TextContext(...)
  |
  v
TextContext
```

## Prompt Or Messages

`TextGenerationRequest` has either `prompt` or `messages`.

```text
TextGenerationRequest
  |
  +-- prompt
  |     direct model input:
  |       - raw prompt string, or
  |       - already-tokenized prompt IDs
  |
  +-- messages
        structured chat conversation:
          - system turn
          - user turn
          - assistant turn
          - tool turn
```

```text
request.prompt present?
  |
  +-- yes
  |     |
  |     +-- use prompt directly
  |     +-- may already be token IDs from prompt_tokens
  |
  +-- no
        |
        +-- use request.messages
        +-- apply model chat template
        +-- tokenize resulting prompt
```

This is why the earlier route code made `prompt` and `messages` mutually
exclusive.

### `prompt`

`prompt` is already the thing the model should consume.

String prompt example:

```python
prompt = "Explain KV cache in one paragraph."
```

Tokenized prompt example:

```python
prompt = [128000, 128006, 882, 128007, ...]
```

Visual:

```text
prompt
  |
  +-- if string:
  |     tokenize string
  |
  +-- if token IDs:
        use token IDs directly
```

This path is useful for legacy completions, benchmarking, manual prompt
construction, or orchestrator flows that already computed prompt tokens.

### `messages`

`messages` are structured chat turns. They are not yet the final model prompt.

Example:

```python
messages = [
    TextGenerationRequestMessage(
        role="system",
        content="You are a concise assistant.",
    ),
    TextGenerationRequestMessage(
        role="user",
        content="Explain KV cache in one paragraph.",
    ),
]
```

Visual:

```text
messages
  |
  +-- role/content structure
  |
  v
model-specific chat template
  |
  v
prompt string
  |
  v
token IDs
```

Example rendered prompt, model-specific:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a concise assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Explain KV cache in one paragraph.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### Main Difference

```text
prompt:
  already a model prompt or token IDs

messages:
  conversation turns that must be rendered with a chat template first
```

They are mutually exclusive because the tokenizer needs one authoritative input
source.

## What Goes Into `TextContext`

```text
TextContext
  |
  +-- request_id
  |     same request identity used throughout serving
  |
  +-- tokens
  |     TokenBuffer wrapping prompt token IDs
  |
  +-- max_length
  |     prompt length + allowed generated tokens
  |
  +-- eos_tracker
  |     EOS token IDs and stop-string/stop-token behavior
  |
  +-- vocab_size
  |     tokenizer vocabulary size
  |
  +-- log_probabilities
  |     whether/how many logprobs to return
  |
  +-- json_schema / grammar / grammar_state
  |     structured-output and constrained-decoding state
  |
  +-- sampling_params
  |     temperature, top_p, max_new_tokens, etc.
  |
  +-- model_name
  |     requested base model or LoRA name
  |
  +-- target_endpoint
  |     optional routing target
  |
  +-- dkv_cache_hint
  |     encoded distributed KV cache hint
  |
  +-- cache_salt
        prefix-cache isolation salt
```

### Q&A

<details>
<summary><strong>Q1. What is <code>vocab_size</code>? Is it related to the vocab dictionary?</strong></summary>

Yes, it is related to the tokenizer vocabulary, but it is only the **count** of
valid token IDs, not the full token dictionary.

```text
tokenizer vocabulary dictionary
  |
  +-- token string -> token id
  |
  +-- examples:
        "Hello"       -> 9906
        " world"      -> 1917
        "<|eot_id|>"  -> 128009

vocab_size
  |
  +-- number of valid token IDs
  |
  +-- example:
        128256 means valid IDs are [0, 128256)
```

Why `TextContext` needs only the size:

```text
Generated token_id
  |
  +-- token_id < 0?
  |     -> invalid
  |
  +-- token_id >= vocab_size?
  |     -> invalid / out of vocabulary
  |
  +-- otherwise
        -> valid token ID
```

It does **not** need the full dictionary here because this context is not doing
token lookup by string. It only needs to validate generated token IDs before
returning output.

```text
Need full vocab dictionary?
  |
  +-- encode text -> token IDs
  |     yes, tokenizer handles this
  |
  +-- decode token IDs -> text
  |     yes, tokenizer handles this
  |
  +-- validate generated ID is in range
        no, only vocab_size is needed
```

So:

```text
vocab dictionary:
  used by tokenizer encode/decode

vocab_size:
  used by TextContext as a cheap bounds check
```

</details>

<details>
<summary><strong>Q2. What does <code>log_probabilities</code> mean here?</strong></summary>

For each generated token, the model produces logits. Sampling turns those logits
into probabilities and chooses the next token. A log probability is the
logarithm of a token's probability.

```text
model logits
  |
  v
probabilities over vocabulary
  |
  +-- "Hello" token probability = 0.70
  +-- "Hi" token probability    = 0.20
  +-- "Hey" token probability   = 0.05
  |
  v
log probabilities
  |
  +-- log(0.70)
  +-- log(0.20)
  +-- log(0.05)
```

In `TextContext`, `log_probabilities` is a request setting that says how much
logprob data to keep and return.

```text
request.logprobs = 0
  |
  +-- do not return log probabilities

request.logprobs = 5
  |
  +-- for each generated token, return:
        sampled token logprob
        top 5 token logprobs
```

Runtime flow:

```text
sampled token
  |
  +-- optional LogProbabilities object
  |
  v
TextContext.advance_token_buffer(new_token, log_probabilities)
  |
  +-- stores logprob data by token position
  |
  v
TextContext.create_generation_output(...)
  |
  +-- attaches logprob data to TextGenerationOutput
```

Why this is useful:

```text
logprobs tell the client:
  "How confident was the model about this generated token?"
```

Example:

```text
generated token: "world"
token_log_probability: -0.12
top_log_probabilities:
  "world" -> -0.12
  "there" -> -2.40
  "!"     -> -3.10
```

More negative means lower probability.

</details>

<details>
<summary><strong>Q3. What does "derive <code>json_schema</code> / <code>grammar</code> and create <code>GrammarEnforcementState</code>" mean?</strong></summary>

This part reads `request.response_format` and converts it into context fields
used later for structured output / constrained decoding.

```text
request.response_format
  |
  +-- json_schema?
  |     -> context.json_schema
  |
  +-- grammar?
  |     -> context.grammar
  |
  +-- enforcement flags?
        -> context.grammar_state
```

### Case 1: Plain Text Response

```text
INPUT: request.response_format          ACTION IN new_context(...)              OUTPUT: TextContext
------------------------------------    -----------------------------------     ------------------------------------
ResponseFormatText(type="text")         no schema / grammar to enforce          context.json_schema = None
                                      ------------------------------->           context.grammar = None
                                                                                 context.grammar_state = default
```

Meaning:

```text
model can generate normal free-form text
```

### Case 2: JSON Schema Response

Example user intent:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "type": "object",
        "properties": {
          "answer": {"type": "string"}
        },
        "required": ["answer"]
      }
    }
  }
}
```

Inside `new_context(...)`:

```text
INPUT: request.response_format          ACTION IN new_context(...)              OUTPUT: TextContext
------------------------------------    -----------------------------------     ------------------------------------
json_schema dict                        json.dumps(json_schema)                 context.json_schema
  {"type":"object", ...}              ------------------------------->           '{"type":"object", ...}'

response_format flags                   GrammarEnforcementState.from...         context.grammar_state
  has_json_schema=True                ------------------------------->           has_json_schema=True
  grammar_enforced=True/False                                                   grammar_enforced=True/False
```

Meaning:

```text
The request carries a JSON schema.
Later, the scheduler/sampler can constrain generated tokens so the output
matches the requested JSON structure.
```

### Case 3: Grammar Response

Some model/tool integrations use a grammar instead of a JSON schema.

```text
INPUT: request.response_format          ACTION IN new_context(...)              OUTPUT: TextContext
------------------------------------    -----------------------------------     ------------------------------------
grammar string                          copy grammar                            context.grammar
  "root ::= ..."                      ------------------------------->           "root ::= ..."

grammar enforcement flags               GrammarEnforcementState.from...         context.grammar_state
  tools_forced=True/False             ------------------------------->           tools_forced=True/False
  grammar_enforced=True/False                                                   grammar_enforced=True/False
```

Meaning:

```text
The request carries a grammar.
Later, constrained decoding uses this grammar to mask invalid next tokens.
```

Concrete example:

```text
Grammar rule:
  output must be either "yes" or "no"

Model vocabulary, simplified:
  token_id 10 -> "yes"
  token_id 11 -> "no"
  token_id 12 -> "maybe"
  token_id 13 -> "hello"
```

At the first generation step, the model may assign logits to many tokens:

```text
raw model logits
  |
  +-- "yes"   ->  3.1
  +-- "no"    ->  2.8
  +-- "maybe" ->  4.5
  +-- "hello" ->  3.7
```

But the grammar says only `"yes"` or `"no"` is valid. The constrained decoding
mask blocks the invalid choices before sampling:

```text
grammar-valid next tokens
  |
  +-- "yes" -> allowed
  +-- "no"  -> allowed

grammar-invalid next tokens
  |
  +-- "maybe" -> masked out
  +-- "hello" -> masked out
```

After masking:

```text
masked logits
  |
  +-- "yes"   ->  3.1
  +-- "no"    ->  2.8
  +-- "maybe" -> -inf
  +-- "hello" -> -inf
```

Then sampling happens only over the allowed tokens:

```text
masked logits
  -> sampler
  -> chosen token is "yes" or "no"
```

For a structured JSON example:

```text
Required output shape:
  {"answer": "<string>"}

Generated so far:
  {

Valid next tokens might be:
  "\""      starts a JSON object key

Invalid next tokens might be:
  "hello"   not valid immediately after {
  "]"       closes an array that was never opened
```

So "mask invalid next tokens" means:

```text
model still computes probabilities for the vocabulary
  |
  v
grammar removes tokens that would break the required format
  |
  v
sampler chooses only from format-valid tokens
```

### Why `GrammarEnforcementState` Exists

The schema/grammar is the rulebook. `GrammarEnforcementState` is the live state
machine that tracks whether the rulebook should be active right now.

```text
grammar / json_schema
  |
  +-- static constraint definition
  |
  v
GrammarEnforcementState
  |
  +-- should grammar be enforced from the first token?
  +-- is tool calling forced?
  +-- is this a JSON-schema request?
  +-- are we inside a thinking region where enforcement is suspended?
  +-- did a tool-call start/end token toggle enforcement?
```

Example:

```text
tool_choice="required"
  |
  +-- grammar_enforced=True from first generated token

tool_choice="auto"
  |
  +-- grammar exists
  +-- grammar_enforced=False initially
  +-- flips on when tool-call start token appears

response_format=json_schema
  |
  +-- has_json_schema=True
  +-- structured-output flag may be required
  +-- enforcement can apply from the start
```

Short version:

```text
json_schema / grammar:
  what shape output must follow

GrammarEnforcementState:
  when and how strongly to enforce that shape during generation
```

</details>

## EOS And Stop Handling

The tokenizer builds an `EOSTracker`.

```text
request.sampling_params
  |
  +-- ignore_eos?
  +-- stop_token_ids?
  +-- stop strings?
  |
  v
create_eos_tracker(request)
  |
  v
context.eos_tracker
```

Later during generation:

```text
new token arrives
  |
  +-- matches EOS token?
  |
  +-- completes stop string?
  |
  v
finish or continue request
```

## Structured Output

`new_context(...)` copies structured-output state from the request into the
context.

```text
request.response_format
  |
  +-- json_schema
  |     -> context.json_schema
  |
  +-- grammar
  |     -> context.grammar
  |
  +-- enforcement options
        -> context.grammar_state
```

The scheduler/batch constructor later uses this state to build grammar matchers
and sampling bitmasks.

## Multimodal Note

Some model-specific tokenizers return `TextAndVisionContext` instead of plain
`TextContext`.

```text
TextGenerationRequest
  |
  +-- messages
  +-- images / decoded_images
  |
  v
TextAndVisionTokenizer.new_context(...)
  |
  +-- apply chat template
  +-- preprocess images
  +-- tokenize text
  |
  v
TextAndVisionContext
```

The high-level idea is the same: convert API-level request data into scheduler
context state.

## Where It Goes Next

In `TokenGeneratorPipeline.next_token_chunk(...)`:

```text
context = await self.tokenizer.new_context(request)
  |
  +-- create buffered detokenizers from context.tokens.prompt
  |
  +-- maybe configure reasoning parser state
  |
  +-- self.model_worker.stream(context.request_id, context)
  |
  v
worker IPC / scheduler
```

Short version:

```text
TextGenerationRequest
  -> tokenizer-specific prompt construction + tokenization
  -> TextContext
  -> model worker queue
```
