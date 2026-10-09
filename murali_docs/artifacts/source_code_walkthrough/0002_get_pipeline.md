# `get_pipeline(...)`

Source:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L458)

## One-Line Purpose

`get_pipeline(...)` validates the requested model name and returns the already
created `TokenGeneratorPipeline`.

```python
def get_pipeline(request: Request, model_name: str) -> TokenGeneratorPipeline:
```

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `request` | FastAPI request; carries the raw HTTP body and gives access to `request.app.state.pipeline`. |
| Input | `model_name` | Model name from `completion_request.model`. |
| Output | `TokenGeneratorPipeline` | Active pipeline object used for tokenizer and worker handoff. |

```text
Raw HTTP request body
          |
          v
CreateChatCompletionRequest.model
          +
FastAPI Request.app.state
          |
          v
get_pipeline(...)
          |
          v
TokenGeneratorPipeline
```

Same raw body example from
[`0001_parse_openai_request_body.md`](0001_parse_openai_request_body.md):

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise assistant."
    },
    {
      "role": "user",
      "content": "Explain KV cache in one paragraph."
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 128,
  "stream": true,
  "stop": ["\n\nUser:"],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_doc",
        "description": "Look up an internal document.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "response_format": {
    "type": "text"
  }
}
```

For `get_pipeline(...)`, the important part of this raw body is:

```text
request body["model"]
  -> parsed into completion_request.model
  -> passed as model_name
```

## Worked Example

Assume this server is already running:

```text
request.app.state.pipeline
  |
  +-- model_name = "meta-llama/Llama-3.1-8B-Instruct"
  +-- tokenizer = TextTokenizer(...)
  +-- lora_queue.list_loras() = [
        "finance-summary-lora",
        "sql-assistant-lora",
      ]
```

```text
INPUT: request + model_name             ACTION IN get_pipeline(...)                 OUTPUT
------------------------------------    ----------------------------------------    ------------------------------------
request.app.state                       read active pipeline                        pipeline
  pipeline=<TokenGeneratorPipeline>   ------------------------------->                TokenGeneratorPipeline(...)

pipeline.model_name                     build valid model list                      models
  "meta-llama/Llama-3.1-8B-Instruct" ------------------------------->                [
lora_queue.list_loras()                                                                 "meta-llama/Llama-3.1-8B-Instruct",
  ["finance-summary-lora",                                                             "finance-summary-lora",
   "sql-assistant-lora"]                                                               "sql-assistant-lora",
                                                                                     ]

model_name                              check request model                         decision
  "finance-summary-lora"             ------------------------------->                accepted:
                                                                                     name is in models

pipeline.tokenizer                      check tokenizer protocol                    decision
  TextTokenizer(...)                  ------------------------------->                accepted:
                                                                                     implements PipelineTokenizer

accepted request                        return existing pipeline                    return value
  model_name is valid                 ------------------------------->                request.app.state.pipeline
```

Failure example:

```text
INPUT model_name                        ACTION                                     OUTPUT
------------------------------------    ----------------------------------------    ------------------------------------
"unknown-model"                         check model_name in models                 ValueError(
                                      ------------------------------->              "Unknown model ..."
                                                                                   )
```

## Sequence View

```text
openai_create_chat_completion(...)
  |
  +-- completion_request = _parse_openai_request_body(...)
  |
  +-- pipeline = get_pipeline(request, completion_request.model)
        |
        +-- app_state = request.app.state
        |
        +-- pipeline = app_state.pipeline
        |
        +-- models = [pipeline.model_name]
        |
        +-- if LoRA queue exists:
        |     models += lora_queue.list_loras()
        |
        +-- if model_name is empty:
        |     model_name = pipeline.model_name
        |
        +-- if model_name not in models:
        |     raise ValueError
        |
        +-- if pipeline.tokenizer is not PipelineTokenizer:
        |     raise ValueError
        |
        +-- return pipeline
```

## Decision Tree

```text
Requested model name
  |
  +-- empty?
  |     |
  |     +-- yes -> use pipeline.model_name
  |     |
  |     +-- no  -> keep requested name
  |
  v
Is name in valid model list?
  |
  +-- no  -> ValueError("Unknown model ...")
  |
  +-- yes
        |
        v
Does tokenizer implement PipelineTokenizer?
  |
  +-- no  -> ValueError("Tokenizer ... does not implement ...")
  |
  +-- yes -> return pipeline
```

## Q&A

<details>
<summary><strong>Q1. <code>pipeline.model_name</code>: when is this set?</strong></summary>

It is set when the serving pipeline object is constructed.

```text
server startup / pipeline creation
  |
  +-- create TokenGeneratorPipeline(...)
        |
        +-- BasePipeline.__init__(model_name=...)
              |
              +-- self.model_name = model_name
```

The relevant code is in `BasePipeline.__init__`:

```python
class BasePipeline(...):
    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer,
        model_worker: ModelWorkerProxy,
        lora_queue: LoRAQueue | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora_queue = lora_queue
        self.model_worker = model_worker
```

So by the time an HTTP request arrives, `pipeline.model_name` is already known.
`get_pipeline(...)` only reads it.

```text
HTTP request time:
  request.app.state.pipeline.model_name
    -> already set during server/pipeline startup
```

</details>

<details>
<summary><strong>Q2. What does "tokenizer implements <code>PipelineTokenizer</code>" mean?</strong></summary>

`PipelineTokenizer` is a protocol: a contract that says, "this tokenizer object
has the methods/properties MAX Serve needs." <span style="color:#0b63ce"><strong><em>See the focused note:
<a href="0002_get_pipeline_0a_PipelineTokenizer.md">0002_get_pipeline_0a_PipelineTokenizer.md</a>.</em></strong></span>

At minimum for this route, the tokenizer must behave like:

```text
PipelineTokenizer
  |
  +-- eos_token_ids
  |
  +-- expects_content_wrapping
  |
  +-- new_context(request)
  |
  +-- decode(...)
```

The check in `get_pipeline(...)` is:

```python
if not isinstance(pipeline.tokenizer, PipelineTokenizer):
    raise ValueError(...)
```

Meaning:

```text
pipeline.tokenizer
  |
  +-- has the tokenizer interface MAX expects?
        |
        +-- yes -> route can safely call tokenizer methods later
        |
        +-- no  -> reject request before deeper failures happen
```

Why it matters:

```text
openai_routes.py
  |
  +-- tokenizer.expects_content_wrapping
  |     used while parsing chat messages
  |
  +-- later pipeline path calls tokenizer.new_context(...)
        used to convert TextGenerationRequest into TextContext
```

</details>

## Valid Model List

```text
valid models
  |
  +-- pipeline.model_name
  |
  +-- lora_queue.list_loras(), if LoRA queue exists
```

Example:

```text
pipeline.model_name:
  meta-llama/Llama-3.1-8B-Instruct

lora_queue.list_loras():
  finance-summary-lora
  sql-assistant-lora

accepted request model names:
  meta-llama/Llama-3.1-8B-Instruct
  finance-summary-lora
  sql-assistant-lora
```

## Success Path

```json
{
  "model": "finance-summary-lora",
  "messages": [
    {"role": "user", "content": "Summarize this earnings report."}
  ]
}
```

```text
completion_request.model
  -> "finance-summary-lora"
  -> found in valid models
  -> return request.app.state.pipeline
```

## Failure Path

```json
{
  "model": "unknown-model",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

```text
completion_request.model
  -> "unknown-model"
  -> not found in valid models
  -> raise ValueError
  -> route returns an OpenAI-style error response
```

## Important Detail

`get_pipeline(...)` does **not** load a new model.

```text
It does:
  validate requested model name
  return existing pipeline

It does not:
  download model weights
  create a new pipeline
  switch the server to a different base model
```

## Where The Output Goes Next

```text
pipeline = get_pipeline(...)
  |
  +-- pipeline.tokenizer
  |     |
  |     +-- used by openai_parse_chat_completion_request(...)
  |
  +-- pipeline / response generator
        |
        +-- tokenizer.new_context(...)
        |
        +-- model_worker.stream(...)
        |
        v
      generated token stream
```

Short version:

```text
requested model name
  -> validate against served model + LoRA names
  -> return active TokenGeneratorPipeline
```
