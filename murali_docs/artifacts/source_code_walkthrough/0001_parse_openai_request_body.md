# `_parse_openai_request_body(...)`

This note explains the first parsing step in the OpenAI-compatible request
path.

Source:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L3070)

<strong><em><a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md"><span style="color:#0b63ce">See complete code walkthrough: 0001_A_1_parse_openai_request_body_http_ingress_and_parser.md</span></a></em></strong>.

## Function Shape

```python
async def _parse_openai_request_body(
    request: Request,
    request_id: str,
    model_cls: type[_TRequest],
) -> _TRequest:
```

## Inputs

`request: Request`

FastAPI's HTTP request object. It provides:

- the raw HTTP body through `await request.body()`
- the FastAPI app through `request.app`
- access to runtime pipeline configuration

`request_id: str`

The request ID assigned earlier by middleware. This function uses it for
logging, especially when unknown request fields are dropped.

`model_cls: type[_TRequest]`

The Pydantic request model class to validate into.

Examples:

- `CreateChatCompletionRequest` for `/v1/chat/completions`
- `CreateCompletionRequest` for `/v1/completions`

For chat completions, the caller passes:

```python
completion_request = await _parse_openai_request_body(
    request,
    request_id,
    CreateChatCompletionRequest,
)
```

So inside the function:

```python
model_cls == CreateChatCompletionRequest
```

## Output

The output is a validated Pydantic request object of type `model_cls`.

For `/v1/chat/completions`, the return value is:

```python
completion_request: CreateChatCompletionRequest
```

Visually:

```text
Raw HTTP body bytes
  -> json.loads(...)
  -> parsed Python dict
  -> model_cls.model_validate(parsed)
  -> typed Pydantic request object
```

For chat completions:

```text
Raw JSON dict
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
        |
        v
CreateChatCompletionRequest(
  model="meta-llama/Llama-3.1-8B-Instruct",
  messages=[
    ChatCompletionMessageParam(
      role="system",
      content="You are a concise assistant.",
    ),
    ChatCompletionMessageParam(
      role="user",
      content="Explain KV cache in one paragraph.",
    ),
  ],
  temperature=0.7,
  top_p=0.9,
  max_tokens=128,
  stream=True,
  stop=["\n\nUser:"],
  tools=[
    ChatCompletionToolParam(
      type="function",
      function=FunctionDefinition(
        name="lookup_doc",
        description="Look up an internal document.",
        parameters={
          "type": "object",
          "properties": {
            "query": {"type": "string"},
          },
          "required": ["query"],
        },
      ),
    ),
  ],
  tool_choice="auto",
  response_format=ResponseFormatText(type="text"),
)
```

The exact nested class names may differ by schema alias, but the important
point is that callers now access validated attributes such as
`completion_request.model`, `completion_request.messages`,
`completion_request.stream`, and `completion_request.tools` instead of reading
untrusted dictionary keys directly.

## Worked Example

```text
INPUT: FastAPI Request / args           ACTION IN _parse_openai_request_body(...)     OUTPUT: CreateChatCompletionRequest
------------------------------------    -----------------------------------------     ------------------------------------
request.body()                          read raw HTTP body bytes                     raw bytes
  b'{"model":"meta-llama/...", ...}'  ------------------------------->                 b'{"model":"meta-llama/...", ...}'

raw bytes                               parse JSON                                  parsed dict
  b'{"model":"meta-llama/...", ...}'  ------------------------------->                 {
                                                                                         "model": "meta-llama/Llama-3.1-8B-Instruct",
                                                                                         "messages": [...],
                                                                                         "stream": true
                                                                                       }

parsed["tools"]                         normalize tool parameters                   normalized tools
  function.parameters may be null     ------------------------------->                 function.parameters becomes {}
                                                                                       when client sent null

request.app.state.pipeline_config       check allow_extra_request_fields             parsed dict
  runtime.allow_extra_request_fields  ------------------------------->                 unknown top-level fields either:
                                                                                         - dropped with warning, or
                                                                                         - left for Pydantic to reject

model_cls                               choose validation schema                    model class
  CreateChatCompletionRequest         ------------------------------->                 validate as chat-completion request

parsed dict + model_cls                 model_cls.model_validate(parsed)             typed request object
  model/messages/stream/tools/...     ------------------------------->                 CreateChatCompletionRequest(
                                                                                         model="meta-llama/Llama-3.1-8B-Instruct",
                                                                                         messages=[...],
                                                                                         stream=True,
                                                                                         tools=[...],
                                                                                       )
```

In this example, the important conversion is:

```text
untrusted JSON dict
  -> schema-validated object
  -> route can safely read completion_request.model,
     completion_request.messages,
     completion_request.stream,
     completion_request.tools
```

## What `model_cls.model_validate(parsed)` Validates

`parsed` is the Python object produced by `json.loads(raw)`.

Example input:

```json
{
  "model": "llama-3",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "stream": true
}
```

`model_cls.model_validate(parsed)` checks that this data matches the expected
OpenAI request schema.

For `CreateChatCompletionRequest`, it validates:

- required fields are present, such as `model` and `messages`
- field types are correct, such as `model` as a string and `stream` as a bool
- nested structures are valid, such as each chat message
- optional fields have acceptable values, such as sampling and tool fields
- unknown top-level fields obey the runtime configuration

Conceptually:

```text
parsed dict
  |
  +-- required fields present?
  |
  +-- field types correct?
  |
  +-- nested objects valid?
  |
  +-- unknown fields allowed or rejected?
  |
  v
CreateChatCompletionRequest instance
```

The function also does two compatibility steps before validation:

1. If a tool schema contains `tools[*].function.parameters: null`, it
   normalizes that to `{}` to match OpenAI API behavior.
2. If `pipeline_config.runtime.allow_extra_request_fields` is enabled, it drops
   unknown top-level fields before validation and logs the dropped names.

## Where The Typed Object Is Used Next

After parsing, `openai_create_chat_completion(...)` uses the typed
`completion_request` object throughout the HTTP route.

It selects the serving pipeline:

```python
pipeline = get_pipeline(request, completion_request.model)
```

It parses chat messages, images, videos, roles, and content:

```python
openai_parse_chat_completion_request(
    completion_request,
    ...
)
```

It reads tool and structured-output controls:

```python
completion_request.tool_choice
completion_request.tools
completion_request.response_format
```

Then the route copies the relevant fields into MAX's internal generation
request:

```python
token_request = TextGenerationRequest(...)
```

So the lifecycle is:

```text
HTTP JSON body
  -> parsed dict
  -> CreateChatCompletionRequest
  -> TextGenerationRequest
  -> tokenizer.new_context(...)
  -> worker queue
  -> scheduler
  -> model execution
```

## Short Version

`_parse_openai_request_body(...)` converts untrusted HTTP JSON into a validated,
typed OpenAI request object.

`model_cls.model_validate(parsed)` is the schema gate. It verifies the JSON
shape, types, and nested fields, then returns a typed object such as
`CreateChatCompletionRequest`.

That typed object is used immediately by the route to select the model
pipeline, parse messages/media/tools, and build the internal
`TextGenerationRequest` that enters token generation.
