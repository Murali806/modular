# Complete Code Walk: `_parse_openai_request_body(...)` - HTTP Ingress And Parser

Sources:
[`openai_routes.py` - route caller](../../../max/python/max/serve/router/openai_routes.py#L2217),
[`openai_routes.py` - parser](../../../max/python/max/serve/router/openai_routes.py#L3061),
[`openai_routes.py` - error mapping](../../../max/python/max/serve/router/openai_routes.py#L2522)

<strong><em><a href="0001_A_2_parse_openai_request_body_tool_parameter_normalization.md"><span style="color:#0b63ce">Next code file: 0001_A_2_parse_openai_request_body_tool_parameter_normalization.md</span></a></em></strong>.

<strong><em><a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md"><span style="color:#0b63ce">Validation section: 0001_B_1_parse_openai_request_body_pydantic_schema_validation.md</span></a></em></strong>.

## 1. Route Entry And Call

```python
# WALKTHROUGH COMMENT (not in source):
# FastAPI/Starlette has already created `request: Request` and middleware has
# already attached `request.state.request_id` before this route starts.

@router.post("/chat/completions", response_model=None)
async def openai_create_chat_completion(
    request: Request,
) -> CreateChatCompletionResponse | EventSourceResponse | Response:
    request_id = request.state.request_id
    try:
        # WALKTHROUGH COMMENT (not in source):
        # Inputs:
        #   request    -> Starlette/FastAPI Request containing HTTP body bytes
        #   request_id -> ID used when warning about dropped extra fields
        #   model_cls  -> CreateChatCompletionRequest validation schema
        # Output:
        #   completion_request -> validated CreateChatCompletionRequest
        completion_request = await _parse_openai_request_body(
            request, request_id, CreateChatCompletionRequest
        )

        # WALKTHROUGH COMMENT (not in source):
        # Parsing has completed. The next route step reads the validated
        # `.model` attribute instead of reading an untrusted dictionary key.
        pipeline = get_pipeline(request, completion_request.model)
```

```text
OpenAI client              FastAPI route                    parser                         next route step
     |                          |                              |                                  |
     | POST /chat/completions   |                              |                                  |
     | JSON request body        |                              |                                  |
     |------------------------->|                              |                                  |
     |                          | request.state.request_id     |                                  |
     |                          |------------------------------|                                  |
     |                          | await _parse_openai_request_body(                               |
     |                          |   request, request_id, CreateChatCompletionRequest)             |
     |                          |----------------------------->|                                  |
     |                          |                              | read + normalize + validate      |
     |                          | completion_request           |                                  |
     |                          |<-----------------------------|                                  |
     |                          | get_pipeline(request, completion_request.model)                 |
     |                          |---------------------------------------------------------------->|
```

## 2. Runtime Config Lookup And Generic Return Type

```python
def get_app_pipeline_config(app: FastAPI) -> PipelineConfig:
    # WALKTHROUGH COMMENT (not in source):
    # `request.app` is the running FastAPI application. Startup code stored
    # the serving PipelineConfig on `app.state.pipeline_config`.
    pipeline_config = app.state.pipeline_config

    # WALKTHROUGH COMMENT (not in source):
    # This assertion narrows the runtime object to the expected config type.
    assert isinstance(pipeline_config, PipelineConfig)
    return pipeline_config


# WALKTHROUGH COMMENT (not in source):
# `_TRequest` can be any Pydantic BaseModel subtype. The input `model_cls`
# and returned object therefore keep the same concrete request type.
_TRequest = TypeVar("_TRequest", bound="BaseModel")
```

```text
model_cls input                                      parser return type
------------------------------------------------    -----------------------------------------
CreateChatCompletionRequest   ------------------>    CreateChatCompletionRequest
CreateCompletionRequest       ------------------>    CreateCompletionRequest

Type relationship:

model_cls: type[_TRequest]
             |
             | model_cls.model_validate(parsed)
             v
return value: _TRequest
```

## 3. Complete Parser Function

```python
async def _parse_openai_request_body(
    request: Request,
    request_id: str,
    model_cls: type[_TRequest],
) -> _TRequest:
    """Parse a JSON request body into a pydantic request model.

    Pre-normalizes ``tools[*].function.parameters: null`` to ``{}`` to
    match OpenAI's API behavior (null parameters is treated as omitted).
    Without this, Pydantic rejects the request before our handler runs.

    Honors ``pipeline_config.runtime.allow_extra_request_fields``: when set,
    unknown top-level fields are dropped (with a warning) before validation
    instead of failing pydantic's ``extra="forbid"`` check.
    """

    # WALKTHROUGH COMMENT (not in source):
    # `await` pauses this coroutine until Starlette has supplied the complete
    # HTTP request body. Other async work may run while this request waits.
    raw = await request.body()

    # WALKTHROUGH COMMENT (not in source):
    # `json.loads` accepts bytes and returns a Python JSON value. The result
    # may be a dict, list, string, number, bool, or None.
    parsed = json.loads(raw)

    # WALKTHROUGH COMMENT (not in source):
    # OpenAI request schemas expect an object. Non-dict JSON bypasses the
    # dict-only compatibility steps and goes directly to Pydantic so it can
    # produce the schema validation error.
    if not isinstance(parsed, dict):
        return model_cls.model_validate(parsed)

    # Pre-normalize tools.parameters: null -> {} before Pydantic validation.
    tools = parsed.get("tools")
    if isinstance(tools, list):
        # WALKTHROUGH COMMENT (not in source):
        # A new top-level dict is created. The helper also copies each tool
        # and function dict, so the original parsed nested dicts are not
        # modified in place.
        parsed = {**parsed, "tools": _normalize_tools_parameters(tools)}

    # WALKTHROUGH COMMENT (not in source):
    # The runtime flag decides whether unknown top-level fields are removed
    # here or left for Pydantic's `extra="forbid"` validation to reject.
    pipeline_config = get_app_pipeline_config(request.app)
    if pipeline_config.runtime.allow_extra_request_fields:
        known = set(model_cls.model_fields)
        extras = [k for k in parsed if k not in known]
        if extras:
            logger.warning(
                "Request %s contained unknown top-level fields %s; dropping "
                "(allow_extra_request_fields=True).",
                request_id,
                extras,
            )

            # WALKTHROUGH COMMENT (not in source):
            # Rebuild `parsed` with only field names declared by model_cls.
            parsed = {k: v for k, v in parsed.items() if k in known}

    # WALKTHROUGH COMMENT (not in source):
    # Pydantic validates required fields, field types, nested message/tool
    # shapes, configured validators, and the extra-field policy. Success
    # returns the concrete model type passed through `model_cls`.
    return model_cls.model_validate(parsed)
```

```text
request: Request
    |
    | await request.body()
    v
raw: bytes
    |
    | json.loads(raw)
    v
parsed: JSON value
    |
    +-- not a dict ---------------------------------------------------+
    |                                                                |
    |                                                                v
    |                                              model_cls.model_validate(parsed)
    |
    +-- dict
         |
         | parsed.get("tools")
         v
       tools: object | None
         |
         +-- list --> _normalize_tools_parameters(tools)
         |               |
         |               v
         |             parsed: new dict with normalized tools
         |
         +-- not list --> parsed remains unchanged
                         |
                         | get_app_pipeline_config(request.app)
                         v
              allow_extra_request_fields?
                         |
               +---------+---------+
               |                   |
              False               True
               |                   |
               |                   | known = set(model_cls.model_fields)
               |                   | extras = unknown parsed keys
               |                   | drop extras when present
               |                   |
               +---------+---------+
                         |
                         v
              model_cls.model_validate(parsed)
                         |
                         v
                 validated _TRequest
```

## 4. Complete Input And Output Variable Values For A Chat Request

```python
# Complete logical snapshot of the Request fields used by this function.
request_values = {
    "method": "POST",
    "url.path": "/v1/chat/completions",
    "headers.content-type": "application/json",
    "state.request_id": "req-chat-001",
    "app.state.pipeline_config.runtime.allow_extra_request_fields": True,
}

raw = b'{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"system","content":"You are a concise assistant."},{"role":"user","content":"Explain KV cache in one sentence."}],"temperature":0.2,"top_p":0.9,"max_tokens":32,"stream":true,"stop":["\\nUser:"],"tools":[{"type":"function","function":{"name":"lookup_doc","description":"Look up a document.","parameters":null}}],"tool_choice":"auto","response_format":{"type":"text"},"client_trace":"trace-77"}'

parsed_after_json_loads = {
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
                "parameters": None,
            },
        }
    ],
    "tool_choice": "auto",
    "response_format": {"type": "text"},
    "client_trace": "trace-77",
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_doc",
            "description": "Look up a document.",
            "parameters": None,
        },
    }
]

parsed_after_tool_normalization = {
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
    "client_trace": "trace-77",
}

# Complete membership result for every top-level key in this example.
known_membership_for_parsed_keys = {
    "model": True,
    "messages": True,
    "temperature": True,
    "top_p": True,
    "max_tokens": True,
    "stream": True,
    "stop": True,
    "tools": True,
    "tool_choice": True,
    "response_format": True,
    "client_trace": False,
}
extras = ["client_trace"]

parsed_after_extra_field_filter = {
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

# Complete non-default/non-None view of the validated object for this input.
completion_request_model_dump = {
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

```text
request_values
    |
    | await request.body()
    v
raw (full bytes value above)
    |
    | json.loads(raw)
    v
parsed_after_json_loads
    |
    | _normalize_tools_parameters(tools)
    v
parsed_after_tool_normalization
    |
    | extras == ["client_trace"]
    v
parsed_after_extra_field_filter
    |
    | CreateChatCompletionRequest.model_validate(...)
    v
completion_request_model_dump
```

<details>
<summary><strong>Q: What does <code>parsed: JSON value</code> look like for a dict versus a non-dict?</strong></summary>

`json.loads(raw)` accepts any valid JSON value. The first JSON character often
makes the resulting Python type clear:

```text
RAW JSON STARTS WITH        PYTHON VALUE RETURNED BY json.loads(raw)
------------------------    -----------------------------------------
{ ... }                     dict
[ ... ]                     list
"..."                       str
42                          int
2.5                         float
true / false                bool
null                        None
```

### Case 1: `parsed` Is A Dict

```python
raw_dict = b'{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello"}]}'

parsed_dict = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Hello",
        }
    ],
}

parsed_dict_is_dict = isinstance(parsed_dict, dict)  # True
```

```text
parsed_dict
    |
    | isinstance(parsed_dict, dict) == True
    v
read and normalize parsed_dict["tools"], when present
    |
    | optionally remove unknown top-level keys
    v
CreateChatCompletionRequest.model_validate(parsed_dict)
    |
    v
validated CreateChatCompletionRequest
```

For this input, a concise non-default/non-`None` view of the validated output
is:

```python
validated_dict_output = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Hello",
        }
    ],
}
```

Being a `dict` only means the value has the required top-level JSON shape. Its
required fields and nested values must still pass Pydantic validation.

### Case 2: `parsed` Is Not A Dict

This example is valid JSON, but its top-level value is an array rather than an
object:

```python
raw_list = b'[{"role":"user","content":"Hello"}]'

parsed_list = [
    {
        "role": "user",
        "content": "Hello",
    }
]

parsed_list_is_dict = isinstance(parsed_list, dict)  # False
```

```text
parsed_list
    |
    | isinstance(parsed_list, dict) == False
    v
skip tools normalization
    |
    | skip unknown-field filtering
    v
CreateChatCompletionRequest.model_validate(parsed_list)
    |
    v
ValidationError
    reason: CreateChatCompletionRequest expects a top-level object/dict,
            but received a list
```

The same direct-validation path applies to every other non-dict JSON value:

```python
non_dict_examples = [
    {
        "raw": b'"hello"',
        "parsed": "hello",
        "python_type": "str",
        "validation_result": "ValidationError",
    },
    {
        "raw": b'42',
        "parsed": 42,
        "python_type": "int",
        "validation_result": "ValidationError",
    },
    {
        "raw": b'true',
        "parsed": True,
        "python_type": "bool",
        "validation_result": "ValidationError",
    },
    {
        "raw": b'null',
        "parsed": None,
        "python_type": "NoneType",
        "validation_result": "ValidationError",
    },
]
```

These values are valid JSON, so `json.loads(...)` succeeds. They fail later
because this request schema requires a JSON object. The surrounding route
converts that Pydantic `ValidationError` into the HTTP 400 response explained
in the next section.

</details>

<details>
<summary><strong>Q: What happens when <code>tools</code> is a list versus a non-list value?</strong></summary>

<strong><em><a href="0001_A_2_parse_openai_request_body_tool_parameter_normalization.md"><span style="color:#0b63ce">See the complete answer: 0001_A_2_parse_openai_request_body_tool_parameter_normalization.md</span></a></em></strong>.

</details>

## 5. Parser Errors Become HTTP 400 Responses

```python
    # WALKTHROUGH COMMENT (not in source):
    # These handlers belong to the route's surrounding `try` block. A JSON
    # decoding failure or Pydantic validation failure from the parser reaches
    # the matching handler before token generation starts.
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except KeyError as e:
        logger.exception("KeyError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValidationError as e:
        logger.warning(
            "Request validation error in request %s: %s", request_id, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TypeError as e:
        logger.exception("TypeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
```

```text
json.loads(raw)              _normalize_tools_parameters(tools)          model_cls.model_validate(parsed)
      |                                      |                                           |
      | malformed JSON                       | invalid dict conversion                   | schema mismatch
      v                                      v                                           v
JSONDecodeError                          TypeError                                ValidationError
      |                                      |                                           |
      v                                      v                                           v
HTTP 400: "Missing JSON."                HTTP 400: "Invalid JSON."                HTTP 400: detail=str(e)
```

<strong><em><a href="0001_A_2_parse_openai_request_body_tool_parameter_normalization.md"><span style="color:#0b63ce">Continue to tool normalization: 0001_A_2_parse_openai_request_body_tool_parameter_normalization.md</span></a></em></strong>.
