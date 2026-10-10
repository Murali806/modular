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

## 4. Variable Flow For A Chat Request

```text
VARIABLE             VALUE BEFORE STEP                         ACTION                              VALUE AFTER STEP
------------------   ---------------------------------------   --------------------------------   ----------------------------------------
request              Request(...)                            await request.body()                 unchanged
raw                  not assigned                            receive body bytes                   b'{"model":"llama",...}'
parsed               not assigned                            json.loads(raw)                      {"model": "llama", ...}
tools                not assigned                            parsed.get("tools")                 list[...] or None
parsed               original JSON dict                     normalize tools                     new dict when tools is a list
pipeline_config      not assigned                            read request.app.state              PipelineConfig(...)
known                not assigned                            set(model_cls.model_fields)          declared top-level field-name set
extras               not assigned                            compare parsed keys with known       unknown top-level field-name list
parsed               may contain unknown fields             filter when runtime flag is True    known fields only
completion_request   not assigned                            model_cls.model_validate(parsed)     CreateChatCompletionRequest(...)
```

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
