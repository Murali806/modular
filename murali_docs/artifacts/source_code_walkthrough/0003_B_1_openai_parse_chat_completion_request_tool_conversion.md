# Complete Code Walk: OpenAI Tool Conversion

Function:
`_convert_chat_completion_tools_to_token_generator_tools(...)`

Sources:
[`openai_routes.py` - caller](../../../max/python/max/serve/router/openai_routes.py#L2269),
[`openai_routes.py` - conversion](../../../max/python/max/serve/router/openai_routes.py#L2558),
[`tool_parsing.py`](../../../max/python/max/pipelines/lib/tool_parsing.py#L45),
[`sampling_config.py`](../../../max/python/max/pipelines/sampling/sampling_config.py#L61),
[`text_generation.py`](../../../max/python/max/pipelines/modeling/types/pipeline_variants/text_generation.py#L59)

<strong><em><a href="0003_A_2_openai_parse_chat_completion_request_media.md"><span style="color:#0b63ce">Previous code walk: 0003_A_2_openai_parse_chat_completion_request_media.md</span></a></em></strong>.

<strong><em><a href="0004_TextGenerationRequest.md"><span style="color:#0b63ce">Next focused note: 0004_TextGenerationRequest.md</span></a></em></strong>.

## One-Line Purpose

The function converts OpenAI-shaped tool definitions into MAX's internal
`TextGenerationRequestTool` dictionaries while validating tool names,
normalizing parameter schemas, and resolving the effective `strict` setting.

It converts tool **definitions**. It does not execute a tool.

```text
OpenAI request tool definitions
              |
              | validate + normalize + resolve strict policy
              v
MAX TextGenerationRequestTool definitions
```

## Where It Runs

The route calls the converter after parsing messages/media and before creating
`TextGenerationRequest`.

```python
# Source: openai_routes.py
# WALKTHROUGH COMMENT (not in source): tool_choice="none" bypasses conversion.
tools = None
if (
    completion_request.tool_choice is None
    or completion_request.tool_choice != "none"
):
    tools = _convert_chat_completion_tools_to_token_generator_tools(
        completion_request.tools,
        valid_tool_name_re,
        policy=pipeline_config.sampling.tool_call_policy,
    )
```

```text
completion_request.tool_choice
              |
        +-----+--------------------------------+
        |                                      |
      "none"                          None / "auto" /
        |                             "required" / named
        |                                      |
        v                                      v
tools = None                         call converter(...)
                                               |
                                               v
                             list[TextGenerationRequestTool]
                                      or None
```

`None` means the OpenAI default behavior, which is effectively automatic tool
selection. Therefore, `tool_choice is None` does not disable the tools list.

## Inputs And Output

| Side | Value | Meaning |
| --- | --- | --- |
| Input | `chat_tools` | OpenAI `ChatCompletionFunctionToolParam` values, or `None`. |
| Input | `valid_tool_name_re` | Model/parser-specific name pattern, falling back to `^[a-zA-Z0-9_-]+$`. |
| Input | `policy` | Server `ToolCallPolicy` used to calculate each tool's effective `strict` flag. |
| Output | `list[TextGenerationRequestTool]` | Internal dictionaries passed to response parsing, grammar handling, the request object, and the tokenizer. |
| Output | `None` | Returned when `chat_tools` is `None` or empty. |

## Complete Conversion Source

```python
def _convert_chat_completion_tools_to_token_generator_tools(
    chat_tools: Iterable[ChatCompletionFunctionToolParam] | None,
    valid_tool_name_re: re.Pattern[str] = _DEFAULT_VALID_TOOL_NAME_RE,
    policy: ToolCallPolicy = ToolCallPolicy.DEFAULT_STRICT_FALSE_AND_BEST_EFFORT,
) -> list[TextGenerationRequestTool] | None:
    """Convert ChatCompletionTool list to TextGenerationRequestTool list."""
    if not chat_tools:
        return None

    token_generator_tools = []
    for tool in chat_tools:
        function = tool["function"]
        name = name_from_tool(tool)
        _validate_tool_function_name(name, valid_tool_name_re)
        token_generator_function = TextGenerationRequestFunction(
            description=function.get("description"),
            name=name,
            parameters=dict(function.get("parameters") or {}),
            strict=policy.resolve_strict(function.get("strict")),
        )
        token_generator_tools.append(
            TextGenerationRequestTool(
                type=tool["type"], function=token_generator_function
            )
        )

    return token_generator_tools
```

## Per-Tool Flow

```text
tool
  |
  +-- tool["function"] ----------------------> function
  |
  +-- function["name"] ----------------------> name_from_tool(...)
  |                                                |
  |                                                v
  |                                      validate name and length
  |
  +-- function.get("description") -----------> description or None
  |
  +-- function.get("parameters") ------------> dict(parameters or {})
  |                                                |
  |                                                +-- missing/None -> {}
  |                                                +-- present -> shallow copy
  |
  +-- function.get("strict") ----------------> policy.resolve_strict(...)
  |                                                |
  |                                                v
  |                                      effective bool strict value
  |
  v
TextGenerationRequestFunction
  |
  | wrap with copied tool["type"]
  v
TextGenerationRequestTool
  |
  | append
  v
token_generator_tools
```

## Internal Output Shape

The internal types are `TypedDict` definitions:

```python
class TextGenerationRequestFunction(TypedDict):
    """Represents a function definition for a text generation request."""

    name: str
    """The name of the function to be invoked."""

    description: str | None
    """A human-readable description of the function's purpose."""

    parameters: dict[str, Any]
    """A dictionary describing the function's parameters, typically following a JSON schema."""

    strict: NotRequired[bool | None]
    """Whether constrained decoding enforces ``parameters`` for this tool.

    ``False`` constrains only the tool-call envelope and leaves the arguments
    free-form; ``True`` enforces the full argument schema. The default value,
    and any potential override, depends on the server's tool-call policy.
    """


class TextGenerationRequestTool(TypedDict):
    """Represents a tool definition for a text generation request."""

    type: str
    """The type of the tool, typically indicating the tool's category or usage."""

    function: TextGenerationRequestFunction
    """The function definition associated with the tool, including its name, description, and parameters."""
```

They provide a typed dictionary contract. The converter does not create a
Pydantic model or a custom runtime class.

```text
TextGenerationRequestTool
  |
  +-- type: "function"
  |
  +-- function
        |
        +-- name
        +-- description
        +-- parameters: JSON Schema dictionary
        +-- strict: effective bool selected by policy
```

## Worked Example

### Complete Input Values

```python
chat_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Colombo.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

valid_tool_name_re = re.compile(r"^[a-zA-Z0-9_-]+$")

policy = ToolCallPolicy.DEFAULT_STRICT_FALSE_AND_BEST_EFFORT
```

### Complete Variable Evolution

```text
INPUT                                   ACTION                                  OUTPUT
------------------------------------    ------------------------------------    ------------------------------------
chat_tools                              check truthiness                         continue conversion
  one tool                            ------------------------------->          chat_tools is not empty

tool                                    tool["function"]                         function
  full tool dictionary                ------------------------------->          {
                                                                                  "name": "get_weather",
                                                                                  "description": "Return the current weather for a city.",
                                                                                  "parameters": {
                                                                                    "type": "object",
                                                                                    "properties": {
                                                                                      "city": {
                                                                                        "type": "string",
                                                                                        "description": "City name, for example Colombo."
                                                                                      },
                                                                                      "unit": {
                                                                                        "type": "string",
                                                                                        "enum": ["celsius", "fahrenheit"]
                                                                                      }
                                                                                    },
                                                                                    "required": ["city"],
                                                                                    "additionalProperties": false
                                                                                  },
                                                                                  "strict": true
                                                                                }

tool                                    name_from_tool(tool)                     name
  function.name="get_weather"        ------------------------------->          "get_weather"

name + regex                            validate name                            valid name
  "get_weather"                      ------------------------------->          no InputError
  ^[a-zA-Z0-9_-]+$

function.parameters                     dict(parameters or {})                  parameters
  complete JSON Schema                ------------------------------->          shallow-copied complete JSON Schema

function.strict + policy                policy.resolve_strict(True)             strict
  True + default-false policy         ------------------------------->          True

normalized fields                       create typed dictionaries                token_generator_tools[0]
                                      ------------------------------->          complete value shown below
```

### Complete Output Value

```python
token_generator_tools = [
    {
        "type": "function",
        "function": {
            "description": "Return the current weather for a city.",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Colombo.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]
```

The values look similar before and after conversion. The important changes are
the ownership boundary, validation, defensive schema copy, and effective
`strict` resolution.

## Name Validation

```python
def name_from_tool(tool: Mapping[str, Any]) -> str:
    """Extracts the function name from an OpenAI-style tool dict."""
    return tool["function"]["name"]
```

```python
def _validate_tool_function_name(
    name: str,
    valid_tool_name_re: re.Pattern[str] = _DEFAULT_VALID_TOOL_NAME_RE,
) -> None:
    """Validate that a tool function name conforms to ``valid_tool_name_re``.

    Raises:
        InputError: If the name is empty, too long, or contains invalid
            characters.
    """
    if not name:
        raise InputError("Invalid tool function name: name cannot be empty.")
    if len(name) > _MAX_TOOL_NAME_LEN:
        raise InputError(
            f"Invalid tool function name: name exceeds the maximum length of "
            f"{_MAX_TOOL_NAME_LEN} characters (was {len(name)})."
        )
    if not valid_tool_name_re.match(name):
        raise InputError(
            f"Invalid tool function name: '{name}'. "
            f"Function names must match {valid_tool_name_re.pattern}."
        )
```

```text
name
  |
  +-- empty? ----------------------------> InputError
  |
  +-- longer than 1024 characters? ------> InputError
  |
  +-- regex does not match? -------------> InputError
  |
  +-- otherwise -------------------------> accepted
```

The parser may supply a model-specific `VALID_TOOL_NAME_RE`; otherwise the
route uses `^[a-zA-Z0-9_-]+$`. An `InputError` is caught by the route and
returned as HTTP 400.

## `strict` Resolution

```python
def resolve_strict(self, requested: bool | None) -> bool:
    """Return the effective per-tool ``strict`` under this policy."""
    if self in (
        ToolCallPolicy.FORCE_STRICT_TRUE_AND_BEST_EFFORT,
        ToolCallPolicy.FORCE_STRICT_TRUE_AND_REJECT_UNSUPPORTED,
    ):
        return True
    if self is ToolCallPolicy.FORCE_STRICT_FALSE:
        return False
    if requested is not None:
        return requested
    return self in (
        ToolCallPolicy.DEFAULT_STRICT_TRUE_AND_BEST_EFFORT,
        ToolCallPolicy.DEFAULT_STRICT_TRUE_AND_REJECT_UNSUPPORTED,
    )
```

```text
requested strict + server policy
              |
              +-- FORCE_STRICT_TRUE_* ----------------> True
              |
              +-- FORCE_STRICT_FALSE -----------------> False
              |
              +-- request supplied True/False --------> requested value
              |
              +-- request omitted strict
                    |
                    +-- DEFAULT_STRICT_TRUE_* --------> True
                    +-- other policies ---------------> False
```

`strict=True` means constrained decoding should enforce the tool argument JSON
Schema. `strict=False` constrains only the tool-call envelope and leaves the
arguments free-form. Whether an unsupported schema is rejected or handled
best-effort is used later during grammar compilation. A
`FORCE_UNCONSTRAINED` server policy disables tool-call grammar generation later,
even if this converter preserves a request's `strict=True` value.

## Where The Output Goes

```text
token_generator_tools
  |
  +-- OpenAIChatResponseGenerator(tools=tools)
  |     uses tool metadata while parsing generated tool calls
  |
  +-- _resolve_grammar_constraints(tools=tools, ...)
  |     selects tools used for constrained-decoding grammar
  |
  +-- TextGenerationRequest(tools=tools)
        |
        v
      tokenizer.new_context(request)
        |
        v
      apply_chat_template(messages, tools, ...)
```

<details>
<summary><strong>Q: What happens when <code>chat_tools</code> is <code>None</code> or empty?</strong></summary>

The function returns `None` immediately. No internal empty list is created.

</details>

<details>
<summary><strong>Q: What happens when <code>tool_choice="none"</code>?</strong></summary>

The caller does not invoke the converter. Its local `tools` variable remains
`None`, which also disables normal tool-call parsing for that request.

</details>

<details>
<summary><strong>Q: Does this function validate the complete JSON Schema?</strong></summary>

No. It converts `parameters` to a fresh dictionary and resolves `strict`.
Schema support and grammar compilation are handled later by the structured
output/tool grammar path.

</details>

<details>
<summary><strong>Q: Does this function call the external tool?</strong></summary>

No. It only prepares tool definitions that the model, grammar system, tokenizer,
and response parser use during generation.

</details>

## Short Version

```text
OpenAI tools
  -> skip entirely when tool_choice="none"
  -> return None when tools are absent/empty
  -> validate each function name
  -> copy description/name/parameters/type
  -> resolve effective strict from request + server policy
  -> produce list[TextGenerationRequestTool]
  -> pass it into grammar, chat-template, and response-parsing paths
```
