# Complete Code Walk: `_normalize_tools_parameters(...)`

Source:
[`tool_call_normalization.py`](../../../max/python/max/serve/parser/tool_call_normalization.py#L62)

<strong><em><a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md"><span style="color:#0b63ce">Previous code file: 0001_A_1_parse_openai_request_body_http_ingress_and_parser.md</span></a></em></strong>.

<strong><em><a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md"><span style="color:#0b63ce">Next section: 0001_B_1_parse_openai_request_body_pydantic_schema_validation.md</span></a></em></strong>.

## 1. Call Site

```python
    # Pre-normalize tools.parameters: null -> {} before Pydantic validation.
    tools = parsed.get("tools")
    if isinstance(tools, list):
        # WALKTHROUGH COMMENT (not in source):
        # Replace only the `tools` value in a newly constructed top-level dict.
        parsed = {**parsed, "tools": _normalize_tools_parameters(tools)}
```

```text
parsed: dict
   |
   | parsed.get("tools")
   v
tools
   |
   +-- not a list --> helper is not called
   |
   +-- list --------> _normalize_tools_parameters(tools)
                           |
                           v
                      normalized list
                           |
                           | {**parsed, "tools": normalized list}
                           v
                      new parsed dict
```

## 2. Complete Normalization Helper

```python
def _normalize_tools_parameters(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns ``tools`` with ``function.parameters`` coerced to a dict.

    OpenAI's API normalizes ``tools[*].function.parameters: null`` to an
    empty parameter list (equivalent to omitting the field) and returns
    200. MAX should match.

    - Dict ``parameters`` are passed through unchanged.
    - ``None`` or missing ``parameters`` becomes ``{}``.
    - Other values pass through unchanged (downstream validation handles
      type errors).
    - Tool entries without a ``function`` dict pass through unchanged.

    The input list and its dicts are not mutated.
    """

    # WALKTHROUGH COMMENT (not in source):
    # The result is always a new list.
    normalized: list[dict[str, Any]] = []

    for tool in tools:
        # WALKTHROUGH COMMENT (not in source):
        # Copy each tool dict before changing nested content.
        out = dict(tool)
        fn = out.get("function")

        if isinstance(fn, dict):
            # WALKTHROUGH COMMENT (not in source):
            # Copy the function dict before assigning `parameters`.
            fn = dict(fn)
            params = fn.get("parameters")

            if params is None:
                # WALKTHROUGH COMMENT (not in source):
                # Both an explicit null and a missing key produce None here.
                fn["parameters"] = {}

            out["function"] = fn

        normalized.append(out)

    return normalized
```

```text
for each tool
    |
    | out = dict(tool)
    +-- tool is dict-compatible --> out: new shallow tool dict
    |
    +-- tool is not dict-compatible --> TypeError --> route returns HTTP 400
    |
    | fn = out.get("function")
    v
fn
    |
    +-- not a dict --------------------------------------------+
    |                                                          |
    |                                                          v
    |                                                normalized.append(out)
    |
    +-- dict
         |
         | fn = dict(fn)
         v
       fn: new shallow function dict
         |
         | params = fn.get("parameters")
         v
       params
         |
         +-- None --> fn["parameters"] = {}
         |
         +-- any other value --> unchanged
         |
         | out["function"] = fn
         v
       normalized.append(out)
```

## 3. Variable And Copy Flow

```text
ORIGINAL OBJECTS                                  NEW OBJECTS
----------------------------------------------    ---------------------------------------------
tools: list[dict]                                 normalized: new list
  |                                                 |
  +-- tool: original dict      -- dict(tool) -->    +-- out: new tool dict
        |                                              |
        +-- function: original -- dict(fn) ---->       +-- fn: new function dict
                                                          |
                                                          +-- parameters: {} when None/missing

No assignment is made into:
  tools
  tool
  original function dict
```

## 4. Concrete Value Flow

```text
tools input
[
  {
    "type": "function",
    "function": {
      "name": "lookup_doc",
      "parameters": null
    }
  }
]
    |
    | _normalize_tools_parameters(tools)
    v
normalized output
[
  {
    "type": "function",
    "function": {
      "name": "lookup_doc",
      "parameters": {}
    }
  }
]
```

## 5. Complete Loop Variable Values

```python
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

tool = {
    "type": "function",
    "function": {
        "name": "lookup_doc",
        "description": "Look up a document.",
        "parameters": None,
    },
}

out_after_dict_tool = {
    "type": "function",
    "function": {
        "name": "lookup_doc",
        "description": "Look up a document.",
        "parameters": None,
    },
}

fn_after_dict_fn = {
    "name": "lookup_doc",
    "description": "Look up a document.",
    "parameters": None,
}

params = None

fn_after_normalization = {
    "name": "lookup_doc",
    "description": "Look up a document.",
    "parameters": {},
}

out_after_function_assignment = {
    "type": "function",
    "function": {
        "name": "lookup_doc",
        "description": "Look up a document.",
        "parameters": {},
    },
}

normalized = [
    {
        "type": "function",
        "function": {
            "name": "lookup_doc",
            "description": "Look up a document.",
            "parameters": {},
        },
    }
]
```

<strong><em><a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md"><span style="color:#0b63ce">Continue to schema validation: 0001_B_1_parse_openai_request_body_pydantic_schema_validation.md</span></a></em></strong>.
