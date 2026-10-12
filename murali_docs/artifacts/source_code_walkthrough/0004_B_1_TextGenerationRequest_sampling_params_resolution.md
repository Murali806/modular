# Complete Code Walk: `SamplingParams.from_input_and_generation_config(...)`

Sources:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L2404),
[`sampling_params.py`](../../../max/python/max/pipelines/context/sampling_params.py#L47),
[`model_config.py`](../../../max/python/max/pipelines/lib/config/model_config.py#L1363)

<strong><em><a href="0003_B_1_openai_parse_chat_completion_request_tool_conversion.md"><span style="color:#0b63ce">Previous code walk: 0003_B_1_openai_parse_chat_completion_request_tool_conversion.md</span></a></em></strong>.

<strong><em><a href="0004_A_1_TextGenerationRequest_construction_and_validation.md"><span style="color:#0b63ce">Continue to TextGenerationRequest construction: 0004_A_1_TextGenerationRequest_construction_and_validation.md</span></a></em></strong>.

## One-Line Purpose

`SamplingParams.from_input_and_generation_config(...)` merges request-level
sampling choices with model defaults and built-in defaults, then creates one
validated `SamplingParams` object used during token selection.

```text
request/server values        model generation_config        class defaults
        |                              |                          |
        +------------------------------+--------------------------+
                                       |
                                       v
                    merge by priority + normalize + validate
                                       |
                                       v
                              SamplingParams
```

## Two Different Objects

```text
SamplingParamsInput
  |
  +-- temporary construction object
  +-- every field is optional
  +-- None means "look for a lower-priority default"
  |
  v
from_input_and_generation_config(...)
  |
  v
SamplingParams
  +-- concrete runtime values
  +-- validated and normalized
  +-- stored in TextGenerationRequest
  +-- later consumed by EOS tracking and sampling
```

`SamplingParamsInput` is not the final scheduler request state. It exists to
represent a partial set of overrides without losing the distinction between
"the user supplied this value" and "use a default."

## Route-Side Input Construction

Before calling the factory, the route resolves server defaults for three
fields and performs a few API-to-internal mappings.

```python
# Source: openai_routes.py
# Use request-level sampling params if provided, else server defaults.
temp = (
    completion_request.temperature
    if completion_request.temperature is not None
    else pipeline_config.runtime.temperature
)
top_k = (
    completion_request.top_k
    if completion_request.top_k is not None
    else pipeline_config.runtime.top_k
)
thinking_temp = (
    completion_request.thinking_temperature
    if completion_request.thinking_temperature is not None
    else pipeline_config.runtime.thinking_temperature
)
max_new_tokens = (
    completion_request.max_completion_tokens
    if completion_request.max_completion_tokens is not None
    else completion_request.max_tokens
)
sampling_params = SamplingParams.from_input_and_generation_config(
    SamplingParamsInput(
        top_k=top_k,
        top_p=completion_request.top_p,
        min_p=completion_request.min_p,
        temperature=temp,
        thinking_temperature=thinking_temp,
        frequency_penalty=completion_request.frequency_penalty,
        presence_penalty=completion_request.presence_penalty,
        repetition_penalty=completion_request.repetition_penalty,
        max_new_tokens=max_new_tokens,
        min_new_tokens=completion_request.min_tokens,
        ignore_eos=completion_request.ignore_eos,
        seed=completion_request.seed or randint(0, 2**63 - 1),
        stop_token_ids=completion_request.stop_token_ids,
        stop=_convert_stop(completion_request.stop),
    ),
    sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
)
```

## Route Mappings

```text
OPENAI / MAX REQUEST FIELD                  INTERNAL INPUT FIELD
----------------------------------------    ------------------------------------
temperature or server temperature       -> temperature
top_k or server top_k                   -> top_k
thinking_temperature or server default  -> thinking_temperature
top_p                                   -> top_p
min_p                                   -> min_p
frequency_penalty                       -> frequency_penalty
presence_penalty                        -> presence_penalty
repetition_penalty                      -> repetition_penalty
max_completion_tokens, else max_tokens  -> max_new_tokens
min_tokens                              -> min_new_tokens
ignore_eos                              -> ignore_eos
seed, else random integer               -> seed
stop_token_ids                          -> stop_token_ids
stop string/list                        -> stop: list[str] | None
```

Only `temperature`, `top_k`, and `thinking_temperature` consult
`pipeline_config.runtime` before the factory is called. Other absent fields go
directly to model defaults or class defaults.

### Stop Normalization

```python
def _convert_stop(stop: str | list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop
```

```text
None                 -> None
"\nUser:"           -> ["\nUser:"]
["END", "STOP"]    -> ["END", "STOP"]
```

### Seed Selection

The current route uses:

```python
completion_request.seed or randint(0, 2**63 - 1)
```

Therefore:

```text
seed = 42    -> 42
seed = None  -> random integer
seed = 0     -> random integer, because 0 is false in Python
```

## Exact Priority Order

For `temperature` and `top_k`:

```text
highest priority
      |
      +-- 1. request value
      +-- 2. server runtime default
      +-- 3. model generation_config value
      +-- 4. SamplingParams class default
      |
lowest priority
```

For `thinking_temperature`, the priority is request value, then server runtime
default, then the `SamplingParams` class default (`None`). The model-default
object does not carry `thinking_temperature`.

For fields without a server runtime default:

```text
highest priority
      |
      +-- 1. request value
      +-- 2. model generation_config value, when supported
      +-- 3. SamplingParams class default
      |
lowest priority
```

The model-default object can carry these `generation_config` values:

```text
temperature
top_p
top_k
repetition_penalty
max_new_tokens
min_new_tokens
do_sample
eos_token_id -> mapped to stop_token_ids
```

Fields such as `min_p`, frequency/presence penalties, `ignore_eos`, `stop`,
and `seed` are not populated from the model generation config by this class.

## Model Default Extraction

Only values explicitly different from Hugging Face's base defaults are copied
from the model's generation config.

```python
@computed_field  # type: ignore[prop-decorator]
@property
def sampling_params_defaults(
    self,
) -> SamplingParamsGenerationConfigDefaults:
    """Returns sampling defaults derived from the generation config."""
    defaults = {}
    for (
        field_name,
        field_value,
    ) in self.generation_config.to_diff_dict().items():
        if (
            field_name
            in SamplingParamsGenerationConfigDefaults.__dataclass_fields__
        ):
            defaults[field_name] = field_value

    return SamplingParamsGenerationConfigDefaults(**defaults)
```

`eos_token_id` receives one special mapping:

```python
@cached_property
def _values_to_update(self) -> dict[str, float | int | list[int]]:
    values: dict[str, float | int | list[int]] = {}
    for _field in fields(self):
        field_value = getattr(self, _field.name)
        if field_value is not None:
            if _field.name == "eos_token_id":
                if isinstance(field_value, int):
                    values["stop_token_ids"] = [field_value]
                elif isinstance(field_value, list):
                    values["stop_token_ids"] = field_value
            else:
                values[_field.name] = field_value
    return values

@property
def values_to_update(self) -> dict[str, float | int | list[int]]:
    """Non-``None`` field values as a dictionary.

    Returns:
        A dictionary mapping field names to their values, excluding any fields
        that are ``None``. This dictionary can be used to update
        :class:`SamplingParams` default values.

    .. code-block:: python

        from max.pipelines.context.sampling_params import (
            SamplingParamsGenerationConfigDefaults,
        )

        defaults = SamplingParamsGenerationConfigDefaults(
            temperature=0.7,
            top_k=50,
        )
        defaults.values_to_update
        # {'temperature': 0.7, 'top_k': 50}
    """
    # Return a copy of the values, because the caller mutates the returned dict.
    return self._values_to_update.copy()
```

```text
eos_token_id = 128009       -> stop_token_ids = [128009]
eos_token_id = [1, 2, 3]    -> stop_token_ids = [1, 2, 3]
```

## Factory Source

```python
@classmethod
def from_input_and_generation_config(
    cls,
    input_params: SamplingParamsInput,
    sampling_params_defaults: SamplingParamsGenerationConfigDefaults,
) -> SamplingParams:
    """Creates a :class:`SamplingParams` instance with defaults from a HuggingFace GenerationConfig.

    Combines three sources of values in priority order (highest to lowest):

    1. User-provided values in ``input_params`` (non-``None``)
    2. Model's GenerationConfig values (only if explicitly set in the model's config)
    3. :class:`SamplingParams` class defaults

    Args:
        input_params: Dataclass containing user-specified parameter values.
            Values of ``None`` will be replaced with model defaults or class defaults.
        sampling_params_defaults: :class:`SamplingParamsGenerationConfigDefaults`
            containing default sampling parameters extracted from the model's
            GenerationConfig.

    Returns:
        A new :class:`SamplingParams` instance with model-aware defaults.

    .. code-block:: python

        from max.pipelines.context.sampling_params import (
            SamplingParams,
            SamplingParamsGenerationConfigDefaults,
            SamplingParamsInput,
        )

        sampling_params_defaults = SamplingParamsGenerationConfigDefaults(
            top_k=50,
        )
        params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(temperature=0.7),
            sampling_params_defaults=sampling_params_defaults,
        )
    """
    # Start with model's generation config values (only if explicitly set)
    defaults: dict[str, Any] = sampling_params_defaults.values_to_update

    # Handle special mappings from GenerationConfig
    if "do_sample" in defaults and defaults["do_sample"] is not None:
        # If do_sample is False, set greedy defaults (unless user overrides)
        if not defaults["do_sample"]:
            defaults["temperature"] = 0
            defaults["top_k"] = 1

        # This isn't included in SamplingParams, therefore we should remove it.
        del defaults["do_sample"]

    # Overlay user-provided values (highest priority)
    for _field in fields(input_params):
        value = getattr(input_params, _field.name)
        if value is not None:
            defaults[_field.name] = value

    return cls(**defaults)
```

## Factory Flow

```text
sampling_params_defaults.values_to_update
              |
              v
defaults: dict
              |
              +-- do_sample=False?
              |      |
              |      +-- defaults["temperature"] = 0
              |      +-- defaults["top_k"] = 1
              |
              +-- remove do_sample
              |
              +-- for every SamplingParamsInput field
              |      |
              |      +-- value is None ----> keep lower-priority value
              |      +-- value is present -> overwrite defaults[field]
              |
              v
SamplingParams(**defaults)
              |
              +-- dataclass class defaults fill omitted fields
              +-- __post_init__ normalizes and validates
              v
final SamplingParams
```

## Final Runtime Fields

```text
SamplingParams
  |
  +-- top_k
  +-- top_p
  +-- min_p
  +-- temperature
  +-- thinking_temperature
  +-- frequency_penalty
  +-- presence_penalty
  +-- repetition_penalty
  +-- max_new_tokens
  +-- min_new_tokens
  +-- ignore_eos
  +-- stop
  +-- stop_token_ids
  +-- detokenize
  +-- seed
  +-- logits_processors
```

## Normalization And Validation

Constructing `SamplingParams(**defaults)` invokes `SamplingParams.__post_init__`.

```text
new SamplingParams
  |
  +-- top_k is None? ----------------------> class default -1
  +-- top_p is None? ----------------------> class default 1.0
  +-- top_k == 0? -------------------------> normalize to -1
  +-- temperature == 0? ------------------> force top_k = 1
  |
  +-- min_p outside [0, 1]? --------------> ValueError
  +-- top_p outside [0, 1]? --------------> ValueError
  +-- top_k < -1? ------------------------> ValueError
  +-- temperature outside [0, 2]? --------> ValueError
  +-- thinking_temperature outside [0, 2]? -> ValueError
  +-- repetition_penalty non-finite/<=0? --> ValueError
  +-- frequency_penalty outside [-2, 2]? --> ValueError
  +-- presence_penalty outside [-2, 2]? ---> ValueError
  +-- max_new_tokens bool/negative? -------> ValueError
  +-- min_new_tokens bool/negative? -------> ValueError
  |
  v
validated SamplingParams
```

The chat route catches these `ValueError` exceptions and returns HTTP 400.

## Worked Example

### Complete Route Inputs

```python
completion_request_values = {
    "temperature": None,
    "top_k": None,
    "top_p": 0.9,
    "min_p": None,
    "thinking_temperature": None,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.0,
    "repetition_penalty": None,
    "max_completion_tokens": 128,
    "max_tokens": 256,
    "min_tokens": 8,
    "ignore_eos": False,
    "seed": 42,
    "stop_token_ids": [128009],
    "stop": "\nUser:",
}

pipeline_runtime_values = {
    "temperature": 0.7,
    "top_k": None,
    "thinking_temperature": None,
}

sampling_params_defaults = SamplingParamsGenerationConfigDefaults(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
    max_new_tokens=512,
    min_new_tokens=0,
    do_sample=True,
    eos_token_id=128009,
)
```

### Step 1: Route Resolution

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    ------------------------------------    -----------------------------------
request temperature=None               use server value when request is None   temp=0.7
server temperature=0.7               ------------------------------->

request top_k=None                     server top_k is also None                top_k=None
server top_k=None                    ------------------------------->          factory may use model top_k

max_completion_tokens=128              prefer max_completion_tokens            max_new_tokens=128
max_tokens=256                        ------------------------------->

stop="\nUser:"                       _convert_stop(...)                      stop=["\nUser:"]
                                      ------------------------------->

seed=42                                seed or random integer                   seed=42
                                      ------------------------------->
```

### Step 2: Complete `SamplingParamsInput`

```python
input_params = SamplingParamsInput(
    top_k=None,
    top_p=0.9,
    min_p=None,
    temperature=0.7,
    thinking_temperature=None,
    frequency_penalty=0.2,
    presence_penalty=0.0,
    repetition_penalty=None,
    max_new_tokens=128,
    min_new_tokens=8,
    ignore_eos=False,
    stop=["\nUser:"],
    stop_token_ids=[128009],
    detokenize=None,
    seed=42,
    logits_processors=None,
)
```

### Step 3: Model Defaults Dictionary

```python
defaults = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 512,
    "min_new_tokens": 0,
    "do_sample": True,
    "stop_token_ids": [128009],
}
```

`do_sample=True` is removed without changing `temperature` or `top_k`.

### Step 4: Overlay Non-`None` Input Values

```python
defaults_after_input_overlay = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 128,
    "min_new_tokens": 8,
    "stop_token_ids": [128009],
    "frequency_penalty": 0.2,
    "presence_penalty": 0.0,
    "ignore_eos": False,
    "stop": ["\nUser:"],
    "seed": 42,
}
```

Important examples:

```text
temperature=0.7       request/server value replaces model 0.8
top_p=0.9             request value replaces model 0.95
top_k=None            does not overwrite model top_k=50
repetition=None       does not overwrite model repetition_penalty=1.1
max_new_tokens=128    request value replaces model 512
```

### Step 5: Complete Final Output

Fields absent from the overlay receive `SamplingParams` class defaults.

```python
sampling_params = SamplingParams(
    top_k=50,
    top_p=0.9,
    min_p=0.0,
    temperature=0.7,
    thinking_temperature=None,
    frequency_penalty=0.2,
    presence_penalty=0.0,
    repetition_penalty=1.1,
    max_new_tokens=128,
    min_new_tokens=8,
    ignore_eos=False,
    stop=["\nUser:"],
    stop_token_ids=[128009],
    detokenize=True,
    seed=42,
    logits_processors=None,
)
```

## Greedy-Decoding Branches

### Model Sets `do_sample=False`

```text
model defaults
  do_sample=False
       |
       +-- temperature=0
       +-- top_k=1
       |
       +-- route input supplied temperature/top_k?
              |
              +-- yes -> request/server values overwrite greedy defaults
              +-- no  -> keep temperature=0 and top_k=1
```

### Final Temperature Is Zero

Even if a different `top_k` survives the merge, `__post_init__` applies:

```text
temperature == 0
      |
      v
top_k = 1
```

This guarantees greedy token selection for a zero-temperature request.

## Where The Output Goes

```text
SamplingParams
  |
  +-- TextGenerationRequest.sampling_params
  |        |
  |        v
  |     tokenizer.new_context(request)
  |        |
  |        +-- max_new_tokens -> maximum context generation length
  |        +-- stop/stop_token_ids/ignore_eos -> EOS tracker
  |
  +-- FusedSamplingProcessor
           |
           +-- temperature
           +-- top_k / top_p / min_p
           +-- penalties
           +-- seed
           v
        sampled next token
```

<details>
<summary><strong>Q: Why not construct <code>SamplingParams(...)</code> directly?</strong></summary>

The factory preserves model-specific generation defaults. Direct construction
would skip values explicitly supplied by the model's Hugging Face
`generation_config` unless every caller manually reproduced the same merge.

</details>

<details>
<summary><strong>Q: Does <code>None</code> become the final value?</strong></summary>

Usually no. In `SamplingParamsInput`, `None` means "do not override." The
factory then keeps a model default or lets the `SamplingParams` class default
fill that field. Fields whose final type allows `None`, such as
`thinking_temperature`, `max_new_tokens`, `stop`, `stop_token_ids`, and
`logits_processors`, may legitimately remain `None`.

</details>

<details>
<summary><strong>Q: Which value wins when both token-limit fields are present?</strong></summary>

`max_completion_tokens` wins over `max_tokens` in the route mapping and becomes
`SamplingParams.max_new_tokens`.

</details>

<details>
<summary><strong>Q: Is <code>SamplingParams</code> only about randomness?</strong></summary>

No. It also carries generation length limits, stop strings and token IDs, EOS
behavior, penalties, detokenization control, the random seed, and optional
custom logits processors.

</details>

## Short Version

```text
OpenAI/MAX request fields
  -> apply selected server defaults
  -> build optional SamplingParamsInput
  -> start with explicit model generation_config defaults
  -> do_sample=False maps to temperature=0, top_k=1
  -> overlay every non-None input field
  -> construct SamplingParams
  -> normalize top_k and zero-temperature behavior
  -> validate ranges and token counts
  -> store in TextGenerationRequest
  -> use during EOS tracking and next-token sampling
```
