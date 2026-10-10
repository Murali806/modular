# Complete Code Walk: `get_pipeline(...)`

Source:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L458)

<strong><em><a href="0002_B_1_get_pipeline_pipeline_tokenizer_protocol.md"><span style="color:#0b63ce">Protocol code walk: 0002_B_1_get_pipeline_pipeline_tokenizer_protocol.md</span></a></em></strong>.

## Route Call

```python
# WALKTHROUGH COMMENT (not in source):
# `completion_request` is already a validated Pydantic request object.
# `completion_request.model` is therefore a typed string field.
pipeline = get_pipeline(request, completion_request.model)
```

```text
request: Request                         completion_request.model: str
      |                                              |
      | request.app.state                            |
      +--------------------+-------------------------+
                           |
                           v
               get_pipeline(request, model_name)
                           |
                           v
                 TokenGeneratorPipeline
```

## Complete Function

```python
def get_pipeline(request: Request, model_name: str) -> TokenGeneratorPipeline:
    # WALKTHROUGH COMMENT (not in source):
    # The pipeline was created during application startup and stored on the
    # FastAPI application state. This function selects it; it does not build it.
    app_state: State = request.app.state
    pipeline: TokenGeneratorPipeline = app_state.pipeline

    # WALKTHROUGH COMMENT (not in source):
    # Start with the base model advertised by the running pipeline.
    models = [pipeline.model_name]

    # WALKTHROUGH COMMENT (not in source):
    # Dynamically loaded LoRA adapter names are also valid request model names.
    if lora_queue := app_state.pipeline.lora_queue:
        models += lora_queue.list_loras()

    # WALKTHROUGH COMMENT (not in source):
    # An empty request model means "use the server's base model."
    if not model_name:
        model_name = pipeline.model_name

    # WALKTHROUGH COMMENT (not in source):
    # Reject names that are neither the base model nor a loaded LoRA.
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}', currently serving '{models}'."
        )

    # WALKTHROUGH COMMENT (not in source):
    # `PipelineTokenizer` is runtime-checkable, so isinstance verifies that the
    # tokenizer exposes the protocol members needed by serving.
    if not isinstance(pipeline.tokenizer, PipelineTokenizer):
        raise ValueError(
            f"Tokenizer for '{model_name}' pipelines does not implement the PipelineTokenizer protocol."
        )

    # WALKTHROUGH COMMENT (not in source):
    # The same startup-created pipeline is returned for the base model and
    # LoRA names. Later request metadata selects the requested adapter.
    return pipeline
```

```text
request.app.state.pipeline
          |
          v
pipeline.model_name ------------------------------+
pipeline.lora_queue.list_loras() -----------------+--> models
                                                       |
model_name --------------------------------------------+
                                                       v
                                             model_name empty?
                                               |            |
                                              yes           no
                                               |            |
                                               v            |
                                      use pipeline.model_name
                                               |            |
                                               +------+-----+
                                                      v
                                          model_name in models?
                                           |              |
                                          no             yes
                                           |              |
                                           v              v
                                      ValueError      tokenizer implements
                                                      PipelineTokenizer?
                                                       |             |
                                                      no            yes
                                                       |             |
                                                       v             v
                                                  ValueError    return pipeline
```

## Variable Flow

```text
VARIABLE        CREATED FROM                         EXAMPLE VALUE
--------------  -----------------------------------  ------------------------------------------
app_state       request.app.state                    State(...)
pipeline        app_state.pipeline                   TokenGeneratorPipeline(...)
models          pipeline.model_name + LoRA names     ["base-llama", "finance-adapter"]
model_name      validated request field/default      "finance-adapter"
return value    existing pipeline object             same object as app_state.pipeline
```
