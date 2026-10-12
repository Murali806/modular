# End-to-End Call Stack: OpenAI Chat Request To Llama And Back

This document traces one OpenAI-compatible chat-completion request through MAX
Serve, from HTTP ingress to Llama model execution and back to the HTTP client.

The tree is based on the current repository source and reconciles the focused
notes and complete-code walkthroughs in this folder.

## Scope

The primary runtime path assumes:

- `POST /v1/chat/completions`
- `PipelineTask.TEXT_GENERATION`
- the normal combined `prefill_and_decode` scheduler
- `TextTokenizer`
- `ZmqModelWorkerProxy`
- `TextGenerationPipeline`
- the Llama 3 graph model and `Llama3BatchProcessor`
- one successfully admitted request

Configuration-dependent alternatives are called out separately. In
particular, disaggregated prefill/decode, speculative decoding, overlap
scheduling, multimodal architectures, and non-Llama models replace parts of
the worker-side tree.

The recursion expands repository-owned code on this path. FastAPI, Starlette,
asyncio, Pydantic internals, ZeroMQ transport internals, and compiled graph or
device-kernel internals are named at their boundary but are not expanded as if
they were ordinary Python calls in this repository.

## Legend

```text
[CALL]       ordinary Python call
[AWAIT]      coroutine call suspended until its awaited work completes
[ITERATE]    async generator is being consumed
[FRAMEWORK]  FastAPI, Starlette, SSE, or asyncio invokes user code
[IPC]        process/queue handoff; not a nested Python call
[BACKGROUND] independently running task or worker loop
[COMPILED]   execution crosses into a compiled MAX graph/runtime
[BRANCH]     only one runtime branch is selected
```

Inline links use the form `[Name:Lnn]`: `Name` is a shortened walkthrough
name, and `Lnn` is the exact line where that function is explained.

## 1. One-Time Startup Wiring

The request path depends on objects and background loops created before the
first HTTP request arrives.

<details>
<summary><strong>Expand: API server startup and API-worker wiring</strong></summary>

<pre>
fastapi_app(settings, serving_settings)  <a href="0000_A_folder_structure.md#expandable-root-level-map">[Map]</a>
  api_server.py:330
    |
    +-- [CALL] FastAPI(title="MAX Serve", ...)
    |
    +-- [CALL] app.include_router(ROUTES[api_type].router)
    |     api_server.py:438
    |       |
    |       +-- OpenAI router prefix = "/v1"
    |             openai_routes.py:235
    |
    +-- [FRAMEWORK] lifespan(...)
          api_server.py:111
            |
            +-- [CALL] ZmqModelWorkerInterface(...)
            |
            +-- [AWAIT] start_model_worker(...)
            |     model_worker.py:613
            |       |
            |       +-- proc.start(ModelWorker(), ...)
            |       |
            |       +-- [FRAMEWORK] spawned child invokes
            |       |     ModelWorker.__call__(...)
            |       |       model_worker.py:564
            |       |         |
            |       |         +-- [CALL] uvloop.run(ModelWorker.run(...))
            |       |
            |       +-- [AWAIT] proc.ready(...)
            |       |
            |       +-- [AWAIT] model_worker_interface.model_worker_proxy()
            |             zmq_interface.py:331
            |               |
            |               +-- create ZmqModelWorkerProxy
            |               |
            |               +-- [BACKGROUND] asyncio.create_task(
            |               |     proxy.response_worker()  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L100">[Route:L100]</a>
            |               |   )
            |               |     zmq_interface.py:247
            |               |
            |               +-- [BACKGROUND] asyncio.create_task(
            |               |     proxy._metrics_worker()
            |               |   )
            |               |
            |               +-- [AWAIT] proxy.wait_until_connected(...)
            |
            +-- [CALL] TokenGeneratorPipeline(...)
            |
            +-- app.state.pipeline = TokenGeneratorPipeline
            +-- app.state.pipeline_config = PipelineConfig
    |
    +-- app.state.settings = settings
    +-- [CALL] register_request(app, ...)
          |
          +-- installs request_session HTTP middleware
</pre>

</details>

The spawned model-worker process builds the model pipeline and then runs the
scheduler forever:

<details>
<summary><strong>Expand: Model-worker startup and scheduler loop</strong></summary>

<pre>
[BACKGROUND] uvloop executes ModelWorker.run(...)
  model_worker.py:232
    |
    +-- [CALL] model_factory()
    |     |
    |     +-- build TextGenerationPipeline
    |     +-- build/load Llama graph model
    |     +-- compile graph and sampler once
    |
    +-- [AWAIT] model_worker_interface.model_worker_queues()
    |     |
    |     +-- request_queue: worker-side ZMQ PULL
    |     +-- response_queue: worker-side ZMQ PUSH
    |     +-- cancel_queue: worker-side ZMQ PULL
    |
    +-- [CALL] load_scheduler(...)
    |     scheduler/__init__.py
    |       |
    |       +-- pipeline_role == "prefill_and_decode"
    |             |
    |             +-- load_text_generation_scheduler(...)
    |                   |
    |                   +-- TokenGenerationScheduler
    |
    +-- [BACKGROUND] while True
          |
          +-- scheduler.run_iteration()  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L64">[Iter:L64]</a>
          +-- sleep_with_backoff(...) when no progress
</pre>

</details>

## 2. Complete Request Call Tree

This is the continuous source-level tree for the normal OpenAI ingress path.

Immediately before route dispatch, MAX Serve's HTTP middleware creates the
request ID and timer consumed by the route:

<details>
<summary><strong>Expand: Request middleware prelude</strong></summary>

```text
[FRAMEWORK] request_session(request, call_next)
  request.py:52
    |
    +-- request_id = uuid.uuid4().hex
    +-- request.state.request_id = request_id
    +-- request.state.request_timer = StopWatch()
    |
    +-- [AWAIT] call_next(request)
          |
          +-- enters the route tree below
```

</details>

<details>
<summary><strong>Expand: Complete HTTP route call tree</strong></summary>

<pre>
POST /v1/chat/completions
    |
    +-- [FRAMEWORK] FastAPI/Starlette route dispatch
          |
          +-- [AWAIT] openai_create_chat_completion(request)  <a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md#L20">[Ingress:L20]</a>
                openai_routes.py:2218
                  |
                  +-- request_id = request.state.request_id
                  |
                  +-- [AWAIT] _parse_openai_request_body(  <a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md#L99">[Parse:L99]</a>
                  |             request,
                  |             request_id,
                  |             CreateChatCompletionRequest,
                  |           )
                  |     openai_routes.py:3070
                  |       |
                  |       +-- [AWAIT] request.body()
                  |       |
                  |       +-- [CALL] json.loads(raw)
                  |       |     |
                  |       |     +-- parsed: dict/list/str/number/bool/None
                  |       |
                  |       +-- [BRANCH] parsed is not dict
                  |       |     |
                  |       |     +-- model_cls.model_validate(parsed)  <a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md#L9">[Validate:L9]</a>
                  |       |           |
                  |       |           +-- usually ValidationError for chat schema
                  |       |
                  |       +-- [BRANCH] parsed is dict
                  |             |
                  |             +-- tools = parsed.get("tools")
                  |             |
                  |             +-- [BRANCH] tools is list
                  |             |     |
                  |             |     +-- [CALL] _normalize_tools_parameters(tools)  <a href="0001_A_2_parse_openai_request_body_tool_parameter_normalization.md#L43">[Tools:L43]</a>
                  |             |           tool_call_normalization.py:62
                  |             |             |
                  |             |             +-- copy each tool dict
                  |             |             +-- copy function dict
                  |             |             +-- parameters None/missing -> {}
                  |             |
                  |             +-- [CALL] get_app_pipeline_config(request.app)  <a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md#L63">[Config:L63]</a>
                  |             |
                  |             +-- [BRANCH] allow_extra_request_fields
                  |             |     |
                  |             |     +-- known = set(model_cls.model_fields)
                  |             |     +-- calculate unknown keys
                  |             |     +-- rebuild parsed without unknown keys
                  |             |
                  |             +-- [CALL] model_cls.model_validate(parsed)  <a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md#L9">[Validate:L9]</a>
                  |                   inherited BaseModel.model_validate
                  |                     |
                  |                     +-- CreateChatCompletionRequest
                  |                     |     schemas/openai.py:448
                  |                     |
                  |                     +-- [CALL] before model validators
                  |                     |     |
                  |                     |     +-- _translate_thinking_to_standard(...)  <a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md#L346">[Before:L346]</a>
                  |                     |
                  |                     +-- validate inherited OpenAI fields
                  |                     +-- validate MAX extension fields
                  |                     +-- validate model and messages
                  |                     +-- reject forbidden extra fields
                  |                     |
                  |                     +-- [CALL] after model validators
                  |                           |
                  |                           +-- _reconcile_max_completion_tokens(...)  <a href="0001_B_1_parse_openai_request_body_pydantic_schema_validation.md#L376">[After:L376]</a>
                  |                           |
                  |                           +-- completion_request:
                  |                               CreateChatCompletionRequest
                  |
                  +-- [CALL] get_pipeline(  <a href="0002_A_1_get_pipeline_model_selection.md#L33">[Select:L33]</a>
                  |             request,
                  |             completion_request.model,
                  |           )
                  |     openai_routes.py:458
                  |       |
                  |       +-- app_state = request.app.state
                  |       +-- pipeline = app_state.pipeline
                  |       +-- models = base model + LoRA names
                  |       +-- validate requested model name
                  |       +-- isinstance(pipeline.tokenizer, PipelineTokenizer)  <a href="0002_B_1_get_pipeline_pipeline_tokenizer_protocol.md#L16">[Protocol:L16]</a>
                  |       +-- return existing TokenGeneratorPipeline
                  |
                  +-- [CALL] get_tool_parser(request.app)
                  |
                  +-- tokenizer = pipeline.tokenizer
                  |
                  +-- [AWAIT] openai_parse_chat_completion_request(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L122">[Chat:L122]</a>
                  |     openai_routes.py:1719
                  |       |
                  |       +-- [CALL] _validate_tool_message_consistency(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L136">[MsgCheck:L136]</a>
                  |       |
                  |       +-- validate every message role
                  |       |
                  |       +-- for each completion_request.messages entry
                  |       |     |
                  |       |     +-- content = message.get("content")
                  |       |     +-- raw_tool_calls = message.get("tool_calls")
                  |       |     |
                  |       |     +-- [CALL] normalize_tool_call_arguments(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L189">[ToolArgs:L189]</a>
                  |       |     |     when tool calls are present
                  |       |     |
                  |       |     +-- [BRANCH] content is list
                  |       |     |     |
                  |       |     |     +-- for each content_part
                  |       |     |           |
                  |       |     |           +-- text -> TextContentPart  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L263">[Text:L263]</a>
                  |       |     |           |
                  |       |     |           +-- image_url
                  |       |     |           |     +-- make_media_ref(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L217">[MediaRef:L217]</a>
                  |       |     |           |     +-- ImageContentPart(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L220">[Image:L220]</a>
                  |       |     |           |
                  |       |     |           +-- video_url
                  |       |     |                 +-- make_media_ref(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L234">[MediaRef:L234]</a>
                  |       |     |                 +-- VideoContentPart(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L237">[Video:L237]</a>
                  |       |     |
                  |       |     +-- [BRANCH] content is scalar/None
                  |       |     |     |
                  |       |     |     +-- use content or ""
                  |       |     |
                  |       |     +-- [CALL] _normalize_openai_role(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L269">[Role:L269]</a>
                  |       |     +-- [CALL] TextGenerationRequestMessage(...)  <a href="0003_A_1_openai_parse_chat_completion_request_messages.md#L268">[Message:L268]</a>
                  |       |
                  |       +-- record image/video count metrics
                  |       +-- enforce model-specific media count limits
                  |       |
                  |       +-- [CALL] _request_media_budget(settings)  <a href="0003_A_2_openai_parse_chat_completion_request_media.md#L59">[Media:L59]</a>
                  |       |     _image_resolution.py:582
                  |       |
                  |       +-- [OPTIONAL MEDIA BRANCH]
                  |       |     |
                  |       |     +-- create resolve_image_from_url(...) tasks  <a href="0003_A_2_openai_parse_chat_completion_request_media.md#L61">[Resolve:L61]</a>
                  |       |     |     _image_resolution.py:839
                  |       |     |       |
                  |       |     |       +-- http/https -> _fetch_url_bytes(...)
                  |       |     |       +-- data URI -> _decode_data_uri_base64(...)
                  |       |     |       +-- file URI -> validate root + read bytes
                  |       |     |
                  |       |     +-- [AWAIT] asyncio.gather(image tasks)
                  |       |     |
                  |       |     +-- [AWAIT] asyncio.to_thread(
                  |       |     |             _resolve_and_decode_images,  <a href="0003_A_2_openai_parse_chat_completion_request_media.md#L74">[Decode:L74]</a>
                  |       |     |           )
                  |       |     |
                  |       |     +-- create/await video resolution tasks
                  |       |
                  |       +-- return _ParsedChatRequest(
                  |             messages,
                  |             request_images,
                  |             request_videos,
                  |             decoded_images,
                  |           )
                  |
                  +-- [CALL] get_app_pipeline_config(request.app)  <a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md#L63">[Config:L63]</a>
                  |
                  +-- [CALL]
                  |     _convert_chat_completion_tools_to_token_generator_tools(...)  <a href="0003_B_1_openai_parse_chat_completion_request_tool_conversion.md#L85">[Tools:L85]</a>
                  |
                  +-- [CALL] _create_response_format(...)
                  |
                  +-- [OPTIONAL GRAMMAR BRANCH]
                  |     |
                  |     +-- _resolve_grammar_constraints(...)
                  |     +-- parser.generate_tool_call_grammar(...)
                  |     +-- TextGenerationResponseFormat(type="grammar", ...)
                  |
                  +-- [CALL] OpenAIChatResponseGenerator(...)
                  |
                  +-- [CALL] SamplingParams.from_input_and_generation_config(...)
                  |
                  +-- [CALL] TextGenerationRequest(...)  <a href="0004_A_1_TextGenerationRequest_construction_and_validation.md#L15">[Request:L15]</a>
                  |     text_generation.py:336
                  |       |
                  |       +-- [CALL] TextGenerationRequest.__post_init__()  <a href="0004_A_1_TextGenerationRequest_construction_and_validation.md#L99">[Validate:L99]</a>
                  |             text_generation.py:479
                  |               |
                  |               +-- convert dict messages to typed messages
                  |               +-- reject prompt + messages together
                  |               +-- reject string prompt with media
                  |               +-- verify media placeholder counts
                  |
                  +-- [BRANCH] completion_request.stream
                        |
                        +-- True  -> streaming path in section 3
                        |
                        +-- False -> non-streaming path in section 4
</pre>

</details>

## 3. Streaming Path: Admission, First Token, And SSE

The streaming branch deliberately submits the request and resolves the first
item before returning the SSE response. This allows admission and early
generation failures to become a real non-200 HTTP response.

<details>
<summary><strong>Expand: Streaming admission and first SSE chunk</strong></summary>

<pre>
openai_create_chat_completion(...)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L12">[Dispatch:L12]</a>
    |
    +-- [AWAIT] response_generator.stream(token_request)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L57">[Stream:L57]</a>
    |     OpenAIChatResponseGenerator.stream
    |     openai_routes.py:687
    |       |
    |       +-- [AWAIT] pipeline.next_token_chunk(token_request)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L65">[Next:L65]</a>
    |       |     llm.py:294
    |       |       |
    |       |       +-- initialize TTFT/ITL/decode timers
    |       |       +-- optionally create/reset reasoning parser
    |       |       |
    |       |       +-- model_worker.note_awaiting_admission(+1)  <a href="0008_A_1_note_awaiting_admission_counter_lifecycle.md#L64">[Await:L64]</a>
    |       |       |
    |       |       +-- [AWAIT] tokenizer.new_context(request)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L189">[Context:L189]</a>
    |       |       |     tokenizer.py:819
    |       |       |       |
    |       |       |       +-- [AWAIT] _generate_prompt_and_token_ids(...)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L95">[Prompt:L95]</a>
    |       |       |       |     tokenizer.py:718
    |       |       |       |       |
    |       |       |       |       +-- [BRANCH] prompt str/list exists
    |       |       |       |       |     +-- [AWAIT] encode(prompt, ...)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L103">[Encode:L103]</a>
    |       |       |       |       |
    |       |       |       |       +-- [BRANCH] use messages
    |       |       |       |             +-- apply_chat_template(...)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L105">[Template:L105]</a>
    |       |       |       |             +-- [AWAIT] _encode_chat_prompt(...)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L139">[Encode:L139]</a>
    |       |       |       |                   +-- custom encoder or encode(...)
    |       |       |       |
    |       |       |       +-- derive json_schema and grammar
    |       |       |       +-- GrammarEnforcementState.from_response_format(...)  <a href="0006_A_2_tokenizer_new_context_metadata_and_state.md#L55">[Grammar:L55]</a>
    |       |       |       +-- max_tokens_to_generate(...)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L219">[MaxTokens:L219]</a>
    |       |       |       +-- TokenBuffer(token_ids)  <a href="0006_A_2_tokenizer_new_context_metadata_and_state.md#L126">[Buffer:L126]</a>
    |       |       |       +-- [AWAIT] create_eos_tracker(request)  <a href="0006_A_2_tokenizer_new_context_metadata_and_state.md#L86">[EOS:L86]</a>
    |       |       |       +-- TextContext(...)  <a href="0006_A_2_tokenizer_new_context_metadata_and_state.md#L84">[TextCtx:L84]</a>
    |       |       |       +-- return context
    |       |       |
    |       |       +-- inject_trace_carrier(context)
    |       |       |
    |       |       +-- [CALL] create_buffered_detokenizer(...)  <a href="0007_A_1_create_buffered_detokenizer_factory_and_decode.md#L47">[Detok:L47]</a>
    |       |       |     incremental_detokenizer.py:343
    |       |       |       +-- content detokenizer
    |       |       |       +-- reasoning detokenizer
    |       |       |
    |       |       +-- [AWAIT] model_worker.stream(  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L34">[IPC:L34]</a>
    |       |       |             context.request_id,
    |       |       |             context,
    |       |       |           )
    |       |       |     ZmqModelWorkerProxy.stream
    |       |       |     zmq_interface.py:105
    |       |       |       |
    |       |       |       +-- reject duplicate request ID
    |       |       |       +-- [AWAIT] acquire _admission_lock
    |       |       |       +-- [AWAIT] request_queue.writable()
    |       |       |       |
    |       |       |       +-- [BRANCH] queue full
    |       |       |       |     +-- raise RequestQueueFull
    |       |       |       |           +-- API exception handler -> HTTP 429
    |       |       |       |
    |       |       |       +-- create per-request asyncio out_queue
    |       |       |       +-- pending_out_queues[request_id] = out_queue
    |       |       |       +-- [AWAIT] request_queue.put(context)
    |       |       |       |     |
    |       |       |       |     +-- [IPC] API process -> model worker process
    |       |       |       |     |
    |       |       |       |     +-- [BRANCH] put raises
    |       |       |       |           +-- delete pending_out_queues[request_id]
    |       |       |       |           +-- send best-effort cancellation
    |       |       |       |           +-- re-raise
    |       |       |       |
    |       |       |       +-- return _drain_responses(request_id, out_queue)  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L126">[Drain:L126]</a>
    |       |       |
    |       |       +-- [BRANCH] tokenization or worker handoff raises
    |       |       |     +-- model_worker.note_awaiting_admission(-1)  <a href="0008_A_1_note_awaiting_admission_counter_lifecycle.md#L64">[Await:L64]</a>
    |       |       |     +-- re-raise
    |       |       |
    |       |       +-- successful worker handoff
    |       |             +-- model_worker.note_awaiting_admission(-1)  <a href="0008_A_1_note_awaiting_admission_counter_lifecycle.md#L64">[Await:L64]</a>
    |       |             +-- return TokenGeneratorPipeline._generate async generator
    |       |
    |       +-- return OpenAIChatResponseGenerator._stream async generator
    |
    +-- [AWAIT] _start_stream(stream)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L87">[Gate:L87]</a>
    |     openai_routes.py:185
    |       |
    |       +-- [AWAIT] stream.__anext__()
    |             |
    |             +-- starts OpenAIChatResponseGenerator._stream(...)
    |             +-- [ITERATE] token_generator
    |                   |
    |                   +-- starts TokenGeneratorPipeline._generate(...)
    |                   +-- [ITERATE] response_stream
    |                         |
    |                         +-- starts ZmqModelWorkerProxy._drain_responses(...)  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L126">[Drain:L126]</a>
    |                         +-- [AWAIT] per-request out_queue.get()
    |                               |
    |                               +-- waits for worker path in section 5
    |
    +-- after first item succeeds:
          return EventSourceResponse(token_stream, ...)
            |
            +-- request_session resumes
            |     +-- add X-Request-ID response header
            |     +-- wrap body iterator for stream-span cleanup
            |
            +-- [FRAMEWORK] ASGI starts the HTTP response
                  |
                  +-- EventSourceResponse iterates token_stream
</pre>

</details>

## 4. Non-Streaming Path

The non-streaming path uses the same admission, tokenization, worker,
scheduler, model, and detokenization path. The difference is that it consumes
the complete token generator before constructing one JSON response.

<details>
<summary><strong>Expand: Non-streaming aggregation path</strong></summary>

<pre>
openai_create_chat_completion(...)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L24">[Dispatch:L24]</a>
    |
    +-- [AWAIT] response_generator.complete([token_request])  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L130">[Complete:L130]</a>
          OpenAIChatResponseGenerator.complete
          openai_routes.py:1159
            |
            +-- [AWAIT] pipeline.all_tokens(token_request)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L156">[All:L156]</a>
            |     llm.py:678
            |       |
            |       +-- [AWAIT] pipeline.next_token_chunk(token_request)  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L65">[Next:L65]</a>
            |       |     +-- same admission path as section 3
            |       |
            |       +-- [ITERATE] generator until request completes
            |       +-- return list[TokenGeneratorOutput]
            |
            +-- concatenate decoded content/reasoning
            +-- process log probabilities
            +-- derive finish_reason
            +-- optionally parse complete tool calls
            +-- build ChatCompletionResponseChoice objects
            +-- build CompletionUsage
            +-- CreateChatCompletionResponse(...)
            +-- return response to FastAPI
</pre>

</details>

## 5. Worker-Side Recursive Execution

The worker-side path is not called directly by the API coroutine. It is
activated by the `TextContext` arriving on the worker's ZMQ request queue.

<details>
<summary><strong>Expand: Worker scheduler through Llama execution</strong></summary>

<pre>
[IPC] TextContext arrives on worker request_queue
    |
    +-- [BACKGROUND] ModelWorker.run(...) loop
          model_worker.py:232
            |
            +-- [CALL] scheduler.run_iteration()  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L64">[Iter:L64]</a>
                  TokenGenerationScheduler.run_iteration
                  text_generation_scheduler.py:205
                    |
                    +-- [CALL] _retrieve_pending_requests()  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L10">[Retrieve:L10]</a>
                    |     text_generation_scheduler.py:161
                    |       |
                    |       +-- drain_queue(request_queue, max_items)  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L24">[Drain:L24]</a>
                    |       |
                    |       +-- for each TextContext
                    |             |
                    |             +-- batch_constructor.enqueue_new_request(ctx)  <a href="0011_A_1_text_batch_constructor_request_admission.md#L113">[Enqueue:L113]</a>
                    |                   text_batch_constructor.py:677
                    |                     |
                    |                     +-- optional grammar gate submit/install
                    |                     |
                    |                     +-- _admit_request(ctx, replica_idx)  <a href="0011_A_1_text_batch_constructor_request_admission.md#L158">[Admit:L158]</a>
                    |                           |
                    |                           +-- optional DP CE pending pool
                    |                           |
                    |                           +-- _bind_request(...)  <a href="0011_A_1_text_batch_constructor_request_admission.md#L195">[Bind:L195]</a>
                    |                                 |
                    |                                 +-- generated_length == 0
                    |                                 |     -> replica.ce_reqs
                    |                                 |
                    |                                 +-- generated_length > 0
                    |                                       -> replica.tg_reqs
                    |
                    +-- [CALL] batch_constructor.construct_batch()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L84">[Batch:L84]</a>
                    |     text_batch_constructor.py:1845
                    |       |
                    |       +-- kv_cache.poll_transfers()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L87">[KV:L87]</a>
                    |       +-- _readmit_completed_onloads()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L89">[Onload:L89]</a>
                    |       +-- _promote_grammar_ready_requests()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L90">[Grammar:L90]</a>
                    |       +-- _plan_ce_step()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L97">[CEPlan:L97]</a>
                    |       +-- _should_backfill_ce()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L98">[Backfill:L98]</a>
                    |       +-- _identify_priority(...) for CE/TG priority  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L105">[Priority:L105]</a>
                    |       |
                    |       +-- for each replica
                    |       |     |
                    |       |     +-- _construct_replica_batch(...)  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L11">[Replica:L11]</a>
                    |       |           text_batch_constructor.py:1534
                    |       |             |
                    |       |             +-- create ReplicaBatch + _create_new_token_budget(...)  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L25">[Budget:L25]</a>
                    |       |             +-- _add_ce_requests(...), or  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L36">[CE:L36]</a>
                    |       |             +-- _add_tg_requests(...)  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L39">[TG:L39]</a>
                    |       |
                    |       +-- TextGenerationInputs(batches=...)
                    |       +-- optional _dp_padder.pad_batch(inputs)  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L144">[DPPad:L144]</a>
                    |       +-- return inputs
                    |
                    +-- [CALL] batch_constructor.take_grammar_failed()  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L74">[Failed:L74]</a>
                    |     |
                    |     +-- failed grammar requests
                    |           -> SchedulerResult.failed(error)
                    |           -> response_queue.put_nowait(...)
                    |
                    +-- [BRANCH] no inputs, no required empty batch,
                    |            and no overlap output pending
                    |     |
                    |     +-- return SchedulerProgress.NO_PROGRESS
                    |
                    +-- [CALL] _schedule(inputs)  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L163">[Schedule:L163]</a>
                          text_generation_scheduler.py
                            |
                            +-- batch_id = monotonic scheduler counter
                            |
                            +-- [CALL] pipeline.execute(inputs)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L76">[Execute:L76]</a>
                            |     TextGenerationPipeline.execute
                            |     text_generation.py:515
                            |       |
                            |       +-- request validator context
                            |       +-- [CALL] _execute(inputs)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L88">[Step:L88]</a>
                            |             text_generation.py:527
                            |               |
                            |               +-- [CALL] prepare_batch(inputs.batches)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L10">[Prepare:L10]</a>
                            |               |     text_generation.py:399
                            |               |       |
                            |               |       +-- _maybe_sort_loras(...) if needed  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L19">[LoRA:L19]</a>
                            |               |       +-- flatten2d(replica batches)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L21">[Flatten:L21]</a>
                            |               |       +-- initialize_bitmask(flat_batch)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L24">[Bitmask:L24]</a>
                            |               |       +-- update_for_structured_output(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L28">[Grammar:L28]</a>
                            |               |       +-- kv_manager.runtime_inputs(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L31">[KVInputs:L31]</a>
                            |               |       |
                            |               |       +-- [CALL]
                            |               |           _pipeline_model.prepare_initial_token_inputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L53">[Stage:L53]</a>
                            |               |             |
                            |               |             +-- LlamaModelBase.prepare_initial_token_inputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L53">[Stage:L53]</a>
                            |               |             |     llama3/model.py:168
                            |               |             |       |
                            |               |             |       +-- super().prepare_initial_token_inputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L53">[Stage:L53]</a>
                            |               |             |             pipeline_model.py:623
                            |               |             |               |
                            |               |             |               +-- batch_processor.prepare_initial_token_inputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L53">[Stage:L53]</a>
                            |               |             |                     llama3/batch_processor.py:99
                            |               |             |                       |
                            |               |             |                       +-- flatten contexts
                            |               |             |                       +-- _stage_ragged_token_inputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L9">[Ragged:L9]</a>
                            |               |             |                       |     +-- pack active token IDs
                            |               |             |                       |     +-- build row offsets
                            |               |             |                       +-- build return_n_logits buffer
                            |               |             |                       +-- optional DP splits
                            |               |             |                       +-- _make_inputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L115">[Inputs:L115]</a>
                            |               |             |                             llama3/batch_processor.py:166
                            |               |             |                               |
                            |               |             |                               +-- Llama3Inputs(...)
                            |               |             |
                            |               |             +-- return model_inputs
                            |               |
                            |               +-- FusedSamplingProcessor(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L107">[Sampler:L107]</a>
                            |               |     sampling_logits_processor.py:97
                            |               |
                            |               +-- [CALL] _launch_forward_pass(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L245">[Forward:L245]</a>
                            |               |     text_generation.py:655
                            |               |       |
                            |               |       +-- _pipeline_model.execute(model_inputs)  <a href="0014_A_1_llama_model_execution_runtime_and_graph.md#L80">[Model:L80]</a>
                            |               |             |
                            |               |             +-- LlamaModelBase.execute(...)  <a href="0014_A_1_llama_model_execution_runtime_and_graph.md#L80">[Model:L80]</a>
                            |               |                   llama3/model.py:183
                            |               |                     |
                            |               |                     +-- model_inputs.buffers
                            |               |                     |
                            |               |                     +-- [COMPILED]
                            |               |                     |     self.model.execute(*buffers)  <a href="0014_A_1_llama_model_execution_runtime_and_graph.md#L83">[Graph:L83]</a>
                            |               |                     |       |
                            |               |                     |       +-- compiled Llama graph
                            |               |                     |       +-- attention/KV/linear/norm ops
                            |               |                     |       +-- MAX runtime + device kernels
                            |               |                     |
                            |               |                     +-- batch_processor.process_outputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L151">[Outputs:L151]</a>
                            |               |                           llama3/batch_processor.py:153
                            |               |                             |
                            |               |                             +-- process_ragged_kv_outputs(...)  <a href="0013_A_1_llama_input_staging_ragged_buffers.md#L154">[RaggedOut:L154]</a>
                            |               |                             +-- ModelOutputs
                            |               |
                            |               +-- sampling_processor.logits_for_sampling(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L138">[Logits:L138]</a>
                            |               |
                            |               +-- apply_logits_processors(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L144">[LogitOps:L144]</a>
                            |               |     sampling/logits_processor.py:27
                            |               |       |
                            |               |       +-- request-specific processors
                            |               |       +-- FusedSamplingProcessor.__call__(...)
                            |               |             |
                            |               |             +-- _sample_logits(...)
                            |               |                   |
                            |               |                   +-- [COMPILED] sampler(*graph_inputs)
                            |               |                   +-- new token IDs
                            |               |
                            |               +-- copy generated tokens device -> host
                            |               |
                            |               +-- update_context_and_prepare_responses(...)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L211">[Update:L211]</a>
                            |               |     pipeline_variants/utils.py:157
                            |               |       |
                            |               |       +-- context.advance_token_buffer(...)
                            |               |       +-- context.advance_fsm(...)
                            |               |       +-- context.to_generation_output()
                            |               |       +-- dict[request_id, TextGenerationOutput]
                            |               |
                            |               +-- kv_manager.step(ctx)  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L219">[KVStep:L219]</a>
                            |               +-- return responses
                            |
                            +-- filter responses for released requests
                            |
                            +-- batch_constructor.advance_requests(inputs)  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L212">[Advance:L212]</a>
                            |     text_batch_constructor.py:808
                            |       |
                            |       +-- completed prefill: CE -> TG queue
                            |       +-- incomplete chunked prefill: back to CE queue
                            |
                            +-- release_request(...) when response.is_done  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L217">[Release:L217]</a>
                            |
                            +-- response_queue.put_nowait(  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L223">[Publish:L223]</a>
                                  request_id -> SchedulerResult(response, batch_id)
                                )
                                  |
                                  +-- [IPC] worker process -> API process
                    |
                    +-- scheduler_logger.log_metrics(...)  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L106">[Metrics:L106]</a>
                    |
                    +-- [CALL] get_cancelled_reqs(cancel_queue)  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L128">[Cancel:L128]</a>
                    |     |
                    |     +-- release matching request
                    |     +-- SchedulerResult.cancelled()
                    |     +-- response_queue.put_nowait(...)
                    |
                    +-- return SchedulerProgress.MADE_PROGRESS
</pre>

</details>

## 6. Response Path Back To The Client

The response follows the reverse queue path and resumes the suspended API-side
generators.

<details>
<summary><strong>Expand: Worker response back to the HTTP client</strong></summary>

<pre>
[IPC] SchedulerResult arrives on API response_queue
    |
    +-- [BACKGROUND] ZmqModelWorkerProxy.response_worker()  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L100">[Route:L100]</a>
    |     zmq_interface.py:247
    |       |
    |       +-- [AWAIT] response_queue.get()
    |       +-- locate pending_out_queues[request_id]
    |       +-- [AWAIT] per-request out_queue.put(response)
    |
    +-- ZmqModelWorkerProxy._drain_responses(...) resumes  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L126">[Drain:L126]</a>
    |     zmq_interface.py:167
    |       |
    |       +-- [AWAIT] out_queue.get()
    |       +-- coalesce already-buffered results
    |       +-- yield (outputs, batch_id)
    |       +-- stop when SchedulerResult.is_done
    |       +-- [BRANCH] consumer abandons/errors
    |       |     +-- send best-effort worker cancellation
    |       |     +-- re-raise
    |       +-- finally remove pending_out_queues[request_id]
    |
    +-- TokenGeneratorPipeline._generate(...) resumes  <a href="0007_A_1_create_buffered_detokenizer_factory_and_decode.md#L288">[Detok:L288]</a>
    |     llm.py, nested in next_token_chunk
    |       |
    |       +-- TextGenerationOutput.merge(responses)
    |       +-- optional reasoning_parser.stream(...)
    |       |
    |       +-- [AWAIT] content_detokenizer.decode(tokens)  <a href="0007_A_1_create_buffered_detokenizer_factory_and_decode.md#L17">[Decode:L17]</a>
    |       +-- [AWAIT] reasoning_detokenizer.decode(tokens)  <a href="0007_A_1_create_buffered_detokenizer_factory_and_decode.md#L17">[Decode:L17]</a>
    |       |
    |       +-- eos_tracker.is_eos_from_string(...)
    |       +-- optional logprob decoding
    |       +-- yield TokenGeneratorOutput(...)
    |
    +-- [BRANCH] streaming
    |     |
    |     +-- OpenAIChatResponseGenerator._stream(...) resumes
    |           |
    |           +-- optional parser.parse_delta(...) for tool calls
    |           +-- get_finish_reason_from_status(...)
    |           +-- ChatCompletionStreamResponseDelta(...)
    |           +-- ChatCompletionStreamResponseChoice(...)
    |           +-- CreateChatCompletionStreamResponse(...)
    |           +-- response.model_dump_json(exclude_none=True)
    |           +-- yield JSON SSE payload
    |           |
    |           +-- repeat sections 5 and 6 until done
    |           +-- optional final usage chunk
    |           +-- yield "[DONE]"
    |                 |
    |                 +-- EventSourceResponse formats SSE frame
    |                 +-- ASGI sends frame -> client
    |
    +-- [BRANCH] non-streaming
          |
          +-- pipeline.all_tokens(...) collects every TokenGeneratorOutput
          +-- OpenAIChatResponseGenerator.complete(...) aggregates them
          +-- CreateChatCompletionResponse(...)
          +-- FastAPI serializes JSON
                |
                +-- request_session adds X-Request-ID
                +-- ASGI sends JSON -> client
</pre>

</details>

## 7. Prefill And Decode Repeat Loop

One call to `scheduler.run_iteration()` normally performs one model step for a
selected batch. Generation therefore loops through the scheduler, pipeline,
model, sampler, and response queue multiple times.

<details>
<summary><strong>Expand: Prefill and decode repeat loop</strong></summary>

<pre>
new TextContext
    |
    +-- CE/prefill queue
          |
          +-- construct_batch()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L84">[Batch:L84]</a>
          +-- pipeline.execute()  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L76">[Execute:L76]</a>
          +-- model forward pass  <a href="0014_A_1_llama_model_execution_runtime_and_graph.md#L80">[Model:L80]</a>
          |
          +-- [BRANCH] prompt not fully processed
          |     |
          |     +-- context returns to CE queue
          |
          +-- [BRANCH] prefill complete
                |
                +-- context moves to TG/decode queue
                      |
                      +-- construct_batch()  <a href="0011_A_2_text_batch_constructor_batch_packing.md#L84">[Batch:L84]</a>
                      +-- pipeline.execute()  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L76">[Execute:L76]</a>
                      +-- sample next token
                      +-- update TextContext  <a href="0012_A_1_pipeline_execution_prepare_forward_sample.md#L211">[Update:L211]</a>
                      +-- send TextGenerationOutput
                      |
                      +-- [BRANCH] status ACTIVE
                      |     |
                      |     +-- remain in TG queue
                      |     +-- next scheduler iteration
                      |
                      +-- [BRANCH] EOS / maximum length / cancelled
                            |
                            +-- release request state and KV cache
                            +-- send terminal SchedulerResult
</pre>

</details>

## 8. Alternate Ingress And Configuration Branches

### SageMaker Ingress

<details>
<summary><strong>Expand: SageMaker ingress path</strong></summary>

<pre>
POST /invocations
    |
    +-- [FRAMEWORK] sagemaker_routes.invocations(request)
          sagemaker_routes.py
            |
            +-- [AWAIT] openai_create_chat_completion(request)  <a href="0001_A_1_parse_openai_request_body_http_ingress_and_parser.md#L20">[Ingress:L20]</a>
                  |
                  +-- same request path from section 2 onward
</pre>

</details>

### Prompt Tokens Already Supplied

<details>
<summary><strong>Expand: Pre-tokenized prompt branch</strong></summary>

<pre>
completion_request.prompt_tokens is non-empty
    |
    +-- TextGenerationRequest.prompt = prompt token IDs  <a href="0004_A_1_TextGenerationRequest_construction_and_validation.md#L15">[Request:L15]</a>
    +-- TextGenerationRequest.messages = []
    |
    +-- tokenizer._generate_prompt_and_token_ids(...)  <a href="0006_A_1_tokenizer_new_context_prompt_and_tokens.md#L95">[Prompt:L95]</a>
          |
          +-- encode(sequence[int])
                |
                +-- np.array(existing IDs)
                +-- no chat-template rendering
</pre>

</details>

### Scheduler/Pipeline Variants

<details>
<summary><strong>Expand: Scheduler and pipeline variants</strong></summary>

```text
pipeline_role == "prefill_and_decode" -> TokenGenerationScheduler
pipeline_role == "decode_only"        -> DecodeScheduler
pipeline_role == "prefill_only"       -> PrefillScheduler

enable_overlap_scheduler              -> OverlapTextGenerationPipeline path
speculative decoding enabled          -> speculative pipeline/context update
non-Llama architecture                -> architecture-specific batch processor/model
```

</details>

These alternatives share the API request parsing and worker IPC boundaries but
replace part of the worker-side execution tree.

## 9. Error Exits

The successful path above has several deliberate exits. These are part of the
runtime call stack because they decide whether the client receives JSON, HTTP
429, or an in-stream error.

<details>
<summary><strong>Expand: Error exit tree</strong></summary>

<pre>
request body / route preparation
    |
    +-- JSONDecodeError
    |     -> route catches it
    |     -> HTTPException(400, "Missing JSON.")
    |
    +-- KeyError or TypeError
    |     -> route catches it
    |     -> HTTPException(400, "Invalid JSON.")
    |
    +-- Pydantic ValidationError
    |     -> route catches it
    |     -> HTTPException(400, validation detail)
    |
    +-- InputError / UndefinedError / ValueError
    |     -> route catches it
    |     -> HTTP 400
    |
    +-- RequestQueueFull from ZmqModelWorkerProxy.stream(...)  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L34">[IPC:L34]</a>
    |     -> not consumed by the route's validation catches
    |     -> app-level _request_queue_full_exception_handler(...)
    |     -> HTTP 429 + Retry-After: 1
    |
    +-- streaming error before the first visible SSE payload
    |     -> OpenAIChatResponseGenerator._stream yields JSONResponse
    |     -> _start_stream(...) returns that real HTTP error response  <a href="0005_A_1_streaming_vs_non_streaming_response_dispatch.md#L87">[Gate:L87]</a>
    |
    +-- streaming error after SSE output has started
          -> HTTP 200 is already committed
          -> stream yields an OpenAI error JSON frame
</pre>

</details>

## 10. Boundaries That Are Not Direct Calls

These distinctions prevent a call-stack diagram from implying false Python
calls:

<details>
<summary><strong>Expand: IPC and compiled-runtime boundaries</strong></summary>

<pre>
API process                    Model worker process
-----------                    --------------------
request_queue.put(context)  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L61">[Push:L61]</a> --> scheduler drains request_queue
                               not a direct call to run_iteration()  <a href="0010_A_1_scheduler_iteration_queue_batch_execute.md#L64">[Iter:L64]</a>

scheduler response_queue   --> proxy.response_worker()  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L100">[Route:L100]</a>
                               not a direct call to _drain_responses()  <a href="0009_A_1_model_worker_stream_admission_and_response_drain.md#L126">[Drain:L126]</a>

LlamaModelBase.execute()  <a href="0014_A_1_llama_model_execution_runtime_and_graph.md#L80">[Model:L80]</a>
  -> self.model.execute(...)   crosses into compiled MAX runtime  <a href="0014_A_1_llama_model_execution_runtime_and_graph.md#L83">[Graph:L83]</a>
                               Python does not recursively call Mojo kernel files
</pre>

</details>

The compiled Llama graph was built during worker startup. At request time,
`self.model.execute(*model_inputs.buffers)` invokes that already-compiled graph.
Graph operators eventually dispatch lower-level kernels, but those kernel
invocations are not represented as ordinary Python frames.

## 11. Source Map

- API application and route registration:
  [`api_server.py`](../../../max/python/max/serve/api_server.py#L330)
- Request ID/timer HTTP middleware:
  [`request.py`](../../../max/python/max/serve/request.py#L45)
- OpenAI route and response generation:
  [`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L2217)
- OpenAI request schemas:
  [`schemas/openai.py`](../../../max/python/max/serve/schemas/openai.py#L448)
- Tool parameter normalization:
  [`tool_call_normalization.py`](../../../max/python/max/serve/parser/tool_call_normalization.py#L62)
- Media resolution:
  [`_image_resolution.py`](../../../max/python/max/serve/router/_image_resolution.py#L839)
- Internal text-generation request:
  [`modeling/types/.../text_generation.py`](../../../max/python/max/pipelines/modeling/types/pipeline_variants/text_generation.py#L336)
- API-side LLM pipeline:
  [`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L294)
- Tokenizer and context creation:
  [`tokenizer.py`](../../../max/python/max/pipelines/lib/tokenizer.py#L718)
- Buffered detokenization:
  [`incremental_detokenizer.py`](../../../max/python/max/serve/pipelines/incremental_detokenizer.py#L343)
- ZMQ worker interface:
  [`zmq_interface.py`](../../../max/python/max/serve/worker_interface/zmq_interface.py#L105)
- Model-worker process loop:
  [`model_worker.py`](../../../max/python/max/serve/pipelines/model_worker.py#L232)
- Scheduler:
  [`text_generation_scheduler.py`](../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L205)
- Scheduler selection:
  [`scheduler/__init__.py`](../../../max/python/max/serve/scheduler/__init__.py#L61)
- Batch constructor:
  [`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L677)
- Text-generation pipeline:
  [`pipeline_variants/text_generation.py`](../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L399)
- Sampling:
  [`sampling_logits_processor.py`](../../../max/python/max/pipelines/sampling/sampling_logits_processor.py#L97)
- Per-request and fused logits-processor dispatch:
  [`logits_processor.py`](../../../max/python/max/pipelines/sampling/logits_processor.py#L27)
- Context update and response construction:
  [`pipeline_variants/utils.py`](../../../max/python/max/pipelines/lib/pipeline_variants/utils.py#L157)
- Pipeline-model delegation to the architecture batch processor:
  [`pipeline_model.py`](../../../max/python/max/pipelines/lib/interfaces/pipeline_model.py#L623)
- Llama input staging:
  [`llama3/batch_processor.py`](../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L99)
- Llama model runtime wrapper:
  [`llama3/model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L168)

## 12. Walkthrough Cross-References

The following documents provide source excerpts, worked values, and focused
explanations for nodes in the call tree:

- Folder map:
  [0000_A_folder_structure.md](0000_A_folder_structure.md)
- HTTP parsing:
  [0001_parse_openai_request_body.md](0001_parse_openai_request_body.md),
  [0001_A_1_parse_openai_request_body_http_ingress_and_parser.md](0001_A_1_parse_openai_request_body_http_ingress_and_parser.md),
  [0001_A_2_parse_openai_request_body_tool_parameter_normalization.md](0001_A_2_parse_openai_request_body_tool_parameter_normalization.md),
  [0001_B_1_parse_openai_request_body_pydantic_schema_validation.md](0001_B_1_parse_openai_request_body_pydantic_schema_validation.md)
- Pipeline selection and tokenizer contract:
  [0002_get_pipeline.md](0002_get_pipeline.md),
  [0002_get_pipeline_0a_PipelineTokenizer.md](0002_get_pipeline_0a_PipelineTokenizer.md),
  [0002_A_1_get_pipeline_model_selection.md](0002_A_1_get_pipeline_model_selection.md),
  [0002_B_1_get_pipeline_pipeline_tokenizer_protocol.md](0002_B_1_get_pipeline_pipeline_tokenizer_protocol.md)
- Chat parsing and media:
  [0003_openai_parse_chat_completion_request.md](0003_openai_parse_chat_completion_request.md),
  [0003_A_1_openai_parse_chat_completion_request_messages.md](0003_A_1_openai_parse_chat_completion_request_messages.md),
  [0003_A_2_openai_parse_chat_completion_request_media.md](0003_A_2_openai_parse_chat_completion_request_media.md),
  [0003_B_1_openai_parse_chat_completion_request_tool_conversion.md](0003_B_1_openai_parse_chat_completion_request_tool_conversion.md)
- Internal request and HTTP response branches:
  [0004_TextGenerationRequest.md](0004_TextGenerationRequest.md),
  [0004_A_1_TextGenerationRequest_construction_and_validation.md](0004_A_1_TextGenerationRequest_construction_and_validation.md),
  [0005_streaming_vs_non_streaming.md](0005_streaming_vs_non_streaming.md),
  [0005_A_1_streaming_vs_non_streaming_response_dispatch.md](0005_A_1_streaming_vs_non_streaming_response_dispatch.md)
- Context creation and output decoding:
  [0006_tokenizer_new_context.md](0006_tokenizer_new_context.md),
  [0006_A_1_tokenizer_new_context_prompt_and_tokens.md](0006_A_1_tokenizer_new_context_prompt_and_tokens.md),
  [0006_A_2_tokenizer_new_context_metadata_and_state.md](0006_A_2_tokenizer_new_context_metadata_and_state.md),
  [0007_create_buffered_detokenizer.md](0007_create_buffered_detokenizer.md),
  [0007_A_1_create_buffered_detokenizer_factory_and_decode.md](0007_A_1_create_buffered_detokenizer_factory_and_decode.md)
- Admission and worker IPC:
  [0008_note_awaiting_admission.md](0008_note_awaiting_admission.md),
  [0008_A_1_note_awaiting_admission_counter_lifecycle.md](0008_A_1_note_awaiting_admission_counter_lifecycle.md),
  [0009_model_worker_stream.md](0009_model_worker_stream.md),
  [0009_A_1_model_worker_stream_admission_and_response_drain.md](0009_A_1_model_worker_stream_admission_and_response_drain.md)
- Scheduler and batch construction:
  [0010_scheduler_iteration.md](0010_scheduler_iteration.md),
  [0010_A_1_scheduler_iteration_queue_batch_execute.md](0010_A_1_scheduler_iteration_queue_batch_execute.md),
  [0011_text_batch_constructor.md](0011_text_batch_constructor.md),
  [0011_A_1_text_batch_constructor_request_admission.md](0011_A_1_text_batch_constructor_request_admission.md),
  [0011_A_2_text_batch_constructor_batch_packing.md](0011_A_2_text_batch_constructor_batch_packing.md)
- Pipeline and Llama execution:
  [0012_pipeline_execution.md](0012_pipeline_execution.md),
  [0012_A_1_pipeline_execution_prepare_forward_sample.md](0012_A_1_pipeline_execution_prepare_forward_sample.md),
  [0013_llama_input_staging.md](0013_llama_input_staging.md),
  [0013_A_1_llama_input_staging_ragged_buffers.md](0013_A_1_llama_input_staging_ragged_buffers.md),
  [0014_llama_model_execution.md](0014_llama_model_execution.md),
  [0014_A_1_llama_model_execution_runtime_and_graph.md](0014_A_1_llama_model_execution_runtime_and_graph.md)
