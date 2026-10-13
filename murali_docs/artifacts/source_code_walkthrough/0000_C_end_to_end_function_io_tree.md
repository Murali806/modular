# End-to-End Function I/O Tree: OpenAI Chat Request To Llama And Back

Companion to
[0000_B_end_to_end_call_stack.md](0000_B_end_to_end_call_stack.md).

Every callable or meaningful runtime step keeps the same execution-order tree
shape and adds four compact branches:

```text
call_or_step
  |
  +-- IN
  +-- STRUCTURES
  +-- LOGIC
  +-- OUT
```

```text
[CONCRETE]     complete value fixed by this request/config fixture
[ILLUSTRATIVE] complete example value whose contents depend on the loaded model/runtime
[RUNTIME]      exact structure shown; storage or large tensor contents exist only at runtime

Tokenizer IDs marked concrete below were generated from:
max/tests/integration/architectures/llama3/testdata/tokenizer.json
EOS IDs use the companion testdata/config.json.
```

## 0. Concrete Request And Data-Structure Spine

<details>
<summary><strong>Expand: Concrete request values used throughout the trees</strong></summary>

<pre>
HTTP request
  |
  +-- IN
  |     method = "POST"
  |     path = "/v1/chat/completions"
  |     content-type = "application/json"
  |     body = {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "messages": [
  |         {"role": "system", "content": "You are a concise assistant."},
  |         {"role": "user", "content": "Explain KV cache in one sentence."}
  |       ],
  |       "temperature": 0.2,
  |       "top_k": 50,
  |       "top_p": 0.9,
  |       "max_completion_tokens": 32,
  |       "seed": 42,
  |       "stop": "\nUser:",
  |       "stream": true,
  |       "stream_options": {"include_usage": true}
  |     }
  |
  +-- STRUCTURES
  |     Request.state.request_id = "req-chat-001"
  |     Request.state.request_timer = StopWatch(start_ns=1730000000000000000)
  |
  +-- LOGIC
  |     HTTP JSON -> typed OpenAI request -> internal generation request
  |
  +-- OUT
        one streamed chat-completion response

CreateChatCompletionRequest
  |
  +-- IN
  |     parsed JSON dict above
  +-- STRUCTURES
  |     model = "meta-llama/Llama-3.1-8B-Instruct"
  |     messages = [system message, user message]
  |     temperature = 0.2
  |     top_k = 50
  |     top_p = 0.9
  |     max_completion_tokens = 32
  |     seed = 42
  |     stop = "\nUser:"
  |     stream = True
  |     stream_options = {"include_usage": True}
  |     tools = None
  |     response_format = None
  |     prompt_tokens = None
  +-- LOGIC
  |     Pydantic validates OpenAI fields and MAX extensions
  +-- OUT
        completion_request: CreateChatCompletionRequest

_ParsedChatRequest
  |
  +-- IN
  |     completion_request.messages
  +-- STRUCTURES
  |     messages = [
  |       TextGenerationRequestMessage(
  |         role="system", content="You are a concise assistant."
  |       ),
  |       TextGenerationRequestMessage(
  |         role="user", content="Explain KV cache in one sentence."
  |       )
  |     ]
  |     request_images = []
  |     request_videos = []
  |     decoded_images = []
  +-- LOGIC
  |     normalize roles/content; resolve no media
  +-- OUT
        request_messages + empty media lists

SamplingParams
  |
  +-- IN
  |     temperature=0.2, top_k=50, top_p=0.9,
  |     max_new_tokens=32, stop=["\nUser:"], seed=42
  +-- STRUCTURES
  |     SamplingParams(
  |       top_k=50,
  |       top_p=0.9,
  |       min_p=0.0,
  |       temperature=0.2,
  |       thinking_temperature=None,
  |       frequency_penalty=0.0,
  |       presence_penalty=0.0,
  |       repetition_penalty=1.0,
  |       max_new_tokens=32,
  |       min_new_tokens=0,
  |       ignore_eos=False,
  |       stop=["\nUser:"],
  |       stop_token_ids=None,
  |       detokenize=True,
  |       seed=42,
  |       logits_processors=None
  |     )
  +-- LOGIC
  |     request values override server/model defaults
  +-- OUT
        validated sampling_params

TextGenerationRequest
  |
  +-- IN
  |     request_id="req-chat-001"
  |     model/messages/sampling values above
  +-- STRUCTURES
  |     prompt=None
  |     messages=[typed system message, typed user message]
  |     images=[]; videos=[]; tools=None; response_format=None
  |     logprobs=0; request_path="/v1/chat/completions"
  +-- LOGIC
  |     reject prompt+messages conflict; verify media placeholders
  +-- OUT
        token_request: TextGenerationRequest

TextContext
  |
  +-- IN
  |     token_request + rendered/tokenized prompt
  +-- STRUCTURES
  |     request_id=RequestID("req-chat-001")
  |     tokens=TokenBuffer(
  |       prompt=[128000,128006,9125,128007,271,38766,1303,33025,2696,25,
  |               6790,220,2366,18,198,15724,2696,25,220,1627,10263,220,
  |               2366,19,271,2675,527,264,64694,18328,13,128009,128006,
  |               882,128007,271,849,21435,85748,6636,304,832,11914,13,
  |               128009,128006,78191,128007,271],
  |       generated=[]
  |     )
  |     eos_tracker=EOSTracker(
  |       eos_token_ids={128001,128008,128009},
  |       eos_sequences=[[198,1502,25]],
  |       eos_stop_strings=["\nUser:"]
  |     )
  |     max_length=81
  |     vocab_size=128256
  |     log_probabilities=0
  |     log_probabilities_echo=False
  |     json_schema=None
  |     grammar=None
  |     grammar_state=GrammarEnforcementState(
  |       grammar_enforced=False,
  |       tools_forced=False,
  |       requires_structured_output_flag=False,
  |       has_json_schema=False
  |     )
  |     sampling_params=SamplingParams(
  |       top_k=50,
  |       top_p=0.9,
  |       min_p=0.0,
  |       temperature=0.2,
  |       thinking_temperature=None,
  |       frequency_penalty=0.0,
  |       presence_penalty=0.0,
  |       repetition_penalty=1.0,
  |       max_new_tokens=32,
  |       min_new_tokens=0,
  |       ignore_eos=False,
  |       stop=["\nUser:"],
  |       stop_token_ids=None,
  |       detokenize=True,
  |       seed=42,
  |       logits_processors=None
  |     )
  |     model_name="meta-llama/Llama-3.1-8B-Instruct"
  |     target_endpoint=None
  |     dkv_cache_hint=None
  |     cache_salt=None
  +-- LOGIC
  |     token IDs are illustrative and tokenizer-specific
  +-- OUT
        context ready for worker admission

Worker/model structures
  |
  +-- IN
  |     context
  +-- STRUCTURES
  |     TextGenerationInputs(
  |       batches=[[context]],
  |       input_tokens=49,
  |       batch_type=BatchType.CE,
  |       per_replica_input_tokens=[49],
  |       per_replica_context_tokens=[0]
  |     )
  |     Llama3Inputs(
  |       tokens=Buffer(shape=(49,), dtype=int64),
  |       input_row_offsets=Buffer([0,49]),
  |       signal_buffers=[],
  |       return_n_logits=Buffer([1]),
  |       data_parallel_splits=None,
  |       kv_cache_inputs=KVCacheInputsPerDevice(
  |         kv_blocks=Buffer(device="gpu:0", values=[RUNTIME]),
  |         cache_lengths=Buffer([0]),
  |         lookup_table=Buffer([[7]]),
  |         max_prompt_length=Buffer([49]),
  |         max_cache_length=Buffer([49]),
  |         page_stride=Buffer([RUNTIME])
  |       )
  |     )
  |     ModelOutputs(
  |       logits=Buffer(shape=(1,128256), dtype=float32),
  |       next_token_logits=None,
  |       logit_offsets=None,
  |       hidden_states=None
  |     )
  |     TextGenerationOutput(
  |       request_id=RequestID("req-chat-001"),
  |       tokens=[83807],
  |       final_status=ACTIVE,
  |       log_probabilities=None,
  |       num_cached_tokens=0
  |     )
  |     SchedulerResult(
  |       is_done=False,
  |       result=TextGenerationOutput(
  |         request_id=RequestID("req-chat-001"),
  |         tokens=[83807],
  |         final_status=ACTIVE,
  |         log_probabilities=None,
  |         num_cached_tokens=0
  |       ),
  |       error=None,
  |       batch_id=1
  |     )
  +-- LOGIC
  |     illustrative generated token 83807 -> decoded text fragment "KV"
  |     illustrative request assumes no prefix-cache hit
  +-- OUT
        TokenGeneratorOutput(
          status=ACTIVE,
          decoded_tokens="KV",
          decoded_reasoning_tokens=None,
          token_count=1,
          token_log_probabilities=None,
          top_log_probabilities=None,
          sampled_tokens=None,
          prompt_token_count=49,
          cached_token_count=0,
          reasoning_token_count=0,
          stop_sequence=None,
          batch_id=1,
          token_ids=[83807],
          prompt_token_ids=[128000,128006,9125,128007,271,38766,1303,33025,
                            2696,25,6790,220,2366,18,198,15724,2696,25,220,
                            1627,10263,220,2366,19,271,2675,527,264,64694,
                            18328,13,128009,128006,882,128007,271,849,21435,
                            85748,6636,304,832,11914,13,128009,128006,78191,
                            128007,271]
        )
</pre>

</details>

## 0.1 Actual HTTP, JSON, Pydantic, And MAX Request Values

<details>
<summary><strong>Expand: Complete normal-text request transformations</strong></summary>

<pre>
[CONCRETE] HTTP wire request
  |
  +-- IN
  |     POST /v1/chat/completions HTTP/1.1
  |     Host: localhost:8000
  |     Content-Type: application/json
  |
  |     {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "messages": [
  |         {
  |           "role": "system",
  |           "content": "You are a concise assistant."
  |         },
  |         {
  |           "role": "user",
  |           "content": "Explain KV cache in one sentence."
  |         }
  |       ],
  |       "temperature": 0.2,
  |       "top_k": 50,
  |       "top_p": 0.9,
  |       "max_completion_tokens": 32,
  |       "seed": 42,
  |       "stop": "\nUser:",
  |       "stream": true,
  |       "stream_options": {
  |         "include_usage": true
  |       }
  |     }
  |
  +-- STRUCTURES
  |     HTTP method/path/headers + UTF-8 JSON body
  +-- LOGIC
  |     Starlette stores the body until request.body() is awaited
  +-- OUT
        request: Request

request.body()
  |
  +-- IN
  |     request receive channel
  +-- STRUCTURES
  |     raw: bytes
  +-- LOGIC
  |     concatenate HTTP body chunks and cache the result
  +-- OUT
        raw = b'{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"system","content":"You are a concise assistant."},{"role":"user","content":"Explain KV cache in one sentence."}],"temperature":0.2,"top_k":50,"top_p":0.9,"max_completion_tokens":32,"seed":42,"stop":"\\nUser:","stream":true,"stream_options":{"include_usage":true}}'

json.loads(raw)
  |
  +-- IN
  |     raw bytes above
  +-- STRUCTURES
  |     parsed: dict[str, object]
  +-- LOGIC
  |     decode UTF-8 JSON into Python values
  +-- OUT
        parsed = {
          "model": "meta-llama/Llama-3.1-8B-Instruct",
          "messages": [
            {
              "role": "system",
              "content": "You are a concise assistant.",
            },
            {
              "role": "user",
              "content": "Explain KV cache in one sentence.",
            },
          ],
          "temperature": 0.2,
          "top_k": 50,
          "top_p": 0.9,
          "max_completion_tokens": 32,
          "seed": 42,
          "stop": "\nUser:",
          "stream": True,
          "stream_options": {"include_usage": True},
        }

CreateChatCompletionRequest.model_validate(parsed)
  |
  +-- IN
  |     parsed dict above
  +-- STRUCTURES
  |     CreateChatCompletionRequest: Pydantic model
  +-- LOGIC
  |     validate OpenAI fields + MAX extensions + model validators
  +-- OUT
        completion_request.model_dump(
          exclude_none=True,
          exclude_defaults=True,
        ) = {
          "model": "meta-llama/Llama-3.1-8B-Instruct",
          "messages": [
            {
              "role": "system",
              "content": "You are a concise assistant.",
            },
            {
              "role": "user",
              "content": "Explain KV cache in one sentence.",
            },
          ],
          "temperature": 0.2,
          "top_k": 50,
          "top_p": 0.9,
          "max_completion_tokens": 32,
          "seed": 42,
          "stop": "\nUser:",
          "stream": True,
          "stream_options": {"include_usage": True},
        }

openai_parse_chat_completion_request(completion_request, ...)
  |
  +-- IN
  |     completion_request.messages shown above
  +-- STRUCTURES
  |     _ParsedChatRequest: NamedTuple
  +-- LOGIC
  |     validate roles/tool history; create internal message models
  +-- OUT
        _ParsedChatRequest(
          messages=[
            TextGenerationRequestMessage(
              role="system",
              content="You are a concise assistant.",
              tool_calls=None,
              tool_call_id=None,
              reasoning_content=None,
            ),
            TextGenerationRequestMessage(
              role="user",
              content="Explain KV cache in one sentence.",
              tool_calls=None,
              tool_call_id=None,
              reasoning_content=None,
            ),
          ],
          request_images=[],
          request_videos=[],
          decoded_images=[],
        )

SamplingParamsInput(...)
  |
  +-- IN
  |     request fields + server defaults
  +-- STRUCTURES
  |     partial sampling input dataclass
  +-- LOGIC
  |     route maps max_completion_tokens -> max_new_tokens
  |     route maps stop string -> one-item stop list
  +-- OUT
        SamplingParamsInput(
          top_k=50,
          top_p=0.9,
          min_p=None,
          temperature=0.2,
          thinking_temperature=None,
          frequency_penalty=None,
          presence_penalty=None,
          repetition_penalty=None,
          max_new_tokens=32,
          min_new_tokens=None,
          ignore_eos=None,
          stop=["\nUser:"],
          stop_token_ids=None,
          detokenize=None,
          seed=42,
          logits_processors=None,
        )

SamplingParams.from_input_and_generation_config(...)
  |
  +-- IN
  |     SamplingParamsInput above
  |     SamplingParamsGenerationConfigDefaults(
  |       temperature=None,
  |       top_p=None,
  |       top_k=None,
  |       repetition_penalty=None,
  |       max_new_tokens=None,
  |       min_new_tokens=None,
  |       do_sample=None,
  |       eos_token_id=None,
  |     )
  +-- STRUCTURES
  |     final runtime SamplingParams dataclass
  +-- LOGIC
  |     request non-None values -> model explicit defaults -> class defaults
  +-- OUT
        SamplingParams(
          top_k=50,
          top_p=0.9,
          min_p=0.0,
          temperature=0.2,
          thinking_temperature=None,
          frequency_penalty=0.0,
          presence_penalty=0.0,
          repetition_penalty=1.0,
          max_new_tokens=32,
          min_new_tokens=0,
          ignore_eos=False,
          stop=["\nUser:"],
          stop_token_ids=None,
          detokenize=True,
          seed=42,
          logits_processors=None,
        )

TextGenerationRequest(...)
  |
  +-- IN
  |     route/parser/sampling outputs above
  +-- STRUCTURES
  |     immutable MAX request dataclass
  +-- LOGIC
  |     remove HTTP-only response mode; preserve model-generation inputs
  +-- OUT
        TextGenerationRequest(
          request_id=RequestID("req-chat-001"),
          model_name="meta-llama/Llama-3.1-8B-Instruct",
          prompt=None,
          messages=[
            TextGenerationRequestMessage(
              role="system",
              content="You are a concise assistant.",
              tool_calls=None,
              tool_call_id=None,
              reasoning_content=None,
            ),
            TextGenerationRequestMessage(
              role="user",
              content="Explain KV cache in one sentence.",
              tool_calls=None,
              tool_call_id=None,
              reasoning_content=None,
            ),
          ],
          images=[],
          videos=[],
          decoded_images=[],
          tools=None,
          response_format=None,
          timestamp_ns=1730000000000000000,
          request_path="/v1/chat/completions",
          logprobs=0,
          echo=False,
          chat_template_options=None,
          sampling_params=SamplingParams(
            top_k=50,
            top_p=0.9,
            min_p=0.0,
            temperature=0.2,
            thinking_temperature=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=32,
            min_new_tokens=0,
            ignore_eos=False,
            stop=["\nUser:"],
            stop_token_ids=None,
            detokenize=True,
            seed=42,
            logits_processors=None,
          ),
          target_endpoint=None,
          dkv_cache_hint=None,
          cache_salt=None,
        )
</pre>

</details>

## 0.2 Actual Prompt Inputs, Rendered Prompt, And Token Buffer

<details>
<summary><strong>Expand: Message, string-prompt, and token-ID prompt forms</strong></summary>

<pre>
[CONCRETE] CHAT MESSAGES BRANCH
  |
  +-- IN
  |     request.prompt = None
  |     request.messages = [
  |       TextGenerationRequestMessage(
  |         role="system",
  |         content="You are a concise assistant.",
  |         tool_calls=None,
  |         tool_call_id=None,
  |         reasoning_content=None,
  |       ),
  |       TextGenerationRequestMessage(
  |         role="user",
  |         content="Explain KV cache in one sentence.",
  |         tool_calls=None,
  |         tool_call_id=None,
  |         reasoning_content=None,
  |       ),
  |     ]
  |     request.tools = None
  |     request.chat_template_options = None
  +-- STRUCTURES
  |     list[TextGenerationRequestMessage]
  +-- LOGIC
  |     flatten messages -> apply model chat template -> tokenize
  +-- OUT
        flattened_messages = [
          {
            "role": "system",
            "content": "You are a concise assistant.",
          },
          {
            "role": "user",
            "content": "Explain KV cache in one sentence.",
          },
        ]
        |
        v
        [CONCRETE model-specific Llama prompt]
        rendered_prompt =
          "&lt;|begin_of_text|&gt;"
          "&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;\n\n"
          "Cutting Knowledge Date: December 2023\n"
          "Today Date: 26 Jul 2024\n\n"
          "You are a concise assistant.&lt;|eot_id|&gt;"
          "&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;\n\n"
          "Explain KV cache in one sentence.&lt;|eot_id|&gt;"
          "&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n"
        |
        v
        [CONCRETE repository Llama 3 tokenizer-fixture output]
        token_ids = np.array([
          128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,
          220,2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,
          2675,527,264,64694,18328,13,128009,128006,882,128007,271,849,
          21435,85748,6636,304,832,11914,13,128009,128006,78191,128007,271
        ], dtype=np.int64)

        token_pieces = [
          "<|begin_of_text|>", "<|start_header_id|>", "system",
          "<|end_header_id|>", "\n\n", "Cut", "ting", " Knowledge", " Date",
          ":", " December", " ", "202", "3", "\n", "Today", " Date", ":",
          " ", "26", " Jul", " ", "202", "4", "\n\n", "You", " are", " a",
          " concise", " assistant", ".", "<|eot_id|>",
          "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n", "Ex",
          "plain", " KV", " cache", " in", " one", " sentence", ".", "<|eot_id|>",
          "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n",
        ]

_generate_prompt_and_token_ids(...)
  |
  +-- IN
  |     prompt=None + messages/tools/options above
  +-- STRUCTURES
  |     tuple[str, np.ndarray[int64]]
  +-- LOGIC
  |     choose messages branch; render then encode
  +-- OUT
        (
          rendered_prompt,
          np.array([
            128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,
            220,2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,
            2675,527,264,64694,18328,13,128009,128006,882,128007,271,849,
            21435,85748,6636,304,832,11914,13,128009,128006,78191,128007,271
          ], dtype=np.int64),
        )

TokenBuffer(token_ids)
  |
  +-- IN
  |     token_ids: np.ndarray(shape=(49,), dtype=int64)
  +-- STRUCTURES
  |     one backing array + processing/completion ranges
  +-- LOGIC
  |     prompt is initially active; no generated token is ready to stream
  +-- OUT
        TokenBuffer(
          array=np.array([
            128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,
            220,2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,
            2675,527,264,64694,18328,13,128009,128006,882,128007,271,849,
            21435,85748,6636,304,832,11914,13,128009,128006,78191,128007,271
          ], dtype=np.int64),
          _processing_range=Range(start=0,end=49),
          _processing_offset=0,
          _completion_range=Range(start=49,end=49),
          _prompt_length=49,
          _current_length=49,
          _actively_chunked=False,
        )

[CONCRETE] LEGACY COMPLETION STRING-PROMPT BRANCH
  |
  +-- IN
  |     POST /v1/completions
  |     {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "prompt": "Explain KV cache directly.",
  |       "temperature": 0.2,
  |       "top_k": 50,
  |       "top_p": 0.9,
  |       "max_tokens": 32,
  |       "seed": 42,
  |       "stream": false
  |     }
  |     get_prompts_from_openai_request(...) = ["Explain KV cache directly."]
  |     request.prompt = "Explain KV cache directly."
  |     request.messages = []
  +-- STRUCTURES
  |     str prompt
  +-- LOGIC
  |     skip chat-template rendering; call encode(prompt, add_special_tokens=True)
  +-- OUT
        token_request.request_id = RequestID("req-completion-001_0")
        token_request.prompt = "Explain KV cache directly."
        token_request.messages = []
        selected_prompt = "Explain KV cache directly."
        [CONCRETE repository Llama 3 tokenizer-fixture output]
        token_pieces = [
          "<|begin_of_text|>", "Ex", "plain", " KV", " cache", " directly", "."
        ]
        token_ids = np.array(
          [128000,849,21435,85748,6636,6089,13],
          dtype=np.int64,
        )

[CONCRETE] LEGACY COMPLETION PRE-TOKENIZED PROMPT BRANCH
  |
  +-- IN
  |     POST /v1/completions
  |     {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "prompt": [128000,128006,882,128007,271],
  |       "temperature": 0.2,
  |       "max_tokens": 32,
  |       "seed": 42,
  |       "stream": false
  |     }
  |     get_prompts_from_openai_request(...) = [
  |       [128000,128006,882,128007,271]
  |     ]
  |     request.prompt = [128000,128006,882,128007,271]
  |     request.messages = []
  +-- STRUCTURES
  |     Sequence[int]
  +-- LOGIC
  |     preserve supplied IDs; skip text tokenization and chat template
  +-- OUT
        token_request.request_id = RequestID("req-completion-002_0")
        token_request.prompt = [128000,128006,882,128007,271]
        token_request.messages = []
        selected_prompt = [128000,128006,882,128007,271]
        token_ids = np.array(
          [128000,128006,882,128007,271],
          dtype=np.int64,
        )
</pre>

</details>

## 0.3 Actual Tool Declaration, Tool Call, And Tool Result Values

<details>
<summary><strong>Expand: Complete tool-call request and response formats</strong></summary>

<pre>
[CONCRETE] TOOL-CALL HTTP REQUEST BODY
  |
  +-- IN
  |     {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "messages": [
  |         {
  |           "role": "user",
  |           "content": "What is the weather in Colombo?"
  |         }
  |       ],
  |       "tools": [
  |         {
  |           "type": "function",
  |           "function": {
  |             "name": "get_weather",
  |             "description": "Return the current weather for a city.",
  |             "parameters": {
  |               "type": "object",
  |               "properties": {
  |                 "city": {
  |                   "type": "string",
  |                   "description": "City name, for example Colombo."
  |                 },
  |                 "unit": {
  |                   "type": "string",
  |                   "enum": ["celsius", "fahrenheit"]
  |                 }
  |               },
  |               "required": ["city"],
  |               "additionalProperties": false
  |             },
  |             "strict": true
  |           }
  |         }
  |       ],
  |       "tool_choice": "auto",
  |       "temperature": 0,
  |       "top_k": 1,
  |       "top_p": 1.0,
  |       "max_completion_tokens": 64,
  |       "seed": 7,
  |       "stream": false
  |     }
  +-- STRUCTURES
  |     OpenAI ChatCompletionFunctionToolParam list
  +-- LOGIC
  |     parse tools -> normalize parameters -> convert to MAX tool dictionaries
  +-- OUT
        completion_request.tools = [
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
          },
        ]

_normalize_tools_parameters(completion_request.tools)
  |
  +-- IN
  |     complete tool dictionary above
  +-- STRUCTURES
  |     copied outer tool + function + parameters dictionaries
  +-- LOGIC
  |     parameters is already a dict; copy it unchanged
  +-- OUT
        normalized_tools = [
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
          },
        ]

_convert_chat_completion_tools_to_token_generator_tools(...)
  |
  +-- IN
  |     chat_tools = normalized_tools above
  |     valid_tool_name_re = re.compile(r"^[a-zA-Z0-9_-]+$")
  |     policy = DEFAULT_STRICT_FALSE_AND_BEST_EFFORT
  +-- STRUCTURES
  |     TextGenerationRequestTool + TextGenerationRequestFunction TypedDicts
  +-- LOGIC
  |     validate name; copy schema; resolve requested strict=True -> True
  +-- OUT
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
          },
        ]

TextGenerationRequest for tool use
  |
  +-- IN
  |     request_messages = [
  |       TextGenerationRequestMessage(
  |         role="user",
  |         content="What is the weather in Colombo?",
  |         tool_calls=None,
  |         tool_call_id=None,
  |         reasoning_content=None,
  |       )
  |     ]
  |     tools = token_generator_tools above
  +-- STRUCTURES
  |     TextGenerationRequest
  +-- LOGIC
  |     preserve tool menu for chat-template rendering and output parsing
  +-- OUT
        TextGenerationRequest(
          request_id=RequestID("req-tools-001"),
          model_name="meta-llama/Llama-3.1-8B-Instruct",
          prompt=None,
          messages=[
            TextGenerationRequestMessage(
              role="user",
              content="What is the weather in Colombo?",
              tool_calls=None,
              tool_call_id=None,
              reasoning_content=None,
            )
          ],
          images=[],
          videos=[],
          decoded_images=[],
          tools=token_generator_tools,
          response_format=None,
          timestamp_ns=1730000001000000000,
          request_path="/v1/chat/completions",
          logprobs=0,
          echo=False,
          chat_template_options=None,
          sampling_params=SamplingParams(
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            temperature=0.0,
            thinking_temperature=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=64,
            min_new_tokens=0,
            ignore_eos=False,
            stop=None,
            stop_token_ids=None,
            detokenize=True,
            seed=7,
            logits_processors=None,
          ),
          target_endpoint=None,
          dkv_cache_hint=None,
          cache_salt=None,
        )

tool-aware chat-template input
  |
  +-- IN
  |     messages = [
  |       {"role":"user","content":"What is the weather in Colombo?"}
  |     ]
  |     tools = token_generator_tools above
  +-- STRUCTURES
  |     tokenizer chat_template(messages, tools)
  +-- LOGIC
  |     exact placement/text is owned by the loaded model's chat template
  +-- OUT
        [CONCRETE repository Llama 3 tokenizer-fixture template output]
        &lt;|begin_of_text|&gt;
        &lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;

        Environment: ipython
        Cutting Knowledge Date: December 2023
        Today Date: 26 Jul 2024

        &lt;|eot_id|&gt;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;

        Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

        Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

        {
            "function": {
                "description": "Return the current weather for a city.",
                "name": "get_weather",
                "parameters": {
                    "additionalProperties": false,
                    "properties": {
                        "city": {
                            "description": "City name, for example Colombo.",
                            "type": "string"
                        },
                        "unit": {
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ],
                            "type": "string"
                        }
                    },
                    "required": [
                        "city"
                    ],
                    "type": "object"
                },
                "strict": true
            },
            "type": "function"
        }

        What is the weather in Colombo?&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;

        prompt_token_count = 239

_resolve_grammar_constraints(...)
  |
  +-- IN
  |     tools = token_generator_tools
  |     tool_choice = "auto"
  |     response_format = None
  +-- STRUCTURES
  |     tuple[tools | None, schema | None, bool, bool]
  +-- LOGIC
  |     auto tools without response format -> conditional grammar enforcement
  +-- OUT
        (
          grammar_tools=token_generator_tools,
          response_format_schema=None,
          tools_forced=False,
          enforce_from_start=False,
        )
        |
        +-- configured parser implements generate_tool_call_grammar?
              |
              +-- no
              |     +-- response_format remains None
              |     +-- tool menu is still rendered into the prompt
              |
              +-- yes
                    +-- TextGenerationResponseFormat(
                          type="grammar",
                          grammar=complete parser-generated grammar string,
                          json_schema=None,
                          grammar_enforced=False,
                          tools_forced=False,
                          requires_structured_output_flag=False,
                          has_json_schema=False,
                        )

        [CONCRETE LlamaToolParser branch]
        LlamaToolParser has no generate_tool_call_grammar method
          -> response_format remains None
          -> tool menu is rendered into the prompt

[ILLUSTRATIVE MODEL OUTPUT, CONCRETE PARSER INPUT SHAPE]
  |
  +-- IN
  |     response_message =
  |       '{"name":"get_weather","parameters":{"city":"Colombo","unit":"celsius"}}'
  +-- STRUCTURES
  |     model-generated JSON text using Llama's name + parameters envelope
  +-- LOGIC
  |     LlamaToolParser.parse_complete uses parse_json_from_text(response_message)
  +-- OUT
        parsed_json_objects = [
          {
            "name": "get_weather",
            "parameters": {
              "city": "Colombo",
              "unit": "celsius",
            },
          }
        ]
        completion_token_ids_with_eos = [
          5018,609,3332,456,70464,2247,14105,23118,9103,
          3332,6255,81487,2247,3928,3332,66,41347,32075,128009
        ]

LlamaToolParser.parse_complete(response_message)
  |
  +-- IN
  |     parsed_json_objects above
  +-- STRUCTURES
  |     ParsedToolResponse + ParsedToolCall
  +-- LOGIC
  |     generate call ID; serialize parameters dict into arguments JSON string
  +-- OUT
        [ILLUSTRATIVE runtime UUID value]
        ParsedToolResponse(
          content=None,
          tool_calls=[
            ParsedToolCall(
              id="call_7d8e9f0011223344",
              name="get_weather",
              arguments='{"city": "Colombo", "unit": "celsius"}',
            )
          ],
        )

OpenAIChatResponseGenerator._tool_response_to_choices(parsed)
  |
  +-- IN
  |     ParsedToolResponse above
  +-- STRUCTURES
  |     ChatCompletionMessageToolCall inside ChatCompletionResponseChoice
  +-- LOGIC
  |     map ParsedToolCall fields to the OpenAI response schema
  +-- OUT
        ChatCompletionMessageToolCall(
          id="call_7d8e9f0011223344",
          type="function",
          function={
            "name": "get_weather",
            "arguments": "{\"city\": \"Colombo\", \"unit\": \"celsius\"}",
          },
        )

[ILLUSTRATIVE RUNTIME ID/TIME/TOKEN COUNTS; CONCRETE HTTP JSON SHAPE]
  |
  +-- IN
  |     parsed tool call above + completed TokenGeneratorOutput chunks
  +-- STRUCTURES
  |     CreateChatCompletionResponse
  +-- LOGIC
  |     complete() emits finish_reason="tool_calls" and assistant content=""
  +-- OUT
        HTTP/1.1 200 OK
        Content-Type: application/json

        {
          "id": "req-tools-001",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                  {
                    "id": "call_7d8e9f0011223344",
                    "type": "function",
                    "function": {
                      "name": "get_weather",
                      "arguments": "{\"city\": \"Colombo\", \"unit\": \"celsius\"}"
                    }
                  }
                ],
                "function_call": null,
                "refusal": ""
              },
              "finish_reason": "tool_calls",
              "logprobs": {
                "content": [],
                "refusal": []
              }
            }
          ],
          "created": 1730000002,
          "model": "meta-llama/Llama-3.1-8B-Instruct",
          "object": "chat.completion",
          "usage": {
            "prompt_tokens": 239,
            "completion_tokens": 19,
            "total_tokens": 258,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0}
          }
        }

[CONCRETE STREAMING LIMITATION FOR DEFAULT LLAMA PARSER]
  |
  +-- IN
  |     stream=True + LlamaToolParser + generated tool JSON text
  +-- STRUCTURES
  |     LlamaToolParser._buffer: str
  +-- LOGIC
  |     parse_delta(delta) appends to _buffer and returns None
  +-- OUT
        no ChoiceDeltaToolCall objects are produced by this parser
        generated JSON currently follows the normal delta.content path

[CONCRETE] PARSED TOOL CALL FIELD MAPPING
  |
  +-- IN
  |     parser output = {
  |       "name": "get_weather",
  |       "arguments": "{\"city\": \"Colombo\", \"unit\": \"celsius\"}"
  |     }
  +-- STRUCTURES
  |     ChatCompletionMessageToolCall
  +-- LOGIC
  |     parser converts model-specific tool syntax to OpenAI tool-call fields
  +-- OUT
        ChatCompletionMessageToolCall(
          id="call_7d8e9f0011223344",
          type="function",
          function={
            "name": "get_weather",
            "arguments": "{\"city\": \"Colombo\", \"unit\": \"celsius\"}",
          },
        )

[CONCRETE] FOLLOW-UP REQUEST WITH TOOL RESULT
  |
  +-- IN
  |     {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "messages": [
  |         {
  |           "role": "user",
  |           "content": "What is the weather in Colombo?"
  |         },
  |         {
  |           "role": "assistant",
  |           "content": null,
  |           "tool_calls": [
  |             {
  |               "id": "call_7d8e9f0011223344",
  |               "type": "function",
  |               "function": {
  |                 "name": "get_weather",
  |                 "arguments": "{\"city\":\"Colombo\",\"unit\":\"celsius\"}"
  |               }
  |             }
  |           ]
  |         },
  |         {
  |           "role": "tool",
  |           "tool_call_id": "call_7d8e9f0011223344",
  |           "content": "{\"temperature\":29,\"unit\":\"celsius\",\"condition\":\"partly cloudy\"}"
  |         }
  |       ],
  |       "tools": [
  |         {
  |           "type": "function",
  |           "function": {
  |             "name": "get_weather",
  |             "description": "Return the current weather for a city.",
  |             "parameters": {
  |               "type": "object",
  |               "properties": {
  |                 "city": {"type": "string"},
  |                 "unit": {
  |                   "type": "string",
  |                   "enum": ["celsius", "fahrenheit"]
  |                 }
  |               },
  |               "required": ["city"],
  |               "additionalProperties": false
  |             },
  |             "strict": true
  |           }
  |         }
  |       ],
  |       "tool_choice": "auto",
  |       "stream": false
  |     }
  +-- STRUCTURES
  |     assistant tool_calls history + matching tool response
  +-- LOGIC
  |     _validate_tool_message_consistency matches call_7d8e9f0011223344 to reply
  +-- OUT
        TextGenerationRequestMessage(
          role="assistant",
          content="",
          tool_calls=[
            {
              "id": "call_7d8e9f0011223344",
              "type": "function",
              "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"Colombo\",\"unit\":\"celsius\"}",
              },
            }
          ],
          tool_call_id=None,
          reasoning_content=None,
        )
        +
        TextGenerationRequestMessage(
          role="tool",
          content="{\"temperature\":29,\"unit\":\"celsius\",\"condition\":\"partly cloudy\"}",
          tool_calls=None,
          tool_call_id="call_7d8e9f0011223344",
          reasoning_content=None,
        )
</pre>

</details>

## 0.4 Actual JSON-Schema And Grammar Values

<details>
<summary><strong>Expand: Complete structured-output request and response formats</strong></summary>

<pre>
[CONCRETE] JSON-SCHEMA HTTP REQUEST BODY
  |
  +-- IN
  |     {
  |       "model": "meta-llama/Llama-3.1-8B-Instruct",
  |       "messages": [
  |         {
  |           "role": "user",
  |           "content": "Describe KV cache as JSON."
  |         }
  |       ],
  |       "response_format": {
  |         "type": "json_schema",
  |         "json_schema": {
  |           "name": "kv_cache_answer",
  |           "strict": true,
  |           "schema": {
  |             "type": "object",
  |             "properties": {
  |               "answer": {"type": "string"},
  |               "benefit": {"type": "string"}
  |             },
  |             "required": ["answer", "benefit"],
  |             "additionalProperties": false
  |           }
  |         }
  |       },
  |       "temperature": 0,
  |       "max_completion_tokens": 96,
  |       "seed": 9,
  |       "stream": false
  |     }
  +-- STRUCTURES
  |     OpenAI ResponseFormatJSONSchema request dictionary
  +-- LOGIC
  |     require server enable_structured_output=True; extract nested schema
  +-- OUT
        completion_request.response_format = {
          "type": "json_schema",
          "json_schema": {
            "name": "kv_cache_answer",
            "strict": True,
            "schema": {
              "type": "object",
              "properties": {
                "answer": {"type": "string"},
                "benefit": {"type": "string"},
              },
              "required": ["answer", "benefit"],
              "additionalProperties": False,
            },
          },
        }

_create_response_format(completion_request.response_format, True)
  |
  +-- IN
  |     OpenAI response_format dictionary above
  +-- STRUCTURES
  |     TextGenerationResponseFormat
  +-- LOGIC
  |     discard OpenAI name/strict wrapper after extracting schema
  |     mark schema enforcement active from first generated token
  +-- OUT
        TextGenerationResponseFormat(
          type="json_schema",
          json_schema={
            "type": "object",
            "properties": {
              "answer": {"type": "string"},
              "benefit": {"type": "string"},
            },
            "required": ["answer", "benefit"],
            "additionalProperties": False,
          },
          grammar=None,
          grammar_enforced=True,
          tools_forced=False,
          requires_structured_output_flag=True,
          has_json_schema=True,
        )

tokenizer.new_context(request)
  |
  +-- IN
  |     request.response_format = TextGenerationResponseFormat above
  +-- STRUCTURES
  |     serialized json_schema string + GrammarEnforcementState
  +-- LOGIC
  |     json.dumps(schema); initialize enforcement state
  +-- OUT
        json_schema = "{\"type\": \"object\", \"properties\": {\"answer\": {\"type\": \"string\"}, \"benefit\": {\"type\": \"string\"}}, \"required\": [\"answer\", \"benefit\"], \"additionalProperties\": false}"

        grammar = None

        grammar_state = GrammarEnforcementState(
          grammar_enforced=True,
          tools_forced=False,
          requires_structured_output_flag=True,
          has_json_schema=True,
          dead_matcher_reported=False,
          tool_region=None,
          thinking_region_delimiters=None,
          _in_thinking_region=False,
          _tool_calling_match_buffer=[],
          _thinking_match_buffer=[],
        )

structured-output matcher
  |
  +-- IN
  |     schema above + tokenizer vocabulary
  +-- STRUCTURES
  |     grammar matcher + per-step token bitmask
  +-- LOGIC
  |     at each decode step, mask tokens that cannot continue valid schema JSON
  +-- OUT
        [ILLUSTRATIVE next-token decision]
        generated_prefix = "{\"answer\":"
        allowed_next_shapes = [JSON string opening quote]
        masked_examples = ["]", "true", "123", "unexpected_property"]
        selected_next_text = "\""

[ILLUSTRATIVE] VALID MODEL CONTENT
  |
  +-- IN
  |     constrained generated text
  +-- STRUCTURES
  |     one JSON object serialized as assistant content
  +-- LOGIC
  |     grammar prevents missing required keys and additional properties
  +-- OUT
        response_message = "{\"answer\":\"KV cache stores attention keys and values from earlier tokens.\",\"benefit\":\"It avoids recomputing prior attention states during decoding.\"}"
        completion_token_ids_with_eos = [
          5018,9399,3332,83807,6636,10756,6666,7039,323,2819,505,6931,
          11460,48991,68244,275,3332,2181,55952,38301,631,287,4972,6666,
          5415,2391,48216,1210,92,128009
        ]

OpenAIChatResponseGenerator.complete(...)
  |
  +-- IN
  |     response_message above
  +-- STRUCTURES
  |     CreateChatCompletionResponse
  +-- LOGIC
  |     keep structured JSON as the assistant message content string
  +-- OUT
        [ILLUSTRATIVE runtime ID/time/token counts; concrete HTTP JSON shape]
        HTTP/1.1 200 OK
        Content-Type: application/json

        {
          "id": "req-json-001",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "{\"answer\":\"KV cache stores attention keys and values from earlier tokens.\",\"benefit\":\"It avoids recomputing prior attention states during decoding.\"}",
                "refusal": ""
              },
              "finish_reason": "stop",
              "logprobs": {
                "content": [],
                "refusal": []
              }
            }
          ],
          "created": 1730000003,
          "model": "meta-llama/Llama-3.1-8B-Instruct",
          "object": "chat.completion",
          "usage": {
            "prompt_tokens": 41,
            "completion_tokens": 30,
            "total_tokens": 71,
            "prompt_tokens_details": {
              "cached_tokens": 0
            },
            "completion_tokens_details": {
              "reasoning_tokens": 0
            }
          }
        }
</pre>

</details>

## 0.5 Actual MAX Worker, Model, And HTTP Response Values

<details>
<summary><strong>Expand: Complete worker request, batch, output, and response formats</strong></summary>

<pre>
[CONCRETE] TextContext BEFORE WORKER HANDOFF
  |
  +-- IN
  |     TextGenerationRequest + rendered prompt/token IDs from sections 0.1/0.2
  +-- STRUCTURES
  |     scheduler/model-oriented TextContext
  +-- LOGIC
  |     replace prompt/message representation with token/state representation
  +-- OUT
        TextContext(
          request_id=RequestID("req-chat-001"),
          max_length=81,
          tokens=TokenBuffer(
            array=np.array([
              128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,
              220,2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,
              2675,527,264,64694,18328,13,128009,128006,882,128007,271,849,
              21435,85748,6636,304,832,11914,13,128009,128006,78191,128007,271
            ], dtype=np.int64),
            _processing_range=Range(start=0,end=49),
            _processing_offset=0,
            _completion_range=Range(start=49,end=49),
            _prompt_length=49,
            _current_length=49,
            _actively_chunked=False,
          ),
          eos_tracker=EOSTracker(
            eos_token_ids={128001,128008,128009},
            eos_sequences=[[198,1502,25]],
            eos_stop_strings=["\nUser:"],
          ),
          vocab_size=128256,
          log_probabilities=0,
          log_probabilities_echo=False,
          ignore_eos=False,
          json_schema=None,
          grammar=None,
          grammar_state=GrammarEnforcementState(
            grammar_enforced=False,
            tools_forced=False,
            requires_structured_output_flag=False,
            has_json_schema=False,
            dead_matcher_reported=False,
            tool_region=None,
            thinking_region_delimiters=None,
            _in_thinking_region=False,
            _tool_calling_match_buffer=[],
            _thinking_match_buffer=[],
          ),
          sampling_params=SamplingParams(
            top_k=50,
            top_p=0.9,
            min_p=0.0,
            temperature=0.2,
            thinking_temperature=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=32,
            min_new_tokens=0,
            ignore_eos=False,
            stop=["\nUser:"],
            stop_token_ids=None,
            detokenize=True,
            seed=42,
            logits_processors=None,
          ),
          model_name="meta-llama/Llama-3.1-8B-Instruct",
          status=GenerationStatus.ACTIVE,
          target_endpoint=None,
          dkv_cache_hint=None,
          cache_salt=None,
          cached_prefix_length=None,
        )

request_queue.put(context)
  |
  +-- IN
  |     TextContext above
  +-- STRUCTURES
  |     MAX queue payload typed as TextContext; not HTTP JSON
  +-- LOGIC
  |     queue serializer transfers the typed context API process -> worker process
  +-- OUT
        logical worker queue item = {
          RequestID("req-chat-001"): TextContext above
        }
        raw ZeroMQ frames = [RUNTIME serializer/transport bytes]

batch_constructor.construct_batch()
  |
  +-- IN
  |     replica_0.ce_reqs = {
  |       RequestID("req-chat-001"): TextContext above
  |     }
  |     replica_0.tg_reqs = {}
  +-- STRUCTURES
  |     ReplicaBatch -> TextGenerationInputs
  +-- LOGIC
  |     fresh generated_length=0 request is CE/prefill work
  +-- OUT
        TextGenerationInputs(
          batches=[[TextContext(req-chat-001)]],
          input_tokens=49,
          context_tokens=0,
          batch_type=BatchType.CE,
          per_replica_input_tokens=[49],
          per_replica_context_tokens=[0],
        )

Llama3BatchProcessor.prepare_initial_token_inputs(...)
  |
  +-- IN
  |     replica_batches = [[TextContext(req-chat-001)]]
  |     return_n_logits = 1
  |     kv allocation = first request, no cached prefix
  +-- STRUCTURES
  |     packed device buffers + KVCacheInputsPerDevice
  +-- LOGIC
  |     flatten active tokens; build cumulative row offsets; attach KV metadata
  +-- OUT
        [CONCRETE request-derived buffers]
        tokens = Buffer(
          shape=[49],
          dtype="int64",
          device="gpu:0",
          values=[
            128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,
            220,2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,
            2675,527,264,64694,18328,13,128009,128006,882,128007,271,849,
            21435,85748,6636,304,832,11914,13,128009,128006,78191,128007,271
          ],
        )

        input_row_offsets = Buffer(
          shape=[2],
          dtype="uint32",
          device="gpu:0",
          values=[0,49],
        )

        return_n_logits = Buffer(
          shape=[1],
          dtype="int64",
          device="cpu",
          values=[1],
        )

        [ILLUSTRATIVE allocator-selected block ID]
        kv_cache_inputs = KVCacheInputsPerDevice(
          kv_blocks=Buffer(
            device="gpu:0",
            shape=[RUNTIME model/cache configuration],
            dtype=[RUNTIME KV dtype],
            values=[RUNTIME model state],
          ),
          cache_lengths=Buffer(
            shape=[1], dtype="uint32", device="gpu:0", values=[0]
          ),
          lookup_table=Buffer(
            shape=[1,1], dtype="int32", device="gpu:0", values=[[7]]
          ),
          max_prompt_length=Buffer(
            shape=[1], dtype="uint32", device="gpu:0", values=[49]
          ),
          max_cache_length=Buffer(
            shape=[1], dtype="uint32", device="gpu:0", values=[49]
          ),
          page_stride=Buffer(
            shape=[1], dtype="int64", device="gpu:0", values=[RUNTIME]
          ),
          kv_scales=None,
          scales_page_stride=None,
          scales_lookup_table=None,
          attention_dispatch_metadata=None,
          draft_attention_dispatch_metadata=None,
          mla_num_partitions=None,
          draft_mla_num_partitions=None,
          kv_blocks_per_layer=None,
          kv_scales_per_layer=None,
        )

        llama3_inputs = Llama3Inputs(
          tokens=tokens,
          input_row_offsets=input_row_offsets,
          signal_buffers=[],
          return_n_logits=return_n_logits,
          data_parallel_splits=None,
          kv_cache_inputs=kv_cache_inputs,
        )

self.model.execute(*llama3_inputs.buffers)
  |
  +-- IN
  |     ordered buffers = (
  |       tokens,
  |       input_row_offsets,
  |       return_n_logits,
  |       *signal_buffers,
  |       *tree.leaves(kv_cache_inputs),
  |     )
  +-- STRUCTURES
  |     compiled Llama graph input ABI
  +-- LOGIC
  |     execute transformer graph; produce last-token vocabulary logits
  +-- OUT
        [RUNTIME full output]
        raw_logits = Buffer(
          shape=[1,128256],
          dtype="float32",
          device="gpu:0",
          values=[128256 runtime-computed floats],
        )

        [ILLUSTRATIVE top-candidate view of that full buffer]
        top_candidates = [
          {"token_id":83807,"decoded":"KV","logit":8.40},
          {"token_id":791,"decoded":"The","logit":7.95},
          {"token_id":32,"decoded":"A","logit":7.31},
          {"token_id":2675,"decoded":"You","logit":6.88}
        ]

process_ragged_kv_outputs(raw_graph_outputs)
  |
  +-- IN
  |     raw logits Buffer above
  +-- STRUCTURES
  |     ModelOutputs
  +-- LOGIC
  |     map compiled output positions to named Python fields
  +-- OUT
        ModelOutputs(
          logits=raw_logits,
          next_token_logits=None,
          logit_offsets=None,
          hidden_states=None,
        )

FusedSamplingProcessor / compiled sampler
  |
  +-- IN
  |     logits shape=[1,128256]
  |     temperature=[0.2]
  |     top_k=[50]
  |     top_p=[0.9]
  |     min_p=[0.0]
  |     seed=[42]
  +-- STRUCTURES
  |     sampling parameter buffers + RNG state + token output Buffer
  +-- LOGIC
  |     filter candidate distribution and sample one token
  +-- OUT
        [ILLUSTRATIVE model-dependent]
        generated_tokens_device = Buffer(
          shape=[1], dtype="int64", device="gpu:0", values=[83807]
        )
        generated_tokens_host = np.array([83807], dtype=np.int64)

update_context_and_prepare_responses(...)
  |
  +-- IN
  |     context=req-chat-001
  |     generated_tokens_host=[83807]
  +-- STRUCTURES
  |     updated TokenBuffer + TextGenerationOutput
  +-- LOGIC
  |     append token; advance status/grammar; expose unstreamed token delta
  +-- OUT
        context.tokens.generated = [83807]
        context.status = GenerationStatus.ACTIVE

        TextGenerationOutput(
          request_id=RequestID("req-chat-001"),
          tokens=[83807],
          final_status=GenerationStatus.ACTIVE,
          log_probabilities=None,
          num_cached_tokens=0,
        )

SchedulerResult.create(output, batch_id=1)
  |
  +-- IN
  |     TextGenerationOutput above
  +-- STRUCTURES
  |     msgspec SchedulerResult
  +-- LOGIC
  |     copy output completion state and stamp scheduler batch ID
  +-- OUT
        SchedulerResult(
          is_done=False,
          result=TextGenerationOutput(
            request_id=RequestID("req-chat-001"),
            tokens=[83807],
            final_status=GenerationStatus.ACTIVE,
            log_probabilities=None,
            num_cached_tokens=0,
          ),
          error=None,
          batch_id=1,
        )

response_queue.put_nowait(response_dict)
  |
  +-- IN
  |     response_dict = {
  |       RequestID("req-chat-001"): SchedulerResult above
  |     }
  +-- STRUCTURES
  |     MAX worker -> API response queue payload; not OpenAI JSON
  +-- LOGIC
  |     serialize typed result map across process boundary
  +-- OUT
        API response_worker receives {
          RequestID("req-chat-001"): SchedulerResult above
        }

TokenGeneratorPipeline._generate(...)
  |
  +-- IN
  |     TextGenerationOutput(tokens=[83807], final_status=ACTIVE)
  +-- STRUCTURES
  |     buffered detokenizer + TokenGeneratorOutput
  +-- LOGIC
  |     decode token ID; attach request/batch/token-count metadata
  +-- OUT
        [ILLUSTRATIVE tokenizer-dependent decoding]
        TokenGeneratorOutput(
          status=GenerationStatus.ACTIVE,
          decoded_tokens="KV",
          decoded_reasoning_tokens=None,
          token_count=1,
          token_log_probabilities=None,
          top_log_probabilities=None,
          sampled_tokens=None,
          prompt_token_count=49,
          cached_token_count=0,
          reasoning_token_count=0,
          stop_sequence=None,
          batch_id=1,
          token_ids=[83807],
          prompt_token_ids=np.array([
            128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,
            220,2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,
            2675,527,264,64694,18328,13,128009,128006,882,128007,271,849,
            21435,85748,6636,304,832,11914,13,128009,128006,78191,128007,271
          ], dtype=np.int64),
        )

[ILLUSTRATIVE] COMPLETE NORMAL-TEXT STREAM
  |
  +-- IN
  |     decoded chunks = [
  |       "KV cache",
  |       " stores reusable attention keys and values from earlier tokens.",
  |     ]
  |     terminal status = END_OF_SEQUENCE
  |     completion_token_ids =
  |       [83807,6636,10756,62671,6666,7039,323,2819,505,6931,11460,13,128009]
  |     prompt_tokens=49; completion_tokens=13; cached_tokens=0
  +-- STRUCTURES
  |     EventSourceResponse over OpenAI stream-response JSON strings
  +-- LOGIC
  |     one JSON object per SSE data frame; usage chunk; [DONE]
  +-- OUT
        HTTP/1.1 200 OK
        Content-Type: text/event-stream; charset=utf-8

        data: {"id":"req-chat-001","choices":[{"index":0,"delta":{"content":"KV cache","role":"assistant"}}],"created":1730000004,"model":"meta-llama/Llama-3.1-8B-Instruct","object":"chat.completion.chunk"}

        data: {"id":"req-chat-001","choices":[{"index":0,"delta":{"content":" stores reusable attention keys and values from earlier tokens.","role":"assistant"}}],"created":1730000004,"model":"meta-llama/Llama-3.1-8B-Instruct","object":"chat.completion.chunk"}

        data: {"id":"req-chat-001","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"created":1730000004,"model":"meta-llama/Llama-3.1-8B-Instruct","object":"chat.completion.chunk"}

        data: {"id":"req-chat-001","choices":[],"created":1730000004,"model":"meta-llama/Llama-3.1-8B-Instruct","object":"chat.completion.chunk","usage":{"completion_tokens":13,"prompt_tokens":49,"total_tokens":62,"completion_tokens_details":{"reasoning_tokens":0},"prompt_tokens_details":{"cached_tokens":0}}}

        data: [DONE]

[ILLUSTRATIVE] COMPLETE NON-STREAMING RESPONSE FOR THE SAME TEXT
  |
  +-- IN
  |     all TokenGeneratorOutput chunks collected by pipeline.all_tokens(...)
  +-- STRUCTURES
  |     CreateChatCompletionResponse
  +-- LOGIC
  |     concatenate decoded content; derive finish_reason; calculate usage
  +-- OUT
        HTTP/1.1 200 OK
        Content-Type: application/json

        {
          "id": "req-chat-001",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "KV cache stores reusable attention keys and values from earlier tokens.",
                "tool_calls": null,
                "function_call": null,
                "refusal": ""
              },
              "finish_reason": "stop",
              "logprobs": {
                "content": [],
                "refusal": []
              }
            }
          ],
          "created": 1730000004,
          "model": "meta-llama/Llama-3.1-8B-Instruct",
          "object": "chat.completion",
          "usage": {
            "prompt_tokens": 49,
            "completion_tokens": 13,
            "total_tokens": 62,
            "prompt_tokens_details": {
              "cached_tokens": 0
            },
            "completion_tokens_details": {
              "reasoning_tokens": 0
            }
          }
        }
</pre>

</details>

## 1. One-Time Startup Wiring

<details>
<summary><strong>Expand: API server startup and API-worker wiring</strong></summary>

<pre>
fastapi_app(settings, serving_settings)
  api_server.py:330
    |
    +-- IN
    |     settings: Settings
    |     serving_settings: ServingTokenGeneratorSettings
    +-- STRUCTURES
    |     FastAPI; ROUTES; app.state; ZMQ endpoint path
    +-- LOGIC
    |     create app -> register routes/middleware -> bind lifespan
    +-- OUT
    |     app: FastAPI
    |
    +-- [CALL] FastAPI(title="MAX Serve", lifespan=lifespan_wrap)
    |     |
    |     +-- IN: title + lifespan callback
    |     +-- STRUCTURES: FastAPI application object
    |     +-- LOGIC: initialize ASGI app and state container
    |     +-- OUT: app
    |
    +-- [CALL] app.include_router(ROUTES[api_type].router)
    |     |
    |     +-- IN: OpenAI router with prefix "/v1"
    |     +-- STRUCTURES: APIRouter + route table
    |     +-- LOGIC: attach /v1/chat/completions and related endpoints
    |     +-- OUT: None; app route table mutated
    |
    +-- [FRAMEWORK] lifespan(app, settings, serving_settings, zmq_endpoint_base)
          api_server.py:111
            |
            +-- IN: app + server/model settings + IPC endpoint base
            +-- STRUCTURES: AsyncExitStack; app.state; model worker interface
            +-- LOGIC: create long-lived serving dependencies; yield while live
            +-- OUT: async context; initialized app state
            |
            +-- [CALL] ZmqModelWorkerInterface(...)
            |     |
            |     +-- IN: task; context type; request_queue_size
            |     +-- STRUCTURES: ZMQ request/response/cancel queue factories
            |     +-- LOGIC: configure typed process boundary
            |     +-- OUT: model_worker_interface
            |
            +-- [AWAIT] start_model_worker(...)
            |     model_worker.py:613
            |       |
            |       +-- IN: model_factory; configs; interface; endpoint; memory_plan
            |       +-- STRUCTURES: ProcessMonitor; Event; spawned ModelWorker process
            |       +-- LOGIC: spawn worker -> wait ready -> create API proxy
            |       +-- OUT: model_worker: ZmqModelWorkerProxy
            |       |
            |       +-- proc.start(ModelWorker(), ...)
            |       |     |
            |       |     +-- IN: callable ModelWorker + startup arguments
            |       |     +-- STRUCTURES: child process + alive/ready event
            |       |     +-- LOGIC: invoke ModelWorker.__call__ in child
            |       |     +-- OUT: running child process handle
            |       |
            |       +-- ModelWorker.__call__(...)
            |       |     model_worker.py:564
            |       |       |
            |       |       +-- IN: same startup arguments in child process
            |       |       +-- STRUCTURES: uvloop event loop
            |       |       +-- LOGIC: run ModelWorker.run coroutine; suppress Ctrl-C noise
            |       |       +-- OUT: None when worker exits
            |       |       |
            |       |       +-- uvloop.run(ModelWorker.run(...))
            |       |             |
            |       |             +-- IN: ModelWorker.run coroutine object
            |       |             +-- STRUCTURES: uvloop runtime
            |       |             +-- LOGIC: own child-process event loop
            |       |             +-- OUT: coroutine completion value None
            |       |
            |       +-- [AWAIT] proc.ready(...)
            |       |     |
            |       |     +-- IN: worker heartbeat/readiness event
            |       |     +-- STRUCTURES: ProcessMonitor state
            |       |     +-- LOGIC: block startup until child signals alive
            |       |     +-- OUT: None or startup exception
            |       |
            |       +-- [AWAIT] model_worker_interface.model_worker_proxy()
            |             zmq_interface.py:331
            |               |
            |               +-- IN: API-side interface configuration
            |               +-- STRUCTURES: ZmqModelWorkerProxy; ZMQ queues; task set
            |               +-- LOGIC: create proxy -> start background consumers -> connect
            |               +-- OUT: connected proxy
            |               |
            |               +-- asyncio.create_task(proxy.response_worker())
            |               |     |
            |               |     +-- IN: response_worker coroutine
            |               |     +-- STRUCTURES: asyncio.Task
            |               |     +-- LOGIC: route worker results continuously
            |               |     +-- OUT: background task handle
            |               |
            |               +-- asyncio.create_task(proxy._metrics_worker())
            |               |     |
            |               |     +-- IN: metrics coroutine
            |               |     +-- STRUCTURES: asyncio.Task
            |               |     +-- LOGIC: consume worker metrics continuously
            |               |     +-- OUT: background task handle
            |               |
            |               +-- [AWAIT] proxy.wait_until_connected(timeout_s)
            |                     |
            |                     +-- IN: optional connection timeout
            |                     +-- STRUCTURES: request/response/cancel MAX queues
            |                     +-- LOGIC: wait until IPC sockets are connected
            |                     +-- OUT: None; queues ready or timeout error
            |
            +-- [CALL] TokenGeneratorPipeline(...)
            |     |
            |     +-- IN: model_name; tokenizer; proxy; LoRA/reasoning options
            |     +-- STRUCTURES: TokenGeneratorPipeline
            |     +-- LOGIC: bind API tokenization/detokenization to worker proxy
            |     +-- OUT: pipeline
            |
            +-- app.state.pipeline/config/settings = initialized objects
                  |
                  +-- IN: pipeline + configuration objects
                  +-- STRUCTURES: Starlette app.state
                  +-- LOGIC: expose dependencies to request routes
                  +-- OUT: None; shared application state populated

register_request(app, structured_logging=...)
  request.py:45
    |
    +-- IN: FastAPI app + logging mode
    +-- STRUCTURES: HTTP middleware closure request_session
    +-- LOGIC: register request identity/timing/trace middleware
    +-- OUT: None; middleware installed
</pre>

</details>

<details>
<summary><strong>Expand: Model-worker startup and scheduler loop</strong></summary>

<pre>
ModelWorker.run(...)
  model_worker.py:232
    |
    +-- IN
    |     model_factory; pipeline_config; settings; worker interface; memory_plan
    +-- STRUCTURES
    |     metric client; model pipeline; WorkerQueues; Scheduler; KV manager
    +-- LOGIC
    |     configure process -> build model -> create queues/scheduler -> loop
    +-- OUT
    |     None; long-running worker coroutine
    |
    +-- [CALL] model_factory()
    |     |
    |     +-- IN: captured architecture/config/session/weights
    |     +-- STRUCTURES: TextGenerationPipeline + compiled Llama model
    |     +-- LOGIC: build/compile/load graph once during startup
    |     +-- OUT: pipeline
    |
    +-- [AWAIT] model_worker_interface.model_worker_queues()
    |     zmq_interface.py:321
    |       |
    |       +-- IN: worker-side interface configuration
    |       +-- STRUCTURES: WorkerQueues(request_queue,response_queue,cancel_queue)
    |       +-- LOGIC: bind worker PULL/PUSH/PULL sockets
    |       +-- OUT: async context yielding WorkerQueues
    |
    +-- [CALL] load_scheduler(...)
    |     scheduler/__init__.py:76
    |       |
    |       +-- IN: pipeline + configs + WorkerQueues + memory_plan
    |       +-- STRUCTURES: Scheduler implementation + batch constructor
    |       +-- LOGIC: select scheduler from task and pipeline role
    |       +-- OUT: TokenGenerationScheduler for prefill_and_decode
    |       |
    |       +-- load_text_generation_scheduler(...)
    |             +-- IN: text pipeline + scheduler settings + worker queues
    |             +-- STRUCTURES: TextBatchConstructor + TokenGenerationScheduler
    |             +-- LOGIC: construct combined prefill/decode scheduler
    |             +-- OUT: TokenGenerationScheduler
    |
    +-- [BACKGROUND] while True
          |
          +-- IN: scheduler + live queues
          +-- STRUCTURES: SchedulerProgress + backoff counter
          +-- LOGIC: execute one iteration; heartbeat; back off when idle
          +-- OUT: repeated request progress
          |
          +-- scheduler.run_iteration()
          |     +-- OUT: MADE_PROGRESS or NO_PROGRESS
          |
          +-- sleep_with_backoff(count_no_progress)
                +-- IN: consecutive idle count
                +-- STRUCTURES: asyncio timer
                +-- LOGIC: avoid busy-spinning when no work exists
                +-- OUT: None after delay
</pre>

</details>

## 2. HTTP Middleware And OpenAI Ingress

<details>
<summary><strong>Expand: Request middleware prelude</strong></summary>

<pre>
request_session(request, call_next)
  request.py:52
    |
    +-- IN
    |     request=Request(POST /v1/chat/completions, body=&lt;concrete JSON&gt;)
    |     call_next=&lt;Starlette route dispatcher&gt;
    +-- STRUCTURES
    |     UUID string; StopWatch; optional server span; Response
    +-- LOGIC
    |     set request state -> invoke route -> add header -> record status/trace
    +-- OUT
    |     Response with X-Request-ID="req-chat-001"
    |
    +-- uuid.uuid4().hex
    |     +-- IN: operating-system/random UUID source
    |     +-- STRUCTURES: UUID -> 32-character hex string
    |     +-- LOGIC: create collision-resistant request identity
    |     +-- OUT: illustrative "req-chat-001" used by this document
    |
    +-- request.state.request_id = request_id
    |     +-- OUT: route-visible request identity
    |
    +-- request.state.request_timer = StopWatch(...)
    |     +-- OUT: timestamp source for TextGenerationRequest
    |
    +-- [AWAIT] call_next(request)
          |
          +-- IN: mutated Request
          +-- STRUCTURES: Starlette middleware chain
          +-- LOGIC: dispatch matching endpoint
          +-- OUT: route Response
</pre>

</details>

<details>
<summary><strong>Expand: Complete HTTP route input, structures, logic, and output</strong></summary>

<pre>
POST /v1/chat/completions
    |
    +-- [FRAMEWORK] FastAPI/Starlette route dispatch
          |
          +-- IN: Request + registered route table
          +-- STRUCTURES: ASGI scope; dependency/middleware stack
          +-- LOGIC: match POST path to openai_create_chat_completion
          +-- OUT: awaited route result
          |
          +-- [AWAIT] openai_create_chat_completion(request)
                openai_routes.py:2218
                  |
                  +-- IN
                  |     request with concrete body/state values
                  +-- STRUCTURES
                  |     CreateChatCompletionRequest; _ParsedChatRequest;
                  |     SamplingParams; TextGenerationRequest; response generator
                  +-- LOGIC
                  |     parse -> select -> normalize -> construct -> dispatch response mode
                  +-- OUT
                  |     EventSourceResponse because stream=true
                  |
                  +-- request_id = request.state.request_id
                  |     |
                  |     +-- IN: request.state
                  |     +-- STRUCTURES: str
                  |     +-- LOGIC: read middleware-created identifier
                  |     +-- OUT: "req-chat-001"
                  |
                  +-- [AWAIT] _parse_openai_request_body(
                  |             request, request_id, CreateChatCompletionRequest
                  |           )
                  |     openai_routes.py:3070
                  |       |
                  |       +-- IN: Request; "req-chat-001"; schema class
                  |       +-- STRUCTURES: bytes; JSON value; Pydantic model class
                  |       +-- LOGIC: read -> decode JSON -> normalize tools/extras -> validate
                  |       +-- OUT: completion_request: CreateChatCompletionRequest
                  |       |
                  |       +-- [AWAIT] request.body()
                  |       |     |
                  |       |     +-- IN: Starlette Request receive channel
                  |       |     +-- STRUCTURES: cached body bytes
                  |       |     +-- LOGIC: consume/cache complete HTTP request body
                  |       |     +-- OUT: b'{"model":"meta-llama/...","messages":[...],...}'
                  |       |
                  |       +-- [CALL] json.loads(raw)
                  |       |     |
                  |       |     +-- IN: raw body bytes
                  |       |     +-- STRUCTURES: Python dict/list/scalar JSON values
                  |       |     +-- LOGIC: JSON decode
                  |       |     +-- OUT: parsed dict matching the concrete body
                  |       |
                  |       +-- [BRANCH] parsed is not dict
                  |       |     |
                  |       |     +-- IN: list/str/number/bool/None
                  |       |     +-- STRUCTURES: Pydantic validation error details
                  |       |     +-- LOGIC: pass directly to model_validate
                  |       |     +-- OUT: normally ValidationError for chat schema
                  |       |
                  |       +-- [BRANCH] parsed is dict
                  |             |
                  |             +-- IN: concrete body dict
                  |             +-- STRUCTURES: tools value; known model fields; extras set
                  |             +-- LOGIC: normalize optional tool definitions and extra fields
                  |             +-- OUT: validation-ready dict
                  |             |
                  |             +-- parsed.get("tools")
                  |             |     +-- IN: parsed request dict
                  |             |     +-- STRUCTURES: list[tool] | None | other JSON value
                  |             |     +-- LOGIC: inspect optional tools without requiring key presence
                  |             |     +-- OUT: None for concrete request
                  |             |
                  |             +-- _normalize_tools_parameters(tools)
                  |             |     tool_call_normalization.py:62
                  |             |       |
                  |             |       +-- IN: list[tool dict] when tools is a list
                  |             |       +-- STRUCTURES: copied tool/function/parameters dicts
                  |             |       +-- LOGIC: replace missing/None function.parameters with {}
                  |             |       +-- OUT: normalized list; concrete request skips this
                  |             |
                  |             +-- get_app_pipeline_config(request.app)
                  |             |     openai_routes.py:3061
                  |             |       |
                  |             |       +-- IN: FastAPI app
                  |             |       +-- STRUCTURES: app.state.pipeline_config
                  |             |       +-- LOGIC: typed state lookup
                  |             |       +-- OUT: PipelineConfig
                  |             |
                  |             +-- [BRANCH] allow_extra_request_fields
                  |             |     |
                  |             |     +-- IN: parsed keys + model_cls.model_fields
                  |             |     +-- STRUCTURES: known/extras sets
                  |             |     +-- LOGIC: optionally remove unknown top-level keys
                  |             |     +-- OUT: unchanged or filtered parsed dict
                  |             |     |
                  |             |     +-- set(model_cls.model_fields)
                  |             |           +-- IN: Pydantic field-name mapping
                  |             |           +-- STRUCTURES: set[str]
                  |             |           +-- LOGIC: enable O(1) known-key membership tests
                  |             |           +-- OUT: known request-field names
                  |             |
                  |             +-- model_cls.model_validate(parsed)
                  |                   inherited Pydantic BaseModel.model_validate
                  |                     |
                  |                     +-- IN: parsed dict + CreateChatCompletionRequest class
                  |                     +-- STRUCTURES: field validators; before/after model validators
                  |                     +-- LOGIC: coerce/validate fields; reject forbidden extras
                  |                     +-- OUT: completion_request
                  |                     |
                  |                     +-- _translate_thinking_to_standard(data)
                  |                     |     +-- IN: raw request mapping
                  |                     |     +-- STRUCTURES: reasoning/thinking compatibility fields
                  |                     |     +-- LOGIC: translate vendor thinking fields before validation
                  |                     |     +-- OUT: normalized mapping
                  |                     |
                  |                     +-- _reconcile_max_completion_tokens(self)
                  |                           +-- IN: validated max_tokens/max_completion_tokens
                  |                           +-- STRUCTURES: mutable Pydantic model during validation
                  |                           +-- LOGIC: max_completion_tokens wins when both differ
                  |                           +-- OUT: validated CreateChatCompletionRequest
                  |
                  +-- [CALL] get_pipeline(request, completion_request.model)
                  |     openai_routes.py:458
                  |       |
                  |       +-- IN: Request; "meta-llama/Llama-3.1-8B-Instruct"
                  |       +-- STRUCTURES: app.state.pipeline; base/LoRA model names; PipelineTokenizer protocol
                  |       +-- LOGIC: validate model name and tokenizer capability
                  |       +-- OUT: existing TokenGeneratorPipeline
                  |       |
                  |       +-- isinstance(pipeline.tokenizer, PipelineTokenizer)
                  |             +-- IN: selected pipeline tokenizer object
                  |             +-- STRUCTURES: runtime-checkable protocol
                  |             +-- LOGIC: verify required tokenizer methods/properties exist
                  |             +-- OUT: true or route configuration error
                  |
                  +-- [CALL] get_tool_parser(request.app)
                  |     openai_routes.py:3110
                  |       |
                  |       +-- IN: FastAPI app
                  |       +-- STRUCTURES: PipelineConfig runtime parser name; parser registry
                  |       +-- LOGIC: resolve optional architecture tool parser
                  |       +-- OUT: ToolParser | None; None for this example
                  |
                  +-- tokenizer = pipeline.tokenizer
                  |     |
                  |     +-- IN: selected pipeline
                  |     +-- STRUCTURES: PipelineTokenizer implementation
                  |     +-- LOGIC: read tokenizer dependency
                  |     +-- OUT: TextTokenizer-compatible object
                  |
                  +-- [AWAIT] openai_parse_chat_completion_request(...)
                  |     openai_routes.py:1719
                  |       |
                  |       +-- IN
                  |       |     completion_request; expects_content_wrapping; media/role limits
                  |       +-- STRUCTURES
                  |       |     TextGenerationRequestMessage; content parts; MediaRef; _ParsedChatRequest
                  |       +-- LOGIC
                  |       |     validate history -> normalize messages -> resolve media
                  |       +-- OUT
                  |       |     _ParsedChatRequest with two typed messages and empty media lists
                  |       |
                  |       +-- _validate_tool_message_consistency(messages)
                  |       |     |
                  |       |     +-- IN: two OpenAI message dicts
                  |       |     +-- STRUCTURES: pending tool_call IDs set
                  |       |     +-- LOGIC: ensure tool responses match earlier assistant tool calls
                  |       |     +-- OUT: None; concrete text-only messages accepted
                  |       |
                  |       +-- normalize_tool_call_arguments(raw_tool_calls)
                  |       |     |
                  |       |     +-- IN: assistant tool_calls | None
                  |       |     +-- STRUCTURES: copied tool-call dictionaries
                  |       |     +-- LOGIC: normalize function arguments representation
                  |       |     +-- OUT: normalized calls | None; None here
                  |       |
                  |       +-- message.get("content") / message.get("tool_calls")
                  |       |     +-- IN: each OpenAI message mapping
                  |       |     +-- STRUCTURES: scalar/multipart content + optional tool-call list
                  |       |     +-- LOGIC: read optional keys without raising KeyError
                  |       |     +-- OUT: concrete content strings and tool_calls=None
                  |       |
                  |       +-- [BRANCH] content is list
                  |       |     |
                  |       |     +-- IN: multipart text/image/video content
                  |       |     +-- STRUCTURES: TextContentPart/ImageContentPart/VideoContentPart
                  |       |     +-- LOGIC: convert every OpenAI part and collect media refs
                  |       |     +-- OUT: typed part list + media tasks
                  |       |     |
                  |       |     +-- TextContentPart(text=...)
                  |       |     |     +-- IN: {"type":"text","text":"..."}
                  |       |     |     +-- STRUCTURES: typed text content part
                  |       |     |     +-- LOGIC: preserve text for chat-template rendering
                  |       |     |     +-- OUT: TextContentPart
                  |       |     |
                  |       |     +-- make_media_ref(url)
                  |       |           +-- IN: image/video URL string
                  |       |           +-- STRUCTURES: MediaRef implementation
                  |       |           +-- LOGIC: preserve parsed or raw media URL form
                  |       |           +-- OUT: MediaRef
                  |       |           |
                  |       |           +-- ImageContentPart(...)
                  |       |           |     +-- IN: normalized image metadata/placeholder
                  |       |           |     +-- STRUCTURES: typed image content part
                  |       |           |     +-- LOGIC: replace URL content with internal image placeholder
                  |       |           |     +-- OUT: ImageContentPart
                  |       |           |
                  |       |           +-- VideoContentPart(...)
                  |       |                 +-- IN: normalized video metadata/placeholder
                  |       |                 +-- STRUCTURES: typed video content part
                  |       |                 +-- LOGIC: replace URL content with internal video placeholder
                  |       |                 +-- OUT: VideoContentPart
                  |       |
                  |       +-- [BRANCH] content is scalar/None
                  |       |     |
                  |       |     +-- IN: concrete message content strings
                  |       |     +-- STRUCTURES: str
                  |       |     +-- LOGIC: use string; map None to empty string
                  |       |     +-- OUT: normalized strings unchanged
                  |       |
                  |       +-- _normalize_openai_role(role, allowed_roles)
                  |       |     |
                  |       |     +-- IN: "system" then "user"
                  |       |     +-- STRUCTURES: allowed role set
                  |       |     +-- LOGIC: validate and normalize role spelling
                  |       |     +-- OUT: "system" then "user"
                  |       |
                  |       +-- TextGenerationRequestMessage(...)
                  |       |     |
                  |       |     +-- IN: role/content/tool metadata
                  |       |     +-- STRUCTURES: frozen Pydantic message model
                  |       |     +-- LOGIC: validate content shape and preserve tool metadata
                  |       |     +-- OUT: two typed internal messages
                  |       |
                  |       +-- record/enforce media counts
                  |       |     |
                  |       |     +-- IN: image_count=0; video_count=0; model limits
                  |       |     +-- STRUCTURES: telemetry counters + validation errors
                  |       |     +-- LOGIC: emit metrics; reject counts above limits
                  |       |     +-- OUT: accepted request
                  |       |
                  |       +-- _request_media_budget(settings)
                  |       |     _image_resolution.py:582
                  |       |       |
                  |       |       +-- IN: Settings
                  |       |       +-- STRUCTURES: _MediaByteBudget(limit)
                  |       |       +-- LOGIC: derive per-request decoded-media byte budget
                  |       |       +-- OUT: budget object; unused for zero media
                  |       |
                  |       +-- [OPTIONAL MEDIA] resolve_image_from_url(media_ref,...)
                  |       |     _image_resolution.py:839
                  |       |       |
                  |       |       +-- IN: MediaRef + security/size settings + budget
                  |       |       +-- STRUCTURES: URL/data/file bytes; media facts
                  |       |       +-- LOGIC: fetch/decode source; enforce host/root/byte policy
                  |       |       +-- OUT: validated image bytes
                  |       |       |
                  |       |       +-- _fetch_url_bytes(...)
                  |       |       |     +-- IN: validated HTTP(S) URL
                  |       |       |     +-- STRUCTURES: streamed HTTP response
                  |       |       |     +-- LOGIC: fetch within redirect/host/byte limits
                  |       |       |     +-- OUT: image bytes
                  |       |       |
                  |       |       +-- _decode_data_uri_base64(...)
                  |       |             +-- IN: data URI
                  |       |             +-- STRUCTURES: base64 payload
                  |       |             +-- LOGIC: clean and decode payload
                  |       |             +-- OUT: image bytes
                  |       |
                  |       +-- asyncio.gather(image_tasks)
                  |       |     +-- IN: zero or more image coroutines
                  |       |     +-- STRUCTURES: ordered result list
                  |       |     +-- LOGIC: resolve images concurrently
                  |       |     +-- OUT: request_images
                  |       |
                  |       +-- asyncio.to_thread(_resolve_and_decode_images,...)
                  |       |     +-- IN: resolved image bytes
                  |       |     +-- STRUCTURES: worker thread + PIL images
                  |       |     +-- LOGIC: keep CPU-heavy image decode off event loop
                  |       |     +-- OUT: decoded_images
                  |       |
                  |       +-- create/await video resolution tasks
                  |       |     +-- IN: video MediaRef values + media budget/settings
                  |       |     +-- STRUCTURES: asynchronous video byte/frame results
                  |       |     +-- LOGIC: resolve and validate every requested video
                  |       |     +-- OUT: request_videos; skipped for concrete request
                  |       |
                  |       +-- return _ParsedChatRequest(...)
                  |             +-- IN: normalized messages and media lists
                  |             +-- STRUCTURES: NamedTuple
                  |             +-- LOGIC: package parser products
                  |             +-- OUT: two messages, [], [], []
                  |
                  +-- get_app_pipeline_config(request.app)
                  |     +-- OUT: PipelineConfig used by tools/sampling/runtime
                  |
                  +-- _convert_chat_completion_tools_to_token_generator_tools(...)
                  |     openai_routes.py:2558
                  |       |
                  |       +-- IN: tools=None; valid name regex; tool-call policy
                  |       +-- STRUCTURES: TextGenerationRequestTool dictionaries
                  |       +-- LOGIC: validate names/types/functions; copy descriptions/parameters
                  |       +-- OUT: None for this request
                  |
                  +-- _create_response_format(...)
                  |     openai_routes.py:2632
                  |       |
                  |       +-- IN: response_format=None; structured-output feature flag
                  |       +-- STRUCTURES: TextGenerationResponseFormat
                  |       +-- LOGIC: translate text/json_object/json_schema request
                  |       +-- OUT: None for this request
                  |
                  +-- [OPTIONAL GRAMMAR] _resolve_grammar_constraints(...)
                  |     openai_routes.py:2109
                  |       |
                  |       +-- IN: tools; tool_choice; response_format
                  |       +-- STRUCTURES: grammar tools/schema/flags tuple
                  |       +-- LOGIC: decide tool/schema grammar and start-enforcement mode
                  |       +-- OUT: grammar inputs; branch skipped here
                  |
                  +-- [OPTIONAL GRAMMAR] parser.generate_tool_call_grammar(...)
                  |     |
                  |     +-- IN: tools/schema/tokenizer/backend/tool choice
                  |     +-- STRUCTURES: architecture-specific grammar string
                  |     +-- LOGIC: encode valid tool-call output language
                  |     +-- OUT: grammar text
                  |
                  +-- TextGenerationResponseFormat(type="grammar", ...)
                  |     |
                  |     +-- IN: generated grammar + enforcement flags
                  |     +-- STRUCTURES: response-format dataclass/model
                  |     +-- LOGIC: carry grammar to tokenizer/scheduler
                  |     +-- OUT: constrained response_format
                  |
                  +-- OpenAIChatResponseGenerator(...)
                  |     openai_routes.py:538
                  |       |
                  |       +-- IN: pipeline; stream options; parser/tools/format options
                  |       +-- STRUCTURES: response generator state
                  |       +-- LOGIC: bind MAX token outputs to OpenAI response formatting
                  |       +-- OUT: response_generator
                  |
                  +-- SamplingParams.from_input_and_generation_config(...)
                  |     sampling_params.py
                  |       |
                  |       +-- IN
                  |       |     SamplingParamsInput(top_k=50,temperature=0.2,top_p=0.9,
                  |       |       max_new_tokens=32,seed=42,others=None)
                  |       |     model generation defaults
                  |       +-- STRUCTURES
                  |       |     request/server/model/built-in precedence map
                  |       +-- LOGIC
                  |       |     merge defaults -> normalize greedy/top-k behavior -> validate ranges
                  |       +-- OUT
                  |             sampling_params shown in section 0
                  |
                  +-- TextGenerationRequest(...)
                  |     text_generation.py:336
                  |       |
                  |       +-- IN
                  |       |     request ID/model/messages/media/tools/format/sampling/routing fields
                  |       +-- STRUCTURES
                  |       |     frozen TextGenerationRequest dataclass
                  |       +-- LOGIC
                  |       |     construct object then run __post_init__
                  |       +-- OUT
                  |       |     token_request shown in section 0
                  |       |
                  |       +-- TextGenerationRequest.__post_init__()
                  |             |
                  |             +-- IN: newly constructed request fields
                  |             +-- STRUCTURES: typed message list + media counts
                  |             +-- LOGIC: convert dict messages; reject invalid prompt/media combinations
                  |             +-- OUT: validated immutable token_request
                  |
                  +-- [BRANCH] completion_request.stream
                        |
                        +-- IN: true
                        +-- STRUCTURES: streaming/non-streaming response paths
                        +-- LOGIC: choose SSE path before tokens are visible
                        +-- OUT: streaming path below
</pre>

</details>

## 3. Streaming Admission And API-To-Worker Handoff

<details>
<summary><strong>Expand: Streaming admission and first SSE chunk</strong></summary>

<pre>
openai_create_chat_completion(...)
    |
    +-- [AWAIT] response_generator.stream(token_request)
    |     OpenAIChatResponseGenerator.stream
    |     openai_routes.py:687
    |       |
    |       +-- IN: token_request from section 0
    |       +-- STRUCTURES: TokenGeneratorOutput async generator; OpenAI stream generator
    |       +-- LOGIC: acquire MAX token generator; wrap it in OpenAI SSE formatter
    |       +-- OUT: AsyncGenerator[str | JSONResponse, None]
    |       |
    |       +-- [AWAIT] pipeline.next_token_chunk(token_request)
    |       |     llm.py:294
    |       |       |
    |       |       +-- IN: TextGenerationRequest
    |       |       +-- STRUCTURES: timers; TextContext; detokenizers; worker stream
    |       |       +-- LOGIC: tokenize -> admit -> return API-side output generator
    |       |       +-- OUT: AsyncGenerator[TokenGeneratorOutput, None]
    |       |       |
    |       |       +-- initialize TTFT/ITL/decode timers
    |       |       |     +-- IN: request timestamp and pipeline metrics
    |       |       |     +-- STRUCTURES: StopWatch/metric state
    |       |       |     +-- LOGIC: start latency accounting
    |       |       |     +-- OUT: initialized timer state
    |       |       |
    |       |       +-- optionally create/reset reasoning parser
    |       |       |     +-- IN: configured parser name
    |       |       |     +-- STRUCTURES: ReasoningParser | None
    |       |       |     +-- LOGIC: prepare parser-specific streaming state
    |       |       |     +-- OUT: parser | None; None in concrete example
    |       |       |
    |       |       +-- model_worker.note_awaiting_admission(+1)
    |       |       |     +-- IN: delta=+1
    |       |       |     +-- STRUCTURES: API-side admission gauge/counter
    |       |       |     +-- LOGIC: count request before tokenization and worker handoff
    |       |       |     +-- OUT: None; gauge incremented
    |       |       |
    |       |       +-- [AWAIT] tokenizer.new_context(request)
    |       |       |     tokenizer.py:819
    |       |       |       |
    |       |       |       +-- IN: TextGenerationRequest
    |       |       |       +-- STRUCTURES: prompt text/IDs; grammar state; TokenBuffer; EOS tracker
    |       |       |       +-- LOGIC: create all scheduler-visible request state
    |       |       |       +-- OUT: TextContext shown in section 0
    |       |       |       |
    |       |       |       +-- [AWAIT] _generate_prompt_and_token_ids(request)
    |       |       |       |     tokenizer.py:718
    |       |       |       |       |
    |       |       |       |       +-- IN: prompt=None; two typed messages
    |       |       |       |       +-- STRUCTURES: rendered prompt string + np.ndarray token IDs
    |       |       |       |       +-- LOGIC: choose prompt branch or messages/chat-template branch
    |       |       |       |       +-- OUT: prompt text + concrete 49 token IDs
    |       |       |       |       |
    |       |       |       |       +-- [BRANCH] prompt exists
    |       |       |       |       |     +-- IN: str or Sequence[int]
    |       |       |       |       |     +-- STRUCTURES: direct tokenizer input
    |       |       |       |       |     +-- LOGIC: encode directly; no chat template
    |       |       |       |       |     +-- OUT: prompt + token IDs
    |       |       |       |       |
    |       |       |       |       +-- [BRANCH] prompt is None; use messages
    |       |       |       |             +-- IN: system/user messages
    |       |       |       |             +-- STRUCTURES: model chat template
    |       |       |       |             +-- LOGIC: render roles/content then tokenize
    |       |       |       |             +-- OUT: rendered Llama prompt + token IDs
    |       |       |       |             |
    |       |       |       |             +-- apply_chat_template(...)
    |       |       |       |             |     +-- IN: flattened messages + template options
    |       |       |       |             |     +-- STRUCTURES: Jinja/chat template
    |       |       |       |             |     +-- LOGIC: insert model control/header tokens
    |       |       |       |             |     +-- OUT: rendered prompt string
    |       |       |       |             |
    |       |       |       |             +-- [AWAIT] _encode_chat_prompt(...)
    |       |       |       |                   tokenizer.py:737
    |       |       |       |                     |
    |       |       |       |                     +-- IN: rendered prompt/messages and media
    |       |       |       |                     +-- STRUCTURES: custom encoder or base tokenizer
    |       |       |       |                     +-- LOGIC: architecture-specific encode when available
    |       |       |       |                     +-- OUT: np.ndarray token IDs
    |       |       |       |
    |       |       |       +-- derive json_schema / grammar
    |       |       |       |     +-- IN: request.response_format=None
    |       |       |       |     +-- STRUCTURES: optional schema/grammar strings
    |       |       |       |     +-- LOGIC: select structured-output constraints
    |       |       |       |     +-- OUT: no constraints for concrete request
    |       |       |       |
    |       |       |       +-- GrammarEnforcementState.from_response_format(...)
    |       |       |       |     +-- IN: response_format=None + tokenizer metadata
    |       |       |       |     +-- STRUCTURES: grammar matcher/build state
    |       |       |       |     +-- LOGIC: create constrained-decoding state only when requested
    |       |       |       |     +-- OUT: None
    |       |       |       |
    |       |       |       +-- max_tokens_to_generate(prompt_size, sampling_params, max_length)
    |       |       |       |     tokenizer.py:347
    |       |       |       |       |
    |       |       |       |       +-- IN: prompt_size=49; requested=32; model limit
    |       |       |       |       +-- STRUCTURES: integer length limits
    |       |       |       |       +-- LOGIC: clamp requested generation to remaining context window
    |       |       |       |       +-- OUT: 32
    |       |       |       |
    |       |       |       +-- TokenBuffer(token_ids)
    |       |       |       |     +-- IN: concrete 49 prompt IDs
    |       |       |       |     +-- STRUCTURES: prompt/active/generated token windows
    |       |       |       |     +-- LOGIC: initialize mutable token lifecycle state
    |       |       |       |     +-- OUT: TokenBuffer(prompt=[...], generated=[])
    |       |       |       |
    |       |       |       +-- [AWAIT] create_eos_tracker(request)
    |       |       |       |     tokenizer.py:809
    |       |       |       |       |
    |       |       |       |       +-- IN: stop strings/token IDs + tokenizer EOS IDs
    |       |       |       |       +-- STRUCTURES: EOSToken/stop-sequence tracker
    |       |       |       |       +-- LOGIC: combine model and request stop conditions
    |       |       |       |       +-- OUT: eos_tracker
    |       |       |       |
    |       |       |       +-- TextContext(...)
    |       |       |             +-- IN: IDs/buffer/sampling/EOS/grammar/routing metadata
    |       |       |             +-- STRUCTURES: TextContext dataclass
    |       |       |             +-- LOGIC: package request state for scheduling and model execution
    |       |       |             +-- OUT: context
    |       |       |
    |       |       +-- inject_trace_carrier(context)
    |       |       |     +-- IN: current API trace + context
    |       |       |     +-- STRUCTURES: serialized trace carrier field
    |       |       |     +-- LOGIC: propagate tracing across IPC
    |       |       |     +-- OUT: None; context mutated
    |       |       |
    |       |       +-- create_buffered_detokenizer(tokenizer, ...)
    |       |       |     incremental_detokenizer.py:343
    |       |       |       |
    |       |       |       +-- IN: tokenizer + EOS IDs + output mode
    |       |       |       +-- STRUCTURES: fast/token-healing/fallback detokenizer implementations
    |       |       |       +-- LOGIC: select incremental decoder that protects partial UTF-8
    |       |       |       +-- OUT: content and reasoning detokenizers
    |       |       |
    |       |       +-- [AWAIT] model_worker.stream(context.request_id, context)
    |       |       |     ZmqModelWorkerProxy.stream
    |       |       |     zmq_interface.py:105
    |       |       |       |
    |       |       |       +-- IN: RequestID("req-chat-001") + TextContext
    |       |       |       +-- STRUCTURES: admission lock; bounded request queue;
    |       |       |       |               pending_out_queues; per-request asyncio.Queue
    |       |       |       +-- LOGIC: atomic writable check/register/push; expose drain generator
    |       |       |       +-- OUT: AsyncGenerator[(list[TextGenerationOutput],batch_id),None]
    |       |       |       |
    |       |       |       +-- duplicate request ID check
    |       |       |       |     +-- IN: pending_out_queues keys
    |       |       |       |     +-- LOGIC: reject ambiguous response routing
    |       |       |       |     +-- OUT: continue or RuntimeError
    |       |       |       |
    |       |       |       +-- [AWAIT] _admission_lock
    |       |       |       |     +-- IN: concurrent submitters
    |       |       |       |     +-- STRUCTURES: asyncio.Lock
    |       |       |       |     +-- LOGIC: serialize writable-check plus push
    |       |       |       |     +-- OUT: exclusive admission section
    |       |       |       |
    |       |       |       +-- [AWAIT] request_queue.writable()
    |       |       |       |     +-- IN: bounded ZMQ request queue state
    |       |       |       |     +-- STRUCTURES: queue capacity/backpressure signal
    |       |       |       |     +-- LOGIC: non-blocking admission decision
    |       |       |       |     +-- OUT: true for concrete request
    |       |       |       |
    |       |       |       +-- [BRANCH] queue full
    |       |       |       |     +-- IN: writable=false
    |       |       |       |     +-- STRUCTURES: RequestQueueFull
    |       |       |       |     +-- LOGIC: reject before response state is committed
    |       |       |       |     +-- OUT: HTTP 429 path
    |       |       |       |
    |       |       |       +-- out_queue = asyncio.Queue()
    |       |       |       |     +-- IN: admitted request ID
    |       |       |       |     +-- STRUCTURES: per-request in-process queue
    |       |       |       |     +-- LOGIC: isolate responses for this request
    |       |       |       |     +-- OUT: pending_out_queues["req-chat-001"]
    |       |       |       |
    |       |       |       +-- [AWAIT] request_queue.put(context)
    |       |       |       |     +-- IN: TextContext
    |       |       |       |     +-- STRUCTURES: serialized ZMQ queue payload
    |       |       |       |     +-- LOGIC: transfer request API process -> worker process
    |       |       |       |     +-- OUT: None after successful enqueue
    |       |       |       |
    |       |       |       +-- [BRANCH] put raises
    |       |       |       |     +-- IN: queue transport exception
    |       |       |       |     +-- STRUCTURES: pending map + cancel queue
    |       |       |       |     +-- LOGIC: remove orphan state; best-effort cancel; re-raise
    |       |       |       |     +-- OUT: exception
    |       |       |       |
    |       |       |       +-- return _drain_responses(request_id, out_queue)
    |       |       |             +-- IN: request ID + registered queue
    |       |       |             +-- STRUCTURES: async generator
    |       |       |             +-- LOGIC: defer waiting until consumer iterates
    |       |       |             +-- OUT: response_stream
    |       |       |
    |       |       +-- model_worker.note_awaiting_admission(-1)
    |       |       |     +-- IN: delta=-1 on success or exception
    |       |       |     +-- STRUCTURES: admission gauge
    |       |       |     +-- LOGIC: close API-side awaiting-admission interval
    |       |       |     +-- OUT: None; gauge decremented
    |       |       |
    |       |       +-- return TokenGeneratorPipeline._generate async generator
    |       |             +-- IN: context + response_stream + detokenizers/parsers
    |       |             +-- STRUCTURES: nested async generator closure
    |       |             +-- LOGIC: defer response processing until iteration
    |       |             +-- OUT: token_generator
    |       |
    |       +-- return OpenAIChatResponseGenerator._stream async generator
    |             +-- IN: token_generator
    |             +-- STRUCTURES: SSE JSON formatter generator
    |             +-- LOGIC: defer OpenAI chunk serialization until iteration
    |             +-- OUT: stream
    |
    +-- [AWAIT] _start_stream(stream)
    |     openai_routes.py:185
    |       |
    |       +-- IN: OpenAI SSE async generator
    |       +-- STRUCTURES: first item + chained async generator
    |       +-- LOGIC: pull first result before committing HTTP 200/SSE headers
    |       +-- OUT: (error=None, token_stream including first item)
    |       |
    |       +-- [AWAIT] stream.__anext__()
    |             +-- IN: unstarted generator chain
    |             +-- STRUCTURES: OpenAI _stream -> pipeline _generate -> _drain_responses
    |             +-- LOGIC: start chain and wait for first worker result
    |             +-- OUT: first SSE payload or pre-header error response
    |
    +-- EventSourceResponse(token_stream, ping=100000, sep="\n")
          |
          +-- IN: primed token stream
          +-- STRUCTURES: SSE response/body iterator
          +-- LOGIC: expose stream to ASGI
          +-- OUT: EventSourceResponse
          |
          +-- request_session adds X-Request-ID
          |     +-- OUT: response header X-Request-ID=req-chat-001
          |
          +-- ASGI starts HTTP response and iterates token_stream
                +-- OUT: SSE frames sent to client
</pre>

</details>

## 4. Non-Streaming Branch

<details>
<summary><strong>Expand: Non-streaming aggregation path</strong></summary>

<pre>
openai_create_chat_completion(...)
    |
    +-- [BRANCH] completion_request.stream == false
          |
          +-- IN: same token_request with stream=false
          +-- STRUCTURES: list[TokenGeneratorOutput] + final OpenAI response models
          +-- LOGIC: consume full generation before returning HTTP response
          +-- OUT: CreateChatCompletionResponse
          |
          +-- [AWAIT] response_generator.complete([token_request])
                OpenAIChatResponseGenerator.complete
                openai_routes.py:1159
                  |
                  +-- IN: one-item request list
                  +-- STRUCTURES: outputs list; choices; usage; tool parser
                  +-- LOGIC: collect -> concatenate -> format final choices/usage
                  +-- OUT: CreateChatCompletionResponse
                  |
                  +-- [AWAIT] pipeline.all_tokens(token_request)
                  |     llm.py:678
                  |       |
                  |       +-- IN: TextGenerationRequest
                  |       +-- STRUCTURES: next_token_chunk async generator + output list
                  |       +-- LOGIC: iterate until terminal output
                  |       +-- OUT: list[TokenGeneratorOutput]
                  |       |
                  |       +-- [AWAIT] pipeline.next_token_chunk(token_request)
                  |       |     +-- OUT: same admitted generator as streaming path
                  |       |
                  |       +-- [ITERATE] generator until done
                  |             +-- IN: active output generator
                  |             +-- LOGIC: append every yielded chunk
                  |             +-- OUT: complete ordered chunk list
                  |
                  +-- concatenate content/reasoning
                  |     +-- IN: decoded fragments
                  |     +-- OUT: full assistant text and optional reasoning text
                  |
                  +-- process log probabilities
                  |     +-- IN: per-token logprob records or None
                  |     +-- OUT: OpenAI logprobs object or None
                  |
                  +-- derive finish_reason
                  |     +-- IN: terminal generation status
                  |     +-- OUT: "stop", "length", "tool_calls", or error mapping
                  |
                  +-- optionally parse complete tool calls
                  |     +-- IN: full generated text + parser/tools
                  |     +-- OUT: parsed tool calls or plain content
                  |
                  +-- ChatCompletionResponseChoice(...)
                  |     +-- IN: message/content/tool calls/finish reason/logprobs
                  |     +-- OUT: final choice model
                  |
                  +-- CompletionUsage(...)
                  |     +-- IN: prompt/generated token counts
                  |     +-- OUT: usage model
                  |
                  +-- CreateChatCompletionResponse(...)
                        +-- IN: ID/model/choices/usage
                        +-- STRUCTURES: OpenAI Pydantic response model
                        +-- LOGIC: validate final response shape
                        +-- OUT: FastAPI-serializable JSON response
</pre>

</details>

## 5. Worker Scheduler, Batch Construction, And Llama Execution

<details>
<summary><strong>Expand: Worker scheduler through Llama execution</strong></summary>

<pre>
[IPC] TextContext arrives on worker request_queue
    |
    +-- IN: serialized context from API process
    +-- STRUCTURES: worker-side MAXPullQueue[TextContext]
    +-- LOGIC: queue delivery activates independent worker loop
    +-- OUT: context available to scheduler drain
    |
    +-- [BACKGROUND] ModelWorker.run(...) loop
          |
          +-- IN: live scheduler and WorkerQueues
          +-- STRUCTURES: heartbeat + idle backoff state
          +-- LOGIC: repeatedly call run_iteration
          +-- OUT: scheduler progress and worker responses
          |
          +-- [CALL] scheduler.run_iteration()
                TokenGenerationScheduler.run_iteration
                text_generation_scheduler.py:205
                  |
                  +-- IN: request/cancel/response queues + batch constructor + pipeline
                  +-- STRUCTURES: TextGenerationInputs; SchedulerResult; SchedulerProgress
                  +-- LOGIC: retrieve -> batch -> execute -> advance/release -> publish
                  +-- OUT: MADE_PROGRESS for the concrete request
                  |
                  +-- [CALL] _retrieve_pending_requests()
                  |     text_generation_scheduler.py:161
                  |       |
                  |       +-- IN: worker request_queue
                  |       +-- STRUCTURES: list[TextContext] + batch constructor queues
                  |       +-- LOGIC: bounded drain; enqueue every context
                  |       +-- OUT: None; constructor owns pending request
                  |       |
                  |       +-- drain_queue(request_queue, max_items)
                  |       |     +-- IN: queue + retrieval cap
                  |       |     +-- STRUCTURES: drained context list
                  |       |     +-- LOGIC: take currently available items without blocking loop
                  |       |     +-- OUT: [context]
                  |       |
                  |       +-- batch_constructor.enqueue_new_request(ctx)
                  |             text_batch_constructor.py:677
                  |               |
                  |               +-- IN: TextContext
                  |               +-- STRUCTURES: grammar pending map; CE pending pool; replica queues
                  |               +-- LOGIC: grammar gate then scheduler admission
                  |               +-- OUT: None; context stored in pending/replica queue
                  |               |
                  |               +-- optional submit_grammar_build(ctx)
                  |               |     +-- IN: context with unresolved grammar
                  |               |     +-- STRUCTURES: async matcher build state
                  |               |     +-- LOGIC: begin grammar compilation before batching
                  |               |     +-- OUT: context parked until ready
                  |               |
                  |               +-- _admit_request(ctx, replica_idx)
                  |                     text_batch_constructor.py:703
                  |                       |
                  |                       +-- IN: context + optional caller replica
                  |                       +-- STRUCTURES: DP CE pending pool; round-robin selector
                  |                       +-- LOGIC: defer fresh prefill for DP balance or choose replica
                  |                       +-- OUT: pooled request or bound request
                  |                       |
                  |                       +-- _bind_request(ctx, replica_idx)
                  |                             text_batch_constructor.py:725
                  |                               |
                  |                               +-- IN: context + selected replica 0
                  |                               +-- STRUCTURES: replica.ce_reqs; replica.tg_reqs
                  |                               +-- LOGIC: generated_length==0 -> CE; otherwise TG
                  |                               +-- OUT: replica.ce_reqs["req-chat-001"] = context
                  |
                  +-- [CALL] batch_constructor.construct_batch()
                  |     text_batch_constructor.py:1845
                  |       |
                  |       +-- IN: all pending CE/TG/grammar/onload state
                  |       +-- STRUCTURES: ReplicaBatch; token budgets; TextGenerationInputs
                  |       +-- LOGIC: settle state -> choose priorities -> pack each replica
                  |       +-- OUT: TextGenerationInputs(batches=[[context]], batch_type=CE)
                  |       |
                  |       +-- kv_cache.poll_transfers()
                  |       |     +-- IN: pending KV onload/offload transfers
                  |       |     +-- STRUCTURES: KV transfer events
                  |       |     +-- LOGIC: settle completed asynchronous transfers
                  |       |     +-- OUT: None; cache transfer state updated
                  |       |
                  |       +-- _readmit_completed_onloads()
                  |       |     +-- IN: _onloading_reqs
                  |       |     +-- STRUCTURES: transfer events + replica CE queues
                  |       |     +-- LOGIC: move settled requests to front of CE queue
                  |       |     +-- OUT: None
                  |       |
                  |       +-- _promote_grammar_ready_requests()
                  |       |     +-- IN: _grammar_pending map
                  |       |     +-- STRUCTURES: grammar matcher build results
                  |       |     +-- LOGIC: install ready matcher; admit request or record failure
                  |       |     +-- OUT: promoted requests + failure list
                  |       |
                  |       +-- _plan_ce_step()
                  |       |     +-- IN: per-replica CE loads and DP balancing settings
                  |       |     +-- STRUCTURES: CE weights; quotas; deferred replica set
                  |       |     +-- LOGIC: choose balanced prefill work and late-bind pooled requests
                  |       |     +-- OUT: _ce_step_quota/_ce_deferred_replicas
                  |       |
                  |       +-- _should_backfill_ce()
                  |       |     +-- IN: pending CE/TG work + coalescing/cadence settings
                  |       |     +-- STRUCTURES: held-step counter
                  |       |     +-- LOGIC: decide whether decode batch may absorb prefill
                  |       |     +-- OUT: bool _ce_backfill_open
                  |       |
                  |       +-- _identify_priority(replica_idx)
                  |       |     +-- IN: replica 0 CE/TG queues + scheduling strategy
                  |       |     +-- STRUCTURES: RequestType.CE/TG/None
                  |       |     +-- LOGIC: choose work class for next batch
                  |       |     +-- OUT: RequestType.CE for fresh request
                  |       |
                  |       +-- _construct_replica_batch(replica_idx, priority_override)
                  |       |     text_batch_constructor.py:1534
                  |       |       |
                  |       |       +-- IN: replica 0 + CE priority
                  |       |       +-- STRUCTURES: ReplicaBatch(batch={}, token_budget=...)
                  |       |       +-- LOGIC: create budget; add CE or TG; optional mixed backfill
                  |       |       +-- OUT: ReplicaBatch containing context
                  |       |       |
                  |       |       +-- _create_new_token_budget(ce_capacity)
                  |       |       |     +-- IN: optional CE quota
                  |       |       |     +-- STRUCTURES: token/batch/chunk capacity counters
                  |       |       |     +-- LOGIC: initialize per-step admission budget
                  |       |       |     +-- OUT: token_budget
                  |       |       |
                  |       |       +-- _add_ce_requests(batch, replica_idx)
                  |       |       |     +-- IN: empty ReplicaBatch + replica CE queue
                  |       |       |     +-- STRUCTURES: KV allocations; LoRA state; budget statuses
                  |       |       |     +-- LOGIC: oldest-first prefill admission within budgets/cache
                  |       |       |     +-- OUT: batch.batch contains req-chat-001
                  |       |       |
                  |       |       +-- _add_tg_requests(batch, replica_idx)
                  |       |             +-- IN: ReplicaBatch + replica TG queue
                  |       |             +-- STRUCTURES: decode candidates; KV blocks; budgets
                  |       |             +-- LOGIC: admit decode work; preempt under KV/LoRA pressure
                  |       |             +-- OUT: decode contexts added; not used on first prefill
                  |       |
                  |       +-- TextGenerationInputs(batches=replica_batches)
                  |       |     +-- IN: list[list[TextContext]]
                  |       |     +-- STRUCTURES: input token counts + BatchType
                  |       |     +-- LOGIC: freeze this step's workload metadata
                  |       |     +-- OUT: inputs
                  |       |
                  |       +-- optional _dp_padder.pad_batch(inputs)
                  |             +-- IN: uneven DP replica batches
                  |             +-- STRUCTURES: padding contexts/batch metadata
                  |             +-- LOGIC: align replica shapes for lockstep execution
                  |             +-- OUT: padded inputs; skipped for one replica
                  |
                  +-- [CALL] batch_constructor.take_grammar_failed()
                  |     text_batch_constructor.py:907
                  |       |
                  |       +-- IN: accumulated grammar failures
                  |       +-- STRUCTURES: list[(RequestID,error)]
                  |       +-- LOGIC: return-and-clear failures
                  |       +-- OUT: [] for concrete request
                  |       |
                  |       +-- SchedulerResult.failed(error)
                  |             +-- IN: grammar error
                  |             +-- STRUCTURES: terminal SchedulerResult
                  |             +-- LOGIC: encode worker-side request failure
                  |             +-- OUT: failed result for response queue
                  |             |
                  |             +-- response_queue.put_nowait(...)
                  |                   +-- IN: request_id -> failed SchedulerResult
                  |                   +-- STRUCTURES: worker MAXPushQueue
                  |                   +-- LOGIC: publish failure without model execution
                  |                   +-- OUT: None
                  |
                  +-- [BRANCH] no executable inputs
                  |     |
                  |     +-- IN: empty batches + no overlap/required empty batch
                  |     +-- STRUCTURES: SchedulerProgress
                  |     +-- LOGIC: skip pipeline execution
                  |     +-- OUT: NO_PROGRESS
                  |
                  +-- [CALL] _schedule(inputs)
                        text_generation_scheduler.py:299
                          |
                          +-- IN: TextGenerationInputs
                          +-- STRUCTURES: batch ID; pipeline responses; SchedulerResult map
                          +-- LOGIC: execute -> advance/release -> publish
                          +-- OUT: number of scheduled contexts
                          |
                          +-- batch_id = next monotonic scheduler counter
                          |     +-- IN: previous counter
                          |     +-- OUT: 1 for illustrative first batch
                          |
                          +-- [CALL] pipeline.execute(inputs)
                          |     TextGenerationPipeline.execute
                          |     text_generation.py:515
                          |       |
                          |       +-- IN: TextGenerationInputs
                          |       +-- STRUCTURES: request-validation context + pipeline metrics
                          |       +-- LOGIC: validate context lifecycle then call _execute
                          |       +-- OUT: dict[RequestID,TextGenerationOutput]
                          |       |
                          |       +-- [CALL] _execute(inputs)
                          |             text_generation.py:527
                          |               |
                          |               +-- IN: scheduled replica batches
                          |               +-- STRUCTURES: model inputs; flat batch; sampler; logits; tokens
                          |               +-- LOGIC: prepare -> forward -> process logits -> sample -> update
                          |               +-- OUT: {"req-chat-001": TextGenerationOutput(...)}
                          |               |
                          |               +-- [CALL] prepare_batch(inputs.batches)
                          |               |     text_generation.py:399
                          |               |       |
                          |               |       +-- IN: [[context]]
                          |               |       +-- STRUCTURES: flat batch; grammar bitmask; KV inputs; Llama3Inputs
                          |               |       +-- LOGIC: convert scheduler contexts to architecture model buffers
                          |               |       +-- OUT: (model_inputs, flat_batch, bitmask)
                          |               |       |
                          |               |       +-- _maybe_sort_loras(replica_batches)
                          |               |       |     +-- IN: batches with optional LoRA IDs
                          |               |       |     +-- STRUCTURES: reordered context lists
                          |               |       |     +-- LOGIC: group/sort only when LoRA execution needs it
                          |               |       |     +-- OUT: unchanged [[context]] here
                          |               |       |
                          |               |       +-- flatten2d(replica_batches)
                          |               |       |     +-- IN: [[context]]
                          |               |       |     +-- STRUCTURES: list[TextContext]
                          |               |       |     +-- LOGIC: concatenate replica lists
                          |               |       |     +-- OUT: [context]
                          |               |       |
                          |               |       +-- initialize_bitmask(flat_batch)
                          |               |       |     +-- IN: [context]
                          |               |       |     +-- STRUCTURES: grammar bitmask Buffer | None
                          |               |       |     +-- LOGIC: allocate mask storage only for constrained outputs
                          |               |       |     +-- OUT: None for concrete request
                          |               |       |
                          |               |       +-- update_for_structured_output(flat_batch, bitmask)
                          |               |       |     +-- IN: contexts + optional mask
                          |               |       |     +-- STRUCTURES: GrammarEnforcementState/matcher
                          |               |       |     +-- LOGIC: fill allowed-token masks for current FSM state
                          |               |       |     +-- OUT: unchanged None mask here
                          |               |       |
                          |               |       +-- kv_manager.runtime_inputs(replica_batches)
                          |               |       |     +-- IN: scheduled contexts and cache block tables
                          |               |       |     +-- STRUCTURES: KVCacheInputs[Buffer,Buffer]
                          |               |       |     +-- LOGIC: stage cache metadata needed by graph
                          |               |       |     +-- OUT: kv_cache_inputs
                          |               |       |
                          |               |       +-- _pipeline_model.prepare_initial_token_inputs(...)
                          |               |             |
                          |               |             +-- IN: replica batches + KV inputs + return_n_logits
                          |               |             +-- STRUCTURES: architecture batch processor
                          |               |             +-- LOGIC: delegate architecture-specific input staging
                          |               |             +-- OUT: Llama3Inputs
                          |               |             |
                          |               |             +-- LlamaModelBase.prepare_initial_token_inputs(...)
                          |               |             |     llama3/model.py:168
                          |               |             |       |
                          |               |             |       +-- IN: same batches/KV/logit count
                          |               |             |       +-- STRUCTURES: GraphPipelineModelWithKVCache base method
                          |               |             |       +-- LOGIC: delegate and narrow result type
                          |               |             |       +-- OUT: asserted Llama3Inputs
                          |               |             |       |
                          |               |             |       +-- super().prepare_initial_token_inputs(...)
                          |               |             |             pipeline_model.py:623
                          |               |             |               |
                          |               |             |               +-- IN: replica batches + KV inputs + logit count
                          |               |             |               +-- STRUCTURES: bound architecture batch_processor
                          |               |             |               +-- LOGIC: forward common model request to processor
                          |               |             |               +-- OUT: ModelInputs implemented as Llama3Inputs
                          |               |             |
                          |               |             +-- batch_processor.prepare_initial_token_inputs(...)
                          |               |                   llama3/batch_processor.py:99
                          |               |                     |
                          |               |                     +-- IN: [[context]], KV inputs, return_n_logits=1
                          |               |                     +-- STRUCTURES: stager buffers; DP split metadata
                          |               |                     +-- LOGIC: flatten -> stage ragged tokens -> package inputs
                          |               |                     +-- OUT: Llama3Inputs
                          |               |                     |
                          |               |                     +-- _stage_ragged_token_inputs(context_batch)
                          |               |                     |     llama3/batch_processor.py:56
                          |               |                     |       |
                          |               |                     |       +-- IN: [context] with active_length=49
                          |               |                     |       +-- STRUCTURES: pinned host/device token and offset buffers
                          |               |                     |       +-- LOGIC: concatenate active tokens; cumulative row offsets
                          |               |                     |       +-- OUT: tokens Buffer(49); offsets Buffer([0,49]); host offsets
                          |               |                     |
                          |               |                     +-- Buffer.from_numpy([return_n_logits])
                          |               |                     |     +-- IN: 1
                          |               |                     |     +-- STRUCTURES: int64 Buffer
                          |               |                     |     +-- LOGIC: graph-readable scalar control
                          |               |                     |     +-- OUT: Buffer([1])
                          |               |                     |
                          |               |                     +-- _make_inputs(...)
                          |               |                           llama3/batch_processor.py:166
                          |               |                             |
                          |               |                             +-- IN: tokens/offsets/logit/signal/KV/DP buffers
                          |               |                             +-- STRUCTURES: Llama3Inputs dataclass
                          |               |                             +-- LOGIC: assign exact graph-signature fields
                          |               |                             +-- OUT: Llama3Inputs shown in section 0
                          |               |
                          |               +-- FusedSamplingProcessor(...)
                          |               |     sampling_logits_processor.py:97
                          |               |       |
                          |               |       +-- IN: flat_batch + devices + sampler + grammar bitmask
                          |               |       +-- STRUCTURES: sampling graph inputs and logit-offset metadata
                          |               |       +-- LOGIC: prepare fused per-request sampling controls
                          |               |       +-- OUT: sampling_processor
                          |               |
                          |               +-- [CALL] _launch_forward_pass(model_inputs, flat_batch)
                          |               |     text_generation.py:655
                          |               |       |
                          |               |       +-- IN: Llama3Inputs + [context]
                          |               |       +-- STRUCTURES: pipeline model + telemetry spans
                          |               |       +-- LOGIC: execute architecture model and record timing
                          |               |       +-- OUT: ModelOutputs
                          |               |       |
                          |               |       +-- _pipeline_model.execute(model_inputs)
                          |               |             +-- IN: Llama3Inputs
                          |               |             +-- STRUCTURES: LlamaModelBase wrapper
                          |               |             +-- LOGIC: architecture runtime dispatch
                          |               |             +-- OUT: ModelOutputs
                          |               |             |
                          |               |             +-- LlamaModelBase.execute(model_inputs)
                          |               |                   llama3/model.py:183
                          |               |                     |
                          |               |                     +-- IN: asserted Llama3Inputs with non-None KV inputs
                          |               |                     +-- STRUCTURES: ordered buffer tuple + compiled Model
                          |               |                     +-- LOGIC: flatten input object -> run graph -> process outputs
                          |               |                     +-- OUT: ModelOutputs
                          |               |                     |
                          |               |                     +-- model_inputs.buffers
                          |               |                     |     +-- IN: Llama3Inputs fields
                          |               |                     |     +-- STRUCTURES: tuple[Buffer,...]
                          |               |                     |     +-- LOGIC: order tokens/offsets/logit/DP-or-signal/KV buffers
                          |               |                     |     +-- OUT: exact compiled graph argument tuple
                          |               |                     |
                          |               |                     +-- [COMPILED] self.model.execute(*buffers)
                          |               |                     |     +-- IN: ordered device buffers
                          |               |                     |     +-- STRUCTURES: compiled Llama graph; MAX runtime; kernels
                          |               |                     |     +-- LOGIC: embeddings -> transformer/KV/attention/MLP/norm/head
                          |               |                     |     +-- OUT: raw output Buffer sequence including logits
                          |               |                     |
                          |               |                     +-- batch_processor.process_outputs(model_outputs)
                          |               |                           llama3/batch_processor.py:153
                          |               |                             |
                          |               |                             +-- IN: raw graph output buffers
                          |               |                             +-- STRUCTURES: ModelOutputs
                          |               |                             +-- LOGIC: map ragged KV graph outputs to named result fields
                          |               |                             +-- OUT: ModelOutputs(logits=...,logit_offsets=...)
                          |               |                             |
                          |               |                             +-- process_ragged_kv_outputs(...)
                          |               |                                   +-- IN: raw buffers + return flags
                          |               |                                   +-- STRUCTURES: logits/offsets/hidden-state slots
                          |               |                                   +-- LOGIC: interpret graph output ordering
                          |               |                                   +-- OUT: ModelOutputs
                          |               |
                          |               +-- sampling_processor.logits_for_sampling(model_outputs)
                          |               |     +-- IN: ModelOutputs
                          |               |     +-- STRUCTURES: logits Buffer + optional offsets
                          |               |     +-- LOGIC: choose last-token or requested multi-token logits
                          |               |     +-- OUT: logits shape (1,128256) + offsets
                          |               |
                          |               +-- apply_logits_processors(context_batch, logits, offsets, ...)
                          |               |     sampling/logits_processor.py:27
                          |               |       |
                          |               |       +-- IN: [context] + batch logits + offsets + fused processor
                          |               |       +-- STRUCTURES: ProcessorInputs; BatchProcessorInputs
                          |               |       +-- LOGIC: mutate per-request logit slices then whole batch
                          |               |       +-- OUT: None; logits modified in place
                          |               |       |
                          |               |       +-- request-specific processors
                          |               |       |     +-- IN: context sampling_params.logits_processors
                          |               |       |     +-- OUT: adjusted request logit slice
                          |               |       |
                          |               |       +-- FusedSamplingProcessor.__call__(...)
                          |               |             +-- IN: full logits/offsets/context batch
                          |               |             +-- STRUCTURES: temperature/top-k/top-p/min-p/penalty arrays
                          |               |             +-- LOGIC: apply batch sampling controls
                          |               |             +-- OUT: sampled token device buffer
                          |               |             |
                          |               |             +-- _sample_logits(...)
                          |               |                   +-- IN: processed logits + graph inputs
                          |               |                   +-- STRUCTURES: compiled sampler
                          |               |                   +-- LOGIC: select one token under sampling policy
                          |               |                   +-- OUT: illustrative token ID [83807]
                          |               |                   |
                          |               |                   +-- [COMPILED] sampler(*graph_inputs)
                          |               |                         +-- IN: logits + sampling parameter buffers + RNG state
                          |               |                         +-- STRUCTURES: compiled sampling graph
                          |               |                         +-- LOGIC: normalize/filter/sample
                          |               |                         +-- OUT: generated token device Buffer([83807])
                          |               |
                          |               +-- copy generated tokens device -> host
                          |               |     +-- IN: generated token device buffer
                          |               |     +-- STRUCTURES: host numpy/int array
                          |               |     +-- LOGIC: make token IDs available to Python context update
                          |               |     +-- OUT: next_tokens=[83807]
                          |               |
                          |               +-- update_context_and_prepare_responses(...)
                          |               |     pipeline_variants/utils.py:157
                          |               |       |
                          |               |       +-- IN: [context] + next_tokens + optional logprobs/steps
                          |               |       +-- STRUCTURES: mutable TextContext + TextGenerationOutput map
                          |               |       +-- LOGIC: append tokens -> advance grammar/status -> build outputs
                          |               |       +-- OUT: {req-chat-001: TextGenerationOutput(tokens=[83807],final_status=ACTIVE)}
                          |               |       |
                          |               |       +-- context.advance_token_buffer(...)
                          |               |       |     +-- IN: token 83807 + generated steps
                          |               |       |     +-- STRUCTURES: TokenBuffer
                          |               |       |     +-- LOGIC: append generated token and advance processed windows
                          |               |       |     +-- OUT: context.tokens.generated=[83807]
                          |               |       |
                          |               |       +-- context.advance_fsm(...)
                          |               |       |     +-- IN: generated token + grammar state
                          |               |       |     +-- STRUCTURES: grammar FSM | None
                          |               |       |     +-- LOGIC: consume token in constrained decoder
                          |               |       |     +-- OUT: unchanged None grammar here
                          |               |       |
                          |               |       +-- context.to_generation_output()
                          |               |             +-- IN: updated context delta/status/logprobs
                          |               |             +-- STRUCTURES: TextGenerationOutput
                          |               |             +-- LOGIC: expose only response-ready generation delta
                          |               |             +-- OUT: TextGenerationOutput(tokens=[83807],final_status=ACTIVE)
                          |               |
                          |               +-- kv_manager.step(ctx)
                          |                     +-- IN: updated context and cache allocation
                          |                     +-- STRUCTURES: paged KV cache state
                          |                     +-- LOGIC: commit cache progress for this model step
                          |                     +-- OUT: None
                          |
                          +-- filter responses for released requests
                          |     +-- IN: pipeline response map + cancellation/release state
                          |     +-- STRUCTURES: filtered dict
                          |     +-- LOGIC: suppress outputs for no-longer-owned requests
                          |     +-- OUT: response for req-chat-001 retained
                          |
                          +-- batch_constructor.advance_requests(inputs)
                          |     text_batch_constructor.py:808
                          |       |
                          |       +-- IN: just-executed inputs
                          |       +-- STRUCTURES: CE/TG replica queues + token windows
                          |       +-- LOGIC: incomplete prefill -> CE; complete prefill/decode -> TG
                          |       +-- OUT: context moved/requeued for next iteration
                          |
                          +-- release_request(request_id) when response.is_done
                          |     text_batch_constructor.py:913
                          |       |
                          |       +-- IN: terminal RequestID
                          |       +-- STRUCTURES: queue maps; grammar build; KV allocation
                          |       +-- LOGIC: remove all scheduler state and release cache
                          |       +-- OUT: None; skipped while ACTIVE
                          |
                          +-- response_queue.put_nowait({request_id: SchedulerResult(...)})
                          |     |
                          |     +-- IN: TextGenerationOutput + batch_id=1
                          |     +-- STRUCTURES: SchedulerResult + worker MAXPushQueue
                          |     +-- LOGIC: publish result worker process -> API process
                          |     +-- OUT: None after enqueue
                          |
                          +-- scheduler_logger.log_metrics(...)
                          |     +-- IN: batch/latency/KV/scheduler state
                          |     +-- STRUCTURES: telemetry counters/histograms
                          |     +-- LOGIC: record iteration observability
                          |     +-- OUT: None
                          |
                          +-- get_cancelled_reqs(cancel_queue)
                          |     +-- IN: worker cancel queue
                          |     +-- STRUCTURES: set/list[RequestID]
                          |     +-- LOGIC: drain cancellation messages
                          |     +-- OUT: [] for concrete request
                          |     |
                          |     +-- SchedulerResult.cancelled()
                          |           +-- IN: cancelled request
                          |           +-- STRUCTURES: terminal cancelled result
                          |           +-- LOGIC: notify API consumer after cleanup
                          |           +-- OUT: cancellation response
                          |           |
                          |           +-- response_queue.put_nowait(...)
                          |                 +-- IN: request_id -> cancelled SchedulerResult
                          |                 +-- STRUCTURES: worker MAXPushQueue
                          |                 +-- LOGIC: publish terminal cancellation
                          |                 +-- OUT: None
                          |
                          +-- return SchedulerProgress.MADE_PROGRESS
                                +-- IN: successful retrieve/batch/execute/publish work
                                +-- STRUCTURES: enum value
                                +-- LOGIC: reset worker idle backoff
                                +-- OUT: MADE_PROGRESS
</pre>

</details>

## 6. Worker Response Back To The Client

<details>
<summary><strong>Expand: Response routing, detokenization, and OpenAI SSE output</strong></summary>

<pre>
[IPC] SchedulerResult arrives on API response_queue
    |
    +-- IN: {RequestID("req-chat-001"): SchedulerResult(output,batch_id=1)}
    +-- STRUCTURES: API-side MAXPullQueue
    +-- LOGIC: transport worker result to proxy background task
    +-- OUT: response available to response_worker
    |
    +-- [BACKGROUND] ZmqModelWorkerProxy.response_worker()
    |     zmq_interface.py:247
    |       |
    |       +-- IN: shared response_queue + pending_out_queues map
    |       +-- STRUCTURES: per-request asyncio.Queue objects
    |       +-- LOGIC: demultiplex response by request ID
    |       +-- OUT: None; per-request queue receives result
    |       |
    |       +-- [AWAIT] response_queue.get()
    |       |     +-- IN: shared worker-response queue
    |       |     +-- OUT: result map for req-chat-001
    |       |
    |       +-- pending_out_queues[request_id]
    |       |     +-- IN: RequestID("req-chat-001")
    |       |     +-- OUT: registered out_queue
    |       |
    |       +-- [AWAIT] out_queue.put(response)
    |             +-- IN: SchedulerResult
    |             +-- STRUCTURES: asyncio.Queue
    |             +-- LOGIC: wake only this request's consumer
    |             +-- OUT: None
    |
    +-- ZmqModelWorkerProxy._drain_responses(request_id, out_queue)
    |     zmq_interface.py:167
    |       |
    |       +-- IN: req-chat-001 + per-request queue
    |       +-- STRUCTURES: SchedulerResult accumulator + async generator
    |       +-- LOGIC: wait/coalesce/yield until terminal result; always clean map
    |       +-- OUT: yields ([TextGenerationOutput], batch_id)
    |       |
    |       +-- [AWAIT] out_queue.get()
    |       |     +-- IN: routed queue
    |       |     +-- OUT: SchedulerResult(is_done=False,result=&lt;token 83807 output&gt;,batch_id=1)
    |       |
    |       +-- coalesce already-buffered results
    |       |     +-- IN: first result + immediately available queue items
    |       |     +-- STRUCTURES: list[TextGenerationOutput]
    |       |     +-- LOGIC: reduce coroutine wakeups while preserving order
    |       |     +-- OUT: [output]
    |       |
    |       +-- yield (outputs, batch_id)
    |       |     +-- IN: [output], 1
    |       |     +-- OUT: worker response tuple to pipeline generator
    |       |
    |       +-- [BRANCH] result.is_done
    |       |     +-- IN: ACTIVE now; terminal later
    |       |     +-- LOGIC: continue while active; stop on terminal status
    |       |     +-- OUT: next wait or generator completion
    |       |
    |       +-- [BRANCH] consumer abandons/errors
    |       |     +-- IN: GeneratorExit/cancellation/exception
    |       |     +-- STRUCTURES: worker cancel queue
    |       |     +-- LOGIC: send best-effort cancel and re-raise
    |       |     +-- OUT: cancelled worker request
    |       |
    |       +-- finally remove pending_out_queues[request_id]
    |             +-- IN: terminal/abandoned generator
    |             +-- OUT: no orphan per-request queue
    |
    +-- TokenGeneratorPipeline._generate(...)
    |     llm.py:448 nested in next_token_chunk
    |       |
    |       +-- IN: response tuples + context + detokenizers/parsers
    |       +-- STRUCTURES: TextGenerationOutput; TokenGeneratorOutput; timers
    |       +-- LOGIC: merge -> decode -> apply stop/reasoning/logprobs -> yield
    |       +-- OUT: TokenGeneratorOutput chunks
    |       |
    |       +-- TextGenerationOutput.merge(responses)
    |       |     +-- IN: one or more output deltas
    |       |     +-- STRUCTURES: merged token/status/logprob output
    |       |     +-- LOGIC: preserve ordered deltas from coalesced scheduler results
    |       |     +-- OUT: merged output tokens=[83807]
    |       |
    |       +-- optional reasoning_parser.stream(...)
    |       |     +-- IN: generated token/text stream
    |       |     +-- STRUCTURES: reasoning/content token partitions
    |       |     +-- LOGIC: split model-specific thinking markers
    |       |     +-- OUT: content/reasoning tokens; skipped here
    |       |
    |       +-- [AWAIT] content_detokenizer.decode([83807])
    |       |     +-- IN: illustrative token IDs [83807]
    |       |     +-- STRUCTURES: incremental tokenizer byte/text buffer
    |       |     +-- LOGIC: decode only safe complete text suffix
    |       |     +-- OUT: illustrative "KV"
    |       |
    |       +-- [AWAIT] reasoning_detokenizer.decode(tokens)
    |       |     +-- IN: reasoning token IDs; [] here
    |       |     +-- OUT: ""
    |       |
    |       +-- eos_tracker.is_eos_from_string(decoded_text)
    |       |     +-- IN: decoded fragment + configured stop strings
    |       |     +-- STRUCTURES: incremental stop matcher
    |       |     +-- LOGIC: detect text stop boundary not visible as one token
    |       |     +-- OUT: false for "KV"
    |       |
    |       +-- optional logprob decoding
    |       |     +-- IN: token/logprob arrays when requested
    |       |     +-- OUT: decoded top-logprob records; skipped because logprobs=0
    |       |
    |       +-- yield TokenGeneratorOutput(...)
    |             +-- IN: decoded content/reasoning/status/token counts/logprobs
    |             +-- STRUCTURES: TokenGeneratorOutput dataclass
    |             +-- LOGIC: expose API-facing generation delta
    |             +-- OUT: TokenGeneratorOutput(decoded_tokens="KV",status=ACTIVE,...)
    |
    +-- [BRANCH] streaming
    |     |
    |     +-- OpenAIChatResponseGenerator._stream(...)
    |           openai_routes.py:698
    |             |
    |             +-- IN: TokenGeneratorOutput stream
    |             +-- STRUCTURES: OpenAI delta/choice/response models
    |             +-- LOGIC: transform each MAX chunk to SSE JSON
    |             +-- OUT: JSON payload strings then "[DONE]"
    |             |
    |             +-- optional parser.parse_delta(...)
    |             |     +-- IN: decoded fragment + parser state
    |             |     +-- OUT: incremental tool-call delta or plain content
    |             |
    |             +-- get_finish_reason_from_status(status)
    |             |     +-- IN: ACTIVE/END_OF_SEQUENCE/MAXIMUM_LENGTH/etc.
    |             |     +-- OUT: None while active; OpenAI finish_reason when terminal
    |             |
    |             +-- ChatCompletionStreamResponseDelta(...)
    |             |     +-- IN: role/content/reasoning/tool delta
    |             |     +-- OUT: delta model
    |             |
    |             +-- ChatCompletionStreamResponseChoice(...)
    |             |     +-- IN: index + delta + finish reason + logprobs
    |             |     +-- OUT: choice model
    |             |
    |             +-- CreateChatCompletionStreamResponse(...)
    |             |     +-- IN: request/model metadata + choice + optional usage
    |             |     +-- OUT: OpenAI stream response model
    |             |
    |             +-- response.model_dump_json(exclude_none=True)
    |             |     +-- IN: response model
    |             |     +-- STRUCTURES: compact JSON string
    |             |     +-- LOGIC: omit absent OpenAI fields
    |             |     +-- OUT: '{"id":"req-chat-001",...,"delta":{"content":"KV"}}'
    |             |
    |             +-- repeat scheduler/response tree until terminal status
    |             |     +-- OUT: ordered content chunks
    |             |
    |             +-- optional final usage chunk
    |             |     +-- IN: stream_options.include_usage
    |             |     +-- OUT: final usage-only chunk when enabled
    |             |
    |             +-- yield "[DONE]"
    |                   +-- IN: terminal generation reached
    |                   +-- OUT: SSE completion sentinel
    |                   |
    |                   +-- EventSourceResponse formats SSE frame
    |                   |     +-- IN: payload string
    |                   |     +-- OUT: "data: &lt;payload&gt;\n\n"
    |                   |
    |                   +-- ASGI sends frame -> client
    |                         +-- IN: encoded SSE bytes
    |                         +-- OUT: network response bytes
    |
    +-- [BRANCH] non-streaming
          |
          +-- pipeline.all_tokens(...)
          |     +-- OUT: complete list[TokenGeneratorOutput]
          |
          +-- OpenAIChatResponseGenerator.complete(...)
          |     +-- OUT: CreateChatCompletionResponse
          |
          +-- FastAPI serializes response model
          |     +-- OUT: application/json body
          |
          +-- request_session adds X-Request-ID
          |     +-- OUT: response header
          |
          +-- ASGI sends JSON -> client
                +-- OUT: complete non-streaming HTTP response
</pre>

</details>

## 7. Prefill And Decode Repeat Loop

<details>
<summary><strong>Expand: Request state across repeated scheduler iterations</strong></summary>

<pre>
new TextContext
    |
    +-- IN: TokenBuffer(prompt=49 IDs, generated=[])
    +-- STRUCTURES: replica.ce_reqs
    +-- LOGIC: fresh requests begin as context-encoding/prefill work
    +-- OUT: CE queue entry
    |
    +-- construct_batch()
    |     +-- IN: CE queue entry
    |     +-- OUT: TextGenerationInputs(batch_type=CE)
    |
    +-- pipeline.execute()
    |     +-- IN: CE inputs
    |     +-- OUT: model output + sampled token/context progress
    |
    +-- model forward pass
    |     +-- IN: active prompt token slice + KV metadata
    |     +-- OUT: logits and updated KV cache
    |
    +-- [BRANCH] prompt not fully processed
    |     +-- IN: processed_length &lt; prompt_length
    |     +-- LOGIC: preserve chunked-prefill state
    |     +-- OUT: context returns to CE queue
    |
    +-- [BRANCH] prefill complete
          +-- IN: processed_length == prompt_length
          +-- LOGIC: future steps need only current decode token
          +-- OUT: context moves to TG queue
          |
          +-- construct_batch()
          |     +-- IN: TG queue entry
          |     +-- OUT: TextGenerationInputs(batch_type=TG)
          |
          +-- pipeline.execute()
          |     +-- IN: active decode token(s)
          |     +-- OUT: next sampled token
          |
          +-- update_context_and_prepare_responses()
          |     +-- IN: sampled token
          |     +-- OUT: updated TokenBuffer + TextGenerationOutput
          |
          +-- [BRANCH] status ACTIVE
          |     +-- IN: no EOS/length/cancel condition
          |     +-- OUT: remain in TG queue; next scheduler iteration
          |
          +-- [BRANCH] EOS / maximum length / cancelled
                +-- IN: terminal status
                +-- LOGIC: release scheduler/grammar/KV state
                +-- OUT: terminal SchedulerResult and completed API stream
</pre>

</details>

## 8. Alternate Paths And Configuration Branches

<details>
<summary><strong>Expand: SageMaker ingress path</strong></summary>

<pre>
POST /invocations
    |
    +-- [FRAMEWORK] sagemaker_routes.invocations(request)
          |
          +-- IN: SageMaker HTTP request carrying OpenAI-compatible body
          +-- STRUCTURES: FastAPI Request
          +-- LOGIC: adapt/forward to shared OpenAI chat route
          +-- OUT: shared route response
          |
          +-- [AWAIT] openai_create_chat_completion(request)
                +-- IN: same Request object
                +-- OUT: same tree from section 2 onward
</pre>

</details>

<details>
<summary><strong>Expand: Pre-tokenized prompt branch</strong></summary>

<pre>
completion_request.prompt_tokens is non-empty
    |
    +-- IN: prompt_tokens=[128000,128006,882,128007,271]
    +-- STRUCTURES: Sequence[int]
    +-- LOGIC: orchestrator-supplied IDs win over messages
    +-- OUT: TextGenerationRequest(prompt=IDs,messages=[])
    |
    +-- tokenizer._generate_prompt_and_token_ids(request)
          |
          +-- IN: request.prompt is Sequence[int]
          +-- STRUCTURES: numpy token array
          +-- LOGIC: route directly to encode(sequence[int])
          +-- OUT: existing IDs; no chat template
          |
          +-- encode(sequence[int])
                +-- IN: existing IDs
                +-- STRUCTURES: np.ndarray(dtype=int64/int32 as tokenizer path requires)
                +-- LOGIC: copy/normalize IDs without text tokenization
                +-- OUT: np.array([128000,128006,882,128007,271])
</pre>

</details>

<details>
<summary><strong>Expand: Scheduler and pipeline variants</strong></summary>

<pre>
load_scheduler(pipeline_config.role, ...)
    |
    +-- IN: pipeline role and runtime feature flags
    +-- STRUCTURES: scheduler/pipeline implementation classes
    +-- LOGIC: select one execution family
    +-- OUT
          |
          +-- role="prefill_and_decode"
          |     +-- OUT: TokenGenerationScheduler
          |
          +-- role="decode_only"
          |     +-- OUT: DecodeScheduler
          |
          +-- role="prefill_only"
                +-- OUT: PrefillScheduler

runtime/model variants
    |
    +-- IN: overlap/speculative/architecture configuration
    +-- STRUCTURES: alternate pipeline/context/batch processor/model classes
    +-- LOGIC: replace worker-side subtree while preserving API and IPC boundaries
    +-- OUT
          +-- enable_overlap_scheduler -> OverlapTextGenerationPipeline path
          +-- speculative decoding    -> speculative sample/context-update path
          +-- non-Llama architecture -> architecture-specific staging/model path
</pre>

</details>

## 9. Error Exit Tree

<details>
<summary><strong>Expand: Inputs, structures, logic, and outputs for error exits</strong></summary>

<pre>
request body / route preparation
    |
    +-- JSONDecodeError
    |     +-- IN: missing/malformed JSON bytes
    |     +-- STRUCTURES: json.JSONDecodeError -> FastAPI HTTPException
    |     +-- LOGIC: route catches decode failure
    |     +-- OUT: HTTP 400 "Missing JSON."
    |     |
    |     +-- HTTPException(status_code=400, detail="Missing JSON.")
    |           +-- IN: caught JSONDecodeError
    |           +-- OUT: FastAPI error response path
    |
    +-- KeyError or TypeError
    |     +-- IN: invalid JSON shape/value use
    |     +-- STRUCTURES: Python exception -> HTTPException
    |     +-- LOGIC: map unsafe request shape to generic client error
    |     +-- OUT: HTTP 400 "Invalid JSON."
    |
    +-- Pydantic ValidationError
    |     +-- IN: schema-invalid parsed value
    |     +-- STRUCTURES: field/model validation details
    |     +-- LOGIC: preserve useful validation detail
    |     +-- OUT: HTTP 400 with validation text
    |
    +-- InputError / UndefinedError / ValueError
    |     +-- IN: semantic request/template/sampling failure
    |     +-- STRUCTURES: route exception handlers
    |     +-- LOGIC: map preparation failures to client-visible 400
    |     +-- OUT: HTTP 400
    |
    +-- RequestQueueFull from ZmqModelWorkerProxy.stream(...)
    |     +-- IN: request_queue.writable() == false
    |     +-- STRUCTURES: RequestQueueFull + app exception handler
    |     +-- LOGIC: load shed before SSE headers/state commit
    |     +-- OUT: HTTP 429 + Retry-After: 1
    |     |
    |     +-- _request_queue_full_exception_handler(request, exc)
    |           api_server.py:307
    |             |
    |             +-- IN: Request + RequestQueueFull
    |             +-- STRUCTURES: OpenAI error envelope + JSONResponse headers
    |             +-- LOGIC: map admission backpressure to standard rate-limit response
    |             +-- OUT: JSONResponse(status=429, Retry-After="1")
    |
    +-- streaming error before first visible payload
    |     +-- IN: generator yields JSONResponse first
    |     +-- STRUCTURES: _start_stream first-item gate
    |     +-- LOGIC: return real error response instead of SSE 200
    |     +-- OUT: non-200 HTTP response
    |
    +-- streaming error after SSE output starts
          +-- IN: exception after HTTP 200 committed
          +-- STRUCTURES: OpenAI error JSON SSE frame
          +-- LOGIC: status cannot change; report error in stream
          +-- OUT: HTTP 200 stream containing error payload
</pre>

</details>

## 10. IPC And Compiled-Runtime Boundaries

<details>
<summary><strong>Expand: Boundaries that are handoffs, not nested Python calls</strong></summary>

<pre>
request_queue.put(context)
    |
    +-- IN: API-process TextContext
    +-- STRUCTURES: ZeroMQ/MAX queue serialization and transport
    +-- LOGIC: cross process boundary; worker scheduler later drains independently
    +-- OUT: worker request_queue item
          |
          +-- not a direct Python call to run_iteration() / scheduler.run_iteration()

response_queue.put_nowait(SchedulerResult)
    |
    +-- IN: worker-process result map
    +-- STRUCTURES: ZeroMQ/MAX queue transport
    +-- LOGIC: cross process boundary; API response_worker later consumes independently
    +-- OUT: API response_queue item
          |
          +-- not a direct Python call to _drain_responses()

LlamaModelBase.execute(model_inputs)
    |
    +-- IN: Llama3Inputs
    +-- STRUCTURES: Python wrapper + compiled MAX Model
    +-- LOGIC: model_inputs.buffers -> self.model.execute(*buffers)
    +-- OUT: ModelOutputs
          |
          +-- compiled graph/runtime/kernel execution is not a Python call-stack recursion
</pre>

</details>
