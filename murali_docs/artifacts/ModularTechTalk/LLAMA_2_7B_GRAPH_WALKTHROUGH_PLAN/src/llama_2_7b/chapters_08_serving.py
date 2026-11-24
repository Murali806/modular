from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('prefill_decode','Prefill → decode: the same model, different active rows',
 'Follow two requests through their first prompt pass and next decode pass, with exact offsets, positions and cache updates.',
 'Repeated serving and lifetime',[
 A('SCHED','Scheduler / batch constructor','request orchestration','model worker','Chooses active prompt/decode rows within token/page/request budgets.'),
 A('BATCH','Active input batch','per-step buffers','batch processor','Flattens active tokens while retaining request row boundaries.'),
 A('M1','M1 + weights','reusable executable','pipeline','Runs 32 layers using the same model/code resources.'),
 A('KV','KV pages and lengths','persistent request state','cache manager','Stores processed input token K/V, not future sampled-token K/V.'),
 A('LOG','Selected logits','per-step output','decoder','LAST_TOKEN selects the final active row of each request.'),
 A('NEXT','Sampled IDs / contexts','request state','pipeline + batch constructor','Updates generated/processed state and supplies the next input token.')],[
 E('Admit two illustrative prompts','Request A has three active prompt tokens; B has two. The batch constructor accounts for input-token and cache capacity before execution.',{'SCHED':B('prefill admitted','A: 3 prompt tokens','B: 2 prompt tokens'),'KV':B('new request pages','old lengths [0,0]')},[R('batch_constructor','def construct_batch('),R('scheduler','self.batch_constructor.construct_batch()')]),
 E('Build ragged prefill inputs','Flatten A0,A1,A2,B0,B1 into five token IDs. Offsets [0,3,5] let every layer find request boundaries without padding.',{'BATCH':B('prefill inputs',('tokens','[A0,A1,A2,B0,B1]'),('offsets','[0,3,5]'))},[R('batch','def prepare_initial_token_inputs(')],edges=[('SCHED','BATCH')]),
 E('Execute the full decoder for active prompt rows','M1 runs embedding and all 32 layers. Each layer writes K/V for its active rows and applies request-local causal attention.',{'M1':B('prefill execution','X [5,4096] through 32 layers'),'KV':B('prompt K/V processed','A positions 0…2','B positions 0…1')},[R('model','self.model.execute('),R('attention','xq = rope_split_store_ragged(')],edges=[('BATCH','M1'),('M1','KV')]),
 E('Select last-token rows for next-token logits','offsets[1:]-1 = [2,4]. The last hidden row per request feeds final norm/head, producing [2,32000] FP32 logits.',{'LOG':B('prefill logits','gather rows [2,4]','[2,32000] FP32')},[R('transformer','input_row_offsets[1:] - 1')],edges=[('M1','LOG')]),
 E('Sample A3 and B2, then update contexts','These labels name illustrative next token IDs, not measured model predictions. Their own K/V has not been computed yet.',{'NEXT':B('sampled next inputs',('next','[A3,B2]')),'KV':B('processed prefix lengths','old lengths for next call [3,2]')},[R('pipeline','update_context_and_prepare_responses('),R('pipeline','self._kv_manager.step(ctx)')],edges=[('LOG','NEXT')]),
 E('Decode with one active token per request','The next call uses tokens [A3,B2], offsets [0,1,2] and old cache lengths [3,2]. Position-dependent RoPE writes A position 3 and B position 2.',{'NEXT':B('handed to decode'),'BATCH':B('decode inputs',('next','[A3,B2]'),('offsets','[0,1,2]')),'M1':B('same executable reused','X [2,4096] through 32 layers')},[R('batch','def prepare_initial_token_inputs('),R('rope_kernel','k_cache.cache_length(bi)')],transfer=['NEXT','new active token IDs','BATCH'],edges=[('BATCH','M1')]),
 E('Read the prefix, append current K/V and produce new logits','Each request attends to its retained prefix plus the newly processed token. No old prompt QKV recomputation is required for the retained cache rows.',{'KV':B('next processed prefix','A positions 0…3','B positions 0…2'),'LOG':B('decode logits','last rows [0,1]','[2,32000] FP32')},[R('ragged_kernel','kv_cache.get_key_cache'),R('transformer','input_row_offsets[1:] - 1')],edges=[('KV','M1'),('M1','LOG')]),
 E('Repeat until a stop/release transition','The scheduler advances requests, responds and handles completion/cancellation. The batch can change between iterations while M1 remains reusable.',{'SCHED':B('next iteration','admit / continue / finish / cancel'),'NEXT':B('next sampled IDs or stop','request lifecycle decision')},[R('scheduler','def run_iteration('),R('batch_constructor','def advance_requests('),R('batch_constructor','def release_request(')])],[
 Q('Two-pass buffer ledger','''
| Quantity | Prefill | Next decode |
|---|---|---|
| active tokens | A0,A1,A2,B0,B1 | A3,B2 (sampled by prefill) |
| T / B | 5 / 2 | 2 / 2 |
| row offsets | `[0,3,5]` | `[0,1,2]` |
| old cache lengths | `[0,0]` | `[3,2]` |
| absolute positions of active rows | A:0,1,2; B:0,1 | A:3; B:2 |
| hidden-state shape | `[5,4096]` | `[2,4096]` |
| Q shape | `[5,32,128]` | `[2,32,128]` |
| last-row gather | `[2,4]` | `[0,1]` |
| logits for LAST_TOKEN | `[2,32000]` | `[2,32000]` |
| processed KV after forward | lengths 3,2 | lengths 4,3 |

Token labels identify logical positions, not actual vocabulary values or a numerical model test. The shape ledger is derived from the configured dimensions and this chosen tiny batch.
''',[R('batch_types','def ragged_batch_inputs('),R('transformer','input_row_offsets[1:] - 1'),R('rope_kernel','k_cache.cache_length(bi)')],open=True),
 Q('Why the newly sampled token is not already in KV','''
The decoder forward processes its **input** IDs and writes their K/V. Its logits predict a next token. Sampling chooses that next ID after the forward has finished, so the sampled token’s embedding/QKV cannot have been part of the earlier forward.

On the next decode, that sampled ID becomes an input and acquires its K/V in all 32 layers. Generated length, processed length and active input length therefore have distinct meanings. A completion criterion may end a request without another forward for the last sampled token.
''',[R('pipeline','model_outputs = self._launch_forward_pass('),R('pipeline','new_tokens = sampling_processor.new_tokens'),R('pipeline','self._kv_manager.step(ctx)')],open=True),
 Q('Chunked prefill and evolving batches','''
Large prompts can be divided according to the enabled chunked-prefill policy and per-step token budget. Intermediate prompt chunks populate cache but do not yet emit a generated response token for that request. The pipeline comments explicitly defer scores until the prompt-completing step.

The batch constructor keeps unfinished context-encoding requests in its CE queue and moves completed prefills to TG/decode; `advance_requests` handles the transition. Request order and membership can change. The next offsets/cache-length/lookup arrays must match the actual next batch, not yesterday’s slot positions.

Continuous serving means orchestration repeats around the same initialized model. It does not imply every server configuration enables overlap, speculative decoding, prefix reuse or disaggregated prefill/decode. Those routes have separate conditions and resource contracts.
''',[R('batch_constructor','def advance_requests('),R('pipeline','A chunked-prefill step emits no output'),R('scheduler','has_pending_outputs')]),
 Q('Compute and memory consequences without invented timing','''
Prefill offers many active Q rows and builds a prefix. Decode usually offers one active Q row per request but reads increasingly long retained K/V. Both still execute all 32 decoder layers, MLPs and output selection.

This explains why production attention dispatch distinguishes token generation from prefill and why cache capacity affects serving admission. It does not by itself establish a measured bottleneck, throughput, TTFT or latency. Those require a compatible checkpoint, real target, workload and profiling evidence.
''',[R('mha','var is_token_generation ='),R('mha','def flash_attention_dispatch['),R('batch_constructor','def _create_new_token_budget(')])],'''
scheduler.run_iteration
├─ retrieve pending requests
├─ batch_constructor.construct_batch → CE/prefill or TG/decode rows
└─ pipeline.execute
   ├─ prepare_batch → offsets / lengths / page lookup
   ├─ same M1.execute → 32 decoder layers → selected logits
   ├─ sampler → host-visible generated IDs
   └─ context update → kv_manager.step
      └─ advance requests / emit responses / release finished or cancelled IDs
''',[R('scheduler','def run_iteration('),R('pipeline','def _execute('),R('batch_constructor','def advance_requests(')])

C('sampler_serving','Sampling graphs and the returned response',
 'Decoder logits are only the middle of serving: inspect graph roles, mutation-sensitive scoring, sampling parameters, D2H and request updates.',
 'Repeated serving and lifetime',[
 A('DEC','Decoder output','GPU/CPU Buffer','M1 execution','Provides logits, optional next_token_logits and offset metadata.'),
 A('SCORE','Label/logit processing','per-batch work','pipeline','Scoring precedes processors that can mutate logits.'),
 A('SAM','Sampler graph','separate compiled Model','pipeline / S1','Runs on configured sampler device; optional bitmask variant is separate.'),
 A('SEED','Sampling state','input/output buffers','FusedSamplingProcessor','Carries per-request controls, previous generated IDs and seed state.'),
 A('COPY','Tracked staging','copy + buffer','pipeline / driver','Makes generated token IDs host-readable at the required boundary.'),
 A('RESP','Contexts / response queue','request output','pipeline + scheduler','Updates request state and emits results; cache lifecycle follows batch construction.')],[
 E('Initialize sampler roles once','load_sampler_graphs builds GraphSpec entries for the ordinary sampler and, only if constraints may fire, the bitmask sampler. They compile/init using S1.',{'SAM':B('initialized models','GraphRole.SAMPLER','optional SAMPLER_BITMASK'),'SEED':B('request controls','temperature / top_k / top_p / min_p / seed')},[R('pipeline','def load_sampler_graphs('),R('pipeline','if pipeline_config.needs_bitmask_constraints:')]),
 E('Choose logits and score before mutation','The pipeline runs label scoring before sampling processors because penalties/min-token processing can edit logits. logits_for_sampling selects next_token_logits/offsets as appropriate.',{'DEC':B('decoder outputs',('logits','logits + optional next_token_logits')),'SCORE':B('scored before mutation','label scoring / optional log probabilities')},[R('pipeline','Scoring precedes sampling'),R('sampling_processor','def logits_for_sampling(')],edges=[('DEC','SCORE')]),
 E('Select the sampler variant and device','A present structured-output bitmask selects the prebuilt bitmask sampler. sample_on_host selects CPU sampling; otherwise use the model device.',{'SAM':B('selected for this batch','ordinary or bitmask sampler','configured sampler device')},[R('pipeline','if bitmask is not None:'),R('pipeline','if pipeline_config.sampling.sample_on_host')]),
 E('Apply sampling controls in the graph','The graph consumes temperature, top-k/p/min-p and seed; optional penalties, min-token masks and bitmask constraints follow configured branches.',{'SAM':B('sampling graph execution','topk_fused_sampling(...)'),'SEED':B('active control buffers','per-request parameters + current seed')},[R('sampler','tokens = topk_fused_sampling('),R('sampling_processor','new_tokens, new_generated_tokens, new_seed =')],edges=[('SCORE','SAM'),('SEED','SAM')]),
 E('Return token buffers and updated state','The graph concatenates new tokens with previous generated tokens and increments seed. The processor retains new_tokens/generated_tokens for the pipeline.',{'SEED':B('updated sampler state','seed + 1','new_tokens / generated_tokens'),'DEC':B('logits consumed')},[R('sampler','all_tokens = ops.concat([prev_tokens, tokens]'),R('sampler','seed = seed + 1'),R('sampling_processor','self.generated_tokens = new_generated_tokens')]),
 E('Stage generated IDs and wait for host access','A tracked staging Buffer receives generated IDs; inplace_copy_from queues the copy and to_numpy makes the host access safe.',{'COPY':B('host-visible token IDs',('ids','generated_tokens_np'))},[R('pipeline','generated_tokens_host = Buffer('),R('pipeline','generated_tokens_host.to_numpy()')],edges=[('SEED','COPY')]),
 E('Update contexts, then step the cache manager','update_context_and_prepare_responses handles per-request token updates. The pipeline then calls kv_manager.step; chunked-prefill requests may emit no response yet.',{'RESP':B('pipeline outputs',('ids','request ID → token response')),'COPY':B('copy consumed')},[R('pipeline','res = update_context_and_prepare_responses('),R('pipeline','self._kv_manager.step(ctx)')],transfer=['COPY','safe host token consumption','RESP']),
 E('Scheduler emits and retires requests','The worker scheduler delivers results, advances the request queues, handles cancellation and releases request-owned state via the batch constructor.',{'RESP':B('serving lifecycle','response queue / continue / stop / cancel')},[R('scheduler','responses = self.pipeline.execute(inputs)'),R('scheduler','get_cancelled_reqs(self.cancel_queue)'),R('batch_constructor','def release_request(')])],[
 Q('Decoder, sampler and optional variants are separate compiled roles','''
`TextGenerationPipeline` first creates S1, then loads the pipeline model and KV manager. It chooses a sampler device and calls `load_sampler_graphs`. The helper constructs ordinary and optional bitmask GraphSpecs and uses the same compile/init helper.

A decoder C1/M1 is therefore not the entire serving executable set. `compiled_modules` combines the decoder’s modules and samplers. The session’s MEF policy must account for multiple graphs and their signatures. A sampler does not need to retrace the 32-layer model to change a request’s temperature value.
''',[R('pipeline','def load_sampler_graphs('),R('pipeline','self.compiled_modules = {')],open=True),
 Q('Which logits row, and why scoring runs first','''
The decoder may expose logits, next_token_logits and logit_offsets depending on its output mode. `logits_for_sampling` chooses the appropriate view; with variable logits and a separate next-token tensor, identity offsets can represent one row per request.

Label scoring happens before sampling processors because penalty/min-token operations can mutate logits in place. Optional log-probability calculation is another pipeline step with its own supported-model checks. Treating all consumers as independent readers could change the scores.

The FP32 logits tensor is `[B,32000]` for the primary LAST_TOKEN route. The selected sampler returns integer vocabulary IDs. Text decoding/stream formatting are later serving work; logits are neither token IDs nor HTTP response bytes.
''',[R('sampling_processor','def logits_for_sampling('),R('pipeline','Scoring precedes sampling'),R('pipeline','if inputs.enable_log_probs:')],open=True),
 Q('Sampling inputs, masks and state','''
| Input | Purpose |
|---|---|
| temperature | scaling policy used by the sampling implementation |
| top_k / max_k | per-request selection controls and batch bound |
| top_p / min_top_p / min_p | configured probability-filter controls |
| seed | stochastic state consumed by sampler and incremented in graph |
| previous generated IDs | permits returned generated-token accumulation |
| optional frequency/penalty data | enabled penalty processing and state updates |
| optional min-token mask | controls premature stop-token choices |
| optional structured-output bitmask | packed int32 vocabulary constraints, unpacked/masked on GPU by the graph |

The graph calls `topk_fused_sampling`; its result is not assumed to be greedy for every request. Deterministic comparison requires fixed request parameters and an appropriate reference run. No token result in this reader is presented as a real Llama prediction.
''',[R('sampler','tokens = topk_fused_sampling('),R('sampler','apply_packed_bitmask('),R('sampling_processor','self.sampler_inputs = SamplerInputs.create')]),
 Q('The final copy in this pipeline versus optional helper paths','''
The ordinary `_execute` implementation copies generated-token buffers to tracked staging memory, then calls `to_numpy()`. This is the source route taught in the main scene. FusedSamplingProcessor also exposes explicit asynchronous-copy/event helpers for other usages; their existence does not prove the main `_execute` call uses them.

CPU sampling is selected by `sample_on_host`; logits can be moved to the sampler device when necessary. Device sampling can keep the vocabulary-sized logits on the GPU and transfer only resulting IDs. The placement and synchronization cost therefore depend on this explicit policy.
''',[R('pipeline','d2h_generated_tokens'),R('sampling_processor','def start_async_token_copy('),R('sampling_processor','if logits.device != self.device:')]),
 Q('Request progression and failure boundaries','''
Empty batches return early. A chunked-prefill step can produce no response token for an unfinished prompt. Normal responses update token/context state, and only afterward does the pipeline step the KV manager.

The scheduler can release cancelled requests through `batch_constructor.release_request` and emit cancellation results. The batch constructor owns primary KV lifecycle and also calls pipeline.release for model-specific state. Client completion, cache-reference release and the device’s last use must remain correctly ordered.
''',[R('pipeline','if len(flat_batch) == 0:'),R('pipeline','A chunked-prefill step emits no output'),R('scheduler','self.batch_constructor.release_request(cancelled_id)'),R('batch_constructor','def release_request(')])],'''
startup: load_sampler_graphs(S1)
├─ ordinary TopKSampler GraphSpec
└─ conditional bitmask TopKSampler GraphSpec

pipeline._execute(batch)
├─ prepare_batch → select sampler → launch decoder
├─ label scoring before logits mutation
├─ logits_for_sampling → apply processors → sampler outputs / seed
├─ tracked staging copy → to_numpy → host IDs
├─ update_context_and_prepare_responses
└─ kv_manager.step → scheduler responses / next batch / release
''',[R('pipeline','def load_sampler_graphs('),R('pipeline','def _execute('),R('sampling_processor','def logits_for_sampling(')])

C('lifetime','Reuse, ownership and release: one complete lifetime map',
 'Track what survives one kernel, one execution, one request and the whole model worker; finish with a concrete verification checklist.',
 'Repeated serving and lifetime',[
 A('LONG','Session / compiled model','long-lived owners','pipeline / model worker','S1, C1 and compiled callable resources survive repeated token execution.'),
 A('M1','Initialized executable','model lifetime','ModelHandle / pipeline','Retains weight registry, loaded host code and setup resources.'),
 A('REQ','Request KV ownership','request lifetime','batch constructor / cache manager','Holds logical page references, lengths and optional shared-prefix ownership.'),
 A('E1','Execution E1','invocation lifetime','MEFExecutor / AsyncValues','Owns input/result references, scratch lifetimes and pending host work.'),
 A('GPU','Pending device use','stream/event lifetime','driver / retained buffers','Queued operations can outlive the submitting host call.'),
 A('FREE','Release / reuse','cleanup transition','last relevant owner','Reuses/frees only after required references and asynchronous uses end.')],[
 E('Retain model resources after initialization','S1/C1/M1 and weight owners are established. They are reused by prefill and decode calls.',{'LONG':B('worker lifetime',('s1','S1'),('c1','C1')),'M1':B('initialized lifetime',('m1','M1 + weights + H1 / resources'))},[R('session_native','ModelHandle handle(compiledModels.copy()')]),
 E('Admit a request and retain cache pages','The request acquires logical cache ownership; the physical pool can outlive many requests.',{'REQ':B('request active',('pages','request A → page references'))},[R('batch_constructor','def _admit_request('),R('batch_constructor','def release_request(')]),
 E('Create an execution and submit device work','E1 retains input/output/resource references while generated host work enqueues commands. Device-use tracking protects the associated buffers.',{'E1':B('in flight',('exec','execution references + scratch')),'GPU':B('pending commands',('use','kernel/copy buffer uses'))},[R('invokable','class MEFExecutor final'),R('primitive','Captures keep the inputs')],edges=[('M1','E1'),('REQ','E1'),('E1','GPU')]),
 E('Retire execution-local work safely','As dependencies and device uses complete, scratch and result owners can be released/reused. C1/M1 and active request KV remain retained.',{'E1':B('completed'),'GPU':B('relevant uses settled'),'FREE':B('temporary reuse eligible',('exec','execution-local scratch / references'))},[R('primitive','Release everything context-reachable BEFORE'),R('device_context','enqueueEventHandler(')],transfer=['E1','last execution ownership/use','FREE']),
 E('Decode with the same code and weights','New E2 inputs reuse M1 and the request’s retained prefix. This is model-object reuse, independent of compile-cache lookup.',{'M1':B('reused unchanged',('m1','same M1 / weight storage')),'REQ':B('prefix retained',('pages','request A → existing + appended KV')),'E1':B('next invocation','E2: new token input / transient state')},[R('model','self.model.execute('),R('mef','ExecutableModel::execute(')]),
 E('Release the request through the owning layer','Completion/cancellation releases cache ownership through the batch constructor and model-specific state through pipeline.release. Shared or pending ownership can defer physical reuse.',{'REQ':B('request retired'),'FREE':B('cache references released as allowed',('pages','page IDs eligible for manager reuse'))},[R('batch_constructor','self.kv_cache.release(bound.ctx)'),R('pipeline','def release(self, request_id')],transfer=['REQ','cache lifecycle','FREE']),
 E('Tear down long-lived resources only at their final owner','Model/session/code/context references eventually drop; CPUDevice shutdown drains work. Loaded CUDA modules unload through retained module owners.',{'LONG':B('last-owner teardown'),'M1':B('model released'),'GPU':B('context/code cleanup'),'FREE':B('lifetime complete','managed tensor deleters','module unload / runtime drain')},[R('cpu','CPUDevice::~CPUDevice'),R('cuda','CUDAModuleFunction::~CUDAModuleFunction()'),R('async','destroyWithRefCountZero()')])],[
 Q('Ownership ledger','''
| Object | Typical scope on this route | Last-use / release rule |
|---|---|---|
| S1 / shared CPUDevice references | pipeline / process facilities | shared owners and runtime shutdown policy |
| G1 / IR1 | construction / compilation | caller graph lifetime versus compiler clone lifetime |
| C1 / compiled archives | compiled callable / model | retained while executable code/resources depend on them |
| M1 / weight registry / DLPack owners | initialized model | native handle retains storage needed by execution |
| H1 / JIT namespace | compiled resource context | JIT/compiled-resource ownership |
| D1 / KD1 / CUDA module | compiled bytes / loaded driver resource | function/module/context owners and last use |
| physical KV pool | cache manager | survives individual request release |
| request page references | request / valid shared prefix | manager reference/usage rules |
| E1 registers/AsyncValues | execution | producer/consumer references and completion |
| temporary activation/scratch | planned invocation uses | dependency-aware last use and device tracking |
| token staging buffer | copy/host consumption | tracked copy completion before read/reuse |

No single `del` statement is a substitute for these ownership relationships. Python names, reference-counted native resources and queued GPU accesses have related but different lifetimes.
''',[R('session_native','ModelHandle handle(compiledModels.copy()'),R('invokable','Make sure the MEF file doesn\'t get deallocated'),R('device_context','enqueueEventHandler(')],open=True),
 Q('Three kinds of reuse and what each saves','''
| Reuse | Retained object | What is avoided |
|---|---|---|
| compiled artifact cache | MEF or kernel archive | some/all compilation work |
| initialized model reuse | same M1, weights, host/device resources | repeated setup/binding/loading where already retained |
| KV/prefix reuse | processed key/value rows with valid ownership | recomputing retained prefix K/V |

Buffer reuse is a fourth, local optimization: scratch memory can serve different non-overlapping lifetimes. It does not mean the old numerical activation remains semantically reusable. These distinctions explain why a warm token loop can retain code and KV while creating new input/result AsyncValues.
''',[R('engine','def compile_reusing_mefs('),R('mef','ExecutableModel::execute('),R('compiler','createSimpleBufferReusePass')],open=True),
 Q('Failure/cancellation do not erase outstanding uses','''
Compiler failure prevents a valid C1; init failure prevents usable M1; execution errors propagate through AsyncValues and native/Python boundaries. Request cancellation changes serving ownership/response state but does not inherently retract every already submitted GPU command.

The scheduler calls batch-constructor release for cancellation. Release paths retire request state and cache references under the manager’s rules; queued work and captured native references still need safe completion. The runtime’s final teardown drains pending work before invalidating the CPUDevice.
''',[R('scheduler','self.batch_constructor.release_request(cancelled_id)'),R('batch_constructor','def release_request('),R('cpu','CPUDevice::~CPUDevice')]),
 Q('What this walkthrough verifies, and what a GPU run would add','''
**Documentation checks:** exact source anchors and saved-source hashes; all Example 04 coverage links; deterministic Step/Back/Reset/seek and route selection; real Play/Pause; movable source/legend cards; local-file and offline loading; narrow-screen/fullscreen behavior; no-JavaScript transcript; file-size limits; KV arithmetic and scalar softmax recurrence.

**GPU-LAB, not performed:** choose an actually accepted checkpoint/encoding and target; record exact resolved config/revision; compile with supported stage diagnostics; inspect real per-stage IR and kernel selection; run prefill/decode parity against a reference; check logits/tokens and cache behavior; capture launches/allocations/completion only if requested. No timing, parity, GPU execution or captured Llama compiler dump is claimed by this course.

The exact original FP16 checkpoint’s acceptance gate must be resolved before calling such a run a success. The structural source explanation remains useful without pretending that blocker disappeared.
'''),
 Q('A final trace to answer any lifetime question','''
For any value, identify its producer, owner, representation, consumers and last safe use. Then ask which boundary proves progress: compiler legality, AsyncValue readiness, driver event completion or host visibility.

```text
new token ID
 → prepared input Buffer
 → runtime input AsyncValue
 → E1 dependency-ready host work
 → H1 descriptors / launch packet
 → KD1 reads weights + old KV, writes new KV / activations
 → logits Buffer → sampler → token staging copy
 → host-visible ID → updated context → next request step
```

The same chain applies to all 32 layers, but each layer uses its own weights and cache index. Code reuse, storage ownership and numerical state remain explicit at every boundary.
''',[R('model','self.model.execute('),R('primitive','mgp_generic_execute('),R('pipeline','generated_tokens_host.to_numpy()')])],'''
worker lifetime: S1 → C1 → M1 → weights / compiled resources
request lifetime: admission → KV references → prefill/decode → release
execution lifetime: inputs → E1 registers/tasks → result references
device-use lifetime: enqueue → completion tracking → safe reuse

last owner + last required use
└─ temporary reuse / request-page return / managed tensor deleter
   └─ eventual module unload and runtime drain at teardown
''',[R('session_native','ModelHandle handle(compiledModels.copy()'),R('batch_constructor','def release_request('),R('cpu','CPUDevice::~CPUDevice')])
