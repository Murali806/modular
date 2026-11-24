from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('index','Llama 2 7B: from session to generated token',
  'A source-guided course through the whole decoder, compiler, AsyncRT and driver. Start at InferenceSession, then follow every retained object.',
  'Orientation',[
 A('S1','S1 · InferenceSession','session','model worker / pipeline','Owns compilation configuration and retained native device contexts; uses a shared CPU runtime.'),
 A('G1','G1 · decoder graph','symbolic IR','Graph / GraphSpec','Describes token, weight and mutable-KV computation. Does not contain a request’s live activations.'),
 A('P1','P1 · implementation packages','package set','installed MAX build','Supplies built-in kernel declarations and parametric Mojo implementations.'),
 A('C1','C1 · CompiledModel','artifact','compiled callable','Retains compiled decoder artifact and expected weights; IR1 is its separate mutable compilation clone.'),
 A('M1','M1 · initialized Model','executable model','compiled callable / pipeline','Binds weights, resources and execution entry points for reuse.'),
 A('RT','AsyncRT · request work','runtime','shared CPUDevice','Schedules readiness continuations and host work. GPU streams remain separate execution mechanisms.'),
 A('GPU','GPU · buffers + code','device context','driver / model / cache manager','Retains weight buffers, KV pages and loaded device handles according to their independent lifetimes.'),
 A('OUT','Token response','request state','pipeline / serving worker','Receives sampled token IDs after the necessary host visibility gate.')], [
 E('1 · Session before decoder graph','The serving pipeline creates S1, then constructs the pipeline model. This is the serving order, even though Example 04 constructs its graph first.',{'S1':B('created',('s1','devices + compile config'))},[R('pipeline','session = InferenceSession(')]),
 E('2 · Trace the decoder','Llama3Model prepares GraphSpec; tracing emits embedding, 32 blocks and logits. P1 supplies the built-in operation contracts.',{'G1':B('symbolic graph',('g1','G1: tokens → 32 blocks → logits')),'P1':B('available',('p1','built-in .mojopkg implementations'))},[R('model','def _single_device_spec('),R('llama','def trace(')],edges=[('S1','G1'),('P1','G1')]),
 E('3 · Compile a clone','The compiler transforms IR1 through MO, MAP, MOGG and MGP. G1 stays the caller’s graph. The resulting artifact becomes C1.',{'G1':B('preserved',('g1','G1: original symbolic graph')),'C1':B('compiled',('c1','C1: MEF + compiled resources'))},[R('frontend','OwningOpRef'),R('compiler','gcPipelineImpl')],transfer=['G1','clone IR1 → compiler','C1'],edges=[('G1','C1'),('P1','C1')]),
 E('4 · Initialize with weights','S1 validates and binds the weight registry, sets up the executable and waits for setup before returning M1.',{'M1':B('initialized',('m1','M1: callable + resource registry')),'GPU':B('weights resident by initialized route',('weights','weight storage'),facts=['Device code can still have lazy first-use loading'])},[R('engine','def init_all('),R('session_native','loadAll(')],edges=[('S1','M1'),('C1','M1')]),
 E('5 · Prepare a ragged batch','The batch processor flattens token rows and binds page tables, lengths and dispatch metadata. KV allocation belongs to cache management.',{'RT':B('inputs prepared',('inputs','tokens [5], offsets [0,3,5]')),'GPU':B('retained + per-request',('weights','weight storage'),('kv','KV pages: A, B'),facts=['Illustrative prompt lengths 3 and 2'])},[R('batch','def prepare_initial_token_inputs('),R('kv','def get_symbolic_inputs(',occurrence=2)],edges=[('RT','GPU')]),
 E('6 · Execute via readiness','Input AsyncValues and chain dependencies make runtime primitives eligible. Host callables submit GPU work with retained resources.',{'RT':B('work scheduled',('exec','operand readiness → H1 → launch')),'M1':B('reused',('m1','same initialized M1'))},[R('primitive','CompiledKernelAsyncABIFn'),R('invokable','numOperands')],edges=[('M1','RT'),('RT','GPU')]),
 E('7 · Sample and make the token visible','A separate sampler graph consumes logits. The pipeline stages generated token IDs and to_numpy() waits for its tracked copy.',{'RT':B('sampling + copy',('sample','sampler graph → generated tokens')),'OUT':B('host visible',('ids','sampled token IDs → request contexts'))},[R('pipeline','d2h_generated_tokens'),R('pipeline','update_context_and_prepare_responses(')],edges=[('GPU','RT'),('RT','OUT')]),
 E('8 · Decode reuses the long-lived objects','The next call supplies new token IDs and updated lengths. Weight buffers, C1, M1 and live KV pages persist; request-local activations change.',{'RT':B('next decode ready',('inputs','tokens [2], offsets [0,1,2]')),'G1':B('no retrace per token',('g1','G1: same symbolic shape contract')),'OUT':B('context advanced',('ids','A length 3; B length 2 → decode inputs'))},[R('pipeline','self._kv_manager.step(ctx)'),R('model','self.model.execute(')],edges=[('OUT','RT'),('RT','GPU')])], [
 Q('Read this course in three passes','''
| Pass | Follow | Question it answers |
|---|---|---|
| Model and construction | lessons 1–8 | Which numbers, tensors, operations and owners make this Llama model? |
| Compiler and initialization | lessons 9–18 | How do symbolic operations become executable code and an initialized model? |
| Runtime and serving | lessons 19–25 | What runs, what waits, what is retained, and when can a token be returned? |

Every lesson starts with a source-ordered call tree. Step through its scene, then open the numbered explanations. Source buttons open the exact line and offer the **complete unchanged file**. The coverage page maps every Example 04 companion to its Llama counterpart.

The primary walkthrough uses **one GPU, one unquantized decoder, ordinary text generation and an explicitly compatible resolved encoding**. NVIDIA illustrates the driver backend; the graph and runtime boundaries remain identifiable for other targets. Two-byte activation/cache examples illustrate BF16-compatible arithmetic. They do not assert that an FP16 checkpoint was accepted or converted.
''',open=True),
 Q('Stable names: do not merge these objects','''
| Identity | What it is | What changes |
|---|---|---|
| G1 | caller-owned Graph / MO module | built during tracing; preserved during native compilation |
| P1 | built-in implementation package set | imported declarations and specialization inputs |
| S1 | InferenceSession | compilation/runtime configuration and device context references |
| IR1 | compiler-owned mutable clone | representation and optimization state |
| C1 | public CompiledModel | compiled artifact, expected-weight contract |
| A1 / H1 | host object bytes / resolved host callable | relocation and symbol resolution produce H1 |
| D1 / KD1 | device executable bytes / loaded device function | driver module load and function lookup produce KD1 |
| M1 | initialized engine Model | weights/resources bound; reused across calls |
| E1 | a single execution | input/result AsyncValues and transient work |

```text
P1 --declarations--> G1 --clone--> IR1 --compile--> C1
S1 -------------------------------init(C1,W)-----> M1
M1 + input buffers + KV references --execute----> E1
E1 --> host H1 --> driver KD1 --> GPU commands --> result buffers
```
''',[R('engine','class CompiledModel'),R('graph','self._module ='),R('primitive','CompiledKernelAsyncABIFn')],open=True),
 Q('Evidence and precision','''
**SOURCE** means that the accompanying code establishes the control flow or contract. Schematic representations show what each compiler stage has to express; they are not a captured Llama compiler dump. Conditional passes and kernel dispatch are marked explicitly. **BOUNDARY** identifies behavior delegated to the vendor driver or information not established by these sources. **GPU-LAB** marks proposed future verification, never a completed run.

The implementation snapshot is `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. The saved model configuration came from the public NousResearch mirror because the official configuration request was gated. The official model ID, mirror configuration and source route stay separate. UI/browser checks validate these learning artifacts; they do not validate model outputs, CUDA execution, latency or numerical parity.
'''),
 Q('Small files and offline navigation','''
Each lesson shares the CSS, animation controller and source manifest. Complete sources are divided into local JavaScript chunks and loaded only when opened, so `file://` reading needs no server, CDN or network access. No model weights, generated binaries or embedded screenshots are included. The builder rejects any generated file larger than 100 KiB.

The whole `llama_2_7b` directory is the portable unit. Keep its asset and `sources` subpaths together. The readable explanations and event transcripts remain available with JavaScript disabled; interactive source lookup uses JavaScript.
''')],'''
TextGenerationPipeline.__init__
├─ InferenceSession(devices, precompiled_mefs, export_mefs) → S1
├─ Llama3Model → load_graphs → GraphSpec.trace → G1
│  ├─ compile_reusing_mefs → IR1 transformations → C1
│  └─ init_all(C1, weights_registry) → M1
├─ KV manager + sampler graphs
└─ execute(batch), repeatedly
   ├─ prepare_batch → ragged inputs + KV metadata
   ├─ Llama3Model.execute → M1.execute → AsyncRT → host H1 → driver KD1
   ├─ sampler → tracked generated-token copy → host-visible IDs
   └─ update contexts → kv_manager.step → next batch
''',[R('pipeline','session = InferenceSession('),R('spec','def compile_and_init_graphs('),R('pipeline','def _execute(')])

C('architecture','The exact Llama 2 7B being explained',
 'Dimensions, weights, residual equations and token semantics before compiler abstractions.',
 'Model and construction',[
 A('CFG','Configuration','metadata','checkpoint / resolved MAX config','Describes dimensions and policies; it is not the tensor payload.'),
 A('W','Weights','persistent buffers','model initialization','Embedding, attention, MLP, norms and output head have named tensor storage.'),
 A('X','Residual stream','activation','one execution','Carries T token rows, each with 4096 features, across 32 blocks.'),
 A('ATT','Attention branch','activation + KV access','current decoder layer','Produces a 4096-wide update from 32 heads of dimension 128.'),
 A('MLP','MLP branch','activation','current decoder layer','Uses gated SiLU and the 11008-wide intermediate dimension.'),
 A('LOG','Logits','output buffer','decoder / sampler','LAST_TOKEN produces B rows of 32000 FP32 logits.')],[
 E('Read dimensions, not a model name heuristic','LlamaForCausalLM selects the shared MAX llama3 implementation. This Llama 2 configuration still supplies 32 layers, 32 heads and no GQA head reduction.',{'CFG':B('resolved dimensions','layers=32','hidden=4096','heads=32; KV heads=32','head_dim=128')},[R('arch','LlamaForCausalLM'),R('config','num_key_value_heads')]),
 E('Bind named matrices','The two embedding-shaped matrices are distinct because embeddings are untied. Each layer owns Q/K/V/O, gate/up/down and two norm scales.',{'W':B('6,738,415,616 architecture parameters','embedding [32000,4096]','32 × decoder weights','lm_head [32000,4096]')},[R('llama','embedding_layer = Embedding('),R('llama','output = Linear(')]),
 E('Look up token embeddings','For the illustrative five-token batch, embedding lookup creates X[5,4096]. Token IDs select rows; token numbers are not floating-point hidden states.',{'X':B('input residual',('x','X [5,4096]'))},[R('transformer','self.embed_tokens(tokens)')],edges=[('W','X')]),
 E('Normalize and attend','RMSNorm feeds QKV projection, RoPE, KV store and causal paged attention. The output projection returns a [5,4096] update.',{'ATT':B('attention output',('att','Attn(RMSNorm(X)) [5,4096]'))},[R('transformer','attn_out ='),R('attention','qkv = self.qkv_proj(x)')],edges=[('X','ATT'),('W','ATT')]),
 E('First residual add','The attention branch updates the residual stream. Keep the original X live until the add is safe.',{'X':B('post-attention residual',('x','H = X + Attn(norm(X))')),'ATT':B('consumed',facts=['Workspace may be reused after its last consumer'])},[R('transformer','h = x + attn_out')],transfer=['ATT','residual add','X']),
 E('Gate and project the MLP','The unquantized MLP can stack gate/up projection weights. Split the result, apply SiLU to the gate half, multiply by up, then project down.',{'MLP':B('MLP intermediate','gate/up [5,22016]','SiLU(gate) × up [5,11008]','down [5,4096]')},[R('linear','def _can_used_fused_mlp('),R('linear','activation_function')],edges=[('X','MLP'),('W','MLP')]),
 E('Second residual, repeated 32 times','Each block returns H + MLP(RMSNorm(H)). The Python trace emits the layer structure; a runtime token does not rerun Python layer construction.',{'X':B('after block 31',('x','X32 [5,4096]')),'MLP':B('consumed',facts=['Different weights and KV layer index per block'])},[R('transformer','for idx, layer in enumerate(self.layers)')],transfer=['MLP','residual add → next layer','X']),
 E('Select final token rows and compute logits','With offsets [0,3,5], LAST_TOKEN gathers rows [2,4]. Final RMSNorm and lm_head produce [2,32000], cast to FP32.',{'X':B('selected rows',('x','gather rows 2,4 → [2,4096]')),'LOG':B('ready for sampler',('logits','logits [2,32000] FP32'))},[R('transformer','input_row_offsets[1:] - 1'),R('transformer','DType.float32')],edges=[('X','LOG')])],[
 Q('Configuration ledger','''
| Field | Llama 2 7B value | Consequence |
|---|---:|---|
| hidden_size / num_hidden_layers | 4096 / 32 | 32 residual blocks of width 4096 |
| intermediate_size | 11008 | gated feed-forward expansion |
| num_attention_heads / num_key_value_heads | 32 / 32 | ordinary multi-head KV storage; no 8-KV-head Llama 3 assumption |
| head dimension | 4096 ÷ 32 = 128 | Q, K and V vector width |
| vocab_size | 32000 | embedding rows and logits columns |
| max_position_embeddings | 4096 | checkpoint context limit; serving configuration is validated separately |
| hidden_act / rms_norm_eps | SiLU / 0.00001 | SwiGLU-style MLP and RMS normalization |
| tie_word_embeddings | false | embedding and output projection weights are distinct |
| rope_scaling | null | do not substitute Llama 3 frequency scaling |
| torch_dtype | float16 | checkpoint metadata; not proof of this MAX revision’s accepted runtime encoding |

Metadata provenance is saved beside the parent guide in `evidence/llama2_7b_public_mirror_config.json` and `evidence/llama2_7b_memory_planning_provenance.json`. The official model is `meta-llama/Llama-2-7b-hf`; the configuration mirror is `NousResearch/Llama-2-7b-hf` at `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
''',[R('config','class Llama3Config')],open=True),
 Q('One block, with dimensions and equations','''
Let `T` be total active token rows across a ragged batch. For every layer `l`:

```text
X_l [T,4096] ───────────────────────────┐
  → RMSNorm → QKV → RoPE/store → MHA → O projection
                                       ↓ add
H_l [T,4096] ───────────────────────────┐
  → RMSNorm → gate/up → SiLU(gate)·up → down
                                       ↓ add
X_(l+1) [T,4096]
```

`RMSNorm(x) = x · rsqrt(mean(x²) + 1e-5) · weight`, with the implementation’s casts and `multiply_before_cast=False` preserved. There is no mean subtraction, LayerNorm bias, dropout, cross-attention or MoE branch on this model route.

Stored projection weights use output-feature × input-feature shapes. Conceptually Q/K/V and O are `[4096,4096]`; gate/up are `[11008,4096]`; down is `[4096,11008]`. Execution can stack compatible projections and specialize layouts. This does not change the checkpoint’s parameter count.
''',[R('llama','multiply_before_cast=False'),R('transformer','class TransformerBlock'),R('linear','class MLP')],open=True),
 Q('Parameter accounting, without rounding 7B into a memory claim','''
| Part | Exact count |
|---|---:|
| embedding + untied lm_head | 2 × 32000 × 4096 = 262,144,000 |
| attention matrices / layer | 4 × 4096² = 67,108,864 |
| MLP matrices / layer | 3 × 4096 × 11008 = 135,266,304 |
| two norm scales / layer | 2 × 4096 = 8,192 |
| all 32 decoder blocks | 32 × 202,383,360 = 6,476,267,520 |
| final norm | 4,096 |
| total architecture parameters | **6,738,415,616** |

The saved safetensors metadata also lists 2,048 FP32 entries in addition to that F16 count. This reader does not assign those entries an unverified semantic role. Serialized files include headers and potentially non-trainable data, so file bytes, parameter payload bytes and peak device memory are different quantities. A two-byte parameter-only estimate is 13,476,831,232 bytes; serving additionally needs KV, workspace, metadata and runtime resources.
''',[R('llama','class Llama3')]),
 Q('The token boundary and output modes','''
Input token IDs are integer vocabulary indices. They are not bytes, characters or one-word-per-token labels. For `[A0,A1,A2,B0,B1]`, the offsets `[0,3,5]` preserve request ownership without padding. A causal mask is local to each request and each absolute position.

The default last-token story uses one logits row per request. Other configured logits modes can return additional positions and offsets; the pipeline chooses `next_token_logits` when appropriate. The full transformer remains necessary to process active prompt rows even when only final-position logits are requested. Sampling and text decoding happen after the decoder’s logits output.
''',[R('transformer','input_row_offsets[1:] - 1'),R('sampling_processor','def logits_for_sampling(')])],'''
LlamaForCausalLM (HF architecture name)
└─ Llama3Model / Llama3Config (MAX implementation names)
   └─ Llama3 → Transformer
      ├─ Embedding: ids [T] → [T,4096]
      ├─ TransformerBlock × 32
      │  ├─ RMSNorm → AttentionWithRope → residual add
      │  └─ RMSNorm → MLP → residual add
      └─ select requested rows → final RMSNorm → lm_head → FP32 logits
''',[R('arch','LlamaForCausalLM'),R('llama','class Llama3'),R('transformer','class TransformerBlock')])

C('build_startup','Build, packages and the serving acceptance gate',
 'How the Python launcher, installed Mojo implementations, model metadata and spawned worker meet.',
 'Model and construction',[
 A('BUILD','Bazel / installation','build process','developer or package build','Produces Python entry points, native bindings and implementation packages.'),
 A('META','Checkpoint metadata','configuration','repository resolver','Identifies architecture, tensor encodings and files; download and memory planning are separate.'),
 A('GATE','Validation','control flow','PipelineConfig','Rejects unsupported architecture/encoding combinations before the downstream model story is assumed.'),
 A('P1','P1 · built-ins','package set','installed MAX runtime','Contains parametric implementations selected later during compilation.'),
 A('WORK','Model worker','process','serving supervisor','Constructs the pipeline/session with serialized configuration and memory plan.'),
 A('S1','S1 · pipeline session','Python/native object','TextGenerationPipeline','Owns decoder and auxiliary graph compilation/initialization policy.')],[
 E('Build the launcher and dependencies','The pipelines Bazel binary depends on the architecture catalog, native runtime and PROD_MOJOPKGS. Packaging makes these available before requests.',{'BUILD':B('build products','pipelines Python launcher','native bindings + runtime'),'P1':B('installed',('p1','PROD_MOJOPKGS'))},[R('build','name = "pipelines"'),R('build','mojo_deps = PROD_MOJOPKGS')],edges=[('BUILD','P1')]),
 E('Resolve the model configuration','The model ID selects metadata and weight files; LlamaForCausalLM dispatches to Llama3Model. Registration is a catalog operation, not a weight load.',{'META':B('resolved metadata','LlamaForCausalLM','dimensions + file encodings')},[R('arch','LlamaForCausalLM'),R('hf','float16')]),
 E('Check the encoding contract','Llama3Config lists float32 and bfloat16 among supported formats but omits float16. Infer an encoding, then validate it against the architecture.',{'GATE':B('explicit acceptance check','resolved encoding ∈ supported encodings?')},[R('config','SUPPORTED_ENCODINGS'),R('pipeline_config','arch.supported_encodings')],edges=[('META','GATE')]),
 E('Continue only for a compatible resolved configuration','This main teaching route assumes compatible unquantized input. No implicit FP16-to-BF16 conversion is asserted. The FP16 rejection branch is selectable above.',{'GATE':B('compatible route assumption','float32 or bfloat16 route'),'WORK':B('configured worker','serialized config + MemoryPlan')},[R('model_config','def _infer_quantization_encoding('),R('planner','class PagedMemoryPlanner')],edges=[('GATE','WORK')]),
 E('Create the session in the pipeline','The worker constructs TextGenerationPipeline. It creates InferenceSession before loading the decoder graph; built-in P1 is available to graph/kernel-library construction.',{'S1':B('created',('s1','InferenceSession(devices=…)'))},[R('pipeline','session = InferenceSession(')],edges=[('WORK','S1'),('P1','S1')]),
 E('Load model state and auxiliary graphs','The pipeline model loads/finalizes its config and graph specs; KV manager and sampler setup are separate pipeline work. A successful build alone does not establish readiness.',{'WORK':B('model construction','Llama3Model + graph specs','KV manager + sampler graphs'),'S1':B('retained',('s1','decoder + auxiliary compilation policy'))},[R('model','def load_graphs('),R('pipeline','load_sampler_graphs(')])],[
 Q('What is built, and what waits until runtime?','''
| Time | Product / action | Relevant distinction from Example 04 |
|---|---|---|
| Bazel / wheel build | Python launcher, architecture modules, native code, production Mojo packages | a model is not baked into the `pipelines` target |
| configuration | resolve model/task/devices, metadata, encoding, memory limits | checkpoint acceptance precedes the source path discussed here |
| model construction | load named weights; initialize/finalize model config; create GraphSpec | Python objects still describe the model |
| tracing | emit G1 with typed inputs, weights and operations | built-in kernels are resolved through the installed library |
| compilation | specialize generated host code and device kernels | shapes/dtypes/target information inform specialization |
| initialization | bind weights and compiled resources | returns reusable M1 |
| request | stage IDs/KV metadata; execute decoder and sampler | no launcher rebuild per token |

The custom-source `mojo precompile` subprocess explained in Example 04 is not automatically repeated for this built-in decoder. `GraphSpec.custom_extensions` defaults to an empty tuple. Backend overlays remain possible through the graph’s resolved extension mechanism; an empty explicit list is not proof that all installed overlays are absent.
''',[R('build','mojo_deps = PROD_MOJOPKGS'),R('spec','custom_extensions:'),R('graph','_resolved_custom_extensions(custom_extensions)')],open=True),
 Q('Why the original FP16 checkpoint cannot be silently assumed to work','''
The saved checkpoint advertises FP16 and its tensor metadata includes F16. This checkout’s `Llama3Config.SUPPORTED_ENCODINGS` does not contain `float16`. The configuration validator checks the effective encoding against the registered architecture and rejects unsupported choices.

The inference code includes specific FP32/BF16 casting decisions; it is not a general promise to convert any FP16 checkpoint. `DEFAULT_ENCODING = "q4_k"` is a class default, not proof that arbitrary safetensors are quantized during startup. A user-specified flag also must resolve to compatible files and supported behavior.

```text
checkpoint files → infer encoding → architecture support check
                                  ├─ supported → model construction
                                  └─ unsupported → configuration error
```

The pages after this gate explain the source-derived unquantized execution path with compatible resolved weights. They do not claim a conversion, successful load, compile or model run. Establishing such a run is a separate concrete validation step.
''',[R('config','SUPPORTED_ENCODINGS'),R('pipeline_config','arch.supported_encodings'),R('model_config','def _infer_quantization_encoding(')],open=True),
 Q('Configuration, planning and worker ownership','''
The parent serving setup resolves model configuration and plans memory; the model worker receives serialized configuration and a bound plan. Parent Python object identity is not shared with the spawned worker. The worker owns the actual serving pipeline, session, initialized models and cache manager used by requests.

Keep startup milestones separate: IPC connection, worker creation, weights loaded, decoder compiled, sampler ready, warm-up and externally visible serving readiness. The startup call-stack reader supplies the larger API/supervisor sequence. This supplement expands the session and model portion of that sequence.

Model metadata includes dimensions and tokenizer/generation policies. It does not report live free GPU memory, allocator reserve, workspace peaks or current cache occupancy. Those belong to device statistics, planning, allocation and runtime accounting.
''',[R('planner','class PagedMemoryPlanner'),R('pipeline','class TextGenerationPipeline')]),
 Q('Scope branches and build verification','''
The primary route is single-device Llama2-style attention without quantization, distributed signal buffers, speculative decoding, sliding-window attention or MLA. These features have explicit branches elsewhere in the shared architecture. Their existence does not make them active for this configuration.

The build wrapper for repository build/test work is `./bazelw`; the documentation builder is a separate Python command documented in the local README. This course does not build MAX or start a server. Its verification checks source anchors, offline assets, state transitions and numerical teaching arithmetic.
''',[R('model','def _single_device_spec('),R('llama','stacked_qkv=config.stacked_qkv')])],'''
//max/python/max/_entrypoints:pipelines
├─ architecture catalog + Python serve dependencies
├─ native engine / driver bindings
└─ PROD_MOJOPKGS → installed P1

max serve --model meta-llama/Llama-2-7b-hf
├─ metadata / task / encoding resolution
├─ supported encoding? → error OR continue
├─ planning and worker setup (see parent startup reader)
└─ TextGenerationPipeline.__init__
   ├─ InferenceSession → S1
   ├─ pipeline model → weights / config / GraphSpec
   └─ KV manager + sampler
''',[R('build','name = "pipelines"'),R('pipeline_config','arch.supported_encodings'),R('pipeline','session = InferenceSession(')],scenarios=[dict(id='fp16',title='Saved FP16 checkpoint: validation stops',steps=[
 E('Saved files resolve to FP16','The saved metadata advertises F16 weights. This is the route that must be checked for the literal original checkpoint.',{'META':B('FP16 checkpoint','torch_dtype=float16','safetensors F16')},[R('hf','float16')]),
 E('FP16 is absent from the architecture support set','The architecture lists supported encodings. FP16 is not in that set in this source revision.',{'GATE':B('unsupported encoding','float16 ∉ SUPPORTED_ENCODINGS')},[R('config','SUPPORTED_ENCODINGS')],edges=[('META','GATE')]),
 E('Stop before asserting successful serving','The validation error means the later source walkthrough is conditional. S1 and model execution are not evidence of a completed run here.',{'GATE':B('configuration rejected','error: unsupported encoding'),'WORK':B('not established'),'S1':B('not reached by this failed route')},[R('pipeline_config','arch.supported_encodings')])])])
