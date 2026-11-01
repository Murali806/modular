## 1. Select and configure Llama

**The checkpoint supplies a model configuration; MAX uses that configuration
to build a decoder graph.** The `llama3` name identifies MAX's implementation
package. Its registration maps the
`LlamaForCausalLM` architecture name to `Llama3Model`.

{{qna:1}}

This guide uses **Llama 2 7B**, one device, and the Graph API path. The working
checkpoint name is `meta-llama/Llama-2-7b-hf`; a checkpoint revision and actual
hardware run have not been captured. Hidden size, heads, vocabulary, and layer
count are read from the checkpoint configuration rather than inferred from
the name “7B.” The Graph API is the default when `prefer_module_v3=False`.

`SOURCE` — the initial serving route is
[`serve_api_server_and_model_worker()`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60)
→ registry factory → model construction in
[`TextGenerationPipeline`](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L274).
The model constructor calls `load_model()`, which selects the decoder returned
by `load_graphs()` on this route.

```text
HF architecture + checkpoint settings + MAX serving settings
                     |
                     v
Llama3Model -> state dictionary -> resolved Llama3Config
                     |
                     v
single-device GraphSpec -> trace -> compile -> initialize
```

| Object | What it contains | When it matters |
|---|---|---|
| `Llama3Config` | Architecture dimensions, dtype, devices, RoPE, KV parameters, output policy | Module construction and tracing |
| `GraphSpec` | Module, graph name, input-type tree, weight registry, optional grouping | Recipe for tracing and compilation |
| `Graph` | Symbolic operations and an MLIR module | Finalization and compilation |
| `CompiledModel` | Compiled artifact handle and expected weight metadata | Initialization and artifact reuse |
| Engine `Model` | Initialized executable state | Execution on concrete buffers |

Expand the registration, then follow the model-loading method. The full
functions include alternative branches so you can see exactly where the
single-device choice is made.

{{cell:registration}}

{{cell:load}}

{{cell:choose}}

### Configuration finalization

`Llama3Config.finalize()` resolves settings that depend on checkpoint metadata:
quantization, normalization, tied embeddings, stacked projections, and output
policy. This mutates a Python configuration object. The symbolic graph has
not been finalized by this call.

{{cell:config_call}}

{{cell:config_finalize}}

<details class="self-check"><summary>Check your understanding: does the name Llama3Model hard-code a Llama 3 8B graph?</summary><p>No. The registration selects this implementation, while the configuration supplies dimensions and model options. A specific checkpoint still needs compatible weights and settings; this source route is not a claim of a completed checkpoint run.</p></details>

## 2. Record the computation

**Tracing calls the Python model on symbolic graph inputs.** The model's tensor
operations append IR operations; they do not consume an actual prompt and
generate tokens at this stage. Python construction and control flow still
execute: the configured layer loop expands the decoder structure into the
graph.

`_single_device_spec()` constructs the module, loads its weights, and captures
its input contract. Weight values can exist before compilation. The later
initialization step binds the named registry to the compiled engine model;
these are different meanings of “loading weights.”

{{cell:single_spec}}

### One block connects the model to the graph

`SOURCE` — the following diagram follows the ordinary residual path in
`TransformerBlock.__call__()`. `T` is the packed token count and `H` is the
configured hidden dimension. This is a shape/dataflow illustration.

```text
tokens [T] -> embedding -> hidden states x [T,H]
                               |
                 +-------------+------------------+
                 |                                |
                 v                                |
             RMSNorm -> attention(Q/K/V, RoPE, KV) |
                 |                                |
                 +---------------------- add <----+
                                          |
                                          h [T,H]
                                          |
                         +----------------+----------------+
                         |                                 |
                         v                                 |
                     RMSNorm -> MLP ---------------------- add
                                                           |
                                                     next block

final block -> selected hidden-state rows -> norm -> LM head -> logits
```

The config chooses attention/linear/MLP implementations and the number of
blocks. `TransformerBlock` exposes both residual paths. The full layer loop
and logits postprocessing show how those blocks become a complete decoder.

{{cell:llama_init}}

{{cell:block}}

{{cell:transform_layers}}

{{cell:logits}}

### Symbolic inputs describe a call boundary

| Input | Declared type / placement | Meaning |
|---|---|---|
| `tokens` | `int64[total_seq_len]`, selected device | All tokens packed into one sequence buffer |
| `row_offsets` | `uint32[input_row_offsets_len]`, selected device | Boundaries between sequences; conventionally `B+1` entries for `B` sequences |
| `return_n_logits` | `int64[return_n_logits]`, CPU | Count input used by the configured variable-logits path |
| KV subtree | Types from `kv_params.get_symbolic_inputs()` | Cache buffers and metadata consumed by attention |

Those are symbolic type declarations. Actual buffer values, their concrete
lengths, and the cache contents arrive during execution. The graph's fixed
architecture dimensions and its variable batch/token dimensions serve
different purposes.

{{cell:ragged}}

### Follow the trace into Graph

```text
GraphSpec.trace()
  -> flatten named input types
  -> Llama3.trace()
     -> define forward adapter for flat inputs and KV reconstruction
     -> Graph(name, forward, input_types=...)
        -> create MLIR module and mo.graph input arguments
        -> forward(*graph.inputs)
        -> Graph.output(returned_values)
```

`Llama3.trace()` reconstructs the nested KV collection from flat graph inputs.
`Graph.__init__()` creates the graph container and its initially empty result
list. Passing `forward` causes automatic tracing and output finalization.
The graph body can initially contain RMO operations; early compiler processing
legalizes them. The container name `mo.graph` does not mean every initial
operation belongs to the MO dialect.

{{cell:spec_trace}}

{{cell:llama_trace}}

{{cell:graph_init}}

<details class="self-check"><summary>Check your understanding: what is executing when the Python layer loop runs during tracing?</summary><p>Python is constructing the symbolic computation. Graph tensor operations record dataflow on symbolic values. Runtime token buffers are supplied to the initialized engine model later. Some host-side setup work can occur during construction; tracing is not equivalent to running Llama inference.</p></details>

## 3. Finalize the graph

**`Graph.output()` completes the graph's output contract and verifies its IR.**
The Llama path reaches it automatically from the `Graph` constructor after
the symbolic forward returns. There is no separate Python `Graph.finalize()`
call on this route.

### Three different objects can be “finalized”

| Name | Object | Meaning |
|---|---|---|
| `Llama3Config.finalize()` | Python model configuration | Resolve checkpoint-dependent settings before tracing |
| `Graph.output()` | Symbolic graph | Install output types, signature, terminator, and names; verify graph IR |
| `FinalizeDeviceGraphsPass` | Marked MGP device-graph model | Finish device-graph collective construction during native compilation |

Select each step below, then open its matching implementation. The state
panels describe the source logic; they are not captured IR from a model run.

{{widget:finalize}}

{{cell:graph_output}}

### What the completed contract provides

- **Output structure:** Python retains a tree describing semantic groups such
  as logits and hidden states, while the engine boundary uses flat values.
- **Types:** `functionType` and `signature` describe actual graph argument and
  result types. Symbolic dimensions are allowed to remain symbolic.
- **Termination:** `mo.output` identifies the SSA values returned by the graph.
- **Names:** engine metadata uses ordinal names such as `input0` and `output0`.
- **Verification:** `_mlir_op.verify()` checks the completed graph. Compiler
  legality, weight-buffer validation, and numerical correctness require
  their corresponding later checks.

### Context exit has a separate job

The `with graph:` context chooses where operations are recorded.
`Graph.__exit__()` unwinds that context through `_enter()` and restores the
previous `CURRENT_GRAPH`. Compilation starts through a later session call.

{{cell:graph_exit}}

{{cell:graph_enter}}

### A small, complete example from the repository

The existing test below exposes finalization without Llama's many layers:
two symbolic inputs → add → output → metadata checks. **`TEST` means the test
definition was inspected; it was not executed in this session.**

{{cell:graph_test}}

<details class="self-check"><summary>Check your understanding: if Graph.output() succeeds, can that Graph already execute a prompt?</summary><p>Its IR contract has been completed and verified. The serving path must still compile or resolve an artifact, initialize it with weights and runtime state, and call the resulting engine Model with concrete buffers.</p></details>

## 4. Cross into the compiler

**The compiled object is produced by the session, after the graph is traced.**
`compile_and_init_graphs()` makes the order explicit: trace → compile or
reuse → gather named weights → initialize. It also retains the input/output
tree information needed by the serving wrapper.

{{cell:compile_graphs}}

### Follow the payload, not just the function names

| Boundary | Payload | Result |
|---|---|---|
| Pipeline → Python session | Completed Graph/Module, extensions | `CompiledModel` handle |
| Python → C++ session | MLIR module capsule, pipeline name, options | Native asynchronous compilation result |
| C++ session → engine | `CompilationRequest` | CPU worker scheduled to process it |
| Engine → frontend | Source pointer, format, settings | Compiled MEF buffer or cached MEF |
| Frontend → compiler | Cloned MLIR module and pipeline options | Lowered module, then MEF serialization |
| MEF reader → session | Parsed compiled-model objects | Artifact available for initialization |

{{widget:native}}

The native implementations are present in this checkout. The Python/C++
crossing is a visible language boundary, not an unexplained stop in the
source trail.

### Python session: dispatch, handoff, and wait

`compile_reusing_mefs()` may resolve a stored artifact. Otherwise
`compile_async()` selects a path or in-memory graph. For a Graph, it collects
expected weight-device metadata and seeds kernel declarations before
`_compile_module()` passes the MLIR capsule into C++.

The process-wide compilation lock spans the asynchronous compile. A completion
callback releases it. The synchronous `compile()` waits on the result and
surfaces errors before returning the artifact.

{{cell:compile_reuse}}

{{cell:compile_async}}

{{cell:compile_module}}

{{cell:compile_wait}}

### C++ session and engine

The C++ session turns the capsule into a `CompilationRequest` with an opaque
source pointer and format. `Engine::Context::compileModel()` schedules a CPU
task. That task calls `compileModelSync()`, which supplies device configuration
and calls `CompiledModels::create()`.

{{cell:native_session}}

{{cell:engine_compile}}

{{cell:engine_sync}}

### Frontend and compiled artifact

For a graph source, `CompiledModels::create()` constructs the MOF frontend,
calls `compileToBinary()`, and reads its returned MEF. For an already compiled
MEF path, it reads the artifact directly.

The frontend clones the source module before applying compiler transformations,
resolves options, and checks the native MEF cache. A miss enters
`convertMOToMEF()`: run the compiler, then serialize MGP to MEF.

{{cell:compiled_create}}

{{cell:frontend}}

{{cell:convert_mef}}

<details class="self-check"><summary>Check your understanding: does every call to a method named compile run compiler passes?</summary><p>No. The configured MEF import path can read an artifact, and the native frontend can return a cached MEF. Both paths ultimately provide compiled-model objects for initialization.</p></details>

## 5. Transform compiler representations

**The graph passes through several representations because different
decisions need different information.** High-level algebra and shapes are
useful early; structured iteration and memory access support fusion; runtime
models expose allocation, lifetime, and execution requirements.

The stage explorer describes the pass schedule proven by this checkout. It
does not assert a particular fused Llama kernel or a GPU speedup. Such claims
need a concrete graph/target dump and, for speed, a measured run.

{{widget:compiler}}

### MO: prepare graph operations

`buildMoPipeline()` loads missing kernel declarations, legalizes RMO operations,
and performs canonicalization and common-subexpression elimination. Options
control shape inference, constant folding, and named pattern fusion. Layout
inference and device validation prepare the graph for subsequent lowering.

This is distinct from `Graph.output()` verification: a structurally valid
graph still needs target-aware compilation and lowering.

{{cell:mo_pipeline}}

### MAP: expose and combine structured computation

After `MOToMAP`, structured kernels make iteration and access relationships
available to the compiler. MAP fusion combines eligible work; later passes
handle accumulating buffers and views before outlining kernel bodies.

A Llama residual addition is visible in `TransformerBlock.__call__()`. Whether
it becomes a separate runtime kernel, merges into an epilogue, or participates
in another transformation depends on the concrete graph, dtype, target, and
eligible patterns. Reading `createFuseKernels()` in the schedule proves that
the fusion pass is invoked, not which final kernels it produces.

{{cell:map_pipeline}}

### MOGG: prepare kernel routines

`MAPToMOGG` lowers structured bodies into the next representation.
`buildMoggPipeline()` legalizes rowwise kernels, conforms call operands,
outlines routines, and wraps functions for tracing. The module is then ready
for `MOGGToMGP`.

{{cell:mogg_pipeline}}

### MGP: plan runtime work and emit Mojo

This stage reasons about model execution resources: buffer reuse, allocation
planning and deferral, model arguments, dependency chains, and runtime
primitives. It also handles marked device graphs and merges equivalent
routines before emitting Mojo.

Graph-level activation/temporary-buffer planning is a different responsibility
from the serving KV-cache manager's request/page policy. Both concern memory,
but their objects and lifetimes differ.

{{cell:mgp_pipeline}}

### JIT and MEF: finish the artifact

`gcPipelineImpl()` connects the builders with explicit conversions. After
MGP processing and Mojo emission, it schedules `JitCompileKernels`, strips
kernel declarations, and finishes JIT lowering. `runGCPipeline()` constructs
the pass manager and calls `pm.run()`; adding passes alone does not execute
them.

MEF serialization occurs in the frontend after successful pipeline execution.
A MEF-backed `CompiledModel` is a compiled artifact. It is not yet the
weight-initialized engine `Model` used for token execution.

{{cell:gc_pipeline}}

{{cell:run_pipeline}}

| Observation point | What to inspect | Evidence needed |
|---|---|---|
| Before native compile | Graph signature, output types, devices, named weight declarations | Source or a graph-building run |
| Stage boundaries | `mo-pre-map`, `mo-to-map`, `map-pre-mogg`, `map-to-mogg`, `mogg-pre-mgp`, `mogg-to-mgp`, `mgp` | Actual IR dumps for the chosen graph/target |
| After Mojo emission | Generated routines; `pre_jit` can intentionally stop before JIT | Source-matched compile inspection |
| After MEF creation | Artifact and model interfaces | Compile result |
| Runtime performance | Kernel launches, allocation/copy behavior, latency | GPU run with workload and target metadata |

`pre_jit` deliberately ends before a runnable artifact can be serialized. The
frontend reports that requested stop; it is different from claiming the
entire graph compiled successfully.

<details class="self-check"><summary>Check your understanding: does one transformer layer imply one GPU kernel?</summary><p>No. A layer contains many operations. Compiler fusion, outlining, and routine reuse determine a different structure, while kernels can launch their own target-specific work. The final kernel set must be inspected for the concrete graph and target.</p></details>

## 6. Initialize, execute, and reuse

**Initialization combines the compiled artifact with weight buffers and
runtime state. Execution supplies the current batch's inputs to that instance.**
The Python session checks expected weight names and CPU/device category when
that metadata is available, checks buffer contiguity, and enters the native
load path. The engine's continuation constructs executable models on a CPU
runtime worker.

The full `init_all()` also contains a virtual-device branch that returns
compile-only stand-ins. This chapter's executable model path assumes the
required real target device is available.

{{cell:init_all}}

{{cell:native_init}}

### Per-batch execution is a small call boundary

`LlamaModelBase.execute()` receives already prepared buffers, calls the
initialized model, and delegates interpretation of its output buffers to the
batch processor. The graph-building and compiler methods are not called in
this method.

{{cell:execute}}

### Prefill and decode change the inputs

Use the two-sequence illustration below to see what changes at the graph
boundary. Token IDs are invented examples; they are not captured model output.
The logical KV history grows between steps, while the runtime supplies the
appropriate KV buffers/metadata through the declared input subtree.

{{widget:buffers}}

Changing concrete token values or lengths within the supported symbolic
contract does not itself alter the Python layer definitions. Changes to
architecture, dtype, devices, output policy, or specialization settings need
the corresponding compatible graph/artifact. This is not a universal promise
that every possible shape change avoids compilation.

### Cached compilation still has a trace in this route

`compile_and_init_graphs()` calls `GraphSpec.trace()` before
`compile_reusing_mefs()`. Therefore a configured MEF import or native cache hit
can skip graph compiler work after tracing has already happened. Other MAX
artifact-loading APIs may enter at a different point.

{{widget:cache}}

### Device-graph finalization is another mechanism

The ordinary `GraphSpec` defaults `is_device_graph=False`.
`FinalizeDeviceGraphsPass` runs within MGP processing but returns immediately
for an unmarked model. For a marked model, it finalizes device-graph
collectives after the staging/build-stub pipeline. That work is separate from
the output signature and `mo.output` installed by Python.

CUDA graph capture/replay also concerns replaying device work. It should not
be used as another name for constructing or finalizing a MAX `Graph`.

{{cell:device_finalize}}

### Diagnose the boundary first

| Symptom | First code to inspect | Question |
|---|---|---|
| Graph output/verification error | `Graph.output()` | Are the output values, chains, and signature valid? |
| Native compilation failure | `_compile_module()` → frontend → compiler diagnostics | Which representation, target rule, or lowering failed? |
| Weight initialization failure | `InferenceSession.init_all()` | Are names, device categories, and buffer layouts compatible? |
| Incorrect inference values | Model execution and a reference implementation | Does numerical behavior match for the same inputs/configuration? |
| Unexpected startup time | Trace, compilation, artifact reuse, and initialization separately | Which phase actually ran, and was an artifact reused? |

The [source map](SOURCE_MAP.md) is the compact reading route. The
[source manifest](source_manifest.json) records every included file and
function range. The [research review](EXPLANATION_RESEARCH.md) explains the
teaching methods used to structure this guide.

<details class="self-check"><summary>Check your understanding: what is the complete lifecycle in one sentence?</summary><p>MAX resolves the Llama configuration, records its symbolic computation, finalizes the graph's interface, compiles or resolves an artifact, initializes executable state with weights, and executes that model on prepared batch buffers.</p></details>
