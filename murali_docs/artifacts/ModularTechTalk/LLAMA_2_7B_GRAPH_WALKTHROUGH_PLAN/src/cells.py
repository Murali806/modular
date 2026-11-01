"""Source selections and authored explanations for the Llama walkthrough."""

REVISION = "610f7111836d9a950d45ba52ec9b3e0ed589fe3f"
LLAMA = "max/python/max/pipelines/architectures/llama3/"
GRAPH = "max/python/max/graph/graph.py"
ENGINE = "max/python/max/engine/api.py"
SPECS = "max/python/max/pipelines/lib/graph_specs.py"
TRANSFORMER = "max/python/max/nn/transformer/transformer.py"
NATIVE = "max/lib/max_license/API/c/"
GC = "GraphCompiler/lib/GraphCompiler.cpp"
FRONTEND = "GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp"


def note(needle, title, text, span=1):
    return dict(needle=needle, title=title, text=text, span=span)


def py(id, file, symbol, purpose, notes=(), evidence="SOURCE"):
    return dict(id=id, file=file, symbol=symbol, purpose=purpose,
                language="python", notes=list(notes), evidence=evidence)


def cpp(id, file, symbol, needle, purpose, notes=(), prefix=0):
    return dict(id=id, file=file, symbol=symbol, needle=needle,
                purpose=purpose, language="cpp", notes=list(notes),
                prefix=prefix, evidence="SOURCE")


CELLS = [
    py("registration", LLAMA + "arch.py", "llama_arch",
       "Map the checkpoint's architecture name to the Llama pipeline implementation.", [
           note('name="LlamaForCausalLM"', "Architecture key", "Lookup uses the architecture name. The example repository IDs are examples, while dimensions come from the selected checkpoint."),
           note("pipeline_model=Llama3Model", "Implementation and adapters", "The registration supplies the pipeline model, tokenizer, supported encodings, and weight adapters.", 8),
       ]),
    py("load", LLAMA + "model.py", "LlamaModelBase.load_graphs",
       "Coordinate model settings, graph recipes, compilation, and initialization.", [
           note("state_dict = self._load_state_dict()", "Resolve the model", "Read/adapt weights, resolve config, and initialize any distributed runtime required by the selected graph.", 3),
           note("specs = self.graph_specs(model_config)", "Trace, compile, initialize", "The helper called here owns tracing as well as compilation and initialization; a GraphSpec is still a recipe.", 3),
           note("self._maybe_release_host_weights", "Finish pipeline wiring", "After initialized models exist, eligible host weights can be released and the batch processor is wired.", 3),
       ]),
    py("choose", LLAMA + "model.py", "LlamaModelBase.graph_specs",
       "Choose the decoder implementation from parallelism and device count.", [
           note("if model_config.data_parallel_degree > 1:", "Ordered selection", "Data parallel takes priority, then multiple-device tensor parallel, then the single-device route used here.", 7),
       ]),
    py("config_call", LLAMA + "model.py", "LlamaModelBase._finalized_config",
       "Initialize model settings and resolve weight-dependent configuration.", [
           note("config = Llama3Config.initialize", "Two configuration steps", "Initialize reads pipeline/HF settings; finalize receives checkpoint state plus output and normalization policy.", 11),
       ]),
    py("config_finalize", LLAMA + "model_config.py", "Llama3Config.finalize",
       "Resolve quantization, projection layout, tied weights, and output policy.", [
           note("has_lm_prefix =", "Normalize names for inspection", "Use canonical Llama-style keys when inspecting checkpoint metadata. This local normalized mapping supports subsequent feature detection.", 17),
           note("quant_config = self._parse_quant_config", "Inspect metadata", "Quantization and normalization information are derived from config and weights before graph modules are constructed.", 13),
           note("self.norm_method =", "Commit resolved settings", "The final fields include tied embeddings, stacked MLP/QKV projections, attention bias, and output policy.", 15),
       ]),
    py("single_spec", LLAMA + "model.py", "LlamaModelBase._single_device_spec",
       "Package the Llama module, input contract, and weight registry into a recipe.", [
           note("single_model = Llama3(model_config)", "Module then recipe", "The module is constructed and weights are loaded before the named input tree and registry are packaged in GraphSpec.", 8),
       ]),
    py("llama_init", LLAMA + "llama3.py", "Llama3.__init__",
       "Assemble the Llama decoder layers from the resolved configuration.", [
           note("rope = create_rope_embedding", "Position encoding", "RoPE configuration includes head count, hidden size, sequence limit, and the target device.", 10),
           note("# Select linear layer class.", "Encoding-specific modules", "Linear, attention, and MLP implementations are chosen from quantization and stacked-projection settings.", 48),
           note("layers = [", "Repeated decoder blocks", "Each block receives attention, MLP, normalization, and KV parameters. The layer count comes from config.", 26),
           note("# Create Embedding and output layers.", "Input/output projections", "Embedding and language-model head dimensions follow config. The output weight can be shared with the embedding.", 25),
       ]),
    py("block", TRANSFORMER, "TransformerBlock.__call__",
       "Record the attention and MLP residual paths for one decoder block.", [
           note("attn_out = self.self_attn(", "Attention path", "Normalize the incoming hidden states and call attention with layer index, KV collection, RoPE data, and sequence boundaries.", 7),
           note("h = x + attn_out", "Two residual additions", "The attention output joins x; the normalized result goes through MLP and joins h. Optional multipliers are explicit branches.", 6),
       ]),
    py("transform_layers", TRANSFORMER, "Transformer._process_hidden_states",
       "Trace the configured layer sequence, then process logits.", [
           note("for idx, layer in enumerate(self.layers):", "Python expands the layer structure", "Each call uses symbolic tensors while tracing. Layer index constants and KV/row-offset inputs are attached to the recorded computation.", 8),
           note("return self._postprocess_logits(", "Decoder output policy", "The final hidden states are passed to the common logits/output-tree construction path.", 6),
       ]),
    py("logits", TRANSFORMER, "logits_postprocess",
       "Select output positions, normalize them, and project to vocabulary logits.", [
           note("last_h = ops.gather", "Last-token positions", "Row offsets identify each sequence's final token. The model gathers those rows before normalization and the language-model head.", 4),
           note("if return_logits == ReturnLogits.VARIABLE:", "Static output-policy branches", "The configured policy selects extra variable-count or all-token logits. return_n_logits supplies values used by the variable path.", 29),
           note("return logits_outputs(", "Named outputs", "Logits, offsets, and optional hidden states form the output tree later flattened by Graph.output.", 6),
       ]),
    py("ragged", "max/python/max/pipelines/lib/interfaces/batch_processor.py",
       "ragged_batch_inputs", "Declare symbolic batch tensor types and placements.", [
           note('"tokens": TensorType(', "Packed tokens and boundaries", "Tokens are int64; row offsets are uint32. Both use the selected device, with symbolic lengths.", 6),
           note('"return_n_logits": TensorType(', "Host-side count input", "This count tensor is int64 on CPU; its symbolic length is declared independently from the token count.", 3),
       ]),
    py("spec_trace", SPECS, "GraphSpec.trace",
       "Flatten the input-type tree and dispatch to the module's trace method.", [
           note("if isinstance(self.module, ModuleV3):", "Variant boundary", "The ModuleV3 branch stages differently. The Graph API route continues to self.module.trace below.", 8),
           note("return self.module.trace(", "Preserve recipe settings", "The graph name, module container, extensions, and device-graph flag are passed with the flattened type leaves.", 7),
       ]),
    py("llama_trace", LLAMA + "llama3.py", "Llama3.trace",
       "Adapt flat graph inputs back into the Llama forward contract.", [
           note("kv_collections = tree.leaves(", "Recover KV structure", "The graph boundary is flat; this adapter reconstructs the per-device KV collection required by the model.", 10),
           note("return Graph(", "Trace through the constructor", "Passing forward to Graph causes the constructor to invoke it on symbolic graph inputs and finalize its returned outputs.", 8),
       ]),
    py("graph_init", GRAPH, "Graph.__init__",
       "Create MLIR graph structure and optionally trace a Python callable.", [
           note("self._params = dict.fromkeys(", "Symbolic dimensions", "Names from symbolic tensor/buffer dimensions become graph parameters.", 7),
           note("self._module = (", "MLIR ownership", "Use a supplied module or create one, then create mo.graph with input types and initially empty result types.", 12),
           note("self._import_kernels", "Kernel declarations and graph mode", "Import the kernel library and mark the graph if explicit device-graph construction was requested.", 7),
           note("if forward is not None:", "Automatic finalization", "Run the callable inside the active graph context. Its return form determines how output is invoked.", 15),
       ]),
    py("graph_output", GRAPH, "Graph.output",
       "Finish the graph's output contract and verify the completed graph.", [
           note("root = (", "1 · Flatten outputs", "Retain the Python output structure; flatten its leaves. A string leaf is rejected because unpacking a dict into its keys is usually a mistake.", 12),
           note("outputs = tuple(", "2 · Normalize values and chains", "Convert leaves to Graph Values and include final chain values when the graph has chain inputs. Chains preserve dependency ordering.", 10),
           note("function_type = mlir.FunctionType.get(", "3 · Set both signatures", "The actual argument and result types define functionType and the kgen generator signature. Symbolic dimensions can remain symbolic.", 9),
           note("self._add_op_generated(", "4 · Append the terminator", "mo.output identifies which SSA values leave the graph. This records IR; it does not launch inference.", 6),
           note('input_names = [f\'"input{i}"\'', "5 · Name the interface", "Ordinal input/result names are written for the engine. The Python output tree is retained separately.", 8),
           note("self._subgraphs = {}", "6 · Verify the graph", "Clear temporary bookkeeping and invoke MLIR graph verification. Failure reports diagnostics. Numerical correctness and target compilation remain separate checks.", 11),
       ]),
    py("graph_exit", GRAPH, "Graph.__exit__",
       "Exit the active graph-construction context.", [
           note("self._context_state.pop()", "Context restoration", "This delegates context exit; output signature construction and native compilation occur elsewhere."),
       ]),
    py("graph_enter", GRAPH, "Graph._enter",
       "Set and restore the active graph while Python operations are recorded.", [
           note("token = CURRENT_GRAPH.set(self)", "Scoped graph state", "The finally block restores the previous graph even when construction raises an exception.", 5),
       ]),
    py("graph_test", "max/tests/tests/graph/test_graph.py", "test_add_op",
       "Read a complete existing test that builds an add graph and checks interface metadata.", [
           note("graph.output(elemwise_sum)", "Finalization in a small graph", "This explicit output call establishes the result contract; the following assertions inspect its names.", 5),
       ], evidence="TEST"),
    py("compile_graphs", SPECS, "compile_and_init_graphs",
       "Trace graph recipes, compile artifacts, then initialize their weights.", [
           note("if len(members) == 1:", "Trace each compilation group", "The ordinary Llama decoder has one spec. Multiple grouped roles can instead share a graph Module.", 13),
           note("compiled = session.compile_reusing_mefs(", "Compilation boundary", "The completed graph or module is compiled or resolved through configured artifact reuse.", 3),
           note("weights: dict[str, DLPackArray] = {}", "Bind after compilation", "Collect weights by name, reject conflicting values, and initialize all models in the compiled artifact.", 8),
       ]),
    py("compile_reuse", ENGINE, "InferenceSession.compile_reusing_mefs",
       "Select normal compilation, configured MEF import, or MEF export.", [
           note("store = (", "Choose the store", "Use the explicit session store if present, otherwise the default. Non-Graph modules take the regular compile route.", 13),
           note("if not store.exporting:", "Import or decline", "A matching artifact path is passed to compile. A store that declines this graph falls back to compiling the graph.", 11),
           note("compiled.export_mef", "Export mode", "Write the compiled artifact under the store's claimed export location."),
       ]),
    py("compile_async", ENGINE, "InferenceSession.compile_async",
       "Prepare a Graph, Module, or artifact path for asynchronous native compilation.", [
           note("if isinstance(model, Path | str):", "Source-kind dispatch", "Paths and in-memory Graphs follow different branches. A Graph contributes its MLIR module and extension libraries.", 7),
           note("expected_weights = {", "Retain initialization expectations", "The graph's weight declarations supply expected device metadata, checked when weight buffers are bound.", 4),
           note("kernel_library = getattr", "Seed declarations", "Reuse kernel declarations already loaded by the Graph's KernelLibrary before the native module handoff.", 7),
           note("compiled = CompiledModel(", "Return an artifact handle", "The handle carries the asynchronous result and graph names. compile() below waits for completion.", 6),
       ]),
    py("compile_wait", ENGINE, "InferenceSession.compile",
       "Wait for asynchronous compilation and surface errors to the caller.", [
           note('with _record_phase("compile_seconds"):', "Synchronous completion", "Schedule compilation, wait on its result, and distinguish requested pre-JIT stopping from a compile failure.", 22),
       ]),
    py("compile_module", ENGINE, "InferenceSession._compile_module",
       "Cross the Python/C++ boundary with the MLIR module capsule.", [
           note("_COMPILATION_LOCK.acquire()", "Serialize native compilation", "The process-wide lock is acquired before scheduling. Synchronous setup errors release it immediately.", 11),
           note("module.mlir_module._CAPIPtr", "Payload", "The capsule identifies the MLIR module; extensions, a pipeline name, and the fusion option accompany it.", 4),
           note("handle.add_done_callback", "Asynchronous lifetime", "Completion releases the lock through a callback. Returning a pending handle does not end the protected compile interval."),
       ]),
    cpp("native_session", "max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp",
        "Core::InferenceSession::compile", "const nb::capsule &model,",
        "Turn the Python module capsule into a native CompilationRequest.", [
            note("compileConfig.setModelSource", "Opaque source and format", "The frontend must interpret the pointer using the matching MLIR implementation and file format.", 4),
            note("compileModel(*runtimeContext", "Schedule engine compilation", "Hand the populated request to the engine and return its asynchronous result.", 2),
        ], prefix=1),
    cpp("engine_compile", NATIVE + "EngineContext.cpp", "Engine::Context::compileModel",
        "Engine::Context::compileModel(CompilationRequest config) const",
        "Schedule compilation on a CPU worker and resolve an asynchronous result.", [
            note("addTask(cpuDevice", "Worker boundary", "The captured request is processed by compileModelSync on the CPU worker; error and telemetry handling resolve the returned handle.", 5),
        ], prefix=1),
    cpp("engine_sync", NATIVE + "EngineContext.cpp", "Engine::Context::compileModelSync",
        "Engine::Context::compileModelSync(CompilationRequest config) const",
        "Supply device configuration and create native compiled-model objects.", [
            note("config.setDevices", "Target context", "Pass configured devices into the compilation request, then dispatch to CompiledModels::create.", 4),
        ], prefix=1),
    cpp("compiled_create", NATIVE + "CompiledModels.cpp", "CompiledModels::create",
        "CompiledModels::create(M::ContextRef context,",
        "Compile in-memory source or read a model artifact from a path.", [
            note("if (auto modelSourceOr", "Graph-source branch", "A source pointer selects the MOF frontend, which produces a MEF buffer. read then builds compiled-model objects from that buffer.", 17),
            note("if (modelCtx.model.format() == FileFormat::kMEFFile)", "Already compiled path", "A MEF path is read directly. Entering an API named compile does not imply compiler passes run.", 3),
        ], prefix=1),
    cpp("frontend", FRONTEND, "MOFContext::compileToBinary(void*, FileFormat)",
        "MOFContext::compileToBinary(void *source,",
        "Clone the graph module, configure lowering, and consult the native MEF cache.", [
            note("ModuleOp::getFromOpaquePointer(source).clone()", "Compiler-owned module", "Lowering transforms a clone of the incoming module. Weight resources are owned by its MLIR context."),
            note("overrideLoweringOptions", "Resolve compiler options", "Collect configuration and detect pre-seeded kernel declarations before deriving cache behavior.", 10),
            note("if (maybeMef) {", "Cache-hit return", "When model-IR caching is enabled and a matching artifact exists, return it before convertMOToMEF runs.", 7),
            note("auto outBufOr =", "Cache miss", "Compile/serialize, then attempt to insert the resulting buffer in the cache.", 10),
        ]),
    cpp("convert_mef", FRONTEND, "convertMOToMEF", "static ErrorOr<BufferRef> convertMOToMEF",
        "Run graph compilation, then serialize the lowered model to MEF.", [
            note("MO::runGCPipeline", "Execute compiler passes", "A failed pipeline returns graph diagnostics rather than an artifact.", 3),
            note("if (options.preJit)", "Intentional early stop", "Pre-JIT inspection ends after Mojo emission; the partially lowered result is not serialized to the MEF cache.", 3),
            note("MO::convertMGPToMEF", "Artifact boundary", "Serialize the successfully lowered module and return its MEF buffer.", 4),
        ]),
    cpp("mo_pipeline", GC, "buildMoPipeline", "void M::MO::buildMoPipeline",
        "Prepare and optimize high-level graph operations before MAP lowering.", [
            note("MOGG::createImportPackages", "Kernel definitions", "Import/verify packages only when declarations were not already seeded by the caller.", 4),
            note("if (opts.useShapeInference)", "Shape and symbolic work", "Shape inference and constant folding are configured independently; symbolic optimization and cleanup run in this stage.", 8),
            note("MO::createInferLayouts", "Layouts and optional patterns", "Infer layouts and optionally run named pattern fusion before validating device information.", 11),
            note("MO::createValidateDevicesPass", "Target legality", "Check device information before the final cleanup and unused-operation removal."),
        ]),
    cpp("map_pipeline", GC, "buildMapPipeline", "void M::MO::buildMapPipeline",
        "Fuse eligible structured computations and outline kernel bodies.", [
            note("MAP::createFuseKernels", "Structural fusion", "Fusion is a transformation of eligible graph structure. The schedule alone does not prove which Llama operations fuse.", 3),
            note("MAP::createAllocateAccumulatingBuffers", "Output storage", "Provide accumulating buffers after fusion, including mixed-precision handling."),
            note("MAP::createOutlineStructuredKernelBodies", "Outline bodies", "Create structured kernel bodies after view handling, then simplify their captures and invariant views.", 10),
        ]),
    cpp("mogg_pipeline", GC, "buildMoggPipeline", "void M::MO::buildMoggPipeline",
        "Legalize kernel calls and outline routines.", [
            note("MOGG::createLegalizeRowwiseKernels", "Call preparation", "Legalize rowwise kernels, conform operands, outline routines, and add tracing wrappers before module cleanup.", 8),
        ]),
    cpp("mgp_pipeline", GC, "buildMgpPipeline", "void M::MO::buildMgpPipeline",
        "Plan runtime resources, lower primitives, and emit Mojo routines.", [
            note("MGP::createSimpleBufferReusePass", "Memory planning", "Buffer reuse, allocation planning, deferral, and argument validation operate on the runtime-model representation.", 7),
            note("MGP::createStageDeviceGraphsPass", "Device-graph path", "Staging, dependency computation, build stubs, and finalization are separate from the frontend Graph.output operation.", 13),
            note("MOGG::createMergeIdenticalFuncs", "Deduplicate routines", "Earlier canonicalization makes equivalent routines eligible for merging. Layer count is not necessarily the count of distinct generated routines."),
            note("MOGG::createEmitMojo", "Emit source for JIT", "Mojo emission is at the end of this builder. The enclosing pipeline schedules kernel JIT afterward."),
        ]),
    cpp("gc_pipeline", GC, "gcPipelineImpl", "static void gcPipelineImpl",
        "Build the ordered sequence of representation conversions and compiler stages.", [
            note('buildMoPipeline(p, moOpts); }, "mo-pre-map"', "High-level graph stage", "The mo-pre-map dump follows MO processing; subsequent stage names distinguish conversions from stage-local transforms."),
            note("p.addPass(MO::createMOToMAP())", "Representation transition", "Stable-tensor marking precedes MO-to-MAP conversion.", 3),
            note("buildMoggPipeline(p, moggOpts)", "Kernel representation", "MOGG preparation precedes conversion into the MGP runtime model.", 7),
            note("if (!skipJit) {", "JIT configuration", "JIT target and compilation options are only configured when JIT is enabled.", 11),
            note("p.addPass(MOGG::createJitCompileKernels", "Compile kernels", "After MGP/Mojo emission, run JIT compilation, strip kernel declarations, and finish runtime-call lowering.", 3),
        ]),
    cpp("run_pipeline", GC, "runGCPipeline", "LogicalResult M::MO::runGCPipeline",
        "Construct the pass manager and run the configured compiler pipeline.", [
            note("gcPipelineImpl(pm, opts", "Build then execute", "Adding passes describes the schedule. pm.run applies that schedule to the module and returns success or failure.", 2),
        ]),
    cpp("device_finalize", "GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp",
        "FinalizeDeviceGraphsPass", "class FinalizeDeviceGraphsPass",
        "Finalize collectives only for an MGP model marked as a device graph.", [
            note("if (!model.getIsDeviceGraph())", "Explicit gate", "An unmarked model returns immediately. The pass's name refers to device-graph collectives, not Python graph output finalization.", 2),
            note("for (DeviceGraphCollectiveOp", "Finalize each collective", "Failures from collective finalization signal a compiler pass failure.", 4),
        ]),
    py("init_all", ENGINE, "InferenceSession.init_all",
       "Validate weight buffers and initialize models in a compiled artifact.", [
           note("if compiled._expected_weights is not None:", "Weight contract", "Check names and expected CPU/device category captured from the graph. These checks occur after graph finalization and compilation.", 20),
           note("_raise_if_not_contiguous(weight_value)", "Buffer layout check", "Non-contiguous registered weights are rejected with the offending weight's name.", 5),
           note("models = self._impl._load_all(", "Native initialization", "Bind the registry through the native load path; return models keyed by their names.", 9),
       ]),
    cpp("native_init", NATIVE + "EngineContext.cpp", "Engine::Context::initModel(CompiledModels)",
        "Engine::Context::initModel(AsyncValueRef<CompiledModels> compiledModels,",
        "Create executable model instances once compilation resolves.", [
            note(".andThenAsync(", "Initialize on a runtime worker", "The continuation uses the compiled artifact and captured weight registry on the CPU runtime worker.", 11),
            note("createExecutableModels(compiledModelsRef.get()", "Executable state", "Create the executable models with runtime configuration and actual weight buffers, then resolve the result.", 7),
        ], prefix=1),
    py("execute", LLAMA + "model.py", "LlamaModelBase.execute",
       "Execute an initialized decoder on the current batch's buffers.", [
           note("self.model.execute(*model_inputs.buffers)", "Per-batch call", "Prepared token/KV buffers enter the existing model. The batch processor interprets its output buffers.", 4),
       ]),
]
