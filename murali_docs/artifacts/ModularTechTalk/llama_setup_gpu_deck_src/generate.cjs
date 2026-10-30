#!/usr/bin/env node

/*
 * Generates a progressive-build PowerPoint and an object-animated HTML deck
 * from one source model. Run: npm install && npm run build
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const AdmZip = require("adm-zip");
const SHAPE = new PptxGenJS().ShapeType;

const OUT = path.resolve(__dirname, "..");
const PPTX = path.join(OUT, "LLAMA_7B_SETUP_TO_GPU_ANIMATED.pptx");
const HTML = path.join(OUT, "LLAMA_7B_SETUP_TO_GPU_INTERACTIVE.html");
const REV = "90ede769dad97c0b1edb3b245b7968f635d77c1b";

const colors = {
  ink: "171717", paper: "F6F4EF", muted: "B7B5AE", line: "3C3B37",
  green: "36D17D", teal: "27C2B8", coral: "FF6B4A", yellow: "FFD166",
  blue: "5AB2FF", violet: "A78BFA", red: "FF4D67"
};

const scenes = [
  {
    phase: "Map", title: "The MAX execution stack",
    lead: "Keep Python graph construction, native compilation, Mojo kernels, and vendor GPU execution as separate layers.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["MAX Serve / Python", "Settings, registry, PipelineConfig, Llama3Model, GraphSpec", "serve_api_and_model_worker.py", "serve_api_server_and_model_worker()", "coral"],
      ["MAX Graph / MLIR", "Value objects, mo.GraphOp, operation IR, signature, verification", "max/python/max/graph/graph.py", "Graph.__init__(), Graph.output()", "teal"],
      ["MAX native engine", "CompiledModels, MEF, ExecutableModel, TensorMap, AsyncValue", "max/python/max/engine/api.py", "compile_async(), init_all(), _Model_execute()", "violet"],
      ["Mojo kernels + driver HAL", "Kernel registrations, Device, Context, Buffer, Queue, Stream", "max/kernels + max/driver", "register(), enqueue_copy(), enqueue_function()", "green"],
      ["Vendor driver + GPU", "Modules, streams, DMA, grids, blocks, SM execution", "Native/vendor boundary", "CUDA/HIP/Metal driver entry points", "red"]
    ]
  },
  {
    phase: "Map", title: "Three meanings of finalization",
    lead: "These functions happen in order but finalize different objects.",
    nodes: [
      ["Llama3Config.finalize()", "Resolve dimensions and policy from HF config plus actual weights.", "llama3/model_config.py", "initialize(), finalize()", "yellow"],
      ["Graph.output()", "Finalize output tree, MLIR signature, terminator, names, and whole-graph verification.", "graph/graph.py", "Graph.output(), Graph._verify_op()", "teal"],
      ["InferenceSession.compile*()", "Hand the finalized MLIR module to the native graph compiler.", "engine/api.py", "compile_async(), _compile_module()", "violet"]
    ]
  },
  {
    phase: "Serve setup", title: "max serve resolves the Llama pipeline",
    lead: "Startup resolves architecture policy before expensive model creation.",
    nodes: [
      ["PipelineArgs", "Model path, devices, encoding, max length, and serving settings.", "serve_api_and_model_worker.py", "serve_api_server_and_model_worker()", "coral"],
      ["PipelineConfig", "Resolved checkpoint and runtime policy.", "pipelines/lib/config/config.py", "PipelineConfig.from_args(), configure_session()", "yellow"],
      ["PipelineRegistry", "LlamaForCausalLM maps to Llama3Model, tokenizer, adapters, and MemoryPlan.", "pipelines/lib/registry.py", "retrieve_factory(), retrieve_architecture()", "teal"],
      ["TextGenerationPipeline", "Creates Device objects, InferenceSession, weights, model, KV manager, and sampler.", "pipeline_variants/text_generation.py", "TextGenerationPipeline.__init__()", "green"]
    ]
  },
  {
    phase: "Serve setup", title: "DeviceSpec, Device, and DeviceRef",
    lead: "CLI intent becomes a concrete runtime device, while the graph stores symbolic placement.",
    nodes: [
      ["DeviceSpec", "Requested device such as gpu:0.", "max/python/max/driver/driver.py", "DeviceSpec.accelerator(), load_device()", "coral"],
      ["Device", "Concrete Accelerator(0) object used by InferenceSession and allocations.", "max/python/max/driver/driver.py", "load_devices(), Accelerator()", "blue"],
      ["DeviceRef", "Symbolic placement embedded in TensorType and BufferType.", "max/graph type APIs", "DeviceRef.from_device()", "teal"]
    ]
  },
  {
    phase: "Model setup", title: "LlamaModelBase.load_graphs()",
    lead: "The central coordinator moves from checkpoint state to compiled decoder Model.",
    nodes: [
      ["Load state", "_load_state_dict() reads and adapts checkpoint entries to MAX names.", "llama3/model.py + weight_adapters.py", "_load_state_dict(), convert_safetensor_state_dict()", "coral"],
      ["Finalize config", "_create_model_config() resolves dimensions, RoPE, dtype, KV policy, and devices.", "llama3/model.py + model_config.py", "_finalized_config(), initialize(), finalize()", "yellow"],
      ["Build GraphSpec", "graph_specs() selects single-device, tensor-parallel, or data-parallel construction.", "llama3/model.py", "graph_specs(), _single_device_spec()", "teal"],
      ["Compile + init", "compile_and_init_graphs() traces, compiles, binds weights, and returns CompiledCallable.", "pipelines/lib/graph_specs.py", "compile_and_init_graphs()", "green"]
    ]
  },
  {
    phase: "Model setup", title: "Typical Llama 7B math",
    lead: "Resolved dimensions determine symbolic shapes and repeated transformer operations.",
    layout: "math",
    nodes: [
      ["Dimensions", "L=32, d=4096, h=32, d_h=d/h=128, d_ff approximately 11008.", "llama3/model_config.py", "Llama3Config.finalize()", "yellow"],
      ["Parameter estimate", "P approximately Vd + L(4d^2 + 3d*d_ff), approximately 6.7B. BF16/FP16 weights approximately 13.4 GB.", "Architecture math", "parameter-count estimate", "green"],
      ["Attention", "Q=aWq, K=aWk, V=aWv; P=softmax(QK^T/sqrt(d_h)+mask); u=x+(PV)Wo.", "llama3/llama3.py", "Llama3.__call__()", "teal"],
      ["Gated MLP", "m=RMSNorm(u); y=u+(SiLU(mWg) elementwise-multiply mWu)Wd.", "llama3/llama3.py", "StackedMLP.__call__()", "coral"]
    ]
  },
  {
    phase: "Graph", title: "GraphSpec is the trace recipe",
    lead: "It packages the model module, graph name, symbolic input tree, weights, extensions, and grouping.",
    nodes: [
      ["module", "Llama3(config), already populated with Weight objects.", "pipelines/lib/graph_specs.py", "GraphSpec.module", "yellow"],
      ["input_types", "Tokens, row offsets, return_n_logits, and KV cache leaves with DeviceRef placement.", "llama3/llama3.py", "input_tree(), input_types()", "blue"],
      ["weights", "Name to DLPack-compatible host or device value.", "pipelines/lib/graph_specs.py", "weights_registry()", "coral"],
      ["trace()", "Calls Llama3.trace() and returns a finalized Graph.", "pipelines/lib/graph_specs.py", "GraphSpec.trace()", "teal"]
    ]
  },
  {
    phase: "Graph", title: "Llama3.trace() records a symbolic program",
    lead: "No model tensor math executes during tracing; Python module calls append MLIR operations.",
    nodes: [
      ["Graph constructor", "Creates builtin.ModuleOp, mo.GraphOp, block arguments, chains, device metadata, and kernel paths.", "max/python/max/graph/graph.py", "Graph.__init__()", "teal"],
      ["forward(Value...)", "Unflattens KV inputs and invokes Llama3.__call__ with symbolic Values.", "llama3/llama3.py", "Llama3.trace()::forward", "blue"],
      ["Record operations", "Embedding, RMSNorm, QKV, RoPE, attention, MLP, and logits become graph IR.", "llama3/llama3.py", "Llama3.__call__()", "yellow"],
      ["Graph.output", "Finalizes the returned output tree and verifies the complete graph.", "max/python/max/graph/graph.py", "Graph.output()", "green"]
    ]
  },
  {
    phase: "Graph", title: "Graph.output() finalizes mo.graph",
    lead: "This is graph finalization, not graph compilation.",
    nodes: [
      ["Flatten outputs", "Preserve the named output-tree structure and collect leaves.", "graph/graph.py", "tree.flatten()", "teal"],
      ["Create MLIR types", "Bridge MAX Value types through C-API MLIR handles.", "graph/graph.py", "Graph.output()", "blue"],
      ["Write signature", "Set functionType and kgen generator signature.", "graph/graph.py", "Graph.output()", "violet"],
      ["Add terminator", "Append mo.OutputOp with final values.", "graph/graph.py", "Graph._add_op_generated()", "coral"],
      ["Name inputs/results", "Write argument_names and result_names for engine execution.", "graph/graph.py", "Graph.output()", "yellow"],
      ["Verify whole graph", "self._mlir_op.verify() checks the completed mo.graph.", "graph/graph.py", "Graph.output(), _capturing_mlir_diagnostics()", "green"]
    ]
  },
  {
    phase: "Compile", title: "Finalized MLIR becomes CompiledModel",
    lead: "The visible Python path ends at a native C-API pointer and asynchronous compile handle.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["Graph.module", "Finalized MLIR plus expected weights, graph names, and custom extension paths.", "max/python/max/engine/api.py", "InferenceSession.compile_async()", "teal"],
      ["Seed kernel declarations", "KernelLibrary analysis seeds declarations and opaque-type mappings.", "max/python/max/graph/graph.py", "KernelLibrary.load_paths(), seed_kernel_decls()", "green"],
      ["_impl.compile", "Acquires the process-global compile lock and calls native compilation with module._CAPIPtr.", "max/python/max/engine/api.py", "InferenceSession._compile_module()", "violet"],
      ["CompiledModel", "Wraps AsyncValue<CompiledModels>/MEF; weights and persistent device memory are not initialized.", "max/include/max/max_license/API/c/CompiledModels.h", "CompiledModels::create()", "yellow"]
    ]
  },
  {
    phase: "Compile", title: "Visible native compilation contract",
    lead: "The repository exposes compilation objects while the exact lowering pass pipeline remains a boundary.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["CompilationRequest", "ModelSource, pipeline name, DeviceConfig, kernel-library paths, and tile-fusion flags.", "CompiledModels.h", "CompilationRequest::setModelSource(), setDevices()", "teal"],
      ["Engine::Context", "Owns program compilation/execution context and calls compileModel().", "EngineContext.h", "Context::compileModel()", "violet"],
      ["CompiledModels", "Contains SourceModel, framework context, compiled model buffer/MEF, and uninitialized models.", "CompiledModels.h", "CompiledModels::create(), read()", "yellow"],
      ["Native boundary", "Exact graph passes, fusion, target code generation, and module creation are not inferred here.", "Native graph compiler", "implementation boundary", "red"]
    ]
  },
  {
    phase: "Compile", title: "Where Mojo kernels enter",
    lead: "Operation intent is matched to device, API, architecture, and model-specific implementations.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["Graph operation", "For example mo.matmul with dtype, shape, layout, and DeviceRef metadata.", "MAX graph IR", "ops.matmul / mo.matmul", "teal"],
      ["Kernel registry", "@extensibility.register() matches operation, type, API, architecture, and model.", "extensibility/decorators.mojo", "register()", "violet"],
      ["Mojo implementation", "Built-in/custom kernel may use native GPU code or a vendor library.", "builtin_kernels/linalg.mojo", "@extensibility.register(\"mo.matmul\")", "green"],
      ["Device function", "Compiled bundle/function is loaded by runtime Context and launched through Queue/Stream.", "driver/_hal/context.mojo", "load_bundle(), load_function()", "red"]
    ]
  },
  {
    phase: "Initialize", title: "CompiledModel is not executable Model",
    lead: "Compilation creates reusable artifact state; initialization binds persistent runtime state.",
    nodes: [
      ["CompiledModel", "Compiled code/artifact, graph names, and expected weight metadata. No bound persistent allocations.", "max/python/max/engine/api.py", "class CompiledModel", "violet"],
      ["init_all()", "Validates the weight registry and crosses to _impl._load_all().", "max/python/max/engine/api.py", "InferenceSession.init_all()", "teal"],
      ["Model", "Initialized ExecutableModel wrapper with resolved devices, weight bindings, and runtime state.", "EngineContext.h + engine/api.py", "Context::initModel(), Model", "green"]
    ]
  },
  {
    phase: "Initialize", title: "Initialization binds weights and devices",
    lead: "Python checks contracts; native init allocates and materializes executable state.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["weights_registry", "Merged GraphSpec name-to-value map. Conflicting values for one name are rejected.", "pipelines/lib/graph_specs.py", "weights_registry(), compile_and_init_graphs()", "yellow"],
      ["Python validation", "Expected name, CPU/GPU DLPack device type, and contiguous storage.", "max/python/max/engine/api.py", "InferenceSession.init_all(), _raise_if_not_contiguous()", "teal"],
      ["Native init", "_impl._load_all() invokes Engine::Context::initModel().", "EngineContext.h", "Context::initModel()", "violet"],
      ["Persistent state", "Resolve DeviceRef, allocate runtime/device storage, bind/copy weights, return Model.", "Native engine/runtime", "initialization boundary", "green"]
    ]
  },
  {
    phase: "Driver", title: "Request inputs move from host to GPU",
    lead: "The Graph-path batch processor explicitly creates host buffers and moves tokens and row offsets.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["NumPy arrays", "tokens_np and offsets_np encode a ragged request batch.", "interfaces/batch_processor.py", "build_single_replica_ragged_token_arrays()", "coral"],
      ["Host Buffer", "Buffer.from_numpy() uses contiguous DLPack-compatible host storage.", "max/python/max/driver/buffer.py", "_from_numpy(), _from_dlpack()", "yellow"],
      ["Buffer.to(device0)", "Native allocation/copy path enqueues host-to-device movement.", "max._core driver boundary", "Buffer.to()", "blue"],
      ["GPU Buffer", "tokens and input_row_offsets now match graph placement on Accelerator(0).", "interfaces/batch_processor.py", "prepare_initial_token_inputs()", "green"]
    ]
  },
  {
    phase: "Driver", title: "Ragged token buffer example",
    lead: "One flat token buffer plus offsets represents variable-length request rows.",
    layout: "math",
    nodes: [
      ["Rows", "Request A has 3 tokens; request B has 2 tokens.", "Batching example", "context_batch", "coral"],
      ["Flat tokens", "tokens = [t0, t1, t2, t3, t4]", "batch_processor.py", "Buffer.from_numpy(tokens_np)", "yellow"],
      ["Offsets", "offsets = [0, 3, 5]", "batch_processor.py", "Buffer.from_numpy(offsets_np)", "teal"],
      ["Lookup", "row r = tokens[offsets[r] : offsets[r+1]]", "Ragged representation", "offset slicing", "green"]
    ]
  },
  {
    phase: "Driver", title: "MAX driver HAL object graph",
    lead: "Mojo exposes ownership, function loading, buffer storage, and ordered submission objects.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["Device", "Device handle, RawDriver ownership, id, and properties.", "max/driver/src/max_license/_hal/device.mojo", "Device._create()", "blue"],
      ["Context", "Device context, function cache, bundle loading, and function loading.", "_hal/context.mojo", "load_bundle(), load_function()", "violet"],
      ["Buffer", "MemoryHandle, byte_size, pinned flag, and owning Context.", "_hal/buffer.mojo", "Buffer, BufferView", "yellow"],
      ["Queue", "Command queue for launches, events, and synchronization.", "_hal/queue.mojo", "Queue", "green"],
      ["Stream", "Queue wrapper with chain event and copy methods.", "_hal/stream.mojo", "copy_to_device(), copy_from_device()", "coral"]
    ]
  },
  {
    phase: "Driver", title: "DeviceContext.enqueue_copy() overloads",
    lead: "Copy direction selects a Stream method and validates device/driver constraints.",
    nodes: [
      ["Host -> Device", "host pointer or HostBuffer to DeviceBuffer calls copy_to_device().", "_device_context_hal/__init__.mojo", "enqueue_copy(), Stream.copy_to_device()", "green"],
      ["Device -> Host", "DeviceBuffer to host pointer or HostBuffer calls copy_from_device().", "_device_context_hal/__init__.mojo", "enqueue_copy(), Stream.copy_from_device()", "coral"],
      ["Same-device D2D", "DeviceBuffer to DeviceBuffer on one device calls copy_intra_device().", "_device_context_hal/__init__.mojo", "Stream.copy_intra_device()", "blue"],
      ["Peer-device D2D", "Same-driver peer copy calls copy_inter_device(); cross-driver is rejected.", "_device_context_hal/__init__.mojo", "Stream.copy_inter_device()", "violet"]
    ]
  },
  {
    phase: "GPU execute", title: "Model.execute() crosses the runtime boundary",
    lead: "Python validates concrete Buffer inputs before native TensorMap and AsyncValue execution.",
    evidence: "SOURCE + BOUNDARY",
    nodes: [
      ["LlamaModelBase.execute", "Forwards model_inputs.buffers to the initialized engine Model.", "llama3/model.py", "LlamaModelBase.execute()", "green"],
      ["_Model_execute", "Checks contiguity and wraps external DLPack values as Buffer.", "max/python/max/engine/api.py", "_Model_execute()", "teal"],
      ["executeFromDriver", "Converts driver-backed inputs into native execution and extracts device results.", "ModelExecution.h", "executeFromDriver(), extractDriverResults()", "violet"],
      ["Queue + GPU", "Runtime schedules compiled functions, dependencies, and output Buffer handles.", "Native runtime + driver", "Context::execute(), queue launch", "red"]
    ]
  },
  {
    phase: "GPU execute", title: "One decode-step dependency timeline",
    lead: "Source proves dependency order; duration and overlap require target-GPU measurement.",
    evidence: "SOURCE + GPU-LAB",
    nodes: [
      ["CPU packing", "Build tokens, offsets, and scalar control buffers.", "batch_processor.py", "prepare_initial_token_inputs()", "coral"],
      ["H2D", "Move request buffers to accelerator storage.", "Buffer / driver HAL", "Buffer.to(), copy_to_device()", "blue"],
      ["Runtime execute", "Resolve dependencies and submit the compiled model.", "engine/api.py + EngineContext.h", "_Model_execute(), Context::execute()", "violet"],
      ["GPU kernels", "RMSNorm, QKV, RoPE, attention, MLP, logits, and KV updates.", "Mojo built-in kernels", "selected device functions", "red"],
      ["D2H token", "Copy generated token through a staging Buffer and read with to_numpy().", "text_generation.py", "inplace_copy_from(), to_numpy()", "green"]
    ]
  },
  {
    phase: "GPU execute", title: "KV cache is persistent device state",
    lead: "The decoder graph receives KV buffers; attention reads prior keys/values and appends the current position.",
    nodes: [
      ["KV manager", "Allocates and owns cache pages according to the memory plan.", "pipelines/kv_cache", "KV cache manager construction", "yellow"],
      ["Graph inputs", "KV leaves are symbolic BufferType/tensor inputs unflattened by Llama3.trace().", "llama3/llama3.py", "unflatten_kv_inputs()", "teal"],
      ["Attention read", "Current queries attend over cached keys and values.", "builtin_kernels/attention.mojo", "paged/ragged attention registrations", "red"],
      ["KV append", "Current-token key/value vectors are stored in the active page/slot.", "builtin_kernels/kv_cache.mojo", "paged KV operations", "green"]
    ]
  },
  {
    phase: "Map", title: "Files that own each phase",
    lead: "Use this as the source-navigation index.",
    layout: "files",
    nodes: [
      ["Serve", "serve_api_and_model_worker.py", "max/python/max/_entrypoints/cli/serve", "serve_api_server_and_model_worker()", "coral"],
      ["Pipeline", "registry.py + text_generation.py", "max/python/max/pipelines/lib", "retrieve_factory(), TextGenerationPipeline.__init__()", "yellow"],
      ["Model", "model.py + model_config.py + llama3.py", "pipelines/architectures/llama3", "load_graphs(), finalize(), trace(), execute()", "green"],
      ["Graph", "graph.py + graph_specs.py", "max/python/max/graph + pipelines/lib", "Graph.output(), compile_and_init_graphs()", "teal"],
      ["Engine", "api.py + EngineContext.h + CompiledModels.h", "max/python/max/engine + max/include", "compile*(), init_all(), Context::initModel()", "violet"],
      ["Driver", "buffer.py + _hal/*.mojo", "max/python/max/driver + max/driver/src", "Buffer.to(), enqueue_copy(), Queue, Stream", "blue"],
      ["Kernels", "builtin_kernels/*.mojo", "max/kernels/src/graph_compiler", "@extensibility.register(...)", "red"]
    ]
  },
  {
    phase: "Map", title: "Debug the phase that owns the failed object",
    lead: "Do not diagnose compilation as checkpoint loading or runtime synchronization as graph tracing without evidence.",
    evidence: "SOURCE + GPU-LAB",
    nodes: [
      ["Architecture/config", "Inspect registry match, dimensions, and adapted weight names/shapes.", "registry.py + model_config.py", "retrieve_architecture(), finalize()", "yellow"],
      ["Graph", "Inspect Graph.output verification, symbolic types, and source tracebacks.", "graph/graph.py", "Graph.output(), _verify_op()", "teal"],
      ["Compile", "Inspect extension paths, device metadata, and native compile error.", "engine/api.py", "_compile_module()", "violet"],
      ["Initialize", "Inspect weight names, DLPack devices, contiguity, and memory plan.", "engine/api.py", "init_all()", "green"],
      ["Execute", "Inspect Buffer dtype/shape/device, KV inputs, and hazard ordering.", "engine/api.py + driver", "_Model_execute(), inplace_copy_from()", "blue"],
      ["Performance", "Measure H2D/D2H, kernel timeline, occupancy, and synchronization.", "GPU profiler", "GPU-LAB measurement", "red"]
    ]
  },
  {
    phase: "Map", title: "Core invariants",
    lead: "These statements prevent the most common mental-model errors.",
    layout: "files",
    nodes: [
      ["01", "DeviceRef is symbolic; Device is concrete.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "teal"],
      ["02", "Config finalization is not graph finalization.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "yellow"],
      ["03", "Graph.output() verifies; compile lowers.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "violet"],
      ["04", "CompiledModel is not executable Model.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "green"],
      ["05", "Inputs execute as device-aware Buffer handles.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "blue"],
      ["06", "Mojo kernels consume compiled arguments, not Python objects.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "coral"],
      ["07", "Queue/Stream work is asynchronous until synchronized.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "red"],
      ["08", "Native passes and vendor calls remain explicit boundaries.", "LLAMA_7B_SETUP_GRAPH_COMPILE_TO_GPU_TREE.md", "Core invariants", "yellow"]
    ]
  }
];

function phaseColor(phase) {
  const p = phase.toLowerCase();
  if (p.includes("serve")) return colors.coral;
  if (p.includes("model")) return colors.yellow;
  if (p.includes("graph")) return colors.teal;
  if (p.includes("compile")) return colors.violet;
  if (p.includes("init")) return colors.green;
  if (p.includes("driver")) return colors.blue;
  if (p.includes("gpu")) return colors.red;
  return colors.green;
}

function addText(slide, text, x, y, w, h, opt) {
  opt = opt || {};
  slide.addText(text, Object.assign({
    x: x, y: y, w: w, h: h, fontFace: opt.mono ? "Aptos Mono" : "Aptos",
    fontSize: opt.size || 15, color: opt.color || colors.paper,
    bold: !!opt.bold, margin: opt.margin === undefined ? 0 : opt.margin,
    valign: opt.valign || "mid", align: opt.align || "left", fit: "shrink"
  }, opt));
}

function rect(slide, x, y, w, h, fill, line) {
  slide.addShape(SHAPE.roundRect, {
    x: x, y: y, w: w, h: h,
    fill: { color: fill }, line: { color: line || fill, width: 1 },
    rectRadius: 0.04
  });
}

function arrow(slide, x, y, w, color) {
  slide.addShape(SHAPE.line, {
    x: x, y: y, w: w, h: 0,
    line: { color: color, width: 2, beginArrowType: "none", endArrowType: "triangle" }
  });
}

function header(slide, scene, build) {
  slide.background = { color: colors.ink };
  slide.addShape(SHAPE.rect, {
    x: 0, y: 0, w: 0.17, h: 7.5,
    fill: { color: phaseColor(scene.phase) }, line: { color: phaseColor(scene.phase) }
  });
  addText(slide, scene.phase.toUpperCase(), 0.48, 0.24, 2.4, 0.26, { size: 10, color: phaseColor(scene.phase), bold: true });
  addText(slide, scene.title, 0.48, 0.57, 11.6, 0.58, { size: 27, bold: true });
  addText(slide, scene.lead, 0.5, 1.18, 11.7, 0.42, { size: 12, color: colors.muted });
  if (build) {
    rect(slide, 11.55, 0.25, 1.2, 0.34, "292927", colors.line);
    addText(slide, build, 11.55, 0.25, 1.2, 0.34, { size: 8.5, align: "center", bold: true });
  }
}

function footer(slide, evidence) {
  addText(slide, evidence || "SOURCE", 0.5, 7.12, 2.0, 0.18, { size: 8, color: (evidence || "").includes("BOUNDARY") ? colors.yellow : colors.green, bold: true });
  addText(slide, "MAX source " + REV.slice(0, 10), 10.5, 7.12, 2.25, 0.18, { size: 8, color: colors.muted, align: "right" });
}

function addNode(slide, node, x, y, w, h) {
  const tone = colors[node[4]] || colors.teal;
  rect(slide, x, y, w, h, "232321", tone);
  slide.addShape(SHAPE.rect, {
    x: x, y: y, w: 0.06, h: h, fill: { color: tone }, line: { color: tone }
  });
  addText(slide, node[0], x + 0.2, y + 0.12, w - 0.36, 0.34, { size: 14, color: tone, bold: true });
  addText(slide, node[1], x + 0.2, y + 0.52, w - 0.36, h - 0.64, { size: 10.2, color: colors.muted, valign: "top" });
}

function noteFor(scene, visible) {
  const current = scene.nodes[Math.min(visible - 1, scene.nodes.length - 1)];
  return scene.title + "\n\n" + scene.lead + "\n\nCurrent build: " + current[0] + "\n" + current[1] + "\n\nFile: " + current[2] + "\nFunctions: " + current[3];
}

function addNotes(slide, text) {
  if (typeof slide.addNotes === "function") slide.addNotes(text);
}

function titleSlide(pptx) {
  const slide = pptx.addSlide();
  slide.background = { color: colors.ink };
  slide.addShape(SHAPE.rect, { x: 0.62, y: 0.55, w: 0.2, h: 5.9, fill: { color: colors.green }, line: { color: colors.green } });
  addText(slide, "LLAMA 7B", 1.15, 0.72, 4.2, 0.35, { size: 14, color: colors.green, bold: true });
  addText(slide, "MAX Serve Setup\nto GPU Execution", 1.1, 1.22, 7.4, 1.7, { size: 39, bold: true, valign: "top" });
  addText(slide, "Model objects, graph tracing, MLIR finalization, compilation, initialization, driver queues, Mojo kernels, and host/device movement", 1.15, 3.28, 7.2, 1.0, { size: 17, color: colors.muted, valign: "top" });
  const labels = [
    ["Python", "serve / pipeline / graph", "coral"],
    ["Native MAX", "compile / init / execute", "violet"],
    ["Mojo + HAL", "kernels / buffers / queues", "teal"],
    ["GPU", "DMA / grids / SMs", "red"]
  ];
  labels.forEach(function(item, i) {
    const y = 1.05 + i * 1.28;
    addNode(slide, [item[0], item[1], "", "", item[2]], 9.15, y, 3.2, 0.82);
    if (i < labels.length - 1) {
      slide.addShape(SHAPE.line, { x: 10.75, y: y + 0.82, w: 0, h: 0.45, line: { color: colors.muted, width: 1.5, endArrowType: "triangle" } });
    }
  });
  addText(slide, "Progressive-build PowerPoint + object-animated HTML", 1.15, 6.15, 5.5, 0.3, { size: 11, color: colors.yellow, bold: true });
  footer(slide, "SOURCE + BOUNDARY");
  addNotes(slide, "Introduce the four layers before following individual functions. The browser companion provides object-level animation; this deck uses progressive build slides and fade transitions.");
}

function sceneSlides(pptx, scene) {
  const total = scene.nodes.length;
  const builds = total <= 2 ? [total] : [Math.ceil(total / 2), total];
  builds.forEach(function(visible, buildIndex) {
    const slide = pptx.addSlide();
    header(slide, scene, builds.length > 1 ? "BUILD " + (buildIndex + 1) + "/" + builds.length : "");
    const count = scene.nodes.length;
    const cols = scene.layout === "files" || scene.layout === "math" ? 2 : Math.min(count, 4);
    const rows = Math.ceil(count / cols);
    const areaX = 0.62, areaY = 1.88, areaW = 12.1, areaH = 4.35;
    const gapX = 0.22, gapY = 0.22;
    const w = (areaW - gapX * (cols - 1)) / cols;
    const h = (areaH - gapY * (rows - 1)) / rows;
    scene.nodes.forEach(function(node, i) {
      const col = i % cols, row = Math.floor(i / cols);
      const x = areaX + col * (w + gapX), y = areaY + row * (h + gapY);
      if (i < visible) addNode(slide, node, x, y, w, h);
      else {
        rect(slide, x, y, w, h, "1E1E1C", colors.line);
        addText(slide, String(i + 1).padStart(2, "0"), x, y, w, h, { size: 22, color: colors.line, bold: true, align: "center" });
      }
    });
    if (scene.layout !== "files" && scene.layout !== "math" && count <= 4 && rows === 1) {
      for (let i = 0; i < visible - 1; i += 1) {
        const x = areaX + (i + 1) * w + i * gapX;
        arrow(slide, x, areaY + h / 2, gapX, colors[scene.nodes[i][4]]);
      }
    }
    const current = scene.nodes[Math.min(visible - 1, total - 1)];
    rect(slide, 0.62, 6.42, 12.1, 0.46, "20201E", colors[current[4]]);
    addText(slide, current[2] + "   |   " + current[3], 0.82, 6.48, 11.7, 0.25, { size: 8.6, mono: true, color: colors.paper });
    footer(slide, scene.evidence || "SOURCE");
    addNotes(slide, noteFor(scene, visible));
  });
}

function injectTransitions(file) {
  const zip = new AdmZip(file);
  zip.getEntries().filter(function(entry) {
    return /^ppt\/slides\/slide\d+\.xml$/.test(entry.entryName);
  }).forEach(function(entry) {
    let xml = entry.getData().toString("utf8");
    if (xml.includes("<p:transition")) return;
    const transition = '<p:transition spd="med" advClick="1"><p:fade/></p:transition>';
    if (xml.includes("<p:clrMapOvr")) xml = xml.replace("<p:clrMapOvr", transition + "<p:clrMapOvr");
    else xml = xml.replace("</p:sld>", transition + "</p:sld>");
    zip.updateFile(entry.entryName, Buffer.from(xml, "utf8"));
  });
  zip.writeZip(file);
}

function esc(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function htmlNode(node, step) {
  return '<button class="node tone-' + node[4] + '" data-step="' + step +
    '" data-file="' + esc(node[2]) + '" data-funcs="' + esc(node[3]) +
    '" data-detail="' + esc(node[1]) + '"><strong>' + esc(node[0]) +
    '</strong><span>' + esc(node[1]) + '</span></button>';
}

function htmlScene(scene, index) {
  const classes = "nodes " + (scene.layout || "flow");
  const nodes = scene.nodes.map(function(node, i) { return htmlNode(node, i + 1); }).join("");
  return '<article class="scene" data-index="' + index + '" data-steps="' + scene.nodes.length + '">' +
    '<header><b>' + esc(scene.phase) + '</b><h1>' + esc(scene.title) + '</h1><p>' + esc(scene.lead) + '</p></header>' +
    '<main class="' + classes + '">' + nodes + '</main>' +
    '<footer><span>' + esc(scene.evidence || "SOURCE") + '</span><code>' + REV.slice(0, 10) + '</code></footer></article>';
}

function htmlDocument() {
  const css = [
    ":root{--ink:#171717;--paper:#f6f4ef;--muted:#b7b5ae;--line:#3c3b37;--green:#36d17d;--teal:#27c2b8;--coral:#ff6b4a;--yellow:#ffd166;--blue:#5ab2ff;--violet:#a78bfa;--red:#ff4d67}",
    "*{box-sizing:border-box;letter-spacing:0}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--ink);color:var(--paper);font-family:Arial,Helvetica,sans-serif}button{font:inherit}",
    ".deck{height:100%;display:grid;grid-template-columns:34px 1fr;grid-template-rows:5px 1fr 58px}.progress{grid-column:1/3;background:#242422}.progress i{display:block;height:100%;width:0;background:var(--green);transition:width .4s}",
    ".nav{grid-row:2;border-right:1px solid #292927;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:6px}.dot{width:8px;height:8px;padding:0;border:1px solid #696762;background:transparent;cursor:pointer}.dot.active{background:var(--green);border-color:var(--green)}",
    ".viewport{grid-row:2;position:relative;overflow:hidden}.scene{position:absolute;inset:0;padding:30px 44px 20px;display:grid;grid-template-rows:auto 1fr auto;opacity:0;transform:translateY(14px);pointer-events:none;transition:opacity .35s,transform .35s}.scene.active{opacity:1;transform:none;pointer-events:auto}",
    ".scene header b{display:block;color:var(--accent,var(--green));font-size:11px;text-transform:uppercase;margin-bottom:8px}.scene h1{font-size:34px;line-height:1.08;margin:0 0 8px}.scene header p{margin:0;color:var(--muted);font-size:15px;line-height:1.45;max-width:980px}.scene footer{display:flex;justify-content:space-between;font-size:10px;color:var(--muted)}.scene footer span{color:var(--green);font-weight:800}",
    ".nodes{min-height:0;display:grid;gap:16px;align-content:center;justify-content:center;padding:22px 0}.nodes.flow{grid-template-columns:repeat(4,minmax(145px,240px))}.nodes.math,.nodes.files{grid-template-columns:repeat(2,minmax(260px,500px))}",
    ".node{--tone:var(--teal);appearance:none;text-align:left;background:#242422;border:1px solid var(--tone);color:var(--paper);padding:18px;min-height:130px;cursor:pointer;position:relative;opacity:0;transform:translateY(13px);filter:saturate(.55);transition:opacity .5s,transform .5s,filter .5s}.node:before{content:'';position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--tone)}.node.revealed{opacity:1;transform:none;filter:none}.node.revealed:after{content:'';position:absolute;right:-17px;top:50%;width:16px;height:2px;background:var(--tone);animation:flow 1.4s linear infinite}.node:last-child:after{display:none}@keyframes flow{50%{background:var(--paper)}}",
    ".node strong{display:block;color:var(--tone);font-size:16px;margin-bottom:9px}.node span{display:block;color:var(--muted);font-size:12px;line-height:1.45}.node:hover{background:#2c2c29}",
    ".tone-green{--tone:var(--green)}.tone-teal{--tone:var(--teal)}.tone-coral{--tone:var(--coral)}.tone-yellow{--tone:var(--yellow)}.tone-blue{--tone:var(--blue)}.tone-violet{--tone:var(--violet)}.tone-red{--tone:var(--red)}",
    ".controls{grid-column:1/3;grid-row:3;border-top:1px solid #292927;background:#1c1c1a;display:flex;align-items:center;justify-content:center;gap:8px;position:relative}.controls button{width:36px;height:36px;border:1px solid #4c4b47;background:#242422;color:var(--paper);cursor:pointer;font-size:17px}.controls button:hover,.controls button.active{border-color:var(--green);color:var(--green)}.count{position:absolute;right:18px;color:var(--muted);font:12px monospace}.build{position:absolute;left:18px;color:var(--muted);font:12px monospace}",
    ".inspector{position:absolute;right:0;top:0;bottom:0;width:365px;background:#20201e;border-left:1px solid var(--teal);padding:24px;transform:translateX(100%);transition:transform .32s;z-index:5}.inspector.open{transform:none}.inspector button{position:absolute;right:14px;top:14px;width:30px;height:30px;border:1px solid var(--line);background:#292927;color:var(--paper)}.inspector h2{font-size:18px;color:var(--teal);margin:38px 0 18px}.inspector h3{font-size:10px;color:var(--muted);text-transform:uppercase;margin:20px 0 7px}.inspector code,.inspector p{font:12px/1.55 monospace;color:var(--paper);white-space:pre-wrap}",
    "@media(max-width:920px){.scene{padding:24px 24px 16px}.scene h1{font-size:27px}.nodes.flow{grid-template-columns:repeat(2,minmax(180px,1fr))}.node.revealed:after{display:none}}",
    "@media(max-width:620px){.deck{grid-template-columns:20px 1fr}.scene{padding:18px 14px 12px}.scene h1{font-size:22px}.scene header p{font-size:12px}.nodes.flow,.nodes.math,.nodes.files{grid-template-columns:1fr;overflow:auto;align-content:start}.node{min-height:105px}.nav{gap:3px}.dot{width:6px;height:6px}.inspector{width:88%}.count,.build{font-size:10px}}"
  ].join("");
  const sceneMarkup = scenes.map(htmlScene).join("");
  const dots = scenes.map(function(scene, i) {
    return '<button class="dot" data-target="' + i + '" title="' + esc((i + 1) + ". " + scene.title) + '"></button>';
  }).join("");
  const script = [
    "(()=>{",
    "const scenes=[...document.querySelectorAll('.scene')],dots=[...document.querySelectorAll('.dot')],bar=document.querySelector('.progress i'),count=document.querySelector('.count'),build=document.querySelector('.build'),inspector=document.querySelector('.inspector');",
    "let index=0,step=0,timer=null;",
    "const max=()=>Number(scenes[index].dataset.steps);",
    "function accent(){const p=scenes[index].querySelector('header b').textContent.toLowerCase();if(p.includes('serve'))return'var(--coral)';if(p.includes('model'))return'var(--yellow)';if(p.includes('graph'))return'var(--teal)';if(p.includes('compile'))return'var(--violet)';if(p.includes('init'))return'var(--green)';if(p.includes('driver'))return'var(--blue)';if(p.includes('gpu'))return'var(--red)';return'var(--green)'}",
    "function render(){scenes.forEach((s,i)=>s.classList.toggle('active',i===index));dots.forEach((d,i)=>d.classList.toggle('active',i===index));scenes[index].style.setProperty('--accent',accent());scenes[index].querySelectorAll('[data-step]').forEach(el=>el.classList.toggle('revealed',Number(el.dataset.step)<=step));bar.style.width=((index+step/Math.max(1,max()))/scenes.length*100)+'%';count.textContent=String(index+1).padStart(2,'0')+' / '+String(scenes.length).padStart(2,'0');build.textContent='BUILD '+step+' / '+max();inspector.classList.remove('open')}",
    "function next(){if(step<max())step++;else if(index<scenes.length-1){index++;step=0}else{index=0;step=0}render()}",
    "function prev(){if(step>0)step--;else if(index>0){index--;step=Number(scenes[index].dataset.steps)}render()}",
    "function play(){const b=document.querySelector('.play');if(timer){clearInterval(timer);timer=null;b.classList.remove('active');b.innerHTML='&#9654;'}else{timer=setInterval(next,1450);b.classList.add('active');b.innerHTML='&#10074;&#10074;'}}",
    "document.querySelector('.next').onclick=next;document.querySelector('.prev').onclick=prev;document.querySelector('.play').onclick=play;document.querySelector('.reset').onclick=()=>{step=0;render()};document.querySelector('.full').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();",
    "dots.forEach(d=>d.onclick=()=>{index=Number(d.dataset.target);step=0;render()});",
    "document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();next()}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();prev()}if(e.key.toLowerCase()==='p')play();if(e.key==='Escape')inspector.classList.remove('open')});",
    "document.addEventListener('click',e=>{const n=e.target.closest('.node');if(!n||!n.classList.contains('revealed'))return;inspector.querySelector('.f').textContent=n.dataset.file||'See the Markdown tree.';inspector.querySelector('.fn').textContent=n.dataset.funcs||'See the Markdown tree.';inspector.querySelector('.d').textContent=n.dataset.detail;inspector.classList.add('open')});",
    "inspector.querySelector('button').onclick=()=>inspector.classList.remove('open');render();",
    "})();"
  ].join("");
  return "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Llama 7B MAX Setup to GPU</title><style>" +
    css + "</style></head><body><div class=\"deck\"><div class=\"progress\"><i></i></div><nav class=\"nav\">" + dots +
    "</nav><div class=\"viewport\">" + sceneMarkup +
    "<aside class=\"inspector\"><button title=\"Close\">x</button><h2>Object details</h2><h3>File</h3><code class=\"f\"></code><h3>Functions</h3><code class=\"fn\"></code><h3>Role</h3><p class=\"d\"></p></aside></div>" +
    "<div class=\"controls\"><span class=\"build\"></span><button class=\"prev\" title=\"Previous\">&#9664;</button><button class=\"play\" title=\"Auto play\">&#9654;</button><button class=\"next\" title=\"Next\">&#9654;&#9654;</button><button class=\"reset\" title=\"Restart scene\">&#8634;</button><button class=\"full\" title=\"Full screen\">&#9974;</button><span class=\"count\"></span></div></div><script>" +
    script + "</script></body></html>";
}

async function main() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Murali Modular Tech Talk";
  pptx.company = "Modular MAX internals study";
  pptx.subject = "Llama 7B setup, graph compilation, driver, Mojo, and GPU call flow";
  pptx.title = "Llama 7B: MAX Serve Setup to GPU Execution";
  pptx.lang = "en-US";
  pptx.theme = { headFontFace: "Aptos Display", bodyFontFace: "Aptos", lang: "en-US" };
  titleSlide(pptx);
  scenes.forEach(function(scene) { sceneSlides(pptx, scene); });
  await pptx.writeFile({ fileName: PPTX });
  injectTransitions(PPTX);
  fs.writeFileSync(HTML, htmlDocument(), "utf8");
  process.stdout.write("Generated " + PPTX + "\nGenerated " + HTML + "\n");
}

main().catch(function(error) {
  console.error(error);
  process.exitCode = 1;
});
