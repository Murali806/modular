/* Source-derived teaching states. No MAX code or GPU work runs in this model. */
(function (root) {
  "use strict";
  const DEVICE_CASES = {
    gpu: {label: "One GPU", devices: ["gpu0"]},
    cpu: {label: "Default CPU", devices: []},
    duplicates: {label: "Repeated devices", devices: ["gpu0", "gpu0", "cpu", "cpu"]},
    host_first: {label: "CPU already listed", devices: ["cpu", "gpu0"]},
    multi: {label: "Two GPUs", devices: ["gpu0", "gpu1"]}
  };
  const COLORS = {
    pipeline: ["#875020", "#fff2e5"], session: ["#245b3b", "#e7f1e5"],
    config: ["#746019", "#fcf5dc"], native: ["#923963", "#fbeaf3"],
    runtime: ["#4144a0", "#eeeeff"], device: ["#176679", "#e5f3f6"],
    graph: ["#5c428e", "#f1ebfc"], artifact: ["#69551c", "#fff4d9"],
    weights: ["#775282", "#f7ecfa"], model: ["#266948", "#e6f4e8"],
    driver: ["#426178", "#edf2f6"], gpu: ["#a03638", "#fff0ef"]
  };
  const clone = x => JSON.parse(JSON.stringify(x));
  const prettyDevice = x => x === "cpu" ? "CPU()" : "Accelerator(" + x.slice(3) + ")";
  function normalizeDevices(devices) {
    const result = [...new Set(devices)];
    if (!result.includes("cpu")) result.push("cpu");
    return result;
  }
  function timeline(initial) {
    const frames = [];
    let state = clone(initial);
    return {
      frames,
      add(title, detail, active, refs, patch = {}, packet = null, kind = "normal") {
        state = {...state, ...clone(patch)};
        frames.push({...clone(state), title, detail, active, refs, packet, kind});
      }
    };
  }
  function constructorFrames(options = {}) {
    const choice = options.devices || "gpu";
    const input = DEVICE_CASES[choice].devices;
    const store = options.store || "none";
    const threads = options.threads === undefined ? null : options.threads;
    const extension = Boolean(options.extension);
    const t = timeline({
      input: [...input], normalized: [], seen: [], current: null, processed: 0,
      python: false, native: false, compileConfig: false, runtimeConfig: false,
      context: false, returned: false, store, threads, nativeThreads: threads || 0,
      extension, models: 0, status: "Arguments ready", error: false,
      devicesStatus: "awaiting constructor", storeStatus: "not checked",
      nativeStatus: "not constructed", compileStatus: "not created",
      runtimeStatus: "not created", contextStatus: "not created"
    });
    t.add("The worker calls the constructor",
      "TextGenerationPipeline has loaded the Device handles selected by MemoryPlan. It passes those handles into InferenceSession.",
      ["pipeline", "arguments"], ["pipeline_session", "pipeline_devices"]);
    t.add("Validate the artifact policy",
      store === "conflict"
        ? "precompiled_mefs and export_mefs are both supplied. The Python constructor raises ValueError before it constructs a native session."
        : "The constructor permits import, export, or neither. The two explicit artifact modes are mutually exclusive.",
      ["python", "store"], ["session"],
      {python: true, error: store === "conflict", storeStatus: store === "conflict" ? "ValueError: conflicting options" : "valid policy",
        status: store === "conflict" ? "Constructor rejected" : "Validating Python arguments"},
      {edge: "call", label: "S1"}, store === "conflict" ? "error" : "normal");
    if (store === "conflict") return t.frames;
    const storeText = {none: "No explicit MEF store", import: "MefStore.for_import", export: "MefStore.for_export"}[store];
    t.add("Remember the session options",
      "Store the thread setting and the optional MEF store. Custom extensions, when supplied, are resolved into the native extension list later in the constructor.",
      ["python", "store"], ["session"],
      {storeStatus: storeText, status: "Recording options"},
      {edge: "options", label: "CFG"});
    const unique = [];
    input.forEach((device, index) => {
      const duplicate = unique.includes(device);
      if (!duplicate) unique.push(device);
      t.add((duplicate ? "Skip repeated " : "Keep ") + prettyDevice(device),
        duplicate ? "This Device is already in seen_devices. The final list keeps its first occurrence."
          : "Append this Device and record it in seen_devices. The original first-occurrence order is preserved.",
        ["normalize"], ["session"],
        {normalized: [...unique], seen: [...unique], current: index, processed: index + 1,
          devicesStatus: duplicate ? "duplicate skipped" : "unique handle appended",
          status: "Normalizing device " + (index + 1) + " / " + input.length},
        {edge: "normalize", label: device}, duplicate ? "branch" : "normal");
    });
    const hasCPU = unique.includes("cpu");
    t.add(hasCPU ? "The host CPU is already present" : "Include the host CPU",
      hasCPU ? "CPU() is already in seen_devices, so its existing position is preserved."
        : "Construct the host CPU handle and append it. Even a GPU-only argument list gives the session a host CPU.",
      ["normalize"], ["session"],
      {normalized: normalizeDevices(input), current: null,
        devicesStatus: hasCPU ? "CPU retained in place" : "host CPU appended",
        status: "Final device list ready"},
      {edge: "normalize", label: "CPU"});
    t.add(extension ? "Resolve the custom extension" : "Use an empty extension list",
      extension ? "The teaching input represents one valid custom-extension package. MAX processes supplied extension objects before entering the native constructor."
        : "No custom extensions were supplied; the native constructor receives an empty list.",
      ["store"], ["session"],
      {storeStatus: storeText + (extension ? " + extension" : ""), status: "Extension arguments ready"});
    t.add("Cross the Python → native boundary",
      "self._impl = _InferenceSession(final_devices, custom_extensions_final, num_threads or 0). This is a native object in the same model-worker process.",
      ["python", "native"], ["session_native"],
      {native: true, nativeStatus: "Core::InferenceSession", status: "Constructing native S1"},
      {edge: "native-call", label: "S1"});
    t.add("Build the compilation configuration",
      "populateCompileConfig records the session's compilation settings and custom-extension library paths. A model graph has not been compiled.",
      ["compile"], ["native_ctor", "compile_config"],
      {compileConfig: true, compileStatus: extension ? "settings + custom library" : "sessionCompileConfig ready",
        status: "Compilation settings ready"},
      {edge: "compile-config", label: "CFG"});
    t.add("Register devices and the thread setting",
      "runtimeConfig receives the thread setting, each DeviceSpec with its existing device context, and the API language 'python'. Device handles are retained by the native session.",
      ["devices", "native"], ["native_ctor"],
      {runtimeConfig: true, runtimeStatus: "numThreads = " + (threads || 0),
        devicesStatus: "DeviceSpec + context registered", status: "Runtime configuration ready"},
      {edge: "register", label: "DEV"});
    t.add("Create the engine runtime context",
      "Engine::Context::create builds the engine context using a MAX context and CPUDevice. The underlying MAX context may be reused; this does not promise a new private thread pool per session.",
      ["context"], ["context_create", "context_ctor", "cpu_options"],
      {context: true, contextStatus: "MAX context + CPUDevice", status: "Runtime context ready"},
      {edge: "context", label: "CTX"});
    t.add("Return the session to the pipeline",
      "The constructor completes its remaining signal/debug setup, and TextGenerationPipeline retains session. Pipeline configuration and model loading follow this call.",
      ["pipeline", "python", "context"], ["session", "pipeline_session", "configure_session"],
      {returned: true, status: "Session ready · decoder not loaded", nativeStatus: "native S1 retained",
        contextStatus: "available for compile / init"},
      {edge: "return", label: "S1"});
    return t.frames;
  }
  function compilationFrames(mode = "cold", options = {}) {
    const reuse = mode === "mef";
    const missing = mode === "missing";
    const t = timeline({
      mode, session: true, graph: false, artifact: false, registry: false,
      initialized: false, error: false, compileCalls: 0, initCalls: 0,
      phase: "ready", weightNames: [], validated: false, repaired: false,
      graphStatus: "recipe only", compilerStatus: "waiting", artifactStatus: "not created",
      registryStatus: "not collected", initStatus: "waiting", modelStatus: "not initialized",
      status: "S1 exists · loading the decoder"
    });
    t.add("Start with the session from lesson 01",
      "The pipeline configures S1, loads checkpoint weights through the model path, and reaches LlamaModelBase.load_graphs. This lesson focuses on one ungrouped decoder graph.",
      ["session", "graph"], ["configure_session", "load_graphs"]);
    t.add("Trace the GraphSpec",
      "compile_and_init_graphs calls GraphSpec.trace(). It records the graph and output structure before consulting the session's MEF store.",
      ["graph"], ["compile_graphs", "graph_trace"],
      {graph: true, phase: "trace", graphStatus: "traced Graph", status: "Graph contract ready"},
      {edge: "trace", label: "IR"});
    t.add("Ask the same session for an artifact",
      "session.compile_reusing_mefs(graph) chooses the compile or artifact-reuse path. The graph remains separate from the session that compiles it.",
      ["session", "compiler"], ["compile_reuse", "compile_graphs"],
      {compilerStatus: reuse ? "matching store entry" : "compile(graph)", status: reuse ? "Stored MEF selected" : "Compile requested",
        compileCalls: 1, phase: "request"}, {edge: "compile", label: reuse ? "MEF" : "IR"});
    t.add(reuse ? "Read the stored MEF" : "Run the native compile work",
      reuse ? "For this matching-store example, the session compiles the artifact path; the native frontend recognizes MEF and reads the compiled artifact."
        : "The engine schedules compilation on its CPU runtime worker. The cold-path illustration assumes no compiled-artifact cache hit.",
      ["compiler"], reuse ? ["compile_reuse", "compiled_create"] : ["compile", "context_compile", "frontend"],
      {phase: reuse ? "import" : "compile", compilerStatus: reuse ? "read compatible artifact" : "native compiler work",
        status: reuse ? "Importing compiled artifact" : "Compiling graph on host"},
      {edge: reuse ? "reuse" : "artifact", label: reuse ? "MEF" : "IR"});
    t.add("CompiledModel is available",
      "The compile result is an artifact. It still requires initialization with actual weight values before an engine Model can execute.",
      ["artifact"], ["compile"],
      {artifact: true, phase: "artifact", artifactStatus: reuse ? "reused CompiledModel" : "CompiledModel",
        status: "Artifact ready · executable model: 0"},
      {edge: "artifact", label: "MEF"});
    t.add("Collect the weight registry",
      "GraphSpec.weights_registry supplies name → value bindings. W_a and W_b in the picture stand for two illustrative required weight names.",
      ["registry"], ["compile_graphs", "weights_registry"],
      {registry: true, phase: "collect", weightNames: missing ? ["W_a"] : ["W_a", "W_b"], registryStatus: missing ? "W_a present · W_b missing" : "W_a + W_b present",
        status: "Weight bindings collected"}, {edge: "weights", label: "W"});
    t.add("Validate before initialization",
      missing ? "In this Graph example, expected-weight metadata names W_b. init_all rejects the missing binding before calling native _load_all."
        : "init_all checks known expected weight names and CPU/device categories, then checks that registered values are contiguous.",
      ["init", "registry"], ["init_all"],
      {error: missing, phase: "validate", validated: !missing, initStatus: missing ? "ValueError: missing W_b" : "weight checks passed",
        status: missing ? "Initialization stopped · model: 0" : "Registry accepted"},
      {edge: "weights", label: "W"}, missing ? "error" : "normal");
    if (missing) {
      if (!options.repaired) return t.frames;
      t.add("Supply the missing binding",
        "Add an illustrative contiguous W_b value with the expected placement to the registry. The compiled artifact and S1 remain available; a failed weight check did not consume them.",
        ["registry", "init"], ["init_all"],
        {phase: "collect", error: false, repaired: true, weightNames: ["W_a", "W_b"],
          registryStatus: "W_a + W_b present", initStatus: "ready to retry validation", status: "W_b supplied · same compiled artifact"});
      t.add("Retry initialization with both bindings",
        "Call init_all again with the completed registry. Both required names are present, so validation permits native initialization. No new compile is requested in this example.",
        ["registry", "init"], ["init_all"],
        {phase: "validate", validated: true, initStatus: "weight checks passed", status: "Both sockets filled · initialization can proceed"});
    }
    t.add("Initialize through the native session",
      "Native loadAll builds the weight registry, waits for the artifact, initializes each executable model, and creates ModelHandles that retain this native session.",
      ["init", "model"], ["init_all", "native_load_all", "native_load", "context_init"],
      {initCalls: 1, phase: "initialize", initStatus: "native loadAll + initModel", status: "Binding executable state"},
      {edge: "initialize", label: "W"});
    t.add("Retain the initialized decoder",
      "The returned engine Model is kept by the compiled callable and the Llama model. Future execute calls use it with S1's runtime context.",
      ["model", "session"], ["compile_graphs", "load_graphs", "model_execute"],
      {initialized: true, phase: "model", modelStatus: "decoder ready", status: "S1 + initialized decoder"},
      {edge: "model", label: "M1"});
    return t.frames;
  }
  function executionFrames(mode = "prefill_decode", options = {}) {
    const examples = [[88, 31], [64, 12], [25, 73], [19, 46], [57, 10], [33, 82]];
    const passes = mode === "decode_only"
      ? [{name: "Decode", tokens: [88, 31], offsets: [0, 1, 2]}]
      : [{name: "Prefill", tokens: [17, 42, 6, 9, 21], offsets: [0, 3, 5]},
        {name: "Decode", tokens: [88, 31], offsets: [0, 1, 2]}];
    const extra = Math.max(0, Math.min(4, Number(options.extraPasses) || 0));
    for (let i = 0; i < extra; i++) {
      const previous = examples[passes.length - 1];
      passes.push({name: "Decode", tokens: previous, offsets: [0, 1, 2]});
    }
    const t = timeline({
      mode, pass: 0, completed: 0, tokens: [], offsets: [], bufferPlace: "not prepared",
      pipelineStatus: "ready for a batch", modelStatus: "initialized decoder M1",
      runtimeStatus: "S1 · ready", gpuStatus: "weights + KV resident",
      resultStatus: "no new result", status: "Session and decoder already initialized",
      calls: 0, compilationCount: 1, initializedCount: 1, inputCount: 0,
      executing: false, output: false, phase: "ready", history: [],
      kvCounts: mode === "decode_only" ? [3, 2] : [0, 0], resultTokens: []
    });
    t.add("Keep the runtime and decoder",
      "This example starts after startup. S1 and its initialized decoder M1 remain in the model worker; the scheduler supplies batches to the pipeline.",
      ["model", "runtime"], ["pipeline_session", "model_execute", "pipeline_execute"]);
    passes.forEach((pass, index) => {
      const number = index + 1;
      t.add("Prepare " + pass.name.toLowerCase() + " inputs",
        pass.name === "Prefill"
          ? "Two illustrative sequences contribute three and two prompt tokens. Row offsets [0, 3, 5] describe the packed token buffer."
          : "The two active sequences now each contribute one token. The input count changes; the existing model and session are reused.",
        ["pipeline", "buffers"], ["pipeline_execute", "prepare_batch"],
        {pass: number, phase: "prepare", tokens: pass.tokens, offsets: pass.offsets, bufferPlace: "prepared model inputs",
          inputCount: pass.tokens.length, output: false, pipelineStatus: pass.name + " · 2 sequences",
          modelStatus: "M1 · ready", runtimeStatus: "S1 · ready", gpuStatus: "weights + KV retained",
          resultStatus: "awaiting this pass", status: pass.name + " inputs ready"},
        {edge: "prepare", label: "B" + number});
      t.add("Call the initialized model",
        "LlamaModelBase.execute forwards prepared model_inputs.buffers to self.model.execute. The buffer metadata determines the required device placements.",
        ["model", "buffers"], ["model_execute"],
        {calls: number, phase: "call", modelStatus: "M1.execute · call " + number, status: "Batch " + number + " enters M1"},
        {edge: "execute", label: "B" + number});
      t.add("Use S1's native runtime context",
        "ModelHandle retrieves its retained session's runtime context. executePreparedDeviceTensors releases the Python GIL around executeFromDriver.",
        ["runtime"], ["native_execute", "model_handle"],
        {phase: "runtime", runtimeStatus: "executeFromDriver", status: "Native execution call"},
        {edge: "runtime", label: "B" + number});
      t.add("Submit device work",
        "The animation expands the device boundary into user-runtime, OS-driver and hardware roles. Its moving packets show causal order, not measured queue or kernel timing.",
        ["runtime", "driver", "gpu"], ["native_execute"],
        {executing: true, phase: "submit", gpuStatus: pass.name + " work in flight", runtimeStatus: "device execution",
          bufferPlace: "consumed by the model", status: pass.name + " execution · BOUNDARY"},
        {edge: "submit", label: "B" + number}, "boundary");
      t.add("Return output Buffers",
        "Native results are wrapped as Python Buffers, retaining their device placement. The picture's 'logits' tile is an output role, not an automatic copy of every output to host RAM.",
        ["gpu", "results"], ["native_execute", "extract_results", "model_execute"],
        {executing: false, output: true, phase: "outputs", resultTokens: examples[index],
          kvCounts: [3 + index + (mode === "decode_only" ? 1 : 0), 2 + index + (mode === "decode_only" ? 1 : 0)],
          gpuStatus: "pass completed", resultStatus: "output Buffers · logits",
          runtimeStatus: "results available", status: "Output buffers returned"},
        {edge: "result", label: "L"});
      t.add("Process results and continue",
        "Llama's batch processor handles outputs; the text-generation pipeline performs its configured sampling and context updates. The example's next-token IDs are illustrative.",
        ["pipeline", "results"], ["model_execute", "pipeline_execute"],
        {completed: number, phase: "sample", history: examples.slice(0, number), pipelineStatus: "process outputs / sampling", modelStatus: "M1 retained",
          runtimeStatus: "S1 retained", gpuStatus: "weights + KV retained",
          status: "Pass " + number + " complete · same S1 and M1"},
        {edge: "feedback", label: "B" + number});
    });
    return t.frames;
  }
  function lifetimeFrames(mode = "caller_first", options = {}) {
    const t = timeline({
      caller: true, decoder: true, sampler: true, references: 3, retained: true,
      nativeStatus: "3 illustrated strong owners", runtimeStatus: "engine context retained",
      releaseStatus: "all pictured owners alive", status: "Minimal ownership example",
      released: null, releaseHistory: []
    });
    t.add("Three owners retain the native session",
      "The Python InferenceSession wrapper owns _impl. Each initialized native ModelHandle also retains a reference to the native InferenceSession. This example draws two models using S1.",
      ["caller", "decoder", "sampler", "native"], ["session_native", "native_load_all", "model_handle"]);
    const order = options.order || (mode === "caller_first" ? ["caller", "decoder", "sampler"] : ["decoder", "sampler", "caller"]);
    if (order.length !== 3 || new Set(order).size !== 3 || order.some(key => !["caller", "decoder", "sampler"].includes(key)))
      throw new Error("Release order must contain each pictured owner exactly once");
    const state = {caller: true, decoder: true, sampler: true};
    order.forEach((owner, index) => {
      state[owner] = false;
      const left = 2 - index;
      const last = left === 0;
      const label = {caller: "the Python session wrapper", decoder: "the decoder ModelHandle", sampler: "the sampler ModelHandle"}[owner];
      t.add("Release " + label,
        last ? "All illustrated owners are gone. With no additional references or outstanding work, native-session destruction can release its engine context and device-handle references. Shared runtime/driver resources may remain."
          : "Removing this owner leaves " + left + " illustrated native-session reference" + (left > 1 ? "s" : "") + ". The retained models can still reach S1's runtime context.",
        [owner, "native", "context"], ["native_load_all", "model_handle", "native_class"],
        {...state, references: left, retained: !last, released: owner, releaseHistory: order.slice(0, index + 1),
          nativeStatus: last ? "eligible for destruction" : left + " illustrated strong owner" + (left > 1 ? "s" : ""),
          runtimeStatus: last ? "owned reference may release" : "engine context retained",
          releaseStatus: last ? "shared resources can outlive S1" : label + " released",
          status: last ? "No pictured owners · lifetime can end" : "S1 stays alive"},
        {edge: owner + "-ref", label: "−1"}, last ? "branch" : "normal");
    });
    return t.frames;
  }
  function framesFor(kind, options = {}) {
    if (kind === "constructor") return constructorFrames(options);
    if (kind === "compilation") return compilationFrames(options.mode, options);
    if (kind === "execution") return executionFrames(options.mode, options);
    if (kind === "lifetime") return lifetimeFrames(options.mode, options);
    throw new Error("Unknown lesson: " + kind);
  }
  const api = {DEVICE_CASES, COLORS, normalizeDevices, constructorFrames, compilationFrames,
    executionFrames, lifetimeFrames, framesFor, prettyDevice};
  root.InferenceSessionLesson = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
