"""Build the offline InferenceSession lesson from pinned Python and C++ source."""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

PAGE_NAME = "max_inference_session_devices_runtime_and_compilation.html"
GUIDE_NAME = "INFERENCE_SESSION_WALKTHROUGH.md"
MANIFEST_NAME = "inference_session_manifest.json"
BASE = "max/python/max/"
FILES = {
    "ENGINE": BASE + "engine/api.py",
    "TEXT": BASE + "pipelines/lib/pipeline_variants/text_generation.py",
    "MODEL": BASE + "pipelines/architectures/llama3/model.py",
    "SPECS": BASE + "pipelines/lib/graph_specs.py",
    "CONFIG": BASE + "pipelines/lib/config/config.py",
    "CORE": BASE + "_core/max_license/lib/Engine/InferenceSession.cpp",
    "HEADER": BASE + "_core/max_license/include/_core/Engine/InferenceSession.h",
    "CONTEXT": "max/lib/max_license/API/c/EngineContext.cpp",
    "COMPILED": "max/lib/max_license/API/c/CompiledModels.cpp",
    "FRONTEND": "GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp",
}


def py(alias, symbol, label, needle=None):
    return dict(alias=alias, symbol=symbol, label=label, language="python", needle=needle)


def cpp(alias, symbol, label, prefix=0):
    return dict(alias=alias, symbol=symbol, label=label, language="cpp", prefix=prefix)


REFS = {
    "session": py("ENGINE", "InferenceSession.__init__", "InferenceSession.__init__"),
    "session_native": py("ENGINE", "InferenceSession.__init__", "self._impl = _InferenceSession", "self._impl = _InferenceSession("),
    "pipeline_session": py("TEXT", "TextGenerationPipeline.__init__", "Pipeline constructs and retains session", "session = InferenceSession("),
    "pipeline_devices": py("TEXT", "TextGenerationPipeline.__init__", "Load MemoryPlan's devices", "self._devices = load_devices("),
    "configure_session": py("CONFIG", "PipelineConfig.configure_session", "Configure the session"),
    "compile_config": cpp("CORE", "CompilationRequest M::Core::InferenceSession::populateCompileConfig(", "populateCompileConfig"),
    "native_ctor": cpp("CORE", "M::Core::InferenceSession::InferenceSession(", "Native InferenceSession constructor"),
    "native_class": cpp("HEADER", "class InferenceSession final", "Native session fields"),
    "model_handle": cpp("HEADER", "class ModelHandle {", "ModelHandle ownership and lifetime"),
    "context_create": cpp("CONTEXT", "Engine::Context::create(const M::Engine::Config &config)", "Engine::Context::create", 1),
    "context_ctor": cpp("CONTEXT", "Engine::Context::Context(const M::Engine::Config &runtimeConfig,", "Engine context and CPUDevice"),
    "cpu_options": cpp("CONTEXT", "getOrCreateCPUDeviceOptions(const Engine::Config &config,", "CPU runtime thread options", 1),
    "load_graphs": py("MODEL", "LlamaModelBase.load_graphs", "Llama load_graphs"),
    "graph_trace": py("SPECS", "GraphSpec.trace", "GraphSpec.trace"),
    "weights_registry": py("SPECS", "GraphSpec.weights_registry", "GraphSpec.weights_registry"),
    "compile_graphs": py("SPECS", "compile_and_init_graphs", "Trace, compile and initialize graphs"),
    "compile_reuse": py("ENGINE", "InferenceSession.compile_reusing_mefs", "Compile or reuse a MEF"),
    "compile": py("ENGINE", "InferenceSession.compile", "Compile an artifact"),
    "init_all": py("ENGINE", "InferenceSession.init_all", "Validate weights and initialize"),
    "native_load_all": cpp("CORE", "M::Core::InferenceSession::loadAll(", "Native loadAll creates ModelHandles", 1),
    "native_load": cpp("CORE", "M::Core::InferenceSession::load(const M::Model *model,", "Native load initializes one model", 1),
    "context_compile": cpp("CONTEXT", "Engine::Context::compileModel(CompilationRequest config) const", "Compile work on the CPU runtime", 1),
    "context_init": cpp("CONTEXT", "Engine::Context::initModel(const M::Model *model,", "Initialize executable model on CPU runtime", 1),
    "compiled_create": cpp("COMPILED", "CompiledModels::create(M::ContextRef context,", "Read MEF or compile model source", 1),
    "frontend": cpp("FRONTEND", "MOFContext::compileToBinary(void *source,", "Native graph compiler and cache"),
    "model_execute": py("MODEL", "LlamaModelBase.execute", "Llama executes its engine Model"),
    "pipeline_execute": py("TEXT", "TextGenerationPipeline._execute", "Prepare, execute and sample"),
    "prepare_batch": py("TEXT", "TextGenerationPipeline.prepare_batch", "Prepare model inputs and KV references"),
    "native_execute": cpp("CORE", "std::vector<Driver::Python::Buffer> ModelHandle::executePreparedDeviceTensors(", "Native execution using the session context"),
    "extract_results": cpp("CORE", "extractPyBufferResults(", "Wrap native results as Python Buffers", 1),
}

COMPONENTS = {
    "pipeline": ("Pipeline / caller", "pipeline", "Python object in the model worker. Creates the session and supplies model work.", ["pipeline_session", "pipeline_execute"]),
    "arguments": ("Constructor arguments", "config", "Device handles and optional thread, artifact and extension settings. Input slots track handles awaiting inspection; the original argument list stays with the caller. Values are illustrative.", ["session"]),
    "python": ("Python InferenceSession", "session", "Normalizes devices, records options and owns the native _impl reference.", ["session", "session_native"]),
    "normalize": ("Device normalization", "device", "Preserves first-occurrence order and includes CPU(). Duplicate Device values do not create extra entries.", ["session"]),
    "store": ("Artifacts and extensions", "config", "Import/export policy and custom-extension inputs. Explicit import and export are mutually exclusive.", ["session", "compile_reuse"]),
    "native": ("Native InferenceSession", "native", "C++ object in the worker. Retains device handles, compilation configuration, runtime configuration and an owned Engine::Context.", ["native_ctor", "native_class"]),
    "devices": ("Retained Device handles", "device", "Each registered runtime device includes a DeviceSpec and its existing device context. Listing two devices alone does not partition a model; Llama's graph construction selects its parallel implementation separately.", ["native_ctor", "load_graphs"]),
    "compile": ("Compilation configuration", "config", "Session-level compilation request settings and custom kernel library paths. A configuration is distinct from a compiled model artifact.", ["compile_config", "native_class"]),
    "context": ("Engine runtime context", "runtime", "An owned Engine::Context carries runtime configuration and references a MAX context and CPUDevice. Shared underlying contexts can outlive a single session.", ["context_create", "context_ctor"]),
    "hardware": ("Selected execution hardware", "gpu", "The device handle names the target. The constructor animation does not depict model-weight allocation, a kernel launch, or measured GPU activity.", ["native_ctor"]),
    "session": ("Reusable session S1", "session", "The same session participates in compilation, initialization and execution through its native runtime context.", ["compile_graphs", "model_handle"]),
    "graph": ("Graph / GraphSpec", "graph", "The recipe traces the model into graph IR. Tracing precedes compile_reusing_mefs in this ungrouped GraphSpec path.", ["graph_trace", "compile_graphs"]),
    "compiler": ("Compiler / MEF import", "graph", "A cold graph path compiles source. A matching stored MEF takes the artifact path; a compile API call need not run graph compiler passes.", ["compile_reuse", "compiled_create", "frontend"]),
    "artifact": ("CompiledModel artifact", "artifact", "Compiled code and metadata require weight initialization before returning an executable Model.", ["compile", "init_all"]),
    "registry": ("Weight registry", "weights", "Named values bind the compiled weight contract. The diagram's W_a and W_b are illustrative names standing for a much larger model registry.", ["weights_registry", "init_all"]),
    "init": ("Initialization", "native", "Validate bindings, then initialize native executable model state on the runtime. Missing required weights stop this path.", ["init_all", "native_load_all", "context_init"]),
    "model": ("Initialized engine Model", "model", "A loaded model is reused for each execution. Its native ModelHandle retains a reference to the native session.", ["native_load_all", "model_execute", "model_handle"]),
    "runtime": ("MAX runtime / user driver", "native", "Host native code executes with the session context and crosses into the device runtime. Driver timing and scheduling remain a BOUNDARY.", ["native_execute"]),
    "driver": ("OS GPU driver", "driver", "Kernel and hardware boundary outside the worker's Python object hierarchy. Submission arrows illustrate causality, not actual kernel traces.", ["native_execute"]),
    "gpu": ("GPU and device buffers", "gpu", "Weights, KV data and model work on the selected hardware. The diagrams assume a supported GPU model configuration and do not measure resource use.", ["model_execute", "prepare_batch"]),
    "buffers": ("Prepared input Buffers", "device", "Packed token IDs, row offsets and KV inputs. Shapes and placement follow the model's input contract; not every buffer has to move to the same device.", ["prepare_batch", "model_execute"]),
    "results": ("Output Buffers / logits", "weights", "Native output buffers are wrapped for Python and processed by the pipeline. A/B tiles show sequence views of the logits; buffer placement is retained until an explicit operation changes it.", ["extract_results", "model_execute", "pipeline_execute"]),
    "caller": ("Python session owner", "session", "The Python wrapper retains the native _impl. Releasing the wrapper does not remove the references held by initialized native models.", ["session_native", "model_handle"]),
    "decoder": ("Decoder ModelHandle", "model", "Retains compiled/executable state, weight ownership and a reference to native S1.", ["native_load_all", "model_handle"]),
    "sampler": ("Sampler ModelHandle", "model", "Illustrates a second engine model using the same session. This is a minimal ownership example, not a count of all models in max serve.", ["pipeline_session", "model_handle"]),
    "release": ("Resource release", "driver", "Only the pictured owners are counted. Actual destruction also depends on other references and outstanding work; shared runtime and driver resources may remain.", ["native_class", "model_handle", "context_create"]),
}

LESSONS = [
    dict(id="constructor", number="01", title="Inside the constructor", subtitle="Python → native · one model-worker process",
         intro="Follow the exact call at ENGINE:L664. Device handles enter; an engine session comes back.",
         detail="Watch the device list normalize, then watch compilation settings and the runtime context appear. The host CPU is included even when the supplied list only contains a GPU.",
         try_it="Choose Repeated devices, then Step through each input. Try CPU already listed to see order preservation. Set MEF policy to Conflicting options to stop before native construction.",
         refs=["pipeline_session", "session", "native_ctor", "context_create"],
         ascii="""SOURCE · illustrative values; logical call order
MODEL WORKER PROCESS
  TextGenerationPipeline.__init__
    -> InferenceSession(devices=[gpu0, gpu0], num_threads=None)
       -> check explicit MEF import/export policy
       -> final_devices: [gpu0, cpu]  (first occurrences + host CPU)
       -> _InferenceSession(final_devices, extensions, 0)
          -> sessionCompileConfig
          -> runtimeConfig: thread setting + DeviceSpec/context pairs
          -> owned Engine::Context
             -> shared/reused MAX context + CPUDevice
    <- session; store pipeline.session
  configure_session(session) -> model loading follows
At constructor return: decoder artifacts = 0; initialized decoders = 0.
Underlying device contexts and CPU runtime resources can be shared."""),
    dict(id="compilation", number="02", title="From a graph to a runnable model", subtitle="One ungrouped decoder graph · same S1",
         intro="A session, a compiled artifact, and an initialized model become ready at different points.",
         detail="Tracing creates the graph. Compilation produces an artifact. Initialization binds real weights and returns the executable engine model. Each object has its own role.",
         try_it="Compare Cold compile with Stored MEF hit: follow the alternate path around compiler work. Choose Missing weight, then Add missing W_b. The empty binding fills and initialization can proceed using the same artifact.",
         refs=["load_graphs", "compile_graphs", "compile_reuse", "init_all"],
         ascii="""SOURCE · ordinary real-device, ungrouped GraphSpec route
Llama model -> GraphSpec.trace() -> Graph
                           |
                           v
                    existing session S1
                     /             \\
          compile(graph)         matching MEF store
          native compiler        compile(artifact_path) -> read MEF
                     \\             /
                       CompiledModel
                            |
GraphSpec.weights_registry() -> init_all(artifact, weights)
                            | required weight missing -> ValueError
                            v
                    native loadAll / initModel
                            |
                  executable ModelHandle -> retains native S1
Returned decoder -> Llama model / compiled callable
All arrows represent logical source order, not captured GPU timing."""),
    dict(id="execution", number="03", title="Change the batch. Reuse the model.", subtitle="Packed inputs → device work → output Buffers",
         intro="Prefill and decode change the inputs while the session and initialized decoder remain available.",
         detail="The Python pipeline prepares Buffers. Llama's engine Model calls the native runtime through its retained session. Output logits are processed and sampled per sequence. A/B tiles depict sequence views; KV tile counts illustrate token entries, not actual cache-page allocation.",
         try_it="Play Prefill + decode. Follow A and B's sampled tokens back into the next input batch. Run another decode to grow both histories and KV entries while S1, M1 and the setup count stay unchanged.",
         refs=["prepare_batch", "model_execute", "native_execute", "extract_results", "pipeline_execute"],
         ascii="""SOURCE + BOUNDARY · illustrative token IDs, placements and event timing
MODEL WORKER PROCESS                            OS / HARDWARE
  Pipeline.prepare_batch()
      prefill: [17 42 6 | 9 21], offsets [0,3,5]
      decode : [88 | 31],        offsets [0,1,2]
          |
  LlamaModelBase.execute -> existing engine Model M1
          |                    |
          |             ModelHandle -> native S1 -> Engine::Context
          |                    |
          |             executeFromDriver ------> driver -> GPU work
          |                    |                             |
          |             output Buffers <--------------------+
          v                    |
  batch output processing / configured sampling
  next model call reuses S1 and M1
Buffer metadata controls placement; an output Buffer can stay on device.
Setup counts are fixture counts, not performance measurements."""),
    dict(id="lifetime", number="04", title="Who keeps the session alive?", subtitle="References → retained context → eligible release",
         intro="Follow native-session ownership after the Python wrapper or a model is released.",
         detail="Native ModelHandles retain S1. Its context must outlive model state that uses the runtime to release buffers. This example shows a wrapper, a decoder and a sampler.",
         try_it="Use the three Release buttons in any order. Each reference leaves its slot; S1 and its context remain while one pictured owner is still alive. Compare with the two automatic release orders.",
         refs=["session_native", "native_load_all", "native_class", "model_handle", "context_create"],
         ascii="""SOURCE · minimal ownership example; only pictured owners are counted
Python InferenceSession._impl -----------+
decoder ModelHandle.inferenceSession ---+--> native session S1
sampler ModelHandle.inferenceSession ---+        |
                                                +-- compile/runtime config
                                                +-- retained Device handles
                                                +-- owned Engine::Context
                                                       |
                                                       +-- MAX context / CPUDevice
Drop wrapper -> model references still retain S1
Drop decoder -> sampler can still retain S1
Drop sampler -> with no other references/work, native destruction is eligible
Shared underlying contexts/driver resources may remain.
ModelHandle member order keeps the session alive through model cleanup."""),
]


def node(key, x, y, w, h, title, subtitle="", status=None, color=None):
    """Render a clickable component; the controller updates its status and state."""
    color = color or COMPONENTS[key][1]
    esc = html.escape
    s = (f'<text x="{x + w / 2}" y="{y + h - 17}" class="node-status" '
         f'data-status="{esc(status or key)}">waiting</text>') if status is not False else ""
    return (f'<g data-node="{key}" data-component="{key}" data-color="{color}" role="button" tabindex="0" '
            f'aria-label="Explain {esc(title, quote=True)}"><rect class="component-box" x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>'
            f'<text class="node-label" x="{x + w / 2}" y="{y + 25}">{esc(title)}</text>'
            f'<text class="node-sub" x="{x + w / 2}" y="{y + 46}">{esc(subtitle)}</text>{s}</g>')


def wire(key, path, label="", x=0, y=0, kind="call"):
    return (f'<path data-wire="{key}" class="wire {kind}" d="{path}"/>'
            + (f'<text class="wire-label" x="{x}" y="{y}">{html.escape(label)}</text>' if label else ""))


def svg_shell(key, height, body):
    label = next(item["title"] for item in LESSONS if item["id"] == key)
    return (f'<svg viewBox="0 0 1240 {height}" role="group" aria-labelledby="svg-{key}-title">'
            f'<title id="svg-{key}-title">{html.escape(label)}</title>'
            f'<desc>Full text equivalent and complete source definitions are in the disclosure below this animation.</desc>'
            f'<defs><marker id="arrow-{key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
            '<path d="M0 0 L7 3.5 L0 7" fill="#667d6c"/></marker></defs>'
            + body + '<g data-actors aria-label="Moving objects and retained references"></g></svg>')


def pockets(key, x, y, count, width=70, gap=12):
    """Empty places remain visible as objects enter, leave or merge."""
    return (f'<g data-slots="{key}" transform="translate({x} {y})">' + ''.join(
        f'<g class="live-slot" data-slot="{i}" data-occupied="false" transform="translate({i * (width + gap)} 0)">'
        f'<rect width="{width}" height="30" rx="5"/><text x="{width / 2}" y="20">—</text></g>'
        for i in range(count)) + '</g>')


def constructor_svg():
    b = '<rect class="process-boundary" x="20" y="24" width="1200" height="536" rx="15"/><text class="process-label" x="620" y="49">MODEL WORKER · ONE HOST PROCESS</text>'
    b += '<rect class="native-boundary" data-boundary="native" x="752" y="74" width="448" height="438" rx="12"/><text class="boundary-label" x="976" y="100">S1 · native InferenceSession</text>'
    b += wire("call", "M320 117 H352")
    b += wire("normalize", "M320 276 H352")
    b += wire("options", "M181 356 V390")
    b += wire("native-call", "M716 117 H752", "_impl", 733, 107)
    b += wire("compile-config", "M716 447 H770")
    b += wire("register", "M716 237 H770")
    b += wire("context", "M1076 269 V304", "create", 1104, 291)
    b += wire("return", "M754 510 H336 V175 H181 V158", "return S1", 547, 526, "return")
    b += node("pipeline", 42, 74, 278, 84, "TextGenerationPipeline", "calls constructor → retains S1", status="pipeline")
    b += node("python", 352, 74, 364, 84, "Python InferenceSession", "normalizes handles + owns _impl", status="python")
    b += node("arguments", 42, 192, 278, 164, "Device references to inspect", "original list stays with caller", status=False)
    b += pockets("input", 62, 262, 4, 52, 10)
    b += '<text class="node-sub" x="181" y="323" data-constructor-input>[gpu0]</text>'
    b += node("normalize", 352, 192, 364, 164, "final_devices", "keep first occurrence · ensure CPU", status="devicesStatus")
    b += pockets("normalized", 378, 262, 3, 92, 18)
    b += '<text class="decision-note" x="534" y="317" data-decision>Choose a device list, then Play.</text>'
    b += node("store", 42, 390, 278, 122, "Artifact / extension policy", "checked before native creation", status="storeStatus")
    b += '<g data-policy-lights><rect class="policy-lamp" x="99" y="448" width="74" height="22" rx="4"/><text class="lamp-label" x="136" y="463">import</text><rect class="policy-lamp" x="189" y="448" width="74" height="22" rx="4"/><text class="lamp-label" x="226" y="463">export</text></g>'
    b += '<rect class="settings-box" x="352" y="390" width="364" height="105" rx="8"/><text class="node-label" x="534" y="415">Arguments passed to native code</text><text class="node-sub" x="534" y="442" data-constructor-threads>num_threads = None → 0</text><text class="node-sub" x="534" y="477" data-constructor-store>extensions = []</text>'
    b += node("devices", 770, 126, 412, 143, "Retained handles + device contexts", "runtimeConfig.devices · existing device contexts", status="registered")
    b += pockets("registered", 792, 191, 3, 112, 16)
    b += node("compile", 770, 304, 184, 158, "Compile config", "settings + library paths", status="compileStatus")
    b += pockets("config", 818, 374, 1, 88)
    b += node("context", 970, 304, 212, 158, "Engine::Context", "owns runtime entry point", status="contextStatus")
    b += pockets("context", 1032, 374, 1, 88)
    b += '<text class="diagram-note" x="976" y="491">Underlying CPU / device resources can be shared.</text>'
    b += '<text class="diagram-note" x="620" y="547">Native session ready ≠ decoder loaded. This constructor creates zero model artifacts.</text>'
    return svg_shell("constructor", 583, b)


def compilation_svg():
    b = '<rect class="process-boundary" x="20" y="24" width="1200" height="507" rx="15"/><text class="process-label" x="620" y="50">MODEL LOADING IN THE WORKER · SOURCE</text>'
    b += node("session", 42, 70, 1158, 65, "S1 · same session throughout", "trace → compile or import → validate weights → initialize", status=False)
    b += wire("compile", "M277 242 H322", "IR", 300, 226)
    b += wire("artifact", "M563 242 H614")
    b += wire("reuse", "M277 417 H581 V333 H734 V314", "MEF path · bypass compile work", 431, 358, "return")
    b += wire("initialize", "M854 242 H897")
    b += wire("weights", "M854 428 H878 V339 H1060 V314")
    b += wire("model", "M1145 314 V362", "initialize", 1175, 345)
    b += node("graph", 42, 150, 235, 164, "GraphSpec → Graph", "schematic graph operations", status="graphStatus")
    b += pockets("graph", 59, 224, 3, 54, 19)
    b += node("compiler", 322, 150, 241, 164, "Native compile work", "runs on the host CPU runtime", status="compilerStatus")
    b += pockets("compiler", 343, 224, 3, 54, 19)
    b += '<g data-compile-work><rect class="work-track" x="343" y="267" width="200" height="7" rx="3"/><rect class="work-fill" data-work-fill x="343" y="267" width="0" height="7" rx="3"/></g>'
    b += node("artifact", 614, 150, 240, 164, "CompiledModel", "compiled code + metadata", status="artifactStatus")
    b += pockets("artifact", 689, 224, 1, 90)
    b += node("init", 897, 150, 303, 164, "init_all · required bindings", "each expected name needs a value", status="initStatus")
    b += pockets("bindings", 932, 224, 2, 104, 30)
    b += '<g data-init-gate><path class="init-gate" d="M913 284 H1184"/><text class="gate-label" x="1048" y="279" data-gate-label>waiting for artifact + weights</text></g>'
    b += '<g data-cache><rect class="cache-box" x="42" y="362" width="235" height="135" rx="9"/><text class="node-label" x="159" y="387">Compatible MEF store</text><text class="node-sub" x="159" y="408">example cache hit</text>' + pockets("cache", 114, 427, 1, 90) + '</g>'
    b += node("registry", 614, 362, 240, 135, "Weight registry", "name → value · illustrative names", status="registryStatus")
    b += pockets("weights", 636, 427, 2, 86, 24)
    b += node("model", 897, 362, 303, 135, "Engine Model M1", "executable decoder · retains native S1", status="modelStatus")
    b += pockets("model", 1004, 427, 1, 88)
    b += '<text class="diagram-note" x="620" y="519">The compiler produces an artifact. Binding valid weights makes an executable model possible.</text>'
    return svg_shell("compilation", 552, b)


def execution_svg():
    b = '<rect class="process-boundary" x="20" y="24" width="947" height="516" rx="15"/><text class="process-label" x="493" y="49">MODEL WORKER · SAME S1 AND M1 ACROSS PASSES</text>'
    b += wire("prepare", "M175 246 V292", "prepare", 210, 272)
    b += wire("execute", "M306 162 H354", "call", 330, 150)
    b += wire("runtime", "M616 162 H664")
    b += wire("submit", "M949 126 H999 M1104 162 V206", "submit", 974, 113)
    b += wire("result", "M999 394 H949", "refs", 974, 381, "return")
    b += wire("feedback", "M806 461 V494 H637 V264 H175 V246", "sample → next model inputs", 581, 514, "return")
    b += node("pipeline", 44, 80, 262, 166, "Text-generation pipeline", "sampled tokens · A and B", status="pipelineStatus")
    b += '<text class="stream-label sequence-a" x="68" y="165">A</text><text class="stream-label sequence-b" x="68" y="203">B</text>'
    b += pockets("history-a", 82, 145, 6, 28, 8) + pockets("history-b", 82, 183, 6, 28, 8)
    b += node("model", 354, 80, 262, 166, "Engine Model M1", "same compiled decoder every pass", status="modelStatus")
    b += '<rect class="identity-badge" x="373" y="146" width="69" height="32" rx="8"/><text class="identity-label" x="407" y="167">M1</text>'
    b += pockets("invocation", 484, 146, 1, 92)
    b += node("runtime", 664, 80, 285, 166, "MAX runtime / user driver", "S1 → retained Engine::Context", status="runtimeStatus")
    b += '<rect class="identity-badge" x="683" y="146" width="69" height="32" rx="8"/><text class="identity-label" x="717" y="167">S1</text>'
    b += pockets("runtime", 800, 146, 1, 104)
    b += node("driver", 999, 90, 210, 72, "OS GPU driver", "kernel / device boundary", status=False)
    b += node("gpu", 999, 206, 210, 322, "GPU device", "schematic buffers + work", status="gpuStatus")
    b += '<rect class="device-memory weight-memory" x="1015" y="269" width="178" height="34" rx="6"/><text class="memory-label" x="1104" y="291">weights · retained</text>'
    b += '<text class="memory-label" x="1104" y="329" data-kv-label>KV entries · A 0 / B 0</text><g data-kv-tiles></g>'
    b += '<g data-gpu-work><rect class="device-work" x="1015" y="388" width="178" height="92" rx="6"/><text class="work-label" x="1104" y="469" data-work-label>between passes</text><rect class="work-track" x="1027" y="443" width="154" height="6" rx="3"/><rect class="work-fill" data-work-fill x="1027" y="443" width="0" height="6" rx="3"/></g>'
    b += node("buffers", 44, 292, 572, 169, "Packed input Buffers", "A then B · shapes change between prefill and decode", status="bufferPlace")
    b += pockets("tokens", 78, 357, 5, 82, 24)
    b += '<text class="input-value" x="330" y="421" data-offsets>row_offsets = []</text>'
    b += node("results", 664, 326, 285, 135, "Result Buffers", "A / B views of logits → sampling", status="resultStatus")
    b += pockets("results", 695, 388, 2, 94, 24)
    b += '<g data-result-tile hidden><text class="diagram-note" x="805" y="478">logits available · placement retained</text></g>'
    b += '<text class="diagram-note" x="1104" y="554">BOUNDARY · no GPU run</text><text class="diagram-note" x="493" y="564">Moving batch labels track calls; GPU tiles track work. Buffer placement still follows the model contract.</text>'
    return svg_shell("execution", 584, b)


def lifetime_svg():
    b = '<rect class="process-boundary" x="20" y="24" width="1200" height="492" rx="15"/><text class="process-label" x="620" y="50">NATIVE OWNERSHIP · RELEASE ANY OWNER · SOURCE</text>'
    b += wire("caller-ref", "M340 134 H433 V208 H532", "_impl", 407, 119, "reference")
    b += wire("decoder-ref", "M340 278 H532", "native reference", 434, 264, "reference")
    b += wire("sampler-ref", "M340 422 H433 V346 H532", "native reference", 426, 457, "reference")
    b += wire("owned", "M816 218 H915", "owns", 864, 202, "reference")
    b += wire("shared", "M1062 306 V351", "may be shared", 1116, 333, "reference")
    b += node("caller", 44, 80, 296, 116, "Python session wrapper", "InferenceSession._impl", status="caller")
    b += node("decoder", 44, 224, 296, 116, "Decoder ModelHandle", "model state + native session reference", status="decoder")
    b += node("sampler", 44, 368, 296, 116, "Sampler ModelHandle", "second illustrative engine model", status="sampler")
    b += '<g data-owner-badges></g>'
    b += node("native", 532, 138, 284, 256, "Native session S1", "pictured strong references", status="nativeStatus")
    b += '<text class="reference-count" x="674" y="241" data-reference-count>3</text>'
    b += pockets("references", 551, 273, 3, 64, 27)
    b += '<text class="diagram-note" x="674" y="329" data-retention>Any remaining owner keeps S1 alive.</text>'
    b += node("context", 915, 122, 285, 184, "Engine::Context", "retained for model work and cleanup", status="runtimeStatus")
    b += pockets("context", 1013, 203, 1, 88)
    b += node("release", 915, 351, 285, 133, "Underlying shared resources", "MAX context / CPUDevice / driver", status="releaseStatus")
    b += '<rect class="released-shelf" x="532" y="432" width="284" height="52" rx="8"/><text class="diagram-note" x="674" y="453">Released references</text>'
    b += '<text class="diagram-note" x="620" y="538">Only these three owners are counted. Destruction also requires no additional references or outstanding work.</text>'
    return svg_shell("lifetime", 562, b)


SVG_BUILDERS = dict(constructor=constructor_svg, compilation=compilation_svg,
                    execution=execution_svg, lifetime=lifetime_svg)


def controls(key):
    if key == "constructor":
        choices = [("gpu", "One GPU"), ("cpu", "Default CPU"), ("duplicates", "Repeated devices"),
                   ("host_first", "CPU already listed"), ("multi", "Two GPUs")]
        buttons = "".join(f'<button data-device-case="{k}" aria-pressed="{str(i == 0).lower()}">{label}</button>' for i, (k, label) in enumerate(choices))
        return (f'<div class="scenario-options" role="group" aria-label="Constructor device examples"><span>Device input</span>{buttons}</div>'
                '<div class="option-fields"><label>CPU runtime threads <select data-option="threads"><option value="default">Default · None → 0</option><option value="1">1</option><option value="4">4</option><option value="8">8</option></select></label>'
                '<label>MEF policy <select data-option="store"><option value="none">Neither</option><option value="import">Import artifacts</option><option value="export">Export artifacts</option><option value="conflict">Conflicting options</option></select></label>'
                '<label class="check-label"><input type="checkbox" data-option="extension"> Custom extension example</label></div>')
    choices = {
        "compilation": [("cold", "Cold compile"), ("mef", "Stored MEF hit"), ("missing", "Missing weight")],
        "execution": [("prefill_decode", "Prefill + decode"), ("decode_only", "Decode only")],
        "lifetime": [("caller_first", "Caller first"), ("models_first", "Models first")],
    }[key]
    experiment = {
        "compilation": '<button data-repair disabled>Add missing W_b</button><span>Fill the missing binding to unblock initialization.</span>',
        "execution": '<button data-another disabled>Run another decode</button><span>Complete a pass, then reuse the same S1 and M1.</span>',
        "lifetime": '<span>Release an owner yourself:</span><button data-drop-owner="caller">Release wrapper</button><button data-drop-owner="decoder">Release decoder</button><button data-drop-owner="sampler">Release sampler</button>',
    }[key]
    return ('<div class="scenario-options" role="group" aria-label="Choose a ' + key + ' scenario">' + "".join(
        f'<button data-mode="{k}" aria-pressed="{str(i == 0).lower()}">{label}</button>' for i, (k, label) in enumerate(choices)) +
        '</div><div class="experiment-actions">' + experiment + '</div>')


PHASES = {
    "constructor": [("options", "Check options"), ("normalize", "Normalize"), ("native", "Create S1"), ("register", "Retain handles"), ("context", "Create context"), ("ready", "Ready")],
    "compilation": [("trace", "Trace"), ("compile", "Compile / import"), ("bind", "Bind weights"), ("initialize", "Initialize"), ("ready", "Model ready")],
    "execution": [("prepare", "Prepare"), ("execute", "Call M1 + S1"), ("submit", "Device work"), ("outputs", "Output Buffers"), ("sample", "Sample / repeat")],
    "lifetime": [("3", "3 owners"), ("2", "2 owners"), ("1", "1 owner"), ("0", "No pictured owners")],
}


def build_inference_session(data, task: Path, src: Path, python_span, cpp_span):
    repo = task.parents[3]
    revision = data["revision"]
    files = {}
    for file in FILES.values():
        raw = (repo / file).read_bytes()
        pinned = subprocess.check_output(["git", "show", f"{revision}:{file}"], cwd=repo)
        if raw != pinned:
            raise ValueError(f"InferenceSession source differs from pinned revision: {file}")
        target = task / "sources" / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[file] = dict(path=file, snapshot=f"sources/{file}", sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode())
    refs = {}
    for key, spec in REFS.items():
        file = FILES[spec["alias"]]
        text = files[file]["text"]
        start, end = python_span(text, spec["symbol"]) if spec["language"] == "python" else cpp_span(text, spec["symbol"], spec.get("prefix", 0))
        code = "".join(text.splitlines(keepends=True)[start - 1:end])
        line = start
        if needle := spec.get("needle"):
            if code.count(needle) != 1:
                raise ValueError(f"Nonunique InferenceSession source target: {key}")
            line += code[:code.index(needle)].count("\n")
        refs[key] = dict(**spec, file=file, start=start, end=end, line=line, sha256=hashlib.sha256(code.encode()).hexdigest())

    def link(key):
        r = refs[key]
        return f'<a data-ref="{key}" href="../sources/{r["file"]}#L{r["line"]}">{html.escape(r["label"])} <small>[{r["alias"]}:L{r["line"]}]</small></a>'

    lessons = []
    for item in LESSONS:
        key = item["id"]
        sources = " · ".join(link(ref) for ref in item["refs"])
        phases = ''.join(f'<li data-phase="{phase}" aria-current="false">{label}</li>' for phase, label in PHASES[key])
        lessons.append(f'''<section class="lesson" id="{key}" data-lesson="{key}" aria-labelledby="heading-{key}">
<div class="experiment">
 <div class="experiment-top"><button class="lesson-badge" data-lesson-card="{key}" aria-label="Explain lesson {item['number']}: {html.escape(item['title'], quote=True)}">{item['number']}</button><h2 id="heading-{key}">{item['title']}</h2><span class="timing-note">{item['subtitle']}</span></div>
 <div class="interactive-settings" data-interactive hidden>{controls(key)}</div>
 <p class="lesson-intro">{item['intro']}</p><p class="pan-hint">Scroll inside the diagram to see all components.</p>
 <div class="visual" tabindex="0" aria-label="{html.escape(item['title'], quote=True)}; scroll horizontally on narrow screens">{SVG_BUILDERS[key]()}</div>
 <div class="playback" data-interactive hidden><button class="play" data-action="play" aria-pressed="false">▶ Play</button><button data-action="back" aria-label="Previous {key} step">Back</button><button data-action="step">Step</button><button data-action="reset">↺ Reset</button><label>Speed <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><output data-progress>Ready</output></div>
 <ol class="phases" aria-label="{key} stages">{phases}</ol>
 <div class="state-strip" data-state-strip aria-label="Illustrated object state"></div>
 <div class="event" data-event role="status" aria-live="polite"><span data-event-title>Ready to explore</span></div>
 <details class="event-explanation"><summary>Why this happened · source</summary><p data-event-detail>{item['detail']}</p><div class="event-sources" data-event-sources>{sources}</div></details>
 <details class="transcript"><summary>Step index, portable flow and source evidence</summary><p>{item['detail']}</p><p><b>Try it:</b> {item['try_it']}</p><ol data-transcript></ol><pre>{html.escape(item['ascii'])}</pre><p>{sources}</p></details>
</div></section>''')
    source_cells = []
    for key in ("session", "native_ctor", "context_create", "native_class", "model_handle", "compile_graphs", "compile", "init_all", "native_load_all", "model_execute", "native_execute"):
        r = refs[key]
        lines = files[r["file"]]["text"].splitlines()
        code = "\n".join(f'<span class="code-line"><span class="line-number">{n}</span>{html.escape(lines[n - 1])}</span>' for n in range(r["start"], r["end"] + 1))
        source_cells.append(f'<details class="code-cell" id="code-{key}"><summary>{html.escape(r["label"])} <span>{r["end"] - r["start"] + 1} complete lines · SOURCE</span></summary><p>{link(key)} · <code>{r["file"]}</code></p><pre><code>{code}</code></pre></details>')
    payload = dict(revision=revision, references=refs, files=files, lessons=LESSONS,
                   components={k: dict(title=v[0], color=v[1], text=v[2], refs=v[3]) for k, v in COMPONENTS.items()})
    template = (src / "inference_session.html").read_text()
    replacements = {
        "{{lessons}}": "\n".join(lessons),
        "{{source-cells}}": "\n".join(source_cells),
        "{{source-links}}": "<ul>" + "".join("<li>" + link(key) + "</li>" for key in refs) + "</ul>",
        "{{revision}}": revision, "{{revision-short}}": revision[:12],
        "{{session-link}}": link("session"),
        "{{css}}": (src / "inference_session.css").read_text(),
        "{{js}}": "\n".join((src / name).read_text() for name in ("inference_session_model.js", "inference_session_visuals.js", "inference_session.js")),
        "{{payload}}": json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
    }
    for token, replacement in replacements.items():
        template = template.replace(token, replacement)
    if re.search(r"\{\{[a-z][a-z-]*\}\}", template):
        raise ValueError("Unresolved InferenceSession template token")
    (task / "link_htmls" / PAGE_NAME).write_text(template)
    manifest = dict(revision=revision, page=PAGE_NAME, model="meta-llama/Llama-2-7b-hf",
                    evidence="SOURCE; GPU/driver animation is illustrative BOUNDARY",
                    references=refs, files=[{k: v for k, v in f.items() if k != "text"} for f in files.values()],
                    lessons=[item["id"] for item in LESSONS])
    (task / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    fence = chr(96) * 3
    guide = ["# InferenceSession: devices, runtime and compilation", "",
             f"[Open the animations](link_htmls/{PAGE_NAME}).", "",
             f"SOURCE revision: {revision}. Model context: meta-llama/Llama-2-7b-hf.", "",
             "All steps are source-derived teaching states. No model inference, compilation, or GPU measurement was run.", ""]
    for item in LESSONS:
        guide += ["## " + item["number"] + " · " + item["title"], "", item["intro"], "",
                  fence + "text", item["ascii"], fence, "", item["detail"], "",
                  " · ".join(f'[{refs[k]["label"]}](sources/{refs[k]["file"]}#L{refs[k]["line"]})' for k in item["refs"]), ""]
    guide += ["## Build and verify", "", fence + "sh",
              "PYTHONDONTWRITEBYTECODE=1 python3 " + str(task.relative_to(repo) / "src/build.py"),
              "node " + str(task.relative_to(repo) / "src/check_inference_session.cjs"),
              "# Set NODE_PATH to an available Playwright installation when necessary.",
              "node " + str(task.relative_to(repo) / "src/check_inference_session_browser.cjs"),
              fence, ""]
    (task / GUIDE_NAME).write_text("\n".join(guide))
    return len(refs), len(LESSONS)
