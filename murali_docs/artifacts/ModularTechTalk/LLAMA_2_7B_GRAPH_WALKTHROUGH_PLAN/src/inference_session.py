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
    "arguments": ("Constructor arguments", "config", "Device handles and optional thread, artifact and extension settings. These are example values, not a measurement.", ["session"]),
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
    "results": ("Output Buffers / logits", "weights", "Native output buffers are wrapped for Python and processed by the pipeline. This does not imply an automatic host copy of every output.", ["extract_results", "model_execute", "pipeline_execute"]),
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
         try_it="Compare Cold compile and Stored MEF hit. Both still need initialization. Missing weight fails before native model initialization.",
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
         detail="The Python pipeline prepares Buffers. Llama's engine Model calls the native runtime through its retained session. Result Buffers come back for output processing and sampling.",
         try_it="Play Prefill + decode. Watch five prompt tokens become two decode tokens while S1, M1 and the setup counts stay unchanged.",
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
         try_it="Compare Caller first and Models first. A remaining pictured model keeps a strong reference to S1. Shared resources can survive the last pictured owner.",
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
            + body + '<g class="motion-packet" hidden aria-hidden="true"><rect x="-20" y="-12" width="40" height="24" rx="6"/>'
            '<text text-anchor="middle" y="4"></text></g></svg>')


def constructor_svg():
    b = '<rect class="process-boundary" x="20" y="26" width="970" height="500" rx="15"/><text class="process-label" x="505" y="51">MODEL WORKER · HOST PROCESS</text>'
    b += '<rect class="native-boundary" data-boundary="native" x="298" y="311" width="666" height="190" rx="12"/><text class="boundary-label" x="631" y="339">Core::InferenceSession · native C++ object</text>'
    b += wire("call", "M266 124 H298", "call", 282, 112)
    b += wire("normalize", "M576 124 H610")
    b += wire("options", "M437 162 V187")
    b += wire("native-call", "M437 287 V311", "_impl", 480, 304)
    b += wire("compile-config", "M614 345 V361")
    b += wire("register", "M787 227 V298 H410 V361", "DeviceSpec + ctx", 748, 286)
    b += wire("context", "M410 477 V489 H835 V477", "runtimeConfig", 620, 486)
    b += wire("device-handle", "M943 441 H1016 V420 H1038", "handles", 1007, 404, "reference")
    b += wire("return", "M320 479 H284 V172 H154 V162", "return S1", 239, 465, "return")
    b += node("pipeline", 42, 82, 224, 80, "TextGenerationPipeline", "creates session", status="pipeline")
    b += node("python", 298, 82, 278, 80, "InferenceSession", "Python wrapper", status="python")
    b += node("normalize", 610, 82, 354, 145, "Normalize devices", "first occurrences + host CPU", status="devicesStatus")
    b += '<g data-slots="normalized" transform="translate(630 153)"></g>'
    b += node("arguments", 42, 190, 224, 163, "Arguments", "illustrative Device handles", status=False)
    b += '<text class="input-value" x="154" y="266" data-constructor-input>[gpu0]</text><text class="node-sub" x="154" y="300" data-constructor-threads>num_threads = None</text><text class="node-sub" x="154" y="326" data-constructor-store>MEF policy: none</text>'
    b += node("store", 298, 187, 278, 100, "Artifact / extension policy", "session options", status="storeStatus")
    b += node("devices", 320, 361, 180, 116, "Device handles", "runtimeConfig.devices", status="registered")
    b += node("compile", 522, 361, 184, 116, "Compile config", "sessionCompileConfig", status="compileStatus")
    b += node("context", 730, 361, 213, 116, "Engine::Context", "runtimeConfig + CPUDevice", status="contextStatus")
    b += node("hardware", 1038, 352, 180, 152, "Selected hardware", "device boundary", status="hardware")
    b += '<text class="diagram-note" x="1128" y="322">BOUNDARY</text><text class="diagram-note" x="631" y="550">At return: compilation context exists; the decoder is still unloaded.</text>'
    return svg_shell("constructor", 574, b)


def compilation_svg():
    b = '<rect class="process-boundary" x="20" y="24" width="1200" height="459" rx="15"/><text class="process-label" x="620" y="50">MODEL LOADING IN THE WORKER · SOURCE</text>'
    b += node("session", 42, 70, 1160, 70, "S1 · the same InferenceSession", "compilation configuration + selected-device runtime", status=False)
    b += wire("trace", "M164 140 V173", "trace", 202, 162)
    b += wire("compile", "M284 229 H330")
    b += wire("artifact", "M592 229 H630")
    b += wire("reuse", "M461 173 V151 H754 V173", "MEF reuse", 680, 164, "reference")
    b += wire("initialize", "M878 229 H923")
    b += wire("weights", "M878 379 H899 V305 H990 V285", "bind", 946, 300)
    b += wire("model", "M1063 285 V326")
    b += node("graph", 42, 173, 242, 112, "GraphSpec → Graph", "trace the model", status="graphStatus")
    b += node("compiler", 330, 173, 262, 112, "Compile / import", "session.compile_reusing_mefs", status="compilerStatus")
    b += node("artifact", 630, 173, 248, 112, "CompiledModel", "code + metadata", status="artifactStatus")
    b += node("init", 923, 173, 279, 112, "session.init_all", "validate → initialize", status="initStatus")
    b += node("registry", 630, 326, 248, 122, "Weight registry", "illustrative name → value pairs", status="registryStatus")
    b += '<g data-slots="weights" transform="translate(656 382)"></g>'
    b += node("model", 923, 326, 279, 122, "Engine Model M1", "executable decoder", status="modelStatus")
    b += '<text class="diagram-note" x="285" y="358">Graph IR records computation.</text><text class="diagram-note" x="285" y="391">Weights provide actual values.</text><text class="diagram-note" x="285" y="424">Initialization joins the two.</text>'
    return svg_shell("compilation", 510, b)


def execution_svg():
    b = '<rect class="process-boundary" x="20" y="28" width="944" height="491" rx="15"/><text class="process-label" x="492" y="54">MODEL WORKER · SAME S1 AND M1 ACROSS PASSES</text>'
    b += wire("prepare", "M166 205 V251", "prepare", 209, 232)
    b += wire("execute", "M289 155 H344", "call", 316, 144)
    b += wire("runtime", "M589 155 H645")
    b += wire("submit", "M939 155 H996 M1104 173 V210", "submit", 981, 145)
    b += wire("result", "M996 404 H939", "Buffers", 980, 389, "return")
    b += wire("feedback", "M645 405 H612 V222 H166 V205", "process / sample / next pass", 418, 214, "return")
    b += node("pipeline", 44, 105, 245, 100, "Text-generation pipeline", "prepare → execute → sample", status="pipelineStatus")
    b += node("model", 344, 105, 245, 100, "Engine Model M1", "retains native session S1", status="modelStatus")
    b += node("runtime", 645, 105, 294, 100, "MAX runtime / user driver", "native code in this process", status="runtimeStatus")
    b += node("driver", 996, 95, 215, 78, "OS GPU driver", "kernel / device boundary", status=False)
    b += node("gpu", 996, 210, 215, 280, "GPU device", "hardware · illustrative work", status="gpuStatus")
    b += '<rect class="device-memory weight-memory" x="1014" y="274" width="179" height="38" rx="6"/><text class="memory-label" x="1103" y="298">weights · retained</text>'
    b += '<rect class="device-memory kv-memory" x="1014" y="324" width="179" height="38" rx="6"/><text class="memory-label" x="1103" y="348">KV data · retained</text>'
    b += '<g data-gpu-work><rect class="device-work" x="1014" y="381" width="179" height="52" rx="6"/><text class="work-label" x="1103" y="412" data-work-label>between passes</text></g>'
    b += node("buffers", 44, 251, 545, 155, "Packed model input view", "A then B · token IDs are illustrative", status="bufferPlace")
    b += '<g data-slots="tokens" transform="translate(91 321)"></g><text class="input-value" x="316" y="372" data-offsets>row_offsets = []</text>'
    b += node("results", 645, 326, 294, 123, "Result Buffers", "batch processor → sampling", status="resultStatus")
    b += '<g data-result-tile hidden><rect class="logits-tile" x="751" y="382" width="83" height="27" rx="5"/><text class="memory-label" x="792" y="401">logits</text></g>'
    b += '<text class="diagram-note" x="316" y="453">Buffer placement follows model input metadata.</text><text class="diagram-note" x="316" y="481">A new batch reuses the initialized decoder.</text><text class="diagram-note" x="1104" y="542">BOUNDARY · no GPU run</text>'
    return svg_shell("execution", 566, b)


def lifetime_svg():
    b = '<rect class="process-boundary" x="20" y="28" width="1200" height="450" rx="15"/><text class="process-label" x="620" y="54">NATIVE OWNERSHIP · THREE PICTURED OWNERS · SOURCE</text>'
    b += wire("caller-ref", "M304 178 H506", "_impl reference", 404, 158, "reference")
    b += wire("decoder-ref", "M304 363 H458 V218 H506", "native ref", 407, 338, "reference")
    b += wire("sampler-ref", "M628 308 V256", "native ref", 675, 291, "reference")
    b += wire("owned", "M778 178 H879", "owns", 829, 163, "reference")
    b += node("caller", 54, 112, 250, 130, "Python session wrapper", "InferenceSession._impl", status="caller")
    b += node("decoder", 54, 308, 250, 124, "Decoder ModelHandle", "inferenceSession reference", status="decoder")
    b += node("sampler", 506, 308, 250, 124, "Sampler ModelHandle", "inferenceSession reference", status="sampler")
    b += node("native", 506, 112, 272, 144, "Native session S1", "illustrated strong owners", status="nativeStatus")
    b += '<text class="reference-count" x="642" y="206" data-reference-count>3</text>'
    b += node("context", 879, 112, 309, 144, "Engine::Context", "config + MAX context / CPUDevice", status="runtimeStatus")
    b += node("release", 879, 308, 309, 124, "After the final owner", "shared runtime / driver resources", status="releaseStatus")
    return svg_shell("lifetime", 503, b)


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
    return '<div class="scenario-options" role="group" aria-label="Choose a ' + key + ' scenario">' + "".join(
        f'<button data-mode="{k}" aria-pressed="{str(i == 0).lower()}">{label}</button>' for i, (k, label) in enumerate(choices)) + "</div>"


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
        lessons.append(f'''<section class="lesson" id="{key}" data-lesson="{key}" aria-labelledby="heading-{key}">
<div class="experiment">
 <div class="experiment-top"><button class="lesson-badge" data-lesson-card="{key}" aria-label="Explain lesson {item['number']}: {html.escape(item['title'], quote=True)}">{item['number']}</button><h2 id="heading-{key}">{item['title']}</h2><span class="timing-note">{item['subtitle']}</span></div>
 <div class="interactive-settings" data-interactive hidden>{controls(key)}</div>
 <p class="lesson-intro">{item['intro']}</p><p class="pan-hint">Scroll inside the diagram to see all components.</p>
 <div class="visual" tabindex="0" aria-label="{html.escape(item['title'], quote=True)}; scroll horizontally on narrow screens">{SVG_BUILDERS[key]()}</div>
 <div class="playback" data-interactive hidden><button class="play" data-action="play" aria-pressed="false">▶ Play</button><button data-action="back" aria-label="Previous {key} step">Back</button><button data-action="step">Step</button><button data-action="reset">↺ Reset</button><label>Speed <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><output data-progress>Ready</output></div>
 <div class="state-strip" data-state-strip aria-label="Illustrated object state"></div>
 <div class="event" data-event role="status" aria-live="polite"><span data-event-title>Ready to explore</span><p data-event-detail>{item['detail']}</p></div>
 <div class="event-sources" data-event-sources>{sources}</div>
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
        "{{js}}": (src / "inference_session_model.js").read_text() + "\n" + (src / "inference_session.js").read_text(),
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
