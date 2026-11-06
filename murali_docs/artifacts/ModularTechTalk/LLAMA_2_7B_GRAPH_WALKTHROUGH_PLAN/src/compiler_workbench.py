"""Build only the offline compiler workbench; optionally refresh its session link."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from build import cpp_span, python_span
from cells import REVISION
from compiler_journey import add_sources, lesson as journey_lesson, panel as journey_panel

TASK = Path(__file__).resolve().parent.parent
SRC = TASK / "src"
REPO = TASK.parents[3]
PAGE_NAME = "max_mlir_mojo_compiler_and_runtime.html"
BASE_REFS = "graph_init graph_output llama_trace frontend mo_pipeline map_pipeline mogg_pipeline mgp_pipeline gc_pipeline compile_async serialize artifact export init_all native_init setup native_execute driver_execute driver_results runtime_execute ready_counts async_users execute mojo_add mojo_compiler".split()
OC = "Mojo/lib/Compiler/ObjectCompiler/"
MOJO_DOC = "Mojo/docs/compiler/MojoCompilerWalkthrough.md"
EXTRA = {
    "backend_select": (OC + "ObjectCompiler.cpp", "ObjectCompiler::create(", "cpp", 1, "Select target backend"),
    "llvm_translation": (OC + "ObjectCompiler.cpp", "translateModuleToLLVMIR(llvm::LLVMContext &ctx, ModuleOp module,", "cpp", 1, "MLIR → LLVM module translation"),
    "host_assembly": (OC + "Target/Host/HostBackend.cpp", "HostBackend::emitAssembly(", "cpp", 0, "Host/target assembly emission"),
    "host_object": (OC + "Target/Host/HostBackend.cpp", "HostBackend::emitObject(", "cpp", 0, "Object emission and linking"),
    "nvptx_object": (OC + "Target/max_license/NVPTX/NVPTXBackend.cpp", "ErrorOr<BufferRef> emitObject(llvm::Module &module,", "cpp", 2, "LLVM PTX → CUDA CompilationDevice → cubin"),
    "amdgpu_backend": (OC + "Target/max_license/AMDGPU/AMDGPUBackend.cpp", "class AMDGPUBackend final", "cpp", 0, "AMDGPU target and ELF linking"),
    "mojo_semantics": (MOJO_DOC, "## Phase 2: Semantic Checking and LIT Lowering", "section", 0, "Mojo lifetimes and LIT lowering"),
    "mojo_elaboration": (MOJO_DOC, "## Phase 4: Elaboration (Monomorphization)", "section", 0, "Mojo compile-time specialization"),
    "mojo_llvm": (MOJO_DOC, "## Phase 6: Lowering to LLVM", "section", 0, "KGEN / POP → LLVM dialect"),
    "compiler_tools": ("Mojo/docs/compiler/README.md", "## Artifacts", "section", 0, "Compiler artifacts and tools"),
    "cross_target": ("GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir", None, "file", 0, "Cross-target codegen test · inspected, not run"),
    "stage_dumps": ("GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir", None, "file", 0, "MAX stage-dump test · inspected, not run"),
}
COLORS = {
    "python": ("#3458a1", "#edf2ff"), "engine": ("#885123", "#fff2e4"),
    "mlir": ("#644895", "#f2ecff"), "mojo": ("#973e28", "#fff0e8"),
    "backend": ("#235f68", "#eaf7f7"), "artifact": ("#6d6220", "#fff8dd"),
    "buffer": ("#774c86", "#f7edff"), "runtime": ("#2e684b", "#eaf5ec"),
    "driver": ("#45566f", "#edf3fa"), "hardware": ("#9b373e", "#fff0f2"),
    "sync": ("#4144a0", "#eeefff"),
}
COMPONENTS = [
    ("python", "Python / Graph builder", "Language frontend · application process", "Records symbolic operators and their input/output contract; recording the graph does not run inference.", ["graph_init", "graph_output"]),
    ("engine", "MAX session / Model", "Python wrappers + native objects", "S1 retains compilation/runtime context; M1 becomes executable after artifact initialization and weight binding.", ["artifact", "init_all"]),
    ("mlir", "MLIR / MAX dialects", "In-memory compiler IR · same host process", "Dialects express different abstraction levels. A compiler-owned clone lowers through MO, MAP, MOGG and MGP.", ["frontend", "gc_pipeline"]),
    ("mojo", "Mojo / KGEN", "Systems language + compiler IR", "Readable syntax, concrete types, ownership and compile-time parameters lower through LIT/KGEN/POP toward LLVM. Typed native code avoids per-operation Python interpreter dispatch.", ["mojo_compiler", "mojo_semantics", "mojo_add"]),
    ("backend", "Target compiler / tools", "Native compiler libraries; tool invocation where applicable", "Target triple/arch select the backend. LLVM emits native objects or assembly; NVIDIA PTX goes through CUDA CompilationDevice. Do not infer a separate process for every compiler stage.", ["backend_select", "host_object", "nvptx_object", "amdgpu_backend"]),
    ("artifact", "Object / MEF / binary", "Byte buffers or explicitly exported files", "A file-shaped card can represent an in-memory IR snapshot. The suffix describes a dump/export format; MEF serialization and model initialization are separate stages.", ["serialize", "export", "compiler_tools"]),
    ("buffer", "Data buffers / references", "Allocation + typed views / handles", "Values, references and owner handles have separate identities. An explicit copy transfers values; moving a handle does not inherently duplicate the allocation.", ["mojo_semantics", "driver_results"]),
    ("runtime", "MAX runtime", "Native host runtime / dependency scheduler", "Ready primitives execute using the session context. AsyncValue readiness and callbacks coordinate dependencies and results.", ["runtime_execute", "ready_counts", "async_users"]),
    ("driver", "User / device driver", "Native driver API and OS/device boundary", "Execution submits work referring to resident code and buffers. Exact command-ring implementation, overlap and timing are BOUNDARY.", ["driver_execute", "driver_results"]),
    ("hardware", "Selected hardware", "CPU or accelerator · physical device", "Codegen can target hardware without executing it. Execution uses compatible real devices. Chip/core/register layouts in these lessons are schematic.", ["cross_target"]),
    ("sync", "Readiness / lifetime gates", "Compiler checks or runtime dependencies", "Borrow checking is a compile-time legality check. Runtime result readiness is a different mechanism and is not inherently a device-wide barrier.", ["mojo_semantics", "driver_results"]),
]
READINGS = [
    ("Modular · Mojo at LLVM 2023", "https://www.modular.com/blog/mojo-llvm-2023", "The author describes an animated, clickable presentation. Inspiration for staged semantic views; its hosted player is not a runtime dependency."),
    ("MLIR interactive pipeline editor", "https://github.com/mortbopet/mlir-interactive", "Primary project: inspect intermediate IR and compose compiler passes interactively."),
    ("VizMLIR", "https://github.com/joepothiboot/vizmlir", "Primary browser visualizer: follow operation ancestry across creation, fusion, lowering and removal."),
    ("MLIR Toy · partial lowering", "https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/", "Mixing dialects lets a compiler preserve useful high-level semantics while lowering selected operations."),
    ("MLIR Toy · LLVM lowering", "https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/", "LLVM-dialect lowering, translation and JIT are distinct activities."),
    ("MLIR · LLVM IR target", "https://mlir.llvm.org/docs/TargetLLVMIR/", "LLVM dialect is MLIR; an LLVM IR module is produced through translation."),
    ("MLIR · GPU compilation", "https://mlir.llvm.org/docs/Dialects/GPU/#gpu-compilation", "Generic MLIR GPU target serialization and offloading. This is a reference route, not a claim that MAX uses every generic GPU pass."),
    ("LLVM · NVPTX backend", "https://llvm.org/docs/NVPTXUsage.html", "The target backend emits PTX virtual-ISA assembly."),
    ("LLVM · CUDA compilation", "https://llvm.org/docs/CompileCudaWithLLVM.html", "PTX assembly and machine-code binaries have different roles. Generic toolchain examples do not override the pinned MAX/Mojo path."),
    ("LLVM · objdump", "https://llvm.org/docs/CommandGuide/llvm-objdump.html", "Disassembly is a read-only view of existing object/binary code."),
    ("NVIDIA · CUDA binary utilities", "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html", "nvdisasm and cuobjdump inspect CUDA cubin/SASS. The workbench keeps that listing separate from PTX text emission."),
    ("LLVM blog · Life of an instruction", "https://blog.llvm.org/2012/11/life-of-instruction-in-llvm.html", "Historical instruction walkthrough: follow values through backend selection and encoding. Current backend documentation establishes today's general mechanisms; this is not the exact MAX pass order."),
    ("Nikita Popov · LLVM middle-end pipeline", "https://www.npopov.com/2023/04/07/LLVM-middle-end-pipeline.html", "The LLVM contributor explains grouped and repeated optimization pipelines. A lowering diagram is not a promise that each pass runs exactly once."),
    ("Jeremy Kun · MLIR lowering through LLVM", "https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/", "Original worked example tracks conversions into LLVM and native output. Before/after fragments here are labeled schematic, not copied compiler dumps."),
    ("Eli Bendersky · Adventures in JIT compilation", "https://eli.thegreenplace.net/2017/adventures-in-jit-compilation-part-3-llvm/", "Original LLVM JIT tutorial separates generated code from calling it. Its older API example does not establish MAX's runtime implementation."),
    ("MLIR · Bufferization", "https://mlir.llvm.org/docs/Bufferization/", "Tensor values and memory buffers have different semantics; aliasing and reads/writes matter. The epilogue counters illustrate intermediate traffic without claiming MAX uses One-Shot Bufferize."),
    ("MLIR · Transform tutorial", "https://mlir.llvm.org/docs/Tutorials/transform/Ch0/", "Payload operations and transformation handles are separate. Inspiration for selecting operation ancestry; not a claim that MAX uses the Transform dialect."),
    ("LLVM · Code generator", "https://llvm.org/docs/CodeGenerator.html", "Instruction selection, virtual/physical registers, scheduling and MC emission. PTX registers are a virtual-ISA boundary; physical GPU allocation happens later."),
    ("LLVM · ORC JIT v2", "https://llvm.org/docs/ORCv2.html", "Symbol lookup/materialization and optional lazy compilation are distinct. MAX's source-backed compile/init/execute and cache routes are authoritative here."),
    ("Clang · Cross compilation", "https://clang.llvm.org/docs/CrossCompilation.html", "Separate compiler host, target triple/features and supporting tools/libraries. CPU/NVIDIA/AMD selectors change the illustrated code artifacts."),
    ("Modular · Mojo optimization pipeline, LLVM 2024", "https://llvm.org/devmtg/2024-10/slides/techtalk/Weiwei-What-We-Learned-Building-Mojo-OptimizationPipeline.pdf", "Compiler authors describe early structured-IR simplification, elaboration and LLVM-module splitting. Historical timing figures are not measurements of these examples."),
    ("Modular · Mojo GPU compilation, LLVM 2025", "https://llvm.org/devmtg/2025-10/slides/lightning_talks/chen_dakkak.pdf", "Compiler authors show unified host/GPU input, GPU-module extraction during elaboration, target compilation and host incorporation of offload output. The workbench separates their artifacts without promising a strictly serial pipeline."),
]
LESSONS = [
    ("graph", "01", "From Python to a verified graph", "Symbolic construction · no model execution", "Watch a definition become named operators, then seal its output contract.", "SOURCE · toy expression is schematic", "Python builder → G1 / native MLIR → Graph.output signature + verify\nVerified Graph ≠ compiled artifact ≠ initialized Model", ["graph_init", "graph_output"]),
    ("lower", "02", "A module changes its language", "MAX compiler · progressive MLIR lowering", "Follow C1 through structured bodies, kernel calls, buffers and generated Mojo.", "SOURCE order · illustrative IR views / fusion", "G1 retained → compiler clone C1\nMO → MAP → MOGG → MGP → symbol DCE → emitted Mojo\nEnclosing pipeline: kernel JIT → final runtime-call lowering", ["frontend", "gc_pipeline", "stage_dumps"]),
    ("mojo", "03", "Readable code. Native control.", "Mojo systems lab · specialization and ownership", "Change the SIMD width, or try a conflicting borrow and watch the legality gate.", "SOURCE concepts · CPU lane and ownership examples", "Mojo → LIT → lifetime/control-flow checks → KGEN\nParameters → elaborated functions → LLVM dialect\nHandle/reference movement ≠ copying allocation cells", ["mojo_compiler", "mojo_semantics", "mojo_add"]),
    ("target", "04", "Pick a target. Change the binary.", "LLVM backend · cross-compilation", "Change the target, choose assembly or a binary, then inspect its disassembly.", "SOURCE backend routes · illustrative files", "LLVM dialect / MLIR → LLVM module\nCPU: LLVM MC → .o → link → .so\nNVIDIA: LLVM NVPTX → PTX → CUDA CompilationDevice → cubin\nAMD: LLVM AMDGPU → .o → ELF link → hsaco\nBinary → disassembly is an inspection branch", ["llvm_translation", "backend_select", "nvptx_object", "amdgpu_backend"]),
    ("package", "05", "Pack the artifact. Bind the values.", "MEF preparation · runtime initialization", "A cutaway archive separates compiled code, its plan and the weight-binding rack.", "SOURCE lifecycle · conceptual MEF cutaway", "Compiled routines + MGP plan/resources → prepared primitives → MEF bytes\nOptional export → model.mef\nCompiledModel + weight registry → init_all → executable M1", ["serialize", "export", "init_all"]),
    ("runtime", "06", "Code stays. Buffers move.", "Native runtime · drivers · selected hardware", "Queue new inputs while the same code, weights, M1 and S1 remain resident.", "BOUNDARY device motion · SOURCE readiness / reuse", "Ready M1/S1 + inputs → optional explicit upload\nDescriptor references code/buffers → native driver → hardware\nAsyncValue ready → result buffers → optional explicit host copy\nNext call reuses code/model/session", ["driver_execute", "driver_results", "ready_counts"]),
]


def owner(key, x, y, w, h, title, subtitle="", color=None):
    color = color or key
    return f'<g data-component="{color}" class="owner"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13"/><text class="owner-title" x="{x+15}" y="{y+24}">{html.escape(title)}</text><text class="owner-subtitle" x="{x+15}" y="{y+43}">{html.escape(subtitle)}</text></g>'


def wire(path, label="", x=0, y=0, key="", kind="call"):
    return f'<g class="wire {kind}" data-wire="{key}"><path d="{path}" marker-end="url(#arrow)"/>' + (f'<text x="{x}" y="{y}">{html.escape(label)}</text>' if label else "") + '</g>'


def slots(name, x, y, count, width=46, gap=9, height=40):
    return f'<g data-slots="{name}">' + ''.join(f'<g data-slot="{i}" data-occupied="false"><rect x="{x+i*(width+gap)}" y="{y}" width="{width}" height="{height}" rx="5"/><text x="{x+i*(width+gap)+width/2}" y="{y+height/2+4}">—</text></g>' for i in range(count)) + '</g>'


def scene(kind):
    if kind == "graph":
        body = owner("python", 24, 64, 268, 350, "PYTHON BUILDER", "one host application process") + owner("mlir", 326, 64, 676, 350, "G1 · NATIVE GRAPH / MLIR", "symbolic values, types and def-use edges") + owner("hardware", 1028, 64, 148, 350, "TARGET DEVICE", "not executing")
        body += wire("M292 221 H326", "record", 295, 207) + wire("M725 228 H762 V278", "depends", 734, 259)
        body += '<rect class="seal-slot" x="365" y="351" width="594" height="43" rx="7"/><text class="static-label" x="390" y="378" data-seal-label>Output contract · open</text><text class="static-label" x="1050" y="292">submitted work</text><text class="zero-count" x="1103" y="330">0</text>'
    elif kind == "lower":
        body = owner("python", 24, 64, 242, 350, "CALLER GRAPH", "G1 stays with its owner") + owner("mlir", 296, 64, 498, 350, "COMPILER WORKBENCH", "C1 · the compiler-owned clone") + owner("artifact", 824, 64, 352, 350, "SNAPSHOT TRAY", "illustrative views · dump optional")
        body += wire("M266 230 H296", "clone", 270, 213) + wire("M794 232 H824", "view", 797, 216)
        body += '<g class="press"><rect x="345" y="135" width="398" height="220" rx="15"/><path d="M365 153H723 M365 332H723"/><circle data-gear cx="716" cy="176" r="13"/></g><text class="static-label" x="361" y="387" data-pass-label>Awaiting C1</text><rect class="bin" x="1017" y="327" width="114" height="60" rx="8"/><path class="bin-lid" d="M1009 320H1139"/><text class="static-label" x="1040" y="406">symbol DCE</text>'
    elif kind == "mojo":
        body = owner("mojo", 24, 64, 262, 350, "TYPED MOJO SOURCE", "readable syntax · explicit semantics") + owner("mlir", 316, 64, 424, 350, "SEMANTICS / SPECIALIZATION", "LIT → KGEN / POP → LLVM") + owner("buffer", 770, 64, 406, 350, "VALUES / NATIVE CONTROL", "CPU SIMD lab or B1 ownership")
        body += wire("M286 232 H316", "lower", 289, 216) + wire("M740 232 H770", "native", 741, 216)
        body += '<g data-lane-view><text class="static-label" x="791" y="170">A values</text>' + slots("lane-input", 790, 186, 8, 39, 6) + '<text class="static-label" x="791" y="267">+ 10 · grouped by WIDTH</text>' + slots("lane-output", 790, 282, 8, 39, 6) + '<g data-lane-groups></g><text class="static-label" x="790" y="362" data-bundles>Compile-time WIDTH = 4</text></g>'
        body += '<g data-owner-view hidden><text class="static-label" x="791" y="188">B1 allocation · four cells</text>' + slots("owned-cells", 803, 216, 4, 71, 12) + '<path class="reference-edge" data-ref-edge d="M696 164 H752 V234 H803"/><text class="static-label" x="791" y="310">owner handle ≠ stored data</text></g>'
    elif kind == "target":
        body = owner("mlir", 24, 64, 250, 350, "COMPILER IR", "in memory · optional .mlir / .ll dumps") + owner("backend", 304, 64, 292, 350, "TARGET CABINET", "triple / arch / features") + owner("backend", 626, 64, 288, 350, "CODE GENERATION", "native libraries / selected tools") + owner("artifact", 944, 64, 232, 350, "OUTPUT TRAY", "binary or text")
        body += wire("M274 234 H304") + wire("M596 234 H626") + wire("M914 234 H944")
        body += '<g class="tool-rack"><path d="M324 333H575M324 354H575"/></g><text class="static-label" x="326" y="386" data-target-arch>sm_90a · CPU host → GPU target</text><text class="static-label" x="643" y="382" data-tool-label>Waiting for target selection</text>'
    elif kind == "package":
        body = owner("artifact", 24, 64, 252, 350, "PREPARED INPUTS", "compiled routines + MGP plan") + owner("artifact", 306, 64, 420, 350, "MEF PACKING BENCH", "conceptual cutaway · not byte layout") + owner("buffer", 756, 64, 226, 350, "WEIGHT BINDINGS", "loaded values · separate registry") + owner("engine", 1012, 64, 164, 350, "MODEL DOCK", "M1 + retained S1")
        body += wire("M276 229 H306") + wire("M726 229 H756") + wire("M982 229 H1012")
        body += slots("bindings", 775, 170, 1, 184, 0, 47) + slots("binding-b", 775, 242, 1, 184, 0, 47) + '<text class="static-label" x="775" y="330" data-binding-gate>Initialization gate · closed</text>'
    else:
        body = owner("python", 24, 64, 306, 350, "CALLER / HOST BUFFERS", "new values for each invocation") + owner("runtime", 360, 64, 260, 350, "NATIVE MAX RUNTIME", "same M1 / S1 · ready dependencies") + owner("hardware", 650, 64, 526, 350, "DRIVER / EXECUTION TARGET", "native submission → physical hardware · schematic")
        body += owner("driver", 674, 131, 180, 100, "DEVICE INTERFACE", "BOUNDARY") + '<g class="chip-board"><rect x="878" y="121" width="273" height="126" rx="12"/><path d="M892 240H1135M892 128H1135"/><text x="894" y="222" data-hardware-label>GPU compute · schematic</text></g>'
        body += '<text class="static-label" x="49" y="156" data-input-label>B1 · input values</text>' + slots("host-input", 49, 170, 4, 56, 9) + '<text class="static-label" x="49" y="310" data-output-label>Host result · only after explicit copy</text>' + slots("host-result", 49, 327, 4, 56, 9)
        body += '<text class="static-label" x="376" y="244">Illustrated dispatch queue</text>' + slots("command", 381, 259, 3, 64, 11, 53)
        body += '<rect class="memory-rack" x="673" y="268" width="478" height="126" rx="9"/><text class="static-label" x="691" y="291" data-memory-label>Device memory · code / weights retained</text><text class="static-label" x="692" y="317" data-device-input-label>D1 · input</text>' + slots("device-input", 691, 330, 4, 46, 6) + '<text class="static-label" x="936" y="317" data-device-output-label>O1 · result</text>' + slots("device-result", 935, 330, 4, 46, 6)
        body += wire("M330 188 H345 V340 H673", "explicit upload", 347, 357, "upload", "data") + wire("M620 281 H644 V181 H674", "submit", 625, 256, "dispatch") + wire("M925 414 V444 H292 V375", "ready result / optional host copy", 525, 461, "completion", "data")
    result = '<svg viewBox="0 0 1200 480" role="img" aria-label="' + html.escape(kind) + ' animation" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#dfe7ed" stroke-width=".7"/></pattern><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#596e79"/></marker></defs><rect class="canvas-backing" width="1200" height="480"/><rect width="1200" height="480" fill="url(#grid)"/><text class="scene-label" x="25" y="31">' + {"graph":"BUILD / VERIFY", "lower":"TRANSFORM / PRESERVE LINEAGE", "mojo":"TYPE / OWN / SPECIALIZE", "target":"CROSS-COMPILE / INSPECT", "package":"SERIALIZE / INITIALIZE", "runtime":"SUBMIT / COMPLETE / REUSE"}[kind] + '</text>' + body + '<g data-actors></g></svg>'
    return result.replace('id="grid"', f'id="grid-{kind}"').replace('url(#grid)', f'url(#grid-{kind})').replace('id="arrow"', f'id="arrow-{kind}"').replace('url(#arrow)', f'url(#arrow-{kind})')


def controls(kind):
    modes = {
        "graph": '<label>Construction <select data-option="seal" data-boolean><option value="true">Trace + finalization</option><option value="false">Trace only</option></select></label><button data-intervention="seal" disabled>Finalize Graph.output</button>',
        "lower": '<label>Illustrative fusion <select data-option="fusion" data-boolean><option value="true">Eligible → grouped body</option><option value="false">Keep separate bodies</option></select></label><span>Grouping demonstrates the mechanism; it is not a MAX configuration flag.</span>',
        "mojo": '<label>Explore <select data-option="lab"><option value="simd">Compile-time SIMD</option><option value="ownership">Ownership / borrow gate</option></select></label><label data-simd-control>WIDTH <select data-option="width"><option value="1">1 · scalar</option><option value="4" selected>4 lanes</option><option value="8">8 lanes</option></select></label><button data-intervention="release" disabled>End immutable borrow</button>',
        "target": '<label>Emit <select data-option="emission"><option value="binary">Executable binary</option><option value="assembly">Assembly / PTX text</option></select></label><label data-cuda-control>CUDA compiler <select data-option="assembler"><option value="api">Default API</option><option value="override">Configured override path</option></select></label><label data-cuda-control>Try <select data-option="failure"><option value="none">Valid toolchain</option><option value="arch">Non-SM target CPU</option><option value="toolkit">Unavailable compilation support</option></select></label><button data-intervention="inspect" disabled>Disassemble binary</button>',
        "package": '<label>Bindings <select data-option="missingWeight" data-boolean><option value="false">Complete registry</option><option value="true">W_b missing</option></select></label><label>MEF <select data-option="export" data-boolean><option value="false">Keep byte buffer</option><option value="true">Export model.mef</option></select></label><button data-intervention="repair" disabled>Supply W_b</button>',
        "runtime": '<label>Input <select data-option="resident" data-boolean><option value="false">Host input · explicit upload</option><option value="true">Already on execution device</option></select></label><label>Output <select data-option="copyResult" data-boolean><option value="false">Keep returned placement</option><option value="true">Explicit host copy</option></select></label><button data-intervention="again" disabled>Queue another call</button>',
    }
    return modes[kind]


def load_sources():
    base = json.loads((TASK / "engine_lifecycle_manifest.json").read_text())
    if base["revision"] != REVISION:
        raise ValueError("Compiler workbench and engine manifest revisions differ")
    refs = {key: dict(base["references"][key]) for key in BASE_REFS}
    for key, (file, needle, language, prefix, label) in EXTRA.items():
        text = (REPO / file).read_text()
        if language == "cpp":
            start, end = cpp_span(text, needle, prefix)
        elif language == "section":
            start = text[:text.index(needle)].count("\n") + 1
            match = re.search(r"\n## ", text[text.index(needle) + len(needle):])
            end = text[:text.index(needle) + len(needle) + match.start()].count("\n") + 1 if match else len(text.splitlines())
        else:
            start, end = 1, len(text.splitlines())
        refs[key] = dict(id=key, file=file, start=start, end=end, line=start, symbol=needle or Path(file).name, label=label, language=language, alias="C++" if language=="cpp" else "DOC" if language=="section" else "TEST")
    files = {}
    for file in sorted({r["file"] for r in refs.values()}):
        raw = (REPO / file).read_bytes()
        pinned = subprocess.check_output(["git", "show", f"{REVISION}:{file}"], cwd=REPO)
        if raw != pinned:
            raise ValueError(f"Source differs from pinned revision: {file}")
        target = TASK / "sources" / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[file] = dict(path=file, text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest())
    for key, r in refs.items():
        r["label"] = r.get("label", r.get("purpose", key))
        r["line"] = r.get("line", r["start"])
        code = "".join(files[r["file"]]["text"].splitlines(keepends=True)[r["start"]-1:r["end"]])
        r["sha256"] = hashlib.sha256(code.encode()).hexdigest()
    add_sources(refs, files, REPO, TASK, REVISION)
    return refs, files


def build_compiler_workbench():
    from compiler_examples import progression_cards
    refs, files = load_sources()
    def link(key):
        r=refs[key]
        return f'<a data-ref="{key}" href="../sources/{r["file"]}#L{r["line"]}">{html.escape(r["label"])} <small>[{r["alias"]}:L{r["line"]}]</small></a>'
    lessons=[journey_panel()]
    for key, number, title, subtitle, intro, evidence, ascii_flow, sources in LESSONS:
        number = f"{int(number)+1:02}"
        lessons.append(f'''<section class="lesson" id="{key}" data-lesson="{key}" aria-labelledby="heading-{key}"><div class="experiment">
<div class="experiment-top"><button class="lesson-badge" data-card="{key}" aria-label="Explain {number}: {title}">{number}</button><h2 id="heading-{key}">{title}</h2><span>{subtitle}</span></div>
<div class="settings" data-interactive hidden>{controls(key)}</div><p class="intro">{intro}</p>
<div class="visual" tabindex="0" aria-label="{title}; scroll inside on narrow screens">{scene(key)}</div>
<div class="playback" data-interactive hidden><button class="play" data-action="play" aria-pressed="false">▶ Play</button><button data-action="back">Back</button><button data-action="step">Step</button><button data-action="reset">↺ Reset</button><label>Speed <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input data-seek type="range" min="0" max="1" value="0" aria-label="{title} step"><output data-progress>Ready</output></div>
<div class="state-strip" data-state-strip></div><div class="event" role="status" aria-live="polite"><strong data-event-title>Ready to explore</strong><span data-event-detail>{intro}</span></div>
<div class="ir-diff" data-ir-diff hidden><div><span data-before-title>Before · schematic IR view</span><pre data-before></pre></div><div><span data-after-title>After · schematic IR view</span><pre data-after></pre></div></div>
<div class="event-sources" data-event-sources>{' · '.join(link(s) for s in sources)}</div><details class="transcript"><summary>Portable flow · steps · evidence</summary><p>{evidence}</p><pre>{html.escape(ascii_flow)}</pre><ol data-transcript></ol></details>
</div></section>''')
    legend=[]
    for key, title, kind, detail, sources in COMPONENTS:
        legend.append(f'<div class="legend-row" data-legend-component="{key}"><span class="swatch" style="background:{COLORS[key][0]}"></span><div><strong>{title}</strong><span class="legend-kind">{kind}</span><p>{detail}</p><div>{" · ".join(link(s) for s in sources)}</div></div></div>')
    source_cells=[]
    for key in ["graph_output", "gc_pipeline", "nvptx_object", "host_object", "serialize", "mojo_add"]:
        r=refs[key]; lines=files[r["file"]]["text"].splitlines()
        code="\n".join(f'<span class="source-line"><span>{n}</span>{html.escape(lines[n-1])}</span>' for n in range(r["start"],r["end"]+1))
        source_cells.append(f'<details class="source-cell"><summary>{html.escape(r["label"])} · complete source</summary><p>{link(key)}</p><pre><code>{code}</code></pre></details>')
    payload=dict(revision=REVISION,references=refs,files=files,lessons=[journey_lesson(),*[dict(id=v[0],number=f"{int(v[1])+1:02}",title=v[2],detail=v[4],evidence=v[5],ascii=v[6],refs=v[7]) for v in LESSONS]],readings=READINGS)
    colors="\n".join(f'[data-component="{key}"], [data-color="{key}"]{{--ink:{ink};--tint:{tint}}}' for key,(ink,tint) in COLORS.items())
    replacements={"css":(SRC/"compiler_workbench.css").read_text()+"\n"+(SRC/"compiler_example_links.css").read_text()+"\n"+(SRC/"compiler_journey.css").read_text()+"\n"+colors,"js":"\n".join((SRC/name).read_text() for name in ("compiler_workbench_model.js","compiler_journey_model.js","compiler_workbench_visuals.js","compiler_journey_visuals.js","compiler_journey.js","compiler_reader_ui.js","compiler_workbench.js")),"lessons":"\n".join(lessons),"examples":progression_cards(),"legend":"\n".join(legend),"source-cells":"\n".join(source_cells),"readings":"".join(f'<li><a href="{url}" target="_blank" rel="noreferrer">{title}</a><p>{note}</p></li>' for title,url,note in READINGS),"revision":REVISION,"payload":json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")}
    template=(SRC/"compiler_workbench.html").read_text()
    # Substitute only the shell's tokens. Original embedded source may contain
    # similar brace sequences and must remain byte-for-byte unchanged.
    tokens = set(re.findall(r"\{\{([a-z-]+)\}\}", template))
    if tokens != replacements.keys():
        raise ValueError(f"Compiler template tokens differ: {tokens ^ replacements.keys()}")
    template = re.sub(r"\{\{([a-z-]+)\}\}", lambda m: replacements[m[1]], template)
    (TASK/"link_htmls"/PAGE_NAME).write_text(template)
    manifest=dict(revision=REVISION,page="link_htmls/"+PAGE_NAME,references=refs,files=[{k:v for k,v in f.items() if k!="text"} for f in files.values()],lessons=payload["lessons"],readings=READINGS)
    (TASK/"compiler_workbench_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    companion=["# MAX / MLIR / Mojo compiler workbench", "", f"[Open the animation](link_htmls/{PAGE_NAME}).", "", f"SOURCE revision: `{REVISION}`. IR views, filenames, lane counts and device motion are illustrations. No MAX compilation or hardware execution was run.", ""]
    for item in payload["lessons"]: companion += ["## "+item["number"]+" · "+item["title"],"",item["detail"],"", "```text",item["ascii"],"```", ""]
    (TASK/"COMPILER_WORKBENCH.md").write_text("\n".join(companion).rstrip()+"\n")
    print(f"Built {PAGE_NAME}: connected journey + {len(LESSONS)} focused lessons, {len(refs)} source targets, {len(files)} complete pinned files")
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-session",action="store_true",help="Also rebuild the session reader from its authoring sources")
    args=parser.parse_args()
    build_compiler_workbench()
    if args.refresh_session:
        from inference_session import build_inference_session
        build_inference_session({"revision":REVISION},TASK,SRC,python_span,cpp_span)
