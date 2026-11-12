"""Build four offline, source-pinned MAX custom-op animation readers."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from build import python_span
from compiler_workbench import COLORS, COMPONENTS, SRC, TASK, REPO, owner, slots, wire
from cells import REVISION
from compiler_journey import add_sources, lesson as journey_lesson, panel as journey_panel
from compiler_toolchain import add_toolchain_sources, render_toolchain_text, portable_frames, transcript_html
from compiler_addition_calltrees import SOURCES as CALLTREE_SOURCES
from compiler_attention_calltrees import SOURCES as ATTENTION_SOURCES
from compiler_capi_sources import add_capi_sources

EXAMPLES = [
    dict(id="addition", number="01", level="Simple", title="Add one. Follow every boundary.",
         short="Add one to a tensor", filename="addition.py", kernel="add_one.mojo",
         task="addition",
         page="max_example_01_add_one_compilation_and_execution.html", expression="Y = X + 1",
         subtitle="One custom op connects Python, a registered Mojo body and native execution.",
         tiny="Tiny view: X [2,2]; original: X [5,10] float32",
         notes="The script prints result and expected values; it does not assert numerical equality. Explicit output types bypass its unimplemented optional shape function.",
         registration="AddOne", math="AddOne", dispatch="AddOne", extra=[("shape_function", "add_one_shape", "Optional shape function")]),
    dict(id="epilogue", number="02", level="Medium", title="Keep the epilogue beside the matmul.",
         short="Matmul + fusible epilogue", filename="elementwise_addition.py", kernel="add_one.mojo",
         task="elementwise_addition",
         page="max_example_02_matmul_fusible_epilogue.html", expression="Y = (A @ B + 1) × 2",
         subtitle="Three mathematical operations expose an eligible fusion opportunity.",
         tiny="Tiny view: A/B [2,2]; original: A [5,8], B [8,10]",
         notes="The source documents epilogue fusion eligibility. Fused/separate routes here are numerical teaching comparisons, not captured compiler dumps or real configuration flags. The original script asserts allclose.",
         registration="AddOneElementwise", math="AddOneElementwise", dispatch="AddOneElementwise", extra=[]),
    dict(id="matmul", number="03", level="Medium-hard", title="Choose a kernel. Coordinate its tiles.",
         short="Custom tiled matrix multiplication", filename="matrix_multiplication.py", kernel="matrix_multiplication.mojo",
         task="matrix_multiplication",
         page="max_example_03_tiled_matmul_compiler_and_memory.html", expression="C = A @ B; algorithm is a compile-time parameter",
         subtitle="Follow target specialization, global/shared storage and two different readiness gates.",
         tiny="Tiny view: A [2,4], B [4,2], BK=2; original: M=K=N=256",
         notes="Original tiled BM=BN=BK is 32 on non-AMD GPUs and 16 on AMD. Main tests several GPU algorithms and a later 128×256×384 case; CPU main uses naive. Each helper call compiles and initializes its own model while reusing the session. The final expected assignment is a@b, overwriting the earlier fp64 reference.",
         registration="MatrixMultiplication", math="tiled_matrix_multiplication", dispatch="MatrixMultiplication",
         extra=[("math_naive_cpu", "naive_matrix_multiplication_cpu", "Naive CPU matmul"), ("math_naive_gpu", "naive_matrix_multiplication", "Naive GPU matmul")]),
    dict(id="attention", number="04", level="Hard", title="Attention, one key tile at a time.",
         short="Fused attention / online softmax", filename="fused_attention.py", kernel="fused_attention.mojo",
         task="fused_attention",
         page="max_example_04_fused_attention_online_softmax.html", expression="O = softmax(Q Kᵀ) V · unscaled, rank 2",
         subtitle="See why a larger running maximum requires rescaling the previous output.",
         tiny="Tiny view: one query, four keys, two features; original CPU N=D=8 / GPU N=D=32",
         notes="This example is unscaled rank-2 attention without mask/dropout/heads. GPU BN=16, BD=8 but its key tile BN_1=8. CPU BN=4, BD=4. The original script prints output Buffers and has no explicit CPU copy or parity assertion. The tiny online-softmax comparison is independently checked in JavaScript.",
         registration="FusedAttention", math="fused_attention_kernel", dispatch="fused_attention_gpu",
         extra=[("math_cpu", "fused_attention_cpu", "CPU online softmax")]),
]

BASE_REFS = "graph_init graph_output frontend mo_pipeline map_pipeline mogg_pipeline mgp_pipeline gc_pipeline compile_async artifact serialize init_all execute native_execute driver_execute driver_results ready_counts mojo_semantics mojo_elaboration mojo_llvm llvm_translation backend_select host_object nvptx_object amdgpu_backend".split()


def mojo_span(text, symbol):
    """Top-level definition, including its decorators and entire indented body."""
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if re.match(r"(?:struct|def) " + re.escape(symbol) + r"(?:\b|\[)", line)]
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level Mojo symbol: {symbol}")
    start = matches[0]
    while start and lines[start-1].startswith("@"):
        start -= 1
    # A top-level comment immediately preceding the next definition belongs to
    # neither body. Trailing blank/comment lines are harmless unchanged source.
    end = next((i for i in range(matches[0]+1, len(lines)) if re.match(r"(?:@|struct |def |comptime |from |import )", lines[i])), len(lines))
    while end > start and (not lines[end-1].strip() or lines[end-1].startswith("#")):
        end -= 1
    return start+1, end


def load_example(example):
    base = json.loads((TASK/"compiler_workbench_manifest.json").read_text())
    if base["revision"] != REVISION:
        raise ValueError("Example and compiler source revisions differ")
    refs = {key:dict(base["references"][key]) for key in BASE_REFS}
    py = "max/examples/custom_ops/" + example["filename"]
    mojo = "max/examples/custom_ops/kernels/" + example["kernel"]
    text = (REPO/mojo).read_text()
    def ref(key,file,start,end,label,language="mojo",symbol="",alias=None):
        refs[key]=dict(id=key,file=file,start=start,end=end,line=start,label=label,symbol=symbol,language=language,alias=alias or ("PY" if language=="python" else "MOJO"))
    ref("entry",py,1,len((REPO/py).read_text().splitlines()),"Complete Python example","python",example["filename"])
    ref("kernel_file",mojo,1,len(text.splitlines()),"Complete Mojo kernel file",symbol=example["kernel"])
    build_file="max/examples/custom_ops/BUILD.bazel"
    build_lines=(REPO/build_file).read_text().splitlines()
    target_line=next(i for i,line in enumerate(build_lines,1) if line.strip()==f'name = "{example["task"]}",')
    ref("bazel_binary",build_file,target_line,target_line+1,"Bazel example binary target",language="text",alias="BUILD")
    macro_file="max/examples/custom_ops/custom_op_example.bzl"
    ref("bazel_test_macro",macro_file,42,50,"Generated .example-test rule",language="text",alias="BUILD")
    pixi_file="max/examples/custom_ops/pixi.toml"
    pixi_lines=(REPO/pixi_file).read_text().splitlines()
    task_line=next(i for i,line in enumerate(pixi_lines,1) if line.strip().startswith(example["task"]+" = "))
    ref("pixi_task",pixi_file,task_line,task_line,"Pixi task mapped to the Python script",language="text",alias="PIXI")
    ref("pixi_dependencies",pixi_file,32,34,"Pixi channels and runtime dependencies",language="text",alias="PIXI")
    readme_file="max/examples/custom_ops/README.md"
    ref("pixi_run_docs",readme_file,55,72,"Documented Pixi setup and run path",language="text",alias="DOC")
    ref("bazel_wrapper","bazelw",1,34,"Repository Bazel bootstrap wrapper",language="text",alias="SH")
    ref("bazel_macro","max/examples/custom_ops/custom_op_example.bzl",12,40,"Custom-op binary: source data, Mojo libraries and Python dependencies",language="text",alias="BUILD")
    ref("bazel_kernel_sources",build_file,9,12,"Raw custom-kernel .mojo sources passed as Bazel data",language="text",alias="BUILD")
    ref("bazel_py_binary","bazel/internal/modular_py_binary.bzl",53,100,"Python binary rule stages the Mojo runtime environment",language="text",alias="BUILD")
    ref("mojo_library_rule","bazel/internal/mojo_library.bzl",35,58,"Bazel Mojo library precompiles source into a .mojoc package",language="text",alias="BUILD")
    ref("mojo_runtime_toolchain","bazel/internal/mojo_test_environment.bzl",78,120,"Mojo toolchain and native runtime files collected for execution",language="text",alias="BUILD")
    ref("mojo_runtime_exports","bazel/internal/mojo_test_environment.bzl",178,198,"Mojo compiler, LLD and C++ linker settings exposed to the process",language="text",alias="BUILD")
    ref("target_info","Mojo/lib/Compiler/KGENCompiler.cpp",213,275,"Per-target triple, architecture and backend lookup",language="cpp",alias="C++")
    ref("offload_targets","Mojo/lib/Compiler/KGENCompiler.cpp",516,568,"Compile each offload for its selected target triple",language="cpp",alias="C++")
    ref("backend_registry","Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp",114,138,"TargetBackendRegistry resolves the LLVM triple",language="cpp",alias="C++")
    for key,symbol,label in [("registration",example["registration"],"Registered Mojo implementation"),("math",example["math"],"Kernel arithmetic"),("dispatch",example["dispatch"],"Target dispatch / launch")]+example["extra"]:
        start,end=mojo_span(text,symbol)
        ref(key,mojo,start,end,label,symbol=symbol)
    custom="max/python/max/graph/ops/custom.py"
    start,end=python_span((REPO/custom).read_text(),"custom")
    ref("custom_api",custom,start,end,"ops.custom graph construction","python","custom")
    files={}
    for file in sorted({r["file"] for r in refs.values()}):
        raw=(REPO/file).read_bytes()
        if raw!=subprocess.check_output(["git","show",f"{REVISION}:{file}"],cwd=REPO):
            raise ValueError(f"Example source differs from pinned revision: {file}")
        target=TASK/"sources"/file
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        files[file]=dict(path=file,text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest())
    for r in refs.values():
        code="".join(files[r["file"]]["text"].splitlines(keepends=True)[r["start"]-1:r["end"]])
        r["sha256"]=hashlib.sha256(code.encode()).hexdigest()
    add_sources(refs, files, REPO, TASK, REVISION)
    add_toolchain_sources(refs, files, REPO, TASK, REVISION)
    if example["id"] in {"addition", "attention"}:
        add_toolchain_sources(refs, files, REPO, TASK, REVISION, sources=CALLTREE_SOURCES)
        add_capi_sources(refs, files, REPO, TASK, REVISION)
    if example["id"] == "attention":
        add_toolchain_sources(refs, files, REPO, TASK, REVISION, sources=ATTENTION_SOURCES)
    return refs,files


def scene(kind,example):
    if kind=="toolchain":
        body=owner("python",24,64,226,172,"BAZEL · TARGET GRAPH","Python binary + declared deps")
        body+=owner("artifact",266,64,258,172,"BUILD OUTPUTS / RUNFILES","Launcher · source · Mojo libraries")
        body+=owner("backend",548,64,246,172,"RUNTIME TOOLCHAIN","Mojo driver · LLD · linker env")
        body+=owner("mojo",818,64,358,172,"GRAPH LOAD · CUSTOM PACKAGE","mojo precompile subprocess → P1")
        body+=wire("M250 155 H266","",0,0)
        body+=wire("M524 155 H548","",0,0)
        body+=wire("M794 155 H818","",0,0)
        body+=owner("mlir",24,264,352,240,"C1 · MAX GRAPH IR / PLAN","MO → MAP → MOGG → MGP · host passes")
        body+=owner("mojo",400,264,358,240,"KIR1 · MOJO IMPLEMENTATION","specialize → KGEN / POP → LLVM dialect")
        body+=owner("backend",782,264,394,240,"TARGET INFO / LLVM1","selected backend + LLVM TargetMachine")
        body+=wire("M1000 236 V250 H200 V264","Graph + package → native compilation",493,250)
        body+=wire("M376 380 H400","",0,0)
        body+=wire("M758 380 H782","",0,0)
        body+=owner("engine",24,534,538,209,"HOST CPU · PROCESS / NATIVE CODE","Python + MAX runtime · host routines / S1")
        body+=owner("hardware",584,534,592,209,"DEVICE OFFLOAD · TARGET IMAGE","NVIDIA PTX → cubin  |  AMD object → ELF")
        body+=wire("M850 504 V520 H294 V534","host control / native routines",421,522)
        body+=wire("M1092 504 V534","",0,0)
        body+=owner("runtime",24,775,1152,160,"ARTIFACT → INITIALIZATION → EXECUTION","CM1 prepared bytes · M1 loaded handles · B1 input / O1 result")
        body+=wire("M1088 743 V760 H176 V775","compiled resources + runtime plan",660,763)
        body+=wire("M294 743 V775","",0,0)
    elif kind=="build":
        body=owner("python",24,64,280,405,"PYTHON BUILDER","one host application process")+owner("mlir",330,64,496,405,"G1 · SYMBOLIC GRAPH","shape / dtype / device · no arithmetic execution")+owner("mojo",856,64,320,405,"REGISTERED MOJO","custom_extensions = kernels/")
        body+=wire("M304 255 H330","record",304,240)+wire("M856 257 H827","resolve",828,241)
    elif kind=="compile":
        body=owner("python",24,64,250,405,"CALLER GRAPH","G1 stays with its owner")+owner("mlir",304,64,420,405,"COMPILER WORKBENCH","in-memory representations / native passes")+owner("backend",750,64,230,405,"TARGET TOOLS","host compilation / target code")+owner("engine",1006,64,170,405,"MODEL DOCK","compiled bytes → init")
        body+=wire("M274 251 H304","clone",276,237)+wire("M724 249 H750","lower",725,234)+wire("M980 250 H1006","prepare",981,235)
    elif kind=="commands":
        body=owner("python",24,64,276,405,"REPOSITORY ROOT","start in the modular/ checkout")+owner("backend",327,64,386,405,"BAZEL · BUILD + TEST","from repository root · ./bazelw")+owner("engine",740,64,436,405,"PIXI · DIRECT SCRIPT","from max/examples/custom_ops/")
        body+=wire("M300 251 H327","binary / test target",301,235)+wire("M300 337 H740","locked env / task",491,325)
    else:
        body=owner("python",24,64,292,490,"HOST CALLER","NumPy arrays / Buffer references")+owner("runtime",344,64,276,490,"NATIVE MAX RUNTIME","M1 / S1 / compiled code stay resident")+owner("hardware",648,64,528,490,"EXECUTION TARGET","CPU or accelerator · motion is schematic")
        body+='<rect class="memory-rack" x="667" y="132" width="166" height="324" rx="8"/><text class="static-label" x="679" y="150">INPUT STORAGE</text><rect class="kernel-bench" x="845" y="132" width="312" height="310" rx="8"/><text class="static-label" x="858" y="155" data-kernel-label>Kernel arithmetic</text><text class="static-label" x="49" y="426">Host result / explicit read</text><text class="static-label" x="856" y="476">Output storage</text>'
        body+=slots("host-output",49,444,4,54,8)+slots("target-output",852,491,4,64,10)
        body+=wire("M316 205 H330 V574 H653 V254 H667","explicit .to(device)",398,592,"upload","data")+wire("M620 290 H648","execute / native boundary",523,275,"submit")+wire("M1160 555 V612 H294 V499","readiness / explicit host copy where present",489,605,"result","data")
    if kind=="run" and example["id"]=="attention":
        body+=owner("artifact",24,650,548,255,"COMPARISON · FULL SCORE MATRIX","hypothetical full-score route · not the provided kernel")
        body+=owner("mojo",600,650,576,255,"THIS KERNEL · REUSE SCORE TILES","local scores expire; running m / l / O survive")
        body+='<text class="static-label" x="49" y="710">Tiny row: four score slots</text><text class="static-label" x="627" y="710">Tiny row: two reusable score slots</text>'
        body+=slots("full-scores",49,729,4,108,20)+slots("online-scores",627,729,2,108,20)
        body+='<text class="static-label" x="49" y="872">Score elements only · excludes Q/K/V/O and E/scratch</text><text class="static-label" x="627" y="872">Running state is retained across key tiles</text>'
    elif kind=="run" and example["id"]=="epilogue":
        body+=owner("artifact",24,650,1152,255,"INTERMEDIATE LIFETIMES / LOGICAL TRAFFIC","tiny 2×2 comparison · eligibility is SOURCE; fusion / allocations are schematic")
        body+='<g data-temporary-edges>'+''.join(wire(path,"",0,0) for path in ["M221 748H240","M407 748H426","M557 748H576","M744 748H762"])+ '</g>'
        body+='<text class="static-label" x="49" y="874">Counters exclude input A/B loads, cache lines and measured hardware traffic</text>'
    height=930 if kind=="run" and example["id"] in {"epilogue","attention"} else 960 if kind=="toolchain" else 630 if kind=="run" else 520
    return f'<svg viewBox="0 0 1200 {height}" role="img" aria-label="{html.escape(example["short"])} · {kind}" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid-{kind}" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#dfe7ed" stroke-width=".7"/></pattern><marker id="arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#596e79"/></marker></defs><rect class="canvas-backing" width="1200" height="{height}"/><rect width="1200" height="{height}" fill="url(#grid-{kind})"/><text class="scene-label" x="25" y="31">{dict(toolchain="FOLLOW BUILD ACTIONS → RUNTIME TARGET CODE",build="READ SOURCE / RECORD CONTRACT",compile="TRANSFORM REPRESENTATIONS / INITIALIZE",run="FOLLOW VALUES / COORDINATE WORK",commands="CHOOSE THE REPOSITORY-RUN PATH")[kind]}</text>{body.replace("url(#arrow)",f"url(#arrow-{kind})")}<g data-actors></g></svg>'


def progression_cards(current=None):
    return '<div class="example-grid">'+"".join(f'<a class="example-link" href="{e["page"]}"'+(' aria-current="page"' if e["id"]==current else '')+f'><span>{e["number"]} · {e["level"]}</span><strong>{e["short"]}</strong><code>{html.escape(e["expression"])}</code><small>{e["filename"]}</small></a>' for e in EXAMPLES)+'</div>'


def build_compiler_examples(example_ids=None):
    manifests=[]
    for example in EXAMPLES:
        if example_ids is not None and example['id'] not in example_ids:
            continue
        refs,files=load_example(example)
        qna_html,qna_meta,call_stack_html=render_toolchain_text(refs,example,TASK,SRC)
        toolchain_frames=portable_frames(example,SRC)
        def link(key):
            r=refs[key]
            return f'<a data-ref="{key}" href="../sources/{r["file"]}#L{r["line"]}">{html.escape(r["label"])} <small>[{r["alias"]}:L{r["line"]}]</small></a>'
        lessons=[dict(id="build",number="01",title="Record this example's graph",detail="Follow the unchanged Python source, typed graph inputs and its Mojo registration.",ascii="Python example + Mojo package → registered implementation\nGraph imports custom_extensions before invoking builder\nTensorType + ops.custom(name, out_types) → Graph.output\nVerified G1; numerical execution has not happened",refs=["entry","registration"],evidence="SOURCE · exact example / graph API"),dict(id="compile",number="02",title="Compile it into an executable model",detail="This focused view expands the cold route. Use the operation journey in 02 for graph-artifact and compiled-kernel cache comparisons.",ascii="S1.compile(G1) → clone C1 → MO / MAP / MOGG / MGP\ndevice-graph finalization → emitted/custom Mojo → kernel JIT\nMojo specialization → LLVM dialect → LLVM IR → target code\nfinal runtime-call lowering → prepared MEF byte buffer\nload modules / resolve callables → initialized M1\nThese examples need no weight registry",refs=["entry","gc_pipeline","registration"],evidence="SOURCE pipeline · schematic IR / backend outputs"),dict(id="run",number="03",title="Watch the kernel's values and decisions",detail=example["tiny"]+". Arithmetic is a small deterministic illustration of the original operation.",ascii="M1 + Buffer inputs → native execution → kernel arithmetic\nRuntime readiness → result Buffer\n"+("print(output); no explicit host copy in this source" if example["id"]=="attention" else "explicit result.to(CPU()) → to_numpy() → print/check")+"\nSame resident code/session, separate per-call data",refs=["entry","math"],evidence="SOURCE kernel logic · browser math illustration · BOUNDARY device timing")]
        lessons[0]["number"]="03"
        lessons[1]["number"]="04"
        lessons[2]["number"]="05"
        toolchain_lesson=dict(id="toolchain",number="01",
            title="From Bazel's build graph to a selected device backend",
            detail="Follow the build/run boundary, each MAX dialect conversion, concrete Mojo specialization, target selection and initialized execution. Text retains the complete previous answers 1–7 and expands their numbered details; each animation step links to its explanation.",
            ascii=(f"1 · ./bazelw build //max/examples/custom_ops:{example['task']}\n"
                   "    Python launcher + runfiles; kernels/*.mojo is data\n"
                   "    Mojo library dependencies → separate .mojoc packages\n"
                   f"2 · ./bazelw run //max/examples/custom_ops:{example['task']}\n"
                   "    Graph(custom_extensions=kernels/) → mojo precompile subprocess\n"
                   "    private staging .mojoc → atomic publication → package P1\n"
                   "    Python entry point stays separate; G1 records shape/dtype/device\n"
                   "3 · InferenceSession.compile(G1); caller retains G1\n"
                   "    C1 / MO: Modular Operations · typed tensor graph\n"
                   "      MOToMAPPass + MojoRegistry → access/kernel structure\n"
                   "    C1 / MAP: Modular Access Primitives · load/compute/store + fusion\n"
                   "      MAPToMOGGPass + MAPToMOGGState → generator/routine calls\n"
                   "    C1 / MOGG: Modular Operation Generator Graph\n"
                   "      MOGGToMGPPass + MOGraphLowering → init/execute resources\n"
                   "    C1 / MGP: Modular Glue Primitives · buffers/dependencies/state\n"
                   "    Device-graph finalization: audit only when getIsDeviceGraph()\n"
                   "4 · EmitMojo → separate KIR1; C1 retains the MGP plan\n"
                   "    kernel JIT miss → KGENCompiler.runElaborationPipeline\n"
                   "    generators + target/dtype/parameters → concrete KGEN / POP\n"
                   "    ObjectCompiler + buildLowerToLLVMPipeline → LLVM dialect (MLIR)\n"
                   "    mlir::translateModuleToLLVMIR → LLVM1 / llvm::Module\n"
                   "5 · TargetInfo triple/arch/features → TargetBackendRegistry lookup\n"
                   "    LLVM TargetRegistry → LLVM TargetMachine\n"
                   "    CPU: HostBackend → host object → linked native code\n"
                   "    NVIDIA: nvptx64-nvidia-cuda / sm_90a → PTX\n"
                   "      CUDA CompilationDevice.compileFunction → same-SM cubin\n"
                   "    AMD: amdgcn-amd-amdhsa / gfx942 + required device bitcode\n"
                   "      AMDGPU object → device ELF link → hsaco\n"
                   "6 · Host Python/MAX/driver stays on CPU; device kernels have their own target\n"
                   "    LowerCompiledRoutines → ModuleLoad / ExecuteGeneric\n"
                   "    prepared CM1 / MEF bytes → session.init → executable M1\n"
                   "    M1 + numerical Buffers → native execution → result readiness\n"
                   "7 · Native MLIR/LLVM APIs run compiler passes in the process\n"
                   "    No standalone mlir-opt command is part of these example targets\n"
                   "    IR/dump names are teaching views; GPU architectures are source-test examples"),
            refs=["bazel_wrapper","bazel_binary","bazel_macro","bazel_kernel_sources","bazel_py_binary","mojo_library_rule","mojo_runtime_toolchain","mojo_runtime_exports","package_load","entry","compile_async","gc_pipeline","mo_pipeline","map_pipeline","mogg_pipeline","mgp_pipeline","mojo_semantics","mojo_elaboration","mojo_llvm","llvm_translation","target_info","offload_targets","backend_registry","backend_select","host_object","nvptx_object","amdgpu_backend"],
            evidence="SOURCE · Bazel rules / MAX and Mojo compiler / target backends")
        toolchain_lesson['refs']=list(dict.fromkeys([
            *toolchain_lesson['refs'],
            *[key for frame in toolchain_frames for key in frame['refs']],
        ]))
        lessons.insert(0,toolchain_lesson)
        bazel_target=f'//max/examples/custom_ops:{example["task"]}'
        commands_ascii=("From the modular/ repository root:\n"
                        f"./bazelw build {bazel_target}\n"
                        f"./bazelw test {bazel_target}.example-test\n\n"
                        "Direct script route (Pixi CLI required; setup once):\n"
                        "cd max/examples/custom_ops\n"
                        f"pixi install\npixi run {example['task']}")
        lessons.append(dict(id="commands",number="00",title="Set up, build and run this exact example",
                            detail="The steps show the repository-root Bazel build and generated example-test, plus the separate Pixi setup and task for running the Python script directly.",
                            ascii=commands_ascii,
                            refs=["bazel_wrapper","bazel_binary","bazel_test_macro","pixi_task","pixi_dependencies","pixi_run_docs"],
                            evidence="SOURCE · repository build rule / locked Pixi environment"))
        if example["id"]=="epilogue":
            next(l for l in lessons if l["id"]=="run")["ascii"]+="\nFused illustration: intermediate reads / writes 0 / 0; final writes 4\nSeparate: TMP1 → TMP2 → Y; intermediate reads / writes 8 / 8\nBoth produce [10, 10; 22, 18]; intermediate lifetimes then end"
        elif example["id"]=="attention":
            next(l for l in lessons if l["id"]=="run")["ascii"]+="\nFull-score comparison: N×N score elements\nProvided tiled route: queryTile×keyTile local score elements\nTiny row: 4 retained scores versus 2 reused slots\nRetain m / l / normalized O; release local score and exponent tiles"
        journey=journey_lesson();journey["number"]="02"
        lessons = [journey, *lessons]
        sections=[]
        panel_order=[lessons[5],lessons[1],lessons[0],lessons[2],lessons[3],lessons[4]]
        for l in panel_order:
            if l["id"] == "journey":
                sections.append(journey_panel(zoom="run",number="02"))
                continue
            transcript=transcript_html(toolchain_frames,refs) if l['id']=='toolchain' else ''
            reading_link='<button type="button" class="qa-step-link" data-read-current data-interactive hidden>Read this step</button>' if l['id']=='toolchain' else ''
            content=f'''<p class="intro">{html.escape(l['detail'])}</p><div class="visual" tabindex="0" aria-label="{html.escape(l['title'])}; scroll inside on narrow screens">{scene(l['id'],example)}</div><div class="playback" data-interactive hidden><button class="play" data-action="play" aria-pressed="false">▶ Play</button><button data-action="back">Back</button><button data-action="step">Step</button><button data-action="reset">↺ Reset</button><label>Speed <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input data-seek type="range" min="0" max="1" value="0" aria-label="{html.escape(l['title'])} step"><output data-progress>Ready</output><button type="button" class="fullscreen-toggle" data-fullscreen-toggle aria-label="Open animation in fullscreen">⛶ Fullscreen</button></div><div class="state-strip" data-state-strip></div><div class="event" role="status" aria-live="polite"><strong data-event-title>Ready to explore</strong><span data-event-detail>{html.escape(l['detail'])}</span>{reading_link}</div><div class="ir-diff"><div><span>Original source · selected current definition</span><pre data-code-preview></pre></div><div><span>Schematic state / numerical work · not a generated dump</span><pre data-state-preview></pre></div></div><div class="event-sources" data-event-sources>{' · '.join(link(r) for r in l['refs'])}</div><details class="transcript"><summary>Portable flow · step-by-step transcript · evidence</summary><pre>{html.escape(l['ascii'])}</pre><ol data-transcript>{transcript}</ol></details>'''
            if l['id']=='toolchain':
                call_stack_tab = ('<button type="button" role="tab" id="toolchain-callstack-tab" '
                                  'data-mode="callstack" aria-controls="toolchain-callstack" '
                                  'aria-selected="false" tabindex="-1">Call Stack Tree</button>'
                                  if call_stack_html else '')
                call_stack_panel = ('<div class="toolchain-reading call-stack-reading" '
                                    'id="toolchain-callstack" data-mode-panel="callstack" '
                                    'role="tabpanel" aria-labelledby="toolchain-callstack-tab">'
                                    + call_stack_html + '</div>' if call_stack_html else '')
                text_view = 'text' in qna_meta.get('reading_views', ['animation', 'text'])
                text_tab = ('<button type="button" role="tab" id="toolchain-text-tab" data-mode="text" '
                      'aria-controls="toolchain-text" aria-selected="false" tabindex="-1">'
                      'Text · <span class="qna-color qna-color--compile">answers 1–7</span></button>'
                      if text_view else '')
                text_panel = ('<div class="toolchain-reading" id="toolchain-text" data-mode-panel="text" '
                         'role="tabpanel" aria-labelledby="toolchain-text-tab">' + qna_html + '</div>'
                         if text_view else '')
                tabs=('<div class="view-switch" role="tablist" aria-label="Lesson 01 reading mode" '
                      'data-interactive hidden><button type="button" role="tab" '
                      'id="toolchain-animation-tab" data-mode="animation" '
                      'aria-controls="toolchain-animation" aria-selected="true">Animation</button>'
                      + text_tab + call_stack_tab + '</div>')
                content=(tabs
                         + '<div id="toolchain-animation" data-mode-panel="animation" role="tabpanel" '
                         'aria-labelledby="toolchain-animation-tab">' + content + '</div>'
                         + text_panel
                         + call_stack_panel)
            else:
                content='<div class="animation-fullscreen-panel" data-mode-panel="animation">'+content+'</div>'
            sections.append(f'''<section class="lesson" id="{l['id']}" data-lesson="{l['id']}" aria-labelledby="heading-{l['id']}"><div class="experiment"><div class="experiment-top"><button class="lesson-badge" data-card="{l['id']}" aria-label="Explain {l['number']}: {html.escape(l['title'])}">{l['number']}</button><h2 id="heading-{l['id']}">{html.escape(l['title'])}</h2><span>{html.escape(l['evidence'])}</span><button type="button" class="fullscreen-toggle" data-fullscreen-toggle aria-label="Open animation in fullscreen">⛶ Fullscreen</button></div>{content}</div></section>''')
            if l["id"]=="commands":
                bazel_target=f'//max/examples/custom_ops:{example["task"]}'
                command_panel=f'''<div class="command-quickstart"><div class="command-route"><h3>Bazel · build, then run the test</h3><p>Run from the repository root, the directory containing <code>bazelw</code>. The wrapper bootstraps its pinned Bazelisk when needed. The test target is generated by <code>custom_op_example_py_binary</code>; it runs the binary and fails if the process exits non-zero. Numerical assertions come from the Python example. The preceding build is optional because <code>bazelw test</code> builds its dependencies.</p><pre><code>./bazelw build {bazel_target}\n./bazelw test {bazel_target}.example-test</code></pre><p class="command-note">The generated Bazel test carries the <code>gpu</code> tag. The Python examples select CPU when no accelerator is detected; otherwise they request an accelerator.</p></div><div class="command-route"><h3>Pixi · set up the environment, then run the script</h3><p>Install the <a href="https://pixi.sh/latest/">Pixi CLI</a> first. Run setup once in the example directory; <code>pixi install</code> resolves the declared Python/MAX dependencies from the project environment and lockfile.</p><pre><code>cd max/examples/custom_ops\npixi install\npixi run {example["task"]}</code></pre><p class="command-note"><code>pixi run {example["task"]}</code> invokes <code>python {example["filename"]}</code>, as declared in <code>pixi.toml</code>. This route runs the example script directly; it does not invoke the Bazel test wrapper.</p></div>'''
                if example["id"]=="attention":
                    command_panel+='''<p class="command-note attention-command-note">This Bazel test is marked <code>manual</code> and disabled in CI in this checkout. Naming its target directly runs the requested test.</p>'''
                command_panel+='</div>'
                sections[-1]=sections[-1].replace('<div class="visual"',command_panel+'<div class="visual"',1)
        legend=[]
        for key,title,kind,detail,sources in COMPONENTS:
            if key=="mojo":detail="The original example's registered struct supplies typed native arithmetic; target/algorithm/tile parameters specialize it. This is not automatic translation of arbitrary Python to Mojo.";sources=["registration","math"]
            if key=="engine":detail="S1 owns compilation/runtime context. Compiled artifact and initialized M1 are separate objects. These four input-tensor examples call init(compiled) without a weight registry.";sources=["entry","init_all"]
            if key=="sync":
                detail="Runtime result readiness coordinates obtaining output Buffers. It does not establish a universal device-wide barrier or automatically copy results to CPU."
                sources=["driver_results","ready_counts"]
                if example["id"]=="matmul":
                    detail="The GPU tiled kernel waits for asynchronous copies and uses block barriers before consuming/reusing shared storage. These kernel gates have a different scope from native runtime result readiness."
                    sources=["math","driver_results"]
                elif example["id"]=="attention":
                    detail="The kernel's m/l/O are arithmetic state. Native result readiness coordinates obtaining output Buffers; no mutex or semaphore is part of the shown online-softmax recurrence."
                    sources=["math","driver_results"]
            if key=="hardware":detail="The original example selects a CPU or accelerator. Compiled target code is separate from execution on that physical device. Chip, storage and tile movement here are schematic.";sources=["entry"]
            sources=[s for s in sources if s in refs]
            legend.append(f'<div class="legend-row" data-legend-component="{key}"><span class="swatch" style="background:{COLORS[key][0]}"></span><div><strong>{title}</strong><span class="legend-kind">{kind}</span><p>{detail}</p><div>{" · ".join(link(s) for s in sources)}</div></div></div>')
        cells=[]
        for key in ["entry","kernel_file","registration","math","gc_pipeline","bazel_binary","bazel_test_macro","bazel_macro","bazel_kernel_sources","bazel_py_binary","mojo_library_rule","mojo_runtime_toolchain","mojo_runtime_exports","target_info","offload_targets","backend_registry","pixi_task","pixi_dependencies","pixi_run_docs","bazel_wrapper"]:
            r=refs[key];lines=files[r["file"]]["text"].splitlines()
            code="\n".join(f'<span class="source-line"><span>{n}</span>{html.escape(lines[n-1])}</span>' for n in range(r["start"],r["end"]+1))
            cells.append(f'<details class="source-cell"><summary>{html.escape(r["label"])} · complete unchanged source</summary><p>{link(key)}</p><pre><code>{code}</code></pre></details>')
        controls='<label>Target illustration <select id="target-select"><option value="cuda">NVIDIA GPU</option><option value="rocm">AMD GPU</option><option value="cpu">CPU fallback</option></select></label>'
        if example["id"]=="epilogue":controls+='<label>Compare <select id="fusion-select"><option value="fused">Eligible fused epilogue</option><option value="separate">Separate intermediates</option></select></label><small>Teaching comparison; not a compiler flag.</small>'
        if example["id"]=="matmul":controls+='<label>GPU algorithm <select id="algorithm-select"><option value="tiled">tiled</option><option value="naive">naive</option></select></label><small>CPU source uses naive. Original GPU main covers more variants.</small>'
        payload=dict(revision=REVISION,example=example,references=refs,files=files,lessons=lessons,toolchain=qna_meta)
        shell=(SRC/"compiler_workbench.html").read_text()
        # Shared source/card markup keeps keyboard/touch/offline behavior equal.
        shared=shell[shell.index('<aside id="explanation-card"'):shell.index('<script id="compiler-workbench-data"')]
        colors="\n".join(f'[data-component="{k}"], [data-color="{k}"]{{--ink:{ink};--tint:{tint}}}' for k,(ink,tint) in COLORS.items())
        window_styles=(SRC/"compiler_card_windows.css").read_text()+"\n"+(SRC/"compiler_card_conversions.css").read_text() if example["id"] in {"addition","attention"} else ""
        window_scripts=("compiler_card_windows.js","compiler_card_conversions.js") if example["id"] in {"addition","attention"} else ()
        if example["id"] == "attention":
            window_styles += "\n" + (SRC/"compiler_attention_mechanisms.css").read_text()
            window_scripts += ("compiler_attention_math.js", "compiler_attention_mechanisms.js")
        replacements=dict(css=(SRC/"compiler_workbench.css").read_text()+"\n"+(SRC/"compiler_example_links.css").read_text()+"\n"+(SRC/"compiler_examples.css").read_text()+"\n"+(SRC/"compiler_journey.css").read_text()+"\n"+colors+"\n"+window_styles,js="\n".join((SRC/name).read_text() for name in ("compiler_examples_model.js","compiler_journey_model.js","compiler_workbench_visuals.js","compiler_examples_visuals.js","compiler_journey_visuals.js","compiler_journey.js",*window_scripts,"compiler_reader_ui.js","compiler_examples.js")),title=html.escape(example["short"]),number=example["number"],level=example["level"],heading=html.escape(example["title"]),subtitle=html.escape(example["subtitle"]),expression=html.escape(example["expression"]),tiny=html.escape(example["tiny"]),notes=html.escape(example["notes"]),controls=controls,sections="\n".join(sections),cards=progression_cards(example["id"]),sources="\n".join(cells),shared=shared.replace('{{legend}}',"\n".join(legend)),payload=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'),revision=REVISION,filename=example["filename"])
        template=(SRC/"compiler_examples.html").read_text()
        tokens=set(re.findall(r"\{\{([a-z-]+)\}\}",template))
        if tokens!=replacements.keys():raise ValueError(f"Example template tokens differ: {tokens^replacements.keys()}")
        page=re.sub(r"\{\{([a-z-]+)\}\}",lambda m:replacements[m[1]],template)
        (TASK/"link_htmls"/example["page"]).write_text(page)
        toolchain_manifest=dict(point1_sha256=qna_meta['point1_sha256'],sections=7,steps=len(toolchain_frames))
        if 'calltrees' in qna_meta:
            toolchain_manifest['calltrees']=qna_meta['calltrees']
        manifests.append(dict(page="link_htmls/"+example["page"],example=example,references=refs,lessons=lessons,toolchain=toolchain_manifest,files=[{k:v for k,v in f.items() if k!='text'} for f in files.values()]))
        print(f"Built {example['page']}: build-to-target flow + journey + 5 focused sections / {len(refs)} source targets")
    manifest_path = TASK/"compiler_examples_manifest.json"
    if example_ids is not None and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        replacements_by_id = {item['example']['id']: item for item in manifests}
        manifests = [replacements_by_id.get(item['example']['id'], item)
                     for item in previous['examples']]
    manifest_path.write_text(json.dumps(dict(revision=REVISION,examples=manifests),indent=2)+"\n")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', nargs='+', choices=[e['id'] for e in EXAMPLES])
    parser.add_argument('--readers-only', action='store_true', help='Build selected readers without compiler supplements')
    args = parser.parse_args()
    build_compiler_examples(args.examples)
    if not args.readers_only:
        from compiler_pipeline_detail import build_pipeline_detail
        build_pipeline_detail()
        from compiler_kernel_loader import build_kernel_loader
        build_kernel_loader()
