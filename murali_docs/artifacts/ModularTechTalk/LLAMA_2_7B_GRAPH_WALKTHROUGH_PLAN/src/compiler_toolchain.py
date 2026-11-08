"""Source evidence and portable reading view for the example toolchain lesson."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

import markdown

from build import cpp_span, python_span


SOURCES = [
    ("bazel_build_file", "max/examples/custom_ops/BUILD.bazel", "", "file", 0, "Complete custom-op BUILD graph"),
    ("bazel_aliases", "bazel/mojo_aliases.bzl", "", "file", 0, "Mojo source-package alias mappings"),
    ("bazel_max_aliases", "max/BUILD.bazel", "", "file", 0, "MAX aliases enter the BUILD graph"),
    ("bazel_layout_package", "max/kernels/src/layout/BUILD.bazel", "", "file", 0, "Layout source package"),
    ("bazel_extensibility_package", "max/kernels/src/graph_compiler/extensibility/BUILD.bazel", "", "file", 0, "Extensibility source package"),
    ("bazel_std_package", "Mojo/stdlib/std/BUILD.bazel", "", "file", 0, "Mojo standard-library source package"),
    ("bazel_mojo_toolchain", "bazel/internal/BUILD.bazel", "185:203", "lines", 0, "Source-built Mojo compiler and LLVM LLD toolchain"),
    ("bazel_binary_test", "bazel/internal/binary_test.bzl", "", "file", 0, "Example test executable and runfiles"),
    ("bazel_run_test", "bazel/internal/modular_run_binary_test.bzl", "", "file", 0, "Example test environment macro"),
    ("package_builder", "Mojo/python/mojo/paths.py", "_build_mojo_source_package", "python", 0, "Actual mojo precompile command and atomic package publication"),
    ("kernel_package_root", "max/examples/custom_ops/kernels/__init__.mojo", "", "file", 0, "Custom kernel package root"),
    ("reference_addition", "max/examples/custom_ops/addition.py", "", "file", 0, "Add-one Python example used in the QnA"),
    ("reference_add_one", "max/examples/custom_ops/kernels/add_one.mojo", "", "file", 0, "Original AddOne and AddOneElementwise definitions"),
    ("mo_dialect", "GraphCompiler/include/GraphCompiler/MODialect/MODialect.td", "", "file", 0, "MO dialect definition"),
    ("map_dialect", "GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPDialect.td", "", "file", 0, "MAP dialect and access-operation families"),
    ("mogg_dialect", "GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGDialect.td", "", "file", 0, "MOGG dialect definition"),
    ("mgp_dialect", "GraphCompiler/include/GraphCompiler/MGPDialect/Dialect/MGPDialect.td", "", "file", 0, "MGP model, context, buffers and state slots"),
    ("graph_compiler_design", "GraphCompiler/docs/GraphCompiler.md", "", "file", 0, "Dialect names and current MAP fusion design"),
    ("mo_to_map", "GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp", "struct MOToMAPPass", "cpp", 0, "MOToMAPPass walks and lowers graph operations"),
    ("map_to_mogg", "GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp", "struct MAPToMOGGPass", "cpp", 0, "MAPToMOGGPass uses MojoRegistry and MAPToMOGGState"),
    ("mogg_to_mgp", "GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp", "void MOGGToMGPPass::runOnOperation()", "cpp", 0, "MOGGToMGPPass and dependency-ordered MOGraphLowering"),
    ("device_finalization", "GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp", "class FinalizeDeviceGraphsPass", "cpp", 0, "Device-graph finalization guard and collective audit"),
    ("emit_mojo", "GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp", "147:190", "lines", 0, "EmitMojo imports packages and generates routine source"),
    ("jit_entry", "GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp", "compileMojoToBinary(ModuleOp module,", "cpp", 1, "MAX kernel JIT entry: parse, LIT lowering, compiler objects and cache routes"),
    ("kgen_elaboration", "Mojo/lib/Compiler/KGENCompiler.cpp", "ErrorOrSuccess KGENCompiler::runElaborationPipeline(", "cpp", 0, "KGENCompiler runs the elaboration pass manager"),
    ("kgen_elaborate_build", "Mojo/lib/Compiler/KGENCompiler.cpp", "void KGEN::populateElaborateModulePasses(", "cpp", 0, "Elaboration and post-elaboration pipeline builders"),
    ("kgen_post_elaboration", "Mojo/lib/Compiler/Pipeline/Pipeline.cpp", "void KGEN::buildPostElaborationPipeline(", "cpp", 0, "Concrete-IR verification, materialization and optimization"),
    ("kgen_llvm_passes", "Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp", "void KGEN::buildLowerToLLVMPipeline(", "cpp", 0, "Exact KGEN / POP / control-flow to LLVM-dialect passes"),
    ("object_lower_to_llvm", "Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp", "ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module)", "cpp", 1, "MLIR lowering, optional dumps and LLVM-module translation"),
    ("backend_registry_declaration", "Mojo/include/Mojo/Compiler/Target/TargetBackend.h", "class TargetBackendRegistry {", "cpp", 0, "Owned backend vector and static registration helper"),
    ("llvm_target_machine", "Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp", "defaultCreateTargetMachine(", "cpp", 1, "LLVM TargetRegistry and triple / CPU / features / ABI"),
    ("llvm_code_gen", "Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp", "runLlcPasses(llvm::Module &module, CompilationOptions &options,", "cpp", 1, "In-process LLVM machine-code pass manager"),
    ("nvptx_backend", "Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp", "", "file", 0, "NVIDIA backend registration, PTX and CUDA compilation"),
    ("host_backend", "Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.h", "", "file", 0, "Host backend interface and target resolution"),
    ("cross_target_example", "GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir", "", "file", 0, "Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen"),
    ("graph_stage_dumps", "GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir", "", "file", 0, "Source test for graph-stage diagnostic dumps"),
]


def add_toolchain_sources(refs, files, repo, task, revision):
    for key, file, symbol, kind, prefix, label in SOURCES:
        raw = (repo / file).read_bytes()
        if raw != subprocess.check_output(["git", "show", f"{revision}:{file}"], cwd=repo):
            raise ValueError(f"Toolchain source differs from pinned revision: {file}")
        text = raw.decode()
        if kind == "cpp":
            start, end = cpp_span(text, symbol, prefix)
        elif kind == "python":
            start, end = python_span(text, symbol)
        elif kind == "lines":
            start, end = map(int, symbol.split(":"))
        else:
            start, end = 1, len(text.splitlines())
        assert 1 <= start <= end <= len(text.splitlines()), (key, start, end)
        snapshot = task / "sources" / file
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
        files[file] = dict(path=file, text=text, sha256=hashlib.sha256(raw).hexdigest())
        span = "".join(text.splitlines(keepends=True)[start - 1:end])
        language = "cpp" if kind == "cpp" or file.endswith((".cpp", ".h")) else "python" if kind == "python" or file.endswith(".py") else "text"
        alias = "C++" if language == "cpp" else "PY" if language == "python" else "BUILD" if file.endswith((".bzl", ".bazel")) else "DOC"
        refs[key] = dict(id=key, file=file, start=start, end=end, line=start,
                         symbol=symbol if kind in {"cpp", "python"} else "", label=label,
                         language=language, alias=alias, sha256=hashlib.sha256(span.encode()).hexdigest())


def diagram(key, title, nodes, edges, height=420):
    """Offline SVG; accompanying Mermaid text remains copyable and portable."""
    body = []
    for path, label, x, y in edges:
        body.append(f'<g class="wire"><path d="{path}" marker-end="url(#qna-arrow-{key})"/></g>')
        if label:
            body.append(f'<text class="qna-edge" x="{x}" y="{y}">{html.escape(label)}</text>')
    for color, x, y, w, label, lines in nodes:
        body.append(f'<g class="qna-node" data-component="{color}"><rect x="{x}" y="{y}" width="{w}" height="90" rx="10"/>')
        body.append(f'<text class="qna-node-title" x="{x+14}" y="{y+26}">{html.escape(label)}</text>')
        body.extend(f'<text class="qna-node-line" x="{x+14}" y="{y+49+i*19}">{html.escape(line)}</text>' for i, line in enumerate(lines))
        body.append('</g>')
    return f'<figure class="qna-figure"><figcaption>{html.escape(title)}</figcaption><div class="qna-diagram-frame" tabindex="0"><svg viewBox="0 0 1060 {height}" role="img" aria-label="{html.escape(title)}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="qna-arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#596e79"/></marker></defs>{"".join(body)}</svg></div></figure>'


def reading_diagrams():
    build = diagram("build", "1.3.1 · One source filegroup, two distinct dependency routes", [
        ("python", 24, 24, 300, "Python-example macro", ["custom_op_example_py_binary", "name = addition"]),
        ("python", 416, 24, 270, ":addition · py_binary", ["launcher + addition.py", "Python dependencies / runfiles"]),
        ("mojo", 778, 24, 254, "Mojo runfiles", ["layout · extensibility · std", "precompiled library packages"]),
        ("mojo", 24, 164, 300, ":kernel_sources · filegroup", ["glob kernels/*.mojo", "the same source files"]),
        ("artifact", 416, 164, 270, ":kernels · mojo_library", ["independent .mojoc build", "used by the Mojo benchmark"]),
        ("backend", 778, 164, 254, ":benchmarks · mojo_binary", ["Mojo executable", "depends on :kernels"]),
        ("runtime", 416, 304, 270, ":addition.example-test", ["sibling binary_test target", "runs the Python launcher"]),
    ], [
        ("M324 69H416", "creates", 342, 58), ("M778 69H686", "deps", 719, 58),
        ("M324 209H369V94H416", "data", 338, 146), ("M324 209H416", "srcs", 348, 198),
        ("M686 209H778", "dep", 719, 198), ("M174 114V134H354V349H416", "creates test", 245, 125),
        ("M686 349H731V105H686", "runs", 738, 285),
    ])
    macro = diagram("macro", "1.3.2 · The local macro expands into build and test targets", [
        ("python", 24, 24, 300, "BUILD.bazel call", ["custom_op_example_py_binary", "name = addition"]),
        ("python", 416, 24, 270, "modular_py_binary", ["prepares the runtime", "creates supporting targets"]),
        ("python", 778, 24, 254, "py_binary :addition", ["Python source + data", "configured launch environment"]),
        ("mojo", 416, 164, 270, "mojo_test_environment", ["libraries · compiler · LLD", "import and linker settings"]),
        ("artifact", 778, 164, 254, "Supporting targets", ["venv · debug · pydeps", "transitive Mojo dependencies"]),
        ("runtime", 24, 304, 300, "modular_run_binary_test", ["addition.example-test", "test environment / constraints"]),
        ("runtime", 416, 304, 270, "binary_test", ["symlinks the executable", "merges its runfiles"]),
    ], [
        ("M324 69H416", "creates", 342, 58), ("M686 69H778", "creates", 699, 58),
        ("M551 114V164", "setup", 558, 145), ("M686 92H719V209H778", "helpers", 726, 146),
        ("M174 114V304", "also creates", 185, 280), ("M324 349H416", "wraps", 342, 338),
        ("M686 349H752V105H778", "runs + runfiles", 785, 289),
    ])
    flow = diagram("ir", "3 · G1 is retained; C1 changes graph representation; KIR1 is separate kernel IR", [
        ("python", 24, 24, 192, "G1 · Python graph", ["shape / dtype / device", "caller retains original"]),
        ("mlir", 280, 24, 192, "C1 · MO", ["Modular Operations", "symbolic graph operations"]),
        ("mlir", 536, 24, 192, "C1 · MAP", ["Access Primitives", "load / compute / store"]),
        ("mlir", 792, 24, 240, "C1 · MOGG", ["generator-oriented calls", "kernel / routine interfaces"]),
        ("mlir", 792, 184, 240, "C1 · MGP", ["Glue Primitives", "buffers / init / execute"]),
        ("mojo", 536, 184, 192, "KIR1 · Mojo", ["generated + imported IR", "specialization / codegen"]),
        ("artifact", 280, 184, 192, "CM1 · MEF bytes", ["runtime-call plan", "compiled-code resources"]),
        ("engine", 24, 184, 192, "M1 · initialized", ["loaded code / handles", "separate input Buffers"]),
    ], [
        ("M216 69H280", "clone", 226, 58), ("M472 69H536", "MOToMAP", 474, 58),
        ("M728 69H792", "lower", 740, 58), ("M912 114V184", "MOGGToMGP", 805, 155),
        ("M792 229H728", "emit + JIT", 730, 218), ("M536 229H472", "prepare", 475, 218),
        ("M280 229H216", "init", 239, 218),
    ], 300)
    kernel = diagram("kernel", "4 · Specialization, MLIR lowering and LLVM translation are separate actions", [
        ("artifact", 24, 24, 280, "P1 · Mojo package", ["post-parse MLIR bytecode", "declarations + bodies"]),
        ("mojo", 390, 24, 280, "KIR1 · parametric KGEN", ["generated + imported bodies", "target / type / parameters"]),
        ("mojo", 756, 24, 280, "KIR1 · concrete KGEN / POP", ["KGENCompiler + elaborator", "native functions / types"]),
        ("mojo", 756, 200, 280, "KIR1 · LLVM dialect", ["MLIR operations / context", "ObjectCompiler lowering"]),
        ("artifact", 390, 200, 280, "LLVM1 · llvm::Module", ["separate LLVM representation", "optional .ll / .bc views"]),
        ("backend", 24, 200, 280, "Compiled target bytes", ["host object / device code", "LLVM TargetMachine + backend"]),
    ], [
        ("M304 69H390", "import", 319, 58), ("M670 69H756", "elaborate", 674, 58),
        ("M896 114V200", "MLIR lowering", 803, 162),
        ("M756 245H670", "translate", 679, 234), ("M390 245H304", "codegen", 317, 234),
    ], 320)
    offload = diagram("offload", "6 · Host control code and GPU images have different targets and roles", [
        ("python", 24, 24, 310, "CPU · Python + MAX process", ["construct graph / call compiler", "supply numerical Buffers"]),
        ("mlir", 396, 24, 310, "Host routines", ["HostBackend / CPU target", "retained native control code"]),
        ("driver", 758, 24, 276, "Host device API", ["kernel handle + Buffer refs", "submit / observe readiness"]),
        ("mojo", 396, 190, 310, "Separate device offload IR", ["compileOffloads + TargetInfo", "selected kernel generators"]),
        ("backend", 24, 332, 310, "NVIDIA · device image", ["nvptx64 / sm → PTX", "CUDA compiler API → cubin"]),
        ("backend", 758, 332, 276, "AMD · device image", ["amdgcn / gfx → object", "device ELF link → hsaco"]),
        ("hardware", 396, 486, 310, "GPU · physical execution", ["loaded compatible kernel code", "device input / result Buffers"]),
    ], [
        ("M334 69H396", "call", 345, 58), ("M706 69H758", "launch", 708, 58),
        ("M551 114V190", "offload", 564, 154),
        ("M551 280V302H179V332", "NVIDIA target", 191, 319),
        ("M551 302H896V332", "AMD target", 772, 319),
        ("M179 422V460H459V486", "device code", 183, 451),
        ("M896 422V460H643V486", "device code", 800, 451),
        ("M1034 69H1047V531H706", "submit", 983, 518),
    ], 600)
    return {"build": build, "macro": macro, "ir": flow, "kernel": kernel, "offload": offload}


def render_toolchain_text(refs, example, task, src):
    originals = json.loads((src / "compiler_toolchain_originals.json").read_text())
    point1 = (src / "compiler_toolchain_point_1.txt").read_text().rstrip("\n")
    source = (src / "compiler_toolchain_qna.md").read_text()
    diagrams = reading_diagrams()
    build_flow = (
        'BUILD.bazel\n'
        '  +-- custom_op_example_py_binary(name="addition")\n'
        '  |     +-- modular_py_binary -> :addition Python launcher / runfiles\n'
        '  |     |     +-- addition.py + Python dependencies\n'
        '  |     |     +-- :kernel_sources -> raw kernels/*.mojo runtime data\n'
        '  |     |     +-- mojo_test_environment -> shared .mojoc packages / tools\n'
        '  |     +-- modular_run_binary_test -> :addition.example-test\n'
        '  |           +-- binary_test -> runs :addition with its runfiles\n'
        '  +-- :kernel_sources -> mojo_library :kernels -> kernels.mojoc\n'
        '        +-- mojo_binary :benchmarks consumes the separate :kernels library\n'
        '\nNo :addition -> :kernels dependency edge.'
    )

    def link(match):
        key = match[1]
        r = refs[key]
        return f'<a data-ref="{key}" href="../sources/{r["file"]}#L{r["line"]}">{html.escape(r["label"])} <small>[{r["alias"]}:L{r["line"]}]</small></a>'

    def render(block):
        # Markdown must not treat source notation such as load[width](idx) as a link.
        block = re.sub(r'<code>(.*?)</code>', lambda m: '`' + html.unescape(m[1]) + '`', block, flags=re.S)
        block = re.sub(r"\[SOURCE:([a-z_]+)\]", link, block)
        result = markdown.markdown(block, extensions=["tables", "fenced_code", "attr_list"])
        result = re.sub(r"<p>\{\{diagram:([a-z]+)\}\}</p>", lambda m: diagrams[m[1]], result)
        return re.sub(r"(<table>.*?</table>)", r'<div class="qna-table-scroll" tabindex="0">\1</div>', result, flags=re.S)

    point1_html = re.sub(
        r'(?m)^(\s*#{3,4}\s+(1(?:\.\d+)+)[^\n]*)$',
        lambda m: '<span class="qna-original-anchor" id="toolchain-q' + m[2].replace('.', '-') + '">' + m[1] + '</span>',
        html.escape(point1),
    )

    nav = '<nav class="qna-nav" aria-label="Compiler questions">' + ''.join(f'<a href="#toolchain-q{i}">{i} · {label}</a>' for i, label in enumerate(["Build", "Package", "MAX IR", "Mojo / JIT", "Backend", "Execution", "Tools"], 1)) + '</nav>'
    body = [nav, '<p class="qna-scope">Complete previous answer, followed by finer numbered explanations. The preserved answer uses <code>addition.py</code>; this page’s animation follows <code>' + html.escape(example["filename"]) + '</code>. Compiler stages run on the host. Evidence is pinned source; the diagrams are teaching views.</p>',
            '<section class="qna-section" id="toolchain-q1"><h3>1. Bazel build, dependencies and macros</h3><div class="qna-original"><strong>Previous answer · unchanged</strong><pre data-original-point="1">' + point1_html + '</pre></div>', diagrams["build"], diagrams["macro"],
            '<details><summary>Portable build and macro diagrams</summary><pre>' + html.escape(build_flow) + '</pre></details>',
            '<p>' + re.sub(r"\[SOURCE:([a-z_]+)\]", link, '[SOURCE:bazel_build_file] · [SOURCE:bazel_macro] · [SOURCE:bazel_aliases] · [SOURCE:bazel_binary_test]') + '</p></section>']
    companion = ["# From Bazel’s build graph to a selected device backend\n\n## 1. Previous answer — unchanged\n\n" + point1 + '\n\n### Portable build and macro diagrams\n\n```text\n' + build_flow + '\n```']
    parts = re.split(r"<!-- section:([2-7]) -->", source)
    for n in range(1, len(parts), 2):
        number, block = parts[n], parts[n+1].strip()
        heading, _, expansion = block.partition("\n")
        body.append(f'<section class="qna-section" id="toolchain-q{number}">' + render(heading) + '<div class="qna-original"><strong>Previous answer · unchanged</strong><pre data-original-point="' + number + '">' + html.escape(originals[number]) + '</pre></div>' + render(expansion) + '</section>')
        companion.append(originals[number] + "\n\n" + expansion)
    full = "\n\n".join(companion) + "\n"
    def markdown_link(match):
        r = refs[match[1]]
        return f'[{r["label"]} ({r["file"]}:{r["line"]})](sources/{r["file"]}#L{r["line"]})'
    portable = re.sub(r"\[SOURCE:([a-z_]+)\]", markdown_link, full)
    portable = re.sub(r"\{\{diagram:[a-z]+\}\}", "", portable)
    portable = re.sub(r'<code>(.*?)</code>', lambda m: '`' + html.unescape(m[1]) + '`', portable, flags=re.S)
    if example["id"] == "addition":
        (task / "COMPILER_TOOLCHAIN_QNA.md").write_text(portable)
    tools = '<div class="qna-tools" data-interactive hidden><button type="button" data-return-animation>Back to Animation</button><button type="button" data-qna-copy>Copy complete 1–7</button><a href="../COMPILER_TOOLCHAIN_QNA.md" download>Download the full text</a><output data-qna-copy-status role="status"></output></div>'
    footer = '<div class="qna-tools" data-interactive hidden><button type="button" data-return-animation>Back to Animation · current step</button><a href="#toolchain-q1">Back to question 1</a></div>'
    return tools + "".join(body) + footer, dict(originals={"1": point1, **originals}, text=portable,
                                      point1_sha256=hashlib.sha256(point1.encode()).hexdigest())


def portable_frames(example, src):
    script = "const fs=require('fs');const {ExampleModel}=require('./compiler_examples_model.js');process.stdout.write(JSON.stringify(new ExampleModel('toolchain',JSON.parse(fs.readFileSync(0,'utf8'))).frames));"
    return json.loads(subprocess.check_output(["node", "-e", script], input=json.dumps(example).encode(), cwd=src))


def transcript_html(frames, refs):
    rows = []
    for i, f in enumerate(frames):
        links = ' · '.join(f'<a data-ref="{key}" href="../sources/{refs[key]["file"]}#L{refs[key]["line"]}">{html.escape(refs[key]["label"])}</a>' for key in f["refs"])
        rows.append(f'<li><button type="button" data-index="{i}" aria-label="Seek to {f["qa"]}">{f["qa"]}</button><strong>{html.escape(f["title"])}</strong><p>{html.escape(f["detail"])}</p><code>{html.escape(f.get("representation", ""))}</code><div class="transcript-sources">{links}</div></li>')
    return "".join(rows)
