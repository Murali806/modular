"""Shared operation journey: source evidence and an offline animation shell."""
from __future__ import annotations

import hashlib
import html
import re
import subprocess

from build import cpp_span, python_span

JIT = "GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp"
DOC = "Mojo/docs/compiler/MojoCompilerWalkthrough.md"
TARGETS = {
    "package_load": ("max/python/max/graph/graph.py", "KernelLibrary.load_paths", "python", 0, "Import source or precompiled Mojo packages"),
    "custom_api": ("max/python/max/graph/ops/custom.py", "custom", "python", 0, "Create and immediately verify a custom graph operation"),
    "mojo_package": (DOC, "## Mojo Packages and Precompiled Files", "section", 0, "Mojo package bytecode and deferred specialization"),
    "artifact_cache": ("GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp", "MOFContext::compileToBinary(void *source,", "cpp", 0, "Native graph-artifact cache hit and miss"),
    "kernel_cache": (JIT, "checkCache(const CompilationState &state,", "cpp", 1, "Per-generator compiled-kernel cache lookup"),
    "kernel_misses": (JIT, "handleCacheMisses(CompilationState &state,", "cpp", 1, "Elaborate and compile cache-miss routines"),
    "kernel_hits": (JIT, "void handleCacheHits(CompilationState &state,", "cpp", 0, "Load compatible compiled-kernel archives"),
    "runtime_calls": ("GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp", "void LowerCompiledRoutinesPass::runOnOperation()", "cpp", 0, "Lower compiled loads/calls to runtime operations"),
    "jit_pipeline": ("GraphCompiler/lib/GraphCompiler.cpp", "void M::MO::buildJitPipeline(", "cpp", 0, "Finalize runtime chains and outline callables"),
}


def add_sources(refs, files, repo, task, revision):
    targets = dict(TARGETS)
    if "entry" not in refs:
        targets["entry"] = ("max/examples/custom_ops/addition.py", "", "file", 0, "Complete original addition example")
        targets["registration"] = ("max/examples/custom_ops/kernels/add_one.mojo", "AddOne", "mojo", 0, "Original AddOne implementation")
        targets["math"] = targets["registration"]
    for key, (file, symbol, language, prefix, label) in targets.items():
        text = (repo / file).read_text()
        if language == "python":
            start, end = python_span(text, symbol)
        elif language == "cpp":
            start, end = cpp_span(text, symbol, prefix)
        elif language == "section":
            offset = text.index(symbol)
            start = text[:offset].count("\n") + 1
            match = re.search(r"\n## ", text[offset + len(symbol):])
            end = text[:offset + len(symbol) + match.start()].count("\n") + 1 if match else len(text.splitlines())
        elif language == "mojo":
            lines = text.splitlines()
            index = next(i for i, line in enumerate(lines) if line.startswith("struct " + symbol + ":"))
            start = index + 1
            while start > 1 and lines[start - 2].startswith("@"):
                start -= 1
            end = next((i for i in range(index + 1, len(lines)) if re.match(r"^(?:@|struct |def |from |import )", lines[i])), len(lines))
            while end > start and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
                end -= 1
        else:
            start, end = 1, len(text.splitlines())
        raw = text.encode()
        if raw != subprocess.check_output(["git", "show", f"{revision}:{file}"], cwd=repo):
            raise ValueError(f"Journey source differs from pinned revision: {file}")
        snapshot = task / "sources" / file
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
        files[file] = dict(path=file, text=text, sha256=hashlib.sha256(raw).hexdigest())
        code = "".join(text.splitlines(keepends=True)[start - 1:end])
        alias = "C++" if language == "cpp" else "DOC" if language == "section" else "MOJO" if language == "mojo" else "PY"
        refs[key] = dict(id=key, file=file, start=start, end=end, line=start, symbol=symbol, label=label, language=language, alias=alias, sha256=hashlib.sha256(code.encode()).hexdigest())


def lesson():
    return dict(id="journey", number="01", title="Follow one operation all the way through", detail="Select an operation or its loads/stores. Follow its source ancestry, changing representations, cache route, compiled code and two calls to the retained model.", evidence="SOURCE lifecycle · schematic transformations / calls", refs=["package_load", "artifact_cache", "kernel_cache", "runtime_calls"], ascii="Python builder + Mojo package → verified G1\nG1 retained → compiler clone C1 → artifact lookup\nmiss: MO → MAP → MOGG → MGP → kernel lookup\nkernel miss: specialize Mojo → LLVM → target code\ncompiled routines → runtime callables → CM1 / MEF bytes\nartifact hit: reuse CM1; kernel hit: reuse code, prepare graph\nS1.init(CM1) → M1 → execute(B1) → O1 ready\nSame S1 / M1 / code → execute(B2) → O2 ready")


def scene():
    def box(color, x, y, w, h, title, subtitle):
        return f'<g class="owner" data-component="{color}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13"/><text class="owner-title" x="{x+14}" y="{y+25}">{title}</text><text class="owner-subtitle" x="{x+14}" y="{y+44}">{subtitle}</text></g>'
    body = box("python",24,72,260,349,"CALLER / ORIGINAL SOURCE","Python builder · G1 retained")
    body += box("mlir",310,72,536,349,"COMPILER · SAME HOST PROCESS","C1 changes · implementation specializes")
    body += box("backend",872,72,304,349,"TARGET CODE / ARTIFACT","host routines + GPU offload where used")
    body += box("engine",24,442,644,171,"HOST RUNTIME / CALLER HANDLES","S1 · M1 · fresh inputs / call references")
    body += box("hardware",694,442,482,171,"EXECUTION TARGET","CPU / GPU · scheduling and timing schematic")
    body += '<g class="journey-edges"><path d="M285 305H309"/><path d="M558 255H582"/><path d="M846 306H871"/><path d="M668 546H692"/></g>'
    return '<svg viewBox="0 0 1200 635" role="group" aria-label="Operation ancestry, compiler representations and repeated execution" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="journey-grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#dfe7ed" stroke-width=".7"/></pattern></defs><rect class="canvas-backing" width="1200" height="635"/><rect width="1200" height="635" fill="url(#journey-grid)"/><text class="scene-label" x="25" y="31">FOLLOW SOURCE ANCESTRY · THEN FOLLOW RUNTIME VALUES</text>'+body+'<g data-actors></g></svg>'


def panel(zoom="runtime"):
    l = lesson()
    stages = [("source", "Package"), ("graph", "Graph / plan"), ("mojo", "Specialize"), ("target", "Target code"), ("artifact", "Artifact / init"), ("runtime", "Execute twice")]
    steps = ''.join(f'<button data-stage="{key}" aria-label="Jump to {label}">{i+1} · {label}</button>' for i,(key,label) in enumerate(stages))
    return f'''<section class="journey lesson" id="journey" data-journey aria-labelledby="heading-journey"><div class="experiment">
<div class="experiment-top"><button class="lesson-badge" data-card="journey" aria-label="Explain 01: operation journey">01</button><h2 id="heading-journey">{l['title']}</h2><span>SOURCE route · schematic IR</span></div>
<div class="journey-toolbar" data-interactive hidden><label>Follow <select data-follow><option value="operation">Source operation</option><option value="load">Its input loads</option><option value="store">Its output stores</option><option value="plan">Graph / readiness plan</option></select></label><label>Cache route <select data-cache><option value="cold">Cold preparation</option><option value="kernel-hit">Compiled kernel hit</option><option value="artifact-hit">Compiled graph artifact hit</option></select></label><span class="journey-clock" data-clock>PREPARATION · runtime calls 0</span></div>
<nav class="journey-map" aria-label="Operation lifecycle" data-interactive hidden>{steps}</nav>
<div class="visual" tabindex="0" aria-label="Scroll inside the operation journey on narrow screens">{scene()}</div>
<div class="playback" data-interactive hidden><button class="play" data-action="play" aria-pressed="false">▶ Play</button><button data-action="back">Back</button><button data-action="step">Step</button><button data-action="nextChange">Next change</button><button data-action="reset">↺ Reset</button><label>Speed <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input data-seek type="range" min="0" max="1" value="0" aria-label="Operation journey step"><output data-progress>Ready</output></div>
<div class="state-strip" data-state-strip></div><div class="event" role="status" aria-live="polite"><strong data-event-title>Follow the source operation</strong><span data-event-detail>{l['detail']}</span></div>
<div class="journey-diff"><div><span>Before · schematic representation</span><pre data-before>Python builder + Mojo implementation package</pre></div><div><span>After · schematic representation</span><pre data-after>Verified graph → compiled artifact → initialized model</pre></div></div>
<details class="journey-source"><summary data-source-summary>Selected source ancestry · complete definition available</summary><pre data-source-preview></pre><div data-source-link></div></details>
<p class="journey-zoom">Explore this mechanism in detail: <a data-zoom href="#{zoom}">kernel arithmetic / memory ↓</a></p>
<div class="event-sources" data-event-sources></div><details class="transcript"><summary>Portable flow · every step · source evidence</summary><pre>{html.escape(l['ascii'])}</pre><ol data-transcript></ol></details>
</div></section>'''
