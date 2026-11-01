"""Source-derived diagrams; these are illustrations, not captured executions."""

import html


def stage_data():
    return {
        "finalize": [
            dict(label="Return tree", title="1. Flatten the returned values", before="The forward call has returned a tree of symbolic outputs.", after="The leaves become the ordered output list. _output_structure preserves their Python grouping.", field="tree.flatten(root)", cell="graph_output", note=0),
            dict(label="Values", title="2. Normalize values and include chains", before="Output leaves may need conversion to Graph Values.", after="The output list contains typed MLIR values, plus final dependency-chain values when chain inputs exist.", field="output_values → mlir_values", cell="graph_output", note=1),
            dict(label="Signature", title="3. Establish the graph signature", before="The new mo.graph was constructed with an empty result-type list.", after="The argument/result types define functionType and the kgen generator signature. Symbolic dimensions can remain symbolic.", field="functionType + signature", cell="graph_output", note=2),
            dict(label="Terminator", title="4. Identify the values that leave the graph", before="The symbolic operations exist; their graph outputs still need a terminator.", after="mo.output names the result SSA values and terminates the graph body.", field="_mo.OutputOp", cell="graph_output", note=3),
            dict(label="Names", title="5. Record engine interface names", before="The types and output values are known.", after="argument_names and result_names receive ordinal names. Python retains the semantic output-tree structure separately.", field="input0… / output0…", cell="graph_output", note=4),
            dict(label="Verify", title="6. Verify the completed graph", before="The output contract and terminator have been installed.", after="MLIR graph verification runs. A valid graph is ready for compiler processing; no inference has been launched by output().", field="self._mlir_op.verify()", cell="graph_output", note=5),
        ],
        "compiler": [
            dict(label="MO", title="Graph-level preparation", input="mo.graph containing high-level operations, types, symbols, and device information", work="Legalize RMO; simplify; run configured shape/constant work; infer layouts and validate devices.", output="Prepared graph before MO-to-MAP conversion", cell="mo_pipeline"),
            dict(label="MO → MAP", title="Enter the structured kernel representation", input="Prepared MO graph", work="Mark stable tensors, then apply MOToMAP. This is a representation conversion in gcPipelineImpl.", output="Graph containing structured MAP computations", cell="gc_pipeline"),
            dict(label="MAP", title="Fuse and outline structured computation", input="Structured operations with iteration and data-access information", work="Fuse eligible kernels; handle accumulation buffers and views; outline structured bodies.", output="Kernel bodies ready for MAP-to-MOGG lowering", cell="map_pipeline"),
            dict(label="MOGG", title="Prepare callable kernel routines", input="MOGG after MAP-to-MOGG conversion", work="Legalize rowwise kernels, conform operands, outline routines, and add tracing wrappers.", output="Kernel routines ready for conversion to MGP", cell="mogg_pipeline"),
            dict(label="MGP", title="Plan execution resources", input="Runtime-model representation after MOGGToMGP", work="Reuse buffers, plan allocations, lower primitives, handle marked device graphs, merge routines, and emit Mojo.", output="Runtime model plus emitted Mojo routines", cell="mgp_pipeline"),
            dict(label="JIT", title="Compile the generated routines", input="Emitted routines and resolved target/compiler options", work="Unless pre-JIT stopping is requested, schedule JitCompileKernels, strip declarations, and finish JIT lowering.", output="Lowered model containing compiled routines and runtime calls", cell="gc_pipeline"),
            dict(label="MEF", title="Produce the compiled artifact", input="Successfully compiled MGP module", work="convertMGPToMEF serializes it. CompiledModels::read consumes the buffer and exposes compiled model objects.", output="MEF-backed artifact; weight initialization follows", cell="convert_mef"),
        ],
        "cache": [
            dict(label="Cold path", title="Trace, compile, initialize", path=["Trace graph", "Native frontend", "Compiler passes", "Serialize MEF", "Read artifact", "Initialize weights"], text="No precompiled store result and no native MEF cache hit: compiler passes and serialization run.", cell="frontend"),
            dict(label="MEF import", title="Trace, resolve a stored artifact, initialize", path=["Trace graph", "Python MEF store", "Read MEF path", "Initialize weights"], text="In this GraphSpec route, tracing happens before the store lookup. The native path recognizes MEF and reads it directly.", cell="compile_reuse"),
            dict(label="Native cache", title="Trace, hit the native cache, initialize", path=["Trace graph", "Native frontend", "MEF cache hit", "Read artifact", "Initialize weights"], text="The frontend derives the cache key and returns its cached MEF before compiler passes and serialization.", cell="frontend"),
        ],
        "buffers": [
            dict(label="Prefill example", tokens=[17, 42, 6, 9, 21], split=3, offsets=[0, 3, 5], total=5, title="Two sequences contribute five prompt tokens", text="Sequence A contributes three tokens; B contributes two. The model uses row offsets to locate each sequence inside the packed token buffer."),
            dict(label="Decode example", tokens=[88, 31], split=1, offsets=[0, 1, 2], total=2, title="Each active sequence contributes its next token", text="Two active sequences contribute one token each. KV inputs refer to the accumulated cache state; the packed token count changes while the symbolic contract can stay the same."),
        ],
    }


def panel(kind, item):
    esc = html.escape
    if kind == "finalize":
        return f'<h4>{esc(item["title"])}</h4><div class="state-pair"><div><small>BEFORE THIS STEP</small><p>{esc(item["before"])}</p></div><div><small>AFTER THIS STEP</small><p>{esc(item["after"])}</p><code>{esc(item["field"])}</code></div></div>'
    if kind == "compiler":
        return f'<h4>{esc(item["title"])}</h4><div class="compiler-contract"><div><small>INPUT</small><p>{esc(item["input"])}</p></div><div><small>TRANSFORMATION</small><p>{esc(item["work"])}</p></div><div><small>OUTPUT</small><p>{esc(item["output"])}</p></div></div>'
    if kind == "cache":
        flow = '<span aria-hidden="true" class="flow-arrow">→</span>'.join(f'<span class="flow-node">{esc(p)}</span>' for p in item["path"])
        return f'<h4>{esc(item["title"])}</h4><div class="flow">{flow}</div><p>{esc(item["text"])}</p>'
    chips = "".join(f'<span class="token {"sequence-a" if i < item["split"] else "sequence-b"}">{value}</span>' for i, value in enumerate(item["tokens"]))
    return f'<h4>{esc(item["title"])}</h4><div class="buffer-row"><strong>tokens</strong><div class="token-strip">{chips}</div><code>int64 [{item["total"]}]</code></div><div class="buffer-row"><strong>row offsets</strong><code>{item["offsets"]}</code><code>uint32 [3]</code></div><p>{esc(item["text"])}</p>'


def explorer(kind, heading, fallback):
    items = stage_data()[kind]
    buttons = "".join(f'<button data-explorer="{kind}" data-step="{i}" aria-pressed="{str(i == 0).lower()}">{i+1 if kind in ("finalize", "compiler") else ""} {html.escape(s["label"])}</button>' for i, s in enumerate(items))
    target = items[0].get("cell", "ragged")
    controls = f'<div class="explorer-footer"><button data-previous="{kind}" aria-label="Previous {kind} stage">← Previous</button><span data-position="{kind}">1 / {len(items)}</span><button data-next="{kind}" aria-label="Next {kind} stage">Next →</button><button class="source-jump" data-explorer-source="{kind}" data-target="{target}">Open matching source ↗</button></div>'
    result = f'<div class="explorer" id="explorer-{kind}"><div class="diagram-heading"><span class="badge">SOURCE</span><h3>{heading}</h3><span class="diagram-caption">Illustration derived from code</span></div><div class="step-tabs" role="group" aria-label="{heading}">{buttons}</div><div class="explorer-panel" id="panel-{kind}" aria-live="polite">{panel(kind, items[0])}</div>{controls}<details class="ascii-fallback"><summary>Static text diagram</summary><pre>{html.escape(fallback)}</pre></details></div>'
    return dict(html=result, markdown="```text\n" + fallback + "\n```\n\n`SOURCE` — conceptual illustration; not an executed graph dump.\n")


FINAL = """Forward outputs -> flatten tree -> normalize Values + conditional chains
                -> functionType + generator signature
                -> mo.output terminator
                -> argument_names + result_names
                -> verify completed graph

Graph.__exit__ restores context. Native compilation is a subsequent call."""
COMPILER = """Graph IR -> MO preparation -> MOToMAP -> MAP fusion/outlining
         -> MAPToMOGG -> MOGG routines -> MOGGToMGP
         -> MGP resource planning + Mojo emission
         -> kernel JIT + runtime-call lowering -> MEF serialization

Pass scheduling is source-proven. A specific fused kernel needs a concrete dump."""
CACHE = """COLD:       trace -> frontend -> passes -> serialize -> read MEF -> init
MEF IMPORT: trace -> Python store -> read MEF path -> init
NATIVE HIT: trace -> frontend -> MEF cache hit -> read MEF -> init

In this GraphSpec route, all three paths have already traced the graph."""
BUFFERS = """Two illustrative sequences; token IDs are examples, not a captured Llama run.

PREFILL: tokens [17, 42, 6 | 9, 21]   row_offsets [0, 3, 5]
DECODE:  tokens [88       | 31   ]   row_offsets [0, 1, 2]

types: tokens=int64[T], offsets=uint32[B+1], return_n_logits=int64[R] on CPU
KV leaves: buffers/metadata declared by kv_params; carry the cache state."""
NATIVE_FLOW = """Pipeline      Python session    C++ session    Engine worker   CompiledModels   MOF frontend   Compiler
   |                |                |              |                |               |             |
   | compile(graph) |                |              |                |               |             |
   |--------------->| capsule        |              |                |               |             |
   |                |--------------->| request      |                |               |             |
   |                |                |------------->| create         |               |             |
   |                |                |              |--------------->| compile       |             |
   |                |                |              |                |-------------->| run passes  |
   |                |                |              |                |               |------------>|
   |                |                |              |                |               |<------------|
   |                |                |              |                |<--------------| MEF buffer  |
   |                |                |              |<---------------| read artifact |             |
   |                |<----------------------------------------------- compiled result              |
   |<---------------| wait complete  |              |                |               |             |
   | init_all(weights)              |              |                |               |             |
   |--------------->| native initialization -> executable models    |               |             |

GraphSpec.trace / Graph.output finish before this cold compilation sequence.
The engine schedules the compile on a CPU worker; arrows show call/result order."""


def sequence_svg():
    actors = ["Pipeline", "Session · Python", "Session · C++", "Engine worker", "CompiledModels", "MOF frontend", "Compiler"]
    xs = [78 + i * 155 for i in range(len(actors))]
    elements = ['<svg viewBox="0 0 1090 455" role="img" aria-labelledby="native-seq-title native-seq-desc"><title id="native-seq-title">Native compilation sequence</title><desc id="native-seq-desc">The Python session hands a capsule to C++. The engine schedules a worker, CompiledModels calls the frontend, and compiler passes produce a model serialized to MEF before the result returns.</desc><defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" class="call-arrowhead"/></marker></defs>']
    for x, name in zip(xs, actors):
        elements.append(f'<rect x="{x-70}" y="10" width="140" height="40" rx="7" class="actor-box"/><text x="{x}" y="35" text-anchor="middle" class="actor-label">{name}</text><line x1="{x}" y1="52" x2="{x}" y2="430" class="lifeline"/>')
    calls = [(0,1,"compile(graph)"),(1,2,"MLIR capsule"),(2,3,"request / schedule"),(3,4,"create"),(4,5,"compileToBinary"),(5,6,"runGCPipeline"),(6,5,"lowered module"),(5,4,"MEF buffer"),(4,3,"compiled objects"),(3,1,"resolve result"),(1,0,"wait returns")]
    for i,(a,b,label) in enumerate(calls):
        y=80+i*31
        dashed=' stroke-dasharray="5 4"' if i>=6 else ''
        elements.append(f'<line x1="{xs[a]}" y1="{y}" x2="{xs[b]}" y2="{y}" class="call-arrow" stroke-width="1.6"{dashed} marker-end="url(#arrowhead)"/><text x="{(xs[a]+xs[b])/2}" y="{y-7}" text-anchor="middle" class="call-label">{label}</text>')
    elements.append('</svg>')
    return ''.join(elements)


WIDGETS = {
    "finalize": explorer("finalize", "What changes inside Graph.output()", FINAL),
    "compiler": explorer("compiler", "Follow the compiler representations", COMPILER),
    "cache": explorer("cache", "Compare artifact reuse paths", CACHE),
    "buffers": explorer("buffers", "Change the batch, keep the symbolic contract", BUFFERS),
    "native": dict(html='<div class="sequence-diagram"><div class="diagram-heading"><span class="badge">SOURCE</span><h3>The native compile handoffs</h3></div><div class="sequence-scroll">'+sequence_svg()+'</div><details class="ascii-fallback"><summary>Portable call sequence</summary><pre>'+html.escape(NATIVE_FLOW)+'</pre></details></div>', markdown="```text\n"+NATIVE_FLOW+"\n```\n"),
}
