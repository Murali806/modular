"""Fused-attention reader: real callers, three inputs, tile parameters and state."""
from __future__ import annotations

import copy
import hashlib
import html
import re

from compiler_calltree_reader import CallTreeReader, EXPLANATION_WINDOW, TOKEN
from compiler_session_calltrees import PANELS

GRAPH = "max/python/max/graph/graph.py"
ENTRY = "max/examples/custom_ops/fused_attention.py"
KERNEL = "max/examples/custom_ops/kernels/fused_attention.mojo"
SOURCES = [
    ("attention_callsite", ENTRY, "41:68", "lines", 0, "Attention context-manager graph and three-input custom call"),
    ("attention_sizes", ENTRY, "26:39", "lines", 0, "Actual CPU and GPU sizes and tile parameters"),
    ("attention_registration", KERNEL, "66:107", "lines", 0, "Namespaced registration, inferred types, explicit parameters and dispatch"),
    ("attention_mma", KERNEL, "234:310", "lines", 0, "GPU TensorCore loads, accumulation, shared store and barrier"),
    ("attention_tiles", KERNEL, "331:346", "lines", 0, "GPU query/output tile selection and key tile BN_1=8"),
    ("attention_max", KERNEL, "348:349", "lines", 0, "Per-row tile maximum and running maximum"),
    ("attention_denominator", KERNEL, "352:363", "lines", 0, "Stable exponentials and accumulated denominator"),
    ("attention_output", KERNEL, "366:378", "lines", 0, "Normalize current tile, rescale retained output, add PV, store final output"),
    ("attention_launch", KERNEL, "381:410", "lines", 0, "Specialize the kernel and enqueue grid (4,2), block 32"),
    ("attention_parameter_attr", "max/python/max/graph/ops/custom.py", "_parameter_attribute", "python", 0, "Python int tile parameters become MLIR index attributes"),
    ("attention_ctor_decorator", GRAPH, "817:819", "lines", 0, "Decorator guarding Graph construction on background threads"),
    ("attention_gc", "GraphCompiler/lib/GraphCompiler.cpp", "199:677", "lines", 0, "Graph-stage builders and pass-manager execution"),
    ("attention_jit", "GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp", "529:594", "lines", 0, "JitCompileKernelsPass invokes native kernel compilation"),
    ("attention_runtime_lower", "GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp", "102:168", "lines", 0, "Lower compiled routines to ModuleLoad and ExecuteGeneric"),
]

GRAPH_TREE = '''caller: with Graph("fused_attention", input_types=[TQ,TK,TV], custom_extensions=[kernels/]) as graph [LINK:entry:41] [CALLER:entry|41]
├─ Evaluate caller arguments before Graph.__init__ [LINK:entry:43]
│  ├─ dtype=DType.float32; device=CPU() or Accelerator() [LINK:entry:26] [LINK:entry:29] [CALLER:entry|29]
│  ├─ CPU: N=D=8, BN=BD=4; GPU: N=D=32, BN=16, BD=8 [LINK:attention_sizes:30] [CARD:toolchain-call-graph-sizes|Sizes · Actual CPU/GPU contracts] [CALLER:entry|30]
│  └─ TQ, TK, TV: TensorType(dtype, shape=[N,D], device=DeviceRef.from_device(device)) [LINK:entry:45] [CALLER:entry|45]
│     OUT: three type descriptions; no NumPy matrices or Buffer objects yet [LINK:entry:44] [CALLER:entry|43]
│
├─ Graph.__init__(name="fused_attention", input_types=[TQ,TK,TV], forward=None) [LINK:graph_init:819]
│  ├─ @in_default_mlir_context → ensure an active MLIR context on this thread [LINK:ct_graph_context:70] [CARD:toolchain-call-graph-context|A.0 · default MLIR context] [CALLER:attention_ctor_decorator|818]
│  ├─ Discover SymbolicDim names; initialize _params and context state [LINK:graph_init:840]
│  │  └─ Shapes are static → _params={}; _data_dependent_dims=0; _should_verify_ops=True [LINK:graph_init:847] [CALLER:graph_init|840]
│  ├─ Enter _location() and choose the top-level module [LINK:graph_init:851] [CARD:toolchain-call-graph-location|_location() · What is loc?]
│  │    it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module. [LINK:graph_init:851] [CALLER:graph_init|851]
│  │  ├─ Reuse module.mlir_module if supplied; otherwise builtin.ModuleOp(location=loc) [LINK:graph_init:856] [CARD:toolchain-call-graph-module-setup|MLIR Module Reuse or create] [CALLER:graph_init|856]
│  │  │    **MLIR.module is IR container** [LINK:ct_builtin_module:62] [CALLER:graph_init|856]
│  │  └─ Place OpBuilder at self._module.body.end [LINK:graph_init:858] [CARD:toolchain-call-graph-op-builder|OpBuilder · Insertion point] [CALLER:graph_init|858]
│  ├─ Insert mo.GraphOp(builder, loc, name, input_types, result_types=[]) [LINK:graph_init:860]
│  │  ├─ builder comes from module.body.end; loc from _location(); name from caller L42 [LINK:graph_init:861] [LINK:entry:42] [CALLER:graph_init|860]
│  │  ├─ Convert three Python input types with t.to_mlir() [LINK:graph_init:864] [CARD:toolchain-call-graph-type-conversion|to_mlir() · Q/K/V type fields] [CALLER:graph_init|864]
│  │  └─ Native builder creates a signature, one region, ONE block, THREE block arguments [LINK:ct_graph_op_build:831] [LINK:ct_graph_op_build:846] [CARD:toolchain-call-graph-native-builder|GraphOp · Signature and body] [CALLER:graph_init|860]
│  │     OUT: %q:TQ, %k:TK, %v:TV in the same entry block; results initially empty [LINK:ct_graph_op_build:848] [CALLER:ct_graph_op_build|848]
│  ├─ Declare runtime input parameters for symbolic dimensions [LINK:graph_init:870] [CARD:toolchain-call-graph-symbolic-params|Shape names · Graph parameters]
│  │  └─ _params={} → empty ParamDeclArrayAttr; BN/BD will be custom-op parameters later [LINK:graph_init:871] [CALLER:graph_init|870]
│  ├─ Retain operation and body handles: _CAPICreate(op._CAPIPtr), regions[0].blocks[0] [LINK:graph_init:874] [CARD:toolchain-call-graph-retain-body|Graph handles · CAPI and body] [CALLER:graph_init|874]
│  ├─ Attach module device-info mapping [LINK:graph_init:877] [CARD:toolchain-call-graph-device-info|Device metadata · Targets] [CALLER:graph_init|877]
│  ├─ Initialize _weights={}, _has_chain_input=False, lazy device_chains [LINK:graph_init:879] [CARD:toolchain-call-graph-tracking|Graph bookkeeping · Weights and chains] [CALLER:graph_init|881]
│  ├─ Insert ChainCreateOp(inputs=[]), retain always-ready CH0, seed CPU ordering [LINK:graph_init:885] [CARD:toolchain-call-graph-ready-chain|Chain · Ordering token] [CALLER:graph_init|886]
│  ├─ Inspect last argument %v: tensor, not a trailing chain → retain CH0 [LINK:graph_init:896] [CARD:toolchain-call-graph-trailing-chain|Last input · Tensor or chain?] [CALLER:graph_init|900]
│  ├─ Set up the custom-kernel library [LINK:graph_init:906]
│  │    self._kernel_library = kernel_library or KernelLibrary() [LINK:graph_init:909] [CALLER:graph_init|909]
│  │    self._import_kernels(_resolved_custom_extensions(custom_extensions)) [LINK:graph_init:910] [CALLER:graph_init|910]
│  │    Resolving defaults also makes backend kernel overlays reachable when explicit paths are absent [LINK:ct_graph_extensions] [CALLER:graph_init|910]
│  │  └─ _import_kernels(...): prepare package P1 [LINK:ct_graph_import:1703] [CALLER:graph_init|910]
│  │     ├─ KernelLibrary.load_paths([kernels/]) [LINK:package_load:269]
│  │     │  └─ _build_mojo_source_package(kernels/) [LINK:package_builder:141]
│  │     │     ├─ kernels/__init__.mojo + referenced child modules, including fused_attention.mojo [LINK:kernel_package_root:1] [CALLER:package_load|289]
│  │     │     ├─ Separate process: Mojo executable + precompile kernels/ -o <private staging path> [LINK:package_builder:169] [CARD:toolchain-call-graph-package|P1 · Executable, parsing and bytecode]
│  │     │     │  └─ parse → resolve → semantic/lifetime checks → MLIR bytecode [LINK:ct_precompile] [LINK:ct_check_lit] [CALLER:package_builder|178]
│  │     │     └─ os.replace(staging_path, package_path) → atomically published P1 .mojoc [LINK:package_builder:179] [CALLER:package_builder|179]
│  │     ├─ Existing .mojoc path → add_path directly [LINK:package_load:286] [CALLER:ct_graph_import|1704]
│  │     └─ Store resolved library paths on the graph op [LINK:ct_graph_import:1706] [CALLER:ct_graph_import|1704]
│  └─ forward=None → no lambda is executed in the constructor [LINK:graph_init:918] [CALLER:entry|41]
│     OUT: graph shell G1 and P1 available; the with-body below records its computation [LINK:entry:55] [CALLER:entry|41]
│
└─ Python enters the with-body after the constructor returns [LINK:entry:55]
   ├─ Graph.__enter__ → _enter → set Graph.current=G1 [LINK:ct_graph_context_manager:1137] [CARD:toolchain-call-graph-forward-context|with Graph · Active recording scope] [CALLER:entry|55]
   ├─ q, k, v, *_ = graph.inputs [LINK:entry:56] [CARD:toolchain-call-graph-forward-inputs|Inputs · Three symbolic wrappers]
   │  └─ Wrap body.arguments → q→%arg0, k→%arg1, v→%arg2; no numeric data [LINK:ct_graph_inputs:943] [CALLER:entry|56]
   ├─ Declare one output TensorType(dtype, [N,D], device) [LINK:entry:63] [CARD:toolchain-call-graph-output-type|Output type · Future O] [CALLER:entry|62]
   ├─ results = ops.custom(name="modular_ops::fused_attention_custom", ...) [LINK:entry:57] [CARD:toolchain-call-graph-custom-insert|mo.custom · Detailed three-input call]
   │  ├─ Resolve Graph.current; name → StringAttr; normalize DeviceRef [LINK:custom_api:94] [CARD:toolchain-call-graph-custom-attributes|Attributes · Name, device, BN/BD] [CALLER:entry|57]
   │  ├─ values=[q,k,v] → existing %arg0,%arg1,%arg2; out_types=[TO] → native result type [LINK:custom_api:105] [CALLER:entry|61]
   │  ├─ Insert mo.custom in G1.body; wrap its one symbolic result %o [LINK:custom_api:109] [CALLER:custom_api|109]
   │  ├─ Attach parameters AFTER insertion: BD/BN → index attributes → dictionary [LINK:custom_api:113] [LINK:attention_parameter_attr:48] [CALLER:entry|60]
   │  ├─ Attach DeviceRefAttr via MLIR Attribute capsule [LINK:custom_api:122] [CALLER:custom_api|122]
   │  ├─ Verify call against loaded P1; resolve FusedAttention.execute and map types/parameters [LINK:custom_api:127] [CARD:toolchain-call-graph-custom-verify|Kernel validation · P1 and MO] [CALLER:custom_api|127]
   │  └─ Return results=[TensorValue(%o: float32[N,D], selected device)] [LINK:custom_api:129] [CARD:toolchain-call-graph-custom-result|Result wrappers · One symbolic output] [CALLER:entry|57]
   ├─ graph.output(*results) → mo.output, result signature, graph verification [LINK:entry:68] [CARD:toolchain-call-graph-forward-output|Graph.output · Complete G1] [CALLER:entry|68]
   └─ Graph.__exit__ restores prior Graph.current [LINK:ct_graph_context_manager:1145] [CALLER:entry|55]
      OUT: verified G1: (TQ,TK,TV) → (TO); compilation and execution are still later [LINK:entry:68] [CALLER:entry|68]'''


def attention_session_panels():
    """Adapt shared API trees at named call sites; execute has its own payload."""
    panels = copy.deepcopy(PANELS[:3])
    entry_lines = {62: 71, 67: 74, 68: 75, 74: 78, 77: 82}
    replacements = {
        "addition.py supplies P1 later via G1": "fused_attention.py supplies P1 later via G1",
        "add-one produces {}": "fused attention produces {} (Q/K/V are call inputs)",
        "including addition;": "including fused_attention;",
        "Add-one has no external weights": "Fused attention has no external weights",
        "Add-one: expected_weights={}": "Attention: expected_weights={}",
        "for add-one": "for fused attention",
        "addition has one model": "fused_attention has one model",
        'user tensor X': 'user inputs Q/K/V',
        '{"addition": M1}': '{"fused_attention": M1}',
        "then M1.execute(x)": "then M1.execute(Q,K,V)",
    }
    for panel in panels:
        panel["line"] = entry_lines[panel["line"]]
        for key in ("tree", "intro"):
            text = panel[key]
            text = re.sub(r"\[LINK:entry:(\d+)\]", lambda m: f"[LINK:entry:{entry_lines[int(m[1])]}]", text)
            text = re.sub(r"\[CALLER:entry\|(\d+)\]", lambda m: f"[CALLER:entry|{entry_lines[int(m[1])]}]", text)
            for old, new in replacements.items():
                text = text.replace(old, new)
            panel[key] = text
    panels[0]["title"] = "InferenceSession constructor · configuration and context → S1"
    panels.append(dict(
        id="model-execute", title="model.execute(Q, K, V) · M1 + three Buffers → [O1]",
        call="output = model.execute(Q, K, V)", line=82,
        intro="IN: initialized M1 and three actual contiguous float32 Buffers, each [8,8] on CPU or [32,32] on the selected GPU. OUT: a list containing O1. The original script prints that list; it does not select [0], copy to CPU, or assert parity.",
        code='''np.random.seed(123)
Q = Buffer.from_numpy(np.random.randn(N, D).astype("f")).to(device)
K = Buffer.from_numpy(np.random.randn(N, D).astype("f")).to(device)
V = Buffer.from_numpy(np.random.randn(N, D).astype("f")).to(device)

output = model.execute(Q, K, V)
print(output)''',
        tree=EXECUTE_TREE))
    return panels


EXECUTE_TREE = '''caller: output = model.execute(Q, K, V) [LINK:entry:82] [CALLER:entry|82]
├─ Before execute: np.random.seed(123); create three real NumPy matrices [LINK:entry:77]
│  ├─ Buffer.from_numpy(Q_array).to(device) → BQ1 [LINK:entry:78] [CALLER:entry|78]
│  ├─ Buffer.from_numpy(K_array).to(device) → BK1 [LINK:entry:79] [CALLER:entry|79]
│  └─ Buffer.from_numpy(V_array).to(device) → BV1 [LINK:entry:80] [CALLER:entry|80]
│     CPU: each has 8×8×4=256 bytes; GPU: each has 32×32×4=4096 bytes [LINK:entry:78] [CALLER:entry|78]
│
└─ Model.execute is installed as _Model_execute; @traced records the call [LINK:ct_execute_install:291] [LINK:ct_execute_py:118]
   ├─ Iterate positional inputs in Q,K,V order [LINK:ct_execute_py:122]
   │  ├─ Validate contiguity; existing Buffer inputs remain Buffers [LINK:ct_execute_py:124] [CALLER:ct_execute_py|124]
   │  ├─ Alternative DLPack/scalar objects use their conversion branches [LINK:ct_execute_py:130] [CALLER:ct_execute_py|130]
   │  └─ input_impls=[BQ1,BK1,BV1]; unsupported types raise before native execution [LINK:ct_execute_py:136] [LINK:ct_execute_py:144] [CALLER:ct_execute_py|144]
   └─ self._execute_device_tensors(input_impls) [LINK:ct_execute_py:145]
      └─ nanobind → ModelHandle::executeDeviceTensors([BQ1,BK1,BV1]) [LINK:ct_execute_binding:565]
         ├─ No loadedModel? empty result; valid initialized M1 continues [LINK:ct_execute_native:457] [CALLER:ct_execute_native|457]
         ├─ validateInputSpecList → count=3, shapes/dtype match compiled signature [LINK:ct_execute_native:462]
         │  └─ Expected CPU example: three float32[8,8]; wrong count or static size raises [LINK:ct_execute_specs:438] [CALLER:ct_execute_native|462]
         ├─ runtimeCtx=S1.native.runtimeContext; create input AsyncValue list [LINK:ct_execute_native:464] [CALLER:ct_execute_native|464]
         ├─ For each of BQ1,BK1,BV1 [LINK:ct_execute_native:475]
         │  ├─ verifyInputDevice against corresponding model input DeviceRef [LINK:ct_execute_native:476] [CALLER:ct_execute_native|476]
         │  ├─ getInputStorage → buffer owner, pointer and byte size [LINK:ct_execute_native:479] [CALLER:ct_execute_native|479]
         │  └─ prepareInput(cpuDevice, storage.buf, tensorSpec) [LINK:ct_execute_native:484]
         │     └─ Driver::prepareInput → retain storage in ready Tensor AsyncValue [LINK:ct_execute_input:112]
         │        ├─ Example GPU input: symbolic pointer pQ, bytes=4096, spec=f32[32,32] [LINK:ct_execute_input:116] [CALLER:ct_execute_native|484]
         │        └─ TensorBufferRef shares same storage; no new transfer in this wrapper step [LINK:ct_execute_input:121] [CALLER:ct_execute_input|121]
         └─ executePreparedDeviceTensors(runtimeCtx, inputValues) [LINK:ct_execute_native:518]
            ├─ Release Python GIL while running native execution [LINK:ct_execute_prepared:529] [CALLER:ct_execute_prepared|529]
            ├─ Driver::executeFromDriver(context, loadedModel.copy(), inputValues) [LINK:ct_execute_prepared:530]
            │  ├─ Await initialized executable; errors stop here [LINK:ct_execute_driver:204] [CALLER:ct_execute_driver|204]
            │  ├─ model->execute(inputs) → ExecutableModel::execute [LINK:ct_execute_driver:212]
            │  │  ├─ passedArgs=[chain,MGP state,weights,extraArgs,...Q/K/V AsyncValues] [LINK:ct_execute_callable:373] [CALLER:ct_execute_callable|378]
            │  │  ├─ Set current AsyncRT CPUDevice for host routines [LINK:ct_execute_callable:382] [CALLER:ct_execute_callable|382]
            │  │  └─ getExecuteFn()->invoke(passedArgs,cpuDevice) [LINK:ct_execute_callable:384] [CARD:toolchain-call-example|E · Buffers, execution and output]
            │  │     ├─ Compiled host routine reaches specialized FusedAttention.execute [LINK:attention_registration:74] [CALLER:ct_execute_callable|384]
            │  │     ├─ CPU target → fused_attention_cpu[BN=4,BD=4] [LINK:attention_registration:101] [CALLER:attention_registration|101]
            │  │     └─ GPU target → fused_attention_gpu[BN=16,BD=8] → enqueue_function [LINK:attention_registration:105] [CARD:toolchain-call-kernel-dispatch|Kernel · CPU/GPU dispatch] [CALLER:attention_registration|105]
            │  │        └─ Kernel computes tiled QKᵀ → online softmax → PV → O [LINK:attention_launch:403] [CARD:toolchain-call-kernel-online|Online softmax · Two tiles] [CALLER:attention_launch|403]
            │  └─ extractDriverResults(context,executable,result AsyncValues) [LINK:ct_execute_driver:213]
            │     ├─ Resolve output device from the session config [LINK:ct_execute_results:125] [CALLER:ct_execute_results|125]
            │     ├─ awaitOrError(results) → result readiness or diagnostic [LINK:ct_execute_results:137] [CALLER:ct_execute_results|137]
            │     ├─ Drop first native result: internal ordering chain [LINK:ct_execute_results:159] [CALLER:ct_execute_results|159]
            │     └─ Tensor result → DeviceBuffer + spec + selected device [LINK:ct_execute_results:166] [CALLER:ct_execute_results|178]
            └─ extractPyBufferResults → output=[O1] [LINK:ct_execute_prepared:538]
               ├─ Preserve storage/aliases; return a Python Buffer wrapper [LINK:ct_execute_py_results:70] [CALLER:ct_execute_prepared|538]
               └─ Caller print(output) prints the list; no explicit to(CPU()) or parity assertion [LINK:entry:83] [CALLER:entry|83]
                  OUT: one float32[N,D] Buffer; S1/M1 and resident code remain available [LINK:ct_execute_py_results:79] [CALLER:entry|82]'''


KERNEL_TREE = '''caller: compiled FusedAttention.execute(output, query, key, value, ctx) [LINK:attention_registration:74]
├─ Infer dtype=float32, rank=2 from tensor types; explicit BN/BD; target from device [LINK:attention_registration:75] [CARD:toolchain-call-kernel-dispatch|Dispatch · Specialization parameters]
│  ├─ CPU: [BN=4,BD=4,target="cpu"]; GPU: [BN=16,BD=8,target="gpu"] [LINK:entry:33] [LINK:entry:38] [CALLER:entry|60]
│  └─ comptime assert rank==2; convert output/Q/K/V to tile tensor views [LINK:attention_registration:88] [CALLER:attention_registration|91]
├─ CPU route: fused_attention_cpu[4,4](Q,K,V,O) [LINK:attention_registration:101]
│  ├─ For each query tile [4,8] and output-column tile [4,4] [LINK:math_cpu] [CALLER:attention_registration|101]
│  ├─ Initialize running state m,l,O; visit two key/value tiles of 4 rows [LINK:kernel_file:185] [CALLER:math_cpu|178]
│  └─ Same stable normalized-output recurrence; final O tile store [LINK:kernel_file:230] [CARD:toolchain-call-kernel-online|Online softmax · Recurrence] [CALLER:kernel_file|198]
└─ GPU route: fused_attention_gpu[16,8](ctx,Q,K,V,O) [LINK:attention_registration:105]
   ├─ Specialize fused_attention_kernel for dtypes/layouts and BN/BD [LINK:attention_launch:391] [CALLER:attention_launch|391]
   ├─ ctx.enqueue_function[kernel_func](Q,K,V,O,grid_dim=(4,2),block_dim=32) [LINK:attention_launch:403] [CARD:toolchain-call-kernel-launch|Launch · Blocks and driver handoff] [CALLER:attention_launch|403]
   └─ Each block selects query rows by block_idx.y and output columns by block_idx.x [LINK:attention_tiles:333]
      ├─ Q_tile=[16,32]; retained m/l=[16], O_i=[16,8] [LINK:attention_tiles:335] [CALLER:kernel_file|333]
      ├─ BN_1=8; for four key/value tiles [LINK:attention_tiles:341] [CARD:toolchain-call-kernel-storage|Storage · Tiles and retained state]
      │  ├─ K_tile=[8,32], V_tile=[8,8]; S=Q_tile×K_tileᵀ=[16,8] [LINK:attention_tiles:344] [CALLER:attention_tiles|346]
      │  ├─ matmul["gpu",transpose_b=True]: fragments → MMA → shared output [LINK:attention_mma:276] [CARD:toolchain-call-kernel-mma|MMA · Loads, accumulation and barrier] [CALLER:attention_tiles|346]
      │  ├─ m_new=max(m_old,row_max(S)) [LINK:attention_max:348] [CALLER:attention_max|349]
      │  ├─ alpha=exp(m_old-m_new); E=exp(S-m_new); l_new=alpha*l_old+sum(E) [LINK:attention_denominator:352] [CALLER:kernel_file|355]
      │  ├─ Normalize E/=l_new; rescale old O*=l_old/l_new*alpha [LINK:attention_output:366] [CALLER:attention_output|367]
      │  └─ PV=E×V_tile; O+=PV; keep m_new and l_new for next tile [LINK:attention_output:374] [CALLER:attention_output|375]
      └─ Store O_i into O.tile[BN,BD](block_idx.y,block_idx.x) once after loop [LINK:attention_output:378] [CALLER:attention_output|378]'''


def card(suffix, title, mechanism, explanation, tree=None, source="attention_callsite", line=41):
    return dict(id="toolchain-call-" + suffix, title=title, mechanism=mechanism,
                explanation=explanation, tree=tree, source=source, line=line)


CARDS = [
    card("graph-sizes", "Sizes · Actual CPU/GPU contracts", "type", '''**The caller selects concrete sizes before constructing G1.** These are the values in [fused_attention.py](sources/max/examples/custom_ops/fused_attention.py#L30).

| Target | Q, K, V and O | Query BN | Output BD | Key rows per iteration |
|---|---|---|---|---|
| CPU | float32 [8,8] | 4 | 4 | 4 |
| GPU | float32 [32,32] | 16 | 8 | BN_1=8 |

**BN and BD are explicit compile-time parameters on the custom call.** They are not graph input tensors or symbolic dimension parameters. See [parameter attachment](sources/max/python/max/graph/ops/custom.py#L113).'''),
    card("graph-context", "A.0 · default MLIR context", "context", '''**The default is a module-level shared MLIR Context object. Current context is active state on this thread.** A context owns dialects and interned types/attributes; it does not contain the numerical Q/K/V matrices.

| Name | Example value | Role |
|---|---|---|
| `_DEFAULT_MLIR_CONTEXT` | context object MCX1 | Shared fallback in this interpreter |
| `mlir.Context.current` | MCX1 on main; None on a new worker | Per-thread active-context lookup |
| `MLIRThreadPoolExecutor` | executor EX1 | Wraps submitted tasks with the context guard |

The Python line `@in_default_mlir_context` is a **decorator**: it wraps the constructor. On a background thread, `call_with_default_mlir_context` preserves an existing current context or temporarily enters the shared default. It restores the previous scope afterward. See [the context module](sources/max/python/max/_mlir_context.py#L26) and [executor](sources/max/python/max/_mlir_context.py#L89).

**Background-thread example:** main submits `Graph(...)` construction to a worker whose current context is None. The guard enters MCX1 on that worker while it constructs IR. This does not copy main's active-context state or allocate another default context.'''),
    card("graph-location", "_location() · What is loc?", "location", '''**loc is an MLIR Location: provenance metadata attached to IR operations and input arguments.**

it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module.

[The helper](sources/max/python/max/graph/graph.py#L438) returns `Location.unknown()` with source-traceback capture disabled. When enabled, it derives metadata from the Python stack. An active `Graph.profile_scope()` can also contribute profiling-scope metadata.

**Example stack:** `fused_attention.py:87 → main() → fused_attention.py:41 → Graph(...)`. The native [frame binding](sources/max/python/max/_core/max_license/modules/graph.cpp#L162) wraps summaries in an opaque location. A readable label such as `main / fused_attention.py:41` illustrates provenance; it is not a captured dump.

Inside `with _location() as loc`, the binding scopes this Location as current. MAX also passes it explicitly to [ModuleOp and GraphOp](sources/max/python/max/graph/graph.py#L856); [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L848) attaches it to each input argument. **Locations remain attached after the with block exits.** loc is metadata, not the MLIR Context or a tensor address.'''),
    card("graph-module-setup", "MLIR Module Reuse or create", "module", '''**MLIR.module is IR container.** The native `builtin.ModuleOp` normally prints as `module { ... }`. It has an operation → body region → body block → contained operations structure.

**Creation:** `module=None` calls [builtin.ModuleOp(location=loc)](sources/max/python/max/graph/graph.py#L856), whose [native constructor](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62) calls `mlir::ModuleOp::create`. The body initially has no graph definitions.

**Reuse:** `module=existing_module` retains `existing_module.mlir_module`. An existing `@encoder` definition remains a sibling of the newly inserted `@fused_attention`; no IR is copied by choosing this branch. See [the public Module wrapper](sources/max/python/max/graph/graph.py#L562).

```mlir
// Schematic nesting after GraphOp insertion
module {
  mo.graph @fused_attention(%q:T, %k:T, %v:T) {
    // one region, one block, three block arguments
  }
}
```

**The module groups definitions; the GraphOp owns the computation's body.**''',
         tree='''caller: self._module = module.mlir_module if module else builtin.ModuleOp(location=loc) [LINK:graph_init:856]
├─ reuse: Module.mlir_module → existing builtin.ModuleOp [LINK:ct_graph_module:581] [CALLER:graph_init|856]
└─ create: native ModuleOp constructor [LINK:ct_builtin_module:62]
   └─ mlir::ModuleOp::create(loc) → operation with body region/block [LINK:ct_builtin_module:65] [CALLER:graph_init|856]'''),
    card("graph-op-builder", "OpBuilder · Insertion point", "builder", '''**An insertion point is a block plus a position in that block.** [body.end](sources/max/python/max/_core/max_license/_core.cpp#L408) creates an end iterator, and the [native OpBuilder constructor](sources/max/python/max/_core/max_license/_core.cpp#L612) receives it.

| Before | Builder action | After |
|---|---|---|
| module body `[encoder]` | insert at end | `[encoder, fused_attention]` |
| graph body `[%chain]` | later insert custom call | `[%chain, %o=custom(%q,%k,%v)]` |

The first builder inserts the graph into its module. [Graph._add_op_get_op_with_results](sources/max/python/max/graph/graph.py#L1284) uses the retained graph body for later call insertion. **These are different insertion blocks.**''',
         tree='''caller: OpBuilder(self._module.body.end) [LINK:graph_init:858]
├─ ModuleOp.body → existing module-body block [LINK:ct_builtin_module:67] [CALLER:graph_init|858]
├─ Block.end → InsertionPoint(block, block.end()) [LINK:ct_block_end:408] [CALLER:graph_init|858]
└─ native OpBuilder(block, iterator) → insertion cursor [LINK:ct_op_builder:612] [CALLER:graph_init|858]'''),
    card("graph-type-conversion", "to_mlir() · Q/K/V type fields", "type", '''**to_mlir converts a Python type description into native MO type fields. It does not convert tensor data.** [TensorType.to_mlir](sources/max/python/max/graph/type.py#L430) constructs `mo.TensorType(shape_attr, dtype, device_attr, metadata=...)`.

| Field | Concrete example | Native meaning |
|---|---|---|
| Shape | GPU `[32,32]` | Two statically known sizes |
| Element dtype | `DType.float32` | Each future element is a 32-bit float |
| Device | `DeviceRef("gpu",0)` | Symbolic placement on gpu:0 |
| Metadata | `{}` here | No explicit layout constraint |

**A size attribute is not an element.** For a static size 32, [StaticDim.to_mlir](sources/max/python/max/graph/dim.py#L562) creates a signed i64 integer attribute, then wraps it with `kgen.CastFromBuiltinAttr`. This says “dimension length is 32”; the data elements are still float32. [Shape.to_mlir](sources/max/python/max/graph/shape.py#L36) collects dimension expressions in `mosh.ShapeAttr`.

**Layout comparison:** a convolution filter may carry `metadata={"layout": ...}` such as HWIO or OIHW. Native [TensorType::getLayoutAttr](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L454) reads that metadata to interpret dimensions. **Attention leaves this dictionary empty**; it does not attach a filter layout. Logical shape is separate from target-specific fragment/storage layout.''',
         tree='''caller: [t.to_mlir() for t in input_types] [LINK:graph_init:864]
└─ TensorType.to_mlir → mo.TensorType(...) [LINK:ct_tensor_to_mlir:430]
   ├─ self.shape.to_mlir() → ShapeAttr(dim attributes, ShapeType()) [LINK:ct_shape_to_mlir]
   │  └─ each StaticDim(32) → signed i64 IntegerAttr(32) → CastFromBuiltinAttr [LINK:ct_static_dim_to_mlir] [CALLER:ct_shape_to_mlir|39]
   ├─ self.dtype=DType.float32 → existing element dtype value [LINK:ct_tensor_to_mlir:441] [CALLER:ct_tensor_to_mlir|439]
   ├─ self.device.to_mlir() → m.DeviceRefAttr("gpu",0) [LINK:ct_device_ref_to_mlir] [CALLER:ct_tensor_to_mlir|442]
   └─ builtin.DictionaryAttr(metadata={}) → no explicit filter layout here [LINK:ct_tensor_to_mlir:443] [CALLER:ct_tensor_to_mlir|443]'''),
    card("graph-native-builder", "GraphOp · Signature and body", "graph", '''**One graph region contains ONE entry block with THREE input arguments.** The caller provides input types, not an existing region. [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L831) records a function type, creates the region/block, then adds one block argument for each input type.

```text
MODULE1: builtin.module
└─ G1: mo.graph @fused_attention
   ├─ signature: (TQ,TK,TV) → () initially
   └─ region R0
      └─ block B0(%q:TQ,%k:TK,%v:TV)
         └─ operations will be inserted here
```

**Why a signature?** It declares the caller/callee contract: count, type, shape and device of inputs and outputs. During output finalization, it becomes `(TQ,TK,TV) → (TO)`. During execute, the runtime checks three real buffers against this contract. A block argument is an SSA value at entry, not a separately allocated block or Buffer.'''),
    card("graph-symbolic-params", "Shape names · Graph parameters", "type", '''**All attention dimensions here are static: _params={}.** The constructor's [discovery](sources/max/python/max/graph/graph.py#L840) and [declaration](sources/max/python/max/graph/graph.py#L870) therefore attach an empty parameter array.

**Comparison:** three input shapes `["seq",32]` would discover and deduplicate `seq`; the graph's input parameter declaration would carry one signed-i64 size name. That is different from `parameters={"BN":16,"BD":8}` on the custom op. BN/BD choose kernel tile sizes; they are not additional Q/K/V inputs.'''),
    card("graph-retain-body", "Graph handles · CAPI and body", "capi", '''**_CAPICreate(op._CAPIPtr) creates another wrapper around the SAME native GraphOp.** CAPI means the MLIR C API used between MAX's nanobind wrapper and MLIR's Python bindings. There is no graph clone or serialized IR here.

```text
MAX GraphOp wrapper op
  → _CAPIPtr: named PyCapsule(pointer=P)
  → MlirOperation {ptr=P}
  → MLIR Python PyOperation(pointer=P, context reference)
  → existing region R0 / block B0
```

The [MAX export](sources/max/python/max/_core/max_license/_core.cpp#L606) wraps the existing operation. The pinned upstream [capsule adapter](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L326) and [factory](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1207) import the handle and recover its context. Native address P is an illustrative identity, not a measured address.

[Graph.__init__](sources/max/python/max/graph/graph.py#L874) then retains `_mlir_op`, `_current_block=regions[0].blocks[0]` and `_graph_body`. **All refer to the already-created graph/body.** G1 is cloned later by compilation, not by this pointer bridge.'''),
    card("graph-device-info", "Device metadata · Targets", "kernels", '''**Tensor placement, module target metadata, and session device handles have different roles.** [The mapping helper](sources/max/python/max/graph/graph.py#L1366) stores CPU information and, when present, one default accelerator's API/architecture/model description on the enclosing module.

| Object | Example | Consumer |
|---|---|---|
| Tensor DeviceRef | `gpu:0` | Placement and type contracts |
| Module device-info entry | GPU API/arch/model | Compiler target decisions |
| Session device list | `[GPU0,CPU0]` | Runtime driver contexts |

If the module already has this mapping, the helper returns. Missing architecture/model queries use unknown. **This metadata step neither compiles nor executes a GPU kernel.**'''),
    card("graph-tracking", "Graph bookkeeping · Weights and chains", "graph", '''**_weights={} records no external weight declarations. Q/K/V are runtime inputs.** [Initialization](sources/max/python/max/graph/graph.py#L879) also creates `_has_chain_input=False` and a lazy `_DeviceChainMap`.

The [chain map](sources/max/python/max/graph/graph.py#L123) retains G1; later device lookups can create dependency seeds. These references order effects. They are separate from matrix storage and from the session's runtime worker or stream.'''),
    card("graph-ready-chain", "Chain · Ordering token", "graph", '''**CH0 is an always-ready dependency token with no producer inputs.** [ChainCreateOp](sources/max/python/max/graph/graph.py#L885) is inserted into G1's body, wrapped as `_always_ready_chain`, then used to seed CPU ordering.

```text
CH0 = mo.chain.create(inputs=[])
device_chains[cpu:0] → CH0
user inputs → %q,%k,%v
```

CH0 is internal bookkeeping, not a fourth input tensor. Later ordered operations may advance per-device chains; the always-ready seed itself remains available.'''),
    card("graph-trailing-chain", "Last input · Tensor or chain?", "inputs", '''**The constructor checks the last existing block argument. In attention it is %v, a tensor.** [The native-type check](sources/max/python/max/graph/graph.py#L896) therefore leaves `_has_chain_input=False` and keeps the initial CPU chain.

**Subgraph comparison:** if a parent passes a trailing chain argument, the child adopts it and excludes internal chain arguments when exposing `Graph.inputs`. It does not create one chain for each Q/K/V matrix.'''),
    card("graph-package", "P1 · Executable, parsing and bytecode", "package", '''**mojo is an executable compiler driver launched in a separate OS process.** [subprocess_run_mojo](sources/Mojo/python/mojo/run.py#L57) chooses `MODULAR_MOJO_MAX_DRIVER_PATH` or falls back to `shutil.which("mojo")`, then invokes `[mojo]+args`.

| Environment | Path selection |
|---|---|
| MAX SDK/wheel | package root `bin/mojo`, unless overridden |
| Bazel runfiles | configured driver path from runtime toolchain |
| PATH fallback | the actual path returned by `shutil.which` |

The exact installed absolute executable path is environment-specific; this document has not run it. See [SDK defaults](sources/Mojo/python/mojo/run.py#L22) and [Bazel exports](sources/bazel/internal/mojo_test_environment.bzl#L178).

**Source package input:** `kernels/__init__.mojo` and referenced modules, including `fused_attention.mojo`. The Python entry file is not compiled into P1. [The helper](sources/Mojo/python/mojo/paths.py#L141) calls `mojo precompile <kernels dir> -o <private staging dir>/mojo_pkg_<path hash>.mojoc` and atomically publishes it. The staging filename uses the final basename.

**Format:** a precompiled parametric Mojo package, not CPU/GPU machine code. [The writer](sources/Mojo/lib/Support/MojoPrecompiledFile.cpp#L137) stores version/header information and compressed MLIR bytecode. A text rendering of a `lit.package` is an inspection view, not the binary bytecode itself. Parsing, declaration resolution and semantic/lifetime checks are separate from later concrete kernel specialization. No full compiler dump is claimed here.'''),
    card("graph-forward-context", "with Graph · Active recording scope", "inputs", '''**Attention uses a with-body, not a forward lambda.** After the constructor finishes importing P1, Python invokes [Graph.__enter__](sources/max/python/max/graph/graph.py#L1137), which activates the MLIR context and sets the graph construction ContextVar.

`Graph.current` resolves G1 while [ops.custom](sources/max/python/max/graph/ops/custom.py#L94) runs. Exit restores the prior graph; this is a scoped graph reference, not the module-level default MLIR Context.'''),
    card("graph-forward-inputs", "Inputs · Three symbolic wrappers", "inputs", '''**q, k, v refer to G1's three block arguments.** [Graph.inputs](sources/max/python/max/graph/graph.py#L943) wraps native SSA values as Python `TensorValue`s.

| Python | Native | Example type |
|---|---|---|
| q | %arg0 | float32 [32,32] gpu:0 |
| k | %arg1 | float32 [32,32] gpu:0 |
| v | %arg2 | float32 [32,32] gpu:0 |

These lowercase values contain shape/type/device information, not numerical matrices. Uppercase `Q/K/V` Buffers are created after init and supplied to execute.'''),
    card("graph-output-type", "Output type · Future O", "type", '''**The declared output type is float32 [N,D] on the selected device.** The [caller](sources/max/examples/custom_ops/fused_attention.py#L63) explicitly constructs it from `dtype`, `N`, `D`, and `DeviceRef.from_device(device)`.

**GPU example:** `out_types=[TensorType(DType.float32,[32,32],gpu:0)]`. **CPU example:** `[8,8]` on cpu:0. This describes the custom call's single future result; no output Buffer is allocated by declaring the type.'''),
    card("graph-custom-attributes", "Attributes · Name, device, BN/BD", "custom", '''**The registration string must match the namespaced Mojo registration exactly.** [ops.custom](sources/max/python/max/graph/ops/custom.py#L96) creates a StringAttr, and the [registered struct](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L66) supplies the definition.

| Python value | MO attribute/value example |
|---|---|
| name | `StringAttr("modular_ops::fused_attention_custom")` |
| GPU device | `DeviceRefAttr("gpu",0)` |
| GPU parameters | `BD=8:index, BN=16:index` |
| CPU parameters | `BD=4:index, BN=4:index` |

**Integer parameters use MLIR IndexType, not tensor float32.** [_parameter_attribute](sources/max/python/max/graph/ops/custom.py#L48) performs that conversion. [The parameters dictionary](sources/max/python/max/graph/ops/custom.py#L113) and [device attribute](sources/max/python/max/graph/ops/custom.py#L122) are attached after operation insertion, before validation.'''),
    card("graph-custom-insert", "mo.custom · Detailed three-input call", "custom", '''**Python records one opaque attention call; it does not run softmax or translate arbitrary Python into Mojo.** [The insertion helper](sources/max/python/max/graph/graph.py#L1284) unwraps operands, converts result types, inserts the operation in G1's body, and creates symbolic result wrappers.

```mlir
// Schematic MO: full printed dialect syntax may differ
module {
  mo.graph @fused_attention(%q:T, %k:T, %v:T) -> T {
    %ch = mo.chain.create()
    %o = mo.custom(%q,%k,%v) {
      symbol = "modular_ops::fused_attention_custom",
      parameters = {BD=8:index, BN=16:index}, device=gpu:0
    } : (T,T,T) -> T
    mo.output %o
  }
}
// T describes float32 [32,32] on gpu:0; no matrix data here.
```

The Mojo algorithm stays in P1 until specialization/JIT. **One result type produces one symbolic `%o`, even though the call has three inputs.**''',
         tree='''caller: results = ops.custom(...) [LINK:entry:57]
└─ custom(name,device,values=[q,k,v],out_types=[TO],parameters={BD:8,BN:16}) [LINK:custom_api:69]
   ├─ Graph.current → G1; StringAttr.get(name,context) [LINK:custom_api:94] [CALLER:custom_api|96]
   ├─ normalize TensorValue operands; convert output TO with to_mlir [LINK:custom_api:105] [CALLER:custom_api|110]
   ├─ Graph._add_op_get_op_with_results(mo.custom,[TO],values,symbol=...) [LINK:custom_api:109]
   │  └─ native operation CUST1 + result %o; wrappers refer to that result [LINK:ct_insert_result] [CALLER:custom_api|109]
   ├─ _parameter_attribute(8/16) → IndexType IntegerAttr; DictAttr attaches parameters [LINK:attention_parameter_attr:48] [CALLER:custom_api|116]
   ├─ device.to_mlir → Attribute._CAPICreate → custom_op.device [LINK:custom_api:122] [CALLER:custom_api|122]
   ├─ KernelLibrary.verify_custom_op(custom_op) [LINK:custom_api:127] [CALLER:custom_api|127]
   └─ return [TensorValue(%o:TO)] [LINK:custom_api:129] [CALLER:entry|57]'''),
    card("graph-custom-verify", "Kernel validation · P1 and MO", "validation", '''**Validation connects a symbolic call to a registered Mojo contract. It does not execute the kernel.** The animation shows registry lookup, device selection, input/result role mapping and parameter checks.

| Graph value | Destination-passing Mojo role |
|---|---|
| %q | query |
| %k | key |
| %v | value |
| %o (call result) | output |

[MojoRegistry](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L48) indexes declarations by registration string. [Operand inference](sources/GraphCompiler/lib/Conversion/OperandInference.cpp#L240) maps graph inputs and result destinations, including the output-first Mojo signature. Dtype/rank are inferred; target follows the device; BN/BD are checked as supplied integer parameters. See [native verifier](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L634).

**Failure example:** `name="fused_attention_typo"` has no matching entry. An extra input, mismatched rank or an unknown parameter also fails validation rather than returning a successful wrapper. `rank==2` is additionally asserted by the specialized Mojo implementation.'''),
    card("graph-custom-result", "Result wrappers · One symbolic output", "custom", '''**The returned value is a list with ONE TensorValue wrapper.** [return results](sources/max/python/max/graph/ops/custom.py#L129) returns the wrapper created during operation insertion.

```text
results = [TensorValue(native=%o,
                       dtype=float32,
                       shape=[32,32], device=gpu:0)]
graph.output(*results) → graph.output(results[0])
```

The fields above describe the wrapper conceptually; they are not a dump of private Python members. `%o` is owned by CUST1 in G1. The later `output=[Buffer O1]` from execute contains actual storage and is a different list.'''),
    card("graph-forward-output", "Graph.output · Complete G1", "output", '''**graph.output(*results) finishes and verifies G1's symbolic contract.** It records the result, adds the output terminator, and updates the declared result signature. See [the call](sources/max/examples/custom_ops/fused_attention.py#L68) and [Graph.output](sources/max/python/max/graph/graph.py#L1396).

```text
before: (TQ,TK,TV) → ()    body contains custom result %o
after:  (TQ,TK,TV) → (TO)  body ends with output(%o)
```

This graph-building finalization is distinct from later MGP device-graph finalization, kernel JIT, and runtime model initialization.'''),
    card("session-ownership", "Session · Configurations, contexts and devices", "session", '''**S1 owns settings and native runtime references as well as device handles.** [The constructor](sources/max/python/max/engine/api.py#L664) normalizes devices and calls the native session.

| Illustrative member | Original-call value |
|---|---|
| S1.num_threads | None |
| S1._impl | native N1 |
| N1.devices | [CPU0] or [GPU0,CPU0] |
| N1.sessionCompileConfig | CFG1; no model source or extension paths yet |
| N1.runtimeConfig | threads=0 (automatic), API=python, registered device/context pairs |
| N1.runtimeContext | Engine::Context E1 → shared MAX ContextRef CX1 → AsyncRT CPUDevice HRT1 |

The [native session](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L546) clones compilation settings and adds supplied session extensions. This caller supplies none: **P1 is attached to G1 and passed later by compile_async**. An extension example `/tmp/addon.mojoc` is an existing compiled package path, not a Python class.

[Context creation/reuse](sources/max/lib/max_license/API/c/EngineContext.cpp#L184) can reuse a matching global runtime CX1 across sessions. **CX1 is MAX's runtime/services context, not Graph's MLIR context.**'''),
    card("compile-passes", "Compiler · MO → MAP → MOGG → MGP", "compiler", '''**Compiler work transforms clone C1 while G1 remains retained.** [MOFContext::compileToBinary](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L381) clones the module and checks the graph-artifact cache.

| Step | Attention example | Why it is needed |
|---|---|---|
| MO | three typed tensors → namespaced custom call → one output | Verify graph/shape/device contracts |
| MAP | opaque iteration with Q/K/V accesses, custom call, O destination | Expose graph-level accesses and implementation boundaries |
| MOGG | bound FusedAttention.execute routine and mapped tensor arguments | Connect registry implementation to executable glue |
| MGP | init/execute regions, buffers, state slots, chains | Express runtime ownership/readiness and calls |
| JIT + final lowering | compiled routine archive; ModuleLoad/ExecuteGeneric; outlined callables | Make the plan callable by init/execute |

**Opaque custom boundary:** MAP does not see or fuse the softmax recurrence inside FusedAttention. Its tiling is Mojo code. Graph canonicalization may remove redundant graph structure when eligible; this example does not prove the custom operation is algebraically simplified.

The [pipeline builder](sources/GraphCompiler/lib/GraphCompiler.cpp#L199) installs MLIR passes and runs them in-process. The detailed pipeline supplement below explains why each pass is needed and labels schematic examples.''',
         tree='''caller: convertMOToMEF(C1,options) [LINK:ct_convert_mef:356]
└─ MO::runGCPipeline(C1,options) → PassManager.run [LINK:ct_run_gc:638]
   ├─ buildMoPipeline → verify/canonicalize MO contracts [LINK:attention_gc:219] [CALLER:ct_run_gc|638]
   ├─ MOToMAPPass → opaque accesses/call for custom attention [LINK:mo_to_map:764] [CALLER:ct_run_gc|638]
   ├─ buildMapPipeline → access-pattern optimization; opaque call retained [LINK:attention_gc:338] [CALLER:ct_run_gc|638]
   ├─ MAPToMOGGPass → MojoRegistry + MAPToMOGGState [LINK:map_to_mogg:53] [CALLER:ct_run_gc|638]
   ├─ buildMoggPipeline → generated functions/calls [LINK:attention_gc:370] [CALLER:ct_run_gc|638]
   ├─ MOGGToMGPPass → MOGraphLowering init/execute state [LINK:mogg_to_mgp:2406] [CALLER:ct_run_gc|638]
   ├─ buildMgpPipeline → guarded device finalization + EmitMojo [LINK:attention_gc:382] [CALLER:ct_run_gc|638]
   ├─ JitCompileKernelsPass → compiled native routines or kernel-cache reuse [LINK:attention_jit] [CALLER:ct_run_gc|638]
   └─ buildJitPipeline → ModuleLoad/ExecuteGeneric + outlined callables [LINK:jit_pipeline] [CALLER:ct_run_gc|638]
      [DETAIL:max_graph_compiler_pipeline_examples.html|Detailed pipeline · transformations and examples]'''),
    card("compile-kernels", "Kernel · Mojo, LLVM and target artifacts", "kernels", '''**JIT means just-in-time compilation: code generation requested by the running MAX process.** This example calls `session.compile(graph)` before supplying numeric buffers. The kernel code is prepared there (or reused from a cache), rather than running `mojo build` for every matrix execution.

```text
P1 parametric FusedAttention.execute
 → dtype=float32,rank=2,BN=16,BD=8,target=gpu
 → KGEN elaboration/specialization
 → MLIR LLVM dialect (still MLIR)
 → LLVM1: llvm::Module (LLVM IR)
 → host object + selected device code
```

[KGENCompiler](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) runs elaboration. [ObjectCompiler](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816) lowers and translates into LLVM IR. The compiler-side [TargetBackendRegistry](sources/Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp#L114) selects a registered backend by target triple; architecture/features configure target codegen.

**NVIDIA example:** `nvptx64-nvidia-cuda`, illustrative arch `sm_90a`, [NVPTX backend](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L1) → PTX and device binary route. **AMD example:** `amdgcn-amd-amdhsa`, illustrative arch `gfx942`, [AMDGPU backend](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77) → AMDGPU object/ELF route. The [cross-target source test](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1) supplies both architecture examples.

The host Python/MAX runtime and compiled host glue stay on CPU. Device kernels have separate offload targets. **P1 .mojoc preparation is a distinct earlier subprocess.** Architecture examples are not GPU detection or execution results.'''),
    card("compile-mef", "Artifact · Prepared CM1 bytes", "compiler", '''**CM1 is a compiled artifact, not an initialized executable Model.** [convertMOToMEF](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356) runs graph/JIT work and exports the runtime plan and code bytes.

```text
CM1 ingredients: MGP plan + model descriptors + compiled routine archive
init(CM1): load symbols / initialize resources / create M1
execute(Q,K,V): invoke M1 with actual data
```

Graph-artifact cache reuse and kernel-code cache reuse are separate. They skip different work; neither makes the Python graph a machine-code file or computes Q/K/V output during construction.'''),
    card("init-resources", "Initialization · Loaded routines and resources", "init", '''**Initialization turns descriptors and compiled bytes into retained runtime state and callable handles.** [Model::setup](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L185) builds initializer arguments, resolves devices, invokes the initializer, and constructs `ExecutableModel(chain,mgpContext,weightsRegistry,extraArgs)`.

| Retained value | Example role |
|---|---|
| chain | asynchronous setup readiness |
| mgpContext | runtime state/module/function slots |
| weightsRegistry | empty external weights here |
| extraArgs | device and initialization state needed by execute |

**Why a CPU worker?** [Engine::Context::initModel](sources/max/lib/max_license/API/c/EngineContext.cpp#L297) schedules setup on the CPU runtime so the task has the correct AsyncRT CPUDevice association. Setup may depend on asynchronous resources; the caller waits for readiness instead of pretending initialization is already complete.

**Host archive loading is separate from driver launch.** `CompiledKernelsContext::load` hashes an archive, installs/reuses its layer, and resolves named native callable pointers. Later host routine execution reaches GPU module/function loading and launch. Use the directly linked loader supplement for that deeper tree and animation.''',
         tree='''caller: initFn->invoke(initArgs,cpuDevice) [LINK:ct_model_setup:220]
├─ mgp_module_load_cpu → runtime module slot [LINK:ct_module_load]
│  └─ CompiledKernelsContext::load(archive_bytes,symbols) [LINK:ct_kernel_load]
│     ├─ hash archive → addIfAbsent<StaticArchiveLayer> [LINK:ct_kernel_load:91] [LINK:ct_kernel_load:160] [CALLER:ct_kernel_load|160]
│     └─ lookup symbols → resident native callable pointers [LINK:ct_kernel_load:165] [CALLER:ct_kernel_load|165]
│        [DETAIL:max_compiled_kernels_context_load_and_driver.html|Archive loading → native symbols → device driver · detailed animation]
└─ ExecutableModel(chain,mgpContext,weightsRegistry,extraArgs) [LINK:ct_model_setup:237]
   └─ waitForModelSetup → readiness/error checks before M1 is returned [LINK:ct_wait_setup] [CALLER:ct_model_setup|237]'''),
    card("init-gates", "Initialization · Bindings and readiness", "init", '''**The attention model has no external weights; Q/K/V are inputs supplied at execute.** [init_all](sources/max/python/max/engine/api.py#L1202) gets `weights_registry_real={}` and checks the expected weight names/devices: also `{}` here.

The real-device route waits for compiled and initialization AsyncValues. Virtual-device mode returns CompileOnlyModel stand-ins that cannot execute. A successful compile is therefore distinct from a ready M1.'''),
    card("example", "E · Buffers, execution and output", "execute", '''**Uppercase Q/K/V become real Buffers after compile and init.** The [original call](sources/max/examples/custom_ops/fused_attention.py#L82) passes three inputs and receives `output=[O1]`.

| Phase | Inputs | Result |
|---|---|---|
| Graph recording | q/k/v symbolic SSA wrappers | symbolic %o |
| compile | G1/P1/types/parameters | CM1 prepared artifact |
| init | CM1/S1 contexts | M1 callable state |
| execute | numeric Q/K/V Buffers | one numeric O1 Buffer |

**Tiny teaching row:** Q=[1,0], scores=[1,0,3,2]. Two tiles update m/l/O; the second maximum grows 1→3. The online-softmax card shows actual browser numbers at every update.

The original source prints the output Buffer list. It has **no explicit host copy and no numerical parity assertion**. Browser reference agreement is an independent tiny arithmetic illustration, not a measured MAX/GPU result.'''),
    card("kernel-dispatch", "Dispatch · CPU/GPU specialization", "kernels", '''**The graph device becomes the compile-time target for the registered routine.** [FusedAttention.execute](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L74) receives output/query/key/value/context in destination-passing form.

```text
dtype=float32 and rank=2 ← tensor types
BN/BD ← Python custom parameters
target="cpu" or "gpu" ← graph device
ctx ← runtime device context
```

The [CPU branch](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) invokes native CPU loops. The [GPU branch](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L105) calls the enqueue wrapper. Specialization removes the unselected compile-time branch; NVIDIA/AMD selection is a lower target-backend decision.'''),
    card("kernel-launch", "Launch · Blocks and driver handoff", "launch", '''**GPU grid=(4,2), block=32 for this caller's sizes.** [The wrapper](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L391) specializes the kernel and passes Q/K/V/O with geometry to `ctx.enqueue_function`.

| Coordinate | Ownership |
|---|---|
| block_idx.y ∈ [0,1] | one group of 16 query rows |
| block_idx.x ∈ [0,3] | one group of 8 output columns |
| eight blocks | cover O [32,32] as 16×8 tiles |
| key loop t ∈ [0,3] | visits four groups of 8 keys |

Host launch metadata, native callable address, GPU function handle and tensor addresses are different values. Driver loading/submission/readiness is described in the linked loader lesson; this page does not run a GPU or claim a measured schedule.'''),
    card("kernel-mma", "MMA · Loads, accumulation and barrier", "mma", '''**S is computed from operand fragments, not a saved full attention matrix.** [GPU matmul](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L234) creates a TensorCore helper with BK=8, a LOCAL accumulator, and SHARED output storage.

```text
Q_tile [16,32] × K_tileᵀ [32,8] → S [16,8]
features split into four BK=8 slices
load_a / load_b → mma_op(A,B,C) → retained C accumulator
store_d(out_sram,C) → barrier() → res.copy_from(out_sram)
```

The same helper is used for E×V in the output update. **barrier() is a block synchronization point for the shared temporary; it is not the global model result-readiness gate.** Exact hardware instructions/residency depend on the target backend and are not measured here.'''),
    card("kernel-storage", "Storage · Current tiles and retained state", "storage", '''**The fused loop keeps one score/weight tile and per-query running state.** Source [tile selection](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) and [updates](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L376) establish the logical shapes.

| GPU block object | Logical shape | Lifetime |
|---|---|---|
| Q tile view | [16,32] | all key iterations for this block |
| K/V views | [8,32] / [8,8] | current key tile |
| S/E temporaries | [16,8] | current key tile |
| running m/l | [16] each | retained between key tiles |
| normalized O_i | [16,8] | retained, then stored once |

Q/K/V tile selection creates views. **The source does not allocate a persistent full N×N score matrix inside this loop.** Logical storage counts do not prove physical register/shared/local memory placement or total peak allocations.'''),
    card("kernel-online", "Online softmax · Two tiles and rescaling", "online", '''**Each key tile updates one stable max, denominator, and normalized output per query row.** The source computes an unscaled, rank-2 `softmax(QKᵀ)V`: no heads, batch, mask, dropout, or 1/√D scale.

```text
m_new = max(m_old, max(S_tile))
alpha = exp(m_old - m_new)
E = exp(S_tile - m_new)
l_new = alpha*l_old + sum(E)
O_new = O_old * alpha*l_old/l_new + (E/l_new) @ V_tile
```

**Why rescale?** When a later maximum grows, earlier exponentials were measured against an older maximum. Multiplying their retained mass by alpha expresses them relative to the new maximum. Multiplying old normalized O by `alpha*l_old/l_new` also adjusts for the new denominator.

**Tiny values:** Q=[1,0]; K=[[1,0],[0,1],[3,0],[2,1]]; V=[[1,0],[0,2],[2,1],[1,3]]. Tile scores are [1,0], then [3,2]. The animation computes all terms and checks against an independent stable full-softmax result. Teaching tile size is 2; source GPU key size is 8.

Source initializes m with the dtype's minimum value; the teaching model uses mathematical −∞ for readability. See [max](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L348), [denominator](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L352), and [output update/store](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L366).'''),
]


def render_attention_calltrees(refs, revision, src):
    from compiler_attention_mechanisms import render_attention_mechanism
    from compiler_capi_conversion import render_capi_conversion
    from compiler_toolchain import color_qna_text

    reader = CallTreeReader(refs)
    def portable_tree(text):
        sources = {}
        def token(m):
            key = m[1] or m[4]
            if key:
                r = refs[key]
                line = int(m[2] or r['line']) if m[1] else int(m[5])
                sources[(key, line)] = r
                return f"{'↗ caller' if m[4] else '↗'} [{r['alias']}:L{line}]"
            return m[7] if m[6] else m[9]
        plain = TOKEN.sub(token, text)
        links = ' · '.join(f'[{r["file"]}:L{line}](sources/{r["file"]}#L{line})' for (key, line), r in sources.items())
        return '```text\n' + plain + '\n```\n\nSource links: ' + links
    panels = attention_session_panels()
    body = ['<div class="attention-calltrees" data-calltrees="attention">',
            '<p class="calltree-scope"><strong>SOURCE</strong> · '
            f'Revision <code>{revision[:12]}</code>. Follow actual source order; cards open beside the tree. '
            'G1/P1 are symbolic/compiler objects; numeric Q/K/V arrive only at execute.</p>',
            '<article class="call-stack-copy"><h2>What happens when with Graph(...) runs</h2>',
            '<p>The constructor creates the graph shell and prepares P1. The following with-body '
            'records and verifies one three-input attention call; no forward lambda is used.</p>',
            reader.tree(GRAPH_TREE, 'graph-constructor-copy', 'Attention Graph construction', 'entry', 41), '</article>']
    portable = ['# Fused attention: source-ordered call stacks and mechanisms', '',
                f'SOURCE · revision {revision}. Schematic examples; no MAX/GPU execution.', '', portable_tree(GRAPH_TREE), '']
    for panel in panels:
        body.append(f'<article class="call-stack-copy session-call-stack" id="{panel["id"]}-section">'
                    f'<h2>{html.escape(panel["title"])}</h2><p class="callsite">'
                    + reader.source('entry', panel['line'], panel['call']) + '</p><p>'
                    + html.escape(panel['intro']) + '</p>'
                    + ('<pre class="example-code"><code>' + html.escape(panel['code']) + '</code></pre>' if panel.get('code') else '')
                    + reader.tree(panel['tree'], panel['id'], panel['title'], 'entry', panel['line']) + '</article>')
        portable.extend(['## ' + panel['title'], '', portable_tree(panel['tree']), ''])
    body.append('<article class="call-stack-copy" id="attention-kernel-section"><h2>Kernel · Dispatch, tile arithmetic and storage</h2>'
                '<p>Compiled code uses the registered destination-passing routine. The GPU route and CPU comparison follow the same online recurrence.</p>'
                + reader.tree(KERNEL_TREE, 'attention-kernel', 'Attention kernel dispatch and updates', 'attention_registration', 74) + '</article>')
    portable.extend(['## Kernel · Dispatch, tile arithmetic and storage', '', portable_tree(KERNEL_TREE), ''])
    launcher = ('<div class="call-card-launchers call-example"><button type="button" hidden class="call-card-launch" '
                'id="toolchain-call-example-launch" data-call-card-launch data-call-explanation="toolchain-call-example">'
                '<span class="call-card-number">E</span><span>Three buffers → execution → output</span>'
                '<span class="call-card-glyph" aria-hidden="true">↗</span></button></div>')
    body.append(launcher + '<div class="call-stack-card-storage">')
    for item in CARDS:
        animation = (render_capi_conversion(refs, graph_name='fused_attention', inputs='%q / %k / %v')
                     if item['mechanism'] == 'capi' else render_attention_mechanism(item['mechanism'], refs, src))
        content = animation + reader.explanation(item['explanation'])
        if item['tree']:
            content += reader.tree(item['tree'], item['id'] + '-tree', item['title'], item['source'], item['line'])
        body.append(f'<details class="call-branch" data-call-card id="{item["id"]}" '
                    f'data-explanation-title="{html.escape(item["title"], quote=True)}">'
                    f'<summary>{html.escape(item["title"])}</summary><div class="call-card-content">{content}</div></details>')
        portable.extend(['### ' + item['title'], '', item['explanation'], '', portable_tree(item['tree']) if item['tree'] else '', ''])
    body.append('</div><p><a href="../ATTENTION_CALL_STACKS.md" download>Download the portable attention call trees</a></p></div>' + EXPLANATION_WINDOW)
    text = '\n'.join(portable)
    metadata = dict(stages=['graph','session','compile','init','execute','kernel'], branches=len(CARDS),
                    cards=[c['id'] for c in CARDS], references=sorted(reader.used_refs),
                    caller_link_counts=reader.caller_counts, caller_links=sum(reader.caller_counts.values()),
                    source_link_counts=reader.source_counts,
                    reading_order=['graph shell → P1','with-body → verified G1','S1','C1 → CM1','M1','Q/K/V → O1'],
                    separate_call_stack_sections=['graph-constructor-copy', *[p['id'] for p in panels], 'attention-kernel'],
                    text_sha256=hashlib.sha256(text.encode()).hexdigest())
    return color_qna_text(''.join(body)), text, metadata
