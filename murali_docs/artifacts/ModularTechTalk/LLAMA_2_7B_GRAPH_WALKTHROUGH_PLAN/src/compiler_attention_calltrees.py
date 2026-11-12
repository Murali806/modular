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
    ("attention_analysis_module", "max/python/max/_core/max_license/lib/Graph/Bindings.cpp", "mlir::ModuleOp M::Graph::Python::Analysis::ensureModule()", "cpp", 0, "Lazily import package declarations into the analysis module"),
    ("attention_validation_parameters", "GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp", "static ErrorOrSuccess validateCustomOpParameters", "cpp", 0, "Complete compile-time parameter union, lookup, native type checks and success return"),
    ("attention_ctor_decorator", GRAPH, "817:819", "lines", 0, "Decorator guarding Graph construction on background threads"),
    ("attention_gc", "GraphCompiler/lib/GraphCompiler.cpp", "199:677", "lines", 0, "Graph-stage builders and pass-manager execution"),
    ("attention_jit", "GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp", "529:594", "lines", 0, "JitCompileKernelsPass invokes native kernel compilation"),
    ("attention_jit_misses", "GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp", "434:503", "lines", 0, "Cache misses elaborate KGEN, emit an archive and install an init resource"),
    ("attention_jit_target", "GraphCompiler/lib/MOGGDialect/Support/MojoStubCompiler.cpp", "118:159", "lines", 0, "Host target description and separate accelerator architecture"),
    ("attention_offloads", "Mojo/lib/Compiler/KGENCompiler.cpp", "489:911", "lines", 0, "Group and slice device offloads, choose per-target compilers and emit kernel payloads"),
    ("attention_host_lower", "Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp", "817:855", "lines", 0, "Run LLVM-dialect lowering then translate the module to LLVM IR"),
    ("attention_runtime_lower", "GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp", "102:168", "lines", 0, "Lower compiled routines to ModuleLoad and ExecuteGeneric"),
    ("attention_graph_output", GRAPH, "1426:1495", "lines", 0, "Output flattening, complete signatures, terminator, names and graph verification"),
    ("attention_stack", "max/kernels/src/layout/tile_tensor.mojo", "3942:3988", "lines", 0, "Logical TileTensor stack allocation defaults to GENERIC address space"),
    ("attention_enqueue", "max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo", "52:191", "lines", 0, "DeviceContext checked explicit-function enqueue and argument-pack handoff"),
    ("attention_load_function", "max/mojo/max/gpu/host/_legacy/device_context.mojo", "2994:3058", "lines", 0, "Device-function payload loading through the native runtime binding"),
    ("attention_pack", "max/mojo/max/gpu/host/_legacy/device_context.mojo", "3503:3681", "lines", 0, "DevicePassable validation, aligned argument slots and packed enqueue"),
    ("attention_packed_tail", "max/mojo/max/gpu/host/_legacy/device_context.mojo", "2924:2966", "lines", 0, "Packed launch tail calls the function enqueuer and checks errors"),
    ("attention_driver_load", "MLRT/lib/Driver/DeviceContext/MojoBindings.cpp", "504:515", "lines", 0, "Native device-function load binding"),
    ("attention_driver_enqueue", "MLRT/lib/Driver/DeviceContext/MojoBindings.cpp", "542:557", "lines", 0, "Native device context direct-enqueue binding"),
    ("attention_cuda_launch", "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp", "1573:1615", "lines", 0, "CUDA device-function launch config and cuLaunchKernelEx"),
    ("attention_hip_launch", "MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp", "1365:1387", "lines", 0, "HIP device-function launch and hipModuleLaunchKernel"),
]

GRAPH_TREE = '''caller: with Graph("fused_attention", input_types=[TQ,TK,TV], custom_extensions=[kernels/]) as graph [LINK:entry:41] [CALLER:entry|41]
├─ Evaluate caller arguments before Graph.__init__ [LINK:entry:43]
│  ├─ dtype=DType.float32; device=CPU() or Accelerator() [LINK:entry:26] [LINK:entry:29] [CALLER:entry|29]
│  ├─ CPU: N=D=8, BN=BD=4; GPU: N=D=32, BN=16, BD=8 [LINK:attention_sizes:30] [CARD:toolchain-call-graph-sizes|Sizes · Actual CPU/GPU contracts] [CALLER:entry|30]
│  └─ TQ, TK, TV: TensorType(dtype, shape=[N,D], device=DeviceRef.from_device(device)) [LINK:entry:45] [CALLER:entry|45]
│     OUT: three type descriptions; no NumPy matrices or Buffer objects yet [LINK:entry:44] [CALLER:entry|43]
│
├─ Graph.__init__(name="fused_attention", input_types=[TQ,TK,TV], forward=None) [LINK:graph_init:820]
│  ├─ @in_default_mlir_context → ensure an active MLIR context on this thread [LINK:ct_graph_context:70] [CARD:toolchain-call-graph-context|A.0 · default MLIR context] [CALLER:attention_ctor_decorator|819]
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
│  │     OUT: %q:TQ, %k:TK, %v:TV in the same entry block; results initially empty [LINK:ct_graph_op_build:849] [CALLER:ct_graph_op_build|849]
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
   ├─ Graph.__enter__ → _enter → set Graph.current=G1 [LINK:ct_graph_context_manager:1134] [CARD:toolchain-call-graph-forward-context|with Graph · Active recording scope] [CALLER:entry|55]
   ├─ q, k, v, *_ = graph.inputs [LINK:entry:56] [CARD:toolchain-call-graph-forward-inputs|Inputs · Three symbolic wrappers]
   │  └─ Wrap body.arguments → q→%arg0, k→%arg1, v→%arg2; no numeric data [LINK:ct_graph_inputs:943] [CALLER:entry|56]
   ├─ Declare one output TensorType(dtype, [N,D], device) [LINK:entry:63] [CARD:toolchain-call-graph-output-type|Output type · Future O] [CALLER:entry|62]
   ├─ results = ops.custom(name="modular_ops::fused_attention_custom", ...) [LINK:entry:57] [CARD:toolchain-call-graph-custom-insert|mo.custom · Detailed three-input call]
   │  ├─ Resolve Graph.current; name → StringAttr; normalize DeviceRef [LINK:custom_api:94] [CARD:toolchain-call-graph-custom-attributes|Attributes · Name, device, BN/BD] [CALLER:entry|57]
   │  ├─ values=[q,k,v] → existing %arg0,%arg1,%arg2; out_types=[TO] → native result type [LINK:custom_api:105] [CALLER:entry|61]
   │  ├─ Insert mo.custom in G1.body; wrap its one symbolic result %o [LINK:custom_api:109] [CALLER:custom_api|109]
   │  ├─ Attach parameters AFTER insertion: BD/BN → index attributes → dictionary [LINK:custom_api:114] [LINK:attention_parameter_attr:49] [CALLER:entry|60]
   │  ├─ Attach DeviceRefAttr via MLIR Attribute capsule [LINK:custom_api:122] [CALLER:custom_api|122]
   │  ├─ Verify call against loaded P1; resolve FusedAttention.execute and map types/parameters [LINK:custom_api:127] [CARD:toolchain-call-graph-custom-verify|Kernel validation · P1 and MO] [CALLER:custom_api|127]
   │  └─ Return results=[TensorValue(%o: float32[N,D], selected device)] [LINK:custom_api:129] [CARD:toolchain-call-graph-custom-result|Result wrappers · One symbolic output] [CALLER:entry|57]
   ├─ graph.output(*results) → mo.output, result signature, graph verification [LINK:entry:68] [CARD:toolchain-call-graph-forward-output|Graph.output · Complete G1] [CALLER:entry|68]
   └─ Graph.__exit__ restores prior Graph.current [LINK:ct_graph_context_manager:1138] [CALLER:entry|55]
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
         ├─ validateInputSpecList → count=3, shapes/dtype match compiled signature [LINK:ct_execute_native:461]
         │  └─ Expected CPU example: three float32[8,8]; wrong count or static size raises [LINK:ct_execute_specs:438] [CALLER:ct_execute_native|461]
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


from compiler_attention_lessons import all_lessons

CARDS = all_lessons()


def render_attention_calltrees(refs, revision, src):
    from compiler_attention_mechanisms import render_attention_mechanism
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
        animation = render_attention_mechanism(item, refs, src)
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
