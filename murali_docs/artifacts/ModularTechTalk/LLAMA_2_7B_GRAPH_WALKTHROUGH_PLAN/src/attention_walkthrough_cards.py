"""Focused explanations for the fresh attention reader; all examples are schematic."""


def card(id, title, intro, tree, body, animation):
    return dict(id=id, title=title, intro=intro, tree=tree.strip(),
                body=body.strip(), animation=animation)


CARDS = [
card('package', 'P1 · From Mojo source to a precompiled package',
     'One subprocess prepares reusable package IR. Concrete GPU instructions are generated later.',
'''Graph.__init__(custom_extensions=[kernels/]) [[graph:910]]
└─ _import_kernels(_resolved_custom_extensions(...)) [[graph:1703]]
   ├─ KernelLibrary.load_paths → classify each path [[graph:269]]
   │  ├─ existing .mojoc → record path directly [[graph:285]]
   │  └─ Mojo source directory → _build_mojo_source_package [[graph:289]]
   │     ├─ hash absolute source path; create per-user destination directory [[paths:147]]
   │     ├─ private .staging.<pid>/ with the SAME final basename [[paths:165]]
   │     ├─ subprocess_run_mojo(["precompile", kernels/, "-o", staging_path]) [[paths:178]]
   │     │  ├─ derive environment from SDK defaults + process environment [[mojo-run:69]]
   │     │  ├─ executable: MODULAR_MOJO_MAX_DRIVER_PATH or which("mojo") [[mojo-run:70]]
   │     │  └─ subprocess.run([absolute_executable, ...], check=True) [[mojo-run:74]]
   │     ├─ invokeMojoParser → LIT::importMojoPackage [[precompile:280]] [[precompile:290]]
   │     ├─ buildPrecompiledPackage produces serializable lit.package [[precompile:308]]
   │     ├─ KGENCompiler::runCheckLITPipeline → semantic/lifetime checks [[precompile:318]]
   │     ├─ writePrecompiledFile → serialized MLIR bytecode [[precompile:322]]
   │     └─ os.replace(staging_path,tmp_path); clean staging directory [[paths:179]]
   ├─ retain the returned P1 path in KernelLibrary [[graph:289]]
   └─ store paths on mo.graph for later compilation [[graph:1706]]''',
'''### What is the executable?

`mojo` is a native command-line compiler driver. The subprocess helper first checks `MODULAR_MOJO_MAX_DRIVER_PATH`, then searches `PATH`. An installed SDK supplies a default `<package root>/bin/mojo`; Bazel maps its configured `MOJO_BINARY_PATH` to that environment variable. These are source-defined resolution rules, not a measured absolute path from a successful run of this example. [[mojo-run:45]] [[mojo-run:70]] [[py-rule:64]] [[env:195]]

### Concrete path example

```
input:  /checkout/max/examples/custom_ops/kernels/
output: /tmp/.modular_<uid>/mojo_pkg/mojo_pkg_<path-hash>.mojoc
stage:  /tmp/.modular_<uid>/mojo_pkg/.staging.<pid>/mojo_pkg_<path-hash>.mojoc
```

The basename must match because the compiler derives package identity from it. The final atomic rename exposes complete bytes to concurrent readers. The hash keys the source path, not a content-addressed “already compiled” check; this source-directory branch still launches precompile. [[paths:158]]

### What P1 contains and who reads it

`kernels/__init__.mojo` and its imported modules form the package. P1 retains declarations and parametric bodies in post-parse IR. The loaded analysis finds the registration name `modular_ops::fused_attention_custom`. Later KGEN specializes the implementation with dtype=float32, rank=2, BN=16, BD=8 and target=gpu. P1 is a binary MLIR serialization; the schematic package shown in the animation is a readable explanation, not a captured bytecode dump. [[package-root:1]] [[kernel:67]] [[walkthrough:599]] [[jit:547]]

The host Python program waits for this subprocess to succeed. A failed precompile raises `MojoCompilationError`, so a valid G1 is not returned with half-written package bytes. [[paths:180]]''', 'package'),

card('context', 'MLIR context · Shared object, current thread',
     'The global context object and the current-context slot are different things.',
'''module import: _DEFAULT_MLIR_CONTEXT = mlir.Context() [[context:26]]
├─ enter it on the importing/main thread; arrange exit at shutdown [[context:32]]
└─ default_mlir_context() returns this same Python context object [[context:41]]

Graph.__init__ is decorated with @in_default_mlir_context [[graph:819]]
└─ wrapper → with ensure_default_mlir_context() [[context:82]]
   ├─ Context.current already set on this thread → keep it [[context:66]]
   └─ no context on this thread → enter _DEFAULT_MLIR_CONTEXT for this call [[context:63]]
      └─ construct IR; exit the scoped context after the call [[context:83]]

MLIRThreadPoolExecutor.submit(fn,...) is an optional helper [[context:90]]
└─ worker runs call_with_default_mlir_context(fn,...) [[context:98]]
   └─ enter the default only if its thread has no current context [[context:48]]''',
'''### Three names, three roles

| Name | Example value | Meaning |
|---|---|---|
| `_DEFAULT_MLIR_CONTEXT` | Context object CTX1 | Module-global Python reference to an MLIR context |
| `mlir.Context.current` | CTX1 on main; initially None on a fresh worker | Thread-local active-context lookup |
| `MLIRThreadPoolExecutor` | Executor E1 with worker threads | Runs functions with a suitable context scope |

The context manages compiler entities such as registered dialects, types and attributes. Reusing it lets IR values share compatible compiler infrastructure. The decorator is Python syntax that replaces the function with a wrapper; it is not a field on G1. A background thread is an ordinary worker thread, separate from the main Python thread, not a GPU thread. [[context:26]] [[context:70]]

### Example: the same graph builder on a worker

```
main thread                 fresh worker thread
current = CTX1              current = None
                            wrapper enters CTX1
                            Graph constructor creates IR in CTX1
                            wrapper exits → current = None
```

`MLIRThreadPoolExecutor.submit(build_graph)` supplies this scope around the submitted function. The exact example calls Graph from `main()` and does not create that executor. The decorator also handles callers that already entered another context: it preserves that active context. Neither this MLIR object nor its executor is the MAX runtime Context or a CUDA context. [[entry:41]] [[context:47]]''', 'context'),

card('module', 'G1 storage · Module, graph, block and insertion point',
     'MLIR’s module is an IR container. Three tensor inputs become three arguments in one graph block.',
'''with _location() as loc: [[graph:851]]
├─ source tracebacks off → Location.unknown(); on → Python frame location [[graph:438]]
├─ module= supplied → reuse module.mlir_module [[graph:856]]
│  otherwise builtin.ModuleOp(location=loc) → native ModuleOp::create [[builtin:65]]
├─ OpBuilder(self._module.body.end) selects the insertion position [[graph:858]]
└─ _mo.GraphOp(builder,loc,name,input_types,result_types=[]) [[graph:860]]
   ├─ name: "fused_attention", from the caller [[entry:42]]
   ├─ input_types: [TQ.to_mlir(),TK.to_mlir(),TV.to_mlir()] [[graph:864]]
   │  └─ mo.TensorType(shape,dtype,device,metadata) [[type:439]]
   ├─ function signature: (T,T,T) → () before graph.output [[graph-op:831]]
   ├─ create body region; add ONE block with THREE arguments [[graph-op:846]]
   └─ insert graph operation at the selected module position [[graph:860]]

self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr) [[graph:874]]
├─ C API capsule transfers the native-operation identity across bindings [[graph:874]]
└─ _graph_body = _mlir_op.regions[0].blocks[0] [[graph:875]]''',
'''### What loc records

`loc` is provenance metadata attached to the module, graph operation and graph input arguments. With traceback capture disabled, it is unknown. When enabled, `_location()` derives it from the Python call stack; profiling scopes can add metadata. Example provenance can include `fused_attention.py:41 → Graph.__init__`, allowing a later diagnostic to point back to the Python graph builder. This is a schematic stack example, not a captured trace. A location is separate from the context that owns IR types. [[graph:438]] [[graph-op:848]]

### Before and after insertion

```
before GraphOp                   after GraphOp
builtin.module                   builtin.module
└─ region                        └─ region
   └─ body block                    └─ body block
      └─ [insertion cursor]            └─ mo.graph @fused_attention
                                           └─ region
                                              └─ B0(%q:T,%k:T,%v:T)
```

The module groups definitions for compiler processing. Reusing a module can hold several named graphs. G1's region contains the computation, and its entry-block arguments model values provided by the caller. It is **one block with three arguments**, not one block for each input. The signature makes the graph interface explicit for verification and later runtime input checking. [[graph-op:831]] [[graph-op:846]]

### Type conversion, with actual example values

```
Python: TensorType(DType.float32, [32,32], DeviceRef("gpu",0))
                         ↓ to_mlir()
MO type fields: shape=[32,32], dtype=float32, device=gpu:0, metadata={}
```

Shape conversion produces dimension attributes; dtype is the existing `DType.float32`; DeviceRef becomes a device attribute. Optional layout metadata is added only when `_layout` exists. This caller supplies no layout override. The conversion records a type description and allocates no 4096-byte tensor. [[type:430]]

### Why the C API capsule appears

`op` is produced by MAX's native binding family, while later Python graph helpers use the MLIR Python binding family. `_CAPIPtr` exposes a capsule for the native operation and `_CAPICreate` creates a wrapper for that same operation. `_graph_body` is an alias of the existing first block. These references preserve identity; this step does not clone IR or serialize a file. The compiler clone is created later in `MOFContext::compileToBinary`. [[graph:874]] [[frontend:381]]''', 'module'),

card('custom', 'mo.custom · Record, bind and verify the call',
     'The registration name and attributes connect symbolic Q/K/V to the Mojo implementation in P1.',
'''ops.custom(name,device,values,out_types,parameters) [[entry:57]]
├─ Graph.current → G1; default_mlir_context() supplies attribute context [[custom:94]]
├─ StringAttr.get("modular_ops::fused_attention_custom") [[custom:96]]
├─ DeviceRef.from_device(GPU0) → DeviceRef("gpu",0) [[custom:97]]
├─ reject mutable-buffer/opaque inputs in this value-style API [[custom:99]]
├─ normalize values → symbolic TensorValue wrappers [%q,%k,%v] [[custom:105]]
├─ _add_op_get_op_with_results(mo.custom,[T],values,symbol=...) [[custom:109]]
│  └─ insert one native mo.custom in G1's current block [[custom:109]]
├─ parameters → DictAttr{BD:8:index, BN:16:index} [[custom:114]]
│  └─ _parameter_attribute(int) → IntegerAttr(IndexType, value) [[custom:49]]
├─ device → native m.DeviceRefAttr(kind="gpu",id=0) [[custom:122]]
├─ KernelLibrary.verify_custom_op(custom_op) [[custom:127]]
│  ├─ path-set analysis supplies P1 declarations / MojoRegistry [[graph:307]]
│  └─ check name, parameter inference and input/result contract [[graph:325]]
└─ return results → [TensorValue(%o)] [[custom:129]]''',
'''### Input → recorded operation → result

```
name        = "modular_ops::fused_attention_custom"
device      = gpu:0
parameters  = {BD:8, BN:16}
operands    = [%q, %k, %v]        each T = float32[32,32] on gpu:0
out_types   = [T]

G1 body B0(%q:T,%k:T,%v:T):
  %o:T = mo.custom(%q,%k,%v)
         symbol="modular_ops::fused_attention_custom"
         parameters={BD=8:index,BN=16:index}, device=gpu:0
  mo.output(%o)

results = [TensorValue(%o)]
results[0].dtype = DType.float32
results[0].shape = Shape([32,32])
results[0].device = DeviceRef("gpu",0)
```

This is schematic readable IR and wrapper state, not a compiler dump. `%o` names the future result; it has no numerical entries yet. `graph.output(*results)` declares this one result as the graph's output. [[entry:63]] [[entry:68]]

### How verification uses P1

The registration maps the string name to `FusedAttention.execute`. Input tensor types infer dtype=float32 and rank=2. Named parameters supply BN and BD; the device determines target=gpu. The implementation's output parameter is an `OutputTensor`, followed by query/key/value `InputTensor`s and a `DeviceContext`. These roles explain how one graph result becomes an output buffer for the native routine. No call to the numerical kernel occurs during this check. [[kernel:67]] [[kernel:74]]

The analysis cache can reuse the declaration registry for the same library paths. It still verifies each custom call. For example an unknown registration name stops resolution; a missing or incompatible required parameter stops parameter matching. The animation shows the name lookup and contract binding before the verified result is exposed. [[graph:307]] [[graph:325]]''', 'custom'),

card('pipeline', 'IR1 · Why MO → MAP → MOGG → MGP → MEF',
     'Different representations answer different questions. The custom kernel body remains Mojo until kernel specialization.',
'''MOFContext::compileToBinary(IR1) [[frontend:377]]
└─ convertMOToMEF → MO::runGCPipeline [[frontend:356]] [[gc:638]]
   ├─ buildMoPipeline: import/verify only if declarations were not seeded [[gc:227]]
   │  └─ CSE + canonicalization + chain simplification [[gc:242]]
   ├─ MOToMAPPass: expose access and computation structure [[moto-map:764]]
   ├─ buildMapPipeline: canonicalize and fuse eligible patterns [[gc:338]]
   ├─ MAPToMOGGPass: MojoRegistry(module) + MAPToMOGGState(registry,op) [[mapto-mogg:57]]
   │  └─ lower each IterOpaqueOp into implementation-oriented operations [[mapto-mogg:66]]
   ├─ buildMoggPipeline: refine generated calls [[gc:370]]
   ├─ MOGGToMGPPass: materialize model init/execute and runtime resources [[moggto-mgp:2406]]
   ├─ buildMgpPipeline: marked device graphs + EmitMojo [[gc:382]]
   ├─ JitCompileKernelsPass: generated Mojo + P1 imports → native resources [[jit:547]]
   ├─ LowerCompiledRoutines: load/call indexes become runtime primitives [[lower-routines:102]]
   └─ serializeToMEF: runtime plan plus resources → bytes owned by C1 [[frontend:371]]''',
'''| Form | Schematic example for this call | Why needed |
|---|---|---|
| MO | `%o = custom(%q,%k,%v), T→T, GPU0, BD=8,BN=16` | Check graph interface and operation contract |
| MAP | typed access boundaries around an opaque computation consuming Q/K/V | Analyze permitted access/fusion without inventing the custom kernel's internals |
| MOGG | selected registered implementation, inferred tensor roles and typed generator call | Connect the graph operation to Mojo code generation |
| MGP | initialize runtime slots; allocate/borrow buffers; order an execution call | Represent resource and execution dependencies |
| Post-JIT MGP | `ModuleLoad(A1,symbols)` + `ExecuteGeneric(slot,module,index)` | Load reusable host routines and invoke them by indexes |
| MEF | serialized functions, primitive references and code resources | Retain/transport a runtime program independently of compiler IR |

### Simplifying MO: concrete comparisons

Canonicalization can remove a supported identity operation, and common-subexpression elimination can share two identical pure computations with identical operands/types. A schematic `t1=identity(q); custom(t1,k,v)` becomes `custom(q,k,v)` when the operation's canonicalization permits it. Two identical constant definitions can become one shared constant. Chain canonicalization reduces redundant ordering structure. These illustrate the passes' purpose; the one-custom-call input graph does not demonstrate those exact rewrites, and side-effectful calls cannot be merged just because their names match. [[gc:242]]

### MAP access/fusion example

For a separate fusible elementwise chain `t=q+1; o=t*2`, compatible per-element accesses may let generated code load `q[i]`, compute `(q[i]+1)*2`, and store `o[i]` without a materialized intermediate t. That is a comparison explaining why access information is useful. This attention example already implements QK/softmax/PV inside its custom Mojo routine; no claim is made that MAP automatically fuses arbitrary custom calls or splits this routine into those Python operations. [[gc:338]] [[kernel:333]]

### MojoRegistry and MAPToMOGGState

`MojoRegistry(module)` is a compiler-side view of imported declaration operations, not a running model or Python dict of GPUs. An illustrative lookup is `"modular_ops::fused_attention_custom" → declaration for FusedAttention.execute`. `MAPToMOGGState(registry, op)` keeps the lowering context for the current opaque iteration; `lowerOp` rewrites that operation with selected implementation calls. The resulting “glue” passes shapes, device context, buffer roles and parameters between generated routines. The package registration and tensor signature provide the concrete values. [[mapto-mogg:57]] [[mapto-mogg:66]] [[registry:48]]

### JIT means just-in-time compilation

Here `session.compile(graph)` asks compiler libraries inside the Python process to produce specialized native routines before the first numerical `execute`. It is still called JIT because native compilation happens as part of program execution. The earlier `mojo precompile` subprocess only prepares package IR. Kernel cache hits can reuse prepared resource bytes/symbol names; they are not yet resident callable pointers. [[jit:248]] [[jit:567]]

The source checks `is_device_graph` before device-graph finalization. This Python Graph does not request that option; an ordinary computation graph on a GPU is not automatically a CUDA Graph capture. [[graph:831]] [[gc:382]]''', 'pipeline'),

card('abi', 'A1 → H1 · Bytes, callable table and host ABI',
     'C1 contains compiled resources; init turns host code bytes into callable addresses. GPU handles remain separate.',
'''M1 initialization: mgp_module_load_cpu(...,image,symbols,...) [[cpu-primitives:56]]
├─ retain CompiledKernelsContext in MGP state's module slot [[cpu-primitives:74]]
└─ load(A1,symbols) → host-link namespace → symbol lookup [[load:85]] [[load:166]]
   └─ kernels[index]=address; symbolTable[index]=name [[load:181]]

M1 execution: mgp_generic_execute(slot,moduleIndex,functionIndex) [[primitives:176]]
├─ getKernelAsyncABI(moduleIndex,functionIndex) → H1 [[primitives:188]]
├─ getKernelInputs(operands,results,constantResources) [[primitives:98]]
│  ├─ addresses of operand AsyncValueRef slots [[primitives:112]]
│  ├─ addresses of result AsyncValueRef slots [[primitives:114]]
│  └─ resource data pointer and byte length pairs [[primitives:117]]
└─ H1(kernelInputs.data()) through void(void**) [[primitives:234]]
   └─ shim unpacks runtime values; generated routine reaches the GPU enqueue [[kernel:105]]''',
'''### Example values at the host call boundary

```
kernel table: modules[0].kernels[0] = H1
H1: host instruction address for a generated routine
kernelInputs = [ &operandSlot0, &operandSlot1, …,
                 &resultSlot0, …, resourceData, resourceByteCount, … ]
operandSlotQ → AsyncValue(Tensor(shape=[32,32],storage=Q_gpu))
resultSlotO  → empty before H1; resolved result after generated work
```

The list's exact order/length depends on the generated routine, including context/chain/resource operands. These values illustrate slot indirection and are not an observed ABI dump. `getKernelInputs` explicitly pushes addresses of AsyncValue references. The invoked type is `CompiledKernelAsyncABIFn`, despite its name being used in both the inline and queued host branches. [[primitives:108]] [[primitives:188]] [[primitives:275]]

### Host linking is a separate mechanism

Normal loading uses ORC/JITLink with an in-process memory mapper. A code section obtains executable host memory; references to other symbols are resolved and relocations applied. The debug branch instead writes an object, invokes the configured linker with `-shared`, and loads its temporary `.so`. Both end with host callable lookup. The GPU D1 payload is subsequently loaded through CUDA/HIP APIs. [[jit-engine:197]] [[load:119]] [[load:144]]

[[driver-guide:host-link]]''', 'abi'),

card('device-loading', 'D1 → device module → kernel function',
     'The driver receives a compiled device image and a symbol name, then returns device handles.',
'''generated enqueue → compile_function[kernel_func]() [[enqueue:171]]
├─ compiler-provided _compile_code[...]() → module/name/code descriptor [[legacy:3151]]
└─ _load_function(...) → AsyncRT native binding [[legacy:3158]] [[legacy:3031]]
   └─ CUDADeviceContext::loadFunction(moduleName,functionName,code,...) [[cuda:2211]]
      ├─ cache key includes module/code identity + function/options [[cuda:2217]]
      ├─ hit → reuse cached module/function; no module load driver call [[cuda:2223]]
      └─ miss → bind device context and load [[cuda:2259]]
         ├─ cuModuleLoadDataEx(D1 bytes) → CUmodule MD1 [[cuda:2262]]
         ├─ cuModuleGetFunction(MD1,name) → CUfunction KD1 [[cuda:2268]]
         └─ record function/cache references; return DeviceFunction [[cuda:2288]]

AMD equivalent: hipModuleLoadData → hipModuleGetFunction [[hip:1968]] [[hip:1972]]''',
'''### How the bytes get to the GPU

MAX passes the image bytes to the vendor's module loader. The image has code, symbols and metadata needed by that loader. The driver creates device-specific executable resources and returns opaque handles. MAX does not allocate a tensor Buffer and memcpy the entire ELF file into it as if it were matrix data. Exact physical placement, page mapping and lazy loading are vendor-driver details. [[cuda:2262]] [[hip:1968]]

`H1` is a host instruction address. `MD1` is a device-module handle, `KD1` is a kernel-function handle, and `Q_gpu` is a tensor allocation. Calling `cuLaunchKernelEx` submits KD1 with a grid, block dimensions, stream and argument pack. The CPU does not directly jump to the GPU instruction pointer. [[cuda:1610]]

[[driver-guide:device-load]]''', 'device-loading'),

card('attention-math', 'Attention · One row, two key tiles',
     'Follow actual numbers through QK, stable normalization, old-output rescaling and the new PV contribution.',
'''GPU example: N=D=32, BN=16, BD=8; BN_1=8 [[entry:36]] [[kernel:333]]
├─ grid=(4,2), block=32; block(x,y) owns O[16y:16y+16,8x:8x+8] [[kernel:403]]
├─ Q_tile:16×32; initialize m=Scalar[f32].MIN, l=0, O_i=0 [[kernel:333]] [[kernel:335]]
└─ for four key tiles of 8 rows [[kernel:343]]
   ├─ S = Q_tile @ K_tileᵀ → 16×8 scores [[kernel:346]]
   ├─ m_new = max(m_old,rowmax(S)) [[kernel:348]]
   ├─ E = exp(S-m_new); l_new=exp(m_old-m_new)*l_old+sum(E) [[kernel:352]]
   ├─ E /= l_new → normalized current-tile weights [[kernel:366]]
   ├─ O_i *= l_old/l_new * exp(m_old-m_new) [[kernel:367]]
   ├─ O_i += E @ V_tile [[kernel:374]]
   └─ retain m_new,l_new,O_i for the next key tile [[kernel:376]]
store the block's output tile in O [[kernel:378]]''',
'''### Small numerical example used by this animation

```
Q = [[1,0], [0,1]]          two independent queries
K = [[1,0], [0,1], [2,0], [0,2]]
V = [[1,0], [0,2], [3,1], [1,4]]
key tiles = [0,1] then [2,3]
```

For query `[1,0]`, scores are `[1,0 | 2,0]`; the maximum grows from 1 to 2. Tile one has `m=1`, `l=1+exp(-1)=1.367879`, and normalized `O=[0.731059,0.537883]`. Tile two sets `alpha=exp(1-2)=0.367879`. Old denominator mass becomes `0.503215`; new tile mass is `1.135335`; new total `l=1.638550`. The old output's multiplier is `alpha*l_old/l_new=0.307110`. Current weights are `[0.610296,0.082595]`. Add their V contribution to the rescaled old output. The browser computes these quantities from the matrices instead of storing rounded captions. [[kernel:348]] [[kernel:366]]

### Why the old result must be rescaled

Changing the maximum changes the reference for every exponential, including keys already processed. Changing the denominator changes every normalized weight. Multiplying the retained output by `alpha*l_old/l_new` accounts for both changes; merely adding a new tile's independently normalized output would be wrong. A second query has its own m/l/O values, so normalizing across all rows together would also be wrong. [[kernel:367]]

### What remains in memory

Q/K/V/O are global tensor buffers. Each block visits local score and exponential tiles, keeping per-row m/l and a partial output tile across iterations. This example calls `stack_allocation` for local tile storage; register allocation/spills and actual occupancy require generated-code analysis. The drawing counts logical elements, not measured device memory consumption. No persistent full 32×32 score buffer is requested by this loop. [[kernel:333]]

The tiny animation uses −∞ as the initial neutral maximum; the source initializes `m_1` with `Scalar[q_dtype].MIN`. For the finite scores in this example, the first tile replaces that sentinel before it contributes mass. The animation is browser arithmetic, not GPU execution or a numerical test of MAX. It follows the same unscaled `softmax(QKᵀ)V` recurrence. [[kernel:335]]''', 'math'),

card('synchronization', 'Readiness · CPU values and GPU streams',
     'Keep argument lifetime, host task completion, device completion and host readability separate.',
'''Buffer.to(GPU0) → allocate + DeviceBuffer.copyTo [[buffer:683]]
└─ transfers and later operations use device-context ordering [[buffer:694]]

mgp_generic_execute selects inline or queued host execution [[primitives:202]]
├─ queued branch retains operand/result/context references [[primitives:212]]
├─ bind context → H1(...) → unbind context [[primitives:229]]
└─ release task references, then publish indirect results [[primitives:254]]

executeFromDriver → extractDriverResults [[driver:212]]
├─ awaitOrError(results): wait for native result values [[driver:137]]
└─ construct Python output Buffers around existing output storage [[driver:173]]

optional O1.to(CPU()).to_numpy() [[py-buffer:186]]
├─ device→host copy when needed [[py-buffer:194]]
└─ DLPack host read → synchronize Buffer storage [[buffer:407]] [[buffer:416]]''',
'''### Three useful readiness statements

| Statement | What it establishes |
|---|---|
| H1 returned | The host shim has returned or arranged its result dependencies; this alone is not a global GPU barrier |
| `awaitOrError(results)` succeeded | The native result AsyncValues needed by the driver wrapper are available without error |
| Buffer synchronization / completed host readback | The relevant storage can be read by the host under that API's synchronization contract |

GPU streams order work submitted to them; event-based dependencies can coordinate different contexts/streams. AsyncRT chains coordinate the host runtime's execution plan. These mechanisms cooperate, but a chain token is not a tensor and a host condition variable is not an online-softmax denominator. [[primitives:176]] [[driver:137]] [[buffer:416]]

### Why references are retained

The queued branch can outlive its caller's stack frame. Its closure retains operands, output slots and a device context while H1 runs. It resolves downstream result slots only after releasing context-reachable task references. This prevents a completion observer from tearing down resources that the host task still uses. Returned O1 owns its storage independently of the temporary launch-argument pack. [[primitives:212]] [[primitives:251]]

The actual example ends with `print(output)` and no explicit readback. The optional readback shown here is clearly separated in the animation. [[entry:83]]

[[driver-guide:completion]]''', 'synchronization'),
]
