"""Fresh, sequential GPU attention reader. Source lines refer to the pinned checkout."""

PAGE = 'max_example_04_fused_attention_online_softmax.html'
DRIVER_PAGE = 'max_example_04_gpu_artifacts_abi_and_driver.html'

SOURCES = {
    'entry': 'max/examples/custom_ops/fused_attention.py',
    'kernel': 'max/examples/custom_ops/kernels/fused_attention.mojo',
    'build': 'max/examples/custom_ops/BUILD.bazel',
    'macro': 'max/examples/custom_ops/custom_op_example.bzl',
    'bazelw': 'bazelw',
    'py-rule': 'bazel/internal/modular_py_binary.bzl',
    'mojo-rule': 'bazel/internal/mojo_library.bzl',
    'test-rule': 'bazel/internal/modular_run_binary_test.bzl',
    'binary-test': 'bazel/internal/binary_test.bzl',
    'env': 'bazel/internal/mojo_test_environment.bzl',
    'graph': 'max/python/max/graph/graph.py',
    'paths': 'Mojo/python/mojo/paths.py',
    'mojo-run': 'Mojo/python/mojo/run.py',
    'package-root': 'max/examples/custom_ops/kernels/__init__.mojo',
    'walkthrough': 'Mojo/docs/compiler/MojoCompilerWalkthrough.md',
    'precompile': 'Mojo/tools/mojo/Precompile/mojo-precompile.cpp',
    'py-buffer': 'max/python/max/driver/buffer.py',
    'api': 'max/python/max/engine/api.py',
    'custom': 'max/python/max/graph/ops/custom.py',
    'native-session': 'max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp',
    'engine-context': 'max/lib/max_license/API/c/EngineContext.cpp',
    'models': 'max/lib/max_license/API/c/CompiledModels.cpp',
    'model': 'GraphCompiler/lib/MEFModel/MEFModel.cpp',
    'gc': 'GraphCompiler/lib/GraphCompiler.cpp',
    'frontend': 'GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp',
    'jit': 'GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp',
    'object': 'Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp',
    'kgen': 'Mojo/lib/Compiler/KGENCompiler.cpp',
    'backend': 'Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp',
    'nvptx': 'Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp',
    'amdgpu': 'Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp',
    'lower-routines': 'GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp',
    'load': 'GraphCompiler/lib/Support/CompiledKernelsContext.cpp',
    'abi': 'GraphCompiler/include/GraphCompiler/Support/CompiledKernelsContext.h',
    'cpu-primitives': 'GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp',
    'primitives': 'GraphCompiler/lib/MGPDialect/Primitives/MGPPrimitives.cpp',
    'mojo-primitives': 'max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo',
    'mgp-device': 'GraphCompiler/lib/MGPDialect/Dialect/MGPDeviceOps.cpp',
    'buffer-builder': 'GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp',
    'buffer': 'max/python/max/_core/max_license/lib/Driver/Buffer.cpp',
    'driver': 'max/lib/max_license/API/c/ModelExecution.cpp',
    'legacy': 'max/mojo/max/gpu/host/_legacy/device_context.mojo',
    'enqueue': 'max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo',
    'bindings': 'MLRT/lib/Driver/DeviceContext/MojoBindings.cpp',
    'cuda': 'MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp',
    'hip': 'MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp',
    'cuda-driver': 'MLRT/lib/Driver/DeviceContext/CUDA/CUDADriver.cpp',
    'hip-driver': 'MLRT/lib/Driver/DeviceContext/HIP/HIPDriver.cpp',
    'cuda-compiler': 'MLRT/lib/Driver/DeviceContext/CUDA/CUDACompilationDevice.cpp',
    'cuda-compiler-driver': 'MLRT/lib/Driver/DeviceContext/CUDA/CUDACompilationDriver.cpp',
    'host-backend': 'Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp',
    'jit-engine': 'Mojo/lib/ExecutionEngine/ExecutionEngine.cpp',
    'jit-engine-h': 'Mojo/include/Mojo/ExecutionEngine/ExecutionEngine.h',
    'archive-layer': 'Mojo/lib/ExecutionEngine/JIT/StaticArchiveLayer.cpp',
    'materialization': 'Mojo/lib/ExecutionEngine/JIT/MaterializationLayer.cpp',
    'device': 'MLRT/lib/Driver/DeviceContext/DeviceContext.cpp',
    'graph-binding': 'max/python/max/_core/max_license/lib/Graph/Bindings.cpp',
    'registry': 'GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp',
}

# Source keys for native interfaces are resolved to their existing pinned files
# by the builder; this avoids a second handwritten path catalog.
ALIASES = {
    'context': 'ct_graph_context', 'builtin': 'ct_builtin_module',
    'graph-op': 'ct_graph_op_build', 'type': 'ct_tensor_to_mlir',
    'moto-map': 'mo_to_map', 'mapto-mogg': 'map_to_mogg',
    'moggto-mgp': 'mogg_to_mgp',
}

def chapter(id, number, title, call, summary, tree, why, *, more=()):
    return dict(id=id, number=number, title=title, call=call, summary=summary,
                tree=tree.strip(), why=why.strip(), more=list(more))

CHAPTERS = [
chapter('build', '01', 'Build the launcher. Then enter Python.',
    './bazelw build //max/examples/custom_ops:fused_attention',
    'Bazel prepares the program and its runfiles. Running that program reaches main(), then Graph() prepares P1.',
    '''caller: ./bazelw build //max/examples/custom_ops:fused_attention [[bazelw:1]]
├─ repository wrapper → pinned Bazelisk executable; exec forwards arguments [[bazelw:33]] [[bazelw:59]]
├─ Bazel loads BUILD.bazel and expands the example macro [[build:73]]
│  └─ custom_op_example_py_binary(name="fused_attention", …) [[macro:5]]
│     ├─ modular_py_binary → Python launcher + fused_attention.py [[py-rule:53]]
│     ├─ data: :kernel_sources → kernels/*.mojo in runfiles [[macro:15]]
│     ├─ Mojo deps: layout, extensibility, std → library packages [[macro:19]]
│     └─ sibling test: :fused_attention.example-test [[macro:43]]
├─ Bazel analysis resolves labels, toolchains and dependency actions [[env:78]]
│  ├─ configured Mojo toolchain precompiles Mojo library dependencies [[mojo-rule:35]]
│  └─ runtime environment exposes compiler, linker and import paths [[env:192]]
└─ OUT: executable launcher + runfiles; example kernel sources still source data [[macro:16]]

caller: ./bazelw run //max/examples/custom_ops:fused_attention [[build:73]]
├─ Bazel ensures the target is built; starts its Python launcher [[py-rule:53]]
├─ Python imports pathlib, NumPy and the MAX native bindings [[entry:14]]
└─ if __name__ == "__main__": main() [[entry:86]]
   ├─ main chooses Accelerator(); GPU sizes N=D=32, BN=16, BD=8 [[entry:29]]
   └─ Graph(..., custom_extensions=[kernels/]) [[entry:41]]
      └─ _import_kernels → load_paths → _build_mojo_source_package [[graph:910]] [[graph:289]]
         ├─ find Mojo executable: MODULAR_MOJO_MAX_DRIVER_PATH, else PATH [[mojo-run:70]]
         ├─ separate process: mojo precompile kernels/ -o <staging>/<package>.mojoc [[paths:169]]
         ├─ Mojo frontend imports package; builds serializable lit.package [[precompile:280]] [[precompile:308]]
         ├─ runCheckLITPipeline checks semantics/lifetimes; write MLIR bytecode [[precompile:318]] [[precompile:322]]
         └─ atomic rename publishes P1: one .mojoc package [[paths:179]]

caller: ./bazelw test //max/examples/custom_ops:fused_attention.example-test [[macro:43]]
└─ generated binary_test runs the same executable/runfiles; nonzero exit fails [[binary-test:3]]
   └─ this script prints output; it has no numerical parity assertion [[entry:82]]''',
    '''### Three tools, three jobs

**Bazel owns the build graph.** It loads Starlark declarations, analyzes configured targets, and schedules only the actions needed for the target. A dependency label describes another build target; it is not a Python object. The example macro supplies Python imports, Mojo libraries and raw custom-kernel source data. [[macro:12]]

**The Python interpreter owns program execution.** Its launcher imports the script and executes the `__main__` guard. Calling `main()` evaluates the GPU choice and constructs the graph. Building the target does not execute these Python statements. The resulting launcher path depends on the selected Bazel configuration; the stable convenience path is `bazel-bin/max/examples/custom_ops/fused_attention`. [[entry:86]]

**The Mojo compiler owns package parsing and later kernel specialization.** At build time, declared shared Mojo library targets have their own precompile actions. At runtime, `Graph` converts the example's `kernels/` source package to P1. This keeps the Python target usable with a runtime-selected GPU. It also keeps package preparation separate from specialization for concrete dtype, shape and device. [[mojo-rule:35]] [[graph:284]]

### Why P1 is a package rather than GPU machine code

P1 holds parsed package IR in MLIR bytecode. It includes the package rooted at `kernels/__init__.mojo` and its referenced modules, including `fused_attention.mojo`; it does not contain `fused_attention.py`. Concrete kernel generation comes later in `session.compile`. Atomic publication prevents another process from reading half-written bytecode. A deterministic output path is not proof of a cache hit: this source-directory route invokes precompile each time it is entered. Supplying an existing `.mojoc` uses the direct-load branch. [[paths:141]] [[graph:285]]

### Why a separate test target

The macro creates a test that launches the example with the same dependencies. `bazel test` builds what it needs and checks the process exit code. This catches build/startup/runtime failures, while numerical correctness requires assertions in the example or a dedicated test. This particular script only prints its returned Buffer list. [[macro:42]] [[entry:83]]''', more=('package',)),

chapter('graph', '02.1', 'G1 · Record the attention computation',
    'with Graph("fused_attention", input_types=[TQ,TK,TV], custom_extensions=[kernels/]) as graph:',
    'G1 is a Python owner of symbolic MO operations in host memory. P1 supplies the custom operation’s declaration and implementation.',
    '''caller: main() → with Graph("fused_attention", …) as graph [[entry:41]]
├─ arguments: three TensorTypes, each float32[32,32] on gpu:0 [[entry:43]]
├─ Graph.__init__(..., forward=None) [[graph:820]]
│  ├─ activate the thread's MLIR context; collect symbolic dimensions [[graph:819]] [[graph:840]] {{context}}
│  │  └─ all dimensions static → _params={} [[graph:840]]
│  ├─ _location() → location metadata; create enclosing builtin.ModuleOp [[graph:851]] {{module}}
│  ├─ OpBuilder at module.body.end → insert mo.GraphOp [[graph:858]]
│  │  ├─ to_mlir() converts type descriptions, not tensor data [[graph:864]]
│  │  └─ GraphOp builder creates ONE region, ONE block, THREE input arguments [[graph-op:846]]
│  ├─ retain _module, _mlir_op and _graph_body; attach device metadata [[graph:874]]
│  ├─ initialize weights={} and CPU/per-device ordering-chain bookkeeping [[graph:879]]
│  └─ KernelLibrary(); _import_kernels(resolved extension paths) [[graph:909]] [[graph:910]] {{package}}
│     ├─ kernels/ → Mojo subprocess → P1.mojoc [[graph:289]] [[paths:178]]
│     └─ analysis keeps P1's path; GraphOp stores _kernel_library_paths [[graph:1703]]
├─ Graph.__enter__ → current graph G1; insertion point is G1's body [[graph:1135]]
├─ q,k,v = graph.inputs → wrappers around B0's three SSA arguments [[entry:56]] [[graph:943]]
├─ ops.custom(name="modular_ops::fused_attention_custom", …) [[entry:57]] {{custom}}
│  ├─ parameters: BD=8, BN=16; operands [%q,%k,%v]; result type T [[entry:60]]
│  ├─ insert mo.custom into B0 and attach attributes [[custom:109]] [[custom:122]]
│  ├─ validate against loaded P1 declarations; return [TensorValue(%o)] [[custom:127]]
│  └─ graph.output(*results) → mo.output + finalized result signature [[entry:68]] [[graph:1469]]
└─ __exit__ restores previous recording context; G1 still owns the graph [[graph:1138]]
   OUT: G1 in host memory; no Q/K/V numerical buffers have been created [[entry:71]]''',
    '''### Why Graph construction precedes numerical inputs

G1 records **what should run** using types and SSA values. `%q`, `%k`, `%v` are arguments in a block; they are not NumPy arrays, GPU pointers or intermediate results of numerical attention. Static `32×32` shapes let the compiler validate and specialize before any random input matrices exist. Actual buffers appear after `session.init` in this example. [[entry:41]] [[entry:78]]

### Where G1 is stored

The Python variable `graph` references a `Graph` instance. Its `_module` refers to the enclosing native `builtin.module`; `_mlir_op` wraps the native `mo.graph`; `_graph_body` aliases that graph's entry block. These wrappers and IR objects live in host process memory. `module.body` is a list of operations, while the graph block contains its arguments and ordered operations. The graph is not automatically written to an `.mlir` file. [[graph:856]] [[graph:874]]

```
Python graph → G1
  ├─ _module → builtin.module
  │             └─ body block
  │                 └─ mo.graph @fused_attention
  │                     └─ region → B0(%q:T,%k:T,%v:T)
  │                         ├─ chain/order operations
  │                         ├─ %o = mo.custom(%q,%k,%v)
  │                         └─ mo.output(%o)
  ├─ _mlir_op / _graph_body → aliases of that same native graph/block
  └─ _kernel_library → analysis + P1 library path
```

### Why the package and graph are separate

G1 describes one call to a registered operation. P1 supplies a reusable parametric Mojo implementation. The name connects them; explicit parameters supply `BN=16, BD=8`, while dtype, rank, layouts and target are inferred from the call contract. Checking this boundary early makes an unknown registration, parameter typo or incompatible result type fail before expensive target compilation. The declaration analysis can be reused; each new custom call is still validated. [[kernel:67]] [[graph:307]]

### Why regions, signatures, contexts and chains exist

A region owns the graph body; one entry block has three arguments because this computation has three inputs. The signature specifies the interface needed by validation, compilation and runtime input checking. The MLIR context owns compiler-wide types/attributes; a location is provenance attached to an operation. Chains represent effect ordering, including work associated with devices. They are internal dependencies and are not extra user tensor inputs. These distinctions let the same symbolic computation be transformed without confusing storage ownership with execution order. [[graph-op:831]] [[graph:879]]

### Why this example uses a with-block

The constructor's `forward` argument is absent. `with graph` makes G1 current while the explicit body inserts the custom op. Exiting restores the previous graph context but does not erase G1. That differs from the add-one example's forward lambda while using the same graph-building machinery. [[entry:55]] [[graph:912]]''', more=('context','module','custom')),

chapter('session', '02.2', 'S1 · Retain compiler and runtime services',
    'session = InferenceSession(devices=[device])',
    'S1 holds configuration and native runtime references. GPU0 is the computation device; CPU services handle compilation, scheduling and driver calls.',
    '''caller: InferenceSession(devices=[GPU0]) [[entry:71]]
├─ Python constructor deduplicates devices; appends CPU if absent [[api:688]]
│  └─ final_devices=[GPU0,CPU0]; custom_extensions=[] for this caller [[api:697]]
└─ native _InferenceSession(final_devices, extensions, num_threads or 0) [[api:707]]
   ├─ populateCompileConfig → sessionCompileConfig [[native-session:546]]
   │  └─ clone current settings; append any explicit extension paths [[native-session:615]]
   ├─ runtimeConfig records device specs + context references [[native-session:549]]
   │  └─ language=Python; thread setting=0 selects runtime defaults [[native-session:553]]
   ├─ Engine::Context::create(runtimeConfig) [[native-session:559]]
   │  ├─ use supplied MAX Context or Init::getOrCreateContext [[engine-context:169]]
   │  ├─ retain MAX Context service store and AsyncRT CPUDevice [[engine-context:214]]
   │  └─ validate configured device labels [[engine-context:200]]
   └─ retain runtimeContext; enable Python stack diagnostics [[native-session:563]]
      OUT: S1._impl → configs, EngineContext1 → MAXContext1 / CPUDevice1 [[native-session:569]]

P1 reaches S1's compiler later through compile(G1), not this constructor [[api:1022]]
G1 remains owned by the caller; S1 has not executed its attention op [[entry:74]]''',
    '''### Why a session is more than a device list

S1 combines **compile policy** with **runtime services**. Compile settings select library paths and compiler options. Runtime settings retain device specifications, actual device contexts and CPU scheduling services. The same session can compile and initialize multiple models without rebuilding every service. The GPU object was already created in `main()`; storing another reference in S1 does not create a second physical GPU. [[entry:29]] [[native-session:546]]

| Object | Example retained contents | Purpose |
|---|---|---|
| Python S1 | `_impl`, selected devices | API owner |
| native S1 | `sessionCompileConfig`, `runtimeConfig`, `runtimeContext` | compiler/runtime configuration |
| EngineContext1 | runtime config, MAX Context reference, CPUDevice reference | compilation and initialization entry points |
| MAXContext1 | typed runtime services including CPUDevice and telemetry | shared process infrastructure |
| GPU0 context | driver context and stream services | device allocation and work submission |

### Why CPU is still required on the GPU path

Python and MAX native orchestration execute on the CPU. A CPU worker can run compilation or model initialization, produce AsyncValues and issue driver calls that enqueue GPU work. The worker is not emulating the attention arithmetic. This division also lets dependent operations wait for values rather than occupying an application thread with all the lower-level scheduling work. [[engine-context:225]] [[engine-context:311]]

### Why there are two kinds of context

The **MLIR context** used by G1 owns compiler IR infrastructure. The **MAX Context** held by EngineContext1 is a native service store, including the CPU runtime. The **GPU driver context** is the device execution/memory domain. Their lifetimes overlap, but they are different objects. `devices=[GPU0]` selects runtime capabilities; G1's `DeviceRef("gpu",0)` is a symbolic reference resolved against those capabilities. [[engine-context:169]] [[model:197]]

### Why P1 is not duplicated here

The caller gave `custom_extensions` to G1 only. During compile, Python gathers G1's package paths and seeds its native declaration analysis. S1 therefore does not need a second explicit `custom_extensions=[P1]` argument for this example. Keeping package ownership at the graph boundary also supports different graphs with different custom libraries in one session. [[api:1022]] [[api:1042]]'''),

chapter('compile', '02.3', 'C1 · Compile G1 for the GPU',
    'compiled = session.compile(graph)',
    'C1 is the returned CompiledModel. IR1 is the separate compiler working clone. P1’s declarations and bodies support lowering and kernel specialization.',
    '''caller: S1.compile(G1) → C1: CompiledModel [[entry:74]] [[api:1104]]
├─ compile_async: select G1.module; append G1's P1 library paths [[api:1020]]
│  ├─ expected_weights={}; capture graph name "fused_attention" [[api:1028]]
│  └─ seed package declarations/type mapping from G1._kernel_library [[api:1042]]
├─ _compile_module → native compile(module capsule, P1 paths, name, options) [[api:1294]]
│  ├─ native request stores the source capsule and compile options [[native-session:670]]
│  └─ Engine::Context::compileModel schedules work on AsyncRT CPU runtime [[engine-context:226]]
├─ MOFContext::compileToBinary clones the source module → IR1 [[frontend:381]]
├─ convertMOToMEF → runGCPipeline (cold graph-artifact route) [[frontend:356]]
│  ├─ MO: verify/simplify graph contracts; imported declarations available [[gc:219]]
│  ├─ MO → MAP: tensor operations become access/computation descriptions [[gc:581]]
│  ├─ MAP: optimize eligible access patterns/fusion; preserve custom contract [[gc:338]]
│  ├─ MAP → MOGG: select Mojo implementations and form implementation calls [[gc:597]]
│  ├─ MOGG → MGP: buffers, state slots, chains, init and execution regions [[gc:611]]
│  └─ MGP: finalize marked device graphs if present; emit Mojo routines [[gc:382]] {{pipeline}}
├─ JitCompileKernelsPass → parse emitted Mojo with P1 imports [[jit:529]]
│  ├─ LIT→KGEN; create host ObjectCompiler before checking native code cache [[jit:567]] [[jit:575]]
│  ├─ cache miss: runElaborationPipeline → specialize and isolate GPU offloads [[jit:440]] [[kgen:610]]
│  ├─ target triple → backend; lower LLVM dialect → LLVM IR [[object:146]] [[object:807]]
│  ├─ NVIDIA: LLVM → PTX → cubin D1; host compiler → code resource A1 [[nvptx:154]] [[object:1023]]
│  └─ matching native-code cache may reuse A1/D1 instead [[jit:248]]
├─ lower routines → ModuleLoad in init, ExecuteGeneric in execute [[lower-routines:102]]
├─ serialize runtime program/resources → MEF bytes; construct native artifact [[frontend:371]]
└─ wait for compile completion/errors → return C1 [[api:1112]]
   C1._compiled retains native artifact; no attention data buffers yet [[api:442]]
   export_mef(path) would write a file; this caller does not call it [[api:449]]''',
    '''### Why C1 and IR1 have different names

**C1 is the public compiled artifact** returned into the Python variable `compiled`. Its `_compiled` member holds a resolved native compiled-model result (through the asynchronous handle), `_expected_weights={}`, and `_graph_names=("fused_attention",)`. **IR1 is the mutable native module clone** created by `MOFContext::compileToBinary` and used by compiler passes. G1 is still the caller's graph. Python may seed declaration metadata onto its module before cloning; compilation does not turn the Python G1 object into an ELF file. [[api:1042]] [[api:1055]] [[frontend:381]]

### Why several compiler representations are useful

**MO — Modular Operations** retains graph-level tensor types, shapes, device placement and custom operation contracts. **MAP — Modular Access Primitives** exposes access/computation structure used by eligible fusion and implementation decisions. **MOGG — Modular Operation Generator Graph** connects selected Mojo generators and their typed calls. **MGP — Modular Glue Primitives** expresses runtime buffers, dependencies, initialization state and execution. These are IR dialects/forms within compiler modules, not four mandatory files on disk. [[gc:219]] [[gc:338]] [[gc:370]] [[gc:382]]

For this graph, the central contract remains three float32[32,32] inputs and one matching output on GPU0, implemented by `fused_attention_custom[BN=16,BD=8,…]`. A custom kernel's internal QK/softmax/PV loop is Mojo code; the graph compiler does not need to reconstruct that loop as ordinary Python graph operators. The companion pipeline card follows a schematic before/after at each layer. Exact textual IR is configuration-dependent. [[entry:57]] [[kernel:74]]

### How P1 is used twice without running it twice

Package analysis first validates the graph call and supplies declarations/type mappings. Later the Mojo compiler consumes package bodies/imports while specializing concrete routines. A `.mojoc` is not a dynamic library loaded into GPU memory. Its role is compile-time code availability. Once C1 owns the emitted resources and runtime program, model initialization operates on that artifact rather than reparsing `kernels/`. [[api:1034]] [[jit:547]]

### Why host and device code are compiled separately

The host routine must obey the host ABI, marshal tensor arguments and invoke runtime/device functions. D1 must obey the GPU instruction set and device argument layout. The compiler isolates the device offload and selects its backend using its target triple and architecture; a GPU override does not change Python/MAX into GPU code. LLVM dialect lowering and translation to `llvm::Module` are in-process compiler APIs. NVIDIA object emission produces PTX then invokes the CUDA compilation device to obtain cubin bytes; the normal Linux path uses the NVPTX compiler library, with an explicit external-ptxas fallback. [[kgen:720]] [[object:807]] [[nvptx:176]] [[cuda-compiler:227]]

### Where .text, .data, stack and heap fit

During compile, A1 and D1 are **bytes in host-owned buffers**. Their object-file sections describe eventual code/constants/globals; they are not yet process instruction addresses. Host executable mapping/linking happens during initialization. Device-module loading happens when the generated load site is reached, with a per-device function cache. Stack frames, launch argument packs and GPU scratch are runtime storage. Q/K/V/O are runtime tensor allocations, not ELF `.data` sections filled by the compiler. See the artifact/ABI guide for each boundary. [[object:1023]] [[load:85]] [[legacy:3158]]

### When MEF and dynamic libraries appear

The post-JIT runtime program and resources are serialized into MEF during compilation. C1 can retain that result in memory; writing `attention.mef` requires an explicit export. Normal initialization uses ORC linking of host code. A debug-information branch writes an object, links a temporary `.so`, and loads it. That `.so` is generated in the loader, whereas vendor libraries such as `libcuda.so.1` are installed runtime dependencies. The artifact, generated debug library, and vendor driver library have different origins. [[frontend:371]] [[api:449]] [[load:98]] [[cuda-driver:61]]''', more=('pipeline','abi')),

chapter('init', '02.4', 'M1 · Initialize the compiled artifact',
    'model = session.init(compiled)',
    'Initialization consumes C1 and S1’s devices/services, resolves native callable resources, and returns a model whose setup dependencies have completed.',
    '''caller: S1.init(C1) → M1 [[entry:75]]
├─ init_all(C1): expected_weights={} and supplied_weights={} [[api:1151]] [[api:1202]]
│  └─ wait/unwrap C1; native load_all returns named models [[api:1244]]
├─ native InferenceSession::loadAll → load(model, weights) [[native-session:290]]
│  └─ Engine::Context::initModel → CPU runtime task [[engine-context:311]]
├─ Model::setup(cpuDevice, weightsRegistry, deviceMap) [[model:184]]
│  ├─ resolve model DeviceRefs using S1's actual device contexts [[model:197]]
│  ├─ prepare configured stream resources; form init arguments [[model:208]]
│  └─ initFn->invoke(initArgs, cpuDevice) [[model:220]]
│     ├─ create MGP state and persistent resource slots [[primitives:89]]
│     └─ module-load primitive → CompiledKernelsContext::load(A1, symbols) [[cpu-primitives:114]]
│        ├─ ordinary: hash namespace → StaticArchiveLayer → JITLink [[load:160]]
│        ├─ debug alternative: write .o → link temporary .so → load library [[load:119]] [[load:152]]
│        └─ symbol lookup fills host callable table H1 [[load:166]] {{abi}}
├─ construct ExecutableModel(chain, mgpContext, weights, extraArgs, …) [[model:237]]
│  └─ retain resources; await fixed/extra setup results and propagate errors [[model:350]] [[engine-context:91]]
└─ native load awaits setup; Python selects the single model → M1 [[native-session:282]] [[api:1158]]
   M1 uses C1's model/resources and S1's runtime; P1/G1 are compile inputs [[native-session:196]]

device-code boundary: generated enqueue path loads D1 on reaching its load site [[legacy:3151]]
└─ D1 bytes → CUDA/HIP module handle → kernel function handle; cache thereafter [[legacy:3158]] [[cuda:2211]]
   This separate device handle is not H1, and it is not a Python callable. {{device-loading}}''',
    '''### Why compile and init are separate

Compilation answers which code and runtime plan are needed. Initialization binds that plan to a concrete runtime and persistent resources. The same C1 can be initialized more than once, while each initialized model has its own resource references and setup state. This example supplies no external weights: Q/K/V are execution inputs, so both expected and supplied weight mappings are empty. [[api:414]] [[api:1202]]

### What the ExecutableModel constructor actually does

The constructor retains the model definition, CPUDevice, initialization chain, MGP context, weights-registry value and extra initialization results. It does not repeat MO→MAP→MOGG→MGP. Resource initialization was requested by `initFn->invoke`. `createExecutableModel` waits for setup before publishing the successful executable, and the Python init call returns only after that path succeeds. A deferred initialization error prevents a valid M1 from being returned. [[model:220]] [[model:350]] [[model:387]]

### Why a CPU worker participates

Initialization executes native primitives, allocates state, links host routines and handles asynchronous dependencies. AsyncRT's worker environment provides the current CPUDevice and scheduling context required by those routines. GPU arithmetic will run after device submission, while these setup tasks use ordinary host execution. A worker task is not a new process or a network RPC. [[engine-context:311]] [[model:187]]

### What gets loaded into CPU memory

The module-load primitive creates `CompiledKernelsContext` in an MGP state slot. The ordinary route hashes A1 into an ORC namespace, registers the object, resolves symbols and stores host function addresses. JITLink arranges allocated sections and resolves relocations. The namespace named `JITDylib` is a symbol/linking unit; it does not by itself imply a `.so` file or `dlopen`. The debug route explicitly creates such a file with the configured linker before loading it. [[cpu-primitives:74]] [[load:91]] [[load:160]] [[load:144]]

### What happens to GPU code

D1 was generated by the device backend during compilation. It is carried as a compiled payload with module/function names. The generated host `DeviceFunction` construction passes those bytes to the device runtime; the CUDA/HIP backend loads the module and finds the kernel entry. The source's enqueue path contains this load/check, so the guide shows a cold first use followed by cache reuse. It does not assume that every GPU kernel has been physically uploaded merely because Python `init` returned. Earlier preload or device-graph capture, when selected, can reach a load site earlier. [[legacy:3151]] [[enqueue:171]] [[cuda:2223]]

### Lifetimes and ownership

M1's native model handle retains its session and initialized model state. The initialized model holds references to callable resources and device contexts needed by outstanding work. G1 and P1 provide compiler inputs; the initialized runtime executes the compiled artifact, not Python's graph-building statements. The detailed guide distinguishes host .text/.data mappings, GPU module allocations, tensor allocations, argument packs and scratch, and shows their release/caching boundaries. [[native-session:196]] [[abi:141]]''', more=('abi','device-loading')),

chapter('execute', '02.5', 'Execute · Buffers, submission and completion',
    'output = model.execute(Q, K, V)',
    'Three 4096-byte input buffers reach GPU0. The host launches the specialized kernel; one GPU output Buffer is returned after runtime readiness checks.',
    '''caller: np.random.seed(123); construct Q, K, V after M1 exists [[entry:77]]
├─ NumPy float32[32,32] arrays → Buffer.from_numpy(...) [[entry:78]] [[py-buffer:183]]
└─ .to(GPU0) → allocate destination; copy 4096 bytes per tensor [[buffer:727]]
   ├─ Buffer::copyTo → device allocator → DeviceBuffer [[buffer:683]]
   └─ copyTo schedules the transfer using the device context [[buffer:694]]

caller: M1.execute(Q, K, V) [[entry:82]]
├─ Python _Model_execute validates contiguous inputs; passes Buffer objects [[api:118]]
├─ ModelHandle::executeDeviceTensors → input-count/type/device validation [[native-session:452]]
│  └─ prepareInput retains existing storage in Tensor AsyncValues [[native-session:484]]
├─ executePreparedDeviceTensors releases GIL → executeFromDriver [[native-session:529]]
├─ ExecutableModel::execute: chain, MGP state, weights, extras, Q/K/V [[model:373]]
│  └─ executeFn->invoke → mgp_generic_execute → host routine H1 [[model:384]] [[primitives:176]]
│     ├─ pack addresses of operand/result AsyncValue slots + constant resources [[primitives:98]]
│     ├─ resolve H1 from module/function indexes; invoke H1(void**) [[primitives:188]] [[primitives:234]]
│     ├─ generated allocation plan obtains O storage; output is a runtime buffer [[mgp-device:202]] [[mojo-primitives:745]]
│     ├─ FusedAttention.execute → fused_attention_gpu[BN=16,BD=8] [[kernel:105]]
│     └─ ctx.enqueue_function[kernel_func](Q,K,V,O, grid=(4,2), block=32) [[kernel:403]]
│        ├─ load/reuse D1's device function handle [[enqueue:171]] [[legacy:3158]]
│        ├─ encode device-layout arguments into aligned host slots [[legacy:3639]]
│        └─ native enqueue → cuLaunchKernelEx / hipModuleLaunchKernel [[bindings:542]] [[cuda:1610]] [[hip:1382]]
├─ GPU blocks read Q/K/V; compute owned output tiles [[kernel:333]]
│  ├─ loop over four key tiles: QKᵀ → running max/denominator → PV [[kernel:343]]
│  └─ rescale earlier output, merge current values, store O tile [[kernel:366]] {{attention-math}}
└─ await native results; drop internal chain; wrap existing output storage [[driver:137]] [[driver:159]] [[native-session:538]]
   └─ Python output=[O1]; print(output) leaves it on the selected device [[entry:83]]

optional readback (not present in the script): output[0].to(CPU()).to_numpy() [[py-buffer:186]]
└─ device→host copy plus host-read synchronization; no second kernel compilation {{synchronization}}''',
    '''### Why input wrapping and input transfer are separate

The example first constructs real NumPy arrays. `Buffer.from_numpy` creates a host Buffer view/owner around that storage; `.to(device)` takes the different-device branch, allocates destination storage, and requests a copy. Each 32×32 float32 tensor has 1024 elements and 4096 bytes. During `execute`, `prepareInput` wraps the existing device storage in native Tensor/AsyncValue objects; it does not upload those 4096 bytes again merely to cross the Python/C++ interface. [[entry:78]] [[buffer:683]] [[native-session:479]]

### Why there are two ABI boundaries

The selected `mgp_generic_execute` primitive calls H1 through `CompiledKernelAsyncABIFn = void(void**)`. `getKernelInputs` collects addresses of operand and result AsyncValue slots, followed by constant-resource data/length entries. H1 unpacks these values; it is not called with four bare Q/K/V/O pointers. The header also declares a different `void(void**, ssize_t*)` ABI, but that is not the call selected by this primitive. [[primitives:98]] [[primitives:188]] [[primitives:234]] [[abi:101]]

H1 then packages arguments for the **GPU launch ABI**: each host slot contains a value encoded to the device layout, and the driver receives addresses of those slots. The slots are not the Q/K/V matrices themselves. A kernel/function handle, stream handle, pointer to an AsyncValue slot, pointer to a launch-argument slot and GPU tensor pointer have different meanings. The specialized TileTensor layout determines the launch payload; the four source operands do not prove a universal four-pointer binary layout. [[legacy:3628]]

### Why submission does not mean completion

The driver enqueues work on a device stream. GPU execution, pending memory transfers, and host continuation have distinct readiness conditions. AsyncRT chains/values order the host plan; device streams/events order actual device operations. A host wrapper can be created before data is safe for an arbitrary host read. The result extraction path waits for its native results, while Buffer synchronization governs access and transfers. The driver guide traces the exact completion and copy sites instead of treating every call as a global GPU barrier. [[model:373]] [[native-session:530]] [[buffer:407]]

### Why output is allocated at runtime

The compiled graph describes the output type and allocation plan. A concrete O buffer is needed for an invocation and remains owned by the returned Buffer after the routine returns. Input buffers, output buffers, host argument metadata, and per-block scratch therefore have different lifetimes. M1 retains reusable code and initialization resources; the next call can supply fresh input buffers. [[abi:89]] [[kernel:74]]

### Why fused attention keeps only a score tile

For GPU N=D=32, each block owns 16 query rows and 8 output columns. It visits four tiles of 8 key/value rows. QK produces current scores; stable exponentials update a running row maximum and denominator; normalized current weights multiply V; the old output is rescaled onto the new normalization before adding PV. This retains m/l/O across key tiles and reuses S/E/PV scratch. It computes unscaled rank-2 attention; this caller has no mask, dropout, multiple heads, paged KV cache or serving protocol. [[kernel:331]] [[kernel:348]] [[kernel:366]]

### Why the final print is not a parity test

The source prints a list of Buffer objects. It does not explicitly copy O to the CPU or compare against a reference. The optional readback shown here is a teaching extension and triggers a transfer/synchronization boundary. A second execution reuses compiled/initialized code, but still needs valid input lifetimes and device ordering. [[entry:82]] [[buffer:727]]''', more=('attention-math','synchronization')),
]

CALLER_CODE = '''def main():
    device = Accelerator()  # Follow the available-GPU branch of the real source.
    N, D, BD, BN = 32, 32, 8, 16

    with Graph("fused_attention", ..., custom_extensions=[mojo_kernels]) as graph:
        q, k, v, *_ = graph.inputs
        results = ops.custom(name="modular_ops::fused_attention_custom", ...)
        graph.output(*results)

    session = InferenceSession(devices=[device])  # S1
    compiled = session.compile(graph)            # C1; graph is G1
    model = session.init(compiled)               # M1

    Q, K, V = ...  # NumPy → Buffer → GPU, three float32[32,32] tensors
    output = model.execute(Q, K, V)              # [O1]
    print(output)'''
