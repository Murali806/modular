"""Attention-specific artifact, ABI and GPU-driver deep dive."""
from attention_walkthrough_content import chapter

CHAPTERS = [
chapter('artifacts', 'A', 'What the compiler produces',
    'P1 + IR1 → generated host routines A1 + device image D1 → C1 / MEF',
    'A1 and D1 have different targets. C1 packages the runtime plan and its resources; the tensors are allocated later.',
'''caller: JitCompileKernelsPass consumes emitted Mojo + custom package P1 [[jit:547]]
├─ LIT→KGEN; select host ObjectCompiler; check native-code cache [[jit:567]] [[jit:575]]
├─ cache miss: KGENCompiler::runElaborationPipeline specializes routines [[jit:440]]
│  ├─ host code: output allocation, tensor adapters, launch/control calls [[kernel:74]]
│  └─ device offload: fused_attention_kernel[...] [[kernel:391]] [[kgen:610]]
├─ create per-target ObjectCompiler with triple/architecture/features [[kgen:720]]
│  └─ TargetBackendRegistry selects a registered backend for the target [[backend:119]]
├─ lower functions to LLVM dialect; translate into llvm::Module [[object:807]]
├─ NVIDIA offload: NVPTXBackend::emitObject [[nvptx:154]]
│  ├─ runLlc → PTX for the selected SM architecture [[nvptx:159]]
│  └─ CUDA CompilationDevice::compileFunction → cubin bytes D1 [[nvptx:204]] [[cuda-compiler:227]]
│     ├─ usual Linux compiler-library route [[cuda-compiler:227]]
│     └─ explicit MODULAR_NVPTX_COMPILER_PATH route can invoke ptxas [[cuda-compiler:139]]
├─ AMD alternative: AMDGPU target → object + device-bitcode libraries → device ELF D1 [[amdgpu:109]]
│  └─ device link uses shared-object backend and no-undefined option [[amdgpu:142]]
├─ host ObjectCompiler::emitArchive → retained native code resource A1 [[object:1023]]
├─ LowerCompiledRoutines → load A1 by symbols; call by module/function indexes [[lower-routines:102]]
└─ convertMOToMEF → serialized program/resources in C1 [[frontend:371]]
   OUT: byte buffers / descriptors, not yet H1 or a GPU function handle [[api:442]]''',
'''### Four formats, four roles

| Identity | Representation | Consumer | What it does not contain |
|---|---|---|---|
| P1 | `.mojoc`, serialized parametric package IR | Mojo analysis and specialization | The Python entry program or a ready-to-launch GPU function |
| IR1 | Mutable MLIR module, transformed through compiler dialects | Native pass managers | Runtime Q/K/V values |
| A1 / D1 | Host object resource / GPU-target executable image | Host linker / vendor module loader | Per-invocation tensor contents |
| C1 | Python CompiledModel holding native compiled artifact/MEF resources | `session.init` | A guaranteed set of resident GPU code pages |

### What selects the backend?

The host routine uses the CPU process's target. Each device offload has its own target triple, architecture and feature options. For NVIDIA a representative triple is `nvptx64-nvidia-cuda`; an architecture such as `sm_90a` is an example setting, not the GPU detected for this document. For AMD the corresponding example is `amdgcn-amd-amdhsa` with `gfx942`. The compiler registry maps the triple to a backend implementation; LLVM's target machinery then generates instructions appropriate to the target. [[object:146]] [[kgen:720]]

### .text and .data are sections, not tensors

An ELF section table organizes code, constants, writable globals, symbols and relocations for tools. A program-header table describes segments for loading executable/shared objects; a `PT_LOAD` segment can cover several sections. Relocatable host objects are handled by JITLink's object/link-graph machinery. NVIDIA cubin and AMD HSA code objects carry vendor-specific GPU code and metadata; their vendor loaders manage the final device resources. [[jit-engine:246]] [[cuda:2262]] [[hip:1968]] [[external:elf]] [[external:cubin]] [[external:amdgpu]]

```
A1 bytes in host memory                 D1 bytes in host memory
  code + constants + symbol references    code + kernel metadata + symbols
                ↓ JITLink                         ↓ CUDA / HIP loader
host executable mappings → H1          device executable resources → KD1

Q/K/V/O allocations are a separate runtime path.
```

### Why two compilers and two loaders?

The CPU must call runtime services and obey a host ABI. The GPU must execute a device kernel with its own data layout, address spaces and calling convention. Their artifacts cannot be exchanged merely because both use ELF containers. The host object may retain embedded image bytes or resource references for D1; exact section names/placement require inspecting emitted artifacts for a concrete compiler configuration. [[kgen:610]] [[legacy:3151]]

### Files versus memory

`emitArchive` names a compiler API returning a code buffer; the current `StaticArchiveLayer` registers an object interface/materialization unit. This does not prove a conventional `.a` containing multiple members. `export_mef(path)` would write C1 to disk, but the source calls only compile/init. The normal route needs no explicit `.so` for every kernel. [[archive-layer:108]] [[api:449]]'''),

chapter('host-link', 'B', 'Load A1 into the host process',
    'CompiledKernelsContext::load(A1, symbols) → host callable table H1',
    'Follow resource bytes through namespace registration, symbol materialization and executable memory. The debug .so branch has a separate origin.',
'''caller: initFn → mgp_module_load_cpu(...,A1,symbols,...) [[model:220]] [[cpu-primitives:56]]
├─ MGP StateContext slot → KernelExecutionState.modules[moduleIndex] [[cpu-primitives:62]]
├─ create CompiledKernelsContext(engine,loadDebuginfo) [[cpu-primitives:74]]
└─ load(A1,[hostSymbol0,...]) [[cpu-primitives:114]] [[load:85]]
   ├─ XXH3-64 of code-resource bytes → namespace key, e.g. "<A1-hash>" [[load:91]]
   ├─ copy bytes into an owned BufferRef [[load:93]]
   ├─ ordinary route: engine.addIfAbsent<StaticArchiveLayer>(key,bytes) [[load:160]]
   │  ├─ mutex protects namespace lookup / creation [[jit-engine-h:159]]
   │  ├─ existing key → no duplicate object registration [[jit-engine-h:160]]
   │  └─ StaticArchiveLayer::add(key,bytes) [[archive-layer:81]]
   │     ├─ getOrCreateDylib(key) → ORC symbol namespace [[archive-layer:82]]
   │     ├─ retain source bytes for non-owning MemoryBuffer view [[archive-layer:95]]
   │     ├─ getObjectFileInterface → symbols/flags for this object [[archive-layer:108]]
   │     └─ define materialization unit under resource tracker [[archive-layer:111]]
   ├─ lookup requested symbols in namespace [[load:166]]
   │  ├─ materialization emits object into ObjectLinkingLayer [[archive-layer:53]]
   │  └─ JITLink allocates, resolves references, applies relocations and finalizes [[jit-engine:246]]
   │     └─ InProcessMemoryMapper owns executable-memory allocation [[jit-engine:211]]
   └─ addKernel(name,index,address) → kernels[] + symbolTable[] [[load:169]] [[load:181]]

debug/profiler branch inside the SAME loader [[load:98]]
├─ write temporary object from A1 [[load:119]]
├─ invoke configured linker with -shared → temporary .so [[load:133]] [[load:144]]
└─ DynamicLibrarySearchGenerator::Load(.so) → then symbol lookup [[load:152]]''',
'''### Example values, one step at a time

| Step | Input | Result | Why |
|---|---|---|---|
| Hash | A1 byte buffer | key `a1_hash` | Name identical bytes consistently in this execution engine |
| Register | key + owned bytes | ORC namespace + pending symbols | Make definitions available without replacing G1 |
| Materialize | lookup of illustrative `host_attention_entry` | Link graph / object allocation | Obtain executable code for the requested symbol |
| Resolve | code reference to illustrative runtime helper | Relocation fixup to helper address | A machine-code call needs an address, not just a symbol name |
| Finalize | allocated sections + fixups | executable code address H1 | Complete memory protections and linker actions before use |
| Retain | H1, symbol name, module index | native function table | Runtime calls avoid repeating symbol-name lookup |

Names and addresses in this table are illustrative. The real mangled symbols are generated and appear in C1's code resource. The loader receives them rather than assuming the Python operation name is the exported machine-code symbol. [[load:166]]

### Memory example

```
object view                        host JIT memory
.text: generated host routine  →   executable allocation; H1 points here
.rodata: read-only constants    →   read-only allocation
.data/.bss: mutable state       →   writable allocation (if present)
relocations: named references   →   patched to linked addresses
```

The source selects an in-process mapper and an ORC ObjectLinkingLayer. LLVM owns object-format-specific linking and memory finalization. The mapping above explains responsibilities; it is not a captured address map, and the source's reservation granularity is not this kernel's code size. The final protection/relocation work crosses into LLVM. [[jit-engine:197]] [[jit-engine:246]] [[external:orc]]

### Why JITDylib does not mean dlopen

An ORC `JITDylib` is a symbol namespace and dependency/search unit. Ordinary A1 loading builds one in memory. By contrast, the debug path explicitly writes `.o`, generates `.so` with the configured linker, and uses a dynamic-library search generator. Installed `libcuda.so.1` or HIP libraries are yet another category: vendor runtime dependencies, not outputs of `session.compile`. [[load:144]] [[load:152]] [[cuda-driver:61]] [[hip-driver:60]]

### Why the engine must outlive the model's work

The code mappings, symbol namespaces and retained source bytes belong to the execution engine/layers. MGP state holds references to CompiledKernelsContext. A task still executing H1 cannot tolerate the engine freeing that code. The header makes the lifetime requirement explicit. Returning a Python Buffer does not grant ownership of the executable memory to that Buffer. [[abi:141]] [[archive-layer:95]]'''),

chapter('device-load', 'C', 'Load D1 through CUDA or HIP',
    'DeviceContext.loadFunction(moduleName, kernelName, imageBytes, options)',
    'A cold first use creates a device module and function handle. A matching function-cache hit reuses them.',
'''caller: generated host routine → fused_attention_gpu → enqueue_function [[kernel:105]] [[kernel:403]]
├─ compile_function[kernel_func] obtains the compiler-provided descriptor [[enqueue:171]]
│  ├─ _compile_code[...] intrinsic → module name, kernel name and D1 bytes [[legacy:3151]]
│  └─ _load_function(...) → native AsyncRT binding [[legacy:3158]] [[legacy:3031]]
├─ vendor runtime library is available through its loader [[cuda-driver:61]] [[hip-driver:60]]
├─ CUDADeviceContext::loadFunction constructs FunctionCacheKey [[cuda:2217]]
│  ├─ key = module/code identity + function name + debug/shared/optimization options [[cuda:2217]]
│  ├─ cache hit → return a DeviceFunction retaining the cached module/function [[cuda:2223]]
│  └─ cache miss → bind CUDA context; set compilation options [[cuda:2259]]
│     ├─ cuModuleLoadDataEx(&MD1,D1.data(),options) [[cuda:2262]]
│     │  └─ vendor loader validates image, resolves device code/global resources [[external:cuda-module]]
│     ├─ cuModuleGetFunction(&KD1,MD1,kernelName) [[cuda:2268]]
│     ├─ set supported function attributes (including optional shared bytes) [[cuda:2271]]
│     └─ retain MD1/KD1 in device function cache; return wrapper [[cuda:2288]]
└─ AMD equivalent [[hip:1940]]
   ├─ hipModuleLoadData(&MD1,D1.data()) [[hip:1968]]
   └─ hipModuleGetFunction(&KD1,MD1,kernelName) [[hip:1972]]

OUT: KD1 plus owning context/module references, ready for launch submission [[cuda:2291]]''',
'''### What is passed to the loader?

```
moduleName   = compiler-generated module identity
functionName = compiler-generated device entry symbol
code         = D1 cubin bytes (NVIDIA) or HSA code-object bytes (AMD)
context      = S1's selected GPU device context
result       = device-module handle MD1 + kernel-function handle KD1
```

The code bytes are already prepared by the device backend in the traced compile path. The CUDA module API also supports other image types, but this walkthrough's NVIDIA backend obtains cubin through the compilation device. Source function names and generated entry symbols are not guaranteed to be identical. [[nvptx:204]] [[cuda:2262]]

### What “place ELF on the GPU” really means

The ELF container remains input to a loader. It describes executable code, symbols, metadata and possibly global data. The driver creates appropriate device resources and returns handles; it need not copy file headers and section tables into an ordinary Q/K/V Buffer. Tensor memory allocations and executable module allocations have separate roles. Exact physical addresses, demand paging and lazy code residency belong to the vendor driver. Module load success does not by itself establish that every code page has been eagerly resident since `init`. [[cuda:2262]] [[external:cuda-module]] [[external:cubin]]

### Which dynamic libraries are involved?

| Library / object | Where it comes from | Why it is loaded |
|---|---|---|
| MAX Python extension | Build/install dependencies | Python-to-native bindings |
| NVPTX compilation library | Installed compiler/runtime toolchain | PTX→cubin compilation route |
| `libcuda.so.1` | NVIDIA driver installation | CUDA module/launch/memory APIs |
| HIP/HSA runtime libraries | ROCm installation | AMD device execution/runtime APIs |
| temporary debug `.so` | Created by CompiledKernelsContext::load from A1 | Optional host code debug/profiler loading |
| D1 cubin / HSA ELF | Device compiler output | GPU module loader input, not a host `dlopen` library |

The native CUDA/HIP driver wrapper locates its installed libraries and resolves entry points. Those library paths may be configurable; a fixed absolute installation path is not assumed. [[cuda-driver:61]] [[hip-driver:107]] [[cuda-compiler-driver:49]] [[load:133]]

### Why cache by code identity and options?

A same-named kernel can have different specializations or code. Shared-memory/debug/optimization options also affect loading. The cache key includes these distinctions and returns owning references, avoiding driver load/get-function calls on the normal hit branch. Loading is reached when this generated site runs; a selected preload/capture route could run it before ordinary execution. [[cuda:2217]] [[cuda:2223]]'''),

chapter('launch', 'D', 'Cross the two ABIs and submit GPU work',
    'H1(AsyncValue slots) → pack device arguments → launch KD1 on a stream',
    'Follow the references and values. The host calls a driver API; the GPU later executes the selected kernel.',
'''runtime execution: mgp_generic_execute(state,operands,results,indexes,...) [[primitives:176]]
├─ select H1 using state slot, module index and function index [[primitives:188]]
├─ choose CPU task affinity from common operand device contexts [[primitives:192]]
├─ queued route retains operand/result/context references [[primitives:212]]
└─ getKernelInputs → H1(void **slotAddresses) [[primitives:225]] [[primitives:234]]
   ├─ H1 unpacks input/output AsyncValues and constant resources [[primitives:98]]
   ├─ output allocation primitive requests 4096 bytes for O1 [[mgp-device:202]] [[mojo-primitives:745]]
   ├─ FusedAttention.execute adapts tensors to TileTensors [[kernel:91]]
   └─ fused_attention_gpu → enqueue_function[kernel_func] [[kernel:105]] [[kernel:403]]
      ├─ dimensions: grid=(32/8,32/16)=(4,2); block=(32,1,1) [[kernel:408]]
      ├─ load/reuse device function KD1 [[enqueue:171]]
      ├─ encode each argument to the target's device layout [[legacy:3628]]
      │  ├─ allocate aligned argument storage; write encoded value [[legacy:3639]]
      │  └─ dense_args_addrs[i] points to the encoded HOST slot [[legacy:3643]]
      └─ native enqueue receives KD1, dimensions, arguments and context/stream [[legacy:3664]]
         ├─ NVIDIA: cuLaunchKernelEx(config,KD1,kernelParams,extra) [[cuda:1610]]
         └─ AMD: hipModuleLaunchKernel(KD1,grid,block,...,stream,args,...) [[hip:1382]]
            └─ driver/runtime queues work; GPU consumes dispatch at its turn [[external:cuda-launch]]

GPU: eight blocks each own one 16×8 output tile [[kernel:378]]
└─ four key tiles → QK / normalization / PV → write O1 [[kernel:343]]''',
'''### Host ABI example

`mgp_generic_execute` obtains a `CompiledKernelAsyncABIFn`, whose C++ type is `void(void**)`. It builds an array containing addresses of operand and result AsyncValueRef slots, then constant-resource pointer/size pairs. The generated shim knows how to unpack these values. An input slot can refer to `Tensor(shape=[32,32], storage=Q_gpu)`. A result slot will be fulfilled with the output's runtime value. [[abi:101]] [[primitives:98]]

### Device argument-pack example

```
GPU data allocation         host argument storage           driver argument array
Q_gpu, 4096 bytes      ←    slotQ = encoded Q descriptor ←   args[0] = &slotQ
K_gpu, 4096 bytes      ←    slotK = encoded K descriptor ←   args[1] = &slotK
V_gpu, 4096 bytes      ←    slotV = encoded V descriptor ←   args[2] = &slotV
O_gpu, 4096 bytes      ←    slotO = encoded O descriptor ←   args[3] = &slotO
```

This is a schematic grouping by the four source TileTensor arguments. Exact binary fields and slot count depend on specialization, device layout and compiler lowering. A descriptor can contain a device pointer and shape/layout fields; some properties are compile-time constants. The driver receives **addresses of encoded argument storage**, not the matrix values and not a direct call into GPU instruction memory. [[legacy:3639]] [[external:cuda-launch]]

### Stack, heap, shared and device memory

| Storage | Example owner / lifetime | Purpose |
|---|---|---|
| Host stack / temporary vectors | Launch helper, until argument values are consumed | Argument metadata and slot-address array |
| Host heap / reference-counted objects | Session/model/task/result owners | Runtime objects, code resources and references |
| GPU tensor allocations | Buffer owners and outstanding runtime work | Q/K/V/O data |
| GPU module resources | Driver module/function cache owners | Device instructions, globals/constants when present |
| Per-thread private storage | Kernel execution | Locals; register/spill placement is compiler-dependent |
| Per-block shared storage | Cooperative work in one block, when the kernel/helpers request it | Block-local sharing; separate from host heap |

The kernel's `stack_allocation` expresses local temporary tensor storage. It does not mean a host CPU stack is transferred to the GPU. Actual register/spill/shared usage must be established from emitted code or device measurement. [[kernel:333]] [[legacy:3628]]

### Is this RPC? What is host↔GPU IPC?

The shown route has a host call into the installed CUDA/HIP runtime, which submits a device command. It is not an HTTP call, a MAX Serve request, a remote Python function or an RPC to a peer process. Device work is asynchronous offload with kernel arguments and stream ordering. The only separate compiler process in this example's normal preparation path is `mojo precompile`. [[mojo-run:74]] [[cuda:1610]]

Below the vendor API, command queues, memory mappings, OS driver interfaces and completion mechanisms connect CPU and GPU. Discrete-GPU data may cross an interconnect; physical transport and placement depend on platform/runtime configuration. AMD's documented AQL queues are one device-dispatch mechanism; this MAX source calls HIP rather than constructing those packets here. CUDA/HIP IPC APIs for sharing resources between processes are separate APIs and are not part of this traced example. [[external:amdgpu]] [[external:cuda-launch]]''', more=('attention-math',)),

chapter('completion', 'E', 'Copies, completion and retained resources',
    'Q/K/V upload → ordered dispatch → native result → optional O1 readback',
    'A returned Buffer, a completed CPU task and a completed host read have distinct contracts.',
'''before execute: NumPy arrays become host Buffers [[entry:78]] [[py-buffer:183]]
├─ Buffer.to(GPU0) detects different devices → copyTo [[buffer:727]]
│  ├─ allocate destination through the target device [[buffer:685]]
│  └─ DeviceBuffer.copyTo requests host→device transfer [[buffer:694]]
└─ execute inputs reuse that destination storage [[native-session:479]]

native execution [[driver:193]]
├─ await initialized executable, invoke runtime execute plan [[driver:204]] [[driver:212]]
├─ queued primitive captures input/output/context references [[primitives:212]]
│  ├─ bind device context; invoke H1; unbind [[primitives:229]]
│  └─ release captured references; then resolve indirect outputs [[primitives:254]]
├─ extractDriverResults → awaitOrError(results) [[driver:137]]
├─ discard leading internal chain; preserve tensor aliases [[driver:151]]
└─ wrap existing result memory as DeviceBuffer, then Python Buffer [[driver:173]] [[native-session:538]]
   OUT: output=[O1], with O1 on GPU0; source only prints this list [[entry:83]]

optional host read (teaching extension) [[py-buffer:186]]
├─ O1.to(CPU()) → destination host buffer + device→host transfer [[buffer:683]]
├─ to_numpy() → np.from_dlpack(cpu_buffer) [[py-buffer:197]]
└─ host-access path synchronizes the Buffer's storage before exposing it [[buffer:416]]

next call may reuse S1, C1, M1, H1 and the cached KD1 [[cuda:2223]]
└─ fresh input/output buffers still require valid ownership and ordering [[primitives:212]]''',
'''### What moves and what remains

For this shape each tensor has `32×32×4 = 4096` bytes. There are three input transfers (12,288 logical bytes) and a 4096-byte output if the reader adds explicit readback. These are tensor payload counts, excluding allocator overhead, protocol traffic, padding and caches. This is arithmetic about the declared shapes, not measured bus traffic. [[entry:36]] [[entry:78]]

G1/P1 need not be rebuilt for each call to M1. Code/resources remain with their runtime owners, while each call has data and result lifetimes. `Buffer.from_numpy` can wrap a contiguous host array without copying; `.to(GPU0)` requests the device transfer. Native `prepareInput` then adapts already-device-resident storage without uploading it again. [[py-buffer:169]] [[buffer:727]] [[native-session:479]]

### Why a CPU task is not a device-completion event

H1 executes CPU instructions that can enqueue GPU work. Host return establishes that this portion of the host routine has returned, not that the entire GPU is idle. Device streams/events and runtime dependencies determine when work and buffers are ready. `awaitOrError(results)` waits for the result AsyncValues used by result extraction. Buffer host access has its own synchronization path. The exact stream/event details are in the native device backend; do not replace them with an invented global synchronization after every API. [[primitives:234]] [[driver:137]] [[buffer:416]]

### Why retain inputs through submission?

The queued primitive owns a closure with operand and output references and a device context. Those objects keep referenced storage/services valid while host work is in progress. The generated routine/runtime associates device operations with their buffer lifetimes. The caller must also obey the API's ownership contract when using external memory. Runtime results may alias; result extraction uses a cache and shallow tensor borrowing to avoid double ownership errors. [[primitives:212]] [[driver:151]]

### Initialization readiness versus execution readiness

`createExecutableModel` already waited for setup before returning M1. This execution path is not the first place that validates a failed initialization. `ExecutableModel::execute` supplies the initial chain, MGP state, weights and extra init results to the execute function, followed by the user inputs. The weights registry is empty for this example; Q/K/V are not loaded model weights. [[engine-context:91]] [[model:373]]

### No server or cross-process transport is added

The Python process, MAX runtime workers and vendor user-space API participate locally. This path contains no API server, KV-cache manager, batching service or network protocol. Vendor-driver internals can use mechanisms beyond what is visible here; the animations stop at the documented dispatch/completion boundary instead of presenting physical DMA scheduling as source-proven MAX behavior. [[entry:23]] [[driver:193]]''', more=('synchronization',)),
]

EXTERNAL = {
 'bazel-macros': dict(title='Bazel · Starlark legacy macros', url='https://bazel.build/extending/legacy-macros',
               note='A def-style Starlark macro runs while loading a BUILD package and declares rule targets. It is separate from rule analysis and action execution.'),
 'bazel-tags': dict(title='Bazel · common target attributes and tags', url='https://bazel.build/reference/be/common-definitions#common-attributes',
               note='The manual tag affects wildcard target/test selection. The local macro forwards the caller’s tags to its generated example test.'),
 'bazel': dict(title='Bazel · phases of a build', url='https://bazel.build/run/build',
               note='Bazel separates loading, target analysis and action execution. The local Starlark macro establishes this example’s dependencies.'),
 'cuda-module': dict(title='NVIDIA CUDA Driver API · module management', url='https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html',
               note='The module API accepts prepared image bytes, returns module/function handles and documents lazy-loading behavior. Physical placement remains driver-managed.'),
 'cuda-launch': dict(title='NVIDIA CUDA Driver API · execution control', url='https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html',
               note='The launch API receives launch dimensions, a function and stream, and pointers to argument storage. Enqueue and completion are separate events.'),
 'cubin': dict(title='NVIDIA · CUDA binary utilities', url='https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html',
               note='A cubin uses ELF to carry CUDA executable code and associated information; it is distinct from the host runtime library.'),
 'amdgpu': dict(title='LLVM · AMDGPU code objects and kernel dispatch', url='https://llvm.org/docs/AMDGPUUsage.html',
               note='AMDHSA ELF metadata and dispatch conventions define GPU code-object and kernel-argument responsibilities. The MAX example reaches them through HIP.'),
 'elf': dict(title='System V ABI · ELF program loading', url='https://gabi.xinuos.com/elf/07-pheader.html',
               note='Sections organize object contents; program headers describe loadable segments. A loadable segment can have a zero-filled tail beyond its file bytes.'),
 'orc': dict(title='LLVM · ORCv2 design', url='https://llvm.org/docs/ORCv2.html',
               note='ORC symbol materialization and JITLink separate registration, symbol resolution and executable-memory finalization.'),
}
