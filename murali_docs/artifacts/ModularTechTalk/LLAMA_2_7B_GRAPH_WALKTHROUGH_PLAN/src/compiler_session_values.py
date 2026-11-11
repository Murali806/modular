"""Schematic constructor values, grounded in native member declarations."""
EXPLANATION = '''

**InferenceSession does more than normalize devices.** It retains compilation settings, device handles, runtime configuration, and a native engine context. Its methods later compile graphs and initialize executable models; it also configures diagnostics/profiling. Constructor completion alone produces neither compiled code nor a Model.

**Example object snapshot · CPU add-one, immediately after construction**

~~~text
S1: Python InferenceSession
├─ num_threads = None                     Python argument, native policy becomes 0
└─ _impl → N1: Core::InferenceSession
   ├─ devices = [CPU0]                    retained driver handle
   ├─ sessionCompileConfig = CFG1: CompilationRequest
   │  ├─ modelSource = monostate          no model supplied yet
   │  ├─ pipelineName = None              chosen later for compilation
   │  └─ passthroughConfigs = []          no constructor extensions
   ├─ runtimeConfig
   │  ├─ numThreads = 0                   automatic selection, not zero workers
   │  ├─ apiLanguage = "python"
   │  └─ configured devices = [{spec=CPU0.deviceSpec, context=CPU0.ctx}]
   └─ runtimeContext → E1: Engine::Context (owned by N1)
      ├─ config = runtimeConfig copy
      ├─ maxContext = ContextRef(CX1)     reference to shared MAX M::Context
      ├─ cpuDevice = reference to HRT1    host AsyncRT CPUDevice
      └─ crashRecoveryContext            configured stack/crash recovery

CX1: MAX M::Context (reference-counted native service container)
├─ cpuDeviceRef → HRT1                   retained runtime reference
└─ typed service storage
   ├─ M::Config                          loaded process configuration
   ├─ Init::Options                      stored options used for reuse checks
   ├─ Telemetry::TelemetryContext        initialized with programName="max-engine"
   └─ GML::MaxEngineTelemetryContext     emplaced by Engine::Context::create

GPU variant: devices=[GPU0,CPU0]; configured-device pairs include both.
E1 and CX1 are still host-side native objects, not GPU memory allocations.
~~~
'''
EXPLANATION += '''
These labels are an explanatory snapshot, not Python `repr`, captured addresses, or an exhaustive list of process-dependent services. Native members are declared in [InferenceSession.h](sources/max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h#L95). The [Engine::Context constructor](sources/max/lib/max_license/API/c/EngineContext.cpp#L217) copies runtime config and retains the MAX context and CPU runtime reference. [M::Context](sources/Support/include/Support/Context.h#L30) is a typed service container; it is distinct from the MLIR Context that owns graph IR.

**Created or reused · same shared object, separate owned engine wrappers**

~~~text
First session:
  getOrCreateContext("max-engine", options)
    → no current global MAX context → create CX1 → retain HRT1 + services
  N1 owns E1; E1.maxContext refers to CX1

Later compatible session:
  getOrCreateContext("max-engine", same options)
    → current CX1 exists → ContextRef::copy(existing)
  N2 owns E2; E2.maxContext also refers to CX1

Different Init::Options with an existing global context:
  → fatal configuration mismatch; no silent incompatible reuse
~~~

The exact reuse decision is in [Init::getOrCreateContext](sources/Init/lib/Init.cpp#L116). A configuration-supplied MAX context follows the [earlier direct-retain branch](sources/max/lib/max_license/API/c/EngineContext.cpp#L175). Shared CX1 does not imply that every session has the same runtime device list: E1/E2 retain their own runtime configuration copies.

**populateCompileConfig · concrete before and after values**

| Moment | CompilationRequest contents |
| --- | --- |
| Original constructor input | `customExtensions=[]` |
| Default CFG1 | `modelSource=monostate`, `pipelineName=None`, `passthroughConfigs=[]` |
| `populateCompileConfig([])` | Copies the default request; adds no kernel-library entries |
| Optional extension illustration | `customExtensions=["/tmp/addon.mojoc"]` (must already exist) |
| Request with that extension | `passthroughConfigs=[KernelLibraryDriverConfig(path="/tmp/addon.mojoc")]`; source/pipeline still unset |
| Later compile of original G1 | Copies session CFG1 and adds G1's P1 path plus model/pipeline settings |

An extension is a **Mojo implementation package path**, not a device, Python class, or machine-code kernel. For example, `/tmp/addon.mojoc` can contain registered custom kernels. This path is illustrative; no such file is claimed to exist here. The [native extension validator](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L600) rejects missing or unsupported package paths. [CompilationRequest](sources/max/include/max/max_license/API/c/CompiledModels.h#L65) copies source/pipeline fields and clones its driver configs; [addKernelLibraryPath](sources/max/include/max/max_license/API/c/CompiledModels.h#L135) stores a KernelLibraryDriverConfig.

The original add-one call supplies no session extensions. Its Graph already owns P1's package paths, which [compile_async](sources/max/python/max/engine/api.py#L1022) supplies later. A separate optional API example is `InferenceSession(devices=[device], custom_extensions=[Path("/tmp/addon.mojoc")])`; Python processes these extensions before entering the native constructor. Session construction prepares the ownership/configuration needed for later work; it does not execute or specialize those kernels.
'''
