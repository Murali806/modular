"""Build the offline MAX engine / graph compiler / Mojo lifecycle reader."""
from __future__ import annotations

import colorsys
import html
import json
from pathlib import Path

from cells import CELLS, ENGINE, GC, NATIVE, cpp, py

PAGE_NAME = 'max_engine_graph_compilation_mef_and_inference.html'
GUIDE_NAME = 'MAX_ENGINE_GRAPH_MEF_AND_INFERENCE.md'
MOJO = 'max/kernels/src/graph_compiler/builtin_kernels/elementwise.mojo'
MOJO_DOC = 'Mojo/docs/compiler/MojoCompilerWalkthrough.md'
CORE = 'max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp'
MEF_MODEL = 'GraphCompiler/lib/MEFModel/MEFModel.cpp'
EXECUTOR = 'MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp'

SELECTED = ['spec_trace', 'llama_trace', 'graph_init', 'graph_output', 'compile_graphs',
            'compile_reuse', 'compile_async', 'compile_wait', 'compile_module',
            'native_session', 'engine_compile', 'engine_sync', 'compiled_create',
            'frontend', 'convert_mef', 'mo_pipeline', 'map_pipeline', 'mogg_pipeline',
            'mgp_pipeline', 'gc_pipeline', 'run_pipeline', 'init_all', 'native_init']
REF_SPECS = {c['id']: c for c in CELLS if c['id'] in SELECTED}
REF_SPECS.update({c['id']: c for c in [
    py('session', ENGINE, 'InferenceSession.__init__', 'InferenceSession'),
    py('artifact', ENGINE, 'CompiledModel', 'CompiledModel'),
    py('export', ENGINE, 'CompiledModel.export_mef', 'export_mef'),
    py('read', ENGINE, 'read', 'read(MEF)'),
    py('execute', ENGINE, '_Model_execute', 'Model.execute'),
    py('compile_only', ENGINE, 'CompileOnlyModel', 'CompileOnlyModel'),
    py('claim_import', 'max/python/max/engine/_precompiled_mefs.py', 'MefStore.claim_import', 'claim_import'),
    py('claim_export', 'max/python/max/engine/_precompiled_mefs.py', 'MefStore.claim_export', 'claim_export'),
    py('store_import', 'max/python/max/engine/_precompiled_mefs.py', 'MefStore.for_import', 'MefStore.for_import'),
    py('write_manifest', 'max/python/max/engine/_precompiled_mefs.py', 'MefStore.write_manifest', 'write_manifest'),
    cpp('serialize', GC, 'convertMGPToMEF', 'ErrorOr<BufferRef> M::MO::convertMGPToMEF', 'Serialize MEF'),
    cpp('setup', NATIVE + 'EngineContext.cpp', 'createExecutableModel', 'createExecutableModel(const M::Model *model,', 'Create executable model', prefix=1),
    cpp('input_validation', CORE, 'validateInputSpecList', 'void validateInputSpecList(', 'Input shape / dtype checks'),
    cpp('native_execute', CORE, 'ModelHandle::executeDeviceTensors', 'M::Core::ModelHandle::executeDeviceTensors(', 'Native input preparation', prefix=1),
    cpp('release_gil', CORE, 'ModelHandle::executePreparedDeviceTensors', 'ModelHandle::executePreparedDeviceTensors(', 'Release GIL / execute', prefix=1),
    cpp('driver_execute', NATIVE + 'ModelExecution.cpp', 'executeFromDriver', 'executeFromDriver(const M::Engine::Context &context,', 'Driver execution', prefix=1),
    cpp('driver_results', NATIVE + 'ModelExecution.cpp', 'extractDriverResults', 'extractDriverResults(const M::Engine::Context &context,', 'Await / extract outputs', prefix=1),
    cpp('runtime_execute', MEF_MODEL, 'ExecutableModel::execute', 'ExecutableModel::execute(MutableArrayRef<AnyAsyncValueRef> args) const', 'ExecutableModel.execute', prefix=1),
    cpp('setup_wait', MEF_MODEL, 'ExecutableModel::waitForModelSetup', 'ErrorOrSuccess ExecutableModel::waitForModelSetup()', 'Setup dependencies'),
    cpp('mef_invoke', EXECUTOR, 'MEFInvokableImpl::invoke', 'void MEFInvokableImpl::invoke(', 'MEF invocation'),
    cpp('ready_counts', EXECUTOR, 'MEFExecutor::decrementOperandsNotReadyCounts', 'void MEFExecutor::decrementOperandsNotReadyCounts(', 'Ready primitive worklist'),
    cpp('async_users', EXECUTOR, 'MEFExecutor::processUsedByRecords', 'void MEFExecutor::processUsedByRecords(', 'Result readiness callbacks'),
]})
REF_SPECS['mojo_add'] = dict(id='mojo_add', file=MOJO, symbol='Add', language='mojo',
    anchor='@extensibility.register("mo.add")', until='@extensibility.register("mo.sub")', purpose='Mojo mo.add registration')
REF_SPECS['mojo_compiler'] = dict(id='mojo_compiler', file=MOJO_DOC, symbol='Mojo compiler phases', language='text',
    anchor='## High-Level Architecture', until='## Phase 1:', purpose='Mojo compiler architecture')
EXTRA_FILES = sorted({c['file'] for c in REF_SPECS.values()})


def obj(key, name, short, owner, kind, role, refs):
    return dict(id=key, name=name, short=short.split('|'), owner=owner, kind=kind, role=role, refs=refs.split())


OBJECTS = [
    obj('session', 'InferenceSession', 'InferenceSession', 'Python', 'object → native session', 'Selects devices, runtime thread settings, extensions and artifact-store policy. Adds the host CPU if absent. A session is neither a model nor a separate OS process.', 'session'),
    obj('devices', 'CPU / Accelerator handles', 'CPU + Accelerator', 'Driver', 'device handles / hardware context', 'Device handles identify configured targets. Compilation may use virtual devices; execution requires real device support. CPU is included in the session even when an accelerator is selected.', 'session'),
    obj('spec', 'GraphSpec + Llama module', 'GraphSpec / module', 'Python', 'recipe / module objects', 'The built-in Llama Graph API route supplies a module and symbolic input types. GraphSpec.trace delegates to Llama3.trace. ModuleV3 takes a different staging route.', 'spec_trace llama_trace compile_graphs'),
    obj('types', 'Symbolic input contract', 'Input types / dims', 'Graph', 'tensor metadata', 'Input types describe dtype, shape (including symbolic dimensions), and placement. They are descriptions rather than per-inference tensor values.', 'graph_init'),
    obj('graph', 'Graph / MLIR module', 'Graph / MLIR', 'Graph', 'Python object + native IR', 'Graph owns symbolic graph structure. Forward runs during construction to record operations; it does not compute model outputs on a GPU.', 'graph_init llama_trace'),
    obj('kernel_library', 'KernelLibrary declarations', 'KernelLibrary', 'Graph', 'compiler metadata', 'The graph imports kernel declarations. compile_async seeds those declarations into the module so native graph compilation can skip repeating package loading.', 'graph_init compile_async mo_pipeline'),
    obj('ops', 'Symbolic graph operations', 'Symbolic operations', 'Graph', 'IR operations / dependencies', 'Embedding, decoder blocks and projections become graph IR on the Llama trace route. The Add example in chapter 03 demonstrates a registration mechanism, not the full Llama kernel schedule.', 'llama_trace graph_output'),
    obj('contract', 'Graph.output contract', 'Verified outputs', 'Graph', 'signature / terminator / verification', 'Flattens outputs, adds dependency chains, sets signatures and mo.output, names the interface, then verifies MLIR. This finalizes the symbolic contract before native compilation.', 'graph_output'),
    obj('lock', '_COMPILATION_LOCK', 'Compilation lock', 'Python', 'process-wide threading lock', 'Serializes Graph/Module native compilation in this process. Acquire before native scheduling; release on setup failure or via the compile completion callback. It is not an IPC semaphore.', 'compile_module'),
    obj('handle', 'Pending compilation result', 'Pending handle', 'Engine', 'asynchronous result / synchronization', 'compile_async returns a CompiledModel backed by a pending native result. compile waits; export or initialization can also resolve pending compilation. A handle does not itself execute the model.', 'compile_async compile_wait export'),
    obj('pool', 'Runtime CPU worker pool', 'CPU runtime pool', 'AsyncRT', 'threads in the application process', 'Native compileModel schedules a CPU runtime task. Native initialization uses a CPU worker continuation. No separate compiler process or fixed thread count is implied.', 'engine_compile native_init'),
    obj('clone', 'Compiler-owned MLIR clone', 'MLIR module clone', 'Compiler', 'native IR ownership', 'The frontend clones the input module before lowering, resolves compiler settings and checks whether declarations are already seeded. The caller graph and this transformed clone are distinct.', 'frontend'),
    obj('cache', 'Native model-IR → MEF cache', 'Native IR cache', 'Compiler', 'optional disk cache', 'If enabled and available, the frontend derives a cache key from the module/options/version and returns a matching MEF before lowering. This cache is separate from Python MefStore.', 'frontend'),
    obj('mo', 'MO stage', 'MO · graph IR', 'Compiler', 'high-level graph passes', 'Imports definitions when needed; performs configured shape/symbolic optimization, layout inference and target validation. Pass scheduling does not establish which exact Llama transformations occur.', 'mo_pipeline gc_pipeline'),
    obj('map', 'MAP stage', 'MAP · structured IR', 'Compiler', 'fusion / kernel-body transforms', 'Fuses eligible structured computations, allocates accumulating buffers and outlines kernel bodies. Several graph operations may share a kernel; this animation makes no kernel-count claim.', 'map_pipeline gc_pipeline'),
    obj('mogg', 'MOGG stage', 'MOGG · kernel calls', 'Compiler', 'kernel / routine IR', 'Legalizes calls and operands, outlines routines, then transitions into the runtime-model representation. These are compiler representations, not OS components.', 'mogg_pipeline gc_pipeline'),
    obj('mgp', 'MGP stage', 'MGP · runtime IR', 'Compiler', 'allocation / dependency planning', 'Plans buffer reuse and allocations, lowers primitives and dependencies, handles configured device-graph work, merges equivalent routines and emits Mojo. Device-graph passes are distinct from Graph.output.', 'mgp_pipeline gc_pipeline'),
    obj('add', 'Mojo Add registration', 'mo.add → Mojo Add', 'Mojo', 'registered kernel implementation', 'The real Add struct is registered as mo.add. Its elementwise overloads use dtype/SIMD-width or tile-layout parameters. Registration provides an implementation building block; it does not imply one launch per graph op.', 'mojo_add'),
    obj('mojo_source', 'Emitted Mojo routines', 'Emitted Mojo', 'Compiler', 'generated routine representation', 'MGP emits Mojo routines before the enclosing graph pipeline schedules kernel JIT and final runtime-call lowering. The handwritten Add registration and emitted model routines have different roles.', 'mgp_pipeline gc_pipeline'),
    obj('lit', 'Mojo parse / semantic checks', 'Parse → LIT checks', 'Mojo', 'compiler phase', 'The compiler walkthrough describes parsing/name/type resolution, semantic control-flow and lifetime checks, then LIT lowering. This is a compiler architecture view, not a captured compile trace.', 'mojo_compiler'),
    obj('kgen', 'KGEN pre-elaboration', 'KGEN / optimization', 'Mojo', 'parametric compiler IR', 'Pre-elaboration optimization reduces parametric IR before specialization. Compile-time generators retain parameters until elaboration.', 'mojo_compiler'),
    obj('concrete', 'Elaboration / concrete functions', 'Elaborate parameters', 'Mojo', 'compile-time specialization', 'Elaboration resolves generators and compile-time parameters into concrete functions/types. dtype, SIMD width and tile layout are illustrated by Add; this page does not choose actual target specializations.', 'mojo_add mojo_compiler'),
    obj('llvm', 'LLVM / target lowering', 'LLVM → target code', 'Mojo', 'compiler backend', 'Concrete functions undergo lowering/optimization and target code generation. CPU and GPU targets have distinct backend requirements. A target binary is code; no inference input has run yet.', 'mojo_compiler gc_pipeline'),
    obj('code', 'Compiled kernel code', 'Target kernel code', 'Compiler', 'compiled routines / target binaries', 'JIT-compiled routines and final runtime calls contribute to the lowered executable artifact. Fusion and routine reuse mean graph-op counts, routine counts and runtime launch counts differ.', 'gc_pipeline'),
    obj('mef', 'MEF bytes / file', 'MEF artifact bytes', 'Artifact', 'serialized executable format', 'convertMGPToMEF prepares device primitives and serializes the lowered model. MEF can include constant resources. External weights, initialization state and per-call inputs remain separate concepts.', 'serialize export compiled_create'),
    obj('store', 'Python MefStore', 'Python MefStore', 'Python', 'artifact selection policy + locked metadata', 'Explicit session store overrides the process-wide default. Graph name/signature select the artifact; optional verify_body checks its fingerprint. The store lock protects its metadata. Declined keys compile normally; missing accepted artifacts raise.', 'compile_reuse claim_import claim_export store_import'),
    obj('manifest', 'Export manifest', 'Manifest / identity', 'Artifact', 'disk metadata', 'Export records key, graph name/signature/body fingerprint, writes MEF and persists the manifest. Exporting and importing cannot both be requested on the same session.', 'claim_export write_manifest session'),
    obj('artifact', 'CompiledModel / CompiledModels', 'CompiledModel', 'Engine', 'uninitialized artifact handle', 'Wraps compiled/deserialized native models. It cannot execute. It can be initialized repeatedly. read(MEF) bypasses rebuilding the Python Graph; a GraphSpec reuse route has already traced.', 'artifact read compile_reuse compiled_create'),
    obj('registry', 'Named weight registry', 'Weights: name → value', 'Python', 'mapping of actual weight buffers', 'init_all checks expected weight names and CPU/device category when Graph metadata is available, then contiguity. read/path-compiled artifacts have no Python expected-weight metadata. Weights are distinct from input buffers.', 'init_all compile_graphs'),
    obj('setup', 'Native model setup', 'Native model setup', 'Engine', 'runtime initialization', 'Model.setup receives CPU runtime, weights registry and device-context map. createExecutableModel waits for setup before returning a usable executable. Compilation alone does not initialize this device state.', 'setup native_init setup_wait'),
    obj('context', 'Persistent executable context', 'Chain / context / weights', 'Runtime', 'AsyncValue state / references', 'ExecutableModel retains chain, MGP context, weights registry and extra initialization/device arguments. Execution passes these fixed values alongside per-call inputs.', 'runtime_execute setup_wait'),
    obj('model', 'Initialized Model / ExecutableModel', 'Initialized Model', 'Engine', 'Python/native executable objects', 'Normal init_all returns Models keyed by sym_name. These hold runnable native models. Virtual mode yields CompileOnlyModel stand-ins when graph names are known; they explicitly refuse execution.', 'init_all compile_only runtime_execute'),
    obj('inputs', 'Per-call input buffers', 'Input Buffers', 'Driver', 'tensor storage / DLPack handles', 'Model.execute accepts Buffer, supported DLPack arrays or scalar inputs. Checks contiguity, wraps DLPack or constructs scalar Buffers, then prepares native inputs. These values differ from symbolic input types.', 'execute native_execute'),
    obj('validation', 'Native input validation', 'Shape / dtype / device', 'Engine', 'checks before executor entry', 'Native input handling checks input count, shape/dtype contract and device placement, extracts buffer storage and marshals AsyncValues. Invalid inputs can fail before graph execution.', 'input_validation native_execute'),
    obj('async', 'AsyncValue register file', 'AsyncValue registers', 'AsyncRT', 'readiness / atomics / reference counts', 'MEF registers hold values that can be pending, ready or errors. Ready values activate users immediately; pending results install readiness callbacks. This is dependency synchronization rather than a MAX Serve request queue.', 'mef_invoke ready_counts async_users'),
    obj('executor', 'MEF invocation / executor', 'MEF executor', 'Runtime', 'per-invocation native execution state', 'Invocation decodes register/primitive tables and allocates executor state. Runnable work follows operand readiness. Execution can dispatch asynchronous work; the executor is not a model-worker process.', 'mef_invoke ready_counts'),
    obj('primitives', 'Ready primitive worklist', 'Ready primitive worklist', 'Runtime', 'dependency-driven dispatch', 'The worklist decrements atomic not-ready counts. A primitive runs when its readiness gate clears. Nonstrict primitives can receive indirect values; readiness and errors govern downstream work.', 'ready_counts async_users'),
    obj('driver', 'Device runtime / driver', 'Device runtime / driver', 'Driver', 'native device context / dispatch boundary', 'Execution uses configured device contexts and device-appropriate primitives. Exact launch streams, driver calls and physical allocation depend on the target; this animation does not measure them.', 'serialize runtime_execute driver_execute'),
    obj('gpu', 'CPU / GPU computation', 'Target device', 'Hardware', 'hardware · BOUNDARY for measurements', 'Target code executes on real configured hardware. GPU kernel overlap, duration, occupancy and memory traffic require runtime evidence. Animated movement is a teaching aid.', 'gc_pipeline driver_execute'),
    obj('outputs', 'Returned output buffers', 'Output Buffers', 'Driver', 'device-associated results', 'extractDriverResults awaits runtime result values, reports errors, drops the internal chain and wraps tensor results into device buffers; Python returns Buffer objects. A return does not mean a universal device-wide barrier or host copy.', 'driver_results release_gil execute'),
]
for i, item in enumerate(OBJECTS):
    hue = (i * 0.61803398875) % 1
    rgb = colorsys.hls_to_rgb(hue, 0.36 + (i % 3) * 0.04, 0.57 + (i % 2) * 0.10)
    item['color'] = '#' + ''.join(f'{round(c * 255):02x}' for c in rgb)


def phase(key, title, lesson, detail, left, middle, right, labels, strip, ascii):
    return dict(id=key, title=title, lesson=lesson, detail=detail,
                left=left.split(), middle=middle.split(), right=right.split(), labels=labels, strip=strip, ascii=ascii)


PHASES = [
    phase('graph', 'Session & graph', 'Describe computation before running it.',
          'Create the session and trace the built-in Llama Graph API route. Input types are symbolic; forward records operations. Graph.output defines/verifies outputs. Session, graph, weight values and runtime inputs are different objects. Direct read(MEF) can bypass graph rebuilding.',
          'session devices spec types', 'graph kernel_library ops contract', 'artifact mef',
          ['PYTHON · SESSION / RECIPE', 'SYMBOLIC GRAPH · MLIR', 'LATER · ARTIFACTS'],
          ['Input contract', 'Trace forward', 'Verify outputs', 'No inference yet'],
          'GraphSpec(module, input_types) → Llama3.trace → Graph(forward)\n  input metadata → symbolic operations → Graph.output → verified IR\nInferenceSession(devices) → native runtime / compilation context\nAlternative: read(existing.mef) → CompiledModel, without a new Graph'),
    phase('compiler', 'Graph compiler', 'Transform graph IR into a runtime program.',
          'Native Graph compilation is scheduled on a CPU runtime worker under a process-wide lock. Lower a clone through MO, MAP, MOGG and MGP. Configured native cache hits bypass lowering. These frames group compiler roles inside the application, not separate OS processes.',
          'session lock handle', 'clone mo map mogg mgp', 'pool cache',
          ['PYTHON → NATIVE HANDOFF', 'MAX GRAPH COMPILER · IR STAGES', 'THREADS / OPTIONAL CACHE'],
          ['MO: shapes / layouts', 'MAP: eligible fusion', 'MOGG: routines', 'MGP: buffers / deps'],
          'compile(Graph) → compile_async → acquire _COMPILATION_LOCK\n  capsule → CompilationRequest → CPU runtime task → clone MLIR\n  native cache HIT → MEF; MISS → MO → MAP → MOGG → MGP\n  MGP → emitted Mojo → JIT → runtime calls → MEF\n  completion callback releases lock; compile() waits'),
    phase('mojo', 'Mojo kernels', 'Connect a graph op to compiled target code.',
          'The real mo.add registration shows parameterized Mojo implementation. MAX emits/compiles routines; the Mojo compiler performs semantic/lifetime checks, parametric optimization, elaboration and target lowering. This chapter combines a source example with documented compiler architecture, rather than claiming a recorded Add or Llama compilation.',
          'add kernel_library mojo_source', 'lit kgen concrete llvm', 'code gpu',
          ['REGISTRATION / GENERATED ROUTINES', 'MOJO COMPILER · ARCHITECTURE', 'COMPILED TARGET CODE'],
          ['mo.add registration', 'dtype / SIMD / tile', 'Specialize parameters', 'Code ≠ computed result'],
          '@extensibility.register("mo.add") → Mojo Add implementation\nMGP → Mojo routines → kernel JIT / runtime-call lowering\nMojo: parse / types → semantic / lifetime checks → LIT lowering\n  → KGEN optimization → elaboration → concrete functions\n  → post-elaboration optimization → LLVM / target code\nAn op can be fused; this is not one GPU launch per op.'),
    phase('mef', 'MEF preparation', 'Store executable code, then choose a reuse route.',
          'Cold compilation serializes MEF. Native model-IR caching and Python MefStore are distinct. Export can occur before runtime initialization. GraphSpec-based import still traces; direct read does not need a new Graph. A declined import key compiles normally, while a missing accepted artifact is an error. Pre-JIT stopping creates no MEF.',
          'session store handle', 'mgp code mef artifact', 'cache manifest',
          ['ARTIFACT POLICY / WAIT', 'LOWERED PROGRAM → MEF', 'CACHE / EXPORTED MANIFEST'],
          ['Native cache ≠ store', 'Export before init', 'Import vs direct read', 'Still uninitialized'],
          'Cold: lowered MGP → prepare primitives → serialize MEF → CompiledModel\nNative cache: matching module/options → cached MEF\nStore export: compile → claim_export → export_mef → manifest\nStore import: trace → claim_import → compile(path) → deserialize\nDirect: read(MEF) → deserialize (no new Graph)\nDeclined key → compile; missing accepted key → MissingArtifactError'),
    phase('init', 'Initialize weights', 'Make a compiled artifact executable.',
          'Bind actual named weight buffers after compilation. Graph metadata supports name/device-category validation; contiguity is checked on real initialization. Native setup binds contexts and waits for setup dependencies. Virtual mode creates explicit compile-only stand-ins if names are known, and rejects path artifacts with unknown names.',
          'session artifact registry', 'setup context model', 'devices driver gpu',
          ['ARTIFACT + ACTUAL WEIGHTS', 'NATIVE INITIALIZATION', 'DEVICE CONTEXTS / HARDWARE'],
          ['Validate weight contract', 'Bind / setup', 'Wait setup state', 'Model may now execute'],
          'CompiledModel + weights_registry(name → buffer)\n  → expected-name/device checks when captured → contiguity checks\n  → _load_all → Model.setup(CPU, registry, device map)\n  → await setup chain/context → Model keyed by sym_name\nVirtual: known names → CompileOnlyModel; execute refuses\nVirtual: path/read artifact with unknown names → ValueError'),
    phase('inference', 'Run inference', 'Let readiness drive one invocation.',
          'Per-call Buffers are validated and marshaled as AsyncValues. Native execution releases the GIL, passes persistent context/weights with inputs, and invokes MEF. Primitive readiness activates users or installs callbacks for pending values. Result extraction awaits runtime results and returns Buffers. Target-level synchronization and transfers are backend-specific.',
          'model inputs outputs', 'validation context executor async primitives', 'pool driver gpu',
          ['PYTHON · ONE MODEL CALL', 'MEF EXECUTOR · DEPENDENCIES', 'RUNTIME THREADS / DEVICE'],
          ['Inputs → AsyncValues', 'Ready primitive runs', 'Pending value → callback', 'Outputs → Buffers'],
          'Model.execute(inputs) → Buffer / DLPack / scalar conversion\n  → validate dtype/shape/device → prepare AsyncValues\n  → release GIL → ExecutableModel.execute\n  fixed chain/context/weights + inputs → MEF invocation\n  readiness counts / worklist → primitive dispatch\n  pending result → andThenSync → downstream users\n  await runtime results → drop chain → return output Buffers'),
]


def event(phase, title, detail, refs, active, create='', edge=None, when=None, **state):
    if isinstance(create, list):
        edge, create = create, ''
    result = dict(phase=phase, title=title, detail=detail, refs=refs.split(), active=active.split(), create=create.split(), **state)
    if edge:
        result['edge'] = edge
    if when:
        result['when'] = when
    return result


EVENTS = [
    event('graph', 'Create InferenceSession', 'The session selects runtime devices and compilation policy. Host CPU is added if absent.', 'session', 'session devices', 'session devices', ['session', 'devices', 'native session / device context']),
    event('graph', 'Use a GraphSpec recipe', 'Module and symbolic input types describe the Llama computation, before per-call buffers exist.', 'spec_trace llama_trace', 'spec types', 'spec types', ['spec', 'types', 'input contract'], when='trace'),
    event('graph', 'Graph records forward symbolically', 'The Graph constructor creates MLIR and runs the forward callback to record operations.', 'graph_init llama_trace', 'graph ops kernel_library', 'graph ops kernel_library', ['spec', 'graph', 'trace → symbolic MLIR'], when='trace', graphState='RECORDING'),
    event('graph', 'Graph.output verifies the contract', 'Set output signatures, chains, mo.output and names; verify MLIR. No numerical inference ran.', 'graph_output', 'graph contract', 'contract', ['graph', 'contract', 'verify / output contract'], when='trace', graphState='VERIFIED', graphReady=True),
    event('graph', 'Direct MEF read needs no new Graph', 'This route starts from an existing artifact. It bypasses rebuilding/tracing the Python Graph.', 'read', 'mef artifact', 'mef', when='read', graphState='BYPASSED'),
    event('compiler', 'Choose an existing artifact route', 'Explicit import chooses an artifact after trace; read starts from bytes. Graph compiler lowering is skipped.', 'compile_reuse read claim_import', 'session', when='artifact-route', compileStage='BYPASSED'),
    event('compiler', 'Import store declines this key', 'claim_import returns None. The Graph follows normal compilation rather than the import path.', 'compile_reuse claim_import', 'session', when='decline'),
    event('compiler', 'Enter compile_async from compile', 'Prepare Graph weight-device expectations, extension libraries and seeded kernel declarations. Native scheduling comes next.', 'compile_async compile_wait', 'session', when='lower-route'),
    event('compiler', 'Acquire the process-wide compilation lock', 'Graph/Module compilation stays protected until the native completion callback releases this lock.', 'compile_module', 'lock', 'lock', ['session', 'lock', 'acquire → release on completion'], when='lower-route', lockState='HELD'),
    event('compiler', 'Schedule native compile on the CPU pool', 'The capsule becomes a CompilationRequest. compileModel adds a CPU runtime task and returns its async result.', 'native_session engine_compile engine_sync', 'pool', 'pool', ['session', 'pool', 'CPU runtime task'], when='lower-route', compileStage='SCHEDULED'),
    event('compiler', 'Return an async-backed CompiledModel handle', 'compile_async wraps the native result. It remains pending in this illustration while native work continues; compile waits for completion.', 'compile_async compile_wait', 'handle', 'handle', ['pool', 'handle', 'native async result → Python handle'], when='lower-route', artifactState='PENDING'),
    event('compiler', 'Clone the input module / resolve options', 'Lowering owns a clone. Seeded kernel declarations prevent repeated package imports.', 'frontend compile_async', 'clone kernel_library', 'clone', ['pool', 'clone', 'compile-owned MLIR clone'], when='lower-route', compileStage='CLONED'),
    event('compiler', 'Native cache hit returns MEF early', 'Assume native model-IR caching is enabled and a matching artifact exists. Compiler passes are bypassed.', 'frontend', 'cache', 'cache', ['clone', 'cache', 'matching native cache key'], when='cache', compileStage='CACHE HIT'),
    event('compiler', 'MO: shapes, layouts and target legality', 'Configured high-level passes prepare graph IR. This is a pass schedule, not measured Llama fusion.', 'mo_pipeline run_pipeline gc_pipeline', 'clone mo', 'mo', ['clone', 'mo', 'MO optimization'], when='lower', compileStage='MO'),
    event('compiler', 'MAP: eligible fusion and outlined bodies', 'Structured computations can fuse. Accumulating buffers and outlined bodies prepare kernel work.', 'map_pipeline gc_pipeline', 'mo map', 'map', ['mo', 'map', 'MO → MAP'], when='lower', compileStage='MAP'),
    event('compiler', 'MOGG: legalize calls and outline routines', 'Prepare kernel calls and operands, then lower into the runtime-model representation.', 'mogg_pipeline gc_pipeline', 'map mogg', 'mogg', ['map', 'mogg', 'MAP → MOGG'], when='lower', compileStage='MOGG'),
    event('compiler', 'MGP: plan buffers, dependencies and primitives', 'Reuse/allocation planning, configured device-graph work and primitive lowering precede Mojo emission.', 'mgp_pipeline gc_pipeline', 'mogg mgp', 'mgp', ['mogg', 'mgp', 'MOGG → MGP'], when='lower', compileStage='MGP'),
    event('mojo', 'Kernel compilation is bypassed on this route', 'Cached/imported/read artifacts already contain compiled content. No new model kernel JIT is shown.', 'frontend compiled_create read', 'code', when='skip-jit'),
    event('mojo', 'Find the real mo.add → Mojo Add registration', 'Add provides parameterized SIMD and tile overloads. This source example illustrates registration rather than a specific Llama launch.', 'mojo_add', 'add kernel_library', 'add', ['kernel_library', 'add', 'registered mo.add implementation'], when='lower'),
    event('mojo', 'MGP emits Mojo routines', 'The enclosing MAX pipeline schedules JIT after Mojo emission, then final runtime-call lowering.', 'mgp_pipeline gc_pipeline', 'mojo_source', 'mojo_source', ['add', 'mojo_source', 'implementation building blocks'], when='lower', mojoStage='EMITTED'),
    event('mojo', 'Mojo compiler: parse / types / lifetime checks', 'The documented compiler architecture checks source/semantics and lowers LIT. Detailed JIT plumbing can vary by entry point.', 'mojo_compiler', 'lit', 'lit', ['mojo_source', 'lit', 'compiler architecture'], when='jit', mojoStage='SEMANTIC'),
    event('mojo', 'Optimize parametric KGEN', 'Reduce and simplify parametric IR before specialization.', 'mojo_compiler', 'lit kgen', 'kgen', ['lit', 'kgen', 'LIT → KGEN'], when='jit', mojoStage='PARAMETRIC'),
    event('mojo', 'Elaborate compile-time parameters', 'Resolve generators into concrete functions/types. Add illustrates dtype, width and layout parameters.', 'mojo_compiler mojo_add', 'kgen concrete', 'concrete', ['kgen', 'concrete', 'specialize → concrete functions'], when='jit', mojoStage='CONCRETE'),
    event('mojo', 'Lower / optimize into target code', 'Concrete functions pass through post-elaboration and target lowering. CPU/GPU backends differ.', 'mojo_compiler gc_pipeline', 'concrete llvm code', 'llvm code', ['concrete', 'llvm', 'LLVM / target lowering'], when='jit', mojoStage='TARGET CODE'),
    event('mef', 'Pre-JIT inspection stops before artifact creation', 'Deliberate stop after Mojo emission. convertMOToMEF returns an error; no MEF is serialized/cached and no model can execute.', 'convert_mef gc_pipeline compile_wait', 'mgp', when='prejit', error='PRE-JIT STOP', artifactState='NO ARTIFACT', lockState='RELEASED', halt=True),
    event('mef', 'Serialize lowered MGP into MEF', 'Prepare device-specific primitives, translate to MEF and copy serialized resources into the artifact buffer.', 'serialize convert_mef', 'mgp code mef', 'mef', ['mgp', 'mef', 'prepare primitives → serialize'], when='jit', artifactState='SERIALIZED'),
    event('mef', 'Resolve cached MEF bytes', 'Native cache supplied the buffer before graph/Mojo lowering. It still represents an uninitialized artifact.', 'frontend compiled_create', 'cache mef', 'mef', ['cache', 'mef', 'native cache hit → MEF'], when='cache', artifactState='CACHED'),
    event('mef', 'Import a matched Graph artifact', 'Graph trace already occurred. claim_import returns a path; compile(path) deserializes the MEF rather than lowering new graph IR.', 'claim_import compile_reuse compiled_create', 'store mef artifact', 'store mef', ['store', 'artifact', 'claim_import → compile(MEF path)'], when='import', artifactState='IMPORTED'),
    event('mef', 'Missing accepted artifact raises', 'An accepted key with no recorded artifact raises MissingArtifactError. It does not silently fall back to compilation.', 'claim_import', 'store', 'store', when='missing', error='MISSING ARTIFACT', artifactState='NO ARTIFACT', halt=True),
    event('mef', 'Read MEF directly into CompiledModel', 'read(path or file-like bytes) creates an uninitialized artifact. No new symbolic Graph or native graph lowering is required.', 'read compiled_create', 'mef artifact', ['mef', 'artifact', 'deserialize existing MEF'], when='read', artifactState='READ'),
    event('mef', 'Compilation result resolves / compile waits', 'Graph compilation completion releases the lock. compile waits on the native result. The compiled artifact still cannot execute.', 'compile_wait compile_module artifact', 'handle artifact', 'artifact', ['handle', 'artifact', 'wait → uninitialized artifact'], when='lower-success', lockState='RELEASED', compiledReady=True, artifactState='COMPILED'),
    event('mef', 'Existing artifact is ready for initialization', 'A deserialized artifact holds compiled models without runtime weight binding.', 'read artifact compiled_create', 'artifact', 'artifact', when='existing-success', compiledReady=True),
    event('mef', 'Claim export identity and write MEF', 'Export before initialization. The store records signature/body fingerprint; export_mef writes the artifact and write_manifest persists metadata.', 'claim_export export write_manifest', 'store artifact manifest mef', 'store manifest', ['artifact', 'manifest', 'MEF export + manifest'], when='export', artifactState='EXPORTED', exported=True),
    event('init', 'Pass actual named weight buffers to init_all', 'Graph-based compilation retains expected weight metadata. Direct read / path compilation lack that Python metadata; native artifact contracts still apply.', 'init_all compile_graphs', 'session artifact registry', 'registry', ['registry', 'artifact', 'weights_registry: names → buffers'], when='success'),
    event('init', 'Reject a noncontiguous registered weight', 'Normal-device initialization checks each registered weight and raises ValueError for this deliberately invalid layout.', 'init_all', 'registry', when='bad-layout', error='NONCONTIGUOUS WEIGHT', modelState='NOT INITIALIZED', halt=True),
    event('init', 'Virtual mode returns compile-only stand-ins', 'Graph names are known for Graph/Module compilation. init_all returns CompileOnlyModel values rather than executable models.', 'init_all compile_only', 'artifact model', 'model', ['artifact', 'model', 'CompileOnlyModel stand-in'], when='virtual-known', modelState='COMPILE-ONLY'),
    event('init', 'Virtual initialization rejects unknown graph names', 'Path/read artifacts carry no Python graph-name list here. Virtual mode raises instead of pretending initialization succeeded.', 'init_all read compile_async', 'artifact', when='virtual-unknown', error='UNKNOWN GRAPH NAMES', halt=True),
    event('init', 'Validate / bind weights on real devices', 'Check captured names and CPU/device category when available; always check registered buffer contiguity on this normal path.', 'init_all', 'registry setup', 'setup', ['registry', 'setup', 'validate → _load_all'], when='real', modelState='SETTING UP'),
    event('init', 'Native setup binds runtime/device context', 'A CPU worker continuation calls model.setup with registry and device map, then awaits setup dependencies.', 'native_init setup setup_wait', 'setup context driver gpu', 'context driver gpu', ['setup', 'context', 'chain / MGP context / weights'], when='real', modelState='AWAITING SETUP'),
    event('init', 'Return initialized Model keyed by sym_name', 'Setup finished. The Model retains its executable context and can now accept per-call inputs.', 'init_all setup runtime_execute', 'context model', 'model', ['context', 'model', 'executable model reference'], when='real', modelState='INITIALIZED', initialized=True),
    event('inference', 'CompileOnlyModel refuses execute', 'Virtual compilation can inspect/export code, but the stand-in cannot execute or allocate device-dependent state.', 'compile_only', 'model', when='virtual-known', error='EXECUTION REFUSED', halt=True),
    event('inference', 'Convert per-call inputs to Buffers', 'Accept Buffer, supported DLPack arrays or scalars after contiguity checks. Inputs are values, unlike symbolic input types.', 'execute', 'inputs model', 'inputs', ['inputs', 'model', 'Model.execute(*inputs)'], when='real', inferenceStage='INPUTS'),
    event('inference', 'Validate shape / dtype / device, marshal AsyncValues', 'Native handling validates input spec and placement, extracts storage and prepares runtime values.', 'input_validation native_execute', 'inputs validation async', 'validation async', ['inputs', 'validation', 'validate → prepareInput'], when='real', inferenceStage='VALIDATED'),
    event('inference', 'Release GIL / invoke the executable', 'Native execution releases the Python GIL. Fixed chain/context/weights and extra setup arguments accompany each call’s inputs.', 'release_gil driver_execute runtime_execute', 'model context executor', 'executor', ['context', 'executor', 'fixed context + inputs → MEF invoke'], when='real', inferenceStage='INVOKED'),
    event('inference', 'Decode registers / primitive tables', 'MEF invocation allocates per-call executor state and starts primitives whose dependency gates permit execution.', 'mef_invoke', 'executor async primitives', 'primitives', ['executor', 'async', 'registers hold AsyncValues'], when='real', inferenceStage='DEPENDENCIES'),
    event('inference', 'Ready operands activate primitive work', 'Atomic not-ready counts drop. A ready primitive gets its implementation and arguments; nonstrict primitives can take indirect values.', 'ready_counts', 'async primitives driver', ['async', 'primitives', 'readiness gate clears → dispatch'], when='real', inferenceStage='DISPATCH'),
    event('inference', 'Device-appropriate work computes results', 'Primitives use target code and configured device context. Arrow movement illustrates dispatch, not a measured GPU stream or timeline.', 'serialize runtime_execute driver_execute', 'primitives driver gpu', ['primitives', 'gpu', 'target computation · illustrative timing'], when='real', inferenceStage='DEVICE WORK'),
    event('inference', 'Pending result installs a readiness callback', 'For a pending AsyncValue, andThenSync retains the executor and processes users when the value becomes ready.', 'async_users', 'async primitives', ['async', 'primitives', 'pending → ready → downstream users'], when='real', inferenceStage='RESULT READINESS'),
    event('inference', 'Await runtime results / return output Buffers', 'Driver result extraction awaits values, propagates errors, drops the internal chain, preserves device placement and wraps returned tensors.', 'driver_results release_gil execute', 'async outputs', 'outputs', ['async', 'outputs', 'await results → Python Buffers'], when='real', inferenceStage='RETURNED', outputsReady=True),
]


def source_span(text, spec, python_span, cpp_span):
    if spec['language'] == 'python':
        return python_span(text, spec['symbol'])
    if spec['language'] == 'cpp':
        return cpp_span(text, spec['needle'], spec.get('prefix', 0))
    anchor = spec['anchor']
    if text.count(anchor) != 1:
        raise ValueError(f'Expected unique source anchor: {anchor}')
    start_offset = text.index(anchor)
    end_offset = text.index(spec['until'], start_offset)
    return text.count('\n', 0, start_offset) + 1, text.count('\n', 0, end_offset)


def build_engine_lifecycle(data, task: Path, src: Path, python_span, cpp_span):
    used = {ref for item in OBJECTS + EVENTS for ref in item['refs']}
    refs, files = {}, {}
    for key in sorted(used):
        spec = REF_SPECS[key]
        file = spec['file']
        record = data['files'][file]
        start, end = source_span(record['text'], spec, python_span, cpp_span)
        refs[key] = dict(**spec, alias={'python':'PY', 'cpp':'C++', 'mojo':'MOJO', 'text':'DOC'}[spec['language']],
                         start=start, end=end, line=start, label=spec['purpose'])
        files[file] = record
    payload = dict(revision=data['revision'], objects=OBJECTS, phases=PHASES, events=EVENTS, references=refs, files=files)
    template = (src / 'engine_lifecycle.html').read_text()
    inventory, fallback = [], []
    for item in OBJECTS:
        links = ' · '.join(f'<a data-ref="{key}" href="../sources/{refs[key]["file"]}#L{refs[key]["line"]}">{html.escape(refs[key]["label"])} [{refs[key]["alias"]}:L{refs[key]["line"]}]</a>' for key in item['refs'])
        inventory.append(f'<details class="inventory-item" data-object="{item["id"]}" style="--object-color:{item["color"]}"><summary><span class="swatch"></span><strong>{html.escape(item["name"])}</strong><span>{html.escape(item["owner"])} · {html.escape(item["kind"])}</span></summary><p>{html.escape(item["role"])}</p><p>{links}</p></details>')
    companion = ['# MAX engine: graph compilation, Mojo, MEF and inference', '', f'[Interactive animation](link_htmls/{PAGE_NAME}).', '', f'SOURCE revision: `{data["revision"]}`.', '', 'Built-in Llama Graph API route plus engine artifact alternatives. Logical source order; no MAX model compilation or inference was run to produce this page.', '']
    for i, p in enumerate(PHASES, 1):
        fallback.append(f'<details class="text-phase"><summary>{i:02d} · {html.escape(p["title"])}</summary><p>{html.escape(p["detail"])}</p><pre>{html.escape(p["ascii"])}</pre></details>')
        companion += [f'## {i:02d} · {p["title"]}', '', p['detail'], '', '```text', p['ascii'], '```', '']
    companion += ['## Object / role atlas', '', '| Component | Owner / type | Role |', '| --- | --- | --- |']
    companion += [f'| {o["name"]} | {o["owner"]} / {o["kind"]} | {o["role"]} |' for o in OBJECTS]
    companion += ['', '## Source event sequence', '']
    for item in EVENTS:
        links = ' · '.join(f'[{refs[k]["label"]}](sources/{refs[k]["file"]}#L{refs[k]["line"]})' for k in item['refs'])
        companion += [f'- **{item["title"]}**{(" (" + item["when"] + " route)") if item.get("when") else ""}: {item["detail"]} {links}']
    template = template.replace('{{inventory}}', '\n'.join(inventory)).replace('{{text-phases}}', '\n'.join(fallback))
    template = template.replace('{{revision}}', data['revision']).replace('{{revision-short}}', data['revision'][:12])
    if '{{' in template:
        raise ValueError('Unfilled engine template')
    template = template.replace('<!-- ENGINE_CSS -->', '<style>' + (src / 'engine_lifecycle.css').read_text() + '</style>')
    template = template.replace('<!-- ENGINE_DATA -->', '<script id="engine-data" type="application/json">' + json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c') + '</script>')
    template = template.replace('<!-- ENGINE_JS -->', ''.join('<script>' + (src / name).read_text() + '</script>' for name in ('engine_lifecycle_model.js', 'engine_lifecycle.js')))
    (task / 'link_htmls' / PAGE_NAME).write_text(template)
    (task / GUIDE_NAME).write_text('\n'.join(companion) + '\n')
    manifest = dict(revision=data['revision'], page='link_htmls/' + PAGE_NAME, objects=OBJECTS, phases=PHASES, events=EVENTS, references=refs, files=[{k:v for k,v in f.items() if k != 'text'} for f in files.values()])
    (task / 'engine_lifecycle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return len(refs), len(OBJECTS)
