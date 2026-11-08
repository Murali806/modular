<!-- section:2 -->
## 2. Python Graph + custom package

### 2.1 Start the Python application {#toolchain-q2-1}

**2.1.1 Build and run are distinct actions.** The Python launcher resolves <code>addition.py</code>, its Python dependencies, the raw kernel source files and the Mojo runtime environment. The application begins when the launcher runs.

~~~bash
# From the modular/ repository root:
./bazelw build //max/examples/custom_ops:addition
./bazelw run //max/examples/custom_ops:addition

# The generated sibling test also runs the example:
./bazelw test //max/examples/custom_ops:addition.example-test
~~~

**2.1.2 The binary's environment supplies the native tools.** <code>modular_py_binary</code> creates a <code>mojo_test_environment</code>, collects transitive <code>MojoInfo</code> packages and expands compiler, import-path, shared-library and linker variables. The configured source-built Mojo tool is <code>//Mojo/tools/mojo</code>; the LLD dependency is <code>@llvm-project//lld:lld</code>.

| Runtime variable | What the Python/native compiler can locate |
| --- | --- |
| <code>MODULAR_MOJO_MAX_DRIVER_PATH</code> | Mojo compiler driver |
| <code>MODULAR_MOJO_MAX_IMPORT_PATH</code> | Directories containing the precompiled Mojo dependencies |
| <code>MODULAR_MOJO_MAX_LLD_PATH</code> | Configured LLD executable |
| <code>MODULAR_MOJO_MAX_LINKER_DRIVER</code> | Host C++ linker driver |
| <code>MODULAR_MOJO_MAX_SHARED_LIBS</code> / <code>...SYSTEM_LIBS</code> | Native libraries and system-link settings |

These are launch settings derived from the build toolchain. The kernel's LLVM target is selected later from target information. [SOURCE:bazel_py_binary] · [SOURCE:mojo_runtime_toolchain] · [SOURCE:mojo_runtime_exports] · [SOURCE:bazel_mojo_toolchain]

### 2.2 Prepare the source package at Graph construction {#toolchain-q2-2}

**2.2.1 Call path.** <code>Graph(custom_extensions=[mojo_kernels])</code> uses <code>KernelLibrary.load_paths</code>. A source-package directory is passed to <code>mojo.paths._build_mojo_source_package</code>; an existing <code>.mojoc</code> path can be added directly.

**2.2.2 Actual subprocess arguments.** The helper constructs the following argument list; the configured Mojo driver is supplied by <code>subprocess_run_mojo</code>:

~~~text
mojo precompile <absolute kernels/ source path> -o <private staging package path>
~~~

The compiler receives the **Mojo source package directory**. <code>addition.py</code> remains the Python entry point. The package includes its modules such as <code>add_one.mojo</code>; this does not mean every packaged function is emitted as machine code.

**2.2.3 Actual output naming and publication.** The helper hashes the absolute source path and constructs:

~~~text
<tempdir>/.modular_<uid>/mojo_pkg/
    .staging.<pid>/mojo_pkg_<path_hash>.mojoc   # private compiler output
    mojo_pkg_<path_hash>.mojoc                 # published path
~~~

It runs precompilation into the staging directory, then <code>os.replace</code> atomically publishes the complete package. The private file uses the final basename because the compiler embeds the module name. This prevents another process from observing a half-written package.

**2.2.4 This hash identifies a source path.** It is not a hash of the kernel's contents, selected GPU architecture or final machine code. The helper shown here invokes precompilation; the presence of that pathname alone is not proof of a skipped compile. [SOURCE:package_load] · [SOURCE:package_builder] · [SOURCE:kernel_package_root]

### 2.3 What is inside .mojoc? {#toolchain-q2-3}

**2.3.1 Representation.** A precompiled Mojo package serializes a <code>lit.package</code> operation as MLIR bytecode. It retains declarations, implementation bodies and package/dependency information in post-parse IR. The compiler walkthrough describes semantic/lifetime checking during package preparation and later loading of those bodies.

**2.3.2 Why bodies remain.** <code>AddOne.execute</code> still has a compile-time <code>target</code>; its nested function still has a compile-time <code>width</code>. Concrete instantiation needs those bodies later. This package is useful compiler input rather than a final CPU/GPU image.

**2.3.3 Two .mojoc routes exist.** The standalone <code>:kernels</code> Bazel library and the runtime source-package helper can precompile the same source group into different output locations. The shared <code>layout</code>, <code>extensibility</code> and <code>std</code> packages are separate dependencies. The dependency edges in point 1 determine which packages Bazel builds.

~~~text
addition.py ── Python program; Graph builder and execute calls
kernels/    ── Mojo source package
  __init__.mojo
  add_one.mojo
  other declared .mojo modules
         │ mojo precompile
         ▼
P1 · .mojoc ── post-parse MLIR bytecode, declarations and bodies
         │ import + concrete specialization later
         ▼
target-specific routines / device images
~~~

[SOURCE:mojo_package] · [SOURCE:reference_add_one] · [SOURCE:bazel_build_file]

### 2.4 Register implementations and record G1 {#toolchain-q2-4}

**2.4.1 Name linkage.** Python records <code>ops.custom(name="add_one", ...)</code>. Mojo declares <code>@extensibility.register("add_one")</code> on <code>AddOne</code>. The matching string connects the graph operation to its registered implementation.

**2.4.2 Input/output contract.** The example records float32 tensors of shape <code>[5,10]</code>, the chosen <code>DeviceRef</code> and an explicit output <code>TensorType</code>. Its custom shape function raises <code>NotImplemented</code>; the explicit output contract makes that optional function unnecessary in this example.

**2.4.3 Recording is symbolic.** G1 stores operations, types, shapes, device placement and def-use relationships. The example creates its random input values later. <code>Graph.output</code> completes/verifies the graph output contract.

**2.4.4 Compiler declarations can be seeded once.** The Graph branch of <code>InferenceSession.compile_async</code> seeds the MLIR module from the Graph's kernel analysis before native compilation. The compiler recognizes the mapping attribute and can avoid repeating package import. [SOURCE:reference_addition] · [SOURCE:custom_api] · [SOURCE:graph_init] · [SOURCE:graph_output] · [SOURCE:compile_async]

<!-- section:3 -->
## 3. MAX C++ / MLIR GraphCompiler: MO → MAP → MOGG → MGP

### 3.1 Which objects own and run the transformations? {#toolchain-q3-1}

**3.1.1 Entry.** The Python <code>InferenceSession</code> passes the Graph's module to the native engine. <code>MOFContext::compileToBinary</code> clones the MLIR module and gathers <code>GCPipelineOptions</code>. G1 remains the caller's graph; C1 is the compiler-owned clone.

**3.1.2 Execution mechanism.** Native <code>mlir::PassManager</code>/<code>mlir::OpPassManager</code> objects run passes assembled by <code>M::MO::buildGCPipeline</code> and <code>gcPipelineImpl</code>. The dialect classes provide operations/types; the conversion pass objects rewrite the module.

**3.1.3 Relation to MAX Serve.** These are MAX/Mojo compiler objects reached through the engine API. A Serve pipeline can call the same engine while initializing a model. Its API server, worker scheduler and batch constructor do not implement the MO/MAP/MOGG/MGP conversions. The four Python examples call the engine directly.

| Layer / object | Responsibility |
| --- | --- |
| Python <code>Graph</code>, <code>Module</code>, <code>KernelLibrary</code> | Symbolic graph and imported kernel declarations |
| Python <code>InferenceSession</code> + native engine/context | Compilation request and selected-device runtime context |
| Native <code>MOFContext</code> | Clone, compiler configuration, artifact lookup and conversion |
| <code>GCPipelineOptions</code>, MLIR pass managers | Configure and execute the ordered compiler transformations |
| <code>MOToMAPPass</code>, <code>MAPToMOGGPass</code>, <code>MOGGToMGPPass</code> | Perform the three named dialect conversions |
| <code>KGENCompiler</code>, <code>ObjectCompiler</code> | Specialize kernel implementations and emit code |

[SOURCE:compile_async] · [SOURCE:frontend] · [SOURCE:gc_pipeline]

### 3.2 MO — Modular Operations {#toolchain-q3-2}

**3.2.1 Format and class.** <code>M::MO::MODialect</code> is the framework-level graph dialect. Operations carry typed tensor values and attributes. The source dialect definition calls it the frontend input to the graph compiler.

**3.2.2 Work at this level.** <code>buildMoPipeline</code> schedules shape/layout inference, symbolic optimization, constant folding when enabled, cleanup, package/kernel verification and device validation. These passes increase the information needed for implementation selection.

**3.2.3 Add-one teaching view.** C1 still describes “apply the registered add_one operation to float32 X[5,10] on the selected device; return Y with the declared type.” It contains tensor contracts and symbolic relationships rather than random X values.

**3.2.4 Artifact distinction.** MO is an in-memory representation. A <code>.mlir</code> file can be a diagnostic dump of it. It is not a mandatory file produced by <code>bazelw build :addition</code>. [SOURCE:mo_dialect] · [SOURCE:mo_pipeline] · [SOURCE:graph_compiler_design]

### 3.3 MO → MAP — Modular Access Primitives {#toolchain-q3-3}

#### 3.3.1 The conversion pass {#toolchain-q3-3-1}

<code>gcPipelineImpl</code> schedules <code>MO::createMarkStableTensorsPass()</code>, then <code>MO::createMOToMAP()</code>. <code>MOToMAPPass::runOnOperation</code> creates a <code>MOGG::MojoRegistry</code>, collects lowerable operations and calls <code>lowerOp</code>. Unsupported ranks or failed lowering signal pass failure.

The result exposes implementation-oriented accesses and kernel structure. It lets later passes reason about how computation loads values and stores outputs. [SOURCE:mo_to_map] · [SOURCE:gc_pipeline]

#### 3.3.2 MAP's operation families and fusion {#toolchain-q3-3-2}

| MAP family | Information made visible |
| --- | --- |
| <code>map.iter.opaque</code> | A hand-written Mojo kernel and a store epilogue/fusion interface |
| <code>map.iter.elementwise</code> | Per-index loads, elementwise calls and stores |
| <code>map.load</code>, <code>map.store</code>, <code>map.call.elementwise</code> | The data accesses and arithmetic used by an explicit body |

<code>buildMapPipeline</code> schedules <code>MAP::createFuseKernels</code>, optionally <code>createFuseMutableLoads</code>, accumulation-buffer allocation and structured-body outlining. Fusion eligibility depends on implementation interfaces and data relationships.

In the epilogue example, <code>AddOneElementwise(ElementwiseUnaryOp)</code> explicitly supplies a fusible <code>x + 1</code> body beside matmul and multiplication. The simple <code>AddOne.execute</code> supplies its hand-written <code>foreach</code> routine. The animations show these structures schematically; they are not captured MAP dumps or proof of a particular fusion result. [SOURCE:map_dialect] · [SOURCE:map_pipeline] · [SOURCE:reference_add_one]

### 3.4 MAP → MOGG — Modular Operation Generator Graph {#toolchain-q3-4}

**3.4.1 Conversion owner.** <code>MAP::createMAPToMOGG(options)</code> constructs <code>MAPToMOGGPass</code>. Its pass creates <code>MojoRegistry</code> and <code>MAPToMOGGState</code> objects to lower outlined opaque kernels and shape-function invocations.

**3.4.2 New representation.** MOGG expresses generator/kernel choices, calls, parameter information and implementation interfaces. The graph now has code-generation-oriented routine boundaries while keeping its input/output dependencies.

**3.4.3 Finishing passes.** <code>buildMoggPipeline</code> legalizes rowwise kernels, conforms call operands, outlines routines and adds trace wrapping. These are compiler-generated call interfaces around the authored implementation.

**3.4.4 Current pipeline evidence.** The pipeline performs advanced kernel fusion in MAP before this conversion. Some MOGG dialect prose contains older fusion motivation; the ordered pipeline and the current GraphCompiler design document establish the current placement of that work. [SOURCE:map_to_mogg] · [SOURCE:mogg_pipeline] · [SOURCE:mogg_dialect] · [SOURCE:graph_compiler_design]

### 3.5 MOGG → MGP — Modular Glue Primitives {#toolchain-q3-5}

**3.5.1 Conversion owner.** <code>MGP::createMOGGToMGP(options)</code> constructs <code>MOGGToMGPPass</code>. <code>runOnOperation</code> uses <code>SymbolTableAnalysis</code>, orders graphs by their call dependencies and invokes <code>MOGraphLowering(...).run()</code>. A dependency cycle stops the pass.

The representation becomes runtime glue: initialized context/state, buffer references, calls, allocation and execution dependencies.

#### 3.5.2 MGP objects and memory/dependency planning {#toolchain-q3-5-2}

| MGP representation | Runtime meaning recorded by the compiler |
| --- | --- |
| <code>mgp.model</code> / <code>MGP::ModelOp</code> | Model initialization and execution invokables |
| <code>!mgp.context</code> | <code>MGP::Context</code> with reusable <code>StateSlot</code> values |
| <code>!mgp.buffer</code> | Byte storage represented by an MLRT <code>BufferRef</code> |
| <code>!mgp.tensor</code>, tensor spec | Shape/dtype metadata associated with buffer storage |
| Chains and call operands | Dependencies and values used to coordinate runtime work |

<code>buildMgpPipeline</code> runs buffer reuse, dynamic memory planning, deferred allocations, model-argument validation, chain simplification, slot globalization and primitive lowering. A slot in this plan is a compiler/runtime resource; it is not a CUDA thread, an IPC queue seat or a physical GPU register.

For add-one, the teaching view records X's input reference, an output resource and the routine/dependency needed to produce Y. Numerical X/Y contents arrive during execution. [SOURCE:mogg_to_mgp] · [SOURCE:mgp_dialect] · [SOURCE:mgp_pipeline]

### 3.6 Device-graph finalization {#toolchain-q3-6}

**3.6.1 Ordered preparation.** The MGP pipeline stages device graphs, partitions collectives, computes/simplifies dependencies, generates build stubs and schedules <code>FinalizeDeviceGraphsPass</code>.

**3.6.2 Actual guard.** The finalization pass reads <code>ModelOp</code> and returns immediately when <code>getIsDeviceGraph()</code> is false. For a device-graph model, it finalizes the <code>DeviceGraphCollectiveOp</code> operations in its execute block. Scheduling this pass does not establish that every ordinary example has a nonempty device graph.

**3.6.3 Milestones.** Python <code>Graph.output</code> verifies the symbolic output contract; this later pass handles compiler device-graph structure; model initialization loads prepared runtime resources; execution submits work. The animation keeps these milestones separate. [SOURCE:mgp_pipeline] · [SOURCE:device_finalization]

### 3.7 Emit routines, then enter kernel JIT {#toolchain-q3-7}

**3.7.1 Generated source.** <code>MOGG::createEmitMojo</code> imports the required standard/custom packages and emits Mojo routine scaffolding for the lowered graph. The emitted source is a compiler resource consumed by the JIT path. The precompiled custom package supplies the authored bodies that the scaffolding references.

**3.7.2 Separate compiler modules.** C1 retains the MGP runtime plan. The JIT entry creates a separate parsed Mojo/KGEN module, KIR1. Rewriting KIR1 toward LLVM does not turn the MGP buffer/dependency plan into GPU assembly.

**3.7.3 Pipeline tail.** On the normal JIT route, <code>gcPipelineImpl</code> adds <code>createJitCompileKernels</code>, strips kernel declarations and calls <code>buildJitPipeline</code>. Final runtime-call lowering and MEF preparation follow compiled-kernel work.

{{diagram:ir}}

~~~mermaid
flowchart LR
    G1["Python Graph G1 · retained"] -->|clone| C1["C1 · MO"]
    C1 -->|MOToMAPPass| MAP["C1 · MAP"]
    MAP -->|MAPToMOGGPass| MOGG["C1 · MOGG"]
    MOGG -->|MOGGToMGPPass| MGP["C1 · MGP plan"]
    MGP --> FINAL["Conditional device-graph finalization"]
    FINAL --> KIR["EmitMojo + JIT · separate KIR1"]
    KIR --> CODE["compiled host / device routines"]
    CODE --> CALLS["runtime call lowering + MEF bytes"]
    CALLS --> M1["session.init · executable M1"]
~~~

~~~text
G1 retained
  └─ clone C1: MO → MAP → MOGG → MGP
                                 ├─ buffer / dependency / init / execute plan
                                 └─ EmitMojo → separate KIR1 → target code
                         runtime calls + code resources → CM1 → init → M1
~~~

[SOURCE:emit_mojo] · [SOURCE:jit_entry] · [SOURCE:gc_pipeline] · [SOURCE:runtime_calls] · [SOURCE:jit_pipeline]

<!-- section:4 -->
## 4. Mojo specialization, LLVM lowering and MAX's JIT path

### 4.1 What JIT means in this example {#toolchain-q4-1}

**4.1.1 Definition.** JIT means **just-in-time compilation**: the running application prepares native code from compiler representations for the required implementation/target. The timing can be explicit preparation before an inference call.

**4.1.2 Concrete MAX path.** In <code>addition.py</code>, the order is <code>Graph(...)</code>, <code>InferenceSession(...)</code>, <code>session.compile(graph)</code>, <code>session.init(compiled)</code>, creation of numerical X, then <code>model.execute(x)</code>. Kernel JIT is reached during graph compilation. This example does not depend on first-execute lazy compilation.

**4.1.3 Actual entry and objects.** <code>JitCompileKernelsPass::compileMojoToBinary</code> retrieves target information, parses generated Mojo, runs LIT-to-KGEN work, creates a <code>KGENCompiler</code> and <code>ObjectCompiler</code>, and stores them with the MGP/KGEN modules in <code>CompilationState</code>.

**4.1.4 Miss/hit behavior.** <code>handleCacheMisses</code> calls <code>compiler-&gt;runElaborationPipeline</code>, then <code>objectCompiler-&gt;emitArchive</code>. It records a compiled-load resource. <code>handleCacheHits</code> loads compatible compiled archives. The surrounding graph/runtime plan still needs preparation. A graph-artifact hit is a separate, broader cache route.

~~~text
Running Python process
  session.compile(G1)
    native GraphCompiler passes on C1
      EmitMojo
        compileMojoToBinary
          parse → LIT-to-KGEN → compiled-kernel lookup
          miss: KGENCompiler.runElaborationPipeline
                ObjectCompiler.emitArchive → compiled resources
          hit:  load compatible compiled resources
      runtime-call lowering → prepared CM1
  session.init(CM1) → M1
  model.execute(Buffer X) → result Buffer Y
~~~

[SOURCE:reference_addition] · [SOURCE:jit_entry] · [SOURCE:kernel_misses] · [SOURCE:kernel_hits] · [SOURCE:artifact_cache]

### 4.2 Which code performs specialization? {#toolchain-q4-2}

**4.2.1 Compiler pipeline.** <code>Mojo/lib/Compiler/KGENCompiler.cpp</code> implements <code>KGENCompiler::runElaborationPipeline</code>. It configures an MLIR pass manager, supplies target information and invokes <code>populateElaborateModulePasses</code>. That builder calls <code>buildElaborateModulePipeline</code> and <code>buildPostElaborationPipeline</code>.

The elaboration machinery instantiates generator functions/types with concrete compile-time values. The compiler walkthrough identifies <code>Mojo/lib/Elaborator/Elaborator.cpp</code> and its IR evaluators; the KGEN compiler supplies the <code>compileOffloads</code> callback used when specialization encounters device offloads.

#### 4.2.2 Add-one specialization example {#toolchain-q4-2-2}

| Authored parameter / operation | Concrete information used during preparation |
| --- | --- |
| <code>AddOne.execute[target]</code> | <code>"cpu"</code> or <code>"gpu"</code> implementation route |
| <code>InputTensor[dtype=output.dtype, rank=output.rank, ...]</code> | Output/input dtype and rank; float32, rank 2 in this example |
| <code>elementwise_add_one[width]</code> | A width selected by the implementation and target |
| <code>SIMD[x.dtype, width]</code> | Concrete scalar/vector type |
| <code>x.load[width](idx) + 1</code> | Typed load and native arithmetic |
| <code>foreach[target=target]</code> | The chosen target's iteration/dispatch machinery |

The animation's small SIMD width is illustrative. The full graph has 50 elements; a source walkthrough alone does not establish a captured vector width, launch count or instruction sequence.

**4.2.3 IR result.** Parametric <code>kgen.generator</code>/<code>kgen.struct.generator</code> forms become concrete <code>kgen.func</code>/<code>kgen.struct.instance</code> forms. Compile-time parameter expressions are evaluated; runtime tensor values remain runtime inputs. [SOURCE:kgen_elaboration] · [SOURCE:kgen_elaborate_build] · [SOURCE:mojo_elaboration] · [SOURCE:reference_add_one]

### 4.3 Post-elaboration optimization {#toolchain-q4-3}

**4.3.1 Actual code.** <code>Mojo/lib/Compiler/Pipeline/Pipeline.cpp</code> implements <code>buildPostElaborationPipeline</code>. It applies floating-point policy, verifies concrete IR, updates materialized compile-time storage, then invokes the first and late optimization pipelines.

**4.3.2 Purpose.** This stage simplifies concrete code, lowers argument/calling conventions and enables optimizations such as scalar replacement, promotion and inlining. It operates on the specialized implementation; it does not invent the tiled matmul algorithm authored in the example or execute the tensor's arithmetic in Python. [SOURCE:kgen_post_elaboration] · [SOURCE:mojo_elaboration]

### 4.4 Which code lowers to LLVM dialect? {#toolchain-q4-4}

**4.4.1 Caller.** <code>ObjectCompiler::lowerAllFuncsToLLVM</code> configures a pass manager and calls <code>buildLowerToLLVMPipeline</code>.

**4.4.2 Exact builder.** <code>Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp</code> schedules:

| Pass / action | Representation change |
| --- | --- |
| <code>createLowerKGENToLLVM</code> | KGEN functions/types toward LLVM-compatible forms |
| Runtime-closure lowering and POP legalization | Make closures and primitive operations lowerable |
| <code>createLowerGlobalPOPToLLVM</code>, <code>createLowerPOPToLLVM</code> | Primitive operations and globals into LLVM dialect |
| <code>createLowerControlFlow</code> | Lower remaining control-flow forms |
| Cast reconciliation, suspension-point lowering, cleanup | Finish legal LLVM-compatible MLIR |
| <code>createTargetSpecificLLVMLowering</code> | Late target-specific MLIR lowering |
| Debug-info lowering | Translate debug representation for LLVM export |

**4.4.3 Format.** LLVM dialect is still **MLIR**: MLIR operations, types, regions and an MLIR context. For example the conceptual SIMD float32 type can become an LLVM-compatible vector representation. A <code>.pre-llvm-ir.mlir</code> diagnostic dump is still MLIR. [SOURCE:object_lower_to_llvm] · [SOURCE:kgen_llvm_passes] · [SOURCE:mojo_llvm]

### 4.5 Which code translates to LLVM IR? {#toolchain-q4-5}

**4.5.1 Exact API.** The file-local <code>translateModuleToLLVMIR</code> helper in <code>ObjectCompiler.cpp</code> invokes <code>mlir::translateModuleToLLVMIR(module, ctx, moduleName)</code>. It produces a separate <code>std::unique_ptr&lt;llvm::Module&gt;</code> using an LLVM context.

**4.5.2 Boundary between IR frameworks.** KIR1 before translation is an MLIR module containing LLVM-dialect operations. LLVM1 after translation is an <code>llvm::Module</code>. Textual LLVM IR can be exported as <code>.ll</code>; LLVM bitcode can be exported as <code>.bc</code>. These exports are optional views of compiler data.

**4.5.3 Code generation.** LLVM optimization and <code>runLlcPasses</code> use the selected <code>llvm::TargetMachine</code>. The helper creates/runs LLVM code-generation passes in the process. The C++ callback name <code>runLlc</code> does not establish that a standalone <code>llc</code> executable is launched for each kernel.

{{diagram:kernel}}

~~~mermaid
flowchart LR
    PKG["P1 · .mojoc / post-parse MLIR"] --> KGEN["KIR1 · parametric KGEN"]
    KGEN -->|KGENCompiler + elaborator| CONCRETE["concrete KGEN / POP"]
    CONCRETE -->|buildLowerToLLVMPipeline| DIALECT["LLVM dialect · MLIR"]
    DIALECT -->|mlir::translateModuleToLLVMIR| LLVM["LLVM1 · llvm::Module"]
    LLVM -->|LLVM target codegen| BYTES["object / PTX bytes"]
~~~

~~~text
post-parse package + generated routines
  → parametric KGEN → elaboration → concrete KGEN / POP
  → LLVM dialect (MLIR module)
  → translation → LLVM IR (llvm::Module)
  → target machine / backend → compiled bytes
~~~

[SOURCE:llvm_translation] · [SOURCE:object_lower_to_llvm] · [SOURCE:llvm_code_gen]

<!-- section:5 -->
## 5. TargetBackendRegistry, target machines and NVIDIA / AMD examples

### 5.1 Where is the registry, and what does it contain? {#toolchain-q5-1}

**5.1.1 Declaration.** <code>Mojo/include/Mojo/Compiler/Target/TargetBackend.h</code> declares <code>M::KGEN::TargetBackendRegistry</code>. Its private storage is <code>std::vector&lt;std::unique_ptr&lt;TargetBackend&gt;&gt; Backends</code>: owned C++ implementation objects.

**5.1.2 Initialization.** A <code>RegisterTargetBackend&lt;BackendT&gt;</code> static object constructs a backend and calls <code>TargetBackendRegistry::get().add(...)</code>. Host, NVPTX and AMDGPU source files contain these registrations. Available implementations depend on which backends the compiler build includes.

**5.1.3 Lookup.** <code>Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp</code> iterates the registered implementations and calls <code>resolve(triple)</code>. It prefers an explicitly resolved backend over a direct self-match. A missing supported implementation produces an error. This is compiler dispatch over target triples.

| Compiler implementation | Source / registration evidence |
| --- | --- |
| <code>HostBackend</code> | <code>Target/Host/HostBackend.h</code> and <code>.cpp</code> |
| <code>NVPTXBackend</code> | <code>Target/max_license/NVPTX/NVPTXBackend.cpp</code> |
| <code>AMDGPUBackend</code> | <code>Target/max_license/AMDGPU/AMDGPUBackend.cpp</code> |
| Other compiled-in backends | Resolved by their registered implementation; this table focuses on the requested CPU/NVIDIA/AMD routes |

[SOURCE:backend_registry_declaration] · [SOURCE:backend_registry] · [SOURCE:host_backend] · [SOURCE:host_object] · [SOURCE:nvptx_backend] · [SOURCE:amdgpu_backend]

### 5.2 How is the backend selected? {#toolchain-q5-2}

**5.2.1 Target record.** <code>TargetInfoAttr</code> describes a triple, architecture, features, ABI, relocation model and data layout. <code>compileOffloads</code> groups device work by target, copies those fields to <code>CompilationOptions</code> and compiles a module sliced for that target.

**5.2.2 Mojo backend selection.** <code>ObjectCompiler::create</code> calls <code>TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple))</code>. The backend is resolved before its lowering/emission hooks are used.

**5.2.3 LLVM machine selection.** <code>defaultCreateTargetMachine</code> then uses the separate LLVM <code>TargetRegistry::lookupTarget</code> and <code>createTargetMachine(triple, targetCpu, targetFeatures, ...)</code>. ABI, relocation, code model and optimization settings also feed machine construction. The Mojo backend controls the artifact route; the LLVM target machine supplies the target's instruction/code-generation machinery.

| Illustrative target record | Selected backend / role |
| --- | --- |
| <code>x86_64-unknown-linux-gnu</code> + a compatible CPU/features | <code>HostBackend</code> for Linux host code |
| <code>nvptx64-nvidia-cuda</code> + <code>sm_90a</code> | <code>NVPTXBackend</code> for an NVIDIA device image |
| <code>amdgcn-amd-amdhsa</code> + <code>gfx942</code> | <code>AMDGPUBackend</code> for an AMD device code object |

The NVIDIA/AMD records come from the repository's cross-codegen test. The animation does not report that those devices are present or that the test was executed. [SOURCE:target_info] · [SOURCE:offload_targets] · [SOURCE:backend_select] · [SOURCE:llvm_target_machine] · [SOURCE:cross_target_example]

### 5.3 NVIDIA: LLVM NVPTX → PTX → cubin {#toolchain-q5-3}

#### 5.3.1 Emit PTX {#toolchain-q5-3-1}

With the <code>nvptx64-nvidia-cuda</code> triple and a selected SM architecture, <code>NVPTXBackend::emitObject</code> calls <code>ctx.runLlc(..., createObjectFile=false)</code>. The LLVM NVPTX backend emits **PTX text** into a byte buffer. PTX is the virtual device instruction representation consumed by NVIDIA's compilation tooling.

#### 5.3.2 Compile PTX for the same SM {#toolchain-q5-3-2}

The backend reads <code>ctx.tm.getTargetCPU()</code>, verifies that it starts with <code>sm_</code>, creates <code>Driver::CompilationDevice</code> with CUDA API and that architecture, then calls <code>compileFunction(ptx, debugLevel, optimizationLevel)</code>. The returned image is a **cubin**, containing native device code for that target.

The source explicitly ties cubin preparation to the SM used to emit PTX. Its Linux branch documents the NVPTXCompiler API and a configurable compiler-path override. The source boundary is the <code>CompilationDevice</code> call; it does not require every run to launch <code>nvcc</code> or a standalone <code>ptxas</code> process.

**5.3.3 Concrete source example.** The cross-codegen test requests <code>default-device-target-triple=nvptx64-nvidia-cuda</code> and <code>default-device-target-arch=sm_90a</code>, then inspects a generated <code>.ptx</code> diagnostic file for the LLVM NVPTX marker. This is a compiler-test specification, not a measured inference run.

~~~text
Mojo/lib/Compiler/ObjectCompiler/
  ObjectCompiler.cpp
  Target/TargetBackend.cpp
  Target/max_license/NVPTX/NVPTXBackend.cpp

TargetInfo(nvptx64-nvidia-cuda, sm_90a, features)
  → NVPTXBackend + LLVM TargetMachine
  → PTX bytes → CUDA CompilationDevice.compileFunction
  → cubin bytes → later module loading / kernel launch
~~~

[SOURCE:nvptx_object] · [SOURCE:nvptx_backend] · [SOURCE:cross_target_example]

### 5.4 AMD: LLVM AMDGPU → object → linked ELF / hsaco {#toolchain-q5-4}

**5.4.1 Target and implementation.** <code>AMDGPUBackend</code> inherits the host backend's object-emission mechanism and supplies AMD target traits, offload behavior, shared-memory address-space information and device-library/link hooks. A target such as <code>amdgcn-amd-amdhsa</code> + <code>gfx942</code> selects this route.

**5.4.2 Device-library preparation.** Its <code>linkRuntimeLibraries</code> constructs a <code>ROCDLTargetAttr</code> and uses <code>AMDGPUModuleLinker</code> for required ROCm bitcode. The shown ASAN case requests OCKL/OCML and <code>/opt/rocm/amdgcn/bitcode/asanrtl.bc</code>; those ASAN libraries are conditional rather than universal inputs for add-one.

**5.4.3 Machine code and link.** LLVM AMDGPU code generation emits target object bytes. The link step resolves the device object into a loadable ELF code object, commonly called an <code>hsaco</code>. The AMD backend adds <code>--no-undefined</code> to catch unresolved symbols during the link.

**5.4.4 Concrete source example.** The repository test selects <code>default-device-target-triple=amdgcn-amd-amdhsa</code>, <code>default-device-target-arch=gfx942</code> and checks an <code>.amdgcn</code> assembly diagnostic for its target marker.

~~~text
Mojo/lib/Compiler/ObjectCompiler/
  ObjectCompiler.cpp
  Target/TargetBackend.cpp
  Target/Host/HostBackend.cpp
  Target/max_license/AMDGPU/AMDGPUBackend.cpp

TargetInfo(amdgcn-amd-amdhsa, gfx942, features)
  → AMDGPUBackend + LLVM TargetMachine
  → required device bitcode linked into LLVM module
  → AMDGPU target .o bytes → device ELF link
  → loadable code object / hsaco → later device launch
~~~

[SOURCE:amdgpu_backend] · [SOURCE:host_object] · [SOURCE:llvm_code_gen] · [SOURCE:cross_target_example]

### 5.5 CPU route and file-format comparison {#toolchain-q5-5}

<code>HostBackend::emitObject</code> calls <code>ctx.runLlc(..., createObjectFile=true)</code> and <code>ctx.linkObject</code>. Host objects are linked into loadable native code. The JIT archive path also uses its MC-linking flow; the backend's <code>createArchive</code> hook itself is marked unwired in this revision.

| Representation | What it contains / when it appears |
| --- | --- |
| <code>.mojo</code> | Source code |
| <code>.mojoc</code> | Precompiled post-parse Mojo package / MLIR bytecode |
| MO / MAP / MOGG / MGP | In-memory MAX graph dialect stages; optional <code>.mlir</code> dumps |
| LLVM dialect | MLIR representation suitable for LLVM translation |
| LLVM IR / <code>.ll</code> / <code>.bc</code> | LLVM module / optional text or bitcode export |
| CPU <code>.o</code> → native linked code | Host instruction bytes and link resources |
| PTX → cubin | NVIDIA virtual instruction text → native device image |
| AMDGPU <code>.o</code> → ELF / hsaco | AMD device object → loadable device image |
| MEF / compiled artifact | Runtime plan and code/resources prepared for initialization |

Object emission does not require writing an assembly file first. Text assembly is an optional output; NVIDIA's PTX route explicitly uses a textual intermediate. The file sheets in the animation represent in-memory compiler data unless a dump/export is named. [SOURCE:host_object] · [SOURCE:nvptx_object] · [SOURCE:amdgpu_backend] · [SOURCE:artifact]

<!-- section:6 -->
## 6. How host and device work are differentiated, initialized and executed

### 6.1 Select graph placement {#toolchain-q6-1}

**6.1.1 Add-one source choice.** The unchanged example uses:

~~~python
device = CPU() if accelerator_count() == 0 else Accelerator()
~~~

It supplies <code>DeviceRef.from_device(device)</code> to the custom operation and tensor types, then constructs <code>InferenceSession(devices=[device])</code>. The compiler's MO device-validation/configuration work supplies the corresponding target information.

**6.1.2 One selected arithmetic device.** This script applies add-one on its chosen CPU or accelerator. A GPU route still has a CPU application and host launch/control routines. A graph can describe multiple placements; this example's single operation has one explicit placement. [SOURCE:reference_addition] · [SOURCE:mo_pipeline] · [SOURCE:gc_pipeline]

### 6.2 Separate host routines from device offloads {#toolchain-q6-2}

**6.2.1 Authored dispatch.** <code>AddOne.execute[target]</code> passes its compile-time target to <code>foreach</code>. CPU iteration and GPU dispatch are target-specific implementations of that operation.

**6.2.2 Compiler mechanism.** Mojo elaboration can encounter device offload compilation. <code>compileOffloads</code> receives a map keyed by <code>TargetInfoAttr</code>, slices the relevant generators into a target-specific module, applies that group's options and compiles the group. The host module retains control/launch code and references to the resulting device image.

**6.2.3 Why the Python program stays on the host.** The compiler only sends selected native routine IR to a device backend. Python constructs the graph and supplies buffers; native host routines interact with device APIs. Device image bytes contain the selected kernels, with the target's ABI, address spaces and entry metadata.

{{diagram:offload}}

~~~mermaid
flowchart TD
    APP["CPU process · Python + MAX"] --> HOST["host routine IR / HostBackend"]
    HOST --> LAUNCH["host driver launch + Buffer references"]
    HOST -->|offload generator + TargetInfo| DEV["separate device IR"]
    DEV --> NV["nvptx64 / sm → PTX → cubin"]
    DEV --> AMD["amdgcn / gfx → object → ELF"]
    NV --> IMAGE["resident compatible device image"]
    AMD --> IMAGE
    IMAGE --> GPU["physical GPU executes kernels"]
    LAUNCH --> GPU
~~~

~~~text
CPU process: Python → MAX host routines → driver API
                    │
                    ├─ host code / runtime state stays on CPU
                    └─ selected offload IR → GPU backend → device image
                                            │
host launch(kernel handle, device Buffers) ───┴─→ GPU execution
~~~

[SOURCE:reference_add_one] · [SOURCE:offload_targets] · [SOURCE:target_info] · [SOURCE:driver_execute]

### 6.3 Finish the runtime call interface {#toolchain-q6-3}

**6.3.1 Prepared placeholders.** Kernel compilation creates compiled-load resources and compiled routine calls in the lowered module.

**6.3.2 Lowering owner.** <code>LowerCompiledRoutinesPass</code> converts these into module loading and runtime execution/call operations such as <code>ModuleLoad</code> and <code>ExecuteGeneric</code>. <code>buildJitPipeline</code> then inlines/cleans up MGP calls, simplifies chains and outlines model callables.

**6.3.3 Resulting contract.** Initialization knows which modules/callables to load. Each execution knows which input references and dependencies feed those callables. This is the graph's runtime interface around compiled routines. [SOURCE:runtime_calls] · [SOURCE:jit_pipeline]

### 6.4 Prepare the compiled artifact / MEF bytes {#toolchain-q6-4}

**6.4.1 Artifact owner.** The engine API exposes a <code>CompiledModel</code> wrapper for prepared compilation. Its runtime plan, code resources and metadata belong to CM1; G1 remains the symbolic graph.

**6.4.2 In-memory preparation.** The example assigns <code>compiled = session.compile(graph)</code>. It does not explicitly export a <code>.mef</code> disk file. A prepared byte buffer is not yet the initialized executable model.

**6.4.3 Data separation.** There are no model weights to bind in these four tensor-input examples. Their numerical arrays are passed to execution. [SOURCE:artifact] · [SOURCE:serialize] · [SOURCE:reference_addition]

### 6.5 Initialize M1 {#toolchain-q6-5}

**6.5.1 Actual call.** <code>model = session.init(compiled)</code> initializes the prepared artifact with the session's runtime/device context.

**6.5.2 Runtime consequences.** Module-load and callable initialization resolve resident code/resources and executable handles. M1 can then accept input Buffers. CM1's compiled bytes, M1's initialized handles and S1's session context are separate identities in the scene.

**6.5.3 Reuse.** The initialized model can be retained for repeated executions with fresh inputs. The matmul example's helper specifically compiles/initializes a new model per helper invocation while its main function reuses the session; the diagram does not change that source ordering. [SOURCE:init_all] · [SOURCE:runtime_calls] · [SOURCE:entry]

### 6.6 Supply buffers and execute {#toolchain-q6-6}

**6.6.1 Input creation.** For add-one, <code>Buffer.from_numpy(x_values).to(device)</code> supplies numerical storage to the selected device. Wrapping/copying data and compilation are distinct operations.

**6.6.2 Actual call.** <code>model.execute(x)</code> hands Buffer references to native execution. MLRT's model/executor dependency machinery coordinates ready operations and result values. This is native graph-runtime work, independent of Serve's request batching scheduler.

**6.6.3 Host/device action.** A CPU routine operates on CPU storage. A GPU routine's host launch references resident device code and accelerator Buffers. Actual driver queues, scheduling overlap and elapsed time are beyond this browser teaching model. [SOURCE:reference_addition] · [SOURCE:execute] · [SOURCE:native_execute] · [SOURCE:driver_execute]

### 6.7 Obtain result readiness and host-readable values {#toolchain-q6-7}

**6.7.1 Readiness.** The runtime returns/coordinates result Buffers through dependency/readiness state. The executor tracks operands that are not ready and triggers dependent work as their values become available.

**6.7.2 Add-one read.** The Python source calls <code>result.to(CPU())</code>, then <code>result.to_numpy()</code>. GPU values must become CPU-readable through that explicit operation. A returned Buffer handle by itself is not a copied NumPy array.

**6.7.3 Verification limits in the originals.** Addition prints result and expected values without a numerical equality assertion. The epilogue and matmul scripts contain numerical checks. The attention script prints output Buffers without an explicit host copy/parity assertion. The browser's small arithmetic checks verify the teaching model rather than execution of MAX on a GPU. [SOURCE:ready_counts] · [SOURCE:driver_results] · [SOURCE:reference_addition] · [SOURCE:entry]

<!-- section:7 -->
## 7. Compiler tools, diagnostics and what the evidence establishes

### 7.1 Which tool runs at each boundary? {#toolchain-q7-1}

| Step | Tool / implementation actually responsible | Invocation boundary |
| --- | --- | --- |
| Build dependency graph | <code>./bazelw</code> → Bazel and declared rules | Build orchestration |
| Shared Mojo packages | Configured Mojo compiler through <code>mojo_library</code> | Bazel build action producing <code>.mojoc</code> |
| Runtime source-package preparation | <code>mojo.paths._build_mojo_source_package</code> → <code>mojo precompile</code> | Explicit subprocess in the Python application's package-preparation path |
| MAX MO/MAP/MOGG/MGP transformations | GraphCompiler C++ library + MLIR pass managers | Native compiler APIs in the application process |
| Kernel elaboration | <code>KGENCompiler</code> + elaboration pipeline/evaluators | Native compiler APIs and their work queues |
| KGEN/POP → LLVM dialect | <code>ObjectCompiler</code> + <code>buildLowerToLLVMPipeline</code> | MLIR passes in the compiler |
| LLVM dialect → LLVM IR | <code>mlir::translateModuleToLLVMIR</code> | MLIR export API producing <code>llvm::Module</code> |
| Target code generation | LLVM <code>TargetMachine</code>, <code>runLlcPasses</code>, selected backend | LLVM code-generation libraries |
| NVIDIA PTX preparation | CUDA <code>CompilationDevice::compileFunction</code> | Driver/compiler implementation boundary; configurable tooling |
| Host / AMD object linking | Object-compiler/link hooks and configured link tools | Link step; precise invocation follows target/tool configuration |
| Initialization / inference | MAX native runtime and device API | Loaded artifact + ready input/result resources |

<code>mo-opt</code>, <code>kgen-opt</code> and <code>mlir-opt</code> are useful compiler development tools. Their existence does not make them required subprocesses in <code>:addition</code>. Likewise, <code>mojo build</code> is a standalone program-build interface; MAX kernel JIT invokes the corresponding compiler machinery through native APIs.

[SOURCE:bazel_macro] · [SOURCE:mojo_library_rule] · [SOURCE:package_builder] · [SOURCE:gc_pipeline] · [SOURCE:kgen_elaboration] · [SOURCE:kgen_llvm_passes] · [SOURCE:llvm_translation] · [SOURCE:llvm_code_gen]

### 7.2 How can each representation be inspected? {#toolchain-q7-2}

**7.2.1 Read the actual definitions first.** Source cards and the offline Source Explorer expose complete unchanged source files, the selected function/pass span, its path and pinned revision.

**7.2.2 Diagnostic files are optional.** Graph-stage dumps can expose MO, MAP, MOGG and MGP views. ObjectCompiler's temporary-output policy can write pre-LLVM-dialect/pre-LLVM-IR <code>.mlir</code> views and LLVM/native diagnostics. These are separate from the ordinary example's required build outputs.

**7.2.3 Target examples are source tests.** <code>GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir</code> supplies the explicit NVIDIA/AMD triples and architectures shown in point 5. <code>gc-pipeline-dump.mlir</code> specifies stage-dump checks. Their <code>RUN</code> lines describe diagnostic compiler-test invocations; they have not been executed as part of this HTML update.

**7.2.4 Evidence categories.**

| Evidence shown here | What it establishes |
| --- | --- |
| Pinned repository source | Call ordering, pass construction, target selection and representation ownership |
| Inspected compiler-test source | Expected cross-target/dump behavior specified by the test |
| Deterministic browser state and arithmetic | The animation's transitions, controls and small numerical illustrations |
| Actual MAX / GPU execution | Requires running the real example and recording its outputs on the selected device; no such execution is claimed by this page |

The previous point 1 explicitly records an interrupted cold Bazel build. This update preserves that statement and does not replace it with a successful build claim.

[SOURCE:graph_stage_dumps] · [SOURCE:cross_target_example] · [SOURCE:object_lower_to_llvm]
