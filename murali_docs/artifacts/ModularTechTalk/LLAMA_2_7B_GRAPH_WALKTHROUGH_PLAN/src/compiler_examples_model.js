/* Repository examples: deterministic teaching state, independent of DOM/time. */
(function(root) {
  'use strict';
  const clone=v=>JSON.parse(JSON.stringify(v));
  const A=[[1,2],[3,4]], B=[[2,0],[1,2]];
  const TA=[[1,2,3,4],[2,1,0,1]], TB=[[1,2],[0,1],[2,0],[1,1]];
  const Q=[1,0], K=[[1,0],[0,1],[3,0],[2,1]], V=[[1,0],[0,2],[2,1],[1,3]];
  const matmul=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0)));
  const fmt=n=>Number.isInteger(n)?String(n):Number(n.toFixed(4)).toString();
  function parameters(id,target,algorithm) {
    const cpu=target==='cpu', amd=target==='rocm';
    if(id==='attention')return cpu?{N:8,D:8,BD:4,BN:4,keyTile:4,grid:'native CPU loops'}:{N:32,D:32,BD:8,BN:16,keyTile:8,grid:'grid (4, 2) · block 32'};
    if(id==='matmul') {const block=amd?16:32;return {M:256,K:256,N:256,BM:cpu?null:algorithm==='tiled'?block:16,BN:cpu?null:algorithm==='tiled'?block:16,BK:cpu?null:algorithm==='tiled'?block:null,grid:cpu?'native CPU loops':algorithm==='tiled'?`grid (${256/block}, ${256/block}) · block ${block*block}`:'grid (16, 16) · block (16, 16)'};}
    return id==='epilogue'?{M:5,K:8,N:10}:{M:5,N:10};
  }
  function descriptions(id,p) {
    return id==='addition'?['X [5,10] f32','Y [5,10] = X + 1']:id==='epilogue'?['A [5,8] · B [8,10]','Y [5,10] = (A @ B + 1) × 2']:id==='matmul'?['A [256,256] · B [256,256]','C [256,256] = A @ B']:[`Q / K / V [${p.N},${p.D}]`,`O [${p.N},${p.D}] = softmax(Q Kᵀ) V`];
  }
  class ExampleModel {
    constructor(kind,example,options={}) {this.kind=kind;this.example=example;this.configure(options);}
    configure(options={}) {this.options={target:'cuda',fusion:'fused',algorithm:'tiled',...options};this.index=0;this.frames=[];this.build();return this.snapshot();}
    build() {
      const id=this.example.id,o=this.options,cpu=o.target==='cpu';
      const algorithm=cpu?'naive':o.algorithm,p=parameters(id,o.target,algorithm);
      const name=id==='addition'?'add_one':id==='epilogue'?'add_one_elementwise':id==='matmul'?'matrix_multiplication':'modular_ops::fused_attention_custom';
      const routine=id==='addition'?'AddOne.execute → foreach':id==='epilogue'?'AddOneElementwise.elementwise':id==='matmul'?cpu?'naive_matrix_multiplication_cpu':algorithm==='tiled'?'tiled_matrix_multiplication':'naive_matrix_multiplication':cpu?'fused_attention_cpu':'fused_attention_kernel';
      let state={id,filename:this.example.filename,phase:'start',target:o.target,cpu,algorithm,params:p,name,routine,signature:descriptions(id,p),fusion:o.fusion==='fused',graph:false,registered:false,verified:false,clone:false,level:0,code:false,mef:false,initialized:false,hostInputs:false,inputs:false,job:false,ready:false,hostRead:false,computed:[],acc:[],tile:-1,copyReady:false,blockReady:false,shared:false,temporary:[],lane:-1,cell:-1,k:-1,oldM:null,m:null,l:0,O:[0,0],oldO:[0,0],scores:[],E:[],alpha:0,tileCount:0};
      const add=(phase,title,detail,patch={},refs=['entry'])=>{refs=refs.map(r=>id==='attention'&&cpu&&r==='math'?'math_cpu':id==='attention'&&cpu&&r==='dispatch'?'registration':r);state={...state,...clone(patch),phase,title,detail,refs};this.frames.push(clone(state));};
      if(this.kind==='build') {
        add('source','Read the actual Python example',this.example.expression);
        if(id==='matmul')add('buffers','Create input buffers before constructing this graph','The helper calls Buffer.from_numpy(...).to(device) first. Graph input types use those buffer shapes.',{hostInputs:true,inputs:true},['entry']);
        if(id==='attention')add('contract','Create the empty attention graph shell','Graph.__init__ converts three input types, creates the module/GraphOp/body arguments, and seeds ordering before it imports the kernel package. The with-body runs after the constructor returns.',{graph:true,shell:true},['attention_callsite','graph_init','ct_graph_op_build']);
        add('package','Prepare the implementation package','KernelLibrary loads the source directory as a precompiled .mojoc package. The package retains MLIR implementation bodies for later specialization.',{package:true},['package_load','mojo_package']);
        if(id!=='attention')add('contract','Describe shapes, dtype and device','TensorType describes symbolic inputs. These are contracts, not numerical inference results.',{graph:true},['entry','graph_init']);
        add('register',id==='attention'?'Resolve the implementation before recording the custom call':'Import implementations before invoking the builder',`Graph imports kernels/. The registration name resolves ${routine} before ops.custom verifies its call.`,{registered:true},['graph_init','registration','custom_api']);
        add('custom','Record a named custom operation',`ops.custom("${name}") records the operation and explicit output type.`,{ops:1},['entry','custom_api']);
        if(id==='epilogue')add('compose','Keep the surrounding math visible','The Python @ operator records matmul; add_one_elementwise and ×2 compose after it. Three mathematical operations can become one eligible epilogue route.',{ops:3},['entry','registration']);
        add('output','Finalize the graph output contract',id==='addition'||id==='epilogue'?'Graph(forward=...) calls the builder and finalizes its returned output through Graph.output.':'The context-manager example explicitly calls graph.output(...) before leaving Graph.',{verified:true},['entry','graph_output','graph_init']);
        add('verified','G1 is verified. No executable model yet.',id==='addition'?'The explicit output TensorType makes the optional add_one_shape function unnecessary here. It is not implemented in this example.':'Typed inputs, explicit output and registered implementation are ready for compilation.',{},['entry',id==='addition'?'shape_function':'registration']);
      } else if(this.kind==='compile') {
        state={...state,graph:true,verified:true,registered:true};
        add('session','S1 owns the selected compilation/runtime context','A verified graph is the input. The session constructor does not execute its arithmetic.',{},['entry','artifact']);
        add('clone','Compile a private graph clone','G1 stays with the Python caller. C1 is the compiler-owned module that changes representation.',{clone:true,level:0},['frontend','compile_async']);
        add('map','MO → MAP: expose implementation bodies',id==='epilogue'?'The elementwise implementation exposes x+1 to composition. Its source documents epilogue fusion eligibility.':'Lower the high-level operation into implementation-oriented bodies. The displayed IR is a schematic view.',{level:1},['mo_pipeline','map_pipeline','registration']);
        if(id==='epilogue')add('fusion',o.fusion==='fused'?'Group the eligible epilogue':'Compare a separate-body route',o.fusion==='fused'?'Illustrative optimized case: keep matmul accumulators live while applying +1 and ×2. No intermediate matrix store is needed in this sketch.':'Illustrative comparison: materialize the matmul result, then add one, then multiply. This selector is a teaching comparison, not a MAX compiler flag.',{level:1},['entry','registration','map_pipeline']);
        add('mogg','MAP → MOGG: express kernel calls','The surrounding graph and custom implementation remain different levels. Actual dialect spellings and chosen passes are not a compiler dump from this example.',{level:2},['mogg_pipeline']);
        add('mgp','MOGG → MGP: buffer/dependency plan','The execution plan describes input/output resources and dependencies; it does not contain the random input values.',{level:3},['mgp_pipeline']);
        add('finalize','Check device-graph finalization, then emit routines','The scheduled pass audits collectives only when getIsDeviceGraph() is true. Ordinary models skip that audit. This later compiler preparation is distinct from Python Graph.output; emitted routines and custom Mojo enter kernel compilation.',{level:4},['mgp_pipeline','gc_pipeline','device_finalization']);
        add('mojo','Mojo specializes the implementation',id==='matmul'?`Compile-time target=${cpu?'cpu':'gpu'}, algorithm=${algorithm}. ${cpu?'CPU uses the naive function.':p.grid+'.'}`:id==='attention'?`BN=${p.BN}, BD=${p.BD}; ${routine}. ${cpu?'CPU key tile BN=4.':'GPU key tile BN_1=8, distinct from BN=16.'}`:'Types, SIMD width and target become concrete native code; the elementwise body does not call Python for every value.',{level:5},['registration','dispatch','mojo_semantics','mojo_elaboration']);
        add('llvm','Mojo MLIR → LLVM dialect → LLVM IR','LIT/KGEN/POP lowering and LLVM-module translation are separate transformations. File sheets represent in-memory views.',{level:6},['mojo_llvm','llvm_translation']);
        add('target',cpu?'Host backend emits and links native code':o.target==='cuda'?'LLVM NVPTX → PTX → cubin':'LLVM AMDGPU → object → linked ELF',cpu?'The host backend can emit an object directly and link native code. Assembly is an optional textual output.':o.target==='cuda'?'The pinned backend passes PTX through CUDA CompilationDevice to produce same-SM code. This is host compilation targeting a GPU.':'The AMDGPU backend emits target object code and links an ELF code object. This is host compilation targeting a GPU.',{level:7,code:true},['backend_select',cpu?'host_object':o.target==='cuda'?'nvptx_object':'amdgpu_backend']);
        add('runtime_calls','Finish the graph’s runtime call interface','The kernel JIT route lowers compiled-load/call placeholders to module loading and runtime calls, then outlines callables. This happens after kernel compilation.',{runtimeCalls:true},['runtime_calls','jit_pipeline']);
        add('mef','Prepare the runtime artifact / MEF bytes','Compiled routines plus prepared execution-plan resources form an in-memory compiled artifact. These scripts do not export a .mef disk file.',{level:8,mef:true},['serialize','artifact','gc_pipeline']);
        add('load_modules','Load modules and resolve executable handles','Initialization uses the prepared module-load and callable plan. Compiled bytes alone are not an initialized executable.',{loading:true},['runtime_calls','init_all']);
        add('init','session.init(compiled) creates executable M1','All four examples take their tensors as call inputs. No model-weight registry is required by these calls.',{initialized:true},['entry','init_all']);
        add('ready','M1 is ready; numerical inputs are separate',id==='matmul'?'The helper already created its buffers. Each new helper invocation compiles and initializes a model, while main reuses S1.':'Random arrays and Buffer arguments are supplied after initialization. Compilation did not compute their results.',{},['entry']);
      } else if(this.kind==='toolchain') {
        const targetName=cpu?'CPU host':o.target==='cuda'?'NVIDIA GPU':'AMD GPU';
        const backend=cpu?'HostBackend':o.target==='cuda'?'NVPTXBackend':'AMDGPUBackend';
        const artifact=cpu?'host object → native code':o.target==='cuda'?'PTX → cubin':'AMDGPU object → ELF / hsaco';
        const commandTarget=`//max/examples/custom_ops:${this.example.task}`;
        const triple=cpu?'x86_64-unknown-linux-gnu':o.target==='cuda'?'nvptx64-nvidia-cuda':'amdgcn-amd-amdhsa';
        const arch=cpu?'compatible host CPU':o.target==='cuda'?'sm_90a':'gfx942';
        state={...state,task:this.example.task,commandTarget,targetName,backend,artifact,triple,arch,
          qa:'1.1',stage:'build',built:false,hostRuntime:false,sourceRef:'bazel_binary',sourceNeedle:'name =',
          graphIR:'absent',kernelIR:'absent',views:[],g1Retained:false,libraries:false,runfiles:false,
          package:false,hostSource:false,deviceIR:false,llvmModule:false,deviceIntermediate:false,
          targetCode:false,hostCode:false,runtimeCalls:false,representation:'Bazel target graph'};
        const f=(qa,phase,title,detail,patch={},refs=['entry'])=>add(phase,title,detail,
          {qa,sourceRef:refs[0],sourceNeedle:'',...patch},refs);
        f('1.1','bazel_target','Resolve the Python example target',
          `./bazelw build ${commandTarget} resolves the macro, Python binary and declared dependencies. `+(id==='addition'?'This frame is a build-graph illustration; the earlier interrupted build remains the recorded result.':'This frame is a source-derived build-graph illustration; no successful example build is claimed.'),
          {command:`./bazelw build ${commandTarget}`,representation:'BUILD.bazel → Python binary target',sourceRef:'bazel_binary',sourceNeedle:'name ='},['bazel_wrapper','bazel_binary','bazel_build_file']);
        f('1.2','bazel_libraries','Precompile the shared Mojo dependencies',
          'The layout, extensibility and std labels resolve to Mojo source-package targets. mojo_library invokes the configured Mojo compiler to create separate .mojoc packages containing post-parse MLIR bytecode.',
          {libraries:true,representation:'Mojo library sources → layout / extensibility / std .mojoc',sourceNeedle:'Precompiles sources'},['mojo_library_rule','bazel_aliases','bazel_layout_package','bazel_extensibility_package','bazel_std_package']);
        f('1.1','bazel_outputs','Stage the launcher and raw kernel data',
          `The ${this.example.task} Python target provides a launcher and runfiles containing ${this.example.filename}, Python dependencies and kernels/*.mojo data. The independently declared :kernels library belongs to a separate dependency route used by :benchmarks.`,
          {built:true,representation:'Python launcher + .py runfiles + raw .mojo data',sourceNeedle:'data ='},['bazel_macro','bazel_kernel_sources','bazel_build_file']);
        f('1.3.2','toolchain_runfiles','Make compiler and linker tools available at runtime',
          'modular_py_binary creates mojo_test_environment. It stages the source-built Mojo driver, LLD, native libraries and import/link settings. The example-test is a sibling target that runs the launcher with its runfiles.',
          {runfiles:true,representation:'runfiles + MODULAR_MOJO_MAX_* environment',sourceNeedle:'extra_env ='},['bazel_py_binary','mojo_runtime_exports','bazel_mojo_toolchain','bazel_run_test','bazel_binary_test']);
        f('2.1','python_process','Run the Python application on the host CPU',
          `./bazelw run ${commandTarget}, or its .example-test, starts ${this.example.filename}. The CPU process owns graph construction, compilation requests, driver interaction and later Buffer arguments.`,
          {stage:'package',hostRuntime:true,command:`./bazelw run ${commandTarget}`,representation:'CPU Python process → Graph(custom_extensions=kernels/)'},['entry','bazel_binary_test']);
        if(id==='attention')f('2.1','graph_shell','Create the empty module and three-argument graph shell',
          'In source order, Graph.__init__ creates the MLIR module, GraphOp signature/body and chain bookkeeping before importing kernels/. The caller supplies three static TensorTypes; no forward lambda or numeric Q/K/V runs here.',
          {shell:true,representation:'MODULE1 → G1 shell: (%q,%k,%v); result types initially empty',sourceNeedle:'with _location()'},['graph_init','ct_graph_op_build','attention_callsite']);
        f('2.2','package_source','Precompile kernels/ into a private staging package',
          'KernelLibrary.load_paths calls mojo.paths._build_mojo_source_package. It invokes mojo precompile <kernels directory> -o <staging path>. The helper atomically publishes mojo_pkg_<path_hash>.mojoc under the temporary per-user package directory.',
          {package:true,representation:'kernels/*.mojo → staging .mojoc → atomically published P1',sourceNeedle:'args ='},['package_builder','package_load','kernel_package_root']);
        f('2.3','package_import','Import declarations and bodies from P1',
          'P1 is a lit.package serialized as MLIR bytecode. It retains implementation bodies for concrete instantiation. The registration string connects the Python custom operation to its Mojo definition; the Python entry-point file remains separate.',
          {registered:true,representation:'P1 post-parse MLIR → registered Mojo implementations'},['mojo_package','registration','custom_api']);
        f('2.4','graph_record','Record G1 with shapes, dtype and device',
          `${this.example.expression}. The original graph contract is ${state.signature.join(' → ')}. Graph.output completes its symbolic contract; numerical execution is a later phase. Kernel declarations can be seeded for the native compiler.`,
          {graph:true,verified:true,g1Retained:true,representation:'G1 · verified symbolic graph; values not supplied',sourceNeedle:'Graph('},['entry','graph_init','graph_output','compile_async']);
        f('3.1','clone','Clone G1 into compiler-owned C1',
          'InferenceSession.compile reaches MOFContext::compileToBinary. The native frontend clones the MLIR module, registers context and gathers GCPipelineOptions. G1 stays with its caller while C1 is transformed by MLIR pass managers.',
          {stage:'graph',clone:true,graphIR:'MO',views:['MO'],representation:'G1 retained → clone C1 / MO',sourceNeedle:'.clone()'},['frontend','compile_async','gc_pipeline']);
        f('3.2','mo_opt','MO: refine graph-level information',
          'MO means Modular Operations. buildMoPipeline schedules symbolic optimization, shape/layout work, kernel verification, device validation and cleanup. C1 still carries graph operations and tensor contracts.',
          {representation:'C1 · MO / typed tensor operations',sourceNeedle:'void M::MO::buildMoPipeline'},['mo_pipeline','mo_dialect','graph_compiler_design']);
        f('3.3.1','mo_to_map','MOToMAPPass exposes implementation accesses',
          'createMOToMAP constructs MOToMAPPass. Its runOnOperation creates MojoRegistry, collects lowerable operations and invokes lowerOp. The resulting MAP representation exposes supported kernel/access structure.',
          {graphIR:'MAP',views:['MO','MAP'],representation:'C1 · MO → MAP / load · compute · store',sourceNeedle:'void runOnOperation'},['mo_to_map','gc_pipeline','map_dialect']);
        f('3.3.2','map_fusion','MAP: optimize accesses and eligible fusion',
          id==='epilogue'?'AddOneElementwise exposes x+1 beside matmul and ×2. buildMapPipeline can fuse eligible bodies, allocate accumulation buffers and outline structured kernels. The drawn fusion structure is schematic.':'MAP exposes opaque or explicit elementwise accesses. buildMapPipeline applies eligible fusion, accumulation-buffer allocation and outlining. The selected example retains its authored kernel algorithm; this is a representation illustration.',
          {representation:'C1 · MAP / accesses + implementation fusion interfaces',sourceNeedle:'MAP::createFuseKernels'},['map_pipeline','map_dialect','registration']);
        f('3.4','map_to_mogg','MAPToMOGGPass builds generator-oriented calls',
          'MAPToMOGGPass uses MojoRegistry and MAPToMOGGState to lower outlined kernels and shape-function invocations. buildMoggPipeline conforms call operands and outlines routines. C1 now expresses code-generation-oriented implementation boundaries.',
          {graphIR:'MOGG',views:['MO','MAP','MOGG'],representation:'C1 · MAP → MOGG / generator calls',sourceNeedle:'MAPToMOGGState state'},['map_to_mogg','mogg_pipeline','mogg_dialect']);
        f('3.5','mogg_to_mgp','MOGGToMGPPass builds model/runtime glue',
          'The conversion orders MO graphs by call dependencies, uses SymbolTableAnalysis and runs MOGraphLowering. MGP expresses model init/execute, buffers, contexts and dependencies around the selected routines.',
          {graphIR:'MGP',views:['MO','MAP','MOGG','MGP'],representation:'C1 · MOGG → MGP / init · execute · buffers',sourceNeedle:'MOGraphLowering('},['mogg_to_mgp','mgp_dialect']);
        f('3.5.2','mgp_plan','MGP: assign reusable resources and dependencies',
          'buildMgpPipeline plans/reuses buffers, defers allocations, validates arguments, simplifies chains, globalizes state slots and lowers primitives. Drawn X / dependency / Y slots represent the runtime plan rather than actual numerical data.',
          {plan:true,representation:'C1 · MGP / input ref → dependency → output resource',sourceNeedle:'createDynamicMemoryPlanningPass'},['mgp_pipeline','mgp_dialect']);
        f('3.6','finalize','Run the conditional device-graph finalization pass',
          'The pipeline stages graphs and computes device dependencies before FinalizeDeviceGraphsPass. Its getIsDeviceGraph guard skips ordinary models; device-graph models finalize their collectives. The guard is shown without claiming a captured device graph for this example.',
          {finalization:true,representation:'C1 · MGP / device-graph pass: audit only if present',sourceNeedle:'getIsDeviceGraph'},['device_finalization','mgp_pipeline']);
        f('3.7','emit_mojo','Emit routine source; keep the MGP plan in C1',
          'EmitMojo imports standard/custom packages and generates routine scaffolding. Kernel preparation creates a separate parsed implementation module KIR1. C1 retains its buffer/dependency plan throughout later kernel lowering.',
          {emitted:true,kernelIR:'generated Mojo / imported bodies',representation:'C1 MGP retained → emitted Mojo + separate KIR1',sourceNeedle:'Import mojo packages'},['emit_mojo','gc_pipeline','jit_entry']);
        f('4.1','jit_lookup','Enter MAX kernel JIT and the cold-cache route',
          'compileMojoToBinary parses generated source, runs LIT-to-KGEN and creates KGENCompiler, ObjectCompiler and CompilationState. This lesson follows a cache-miss illustration; compatible compiled archives can skip fresh kernel preparation.',
          {stage:'mojo',kernelIR:'parametric KGEN',representation:'KIR1 · LIT → parametric KGEN; cache miss selected',sourceNeedle:'KGEN::KGENCompiler compiler'},['jit_entry','kernel_misses','kernel_hits']);
        f('6.2','offload_split',cpu?'Keep the native operation on its CPU target':'Separate host routine IR from device offload IR',
          cpu?'The selected operation uses native CPU iteration. There is a host routine and no GPU image in this target route.':'During elaboration, compileOffloads groups selected generators by TargetInfoAttr and slices a separate device module. The host module retains launch/control routines and references to the device image.',
          {hostSource:true,deviceIR:!cpu,representation:cpu?'host native routine / CPU placement':'host launch module + separate device kernel module',sourceNeedle:'targetOffloadInfos'},['offload_targets','registration','target_info']);
        f('5.2','target_info','Resolve the selected target record',
          `The focused target record is ${triple}, architecture ${arch}, plus features/ABI/layout. The NVIDIA sm_90a and AMD gfx942 examples are taken from the cross-codegen source test, not detected hardware. Host compilation keeps its own host target.`,
          {stage:'target',targetInfo:true,representation:`TargetInfo: ${triple} / ${arch} / features`,sourceNeedle:'targetTriple ='},['target_info','offload_targets','cross_target_example']);
        f('5.1','select_backend',`Select ${backend} and its LLVM target machine`,
          'ObjectCompiler::create resolves the triple through TargetBackendRegistry. Its owned C++ backend implementations register during static initialization. LLVM TargetRegistry separately supplies a TargetMachine using the triple, CPU, features and ABI.',
          {selectedBackend:backend,representation:`TargetBackendRegistry → ${backend} → LLVM TargetMachine`,sourceNeedle:'TargetBackendRegistry::get().lookup'},['backend_select','backend_registry_declaration','backend_registry','llvm_target_machine']);
        f('4.2.2','mojo_specialize','Instantiate concrete kernel parameters',
          `${routine}. KGENCompiler::runElaborationPipeline supplies the target to the elaborator. Function/type generators acquire concrete dtype, rank, compile-time parameters and implementation choices; the small vector grouping is illustrative.`,
          {stage:'mojo',kernelIR:'concrete KGEN / POP',specialized:true,representation:'KIR1 · parametric generators → concrete functions / types',sourceNeedle:'populateElaborateModulePasses'},['kgen_elaboration','kgen_elaborate_build','registration','mojo_elaboration']);
        f('4.3','post_elaboration','Optimize and lower the concrete implementation',
          'buildPostElaborationPipeline applies floating-point policy, verifies concrete IR, updates compile-time materialization and invokes its optimization pipelines. Runtime tensor contents remain separate inputs.',
          {postOptimized:true,representation:'KIR1 · concrete native operations / conventions',sourceNeedle:'buildFirstOptPipeline'},['kgen_post_elaboration','mojo_elaboration']);
        f('4.4','llvm_dialect','Lower KGEN / POP / control flow to LLVM dialect',
          'ObjectCompiler::lowerAllFuncsToLLVM invokes buildLowerToLLVMPipeline. LowerKGENToLLVM, POP lowering, control-flow/closure lowering and target-specific MLIR passes finish an LLVM-compatible MLIR module.',
          {kernelIR:'LLVM dialect / MLIR',representation:'KIR1 · LLVM dialect, still an MLIR module',sourceNeedle:'createLowerKGENToLLVM'},['kgen_llvm_passes','object_lower_to_llvm','mojo_llvm']);
        f('4.5','llvm_translate','Create a separate LLVM IR module',
          'The ObjectCompiler helper calls mlir::translateModuleToLLVMIR and returns an llvm::Module in an LLVM context. LLVM1 is a distinct representation; optional .ll/.bc exports are views of it.',
          {llvmModule:true,representation:'KIR1 MLIR → LLVM1 / llvm::Module',sourceNeedle:'mlir::translateModuleToLLVMIR'},['llvm_translation','object_lower_to_llvm']);
        f(cpu?'5.5':o.target==='cuda'?'5.3.1':'5.4','llvm_codegen',cpu?'LLVM emits host object bytes':o.target==='cuda'?'LLVM NVPTX emits PTX text':'LLVM AMDGPU emits device object bytes',
          cpu?'HostBackend runs the LLVM machine-code passes with object output enabled. The native object can then be linked for loading.':o.target==='cuda'?'NVPTXBackend runs LLVM code generation with textual output, producing PTX in a byte buffer for the selected SM.':'AMDGPU supplies device-library/link hooks and uses LLVM target object emission. Required bitcode is prepared before the final object/link route.',
          {stage:'target',deviceIntermediate:true,representation:cpu?'LLVM1 → host .o bytes':o.target==='cuda'?'LLVM1 → PTX byte buffer':'LLVM1 → AMDGPU .o byte buffer',sourceNeedle:cpu?'createObjectFile=*/true':'emitObject'},['llvm_code_gen',cpu?'host_object':o.target==='cuda'?'nvptx_object':'amdgpu_backend']);
        f(cpu?'5.5':o.target==='cuda'?'5.3.2':'5.4','emit_target',cpu?'Link native CPU code':o.target==='cuda'?'CUDA CompilationDevice prepares the cubin':'Link AMD device ELF / hsaco',
          cpu?'The host object is linked into loadable native code. The JIT archive flow can also use its MC linking path.':o.target==='cuda'?'NVPTXBackend verifies an sm_ target and creates CUDA CompilationDevice for that same architecture. compileFunction converts PTX into cubin bytes through the configured compiler API.':'AMDGPU target object bytes are linked into a loadable ELF device code object. The backend adds --no-undefined; ROCm bitcode requirements depend on the selected code and configuration.',
          {targetCode:true,hostCode:cpu,code:true,representation:cpu?'host .o → loadable native code':o.target==='cuda'?'PTX → same-SM cubin bytes':'AMDGPU .o → linked ELF / hsaco',sourceNeedle:cpu?'ctx.linkObject':o.target==='cuda'?'compileFunction':'appendLinkArgs'},[cpu?'host_object':o.target==='cuda'?'nvptx_object':'amdgpu_backend','backend_select']);
        f('6.2','host_device','Retain the host code and device image in their own roles',
          cpu?'The Python process and this native operation use CPU resources. The GPU image rack stays empty for the CPU route.':'The compiled host routine retains the selected GPU image and launch logic. Python, MAX runtime and driver calls stay in the host process; the device image contains selected native kernels.',
          {hostCode:true,hostDispatch:true,representation:cpu?'host CPU process + native routine':'HostBackend control code + selected GPU image',sourceNeedle:'for (auto [target, info]'},['offload_targets','host_object','entry']);
        f('6.3','runtime_calls','Lower compiled placeholders into runtime calls',
          'LowerCompiledRoutines converts compiled-load/call placeholders to module-load and execution operations. buildJitPipeline finishes chains and outlines model callables. C1 now describes how to initialize and invoke the compiled routines.',
          {stage:'runtime',runtimeCalls:true,representation:'C1 · module-load / ExecuteGeneric / outlined callables',sourceNeedle:'void LowerCompiledRoutinesPass::runOnOperation'},['runtime_calls','jit_pipeline']);
        f('6.4','mef','Prepare CM1: runtime plan and compiled resources',
          'The compiled artifact combines prepared runtime-call resources with compiled code. The example receives a CompiledModel; it does not export a .mef disk file. G1, C1, KIR1, LLVM1 and CM1 keep distinct identities.',
          {mef:true,representation:'CM1 · prepared MEF / compiled artifact bytes'},['artifact','serialize','entry']);
        f('6.5','init','Initialize executable M1 from CM1',
          'session.init(compiled) loads/resolves runtime modules and callable handles using S1. These tensor-input examples do not require a model-weight registry. M1 becomes executable only after this initialization.',
          {initialized:true,representation:'CM1 + S1 → loaded modules / callable handles → M1',sourceRef:'entry',sourceNeedle:'session.init'},['entry','init_all','runtime_calls']);
        f('6.6','execute','Supply numerical Buffers and execute M1',
          cpu?'The application supplies CPU Buffers to model.execute. Native runtime dependencies coordinate the operation using resident host code.':'The application supplies accelerator Buffers to model.execute. Host routines submit resident device kernels with those Buffer references; native runtime dependencies coordinate the work.',
          {inputs:true,job:true,representation:cpu?'M1.execute(CPU Buffer inputs)':'host launch(device kernel handle, device Buffer inputs)',sourceNeedle:'model.execute'},['entry','execute','native_execute','driver_execute']);
        f('6.7','result_ready','Resolve O1 readiness and the source-specific read',
          id==='attention'?'Result Buffer readiness is coordinated by the runtime. This attention source prints output Buffers; it has no explicit CPU copy/parity assertion.':cpu?'The result already has CPU placement. The source reads its result through the explicit result/NumPy path; the compiler did not compute input values during preparation.':'The runtime coordinates the result Buffer; the example explicitly copies it to CPU before reading NumPy values. Returning a handle and copying numerical storage are different events.',
          {ready:true,job:false,hostRead:id!=='attention',representation:id==='attention'?'O1 · ready Buffer → print(output)':'O1 · ready Buffer → explicit CPU read',sourceRef:'driver_results'},['driver_results','ready_counts','entry']);
        f('7.1','evidence','Keep each tool boundary and evidence visible',
          'Bazel builds the dependency graph; mojo precompile prepares source packages; MAX/Mojo APIs run native compiler pipelines; LLVM libraries and target/link hooks prepare code. The source tests specify optional diagnostic dumps. This browser scene illustrates the route without claiming a completed Bazel or GPU run.',
          {stage:'evidence',evidenceReady:true,representation:'SOURCE: pinned code / optional dumps · browser teaching model',sourceNeedle:'buildStage'},['gc_pipeline','package_builder','llvm_code_gen','cross_target_example','graph_stage_dumps']);
      } else if(this.kind==='commands') {
        const target=`//max/examples/custom_ops:${this.example.task}`;
        state={...state,graph:true,verified:true,registered:true,task:this.example.task,commandTarget:target,built:false,testCommandSelected:false,pixiInstalled:false,pixiRun:false};
        add('prerequisites','Start from the repository root','The Bazel commands run in modular/, the checkout directory that contains bazelw. Bazel uses the checked-in wrapper; Pixi is a separate direct-script route.',{command:'# From the modular/ repository root\n# ./bazelw bootstraps its pinned Bazelisk when needed',route:'Choose Bazel or Pixi'},['bazel_wrapper']);
        add('bazel_build','Build this example binary',`Build the exact Python binary target: ${target}.`,{command:`# From the modular/ repository root\n./bazelw build ${target}`,route:'Bazel',built:true},['bazel_wrapper','bazel_binary']);
        add('bazel_test','Run the generated example test',`This is the test rule generated for ${target}. It runs the binary and fails on a non-zero exit.`,{command:`# From the modular/ repository root\n./bazelw test ${target}.example-test`,route:'Bazel test',testCommandSelected:true},['bazel_test_macro','bazel_binary']);
        add('pixi_setup','Install the Pixi environment once','Pixi is an alternative route for running the Python script directly. Run this from max/examples/custom_ops/; pixi install reads the environment definition and lockfile.',{command:'cd max/examples/custom_ops\npixi install  # first setup only',route:'Pixi setup',pixiInstalled:true},['pixi_dependencies','pixi_task']);
        add('pixi_run','Run the exact Python example task',`The task ${this.example.task} maps to python ${this.example.filename}.`,{command:`cd max/examples/custom_ops\npixi run ${this.example.task}`,route:'Pixi run',pixiRun:true},['pixi_task','pixi_run_docs','entry']);
      } else {
        state={...state,graph:true,verified:true,registered:true,code:true,mef:true,initialized:true};
        add('resident','Begin with the compiled model M1',`This independent scene starts after initialization. ${this.example.tiny}. Values below are a reduced numerical illustration.`,{},['entry']);
        add('arrays','Create host arrays',id==='matmul'?'The original helper makes buffers before constructing/compiling the graph. This scene isolates their execution lifecycle.':'The original Python script creates NumPy arrays and wraps them in driver Buffers.',{hostInputs:true},['entry']);
        add('upload',cpu?'Use the CPU buffers':'Explicit .to(device) supplies accelerator inputs',cpu?'CPU placement needs no GPU transfer. The target view refers to host values.':'Data is explicitly moved to the selected accelerator. The arrows are schematic, not measured transfer timings.',{inputs:true},['entry']);
        add('execute','model.execute(...) hands buffer references to native execution','M1 and compiled code stay resident. Runtime dependencies coordinate execution and result readiness.',{job:true},['entry','execute','native_execute','driver_execute']);
        if(id==='addition') {
          const values=[0,.25,.5,.75];
          add('load','Load a small group of X values','The Mojo body loads SIMD[x.dtype,width], adds one, and foreach stores the results. Displayed two-lane groups are illustrative.',{values,lane:0,operands:values.slice(0,2)},['math']);
          add('add','Native +1 computes the first group','The output lanes are numerical values. There is no per-value Python interpreter call.',{computed:values.slice(0,2).map(x=>x+1)},['math']);
          add('next','foreach advances to another group','The same compiled body handles the remaining positions.',{lane:1,operands:values.slice(2)},['math']);
          add('store','Write all four illustrated output values','The real example applies this operation to 5×10 float32 values.',{computed:values.map(x=>x+1)},['math']);
        } else if(id==='epilogue') {
          const product=matmul(A,B),flat=product.flat(),result=flat.map(x=>(x+1)*2);
          add('traffic','Compare intermediate storage for the same expression','Counters below count scalar intermediate reads/writes in the tiny 2×2 comparison. They exclude A/B reads, caches and physical bus traffic. Fusion eligibility does not prove that this example compiled into that route.',{traffic:{reads:0,writes:0,output:0},tempLive:0},['registration','entry']);
          if(state.fusion) {
            for(let i=0;i<4;i++) {
              const r=Math.floor(i/2),c=i%2;
              add('dot',`Accumulate Y[${r},${c}]`,`${A[r][0]}×${B[0][c]} + ${A[r][1]}×${B[1][c]} = ${flat[i]}. Keep the value in the illustrative accumulator.`,{cell:i,acc:[flat[i]],operands:[A[r][0],B[0][c],A[r][1],B[1][c]]},['entry','math']);
              add('epilogue','Apply +1, then ×2 before the final store',`(${flat[i]} + 1) × 2 = ${result[i]}. This eligible fusion sketch avoids a full intermediate tensor.`,{cell:i,acc:[result[i]],computed:result.slice(0,i+1),traffic:{reads:0,writes:0,output:i+1}},['entry','math']);
            }
          } else {
            add('matmul','Store the matmul intermediate','Illustrative separate route: materialize A@B in TMP1. Four intermediate values are written.',{acc:flat,temporary:flat,tempName:'TMP1',tempLive:1,traffic:{reads:0,writes:4,output:0}},['entry']);
            add('add_one','Read TMP1 and store the add-one intermediate','Four TMP1 values are read; four TMP2 values are written. This comparison uses distinct allocations.',{acc:flat.map(x=>x+1),temporary:flat.map(x=>x+1),tempName:'TMP2',tempLive:2,traffic:{reads:4,writes:8,output:0}},['math']);
            add('release_tmp1','TMP1 has no remaining consumer','Release the first illustrative intermediate after producing TMP2. Allocation reuse is a separate possible implementation policy.',{tempLive:1},['entry']);
            add('multiply','Read TMP2 and store the final output','Four more intermediate reads and four final output writes. Both routes produce the same values.',{acc:result,computed:result,tempName:'TMP2',traffic:{reads:8,writes:8,output:4}},['entry']);
            add('release_tmp2','The last intermediate is no longer live','Keep the result buffer; release TMP2. The accumulated counters retain the comparison.',{temporary:[],tempLive:0},['entry']);
          }
        } else if(id==='matmul') {
          const result=matmul(TA,TB).flat();
          if(!cpu)add('memset','enqueue_memset zeros the output first','The GPU dispatch uses a non-owning DeviceBuffer view. The tiled kernel later performs dst += dst_reg.',{computed:[0,0,0,0]},['dispatch']);
          if(!cpu&&algorithm==='tiled') {
            add('registers','Allocate shared tiles; initialize accumulators','Source uses BM×BK and BK×BN shared storage. Four drawn accumulators represent a reduced 2×2 output tile.',{acc:[0,0,0,0]},['math']);
            let acc=[0,0,0,0];
            for(let t=0;t<2;t++) {
              add('copy',`Tile ${t+1}: global → shared copies`,`${t===0?'k=0,1':'k=2,3'} in the reduced example. Shared slots are not consumable until copy completion and the block barrier.`,{tile:t,shared:true,copyReady:false,blockReady:false},['math']);
              add('commit','Commit the asynchronous copy group','Group the issued copies before waiting.',{},['math']);
              add('wait','async_copy_wait_all completes this copy group','Copy readiness alone is distinct from coordinating all cooperating block threads.',{copyReady:true},['math']);
              add('barrier','barrier() opens the block consumption gate','All cooperating block threads reach the barrier before shared-tile consumption.',{blockReady:true},['math']);
              for(let k=t*2;k<t*2+2;k++) {
                acc=acc.map((v,i)=>v+TA[Math.floor(i/2)][k]*TB[k][i%2]);
                add('mac',`Accumulate products for k=${k}`,`Per-thread register += A[row,${k}] × B[${k},col]. Shared tiles can serve several output positions.`,{k,acc},['math']);
              }
              add('reuse_barrier','A second barrier protects shared-storage reuse','Finish all readers before the next tile overwrites shared memory.',{blockReady:false,shared:false},['math']);
            }
            add('store','Add the register results to the zeroed output','C = [11, 8; 3, 6] in this full-tile numerical sketch. The original kernel uses K//BK iterations, so its example uses full tiles.',{computed:result},['math','dispatch']);
          } else {
            for(let i=0;i<4;i++) {
              const r=Math.floor(i/2),c=i%2;let acc=0;
              for(let k=0;k<4;k++){acc+=TA[r][k]*TB[k][c];add('naive',`Output [${r},${c}], k=${k}`,`${TA[r][k]} × ${TB[k][c]} adds to the accumulator → ${acc}. The naive path reads operands without the shared-tile loop.`,{cell:i,k,acc:[acc],operands:[TA[r][k],TB[k][c]]},[cpu?'math_naive_cpu':'math_naive_gpu']);}
              add('store',`Store C[${r},${c}] = ${acc}`,'Advance to the next output element in this schematic view.',{computed:result.slice(0,i+1)},[cpu?'math_naive_cpu':'math_naive_gpu']);
            }
          }
        } else {
          if(!cpu)add('gpu_launch','Specialize and enqueue the GPU kernel','The source binds BN=16, BD=8 and concrete tensor types/layouts. grid=(D/BD,N/BN)=(4,2), block=32. Each block computes one 16×8 output tile.',{launch:true},['attention_launch']);
          add('initialize','Initialize online softmax state','For this query row: m=MIN, l=0, O=[0,0]. The animation uses the mathematical −∞ limit for the finite MIN sentinel.',{m:null,l:0,O:[0,0]},['math']);
          let m=-Infinity,l=0,out=[0,0];
          for(let t=0;t<2;t++) {
            const start=t*2,scores=K.slice(start,start+2).map(row=>Q.reduce((s,q,j)=>s+q*row[j],0));
            const oldM=m,oldL=l,oldO=out.slice();m=Math.max(m,...scores);
            const alpha=Number.isFinite(oldM)?Math.exp(oldM-m):0,E=scores.map(s=>Math.exp(s-m));l=alpha*oldL+E.reduce((a,b)=>a+b,0);
            const rescaled=oldO.map(x=>x*alpha*oldL/l),weights=E.map(x=>x/l);
            out=rescaled.map((x,j)=>x+weights.reduce((s,w,i)=>s+w*V[start+i][j],0));
            if(!cpu){
              add('mma_load',`Key tile ${t+1}: load TensorCore operand fragments`,'The original GPU score matmul splits D=32 into four BK=8 slices. load_a/load_b make fragments; the tiny two-key arithmetic below is a teaching reduction.',{tile:t,local:false,mmaPhase:'load'},['attention_mma']);
              add('mma_accumulate','Multiply fragments and accumulate C','mma_op(A,B,C) produces partial scores; c_reg retains the accumulator across reduction slices.',{mmaPhase:'accumulate'},['attention_mma']);
              add('mma_store','Store shared scores, synchronize, then read the tile','store_d(out_sram,c_reg); barrier(); res.copy_from(out_sram). Block synchronization differs from later result readiness.',{mmaPhase:'store'},['attention_mma']);
            }
            add('score',`Key tile ${t+1}: Q K_tileᵀ → [${scores}]`,'Scores are local to this key tile. The original operation is unscaled and has no mask, dropout or head/batch axis.',{tile:t,scores,E:[],oldM:Number.isFinite(oldM)?oldM:null,oldL,oldO,local:true},['math']);
            add('max',`Update the running maximum: m=${fmt(m)}`,t?'The larger second-tile maximum changes the reference scale. Older contributions must be rescaled.':'The first tile establishes a finite reference maximum.',{m,alpha},['math']);
            add('exp','Compute E = exp(scores − m_new)',`E=[${E.map(fmt)}], α=exp(m_old−m_new)=${fmt(alpha)}.`,{E},['math']);
            add('sum','Update l_new = α l_old + sum(E)',`${fmt(alpha)}×${fmt(oldL)} + ${fmt(E.reduce((a,b)=>a+b,0))} = ${fmt(l)}.`,{l},['math']);
            add('rescale','Rescale the previous normalized output',`O_old × α × l_old/l_new → [${rescaled.map(fmt)}]. This preserves its probability mass on the new scale.`,{O:rescaled},['math']);
            add('weighted','Add (E/l_new) @ V_tile',`O becomes [${out.map(fmt)}]. Retain m, l and O; release the local score tile.`,{O:out,tileCount:t+1,local:false},['math']);
          }
          add('store','Write the final attention output','This row agrees with independently evaluating full softmax([1,0,3,2]) @ V in the reduced numerical example.',{computed:out},['math']);
        }
        add('ready','Result buffer readiness is separate from arithmetic','The native execution path returns driver Buffers coordinated through runtime readiness. This scene does not imply a device-wide barrier.',{ready:true,job:false},['driver_results','ready_counts']);
        if(id!=='attention')add('host',cpu?'Read the CPU result view':'Explicit result.to(CPU()) makes host values readable',cpu?'The result already has CPU placement; no accelerator-to-host transfer is depicted.':'The original script explicitly copies the result to CPU, then calls to_numpy(). Moving only a result handle would not copy its values.',{hostRead:true},['entry']);
        add('complete',id==='attention'?'The script prints the returned output':id==='addition'?'The script prints result and expected X+1':'The script checks the numerical result',id==='attention'?'The original example calls print(output). It has no explicit host copy or NumPy parity assertion. The independent tiny-math check belongs to this animation.':id==='addition'?'The source prints both arrays but contains no numerical equality assertion.':'The source uses NumPy assert_allclose. This browser animation is not an execution of the MAX example.',{},['entry']);
      }
    }
    snapshot(){return clone({...this.frames[this.index],index:this.index,total:this.frames.length,done:this.index===this.frames.length-1,lesson:this.kind});}
    seek(index){this.index=Math.max(0,Math.min(this.frames.length-1,Number(index)));return this.snapshot();}
    step(){return this.seek(this.index+1);}
    back(){return this.seek(this.index-1);}
    reset(){return this.seek(0);}
  }
  const api={ExampleModel,parameters,matmul,fmt,A,B,TA,TB,Q,K,V};
  if(typeof module==='object'&&module.exports)module.exports=api;else root.CompilerExamplesModel=api;
})(typeof window==='object'?window:globalThis);
