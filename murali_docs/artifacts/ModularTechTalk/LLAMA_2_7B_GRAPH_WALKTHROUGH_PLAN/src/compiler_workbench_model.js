/* Source-derived teaching states. This runs neither MAX nor a compiler. */
(function (root) {
  'use strict';
  const clone = x => JSON.parse(JSON.stringify(x));
  const TARGETS = {
    cpu: {name:'x86-64 CPU', triple:'x86_64-unknown-linux-gnu', arch:'x86-64', backend:'HostBackend / LLVM x86', assembly:'kernel.s', object:'kernel.o', binary:'kernel.so', tool:'LLVM MC + host linker', inspection:'llvm-objdump', driver:'CPU runtime', memory:'Host RAM'},
    cuda: {name:'NVIDIA GPU', triple:'nvptx64-nvidia-cuda', arch:'sm_90a', backend:'NVPTXBackend / LLVM NVPTX', assembly:'kernel.ptx', object:null, binary:'kernel.cubin', tool:'CUDA CompilationDevice', inspection:'nvdisasm / cuobjdump', driver:'CUDA / OS driver', memory:'Device VRAM'},
    rocm: {name:'AMD GPU', triple:'amdgcn-amd-amdhsa', arch:'gfx942', backend:'AMDGPUBackend / LLVM AMDGPU', assembly:'kernel.amdgcn', object:'kernel.o', binary:'kernel.hsaco', tool:'LLVM MC + ELF linker', inspection:'llvm-objdump', driver:'HIP / OS driver', memory:'Device VRAM'}
  };
  function timeline(initial) {
    const frames=[]; let state=clone(initial);
    return {frames, add(stage,title,detail,patch={},refs=[],kind='normal') {
      state={...state,...clone(patch)};
      frames.push({...clone(state),stage,title,detail,refs:[...refs],kind});
    }};
  }
  function graphFrames(o) {
    const t=timeline({ops:0,graph:false,sealed:false,contract:false,compileCalls:0,phase:'source'});
    t.add('source','A Python definition describes the computation','The paper is model-building code. Running its builder records symbolic operations; no tensor arithmetic has run on a GPU.',{},['graph_init','llama_trace']);
    t.add('graph','Create G1 and its input contract','Graph owns native MLIR. Dtype, shape and device are metadata; actual inference buffers arrive later.',{graph:true,phase:'graph'},['graph_init']);
    t.add('record','Record ADD1','A + B becomes a symbolic operation and def-use edges in G1.',{ops:1},['graph_init','graph_output']);
    t.add('record','Record MUL1','The multiply depends on the add. Operation tiles show the illustrative expression (A+B)×C.',{ops:2},['graph_init']);
    if(o.seal === false) {
      t.add('waiting','The output contract is still open','Use Finalize Graph.output to declare and verify the outputs. This lesson pauses before handing G1 to compilation.',{phase:'waiting'},['graph_output'],'wait');
    } else {
      t.add('signature','Graph.output declares the interface','The graph adds its output terminator, signature and dependency chains.',{contract:true,phase:'signature'},['graph_output']);
      t.add('verify','Verify and seal the symbolic graph','Graph.output verifies the MLIR contract. This construction milestone is separate from MAX dialect lowering and device-graph finalization passes.',{sealed:true,phase:'verified'},['graph_output']);
      t.add('ready','G1 is ready for the compiler','S1 retains its context. The graph is symbolic and verified; there is still no executable model.',{},['compile_async']);
    }
    return t.frames;
  }
  const LOWER_STAGES=[
    ['mo','MO','Shape / layout / symbolic work','mo.add → mo.mul\nTensor shapes + def-use edges','mo_pipeline'],
    ['map','MAP','Structured bodies / fusion','MAP structured computation\nLoop / accumulation body','map_pipeline'],
    ['mogg','MOGG','Outlined kernel calls','MOGG routines + call operands\nKernel implementation references','mogg_pipeline'],
    ['mgp','MGP','Buffers / primitives / dependencies','MGP model: slots + primitives\nAllocation and dependency plan','mgp_pipeline'],
    ['mojo','Mojo emission','Generate routine code','Generated routine representation\nbuiltin_kernels.elementwise.Add','mgp_pipeline']
  ];
  function lowerFrames(o) {
    const t=timeline({clone:false,level:-1,history:[],fused:false,discarded:false,deviceFinalized:false,phase:'input',before:'Verified G1 / MO\n(A+B)×C',after:'Waiting for compiler-owned C1'});
    t.add('input','The caller keeps G1','Compilation receives the symbolic module. Displayed stage text is a schematic view, not a recorded compiler dump.',{},['frontend']);
    t.add('clone','Clone the native module as C1','The frontend transforms its own clone. G1 remains the caller graph.',{clone:true,phase:'clone'},['frontend']);
    LOWER_STAGES.forEach((s,i)=>{
      if(i===4) t.add('dce','Remove an unused routine symbol','MGP runs symbol DCE before Mojo emission. The discarded illustrative routine is neither a checkpoint weight nor a device buffer.',{discarded:true,phase:'dce'},['mgp_pipeline']);
      const before=i ? LOWER_STAGES[i-1][3] : 'Verified symbolic Graph / MO';
      t.add(s[0],s[1]+': '+s[2],i===1 ? 'Eligible structured operations can fuse. The two-to-one grouping is an illustrative case; actual fusion depends on the optimizer and graph.' : s[2]+'. The same compiler-owned module C1 now exposes a different representation.',
        {level:i,phase:s[0],fused:i>=1 && o.fusion !== false,history:LOWER_STAGES.slice(0,i+1).map(v=>v[1]),before,after:s[3]},[s[4],'gc_pipeline']);
      if(i===3) t.add('device-finalize','Finalize device-graph structures','Within MGP, dependency work and build-graph stubs precede the device-graph finalization pass. It handles staged device-graph structures where present; this is separate from Python Graph.output and from hardware execution.',{deviceFinalized:true,phase:'device-finalize',before:'MGP: device graph dependencies / builder stubs',after:'MGP: device graph finalization pass\nContinue toward emission and kernel JIT'},['mgp_pipeline']);
    });
    t.add('ready','Routines and the MGP plan can proceed','The enclosing MAX graph pipeline schedules kernel JIT after emission, then its final runtime-call lowering. The next lesson opens the Mojo compiler.',{phase:'ready'},['gc_pipeline']);
    return t.frames;
  }
  function mojoFrames(o) {
    if(o.lab==='ownership') return ownershipFrames(o);
    const width=Number(o.width||4), values=Array.from({length:8},(_,i)=>i+1);
    const t=timeline({phase:'source',level:0,width,values,results:[],checked:false,concrete:false,bundles:0,owner:'caller',borrow:null,blocked:false});
    t.add('source','Readable syntax, explicit types','The source card represents a typed Mojo routine. See the unchanged Add implementation through its source link.',{},['mojo_add','mojo_compiler']);
    t.add('lit','The parser emits LIT MLIR','Mojo uses source-level LIT IR, rather than requiring a separate traditional AST-to-IR pipeline.',{level:1,phase:'lit'},['mojo_compiler']);
    t.add('check','Check lifetimes and semantic control flow','The compiler checks references and lowers semantic control flow before lowering LIT.',{level:2,phase:'checked',checked:true},['mojo_semantics']);
    t.add('kgen','Keep the routine parametric in KGEN','Types and compile-time parameters remain available to pre-elaboration optimization.',{level:3,phase:'kgen'},['mojo_compiler']);
    t.add('specialize','Instantiate WIDTH = '+width,'Elaboration resolves compile-time parameters into concrete functions/types. WIDTH here is an illustrative CPU SIMD specialization.',{level:4,phase:'concrete',concrete:true},['mojo_elaboration','mojo_add']);
    t.add('llvm','Lower concrete operations toward LLVM','Argument/calling conventions and low-level operations lower before LLVM dialect translation and native target codegen.',{level:5,phase:'llvm'},['mojo_llvm','llvm_translation']);
    t.add('lanes','Map eight values into '+(8/width)+' instruction bundle'+(width===8?'':'s'),'The lane map explains a CPU SIMD add using width '+width+'. It is not a measured instruction count or speedup; the compiler may further transform it.',{phase:'lanes',bundles:8/width,results:values.map(v=>v+10)},['mojo_add']);
    t.add('ready','Typed abstractions become native operations','Python-like syntax does not imply Python object/interpreter dispatch inside every native arithmetic operation. Concrete types, parameters and ownership allow native code generation.',{},['mojo_compiler']);
    return t.frames;
  }
  function ownershipFrames(o) {
    const t=timeline({phase:'owner',lab:'ownership',owner:'caller',borrow:null,blocked:false,moved:false,values:[1,2,3,4],results:[],checked:false});
    t.add('owner','B1 has one owner and four cells','B1 is an illustrative allocation. Handle/reference cards are separate from its stored values.',{},['mojo_semantics']);
    t.add('borrow','An immutable borrow refers to B1','The dashed edge creates access to the same allocation, without copying its four cells.',{phase:'borrow',borrow:'imm'},['mojo_semantics']);
    t.add('conflict','A simultaneous mutation would conflict','The attempted mutable access conflicts with the live immutable borrow in this teaching case. The compile-time lifetime gate blocks it before execution.',{phase:'blocked',blocked:true},['mojo_semantics'],'error');
    if(!o.released) return t.frames;
    t.add('release','End the immutable borrow','Release borrow is a teaching intervention: it changes the program lifetime before recompilation, rather than bypassing the compiler checks.',{phase:'release',borrow:null,blocked:false},['mojo_semantics']);
    t.add('exclusive','Grant exclusive mutable access','The mutable reference now accesses the same B1 cells. The access does not allocate another buffer.',{phase:'exclusive',borrow:'mut',checked:true},['mojo_semantics']);
    t.add('write','One cell changes in place','The example changes B1[0]. Ownership and aliasing controls determine which access is legal.',{phase:'write',values:[11,2,3,4]},['mojo_semantics']);
    t.add('move','Move ownership, keep B1','The owner handle transfers to the callee. The cells and allocation identity stay B1; moving a handle does not inherently copy its data.',{phase:'move',borrow:null,owner:'callee',moved:true},['mojo_semantics']);
    return t.frames;
  }
  function targetFrames(o) {
    const target=TARGETS[o.target||'cuda'];
    const t=timeline({target:clone(target),targetKey:o.target||'cuda',phase:'mlir',llvm:false,selected:false,assembly:false,object:false,binary:false,inspect:false,error:null,mode:o.emission||'binary',assembler:o.assembler||'api'});
    t.add('mlir','LLVM dialect is still MLIR','Lowered MLIR operations and an LLVM IR module are different representations.',{},['mojo_llvm','llvm_translation']);
    t.add('select','Select '+target.backend,'Backend lookup uses the target triple; arch/features refine codegen. The illustrated compiler host is a CPU. GPU target selection is not GPU execution.',{phase:'select',selected:true},['backend_select','cross_target']);
    t.add('translate','Translate into LLVM IR','Module translation produces an llvm::Module. The .ll suffix represents an optional textual dump of it.',{phase:'llvm',llvm:true},['llvm_translation']);
    if(o.target==='cuda' && o.invalidArch) {
      t.add('error','Reject a non-SM NVIDIA target CPU','NVPTXBackend requires the TargetMachine CPU to start with sm_ for PTX-to-cubin compilation.',{phase:'error',assembly:true,error:'Target CPU must start with sm_'},['nvptx_object'],'error');
      return t.frames;
    }
    const assemblyOnly=o.emission==='assembly';
    if(o.target==='cuda' || assemblyOnly) {
      t.add('assembly','Emit '+target.assembly,o.target==='cuda' ? 'LLVM NVPTX emits PTX, a textual virtual ISA representation. PTX is not the final SASS machine-code payload.' : 'Assembly emission is an alternative output of target codegen. It stops before producing the linked binary in this route.',{phase:'assembly',assembly:true},[o.target==='cuda'?'nvptx_object':'host_assembly']);
    }
    if(assemblyOnly) {
      t.add('ready','Text emission stops here','A listing can be inspected or assembled through a separate toolchain route. This lesson selected text output, so there is no executable artifact to load.',{phase:'text'},['host_assembly']);
      return t.frames;
    }
    if(o.target==='cuda') {
      t.add('assemble',o.assembler==='override'?'Use the configured CUDA compiler path':'Use the CUDA compilation API','NVPTXBackend creates a CUDA CompilationDevice for the same SM target as the PTX. On Linux, the default uses NVPTXCompiler when no override path is set.',{phase:'assemble'},['nvptx_object']);
      if(o.missingToolkit) {
        t.add('error','Compilation device creation failed','A toolchain/CompilationDevice failure returns an error. Cross-compilation avoids needing a physical GPU, but still requires the relevant compilation support.',{phase:'error',error:'CUDA compilation support unavailable'},['nvptx_object','cross_target'],'error');
        return t.frames;
      }
    } else {
      t.add('object','Emit relocatable '+target.object,'LLVM MC codegen can emit object bytes directly; it does not have to save a textual assembly file first.',{phase:'object',object:true},['host_object']);
      t.add('link',o.target==='rocm'?'Link an AMDGPU ELF code object':'Link the native host object','The source HostBackend links its object; AMDGPU inherits this emission path with device-library/link customizations.',{phase:'link'},[o.target==='rocm'?'amdgpu_backend':'host_object']);
    }
    t.add('binary','Produce '+target.binary,'The binary cartridge contains target code. An actual compiler may keep the bytes in memory; filenames are representative exported artifacts.',{phase:'binary',binary:true},[o.target==='cuda'?'nvptx_object':o.target==='rocm'?'amdgpu_backend':'host_object']);
    if(o.inspect) t.add('inspect','Disassembly is an inspection branch',target.inspection+' can inspect this target binary. The listing decodes existing machine-code bytes; the original binary remains unchanged. This is an illustrative read-only view, not a tool execution.',{phase:'inspect',inspect:true},['compiler_tools']);
    return t.frames;
  }
  function packageFrames(o) {
    const target=TARGETS[o.target||'cuda'];
    const t=timeline({phase:'parts',target:clone(target),parts:0,mef:false,exported:false,compiled:false,weights:0,initialized:false,codeLoaded:false,error:null});
    t.add('parts','Start from compiled routines and a runtime plan','This lesson assumes successful compatible target compilation. Code, resources and the MGP dependency/primitive plan feed MEF preparation; checkpoint weight bindings remain separate.',{},['serialize','gc_pipeline']);
    t.add('resources','Clone the module and collect resources','The converter retains the input module, clones it and obtains its resource blobs before running device-primitive preparation. The cutaway is a conceptual grouping, not a byte-accurate MEF section layout.',{phase:'collect',parts:1},['serialize']);
    t.add('primitives','Prepare device-appropriate primitives','The preparation pass adapts device-generic primitives. Translation and serialization then produce MEF bytes from the prepared module and its resources.',{phase:'pack',parts:3},['serialize']);
    t.add('mef','Create the MEF byte buffer','MEF is a target-bearing executable-format artifact. It is separate from the initialized engine Model and the Python Graph.',{phase:'mef',mef:true,compiled:true,parts:4},['serialize','artifact']);
    if(o.export) t.add('export','Export the MEF to a file','export_mef writes artifact bytes. Showing model.mef here means an explicit export, rather than assuming every internal IR stage is a disk file.',{phase:'export',exported:true},['export']);
    t.add('weights','Supply W_a','The illustration starts with one registry binding. Compiling the graph did not supply the model checkpoint values.',{phase:'bind',weights:1},['init_all']);
    if(o.missingWeight && !o.repaired) {
      t.add('blocked','W_b is missing: hold initialization','This named-binding gate prevents the illustrated model from becoming executable. Supply binding is a teaching action that changes the prerequisite.',{phase:'blocked',error:'W_b missing'},['init_all'],'error');
      return t.frames;
    }
    t.add('weights','Supply W_b','Both illustrated bindings are now available. The actual model has its own complete weight registry.',{phase:'bind',weights:2,error:null},['init_all']);
    t.add('init','Initialize M1 from artifact and bindings','init_all binds registries and initializes runtime models. The native model handle retains its session context.',{phase:'init',initialized:true,codeLoaded:true},['init_all','native_init','setup']);
    t.add('ready','M1 is executable and S1 is retained','The prepared artifact, initialized Model and resident runtime resources are distinct. The next lesson starts from this ready state.',{phase:'ready'},['artifact','setup']);
    return t.frames;
  }
  function runtimeFrames(o) {
    const target=TARGETS[o.target||'cuda'], cpu=o.target==='cpu', resident=o.resident || cpu;
    const calls=Number(o.calls||1);
    const t=timeline({phase:'ready',target:clone(target),cpu,resident,call:0,inputs:[],deviceInputs:[],results:[],hostResults:[],queued:false,running:false,ready:false,codeResident:true,weightsResident:true,model:'M1',session:'S1',copies:0,dispatches:0,history:[]});
    t.add('ready','M1, code and weights are already resident','This lesson starts after initialization. The same code/model/context serve each dispatch; no code file is uploaded for every input.',{},['setup','runtime_execute']);
    let copies=0;
    for(let c=1;c<=calls;c++) {
      const values=[c,2+c,4+c,6+c], results=values.map(v=>(v+10)*2);
      t.add('inputs','Call '+c+': prepare B'+c,'Input placement is part of the model contract. The host-to-device copy in this example is explicit; resident inputs avoid that copy.',{phase:'inputs',call:c,inputs:values,deviceInputs:resident?values:[],results:[],hostResults:[],queued:false,running:false,ready:false},['native_execute']);
      if(!resident) t.add('copy','Copy input values into D'+c,'Only the input data crosses the explicit upload route. M1, code and weights remain resident. The depicted transfer is schematic.',{phase:'copy',deviceInputs:values,copies:++copies},['driver_execute']);
      t.add('queue','Queue a dispatch that refers to code and buffers','The descriptor carries references to resident code and buffers. Queue slots illustrate causality; they are not a measured driver command-ring layout.',{phase:'queue',queued:true,dispatches:c},['driver_execute','runtime_execute']);
      t.add('driver',cpu?'Native CPU work becomes runnable':'Submit through the user/device driver',cpu?'CPU runtime execution uses ready operands. No GPU driver is introduced on this host route.':'Native MAX/user-driver code crosses the device driver boundary. Device-specific scheduling/overlap is BOUNDARY here.',{phase:'driver'},['driver_execute']);
      t.add('execute','Execute the ready work','The tile values show the illustrative expression (A+10)×2. The source runtime schedules primitives when AsyncValue dependencies become ready; these are not measured GPU kernel counts or timings.',{phase:'execute',queued:false,running:true},['ready_counts','async_users']);
      t.add('results','Result buffers become ready','Result extraction awaits runtime values. Readiness does not inherently imply a device-wide barrier or a copy to host memory.',{phase:'results',running:false,ready:true,results,history:Array.from({length:c},(_,i)=>'Call '+(i+1)+' complete')},['driver_results']);
      if(o.copyResult && !cpu) t.add('host','Explicitly copy the output to host','This teaching branch asks for a host result copy. It is separate from obtaining a ready device result and is not automatic for every Model.execute.',{phase:'host',hostResults:results,copies:++copies},['driver_results']);
      t.add('complete','Call '+c+' completed; reuse M1 and S1','The next call supplies new input/result buffers. The initialized model, code and weights keep their identities.',{phase:'complete'},['execute','runtime_execute']);
    }
    return t.frames;
  }
  const BUILDERS={graph:graphFrames,lower:lowerFrames,mojo:mojoFrames,target:targetFrames,package:packageFrames,runtime:runtimeFrames};
  class WorkbenchModel {
    constructor(kind,options={}) { this.kind=kind; this.options={target:'cuda',...clone(options)}; this.frames=BUILDERS[kind](this.options); this.index=0; }
    snapshot() { return {...clone(this.frames[this.index]),index:this.index,total:this.frames.length,done:this.index===this.frames.length-1,options:clone(this.options),kindName:this.kind}; }
    seek(i) { this.index=Math.max(0,Math.min(this.frames.length-1,Number(i))); return this.snapshot(); }
    step() { return this.seek(this.index+1); }
    back() { return this.seek(this.index-1); }
    reset() { return this.seek(0); }
    configure(patch) { this.options={...this.options,released:false,repaired:false,inspect:false,calls:1,...clone(patch)};this.frames=BUILDERS[this.kind](this.options);this.index=0;return this.snapshot(); }
    intervene(action) {
      const s=this.snapshot(), old=this.index;
      if(action==='seal' && this.kind==='graph' && s.graph && !s.sealed) this.options.seal=true;
      else if(action==='release' && this.kind==='mojo' && s.blocked) this.options.released=true;
      else if(action==='inspect' && this.kind==='target' && s.binary && !s.inspect) this.options.inspect=true;
      else if(action==='repair' && this.kind==='package' && s.error) this.options.repaired=true;
      else if(action==='again' && this.kind==='runtime' && s.done) this.options.calls=Number(this.options.calls||1)+1;
      else return s;
      this.frames=BUILDERS[this.kind](this.options); this.index=Math.min(old+1,this.frames.length-1);return this.snapshot();
    }
  }
  root.CompilerWorkbenchModel={WorkbenchModel,TARGETS,LOWER_STAGES};
  if(typeof module!=='undefined' && module.exports) module.exports=root.CompilerWorkbenchModel;
})(typeof window==='undefined'?globalThis:window);
