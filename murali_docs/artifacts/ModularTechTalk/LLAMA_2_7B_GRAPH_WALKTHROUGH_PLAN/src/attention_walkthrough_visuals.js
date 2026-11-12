/* Original SVG diagrams: persistent actors, changing ownership, IR and arithmetic. */
(function(root){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const palette={pink:['#982258','#f5e7ee'],green:['#076045','#e2eee6'],blue:['#315b85','#e5edf3'],orange:['#805421','#f2ebdf'],gray:['#596b60','#edf0ed']};
  const M=root.AttentionModel;
  function n(id,x,y,w,h,title,rows=[],kind='card',tone='green',extra={}){return {id,x,y,w,h,title,rows,kind,tone,...extra};}
  function e(from,to,label='',extra={}){return {id:from+'-'+to,from,to,label,...extra};}
  const markIndex=(mark,marks)=>Math.max(0,marks.indexOf(mark));
  function standard(a,b,c){return [{x:16,y:22,w:202,h:464,title:a},{x:234,y:22,w:514,h:464,title:b},{x:764,y:22,w:260,h:464,title:c}];}
  function scene(lanes,nodes,edges=[],footer='SOURCE · schematic state; identities are teaching labels'){return {lanes,nodes,edges,footer};}
  const buildMarks=['command','bazel','targets','analysis','libraries','launcher','built','run','main','package-input','precompile','package-ready','test'];
  function build(f){
    const p=['package-parse','package-checks','package-write'].includes(f.mark)?10:markIndex(f.mark,buildMarks),nodes=[n('request',30,62,174,90,'Requested target',['fused_attention','Bazel label'],'file','pink')],edges=[];
    if(p>=1)nodes.push(n('bazel',250,62,228,95,'Bazel / Bazelisk',['loads BUILD.bazel',p>=3?'analyzed action graph':'loading target'],'card','blue'));
    if(p>=2){nodes.push(n('targets',250,181,228,150,'Target dependency graph',[':fused_attention → py_binary','├─ fused_attention.py','├─ kernel_sources (data)','└─ layout/extensibility/std'],'card','blue'));edges.push(e('request','bazel','resolve'),e('bazel','targets','expand'));}
    if(p>=4)nodes.push(n('libraries',250,357,228,95,'Shared package outputs',['layout / std / extensibility','.mojoc library dependencies'],'file','green'));
    if(p>=5){nodes.push(n('launcher',p<7?520:30,p<7?62:193,p<7?212:174,128,'L1 · Python launcher',['fused_attention.py','kernels/*.mojo (raw)','runtime paths / libraries'],'file','green'));edges.push(e('targets','launcher','assemble'));}
    if(p>=7){nodes.push(n('python',520,62,212,125,'Python process',[p>=8?'__main__ → main()':'import modules',p>=8?'GPU0; N=D=32':'NumPy + MAX','BN=16, BD=8'],'card','pink'));edges.push(e('launcher','python','run'));}
    if(p>=9){nodes.push(n('raw',30,354,174,105,'kernels/',['__init__.mojo','fused_attention.mojo'],'file','pink'));nodes.push(n('graph-request',520,211,212,85,'Graph constructor',['load_paths(kernels/)','P1 requested'],'card','pink'));edges.push(e('raw','graph-request','package path'));}
    if(p===10){nodes.push(n('mojo-child',780,83,228,205,'Mojo compiler process',['native executable','precompile kernels/',f.mark==='package-parse'?'LIT → lit.package':f.mark==='package-checks'?'semantic / lifetime checks':f.mark==='package-write'?'serialize MLIR bytecode':'parse → package IR','private staging output'],'process','blue'));edges.push(e('graph-request','mojo-child','spawn + wait'));}
    if(p>=11){nodes.push(n('P1',780,83,228,155,'P1 · .mojoc',['lit.package','parametric declarations','and implementation bodies','MLIR bytecode'],'file','green'));nodes.push(n('published',780,279,228,103,'Atomic publication',['.staging.<pid> → final','same package basename'],'card','green'));edges.push(e('graph-request','P1','prepare'),e('published','P1','publish'));}
    if(p===6)nodes.push(n('build-end',780,98,228,115,'Build complete',['launcher / runfiles ready','main() has not run','P1 not prepared yet'],'gate','green'));
    if(p===12){nodes.push(n('test',520,348,212,105,'Sibling test launcher',['same executable / runfiles','exit status → test result','manual target in BUILD'],'card','orange'));edges.push(e('test','python','run again'));}
    return scene(standard('INPUTS / PROGRAM','BUILD AND HOST EXECUTION','PACKAGE RESULT'),nodes,edges);
  }
  const graphMarks=['arguments','context','dimensions','module','graph-op','aliases','devices','chains','package','enter','custom','attributes','verify','output'];
  function graph(f){
    const p=markIndex(f.mark,graphMarks),nodes=[n('types',30,62,174,112,'Python input types',['TQ = TK = TV','f32[32,32]','DeviceRef(gpu,0)'],'card','pink')],edges=[];
    if(p>=1)nodes.push(n('context',30,198,174,90,'Active context',['thread.current → CTX1','compiler IR owner'],'card','blue'));
    if(p>=2)nodes.push(n('params',30,316,174,106,'G1 fields',['_params = {}','weights = {}',p>=5?'_graph_body → B0':'body: pending'],'card','pink'));
    if(p>=3)nodes.push(n('module',250,62,482,365,'builtin.module',[],'module','blue',{phase:p}));
    if(p>=5)edges.push(e('params','module','aliases same IR'));
    if(p>=8){nodes.push(n('P1',780,62,228,135,'P1 · library analysis',['package path retained','FusedAttention.execute','dtype / rank / BN / BD'],'file','green'));edges.push(e('P1','module','declarations',{reverse:true}));}
    if(p>=10)nodes.push(n('contract',780,226,228,144,p>=12?'Contract verified':'Bind custom contract',['3 inputs : T','1 output : T',p>=11?'BD=8, BN=16; gpu:0':'name → StringAttr',p>=12?'result = TensorValue(%o)':'verification pending'],'card',p>=12?'green':'pink'));
    if(p>=13)nodes.push(n('G1',780,394,228,67,'G1 retained',['signature (T,T,T) → T'],'gate','green'));
    if(p>=4)edges.push(e('types','module','convert types'));
    return scene(standard('PYTHON DESCRIPTIONS','NATIVE IR · HOST MEMORY','PACKAGE / VERIFIED RESULT'),nodes,edges);
  }
  function session(f){
    const p=markIndex(f.mark,['argument','normalize','compile-config','runtime-config','max-context','cpu','diagnostics','ready']);
    const nodes=[n('devices',30,66,174,150,'Device arguments',['requested: GPU0',p>=1?'dedup → GPU0':'',p>=1?'append → CPU0':''],'slots','pink')],edges=[];
    if(p>=2){nodes.push(n('S1',250,62,482,348,'S1._impl · Native session',[],'session','blue',{phase:p}));edges.push(e('devices','S1','retain refs'));}
    if(p>=4){nodes.push(n('MAXContext',780,62,228,126,'MAXContext1',['runtime service store','shared/native reference','not an MLIR context'],'card','green'));edges.push(e('S1','MAXContext','runtimeContext'));}
    if(p>=5){nodes.push(n('CPUDevice',780,224,228,95,'CPUDevice1',['AsyncRT scheduling','worker thread context'],'card','blue'));nodes.push(n('GPU0',780,354,228,95,'GPU0 device context',['streams / allocations','vendor API handles'],'chip','green'));edges.push(e('MAXContext','CPUDevice','service'));
    }
    if(p>=7)nodes.push(n('ready',250,431,482,38,'S1 ready: compiler policy + runtime services',[],'gate','green'));
    return scene(standard('SELECTED CAPABILITIES','CONFIGURATION / OWNERSHIP','NATIVE SERVICES'),nodes,edges);
  }
  const compileMarks=['gather','decls','clone','MO','MAP','MAP-opt','MOGG','MOGG-opt','MGP','emit','cache','specialize','llvm','device-code','host-code','calls','MEF','C1'];
  function compile(f,opts){
    const p=f.mark==='cache-hit'?14:markIndex(f.mark,compileMarks),vendor=opts.vendor==='amd'?'AMD':'NVIDIA';
    const nodes=[n('G1',30,62,174,146,'G1 · retained graph',['3 inputs → 1 output','BD=8, BN=16','f32[32,32] @ gpu:0','kept by Python'],'file','pink'),n('P1',30,242,174,117,'P1 · package',['declarations + bodies','parametric source IR','no concrete target'],'file','green')],edges=[];
    if(p<2)nodes.push(n('request',250,62,482,242,'Compile request',['S1 compile config + G1 module','P1 paths; expected_weights={}',...f.code.split('\n')],'program','blue'));
    else{nodes.push(n('IR1',250,62,482,252,p>=16?'Serialized runtime program':p>=9?'Native routine / runtime plan':`IR1 · ${p>=8?'MGP':p>=6?'MOGG':p>=4?'MAP':'MO'}`,f.code.split('\n'),'program','blue',{phase:p}));edges.push(e('G1','IR1',p===2?'clone':'G1 retained'));}
    if(p>=1)edges.push(e('P1',p<2?'request':'IR1',p>=9?'bodies / imports':'declarations'));
    if(p>=11&&p<15&&f.mark!=='cache-hit'){nodes.push(n('host-IR',250,346,225,112,'Host target',['CPU native routine',p>=12?'LLVM IR':'specialized KGEN','adapters + enqueue'],'program','pink'));nodes.push(n('device-IR',495,346,237,112,`${vendor} target`,[opts.vendor==='amd'?'amdgcn + gfx target':'nvptx + SM target',p>=13?(opts.vendor==='amd'?'HSA ELF D1':'PTX → cubin D1'):'device kernel IR'],'program','green'));}
    if(p>=10)nodes.push(n('cache',780,62,228,75,f.mark==='cache-hit'?'Native code cache · HIT':'Native code cache',['bytes + symbol names',f.mark==='cache-hit'?'specialization skipped':'cold route illustrated'],'gate',f.mark==='cache-hit'?'green':'orange'));
    if(p>=13)nodes.push(n('D1',780,168,228,104,`D1 · ${opts.vendor==='amd'?'HSA ELF':'cubin'}`,['device image bytes','kernel code + metadata'],'file','green'));
    if(p>=14)nodes.push(n('A1',780,300,228,96,'A1 · host code resource',['object bytes / symbols','unlinked native code'],'file','pink'));
    if(p>=15)nodes.push(n('plan',250,344,482,112,'Runtime calls after lowering',['init → ModuleLoad(A1, symbols)','execute → ExecuteGeneric(slot,module,index)','init / execute ordering retained'],'program','blue'));
    if(p>=17){nodes.push(n('C1',780,416,228,54,'C1 · compiled artifact',['MEF resources + graph names'],'gate','green'));edges.push(e('IR1','C1','serialize'));}
    return scene(standard('INPUTS REMAIN AVAILABLE','IN-PROCESS COMPILER TRANSFORMS','PREPARED BYTE RESOURCES'),nodes,edges);
  }
  function init(f){
    const p=markIndex(f.mark,['inputs','task','devices','invoke','slot','register','link','debug','model','ready','M1']);
    const nodes=[n('C1',30,62,174,133,'C1 · artifact',['init / execute plan','A1 + D1 resources','expected_weights={}'],'file','green'),n('S1',30,232,174,111,'S1 · runtime',['GPU0, CPUDevice1','native services'],'card','pink')],edges=[];
    if(p>=1){nodes.push(n('worker',250,62,482,100,'CPU runtime worker',[p>=3?'running initFn plan':'setup task queued','current CPUDevice1 in thread-local state'],'queue','blue'));edges.push(e('C1','worker','setup'));}
    if(p>=2)edges.push(e('S1','worker','contexts'));
    if(p>=3)nodes.push(n('slots',250,168,228,273,'MGP init-result slots',['chain: '+(p>=9?'ready':'pending'),'state context: '+(p>=4?'allocated':'pending'),'weights: {}',p>=4?'modules[0] → CKC1':'modules[0]: empty',p>=6?'kernels[0] → H1':'kernel address: pending'],'slots','blue'));
    if(p>=5)nodes.push(n('linker',504,168,228,125,'Host linker',[p>=6?'H1 resolved':'A1 registered / symbols pending',p===7?'optional: A1→.o→.so':'ordinary route: ORC / JITLink'],'card',p>=6?'green':'orange'));
    if(p>=6){nodes.push(n('memory',780,62,228,153,'Host code memory',['executable code → H1','read-only constants','writable globals if present'],'memory','pink'));edges.push(e('linker','memory','link / lookup'));}
    if(p>=8){nodes.push(n('M1',780,250,228,158,p>=10?'M1 · initialized model':'ExecutableModel owner',['model + cpuDevice','chain / MGP state / weights','extra init resources',p>=9?'setup ready, no error':'setup result: pending'],'card',p>=9?'green':'orange'));edges.push(e('slots','M1','retain refs'));}
    nodes.push(n('KD1',780,431,228,45,'GPU function load site: later',[],'gate','gray'));
    return scene(standard('COMPILED INPUT / SESSION','CPU SETUP AND RESOURCE SLOTS','RESIDENT CODE / MODEL'),nodes,edges);
  }
  const executeMarks=['host','Q-copy','K-copy','V-copy','inputs','invoke','host-abi','output-alloc','device-load','launch-pack','launch','tile0','tile1','tile2','tile3','ready','return','readback'];
  function execute(f,opts){
    const p=markIndex(f.mark,executeMarks),nodes=[n('M1',30,57,180,68,'M1 · resident model',['H1 + runtime services'],'card','pink'),n('host-inputs',30,151,180,192,'Host inputs',[],'host-matrices','gray',{copies:Math.min(p,3)})],edges=[];
    nodes.push(n('gpu-buffers',252,57,302,248,'GPU0 · tensor buffers',[],'gpu-buffers','green',{phase:p}));
    if(p>0&&p<=3){const name=['','Q','K','V'][p];nodes.push(n('transfer',174,365,262,67,`${name} payload → GPU0`,['4096 bytes, separate destination'],'packet','green'));edges.push(e('host-inputs','gpu-buffers',`${name}: copy 4096 B`));}
    if(p>=4)nodes.push(n('runtime',584,57,428,126,'Native host execution',[p>=5?'execute(chain,state,weights,extras,Q,K,V)':'validate and wrap device storage',p>=6?'H1(void**) ← addresses of AsyncValue slots':'input Tensor AsyncValues',p>=7?'O allocation plan obtains 4096-byte storage':'output slot pending'],'program','blue'));
    if(p>=8)nodes.push(n('KD1',584,209,205,102,'KD1 · GPU function',[opts.vendor==='amd'?'HSA ELF → HIP handles':'cubin → CUDA handles','code cached separately'],'chip','green'));
    if(p>=9)nodes.push(n('args',807,209,205,102,'Launch arguments',['encoded host slots','grid=(4,2), block=32','stream + KD1'],'slots','pink'));
    if(p>=10){nodes.push(n('gpu-work',252,335,508,133,p>=15?'Dispatch complete / results observed':'GPU · output tile ownership',[],'tile-grid','green',{keyTile:Math.min(3,Math.max(-1,p-11)),phase:p}));edges.push(e('args','gpu-work',opts.vendor==='amd'?'hipModuleLaunchKernel':'cuLaunchKernelEx'));}
    if(p>=11&&p<=14)nodes.push(n('math',30,369,180,103,'Per-block recurrence',[`key tile ${p-11}: 8 rows`,'retain m / l / O_i','reuse S / E / PV'],'card','pink'));
    if(p>=15)nodes.push(n('result',787,348,225,110,p>=16?'Python output = [O1]':'Native results ready',['O1: f32[32,32] @ GPU0','internal chain removed',p>=17?'optional CPU readback':'no automatic CPU copy'],'file','green'));
    if(p>=16)edges.push(e('gpu-buffers','result','wrap same O storage'));
    if(p>=17){nodes.push(n('host-read',30,374,180,90,'Host-readable O',['4096-byte readback','Buffer sync / DLPack'],'file','green'));edges.push(e('result','host-read','optional readback',{bottom:true}));}
    return scene([{x:16,y:22,w:210,h:464,title:'HOST PROGRAM / INPUTS'},{x:238,y:22,w:328,h:295,title:'DEVICE DATA ALLOCATIONS'},{x:570,y:22,w:454,h:295,title:'HOST RUNTIME / DRIVER CALL'},{x:238,y:325,w:786,h:161,title:'GPU DISPATCH / RESULT'}],nodes,edges,'SOURCE · geometry exact for this example; motion and scheduling schematic');
  }
  function packageScene(f){
    const p=markIndex(f.mark,['package-input','path','staging','executable','precompile','package-ir','checks','bytecode','package-ready','attach']);
    const nodes=[n('source',30,68,174,145,'kernels/ package',['__init__.mojo','fused_attention.mojo','other package modules'],'file','pink')],edges=[];
    if(p>=1)nodes.push(n('final-path',30,264,174,135,'Destination identity',['per-user temp directory','path-derived hash','same final basename'],'file','gray'));
    if(p>=2)nodes.push(n('staging',250,298,482,131,'Private staging output',['.staging.<pid>/mojo_pkg_<hash>.mojoc',p>=7?'complete MLIR bytecode bytes':'not yet published'],'file',p>=7?'green':'gray'));
    if(p>=3)nodes.push(n('compiler',250,68,482,184,'Separate Mojo executable process',['MODULAR_MOJO_MAX_DRIVER_PATH or PATH',p>=4?'precompile kernels/ -o <staging path>':'resolved native compiler driver',p>=6?'semantic/lifetime checks completed':p>=5?'lit.package: declarations + bodies':'package parser / frontend',p>=8?'child process exited successfully':'host Python waits for process success'],'process','blue'));
    if(p>=4)edges.push(e('source','compiler','source files'),e('compiler','staging','write'));
    if(p>=8){nodes.push(n('P1',780,68,228,154,'Published P1',['MLIR bytecode package','complete file at final path','atomic os.replace'],'file','green'));edges.push(e('staging','P1','atomic rename'));}
    if(p>=9){nodes.push(n('graph-library',780,276,228,136,'G1 library binding',['KernelLibrary paths=[P1]','GraphOp path attribute','later compile consumes P1'],'card','pink'));edges.push(e('P1','graph-library','record path'));}
    return scene(standard('INPUT / DESTINATION NAME','CHILD PROCESS AND STAGING','PUBLICATION / GRAPH LINK'),nodes,edges);
  }
  function contextScene(f){
    const p=markIndex(f.mark,['global','main','worker','enter','exit','executor']);
    const nodes=[n('CTX1',250,78,482,146,'CTX1 · module-global MLIR context',['dialect / type / attribute infrastructure','_DEFAULT_MLIR_CONTEXT refers to CTX1','default_mlir_context() returns the same object'],'card','blue'),n('main-thread',30,78,174,174,'Main Python thread',['current-context slot',p>=1?'CTX1':'not entered yet','ordinary example caller'],'slots','pink')],edges=[];
    if(p>=1)edges.push(e('main-thread','CTX1','current ref'));
    if(p>=2){nodes.push(n('worker-thread',780,78,228,174,'Illustrative worker thread',['current-context slot',p===3?'CTX1':'None',p===3?'Graph constructor running':p>=4?'scope exited':'guard checks for None'],'slots',p===3?'green':'gray'));if(p===3)edges.push(e('CTX1','worker-thread','scoped reference'));}
    if(p>=2)nodes.push(n('decorator',250,296,482,139,'@in_default_mlir_context',['wrapper enters ensure_default_mlir_context()',p>=3?'enter default only when current is None':'inspect current context on calling thread','reuse an already active context'],'program','green'));
    if(p>=5){nodes.push(n('executor',30,330,174,104,'Optional executor',['submit guarded task','not another context'],'card','orange'));edges.push(e('executor','decorator','task wrapper'));}
    return scene(standard('MAIN THREAD','SHARED OBJECT / PYTHON GUARD','BACKGROUND THREAD'),nodes,edges,'SOURCE · worker case is a teaching comparison; this script calls Graph in main()');
  }
  function hostLink(f){
    const p=markIndex(f.mark,['receive','hash','namespace','interface','define','lookup','allocate','relocate','table','debug']);
    const nodes=[n('A1',30,66,174,180,'A1 · host object',['code bytes','constant/data bytes','symbols / relocations','generated host entry name'],'file','pink')],edges=[];
    if(p>=1)nodes.push(n('hash',30,300,174,120,'Owned resource',['copy of A1 bytes','key = hash(A1)'],'card','green'));
    if(p>=2){nodes.push(n('namespace',250,66,482,329,'ORC JITDylib · symbol namespace',[],'symbol-table','blue',{phase:p}));edges.push(e('A1','namespace','register object'));}
    if(p>=6){nodes.push(n('memory',780,66,228,212,'Host memory allocations',['code: '+(p>=7?'H1 executable':'allocated, pending fixups'),'constants: read-only','globals: writable if present','not Q/K/V tensor storage'],'memory','pink'));edges.push(e('namespace','memory',p>=7?'resolve / finalize':'allocate'));}
    if(p>=8){nodes.push(n('table',780,324,228,104,'CKC1 callable table',['kernels[0] = H1','symbolTable[0] = name'],'slots','green'));edges.push(e('memory','table','resolved address'));}
    if(p>=9)nodes.push(n('debug-path',250,418,482,57,'Debug branch: A1 → temp.o → linker → temp.so',['dynamic-library load then symbol lookup'],'gate','orange'));
    return scene(standard('RETAINED BYTE BUFFER','REGISTRATION / MATERIALIZATION','MAPPINGS / CALLABLE ADDRESS'),nodes,edges,'SOURCE · symbol names/addresses illustrative; LLVM performs relocation/finalization');
  }
  function deviceLoad(f,opts){
    const p=markIndex(f.mark,['descriptor','binding','cache','module','function','attributes','retain','hit']),amd=opts.vendor==='amd';
    const nodes=[n('D1',30,65,174,144,`D1 · ${amd?'HSA ELF':'cubin'}`,['device code bytes','entry symbol / metadata','host-owned image payload'],'file','green')],edges=[];
    if(p>=1){nodes.push(n('descriptor',250,65,482,126,'Host descriptor / native binding',['moduleName, functionName, codeBytes','GPU0 context + load options',amd?'HIPDeviceContext::loadFunction':'CUDADeviceContext::loadFunction'],'program','blue'));edges.push(e('D1','descriptor','image + entry'));}
    if(p>=2)nodes.push(n('cache',250,226,482,100,'Device function cache',[p>=7?'matching key → HIT':'key: module/code + name + load options',p>=6?'entry retains module/function refs':'cold lookup → MISS'],'slots',p>=6?'green':'orange'));
    if(p>=3){nodes.push(n('module',780,65,228,115,'MD1 · device module',['opaque vendor handle',amd?'hipModuleLoadData':'cuModuleLoadDataEx'],'chip','green'));edges.push(e('descriptor','module',p>=7?'reuse, no load':'load D1'));}
    if(p>=4){nodes.push(n('KD1',780,224,228,112,'KD1 · kernel function',['inside MD1',amd?'hipModuleGetFunction':'cuModuleGetFunction'],'chip','pink'));edges.push(e('module','KD1','lookup entry'));}
    if(p>=5)nodes.push(n('options',250,365,482,88,p>=7?'Cache hit skips vendor load/get-function calls':'DeviceFunction owns module/context references',[p>=7?'next invocation still supplies new data':'function/shared-memory attributes where supported'],'gate','green'));
    return scene(standard('PREPARED IMAGE','HOST RUNTIME / CACHE','DEVICE EXECUTABLE HANDLES'),nodes,edges,'BOUNDARY · physical GPU placement/lazy loading are vendor-driver responsibilities');
  }
  function abi(f,opts){
    const p=markIndex(f.mark,['host-pointer','operands','results','host-call','device-handle','encode','addresses','launch']);
    const nodes=[n('H1',30,67,174,105,'H1 · host callable',['void(void**)','host instruction address'],'card','pink')],edges=[];
    if(p>=1){nodes.push(n('host-slots',250,67,482,165,'Host compiled-routine ABI',p>=2?['&operandSlot → AV(Tensor Q_gpu)','&operandSlot → AV(Tensor K_gpu / V_gpu)','&resultSlot → output AsyncValue','resourceData, resourceByteCount, …']:['&operandSlot → AV(Tensor Q_gpu)','&operandSlot → AV(Tensor K_gpu / V_gpu)'],'slots','blue'));edges.push(e('host-slots','H1','invoke / unpack',{reverse:true}));}
    if(p>=3)nodes.push(n('buffers',30,249,174,184,'GPU tensor storage',['Q_gpu: 4096 B','K_gpu: 4096 B','V_gpu: 4096 B','O_gpu: 4096 B'],'memory','green'));
    if(p>=4)nodes.push(n('KD1',780,67,228,105,'KD1 · GPU function',['device handle',opts.vendor==='amd'?'HIP context / stream':'CUDA context / stream'],'chip','green'));
    if(p>=5){nodes.push(n('device-slots',250,275,482,181,'Device launch ABI',[],'argument-pack','pink',{phase:p}));edges.push(e('buffers','device-slots','encode descriptors'));}
    if(p>=7){nodes.push(n('dispatch',780,278,228,160,'GPU dispatch',['grid (4,2)','32 threads per block','8 output tiles',opts.vendor==='amd'?'hipModuleLaunchKernel':'cuLaunchKernelEx'],'queue','green'));edges.push(e('device-slots','dispatch','argument addresses'),e('KD1','dispatch','kernel handle'));}
    return scene(standard('CALLABLE / DEVICE STORAGE','TWO DIFFERENT ARGUMENT PACKS','DRIVER HANDLE / DISPATCH'),nodes,edges,'SOURCE · TileTensor binary fields vary by specialization; addresses are schematic');
  }
  function math(f){
    const s=f.math,fmt=M.fmt,rows=s.tile<0?[]:[`tile ${s.tile+1}: keys [${s.indexes}]`,`scores: [${(s.scores||[null,null]).map(fmt)}]`];
    const nodes=[n('matrices',25,59,282,293,`q${s.query} = [${s.Q[s.query]}] · inputs`,[],'math-inputs','pink',{math:s}),n('scores',331,59,337,150,'QK → stable exponentials',rows,'math-scores','blue',{math:s}),n('mass',691,59,324,205,'Denominator mass',[],'math-mass','green',{math:s}),n('output',331,243,337,234,'Rescale old O, then add PV',[],'math-output','pink',{math:s}),n('retained',25,379,282,98,'Retained across tiles',[`m=${fmt(s.m)}  l=${fmt(s.l)}`,`O=[${s.O.map(fmt)}]`,`${s.seen} keys committed`],'card','green'),n('formula',691,306,324,171,'Normalized-output recurrence',['α = exp(m_old − m_new)','l_new = α·l_old + ΣE','O_new = O_old·α·l_old/l_new','        + (E/l_new) @ V_tile',s.part==='store'?'Full-softmax reference agrees':'each query has independent m/l/O'],'program','green')];
    return scene([{x:12,y:22,w:300,h:464,title:'CONCRETE MATRIX VALUES'},{x:320,y:22,w:358,h:464,title:'CURRENT TILE / OUTPUT UPDATE'},{x:685,y:22,w:342,h:464,title:'NORMALIZATION / EQUATION'}],nodes,[], 'BROWSER ARITHMETIC · 2 query rows × 4 keys × 2 features; not GPU execution');
  }
  function pose(key,f,options={}){
    if(f.math)return math(f);
    if(key==='build')return build(f);
    if(['graph','module','custom'].includes(key))return graph(f);
    if(key==='context')return contextScene(f);
    if(key==='package')return packageScene(f);
    if(key==='session')return session(f);
    if(['compile','pipeline','artifacts'].includes(key))return compile(f,options);
    if(key==='init')return init(f);
    if(['execute','completion','synchronization'].includes(key))return execute(f,options);
    if(key==='host-link')return hostLink(f);
    if(['device-load','device-loading'].includes(key))return deviceLoad(f,options);
    return abi(f,options);
  }

  function part(parent,tag,key,attrs={},text){
    parent.__parts ||= new Map();
    let el=parent.__parts.get(key);
    if(!el){el=document.createElementNS(NS,tag);el.dataset.part=key;parent.__parts.set(key,el);parent.append(el);}
    el.style.display='';el.__touched=true;
    for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v));
    if(text!==undefined)el.textContent=text;
    return el;
  }
  function begin(parent){for(const el of parent.__parts?.values()||[])el.__touched=false;}
  function end(parent){for(const el of parent.__parts?.values()||[])if(!el.__touched)el.style.display='none';}
  function text(parent,key,x,y,value,cls='node-line',attrs={}){return part(parent,'text',key,{x,y,class:cls,...attrs},value);}
  function rect(parent,key,x,y,w,h,fill,stroke='none',extra={}){return part(parent,'rect',key,{x,y,width:w,height:h,rx:5,fill,stroke,...extra});}
  function wrap(value,length){
    const rows=[];
    for(const line of String(value).split('\n')){
      let remaining=line;
      while(remaining.length>length){let cut=remaining.lastIndexOf(' ',length);if(cut<length*.45)cut=length;rows.push(remaining.slice(0,cut));remaining=remaining.slice(cut).trimStart();}
      rows.push(remaining);
    }
    return rows;
  }
  function lines(g,key,values,x,y,w,lineHeight=19,cls='node-line',max=99){
    const split=values.flatMap(v=>wrap(v,Math.max(12,Math.floor(w/7.0)))).slice(0,max);
    split.forEach((v,i)=>text(g,key+i,x,y+i*lineHeight,v,cls));
  }
  function matrix(g,key,x,y,values,{cell=28,active=[],label='',fmt=M.fmt}={}){
    if(label)text(g,key+'label',x,y-8,label,'matrix-label');
    values.forEach((row,i)=>row.forEach((value,j)=>{
      const selected=active.some(a=>a[0]===i&&a[1]===j);
      rect(g,`${key}r${i}c${j}`,x+j*cell,y+i*cell,cell-2,cell-2,selected?'#edc5d6':'#f8fbf8',selected?'#a33667':'#bdcdbf');
      text(g,`${key}t${i}c${j}`,x+j*cell+(cell-2)/2,y+i*cell+cell*.65,fmt(value),'matrix-value',{'text-anchor':'middle','font-size':cell<25?10:12});
    }));
  }
  function contents(g,d){
    const [ink,tint]=palette[d.tone]||palette.green;
    rect(g,'body',0,0,d.w,d.h,d.kind==='process'?'#e4eaf0':tint,ink,{'stroke-width':1.15,'stroke-dasharray':d.kind==='process'?'6 3':'0'});
    if(d.kind==='file')part(g,'path','fold',{d:`M${d.w-23} 0V22H${d.w}Z`,fill:'#fff',stroke:ink,'stroke-width':.7});
    if(d.kind==='chip')for(let i=0;i<7;i++){rect(g,'pin'+i,16+i*27,-4,10,5,ink);rect(g,'pinb'+i,16+i*27,d.h-1,10,5,ink);}
    text(g,'title',13,24,d.title,'node-title',{fill:ink});
    const p=d.phase||0;
    if(d.kind==='module'){
      text(g,'loc',15,48,'loc: unknown or Python provenance','small');
      if(p<4){text(g,'empty',20,90,'module body: [insertion cursor]','node-line');return;}
      rect(g,'graph',15,69,d.w-30,d.h-87,'#f4f6f4','#679284');
      text(g,'glabel',28,93,'mo.graph @fused_attention','node-title');
      text(g,'sig',28,115,p>=13?'signature: (T,T,T) → T':'signature: (T,T,T) → ()','node-line');
      text(g,'region',29,141,'region 0','small');
      rect(g,'block',29,152,d.w-58,184,'#e6ece7','#9db7a7');
      text(g,'btitle',42,174,'B0 · one entry block','node-title');
      for(let i=0;i<3;i++){rect(g,'arg'+i,42+i*133,186,121,37,'#f3dfe9','#b67a97');text(g,'argtext'+i,51+i*133,208,['%q:T','%k:T','%v:T'][i],'node-title',{fill:'#9b255c'});}
      if(p>=7)text(g,'chain',44,249,'%ready = chain.create','node-line');
      if(p>=10)text(g,'custom',44,274,'%o:T = mo.custom(%q,%k,%v)','node-line',{fill:'#922052','font-weight':800});
      if(p>=11)text(g,'attrs',44,296,'BD=8, BN=16, device=gpu:0','node-line');
      if(p>=13)text(g,'return',44,320,'mo.output(%o)','node-line',{fill:'#056044','font-weight':800});
      else text(g,'cursor',44,320,p>=9?'↑ recording insertion cursor':'body retained for later recording','small');
    } else if(d.kind==='session'){
      const rows=[['Compile config', 'custom extension paths=[]'],['Runtime config',p>=3?'GPU0 + CPU0; numThreads=0':'pending'],['EngineContext1',p>=4?'MAXContext1 + CPUDevice1':'pending'],['Diagnostics',p>=6?'Python source callback enabled':'pending']];
      rows.forEach(([a,b],i)=>{rect(g,'slot'+i,14,43+i*72,d.w-28,58,'#f3f7f4','#b4c9b9');text(g,'slab'+i,26,64+i*72,a,'node-title');text(g,'svalue'+i,26,84+i*72,b,'node-line');});
    } else if(d.kind==='symbol-table'){
      const rows=[['object bytes',p>=3?'retained; interface read':'pending'],['host entry symbol',p>=7?'resolved → H1':p>=3?'declared, address pending':'pending'],['runtime helper ref',p>=7?'resolved relocation':p>=3?'external symbol reference':'pending'],['materialization unit',p>=5?'emitted → ObjectLinkingLayer':p>=4?'defined in namespace':'pending'],['finalized state',p>=7?'callable code ready':'awaiting symbol lookup / linking']];
      rows.forEach(([a,b],i)=>{rect(g,'sym'+i,14,46+i*51,d.w-28,43,'#f6f8f5','#bacbbb');text(g,'symn'+i,24,63+i*51,a,'small');text(g,'symv'+i,24,80+i*51,b,'node-line');});
    } else if(d.kind==='gpu-buffers'){
      ['Q_gpu','K_gpu','V_gpu','O_gpu'].forEach((label,i)=>{
        const allocated=i<3?p>i:p>=7,x=13+(i%2)*143,y=48+Math.floor(i/2)*97;
        text(g,'buflabel'+i,x,y,label+(allocated?' · 4096 B':' · not allocated'),'small');
        for(let r=0;r<2;r++)for(let c=0;c<4;c++){const filled=allocated&&(i<3||p>=14),key='buf'+i+r+c;rect(g,key,x+c*30,y+9+r*26,27,23,filled?'#bedac6':allocated?'#e1e9e2':'#f4f5f3',allocated?'#659e77':'#bcc8bf');}
        text(g,'buftype'+i,x,y+75,allocated?'float32[32,32]':'','small');
      });
    } else if(d.kind==='host-matrices'){
      ['Q_host','K_host','V_host'].forEach((name,i)=>{text(g,'hname'+i,13,52+i*47,name,'small');for(let j=0;j<5;j++)rect(g,'hcell'+i+j,13+j*29,59+i*47,25,16,d.copies>i?'#d3ded5':'#e7c5d4','#acbcb0');});
    } else if(d.kind==='tile-grid'){
      for(let y=0;y<2;y++)for(let x=0;x<4;x++){rect(g,'tile'+x+y,14+x*57,40+y*35,51,29,p>=14?'#b9dcc4':p>=11?'#ead0dc':'#eff2ee','#8fa99a');text(g,'tiletext'+x+y,38+x*57,60+y*35,`(${x},${y})`,'small',{'text-anchor':'middle'});}
      lines(g,'own',['one block owns 16×8 output',d.keyTile>=0?`current key tile: ${d.keyTile} / 3`:'dispatch queued','per block: QK → m/l → rescale O → PV'],258,51,239,22,'node-line');
    } else if(d.kind==='argument-pack'){
      for(let i=0;i<4;i++){const x=14+i*113;rect(g,'slot'+i,x,48,102,69,'#f6f3f4','#c4a5b2');text(g,'slotlabel'+i,x+9,71,'slot'+['Q','K','V','O'][i],'node-title');text(g,'slotptr'+i,x+9,93,['Q_gpu','K_gpu','V_gpu','O_gpu'][i],'node-line');}
      text(g,'args',14,143,p>=6?'args[i] = address of encoded HOST slot':'_to_device_type → encoded value / layout fields','node-line');
      text(g,'schematic',14,163,'Schematic source-argument grouping; concrete ABI may flatten differently','small');
    } else if(d.kind.startsWith('math-'))renderMath(g,d);
    else if(d.kind==='memory'){
      d.rows.forEach((v,i)=>{rect(g,'mem'+i,12,40+i*35,d.w-24,29,i===0?'#e4ccda':i===1?'#d0e0e6':'#d2e3d6',i===0?'#a2446f':'#afbfaf');text(g,'memtext'+i,20,59+i*35,v,'small');});
    } else if(d.kind==='gate')lines(g,'row',d.rows,13,45,d.w-26,17,'small');
    else lines(g,'row',d.rows,13,48,d.w-26,d.kind==='program'?22:20,'node-line',Math.floor((d.h-42)/19));
  }
  function renderMath(g,d){
    const s=d.math,fmt=M.fmt;
    if(d.kind==='math-inputs'){
      matrix(g,'Q',14,55,s.Q,{label:'Q',active:[[s.query,0],[s.query,1]],cell:27});
      const active=s.indexes?.flatMap(i=>[[i,0],[i,1]])||[];
      matrix(g,'K',90,55,s.K,{label:'K',active,cell:27});matrix(g,'V',181,55,s.V,{label:'V',active,cell:27});
      lines(g,'legend',[s.tile<0?'No tile selected':`Selected key rows: ${s.indexes.join(', ')}`,'Q stays fixed while K/V tiles change.',s.products?`products: ${s.products.map(p=>'['+p.map(fmt)+']').join('  ')}`:'dot product: q[0]k[0] + q[1]k[1]'],14,203,d.w-28,22,'small');
    } else if(d.kind==='math-scores'){
      const pairs=[['S',s.scores],['S − m_new',s.shifted],['E = exp(shift)',s.E],['P = E/l_new',s.P]];
      pairs.forEach(([label,v],i)=>{text(g,'key'+i,14,52+i*25,label,'small');text(g,'val'+i,135,52+i*25,v?'['+v.map(fmt).join(', ')+']':'—','state-value');});
    } else if(d.kind==='math-mass'){
      [['m_old → m_new',fmt(s.oldM??s.m)+' → '+fmt(s.newM)],['α = exp(Δm)',fmt(s.alpha)],['old mass = α·l_old',fmt(s.oldMass)],['new mass = ΣE',fmt(s.newMass)],['l_new = old + new',fmt(s.newL)]].forEach(([a,b],i)=>{text(g,'massk'+i,14,50+i*24,a,'small');text(g,'massv'+i,203,50+i*24,b,'state-value');});
      const old=s.oldMass||0,fresh=s.newMass||0,total=old+fresh,available=d.w-28;
      rect(g,'barbg',14,175,available,16,'#f5f7f3','#adc1b0');
      if(total>0){rect(g,'barold',14,175,available*old/total,16,'#a6356a');rect(g,'barnew',14+available*old/total,175,available*fresh/total,16,'#298b5c');}
    } else if(d.kind==='math-output'){
      const entries=[['O at start of this tile',s.oldO||s.O],['rescaled old term',s.scaledO],['current PV term',s.PV],['O = old term + PV',s.part==='merge'||s.part==='commit'||s.part==='store'?s.O:null]];
      entries.forEach(([a,b],i)=>{text(g,'outkey'+i,14,52+i*45,a,'small');text(g,'outval'+i,14,72+i*45,b?'['+b.map(fmt).join(', ')+']':'—','state-value');});
      text(g,'outnote',14,221,'Terms appear only after their computation.','small');
    }
  }
  function render(svg,key,frame,next,t,options={}){
    const a=pose(key,frame,options),b=next?pose(key,next,options):a,current=t>=.5?b:a;
    const bg=svg.querySelector('[data-scene-background]'),actorRoot=svg.querySelector('[data-scene-actors]'),edgeRoot=svg.querySelector('[data-scene-edges]');
    begin(bg);current.lanes.forEach((l,i)=>{rect(bg,'lane'+i,l.x,l.y,l.w,l.h,'#f0f3ef','#ced9d0');text(bg,'lanetitle'+i,l.x+12,l.y+20,l.title,'lane-label');});text(bg,'footer',19,503,current.footer,'small');end(bg);
    const byA=new Map(a.nodes.map(n=>[n.id,n])),byB=new Map(b.nodes.map(n=>[n.id,n])),all=new Set([...byA.keys(),...byB.keys()]),positions=new Map();
    begin(actorRoot);
    for(const id of all){
      const left=byA.get(id),right=byB.get(id),d=(t>=.5?right||left:left||right),from=left||{...right,x:right.x-25},to=right||{...left,x:left.x+15};
      const x=from.x+(to.x-from.x)*t,y=from.y+(to.y-from.y)*t,opacity=left&&right?1:left?1-t:t;
      if(opacity<=0)continue;
      const g=part(actorRoot,'g',id,{transform:`translate(${x.toFixed(3)},${y.toFixed(3)})`,opacity:opacity.toFixed(4),'data-actor-id':id});
      begin(g);contents(g,d);end(g);positions.set(id,{...d,x,y});
    }
    end(actorRoot);begin(edgeRoot);
    for(const edge of current.edges){const from=positions.get(edge.from),to=positions.get(edge.to);if(!from||!to)continue;
      let x1=from.x+from.w,y1=from.y+from.h/2,x2=to.x,y2=to.y+to.h/2;
      if(Math.abs(from.x-to.x)<50){x1=from.x+from.w/2;y1=from.y+from.h;x2=to.x+to.w/2;y2=to.y;}
      else if(from.x>to.x){x1=from.x;x2=to.x+to.w;}
      const middle=(x1+x2)/2,mark=svg.querySelector('marker').id;
      const d=edge.bottom?`M${x1} ${y1}V481H${x2}V${y2}`:`M${x1} ${y1}C${middle} ${y1},${middle} ${y2},${x2} ${y2}`;
      const path=part(edgeRoot,'path',edge.id,{d,fill:'none',stroke:'#72927c','stroke-width':1.3,'marker-end':`url(#${mark})`,'stroke-dasharray':edge.bottom?'4 3':'0'});
      part(path,'title','label',{},edge.label||`${edge.from} → ${edge.to}`);
      if(edge.label){const lx=(x1+x2)/2,ly=edge.bottom?479:(y1+y2)/2-8;
        const box={x:lx-edge.label.length*2.7-4,y:ly-11,w:edge.label.length*5.4+8,h:16};
        // Keep a relation's accessible title when its visible label would
        // collide with an actor in a narrow inter-column channel.
        if([...positions.values()].some(p=>box.x<p.x+p.w&&box.x+box.w>p.x&&box.y<p.y+p.h&&box.y+box.h>p.y))continue;
        rect(edgeRoot,edge.id+'bg',box.x,box.y,box.w,box.h,'#e9eeea');
        text(edgeRoot,edge.id+'label',lx,ly,edge.label,'edge-label',{'text-anchor':'middle'});
      }
    }end(edgeRoot);
  }
  root.AttentionVisuals={pose,render};
})(window);
