/* Attention mechanisms: deterministic frames, numerical state, retained actors. */
(function(root){
'use strict';
const clone=v=>JSON.parse(JSON.stringify(v));
const frame=(title,before,after,fields,refs,detail='')=>({title,before,after,fields,refs,detail});
const shapes={cpu:{N:8,D:8,BN:4,BD:4,key:4},cuda:{N:32,D:32,BN:16,BD:8,key:8},rocm:{N:32,D:32,BN:16,BD:8,key:8}};
const stories={
 package:[
  frame('Resolve the source package','kernels/__init__.mojo + child modules','absolute directory → package output path',{source:'raw .mojo files',process:'Python caller',P1:'not created'},[['package_builder',141],['kernel_package_root',1]]),
  frame('Resolve and launch the Mojo executable','DRIVER_PATH or shutil.which("mojo")','subprocess.run([mojo,"precompile",path,"-o",staging])',{source:'same package files',process:'separate compiler process',P1:'private staging file'},[['ct_mojo_run',70],['package_builder',178]]),
  frame('Parse, resolve and check package IR','Mojo source → AST / LIT IR','referenced declarations + semantic/lifetime checks',{source:'parametric implementation IR',process:'compiler precompile pipeline',P1:'checked package IR'},[['ct_precompile'],['ct_check_lit']]),
  frame('Serialize precompiled package','checked parametric package IR','version/header + compressed MLIR bytecode',{source:'bodies retained for specialization',process:'compiler writes staging',P1:'.mojoc package bytes'},[['ct_package_write']]),
  frame('Publish atomically and attach paths','private staging filename = final basename','os.replace(staging,final); G1 stores package paths',{source:'Python file remains separate',process:'Python caller resumes',P1:'published package; no machine code'},[['package_builder',179],['ct_graph_import',1706]])],
 context:[
  frame('Read context on the calling thread','main: current → DEFAULT\nworker: current → None','Shared DEFAULT object exists; worker has no active scope',{CURRENT:'None',DEFAULT:'shared object',G1:'not yet built'},[['ct_graph_context_state',26]]),
  frame('Decorator calls the context guard','@in_default_mlir_context\nGraph.__init__(...)','call_with_default_mlir_context → enter DEFAULT',{CURRENT:'DEFAULT on worker',DEFAULT:'same shared object',G1:'construction scope'},[['ct_graph_context',70],['ct_graph_context_state',56]]),
  frame('Build types and IR in the selected context','current Context → dialects / types / attrs','MLIR types and G1 operations use this context',{CURRENT:'DEFAULT on worker',DEFAULT:'same shared object',G1:'IR owned by context'},[['graph_init',860]]),
  frame('Leave the temporary scope','worker exits with DEFAULT','worker current returns to None; shared object survives',{CURRENT:'None on worker',DEFAULT:'still retained',G1:'still retained'},[['ct_graph_context_state',65]])],
 location:[
  frame('Read traceback capture policy','with _location() as loc','capture disabled → Location.unknown()',{loc:'unknown',module:'pending',graph:'pending'},[['ct_graph_location',438]]),
  frame('Captured-stack alternative','main() → Graph("fused_attention") at L41','frame summaries → opaque MLIR Location',{loc:'Python call-site metadata',module:'pending',graph:'pending'},[['ct_graph_frame_loc',162],['entry',41]],'Example label: fused_attention.py:41 in main; this is illustrative provenance, not an IR dump.'),
  frame('Attach location to the container','builtin.ModuleOp(location=loc)','module operation retains the Location',{loc:'same handle',module:'loc attached',graph:'pending'},[['graph_init',856]]),
  frame('Attach to graph and input arguments','mo.GraphOp(builder,loc,...)','GraphOp + %q/%k/%v carry provenance',{loc:'retained after scope',module:'loc attached',graph:'loc on op + args'},[['graph_init',862],['ct_graph_op_build',848]])],
 module:[
  frame('Choose reuse or creation','module=None OR module=existing','source branch selects one native container',{MODULE1:'absent or existing',body:'not selected',G1:'not inserted'},[['graph_init',856]]),
  frame('Create the empty module','builtin.ModuleOp(location=loc)','builtin.module {\n  // one empty body block\n}',{MODULE1:'one IR container',body:'one block',G1:'not inserted'},[['ct_builtin_module',62]]),
  frame('Reuse alternative retains siblings','existing module contains @encoder','Reuse the same module; keep @encoder\nappend @fused_attention later',{MODULE1:'same container',body:'existing operations kept',G1:'pending sibling'},[['graph_init',856]]),
  frame('Insert named graph','builder points at body.end','module { @encoder; mo.graph @fused_attention {...} }',{MODULE1:'same container',body:'two sibling definitions',G1:'inserted in body'},[['graph_init',858],['graph_init',860]])],
 builder:[
  frame('Read the module body','MODULE1 → region → block','body.end yields block + end iterator',{block:'module body',iterator:'end',builder:'pending'},[['ct_block_end',408]]),
  frame('Create the native builder','OpBuilder(self._module.body.end)','C++ mlir::OpBuilder(block,end)',{block:'same body',iterator:'after siblings',builder:'ready'},[['ct_op_builder',612]]),
  frame('Create at that insertion point','[ @encoder | insertion cursor ]','[ @encoder | @fused_attention | cursor ]',{block:'same body',iterator:'after new graph',builder:'retained'},[['graph_init',860]]),
  frame('Later insertion uses graph body','Graph._current_block → G1 body','mo.custom / mo.output go INSIDE G1',{block:'graph entry block',iterator:'graph body end',builder:'graph operation builder'},[['ct_insert_result']])],
 type:[
  frame('Convert shape dimensions','TensorType(float32,[32,32],gpu:0)','32 → IntegerAttr(si64,32) → CastFromBuiltinAttr',{shape:'two static sizes',dtype:'float32 unchanged',device:'pending'},[['ct_static_dim_to_mlir']]),
  frame('Make MOSH shape attribute','two typed static dimension attributes','mosh.ShapeAttr([dim32,dim32],ShapeType())',{shape:'ShapeAttr [32,32]',dtype:'float32 unchanged',device:'pending'},[['ct_shape_to_mlir']]),
  frame('Convert device and metadata','DeviceRef(kind="gpu",id=0); metadata={}','m.DeviceRefAttr("gpu",0); DictionaryAttr({})',{shape:'ShapeAttr [32,32]',dtype:'float32',device:'gpu:0 + empty layout'},[['ct_device_ref_to_mlir'],['ct_tensor_to_mlir',430]]),
  frame('Create three native types','TQ/TK/TV Python descriptions','TQ=TK=TV=MO tensor<f32,[32,32],gpu:0>',{shape:'three typed inputs',dtype:'all float32',device:'all gpu:0'},[['graph_init',864]],'CPU branch uses [8,8] and cpu:0. Types contain no matrix data.')],
 graph:[
  frame('Record input signature','input_types=[TQ,TK,TV]; results=[]','signature (TQ,TK,TV) → ()',{signature:'three inputs / no results',region:'pending',block:'pending'},[['ct_graph_op_build',831]]),
  frame('Create one graph region','OperationState for @fused_attention','state.addRegion() creates R0',{signature:'unchanged',region:'R0 created',block:'pending'},[['ct_graph_op_build',846]]),
  frame('Create one entry block','R0 has no blocks','R0.emplaceBlock() → B0',{signature:'unchanged',region:'R0',block:'ONE B0'},[['ct_graph_op_build',847]]),
  frame('Append one argument per input','B0 empty','B0(%q:TQ,%k:TK,%v:TV)',{signature:'(TQ,TK,TV) → ()',region:'same R0',block:'three arguments in B0'},[['ct_graph_op_build',848]])],
 inputs:[
  frame('Read declared block arguments','G1._graph_body.arguments','[%arg0:TQ,%arg1:TK,%arg2:TV]',{q:'native arg0',k:'native arg1',v:'native arg2'},[['ct_graph_inputs',943]]),
  frame('Omit internal chain inputs when present','_has_chain_input=False in this example','all three tensor arguments remain',{q:'arg0 kept',k:'arg1 kept',v:'arg2 kept'},[['ct_graph_inputs',944]]),
  frame('Wrap existing native values','Value.from_mlir(...) per argument','TensorValue wrappers refer to the same SSA values',{q:'wrapper → %arg0',k:'wrapper → %arg1',v:'wrapper → %arg2'},[['ct_graph_inputs',948]]),
  frame('Python assignment receives wrappers','q,k,v,*_=graph.inputs','ops.custom(values=[q,k,v]) can record a call',{q:'symbolic; no data',k:'symbolic; no data',v:'symbolic; no data'},[['entry',56],['entry',61]])],
 custom:[
  frame('Find graph and registration name','name="modular_ops::fused_attention_custom"','Graph.current=G1; symbol=StringAttr(name)',{symbol:'namespaced registration',operands:'pending',result:'pending'},[['custom_api',94],['custom_api',96]]),
  frame('Unwrap three operand values','q,k,v are Python TensorValue wrappers','operands=[%arg0,%arg1,%arg2]; out_types=[TO]',{symbol:'same StringAttr',operands:'Q / K / V SSA values',result:'type TO'},[['custom_api',105]]),
  frame('Insert new mo.custom operation','G1 body: %q,%k,%v and CH0','%o = mo.custom(%q,%k,%v) {symbol=...}',{symbol:'same registration',operands:'attached to CUST1',result:'%o exists in MO'},[['custom_api',109]]),
  frame('Attach compile-time tile parameters','parameters={BD:8,BN:16}','BD=8:index; BN=16:index; parameters dictionary',{symbol:'same registration',operands:'unchanged',result:'%o; call gains BN/BD'},[['attention_parameter_attr',48],['custom_api',113]]),
  frame('Attach device after parameter dictionary','DeviceRef.from_device(Accelerator())','custom_op.device=DeviceRefAttr("gpu",0)',{symbol:'same registration',operands:'unchanged',result:'%o; gpu:0 call'},[['custom_api',122]]),
  frame('Verify, then return one wrapper','P1 registry + CUST1 contracts','results=[TensorValue(%o:TO)]',{symbol:'verified against P1',operands:'three inputs checked',result:'ONE symbolic wrapper'},[['custom_api',127],['custom_api',129]])],
 validation:[
  frame('Index declarations loaded from P1','P1: precompiled Mojo package','registry["modular_ops::fused_attention_custom"] → candidates',{name:'pending',mapping:'pending',params:'pending'},[['ct_validation_lookup',48]]),
  frame('Match exact namespaced registration','CUST1.symbol StringAttr','candidate FusedAttention.execute matches string',{name:'matched',mapping:'pending',params:'pending'},[['ct_validation_native',634],['attention_registration',66]]),
  frame('Select a device-compatible candidate','CUST1.device=gpu:0 OR cpu:0','target parameter becomes "gpu" OR "cpu"',{name:'matched',mapping:'device checked',params:'target inferred'},[['ct_validation_device',157]]),
  frame('Map inputs and destination output','CUST1: inputs=[q,k,v], results=[o]','query←q; key←k; value←v; output←o',{name:'matched',mapping:'3 inputs + 1 output',params:'dtype=float32,rank=2'},[['ct_validation_mapping',240]],'Destination-passing Mojo puts output first; inference maps the graph inputs and result to their declared roles.'),
  frame('Check explicit parameter names/types','parameters={BD:8,BN:16}','BD:Int=8, BN:Int=16; inferred dtype/rank retained',{name:'matched',mapping:'all operands consumed',params:'BN/BD accepted'},[['ct_validation_params',426]]),
  frame('Success permits wrappers to return','verified CUST1 remains in G1','ops.custom returns [TensorValue(%o)]',{name:'passed',mapping:'passed',params:'passed'},[['custom_api',129]],'Wrong name, type, missing or extra operands/parameters raise; no kernel arithmetic is executed by validation.')],
 output:[
  frame('Gather symbolic output','results=[TensorValue(%o:TO)]','graph.output(*results)',{G1:'three inputs',CUST1:'verified',output:'%o selected'},[['entry',68]]),
  frame('Insert graph terminator','G1 body contains CUST1','mo.output(%o) in the graph body',{G1:'output recorded',CUST1:'retained',output:'mo.output'},[['graph_output']]),
  frame('Finalize the result signature','(TQ,TK,TV) → ()','(TQ,TK,TV) → (TO)',{G1:'verified signature',CUST1:'retained',output:'one result type'},[['graph_output']]),
  frame('Leave with scope','Graph.current=G1','Graph.current restored; G1 and P1 retained',{G1:'ready for compile',CUST1:'symbolic only',output:'no Buffer yet'},[['ct_graph_context_manager',1145]])],
 compiler:[
  frame('Clone before lowering','G1 verified + seeded P1 declarations','C1=clone(G1); G1 remains intact',{G1:'retained MO',C1:'MO clone',kernel:'P1 bodies'},[['ct_framework_compile',381]]),
  frame('Verify and simplify MO contracts','CUST1(Q,K,V) → O; BN/BD and device','shape/device/contracts checked; canonicalization runs',{G1:'retained',C1:'verified MO',kernel:'opaque custom contract'},[['attention_gc',219]]),
  frame('Lower MO to opaque MAP call','tensor inputs + custom result type','opaque iteration wraps Q/K/V accesses + call + O',{G1:'retained',C1:'MAP accesses and call',kernel:'P1 implementation'},[['mo_to_map',764]],'This custom attention call is opaque to surrounding MAP fusion; its online recurrence remains inside the Mojo implementation.'),
  frame('Select implementation and glue','MAP opaque call + MojoRegistry','MOGG routine calls FusedAttention.execute with mapped tensors',{G1:'retained',C1:'MOGG routine',kernel:'BN/BD + dtype/rank bound'},[['map_to_mogg',53]]),
  frame('Lower graph to runtime plan','MOGG graph + routine calls','MGP init/execute regions, buffers, state and chains',{G1:'retained',C1:'MGP plan',kernel:'routine code still separate'},[['mogg_to_mgp',2406]]),
  frame('Compile routines and lower their runtime calls','JitCompileKernels: Mojo IR → native code','ModuleLoad + ExecuteGeneric; outlined callable plan',{G1:'retained',C1:'final runtime-call plan',kernel:'host/device artifacts'},[['attention_jit'],['attention_runtime_lower']])],
 kernels:[
  frame('Specialize parametric implementation','P1: execute[dtype,rank,BN,BD,target]','KIR1: float32,rank=2,BN=16,BD=8,target=gpu',{KIR1:'concrete parameters',LLVM1:'absent',code:'absent'},[['kgen_elaboration'],['attention_registration',74]]),
  frame('Lower through KGEN/POP/LLVM dialect','concrete Mojo implementation IR','MLIR module contains lowered llvm.* operations',{KIR1:'MLIR LLVM dialect',LLVM1:'absent',code:'absent'},[['kgen_llvm_passes']]),
  frame('Translate MLIR into LLVM IR','MLIR llvm dialect is still MLIR','LLVM1 = llvm::Module; typed LLVM instructions',{KIR1:'translated',LLVM1:'new LLVM module',code:'absent'},[['object_lower_to_llvm'],['llvm_translation']]),
  frame('Select backend by triple/arch/features','nvptx64-nvidia-cuda OR amdgcn-amd-amdhsa','NVPTXBackend OR AMDGPUBackend; host backend for host code',{KIR1:'compiled',LLVM1:'target metadata',code:'host + device artifacts'},[['target_info',241],['backend_registry',114]],'Example architectures sm_90a/gfx942 illustrate cross-codegen. This document does not query or run a GPU.')],
 session:[
  frame('Normalize device references','devices=[Accelerator()]','final_devices=[GPU0,CPU0]; first occurrences kept',{S1:'Python wrapper',N1:'pending',context:'pending'},[['ct_session',689]]),
  frame('Populate configurations','custom_extensions=None; num_threads=None','CFG1.paths=[]; runtime threads=0, API=python',{S1:'owns N1',N1:'configs + device refs',context:'pending'},[['ct_session_native',546]]),
  frame('Get or create runtime context','Engine::Context::create(runtimeConfig)','E1 → MAX ContextRef(CX1) → AsyncRT CPUDevice HRT1',{S1:'same wrapper',N1:'runtimeContext=E1',context:'CX1 may be shared'},[['ct_context_create',184]]),
  frame('Retain ownership for compile/init','N1 keeps configs/devices/runtimeContext','S1 usable; no graph compilation or model execution yet',{S1:'retained',N1:'ready',context:'not the MLIR Context'},[['ct_session_native',563]])],
 init:[
  frame('Resolve artifact and binding prerequisites','S1 + CM1 + weights_registry=None','expected_weights={} and registry={} → pass',{CM1:'resolved descriptors',M1:'pending',handles:'pending'},[['init_all',1202]]),
  frame('Schedule model setup on CPU runtime','runtimeContext.initModel(...)','worker has AsyncRT CPUDevice in thread-local state',{CM1:'retained',M1:'setup pending',handles:'pending'},[['ct_context_init',297]],'The worker executes initialization with the runtime association its primitives need; GPU arithmetic has not started.'),
  frame('Load routines and initialize state','initFn.invoke(initArgs,cpuDevice)','map device refs; load archive symbols; create MGP state',{CM1:'retained',M1:'state constructed',handles:'native callables resolved'},[['ct_model_setup',220],['ct_kernel_load']]),
  frame('Await setup and return model wrapper','ExecutableModel(chain,mgpContext,weights,extraArgs)','setup chain/context/state ready → M1',{CM1:'retained',M1:'initialized',handles:'retained for execute'},[['ct_model_setup',237],['ct_wait_setup']])],
 execute:[
  frame('Supply actual buffers','BQ1/BK1/BV1 each f32[32,32] on GPU0','Model.execute(Q,K,V) receives real storage',{M1:'same initialized code',inputs:'three fresh Buffers',output:'pending'},[['entry',78],['entry',82]]),
  frame('Prepare native Tensor AsyncValues','validate count/shape/dtype/device','retain buffer storage + spec; no new copy here',{M1:'retained',inputs:'Tensor AsyncValues',output:'pending'},[['ct_execute_native',462],['ct_execute_input',121]]),
  frame('Invoke resident executable','retained state + Q/K/V input references','host routine → CPU loops OR GPU enqueue → kernel',{M1:'same resident code',inputs:'consumed by invocation',output:'device computation'},[['ct_execute_callable',384],['attention_launch',403]]),
  frame('Await result and wrap output','native results = [ordering chain,tensor O]','drop chain → output=[Buffer O1]; print(output)',{M1:'retained for reuse',inputs:'owners retained as needed',output:'one selected-device Buffer'},[['ct_execute_results',159],['entry',83]])],
 launch:[
  frame('Bind concrete GPU function','BN=16,BD=8; Q/K/V/O layouts','kernel_func=fused_attention_kernel[types,layouts,16,8]',{function:'specialized device kernel',grid:'pending',block:'pending'},[['attention_launch',391]]),
  frame('Derive launch geometry','N=D=32','grid=(D/BD,N/BN)=(4,2); block=32',{function:'same specialization',grid:'8 work blocks',block:'32 threads'},[['attention_launch',408],['attention_launch',409]]),
  frame('Host enqueues kernel and buffer arguments','ctx.enqueue_function[kernel_func](Q,K,V,O)','driver route receives resident function + grid/block + buffers',{function:'resident device handle',grid:'(4,2)',block:'32 threads'},[['attention_launch',403]],'Native module/function loading and driver details are expanded in the linked loader lesson; motion here is a source illustration.'),
  frame('Each block owns one output tile','block=(x,y)','Q rows y*16..; O columns x*8..; O tile [16,8]',{function:'same kernel',grid:'4 column × 2 row tiles',block:'tile computation'},[['attention_tiles',333],['attention_output',378]])],
 mma:[
  frame('Create matmul fragments','Q_tile [16,32]; K_tile [8,32]','TensorCore(M=16,N=8,BK=8,transpose_b=True)',{A:'pending',B:'pending',C:'LOCAL accumulator=0'},[['attention_mma',272],['attention_mma',276]]),
  frame('Load one reduction slice','k_i=0; features 0..7','load_a(Q[16,8]); load_b(K[8,8])',{A:'A fragment',B:'B fragment',C:'same accumulator'},[['attention_mma',296],['attention_mma',301]]),
  frame('Multiply and accumulate','A_fragment × B_fragment + C','d_reg=mma_op(...); c_reg.copy_from(d_reg)',{A:'consumed',B:'consumed',C:'partial scores'},[['attention_mma',305]]),
  frame('Repeat four BK slices','D/BK=32/8=4','C accumulates complete S [16,8]',{A:'next slices reused',B:'next slices reused',C:'final S tile'},[['attention_mma',295]]),
  frame('Store shared tile and synchronize block','mma.store_d(out_sram,c_reg)','barrier(); res.copy_from(out_sram)',{A:'no full score matrix',B:'shared temporary S',C:'result available'},[['attention_mma',307],['attention_mma',309]],'These are source storage scopes and block synchronization; exact register residency and hardware instruction choice require target evidence.')],
 storage:[
  frame('Read one query tile','GPU block=(x,y); N=D=32','Q_tile [16,32] is a view of Q',{Q:'512 logical elements',KV:'views not copied here',state:'m/l/O initialized'},[['attention_tiles',333]]),
  frame('Select one key/value tile','key iteration t=0..3; BN_1=8','K [8,32], V [8,8]; S [16,8]',{Q:'same query tile',KV:'current key/value views',state:'S temporary=128 elements'},[['attention_tiles',344]]),
  frame('Keep normalized-output state','per-row max/denominator + partial output','m [16], l [16], O_i [16,8] survive key iteration',{Q:'same view',KV:'next tile selected',state:'160 logical elements'},[['attention_output',376],['attention_output',377]]),
  frame('Store once all key tiles contribute','4 key tiles consumed','O_i copied into final [16,8] destination view',{Q:'input retained',KV:'temporary tiles reused',state:'final O tile stored'},[['attention_output',378]],'No persistent full [32,32] score tensor is allocated by this fused loop. Logical counts are not measured GPU memory residency.')],
};
function online(){
 const q=[1,0],k=[[1,0],[0,1],[3,0],[2,1]],v=[[1,0],[0,2],[2,1],[1,3]],scores=k.map(row=>q.reduce((s,n,j)=>s+n*row[j],0));
 let m=-Infinity,l=0,O=[0,0],out=[];
 const push=(title,patch,refs,detail)=>out.push({title,kind:'online',q,k,v,scores,m:Number.isFinite(m)?m:null,l,O:[...O],tile:-1,alpha:0,E:[],prob:[],oldContribution:[0,0],newContribution:[0,0],...patch,refs,detail});
 push('Start one query row',{},{},'Teaching sizes: Q row [1,0], four keys, two key tiles of 2; original GPU key tile is 8.');
 out[0].refs=[['attention_tiles',335]];
 for(let tile=0;tile<2;tile++){
  const oldM=m,oldL=l,oldO=[...O],S=scores.slice(tile*2,tile*2+2),newM=Math.max(m,...S),alpha=Number.isFinite(m)?Math.exp(m-newM):0,E=S.map(x=>Math.exp(x-newM)),newL=alpha*l+E.reduce((a,b)=>a+b,0),prob=E.map(x=>x/newL),oldContribution=O.map(x=>x*alpha*l/newL),newContribution=[0,1].map(j=>prob.reduce((s,p,i)=>s+p*v[tile*2+i][j],0)),newO=oldContribution.map((x,j)=>x+newContribution[j]);
  const base={tile,oldM:Number.isFinite(oldM)?oldM:null,oldL,oldO,S,newM,newL,alpha,E,prob,oldContribution,newContribution,newO};
  push('Load scores for key tile '+(tile+1),{...base,visible:'scores'},[['attention_tiles',346]],'Dot Q against each K row; this row gives scores '+S.join(', ')+'.');
  push('Update running row maximum',{...base,visible:'maximum',m:newM},[['attention_max',348]],tile?'Maximum grows 1 → 3; earlier weights must be rescaled.':'First maximum is 1.');
  push('Compute stable exponentials and denominator',{...base,visible:'denominator',m:newM,l:newL},[['attention_denominator',352]],'alpha*l_old + sum(exp(S-m_new)) keeps all visited keys in one stable denominator.');
  push('Normalize and rescale previous output',{...base,visible:'rescale',m:newM,l:newL,O:oldContribution},[['attention_output',366]],'E/l_new are this tile’s weights; O_old is multiplied by alpha*l_old/l_new.');
  m=newM;l=newL;O=newO;
  push('Add this tile’s weighted V contribution',{...base,visible:'add',m,l,O:[...O]},[['attention_output',374]],'O_new = rescaled O_old + (E/l_new) @ V_tile; retain m,l,O for the next tile.');
 }
 const exp=scores.map(s=>Math.exp(s-Math.max(...scores))),sum=exp.reduce((a,b)=>a+b,0),reference=[0,1].map(j=>exp.reduce((s,e,i)=>s+e/sum*v[i][j],0));
 push('Store final row and compare stable full softmax',{...out[out.length-1],title:'Store final row and compare stable full softmax',tile:1,visible:'store',reference,match:O.every((x,j)=>Math.abs(x-reference[j])<1e-12)},[['attention_output',378]],'Browser arithmetic agrees with the independent full softmax for the same tiny inputs; this is not MAX/GPU parity evidence.');
 return out;
}
function frames(kind){return clone(kind==='online'?online():stories[kind]||stories.graph);}
root.AttentionMechanisms={frames,shapes};
if(typeof module==='object')module.exports=root.AttentionMechanisms;
})(typeof globalThis==='object'?globalThis:this);
