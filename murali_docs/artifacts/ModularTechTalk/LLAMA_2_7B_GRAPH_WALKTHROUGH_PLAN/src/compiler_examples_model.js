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
        add('contract','Describe shapes, dtype and device','TensorType describes symbolic inputs. These are contracts, not numerical inference results.',{graph:true},['entry','graph_init']);
        add('custom','Record a named custom operation',`ops.custom("${name}") records the operation and explicit output type.`,{ops:1},['entry','custom_api']);
        if(id==='epilogue')add('compose','Keep the surrounding math visible','The Python @ operator records matmul; add_one_elementwise and ×2 compose after it. Three mathematical operations can become one eligible epilogue route.',{ops:3},['entry','registration']);
        add('register','Connect the string to its Mojo implementation',`custom_extensions points at kernels/. The registered name selects ${routine}.`,{registered:true},['registration','custom_api']);
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
        add('finalize','Finalize the device graph and emit routines','This later compiler finalization is distinct from Python Graph.output. Emitted routines and custom Mojo enter kernel compilation.',{level:4},['mgp_pipeline','gc_pipeline']);
        add('mojo','Mojo specializes the implementation',id==='matmul'?`Compile-time target=${cpu?'cpu':'gpu'}, algorithm=${algorithm}. ${cpu?'CPU uses the naive function.':p.grid+'.'}`:id==='attention'?`BN=${p.BN}, BD=${p.BD}; ${routine}. ${cpu?'CPU key tile BN=4.':'GPU key tile BN_1=8, distinct from BN=16.'}`:'Types, SIMD width and target become concrete native code; the elementwise body does not call Python for every value.',{level:5},['registration','dispatch','mojo_semantics','mojo_elaboration']);
        add('llvm','Mojo MLIR → LLVM dialect → LLVM IR','LIT/KGEN/POP lowering and LLVM-module translation are separate transformations. File sheets represent in-memory views.',{level:6},['mojo_llvm','llvm_translation']);
        add('target',cpu?'Host backend emits and links native code':o.target==='cuda'?'LLVM NVPTX → PTX → cubin':'LLVM AMDGPU → object → linked ELF',cpu?'The host backend can emit an object directly and link native code. Assembly is an optional textual output.':o.target==='cuda'?'The pinned backend passes PTX through CUDA CompilationDevice to produce same-SM code. This is host compilation targeting a GPU.':'The AMDGPU backend emits target object code and links an ELF code object. This is host compilation targeting a GPU.',{level:7,code:true},['backend_select',cpu?'host_object':o.target==='cuda'?'nvptx_object':'amdgpu_backend']);
        add('mef','Prepare the runtime artifact / MEF bytes','Compiled routines plus prepared execution-plan resources form an in-memory compiled artifact. These scripts do not export a .mef disk file.',{level:8,mef:true},['serialize','artifact','gc_pipeline']);
        add('init','session.init(compiled) creates executable M1','All four examples take their tensors as call inputs. No model-weight registry is required by these calls.',{initialized:true},['entry','init_all']);
        add('ready','M1 is ready; numerical inputs are separate',id==='matmul'?'The helper already created its buffers. Each new helper invocation compiles and initializes a model, while main reuses S1.':'Random arrays and Buffer arguments are supplied after initialization. Compilation did not compute their results.',{},['entry']);
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
          if(state.fusion) {
            for(let i=0;i<4;i++) {
              const r=Math.floor(i/2),c=i%2;
              add('dot',`Accumulate Y[${r},${c}]`,`${A[r][0]}×${B[0][c]} + ${A[r][1]}×${B[1][c]} = ${flat[i]}. Keep the value in the illustrative accumulator.`,{cell:i,acc:[flat[i]],operands:[A[r][0],B[0][c],A[r][1],B[1][c]]},['entry','math']);
              add('epilogue','Apply +1, then ×2 before the final store',`(${flat[i]} + 1) × 2 = ${result[i]}. This eligible fusion sketch avoids a full intermediate tensor.`,{cell:i,acc:[result[i]],computed:result.slice(0,i+1)},['entry','math']);
            }
          } else {
            add('matmul','Store the matmul intermediate','Illustrative separate route: materialize A@B in TMP1.',{acc:flat,temporary:flat,tempName:'TMP1'},['entry']);
            add('add_one','Read TMP1 and store the add-one intermediate','Elementwise +1 produces TMP2 in this comparison.',{acc:flat.map(x=>x+1),temporary:flat.map(x=>x+1),tempName:'TMP2'},['math']);
            add('multiply','Read TMP2 and store the final output','The two intermediate matrices belong only to the unfused teaching comparison.',{acc:result,computed:result,tempName:'TMP2'},['entry']);
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
          add('initialize','Initialize online softmax state','For this query row: m=MIN, l=0, O=[0,0]. The animation uses the mathematical −∞ limit for the finite MIN sentinel.',{m:null,l:0,O:[0,0]},['math']);
          let m=-Infinity,l=0,out=[0,0];
          for(let t=0;t<2;t++) {
            const start=t*2,scores=K.slice(start,start+2).map(row=>Q.reduce((s,q,j)=>s+q*row[j],0));
            const oldM=m,oldL=l,oldO=out.slice();m=Math.max(m,...scores);
            const alpha=Number.isFinite(oldM)?Math.exp(oldM-m):0,E=scores.map(s=>Math.exp(s-m));l=alpha*oldL+E.reduce((a,b)=>a+b,0);
            const rescaled=oldO.map(x=>x*alpha*oldL/l),weights=E.map(x=>x/l);
            out=rescaled.map((x,j)=>x+weights.reduce((s,w,i)=>s+w*V[start+i][j],0));
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
