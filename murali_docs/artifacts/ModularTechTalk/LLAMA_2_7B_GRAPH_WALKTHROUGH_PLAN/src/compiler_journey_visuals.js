(function(root) {
  'use strict';
  const point=(x,y)=>({x,y});
  const actor=(id,kind,label,x,y,w,h,color,extra={})=>({id,kind,label,x,y,w,h,color,opacity:1,...extra});
  class JourneyVisual extends root.CompilerWorkbenchVisual {
    targets(s) {
      const list=[],add=(...v)=>list.push(actor(...v)),subject=s.subject;
      const filename={addition:'addition.py',epilogue:'elementwise_addition.py',matmul:'matrix_multiplication.py',attention:'fused_attention.py'}[s.id];
      add('source','file',filename,43,133,222,102,'python',{tag:'ORIGINAL PYTHON BUILDER',lines:[subject.expression],place:'caller source'});
      add('S1','handle','S1 · retained context',43,502,172,79,'engine',{tag:'HOST SESSION',lines:['runtime / compilation'],place:'host context'});
      if(s.package&&!s.kernelIR)add('P1','file','P1 · .mojoc package',585,134,242,169,'mojo',{tag:'MLIR BYTECODE · PARAMETRIC',lines:[subject.name,'bodies available on import','specialization deferred'],place:'implementation package',lineage:'operation'});
      if(s.graph)add('G1','file','G1 · '+(s.verified?'verified':'recording'),43,271,222,124,'python',{tag:'CALLER GRAPH · RETAINED',lines:['typed inputs / output',subject.op],place:'caller owns graph'});
      if(s.clone&&!s.compiled)add('C1','file','C1 · '+s.graphLevel,331,133,229,264,'mlir',{tag:'COMPILER-OWNED GRAPH / PLAN',lines:['inputs → producer → output','kernel IR is a separate object'],origin:point(43,270),place:'compiler graph / plan',lineage:'plan'});
      if(s.graphLevel==='MGP'&&s.kernelIR&&!s.compiled)add('plan-fields','crate','refs / dependencies',348,283,193,85,'sync',{tag:'C1 · RETAINED PLAN',lines:['input → routine → output'],place:'MGP plan',lineage:'plan'});
      if(s.clone&&!s.kernelIR&&!s.hostCode&&!s.compiled){
        if(s.id==='epilogue') {
          add('matmul','tile','MATMUL → accumulator',348,245,193,37,'mlir',{tag:'PRODUCER',place:'C1',lineage:'operation'});
          add('op-main','tile',s.phase==='fusion'&&s.fusion==='fused'?'epilogue · +1, ×2':'+1 → ×2',348,294,193,37,'mojo',{origin:point(54,330),place:'C1',lineage:'operation'});
          if(s.phase==='fusion'&&s.fusion==='separate')add('TMP','handle','TMP1 → TMP2 → Y',348,344,193,34,'buffer',{place:'illustrative temporaries',lineage:'plan'});
        } else add('op-main','tile',subject.op+' · '+s.level,348,270,193,47,'mojo',{origin:point(54,330),place:'C1',lineage:'operation'});
        if(s.level==='MGP')add('dependencies','handle','input refs | chain | output',348,344,193,34,'sync',{place:'readiness plan',lineage:'plan'});
      }
      if(s.deviceFinalized&&!s.compiled)add('finalized-graph','stamp','DEVICE FINALIZED',348,385,193,24,'sync',{place:'later compiler finalization'});
      if(s.emitted&&!s.kernelIR&&!s.hostCode&&!s.compiled)add('emitted-routine','file','Emitted graph routines',585,319,242,78,'mojo',{tag:'ALONGSIDE CUSTOM BODIES',place:'generated Mojo input to JIT',lineage:'operation'});
      if(s.kernelIR&&!s.hostCode) {
        add('KIR1','file','KIR1 · '+(s.llvm?'LLVM dialect':s.level),585,133,242,264,'mojo',{tag:s.concrete?'CONCRETE IMPLEMENTATION':'PARAMETRIC IMPLEMENTATION',lines:[s.algorithm==='naive'&&s.id==='matmul'?'authored naive body':subject.body,s.concrete?'parameters resolved':'parameters still visible'],origin:point(585,134),place:'kernel IR'});
        if(s.concrete) {
          add('op-load','tile',s.id==='attention'?'load Q / K / V':s.id==='matmul'?'load / copy operand tiles':'load typed input',602,252,207,35,'buffer',{origin:point(355,273),place:'implementation',lineage:'load'});
          add('op-main','tile',s.operationForm,602,300,207,37,'mojo',{origin:point(355,273),place:'implementation',lineage:'operation'});
          add('op-store','tile',s.id==='attention'?'rescale / store running O':'store typed output',602,350,207,35,'runtime',{origin:point(355,273),place:'implementation',lineage:'store'});
        } else add('op-main','tile',subject.op+' · generator',602,281,207,49,'mojo',{origin:point(355,273),place:'parametric implementation',lineage:'operation'});
      }
      if(!s.compiled) {
        add('target-policy','tool',s.cpu?'CPU · host target':s.target==='cuda'?'NVIDIA · NVPTX':'AMD · AMDGPU',890,133,266,90,'backend',{tag:'COMPILATION TARGET',lines:[s.cpu?'CPU objects / native routines':s.target==='cuda'?'PTX → same-SM cubin':'target object → ELF / hsaco'],place:'target compiler'});
      } else add('CM1','crate','CM1 · prepared artifact',890,134,266,263,'artifact',{tag:s.cache==='artifact-hit'?'COMPATIBLE CACHE HIT':'MEF BYTES · IN MEMORY',lines:['code archives + runtime plan'],origin:point(895,270),place:'compiled artifact',lineage:'plan'});
      if(s.hostCode)add('HOST1','binary','HOST1',902,251,119,107,'artifact',{tag:'HOST ROUTINES',place:'host code'});
      if(s.deviceCode)add('GPU1','binary','GPU1',1035,251,109,107,'backend',{tag:s.target==='cuda'?'CUBIN':'HSACO',place:'GPU code'});
      if(s.llvm&&!s.compiled)add('LLVM1','file','LLVM1 · llvm::Module',890,s.hostCode||s.deviceCode?229:247,266,s.hostCode||s.deviceCode?20:145,'mlir',{tag:'TRANSLATED LLVM IR',lines:['distinct from KIR1 / MLIR','selected target backend input'],origin:point(585,133),place:'LLVM module'});
      if(s.hostCode)add('op-main','tile',s.id==='addition'?'+ 1':s.id==='epilogue'?'+1 ×2':s.id==='matmul'?'matmul':'attn',s.cpu?910:1043,313,s.cpu?103:93,29,'mojo',{origin:point(608,298),place:s.cpu?'host code ancestry':'device code ancestry',lineage:'operation'});
      if(s.compiled)add('plan-ref','handle','C1 → runtime callables',909,361,228,28,'mlir',{place:'artifact plan',lineage:'plan'});
      if(s.initialized)add('M1','chip','M1 · executable',236,495,168,94,'engine',{tag:'RETAINED MODEL',origin:point(894,366),via:[point(867,431),point(281,431)],place:'host model'});
      if(s.loading)add('H1','handle',s.initialized?'H1 · callables':'H1 · resolve refs',s.initialized?247:236,s.initialized?551:500,s.initialized?146:168,s.initialized?28:84,'runtime',{tag:'INITIALIZATION',lines:s.initialized?[]:['module / routine refs'],origin:point(902,291),place:s.initialized?'M1 runtime handles':'initializing runtime',lineage:'plan'});
      if(s.input)add('B'+(s.call||1),'crate','B'+(s.call||1),426,499,113,87,'buffer',{tag:s.cpu?'CPU ARGS':'GPU ARGS',lines:[s.id==='addition'?(s.call===2?'X=[4, 5]':'X=[0, 1]'):'runtime tensors'],origin:point(128,431),place:'caller buffer arguments',lineage:'load'});
      if(s.submitted||s.result)add('JOB'+s.call,'descriptor','JOB'+s.call+' · refs',553,499,96,87,'runtime',{tag:s.result?'COMPLETED':'EXECUTE',lines:['code + B'+s.call],origin:point(427,512),place:'native call',lineage:'plan'});
      if(s.initialized)add('driver','tool',s.cpu?'CPU native call':'Driver API',711,500,135,87,'driver',{tag:'RUNTIME BOUNDARY',lines:[s.result?'result ready':s.submitted?'work submitted':'waiting for call'],place:'execution interface'});
      if(s.submitted||s.result)add('device-input','crate',s.cpu?'ref(B'+s.call+')':'D'+s.call+' · inputs',863,500,126,87,'buffer',{tag:s.cpu?'SAME HOST DATA':'DEVICE STORAGE',lines:[s.id==='addition'?(s.call===2?'4, 5':'0, 1'):'input tensor views'],origin:point(426,499),via:[point(674,622),point(849,622)],place:s.cpu?'CPU input views':'explicit device placement',lineage:'load'});
      if(s.phase==='compute'+s.call||s.result)add('O'+s.call,'crate','O'+s.call+' · '+(s.result?'ready':'values'),1005,500,150,87,'runtime',{tag:s.result?'RESULT BUFFER':'NATIVE ARITHMETIC',lines:[s.id==='addition'?(s.call===2?'5, 6':'1, 2'):subject.op],origin:point(864,502),place:'result allocation',lineage:'store'});
      return list;
    }
    draw(item,target) {
      super.draw(item,target);
      if(target.id==='KIR1'&&this.state.concrete){
        const path=document.createElementNS('http://www.w3.org/2000/svg','path');
        path.setAttribute('d','M120 154V167M120 204V217');path.setAttribute('class','journey-def-use');item.g.append(path);
      }
      item.g.dataset.lineage=target.lineage||'';
      item.g.classList.toggle('lineage-selected',Boolean(target.lineage&&target.lineage===this.state.selection));
      if(target.lineage){item.g.setAttribute('role','button');item.g.setAttribute('tabindex','0');item.g.setAttribute('aria-label','Follow '+target.lineage+': '+target.label);item.g.setAttribute('aria-pressed',String(target.lineage===this.state.selection));}
    }
  }
  root.CompilerJourneyVisual=JourneyVisual;
})(window);
