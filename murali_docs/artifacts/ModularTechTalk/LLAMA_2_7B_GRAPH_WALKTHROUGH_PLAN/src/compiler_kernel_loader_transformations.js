/* Focused cutaways. Payload actors retain their DOM identity; copies get new IDs.
   Positions and timing are teaching geometry, never hardware measurements. */
(function(root){
  'use strict';
  const NS='http://www.w3.org/2000/svg',ease=t=>t*t*(3-2*t);
  function el(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
  class TransformVisual{
    constructor(svg,reduced){
      this.svg=svg;this.layer=svg.querySelector('[data-payloads]');this.edges=svg.querySelector('[data-connections]');this.backgrounds=svg.querySelector('[data-owners]');
      this.reduce=reduced;this.actors=new Map();this.raf=0;this.paused=true;this.last=0;this.lens='';
    }
    targets(f){
      const d=f.detail,s=f.state,o=f.options,items=[],a=(id,title,lines,x,y,w=300,h=92,color='bytes',extra={})=>items.push({id,title,lines,x,y,w,h,color,kind:'card',opacity:1,...extra});
      const panel=(id,title,x,y,w,h,color='bytes')=>a(id,title,[],x,y,w,h,color,{kind:'panel'});
      const cell=(id,value,label,x,y,color='buffer',seed=null,w=70)=>a(id,String(value??'—'),[label],x,y,w,67,color,{kind:'cell',seed});
      const wire=(id,points,label,x,y,color='table',dashed=false)=>items.push({id,kind:'edge',points,title:label,lines:[],x:0,y:0,w:0,h:0,labelX:x,labelY:y,color,dashed,opacity:1});
      const note=(id,text,x,y,color='table',w=600)=>a(id,text,[],x,y,w,35,color,{kind:'tag'});
      const code=()=>a('resident-code',o.target==='cpu'?'P0 · resident host routine':'P0 host routine → DF1 device function',
        ['Code and handles survive; this call has fresh data.'],36,497,1327,66,'code');
      const stateStrip=(y)=>{
        const states=['NeverSearched','Materializing','Resolved','Emitted','Ready'],cur=states.indexOf(d.orc.state);
        states.forEach((state,i)=>a('orc-state-'+state,state,[],35+i*269,y,251,35,i<=cur?'table':'namespace',{kind:'tag',muted:i>cur,active:i===cur}));
      };
      const tensor=(id,title,values,x,y,address,ready)=>{
        panel('tensor-'+id,title,x,y,357,205,'buffer');
        note('tensor-address-'+id,address,x+14,y+40,'buffer',328);
        values.forEach((v,i)=>cell(id+'-cell-'+i,v,'f32 · '+i,x+14+i*82,y+90));
        note('tensor-ready-'+id,ready,x+14,y+165,'buffer',328);
      };
      switch(d.lens){
      case 'bytes':
        panel('caller-object','A0 · caller-owned compiled object',30,100,430,375);
        panel('owned-object','A1 · BufferRef-owned copy',938,100,430,375);
        a('byte-copy-action',s.copy?'Buffer::get · copy bytes':'Awaiting owned byte copy',
          ['A0 stays with the caller.',s.copy?'A1 contains the same 12,288 bytes.':'ArrayRef / StringRef only view A0.'],500,205,390,110,'table');
        d.object.sections.forEach((name,i)=>{
          a('A0-section-'+i,name,[['code bytes','constant data','names + offsets','relocation records'][i]],50,154+i*59,389,51,'bytes',{compact:true});
          if(s.copy)a('A1-section-'+i,name,[['code bytes','constant data','names + offsets','relocation records'][i]],958,154+i*59,389,51,'bytes',{compact:true,seed:{x:50,y:154+i*59}});
        });
        d.object.magic.forEach((b,i)=>{
          cell('A0-header-'+i,b,'ELF byte',50+i*92,393,'bytes',null,82);
          if(s.copy)cell('A1-header-'+i,b,'same byte',958+i*92,393,'bytes',{x:50+i*92,y:393},82);
        });
        wire('copy-edge',[[462,178],[938,178]],'copy into owned storage',526,162,'bytes');
        if(s.hash)note('namespace-hash','xxh3(all bytes) → 8a3f10c492be760d',35,478,'namespace',850);
        if(s.arguments)note('ordered-names','names[0]=add_one_entry · names[1]='+d.object.requestedNames[1],35,516,'table',1300);
        break;
      case 'namespace':
        panel('retained-storage','OBJECT LAYER · retained inputs',30,103,342,364);
        panel('orc-namespace','ENGINE · JD1 namespace',410,103,585,364,'namespace');
        panel('registration-decision','REGISTRATION RESULT',1033,103,335,364,'table');
        a('A1-reference','A1 · new owned object storage',
          [s.copyReleased?'Local A1 reference released.':o.route==='debug'?'Debug copy supplies temporary .o bytes.':s.bufferRetained?'objectBuffers retains ref(A1)':'A1 copied; layer reference pending.',
            'Copying a BufferRef retains a reference.'],50,161,302,110,'bytes',{muted:s.copyReleased});
        if(o.route==='reuse')a('Aprev-reference','Aprev · earlier object retained',
          ['Same binary hash; not the new A1.','Engine retains previously linked code.'],50,297,302,106);
        else if(s.bufferRetained)a('resource-tracker','ResourceTracker',
          ['JD1.defaultResourceTracker','Tracks registered JIT resources.'],50,297,302,106,'namespace');
        a('namespace-key','JD1: 8a3f10c492be760d',[s.namespace?'namespace exists':'namespace absent',
          s.mutex?'mu: LOCK HELD':'mu: unlocked'],433,154,538,96,'namespace');
        a('registered-definitions',o.route==='reuse'?'JD1 · existing ready definitions':s.unit?'MU1 · definitions + materialization recipe':s.generator?'DynamicLibrarySearchGenerator':'No definitions registered',
          [s.unit?'add_one_entry | shape_entry':s.generator?'exports from the loaded .so':'table has no callable addresses',
            o.route==='reuse'?'No new MU1 or object linking.':s.unit?'callback → ObjectLayer.emit(A1)':'host loading work only'],433,273,538,103,'namespace');
        note('link-order','Link order → CompilerRT → mlirc',433,397,'namespace',538);
        a('registration-gate',s.reused?'HIT · reuse JD1':s.generator?'Shared library route':s.unit?'Definitions registered':s.namespace?'Namespace created':'Check name under mu',
          [s.reused?'Skip new object + MU1 + linking.':s.generator?'Configured linker → .so generator.':'Later lookup can request materialization.',
            'New context table: '+s.table+' / 2 entries'],1052,171,296,145,'table');
        if(s.debugFile)a('debug-object-file','Temporary .o',[s.debugShared?'configured linker → temporary .so':'recognized object bytes written to disk'],1052,344,296,69,'bytes');
        wire('retain-reference',[[372,282],[410,282]],'ref',378,267,'bytes',true);
        break;
      case 'linking':
        panel('object-cutaway','A1 · retained ELF object',30,94,302,366);
        panel('link-graph','JITLink · LinkGraph',364,94,454,366,'namespace');
        panel('host-memory','CODE1 · host process memory',850,94,518,378,'code');
        a('object-text','.text',['compiled code bytes'],49,146,264,66);
        a('object-rodata','.rodata',['pointer slot: 00 00 00 00 ...'],49,238,264,66);
        a('object-relocations','.rela.rodata',['offset 0x20 → runtime_helper','edge kind: Pointer64 · addend 0x30'],49,330,264,97);
        if(d.link.graph){
          a('graph-text','T0 · .text block',['add_one_entry: +0x00','shape_entry: +0x80'],384,146,414,91,'namespace');
          a('graph-data','D0 · .rodata block',['pointer64 slot at D0 + 0x20'],384,269,414,66,'namespace');
          a('dead-block',d.link.pruned?'Tdead · pruned':'Tdead · unreachable local block',
            [d.link.pruned?'Removed; occupies no linked slot.':'Illustrative block, not an add-one dump.'],
            d.link.pruned?49:384,d.link.pruned?485:365,d.link.pruned?285:414,60,'namespace',{muted:d.link.pruned});
          wire('graph-parse',[[333,182],[364,182]],'decode',333,166,'namespace');
        }
        if(d.link.allocated){
          a('memory-text','.text · '+d.link.textProtection,
            ['base='+d.link.textAddress,d.link.copied?'':'address space assigned; copy pending'],
            873,146,472,92,'code');
          a('memory-rodata','.rodata · '+d.link.dataProtection,
            ['base='+d.link.dataAddress,'fixup address='+d.link.fixup.slotAddress],873,258,472,87,'code');
          wire('allocation-edge',[[819,188],[850,188]],'layout',817,173,'code');
        }
        if(d.link.copied){
          a('working-text-bytes','T0 bytes · copied, unchanged',[],887,207,444,31,'bytes',
            {kind:'tag',seed:{x:49,y:146}});
          note('patch-byte-label','D0 + 0x20 · eight little-endian bytes',874,355,'code',472);
          (d.link.patched?d.link.fixup.after:d.link.fixup.before).forEach((b,i)=>
            cell('patch-byte-'+i,b,'+'+i,879+i*58,395,d.link.patched?'table':'code',{x:49+i*29,y:269},50));
        }
        if(d.link.external)a('external-symbol','R0 · runtime_helper',
          [d.link.fixup.targetAddress,'Resolved through dependency link order.'],384,470,414,80,'namespace');
        if(s.relocationComputed||d.link.patched)note('fixup-formula',
          d.link.fixup.targetAddress+' + '+d.link.fixup.addend+' = '+d.link.fixup.value,851,484,'table',518);
        if(d.link.patched)wire('resolved-fixup',[[798,507],[829,507],[829,438],[873,438]],'write pointer',812,470,'table');
        stateStrip(558);
        break;
      case 'symbols':
        panel('name-scope','INPUT · name + search scope',30,103,386,350,'namespace');
        panel('resolved-symbol','ORC · symbol address',454,103,462,350,'code');
        panel('callable-table','CONTEXT · indexed callable table',954,103,414,350,'table');
        a('symbol-query',d.orc.name,['search order: [JD1]',s.interned?'target linker spelling + intern pool':'StringRef supplied by the caller'],
          50,160,346,107,'namespace');
        if(s.interned)a('interned-symbol',(d.orc.name==='shape_entry'?'SYMBOL2':s.error?'SYMBOL3':'SYMBOL1')+' · interned name',
          ['ELF spelling: '+d.orc.name,'target prefix may change the spelling'],50,311,346,97,'namespace');
        a('lookup-gate',s.error?'ERROR · missing_entry':d.orc.canReturn?'Ready · lookup may return':'Lookup waits for Ready',
          [s.error?'Earlier table append remains.':'SymbolState: '+d.orc.state,
            'Minimum requested state: Ready'],477,158,416,107,s.error?'bytes':'code');
        if(d.orc.canReturn||s.resolved)a('resolved-address',
          s.error?'Earlier P0 · remains in the table':s.resolved>1?'P1 · CompiledFunc.fn':'P0 · CompiledFunc.fn',
          [s.resolved>1?'0x00007f1200100080':'0x00007f1200100000','Wrapper contains a host address.'],477,312,416,97,'code');
        for(let i=0;i<2;i++){
          const entry=d.table.entries[i];
          a('callable-entry-'+i,entry?i+' · '+entry.name:i+' · empty',
            [entry?entry.address:'No pointer stored yet.'],976,164+i*112,370,89,'table',{muted:!entry});
        }
        note('table-next-index','kernelIndex='+d.table.nextIndex+' · append only in name order',976,396,'table',371);
        if(d.table.entries.length)wire('address-to-slot',[[917,328],[934,328],[934,204],[954,204]],'store ref(P0)',870,475,'table',true);
        stateStrip(522);
        break;
      case 'invoke':
        panel('indexed-selection','MGP · index selection',30,103,374,328,'table');
        panel('host-abi-slots','HOST ABI · async-value slots',442,103,480,328,'buffer');
        panel('native-entry','HOST · native routine',960,103,408,328,'code');
        a('runtime-indices','runtimeSlot=3 · moduleIndex=0',
          ['KernelExecutionState.modules[0]','functionIndex=0 → kernels[0]'],50,155,334,111,'table');
        a('selected-host-pointer','P0 · add_one_entry',['0x00007f1200100000','getKernelAsyncABI(0) → void(void**)'],50,299,334,93,'table');
        a('host-argument-vector','kernelInputs: void**',
          [s.hostInvoked?'Fresh operand / result-value addresses':'No routine invocation yet.',
            'Generated host ABI includes runtime values.','These slots differ from GPU kernelParams.'],465,155,434,126,'buffer');
        a('host-execution',s.hostInvoked?'Invoke P0(kernelInputs.data())':'Loaded routine · not invoked',
          [s.repeat?'Second call uses B2 / O2.':'This call uses B1 / O1.',
            o.target==='cpu'?'foreach[target="cpu"]':'foreach[target="gpu"] → DeviceContext'],982,155,364,126,'code');
        if(s.hostInvoked)wire('host-abi-invoke',[[899,215],[982,215]],'call',927,197,'code');
        wire('index-reference',[[405,341],[424,341],[424,307],[960,307]],'ref(P0); code remains resident',459,329,'table',true);
        note('ownership-lifetime',s.repeat?'M1 / P0 / DF1 retained · B2 / O2 fresh':'Engine owns code · context borrows engine · table stores pointers',
          34,464,'code',1327);
        note('load-not-whole-init','load success makes this module chain ready; overall initialization may have further work.',34,509,'table',1327);
        break;
      case 'device':
        panel('gpu-image-input','HOST ROUTINE · compiled GPU image',30,101,371,395,'driver');
        panel('device-cache-key','DEVICE · function cache key',439,101,489,395,'namespace');
        panel('vendor-handles',o.target==='hip'?'AMD HIP · opaque handles':'NVIDIA CUDA · opaque handles',966,101,402,395,'driver');
        a('GPUIMAGE',d.device.image,['Separate from A1 host object bytes.',
          'module: demo_gpu_module','entry: add_one_gpu'],52,156,327,123,'driver');
        a('driver-bridge','Mojo → C ABI → C++',
          ['_load_function(...)',
            'loadFunctionWithDebugName · C ABI',
            'virtual DeviceContext::loadFunction'],52,320,327,124,'driver');
        d.device.keyFields.forEach((text,i)=>note('device-key-'+i,text,461,155+i*42,'namespace',445));
        note('device-cache-verdict',!d.device.key?'Cache lookup pending':d.device.cacheHit?'HIT · vendor load calls skipped':'MISS · vendor module/function load',461,386,'table',445);
        if(s.deviceModule)a('device-module-handle','DM1 · '+KernelLoaderModel.TARGETS[o.target].MODULE_TYPE,
          [d.device.cacheHit?'retained cached module handle':KernelLoaderModel.TARGETS[o.target].MODULE_API,
            'Driver-managed module; not a host pointer.'],988,156,358,119,'driver');
        if(s.deviceFunction)a('device-function-handle','DF1 · '+KernelLoaderModel.TARGETS[o.target].FUNCTION_TYPE,
          ['name=add_one_gpu',d.device.cacheHit?'cached handle retained':KernelLoaderModel.TARGETS[o.target].FUNCTION_API],988,309,358,113,'driver');
        if(d.device.cached)wire('device-cache-reference',[[928,350],[946,350],[946,365],[966,365]],'retained',916,293,'driver',true);
        note('vendor-loading-limit','The driver decides when code becomes resident; module/function handles do not imply a launch.',
          35,529,'driver',1325);
        break;
      case 'launch':
        panel('launch-host-slots','HOST MEMORY · argument pointers',30,100,483,401,'buffer');
        panel('driver-launch-command','DRIVER · launch descriptor',550,100,354,401,'stream');
        panel('launch-device-memory','GPU · per-call allocations',941,100,427,401,'buffer');
        d.launch.arguments.forEach((arg,i)=>{
          const y=157+i*162;
          a('kernel-param-'+i,'args['+i+'] = &'+arg.slot,
            [arg.slotAddress],51,y,440,70,'buffer');
          a('argument-slot-'+i,arg.slot+' contains a device pointer',
            [arg.value+' → '+arg.referent],51,y+88,440,60,'buffer',{compact:true});
          wire('slot-indirection-'+i,[[280,y+71],[280,y+88]],'dereference',300,y+84,'buffer');
          a('device-allocation-'+i,arg.referent+' · '+(i?'output':'input'),
            [arg.value,i?d.data.stored?JSON.stringify(d.data.output):'[ —, —, —, — ]':JSON.stringify(d.data.input)],
            963,y+77,383,77,'buffer');
          wire('device-pointer-'+i,[[491,y+118],[533,y+118],[533,y+153],[928,y+153],[928,y+116],[963,y+116]],
            'device address',691,y+146,'buffer',true);
        });
        a('launch-config','DF1 · stream S0',
          ['grid=(1,1,1)','block=(32,1,1) · shared=0','args: void** · count=2'],571,154,312,127,'stream');
        a('stream-command',d.launch.submitted?'L'+d.launch.call+' · submitted to S0':'S0 · no launch submitted',
          [d.launch.submitted?KernelLoaderModel.TARGETS[o.target].LAUNCH_API:'Argument vector prepared first.',
            d.launch.complete?'Completion observed.':'Return / submission does not finish O'+d.launch.call+'.'],571,341,312,123,'stream');
        note('argument-pack-boundary','Two-pointer device ABI illustration; the generated AddOne pack may contain further fields.',
          35,534,'buffer',1325);
        break;
      case 'arithmetic': {
        const call=d.data.call;
        tensor(d.data.inputId,d.data.inputId+' · '+d.data.location,d.data.input,35,117,d.data.inputAddress,'Input stays in its allocation.');
        panel('register-bank-'+call,o.target==='cpu'?'HOST · working values':'DEVICE · working values',520,117,357,205,'code');
        tensor(d.data.outputId,d.data.outputId+' · output allocation',d.data.output,1005,117,d.data.outputAddress,
          d.data.complete?'Complete; safe under runtime ordering.':d.data.stored?'Written; completion is still pending.':'No output values written.');
        note('numeric-source','x.load[width](idx) + 1',535,160,'code',327);
        if(d.data.loaded)d.data.registers.forEach((v,i)=>cell('register-'+call+'-'+i,v,
          d.data.added?d.data.input[i]+' + 1':'load',534+i*82,207,'code',{x:49+i*82,y:207}));
        if(d.data.stored)d.data.output.forEach((v,i)=>cell('store-'+call+'-'+i,v,'store',1019+i*82,207,'buffer',{x:534+i*82,y:207}));
        wire('load-values',[[394,351],[520,351]],'load: copy values',396,333,'code');
        wire('store-values',[[879,351],[1005,351]],'store: copy values',879,333,'buffer');
        a('numeric-transformation',d.data.stored?'STORE → output values':d.data.added?'ADD → each value increases by one':d.data.loaded?'LOAD → working values':'Input values ready',
          [JSON.stringify(d.data.input)+' → '+JSON.stringify(d.data.expected),
            'Four float32 elements shown; source graph is [5,10].'],35,379,1327,93,'table');
        code();
        break;
      }
      case 'completion':
        tensor(d.data.outputId,d.data.outputId+' · '+d.data.location,d.data.output,35,142,d.data.outputAddress,'Values remain in the target allocation.');
        panel('completion-gate','RUNTIME · completion / ordering',520,142,357,205,'stream');
        a('completion-status',d.data.complete?'Completion observed':'Awaiting completion',
          [o.target==='cpu'?'CPU runtime task / dependency ordering':'S0 + event / dependency ordering',
            'Written data; readiness is separate.'],540,200,317,102,'stream');
        if(o.target==='cpu'){
          panel('cpu-readable-result','CPU RESULT · same allocation',1005,142,357,205,'buffer');
          a('cpu-output-reference','ref('+d.data.outputId+') · already on CPU',
            [JSON.stringify(d.data.output),'No device-to-host copy required.'],1025,201,317,106,'buffer');
        }else tensor(d.data.hostId,d.data.hostId+' · CPU-readable copy',
            d.data.hostCopy?d.data.expected:[null,null,null,null],1005,142,'result.to(CPU())',
            d.data.hostCopy?'Host values can be read as NumPy.':'No host copy shown until requested.');
        if(d.data.hostCopy)d.data.expected.forEach((v,i)=>cell('host-copy-'+d.data.call+'-'+i,v,'copy',1019+i*82,232,'buffer',{x:49+i*82,y:232}));
        wire('completion-ordering',[[393,380],[520,380]],'observe ordering',394,362,'stream');
        if(d.data.hostCopy)wire('explicit-copy',[[879,380],[1005,380]],'explicit copy',879,362,'buffer');
        note('completion-limit','No device-wide barrier is implied; the example explicitly requests result.to(CPU()).',35,409,'stream',1327);
        code();
        break;
      }
      return items;
    }
    node(a){
      const g=el('g',{'data-payload':a.id,'data-color':a.color,role:'img'});
      let shape,title,lines=[];
      if(a.kind==='edge'){shape=el('path',{class:'payload-wire','marker-end':'url(#transform-arrow)'});title=el('text',{class:'payload-edge-label'});this.edges.append(g);}
      else{
        shape=el('rect',{rx:a.kind==='panel'?10:7,class:'payload-'+a.kind});
        title=el('text',{class:a.kind==='cell'?'payload-number':a.kind==='panel'?'payload-owner':'payload-title'});
        for(let i=0;i<4;i++){const t=el('text',{class:'payload-line'});lines.push(t);}
        (a.kind==='panel'?this.backgrounds:this.layer).append(g);
      }
      g.append(shape,title,...lines);
      return {g,shape,title,lines,x:a.x,y:a.y,opacity:0,target:a,time:0};
    }
    set(f,animate){
      this.pause();const desired=this.targets(f),visible=new Set(desired.map(a=>a.id)),newLens=f.detail.lens!==this.lens;
      for(const a of desired){
        let r=this.actors.get(a.id);if(!r){r=this.node(a);this.actors.set(a.id,r);}
        const entering=r.opacity<.01;
        if(entering&&a.seed){r.x=a.seed.x;r.y=a.seed.y;}
        else if(newLens){r.x=a.x;r.y=a.y;r.opacity=1;}
        r.target=a;r.from={x:r.x,y:r.y,opacity:r.opacity};r.time=0;
        r.g.dataset.color=a.color;
        r.g.dataset.muted=String(!!a.muted);r.g.dataset.active=String(!!a.active);
        r.g.setAttribute('aria-label',a.title+' · '+a.lines.join(' · '));
        r.delayedTitle=animate&&!this.reduce.matches&&a.kind==='cell'&&(
          f.detail.lens==='arithmetic'&&f.detail.data.stored&&a.id.startsWith(f.detail.data.outputId+'-cell-')
          ||f.detail.lens==='completion'&&f.detail.data.hostCopy&&a.id.startsWith(f.detail.data.hostId+'-cell-'));
        r.title.textContent=r.delayedTitle?'—':a.title;
        if(a.kind==='edge'){
          r.shape.setAttribute('d',a.points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '));
          r.shape.setAttribute('stroke-dasharray',a.dashed?'6 5':'none');
          r.title.setAttribute('x',a.labelX);r.title.setAttribute('y',a.labelY);
        }else{
          r.shape.setAttribute('width',a.w);r.shape.setAttribute('height',a.h);
          const centered=a.kind==='cell';
          r.title.setAttribute('x',centered?a.w/2:14);
          r.title.setAttribute('y',centered?35:a.kind==='tag'?23:25);
          r.title.setAttribute('text-anchor',centered?'middle':'start');
          r.lines.forEach((t,i)=>{
            t.textContent=a.lines[i]||'';t.setAttribute('x',centered?a.w/2:14);
            t.setAttribute('y',centered?56:a.compact?43:48+i*20);
            t.setAttribute('text-anchor',centered?'middle':'start');
          });
          for(const t of [r.title,...r.lines]){
            t.style.fontSize='';const actual=t.getComputedTextLength(),limit=a.w-27;
            if(actual>limit){const size=parseFloat(getComputedStyle(t).fontSize);t.style.fontSize=Math.max(11,size*limit/actual)+'px';}
          }
        }
      }
      for(const [id,r]of this.actors)if(!visible.has(id)){r.from={x:r.x,y:r.y,opacity:r.opacity};r.target={...r.target,x:r.x,y:r.y,opacity:0};r.time=0;if(newLens)r.opacity=0;}
      this.lens=f.detail.lens;this.svg.dataset.lens=this.lens;this.svg.dataset.patched=String(f.detail.link.patched);
      this.svg.dataset.symbolState=f.detail.orc.state;this.svg.dataset.compute=f.state.computeStage;
      this.svg.dataset.hostCopy=String(f.detail.data.hostCopy);
      const title=KernelLoaderModel.LENSES.find(([key])=>key===this.lens)[1];
      this.svg.querySelector('#transform-title').textContent=title;
      this.svg.querySelector('#transform-kind').textContent=['bytes','linking'].includes(this.lens)?
        'ELF64 / x86-64 illustration · '+f.event.evidence:f.event.evidence+' · illustrative payload / addresses';
      this.paused=false;this.last=performance.now();
      if(!animate||this.reduce.matches){
        for(const r of this.actors.values()){r.x=r.target.x;r.y=r.target.y;r.opacity=r.target.opacity;r.time=1050;this.paint(r);}
        return;
      }
      this.schedule();
    }
    paint(r){if(r.delayedTitle&&r.time>=1050){r.title.textContent=r.target.title;r.delayedTitle=false;}r.g.setAttribute('transform','translate('+r.x+','+r.y+')');r.g.setAttribute('opacity',String(r.opacity));r.g.setAttribute('aria-hidden',String(r.opacity<.01));r.g.style.pointerEvents=r.opacity>.1?'auto':'none';}
    tick(now){
      this.raf=0;if(this.paused)return;const dt=Math.min(60,Math.max(0,now-this.last));this.last=now;let active=false;
      for(const r of this.actors.values()){
        r.time=Math.min(1050,r.time+dt);const t=ease(r.time/1050);
        r.x=r.from.x+(r.target.x-r.from.x)*t;r.y=r.from.y+(r.target.y-r.from.y)*t;r.opacity=r.from.opacity+(r.target.opacity-r.from.opacity)*t;
        this.paint(r);if(r.time<1050)active=true;
      }
      if(active)this.schedule();
    }
    schedule(){if(!this.raf)this.raf=requestAnimationFrame(t=>this.tick(t));}
    pause(){this.paused=true;if(this.raf)cancelAnimationFrame(this.raf);this.raf=0;}
    resume(){if(!this.paused)return;this.paused=false;this.last=performance.now();if([...this.actors.values()].some(r=>r.time<1050))this.schedule();}
  }
  root.KernelLoaderTransformVisual=TransformVisual;
})(window);
