/* Stable named actors: copied bytes, names, code, driver handles and buffers. */
(function(root){
  'use strict';
  const ns='http://www.w3.org/2000/svg',ease=t=>t*t*(3-2*t);
  function element(tag,attrs={},text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e;}
  class Visual{
    constructor(svg,reduced){this.svg=svg;this.layer=svg.querySelector('[data-actors]');this.reduce=reduced;this.actors=new Map();this.raf=0;this.paused=false;this.last=0;}
    text(id,text){this.svg.querySelector('#'+id).textContent=text;}
    targets(frame){const s=frame.state,o=frame.options,items=[],add=(id,color,title,lines,x,y,w=222,h=74)=>items.push({id,color,title,lines,x,y,w,h,opacity:1});
      add('A0','bytes','A0 · caller bytes',['12,288 bytes · compiled object','not the input tensor'],44,102);
      if(s.primitive)add('STATE','table','runtime slot 3 · modules[0]',['CompiledKernelsContext','borrowed engine reference'],1035,350,303,68);
      if(s.arguments)add('NAMES','bytes','kernelNames · ordered names',[frame.detail.object.requestedNames.join(' · ')],44,363,242,56);
      if(s.hash)add('HASH','namespace','Binary namespace',['8a3f10c492be760d'],44,196,222,49);
      if(s.copy&&!s.copyReleased&&!(o.route==='reuse'&&s.loadStatus==='success'))add('A1','bytes','A1 · owned byte copy',[s.bufferRetained?'retained ref(A1) in layer':'BufferRef owns copied bytes','same 12,288-byte contents'],s.bufferRetained?367:44,s.bufferRetained?196:265);
      if(s.namespace)add('JD1','namespace','JD1 · ORC namespace',[s.unit?'2 definitions registered':s.generator?'shared-library generator':'definitions not registered',s.dependencies?'links CompilerRT + mlirc':'dependencies pending'],367,101);
      if(o.route==='reuse')add('PREV','bytes','Prior object retained',['same hash · engine owns code'],367,196);
      if(s.interface)add(o.route==='reuse'?'PREVDEFS':'MU1','namespace',o.route==='reuse'?'Existing ready definitions':s.generator?'Dynamic symbol generator':s.unit?'MU1 · materialization recipe':'Object interface',[o.route==='reuse'?'prior linked code; no new MU1':s.generator?'exports from loaded .so':s.unit?'recipe: ObjectLayer.emit(A1)':'add_one_entry · shape_entry'],367,271);
      if(s.debugFile)add('OBJFILE','bytes','Temporary .o',['kept for debug information'],s.debugShared?44:367,s.debugShared?363:365,222,57);
      if(s.debugShared)add('SOFILE','code','Linked .so',['configured linker subprocess'],367,365,222,57);
      if(s.memory!=='empty')add('CODE1','code','CODE1 · host code',[frame.detail.resources.code?'linked · ready in engine':s.relocationPatched?'fixups applied · finalize pending':'allocated · work pending',s.externalResolved?'runtime_helper → R0':'object sections → memory'],704,103,271);
      if(s.materialized&&!s.executing)add('RELOC1','code','RELOC1 · object reference',[s.externalResolved?'runtime_helper → R0':'runtime_helper → unresolved',s.relocationPatched?'patch bytes written':'patch slot waits for its target'],704,282,271,68);
      if(s.lifetime)add('OWNERS','code','Engine owns · context borrows',['loaded code survives load return'],704,370,271,53);
      if(s.lookup)add('SYMBOL','namespace',s.lookup,[s.error?'lookup failed':s.resolved?'resolved entry available':s.interned?'mangled + interned':'name + JD1 search scope'],s.resolved&&!s.error?1035:704,s.resolved&&!s.error?277:197,244,56);
      if(s.table>=1)add('P0','table','0 · add_one_entry → P0',['0x00007f1200100000'],1035,101,303,68);
      if(s.table>=2)add('P1','table','1 · shape_entry → P1',['0x00007f1200100080'],1035,190,303,68);
      if(s.executing)add('CALL','code','Invoke P0 · host ABI',['void(void**) · fresh operand slots'],s.hostInvoked?704:1035,s.hostInvoked?298:302,s.hostInvoked?271:303,62);
      if(s.deviceImage)add('GPUIMAGE','driver','GPU image · separate bytes',[o.target==='hip'?'hsaco · AMD target':'cubin · NVIDIA target','module=demo_gpu_module'],s.deviceModule?368:44,s.deviceModule?638:577,263,69);
      if(s.deviceModule)add('DM1','driver','DM1 · device module',[o.target==='hip'?'opaque hipModule_t':'opaque CUmodule'],368,553,263,55);
      if(s.deviceFunction)add('DF1','driver','DF1 · device function',[o.target==='hip'?'opaque hipFunction_t':'opaque CUfunction','add_one_gpu · retained handle'],706,581,270,71);
      if(s.executing){add(s.repeat?'B2':'B1','buffer',s.repeat?'B2 · fresh input':'B1 · input',[s.repeat?'[10, 11, 12, 13]':'[0, 1, 2, 3]','tiny [2,2] · already on target'],1040,517,288,66);add(s.repeat?'O2':'O1','buffer',s.repeat?'O2 · fresh output':'O1 · output',[s.computeStage==='stored'?s.repeat?'[11, 12, 13, 14]':'[1, 2, 3, 4]':'[ —, —, —, — ]',s.complete?'completion observed':'contents/readiness pending'],1040,650,288,66);}
      if(s.launchPacked)add('ARGVALUES','buffer','args · value slots',['slot0 contains device pointer B1','slot1 contains device pointer O1'],44,577,250,70);
      if(s.submitted)add(s.repeat?'L2':'L1','stream',s.repeat?'L2 · next call':'L1 · launch submission',[s.complete?'completion observed':s.computeStage==='stored'?'device write · readiness separate':'S0 queue · pending'],s.computeStage==='stored'?1040:707,s.computeStage==='stored'?590:669,s.computeStage==='stored'?288:269,55);
      return items;
    }
    node(a){const g=element('g',{'data-actor':a.id,'data-color':a.color,role:'img','aria-label':a.title}),box=element('rect',{rx:8}),title=element('text',{x:12,y:24,class:'actor-title'}),l1=element('text',{x:12,y:45,class:'actor-line'}),l2=element('text',{x:12,y:62,class:'actor-line'});g.append(box,title,l1,l2);this.layer.append(g);return {g,box,title,l1,l2,x:a.x,y:a.y,opacity:0,target:a};}
    set(frame,animate){this.pause();const desired=this.targets(frame),present=new Set(desired.map(a=>a.id));for(const a of desired){let r=this.actors.get(a.id);if(!r){r=this.node(a);this.actors.set(a.id,r);if(a.id==='A1'){r.x=44;r.y=102;}if(a.id==='GPUIMAGE'){r.x=704;r.y=298;}}
        r.title.textContent=a.title;r.l1.textContent=a.lines[0]||'';r.l2.textContent=a.lines[1]||'';r.g.setAttribute('aria-label',a.title+' · '+a.lines.join(' · '));r.box.setAttribute('width',a.w);r.box.setAttribute('height',a.h);r.target=a;r.g.dataset.place=frame.event.focus;r.from={x:r.x,y:r.y,opacity:r.opacity};r.time=0;}
      for(const [id,r]of this.actors)if(!present.has(id)){r.from={x:r.x,y:r.y,opacity:r.opacity};r.target={...r.target,x:r.x,y:r.y,opacity:0};r.time=0;}
      this.text('mutex-label',frame.state.mutex?'mu · LOCK HELD':'mu · unlocked');this.svg.querySelector('#mutex-gate').dataset.locked=String(frame.state.mutex);
      this.text('gate-label',frame.state.gate);this.text('code-status',frame.state.memory==='empty'?'Code memory · not loaded':frame.detail.resources.code?'Code memory · ready':frame.state.relocationPatched?'Code memory · fixups applied':'Code memory · work pending');
      for(let i=0;i<2;i++){const filled=frame.state.table>i;const row=this.svg.querySelector('[data-table-index="'+i+'"]');row.dataset.occupied=String(filled);row.querySelector('text').textContent=filled?'':i+' · empty callable slot';}
      this.text('host-phase',frame.state.phase);this.text('driver-label',frame.options.target==='cpu'?'GPU driver · inactive (CPU target)':frame.options.target==='hip'?'AMD HIP driver · separate load path':'NVIDIA CUDA driver · separate load path');this.text('driver-status',frame.state.driver);this.text('stream-label',frame.state.submitted?'S0 · '+(frame.state.complete?'completed':'submitted; pending'):'S0 · no launch submitted');
      this.svg.dataset.table=String(frame.state.table);this.svg.dataset.error=String(frame.state.error);this.svg.dataset.complete=String(frame.state.complete);
      this.paused=false;this.last=performance.now();if(!animate||this.reduce.matches){for(const r of this.actors.values()){r.x=r.target.x;r.y=r.target.y;r.opacity=r.target.opacity;r.time=850;this.paint(r);}return;}this.schedule();
    }
    paint(r){r.g.setAttribute('transform',`translate(${r.x},${r.y})`);r.g.setAttribute('opacity',String(r.opacity));r.g.style.pointerEvents=r.opacity>.1?'auto':'none';}
    schedule(){if(!this.raf)this.raf=requestAnimationFrame(t=>this.tick(t));}
    tick(now){this.raf=0;if(this.paused)return;const dt=Math.min(60,Math.max(0,now-this.last));this.last=now;let active=false;for(const r of this.actors.values()){r.time=Math.min(850,r.time+dt);const t=ease(r.time/850);r.x=r.from.x+(r.target.x-r.from.x)*t;r.y=r.from.y+(r.target.y-r.from.y)*t;r.opacity=r.from.opacity+(r.target.opacity-r.from.opacity)*t;this.paint(r);if(r.time<850)active=true;}if(active)this.schedule();}
    pause(){this.paused=true;if(this.raf)cancelAnimationFrame(this.raf);this.raf=0;}
    resume(){if(!this.paused)return;this.paused=false;this.last=performance.now();if([...this.actors.values()].some(r=>r.time<850))this.schedule();}
  }
  root.KernelLoaderVisual=Visual;
})(window);
