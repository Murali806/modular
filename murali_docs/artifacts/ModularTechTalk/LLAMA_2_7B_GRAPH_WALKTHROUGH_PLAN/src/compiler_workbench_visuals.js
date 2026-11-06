/* Persistent, labeled SVG actors. Motion illustrates transformations, not a trace. */
(function(root) {
  'use strict';
  const ns='http://www.w3.org/2000/svg', clamp=n=>Math.max(0,Math.min(1,n));
  const point=(x,y)=>({x,y}), ease=t=>t*t*(3-2*t);
  function el(tag,attrs={},text) { const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e; }
  function a(id,kind,label,x,y,w=160,h=90,color='artifact',extra={}) { return {id,kind,label,x,y,w,h,color,opacity:1,...extra}; }
  function route(points,t) {
    const lengths=points.slice(1).map((p,i)=>Math.hypot(p.x-points[i].x,p.y-points[i].y));
    let d=lengths.reduce((x,y)=>x+y,0)*t;
    for(let i=0;i<lengths.length;i++) { if(d<=lengths[i]||i===lengths.length-1) { const k=lengths[i]?d/lengths[i]:1;return point(points[i].x+(points[i+1].x-points[i].x)*k,points[i].y+(points[i+1].y-points[i].y)*k); }d-=lengths[i]; }
    return points[0];
  }
  class WorkbenchVisual {
    constructor(node,reduced) { this.node=node;this.kind=node.dataset.lesson;this.layer=node.querySelector('[data-actors]');this.reduced=reduced;this.actors=new Map();this.frozenAt=null; }
    slots(name,values,missing=false) {
      const group=this.node.querySelector('[data-slots="'+name+'"]');if(!group)return;
      group.dataset.values=JSON.stringify(values);
      group.querySelectorAll('[data-slot]').forEach((slot,i)=>{
        const filled=values[i]!==undefined&&values[i]!==null;
        slot.dataset.occupied=String(filled);slot.dataset.missing=String(missing&&!filled);
        slot.setAttribute('aria-label',filled?String(values[i]):missing?'missing binding':'empty slot');
        slot.querySelector('text').textContent=filled?'':missing?'W_b ?':'—';
      });
    }
    text(selector,value) { const e=this.node.querySelector(selector);if(e)e.textContent=value; }
    targets(s) {
      const list=[],kind=this.kind;
      if(kind==='graph') {
        list.push(a('source-model','file','model.py',65,150,198,174,'python',{tag:'SOURCE',lines:['typed input contract','A + B → multiply C','symbolic builder']}));
        if(s.graph) list.push(a('G1','file','G1 · Graph',382,143,327,182,'mlir',{tag:'MO / MLIR',lines:['dtype / shape / device','symbolic values','def-use relationships'],origin:point(180,230)}));
        if(s.ops>=1) list.push(a('ADD1','tile','ADD1 · A + B',770,173,174,56,'mojo',{origin:point(186,211),via:[point(312,238),point(741,238)],place:'graph'}));
        if(s.ops>=2) list.push(a('MUL1','tile','MUL1 · × C',770,269,174,56,'backend',{origin:point(186,254),via:[point(312,336),point(741,336)],place:'graph'}));
        if(s.contract) list.push(a('signature','handle','mo.output / signature',395,350,260,44,'engine',{origin:point(262,367),place:'contract'}));
        if(s.sealed) list.push(a('verified','stamp','VERIFIED',776,350,161,43,'runtime',{origin:point(658,372),place:'verified'}));
        list.push(a('idle-device','chip','IDLE',1050,148,106,90,'hardware',{lines:['no submissions'],tag:'TARGET'}));
        this.text('[data-seal-label]',s.contract?'':'Output contract · open');
      } else if(kind==='lower') {
        list.push(a('G1','file','G1 · retained',50,152,192,199,'python',{tag:'MO / MLIR',lines:['ADD1 → MUL1','caller-owned graph','not rewritten']}));
        if(s.clone) list.push(a('C1','file','C1 · '+(s.level<0?'module':s.level===4?'MGP + routines':root.CompilerWorkbenchModel.LOWER_STAGES[s.level][1]),386,151,265,204,'mlir',{tag:'COMPILER IR',lines:[s.level<0?'native module clone':root.CompilerWorkbenchModel.LOWER_STAGES[s.level][2]],origin:point(145,236),via:[point(280,233)],place:s.phase}));
        if(s.clone&&s.level<=0) {
          list.push(a('C1-ADD1','tile','ADD1',433,250,120,33,'mojo',{origin:point(180,228),place:'C1'}));
          list.push(a('C1-MUL1','tile','MUL1',433,294,120,33,'backend',{origin:point(180,266),place:'C1'}));
        } else if(s.clone) {
          if(s.fused) list.push(a('C1-F1',s.level===4?'file':'tile',s.level===4?'emitted Mojo routine':s.level===3?'primitive → F1':s.level===2?'kernel call → F1':'ADD1 + MUL1',408,s.level===1?270:238,214,s.level===4?92:38,'mojo',{origin:point(493,258),tag:s.level===4?'ADD IMPLEMENTATION':null,place:s.level===1?'grouped-body':s.level===2?'kernel-call':s.level===3?'primitive':'emitted-routine',description:'Illustrative routine F1; provenance ADD1 and MUL1'}));
          else {
            list.push(a('C1-ADD1','tile','body / call ADD1',408,258,214,32,'mojo',{place:'separate'}));
            list.push(a('C1-MUL1','tile','body / call MUL1',408,300,214,32,'backend',{place:'separate'}));
          }
        }
        if(s.clone&&s.level===3) ['input ref','dependency','output ref'].forEach((name,i)=>list.push(a('plan-slot-'+i,'tile',name,401+i*78,306,72,31,'buffer',{origin:point(603,264),place:'MGP plan'})));
        s.history.forEach((stage,i)=>list.push(a('view-'+i,'file',stage,846+(i%2)*151,135+Math.floor(i/2)*87,129,75,'artifact',{tag:'VIEW',lines:['C1 snapshot'],origin:point(518,241),via:[point(810,239)],place:'snapshot',delay:(i===s.history.length-1?80:0)})));
        if(s.clone) list.push(a('unused-routine','tile','unused symbol',s.discarded?1029:654,s.discarded?342:265,s.discarded?92:112,31,'driver',{via:[point(781,382),point(1000,382)],place:s.discarded?'DCE bin':'C1'}));
        if(s.deviceFinalized) list.push(a('device-graph-seal','stamp','FINALIZED',655,316,118,47,'sync',{tag:'DEVICE GRAPH',origin:point(720,349),place:'device-graph-pass',description:'MGP device-graph finalization pass; not Python Graph.output or device execution'}));
        this.text('[data-pass-label]',s.phase==='ready'?'Next: kernel JIT → runtime calls':s.clone?'Active: '+s.phase.toUpperCase():'Awaiting C1');
      } else if(kind==='mojo') {
        const ownership=s.lab==='ownership';
        this.node.querySelector('[data-lane-view]').toggleAttribute('hidden',ownership);
        this.node.querySelector('[data-owner-view]').toggleAttribute('hidden',!ownership);
        if(ownership) {
          list.push(a('owner-handle','handle','B1 owner · '+s.owner,s.moved?420:54,s.moved?326:194,206,74,'mojo',{tag:'OWNS',lines:['same allocation B1'],via:[point(290,363)],place:s.owner}));
          if(s.borrow) list.push(a('borrow-'+s.borrow,'handle',s.borrow+' reference → B1',415,129,248,85,'buffer',{tag:'BORROW',lines:[s.borrow==='imm'?'read access':'exclusive mutation'],origin:point(242,242),place:'borrow'}));
          list.push(a('lifetime-gate','stamp',s.blocked?'BORROW CONFLICT':s.checked?'LEGAL ACCESS':'CHECK LIFETIMES',414,244,263,58,s.blocked?'hardware':'sync',{description:'Compile-time program legality gate',place:s.blocked?'blocked':'checking'}));
          this.slots('owned-cells',s.values);
          s.values.forEach((v,i)=>list.push(a('B1-cell-'+i,'cell',String(v),803+i*83,216,71,40,'buffer',{place:'B1',description:'Stored cell in the same B1 allocation'})));
          this.node.querySelector('[data-ref-edge]').toggleAttribute('hidden',!s.borrow);
          this.node.querySelector('[data-ref-edge]').setAttribute('d','M663 169H752V236H803');
        } else {
          list.push(a('source-add','file','Add · typed Mojo',49,147,215,195,'mojo',{tag:'MOJO',lines:['dtype + width parameters','typed inputs / output','registered implementation']}));
          if(s.level>=1) list.push(a('K1-IR','file',s.level===1?'LIT':s.level<=3?'KGEN / POP':s.level===4?'Concrete KGEN':'LLVM dialect',383,135,246,209,'mlir',{tag:'MLIR',lines:[s.concrete?'WIDTH = '+s.width:'WIDTH still parametric',s.checked?'lifetimes checked':'source semantics','types / operations'],origin:point(164,240),place:s.phase}));
          list.push(a('width','tile','WIDTH = '+s.width,391,360,174,33,'backend',{place:s.concrete?'specialized':'parameter'}));
          this.slots('lane-input',s.values);this.slots('lane-output',s.results);
          s.values.forEach((v,i)=>list.push(a('lane-in-'+i,'cell',String(v),790+i*45,186,39,40,'buffer',{origin:point(238,300),place:'CPU lane'})));
          s.results.forEach((v,i)=>list.push(a('lane-out-'+i,'cell',String(v),790+i*45,282,39,40,'runtime',{origin:point(809+i*45,206),place:'native-result',delay:Math.floor(i/s.width)*100})));
          const group=this.node.querySelector('[data-lane-groups]');group.replaceChildren();
          for(let i=0;i<8;i+=s.width) group.append(el('path',{d:'M'+(790+i*45)+' 330v8h'+(s.width*45-6)+'v-8',class:'lane-bracket'}));
          this.text('[data-bundles]',s.results.length?'Illustrative native bundles: '+s.bundles:'Compile-time WIDTH = '+s.width);
          if(s.checked) list.push(a('type-check','stamp','CHECKED',650,298,67,65,'sync'));
        }
      } else if(kind==='target') {
        if(!s.llvm) list.push(a('L1','file','LLVM dialect',48,135,202,215,'mlir',{tag:'.MLIR VIEW',lines:['llvm operations in MLIR','target metadata','ready for translation']}));
        else {
          list.push(a('L1','file','MLIR retained',44,351,210,46,'mlir',{tag:'VIEW',place:'source'}));
          list.push(a('LLVM1','file','LLVM module',47,130,206,203,'mlir',{tag:'.LL VIEW',lines:['llvm::Module','target = '+s.target.arch,'native codegen input'],origin:point(145,241),place:'LLVM'}));
        }
        if(s.selected) list.push(a('backend','chip',s.targetKey==='cpu'?'HostBackend':s.targetKey==='cuda'?'NVPTXBackend':'AMDGPUBackend',326,140,248,154,'backend',{tag:'SELECTED',lines:[s.target.arch,s.targetKey==='cuda'?'LLVM NVPTX':s.targetKey==='rocm'?'LLVM AMDGPU':'LLVM x86'],origin:point(450,365),place:s.targetKey}));
        if(s.selected) list.push(a('tool','tool',s.targetKey==='cuda'?(s.assembler==='override'?'Configured CUDA path':'CUDA CompilationDevice'):'LLVM MC / linker',646,129,246,76,'backend',{tag:'HOST CODEGEN',lines:[s.phase==='assemble'?'assembling target code':'selected toolchain'],place:s.phase}));
        if(s.assembly) list.push(a('ASM1','file',s.target.assembly,s.mode==='assembly'?966:656,s.mode==='assembly'?138:227,s.mode==='assembly'?189:237,s.mode==='assembly'?198:117,'artifact',{tag:s.targetKey==='cuda'?'PTX TEXT':'ASSEMBLY TEXT',lines:[s.targetKey==='cuda'?'.target '+s.target.arch:'target: '+s.target.arch,'readable instructions'],origin:point(150,234),via:[point(610,235)],place:'assembly'}));
        if(s.object) list.push(a('OBJ1','file',s.target.object,660,235,232,111,'artifact',{tag:'RELOCATABLE OBJECT',lines:['machine code + relocations','input to the linker'],origin:point(150,234),place:'object'}));
        if(s.binary) list.push(a('BIN1','binary',s.target.binary,964,132,194,124,'artifact',{tag:'TARGET CODE',lines:['0101 1100 1010','symbol / code bytes'],origin:point(779,281),via:[point(929,280)],place:'binary'}));
        if(s.inspect) list.push(a('DISASM1','file','Disassembly view',963,286,194,111,'driver',{tag:'READ ONLY',lines:[s.target.inspection,'binary bytes unchanged'],origin:point(1061,231),place:'inspection'}));
        if(s.error) list.push(a('tool-error','stamp','STOP',714,351,123,47,'hardware',{description:s.error,place:'error'}));
        this.text('[data-target-arch]',s.target.arch+' · CPU host → '+s.target.name);
        this.text('[data-tool-label]',s.error?'Codegen stopped':s.selected?s.targetKey==='cuda'?'PTX → same-SM cubin':'MC object emission → link':'Waiting for target selection');
      } else if(kind==='package') {
        if(s.mef) list.push(a('MEF1','crate','MEF byte buffer',343,123,347,232,'artifact',{tag:'SERIALIZED',lines:['conceptual ingredients / cutaway'],origin:point(515,252),place:'MEF'}));
        list.push(a('S1','handle','S1 · retained',1027,127,133,65,'engine',{tag:'CONTEXT',lines:['native session'],place:'session'}));
        list.push(a('compiled-code','binary',s.mef?'code / resources':'Code',s.mef?361:s.parts?347:50,s.mef?226:s.parts?206:148,s.mef?311:s.parts?147:199,s.mef?38:s.parts?64:88,'backend',{tag:s.target.binary,origin:point(144,192),via:[point(294,194)],place:s.mef?'MEF content':s.parts?'MEF input':'code'}));
        list.push(a('plan','file',s.mef?'primitives / dependencies':'MGP plan',s.mef?361:s.parts>=3?515:51,s.mef?277:s.parts>=3?211:272,s.mef?311:s.parts>=3?163:199,s.mef?38:s.parts>=3?61:104,'mlir',{tag:'PRIMITIVES',lines:s.parts>=3?[]:['slots / dependencies','resources'],origin:point(148,324),via:[point(290,324),point(294,278)],place:s.mef?'MEF content':s.parts>=3?'MEF input':'plan'}));
        if(s.exported) list.push(a('exported-mef','file','model.mef',381,363,267,40,'artifact',{tag:'EXPLICIT EXPORT',origin:point(515,305),place:'disk'}));
        this.slots('bindings',s.weights>=1?['W_a']:[]);this.slots('binding-b',s.weights>=2?['W_b']:[],Boolean(s.error));
        if(s.weights>=1)list.push(a('W_a','handle','W_a → binding',778,173,177,39,'buffer',{origin:point(863,370),place:'registry'}));
        if(s.weights>=2)list.push(a('W_b','handle','W_b → binding',778,245,177,39,'buffer',{origin:point(863,370),place:'registry'}));
        if(s.initialized)list.push(a('M1','chip','M1 · executable',1027,286,133,99,'engine',{tag:'INITIALIZED',lines:['S1 retained'],origin:point(960,299),place:'model'}));
        this.text('[data-binding-gate]',s.error?'W_b missing · gate closed':s.initialized?'Both bound · M1 ready':s.weights===2?'Both bindings available':'Initialization gate · closed');
      } else {
        list.push(a('M1','chip','M1',376,127,100,85,'engine',{tag:'REUSED',lines:['model'],place:'resident'}));
        list.push(a('S1','handle','S1',488,127,115,85,'runtime',{tag:'RETAINED',lines:['context'],place:'resident'}));
        list.push(a('K1','binary','K1 · code',895,143,114,61,'backend',{tag:'RESIDENT',place:'code'}));
        list.push(a('W1','handle','W1 · weights',1025,143,112,61,'buffer',{tag:'VALUES',place:'weights'}));
        const host=(!s.resident||s.cpu)?s.inputs:[];
        this.slots('host-input',host);this.slots('device-input',s.deviceInputs);this.slots('device-result',s.results);this.slots('host-result',s.cpu?s.results:s.hostResults);this.slots('command',s.queued?['JOB'+s.call]:[]);
        host.forEach((v,i)=>list.push(a('B'+s.call+'-'+i,'cell',String(v),49+i*65,170,56,40,'buffer',{place:'host-input',origin:point(173,141)})));
        s.deviceInputs.forEach((v,i)=>list.push(a('D'+s.call+'-'+i,'cell',String(v),691+i*52,330,46,40,'buffer',{place:s.cpu?'ref(B'+s.call+')':'execution-input',origin:s.cpu||s.resident?point(691+i*52,330):point(77+i*65,190),via:s.cpu||s.resident?[]:[point(340,340),point(660,340)]})));
        if(s.queued)list.push(a('JOB'+s.call,'descriptor','JOB'+s.call,s.phase==='driver'?683:384,s.phase==='driver'?158:264,s.phase==='driver'?157:58,s.phase==='driver'?51:43,'runtime',{tag:'REFS',lines:s.phase==='driver'?['K1 + D'+s.call]:[],origin:point(295,201),via:[point(345,231),point(348,280)],place:s.phase==='driver'?'driver':'queue'}));
        if(s.running)list.push(a('work'+s.call,'tile','(A + 10) × 2',896,207,241,34,'hardware',{origin:point(760,181),place:'executing',description:'Illustrative work, not a measured kernel or launch count'}));
        s.results.forEach((v,i)=>list.push(a('O'+s.call+'-'+i,'cell',String(v),935+i*52,330,46,40,'runtime',{origin:point(1010,225),place:'result',delay:i*70})));
        if(s.ready)list.push(a('result-handle-'+s.call,'handle','O'+s.call+' · ready',394,345,202,42,'runtime',{tag:'BUFFER REFERENCE',origin:point(1040,351),via:[point(1030,442),point(590,442)],place:'returned handle'}));
        (s.cpu?s.results:s.hostResults).forEach((v,i)=>list.push(a('host-O'+s.call+'-'+i,'cell',String(v),49+i*65,327,56,40,'runtime',{origin:s.cpu?point(49+i*65,327):point(958+i*52,350),via:s.cpu?[]:[point(1060,442),point(304,442)],place:s.cpu?'host view':'explicit host-copy',delay:i*70})));
        this.text('[data-input-label]','B'+(s.call||1)+' · '+(s.resident&&!s.cpu?'input already on target':'input values'));
        this.text('[data-output-label]',s.cpu?'Host result view · no transfer':'Host result · only after explicit copy');
        this.text('[data-device-input-label]',s.cpu?'ref(B'+(s.call||1)+') · same host values':'D'+(s.call||1)+' · input');
        this.text('[data-device-output-label]','O'+(s.call||1)+' · '+(s.cpu?'reference view':'result'));
        this.text('[data-hardware-label]',s.running?'':s.cpu?'Native CPU · schematic':s.target.name+' · schematic');
        this.text('[data-memory-label]',s.target.memory+' · code / weights retained');
        this.node.querySelector('[data-wire="upload"]').toggleAttribute('hidden',s.resident);
        this.node.querySelector('[data-component="driver"] .owner-title').textContent=s.cpu?'CPU EXECUTION':'DEVICE INTERFACE';
        this.node.querySelector('[data-component="driver"] .owner-subtitle').textContent=s.cpu?'native runtime':'BOUNDARY';
      }
      return list;
    }
    draw(item,target) {
      const g=item.g;g.replaceChildren();g.append(el('title',{},target.description||target.label+' · '+(target.place||target.kind)));
      const w=target.w,h=target.h;
      if(target.kind==='file') {
        g.append(el('path',{class:'actor-paper',d:'M0 0H'+(w-20)+'L'+w+' 20V'+h+'H0Z'}));
        g.append(el('path',{class:'paper-fold',d:'M'+(w-20)+' 0V20H'+w}));
      } else if(target.kind==='binary') {
        g.append(el('rect',{class:'actor-box cartridge',width:w,height:h,rx:9}));
        for(let i=0;i<5;i++)g.append(el('rect',{class:'binary-pin',x:10+i*(w-20)/5,y:h-8,width:9,height:12,rx:1}));
      } else if(target.kind==='chip') {
        g.append(el('rect',{class:'actor-box microchip',x:5,y:5,width:w-10,height:h-10,rx:9}));
        for(let i=1;i<=4;i++){const y=i*h/5;g.append(el('path',{class:'chip-pin',d:'M0 '+y+'H6M'+(w-6)+' '+y+'H'+w}));}
      } else if(target.kind==='crate') {
        g.append(el('rect',{class:'actor-box crate',width:w,height:h,rx:12}));
        g.append(el('path',{class:'crate-rail',d:'M9 16H'+(w-9)+'M9 '+(h-16)+'H'+(w-9)}));
      } else if(target.kind==='tool') {
        g.append(el('rect',{class:'actor-box tool-box',width:w,height:h,rx:9}));
        g.append(el('path',{class:'tool-handle',d:'M'+(w/2-28)+' 0V-7H'+(w/2+28)+'V0'}));
      } else g.append(el('rect',{class:'actor-box',width:w,height:h,rx:target.kind==='cell'?5:9}));
      const cell=target.kind==='cell', compact=h<=48;
      const fit=(element,max)=>{const original=element.textContent;let n=original.length;while(n>1&&element.getComputedTextLength()>max){element.textContent=original.slice(0,--n)+'…';}};
      if(cell)g.append(el('text',{class:'cell-label',x:w/2,y:h/2+4},target.label));
      else {
        const title=el('text',{class:'actor-label',x:10,y:compact?h/2+4:25},target.label);g.append(title);fit(title,w-24);
        if(target.tag&&!compact){const tag=el('text',{class:'actor-tag',x:10,y:45},target.tag);g.append(tag);fit(tag,w-24);}
        if(!compact)(target.lines||[]).slice(0,Math.min(3,Math.max(0,Math.floor((h-57)/20)))).forEach((line,i)=>{const text=el('text',{class:'actor-line',x:10,y:67+i*20},line);g.append(text);fit(text,w-24);});
      }
      g.dataset.color=target.color;g.dataset.kind=target.kind;g.dataset.place=target.place||'';
      g.dataset.label=target.label;g.dataset.width=w;g.dataset.height=h;
    }
    render(s,animate,speed=1) {
      const now=performance.now(),before=this.frozenAt??now;this.frozenAt=null;this.state=s;
      const targets=this.targets(s), ids=new Set(targets.map(v=>v.id));
      this.actors.forEach(item=>{if(!ids.has(item.id))targets.push({...item.target,opacity:0,via:[]});});
      for(const target of targets) {
        let item=this.actors.get(target.id), from;
        if(!item) { const g=el('g',{class:'actor','data-actor':target.id});this.layer.append(g);item={id:target.id,g};this.actors.set(target.id,item);from={...(target.origin||target),opacity:1}; }
        else from=this.sample(item,before);
        const duration=animate&&!this.reduced.matches?1000/speed:0;
        Object.assign(item,{from,target,started:now+(target.delay||0)/speed,duration,points:[from,...(Math.hypot(target.x-from.x,target.y-from.y)>1?target.via||[]:[]),target]});
        this.layer.append(item.g);
        this.draw(item,target);
      }
      this.tick(now);
    }
    sample(item,now) { const t=item.duration?clamp((now-item.started)/item.duration):1;return {...route(item.points,ease(t)),opacity:item.from.opacity+(item.target.opacity-item.from.opacity)*ease(t),done:t===1}; }
    tick(now) {
      if(this.frozenAt!==null)return false;let moving=false;
      this.actors.forEach(item=>{const p=this.sample(item,now);item.g.toggleAttribute('hidden',p.opacity<.005);item.g.setAttribute('transform','translate('+p.x.toFixed(2)+' '+p.y.toFixed(2)+')');item.g.style.opacity=p.opacity;item.g.dataset.moving=String(!p.done);moving ||= !p.done;});
      return moving;
    }
    pause() { if(this.frozenAt!==null||!this.state)return;const now=performance.now();this.tick(now);this.frozenAt=now; }
    resume() { if(this.frozenAt===null)return;const elapsed=performance.now()-this.frozenAt;this.actors.forEach(a=>a.started+=elapsed);this.frozenAt=null; }
  }
  root.CompilerWorkbenchVisual=WorkbenchVisual;
})(window);
