(function() {
  'use strict';
  const data=JSON.parse(document.getElementById('compiler-workbench-data').textContent);
  const {WorkbenchModel}=window.CompilerWorkbenchModel;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)'), controllers=[];
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function sourceLinks(refs) { return refs.map(key=>{const r=data.references[key];return '<a data-ref="'+key+'" href="../sources/'+r.file+'#L'+r.line+'">'+escape(r.label)+' <small>['+r.alias+':L'+r.line+']</small></a>';}).join(' · '); }
  let raf=null;
  function schedule() { if(raf===null)raf=requestAnimationFrame(tick); }
  function tick(now) {
    raf=null;let active=false;
    for(const c of controllers) {
      if(c.playing&&now>=c.nextAt) {c.model.step();c.render(true);c.nextAt=now+1450/c.speed;if(c.model.snapshot().done)c.playing=false;c.updatePlay();}
      active=c.visual.tick(now)||c.playing||active;
    }
    if(active)schedule();
  }
  class Lesson {
    constructor(node) {
      this.node=node;this.kind=node.dataset.lesson;this.speed=1;this.playing=false;
      this.model=new WorkbenchModel(this.kind,this.options());this.visual=new window.CompilerWorkbenchVisual(node,reduced);
      node.addEventListener('click',e=>{
        const action=e.target.closest('[data-action]')?.dataset.action;
        if(action==='play')this.setPlaying(!this.playing);
        else if(action==='step'){this.setPlaying(false);this.model.step();this.render(true);}
        else if(action==='back'){this.setPlaying(false);this.model.back();this.render(false);}
        else if(action==='reset'){this.setPlaying(false);this.model.configure(this.options());this.render(false);}
        const intervention=e.target.closest('[data-intervention]')?.dataset.intervention;
        if(intervention){this.setPlaying(false);this.model.intervene(intervention);this.render(true);}
        const index=e.target.closest('button[data-index]')?.dataset.index;
        if(index!==undefined){this.setPlaying(false);this.model.seek(Number(index));this.render(false);}
      });
      node.addEventListener('change',e=>{
        if(e.target.matches('[data-option]')){this.setPlaying(false);this.model.configure(this.options());this.render(false);}
        if(e.target.matches('[data-speed]')){this.speed=Number(e.target.value);if(this.playing)this.nextAt=performance.now()+1450/this.speed;}
      });
      node.querySelector('[data-seek]').addEventListener('input',e=>{this.setPlaying(false);this.model.seek(Number(e.target.value));this.render(false);});
      node.querySelectorAll('details').forEach(el=>el.addEventListener('toggle',()=>{if(!el.open&&el.contains(node.querySelector('.visual')))this.setPlaying(false);}));
      this.render(false);
    }
    options() {
      const result={target:document.getElementById('target-select').value};
      this.node.querySelectorAll('[data-option]').forEach(el=>{result[el.dataset.option]=el.hasAttribute('data-boolean')?el.value==='true':el.dataset.option==='width'?Number(el.value):el.value;});
      result.invalidArch=result.failure==='arch';result.missingToolkit=result.failure==='toolkit';return result;
    }
    setPlaying(value) {
      if(value&&this.model.snapshot().done){this.model.configure(this.options());this.render(false);}
      this.playing=value;
      if(value){this.visual.resume();this.nextAt=performance.now()+350;this.node.querySelectorAll('.transcript').forEach(e=>{if(e.open)e.open=false;});schedule();}
      else this.visual.pause();
      this.updatePlay();
    }
    updatePlay(){const b=this.node.querySelector('[data-action="play"]');b.textContent=this.playing?'Ⅱ Pause':'▶ Play';b.setAttribute('aria-pressed',String(this.playing));this.node.dataset.playing=String(this.playing);}
    render(animate) {
      const s=this.model.snapshot();this.node.dataset.index=s.index;this.node.dataset.frames=s.total;this.node.dataset.phase=s.phase||s.stage;this.node.dataset.state=JSON.stringify(s);
      this.visual.render(s,animate,this.speed);if(animate)schedule();
      this.node.querySelector('[data-progress]').textContent=(s.index+1)+' / '+s.total;
      const seek=this.node.querySelector('[data-seek]');seek.max=s.total-1;seek.value=s.index;
      this.node.querySelector('[data-action="step"]').disabled=s.done;this.node.querySelector('[data-action="back"]').disabled=s.index===0;
      this.node.querySelector('[data-event-title]').textContent=s.title;this.node.querySelector('[data-event-detail]').textContent=s.detail;this.node.querySelector('.event').dataset.kind=s.kind;
      this.node.querySelector('[data-event-sources]').innerHTML=sourceLinks(s.refs);
      const strips={
        graph:[['Native G1',s.graph?1:0],['Recorded ops',s.ops],['Contract',s.sealed?'verified':s.contract?'declared':'open'],['Executable models',0]],
        lower:[['Compiler clone',s.clone?'C1':'not created'],['Representation',window.CompilerWorkbenchModel.LOWER_STAGES[s.level]?.[1]||'MO'],['Device-graph pass',s.deviceFinalized?'finalized':'pending'],['Views',s.history?.length||0],['Grouping',s.fused?'illustrative fused body':'separate']],
        mojo:s.lab==='ownership'?[['Allocation','B1'],['Owner',s.owner],['Borrow',s.borrow||'none'],['Legality',s.blocked?'blocked':'checked / pending']]:[['WIDTH',s.width],['Concrete',s.concrete?'yes':'pending'],['Native bundles',s.bundles||'pending'],['Python interpreter calls','no per-lane dispatch']],
        target:[['Target',s.target?.arch],['LLVM IR',s.llvm?'created':'pending'],['Object bytes',s.object?'created':'none'],['Binary',s.binary?'created':'none'],['Inspection',s.inspect?'disassembly':'none']],
        package:[['MEF bytes',s.mef?'created':'pending'],['Export',s.exported?'file written':'not requested'],['Bindings',s.weights+' / 2'],['M1',s.initialized?'executable':'not initialized']],
        runtime:[['Dispatches',s.dispatches],['Explicit copies',s.copies],['Code / weights','K1 / W1 retained'],['Session / model','S1 / M1 reused'],['Results',s.ready?'ready':'pending']]
      };
      this.node.querySelector('[data-state-strip]').innerHTML=strips[this.kind].map(([k,v])=>'<span>'+escape(k)+' <b>'+escape(v)+'</b></span>').join('');
      for(const button of this.node.querySelectorAll('[data-intervention]')) {
        const action=button.dataset.intervention;
        button.disabled=action==='seal'?!(s.graph&&!s.sealed):action==='release'?!s.blocked:action==='inspect'?!(s.binary&&!s.inspect):action==='repair'?!s.error:action==='again'?!s.done:true;
      }
      this.node.querySelectorAll('[data-cuda-control]').forEach(e=>e.hidden=document.getElementById('target-select').value!=='cuda');
      this.node.querySelectorAll('[data-simd-control]').forEach(e=>e.hidden=s.lab==='ownership');
      const diff=this.node.querySelector('[data-ir-diff]');
      let before=s.before,after=s.after;
      if(this.kind==='mojo'&&s.lab!=='ownership'){before='Source-level intent\nAdd(dtype, WIDTH) · typed inputs';after=s.concrete?'Concrete native view\nWIDTH = '+s.width+' · '+(s.bundles||'pending')+' schematic bundles':'LIT / parametric KGEN\nKeep types + compile-time parameters';}
      if(this.kind==='target'){before='LLVM dialect (MLIR)\nllvm operations + target metadata';after=s.binary?'Target code bytes\n'+s.target.binary:s.assembly?'Textual target assembly\n'+s.target.assembly:s.llvm?'LLVM IR module\ntriple: '+s.target.triple:'Translation pending';}
      diff.hidden=!(before&&after);if(!diff.hidden){this.node.querySelector('[data-before]').textContent=before;this.node.querySelector('[data-after]').textContent=after;}
      this.node.querySelector('[data-transcript]').innerHTML=this.model.frames.map((f,i)=>'<li><button data-index="'+i+'" aria-label="Go to step '+(i+1)+'">'+String(i+1).padStart(2,'0')+'</button><strong>'+escape(f.title)+'</strong> '+escape(f.detail)+'</li>').join('');
      this.updatePlay();
    }
  }
  document.querySelectorAll('[data-interactive]').forEach(e=>e.hidden=false);
  document.querySelectorAll('[data-lesson]').forEach(node=>controllers.push(new Lesson(node)));
  document.getElementById('target-select').addEventListener('change',()=>controllers.filter(c=>['target','package','runtime'].includes(c.kind)).forEach(c=>{c.setPlaying(false);c.model.configure(c.options());c.render(false);}));
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(!entry.isIntersecting)controllers.find(c=>c.node===entry.target)?.setPlaying(false);}),{threshold:.1});controllers.forEach(c=>observer.observe(c.node));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)controllers.forEach(c=>c.setPlaying(false));});
  reduced.addEventListener('change',()=>controllers.forEach(c=>c.render(false)));
  document.querySelector('[data-print]').addEventListener('click',()=>print());

  window.CompilerReaderUI(data,controllers);
  window.compilerWorkbenchSnapshot=key=>controllers.find(c=>c.kind===key)?.model.snapshot();
})();
