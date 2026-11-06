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

  const explanation=document.getElementById('explanation-card'),legend=document.getElementById('legend-card'),legendButton=document.getElementById('legend-button');
  let badge=null,dismissTimer=null,suppressFocus=false;
  function positionCard(card,trigger,alignRight=false) {
    const b=trigger.getBoundingClientRect(),gap=14,w=card.offsetWidth,h=card.offsetHeight;
    card.style.left=Math.max(gap,Math.min(innerWidth-w-gap,alignRight?innerWidth-w-gap:b.left))+'px';
    card.style.top=Math.max(68,Math.min(innerHeight-h-gap,b.bottom+10))+'px';
  }
  function showExplanation(button,focus=false) {
    clearTimeout(dismissTimer);badge=button;
    const key=button.dataset.card,l=data.lessons.find(l=>l.id===key),c=controllers.find(c=>c.kind===key),s=c.model.snapshot();
    document.getElementById('explanation-kicker').textContent=l.number+' · '+l.evidence;
    document.getElementById('explanation-title').textContent=l.title;
    document.getElementById('explanation-body').innerHTML='<p>'+escape(l.detail)+'</p><p><b>Current step:</b> '+escape(s.title)+'</p><p>'+escape(s.detail)+'</p><pre>'+escape(l.ascii)+'</pre><p>'+sourceLinks(l.refs)+'</p>';
    explanation.hidden=false;button.setAttribute('aria-expanded','true');positionCard(explanation,button);if(focus)explanation.focus({preventScroll:true});
  }
  function closeExplanation(restore=false) {
    if(explanation.hidden)return;explanation.hidden=true;
    const old=badge;old?.setAttribute('aria-expanded','false');badge=null;
    if(restore&&old){suppressFocus=true;old.focus({preventScroll:true});queueMicrotask(()=>suppressFocus=false);}
  }
  function maybeDismiss() {
    clearTimeout(dismissTimer);dismissTimer=setTimeout(()=>{
      if(!badge)return;
      if(document.getElementById('source-dialog').open)return;
      if(badge.matches(':hover')||explanation.matches(':hover')||explanation.contains(document.activeElement))return;
      if(document.activeElement===badge && badge.matches(':focus-visible'))return;
      closeExplanation();
    },130);
  }
  document.querySelectorAll('[data-card]').forEach(button=>{
    button.setAttribute('aria-expanded','false');
    button.addEventListener('pointerenter',e=>{if(e.pointerType!=='touch')showExplanation(button);});button.addEventListener('pointerleave',maybeDismiss);
    button.addEventListener('focus',()=>{if(!suppressFocus)showExplanation(button);});button.addEventListener('blur',maybeDismiss);
    button.addEventListener('pointerdown',()=>{button.dataset.wasOpen=String(badge===button&&!explanation.hidden);});
    button.addEventListener('click',e=>{if(e.detail===0)showExplanation(button,true);else if(e.pointerType==='touch'&&button.dataset.wasOpen==='true')closeExplanation();else showExplanation(button);});
  });
  explanation.addEventListener('pointerenter',()=>clearTimeout(dismissTimer));explanation.addEventListener('pointerleave',maybeDismiss);explanation.addEventListener('focusout',maybeDismiss);
  document.querySelector('[data-close-card]').addEventListener('click',()=>closeExplanation(true));
  function closeLegend(restore=false){legend.hidden=true;legendButton.setAttribute('aria-expanded','false');if(restore)legendButton.focus({preventScroll:true});}
  legendButton.addEventListener('click',()=>{if(!legend.hidden)closeLegend();else{closeExplanation();legend.hidden=false;legendButton.setAttribute('aria-expanded','true');positionCard(legend,legendButton,true);}});
  document.querySelector('[data-close-legend]').addEventListener('click',()=>closeLegend(true));
  document.addEventListener('pointerdown',e=>{if(document.getElementById('source-dialog').open)return;if(!legend.hidden&&!legend.contains(e.target)&&e.target!==legendButton)closeLegend();});
  addEventListener('resize',()=>{if(!legend.hidden)positionCard(legend,legendButton,true);if(badge&&!explanation.hidden)positionCard(explanation,badge);});
  addEventListener('scroll',()=>{if(!legend.hidden)positionCard(legend,legendButton,true);if(badge&&!explanation.hidden)positionCard(explanation,badge);},{passive:true});

  const dialog=document.getElementById('source-dialog'),sourceFile=document.getElementById('source-file');
  let launch=null,definition='',currentFile=null,matches=[],matchIndex=-1,url=null;
  function searchSource(){sourceFile.querySelectorAll('.search-hit').forEach(e=>e.classList.remove('search-hit'));const q=document.getElementById('source-search').value.toLowerCase();matches=q?[...sourceFile.querySelectorAll('.source-line')].filter(e=>e.textContent.toLowerCase().includes(q)):[];matchIndex=-1;document.getElementById('source-count').textContent=matches.length?matches.length+' matches':q?'No matches':'';}
  function jumpMatch(delta){if(!matches.length)return;matchIndex=(matchIndex+delta+matches.length)%matches.length;matches[matchIndex].classList.add('search-hit');matches[matchIndex].scrollIntoView({block:'center'});document.getElementById('source-count').textContent=(matchIndex+1)+' / '+matches.length;}
  function openSource(link) {
    const r=data.references[link.dataset.ref];if(!r)return;
    launch=link;currentFile=data.files[r.file];const lines=currentFile.text.split('\n');definition=lines.slice(r.start-1,r.end).join('\n');
    document.getElementById('source-title').textContent=r.label;document.getElementById('source-location').textContent=r.file+' · lines '+r.start+'–'+r.end+' · '+data.revision;
    sourceFile.querySelector('code').innerHTML=lines.map((line,i)=>'<span class="source-line'+(i+1>=r.start&&i+1<=r.end?' selected':'')+(i+1===r.line?' exact':'')+'" data-line="'+(i+1)+'"><span>'+(i+1)+'</span>'+escape(line)+'</span>').join('');
    if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(new Blob([currentFile.text],{type:'text/plain'}));const download=document.getElementById('source-download');download.href=url;download.download=r.file.split('/').pop();
    document.getElementById('source-search').value='';searchSource();dialog.showModal();requestAnimationFrame(()=>sourceFile.querySelector('[data-line="'+r.line+'"]')?.scrollIntoView({block:'center'}));
  }
  document.addEventListener('click',e=>{const link=e.target.closest('a[data-ref]');if(link){e.preventDefault();openSource(link);}});
  document.getElementById('source-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('close',()=>{launch?.focus({preventScroll:true});});
  document.getElementById('source-search').addEventListener('input',searchSource);document.getElementById('source-next').addEventListener('click',()=>jumpMatch(1));document.getElementById('source-previous').addEventListener('click',()=>jumpMatch(-1));
  document.getElementById('source-copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(definition);document.getElementById('source-copy').textContent='Copied';setTimeout(()=>document.getElementById('source-copy').textContent='Copy definition',1200);}catch{document.getElementById('source-copy').textContent='Select code to copy';}});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!dialog.open){if(!legend.hidden){closeLegend(true);e.preventDefault();}else if(!explanation.hidden){closeExplanation(true);e.preventDefault();}}});
  window.compilerWorkbenchSnapshot=key=>controllers.find(c=>c.kind===key)?.model.snapshot();
})();
