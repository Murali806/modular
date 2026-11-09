(function() {
  'use strict';
  const data=JSON.parse(document.getElementById('compiler-examples-data').textContent);
  const {ExampleModel,fmt}=window.CompilerExamplesModel,reduced=matchMedia('(prefers-reduced-motion: reduce)'),controllers=[];
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const sourceLinks=refs=>refs.map(k=>{const r=data.references[k];if(!r)throw Error('Unknown source '+k);return '<a data-ref="'+k+'" href="../sources/'+r.file+'#L'+r.line+'">'+escape(r.label)+' <small>['+r.alias+':L'+r.line+']</small></a>';}).join(' · ');
  let raf=null;
  function schedule(){if(raf===null)raf=requestAnimationFrame(tick);}
  function tick(now){raf=null;let active=false;for(const c of controllers){if(c.playing&&now>=c.nextAt){c.model.step();c.render(true);c.nextAt=now+1650/c.speed;if(c.model.snapshot().done)c.playing=false;c.updatePlay();}active=c.visual.tick(now)||c.playing||active;}if(active)schedule();}
  function options(){return {target:document.getElementById('target-select').value,fusion:document.getElementById('fusion-select')?.value||'fused',algorithm:document.getElementById('algorithm-select')?.value||'tiled'};}
  function codePreview(s) {
    let key=s.refs[0],needle='';
    if(s.lesson==='commands')return s.command;
    if(s.lesson==='toolchain'){key=s.sourceRef||s.refs[0];needle=s.sourceNeedle||'';}
    if(s.lesson==='build') {key=s.phase==='register'?'registration':'entry';needle={source:'Graph(',buffers:'a_tensor =',contract:'TensorType(',custom:'name=',compose:'forward=',output:data.example.id==='addition'||data.example.id==='epilogue'?'forward=':'graph.output',verified:'out_types='}[s.phase]||'';}
    if(s.lesson==='compile'&&['session','clone','init','ready'].includes(s.phase)){key='entry';needle={session:'InferenceSession(',clone:'compiled = session.compile',init:'model = session.init',ready:'model = session.init'}[s.phase];}
    if(s.lesson==='run') {
      if(['resident','arrays','upload','execute','host','complete'].includes(s.phase)){key='entry';needle={resident:'session.init',arrays:'np.random',upload:'Buffer.from_numpy',execute:'model.execute',host:'result.to(CPU())',complete:data.example.id==='attention'?'print(output)':data.example.id==='addition'?'Expected result':'assert_allclose'}[s.phase];}
      else if(data.example.id==='addition'&&['load','add','next','store'].includes(s.phase)){key='math';needle=s.phase==='load'?'def elementwise_add_one':s.phase==='next'?'foreach[target=target]':'return x.load';}
      else if(data.example.id==='epilogue'&&['dot','epilogue','matmul','add_one','multiply'].includes(s.phase)){key=['epilogue','add_one'].includes(s.phase)?'math':'entry';needle=key==='math'?'return x + 1':'forward=lambda';}
      else if(data.example.id==='matmul'&&!s.cpu&&s.algorithm==='tiled'){key=s.phase==='memset'?'dispatch':'math';needle={memset:'gpu_ctx.enqueue_memset',registers:'var a_smem',copy:'GenericToSharedAsyncTileCopier',commit:'async_copy_commit_group()',wait:'async_copy_wait_all()',barrier:'barrier()',mac:'comptime for k',reuse_barrier:'barrier()',store:'dst[row, col]'}[s.phase]||'';}
      else if(data.example.id==='attention'&&!['ready'].includes(s.phase)){key=s.cpu?'math_cpu':'math';needle={initialize:'var m_1',score:'var S =',max:'m_2.max',exp:'E.copy_from',sum:'l_2 +=',rescale:'O_i *= l_1',weighted:'O_i += PV',store:'copy_from(O_i)'}[s.phase]||'';}
    }
    const r=data.references[key],lines=data.files[r.file].text.split('\n');let start=r.start-1;
    if(needle){const hits=lines.map((line,i)=>i>=r.start-1&&i<r.end&&line.includes(needle)?i:-1).filter(i=>i>=0);if(hits.length)start=s.phase==='reuse_barrier'?hits[hits.length-1]:hits[0];}
    return r.file+'\n'+lines.slice(start,Math.min(r.end,start+9)).map((line,i)=>String(start+i+1).padStart(4)+'  '+line).join('\n');
  }
  function statePreview(s) {
    if(s.lesson==='build')return 'G1: '+(s.graph?'created':'not yet created')+'\n'+s.signature.join('\n')+'\nMojo registration: '+(s.registered?'resolved':'pending')+'\nOutput contract: '+(s.verified?'verified':'open')+'\nExecutable model: absent';
    if(s.lesson==='compile')return 'G1 stays with caller\nC1: '+(s.clone?'compiler-owned':'not created')+'\n'+['MO','MAP','MOGG','MGP','MGP finalization','Mojo LIT/KGEN/POP','LLVM dialect → LLVM IR','target code','prepared artifact'][s.level]+'\n'+s.routine+'\nRuntime calls: '+(s.runtimeCalls?'outlined':'pending')+'\nMEF bytes: '+(s.mef?'prepared':'pending')+'\nM1: '+(s.initialized?'initialized':s.loading?'loading / resolving':'absent')+'\nWeight registry: not required by this example';
    if(s.lesson==='commands')return 'Example: '+s.filename+'\nBazel binary: '+s.commandTarget+'\nBazel test: '+s.commandTarget+'.example-test\nPixi task: '+s.task+' → python '+s.filename+'\nSelected route: '+s.route+'\nBazel test command shown: '+(s.testCommandSelected?'yes':'not yet')+'\nPixi install shown: '+(s.pixiInstalled?'yes':'not yet');
    if(s.lesson==='toolchain')return 'Question '+s.qa+' · '+s.representation+'\nExample target: '+s.commandTarget+'\nBuild outputs: '+(s.built?'launcher + source runfiles':'not yet')+'\nShared Mojo dependencies: '+(s.libraries?'.mojoc packages':'not yet')+'\nCustom package P1: '+(s.package?'post-parse MLIR bytecode':'raw .mojo data')+'\nG1: '+(s.g1Retained?'retained by caller':'not recorded')+'\nC1 graph / runtime plan: '+s.graphIR+'\nSeparate KIR1: '+s.kernelIR+'\nLLVM1: '+(s.llvmModule?'llvm::Module':'not translated')+'\nTargetInfo: '+(s.targetInfo?s.triple+' / '+s.arch:'not resolved')+'\nSelected backend: '+(s.selectedBackend||'not resolved')+'\nTarget output: '+(s.targetCode?s.artifact:'not emitted')+'\nRuntime-call plan: '+(s.runtimeCalls?'outlined':'not lowered')+'\nCM1: '+(s.mef?'prepared bytes':'not prepared')+' · M1: '+(s.initialized?'initialized handles':'not initialized')+'\nNumerical inputs: '+(s.inputs?'supplied to execute':'separate from compilation')+'\nResult Buffer: '+(s.ready?'ready':'not ready');
    if(s.id==='attention')return 'Tile '+(s.tile+1)+' / 2 (tiny view)\nscores: ['+s.scores+']\nm='+ (s.m===null?'MIN':fmt(s.m))+'   α='+fmt(s.alpha)+'\nl='+fmt(s.l)+'\nO=['+s.O.map(fmt)+']\nFinal result: ['+s.computed.map(fmt)+']\nOnly local score tiles; running m/l/O survive\nSource score elements: full comparison '+s.params.N*s.params.N+'; provided local tile '+s.params.BN*s.params.keyTile;
    if(s.id==='epilogue'&&s.traffic)return 'Route: '+(s.fusion?'eligible epilogue sketch':'separate-body sketch')+'\nTiny logical intermediate reads: '+s.traffic.reads+'\nTiny logical intermediate writes: '+s.traffic.writes+'\nFinal output writes: '+s.traffic.output+'\nLive intermediate allocations: '+s.tempLive+'\nResult: ['+s.computed.map(fmt)+']\nCounts exclude A/B reads and measured bus/cache traffic';
    return 'Inputs: '+(s.inputs?s.cpu?'CPU references':'device buffers':'not supplied')+'\n'+(s.id==='matmul'?`Selected ${s.algorithm}; k=${s.k}\nCopy ready: ${s.copyReady}; block ready: ${s.blockReady}\n`:s.id==='epilogue'?`Route: ${s.fusion?'eligible epilogue sketch':'separate-body sketch'}\n`:'Illustrative SIMD group: '+(s.lane+1)+'\n')+'Accumulator: ['+s.acc.map(fmt)+']\nResult: ['+s.computed.map(fmt)+']\nBuffer ready: '+s.ready+'\nExplicit host read: '+s.hostRead;
  }
  function isTextHash(hash){return /^#toolchain-(?:q\d|call-|reference(?:$|-))/.test(hash);}
  function revealReadingTarget(target){for(let parent=target;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;}
  class Lesson {
    constructor(node){this.node=node;this.kind=node.dataset.lesson;this.speed=1;this.playing=false;this.model=new ExampleModel(this.kind,data.example,options());this.visual=new window.CompilerExamplesVisual(node,reduced);
      node.addEventListener('click',e=>{const mode=e.target.closest('button[data-mode]')?.dataset.mode;if(mode)this.setMode(mode);if(e.target.closest('[data-return-animation]')){this.setMode('animation');this.node.scrollIntoView({block:'start',behavior:'instant'});this.node.querySelector('#toolchain-animation-tab').focus({preventScroll:true});}const qa=e.target.closest('[data-read-qa]')?.dataset.readQa;if(qa||e.target.closest('[data-read-current]')){this.setMode('text');const target=questionNode(qa||this.model.snapshot().qa);if(target){target.tabIndex=-1;target.focus({preventScroll:true});target.scrollIntoView({block:'start',behavior:'instant'});}}const action=e.target.closest('[data-action]')?.dataset.action;if(action==='play')this.setPlaying(!this.playing);else if(action){this.setPlaying(false);if(action==='reset')this.model.configure(options());else this.model[action]();this.render(action==='step');}const index=e.target.closest('button[data-index]')?.dataset.index;if(index!==undefined){this.setPlaying(false);this.model.seek(Number(index));this.render(false);}});
      if(this.kind==='toolchain')node.addEventListener('click',e=>{const anchor=e.target.closest('a[href^="#toolchain-"]');if(!anchor||anchor.hasAttribute('data-call-explanation')||!isTextHash(anchor.hash))return;const target=document.getElementById(anchor.hash.slice(1));if(target){this.setMode('text');revealReadingTarget(target);}});
      node.querySelector('[role="tablist"]')?.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...node.querySelectorAll('[data-mode]')],current=tabs.indexOf(document.activeElement),next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(current+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;this.setMode(tabs[next].dataset.mode);tabs[next].focus();});
      node.querySelector('[data-seek]').addEventListener('input',e=>{this.setPlaying(false);this.model.seek(Number(e.target.value));this.render(false);});
      node.querySelector('[data-speed]').addEventListener('change',e=>{this.speed=Number(e.target.value);if(this.playing)this.nextAt=performance.now()+1650/this.speed;});this.render(false);if(this.kind==='toolchain')this.setMode(isTextHash(location.hash)?'text':'animation');
    }
    setMode(mode){this.setPlaying(false);this.mode=mode;this.node.dataset.mode=mode;this.node.querySelectorAll('[data-mode-panel]').forEach(p=>p.hidden=p.dataset.modePanel!==mode);this.node.querySelectorAll('[data-mode]').forEach(b=>{const selected=b.dataset.mode===mode;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;});}
    setPlaying(value){if(value&&this.mode==='text')return;if(value&&this.model.snapshot().done){this.model.configure(options());this.render(false);}this.playing=value;if(value){this.visual.resume();this.nextAt=performance.now()+350;schedule();}else this.visual.pause();this.updatePlay();}
    updatePlay(){const b=this.node.querySelector('[data-action="play"]');b.textContent=this.playing?'Ⅱ Pause':'▶ Play';b.setAttribute('aria-pressed',String(this.playing));this.node.dataset.playing=String(this.playing);}
    render(animate){const s=this.model.snapshot();Object.assign(this.node.dataset,{index:s.index,frames:s.total,phase:s.phase,state:JSON.stringify(s)});this.visual.render(s,animate,this.speed);if(animate)schedule();
      this.node.querySelector('[data-progress]').textContent=(s.index+1)+' / '+s.total;const seek=this.node.querySelector('[data-seek]');seek.max=s.total-1;seek.value=s.index;
      this.node.querySelector('[data-action="back"]').disabled=s.index===0;this.node.querySelector('[data-action="step"]').disabled=s.done;
      this.node.querySelector('[data-event-title]').textContent=s.title;this.node.querySelector('[data-event-detail]').textContent=s.detail;this.node.querySelector('[data-event-sources]').innerHTML=sourceLinks(s.refs);
      const pairs=s.lesson==='build'?[['Input types',s.signature[0]],['G1',s.verified?'verified':s.graph?'recording':'not created'],['Executable',0]]:s.lesson==='compile'?[['Target',s.target],['Selected body',s.routine],['Compiled bytes',s.code?'ready':'pending'],['M1',s.initialized?'ready':'not initialized']]:s.lesson==='commands'?[['Route',s.route],['Bazel build',s.built?'command shown':'next'],['Bazel test',s.testCommandSelected?'command shown':'next'],['Pixi',s.pixiInstalled?'setup shown':s.pixiRun?'run shown':'optional']]:s.lesson==='toolchain'?[['QnA',s.qa],['C1',s.graphIR],['KIR1',s.kernelIR],['Backend',s.selectedBackend||'pending'],['Code',s.targetCode?s.artifact:'pending'],['M1',s.initialized?'initialized':'absent']]:[['Placement',s.cpu?'CPU':'accelerator'],['Output cells',s.computed.length],['Result',s.ready?'ready':'pending'],[s.id==='attention'?'Key tiles (tiny)':'Host read',s.id==='attention'?s.tileCount:s.hostRead?'explicit':'not yet']];
      this.node.querySelector('[data-state-strip]').innerHTML=pairs.map(([k,v])=>'<span>'+escape(k)+' <b>'+escape(v)+'</b></span>').join('');
      this.node.querySelector('[data-code-preview]').textContent=codePreview(s);this.node.querySelector('[data-state-preview]').textContent=statePreview(s);
      const codeHeading=this.node.querySelector('.ir-diff > div:first-child > span'),stateHeading=this.node.querySelector('.ir-diff > div:nth-child(2) > span');
      codeHeading.textContent=s.lesson==='commands'?'Command · copy from the selected step':'Original source · selected current definition';
      stateHeading.textContent=s.lesson==='commands'?'Targets and setup · repository evidence':'Schematic state / numerical work · not a generated dump';
      this.node.querySelector('[data-read-current]')?.replaceChildren('Read '+s.qa+' in Text');
      this.node.querySelector('[data-transcript]').innerHTML=this.model.frames.map((f,i)=>'<li'+(i===s.index?' aria-current="step"':'')+'><button type="button" data-index="'+i+'" aria-label="Seek to step '+(i+1)+'">'+String(i+1).padStart(2,'0')+'</button><strong>'+escape(f.title)+'</strong><p>'+escape(f.detail)+'</p>'+(f.representation?'<code>'+escape(f.representation)+'</code><button type="button" class="qa-step-link" data-read-qa="'+f.qa+'">Read '+f.qa+'</button>':'')+'<div class="transcript-sources">'+sourceLinks(f.refs)+'</div></li>').join('');this.updatePlay();
    }
  }
  function questionNode(qa){const parts=qa.split('.');while(parts.length){const target=document.getElementById('toolchain-q'+parts.join('-'));if(target){revealReadingTarget(target);return target;}parts.pop();}return null;}
  document.querySelectorAll('[data-interactive]').forEach(e=>e.hidden=false);document.querySelectorAll('[data-lesson]').forEach(n=>controllers.push(new Lesson(n)));
  function refreshOptions(){for(const c of controllers){c.setPlaying(false);c.model.configure(options());c.render(false);}const s=controllers.find(c=>c.kind==='compile').model.snapshot();document.getElementById('algorithm-select')?.toggleAttribute('disabled',s.cpu);document.querySelector('[data-actual]').textContent='Original dimensions: '+s.signature.join(' → ')+(data.example.id==='matmul'||data.example.id==='attention'?' · '+s.params.grid:'')+(data.example.id==='attention'?' · query BN='+s.params.BN+', feature BD='+s.params.BD+', key tile='+s.params.keyTile:'');}
  document.querySelectorAll('.example-options select').forEach(e=>e.addEventListener('change',refreshOptions));refreshOptions();
  document.querySelector('[data-qna-copy]').addEventListener('click',async()=>{const status=document.querySelector('[data-qna-copy-status]');try{if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(data.toolchain.text);else{const field=document.createElement('textarea');field.value=data.toolchain.text;field.className='copy-fallback';document.body.append(field);field.select();const copied=document.execCommand('copy');field.remove();if(!copied)throw Error('Clipboard unavailable');}status.textContent=data.toolchain.calltrees?'Copied call trees and complete answers 1–7.':'Copied complete answers 1–7.';}catch{status.textContent='Use Download the full text to save the complete text.';}});
  function syncTextHash(){if(isTextHash(location.hash)){controllers.find(c=>c.kind==='toolchain').setMode('text');const target=document.getElementById(location.hash.slice(1));if(target){revealReadingTarget(target);target.scrollIntoView({block:'start',behavior:'instant'});}}}
  window.addEventListener('hashchange',syncTextHash);
  function setupCallExplanationCard(){
    const card=document.getElementById('call-explanation-card');
    if(!card)return;
    document.body.append(card);
    const body=card.querySelector('.call-explanation-card-body'),closeButton=card.querySelector('[data-close-call-explanation]'),title=card.querySelector('#call-explanation-title');
    let trigger=null;
    const sourceOpen=()=>document.getElementById('source-dialog').open;
    function close(restore=false){
      if(card.hidden)return;
      card.hidden=true;
      const previous=trigger;trigger=null;
      previous?.setAttribute('aria-expanded','false');
      body.replaceChildren();
      if(restore)previous?.focus({preventScroll:true});
    }
    function position(){
      if(card.hidden||!trigger)return;
      card.style.maxHeight='calc(100dvh - 82px)';
      const bounds=trigger.getBoundingClientRect(),gap=14;
      const below=Math.max(68,bounds.bottom+10),belowSpace=innerHeight-below-gap,aboveSpace=bounds.top-68-gap;
      const useBelow=belowSpace>=240||belowSpace>=aboveSpace;
      const top=useBelow?below:Math.max(68,bounds.top-card.offsetHeight-gap);
      card.style.left=Math.max(gap,Math.min(innerWidth-card.offsetWidth-gap,bounds.left))+'px';
      card.style.top=top+'px';
      card.style.maxHeight=Math.max(80,useBelow?belowSpace:bounds.top-top-gap)+'px';
    }
    function open(link){
      const explanation=document.getElementById(link.dataset.callExplanation),content=explanation?.querySelector('.call-explanation');
      if(!content)return;
      if(!card.hidden&&trigger===link){close(true);return;}
      close();trigger=link;
      title.textContent=explanation.dataset.explanationTitle||'Explanation';
      body.replaceChildren(content.cloneNode(true));body.scrollTop=0;
      controllers.forEach(controller=>controller.setPlaying(false));
      card.hidden=false;link.setAttribute('aria-expanded','true');position();
      closeButton.focus({preventScroll:true});
    }
    document.querySelectorAll('[data-call-explanation]').forEach(link=>{
      link.setAttribute('aria-haspopup','dialog');link.setAttribute('aria-controls',card.id);link.setAttribute('aria-expanded','false');
    });
    document.addEventListener('click',event=>{
      const link=event.target.closest('[data-call-explanation]');
      if(link){event.preventDefault();open(link);return;}
      if(!card.hidden&&!sourceOpen()&&!card.contains(event.target))close();
    });
    closeButton.addEventListener('click',()=>close(true));
    document.addEventListener('pointerdown',event=>{
      if(!card.hidden&&!sourceOpen()&&!card.contains(event.target)&&!trigger.contains(event.target))close();
    });
    document.addEventListener('focusin',event=>{
      if(!card.hidden&&!sourceOpen()&&!card.contains(event.target)&&!trigger.contains(event.target))close();
    });
    document.addEventListener('keydown',event=>{
      if(event.key==='Escape'&&!card.hidden&&!sourceOpen()){event.preventDefault();close(true);}
    });
    addEventListener('resize',position);
    addEventListener('scroll',()=>{
      if(card.hidden||sourceOpen())return;
      const bounds=trigger.getBoundingClientRect();
      if(bounds.bottom<68||bounds.top>innerHeight-14)close();else position();
    },{passive:true});
    addEventListener('beforeprint',()=>close());
  }
  let printClosed=[];
  window.addEventListener('beforeprint',()=>{printClosed=[...document.querySelectorAll('.addition-calltrees details:not([open]),#toolchain-reference:not([open])')];printClosed.forEach(detail=>detail.open=true);});
  window.addEventListener('afterprint',()=>{printClosed.forEach(detail=>detail.open=false);printClosed=[];});
  const observer=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)controllers.find(c=>c.node===e.target)?.setPlaying(false);}),{threshold:.1});controllers.forEach(c=>observer.observe(c.node));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)controllers.forEach(c=>c.setPlaying(false));});reduced.addEventListener('change',()=>controllers.forEach(c=>c.render(false)));document.querySelector('[data-print]').addEventListener('click',()=>print());
  const journey=window.CompilerJourney.attach(data);
  window.CompilerReaderUI(data,[journey,...controllers]);setupCallExplanationCard();window.compilerExampleSnapshot=key=>key==='journey'?journey.model.snapshot():controllers.find(c=>c.kind===key)?.model.snapshot();
  syncTextHash();
})();
