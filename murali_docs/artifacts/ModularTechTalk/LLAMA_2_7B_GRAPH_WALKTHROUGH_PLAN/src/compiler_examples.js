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
    if(s.lesson==='compile')return 'G1 stays with caller\nC1: '+(s.clone?'compiler-owned':'not created')+'\n'+['MO','MAP','MOGG','MGP','MGP finalization','Mojo LIT/KGEN/POP','LLVM dialect → LLVM IR','target code','prepared artifact'][s.level]+'\n'+s.routine+'\nMEF bytes: '+(s.mef?'prepared':'pending')+'\nM1: '+(s.initialized?'initialized':'absent')+'\nWeight registry: not required by this example';
    if(s.id==='attention')return 'Tile '+(s.tile+1)+' / 2 (tiny view)\nscores: ['+s.scores+']\nm='+ (s.m===null?'MIN':fmt(s.m))+'   α='+fmt(s.alpha)+'\nl='+fmt(s.l)+'\nO=['+s.O.map(fmt)+']\nFinal result: ['+s.computed.map(fmt)+']\nOnly local score tiles; running m/l/O survive';
    return 'Inputs: '+(s.inputs?s.cpu?'CPU references':'device buffers':'not supplied')+'\n'+(s.id==='matmul'?`Selected ${s.algorithm}; k=${s.k}\nCopy ready: ${s.copyReady}; block ready: ${s.blockReady}\n`:s.id==='epilogue'?`Route: ${s.fusion?'eligible epilogue sketch':'separate-body sketch'}\n`:'Illustrative SIMD group: '+(s.lane+1)+'\n')+'Accumulator: ['+s.acc.map(fmt)+']\nResult: ['+s.computed.map(fmt)+']\nBuffer ready: '+s.ready+'\nExplicit host read: '+s.hostRead;
  }
  class Lesson {
    constructor(node){this.node=node;this.kind=node.dataset.lesson;this.speed=1;this.playing=false;this.model=new ExampleModel(this.kind,data.example,options());this.visual=new window.CompilerExamplesVisual(node,reduced);
      node.addEventListener('click',e=>{const action=e.target.closest('[data-action]')?.dataset.action;if(action==='play')this.setPlaying(!this.playing);else if(action){this.setPlaying(false);if(action==='reset')this.model.configure(options());else this.model[action]();this.render(action==='step');}const index=e.target.closest('button[data-index]')?.dataset.index;if(index!==undefined){this.setPlaying(false);this.model.seek(Number(index));this.render(false);}});
      node.querySelector('[data-seek]').addEventListener('input',e=>{this.setPlaying(false);this.model.seek(Number(e.target.value));this.render(false);});
      node.querySelector('[data-speed]').addEventListener('change',e=>{this.speed=Number(e.target.value);if(this.playing)this.nextAt=performance.now()+1650/this.speed;});this.render(false);
    }
    setPlaying(value){if(value&&this.model.snapshot().done){this.model.configure(options());this.render(false);}this.playing=value;if(value){this.visual.resume();this.nextAt=performance.now()+350;schedule();}else this.visual.pause();this.updatePlay();}
    updatePlay(){const b=this.node.querySelector('[data-action="play"]');b.textContent=this.playing?'Ⅱ Pause':'▶ Play';b.setAttribute('aria-pressed',String(this.playing));this.node.dataset.playing=String(this.playing);}
    render(animate){const s=this.model.snapshot();Object.assign(this.node.dataset,{index:s.index,frames:s.total,phase:s.phase,state:JSON.stringify(s)});this.visual.render(s,animate,this.speed);if(animate)schedule();
      this.node.querySelector('[data-progress]').textContent=(s.index+1)+' / '+s.total;const seek=this.node.querySelector('[data-seek]');seek.max=s.total-1;seek.value=s.index;
      this.node.querySelector('[data-action="back"]').disabled=s.index===0;this.node.querySelector('[data-action="step"]').disabled=s.done;
      this.node.querySelector('[data-event-title]').textContent=s.title;this.node.querySelector('[data-event-detail]').textContent=s.detail;this.node.querySelector('[data-event-sources]').innerHTML=sourceLinks(s.refs);
      const pairs=s.lesson==='build'?[['Input types',s.signature[0]],['G1',s.verified?'verified':s.graph?'recording':'not created'],['Executable',0]]:s.lesson==='compile'?[['Target',s.target],['Selected body',s.routine],['Compiled bytes',s.code?'ready':'pending'],['M1',s.initialized?'ready':'not initialized']]:[['Placement',s.cpu?'CPU':'accelerator'],['Output cells',s.computed.length],['Result',s.ready?'ready':'pending'],[s.id==='attention'?'Key tiles (tiny)':'Host read',s.id==='attention'?s.tileCount:s.hostRead?'explicit':'not yet']];
      this.node.querySelector('[data-state-strip]').innerHTML=pairs.map(([k,v])=>'<span>'+escape(k)+' <b>'+escape(v)+'</b></span>').join('');
      this.node.querySelector('[data-code-preview]').textContent=codePreview(s);this.node.querySelector('[data-state-preview]').textContent=statePreview(s);
      this.node.querySelector('[data-transcript]').innerHTML=this.model.frames.map((f,i)=>'<li><button data-index="'+i+'">'+String(i+1).padStart(2,'0')+'</button><strong>'+escape(f.title)+'</strong> '+escape(f.detail)+'</li>').join('');this.updatePlay();
    }
  }
  document.querySelectorAll('[data-interactive]').forEach(e=>e.hidden=false);document.querySelectorAll('[data-lesson]').forEach(n=>controllers.push(new Lesson(n)));
  function refreshOptions(){for(const c of controllers){c.setPlaying(false);c.model.configure(options());c.render(false);}const s=controllers[1].model.snapshot();document.getElementById('algorithm-select')?.toggleAttribute('disabled',s.cpu);document.querySelector('[data-actual]').textContent='Original dimensions: '+s.signature.join(' → ')+(data.example.id==='matmul'||data.example.id==='attention'?' · '+s.params.grid:'')+(data.example.id==='attention'?' · query BN='+s.params.BN+', feature BD='+s.params.BD+', key tile='+s.params.keyTile:'');}
  document.querySelectorAll('.example-options select').forEach(e=>e.addEventListener('change',refreshOptions));refreshOptions();
  const observer=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)controllers.find(c=>c.node===e.target)?.setPlaying(false);}),{threshold:.1});controllers.forEach(c=>observer.observe(c.node));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)controllers.forEach(c=>c.setPlaying(false));});reduced.addEventListener('change',()=>controllers.forEach(c=>c.render(false)));document.querySelector('[data-print]').addEventListener('click',()=>print());
  window.CompilerReaderUI(data,controllers);window.compilerExampleSnapshot=key=>controllers.find(c=>c.kind===key)?.model.snapshot();
})();
