/* Stage navigation, offline source explorer and independently controlled cards. */
(function(){
  'use strict';
  const data=JSON.parse(document.getElementById('attention-data').textContent),M=window.AttentionModel,V=window.AttentionVisuals,W=window.CompilerCardWindows;
  const players=new Set(),windows=new Map(),settings={vendor:'nvidia',cache:'miss'},reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let selectedStage=data.chapters[0].id,serial=0;
  const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function refHTML(raw){
    return raw.replace(/\[\[([^:\]]+):([^\]]+)\]\]/g,(_,key,line)=>{
      if(key==='external'){const ref=data.external[line];return ref?`<a class="source-link external" href="${escape(ref.url)}" target="_blank" rel="noopener">↗ ${escape(ref.title)}</a>`:'';}
      const path=data.sourcePaths[key];return path?`<a class="source-link" data-source="${escape(key+':'+line)}" href="../sources/${escape(path)}#L${Number(line)}" title="${escape(path+':'+line)}">↗ ${escape(path.split('/').at(-1))}:${Number(line)}</a>`:'';
    });
  }
  function shown(el){return !!el.getClientRects().length&&!el.closest('[hidden],details:not([open])');}
  class Player{
    constructor(el){
      this.el=el;this.key=el.dataset.animation;this.options={...settings,query:0};this.index=0;this.phase=0;this.wait=0;this.playing=false;this.raf=0;this.last=0;this.speed=1;
      this.svg=el.querySelector('svg');this.svg.querySelector('marker').id='aw-arrow-'+(++serial);
      this.controls=el.querySelector('.player-controls');this.seek=el.querySelector('[data-seek]');
      this.frames=M.frames(data,this.key,this.options);this.setupScenarios();
      this.el.querySelector('.portable').open=false;
      this.controls.addEventListener('click',event=>{
        const b=event.target.closest('button');if(!b)return;
        if(b.hasAttribute('data-legend')){openLegend(b);return;}
        const action=b.dataset.action;
        if(action==='play'){this.playing?this.pause():this.play();return;}
        this.pause();this.phase=0;this.wait=0;
        if(action==='step')this.index=M.clampIndex(this.index+1,this.frames.length);
        if(action==='back')this.index=M.clampIndex(this.index-1,this.frames.length);
        if(action==='reset')this.index=0;
        this.paint();
      });
      this.seek.addEventListener('input',()=>{this.pause();this.index=Number(this.seek.value);this.phase=0;this.wait=0;this.paint();});
      el.closest('.floating-note')?.querySelector('[data-query]')?.addEventListener('change',event=>{this.options.query=Number(event.target.value);this.refresh(true);});
      const speed=document.createElement('label');speed.className='speed-label';speed.style.fontSize='11px';speed.innerHTML='Speed <select aria-label="Animation speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>';
      speed.querySelector('select').addEventListener('change',event=>{this.speed=Number(event.target.value);});
      this.controls.insertBefore(speed,this.controls.querySelector('[data-legend]'));
      this.fillMathTranscript();
      players.add(this);this.paint();
      this.observer=new IntersectionObserver(entries=>{if(entries[0]&&!entries[0].isIntersecting)this.pause();},{threshold:0});this.observer.observe(el);
      el.__player=this;
    }
    setupScenarios(){
      if(!['compile','pipeline','artifacts','init','execute','host-link','device-load','device-loading','launch','abi','completion','synchronization'].includes(this.key))return;
      const controls=document.createElement('div');controls.className='scenario-controls';
      controls.innerHTML='<label>GPU backend <select data-vendor aria-label="GPU backend illustration"><option value="nvidia">NVIDIA / CUDA</option><option value="amd">AMD / HIP</option></select></label>';
      if(['compile','pipeline','artifacts'].includes(this.key))controls.innerHTML+='<label>Native code cache <select data-cache aria-label="Native code cache route"><option value="miss">Cold compilation</option><option value="hit">Matching code hit</option></select></label>';
      controls.innerHTML+='<span class="scenario-note">Source-based route; no GPU is queried.</span>';
      controls.querySelector('[data-vendor]').value=settings.vendor;
      controls.querySelector('[data-cache]')&&(controls.querySelector('[data-cache]').value=settings.cache);
      controls.addEventListener('change',event=>{
        if(event.target.hasAttribute('data-vendor'))settings.vendor=event.target.value;
        if(event.target.hasAttribute('data-cache'))settings.cache=event.target.value;
        for(const p of players){p.options={...p.options,...settings};p.refresh(false);}
      });
      this.el.insertBefore(controls,this.controls);this.scenarios=controls;
    }
    refresh(reset){
      const mark=this.frames[this.index]?.mark;this.pause();this.frames=M.frames(data,this.key,this.options);
      const mapped=this.frames.findIndex(f=>f.mark===mark);this.index=reset?0:mapped<0?M.clampIndex(this.index,this.frames.length):mapped;this.phase=0;this.wait=0;
      if(this.scenarios){this.scenarios.querySelector('[data-vendor]').value=settings.vendor;const c=this.scenarios.querySelector('[data-cache]');if(c)c.value=settings.cache;}
      this.fillMathTranscript();
      this.paint();
    }
    fillMathTranscript(){
      this.el.querySelector('.portable ol').innerHTML=this.frames.map(f=>`<li><strong>${escape(f.title)}</strong><p>${escape(f.operation+' '+f.outgoing)} ${refHTML(f.refs)}</p><p class="reason">Why: ${escape(f.why)}</p></li>`).join('');
    }
    play(){
      if(!shown(this.el))return;
      if(this.index>=this.frames.length-1){this.index=0;this.phase=0;this.wait=0;}
      this.playing=true;this.last=0;this.button();this.raf=requestAnimationFrame(t=>this.tick(t));
    }
    pause(){this.playing=false;this.last=0;cancelAnimationFrame(this.raf);this.raf=0;this.button();}
    button(){const button=this.controls.querySelector('[data-action=play]');button.textContent=this.playing?'Pause':'Play';button.setAttribute('aria-pressed',String(this.playing));}
    tick(now){
      if(!this.playing)return;
      if(document.hidden||!shown(this.el)){this.pause();return;}
      const dt=this.last?Math.min(90,now-this.last)*this.speed:0;this.last=now;
      if(this.wait<1550){this.wait+=dt;}
      else if(this.index<this.frames.length-1){
        if(reduced.matches){this.index++;this.phase=0;this.wait=0;}
        else{this.phase=Math.min(1,this.phase+dt/950);if(this.phase>=1){this.index++;this.phase=0;this.wait=0;}}
        this.paint();
      }else{this.pause();return;}
      this.raf=requestAnimationFrame(t=>this.tick(t));
    }
    paint(){
      if(!this.frames.length)return;
      const current=this.frames[this.index],next=this.frames[this.index+1],f=this.phase>=.5&&next?next:current;
      const smooth=this.phase*this.phase*(3-2*this.phase);
      V.render(this.svg,this.key,current,next,smooth,this.options);
      this.el.querySelector('[data-event-title]').textContent=f.title;
      this.el.querySelector('[data-event-action]').textContent=f.operation;
      this.el.querySelector('[data-in]').textContent=f.incoming;
      this.el.querySelector('[data-out]').textContent=f.outgoing;
      this.el.querySelector('[data-why]').textContent=f.why;
      this.el.querySelector('[data-evidence]').innerHTML=f.source_html||refHTML(f.refs);
      this.el.querySelector('[data-count]').textContent=`${this.index+1} / ${this.frames.length}`;
      this.el.querySelector('[data-step-number]').textContent=String(this.index+1).padStart(2,'0');
      this.seek.max=this.frames.length-1;this.seek.value=this.index;
      this.controls.querySelector('[data-action=back]').disabled=this.index===0&&this.phase===0;
      this.controls.querySelector('[data-action=step]').disabled=this.index===this.frames.length-1;
      this.el.dataset.frame=String(this.index);this.el.dataset.mark=current.mark;this.el.dataset.progress=this.phase.toFixed(4);this.button();
    }
  }
  function initPlayers(scope){for(const el of scope.querySelectorAll('[data-animation]'))if(!el.__player&&!el.closest('.note-bank'))new Player(el);}
  function closeWindow(win,force=false){
    if(!force&&W.isHeld(win))return;
    win.querySelectorAll('[data-animation]').forEach(el=>el.__player?.pause());win.hidden=true;W.closed(win);
    const origin=win.__origin;if(force&&origin?.isConnected)origin.focus({preventScroll:true});
  }
  function createWindow(key,title,content,origin,cls=''){
    for(const player of players)player.pause();
    let win=windows.get(key);
    if(!win){
      win=document.createElement('aside');win.className='floating-note '+cls;win.setAttribute('role','dialog');win.setAttribute('aria-modal','false');
      const heading='note-title-'+(++serial);win.setAttribute('aria-labelledby',heading);win.dataset.windowKey=key;win.hidden=true;
      const head=document.createElement('header');head.className='note-header';head.innerHTML=`<h2 id="${heading}">${escape(title)}</h2><button type="button" data-window-close>Close</button>`;
      const body=document.createElement('div');body.className='reader-window-content';body.innerHTML=content;
      win.append(head,body);(document.fullscreenElement||document.body).append(win);
      W.register(win,{header:head,body,closeButton:head.querySelector('button'),close:()=>closeWindow(win,true)});windows.set(key,win);initPlayers(body);
      win.addEventListener('toggle',event=>{if(event.target instanceof HTMLDetailsElement&&!event.target.open)event.target.querySelectorAll('[data-animation]').forEach(el=>el.__player?.pause());},true);
    }
    if(win.hidden)W.opening(win);win.hidden=false;win.__origin=origin;W.sourceShown(win);
    // The shared helper owns stacking for opening, pointer and keyboard focus.
    // A separate z-index counter would prevent held windows coming to the front.
    win.querySelector('.reader-window-header').focus({preventScroll:true});
    return win;
  }
  function openCard(id,origin){const card=data.cards.find(c=>c.id===id);if(card)createWindow('card-'+id,card.title,card.html,origin);}
  function openLegend(origin){
    const content='<p>Identity labels stay stable while fields, references and resources change. Arrows name a call, reference, transformation or copy; they do not imply measured timing.</p><div class="legend-list">'+[
      ['#982258','Python / caller / host interface','Graph types, host callables and the current operation. P1/G1/C1/M1 are different objects.'],
      ['#315b85','Compiler / runtime structure','Native IR containers, context services, symbol namespaces and runtime slots. These run inside the host process unless a process boundary is explicitly drawn.'],
      ['#076045','Produced artifact / data / ready state','Prepared packages, byte resources, linked handles and device buffers. Code bytes, function handles and tensor pointers are distinct.'],
      ['#805421','Decision / pending work / alternate route','Cache or setup gate; the label states the condition. A cache hit skips specific work, not data execution.'],
      ['#596b60','Not yet present / external boundary','Empty slots and driver-managed physical details. GPU scheduling and code residency are not measured here.']
    ].map(([color,title,body])=>`<div><span class="legend-swatch" style="background:${color}"></span><section><strong>${title}</strong><p>${body}</p></section></div>`).join('')+'</div><p><strong>SOURCE</strong>: verified repository calls. <strong>BOUNDARY</strong>: LLVM/vendor implementation details. Tiny matrix arithmetic is calculated in this browser.</p>';
    createWindow('legend','Diagram legend · identity, ownership and readiness',content,origin);
  }
  function sourceRef(token){
    if(data.references[token])return data.references[token];
    const split=token.lastIndexOf(':'),key=token.slice(0,split),line=Number(token.slice(split+1)),file=data.sourcePaths[key];
    if(file&&Number.isInteger(line))return {file,line};
    return null;
  }
  function openSource(token,origin){
    let ref=sourceRef(token)||{file:data.sourcePaths.entry,line:41};
    let key='source';const existing=windows.get(key);
    if(existing&&!existing.hidden&&W.isHeld(existing))key='source-'+(++serial);
    const content='<p class="source-path"></p><div class="source-toolbar-fields"><select aria-label="Source file"></select><label>Line <input type="number" min="1" aria-label="Source line"></label><button type="button" data-go-line>Go</button></div><div class="source-scroll" tabindex="0" aria-label="Complete pinned source"><pre class="source-lines"></pre></div>';
    const win=createWindow(key,'Source explorer · pinned complete file',content,origin,'source-window'),select=win.querySelector('select'),input=win.querySelector('input');
    if(!select.options.length){
      for(const file of Object.keys(data.files).sort()){const opt=document.createElement('option');opt.value=file;opt.textContent=file;select.append(opt);}
      select.addEventListener('change',()=>show(select.value,1));
      win.querySelector('[data-go-line]').addEventListener('click',()=>show(select.value,Number(input.value)));
      input.addEventListener('keydown',event=>{if(event.key==='Enter')show(select.value,Number(input.value));});
    }
    function show(file,line){
      const item=data.files[file];if(!item)return;
      const lines=item.text.split('\n');line=Math.max(1,Math.min(lines.length,Math.trunc(line)||1));
      select.value=file;input.value=line;input.max=lines.length;
      win.querySelector('.source-path').textContent=`${file}:${line} · revision ${data.revision}`;
      const pre=win.querySelector('.source-lines');
      if(pre.dataset.file!==file){
        const fragment=document.createDocumentFragment();
        lines.forEach((text,i)=>{const span=document.createElement('span');span.className='source-line';span.dataset.line=String(i+1);const number=document.createElement('span');number.className='source-line-number';number.textContent=i+1;span.append(number,document.createTextNode(text||' '));fragment.append(span);});
        pre.replaceChildren(fragment);pre.dataset.file=file;
      }
      pre.querySelector('.is-selected')?.classList.remove('is-selected');const target=pre.children[line-1];target.classList.add('is-selected');
      requestAnimationFrame(()=>{const scroll=win.querySelector('.source-scroll');scroll.scrollTop=target.offsetTop-scroll.offsetTop-scroll.clientHeight/2+target.offsetHeight/2;});
      win.dataset.sourceFile=file;win.dataset.sourceLine=String(line);
    }
    show(ref.file,ref.line);
  }
  function setView(section,view){
    for(const tab of section.querySelectorAll('[data-view]')){const active=tab.dataset.view===view;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;}
    for(const panel of section.querySelectorAll('[data-view-panel]')){panel.hidden=panel.dataset.viewPanel!==view;if(panel.hidden)panel.querySelectorAll('[data-animation]').forEach(el=>el.__player?.pause());}
    section.dataset.selectedView=view;
  }
  function selectStage(id,{hash=true,scroll=false}={}){
    if(!data.chapters.some(c=>c.id===id))id=data.chapters[0].id;
    selectedStage=id;
    for(const section of document.querySelectorAll('main>.chapter')){
      section.hidden=section.dataset.stage!==id;
      if(section.hidden)section.querySelectorAll('[data-animation]').forEach(el=>el.__player?.pause());
    }
    for(const link of document.querySelectorAll('[data-stage-link]')){if(link.dataset.stageLink===id)link.setAttribute('aria-current','step');else link.removeAttribute('aria-current');}
    if(hash)history.replaceState(null,'','#stage-'+id);
    if(scroll)document.getElementById('stage-'+id).scrollIntoView({block:'start',behavior:'auto'});
  }
  function fromHash(){const hash=location.hash.slice(1);const id=hash.replace(/^stage-/,'');const aliases={toolchain:'build',commands:'build',run:'execute'};selectStage(aliases[id]||id,{hash:false});}
  document.documentElement.classList.add('js');
  for(const chapter of document.querySelectorAll('main>.chapter'))setView(chapter,'tree');
  initPlayers(document.querySelector('main'));fromHash();
  document.addEventListener('click',event=>{
    const button=event.target.closest('button,a');if(!button)return;
    if(button.hasAttribute('data-source')){event.preventDefault();openSource(button.dataset.source,button);return;}
    if(button.hasAttribute('data-open-source')){openSource('entry:41',button);return;}
    if(button.hasAttribute('data-card')){event.preventDefault();openCard(button.dataset.card,button);return;}
    if(button.hasAttribute('data-stage-link')){event.preventDefault();selectStage(button.dataset.stageLink);return;}
    if(button.hasAttribute('data-view')){const section=button.closest('.chapter');setView(section,button.dataset.view);if(button.dataset.view==='animation')section.scrollIntoView({block:'start',behavior:'auto'});return;}
    if(button.hasAttribute('data-prev-stage')||button.hasAttribute('data-next-stage')){const index=data.chapters.findIndex(c=>c.id===selectedStage)+(button.hasAttribute('data-next-stage')?1:-1);selectStage(data.chapters[M.clampIndex(index,data.chapters.length)].id,{scroll:true});return;}
    if(button.hasAttribute('data-fullscreen')){
      const el=button.closest('.chapter');
      (document.fullscreenElement?document.exitFullscreen():el.requestFullscreen()).catch(()=>{button.textContent='Fullscreen unavailable';});return;
    }
  });
  document.addEventListener('pointerdown',event=>{
    if(event.target.closest('.floating-note,[data-card],[data-source],[data-open-source],[data-legend]'))return;
    for(const win of windows.values())if(!win.hidden)closeWindow(win);
  });
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      const open=[...windows.values()].filter(w=>!w.hidden&&!W.isHeld(w));
      if(open.length){event.preventDefault();open.sort((a,b)=>Number(b.style.zIndex)-Number(a.style.zIndex));closeWindow(open[0],true);}return;
    }
    const tab=event.target.closest('[role=tab]');if(!tab||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const tabs=[...tab.parentElement.querySelectorAll('[role=tab]')],i=tabs.indexOf(tab);
    const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    tabs[next].click();tabs[next].focus();
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)for(const p of players)p.pause();});
  addEventListener('hashchange',fromHash);
  document.addEventListener('fullscreenchange',()=>{for(const b of document.querySelectorAll('[data-fullscreen]'))b.textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';});
  let printOpen=[];
  addEventListener('beforeprint',()=>{for(const p of players)p.pause();printOpen=[...document.querySelectorAll('main .portable:not([open]),.note-bank>details:not([open])')];for(const d of printOpen)d.open=true;});
  addEventListener('afterprint',()=>{for(const d of printOpen)d.open=false;printOpen=[];});
  // Small inspection interface used by focused browser checks; it never runs MAX.
  window.AttentionReader={data,players,windows,selectStage,setView,openCard,openSource,settings};
})();
