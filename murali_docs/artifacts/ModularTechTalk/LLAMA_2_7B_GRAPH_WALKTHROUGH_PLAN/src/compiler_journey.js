(function(root) {
  'use strict';
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  root.CompilerJourney={attach(data) {
    const node=document.querySelector('[data-journey]');
    const reduced=matchMedia('(prefers-reduced-motion: reduce)');
    const links=keys=>keys.map(key=>{const r=data.references[key];if(!r)throw Error('Unknown journey source '+key);return '<a data-ref="'+key+'" href="../sources/'+r.file+'#L'+r.line+'">'+escape(r.label)+' ['+r.alias+':L'+r.line+']</a>';}).join(' · ');
    let raf=null;
    const schedule=()=>{if(raf===null)raf=requestAnimationFrame(tick);};
    function tick(now){raf=null;if(c.playing&&now>=c.nextAt){c.model.step();c.render(true);c.nextAt=now+1600/c.speed;if(c.model.snapshot().done){c.playing=false;c.updatePlay();}}const moving=c.visual.tick(now);if(c.playing||moving)schedule();}
    const c={node,kind:'journey',playing:false,speed:1,
      options(){return {target:document.getElementById('target-select').value,cache:node.querySelector('[data-cache]').value,fusion:document.getElementById('fusion-select')?.value||'fused',algorithm:document.getElementById('algorithm-select')?.value||'tiled'};},
      setPlaying(value){if(value&&this.model.snapshot().done){this.model.reset();this.render(false);}this.playing=value;if(value){this.visual.resume();this.nextAt=performance.now()+350;schedule();}else this.visual.pause();this.updatePlay();},
      updatePlay(){const b=node.querySelector('[data-action="play"]');b.textContent=this.playing?'Ⅱ Pause':'▶ Play';b.setAttribute('aria-pressed',String(this.playing));node.dataset.playing=String(this.playing);},
      refresh(){const selected=this.model.selection;this.setPlaying(false);this.model.configure(this.options());this.model.select(selected);this.render(false);},
      render(animate){const s=this.model.snapshot();Object.assign(node.dataset,{state:JSON.stringify(s),index:s.index,frames:s.total,phase:s.phase});this.visual.render(s,animate,this.speed);if(animate)schedule();
        node.querySelector('[data-progress]').textContent=(s.index+1)+' / '+s.total;const seek=node.querySelector('[data-seek]');seek.max=s.total-1;seek.value=s.index;
        for(const action of ['step','nextChange'])node.querySelector('[data-action="'+action+'"]').disabled=s.done;
        node.querySelector('[data-action="back"]').disabled=s.index===0;
        node.querySelector('[data-event-title]').textContent=s.title;node.querySelector('[data-event-detail]').textContent=s.detail;
        node.querySelector('[data-event-sources]').innerHTML=links(s.refs);
        node.querySelector('[data-before]').textContent=s.before;node.querySelector('[data-after]').textContent=s.after;
        const clock=node.querySelector('[data-clock]');clock.dataset.runtime=String(s.stage==='runtime');clock.textContent=(s.stage==='runtime'?'EXECUTION':'PREPARATION')+' · call '+s.call+' · new kernel builds total '+s.compileRuns;
        node.querySelector('[data-follow]').value=s.selection;
        const current=root.CompilerJourneyModel.STAGES.indexOf(s.stage);
        node.querySelectorAll('[data-stage]').forEach(button=>{const n=root.CompilerJourneyModel.STAGES.indexOf(button.dataset.stage);button.dataset.complete=String(n<current);if(n===current)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');button.disabled=!this.model.frames.some(f=>f.stage===button.dataset.stage);});
        node.querySelector('[data-state-strip]').innerHTML=[['Follow',s.selection],['C1',s.graphLevel],['Kernel view',s.level],['CM1',s.compiled?'prepared':'pending'],['M1',s.initialized?'executable':'pending'],['Preparation cache',s.cacheHit?s.cache:'miss']].map(([k,v])=>'<span>'+escape(k)+' <b>'+escape(v)+'</b></span>').join('');
        let key=s.selection==='plan'?s.refs[0]:s.stage==='source'||s.phase==='graph'?'entry':'math';
        if(s.id==='attention'&&s.cpu&&key==='math')key='math_cpu';
        const ref=data.references[key],lines=data.files[ref.file].text.split('\n');
        const needles=s.selection==='load'?['.load','copy','InputTensor']:s.selection==='store'?['foreach[','dst[','copy_from(O_i)','OutputTensor']:s.selection==='operation'?['return x','def elementwise','def execute','Graph(']:[];
        let start=ref.start-1;for(const needle of needles){const i=lines.findIndex((line,i)=>i>=ref.start-1&&i<ref.end&&line.includes(needle));if(i>=0){start=Math.max(ref.start-1,i-2);break;}}
        node.querySelector('[data-source-summary]').textContent='SOURCE ancestry · '+s.selection+' · '+ref.file;
        node.querySelector('[data-source-preview]').textContent=lines.slice(start,Math.min(ref.end,start+11)).map((line,i)=>String(start+i+1).padStart(4)+'  '+line).join('\n');
        node.querySelector('[data-source-link]').innerHTML=links([key]);
        const zoom=node.querySelector('[data-zoom]');zoom.href='#'+(data.example?'run':s.stage==='graph'?'lower':s.stage==='target'?'target':s.stage==='artifact'?'package':s.stage==='runtime'?'runtime':'mojo');
        zoom.textContent=data.example?(s.id==='attention'?'online-softmax state / tile storage ↓':s.id==='matmul'?'tile copies / barriers / accumulators ↓':s.id==='epilogue'?'fusion / intermediate allocations ↓':'typed loads / +1 / stores ↓'):'open the focused '+s.stage+' view ↓';
        node.querySelector('[data-transcript]').innerHTML=this.model.frames.map((f,i)=>'<li><button data-index="'+i+'">'+String(i+1).padStart(2,'0')+'</button><strong>'+escape(f.title)+'</strong> '+escape(f.detail)+' '+links(f.refs)+'</li>').join('');this.updatePlay();
      }
    };
    c.model=new root.CompilerJourneyModel.JourneyModel(data.example?.id||'addition',c.options());c.visual=new root.CompilerJourneyVisual(node,reduced);
    node.addEventListener('click',e=>{const action=e.target.closest('[data-action]')?.dataset.action;if(action==='play')c.setPlaying(!c.playing);else if(action){c.setPlaying(false);c.model[action]();c.render(action==='step'||action==='nextChange');}
      const stage=e.target.closest('[data-stage]')?.dataset.stage;if(stage){c.setPlaying(false);c.model.seek(c.model.frames.findIndex(f=>f.stage===stage));c.render(false);}
      const lineage=e.target.closest('[data-lineage]')?.dataset.lineage;if(lineage){c.setPlaying(false);c.model.select(lineage);c.render(false);}
      const index=e.target.closest('button[data-index]')?.dataset.index;if(index!==undefined){c.setPlaying(false);c.model.seek(Number(index));c.render(false);}
    });
    node.addEventListener('keydown',e=>{const lineage=e.target.closest('[data-lineage]')?.dataset.lineage;if(lineage&&['Enter',' '].includes(e.key)){e.preventDefault();c.setPlaying(false);c.model.select(lineage);c.render(false);}});
    node.querySelector('[data-follow]').addEventListener('change',e=>{c.setPlaying(false);c.model.select(e.target.value);c.render(false);});
    node.querySelector('[data-cache]').addEventListener('change',()=>c.refresh());
    node.querySelector('[data-seek]').addEventListener('input',e=>{c.setPlaying(false);c.model.seek(Number(e.target.value));c.render(false);});
    node.querySelector('[data-speed]').addEventListener('change',e=>{c.speed=Number(e.target.value);if(c.playing)c.nextAt=performance.now()+1600/c.speed;});
    document.querySelectorAll('#target-select,#fusion-select,#algorithm-select').forEach(e=>e.addEventListener('change',()=>c.refresh()));
    new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)c.setPlaying(false);},{threshold:.1}).observe(node);
    document.addEventListener('visibilitychange',()=>{if(document.hidden)c.setPlaying(false);});reduced.addEventListener('change',()=>c.render(false));
    c.render(false);root.compilerJourneySnapshot=()=>c.model.snapshot();return c;
  }};
})(window);
