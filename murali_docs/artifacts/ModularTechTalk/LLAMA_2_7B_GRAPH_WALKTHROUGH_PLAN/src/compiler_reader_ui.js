/* Shared transient cards and offline original-source explorer. */
(function(root) {
  root.CompilerReaderUI=function(data,controllers) {
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;", "'":"&#39;"}[c]));
  function sourceLinks(refs) { return refs.map(key=>{const r=data.references[key];return '<a data-ref="'+key+'" href="../sources/'+r.file+'#L'+r.line+'">'+escape(r.label)+' <small>['+r.alias+':L'+r.line+']</small></a>';}).join(' · '); }
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
  };
})(window);
