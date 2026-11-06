(function() {
  "use strict";
  const data = JSON.parse(document.getElementById("engine-data").textContent);
  const objects = new Map(data.objects.map(o=>[o.id,o]));
  const $ = selector => document.querySelector(selector);
  const svg = $("#engine-svg"), nodes = $("#object-nodes"), edgeLayer = $("#active-edge");
  const ns = "http://www.w3.org/2000/svg";
  let model = new window.EngineLifecycleModel(data), playing = false, timer = null, raf = null;
  let cardContext = null, closeTimer = null, sourceOrigin = null, marker = null;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const make = (tag,attrs={},text="") => {
    const element=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([k,v])=>element.setAttribute(k,String(v)));
    if(text)element.textContent=text;
    return element;
  };
  function sourceLinks(keys) {
    const wrap=document.createElement("div");wrap.className="source-links";
    keys.forEach(key=>{
      const ref=data.references[key], link=document.createElement("a");
      link.href="../sources/"+ref.file+"#L"+ref.line;link.dataset.ref=key;
      link.textContent=ref.label+" ["+ref.alias+":L"+ref.line+"]";wrap.append(link);
    });
    return wrap;
  }
  data.phases.forEach((phase,i)=>{
    const button=document.createElement("button");button.textContent=String(i+1).padStart(2,"0")+" "+phase.title;
    button.dataset.phase=phase.id;button.addEventListener("click",()=>{pause();hideHover();model.jump(phase.id);render();});
    $("#chapter-nav").append(button);
  });
  data.objects.forEach(o=>{
    const row=document.createElement("div");row.className="legend-row";row.dataset.object=o.id;row.style.setProperty("--object-color",o.color);
    const swatch=document.createElement("span");swatch.className="swatch";row.append(swatch);
    const content=document.createElement("div"), title=document.createElement("strong"), kind=document.createElement("small"), role=document.createElement("p");
    title.textContent=o.name;kind.textContent=o.owner.toUpperCase()+" · "+o.kind;role.textContent=o.role;
    content.append(title,kind,role,sourceLinks(o.refs));row.append(content);$("#legend-rows").append(row);
  });
  document.querySelectorAll("[data-interactive]").forEach(n=>n.hidden=false);
  $("#svg-static-hint").remove();
  const alwaysPresent=new Set(['devices','pool','driver','gpu']);
  function nodeState(o,s) {
    if(o.id==='graph'||o.id==='contract')return s.graphState;
    if(o.id==='artifact')return s.artifactState;
    if(o.id==='lock')return s.lockState;
    if(o.id==='model')return s.modelState;
    if(o.id==='outputs')return s.outputsReady?'RETURNED':'NOT RETURNED';
    if(o.id==='handle')return s.compiledReady?'RESOLVED':s.created.includes(o.id)?'PENDING':'NO RESULT YET';
    if(o.id==='gpu')return s.inferenceStage==='DEVICE WORK'?'DEVICE WORK · SOURCE':'HARDWARE / BOUNDARY';
    if(o.id==='pool')return 'RUNTIME WORKER THREADS';
    if(o.id==='devices')return 'CPU / TARGET CONTEXT';
    if(s.active.includes(o.id))return 'ACTIVE CALL / DATA';
    return s.created.includes(o.id)?'INTRODUCED':'NOT YET INTRODUCED';
  }
  function placeColumn(ids,x,width,columns,s,positions,owner) {
    const rows=Math.ceil(ids.length/columns),h=80;
    const gapX=16,w=(width-(columns-1)*gapX)/columns;
    const pitch=rows>=5?99:rows===4?124:148;
    ids.forEach((id,i)=>{
      const o=objects.get(id),box={x:x+(i%columns)*(w+gapX),y:96+Math.floor(i/columns)*pitch,w,h};
      positions.set(id,box);
      const g=make('g',{class:'object-node '+owner+'-node',transform:`translate(${box.x} ${box.y})`,'data-object':id,
        'data-exists':alwaysPresent.has(id)||s.created.includes(id),'data-active':s.active.includes(id),tabindex:0,role:'button',
        'aria-label':o.name+'. '+o.owner+'. '+o.kind+'. '+nodeState(o,s)+'. Hover or focus for source.','aria-haspopup':'true'});
      g.style.setProperty('--object-color',o.color);
      g.append(make('rect',{width:w,height:h,rx:9}),make('rect',{class:'node-stripe',x:0,y:12,width:4,height:h-24,rx:2}));
      o.short.forEach((line,k)=>g.append(make('text',{class:'node-label',x:12,y:23+k*17},line)));
      g.append(make('text',{class:'node-kind',x:12,y:44},o.owner.toUpperCase()),make('text',{class:'node-state',x:12,y:65},nodeState(o,s)));
      g.addEventListener('mouseenter',()=>showHover('object',id,g));g.addEventListener('mouseleave',scheduleHide);
      g.addEventListener('focus',()=>showHover('object',id,g));g.addEventListener('blur',scheduleHide);
      g.addEventListener('click',()=>showHover('object',id,g,true));
      g.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();showHover('object',id,g,true);}});
      nodes.append(g);
    });
  }
  function drawEdge(edge,positions) {
    edgeLayer.replaceChildren();marker=null;
    if(!edge||!positions.has(edge[0])||!positions.has(edge[1]))return;
    const a=positions.get(edge[0]),b=positions.get(edge[1]);
    // Route active calls through the gutters so they do not cover object text.
    const ports=r=>[
      [{x:r.x+r.w,y:r.y+r.h/2},{x:r.x+r.w+6,y:r.y+r.h/2}],
      [{x:r.x,y:r.y+r.h/2},{x:r.x-6,y:r.y+r.h/2}],
      [{x:r.x+r.w/2,y:r.y},{x:r.x+r.w/2,y:r.y-6}],
      [{x:r.x+r.w/2,y:r.y+r.h},{x:r.x+r.w/2,y:r.y+r.h+6}]
    ];
    const blocked=(p,q)=>[...positions.values()].some(r=>{
      const left=r.x-2,right=r.x+r.w+2,top=r.y-2,bottom=r.y+r.h+2;
      if(p.x===q.x)return p.x>left&&p.x<right&&Math.max(p.y,q.y)>top&&Math.min(p.y,q.y)<bottom;
      return p.y>top&&p.y<bottom&&Math.max(p.x,q.x)>left&&Math.min(p.x,q.x)<right;
    });
    let best=null,score=Infinity,end;
    function consider(points,anchors){
      if(points.slice(1).some((p,i)=>blocked(points[i],p)))return;
      const length=points.slice(1).reduce((sum,p,i)=>sum+Math.abs(p.x-points[i].x)+Math.abs(p.y-points[i].y),0)+points.length*9;
      if(length<score){score=length;best=[anchors[0],...points,anchors[1]];end=anchors[1];}
    }
    const lanes=[5,24,186,367,386,676,985,1004,1235];
    for(const [sa,s] of ports(a))for(const [ta,t] of ports(b)){
      const anchors=[sa,ta];
      if(s.x===t.x||s.y===t.y)consider([s,t],anchors);
      consider([s,{x:t.x,y:s.y},t],anchors);consider([s,{x:s.x,y:t.y},t],anchors);
      for(const x of lanes)consider([s,{x,y:s.y},{x,y:t.y},t],anchors);
      for(const y of [84,618]){
        consider([s,{x:s.x,y},{x:t.x,y},t],anchors);
        for(const x1 of lanes)for(const x2 of lanes)consider([s,{x:x1,y:s.y},{x:x1,y},{x:x2,y},{x:x2,y:t.y},t],anchors);
      }
    }
    if(!best)return;
    const d=best.map((p,i)=>(i?"L":"M")+p.x+","+p.y).join(" ");
    const path=make("path",{class:"active-wire",d});edgeLayer.append(path);
    const label=make("text",{class:"edge-label",x:620,y:704,"text-anchor":"middle"},edge[2]);edgeLayer.append(label);
    const dot=make("circle",{class:"packet",cx:end.x,cy:end.y,r:6});edgeLayer.append(dot);
    marker={path,dot};
  }
  function detailStrip(s) {
    const strip=$('#detail-strip');strip.replaceChildren();
    const phase=data.phases.find(p=>p.id===s.phase);
    const phaseEvents=model.events.filter(e=>e.phase===s.phase),current=model.events[s.index];
    const stage=Math.max(0,Math.floor(phaseEvents.indexOf(current)*4/Math.max(1,phaseEvents.length)));
    const labels=s.phase==='inference'&&s.inferenceStage==='RESULT READINESS'
      ?['Input: ready','Primitive: dispatched','Result: pending → ready','User: callback activates']:phase.strip;
    labels.forEach((label,i)=>{
      const g=make('g',{class:'tile',transform:`translate(${24+i*304} 645)`,'data-recorded':i<=stage&&s.index>=0});
      g.append(make('rect',{width:280,height:37,rx:6}),make('text',{x:140,y:23,'text-anchor':'middle'},label));strip.append(g);
      if(i<3)strip.append(make('text',{class:'detail-label',x:316+i*304,y:669,'text-anchor':'middle'},'→'));
    });
  }
  function render() {
    cancelAnimation();
    const s=model.snapshot(),phase=data.phases.find(p=>p.id===s.phase),n=data.phases.indexOf(phase)+1;
    $('#animation').dataset.phase=s.phase;$('#animation').dataset.playing=playing;$('#animation').dataset.step=s.index;
    $('#left-label').textContent=phase.labels[0];$('#middle-label').textContent=phase.labels[1];$('#right-label').textContent=phase.labels[2];
    $('#left-status').textContent='ROUTE · '+s.options.route.toUpperCase();
    $('#middle-status').textContent=s.error||({graph:s.graphState,compiler:s.compileStage,mojo:s.mojoStage,mef:s.artifactState,init:s.modelState,inference:s.inferenceStage})[s.phase];
    $('#phase-title').textContent=phase.title;$('#lesson-button').textContent=String(n).padStart(2,'0');
    $('#lesson-button').setAttribute('aria-label','Lesson '+n+': '+phase.title);
    $('#current-event').textContent=s.title;$('#current-event').dataset.error=Boolean(s.error);
    $('#step-count').textContent=(s.index+1)+' / '+s.total;
    $('#timeline').max=s.total-1;$('#timeline').value=s.index;$('#timeline-label').textContent='Step '+(s.index+1)+' / '+s.total;
    $('#event-source-links').replaceChildren(sourceLinks(s.refs));
    $('#previous').disabled=s.index<0;$('#step').disabled=s.done;
    $('#play').setAttribute('aria-pressed',String(playing));$('#play').textContent=playing?'Pause':'Play';
    document.querySelectorAll('#chapter-nav [data-phase]').forEach(button=>{
      button.setAttribute('aria-current',String(button.dataset.phase===s.phase));
      button.disabled=!model.events.some(e=>e.phase===button.dataset.phase);
    });
    document.querySelectorAll('[data-gate]').forEach(gate=>gate.dataset.ready=String(s[gate.dataset.gate]));
    const positions=new Map();nodes.replaceChildren();
    placeColumn(phase.left,32,308,1,s,positions,'left');
    placeColumn(phase.middle,396,560,2,s,positions,'middle');
    placeColumn(phase.right,1012,196,1,s,positions,'right');
    drawEdge(s.edge,positions);detailStrip(s);
    const list=$('#event-history');list.replaceChildren();
    if(!s.history.length){const li=document.createElement('li');li.textContent='Ready to start. Animation replays source logic.';list.append(li);}
    s.history.forEach((title,i)=>{const li=document.createElement('li');li.textContent=(i+1)+'. '+title;list.append(li);});
    list.scrollTop=list.scrollHeight;
    if(cardContext?.type==='lesson')showHover('lesson',s.phase,$('#lesson-button'));
    if(s.done&&playing)pause();
  }
  function cancelAnimation(){if(raf!==null)cancelAnimationFrame(raf);raf=null;}
  function animateMarker() {
    if(!marker||reduced.matches)return;
    const {path,dot}=marker,length=path.getTotalLength(),start=performance.now();
    const tick=now=>{const progress=Math.min(1,(now-start)/650),point=path.getPointAtLength(length*progress);dot.setAttribute("cx",point.x);dot.setAttribute("cy",point.y);if(progress<1)raf=requestAnimationFrame(tick);else raf=null;};
    raf=requestAnimationFrame(tick);
  }
  function pause(){playing=false;clearTimeout(timer);timer=null;cancelAnimation();$("#play").textContent="Play";$("#play").setAttribute("aria-pressed","false");$("#animation").dataset.playing="false";}
  function next(){hideHover();model.step();render();animateMarker();}
  function play(){if(model.snapshot().done)model.reset();playing=true;next();if(playing)timer=setTimeout(()=>{if(playing)play();},1500);}
  $("#step").addEventListener("click",()=>{pause();next();});
  $("#previous").addEventListener("click",()=>{pause();hideHover();model.previous();render();});
  $("#reset").addEventListener("click",()=>{pause();hideHover();model.reset();render();});
  $("#play").addEventListener("click",()=>playing?pause():play());
  $("#timeline").addEventListener("input",()=>{pause();hideHover();model.seek($("#timeline").value);render();});
  ['route','initialization'].forEach(key=>$('#'+key+'-mode').addEventListener('change',()=>{
    pause();hideHover();model=new window.EngineLifecycleModel(data,{route:$('#route-mode').value,initialization:$('#initialization-mode').value});render();
  }));
  function hideHover(returnFocus=false){clearTimeout(closeTimer);const previous=cardContext;cardContext=null;$("#hover-card").hidden=true;$("#lesson-button").setAttribute("aria-expanded","false");if(returnFocus&&previous?.trigger?.isConnected){previous.trigger.focus();$("#hover-card").hidden=true;$("#lesson-button").setAttribute("aria-expanded","false");cardContext=null;}}
  function scheduleHide(){clearTimeout(closeTimer);closeTimer=setTimeout(()=>{
    if($("#source-dialog").open)return;
    if(!$("#hover-card").matches(":hover")&&!$("#hover-card").contains(document.activeElement)&&!cardContext?.trigger?.matches(":hover")&&document.activeElement!==cardContext?.trigger)hideHover();
  },140);}
  function showHover(type,id,trigger,pinned=false){
    if($("#source-dialog").open)return;clearTimeout(closeTimer);
    const s=model.snapshot(),o=type==="object"?objects.get(id):data.phases.find(p=>p.id===id);
    cardContext={type,id,trigger,pinned};const card=$("#hover-card");card.hidden=false;
    card.style.setProperty("--object-color",type==="object"?o.color:"#263c5b");
    $("#hover-title").textContent=type==="object"?o.name:o.lesson;
    $("#hover-kind").textContent=type==="object"?o.owner.toUpperCase()+" · "+o.kind:"LESSON "+String(data.phases.indexOf(o)+1).padStart(2,"0");
    $("#hover-role").textContent=type==="object"?o.role:o.detail;
    $("#hover-sources").replaceChildren(sourceLinks(type==="object"?o.refs:s.refs));
    const box=trigger.getBoundingClientRect(),head=$(".topbar").getBoundingClientRect(),width=card.getBoundingClientRect().width,height=card.getBoundingClientRect().height;
    card.style.left=Math.max(12,Math.min(innerWidth-width-12,box.x)) +"px";
    const desired=box.bottom+7;card.style.top=Math.max(head.bottom+8,Math.min(innerHeight-height-12,desired))+"px";
    if(type==="lesson")trigger.setAttribute("aria-expanded","true");if(pinned)card.focus();
  }
  $("#lesson-button").addEventListener("mouseenter",e=>showHover("lesson",model.snapshot().phase,e.currentTarget));
  $("#lesson-button").addEventListener("focus",e=>showHover("lesson",model.snapshot().phase,e.currentTarget));
  $("#lesson-button").addEventListener("click",e=>showHover("lesson",model.snapshot().phase,e.currentTarget,true));
  $("#lesson-button").addEventListener("mouseleave",scheduleHide);$("#lesson-button").addEventListener("blur",scheduleHide);
  $("#hover-card").addEventListener("mouseenter",()=>clearTimeout(closeTimer));$("#hover-card").addEventListener("mouseleave",scheduleHide);$("#hover-card").addEventListener("focusout",scheduleHide);
  function closeLegend(returnFocus=false){$("#legend-card").hidden=true;$("#legend-button").setAttribute("aria-expanded","false");if(returnFocus)$("#legend-button").focus();}
  $("#legend-button").addEventListener("click",()=>{
    if(!$("#legend-card").hidden){closeLegend();return;}hideHover();$("#legend-card").hidden=false;$("#legend-button").setAttribute("aria-expanded","true");$("#legend-card").focus();
  });
  $("#legend-filter").addEventListener("input",()=>{const value=$("#legend-filter").value.toLowerCase();document.querySelectorAll(".legend-row").forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(value));});
  document.querySelector('[data-close="legend"]').addEventListener("click",()=>closeLegend(true));
  document.querySelector('[data-close="hover"]').addEventListener("click",()=>hideHover(true));
  function openSource(key,origin){
    const ref=data.references[key];sourceOrigin=origin;pause();
    $("#source-title").textContent=ref.symbol;$("#source-meta").textContent=ref.file+" · complete original file · definition L"+ref.start+"–L"+ref.end+" · selected L"+ref.line+" · SOURCE "+data.revision;
    const code=$("#source-code code");code.replaceChildren();
    data.files[ref.file].text.split("\n").forEach((text,i)=>{
      const row=document.createElement("span");row.className="source-line"+(i+1>=ref.start&&i+1<=ref.end?" in-definition":"")+(i+1===ref.line?" at-call":"");
      row.dataset.line=i+1;const number=document.createElement("i"),content=document.createElement("b");number.textContent=i+1;content.textContent=text;row.append(number,content);code.append(row);
    });
    $("#source-dialog").showModal();const container=$("#source-code"),target=code.querySelector(`[data-line="${ref.line}"]`);
    container.scrollTop=target.offsetTop-container.offsetTop-50;
  }
  document.addEventListener("click",e=>{
    const link=e.target.closest("a[data-ref]");if(link){e.preventDefault();openSource(link.dataset.ref,link);return;}
    if(!$("#source-dialog").open){if(!$("#legend-card").contains(e.target)&&!$("#legend-button").contains(e.target))closeLegend();if(!$("#hover-card").contains(e.target)&&!e.target.closest(".object-node,.ipc-slots,#lesson-button"))hideHover();}
  });
  $("#close-source").addEventListener("click",()=>$("#source-dialog").close());
  $("#source-dialog").addEventListener("close",()=>sourceOrigin?.focus());
  document.addEventListener("keydown",e=>{if(e.key!=="Escape"||$("#source-dialog").open)return;if(!$("#hover-card").hidden)hideHover(true);else if(!$("#legend-card").hidden)closeLegend(true);});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)pause();});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();},{threshold:.08}).observe($("#animation"));
  reduced.addEventListener("change",cancelAnimation);
  let printState=[];
  window.addEventListener("beforeprint",()=>{pause();printState=[...document.querySelectorAll("details")].map(n=>[n,n.open]);printState.forEach(([n])=>n.open=true);});
  window.addEventListener("afterprint",()=>{printState.forEach(([n,open])=>n.open=open);printState=[];});
  window.engineLifecycleSnapshot=()=>model.snapshot();
  render();
})();
