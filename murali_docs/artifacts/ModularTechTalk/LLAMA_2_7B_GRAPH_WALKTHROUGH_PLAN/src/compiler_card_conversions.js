/* Per-card symbolic construction: retained source, moving record, accumulating output. */
(()=>{
'use strict';
const states=new WeakMap(),active=new Set();
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function visible(s){return s.root.isConnected&&!document.hidden&&s.root.checkVisibility({checkVisibilityCSS:true})&&s.root.getBoundingClientRect().bottom>0&&s.root.getBoundingClientRect().top<innerHeight;}
function pose(s,x,y){s.x=x;s.y=y;s.packet.setAttribute('transform',`translate(${x},${y})`);}
function playing(s,on){s.playing=on;s.root.dataset.conversionPlaying=String(on);s.root.querySelector('[data-conversion-action="play"]').textContent=on?'Pause':'Play';if(on){if(s.paused){s.start=performance.now()-s.elapsed;s.paused=false;}active.add(s);s.last=performance.now();schedule();}else{if(s.moving){s.paused=true;s.elapsed=performance.now()-s.start;}active.delete(s);}}
function render(s){
 const f=s.frames[s.index];s.root.dataset.conversionIndex=String(s.index);
 s.root.querySelector('[data-conversion-title]').textContent=f.title;s.root.querySelector('[data-conversion-detail]').textContent=f.detail;
 s.root.querySelector('[data-conversion-count]').textContent=`${s.index} / ${s.frames.length-1}`;
 s.root.querySelector('[data-conversion-packet-label]').textContent=f.packet;
 s.root.querySelectorAll('[data-capi-phase]').forEach(r=>r.setAttribute('opacity',Number(r.dataset.capiPhase)<=s.index?'1':'0'));
 s.root.querySelectorAll('[data-conversion-record]').forEach((r,i)=>r.setAttribute('opacity',i<s.index?'1':'0'));
 s.root.querySelector('[data-conversion-action="back"]').disabled=s.index===0;
 s.root.querySelector('[data-conversion-action="step"]').disabled=s.index===s.frames.length-1;
 if(s.root.dataset.validationScenarios)renderValidation(s);
}
function renderValidation(s){
 const root=s.root,i=s.index,f=s.frames[i],scenario=root.dataset.validationScenario;
 const show=(sel,on)=>root.querySelectorAll(sel).forEach(e=>e.setAttribute('opacity',on?'1':'0'));
 show('[data-validation-phase="1"]',i>=1);show('[data-validation-phase="2"]',i>=2);show('[data-validation-phase="4"]',i>=4);
 show('[data-validation-extra]',scenario==='extra');show('[data-validation-input-map]',i>=5);
 show('[data-validation-output-map]',i>=6&&scenario!=='extra');show('[data-validation-rejected-input]',i>=5&&scenario==='extra');
 const gate=(key,status,label)=>{root.querySelector(`[data-validation-gate="${key}"]`).setAttribute('class','validation-'+status);root.querySelector(`[data-validation-label="${key}"]`).textContent=label;};
 gate('name',i<3?'pending':scenario==='unknown'?'error':'pass',i<3?'Name lookup pending':scenario==='unknown'?'NO ENTRY · add_one_typo':'MATCH · add_one → AddOne');
 gate('device',i<4?'pending':'pass',i<4?'Device selection pending':'MATCH · device-compatible decl');
 gate('params',i<7?'pending':'pass',i<7?'Parameter check pending':'PASS · supplied parameters={}');
 const error=!!f.failure,done=i===8;
 gate('outcome',error?'error':done?'pass':'pending',error?'ValueError → STOP; no successful return':done?'SUCCESS → [TensorValue(%r)]':'Validation pending · wrappers not returned');
 root.dataset.validationOutcome=error?'error':done?'success':'pending';
 const result=root.querySelector('[data-validation-result]');result.querySelector('text:last-child').textContent=error?'Exception; no successful result':done?'[TensorValue(%r)] · symbolic':'Not returned yet';
 root.querySelector('[data-validation-symbol]').textContent='symbol = '+(scenario==='unknown'?'add_one_typo':'add_one');
 root.querySelectorAll('[data-validation-record-label]').forEach(e=>{e.textContent=s.frames[Number(e.dataset.validationRecordLabel)]?.record||'';});
 const source=root.querySelector('[data-validation-source]');source.dataset.ref=f.source;source.dataset.line=f.line;source.href='../sources/'+f.sourceFile+'#L'+f.line;source.textContent=f.sourceLabel+' · L'+f.line;
}
function target(s,index){return s.frames[index].point||[index?334:22,index?215:110];}
function move(s,index,animate){
 s.from=s.root.dataset.validationScenarios?[s.x,s.y]:[22,110];s.to=target(s,index);
 s.index=index;s.paused=false;render(s);s.moving=animate&&!reduced.matches;s.start=performance.now();s.last=s.start;
 if(s.moving){pose(s,...s.from);s.root.querySelectorAll('[data-conversion-record]')[index-1]?.setAttribute('opacity','0');active.add(s);schedule();}
 else{pose(s,...s.to);if(!s.playing)active.delete(s);}
}
let raf=0;
function schedule(){if(!raf&&active.size)raf=requestAnimationFrame(tick);}
function tick(now){
 raf=0;
 for(const s of [...active]){
  if(!visible(s)){playing(s,false);continue;}
  if(s.moving){const t=Math.min(1,(now-s.start)/700),ease=t*t*(3-2*t);pose(s,s.from[0]+(s.to[0]-s.from[0])*ease,s.from[1]+(s.to[1]-s.from[1])*ease);if(t===1){s.moving=false;render(s);s.last=now;if(!s.playing)active.delete(s);}}
  else if(s.playing&&now-s.last>800){if(s.index<s.frames.length-1)move(s,s.index+1,true);else playing(s,false);}
 }
 schedule();
}
function mount(root){if(states.has(root))return;const frames=JSON.parse(root.dataset.conversionFrames);const s={root,frames,index:Math.min(Number(root.dataset.conversionIndex)||0,frames.length-1),packet:root.querySelector('[data-conversion-packet]'),playing:false,moving:false};states.set(root,s);render(s);pose(s,...target(s,s.index));root.dataset.conversionPlaying='false';root.querySelector('[data-conversion-action="play"]').textContent='Play';}
function scan(node){if(node.nodeType!==1)return;if(node.matches('[data-card-conversion]'))mount(node);node.querySelectorAll('[data-card-conversion]').forEach(mount);}
document.addEventListener('click',e=>{const b=e.target.closest('[data-conversion-action]');if(!b)return;const root=b.closest('[data-card-conversion]');mount(root);const s=states.get(root),action=b.dataset.conversionAction;if(action==='play'){if(!s.playing&&s.index===s.frames.length-1)move(s,0,false);playing(s,!s.playing);return;}playing(s,false);if(action==='reset')move(s,0,false);else if(action==='back')move(s,Math.max(0,s.index-1),false);else move(s,Math.min(s.frames.length-1,s.index+1),!!root.dataset.validationScenarios);});
document.addEventListener('change',e=>{if(!e.target.matches('[data-validation-select]'))return;const root=e.target.closest('[data-card-conversion]');mount(root);const s=states.get(root);playing(s,false);root.dataset.validationScenario=e.target.value;s.frames=JSON.parse(root.dataset.validationScenarios)[e.target.value];root.dataset.conversionFrames=JSON.stringify(s.frames);move(s,0,false);});
new MutationObserver(records=>{for(const r of records)for(const n of r.addedNodes)scan(n);}).observe(document.body,{childList:true,subtree:true});
scan(document.body);
document.addEventListener('visibilitychange',()=>{if(document.hidden)for(const s of [...active])playing(s,false);});
})();
