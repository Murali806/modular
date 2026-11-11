/* Lazy card controllers also work for preserved Hold windows and offline reading. */
(()=>{
'use strict';
const states=new WeakMap(),active=new Set(),reduced=matchMedia('(prefers-reduced-motion: reduce)');
let scheduled=false;
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(tick);}}
const fmt=n=>n===null?'−∞':Number.isInteger(n)?String(n):Number(n.toFixed(6)).toString(),vec=v=>'['+v.map(fmt).join(', ')+']';
const shown=s=>s.root.isConnected&&!document.hidden&&s.root.checkVisibility({checkVisibilityCSS:true})&&s.root.getBoundingClientRect().bottom>0&&s.root.getBoundingClientRect().top<innerHeight;
function get(root){
 if(states.has(root))return states.get(root);
 const frames=JSON.parse(root.dataset.amFrames),s={root,frames,index:Math.min(Number(root.dataset.amIndex)||0,frames.length-1),playing:false,animation:null,frozen:null,motionTo:'translateX(0)',last:0};
 states.set(root,s);render(s,false);return s;
}
function play(s,on){
 s.playing=on;s.root.dataset.amPlaying=String(on);s.root.querySelector('[data-am-action="play"]').textContent=on?'Pause':'Play';
 const packet=s.root.querySelector('[data-am-packet]');
 if(on){
  if(s.frozen&&packet&&s.frozen.remaining>0){s.animation=packet.animate([{transform:s.frozen.pose},{transform:s.motionTo}],{duration:s.frozen.remaining,fill:'forwards',easing:'ease-in-out'});s.frozen=null;}
  else s.animation?.play();
  s.last=performance.now();active.add(s);schedule();
 }else{
  if(packet&&s.animation){
   const pose=getComputedStyle(packet).transform,remaining=Math.max(0,s.animation.effect.getComputedTiming().endTime-(Number(s.animation.currentTime)||0));
   s.animation.cancel();s.animation=null;packet.style.transform=pose;s.frozen={pose,remaining};
  }
  active.delete(s);
 }
}
function sourceLinks(s,f){
 const target=s.root.querySelector('[data-am-sources]');target.replaceChildren();
 f.refs.forEach((r,i)=>{if(i)target.append(' · ');const a=document.createElement('a');a.dataset.ref=r.key;a.dataset.line=r.line;a.href='../sources/'+r.file+'#L'+r.line;a.textContent=r.label+' · L'+r.line;target.append(a);});
}
function numeric(s,f){
 const el=s.root;el.querySelector('[data-am-m]').textContent=fmt(f.m);el.querySelector('[data-am-l]').textContent=fmt(f.l);el.querySelector('[data-am-o]').textContent=vec(f.O);
 el.querySelector('[data-am-tile]').textContent=f.tile<0?'Q row [1,0] · inputs ready':`Key tile ${f.tile+1} / 2 · rows ${f.tile*2} and ${f.tile*2+1}`;
 const weights=['denominator','rescale','add','store'].includes(f.visible),rescale=['rescale','add','store'].includes(f.visible),added=['add','store'].includes(f.visible);
 el.querySelectorAll('[data-am-key]').forEach(row=>{const i=Number(row.dataset.amKey),offset=i-f.tile*2,isActive=offset>=0&&offset<2;row.dataset.amActive=String(isActive);row.querySelector('[data-am-score]').textContent=fmt(f.scores[i]);row.querySelector('[data-am-value]').textContent=vec(f.v[i]);row.querySelector('[data-am-exp]').textContent=isActive&&weights?fmt(f.E?.[offset]??0):'—';row.querySelector('[data-am-weight]').textContent=isActive&&weights?fmt(f.prob?.[offset]??0):'—';row.querySelector('[data-am-contribution]').textContent=isActive&&added?vec(f.v[i].map(x=>x*(f.prob?.[offset]??0))):'—';});
 const oldMass=weights?(f.alpha??0)*(f.oldL??0):0,newMass=weights?(f.E||[]).reduce((a,b)=>a+b,0):0,total=oldMass+newMass;
 el.querySelector('[data-am-old-mass]').style.width=(total?oldMass/total*100:0)+'%';el.querySelector('[data-am-new-mass]').style.width=(total?newMass/total*100:0)+'%';
 el.querySelector('[data-am-mass-text]').textContent=`Prior mass (pink) ${fmt(oldMass)} + tile mass (green) ${fmt(newMass)} = ${fmt(total)}; alpha=${fmt(weights?f.alpha??0:0)}`;
 el.querySelector('[data-am-old-o]').textContent=vec(rescale?f.oldContribution||[0,0]:f.oldO||[0,0]);el.querySelector('[data-am-new-o]').textContent=added?vec(f.newContribution||[0,0]):'pending PV';el.querySelector('[data-am-sum-o]').textContent=added?vec(f.newO||f.O):'pending add';
 el.querySelector('[data-am-reference]').textContent=f.reference?`Stable full-softmax reference ${vec(f.reference)} · ${f.match?'MATCH':'MISMATCH'} (browser check)`:'Retain m, l and normalized O between key tiles; no full score matrix required.';
}
function render(s,animate){
 const f=s.frames[s.index],previous=s.frames[Math.max(0,s.index-1)],el=s.root;el.dataset.amIndex=String(s.index);
 el.querySelector('[data-am-title]').textContent=f.title;el.querySelector('[data-am-detail]').textContent=f.detail;el.querySelector('[data-am-count]').textContent=`${s.index} / ${s.frames.length-1}`;
 el.querySelector('[data-am-action="back"]').disabled=s.index===0;el.querySelector('[data-am-action="step"]').disabled=s.index===s.frames.length-1;sourceLinks(s,f);
 if(f.kind==='online'){numeric(s,f);return;}
 el.querySelector('[data-am-before]').textContent=f.before;el.querySelector('[data-am-after]').textContent=f.after;
 el.querySelectorAll('[data-am-field]').forEach(row=>{const key=row.dataset.amField;row.querySelector('b').textContent=f.fields[key]||'pending';row.dataset.amChanged=String(f.fields[key]!==previous.fields[key]);});
 const packet=el.querySelector('[data-am-packet]');packet.querySelector('[data-am-packet-text]').textContent=f.title;
 const width=packet.parentElement.clientWidth-packet.offsetWidth;s.animation?.cancel();s.frozen=null;s.motionTo=`translateX(${Math.max(0,width)}px)`;
 s.animation=packet.animate([{transform:'translateX(0)'},{transform:s.motionTo}],{duration:animate&&!reduced.matches?850:1,fill:'forwards',easing:'ease-in-out'});
}
function tick(now){
 scheduled=false;
 for(const s of active){if(!shown(s)){play(s,false);continue;}if(now-s.last>1750){if(s.index===s.frames.length-1){play(s,false);continue;}s.index++;render(s,true);s.last=now;}}
 if(active.size)schedule();
}
document.addEventListener('click',e=>{
 const action=e.target.closest('[data-am-action],[data-am-seek]');if(!action)return;
 const root=action.closest('[data-attention-mechanism]');if(!root)return;
 const s=get(root),key=action.dataset.amAction;if(key==='play'){play(s,!s.playing);return;}
 play(s,false);
 if(action.dataset.amSeek!==undefined)s.index=Number(action.dataset.amSeek);
 else if(key==='reset')s.index=0;else s.index=Math.max(0,Math.min(s.frames.length-1,s.index+(key==='back'?-1:1)));
 render(s,true);
});
new MutationObserver(()=>{document.querySelectorAll('.reader-card-window:not([hidden]) [data-attention-mechanism]').forEach(root=>get(root));for(const s of active)if(!shown(s))play(s,false);}).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['hidden']});
document.addEventListener('visibilitychange',()=>{if(document.hidden)for(const s of active)play(s,false);});
document.addEventListener('toggle',()=>{for(const s of active)if(!shown(s))play(s,false);},true);
addEventListener('scroll',()=>{for(const s of active)if(!shown(s))play(s,false);},{passive:true});
window.attentionMechanismSnapshot=root=>{const el=typeof root==='string'?document.querySelector(root):root;if(!el)return null;const s=get(el);return structuredClone({...s.frames[s.index],index:s.index,playing:s.playing});};
})();
