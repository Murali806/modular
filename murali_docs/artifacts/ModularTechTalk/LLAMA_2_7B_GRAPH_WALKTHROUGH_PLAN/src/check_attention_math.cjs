'use strict';
const assert=require('node:assert/strict'),m=require('./compiler_attention_math.js');
const near=(x,y)=>assert.ok(Math.abs(x-y)<1e-12,`${x} != ${y}`),vector=(a,b)=>a.forEach((x,d)=>near(x,b[d]));
function prefix(q,count,max){
 const scores=m.K.slice(0,count).map(k=>m.dot(q,k)),peak=max??Math.max(...scores),e=scores.map(x=>Math.exp(x-peak)),l=e.reduce((a,b)=>a+b,0);
 return {l,O:[0,1].map(d=>e.reduce((s,x,j)=>s+x*m.V[j][d]/l,0))};
}
for(let row=0;row<2;row++){
 const frames=m.onlineFrames(row);assert.deepEqual(frames,m.onlineFrames(row));
 for(const f of frames){
  const line=f.refs[0][1],a=f.scene,v=f.values;
  if(line===360){v.E.forEach(x=>assert.ok(x>0&&x<=1));}
  if(line===363)near(a.lNew,prefix(a.q,(a.tile+1)*2,a.mNew).l);
  if(line===366)near(v.E.reduce((x,y)=>x+y,0)+(a.priorMass/a.lNew),1);
  if(line===367)vector(v.o,a.oldO.map(x=>x*a.oldL));
  if(line===368)vector(v.o,a.oldO.map(x=>x*a.oldL/a.lNew));
  if(line===373)vector(v.o,a.oldO.map(x=>x*a.alpha*a.oldL/a.lNew));
  if(line===374)vector(v.PV,[0,1].map(d=>v.E.reduce((s,p,i)=>s+p*m.V[a.tile*2+i][d],0)));
  if(line===375)vector(v.o,prefix(a.q,(a.tile+1)*2).O);
  if(line===377){near(v.l,prefix(a.q,(a.tile+1)*2).l);near(v.m,Math.max(...m.K.slice(0,(a.tile+1)*2).map(k=>m.dot(a.q,k))));}
 }
 const last=frames.at(-1);assert.equal(last.scene.match,true);vector(last.values.o,m.reference(m.Q[row]).O);
 const scale=frames.find(f=>f.refs[0][1]===355&&f.scene.tile===1).scene.alpha;near(scale,row===0?Math.exp(-2):1);
 const initial=frames[0];assert.deepEqual(initial.scene.scores,[null,null,null,null]);assert.equal(initial.scene.reference,null);
}
console.log('Attention math: every source-step intermediate, each tile prefix, growing/unchanged maxima and both query outputs match independent stable-softmax arithmetic.');
