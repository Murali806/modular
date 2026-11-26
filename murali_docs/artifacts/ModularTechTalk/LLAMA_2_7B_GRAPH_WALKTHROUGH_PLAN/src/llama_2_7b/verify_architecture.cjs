/* Independent arithmetic and numerical checks; this does not run MAX. */
'use strict';
const assert=require('node:assert/strict');
const cost=require('./architecture_model.js'),tensor=require('./architecture_tensor.js');
const report={costCases:0,completeReductions:0,rotaryPairs:0,attentionRows:0,onlineVectors:0,cacheStates:0,addressCases:0};
const close=(a,b,tol=3e-11)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
function arrayClose(a,b){assert.equal(a.length,b.length);a.forEach((x,i)=>Array.isArray(x)?arrayClose(x,b[i]):close(x,b[i]));}
const row=(r,id)=>r.rows.find(x=>x.id===id);

// Golden counts independently worked through in architecture_math_reference_notes.md.
const prefill=cost.calculate(),decode=cost.calculate({T:1,C:512}),rect=cost.calculate({pairs:'rectangle'});
assert.equal(prefill.totals.mac,3350272671744);
assert.equal(rect.totals.mac,3384565301248);
assert.equal(decode.totals.mac,6741557248);
assert.equal(row(prefill,'q').mac,8589934592);
assert.equal(row(prefill,'gate').mac,23085449216);
assert.equal(row(prefill,'qk').mac,537919488);
assert.equal(row(prefill,'head').mac,131072000);
assert.equal(prefill.memory.weights,13476831232);
assert.equal(row(prefill,'embedding').weight,250*2**20);
assert.equal(row(prefill,'head').weight,250*2**20);
assert.equal(row(prefill,'q').weight,32*2**20);
assert.equal(row(prefill,'gate').weight,86*2**20);
assert.equal(row(prefill,'norm1').weight,8192);
assert.equal(prefill.memory.usedKV,256*2**20);
assert.equal(prefill.memory.allocatedKV,256*2**20);
assert.equal(decode.memory.usedKV,256.5*2**20);
assert.equal(decode.memory.allocatedKV,320*2**20);
assert.equal(decode.memory.paddingKV,63.5*2**20);
assert.equal(decode.memory.addedKV,524288);
assert.equal(row(decode,'kv').write,16384);
assert.equal(cost.calculate({T:1,C:4095}).memory.usedKV,2*2**30);
assert.equal(prefill.totals.flops,prefill.totals.mac*2);
assert.equal(row(prefill,'q').ins,'[512, 4096]','shape dimensions must not contain thousands commas');

for(const T of [1,7,128,512])for(const C of [0,127,512])for(const B of [1,3])for(const traffic of ['boundary','materialized']){
  const r=cost.calculate({T,C,B,traffic});report.costCases++;
  let pairs=0;
  // Count actual allowed positions, independently of the closed-form equation.
  for(let query=0;query<T;query++)for(let key=0;key<C+T;key++)if(key<=C+query)pairs++;
  assert.equal(row(r,'qk').mac,pairs*B*32*128);
  assert.equal(row(r,'pv').mac,pairs*B*32*128);
  const pageSet=new Set();for(let token=0;token<C+T;token++)pageSet.add(Math.floor(token/128));
  assert.equal(r.memory.pages,pageSet.size*B);
  assert.equal(row(r,'kv').write*32,r.memory.addedKV);
  assert.equal(row(r,'head').mac,131072000*B,'LAST_TOKEN head processes B rows, not all active tokens');
  assert.equal(r.memory.weights,13476831232,'weight residency is independent of batch, prompt and context');
  assert.equal(r.totals.traffic,r.totals.read+r.totals.write);
  const macRows=r.rows.filter(x=>x.id!=='attention').reduce((n,x)=>n+x.mac*x.repeat,0);
  assert.equal(r.totals.mac,macRows,'fused group must not duplicate MACs');
  close(r.totals.ai*r.totals.traffic,r.totals.flops);
  close(r.totals.requiredGBs*10e6,r.totals.traffic);
  assert.equal(row(r,'tokenizer').traffic,null);
  assert.equal(row(r,'sample').traffic,null);
  if(T===1)assert.equal(r.totals.mac,cost.calculate({T,C,B,traffic,pairs:'rectangle'}).totals.mac);
  if(traffic==='materialized'){
    const fused=cost.calculate({T,C,B});
    assert.equal(r.totals.mac,fused.totals.mac);
    assert.equal(r.totals.traffic-fused.totals.traffic,4*r.score*32,'S write/read + P write/read must account exactly once');
  }
}
for(const P of [16,32,64,128,256])for(const C of [P-1,P,2*P-1]){
  const r=cost.calculate({T:1,C,P,B:3});
  const allocatedBefore=new Set(Array.from({length:C},(_,i)=>Math.floor(i/P))).size;
  const allocatedAfter=new Set(Array.from({length:C+1},(_,i)=>Math.floor(i/P))).size;
  assert.equal(r.memory.pages-r.memory.pagesBefore,3*(allocatedAfter-allocatedBefore));report.costCases++;
}
for(const w of [.5,1,2,4])for(const k of [1,2,4]){
  const r=cost.calculate({w,k});assert.equal(r.totals.mac,prefill.totals.mac);
  assert.equal(r.memory.usedKV,256*2**20*k/2);
  assert.equal(row(r,'norm1').weight,8192,'low-bit matrix control does not quantize gamma');
  assert.equal(row(r,'q').weight,4096*4096*w);report.costCases++;
}
const extraReads=cost.calculate({T:1,C:512,kvReads:3});
assert.equal(extraReads.totals.traffic-decode.totals.traffic,2*decode.memory.usedKV);
const ms=cost.calculate({budgetMs:5});assert.equal(ms.totals.requiredGBs,2*prefill.totals.requiredGBs);
const rate=cost.calculate({bandwidth:500,tflops:50});assert.equal(rate.totals.memoryMs,2*prefill.totals.memoryMs);assert.equal(rate.totals.computeMs,2*prefill.totals.computeMs);
for(const invalid of [{T:0},{T:512,C:4000},{T:1.5},{B:0},{B:65},{k:0},{w:3},{P:17},{budgetMs:0},{bandwidth:1e-300},{bandwidth:NaN},{kvReads:1.5},{traffic:'unknown'}])assert.throws(()=>cost.calculate(invalid));


// Full-size operation probes: independent reductions over literal model extents.
assert.deepEqual([tensor.spec.D,tensor.spec.F,tensor.spec.H,tensor.spec.d,tensor.spec.N,tensor.spec.V,tensor.spec.P],[4096,11008,32,128,32,32000,128]);
assert.equal(tensor.events.length,581);
const event=(stage,layer=0)=>tensor.events.find(e=>e.stage===stage&&(e.layer===layer||e.layer===null));
for(let l=0;l<32;l++)assert.deepEqual(tensor.events.filter(e=>e.layer===l).map(e=>e.stage),tensor.block);
for(const pass of ['prefill','decode','next'])for(const l of [0,7,31])for(const stage of ['q','k','v','o','gate','up','down','head']){
  const op=tensor.operation(pass,event(stage,l)),K=stage==='down'?11008:4096;
  assert.equal(op.inner,K);assert.equal(op.left.cols,K);assert.equal(op.right.rows,K);
  assert.deepEqual(op.weightShape,[op.result.cols,K]);
  for(const [i,j]of [[0,0],[op.result.rows-1,op.result.cols-1]]){
    const p=tensor.probe(op,i,j);let dot=0;
    for(let k=0;k<K;k++){dot+=op.left.get(i,k)*op.right.get(k,j);if([0,127,128,K-2,K-1].includes(k))close(p.prefix[k+1],dot);}
    close(p.expected,dot);close(op.result.get(i,j),dot);
    assert.equal(p.prefix.length,K+1);report.completeReductions++;
  }
  const again=tensor.operation(pass,{...event(stage,l)});close(again.result.get(0,0),op.result.get(0,0));
}
for(const pass of ['prefill','decode','next'])for(const l of [0,15,31])for(const stage of ['norm1','norm2','finalnorm']){
  const op=tensor.operation(pass,event(stage,l)),i=op.result.rows-1,p=tensor.probe(op,i,4095);
  let squares=0;for(let c=0;c<4096;c++)squares+=op.left.get(i,c)**2;
  close(p.prefix[4096],squares);close(p.expected,op.left.get(i,4095)*op.right.get(0,4095)/Math.sqrt(squares/4096+1e-5));report.completeReductions++;
}
for(const l of [0,17,31])for(const h of [0,13,31])for(const pos of [0,127,511,512,513])for(const pair of [0,1,31,63])for(const kind of ['Q','K']){
  const a=tensor.raw(kind,l,h,pos,pair),b=tensor.raw(kind,l,h,pos,pair+64),theta=pos*10000**(-pair/64);
  const x=tensor.rotated(kind,l,h,pos,pair),y=tensor.rotated(kind,l,h,pos,pair+64);
  close(x,a*Math.cos(theta)-b*Math.sin(theta));close(y,a*Math.sin(theta)+b*Math.cos(theta));close(x*x+y*y,a*a+b*b);report.rotaryPairs++;
}
for(const pass of ['prefill','decode','next'])for(const l of [0,31])for(const h of [0,31]){
  const r=tensor.route(pass),a=tensor.attention(pass,l,h);
  for(const q of new Set([0,Math.floor((r.T-1)/2),r.T-1])){
    // Independent stable causal softmax followed by a complete 128-channel PV.
    const scores=Array.from({length:r.L},(_,j)=>{if(j>r.C+q)return -Infinity;let s=0;for(let c=0;c<128;c++)s+=a.q(q,c)*a.k(j,c);return s/Math.sqrt(128);});
    const m=Math.max(...scores),exp=scores.map(x=>Math.exp(x-m)),denom=exp.reduce((x,y)=>x+y,0),prob=exp.map(x=>x/denom);
    const full=Array.from({length:128},(_,c)=>prob.reduce((s,p,j)=>s+p*a.v(j,c),0));
    for(let j=0;j<r.L;j++){close(a.row(q).p[j],prob[j]);if(j>r.C+q)assert.equal(prob[j],0);}
    close(prob.reduce((x,y)=>x+y,0),1);
    for(const width of [1,127,128,256]){
      const tiles=tensor.online(a,q,width);arrayClose(tiles.at(-1).output,full);assert.equal(tiles.at(-1).end,r.L);report.onlineVectors++;
    }
    const pv=tensor.operation(pass,event('pv',l),h);for(const c of [0,127])close(tensor.probe(pv,q,c).expected,full[c]);
    const qk=tensor.operation(pass,event('qk',l),h),last=r.C+q;close(tensor.probe(qk,q,last).prefix[128]/Math.sqrt(128),scores[last]);report.attentionRows++;
  }
}
// RoPE → K store → QK and probabilities → PV share the same generated values.
for(const pass of ['prefill','decode','next']){
  const r=tensor.route(pass),rope=tensor.operation(pass,event('rope',31),31,'K'),kv=tensor.operation(pass,event('kv',31),31),qk=tensor.operation(pass,event('qk',31),31);
  for(const c of [0,63,64,127]){
    close(rope.result.get(r.T-1,c),kv.left.get(r.T-1,c));close(kv.left.get(r.T-1,c),qk.right.get(c,r.L-1));
  }
  for(let l=0;l<32;l++)for(const stage of tensor.block){
    const e=event(stage,l),op=tensor.operation(pass,e);
    for(const step of new Set([0,Math.floor(op.steps/2),op.steps])){
      const cache=tensor.cacheState(pass,e,step),at=tensor.block.indexOf(stage),store=tensor.block.indexOf('kv');
      const expected=Array.from({length:32},(_,k)=>k<l?r.L:k>l?r.C:at<store?r.C:at>store?r.L:r.C+step);
      assert.deepEqual(cache.lengths,expected);
      assert.equal(cache.used,expected.reduce((a,b)=>a+b,0)*16384);assert.equal(cache.assigned,Math.ceil(r.L/128)*64*2**20);report.cacheStates++;
    }
  }
  const end=tensor.cacheState(pass,event('sample'),32000);assert.deepEqual(end.lengths,Array(32).fill(r.L));
  assert.deepEqual(tensor.cacheState(pass,event('head'),0),end,'sampling never appends a new KV position');
}
for(const pos of [0,127,128,511,512,513,639])for(const l of [0,15,31])for(const h of [0,31])for(const c of [0,127])for(const kv of [0,1]){
  const a=tensor.address(pos,l,h,c,kv),physical=[3,1,5,2,7][Math.floor(pos/128)];
  // Independent strides for dimensions [page,kv,layer,slot,head,channel].
  const expected=physical*33554432+kv*16777216+l*524288+(pos%128)*4096+h*128+c;
  assert.equal(a.element,expected);assert.equal(a.byte,expected*2);assert.equal(a.slot,pos%128);report.addressCases++;
}
for(const bad of [[640,0,0,0],[-1,0,0,0],[0,32,0,0],[0,0,32,0],[0,0,0,128],[0,0,0,0,2]])assert.throws(()=>tensor.address(...bad));
for(const pass of ['prefill','decode','next']){
  const e=tensor.operation(pass,event('embedding'));assert.equal(e.right.rows,32000);assert.equal(e.right.cols,4096);
  close(e.result.get(e.result.rows-1,4095),e.right.get(tensor.token(e.C+e.T-1),4095));
  const s=tensor.operation(pass,event('swiglu',31)),x=s.left.get(s.T-1,11007),u=s.right.get(s.T-1,11007);
  close(s.result.get(s.T-1,11007),x/(1+Math.exp(-x))*u);
  const g=tensor.operation(pass,event('gather'));close(g.result.get(0,4095),g.left.get(g.T-1,4095));
  const sample=tensor.operation(pass,event('sample')),p=tensor.probe(sample,0,0);
  let best=-Infinity,id=-1;for(let j=0;j<32000;j++){const x=sample.left.get(0,j);if(x>best){best=x;id=j;}}
  assert.equal(p.ids[32000],id);assert.equal(p.expected,id);close(p.prefix[32000],best);
}
report.independentOracle='Full-size selected contractions and 128-channel stable attention; synthetic operands, not a model forward';
report.golden={prefillMACs:prefill.totals.mac,decodeMACs:decode.totals.mac,parameterBytes:prefill.memory.weights};
report.result='PASS';console.log(JSON.stringify(report,null,2));
