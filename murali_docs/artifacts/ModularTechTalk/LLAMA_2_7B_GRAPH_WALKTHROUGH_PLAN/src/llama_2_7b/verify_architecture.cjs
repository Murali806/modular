/* Independent arithmetic and numerical checks; this does not run MAX. */
'use strict';
const assert=require('node:assert/strict');
const cost=require('./architecture_model.js'),toy=require('./architecture_toy.js');
const report={costCases:0,oracleForwards:0,cachedForwards:0,onlineCases:0,timelineEvents:0};
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

// A separate full-sequence decoder oracle: use loops and parameter tensors,
// without calling the toy's matmul/norm/RoPE/softmax/forward implementations.
const parameters=toy.parameters();
function denseOracle(input,weight){return input.map(x=>weight.map(w=>{let s=0;for(let k=0;k<x.length;k++)s+=x[k]*w[k];return s;}));}
function normOracle(input,gamma){return input.map(x=>{let squares=0;for(const v of x)squares+=v*v;const rms=Math.sqrt(squares/x.length+1e-5);return x.map((v,j)=>v/rms*gamma[j]);});}
function oracle(tokens){
  let x=tokens.map(id=>parameters.embedding[id].slice());const cache=[];
  for(let l=0;l<2;l++){
    const w=parameters.weights[l],n=normOracle(x,w.norm1),q=denseOracle(n,w.q),k=denseOracle(n,w.k),v=denseOracle(n,w.v);
    for(let i=0;i<tokens.length;i++)for(let h=0;h<2;h++)for(const vector of [q[i],k[i]]){
      const j=h*2,a=vector[j],b=vector[j+1];vector[j]=a*Math.cos(i)-b*Math.sin(i);vector[j+1]=a*Math.sin(i)+b*Math.cos(i);
    }
    cache.push({K:k,V:v});
    const context=Array.from({length:tokens.length},()=>Array(4).fill(0));
    for(let i=0;i<tokens.length;i++)for(let h=0;h<2;h++){
      const scores=[];for(let j=0;j<=i;j++)scores.push((q[i][h*2]*k[j][h*2]+q[i][h*2+1]*k[j][h*2+1])/Math.sqrt(2));
      const maximum=Math.max(...scores),e=scores.map(s=>Math.exp(s-maximum)),sum=e.reduce((a,b)=>a+b,0);
      for(let j=0;j<=i;j++)for(let d=0;d<2;d++)context[i][h*2+d]+=e[j]/sum*v[j][h*2+d];
    }
    const projected=denseOracle(context,w.o),residual=x.map((r,i)=>r.map((z,j)=>z+projected[i][j]));
    const normalized=normOracle(residual,w.norm2),g=denseOracle(normalized,w.gate),u=denseOracle(normalized,w.up);
    const z=g.map((r,i)=>r.map((g,j)=>g/(1+Math.exp(-g))*u[i][j])),down=denseOracle(z,w.down);
    x=residual.map((r,i)=>r.map((z,j)=>z+down[i][j]));
  }
  return {hidden:x,cache,logits:denseOracle(normOracle([x.at(-1)],parameters.finalGamma),parameters.outputWeight)};
}
let seed=1841;function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;}
for(let n=1;n<=6;n++)for(let trial=0;trial<10;trial++){
  const tokens=Array.from({length:n},()=>Math.floor(random()*6)),full=toy.forward(tokens),ref=oracle(tokens);
  arrayClose(full.hidden,ref.hidden);arrayClose(full.logits,ref.logits);
  for(let l=0;l<2;l++){arrayClose(full.cache[l].K,ref.cache[l].K);arrayClose(full.cache[l].V,ref.cache[l].V);}
  report.oracleForwards++;
  for(let split=1;split<n;split++){
    const prefix=toy.forward(tokens.slice(0,split)),saved=structuredClone(prefix.cache),next=toy.forward(tokens.slice(split),prefix.cache);
    assert.deepEqual(prefix.cache,saved,'a call must not mutate its caller-owned prefix snapshot');
    arrayClose(next.logits,ref.logits);arrayClose(next.hidden,ref.hidden.slice(split));
    for(let l=0;l<2;l++){arrayClose(next.cache[l].K.slice(0,split),saved[l].K);arrayClose(next.cache[l].V.slice(0,split),saved[l].V);}
    report.cachedForwards++;
  }
  if(n>1){const changed=tokens.slice();changed[n-1]=(changed[n-1]+1)%6;arrayClose(toy.forward(changed).hidden.slice(0,n-1),full.hidden.slice(0,n-1));}
}
for(const name of ['prefill','decode','crosspage']){
  const data=toy.route(name),frames=toy.timeline(data);assert.deepEqual(toy.route(name),data,'route reset is deterministic');
  let lengths=[data.C,data.C];
  for(const event of data.events){
    if(event.written)lengths[event.layer]++;
    assert.deepEqual(event.cache.map(c=>c.K.length),lengths);
    assert.deepEqual(event.cache.map(c=>c.V.length),lengths);
    if(event.kind==='matmul')for(const h of event.headMatrices||[event]){
      const a=h.left.values,b=h.right.values,out=h.result.values;
      for(let i=0;i<a.length;i++)for(let j=0;j<b[0].length;j++){
        let dot=0;for(let k=0;k<a[0].length;k++)dot+=a[i][k]*b[k][j];close(dot*(event.scale||1),out[i][j]);
      }
    }
    if(event.kind==='rope')for(const h of event.headMatrices)for(let i=0;i<h.left.values.length;i++){
      const x=h.left.values[i],y=h.result.values[i];for(const off of [0,2])close(x[off]**2+x[off+1]**2,y[off]**2+y[off+1]**2);
    }
    report.timelineEvents++;
  }
  assert.deepEqual(lengths,[data.C+data.T,data.C+data.T]);
  assert.equal(data.events.at(-1).stage,'sample');
  assert.deepEqual(data.events.at(-1).cache,data.events.at(-2).cache,'sampling must not append a future token');
  assert.equal(frames.filter(f=>f.term>0).length,frames.length-data.events.length);
  for(const ctx of data.contexts)for(let h=0;h<2;h++)for(let i=0;i<ctx.T;i++){
    close(ctx.probabilities[h][i].reduce((a,b)=>a+b,0),1);
    ctx.probabilities[h][i].forEach((v,j)=>{if(j>ctx.C+i)assert.equal(v,0);});
  }
}
// Online attention tested against independently normalized full weighted sums.
for(const n of [1,2,3,7,17])for(const dimension of [2,4])for(let query=0;query<n;query++)for(const tile of [1,2,3,8]){
  const q=Array.from({length:dimension},()=>random()*20-10),k=Array.from({length:n},()=>q.map(()=>random()*20-10));
  const v=k.map(()=>Array.from({length:3},()=>random()*10-5));
  const logits=k.slice(0,query+1).map(key=>q.reduce((s,x,i)=>s+x*key[i],0)/Math.sqrt(dimension)),max=Math.max(...logits);
  const p=logits.map(s=>Math.exp(s-max)),sum=p.reduce((a,b)=>a+b,0),expected=[0,0,0];
  for(let i=0;i<p.length;i++)for(let j=0;j<3;j++)expected[j]+=p[i]/sum*v[i][j];
  const actual=toy.online(q,k,v,query,tile);arrayClose(actual.output,expected);
  assert.equal(actual.tiles.at(-1).end,query+1);report.onlineCases++;
}
const extreme=toy.online([10000,10000],[[10000,10000],[-10000,-10000]],[[3,-2],[7,1]],1,1);arrayClose(extreme.output,[3,-2]);
report.independentOracle='Full-sequence, two-layer FP64 decoder with separate loops; cached-prefix and causality checks';
report.golden={prefillMACs:prefill.totals.mac,decodeMACs:decode.totals.mac,parameterBytes:prefill.memory.weights};
report.result='PASS';console.log(JSON.stringify(report,null,2));
