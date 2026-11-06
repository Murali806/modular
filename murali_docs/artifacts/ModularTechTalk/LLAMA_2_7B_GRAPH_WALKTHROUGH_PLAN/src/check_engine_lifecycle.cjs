'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {EngineLifecycleModel}=require('./engine_lifecycle_model.js');
const task=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(task,'engine_lifecycle_manifest.json'),'utf8'));
const ids=new Set(data.objects.map(o=>o.id)),refs=new Set(Object.keys(data.references));
assert.equal(ids.size,data.objects.length);assert.equal(new Set(data.objects.map(o=>o.color)).size,data.objects.length);
for(const e of data.events){
  e.create.concat(e.active).forEach(id=>assert.ok(ids.has(id),id));
  e.refs.forEach(ref=>assert.ok(refs.has(ref),ref));
  if(e.edge)e.edge.slice(0,2).forEach(id=>assert.ok(ids.has(id),id));
}
let scenarios=0,transitions=0;
for(const route of ['cold','cache','export','import','read','decline','missing','prejit']){
  for(const initialization of ['device','virtual','bad-layout']){
    const m=new EngineLifecycleModel(data,{route,initialization});scenarios++;
    assert.equal(m.snapshot().index,-1);const snapshots=[];
    while(!m.snapshot().done){
      const s=m.step();snapshots.push(s);transitions++;
      if(s.initialized)assert.ok(s.compiledReady,'initialization requires artifact');
      if(s.outputsReady)assert.ok(s.initialized,'outputs require executable model');
      if(s.compiledReady)assert.notEqual(s.lockState,'HELD','completed artifact releases graph compilation lock');
      if(s.options.route==='read')assert.equal(s.graphReady,false,'read bypasses Graph');
    }
    const final=m.snapshot();
    if(['missing','prejit'].includes(route))assert.ok(final.error && !final.compiledReady && !final.initialized && !final.outputsReady);
    else if(initialization==='bad-layout')assert.ok(final.compiledReady && final.error==='NONCONTIGUOUS WEIGHT' && !final.initialized);
    else if(initialization==='virtual'){
      assert.ok(!final.initialized && !final.outputsReady);
      assert.equal(final.error,['read','import'].includes(route)?'UNKNOWN GRAPH NAMES':'EXECUTION REFUSED');
    }else assert.ok(final.compiledReady && final.initialized && final.outputsReady && !final.error);
    if(route==='read')assert.ok(!final.history.some(t=>t.includes('records forward')));
    if(['cache','import','read','missing'].includes(route))assert.ok(!final.history.some(t=>t.startsWith('MAP:')||t.startsWith('Optimize parametric')));
    if(route==='export')assert.ok(final.exported);
    if(route==='prejit')assert.ok(final.history.includes('MGP emits Mojo routines')&&!final.history.includes('Lower / optimize into target code'));
    for(let i=snapshots.length-1;i>=0;i--)assert.deepEqual(m.seek(i),snapshots[i],'seek deterministically replays state');
    m.reset();assert.equal(m.snapshot().created.length,0);assert.equal(m.snapshot().compiledReady,false);
    m.step();m.previous();assert.equal(m.snapshot().index,-1);
  }
}
console.log(JSON.stringify({result:'PASS',scenarios,transitions,objects:data.objects.length,sourceTargets:refs.size,
  checks:['lifecycle gates','cache/import/read bypasses','export before initialization','explicit failure termination','virtual execution refusal','deterministic seek/reset'],
  evidence:'CPU-RUN animation model; no MAX execution'},null,2));
