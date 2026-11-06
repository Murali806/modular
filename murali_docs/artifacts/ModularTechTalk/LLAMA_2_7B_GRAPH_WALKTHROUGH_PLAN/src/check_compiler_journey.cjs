'use strict';
const assert=require('assert/strict'),fs=require('fs'),path=require('path');
const {JourneyModel}=require('./compiler_journey_model.js');
const task=path.resolve(__dirname,'..'),manifest=JSON.parse(fs.readFileSync(path.join(task,'compiler_examples_manifest.json'))),routes=[];
for(const e of manifest.examples)for(const target of ['cpu','cuda','rocm'])for(const cache of ['cold','kernel-hit','artifact-hit']){
  const m=new JourneyModel(e.example.id,{target,cache}),phases=m.frames.map(f=>f.phase);
  assert.ok(phases.indexOf('package')<phases.indexOf('import'));
  assert.ok(phases.indexOf('import')<phases.indexOf('graph'));
  assert.ok(phases.indexOf('verify')<phases.indexOf('clone'));
  assert.ok(phases.indexOf('init')<phases.indexOf('dispatch1'));
  const last=m.seek(999);assert.equal(last.call,2);assert.ok(last.compiled&&last.initialized&&last.result);
  assert.equal(last.compileRuns,cache==='cold'?1:0);assert.equal(last.graphLowerings,cache==='artifact-hit'?0:1);
  assert.equal(last.deviceCode,target!=='cpu');
  if(cache==='artifact-hit')assert.ok(!phases.includes('map')&&!phases.includes('kernel_lookup'));
  if(cache==='kernel-hit')assert.ok(phases.includes('map')&&!phases.includes('elaborate')&&phases.includes('runtime_calls'));
  if(cache==='cold')assert.ok(phases.indexOf('elaborate')<phases.indexOf('runtime_calls'));
  if(cache!=='artifact-hit')assert.ok(phases.indexOf('verify')<phases.indexOf('device_finalize')&&phases.indexOf('device_finalize')<phases.indexOf('emit_mojo')&&phases.indexOf('emit_mojo')<phases.indexOf('kernel_lookup'));
  assert.ok(phases.indexOf('load_modules')<phases.indexOf('init'));
  if(e.example.id==='attention'&&target==='cpu')assert.deepEqual(m.frames.find(f=>f.phase==='compute1').refs,['math_cpu']);
  for(const f of m.frames){assert.ok(f.refs.every(key=>e.references[key]),f.refs.join(','));assert.ok(f.before&&f.after);if(f.submitted)assert.ok(f.initialized&&f.input);}
  for(const selection of ['operation','load','store','plan']){m.reset();m.select(selection);let old=m.index;const s=m.nextChange();assert.ok(s.index>old);assert.ok(s.changed.includes(selection));const index=s.index;m.back();assert.equal(m.index,index-1);assert.equal(m.selection,selection);}
  const untouched=m.seek(7);untouched.history.push('corrupt');assert.ok(!m.snapshot().history.includes('corrupt'));
  m.reset();assert.equal(m.index,0);assert.equal(m.snapshot().call,0);
  if(e.example.id==='matmul')assert.ok(phases.indexOf('early_buffers')<phases.indexOf('graph'));
  routes.push({example:e.example.id,target,cache,frames:m.frames.length});
}
for(const e of manifest.examples){const {ExampleModel}=require('./compiler_examples_model.js');const m=new ExampleModel('build',e.example);assert.ok(m.frames.findIndex(f=>f.phase==='register')<m.frames.findIndex(f=>f.phase==='custom'));}
fs.writeFileSync(path.join(task,'validation/compiler_journey_model.json'),JSON.stringify({evidence:'CPU-RUN · deterministic JavaScript teaching model; no MAX compilation or inference',routes,checks:['Package/import/custom-call order','Artifact hit bypasses graph lowering; kernel hit retains new graph planning','Init precedes numerical calls; code reused across two calls','Source references resolve; next-change filtering and immutable seek/reset','CPU/GPU target code roles and original matmul buffer ordering']},null,2)+'\n');
console.log('Compiler journey model passed:',routes.length,'routes');
