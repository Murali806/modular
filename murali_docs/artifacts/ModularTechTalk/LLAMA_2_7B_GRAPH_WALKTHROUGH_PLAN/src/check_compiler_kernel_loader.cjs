'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),{execFileSync}=require('node:child_process');
const task=path.resolve(__dirname,'..'),repo=path.resolve(task,'../../../..'),page=fs.readFileSync(path.join(task,'link_htmls/max_compiled_kernels_context_load_and_driver.html'),'utf8'),data=JSON.parse(page.match(/<script type="application\/json" id="loader-data">([\s\S]*?)<\/script>/)[1]),model=require('./compiler_kernel_loader_model.js');
assert.equal(new Set(data.events.map(e=>e.id)).size,data.events.length);assert.equal(data.sections.length,16);assert.ok(data.sections.every(([section])=>data.events.some(e=>e.section===section)));
const hashes=new Map();for(const [file,f]of Object.entries(data.files)){
  const original=fs.readFileSync(f.origin==='LLVM upstream'?path.join(task,f.snapshot):path.join(repo,file));
  if(f.origin==='LLVM upstream'){assert.equal(f.revision,data.llvm_revision);assert.ok(f.url.includes('/'+data.llvm_revision+'/'));assert.ok(f.text.startsWith('//==='));}
  else assert.deepEqual(original,execFileSync('git',['show',data.revision+':'+file],{cwd:repo,maxBuffer:12*1024*1024}));
  assert.equal(original.toString(),f.text);assert.equal(crypto.createHash('sha256').update(original).digest('hex'),f.sha256);assert.deepEqual(fs.readFileSync(path.join(task,'sources',file)),original);hashes.set(file,f.sha256);
}
assert.ok(data.files[data.sources['llvm-config']].text.includes('LLVM_COMMIT = "'+data.llvm_revision+'"'));
const answer=fs.readFileSync(path.join(__dirname,'compiler_kernel_loader_answer.md'));assert.equal(crypto.createHash('sha256').update(answer).digest('hex'),data.answer_sha256);
for(const e of data.events){assert.ok(e.inputs.length>25&&e.outputs.length>25&&e.why.length>25);assert.ok(e.tree.length>=2);for(const [,key,line]of e.tree){assert.ok(data.sources[key]);assert.ok(line>=1&&line<=data.files[data.sources[key]].text.split('\n').length);}}
const checks=[],scenarios=[];
for(const route of ['cold','reuse','debug','missing'])for(const target of ['cuda','hip','cpu'])for(const deviceCache of ['miss','hit']){
  const frames=model.frames(data,{route,target,deviceCache}),ids=frames.map(f=>f.event.id),end=frames.at(-1);let size=0;
  for(const f of frames){
    assert.ok(f.state.table>=size&&f.state.table<=size+1);size=f.state.table;assert.ok(f.state.resolved>=f.state.table);if(f.state.table)assert.ok(f.state.namespace);if(f.state.loadStatus==='success')assert.equal(f.state.table,2);if(f.state.submitted)assert.ok(f.state.deviceFunction&&f.state.hostInvoked);
    const d=f.detail;assert.ok(model.LENSES.some(([key])=>key===d.lens));
    if(f.state.relocationPatched)assert.ok(d.link.copied&&d.link.external);
    if(f.state.finalized)assert.ok(d.link.patched);
    if(f.state.symbolStage==='Resolved'){assert.equal(d.orc.canReturn,false);assert.equal(d.table.entries.length,0);}
    if(f.state.symbolStage==='Emitted'){assert.ok(d.link.finalized);assert.equal(d.orc.canReturn,false);}
    assert.deepEqual(d.data.input,f.state.repeat?[10,11,12,13]:[0,1,2,3]);
    if(d.data.added)assert.deepEqual(d.data.registers,d.data.input.map(x=>x+1));
    if(d.data.stored)assert.deepEqual(d.data.output,d.data.input.map(x=>x+1));
    if(d.data.hostCopy)assert.ok(d.data.complete&&target!=='cpu');
    assert.equal(d.table.entries.length,f.state.table);
  }
  if(route==='missing'){assert.equal(end.event.id,'missing');assert.equal(end.state.table,1);assert.equal(end.state.loadStatus,'error');assert.equal(end.state.executing,false);assert.ok(!ids.includes('execute'));assert.ok(frames.find(f=>f.event.id==='arguments').event.outputs.includes('missing_entry'));}
  else {assert.equal(end.state.table,2);assert.equal(end.state.loadStatus,'success');assert.equal(end.state.repeat,true);assert.equal(end.state.complete,true);}
  if(route==='reuse'){assert.ok(ids.includes('copy')&&ids.includes('reuse')&&ids.includes('release-copy')&&ids.includes('lookup'));assert.ok(!ids.includes('define')&&!ids.includes('materialize')&&!ids.includes('relocations'));assert.equal(end.state.copyReleased,true);}
  if(route==='debug'){assert.ok(ids.includes('debug-link')&&ids.includes('debug-generator'));assert.ok(!ids.includes('lock')&&!ids.includes('materialize'));assert.equal(end.state.generator,true);}
  if(route==='cold'||route==='missing'){
    const ordered=['materialize','link-graph','prune','memory','section-copy','assigned-addresses','dependency-lookup','relocation-value','relocations','finalize-memory','emitted','link-ready','resolve','append'];
    for(let i=1;i<ordered.length;i++)assert.ok(ids.indexOf(ordered[i-1])<ids.indexOf(ordered[i]),ordered[i]);
    assert.equal(frames.find(f=>f.event.id==='assigned-addresses').detail.orc.required,'Ready');
  }
  if(target==='cpu'){assert.ok(!ids.includes('device-load')&&!ids.includes('launch'));assert.equal(end.state.deviceModule,false);}
  else if(route!=='missing'){assert.equal(ids.includes('device-load'),deviceCache==='miss');assert.equal(ids.includes('device-reuse'),deviceCache==='hit');assert.ok(ids.indexOf('launch')<ids.indexOf('completion'));if(target==='hip')assert.equal(frames.find(f=>f.event.id==='device-cache').event.source,'hip');}
  if(route!=='missing'){
    const repeated=frames.find(f=>f.event.id==='repeat');assert.deepEqual(repeated.detail.data.output,[null,null,null,null]);assert.equal(repeated.state.complete,false);
    assert.equal(repeated.detail.launch.arguments[0].value,'0x80030000');assert.equal(repeated.detail.launch.arguments[1].value,'0x80040000');
    assert.ok(ids.indexOf('repeat-read')<ids.indexOf('repeat-add')&&ids.indexOf('repeat-add')<ids.indexOf('repeat-store')&&ids.indexOf('repeat-store')<ids.indexOf('repeat-completion'));
    if(target!=='cpu'){assert.ok(ids.indexOf('pack')<ids.indexOf('launch')&&ids.indexOf('launch')<ids.indexOf('work-read')&&ids.indexOf('device-work')<ids.indexOf('completion')&&ids.indexOf('completion')<ids.indexOf('transfer'));assert.equal(end.detail.data.hostId,'H2');}
  }
  const t=new model.Timeline(data,{route,target,deviceCache});t.seek(frames.length-1);const snap=t.snapshot();snap.state.table=-1;snap.detail.link.fixup.after[0]='ff';assert.equal(t.snapshot().state.table,end.state.table);assert.equal(t.snapshot().detail.link.fixup.after[0],'30');t.back();t.seek(0);assert.equal(t.snapshot().state.table,0);
  scenarios.push({route,target,deviceCache,steps:frames.length-1});
}
checks.push('24 scenarios: ordered appends, partial error, exact cache skips, host/device distinction, debug branch, later submission/completion, immutable snapshots and Reset');
const fix=model.FIXUP;assert.equal(BigInt(fix.targetAddress)+BigInt(fix.addend),BigInt(fix.value));assert.equal(BigInt('0x'+fix.after.slice().reverse().join('')),BigInt(fix.value));assert.deepEqual(fix.after,['30','04','00','10','12','7f','00','00']);
checks.push('9 transformation lenses: exact Pointer64 arithmetic/encoding, correct pinned LLVM phase order, Resolved versus Ready gate, value loads/adds/stores, explicit host copy and fresh second-call argument pointers');
assert.ok(!page.includes('@@')&&!page.includes('[DETAIL:'));assert.ok(page.includes('id="preserved-answer"'));assert.ok(page.includes('id="tree-view"'));
const report={result:'PASS',sourceFiles:hashes.size,events:data.events.length,scenarios:scenarios.length,checks,evidence:'SOURCE and teaching-state checks; no MAX compilation or GPU execution'};fs.writeFileSync(path.join(task,'validation/compiler_kernel_loader_model.json'),JSON.stringify({...report,routes:scenarios},null,2)+'\n');console.log(JSON.stringify(report,null,2));
