/* Numerical and lifecycle invariants; these checks do not run MAX or Mojo. */
'use strict';
const assert=require('assert/strict'),fs=require('fs'),path=require('path');
const {ExampleModel,matmul,A,B,TA,TB,Q,K,V}=require('./compiler_examples_model.js');
const task=path.resolve(__dirname,'..'),manifest=JSON.parse(fs.readFileSync(path.join(task,'compiler_examples_manifest.json'),'utf8'));
const checks=[],results=[];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const scores=K.map(k=>dot(Q,k)),max=Math.max(...scores),weights=scores.map(x=>Math.exp(x-max));
const reference=V[0].map((_,j)=>weights.reduce((s,w,i)=>s+w*V[i][j],0)/weights.reduce((s,w)=>s+w,0));
for(const e of manifest.examples)for(const target of ['cuda','rocm','cpu'])for(const variant of e.example.id==='epilogue'?['fused','separate']:e.example.id==='matmul'?['tiled','naive']:['default']) {
  const options={target,fusion:variant,algorithm:variant==='default'?'tiled':variant};
  const toolchain=new ExampleModel('toolchain',e.example,options),build=new ExampleModel('build',e.example,options),compile=new ExampleModel('compile',e.example,options),run=new ExampleModel('run',e.example,options);
  for(const model of [toolchain,build,compile,run]) {
    for(const [i,f] of model.frames.entries()){assert.ok(f.title&&f.detail);assert.ok(f.refs.every(r=>e.references[r]),f.refs.join(','));model.seek(i);assert.equal(model.snapshot().index,i);}
    model.seek(model.frames.length-1);const last=model.snapshot();last.computed.push(999);assert.ok(!model.snapshot().computed.includes(999));
    model.back();assert.equal(model.snapshot().done,false);model.reset();assert.equal(model.snapshot().index,0);
  }
  const backend=target==='cpu'?'HostBackend':target==='cuda'?'NVPTXBackend':'AMDGPUBackend';
  const required=['bazel_target','bazel_libraries','bazel_outputs','toolchain_runfiles','python_process','package_source','package_import','graph_record','clone','mo_opt','mo_to_map','map_fusion','map_to_mogg','mogg_to_mgp','mgp_plan','finalize','emit_mojo','jit_lookup','offload_split','target_info','select_backend','mojo_specialize','post_elaboration','llvm_dialect','llvm_translate','llvm_codegen','emit_target','host_device','runtime_calls','mef','init','execute','result_ready','evidence'];
  assert.deepEqual(toolchain.frames.map(f=>f.phase),required);
  assert.equal(toolchain.frames.length,e.toolchain.steps);
  const at=phase=>toolchain.frames.find(f=>f.phase===phase);
  assert.equal(at('bazel_outputs').package,false);
  assert.equal(at('package_source').graph,false);
  assert.deepEqual(['clone','mo_to_map','map_to_mogg','mogg_to_mgp'].map(p=>at(p).graphIR),['MO','MAP','MOGG','MGP']);
  assert.ok(toolchain.frames.slice(required.indexOf('graph_record')).every(f=>f.g1Retained));
  assert.ok(toolchain.frames.slice(required.indexOf('mogg_to_mgp')).every(f=>f.graphIR==='MGP'));
  assert.equal(at('llvm_dialect').kernelIR,'LLVM dialect / MLIR');
  assert.equal(at('llvm_dialect').llvmModule,false);
  assert.equal(at('llvm_translate').llvmModule,true);
  assert.equal(at('llvm_codegen').targetCode,false);
  assert.equal(at('mef').initialized,false);
  assert.equal(at('init').inputs,false);
  assert.equal(at('execute').ready,false);
  assert.equal(at('result_ready').ready,true);
  assert.ok(at('finalize').detail.includes('getIsDeviceGraph'));
  assert.equal(at('offload_split').deviceIR,target!=='cpu');
  assert.ok(toolchain.frames.every(f=>/^\d+(\.\d+)*$/.test(f.qa)&&f.representation&&f.refs.length));
  assert.equal(toolchain.frames.find(f=>f.phase==='select_backend').selectedBackend,backend);
  const emitted=toolchain.frames.find(f=>f.phase==='emit_target');
  assert.ok(emitted.detail.includes(target==='cuda'?'cubin':target==='rocm'?'ELF':'host object'));
  assert.equal(at('target_info').triple,target==='cuda'?'nvptx64-nvidia-cuda':target==='rocm'?'amdgcn-amd-amdhsa':'x86_64-unknown-linux-gnu');
  assert.equal(at('host_device').hostCode,true);
  assert.ok(toolchain.frames.at(-1).detail.includes('without claiming a completed Bazel or GPU run'));
  assert.equal(toolchain.frames[0].command,`./bazelw build //max/examples/custom_ops:${e.example.task}`);
  build.seek(999);assert.equal(build.snapshot().verified,true);assert.equal(build.snapshot().initialized,false);
  const beforeInit=compile.frames.find(f=>f.phase==='mef');assert.ok(beforeInit.mef&&!beforeInit.initialized);compile.seek(999);assert.ok(compile.snapshot().initialized&&compile.snapshot().code);
  run.seek(999);const s=run.snapshot();assert.ok(s.ready&&s.initialized);assert.equal(s.hostRead,e.example.id!=='attention');
  const expected=e.example.id==='addition'?[1,1.25,1.5,1.75]:e.example.id==='epilogue'?matmul(A,B).flat().map(x=>(x+1)*2):e.example.id==='matmul'?matmul(TA,TB).flat():reference;
  assert.equal(s.computed.length,expected.length);s.computed.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12,`${e.example.id} ${i}: ${v} / ${expected[i]}`));
  if(target==='cpu'){assert.ok(!run.frames.some(f=>['copy','barrier','wait','memset'].includes(f.phase)));if(e.example.id==='matmul')assert.equal(s.routine,'naive_matrix_multiplication_cpu');}
  if(e.example.id==='matmul'&&target!=='cpu'&&variant==='tiled') {
    for(const f of run.frames.filter(f=>f.phase==='mac'))assert.ok(f.shared&&f.copyReady&&f.blockReady);
    const waits=run.frames.filter(f=>f.phase==='wait');assert.equal(waits.length,2);assert.ok(waits.every(f=>f.copyReady&&!f.blockReady));
    assert.ok(run.frames.filter(f=>f.phase==='copy').every(f=>!f.copyReady&&!f.blockReady));assert.equal(s.params.BK,target==='rocm'?16:32);
  }
  if(e.example.id==='attention') {
    const second=run.frames.find(f=>f.phase==='max'&&f.tile===1);assert.equal(second.m,3);assert.ok(second.alpha<1&&second.alpha>0);
    assert.equal(s.tileCount,2);assert.equal(s.params.keyTile,target==='cpu'?4:8);assert.equal(s.params.BN,target==='cpu'?4:16);
    const rescaled=run.frames.find(f=>f.phase==='rescale'&&f.tile===1);assert.ok(rescaled.O.every((v,i)=>v<rescaled.oldO[i]));
  }
  if(e.example.id==='epilogue') {
    assert.equal(run.frames.some(f=>f.temporary.length),variant==='separate');
    assert.equal(s.tempLive,0);assert.equal(s.traffic.output,4);
    assert.deepEqual([s.traffic.reads,s.traffic.writes],variant==='separate'?[8,8]:[0,0]);
  }
  results.push({example:e.example.id,target,variant,frames:[toolchain.frames.length,build.frames.length,compile.frames.length,run.frames.length],backend,result:s.computed});
}
checks.push('Independent tiny arithmetic: add-one, composed matmul, full-K tiled matmul and full-softmax attention');
checks.push('Graph, compiled artifact and initialized executable are distinct; readiness/host copying follows each source');
checks.push('CPU routes omit GPU transfers/coordination; NVIDIA/AMD tile parameters follow source');
checks.push('Async copy completion precedes block consumption; shared reuse closes the gate');
checks.push('Bazel build inputs stay distinct from runtime graph packaging; TargetInfo selects CPU/NVIDIA/AMD LLVM backend routes');
checks.push('Thirty-four source-linked toolchain steps preserve G1 and the MGP plan while kernel/LLVM modules, code, artifact, handles and data become ready separately');
checks.push('Online-softmax second-tile rescaling preserves agreement with independent full softmax');
checks.push('Deterministic seek/back/reset and immutable snapshots; all source references resolve');
fs.writeFileSync(path.join(task,'validation/compiler_examples_model.json'),JSON.stringify({evidence:'CPU-RUN · JavaScript math/model checks only; no MAX/GPU execution',checks,results},null,2)+'\n');
console.log('Example model checks passed:',results.length,'routes');
