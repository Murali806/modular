'use strict';
// Independent arithmetic and scenario checks; this never runs a GPU kernel.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const M = require('./attention_walkthrough_model.js');
const task = path.resolve(__dirname, '..');

function inputData() {
  const text = fs.readFileSync(path.join(task, 'link_htmls/max_example_04_fused_attention_online_softmax.html'), 'utf8');
  return JSON.parse(text.match(/<script type="application\/json" id="attention-data">([\s\S]*?)<\/script>/)[1]);
}

function near(a, b, label) {
  assert.ok(Math.abs(a - b) < 1e-12, `${label}: ${a} versus ${b}`);
}

function fullAttention(q, k, v) {
  const scores = k.map(row => row.reduce((s, x, i) => s + x * q[i], 0));
  const max = Math.max(...scores);
  const masses = scores.map(x => Math.exp(x - max));
  const sum = masses.reduce((s, x) => s + x, 0);
  return v[0].map((_, d) => v.reduce((s, row, i) => s + masses[i] * row[d] / sum, 0));
}

function run() {
  const checks = [], data = inputData(), numeric = [];
  for (const q of [0, 1]) {
    const frames = M.mathFrames(q);
    let lastSeen = 0;
    for (const frame of frames) {
      const s = frame.math;
      assert.ok(s.seen >= lastSeen);
      lastSeen = s.seen;
      if (s.E) s.E.forEach(x => assert.ok(x > 0 && x <= 1));
      if (s.part === 'old-scaled') s.scaledO.forEach((x, d) => near(x, s.oldO[d] * s.alpha * s.oldL / s.newL, 'rescaled old output'));
      if (s.part === 'merge') s.O.forEach((x, d) => near(x, s.scaledO[d] + s.PV[d], 'old/current merge'));
      if (s.part === 'commit') {
        const reference = fullAttention(M.Q[q], M.K.slice(0, s.seen), M.V.slice(0, s.seen));
        s.O.forEach((x, d) => near(x, reference[d], `${s.seen}-key prefix`));
      }
    }
    const expected = fullAttention(M.Q[q], M.K, M.V), final = frames.at(-1).math.O;
    final.forEach((x, d) => near(x, expected[d], 'full attention'));
    // Old snapshots must be immutable when newer snapshots or runs are edited.
    const original = M.mathFrames(q)[0].math.Q[0][0];
    frames.at(-1).math.Q[0][0] = 999;
    assert.equal(frames[0].math.Q[0][0], original);
    assert.equal(M.Q[0][0], original);
    numeric.push({query:q, steps:frames.length, output:final, reference:expected});
  }
  checks.push('Both query rows agree with independent stable softmax at each committed key prefix and final output.');
  checks.push('Exponentials, old-output rescaling, merge arithmetic and snapshot independence are verified.');
  for (const key of ['compile', 'pipeline', 'artifacts']) {
    const cold = M.frames(data, key, {cache:'miss'}), warm = M.frames(data, key, {cache:'hit'});
    assert.ok(cold.length > warm.length);
    assert.ok(warm.some(f => f.mark === 'cache-hit'));
    for (const mark of ['specialize', 'llvm', 'device-code', 'host-code']) assert.ok(!warm.some(f => f.mark === mark));
    for (const mark of ['calls', 'MEF', 'C1']) assert.ok(warm.some(f => f.mark === mark));
    const nvidia = M.frames(data, key, {vendor:'nvidia'}).find(f => f.mark === 'device-code');
    const amd = M.frames(data, key, {vendor:'amd'}).find(f => f.mark === 'device-code');
    assert.match(nvidia.code, /cubin/);
    assert.match(amd.code, /amdgcn-amd-amdhsa/);
    assert.match(amd.refs, /amdgpu/);
    assert.equal(amd.source_html, null, 'AMD must not retain the CUDA source link');
  }
  const hip = M.frames(data, 'device-load', {vendor:'amd'});
  assert.equal(hip.find(f => f.mark === 'module').operation, 'hipModuleLoadData.');
  assert.match(hip.find(f => f.mark === 'attributes').operation, /NVIDIA-specific/);
  assert.doesNotMatch(hip.at(-1).code, /cuModule/);
  assert.deepEqual(M.frames(data, 'execute', {cache:'hit'}), M.frames(data, 'execute', {cache:'miss'}));
  checks.push('A native-code hit skips code generation while retaining runtime lowering; vendor changes select the correct backend, module APIs and source evidence.');
  const report={result:'PASS',checks,numeric,evidence:'JavaScript arithmetic and documentation scenarios only; no MAX/GPU parity claim'};
  fs.writeFileSync(path.join(task, 'validation/attention_walkthrough_model.json'), JSON.stringify(report, null, 2) + '\n');
  return report;
}
module.exports={run};
if (require.main === module) console.log(JSON.stringify(run(), null, 2));
