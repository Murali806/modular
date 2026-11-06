"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const Model = require("./inference_session_model.js");
const task = path.resolve(__dirname, "..");
const page = fs.readFileSync(path.join(task, "link_htmls/max_inference_session_devices_runtime_and_compilation.html"), "utf8");
const payload = JSON.parse(page.match(/<script id="inference-session-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const manifest = JSON.parse(fs.readFileSync(path.join(task, "inference_session_manifest.json"), "utf8"));
const digest = text => crypto.createHash("sha256").update(text).digest("hex");
const checks = [], allFrames = [];

assert.deepEqual(payload.references, manifest.references);
for (const [name, file] of Object.entries(payload.files)) {
  const snapshot = fs.readFileSync(path.join(task, file.snapshot), "utf8");
  assert.equal(file.text, snapshot, "Embedded source equals snapshot: " + name);
  assert.equal(digest(snapshot), file.sha256);
}
for (const [key, ref] of Object.entries(payload.references)) {
  const source = payload.files[ref.file].text.split(/(?<=\n)/).slice(ref.start - 1, ref.end).join("");
  assert.equal(digest(source), ref.sha256, "Complete definition: " + key);
  assert.ok(ref.start <= ref.line && ref.line <= ref.end, "Source call is inside definition");
}
assert.equal(payload.references.session.line, 664);
checks.push("10 complete embedded source files and 30 exact definition/call-site hashes");

let constructorCases = 0;
for (const [devices, record] of Object.entries(Model.DEVICE_CASES)) {
  const original = record.devices.slice();
  for (const store of ["none", "import", "export", "conflict"]) {
    for (const threads of [null, 1, 4, 8]) {
      for (const extension of [false, true]) {
        const frames = Model.constructorFrames({devices, store, threads, extension});
        const final = frames.at(-1);
        allFrames.push(...frames);
        constructorCases++;
        assert.deepEqual(record.devices, original, "An example must not mutate the inputs");
        assert.ok(frames.every(frame => frame.models === 0), "Construction never loads a model");
        assert.ok(frames.every(frame => !frame.context || frame.runtimeConfig), "Context follows runtime configuration");
        assert.ok(frames.every(frame => !frame.runtimeConfig || frame.native), "Runtime config belongs to native session");
        if (store === "conflict") {
          assert.ok(final.error && !final.native && !final.context && !final.returned);
          assert.deepEqual(final.normalized, [], "Conflict rejected before device loop");
        } else {
          assert.ok(final.returned && final.native && final.context && final.compileConfig);
          assert.equal(final.normalized.filter(d => d === "cpu").length, 1, "Exactly one host CPU");
          assert.equal(new Set(final.normalized).size, final.normalized.length, "Unique Device values");
          const firstOrder = original.filter((device, i) => original.indexOf(device) === i);
          assert.deepEqual(final.normalized.filter(d => original.includes(d)), firstOrder, "First occurrence order preserved");
          assert.equal(final.nativeThreads, threads === null ? 0 : threads);
          assert.equal(final.extension, extension);
        }
      }
    }
  }
}
assert.deepEqual(Model.normalizeDevices(["cpu", "gpu0", "cpu"]), ["cpu", "gpu0"]);
assert.deepEqual(Model.normalizeDevices([]), ["cpu"]);
checks.push(constructorCases + " constructor combinations: CPU inclusion, deduplication, input order, thread mapping, extension and MEF conflict");

for (const mode of ["cold", "mef", "missing"]) {
  const frames = Model.compilationFrames(mode), final = frames.at(-1);
  allFrames.push(...frames);
  let sawArtifactWithoutModel = false;
  for (const frame of frames) {
    assert.ok(frame.session);
    if (frame.artifact && !frame.initialized) sawArtifactWithoutModel = true;
    if (frame.initCalls) assert.ok(frame.artifact && frame.registry && !frame.error);
    if (frame.initialized) assert.ok(frame.initCalls && frame.artifact && frame.registry);
  }
  assert.ok(sawArtifactWithoutModel, "Artifact ready is distinct from executable ready");
  assert.equal(final.initialized, mode !== "missing");
  assert.equal(final.initCalls, mode === "missing" ? 0 : 1);
  if (mode === "missing") assert.ok(final.error && final.artifact && !final.initialized);
}
assert.ok(!Model.compilationFrames("mef").some(f => f.compilerStatus === "native compiler work"));
checks.push("Cold compile and MEF reuse both initialize; missing weights stop before native model initialization");

for (const mode of ["prefill_decode", "decode_only"]) {
  const frames = Model.executionFrames(mode);
  allFrames.push(...frames);
  let calls = 0;
  for (const f of frames) {
    assert.equal(f.compilationCount, 1);
    assert.equal(f.initializedCount, 1);
    assert.ok(f.calls >= calls && f.calls <= calls + 1);
    assert.ok(f.completed <= f.calls);
    calls = f.calls;
    if (f.tokens.length) {
      assert.equal(f.offsets[0], 0);
      assert.equal(f.offsets.at(-1), f.tokens.length);
      assert.equal(f.offsets.length, 3);
      assert.ok(f.offsets.every((n, i) => i === 0 || n >= f.offsets[i - 1]));
    }
  }
  assert.equal(frames.at(-1).completed, mode === "decode_only" ? 1 : 2);
  assert.equal(frames.at(-1).calls, frames.at(-1).completed);
}
checks.push("Packed prefill/decode buffers obey row offsets; batch execution reuses the decoder and session");

for (const mode of ["caller_first", "models_first"]) {
  const frames = Model.lifetimeFrames(mode);
  allFrames.push(...frames);
  for (const f of frames) {
    const owners = [f.caller, f.decoder, f.sampler].filter(Boolean).length;
    assert.equal(f.references, owners);
    assert.equal(f.retained, owners > 0, "A remaining pictured model retains native S1");
  }
  assert.deepEqual(frames.map(f => f.references), [3, 2, 1, 0]);
}
assert.ok(Model.lifetimeFrames("caller_first")[1].decoder && Model.lifetimeFrames("caller_first")[1].retained);
checks.push("Both release orders preserve the native context until the final pictured owner is gone");

for (const frame of allFrames) {
  assert.ok(frame.title && frame.detail && frame.refs.length > 0);
  for (const ref of frame.refs) assert.ok(payload.references[ref], "Frame has an exact source target: " + ref);
  for (const key of frame.active) assert.ok(payload.components[key], "Active component has a legend definition: " + key);
}
for (const component of Object.values(payload.components)) {
  assert.ok(Model.COLORS[component.color]);
  component.refs.forEach(ref => assert.ok(payload.references[ref]));
}
const tree = fs.readFileSync(path.join(task, "max_serve_llama2_7b_startup_call_stack.html"), "utf8");
assert.ok(tree.includes('href="link_htmls/max_inference_session_devices_runtime_and_compilation.html"'));
assert.ok(tree.indexOf("Animate InferenceSession: devices") > tree.indexOf("owns runtime/compilation context"));
assert.ok(!/<(?:script|link)[^>]+(?:src|href)="https?:/.test(page), "No external runtime dependencies");
checks.push("Every teaching state has source/legend coverage; generated call-tree link and offline dependencies checked");
const result = {checked_at: new Date().toISOString(), result: "PASS", revision: payload.revision,
  command: "node src/check_inference_session.cjs", evidence: "CPU-RUN JavaScript/document checks; no MAX or GPU execution",
  constructor_cases: constructorCases, checked_frames: allFrames.length, source_targets: Object.keys(payload.references).length, checks};
fs.writeFileSync(path.join(task, "validation/inference_session_model.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result, null, 2));
