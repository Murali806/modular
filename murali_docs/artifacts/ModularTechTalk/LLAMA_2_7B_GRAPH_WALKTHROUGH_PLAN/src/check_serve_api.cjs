"use strict";
// Check the API lesson's consequences and boundaries, not production timings.
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {ApiProcess} = require("./serve_api_model.js");
const checks = [];
function run(mode) {
  const model = new ApiProcess(mode), trace = [model.snapshot()];
  while (!model.finished && trace.length < 50) {
    const s = model.next();
    assert.ok(s.queue.length <= 1, "The illustrative slot never overflows");
    assert.equal(new Set(s.queue).size, s.queue.length);
    assert.ok([null, "R1", "R2"].includes(s.lockOwner));
    trace.push(s);
  }
  assert.ok(model.finished, "A finite teaching scene reaches its stated outcome");
  return {model, trace, final:trace.at(-1)};
}
const normal = run("normal");
assert.ok(normal.trace.some(s => s.active === "tokenizer" && s.requests.every(r => r.state === "awaiting CPU tokenization")));
assert.ok(normal.trace.some(s => s.lockOwner === "R1" && s.requests[1].state === "awaiting admission lock"));
assert.ok(normal.trace.some(s => s.lockOwner === "R2" && s.requests[0].state.includes("awaiting model output")), "Admission moves on before R1 produces output");
assert.ok(normal.trace.some(s => s.lockOwner === null && s.outputChunks === 0 && s.queue[0] === "R1"), "The lock does not cover the whole stream");
assert.ok(normal.trace.every(s => !s.queue.includes("R2")), "The rejected request never occupies the IPC slot");
assert.equal(normal.final.requests[1].state, "HTTP 429 returned");
assert.equal(normal.final.outputChunks, 1);
assert.equal(normal.final.requests[0].state, "first chunk delivered · streaming");
assert.deepEqual(normal.final.queue, []);
assert.equal(normal.final.lockOwner, null);
assert.ok(normal.trace.findIndex(s => s.outputChunks === 1) > normal.trace.findIndex(s => s.event.kind === "reject"));
checks.push("Shared-loop tasks await CPU work; admission lock serializes requests and ends before output; R2 gets 429 while R1 later streams");

for (const mode of ["no-server", "no-app", "no-loop"]) {
  const {final, trace} = run(mode);
  assert.equal(final.event.kind, "blocked");
  assert.equal(final.outputChunks, 0);
  assert.deepEqual(final.probes, {R1:null, R2:null});
  assert.ok(trace.every(s => s.queue.length === 0 && s.requests.every(r => r.location === "client")));
  assert.ok(final.requests.every(r => !r.state.includes("429")), "Missing startup roles do not invent HTTP status codes");
}
checks.push("Removing server, app or event loop prevents request progress before tokenization/admission without inventing HTTP errors");

const unlocked = run("no-lock");
const bothSawRoom = unlocked.trace.find(s => s.probes.R1 && s.probes.R2 && s.queue.length === 0);
assert.ok(bothSawRoom, "The possible interleaving exposes two positive probes before any send");
assert.ok(unlocked.trace.some(s => s.queue.length === 1 && s.probes.R2 === true));
assert.ok(unlocked.trace.every(s => s.lockOwner === null && s.event.kind !== "reject"));
assert.equal(unlocked.final.event.kind, "race");
assert.equal(unlocked.final.requests[1].state, "send can wait for room");
assert.deepEqual(unlocked.final.queue, ["R1"]);
assert.equal(unlocked.final.outputChunks, 0);
assert.match(unlocked.model.purpose, /Possible interleaving/);
checks.push("Without the lock, two earlier probes can pass; R2's later send faces backpressure without overflow or a fabricated guaranteed 429");

const replacement = run("asyncio");
assert.deepEqual(replacement.trace.map(({mode, ...s}) => s), normal.trace.map(({mode, ...s}) => s));
assert.match(replacement.model.purpose, /startup\/configuration changes/);
checks.push("A compatible replacement loop preserves the teaching outcomes and explicitly requires startup/configuration changes");

const copy = normal.model.snapshot();
copy.requests[0].state = "tampered"; copy.queue.push("R2"); copy.probes.R1 = false;
assert.deepEqual(normal.model.snapshot(), normal.final, "Read-only snapshots cannot change the lesson model");
assert.deepEqual(normal.model.next(), normal.final, "Finished scenes stay stable until reset/replay");
const fresh = new ApiProcess();
assert.equal(fresh.index, 0); assert.equal(fresh.outputChunks, 0);
assert.deepEqual(fresh.queue, []); assert.deepEqual(fresh.probes, {R1:null, R2:null});
assert.throws(() => new ApiProcess("unknown"), /Unknown API teaching mode/);
checks.push("Snapshots, completed scenes and fresh replays remain isolated and deterministic");

const result = {result:"PASS", checks, evidence:"CPU-RUN deterministic teaching trace; SOURCE control flow and BOUNDARY counterfactuals, no real HTTP/ZMQ race or GPU measurement"};
fs.writeFileSync(path.join(__dirname, "../validation/SERVE_API_MODEL_CHECKS.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result, null, 2));
