"use strict";
// Verify the close-up's selected starting states and teaching consequences.
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {WorkerProcess} = require("./serve_worker_model.js");
const checks = [];
function run(mode) {
  const model = new WorkerProcess(mode), trace = [model.snapshot()];
  while (!model.finished && trace.length < 150) trace.push(model.next());
  assert.ok(model.finished, "The scene reaches its stated outcome");
  for (const s of trace) {
    assert.ok(s.pages <= s.capacity.pages);
    assert.ok(s.batch.length <= s.capacity.batch);
    assert.ok(s.batch.reduce((n, op) => n+op.tokens, 0) <= s.capacity.budget);
    assert.equal(s.rejected, 0, "These scenes start after admission and do not reject cache waits");
    assert.ok(s.requests.every(r => r.delivered <= r.generated));
    assert.ok(s.requests.every(r => r.promptDone === r.prompt || r.generated === 0), "Partial prefill has no output token");
  }
  return {model, trace, final:trace.at(-1)};
}
const normal = run("normal");
assert.equal(normal.trace[0].ipc, 2);
assert.equal(normal.trace[1].pending, 2);
assert.equal(normal.trace[1].ipc, 0);
assert.ok(normal.trace.some(s => s.requests[0].promptDone === 8 && s.generated === 0));
assert.ok(normal.trace.some(s => s.batch.length === 2));
assert.ok(normal.trace.some(s => s.results.length === 2), "A result map can carry both request IDs");
assert.ok(normal.trace.every(s => s.results.every(result => s.requests.find(r => r.id === result.id).generated >= result.token)));
assert.equal(normal.final.done, 2);
assert.equal(normal.final.generated, 5);
assert.equal(normal.final.pages, 0);
checks.push("Two admitted contexts drain, partial prefill waits, mixed batch membership changes, results preserve IDs and finished requests release KV ownership");

const busy = run("busy");
assert.equal(busy.trace[0].phase, "execute");
assert.equal(busy.trace[0].requests[1].state, "ipc");
assert.deepEqual(busy.trace[0].batch.map(op => op.id), ["R1"]);
const firstR2 = busy.trace.find(s => s.batch.some(op => op.id === "R2"));
assert.ok(firstR2.pass > busy.trace[0].pass, "A late request cannot join the running pass");
assert.ok(firstR2.requests[0].generated < firstR2.requests[0].output, "R2 can join a later pass while R1 is still generating");
assert.equal(busy.final.done, 2);
checks.push("A busy pass keeps its membership; R2 waits in IPC, then joins a later pass before R1 finishes");

const cache = run("cache");
assert.equal(cache.trace[0].pending, 1);
assert.equal(cache.trace[0].pages, 2);
assert.equal(cache.trace[0].batch.length, 1, "KV pressure leaves one batch slot unused");
assert.equal(cache.trace[0].requests[1].pages, 0);
const cacheR2 = cache.trace.find(s => s.requests[1].selected !== null);
assert.equal(cacheR2.requests[0].pages, 0);
assert.equal(cacheR2.requests[0].generated, cacheR2.requests[0].output);
assert.equal(cache.final.pages, 0);
assert.equal(cache.final.capacity.pages, 2, "Releasing ownership retains the backing pool");
checks.push("KV pressure defers admitted R2 despite a free batch slot; R1's release enables progress without an HTTP 429");

const idle = run("idle");
assert.deepEqual(idle.trace.map(s => s.phase), ["ready", "drain", "select", "idle"]);
assert.ok(idle.trace.every(s => s.generated === 0 && s.pass === 0));
assert.match(idle.final.event.text, /NO_PROGRESS.*backoff/);
checks.push("An empty worker checks input, finds no eligible batch and yields with backoff without launching model work");

const snapshot = normal.model.snapshot();
snapshot.requests[0].generated = 999; snapshot.results.push({id:"fake"});
assert.deepEqual(normal.model.snapshot(), normal.final);
assert.deepEqual(normal.model.next(), normal.final);
assert.deepEqual(new WorkerProcess().snapshot(), normal.trace[0]);
assert.throws(() => new WorkerProcess("invalid"), /Unknown worker teaching mode/);
checks.push("Completed scenes, snapshots and reset are isolated and deterministic");

const result = {result:"PASS", checks, evidence:"CPU-RUN teaching model; reuses ServeStory.Worker, no model weights, GPU timings or native execution"};
fs.writeFileSync(path.join(__dirname, "../validation/SERVE_WORKER_MODEL_CHECKS.json"), JSON.stringify(result, null, 2)+"\n");
console.log(JSON.stringify(result, null, 2));
