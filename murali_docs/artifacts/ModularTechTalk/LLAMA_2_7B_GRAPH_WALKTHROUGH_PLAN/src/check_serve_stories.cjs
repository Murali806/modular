"use strict";
// Behavioral checks for the lesson's causal claims, not GPU performance tests.
const assert = require("assert/strict");
const {Worker, Fleet, TERMINAL} = require("./serve_story_model.js");
const checks = [];
function until(worker, predicate, limit = 2000) {
  for (let i = 0; i < limit; i++) {
    if (predicate()) return;
    worker.next();
    assert.ok(worker.usedPages <= worker.config.pages, "KV ownership stays within the toy pool");
    assert.ok(worker.active.length <= worker.config.batch, "batch membership stays bounded");
    assert.ok(worker.at("ipc").length <= worker.config.queue, "transport capacity stays bounded");
    assert.ok(worker.ceCount <= worker.config.pending, "prefill membership stays within M");
    for (const r of worker.requests) {
      assert.ok(r.delivered <= r.generated && r.generated <= r.output);
      if (!r.cancellation) assert.equal(r.generated, r.delivered + r.buffered + r.unpublished);
    }
  }
  throw new Error("Scenario failed to reach its expected outcome");
}
function busy(config = {}) {
  const worker = new Worker(config);
  worker.seedDecoder({output:9});
  until(worker, () => worker.busy);
  return worker;
}
{
  const w = busy({batch:2});
  const oldMembers = w.batch.map(op => op.id);
  const r = w.submit();
  until(w, () => r.state === "ipc");
  assert.ok(w.busy, "admission continues during the selected model pass");
  assert.deepEqual(w.batch.map(op => op.id), oldMembers, "the selected pass cannot gain a late arrival");
  until(w, () => r.selected !== null);
  assert.ok(w.requests[0].generated < w.requests[0].output, "R2 joins before R1 finishes");
  until(w, () => w.live.length === 0);
  assert.equal(w.usedPages,0);
  checks.push("Admission during execution; late request joins a later pass; final KV release");
}
{
  const w = busy({batch:1}); const r = w.submit();
  until(w, () => r.state === "pending");
  assert.equal(r.accepted !== null,true);
  assert.equal(r.selected,null);
  until(w, () => r.selected !== null);
  assert.equal(w.requests[0].generated,w.requests[0].output);
  checks.push("One batch slot makes admitted R2 wait for R1's generation to finish");
}
{
  const outcomes = [];
  for (const queue of [2,6]) {
    const w = busy({batch:1, pending:2, queue});
    for (let i=0;i<6;i++) w.submit({admissionDelay:1+i*4});
    let rejectedAtFullM = false;
    until(w, () => {
      if (w.last.kind === "reject") {
        assert.equal(w.ceCount,2, "this scenario fills M before rejecting at N");
        assert.equal(w.at("ipc").length,queue);
        const r = w.find(w.last.id);
        assert.equal(r.accepted,null); assert.equal(r.responseStarted,false); assert.equal(r.pages,0);
        rejectedAtFullM = true;
      }
      return w.live.length === 0;
    });
    outcomes.push(w.snapshot());
    if (queue === 2) assert.ok(rejectedAtFullM);
  }
  assert.ok(outcomes[0].rejected > 0);
  assert.equal(outcomes[1].rejected,0);
  assert.equal(outcomes[1].accepted,7);
  assert.equal(outcomes[1].done,7);
  checks.push("M → N → 429 before streaming; extra queue room changes acceptance; admitted work completes");
}
{
  const w = new Worker({budget:8, pending:4});
  w.seedDecoder({output:7}); const long = w.submit({prompt:24,output:3});
  let partial = false, mixed = false;
  until(w, () => {
    for (const h of w.history) {
      assert.ok(h.ops.reduce((n,op)=>n+op.tokens,0) <= 8);
      mixed ||= h.ops.some(op=>op.kind==="prefill") && h.ops.some(op=>op.kind==="decode");
    }
    if (long.promptDone > 0 && long.promptDone < long.prompt) { partial = true; assert.equal(long.generated,0); }
    return w.live.length === 0;
  });
  assert.ok(partial && mixed);
  assert.equal(long.delivered,3);
  checks.push("Chunked prefill consumes the shared token budget; mixed passes preserve per-request output counts");
}
{
  const w = new Worker({readEvery:12}); const r = w.submit();
  until(w, () => r.state === "streaming" && r.buffered > 0);
  assert.equal(r.pages,0); assert.equal(r.generated,r.output); assert.ok(r.delivered<r.generated);
  until(w, () => r.state === "done");
  assert.equal(r.delivered,5);
  until(w, () => w.phase === "idle");
  assert.match(w.last.text,/backoff/);
  checks.push("Slow reader buffers output after KV release; publication precedes delivery; idle worker yields");
}
{
  const w = busy(); const r = w.requests[0]; const pages = r.pages;
  assert.ok(w.cancel(r.id)); assert.equal(r.pages,pages);
  until(w, () => r.state === "cancelled");
  assert.equal(r.pages,0);
  const after = busy();
  until(after, () => after.phase === "advance");
  assert.ok(after.cancel("R1"));
  assert.equal(after.find("R1").state,"cancelled");
  assert.equal(after.find("R1").unpublished,0);
  checks.push("Cancellation preserves selected-pass ownership through safe cleanup");
}
{
  const w = busy({pages:3}); const r = w.submit({prompt:8,output:9});
  until(w, () => r.state === "pending");
  assert.equal(r.accepted !== null,true);
  until(w, () => /KV pages/.test(r.reason));
  assert.equal(w.at("rejected").length,0,"KV wait is not an API queue rejection");
  until(w, () => r.state === "done");
  checks.push("Cache pressure delays already-admitted work without producing HTTP 429");
}
{
  const f = new Fleet(); const first = f.submit(); f.addReplica();
  const during = f.submit();
  assert.ok(f.replicas[0].worker.find(first.id));
  assert.ok(f.replicas[0].worker.find(during.id));
  while(f.time < f.replicas[1].readyAt) f.next();
  for(let i=0;i<4;i++) f.submit();
  assert.ok(f.replicas[1].worker.requests.length > 0);
  assert.ok(f.replicas[0].worker.find(first.id),"existing work never migrates");
  assert.equal(f.replicas[1].worker.find(first.id),undefined);
  assert.equal(new Set(f.replicas.flatMap(r=>r.worker.requests.map(q=>q.id))).size,6);
  checks.push("External routing excludes warming replicas and preserves existing stream ownership");
}
console.log(JSON.stringify({result:"PASS", evidence:"CPU-RUN discrete-event teaching model; no MAX inference", checks},null,2));
