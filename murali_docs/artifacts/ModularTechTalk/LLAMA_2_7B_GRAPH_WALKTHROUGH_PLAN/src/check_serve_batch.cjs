"use strict";
const assert = require("assert/strict");
const {BatchConstructor, MODES} = require("./serve_batch_model.js");

function play(mode, options = {}) {
  const m = new BatchConstructor(mode, options), trace = [m.snapshot()];
  for (let i = 0; i < 600 && !m.finished; i++) {
    const previous = m.snapshot(), s = m.next();
    assert.ok(s.pages >= 0 && s.pages <= s.config.pages, "KV ownership fits the retained pool");
    assert.ok(s.batch.length <= s.config.slots, "Selected requests fit batch seats");
    assert.equal(new Set(s.batch.map(op => op.id)).size, s.batch.length, "A request appears once per pass");
    if (s.config.chunking) assert.ok(s.used <= s.config.budget, "Chunked input work respects the active target");
    for (const r of s.requests) {
      const before = previous.requests.find(q => q.id === r.id);
      assert.ok(r.emitted >= before.emitted, "Preemption preserves already emitted token history");
      assert.ok(r.processed <= r.prompt, "Prefill progress is bounded by the current context");
      if (r.phase === "DONE") assert.equal(r.pages, 0, "Finished requests release KV ownership");
      if (r.phase === "HOLD") assert.ok(!s.batch.some(op => op.id === r.id), "Unready work stays out of inputs");
      if (s.phase === "advance" && r.phase === "CE") assert.equal(r.emitted, before.emitted, "Partial prefill emits no token");
    }
    if (s.phase === "execute") assert.deepEqual(s.batch, previous.inputs, "Execution keeps the returned input membership");
    trace.push(s);
  }
  assert.ok(m.finished, mode + " converges");
  if (mode !== "impossible") assert.ok(m.requests.every(r => r.phase === "DONE"), mode + " eventually releases all work");
  return {m, trace};
}

let runs = 0;
for (const mode of Object.keys(MODES)) for (const policy of ["mixed", "prefill", "strict"]) {
  for (const chunking of [true, false]) for (const budget of [8, 12]) {
    play(mode, {policy, chunking, budget}); runs++;
  }
}
const mixed = play("mixed").trace;
const first = mixed.find(s => s.phase === "pack");
assert.deepEqual(first.batch.map(op => [op.id, op.kind, op.tokens]), [["R1","TG",1],["R2","CE",7]]);
assert.ok(first.tg.includes("R1"), "Selected TG remains tracked");
assert.ok(!first.ce.includes("R2"), "Selected CE is popped until feedback");
assert.ok(mixed.some(s => s.phase === "advance" && s.ce[0] === "R2" && s.requests[1].processed === 7 && s.requests[1].emitted === 0), "Partial prefill returns to the CE head");

const prefill = play("mixed", {policy:"prefill"}).trace.find(s => s.phase === "pack");
assert.ok(prefill.batch.every(op => op.kind === "CE"), "Per-replica CE priority builds a CE-only pass");
const strict = play("mixed", {policy:"strict"}).trace.find(s => s.phase === "pack");
assert.deepEqual(strict.batch.map(op => op.kind), ["TG"], "Global TG override suppresses CE backfill");
const whole = play("mixed", {chunking:false}).trace.find(s => s.phase === "pack");
assert.equal(whole.used, 13, "Chunking OFF admits the whole prompt beyond the soft active target");

const seats = play("seats").trace.find(s => s.event.kind === "wait");
assert.equal(seats.checks.seats, "wait");
assert.ok(seats.used < seats.config.budget && seats.pages < seats.config.pages, "Seats can bind despite spare token and KV capacity");
const cache = play("cache").trace;
const cacheWait = cache.find(s => s.event.kind === "wait");
assert.equal(cacheWait.checks.kv, "wait");
assert.ok(!cacheWait.batch.some(op => op.id === "R2"));
const cacheStart = cache.find(s => s.batch.some(op => op.id === "R2"));
assert.equal(cacheStart.requests[0].pages, 0, "Waiting CE runs after the decoder releases cache ownership");

const preempt = play("preempt").trace.find(s => s.event.kind === "preempt");
assert.equal(preempt.requests[1].phase, "CE");
assert.equal(preempt.requests[1].prompt, 5, "Recompute combines the original prompt and existing generated token");
assert.equal(preempt.requests[1].emitted, 1);
assert.equal(preempt.requests[1].pages, 0);
assert.equal(preempt.ce[0], "R2");
const context = play("context").trace.find(s => s.checks.context === "wait");
assert.ok(context.used < context.config.budget);
assert.equal(context.totalUsed, 4);
assert.equal(context.requests[1].pages, 0, "Rejected context-budget candidate releases its cache claim");
assert.ok(context.ce.includes("R2"));

const ready = play("readiness").trace;
assert.ok(!ready.filter(s => s.pass < 2).some(s => s.batch.some(op => op.id === "R2")));
assert.ok(!ready.filter(s => s.pass < 3).some(s => s.batch.some(op => op.id === "R3")));
const fatal = play("impossible").m.snapshot();
assert.equal(fatal.phase, "fatal");
assert.equal(fatal.inputs.length, 0);
assert.match(fatal.event.text, /InsufficientBlocksError/);
const idle = play("idle").m.snapshot();
assert.equal(idle.phase, "idle");
assert.match(idle.event.text, /NO_PROGRESS/);
assert.equal(idle.inputs.length, 0);
console.log(JSON.stringify({result:"PASS",configuration_runs:runs,
  checks:["owned CE/TG state","chunked vs whole active windows","priority/backfill","seat limits",
    "KV waiting/reuse","preemption/token preservation/recompute","context-budget rejection",
    "readiness","immutable pass membership","irrecoverable error","empty-input backoff"]}));
