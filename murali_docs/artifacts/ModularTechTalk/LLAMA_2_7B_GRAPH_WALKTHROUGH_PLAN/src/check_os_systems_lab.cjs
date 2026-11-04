"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {Simulation, DEFAULT_EXAMPLE, PRESETS, GIB, TERMINAL} = require("./os_systems_lab.js");
const task = path.resolve(__dirname, "..");
const evidence = JSON.parse(fs.readFileSync(path.join(task, "os_systems_manifest.json")));
const checks = [];
function test(name, run) {run(); checks.push(name);}
function invariant(simulation) {
  const s = simulation.snapshot();
  const live = s.requests.filter(r => !TERMINAL.has(r.state));
  assert.equal(live.length, s.unfinished);
  assert.equal(new Set(live.map(r => r.clientId)).size, live.length);
  assert.equal(s.idle + s.unfinished, s.config.clients);
  assert.equal(s.stats.generated, s.stats.delivered + s.counts.buffered + s.stats.discarded);
  assert.equal(s.stats.submitted, s.unfinished + s.stats.completed + s.stats.rejected + s.stats.cancelled + s.stats.failed);
  for (const r of s.requests) {
    assert.ok(r.delivered >= 0 && r.buffered >= 0 && r.discarded >= 0);
    assert.equal(r.generated, r.delivered + r.buffered + r.discarded);
    assert.ok(r.generated <= r.outputLimit && r.promptDone <= r.promptTokens);
    assert.ok(r.assignedPages <= r.reservedPages);
    if (TERMINAL.has(r.state) || r.state === "sending") assert.equal(r.reservedPages + r.assignedPages, 0);
    if (r.firstDeliveredAt !== null) assert.ok(r.firstDeliveredAt >= r.firstProducedAt);
  }
  for (const replica of s.replicas) {
    assert.ok(replica.active.length <= s.config.batchSize);
    assert.ok(replica.reserved <= s.memory.usablePages && replica.assigned <= replica.reserved);
    assert.equal(replica.reusable + replica.reserved, replica.state === "ready" ? s.memory.usablePages : 0);
    if (s.config.pendingLimit) assert.ok(replica.ce <= s.config.pendingLimit);
    if (s.config.transitLimit) assert.ok(replica.transit <= s.config.transitLimit);
    if (s.config.apiLimit) assert.ok(replica.api <= s.config.apiLimit);
    if (replica.state !== "ready") assert.equal(replica.active.length, 0);
  }
  assert.ok(s.history.length <= 240 && s.events.length <= 60);
  return s;
}

test("saved checkpoint geometry drives the simulator", () => {
  assert.deepEqual(DEFAULT_EXAMPLE, evidence.example);
  const s = invariant(new Simulation());
  assert.equal(s.memory.tokenBytes, 512 * 1024);
  assert.equal(s.memory.pageBytes, 64 * 1024 ** 2);
  assert.equal(s.memory.poolBlocks, 129);
  assert.equal(s.memory.allocatedBytes, 8.0625 * GIB);
  assert.equal(s.memory.usablePages, 128);
});
test("opening snapshot separates 1,000 connections, 100 requests and batch 4", () => {
  const s = invariant(new Simulation());
  assert.equal(s.idle, 900);
  assert.equal(s.stats.submitted, 100);
  assert.equal(s.counts.waiting, 96);
  assert.equal(s.counts.active, 4);
  assert.equal(s.replicas[0].ce, 100);
  assert.equal(s.replicas[0].reserved, 20);
  assert.equal(s.replicas[0].assigned, 16);
});
test("a 1 GiB pool admits three five-page requests, not four", () => {
  const sim = new Simulation(PRESETS.memory.config);
  const s = invariant(sim);
  assert.equal(s.memory.poolBlocks, 16);
  assert.equal(s.memory.usablePages, 15);
  assert.equal(s.counts.active, 3);
  assert.equal(s.replicas[0].reserved, 15);
  assert.equal(s.replicas[0].assigned, 12);
  sim.advance(20);
  invariant(sim);
  assert.ok(sim.stats.completed > 3, "released reservations serve later waiting requests");
});
test("full CE M backs up IPC N and rejects before generation", () => {
  const sim = new Simulation(PRESETS.overload.config);
  for (let i = 0; i < 500; i++) {sim.tick(); invariant(sim);}
  assert.ok(sim.stats.rejected429 > 0);
  assert.equal(sim.snapshot().replicas[0].transit, 4);
  const rejects = [...sim.requests.values()].filter(r => r.code === 429);
  assert.ok(rejects.length);
  rejects.forEach(r => {assert.equal(r.generated, 0); assert.equal(r.firstProducedAt, null);});
});
test("tokenizer CPU pressure precedes worker/GPU admission", () => {
  const sim = new Simulation({seedRequests: 0, tokenizerRate: 100, arrivalRate: 30,
    promptTokens: 3072, tokenizerThreads: 1});
  sim.advance(1);
  const s = invariant(sim);
  assert.ok(s.counts.api > 10);
  assert.equal(s.counts.tokenizing, 1);
  assert.equal(s.counts.active, 0);
});
test("slow sending requests no longer own KV pages", () => {
  const sim = new Simulation({arrivalRate: 0, seedRequests: 4, slowPercent: 100});
  sim.advance(3);
  const s = invariant(sim);
  assert.equal(s.counts.active, 0);
  assert.equal(s.counts.sending, 4);
  assert.ok(s.counts.buffered > 200);
  assert.equal(s.replicas[0].reserved, 0);
  assert.equal(s.replicas[0].allocatedBytes, 8.0625 * GIB);
  assert.equal(s.replicas[0].reusable, 128);
  sim.advance(64);
  assert.equal(invariant(sim).stats.completed, 4);
});
test("in-flight cancellation holds pages until the pass boundary", () => {
  const sim = new Simulation({arrivalRate: 0, seedRequests: 1});
  const r = sim.requests.get("R1");
  const pages = r.reservedPages;
  assert.ok(sim.cancel("R1"));
  assert.equal(r.state, "cancel_pending");
  assert.equal(r.reservedPages, pages);
  sim.tick();
  assert.equal(r.reservedPages, pages);
  sim.advance(1);
  invariant(sim);
  assert.equal(r.state, "cancelled");
  assert.equal(r.reservedPages, 0);
  assert.equal(r.generated, 0);
});
test("cancelling a sending request discards only undelivered tokens", () => {
  const sim = new Simulation({arrivalRate: 0, seedRequests: 1, slowPercent: 100});
  sim.advance(2);
  const r = sim.requests.get("R1");
  assert.equal(r.state, "sending");
  const delivered = r.delivered, buffered = r.buffered;
  assert.ok(buffered > 0 && delivered > 0);
  sim.cancel("R1");
  invariant(sim);
  assert.equal(r.delivered, delivered);
  assert.equal(r.discarded, buffered);
  assert.equal(r.buffered, 0);
});
test("effective output limit is capped to remaining context", () => {
  const sim = new Simulation({seedRequests: 0, arrivalRate: 0, batchSize: 1});
  const r = sim.submit({promptTokens: 4080, outputTokens: 64});
  assert.equal(r.requestedOutput, 64);
  assert.equal(r.outputLimit, 16);
  assert.equal(r.neededPages, 32);
  sim.advance(3);
  invariant(sim);
  assert.equal(r.generated, 16);
  assert.equal(r.state, "completed");
});
test("warm replicas receive new traffic only after readiness; old owners stay", () => {
  const sim = new Simulation({arrivalRate: 0, seedRequests: 4, outputTokens: 256, routing: "least_outstanding"});
  const oldOwners = [...sim.requests.values()].map(r => [r.id, r.replicaId]);
  const added = sim.addReplica();
  assert.equal(added.state, "warming");
  assert.equal(sim.selectReplica().label, "A");
  sim.advance(1.99);
  assert.equal(added.state, "warming");
  sim.tick();
  assert.equal(added.state, "ready");
  const r = sim.submit({promptTokens: 1, outputTokens: 2});
  assert.equal(r.replicaId, added.id, "new traffic chooses the newly ready, empty replica");
  oldOwners.forEach(([id, owner]) => assert.equal(sim.requests.get(id).replicaId, owner));
  invariant(sim);
});
test("worker crash fails streams, excludes routing and replaces fresh state", () => {
  const sim = new Simulation({replicas: 2, arrivalRate: 0, seedRequests: 8});
  const first = sim.replicas[0];
  const failedIds = [...sim.requests.values()].filter(r => r.replicaId === first.id).map(r => r.id);
  sim.crashReplica(first.id);
  failedIds.forEach(id => assert.equal(sim.requests.get(id).state, "failed"));
  assert.equal(sim.selectReplica().label, "B");
  assert.equal(invariant(sim).stats.failed, failedIds.length);
  sim.replaceReplica(first.id);
  assert.equal(first.state, "warming");
  sim.advance(2);
  assert.equal(first.state, "ready");
  failedIds.forEach(id => assert.equal(sim.requests.get(id).state, "failed"));
  invariant(sim);
});
test("estimated-work routing distinguishes equal counts with unequal tokens", () => {
  const sim = new Simulation({replicas: 2, arrivalRate: 0, seedRequests: 0, routing: "token_work"});
  sim.submit({replicaId: sim.replicas[0].id, promptTokens: 3072, outputTokens: 384});
  sim.submit({replicaId: sim.replicas[1].id, promptTokens: 256, outputTokens: 32});
  assert.equal(sim.outstanding(sim.replicas[0]), 1);
  assert.equal(sim.outstanding(sim.replicas[1]), 1);
  assert.ok(sim.estimatedWork(sim.replicas[0]) > sim.estimatedWork(sim.replicas[1]) * 10);
  assert.equal(sim.selectReplica().label, "B");
  invariant(sim);
});
test("power-of-two routing and mixed workloads replay deterministically", () => {
  const config = {...PRESETS.skew.config, routing: "power_two", slowPercent: 40};
  const a = new Simulation(config), b = new Simulation(config);
  a.advance(8); b.advance(8);
  assert.deepEqual(a.snapshot(), b.snapshot());
  invariant(a);
  assert.ok([...a.requests.values()].some(r => r.promptTokens === 3072));
  assert.ok([...a.requests.values()].some(r => r.promptTokens === 256));
});
test("client population caps unfinished streams without inventing requests", () => {
  const sim = new Simulation({clients: 2, seedRequests: 0, arrivalRate: 50,
    slowPercent: 100, readerRate: 1, slowReaderRate: 1});
  sim.advance(2);
  const s = invariant(sim);
  assert.equal(s.unfinished, 2);
  assert.ok(s.stats.busyClients > 0);
  assert.equal(s.stats.submitted, 2);
});
test("invalid memory cannot over-assign pages; long histories stay bounded", () => {
  const impossible = new Simulation({gpuGiB: 13, utilization: 0.5, seedRequests: 0, arrivalRate: 0});
  assert.equal(impossible.memory.usablePages, 0);
  const r = impossible.submit();
  assert.equal(r.state, "rejected");
  assert.equal(r.code, null, "memory rejection is lab policy, not a forged MAX HTTP 429");
  invariant(impossible);
  const sim = new Simulation({seedRequests: 0, arrivalRate: 40, outputTokens: 1,
    promptTokens: 32, readerRate: 10000, tokenizerRate: 200000});
  sim.advance(30);
  const s = invariant(sim);
  assert.equal(s.history.length, 240);
  assert.ok(s.requests.filter(r => TERMINAL.has(r.state)).length <= 256);
  assert.ok(s.stats.completed > 300);
});

const result = {checked_at: new Date().toISOString(), result: "PASS", tests: checks.length,
  checks, command: "node src/check_os_systems_lab.cjs", evidence: "CPU-RUN: deterministic simulator checks; no model/device execution"};
fs.writeFileSync(path.join(task, "validation/os_systems_simulation.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result, null, 2));
