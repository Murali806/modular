/* Pure deterministic teaching model. No network, devices or wall-clock timers. */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.OSSystemsLab = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const GIB = 1024 ** 3;
  const TICK_MS = 10;
  const TERMINAL = new Set(["completed", "rejected", "failed", "cancelled"]);
  const ROUTES = ["round_robin", "least_outstanding", "token_work", "power_two"];
  const DEFAULT_EXAMPLE = Object.freeze({weightBytes: 13476876272, layers: 32,
    kvHeads: 32, headDim: 128, cacheBytes: 2, pageSize: 128, maxContext: 4096,
    tokenBytes: 524288, pageBytes: 67108864});
  const DEFAULTS = Object.freeze({clients: 1000, arrivalRate: 6, promptTokens: 512,
    outputTokens: 64, shape: "uniform", slowPercent: 0, replicas: 1,
    routing: "round_robin", transitLimit: 32, pendingLimit: 128, apiLimit: 256,
    batchSize: 4, inputBudget: 1024, gpuGiB: 24, utilization: 0.9, poolCapGiB: 0,
    tokenizerRate: 20000, tokenizerThreads: 2, prefillRate: 12000, decodeRate: 80,
    launchMs: 2, readerRate: 120, slowReaderRate: 1, seedRequests: 100, seed: 903});
  const PRESETS = Object.freeze({
    connections: {title: "1,000 connections", config: {}},
    overload: {title: "N fills behind M", config: {arrivalRate: 80, pendingLimit: 8,
      transitLimit: 4, seedRequests: 8, outputTokens: 256}},
    memory: {title: "1 GiB KV pool", config: {poolCapGiB: 1, arrivalRate: 3}},
    slow: {title: "Slow readers", config: {clients: 200, arrivalRate: 4,
      slowPercent: 100, seedRequests: 12, outputTokens: 64}},
    skew: {title: "Unequal request costs", config: {replicas: 3, shape: "mixed",
      seedRequests: 0, arrivalRate: 12, promptTokens: 256, outputTokens: 32,
      routing: "round_robin"}},
  });
  const limits = {
    clients: [1, 10000, true], arrivalRate: [0, 1000], promptTokens: [1, 4095, true],
    outputTokens: [1, 4096, true], slowPercent: [0, 100], replicas: [1, 8, true],
    transitLimit: [0, 1024, true], pendingLimit: [0, 2048, true], apiLimit: [0, 2048, true],
    batchSize: [1, 32, true], inputBudget: [32, 8192, true], gpuGiB: [13, 96],
    utilization: [0.5, 1], poolCapGiB: [0, 64], tokenizerRate: [100, 200000],
    tokenizerThreads: [1, 16, true], prefillRate: [100, 200000], decodeRate: [1, 1000],
    launchMs: [0, 100], readerRate: [1, 10000], slowReaderRate: [0.1, 100],
    seedRequests: [0, 500, true], seed: [0, 0xffffffff, true],
  };
  const cap = n => n === 0 ? Infinity : n;
  const pagesFor = (tokens, example) => Math.ceil(tokens / example.pageSize);
  function normalize(input) {
    const config = {...DEFAULTS, ...input};
    for (const [key, [low, high, integer]] of Object.entries(limits)) {
      const value = config[key];
      if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high || (integer && !Number.isInteger(value))) {
        throw new RangeError(key + " must be " + (integer ? "an integer" : "a number") + " in [" + low + ", " + high + "]");
      }
    }
    if (!ROUTES.includes(config.routing)) throw new RangeError("Unknown routing policy");
    if (!["uniform", "mixed"].includes(config.shape)) throw new RangeError("Unknown request shape");
    return Object.freeze(config);
  }
  function memoryPlan(config, example = DEFAULT_EXAMPLE) {
    const tokenBytes = 2 * example.layers * example.kvHeads * example.headDim * example.cacheBytes;
    const pageBytes = tokenBytes * example.pageSize;
    const budgetBytes = Math.max(0, Math.floor(config.gpuGiB * GIB * config.utilization) - example.weightBytes);
    const fullContextBlocks = config.batchSize * pagesFor(example.maxContext, example) + 1;
    const capBytes = config.poolCapGiB > 0 ? config.poolCapGiB * GIB : Infinity;
    const poolBlocks = Math.min(fullContextBlocks, Math.floor(Math.min(budgetBytes, capBytes) / pageBytes));
    return Object.freeze({weightBytes: example.weightBytes, tokenBytes, pageBytes,
      budgetBytes, fullContextBlocks, poolBlocks, allocatedBytes: poolBlocks * pageBytes,
      usablePages: Math.max(0, poolBlocks - 1), nullBlocks: poolBlocks > 0 ? 1 : 0});
  }
  function randomGenerator(seed) {
    let value = seed >>> 0;
    return () => {
      value = (value + 0x6D2B79F5) | 0;
      let t = Math.imul(value ^ (value >>> 15), 1 | value);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function percentile(values, p) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)];
  }
  class Simulation {
    constructor(config = {}, example = DEFAULT_EXAMPLE) {
      this.config = normalize(config);
      this.example = Object.freeze({...example});
      this.memory = memoryPlan(this.config, this.example);
      this.rng = randomGenerator(this.config.seed);
      this.ticks = 0;
      this.now = 0;
      this.arrivalCredit = 0;
      this.requestSequence = 0;
      this.replicaSequence = 0;
      this.routeCursor = 0;
      this.clientCursor = 0;
      this.requests = new Map();
      this.clientOwners = new Map();
      this.replicas = [];
      this.events = [];
      this.history = [];
      this.ttftSamples = [];
      this.stats = {demand: 0, submitted: 0, admitted: 0, completed: 0,
        rejected: 0, rejected429: 0, failed: 0, cancelled: 0, busyClients: 0,
        generated: 0, delivered: 0, discarded: 0, seeded: 0};
      for (let i = 0; i < this.config.replicas; i++) this._newReplica(0);
      this._seedSnapshot();
      this._recordHistory();
    }
    _event(message, request = null) {
      const record = {at: this.now, message, request: request?.id || null};
      this.events.push(record);
      if (this.events.length > 60) this.events.shift();
      if (request) {
        request.history.push(record);
        if (request.history.length > 24) request.history.shift();
      }
    }
    _newReplica(warmMs) {
      const id = "replica-" + (++this.replicaSequence);
      const replica = {id, label: String.fromCharCode(64 + this.replicaSequence),
        state: warmMs ? "warming" : "ready", readyAt: this.now + warmMs / 1000,
        apiQueue: [], cpu: [], transit: [], waiting: [], active: [], flight: null,
        busySeconds: 0, prefillCursor: 0, batches: 0, completed: 0, failures: 0};
      this.replicas.push(replica);
      this._event("Replica " + replica.label + (warmMs ? " warming; excluded from routing" : " ready"));
      return replica;
    }
    _ceCount(replica) {
      return replica.waiting.length + replica.active.filter(id => {
        const r = this.requests.get(id);
        return r && r.promptDone < r.promptTokens;
      }).length;
    }
    outstanding(replica) {
      let count = 0;
      for (const request of this.requests.values()) {
        if (request.replicaId === replica.id && !TERMINAL.has(request.state)) count++;
      }
      return count;
    }
    estimatedWork(replica) {
      let seconds = 0;
      for (const r of this.requests.values()) {
        if (r.replicaId !== replica.id || TERMINAL.has(r.state)) continue;
        seconds += (r.promptTokens - r.promptDone) / this.config.prefillRate;
        seconds += Math.max(0, r.outputLimit - r.generated) / this.config.decodeRate;
      }
      return seconds;
    }
    selectReplica() {
      const ready = this.replicas.filter(r => r.state === "ready");
      if (!ready.length) return null;
      if (this.config.routing === "round_robin") return ready[(this.routeCursor++) % ready.length];
      if (this.config.routing === "power_two") {
        if (ready.length === 1) return ready[0];
        const first = Math.floor(this.rng() * ready.length);
        let second = Math.floor(this.rng() * (ready.length - 1));
        if (second >= first) second++;
        return this.outstanding(ready[first]) <= this.outstanding(ready[second]) ? ready[first] : ready[second];
      }
      const cost = this.config.routing === "token_work" ? r => this.estimatedWork(r) : r => this.outstanding(r);
      return ready.reduce((best, next) => cost(next) < cost(best) ? next : best);
    }
    _freeClient() {
      if (this.clientOwners.size >= this.config.clients) return null;
      for (let i = 0; i < this.config.clients; i++) {
        const id = this.clientCursor;
        this.clientCursor = (this.clientCursor + 1) % this.config.clients;
        if (!this.clientOwners.has(id)) return id;
      }
      return null;
    }
    _shape() {
      if (this.config.shape === "mixed" && (this.requestSequence + 1) % 5 === 0) {
        return {promptTokens: 3072, outputTokens: 384};
      }
      return {promptTokens: this.config.promptTokens, outputTokens: this.config.outputTokens};
    }
    _createRequest(replica, clientId, overrides = {}) {
      const shape = {...this._shape(), ...overrides};
      const promptTokens = Math.max(1, Math.min(this.example.maxContext - 1, Math.floor(shape.promptTokens)));
      const requestedOutput = Math.max(1, Math.floor(shape.outputTokens));
      const outputLimit = Math.min(requestedOutput, this.example.maxContext - promptTokens);
      const slow = shape.slow === undefined ? this.rng() * 100 < this.config.slowPercent : Boolean(shape.slow);
      const request = {id: "R" + (++this.requestSequence), number: this.requestSequence,
        replicaId: replica?.id || null, clientId, state: "api_wait", promptTokens,
        requestedOutput, outputLimit, promptDone: 0, generated: 0, delivered: 0,
        buffered: 0, discarded: 0, readerCredit: 0, slow,
        readerRate: slow ? this.config.slowReaderRate : this.config.readerRate,
        neededPages: pagesFor(promptTokens + outputLimit, this.example), reservedPages: 0,
        assignedPages: 0, createdAt: this.now, tokenStartAt: null, admittedAt: null,
        selectedAt: null, prefillDoneAt: null, firstProducedAt: null,
        firstDeliveredAt: null, generationDoneAt: null, doneAt: null,
        code: null, reason: "", history: []};
      this.requests.set(request.id, request);
      this.clientOwners.set(clientId, request.id);
      this.stats.submitted++;
      return request;
    }
    _mark(request, state, reason) {
      request.state = state;
      this._event(request.id + " -> " + state + ": " + reason, request);
    }
    _seedSnapshot() {
      for (let i = 0; i < this.config.seedRequests; i++) {
        const clientId = this._freeClient();
        if (clientId === null) break;
        const available = this.replicas.filter(r => this._ceCount(r) < cap(this.config.pendingLimit));
        if (!available.length) break;
        const replica = available[i % available.length];
        const request = this._createRequest(replica, clientId);
        request.admittedAt = this.now;
        this._mark(request, "waiting", "opening snapshot: tokenization and IPC already complete");
        replica.waiting.push(request.id);
        this.stats.admitted++;
        this.stats.seeded++;
      }
      for (const replica of this.replicas) this._prepareFlight(replica);
      this._event("Snapshot seeds " + this.stats.seeded + " already-tokenized contexts");
    }
    submit(overrides = {}) {
      this.stats.demand++;
      const clientId = this._freeClient();
      if (clientId === null) {
        this.stats.busyClients++;
        return null;
      }
      const replica = overrides.replicaId ? this.replicas.find(r => r.id === overrides.replicaId && r.state === "ready") : this.selectReplica();
      const request = this._createRequest(replica, clientId, overrides);
      if (!replica) {
        this._terminal(request, "rejected", "lab frontend: no ready replica", 503);
      } else if (replica.apiQueue.length >= cap(this.config.apiLimit)) {
        this._terminal(request, "rejected", "lab external API waiting gate full", 503);
      } else if (request.neededPages > this.memory.usablePages) {
        this._terminal(request, "rejected", "lab conservative reservation exceeds usable KV pool", null);
      } else {
        replica.apiQueue.push(request.id);
        this._event(request.id + " -> API wait on " + replica.label, request);
      }
      return request;
    }
    _removeFromQueues(request) {
      const replica = this.replicas.find(r => r.id === request.replicaId);
      if (!replica) return;
      for (const key of ["apiQueue", "transit", "waiting", "active"]) {
        replica[key] = replica[key].filter(id => id !== request.id);
      }
      replica.cpu = replica.cpu.filter(job => job.id !== request.id);
    }
    _release(request) {
      const replica = this.replicas.find(r => r.id === request.replicaId);
      if (replica) replica.active = replica.active.filter(id => id !== request.id);
      request.reservedPages = 0;
      request.assignedPages = 0;
    }
    _terminal(request, state, reason, code = null) {
      if (TERMINAL.has(request.state)) return;
      if (request.buffered) {
        request.discarded += request.buffered;
        this.stats.discarded += request.buffered;
        request.buffered = 0;
      }
      this._removeFromQueues(request);
      this._release(request);
      request.code = code;
      request.reason = reason;
      request.doneAt = this.now;
      this.clientOwners.delete(request.clientId);
      this.stats[state]++;
      if (state === "rejected" && code === 429) this.stats.rejected429++;
      const replica = this.replicas.find(r => r.id === request.replicaId);
      if (replica && state === "completed") replica.completed++;
      if (replica && state === "failed") replica.failures++;
      this._mark(request, state, reason);
    }
    cancel(id) {
      const request = this.requests.get(id);
      if (!request || TERMINAL.has(request.state) || request.state === "cancel_pending") return false;
      const replica = this.replicas.find(r => r.id === request.replicaId);
      const inFlight = replica?.flight?.ops.some(op => op.id === id);
      if (inFlight) this._mark(request, "cancel_pending", "cancel received; retain pages through current pass");
      else this._terminal(request, "cancelled", "cancelled at safe scheduling boundary");
      return true;
    }
    addReplica(warmMs = 2000) {
      if (this.replicas.length >= 12) return null;
      return this._newReplica(warmMs);
    }
    crashReplica(id) {
      const replica = this.replicas.find(r => r.id === id);
      if (!replica || replica.state === "down") return false;
      replica.state = "down";
      replica.flight = null;
      for (const request of this.requests.values()) {
        if (request.replicaId === id && !TERMINAL.has(request.state)) {
          this._terminal(request, "failed", "worker exit tears down replica; stream not replayed");
        }
      }
      this._event("Replica " + replica.label + " down: worker/API serving context lost");
      return true;
    }
    replaceReplica(id, warmMs = 2000) {
      const replica = this.replicas.find(r => r.id === id);
      if (!replica || replica.state !== "down") return false;
      replica.state = warmMs ? "warming" : "ready";
      replica.readyAt = this.now + warmMs / 1000;
      replica.busySeconds = 0;
      this._event("External replacement for " + replica.label + ": new serving state; old streams stay failed");
      return true;
    }
    _runTokenizer(replica) {
      const elapsedMs = TICK_MS;
      for (const job of [...replica.cpu]) {
        job.remainingMs -= elapsedMs;
        if (job.remainingMs > 0) continue;
        replica.cpu = replica.cpu.filter(other => other !== job);
        const request = this.requests.get(job.id);
        if (replica.transit.length >= cap(this.config.transitLimit)) {
          this._terminal(request, "rejected", "connected IPC request channel full: MAX-style admission gate", 429);
        } else {
          request.admittedAt = this.now;
          replica.transit.push(request.id);
          this.stats.admitted++;
          this._mark(request, "transit", "tokenized context admitted to exact lab N-slot IPC queue");
        }
      }
      while (replica.cpu.length < this.config.tokenizerThreads && replica.apiQueue.length) {
        const request = this.requests.get(replica.apiQueue.shift());
        request.tokenStartAt = this.now;
        this._mark(request, "tokenizing", "default-executor CPU work (linear lab rate)");
        replica.cpu.push({id: request.id, remainingMs: Math.max(TICK_MS, request.promptTokens / this.config.tokenizerRate * 1000)});
      }
    }
    _drainIPC(replica) {
      while (replica.transit.length && this._ceCount(replica) < cap(this.config.pendingLimit)) {
        const request = this.requests.get(replica.transit.shift());
        replica.waiting.push(request.id);
        this._mark(request, "waiting", "worker drains context within pending CE cap M");
      }
    }
    _reserved(replica) {
      return replica.active.reduce((sum, id) => sum + this.requests.get(id).reservedPages, 0);
    }
    _fillActive(replica) {
      while (replica.active.length < this.config.batchSize && replica.waiting.length) {
        const request = this.requests.get(replica.waiting[0]);
        if (request.neededPages > this.memory.usablePages) {
          this._terminal(request, "rejected", "lab conservative reservation exceeds usable KV pool");
          continue;
        }
        if (this._reserved(replica) + request.neededPages > this.memory.usablePages) break;
        replica.waiting.shift();
        request.reservedPages = request.neededPages;
        request.assignedPages = pagesFor(request.promptTokens, this.example);
        request.selectedAt = this.now;
        replica.active.push(request.id);
        this._mark(request, "prefill", "batch slot + conservative KV reservation claimed");
      }
    }
    _prepareFlight(replica) {
      if (replica.state !== "ready" || replica.flight) return;
      this._fillActive(replica);
      const decoders = replica.active.map(id => this.requests.get(id)).filter(r => r.state === "decode");
      const prefills = replica.active.map(id => this.requests.get(id)).filter(r => r.state === "prefill");
      const ops = decoders.map(r => ({id: r.id, type: "decode", tokens: 1}));
      let budget = Math.max(0, this.config.inputBudget - ops.length);
      const quota = prefills.length ? Math.max(1, Math.floor(budget / prefills.length)) : 0;
      if (prefills.length) {
        const start = replica.prefillCursor % prefills.length;
        for (let i = 0; i < prefills.length && budget > 0; i++) {
          const request = prefills[(start + i) % prefills.length];
          const tokens = Math.min(budget, quota, request.promptTokens - request.promptDone);
          if (tokens > 0) ops.push({id: request.id, type: "prefill", tokens});
          budget -= tokens;
        }
        replica.prefillCursor++;
      }
      if (!ops.length) return;
      const promptWork = ops.filter(op => op.type === "prefill").reduce((sum, op) => sum + op.tokens, 0);
      const modeledMs = this.config.launchMs + promptWork / this.config.prefillRate * 1000 + (decoders.length ? 1000 / this.config.decodeRate : 0);
      const durationMs = Math.max(TICK_MS, modeledMs);
      replica.flight = {batchId: ++replica.batches, ops, durationMs, remainingMs: durationMs,
        startedAt: this.now, promptTokens: promptWork, decoders: decoders.length};
    }
    _emitToken(request) {
      if (request.generated >= request.outputLimit) return;
      request.generated++;
      request.buffered++;
      this.stats.generated++;
      request.assignedPages = pagesFor(request.promptTokens + request.generated, this.example);
      if (request.firstProducedAt === null) {
        request.firstProducedAt = this.now;
        this._event(request.id + " first token produced; not yet client delivery", request);
      }
      if (request.generated === request.outputLimit) {
        request.generationDoneAt = this.now;
        this._release(request);
        this._mark(request, "sending", "generation complete; KV ownership returned; output still drains");
      }
    }
    _finishFlight(replica) {
      const flight = replica.flight;
      replica.flight = null;
      for (const op of flight.ops) {
        const request = this.requests.get(op.id);
        if (!request || TERMINAL.has(request.state)) continue;
        if (request.state === "cancel_pending") {
          this._terminal(request, "cancelled", "in-flight pass completed; pages now safe to reuse");
          continue;
        }
        if (op.type === "prefill") {
          request.promptDone = Math.min(request.promptTokens, request.promptDone + op.tokens);
          if (request.promptDone === request.promptTokens) {
            request.prefillDoneAt = this.now;
            this._mark(request, "decode", "prompt prefill complete; first-token result available");
            this._emitToken(request);
          }
        } else this._emitToken(request);
      }
    }
    _readOutputs() {
      for (const request of this.requests.values()) {
        if (TERMINAL.has(request.state) || request.state === "cancel_pending") continue;
        if (!request.buffered) {
          request.readerCredit = 0;
          continue;
        }
        request.readerCredit += request.readerRate * TICK_MS / 1000;
        const count = Math.min(request.buffered, Math.floor(request.readerCredit + 1e-9));
        if (!count) continue;
        request.readerCredit -= count;
        request.buffered -= count;
        request.delivered += count;
        this.stats.delivered += count;
        if (request.firstDeliveredAt === null) {
          request.firstDeliveredAt = this.now;
          this.ttftSamples.push(this.now - request.createdAt);
          if (this.ttftSamples.length > 2048) this.ttftSamples.shift();
          this._event(request.id + " first token delivered to client", request);
        }
        if (request.state === "sending" && !request.buffered) {
          this._terminal(request, "completed", "all generated output delivered");
        }
      }
    }
    _step() {
      this.ticks++;
      this.now = this.ticks * TICK_MS / 1000;
      this._readOutputs();
      this.arrivalCredit += this.config.arrivalRate * TICK_MS / 1000;
      while (this.arrivalCredit + 1e-9 >= 1) {
        this.arrivalCredit -= 1;
        this.submit();
      }
      for (const replica of this.replicas) {
        if (replica.state === "warming" && this.now + 1e-9 >= replica.readyAt) {
          replica.state = "ready";
          this._event("Replica " + replica.label + " ready; eligible for new traffic");
        }
        if (replica.state !== "ready") continue;
        this._runTokenizer(replica);
        this._drainIPC(replica);
        if (replica.flight) {
          replica.busySeconds += Math.min(TICK_MS, Math.max(0, replica.flight.remainingMs)) / 1000;
          replica.flight.remainingMs -= TICK_MS;
          if (replica.flight.remainingMs <= 1e-9) this._finishFlight(replica);
        }
        this._prepareFlight(replica);
      }
      if (this.ticks % 10 === 0) this._recordHistory();
      if (this.ticks % 100 === 0) this._prune();
    }
    tick(count = 1) {
      if (!Number.isInteger(count) || count < 0) throw new RangeError("Tick count must be a nonnegative integer");
      for (let i = 0; i < count; i++) this._step();
      return this;
    }
    advance(seconds) { return this.tick(Math.round(seconds * 1000 / TICK_MS)); }
    _prune() {
      const terminal = [...this.requests.values()].filter(r => TERMINAL.has(r.state));
      for (const request of terminal.slice(0, Math.max(0, terminal.length - 256))) this.requests.delete(request.id);
    }
    _queueCounts() {
      const counts = {api: 0, tokenizing: 0, transit: 0, waiting: 0, active: 0, sending: 0, buffered: 0};
      for (const replica of this.replicas) {
        counts.api += replica.apiQueue.length;
        counts.tokenizing += replica.cpu.length;
        counts.transit += replica.transit.length;
        counts.waiting += replica.waiting.length;
        counts.active += replica.active.length;
      }
      for (const request of this.requests.values()) {
        counts.buffered += request.buffered;
        if (request.state === "sending") counts.sending++;
      }
      return counts;
    }
    deliveredRate() {
      if (!this.history.length || this.now === 0) return 0;
      const target = Math.max(0, this.now - 2);
      const first = this.history.find(entry => entry.at >= target) || this.history[this.history.length - 1];
      const elapsed = this.now - first.at;
      return elapsed > 0 ? (this.stats.delivered - first.delivered) / elapsed : 0;
    }
    _recordHistory() {
      const counts = this._queueCounts();
      const entry = {at: this.now, waiting: counts.waiting + counts.transit,
        api: counts.api + counts.tokenizing, buffered: counts.buffered,
        delivered: this.stats.delivered, rate: this.deliveredRate(), active: counts.active};
      this.history.push(entry);
      if (this.history.length > 240) this.history.shift();
    }
    snapshot() {
      const counts = this._queueCounts();
      return {now: this.now, config: this.config, memory: this.memory, stats: {...this.stats},
        counts, idle: this.config.clients - this.clientOwners.size,
        unfinished: this.clientOwners.size, deliveredRate: this.deliveredRate(),
        ttftP95: percentile(this.ttftSamples, 0.95), history: this.history.map(item => ({...item})),
        events: this.events.map(item => ({...item})),
        requests: [...this.requests.values()].map(r => ({...r, history: r.history.map(item => ({...item}))})),
        replicas: this.replicas.map(replica => {
          const assigned = replica.active.reduce((sum, id) => sum + this.requests.get(id).assignedPages, 0);
          return {id: replica.id, label: replica.label, state: replica.state,
            readyAt: replica.readyAt, api: replica.apiQueue.length, tokenizing: replica.cpu.length,
            transit: replica.transit.length, waiting: replica.waiting.length,
            ce: this._ceCount(replica), active: [...replica.active], reserved: this._reserved(replica),
            assigned, reusable: replica.state === "ready" ? this.memory.usablePages - this._reserved(replica) : 0,
            allocatedBytes: replica.state === "ready" ? this.memory.allocatedBytes : 0,
            flight: replica.flight ? {...replica.flight, ops: replica.flight.ops.map(op => ({...op}))} : null,
            busyFraction: this.now > replica.readyAt ? Math.min(1, replica.busySeconds / (this.now - replica.readyAt)) : 0,
            outstanding: this.outstanding(replica), workSeconds: this.estimatedWork(replica),
            completed: replica.completed, failures: replica.failures,
            buffered: [...this.requests.values()].filter(r => r.replicaId === replica.id).reduce((sum, r) => sum + r.buffered, 0)};
        })};
    }
  }
  return Object.freeze({Simulation, DEFAULTS, DEFAULT_EXAMPLE, PRESETS, GIB, TICK_MS,
    TERMINAL, normalize, memoryPlan, pagesFor, percentile});
});
