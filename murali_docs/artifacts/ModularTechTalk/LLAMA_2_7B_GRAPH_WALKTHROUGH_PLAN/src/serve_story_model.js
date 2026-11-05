/* A discrete-event teaching model of one ordinary text-generation worker.
 * Beats are presentation units, never seconds or throughput predictions.
 * Queue N is exact here; real ZeroMQ HWM is approximate. Scheduling is a
 * deliberately small, chunked, decode-first example, not MAX's full policy.
 */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ServeStory = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const TERMINAL = new Set(["done", "rejected", "cancelled"]);
  const PHASES = ["drain", "select", "execute", "advance", "publish"];
  const DEFAULTS = Object.freeze({queue: 2, pending: 2, batch: 2, budget: 8,
    pages: 24, pageTokens: 8, readEvery: 1, prompt: 8, output: 5});
  class Worker {
    constructor(options = {}) {
      this.config = {...DEFAULTS, ...options};
      for (const key of Object.keys(DEFAULTS)) {
        if (!Number.isInteger(this.config[key]) || this.config[key] < 1) throw new RangeError(key);
      }
      this.time = 0;
      this.serial = 0;
      this.sequence = 0;
      this.requests = [];
      this.events = [];
      this.log = [];
      this.last = {kind: "idle", text: "Ready. The worker has no batch to execute."};
      this.phase = "idle";
      this.pass = 0;
      this.batch = [];
      this.history = [];
      this.idlePolls = 0;
      this.wakeScheduled = false;
      this._schedule(1, "worker");
    }
    _schedule(delay, kind, id = null) {
      this.events.push({at: this.time + delay, serial: this.serial++, kind, id});
      this.events.sort((a, b) => a.at - b.at || a.serial - b.serial);
      if (kind === "worker") this.wakeScheduled = true;
    }
    _note(kind, text, id = null) {
      this.last = {kind, text, id, at: this.time};
      this.log.push(this.last);
      if (this.log.length > 80) this.log.shift();
    }
    find(id) { return this.requests.find(r => r.id === id); }
    at(state) { return this.requests.filter(r => r.state === state); }
    get active() { return this.requests.filter(r => ["prefill", "decode", "cancel_pending"].includes(r.state)); }
    get ceCount() { return this.requests.filter(r => ["pending", "prefill"].includes(r.state) || (r.state === "cancel_pending" && r.promptDone < r.prompt)).length; }
    get usedPages() { return this.requests.reduce((n, r) => n + r.pages, 0); }
    get live() { return this.requests.filter(r => !TERMINAL.has(r.state)); }
    get busy() { return this.phase === "execute"; }
    get writable() { return this.at("ipc").length < this.config.queue; }
    submit(options = {}) {
      const number = ++this.sequence;
      const r = {id: options.id || "R" + number, number, prompt: options.prompt || this.config.prompt,
        output: options.output || this.config.output, promptDone: 0, generated: 0, delivered: 0,
        buffered: 0, unpublished: 0, pages: 0, state: "api", arrived: this.time, accepted: null, selected: null,
        ended: null, responseStarted: false, cancellation: false, reading: false, reason: "Tokenizing the prompt in the API process."};
      if (this.find(r.id)) throw new Error("Duplicate request " + r.id);
      this.requests.push(r);
      this._schedule(options.admissionDelay || 1, "admit", r.id);
      this._note("arrival", r.id + " reaches the API. Its prompt must be tokenized before worker admission.", r.id);
      return r;
    }
    seedDecoder(options = {}) {
      const r = this.submit(options);
      this.events = this.events.filter(e => !(e.kind === "admit" && e.id === r.id));
      r.state = "decode"; r.promptDone = r.prompt; r.generated = 1; r.delivered = 1;
      r.responseStarted = true; r.accepted = 0; r.selected = 0;
      r.pages = Math.ceil((r.prompt + r.output) / this.config.pageTokens);
      if (r.pages > this.config.pages) throw new RangeError("Opening decoder does not fit");
      this._note("seed", r.id + " is already decoding in this opening scene.", r.id);
      return r;
    }
    cancel(id) {
      const r = this.find(id);
      if (!r || TERMINAL.has(r.state) || r.cancellation) return false;
      r.cancellation = true;
      if (this.batch.some(op => op.id === id) && ["select", "execute"].includes(this.phase)) {
        r.state = "cancel_pending";
        r.reason = "Cancellation requested; finish the selected pass before releasing ownership.";
      } else this._cancel(r);
      this._note("cancel", r.id + ": cancellation travels to the worker; cleanup occurs at a safe boundary.", id);
      return true;
    }
    _cancel(r) {
      r.state = "cancelled"; r.pages = 0; r.buffered = 0; r.unpublished = 0; r.ended = this.time;
      r.reason = "Worker released this request at a safe boundary.";
    }
    _admit(r) {
      if (!r || r.state !== "api") return;
      if (!this.writable) {
        r.state = "rejected"; r.ended = this.time;
        r.reason = "Request channel unwritable: HTTP 429 before streaming starts.";
        this._note("reject", r.id + " → HTTP 429. The connected request channel is full; no output queue or GPU slot is assigned.", r.id);
      } else {
        r.state = "ipc"; r.accepted = this.time;
        r.reason = "Accepted at the worker handoff. Waiting for the worker to pull the context.";
        this._note("admit", r.id + " accepted. The request channel has room, even if the worker is executing another batch.", r.id);
      }
    }
    _select() {
      const c = this.config;
      // Reserve future pages in this lesson so the finite toy pool is safe.
      // MAX allocates per pass and can preempt; see the page's SOURCE contract.
      for (const r of this.at("pending")) {
        if (this.active.length >= c.batch) {
          r.reason = "Waiting for a batch slot; admission has already succeeded.";
          continue;
        }
        const needed = Math.ceil((r.prompt + r.output) / c.pageTokens);
        if (this.usedPages + needed > c.pages) {
          r.reason = "Waiting for reusable KV pages in this teaching reservation model.";
          continue;
        }
        r.pages = needed; r.selected = this.time; r.state = "prefill";
      }
      let remaining = c.budget;
      this.batch = [];
      for (const r of this.at("decode")) {
        if (!remaining) break;
        this.batch.push({id: r.id, kind: "decode", tokens: 1}); remaining--;
      }
      for (const r of this.at("prefill")) {
        if (!remaining) { r.reason = "This pass used its input-token budget; prefill waits for another pass."; continue; }
        const tokens = Math.min(remaining, r.prompt - r.promptDone);
        this.batch.push({id: r.id, kind: "prefill", tokens}); remaining -= tokens;
      }
      if (this.batch.length) {
        this.pass++;
        for (const op of this.batch) this.find(op.id).reason = "Selected for pass " + this.pass + ": " + op.kind + " (" + op.tokens + " input tokens).";
        this.history.push({pass: this.pass, ops: this.batch.map(op => ({...op})), finished: false});
        if (this.history.length > 8) this.history.shift();
      }
    }
    _worker() {
      this.wakeScheduled = false;
      switch (this.phase) {
        case "idle":
        case "publish": {
          this.phase = "drain";
          const available = Math.max(0, this.config.pending - this.ceCount);
          const pulled = this.at("ipc").slice(0, available);
          for (const r of pulled) { r.state = "pending"; r.reason = "Inside the worker's pending-prefill queue. Batch and KV eligibility come next."; }
          this._note("drain", pulled.length ? "Worker pulls " + pulled.map(r => r.id).join(", ") + " into pending prefill, within M = " + this.config.pending + "." : (available ? "Worker checks the request channel. No new contexts to pull." : "M is full. The worker stops pulling; excess requests stay in IPC."));
          this._schedule(1, "worker");
          break;
        }
        case "drain":
          this.phase = "select";
          this._select();
          this._note("select", this.batch.length ? "Pass " + this.pass + " selects " + this.batch.map(op => op.id + " " + op.kind + " ×" + op.tokens).join(" + ") + ". Check batch slots, token budget and KV space." : "No eligible batch. The worker returns NO_PROGRESS and yields with backoff.");
          this._schedule(1, "worker");
          break;
        case "select":
          if (!this.batch.length) {
            this.phase = "idle"; this.idlePolls++;
            this._note("idle", "NO_PROGRESS: yield with backoff, then check again. The API process can still handle requests.");
            this._schedule(Math.min(4, this.idlePolls + 1), "worker");
          } else {
            this.phase = "execute"; this.idlePolls = 0;
            this._note("execute", "The worker calls pipeline.execute() for pass " + this.pass + ". The native runtime and device execute the selected work; the API remains independent.");
            this._schedule(2, "worker");
          }
          break;
        case "execute":
          this.phase = "advance";
          for (const op of this.batch) {
            const r = this.find(op.id);
            if (r.cancellation) { this._cancel(r); continue; }
            if (op.kind === "prefill") {
              r.promptDone += op.tokens;
              r.reason = "Prefill has consumed " + r.promptDone + " / " + r.prompt + " prompt tokens. Unfinished input returns for another pass.";
            }
            if (r.promptDone === r.prompt) {
              r.state = "decode"; r.generated++; r.unpublished++;
              r.reason = "Generated " + r.generated + " / " + r.output + " output tokens; eligible for another decode pass.";
              if (r.generated === r.output) { r.state = "streaming"; r.pages = 0; r.reason = "Generation finished. KV ownership is reusable; buffered output is still draining."; }
            }
          }
          this.history[this.history.length - 1].finished = true;
          this._note("advance", "Pass " + this.pass + " completed. Advance prompt/decode state and release finished requests' KV ownership.");
          this._schedule(1, "worker");
          break;
        case "advance":
          this.phase = "publish";
          for (const op of this.batch) {
            const r = this.find(op.id);
            r.buffered += r.unpublished; r.unpublished = 0;
            if (r.buffered && !r.reading && !TERMINAL.has(r.state)) { r.reading = true; this._schedule(this.config.readEvery, "read", r.id); }
          }
          this._note("publish", "Send results over response IPC, correlated by request ID. The API streams tokens; the next worker iteration can choose a different batch.");
          this._schedule(1, "worker");
          break;
      }
    }
    _read(r) {
      if (!r || TERMINAL.has(r.state)) return;
      r.reading = false;
      if (r.buffered) {
        r.buffered--; r.delivered++; r.responseStarted = true;
        this._note("token", r.id + ": token " + r.delivered + " reaches the client through response IPC → API/SSE → TCP.", r.id);
      }
      if (r.state === "streaming" && !r.buffered && !r.unpublished) { r.state = "done"; r.ended = this.time; r.reason = "All output delivered; request complete."; }
      else if (r.buffered) { r.reading = true; this._schedule(this.config.readEvery, "read", r.id); }
    }
    next() {
      const event = this.events.shift();
      if (!event) return false;
      this.time = event.at;
      if (event.kind === "worker") this._worker();
      else if (event.kind === "admit") this._admit(this.find(event.id));
      else if (event.kind === "read") this._read(this.find(event.id));
      return true;
    }
    advanceTo(time) {
      if (time < this.time) throw new RangeError("Cannot run backwards");
      while (this.events.length && this.events[0].at <= time) this.next();
      this.time = time;
      return this;
    }
    snapshot() {
      const count = state => this.at(state).length;
      return {time: this.time, phase: this.phase, pass: this.pass, writable: this.writable,
        ipc: count("ipc"), pending: count("pending"), ce: this.ceCount, active: this.active.length,
        accepted: this.requests.filter(r => r.accepted !== null).length, rejected: count("rejected"),
        done: count("done"), cancelled: count("cancelled"), live: this.live.length,
        generated: this.requests.reduce((n, r) => n + r.generated, 0),
        delivered: this.requests.reduce((n, r) => n + r.delivered, 0),
        buffered: this.requests.reduce((n, r) => n + r.buffered, 0), pages: this.usedPages,
        last: {...this.last}, batch: this.batch.map(op => ({...op})), requests: this.requests.map(r => ({...r}))};
    }
  }
  class Fleet {
    constructor() {
      this.time = 0; this.sequence = 0; this.cursor = 0;
      this.replicas = [{name: "A", readyAt: 0, worker: new Worker({batch: 1, pending: 2, queue: 2, output: 5})}];
      this.last = "One ready replica. Routing here is an external example policy.";
    }
    addReplica() {
      if (this.replicas.length > 1) return false;
      const worker = new Worker({batch: 1, pending: 2, queue: 2, output: 5});
      worker.time = this.time;
      worker.events = [];
      worker._schedule(5, "worker");
      this.replicas.push({name: "B", readyAt: this.time + 4, worker});
      this.last = "Replica B is initializing. New requests continue to A until B is ready.";
      return true;
    }
    submit(options = {}) {
      const ready = this.replicas.filter(r => r.readyAt <= this.time);
      const replica = ready[this.cursor++ % ready.length];
      const request = replica.worker.submit({...options, id: "R" + (++this.sequence)});
      this.last = request.id + " routed to ready replica " + replica.name + ". Existing requests stay on their original replica.";
      return request;
    }
    next() {
      const time = Math.min(...this.replicas.flatMap(r => r.worker.events.map(e => e.at)));
      this.time = time;
      for (const r of this.replicas) r.worker.advanceTo(time);
      this.last = this.replicas.length > 1 && this.replicas[1].readyAt <= time ? "A and B are ready. Each worker selects batches and owns its own KV pool." : "A handles requests while the external router waits for another ready replica.";
    }
  }
  return {Worker, Fleet, PHASES, TERMINAL, DEFAULTS};
});
