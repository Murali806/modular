/* A focused view of the existing teaching worker, not a second scheduler.
 * The four starting snapshots and event durations are illustrative.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./serve_story_model.js"));
  else root.ServeWorker = factory(root.ServeStory);
})(typeof globalThis !== "undefined" ? globalThis : this, function (engine) {
  "use strict";
  const {Worker, TERMINAL} = engine;
  const MODES = {
    normal: "R1 has 12 prompt tokens; R2 has 4. Follow their work through two batch slots and an eight-token input budget.",
    busy: "R1's selected pass is running. R2 is already admitted and waiting in request IPC; watch which pass can include it.",
    cache: "A batch slot is free, but R1 owns both teaching KV blocks. R2 waits for reusable blocks before it can start.",
    idle: "No requests are waiting. Follow the worker's empty check, NO_PROGRESS result and backoff.",
  };

  class WorkerProcess {
    constructor(mode = "normal") {
      if (!Object.hasOwn(MODES, mode)) throw new Error("Unknown worker teaching mode: " + mode);
      this.mode = mode;
      this.index = 0;
      this.finished = false;
      this.log = [];
      this.results = [];
      this.worker = new Worker({batch:2, budget:8, pending:2, queue:2, pages:mode === "cache" ? 2 : 4});
      const w = this.worker;
      // Enter through the model's normal transitions. The close-up starts
      // after API admission, with no HTTP server drawn inside the worker.
      if (mode === "normal") {
        w.submit({prompt:12, output:3});
        w.submit({prompt:4, output:2});
        this.prepare(() => w.at("ipc").length === 2);
      } else if (mode === "busy" || mode === "cache") {
        w.seedDecoder({prompt:8, output:4});
        this.prepare(() => w.busy);
        w.submit({prompt:4, output:2});
        this.prepare(() => w.at("ipc").length === 1);
        if (mode === "cache") this.prepare(() => w.phase === "select" && w.at("pending").length === 1);
      }
      this.event = {kind:mode === "busy" ? "execute" : mode === "cache" ? "cache" : "ready", text:MODES[mode]};
    }
    prepare(until) {
      for (let i = 0; !until(); i++) {
        if (i === 100) throw new Error("Worker opening snapshot did not converge");
        this.worker.next();
      }
    }
    get purpose() { return MODES[this.mode]; }
    get phase() { return this.index === 0 && ["normal", "idle"].includes(this.mode) ? "ready" : this.worker.phase; }
    next() {
      if (this.finished) return this.snapshot();
      const w = this.worker;
      if (this.index === 0 && this.mode === "normal") {
        // The opening IPC contexts arrived after an empty poll. Start this
        // close-up at the next drain, without replaying API-side events.
        do { w.next(); } while (w.last.kind !== "drain");
      } else w.next();
      this.index++;
      this.event = {...w.last};
      if (w.last.kind === "select" && w.at("pending").some(r => /KV pages/.test(r.reason))) {
        this.event = {kind:"cache", text:"R2 is admitted, but cannot reserve KV blocks. The free batch slot stays empty while R1 continues decoding; this wait is not an HTTP 429."};
      }
      if (w.last.kind === "publish") {
        this.results = w.requests.filter(r => r.buffered > 0).map(r => ({id:r.id, token:r.generated, done:r.generated === r.output}));
        if (!this.results.length) this.event.text = "This pass processed only part of a prompt. No output token is ready, so there is no response map to publish yet.";
      }
      this.log.push({...this.event, index:this.index});
      this.finished = w.phase === "idle" && w.requests.every(r => TERMINAL.has(r.state));
      return this.snapshot();
    }
    snapshot() {
      return {...this.worker.snapshot(), phase:this.phase, mode:this.mode, index:this.index,
        finished:this.finished, event:{...this.event}, results:this.results.map(r => ({...r})),
        capacity:{batch:this.worker.config.batch, budget:this.worker.config.budget, pages:this.worker.config.pages}};
    }
  }
  return {WorkerProcess, MODES};
});
