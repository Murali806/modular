/* Deterministic API teaching trace. No real HTTP, threads, sockets or timings. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ServeApi = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const MODES = {
    normal: {purpose:"All roles present. Follow two requests sharing one event-loop thread."},
    "no-server": {missing:"uvicorn", purpose:"Uvicorn's Server supplies HTTP ↔ ASGI. Remove that role: no configured HTTP service accepts these requests."},
    "no-app": {missing:"fastapi", purpose:"FastAPI supplies MAX's routes, validation and handlers. Remove that role: there is no MAX ASGI app to mount."},
    "no-loop": {missing:"loop", purpose:"The event loop runs ready async work. Remove the loop: startup, request handling and response delivery cannot advance."},
    "no-lock": {missing:"admission", purpose:"Possible interleaving without the lock: two probes see room, then a later send faces backpressure. Watch R2."},
    asyncio: {purpose:"A compatible asyncio loop can fill uvloop's role after startup/configuration changes. Removing the uvloop package alone does not provide a fallback."},
  };

  class ApiProcess {
    constructor(mode = "normal") {
      if (!Object.hasOwn(MODES, mode)) throw new Error("Unknown API teaching mode: " + mode);
      this.mode = mode;
      this.index = 0;
      this.finished = false;
      this.active = null;
      this.lockOwner = null;
      this.probes = {R1:null, R2:null};
      this.queue = [];
      this.outputChunks = 0;
      this.loopTask = mode === "no-loop" ? "No running loop · async work cannot start" : "Ready to run · awaiting your first step";
      const missingStartupRole = ["no-server", "no-app", "no-loop"].includes(mode);
      this.worker = missingStartupRole ? "outside this startup case" : "connected · not pulling yet";
      this.requests = ["R1", "R2"].map(id => ({id, state:"at client", location:"client"}));
      this.event = {kind:"ready", text:missingStartupRole
        ? "Press Play or Step to see why this role is needed before HTTP serving."
        : "Ready. Press Play or Step. The worker is connected, but is not pulling requests yet."};
      this.log = [];
    }
    get missing() { return MODES[this.mode].missing || null; }
    get purpose() { return MODES[this.mode].purpose; }
    setRequest(index, state, location) { Object.assign(this.requests[index], {state, location}); }
    record(active, loopTask, text, kind = "progress") {
      this.active = active;
      this.loopTask = loopTask;
      this.event = {kind, text};
      this.log.push({index:this.index, ...this.event});
    }
    next() {
      if (this.finished) return this.snapshot();
      this.index++;
      if (["no-server", "no-app", "no-loop"].includes(this.mode)) {
        const outcomes = {
          "no-server": ["No HTTP service", "The FastAPI object can exist, but without Uvicorn or a replacement HTTP/ASGI server, this configuration has no HTTP service to receive the requests."],
          "no-app": ["No MAX ASGI app", "The configured service cannot be assembled without an ASGI app. MAX's routes, validation and handlers are missing; neither request reaches tokenization or admission."],
          "no-loop": ["No async progress", "With no running event loop, serve() cannot advance through async startup. No request callback runs and no awaited response can resume."],
        };
        const [state, text] = outcomes[this.mode];
        this.requests.forEach(r => r.state = state);
        this.record(this.missing, this.mode === "no-loop" ? "No running loop · zero request callbacks" : "Service cannot start with this role absent", text, "blocked");
        this.finished = true;
        return this.snapshot();
      }

      // Shared setup: the loop schedules the request tasks; CPU tokenization is
      // offloaded. Each step groups operations for teaching, not wall-clock time.
      switch (this.index) {
        case 1:
          this.requests.forEach((r, i) => this.setRequest(i, "HTTP received", "server"));
          this.record("uvicorn", "Uvicorn: handle ready HTTP connections", "Uvicorn receives the HTTP requests and invokes the ASGI app. Server is its Python object inside the API process.");
          break;
        case 2:
          this.requests.forEach((r, i) => this.setRequest(i, "validated · handler running", "app"));
          this.record("fastapi", "Run each ready request task in turn", "FastAPI selects the MAX route, validates the request and calls its handler. R1 and R2 take turns on the same loop thread.");
          break;
        case 3:
          this.requests.forEach((r, i) => this.setRequest(i, "awaiting CPU tokenization", "executor"));
          this.record("tokenizer", "Request tasks suspended · other ready work may run", "The request tasks await tokenization in CPU executor threads. Their suspension leaves the loop available for other ready tasks and I/O.");
          break;
        case 4:
          this.requests.forEach((r, i) => this.setRequest(i, "tokenized · ready for admission", "app"));
          this.record("fastapi", "CPU results ready → resume request tasks", "Tokenization finishes. The loop resumes the request coroutines, which now have contexts to submit to the worker.");
          break;
        default:
          if (this.mode === "no-lock") this.unlockedAdmission();
          else this.lockedAdmission();
      }
      return this.snapshot();
    }
    lockedAdmission() {
      switch (this.index) {
        case 5:
          this.lockOwner = "R1";
          this.setRequest(0, "owns lock · probing IPC", "probe");
          this.setRequest(1, "awaiting admission lock", "app");
          this.record("admission", "R1 awaits probe; R2 suspends on the held lock", "R1 holds the admission lock while checking writability. An await can let R2 run, but R2 must wait for that same lock before checking the channel.");
          break;
        case 6:
          this.probes.R1 = true;
          this.setRequest(0, "probe passed · owns lock", "send");
          this.record("admission", "Resume R1: channel has room", "R1's probe reports room. The lock prevents R2 from taking that space between R1's check and send.");
          break;
        case 7:
          this.queue.push("R1");
          this.setRequest(0, "sent to request IPC", "ipc");
          this.record("ipc", "R1: register output queue → send context", "R1 registers its output queue and sends its context. The only illustrated request slot is now occupied; the admission lock still covers this send.");
          break;
        case 8:
          this.lockOwner = "R2";
          this.setRequest(0, "admitted · awaiting model output", "ipc");
          this.setRequest(1, "owns lock · probing IPC", "probe");
          this.record("admission", "R1 releases lock; R2 acquires it", "R1 releases the lock before waiting for output. R2 can now enter admission even though R1 is still waiting on the worker.");
          break;
        case 9:
          this.probes.R2 = false;
          this.lockOwner = null;
          this.setRequest(1, "RequestQueueFull → HTTP 429", "app");
          this.record("fastapi", "R2: full channel → exception handler", "R2's probe sees the full, connected channel and raises RequestQueueFull. The API prepares HTTP 429 with Retry-After: 1 before streaming R2.", "reject");
          break;
        case 10:
          this.setRequest(1, "HTTP 429 returned", "client-error");
          this.record("uvicorn", "Write R2's HTTP response", "R2 receives HTTP 429. R1 remains admitted. Rejecting R2 did not require waiting for R1's generation to finish.", "reject");
          break;
        case 11:
          this.record("loop", "No ready work → await I/O; resume when ready", "R1's response task is suspended. The event loop can service other ready work, or wait for I/O if nothing is ready. The admission lock is free.");
          break;
        case 12:
          this.queue = [];
          this.worker = "worker took R1 · slot free";
          this.setRequest(0, "awaiting response data", "app");
          this.record("ipc", "R1 still suspended · response not ready", "The offstage worker eventually pulls R1, freeing the IPC slot. R1 still awaits output; worker scheduling and GPU execution are outside this lesson.");
          break;
        case 13:
          this.worker = "first response arrived";
          this.setRequest(0, "response ready · task resumes", "response");
          this.record("loop", "Response data ready → resume R1", "Worker output arrives through response IPC. The API routes it to R1's output queue, making the waiting response path ready to resume.");
          break;
        case 14:
          this.setRequest(0, "sending first output chunk", "server");
          this.record("uvicorn", "Resume response path → ASGI send → HTTP", "The API prepares the streamed response and Uvicorn writes it to the HTTP connection. The same loop advances ready response work.");
          break;
        case 15:
          this.setRequest(0, "first chunk delivered · streaming", "client-output");
          this.outputChunks = 1;
          this.record("loop", "Ready for the next request or output event", "R1's first output chunk reaches the client. Later chunks repeat the wake → resume → send cycle. R2 was rejected at admission while R1 continued.");
          this.finished = true;
          break;
      }
    }
    unlockedAdmission() {
      // An intentionally selected, possible interleaving, not a claim that every
      // unlocked run races. Capacity is an exact teaching slot, unlike ZMQ HWM.
      switch (this.index) {
        case 5:
          this.probes.R1 = true;
          this.setRequest(0, "probe passed · no lock", "probe");
          this.record("admission", "R1 probes; a coroutine can yield before sending", "In this possible interleaving, R1's writability probe observes room. No lock protects the interval before it sends.");
          break;
        case 6:
          this.probes.R2 = true;
          this.setRequest(1, "probe also passed · no lock", "probe");
          this.record("admission", "R2 also probes before R1 sends", "R2 runs before R1 enqueues and also observes room. Both positive checks refer to the same available capacity, even on a single loop thread.");
          break;
        case 7:
          this.queue.push("R1");
          this.setRequest(0, "admitted · awaiting model output", "ipc");
          this.setRequest(1, "earlier probe passed · about to send", "send");
          this.record("ipc", "R1 sends → the illustrated slot fills", "R1 sends first and occupies the slot. R2 still carries its earlier positive probe result; that result no longer guarantees room for its send.");
          break;
        case 8:
          this.setRequest(1, "send can wait for room", "send");
          this.record("admission", "R2 can suspend on send · the loop still runs", "R2's later send can wait under backpressure instead of being rejected immediately. The loop still works; the lost guarantee is atomic probe → send. Compare All present to see R2 get HTTP 429.", "race");
          this.finished = true;
          break;
      }
    }
    snapshot() {
      return {
        mode:this.mode, missing:this.missing, index:this.index, finished:this.finished,
        active:this.active, lockOwner:this.lockOwner, probes:{...this.probes},
        queue:[...this.queue], outputChunks:this.outputChunks, loopTask:this.loopTask,
        worker:this.worker, requests:this.requests.map(r => ({...r})), event:{...this.event},
      };
    }
  }
  return {ApiProcess, MODES};
});
