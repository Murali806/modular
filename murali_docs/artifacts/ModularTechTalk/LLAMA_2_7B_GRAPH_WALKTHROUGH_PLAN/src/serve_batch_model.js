/* SOURCE branch projection of TextBatchConstructor, for one ordinary replica.
 * Counts, page costs and readiness events are teaching inputs, not GPU traces.
 * No adapters, speculative drafts, DP padding or external KV store is modeled.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ServeBatch = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const MODES = {
    mixed: "R1 is decoding; R2 needs 12 prompt tokens; R3 needs 3. Build each pass from the same owned request state.",
    seats: "Two active decoders occupy both batch seats. Spare token budget and KV blocks do not create a third seat.",
    cache: "R1 needs the last free KV block for its next decode input. R2 is admitted but its prefill must wait for reusable blocks.",
    preempt: "All three KV blocks are owned. The oldest decoder needs another block; reclaim the newest decoder's cache and requeue it for recomputation.",
    context: "The optional total-context budget is six tokens, with costs rounded to four-token pages. A short active chunk can still fail that separate budget.",
    readiness: "R2 awaits a grammar; R3 awaits a KV onload. Poll readiness each iteration while ready R1 continues.",
    impossible: "One KV block cannot hold R1's 12-token context. There is no other work, transfer or reclaimable owner to relieve the shortage.",
    idle: "No CE, TG or held requests exist. Return empty inputs; the caller reports NO_PROGRESS and yields.",
  };
  const roundPage = n => Math.ceil(n / 4);
  function request(id, phase, prompt, target = 3, extra = {}) {
    return Object.assign({id, phase, prompt, originalPrompt:prompt, processed:phase === "TG" ? prompt : 0,
      emitted:phase === "TG" ? 1 : 0, sinceReset:phase === "TG" ? 1 : 0, target,
      pages:phase === "TG" ? roundPage(prompt) : 0, reason:"tracked for a later pass", resets:0}, extra);
  }
  class BatchConstructor {
    constructor(mode = "mixed", options = {}) {
      if (!Object.hasOwn(MODES, mode)) throw new Error("Unknown batch scene: " + mode);
      this.mode = mode;
      this.config = Object.assign({slots:mode === "seats" ? 2 : 3, budget:8, chunking:true,
        policy:"mixed", pages:mode === "impossible" ? 1 : ["cache", "preempt"].includes(mode) ? 3 : 12,
        total:mode === "context" ? 6 : null}, options);
      if (!["mixed", "prefill", "strict"].includes(this.config.policy)) throw new Error("Unknown policy");
      this.requests = mode === "idle" ? [] : mode === "impossible" ? [request("R1", "CE", 12)] : mode === "context"
        ? [request("R1", "CE", 3, 1), request("R2", "CE", 4, 2)]
        : mode === "cache"
          ? [request("R1", "TG", 8, 3), request("R2", "CE", 4, 2)]
          : mode === "preempt"
            ? [request("R1", "TG", 8, 2), request("R2", "TG", 4, 3)]
            : mode === "readiness"
              ? [request("R1", "TG", 4, 5), request("R2", "HOLD", 4, 2, {hold:"grammar", readyAt:2}),
                request("R3", "HOLD", 8, 2, {hold:"KV onload", readyAt:3, pages:1, processed:4})]
              : mode === "seats"
                ? [request("R1", "TG", 4, 3), request("R2", "TG", 4, 4), request("R3", "CE", 4, 2)]
                : [request("R1", "TG", 4, 4), request("R2", "CE", 12, 2), request("R3", "CE", 3, 2)];
      this.ce = this.requests.filter(r => r.phase === "CE").map(r => r.id);
      this.tg = this.requests.filter(r => r.phase === "TG").map(r => r.id);
      this.batch = []; this.inputs = []; this.history = []; this.log = [];
      this.used = 0; this.totalUsed = 0; this.pass = 0; this.index = 0;
      this.phase = "ready"; this.focus = null; this.location = "sets"; this.checks = {};
      this.priority = "not chosen"; this.finished = false;
      this.event = {kind:"ready", text:MODES[mode]};
      this.runner = this.run();
    }
    get purpose() { return MODES[this.mode]; }
    find(id) { return this.requests.find(r => r.id === id); }
    get pages() { return this.requests.reduce((n, r) => n + r.pages, 0); }
    get remaining() {
      return Math.min(this.config.budget - this.used,
        this.config.total === null ? Infinity : this.config.total - this.totalUsed);
    }
    *eventAt(phase, text, kind = phase) {
      this.phase = phase;
      yield {kind, text};
    }
    resetContext(r) {
      // TextContext.reset() makes the existing sequence a fresh prompt.
      r.prompt += r.sinceReset;
      r.sinceReset = 0; r.processed = 0; r.pages = 0; r.phase = "CE"; r.resets++;
      this.tg = this.tg.filter(id => id !== r.id);
      this.ce = [r.id, ...this.ce.filter(id => id !== r.id)];
    }
    *cacheAllocation(r, kind, candidates = []) {
      const needed = roundPage(kind === "TG" ? r.prompt + r.sinceReset : r.prompt);
      let extra = Math.max(0, needed - r.pages);
      while (this.pages + extra > this.config.pages && kind === "TG" && candidates.length) {
        const victim = this.find(candidates.pop());
        const released = victim.pages;
        this.resetContext(victim);
        victim.reason = "preempted: recompute " + victim.prompt + " existing tokens as a fresh prompt";
        this.checks.kv = "wait";
        yield* this.eventAt("decode", victim.id + " is the newest remaining TG candidate. Preempt it, release " + released + " KV block(s), preserve existing tokens, and requeue it in CE.", "preempt");
        extra = Math.max(0, needed - r.pages);
      }
      if (this.pages + extra > this.config.pages) {
        this.checks.kv = "wait";
        r.reason = "waiting: need " + extra + " additional KV block(s), " + (this.config.pages - this.pages) + " free";
        if (this.mode === "impossible") {
          r.reason = "fatal: no other work or in-flight transfer can free enough blocks";
          yield* this.eventAt("fatal", "InsufficientBlocksError: R1 cannot fit and nothing can resolve this shortage. The constructor raises an error instead of returning an endlessly waiting batch.", "fatal");
          return false;
        }
        yield* this.eventAt(kind === "TG" ? "decode" : "prefill", r.id + " cannot allocate KV: " + r.reason + ". Keep it for another pass; this is already-admitted work.", "wait");
        return false;
      }
      r.pages += extra;
      this.checks.kv = "pass";
      yield* this.eventAt(kind === "TG" ? "decode" : "prefill", r.id + ": KV allocation succeeds (" + extra + " additional block(s)); " + this.pages + "/" + this.config.pages + " owned.");
      return true;
    }
    *decode() {
      const candidates = this.tg.slice();
      while (candidates.length && this.batch.length < this.config.slots) {
        const r = this.find(candidates.shift());
        this.focus = r.id; this.location = "checks"; this.checks = {seats:"pass"};
        yield* this.eventAt("decode", "Consider oldest TG " + r.id + ". Ordinary decode needs one active input token.");
        if (this.config.budget - this.used <= 0) {
          this.checks.tokens = "wait"; r.reason = "active-token budget exhausted";
          yield* this.eventAt("decode", "BUDGET_EXHAUSTED: no active input room for " + r.id + ". Stop adding TG work.", "wait");
          break;
        }
        this.checks.tokens = "pass";
        yield* this.eventAt("decode", "Active-token check passes for " + r.id + ": 1 input token fits before the KV allocation.");
        if (!(yield* this.cacheAllocation(r, "TG", candidates))) break;
        this.used++; this.totalUsed += roundPage(r.prompt + r.sinceReset) * 4;
        this.batch.push({id:r.id, kind:"TG", tokens:1, start:r.prompt + r.sinceReset - 1});
        r.reason = "selected for this decode pass"; this.location = "batch";
        yield* this.eventAt("decode", "Select " + r.id + " · D × 1. Record its token cost; keep its context in TG for later passes.");
      }
    }
    *prefill() {
      while (this.ce.length) {
        const activeSeats = this.batch.filter(op => op.kind === "CE").length + this.tg.length;
        if (this.batch.length >= this.config.slots || activeSeats >= this.config.slots) {
          const r = this.find(this.ce[0]);
          this.focus = r.id; this.location = "checks"; this.checks = {seats:"wait"};
          r.reason = "waiting for an active request seat";
          yield* this.eventAt("prefill", "No CE seat for " + r.id + ": TG contexts also occupy active seats, including decoders not selected in this pass.", "wait");
          break;
        }
        const r = this.find(this.ce[0]);
        this.focus = r.id; this.location = "checks"; this.checks = {seats:"pass"};
        yield* this.eventAt("prefill", "Consider CE queue head " + r.id + ": " + r.processed + "/" + r.prompt + " prompt tokens processed. A request seat is available.");
        if (this.tg.length && this.batch.some(op => op.kind === "TG") && this.pages / this.config.pages > 0.9 && r.pages === 0) {
          this.checks.kv = "wait"; r.reason = "KV pressure: let active TG drain before new prefill";
          yield* this.eventAt("prefill", "KV pressure holds new " + r.id + " outside this pass. Let existing TG work drain; a prefill already holding KV can continue.", "wait");
          break;
        }
        if (this.remaining <= 0) {
          this.checks.tokens = "wait"; r.reason = "no remaining token budget";
          yield* this.eventAt("prefill", "No remaining budget for " + r.id + ". Leave it at the CE head for another pass.", "wait");
          break;
        }
        if (!(yield* this.cacheAllocation(r, "CE"))) break;
        const active = r.prompt - r.processed;
        // The real collection can shorten the active window using its minimum
        // remaining budget. Alignment/min-chunk/per-context caps are disabled here.
        const cut = this.config.chunking ? Math.min(active, this.remaining) : active;
        const activeStatus = cut >= this.config.budget - this.used ? "REACHED" : "AVAILABLE";
        this.checks.tokens = "pass";
        yield* this.eventAt("prefill", r.id + (cut < active ? ": chunk " + cut + " of " + active + " remaining prompt tokens" : ": take " + cut + " prompt tokens") + ". Active budget " + activeStatus + (cut > this.config.budget - this.used ? " · chunking OFF makes this target soft." : "."));
        const cost = roundPage(r.prompt + r.sinceReset) * 4;
        // TokenBudgetCollection stops on the first non-AVAILABLE status.
        // Evaluate the optional total-context budget only when active reports AVAILABLE.
        if (activeStatus === "AVAILABLE" && this.config.total !== null && this.totalUsed + cost > this.config.total) {
          this.checks.context = "wait";
          r.reason = "aligned total-context cost " + cost + " exceeds remaining " + (this.config.total - this.totalUsed);
          r.pages = 0;
          yield* this.eventAt("prefill", "BUDGET_EXHAUSTED: " + r.id + "'s total-context cost is " + cost + " even with a " + cut + "-token chunk. Release its cache claim and retry from the CE head.", "wait");
          break;
        }
        this.checks.context = this.config.total === null ? "off" : "pass";
        this.ce.shift(); this.used += cut; this.totalUsed += cost;
        this.batch.push({id:r.id, kind:"CE", tokens:cut, start:r.processed});
        r.phase = "SELECTED_CE"; r.reason = "selected prefill window " + r.processed + "…" + (r.processed + cut);
        this.location = "batch";
        yield* this.eventAt("prefill", "Select " + r.id + " · P × " + cut + ". Pop it from CE; the selected batch keeps the same context reference.", "select");
        if (activeStatus === "REACHED" || this.remaining <= 0) break;
      }
    }
    *run() {
      while (true) {
        this.pass++; this.batch = []; this.inputs = []; this.used = 0; this.totalUsed = 0;
        this.focus = null; this.location = "sets"; this.checks = {};
        for (const r of this.requests.filter(r => r.phase === "HOLD" && r.readyAt <= this.pass)) {
          r.phase = "CE"; this.ce.push(r.id); r.reason = r.hold + " ready: admitted to CE";
        }
        yield* this.eventAt("readiness", "Pass " + this.pass + ": poll completed KV transfers; readmit landed onloads; promote grammar-ready requests.");
        this.priority = this.tg.length ? (this.config.policy === "prefill" && this.ce.length ? "CE" : "TG") : this.ce.length ? "CE" : "NONE";
        yield* this.eventAt("policy", "Start an empty batch and fresh token budgets. Priority: " + this.priority + (this.priority === "TG" ? this.config.policy === "strict" ? " · global override; no CE backfill." : " · per replica, in-flight batching enabled." : this.priority === "CE" ? " · prefill first; TG fallback only if CE adds nothing." : " · no runnable request."));
        if (this.priority === "TG") {
          yield* this.decode();
          if (this.config.policy === "mixed" && this.batch.length) yield* this.prefill();
        } else if (this.priority === "CE") {
          yield* this.prefill();
          if (!this.batch.length && this.tg.length) yield* this.decode();
        }
        this.focus = null; this.location = "batch";
        this.inputs = this.batch.map(op => ({...op}));
        this.history.push({pass:this.pass, batch:this.inputs.map(op => ({...op})), used:this.used});
        yield* this.eventAt("pack", "Return TextGenerationInputs: batches = [[" + this.inputs.map(op => op.id).join(", ") + "]]. Context references are packed; the pipeline prepares tensors next.");
        if (!this.batch.length) {
          yield* this.eventAt("idle", "Empty inputs → caller returns NO_PROGRESS → worker awaits backoff, then checks again.", "idle");
          if (this.requests.every(r => r.phase === "DONE")) return;
          continue;
        }
        yield* this.eventAt("execute", "The caller executes this fixed input batch. The constructor makes no change to its membership while this ordinary pass runs.");
        const progress = [];
        for (const op of this.batch) {
          const r = this.find(op.id);
          if (op.kind === "CE") r.processed += op.tokens;
          if (op.kind === "TG" || r.processed === r.prompt) {
            r.emitted++; r.sinceReset++;
            r.phase = "TG";
            if (!this.tg.includes(r.id)) this.tg.push(r.id);
            progress.push(r.id + ": output token " + r.emitted);
          } else {
            r.phase = "CE";
            progress.push(r.id + ": partial prefill, no output");
          }
        }
        const partial = this.batch.filter(op => this.find(op.id).phase === "CE").map(op => op.id);
        this.ce = [...partial, ...this.ce];
        for (const r of this.requests.filter(r => r.emitted >= r.target && r.phase === "TG")) {
          r.phase = "DONE"; r.pages = 0; r.reason = "finished: request and KV ownership released";
          this.tg = this.tg.filter(id => id !== r.id);
          progress.push(r.id + ": release KV");
        }
        yield* this.eventAt("advance", "After execution, advance_requests() returns partial prompts to the CE head and promotes completed prefill to TG. The scheduler releases terminated work. " + progress.join("; ") + ".");
        if (this.requests.every(r => r.phase === "DONE")) {
          yield* this.eventAt("complete", "All requests finished. KV block ownership is reusable; the backing pool is retained. The next constructor call starts with empty request sets.");
          return;
        }
      }
    }
    next() {
      if (this.finished) return this.snapshot();
      const next = this.runner.next();
      if (next.done) { this.finished = true; return this.snapshot(); }
      this.index++; this.event = next.value; this.log.push({...next.value, index:this.index, pass:this.pass});
      if (["complete", "fatal"].includes(this.phase) || (this.phase === "idle" && this.requests.every(r => r.phase === "DONE"))) this.finished = true;
      if (this.pass > 50) throw new Error("Batch teaching scenario failed to converge");
      return this.snapshot();
    }
    snapshot() {
      return {mode:this.mode, config:{...this.config}, index:this.index, pass:this.pass, phase:this.phase,
        priority:this.priority, finished:this.finished, event:{...this.event},
        requests:this.requests.map(r => ({...r})), ce:this.ce.slice(), tg:this.tg.slice(),
        batch:this.batch.map(op => ({...op})), inputs:this.inputs.map(op => ({...op})),
        history:this.history.map(h => ({...h, batch:h.batch.map(op => ({...op}))})),
        used:this.used, totalUsed:this.totalUsed, pages:this.pages,
        focus:this.focus, location:this.location, checks:{...this.checks}};
    }
  }
  return {BatchConstructor, MODES};
});
