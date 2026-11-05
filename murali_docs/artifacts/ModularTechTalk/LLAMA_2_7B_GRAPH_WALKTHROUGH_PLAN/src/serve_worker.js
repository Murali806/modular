/* Presentation of the model-worker close-up; shares the article frame loop. */
(function () {
  "use strict";
  window.ServeWorkerView = class WorkerStory {
    constructor(node, {reduced, startFrames, colors, escape, element}) {
      Object.assign(this, {node, reduced, startFrames, colors, escape, element});
      this.key = "worker";
      this.interval = 1300;
      this.mode = "normal";
      this.playing = false;
      this.dots = new Map();
      this.tokens = [];
      node.querySelectorAll("[data-worker-controls]").forEach(el => el.hidden = false);
      node.addEventListener("click", e => {
        const mode = e.target.closest("button[data-worker-mode]");
        if (mode) { this.mode = mode.dataset.workerMode; this.reset(); return; }
        const action = e.target.closest("[data-worker-action]")?.dataset.workerAction;
        if (action === "reset") this.reset();
        if (action === "step") { this.setPlaying(false); this.step(true); }
        if (action === "play") {
          if (this.model.finished) this.reset();
          this.setPlaying(!this.playing);
        }
      });
      this.reset();
    }
    reset() {
      this.model = new window.ServeWorker.WorkerProcess(this.mode);
      this.dots.clear(); this.tokens = [];
      this.node.querySelector("[data-worker-dots]").replaceChildren();
      this.node.querySelector("[data-worker-token-dots]").replaceChildren();
      this.setPlaying(false);
      this.render(true);
    }
    setPlaying(value) {
      this.playing = value;
      this.node.dataset.playing = String(value);
      const button = this.node.querySelector('[data-worker-action="play"]');
      button.textContent = value ? "Ⅱ Pause" : this.model.finished ? "▶ Replay" : "▶ Play";
      button.setAttribute("aria-pressed", String(value));
      this.node.querySelector("[data-current-event]").setAttribute("aria-live", value ? "off" : "polite");
      this.nextAt = performance.now() + this.interval;
      if (value) this.startFrames();
    }
    step(snap = false) {
      this.model.next();
      this.render(snap || this.reduced.matches);
      if (this.model.finished) this.setPlaying(false);
    }
    render(snap = false) {
      const s = this.model.snapshot(), node = this.node;
      const text = (selector, value) => node.querySelector(selector).textContent = value;
      Object.assign(node.dataset, {workerMode:s.mode, workerStep:String(s.index), phase:s.phase, pages:String(s.pages), ipc:String(s.ipc), pass:String(s.pass)});
      text("[data-worker-purpose]", this.model.purpose);
      node.querySelectorAll("[data-worker-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.workerMode === s.mode)));
      node.querySelectorAll("[data-worker-phase]").forEach(el => el.dataset.current = String(el.dataset.workerPhase === s.phase));
      const active = {drain:["ipc", "sets"], select:["batch", "kv"], execute:["pipeline", "runtime", "gpu"], advance:["results", "kv"], publish:s.results.length ? ["results", "response"] : ["results"], idle:["loop"]}[s.phase] || [];
      node.querySelectorAll("[data-worker-component]").forEach(el => el.dataset.active = String(active.includes(el.dataset.workerComponent)));
      text("[data-worker-loop]", s.phase === "execute" ? "Inside pipeline.execute() · this pass keeps its selected members" : s.phase === "idle" ? "NO_PROGRESS → await backoff → check again" : "One loop thread · " + ({ready:"ready for the next iteration", drain:"pull admitted contexts", select:"choose eligible work", advance:"update request and cache ownership", publish:"send results; then repeat"}[s.phase] || s.phase));
      text("[data-worker-ipc-count]", s.ipc + " / 2 used");
      text("[data-worker-set-count]", "CE " + s.ce + " · TG " + s.requests.filter(r => r.state === "decode").length + " · M = 2");
      text("[data-worker-budget]", (["select", "execute", "advance", "publish"].includes(s.phase) ? s.batch.reduce((n, op) => n + op.tokens, 0) : 0) + " / 8 input tokens");
      text("[data-worker-pipeline]", s.phase === "execute" ? "execute pass " + s.pass : ["advance", "publish"].includes(s.phase) ? "pass " + s.pass + " returned" : "awaiting a batch");
      text("[data-worker-gpu]", s.phase === "execute" ? "pass " + s.pass + " running" : "between passes");
      text("[data-worker-page-count]", s.pages + " / " + s.capacity.pages + " owned · pool retained");
      text("[data-worker-progress]", (s.finished ? "Scene complete" : "Pass " + s.pass) + " · event " + s.index);
      const event = node.querySelector("[data-current-event]");
      event.textContent = s.event.text; event.dataset.kind = s.event.kind;
      node.querySelector('[data-worker-action="step"]').disabled = s.finished;
      node.querySelector("[data-worker-transcript]").innerHTML = this.model.log.map(e => '<li>' + this.escape(e.text) + '</li>').join("");
      const selected = ["select", "execute", "advance", "publish"].includes(s.phase) ? s.batch : [];
      const selectedIDs = new Set(selected.map(op => op.id));
      const inResults = r => ["done", "streaming"].includes(r.state) || (selectedIDs.has(r.id) && ["advance", "publish"].includes(s.phase));
      const waiting = s.requests.filter(r => r.state !== "ipc" && !selectedIDs.has(r.id) && !inResults(r));
      const queued = s.requests.filter(r => r.state === "ipc");
      this.rows("[data-worker-ipc-slots]", [0, 1].map(i => ({x:72+i*62, y:234, width:52, occupied:!!queued[i], kind:"ipc", label:queued[i] ? "" : "empty"})));
      this.rows("[data-worker-set-slots]", [0, 1].map(i => ({x:332, y:229+i*40, width:158, label:waiting[i] ? (waiting[i].promptDone < waiting[i].prompt ? "CE " + waiting[i].promptDone + "/" + waiting[i].prompt : "TG ready") : "empty", labelX:waiting[i] ? 28 : 0})));
      this.rows("[data-worker-batch-slots]", [0, 1].map(i => ({x:562, y:229+i*40, width:198, label:selected[i] ? (selected[i].kind === "prefill" ? "P" : "D") + " × " + selected[i].tokens : "free slot", labelX:selected[i] ? 27 : 0})));
      this.rows("[data-worker-gpu-work]", [0, 1].map(i => ({x:1134, y:390+i*43, width:146, label:s.phase === "execute" && selected[i] ? selected[i].id + " · " + (selected[i].kind === "prefill" ? "P" : "D") + " × " + selected[i].tokens : "—"})));
      const owners = s.requests.flatMap(r => Array(r.pages).fill(r.id));
      this.rows("[data-worker-pages]", Array.from({length:s.capacity.pages}, (_, i) => ({x:362+(i-(s.capacity.pages-1)/2)*49, y:461, width:43, label:owners[i] || "free", kind:"page", occupied:!!owners[i], owner:owners[i] || ""})));
      for (const [i, id] of ["R1", "R2"].entries()) {
        const r = s.requests.find(r => r.id === id);
        const result = s.results.find(result => result.id === id);
        text('[data-worker-published="' + id + '"]', result ? id + ": t" + result.token + (result.done ? " · done" : "") : "—");
        text('[data-worker-result="' + id + '"]', r ? (inResults(r) ? "" : id + ": ") + r.generated + "/" + r.output + " tokens · " + (r.pages ? r.pages + " KV" : r.generated === r.output ? "released" : "no KV yet") : "no request");
        let state = "no request";
        if (r) state = r.state === "ipc" ? "admitted · waiting in IPC" : r.state === "pending" ? (/KV pages/.test(r.reason) ? "waiting for reusable KV blocks" : "waiting for batch selection") : r.state === "done" ? "complete · KV ownership released" : r.state === "streaming" ? "generation done · result returning" : r.promptDone < r.prompt ? "prefill " + r.promptDone + "/" + r.prompt + " · output " + r.generated : "decode · " + r.generated + "/" + r.output + " tokens";
        text('[data-worker-state="' + id + '"]', state);
        if (!r) continue;
        const point = r.state === "ipc" ? {x:72+queued.findIndex(q => q.id === id)*62, y:234}
          : inResults(r) ? {x:551+i*221, y:466}
          : selectedIDs.has(id) ? {x:491, y:229+selected.findIndex(op => op.id === id)*40}
          : {x:280, y:229+waiting.findIndex(q => q.id === id)*40};
        const color = r.state === "ipc" || r.state === "pending" ? this.colors.ipc : ["streaming", "done"].includes(r.state) ? this.colors.done : r.promptDone < r.prompt ? this.colors.prefill : this.colors.decode;
        this.moveRequest(r, point, color, snap);
      }
      if (!snap && s.event.kind === "publish" && this.publishedStep !== s.index) this.publishTokens(s);
      this.publishedStep = s.index;
      this.animate(performance.now());
      if (!snap) this.startFrames();
    }
    rows(selector, rows) {
      const layer = this.node.querySelector(selector);
      layer.replaceChildren();
      for (const row of rows) {
        const attrs = {class:"ss-worker-slot", transform:"translate(" + row.x + " " + row.y + ")", "data-occupied":String(!!row.occupied)};
        if (row.kind) attrs["data-worker-" + row.kind + "-slot"] = "";
        if (row.owner !== undefined) attrs["data-owner"] = row.owner;
        const g = this.element("g", attrs);
        g.append(this.element("rect", {x:-row.width/2, y:-15, width:row.width, height:30, rx:5}), this.element("text", {x:row.labelX || 0, y:4}, row.label));
        layer.append(g);
      }
    }
    moveRequest(request, point, color, snap) {
      const now = performance.now();
      let dot = this.dots.get(request.id);
      if (!dot) {
        const g = this.element("g", {class:"ss-request-dot ss-worker-request", "data-worker-request":request.id});
        g.append(this.element("title"), this.element("rect", {x:-21, y:-12, width:42, height:24, rx:12}), this.element("text", {y:4}, request.id));
        this.node.querySelector("[data-worker-dots]").append(g);
        dot = {g, x:point.x, y:point.y, tx:point.x, ty:point.y, at:now};
        this.dots.set(request.id, dot);
      }
      const p = this.interpolate(dot, now);
      Object.assign(dot, {x:p.x, y:p.y, tx:point.x, ty:point.y, at:snap ? now-1000 : now});
      dot.g.querySelector("title").textContent = request.id + ": " + request.reason;
      dot.g.querySelector("rect").setAttribute("fill", color[0]);
      dot.g.querySelector("rect").setAttribute("stroke", color[1]);
      dot.g.querySelector("text").setAttribute("fill", color[2]);
    }
    publishTokens(s) {
      for (const [i, result] of s.results.entries()) {
        const g = this.element("g", {class:"ss-worker-token"});
        g.append(this.element("rect", {x:-26, y:-10, width:52, height:20, rx:8}), this.element("text", {y:4}, result.id + ":t" + result.token));
        this.node.querySelector("[data-worker-token-dots]").append(g);
        this.tokens.push({g, at:performance.now()+i*90, points:[{x:741,y:520},{x:741,y:540},{x:102,y:540},{x:102,y:505}]});
      }
    }
    interpolate(dot, now) {
      const p = this.reduced.matches ? 1 : Math.min(1, Math.max(0, (now-dot.at)/650));
      const t = p*p*(3-2*p);
      return {x:dot.x+(dot.tx-dot.x)*t, y:dot.y+(dot.ty-dot.y)*t, done:p===1};
    }
    animate(now) {
      let moving = false;
      for (const dot of this.dots.values()) {
        const p = this.interpolate(dot, now);
        dot.g.setAttribute("transform", "translate(" + p.x.toFixed(1) + "," + p.y.toFixed(1) + ")");
        moving ||= !p.done;
      }
      this.tokens = this.tokens.filter(token => {
        const progress = this.reduced.matches ? 1 : Math.max(0, (now-token.at)/1050);
        if (progress >= 1) { token.g.remove(); return false; }
        const lengths = token.points.slice(1).map((p, i) => Math.hypot(p.x-token.points[i].x, p.y-token.points[i].y));
        let distance = progress * lengths.reduce((a, b) => a+b, 0), index = 0;
        while (index < lengths.length-1 && distance > lengths[index]) distance -= lengths[index++];
        const a = token.points[index], b = token.points[index+1], t = distance/lengths[index];
        token.g.setAttribute("transform", "translate(" + (a.x+(b.x-a.x)*t) + "," + (a.y+(b.y-a.y)*t) + ")");
        moving = true; return true;
      });
      return moving;
    }
  };
})();
