/* Presentation for the shared, offline request stories. */
(function () {
  "use strict";
  const {Worker, Fleet, TERMINAL} = window.ServeStory;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const SVG = "http://www.w3.org/2000/svg";
  const STATES = {api:"API / tokenize", ipc:"IPC waiting", pending:"Pending prefill", prefill:"Prefill", decode:"Decode", streaming:"Output draining", done:"Completed", rejected:"HTTP 429", cancel_pending:"Cancel pending", cancelled:"Cancelled"};
  const COLORS = {api:["#edf0e2", "#778662", "#414f33"], ipc:["#f4e8c9", "#a28950", "#674d21"],
    pending:["#f4e8c9", "#a28950", "#674d21"], prefill:["#dfe9f4", "#6a88a0", "#37556d"],
    decode:["#dbedd6", "#6b8d5e", "#34572d"], streaming:["#dceee4", "#6b907c", "#31573f"],
    done:["#e1ecdb", "#829976", "#3b5c30"], rejected:["#f7dfd1", "#bd8267", "#813c22"],
    cancelled:["#ece5df", "#a08e7c", "#64503e"], cancel_pending:["#f2e4d7", "#ad8968", "#6c4d34"]};
  const controllers = [];
  const cards = [];
  let frame = null;

  class FloatingCard {
    constructor(details, hover) {
      this.details = details;
      this.hover = hover;
      this.trigger = details.querySelector(":scope > summary");
      this.card = details.querySelector(":scope > aside");
      this.mode = null;
      this.timer = null;
      this.restoring = false;
      details.open = false;
      this.trigger.setAttribute("aria-expanded", "false");
      this.trigger.addEventListener("pointerdown", e => this.pointer = e.pointerType);
      this.trigger.addEventListener("pointerenter", e => {
        if (hover && e.pointerType === "mouse") this.open("pointer");
      });
      details.addEventListener("pointerenter", () => clearTimeout(this.timer));
      details.addEventListener("pointerleave", e => {
        if (!hover || e.pointerType === "touch") return;
        this.timer = setTimeout(() => {
          if (this.mode === "keyboard" && details.contains(document.activeElement)) return;
          if (!document.querySelector("#source-dialog[open]")) this.close();
        }, 140);
      });
      this.card.addEventListener("pointerenter", () => clearTimeout(this.timer));
      this.trigger.addEventListener("click", e => {
        e.preventDefault();
        if (!hover || (e.detail !== 0 && this.pointer === "touch")) {
          if (details.open) this.close(true);
          else this.open(e.detail === 0 ? "keyboard" : "touch");
        } else this.open(e.detail === 0 ? "keyboard" : "pointer");
        if (e.detail === 0 && details.open) this.card.focus({preventScroll:true});
      });
      details.addEventListener("focusin", e => {
        if (hover && !this.restoring && e.target === this.trigger && (this.pointer !== "touch" || this.trigger.matches(":focus-visible"))) this.open("keyboard");
      });
      details.addEventListener("focusout", () => setTimeout(() => {
        if (!details.contains(document.activeElement) && !document.querySelector("#source-dialog[open]") && (this.mode === "keyboard" || !details.matches(":hover"))) this.close();
      }, 0));
      details.querySelector("[data-card-dismiss]").addEventListener("click", () => this.close(true));
      details.addEventListener("toggle", () => {
        this.trigger.setAttribute("aria-expanded", String(details.open));
        if (details.open) this.position();
      }, true);
    }
    open(mode) {
      clearTimeout(this.timer);
      cards.forEach(c => { if (c !== this) c.close(); });
      this.mode = mode;
      this.details.open = true;
      this.trigger.setAttribute("aria-expanded", "true");
      this.position();
    }
    close(restore = false) {
      clearTimeout(this.timer);
      if (!this.details.open) return;
      restore ||= this.card.contains(document.activeElement);
      this.details.open = false;
      this.trigger.setAttribute("aria-expanded", "false");
      this.mode = null;
      if (restore) {
        this.restoring = true;
        this.trigger.focus({preventScroll:true});
        this.restoring = false;
      }
    }
    position() {
      if (!this.details.open) return;
      const r = this.trigger.getBoundingClientRect();
      const below = innerHeight - r.bottom - 20;
      const above = r.top - 20;
      const down = !this.hover || below >= 260 || below >= above;
      this.card.style.maxHeight = Math.max(100, Math.min(innerHeight - 24, down ? below : above)) + "px";
      const box = this.card.getBoundingClientRect();
      const left = this.hover ? r.left : r.right - box.width;
      const top = down ? r.bottom + 8 : r.top - box.height - 8;
      this.card.style.left = Math.max(12, Math.min(innerWidth - box.width - 12, left)) + "px";
      this.card.style.top = Math.max(12, Math.min(innerHeight - box.height - 12, top)) + "px";
      this.card.style.right = "auto";
    }
  }

  function slotPoint(index, capacity, fleet = false) {
    return fleet ? {x:638 + index*52, y:195} : {
      x:486 + (index%2)*52,
      y:196 + (3-Math.ceil(capacity/2))*15.5 + Math.floor(index/2)*31,
    };
  }
  function renderSlots(layer, worker, fleet = false) {
    const waiting = worker.at("ipc");
    if (layer.children.length !== worker.config.queue) {
      layer.replaceChildren();
      for (let i = 0; i < worker.config.queue; i++) {
        const p = slotPoint(i, worker.config.queue, fleet);
        const slot = element("g", {class:"ss-message-slot", "data-ipc-slot":i, transform:"translate(" + p.x + " " + p.y + ")"});
        slot.append(element("rect", {x:-23,y:-15,width:46,height:30,rx:5}), element("text", {y:4}, i+1));
        layer.append(slot);
      }
    }
    for (const [i, slot] of Array.from(layer.children).entries()) {
      slot.dataset.occupied = String(Boolean(waiting[i]));
      slot.dataset.request = waiting[i]?.id || "";
      slot.setAttribute("aria-label", "Slot " + (i+1) + ": " + (waiting[i]?.id || "empty"));
    }
    layer.setAttribute("aria-label", waiting.length + " of " + worker.config.queue + " request message slots occupied");
  }

  function element(tag, attrs = {}, text = null) {
    const el = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== null) el.textContent = text;
    return el;
  }
  function modelFor(key, value) {
    const config = {};
    if (key === "busy") config.batch = value || 1;
    if (key === "pressure") Object.assign(config, {batch:1, queue:value || 2, output:9});
    if (key === "scheduler") Object.assign(config, {pending:4, budget:value || 8});
    if (key === "reader") config.readEvery = value || 12;
    const model = key === "replicas" ? new Fleet() : new Worker(config);
    if (key === "replicas") model.submit({output:7});
    else if (["busy", "pressure", "clocks"].includes(key)) {
      model.seedDecoder({output:key === "pressure" ? 9 : 7});
      while (!model.busy) model.next();
    } else if (key === "scheduler") {
      model.seedDecoder({output:7}); model.submit({prompt:24, output:3});
    } else model.submit();
    return model;
  }
  function selectedState(r) {
    if (r.cancellation && r.state !== "cancelled") return "cancel_pending";
    return r.state;
  }
  class Story {
    constructor(node) {
      this.node = node;
      this.key = node.dataset.serveStory;
      this.playing = false;
      this.selected = null;
      this.dots = new Map();
      this.tokenDots = [];
      this.delivered = new Map();
      this.nextAt = 0;
      this.option = node.querySelector("[data-ss-option]");
      this.node.querySelector("[data-ss-controls]").hidden = false;
      this.node.addEventListener("click", e => {
        const button = e.target.closest("[data-ss-action]");
        if (button) this.action(button.dataset.ssAction);
        const request = e.target.closest("[data-ss-request]");
        if (request) { this.selected = request.dataset.ssRequest; this.render(); }
      });
      this.node.addEventListener("keydown", e => {
        const request = e.target.closest("svg [data-ss-request]");
        if (request && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault(); this.selected = request.dataset.ssRequest; this.render();
        }
      });
      this.option?.addEventListener("change", () => this.reset());
      this.reset();
    }
    get workers() { return this.key === "replicas" ? this.model.replicas.map(r => r.worker) : [this.model]; }
    get requests() { return this.workers.flatMap(w => w.requests); }
    reset() {
      this.setPlaying(false);
      this.model = modelFor(this.key, Number(this.option?.value) || undefined);
      this.selected = this.requests[0]?.id || null;
      this.dots.clear(); this.tokenDots = []; this.delivered.clear();
      this.node.querySelector("[data-request-dots]").replaceChildren();
      this.node.querySelector("[data-output-dots]")?.replaceChildren();
      this.render(true);
      this.node.querySelector('[data-ss-action="replica"]')?.removeAttribute("disabled");
    }
    setPlaying(value) {
      this.playing = value;
      this.node.dataset.playing = String(value);
      const button = this.node.querySelector('[data-ss-action="play"]');
      button.textContent = value ? "Ⅱ Pause" : "▶ Play";
      button.setAttribute("aria-pressed", String(value));
      this.node.querySelector("[data-current-event]").setAttribute("aria-live", value ? "off" : "polite");
      this.nextAt = performance.now() + 850;
      if (value) startFrames();
    }
    action(action) {
      if (action === "reset") { this.reset(); return; }
      if (action === "play") { this.setPlaying(!this.playing); return; }
      if (action === "step") { this.setPlaying(false); this.step(); return; }
      if (action === "replica") {
        this.model.addReplica();
        this.node.querySelector('[data-ss-action="replica"]').disabled = true;
      }
      if (action === "send") {
        const count = this.key === "pressure" ? 6 : this.key === "replicas" ? 4 : 1;
        for (let i = 0; i < count; i++) {
          const options = this.key === "scheduler" ? {prompt:8, output:3} : {};
          // Stagger tokenizer completions to expose M filling before N backs up.
          if (this.key === "pressure") options.admissionDelay = 1 + i*4;
          const r = this.model.submit(options);
          this.selected = r.id;
        }
      }
      this.render(); this.setPlaying(true);
    }
    step() {
      this.model.next();
      if (this.key !== "replicas" && this.model.last.kind === "reject") this.selected = this.model.last.id;
      this.render();
      const pendingWarmup = this.key === "replicas" && this.model.replicas.some(r => r.readyAt > this.model.time);
      if (!this.requests.some(r => !TERMINAL.has(r.state)) && !pendingWarmup && this.workers.every(w => w.phase === "idle")) this.setPlaying(false);
    }
    render(snap = false) {
      const now = performance.now();
      const snapshots = this.workers.map(w => w.snapshot());
      const sum = key => snapshots.reduce((n, s) => n + s[key], 0);
      this.node.dataset.time = String(this.model.time);
      this.node.dataset.accepted = String(sum("accepted"));
      this.node.dataset.rejected = String(sum("rejected"));
      this.node.dataset.done = String(sum("done"));
      this.node.dataset.phase = snapshots[0].phase;
      this.node.dataset.ipc = String(sum("ipc"));
      this.node.dataset.pending = String(sum("pending"));
      this.node.dataset.pages = String(sum("pages"));
      this.node.dataset.buffered = String(sum("buffered"));
      const meter = this.node.querySelector("[data-meters]");
      const pairs = this.key === "reader" ? [["Generated",sum("generated")], ["Delivered",sum("delivered")], ["Buffered",sum("buffered")], ["KV pages owned",sum("pages")]] : [["Accepted",sum("accepted")], ["Waiting · IPC + pending",sum("ipc") + sum("pending")], ["Active sequences",sum("active")], ["Rejected · 429",sum("rejected")]];
      meter.innerHTML = pairs.map(([label, count], i) => '<span' + (i === 3 && this.key !== "reader" ? ' class="ss-negative"' : '') + '>' + label + '<b>' + count + '</b></span>').join("");
      const current = this.node.querySelector("[data-current-event]");
      const event = this.key === "replicas" ? {text:this.model.last, kind:"fleet"} : this.model.last;
      if (current.textContent !== event.text) current.textContent = event.text;
      current.dataset.kind = event.kind;
      if (this.key === "replicas") this.renderFleet();
      else {
        const s = snapshots[0], c = this.model.config;
        for (const [key, count] of Object.entries({api:this.model.at("api").length, ipc:s.ipc + " / " + c.queue,
          pending:s.pending + " waiting", batch:s.active + " / " + c.batch})) this.node.querySelector('[data-count="' + key + '"]').textContent = count;
        this.node.querySelector('[data-cap="pending"]').textContent = "M " + c.pending + " · CE " + s.ce;
        this.node.querySelector('[data-cap="batch"]').textContent = s.pages + " / " + c.pages + " KV pages";
        if (this.key === "busy") this.node.querySelector('[data-ss-action="send"]').textContent = "Send R" + (this.model.sequence+1);
        const gate = this.node.querySelector("[data-gate]");
        gate.dataset.gate = s.writable ? "open" : "closed";
        gate.querySelector("text").textContent = s.writable ? "✓" : "×";
        this.node.querySelector("[data-gpu]").textContent = this.model.busy ? "Pass " + s.pass : "Idle";
        this.node.querySelector(".ss-gpu").dataset.busy = String(this.model.busy);
        renderSlots(this.node.querySelector("[data-ipc-slots]"), this.model);
        this.node.querySelectorAll("[data-phase]").forEach(el => el.dataset.current = String(el.dataset.phase === s.phase));
        this.renderLanes(s);
        this.renderHistory();
      }
      const requestKey = this.requests.slice(-24).map(r => r.id + ":" + r.state).join("|");
      const strip = this.node.querySelector("[data-request-list]");
      if (requestKey !== this.requestKey) {
        strip.innerHTML = this.requests.slice(-24).map(r => '<button data-ss-request="' + r.id + '" data-terminal="' + r.state + '" aria-label="Inspect ' + r.id + ', ' + STATES[r.state] + '">' + r.id + '</button>').join("");
        this.requestKey = requestKey;
      }
      strip.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.ssRequest === this.selected)));
      this.renderInspector();
      const events = this.workers.flatMap(w => w.log).sort((a, b) => a.at - b.at).slice(-16);
      this.node.querySelector("[data-transcript]").innerHTML = events.map(e => '<li>' + escape(e.text) + '</li>').join("");
      this.updateDots(now, snap || reduced.matches);
      if (!reduced.matches) startFrames();
    }
    renderLanes(s) {
      const box = this.node.querySelector("[data-owner-lanes]");
      if (!box) return;
      const r = this.requests.find(r => r.id === this.selected) || this.requests[0];
      const lanes = this.key === "reader" ? [
        ["Generation", s.generated + " tokens produced; " + s.pages + " KV pages still owned by requests"],
        ["Output path", s.buffered + " tokens buffered across response IPC, API and transport (grouped)"],
        ["Client", s.delivered + " tokens consumed; " + (this.model.config.readEvery > 1 ? "slow reader" : "normal reader")],
      ] : [
        ["API process", this.model.at("api").length ? "Tokenization work can proceed while the worker's pass runs." : (s.buffered ? "Output available: resume streaming tasks." : "Awaiting socket readiness, tokenizer work or response IPC.")],
        ["MAX worker", this.model.busy ? "Inside pipeline.execute(); the current batch is already selected." : (s.phase === "idle" ? "No eligible work: yield with backoff, then check again." : "Running scheduler phase: " + s.phase + ".")],
        ["GPU / runtime", this.model.busy ? "Executing the selected pass; completion is managed by the native runtime." : "Between model passes in this ordinary-pipeline illustration."],
        ["OS threads", "Linux gives CPU time to runnable threads. " + r.id + "'s request state (" + STATES[r.state] + ") is a separate lifecycle."],
      ];
      box.innerHTML = lanes.map(([owner, meaning]) => '<div><b>' + owner + '</b><span>' + meaning + '</span></div>').join("");
    }
    renderHistory() {
      const box = this.node.querySelector("[data-batch-history]");
      if (!box) return;
      box.innerHTML = this.model.history.slice(-5).map(h => '<div><strong>Pass ' + h.pass + (h.finished ? ' ✓' : '') + '</strong>' + h.ops.map(op => escape(op.id) + ' ' + (op.kind === "decode" ? "D" : "P") + ' ×' + op.tokens).join('<br>') + '</div>').join("");
    }
    renderInspector() {
      const r = this.requests.find(r => r.id === this.selected) || this.requests[0];
      if (!r) return;
      const replica = this.key === "replicas" ? ' · replica ' + this.model.replicas.find(rep => rep.worker.find(r.id)).name : '';
      const state = selectedState(r);
      const trail = ["api", "ipc", "pending", "prefill", "decode", "streaming", "done"];
      if (["rejected", "cancelled", "cancel_pending"].includes(state)) trail.push(state);
      const link = document.getElementById("machine-request") ? '<a href="#machine-request">Explore the request state machine ↓</a>' : '';
      this.node.querySelector("[data-request-detail]").innerHTML = '<strong>' + escape(r.id) + replica + ' · ' + STATES[state] + '</strong><div class="ss-state-trail">' + trail.map(s => '<span data-current="' + (s === state) + '">' + STATES[s] + '</span>').join("") + '</div><p>' + escape(r.reason) + '</p><p>Prompt ' + r.promptDone + '/' + r.prompt + ' · Output ' + r.generated + '/' + r.output + ' · Delivered ' + r.delivered + ' · KV pages ' + r.pages + '</p>' + link;
    }
    renderFleet() {
      for (let i = 0; i < 2; i++) {
        const rep = this.model.replicas[i];
        const status = this.node.querySelector('[data-fleet-state="' + i + '"]');
        const detail = this.node.querySelector('[data-fleet-detail="' + i + '"]');
        detail.dataset.fleetPresent = String(Boolean(rep));
        if (!rep) {
          status.textContent = "Not added · use the button below";
          for (const field of ["api", "ipc", "pending", "batch", "gpu"]) detail.querySelector('[data-fleet-' + field + ']').textContent = "—";
          detail.querySelector("[data-ipc-slots]").setAttribute("aria-label", "Replica not added: no request message queue");
          detail.querySelectorAll("[data-ipc-slot]").forEach(slot => { slot.dataset.occupied = "false"; slot.dataset.request = ""; slot.setAttribute("aria-label", "Replica not added"); });
          detail.querySelector(".ss-gpu").dataset.busy = "false";
          continue;
        }
        const ready = rep.readyAt <= this.model.time;
        status.textContent = ready ? "Ready · its own worker and GPU" : "Initializing · excluded from routing";
        const s = rep.worker.snapshot();
        const values = {api:rep.worker.at("api").length + " requests", ipc:s.ipc + " / " + rep.worker.config.queue,
          pending:"Pending · " + s.pending, batch:"Batch · " + s.active, gpu:rep.worker.busy ? "Pass " + s.pass : "Idle"};
        for (const [key, value] of Object.entries(values)) detail.querySelector('[data-fleet-' + key + ']').textContent = ready ? value : "Warming";
        detail.querySelector(".ss-gpu").dataset.busy = String(rep.worker.busy);
        renderSlots(detail.querySelector("[data-ipc-slots]"), rep.worker, true);
      }
    }
    positions() {
      const positions = new Map();
      if (this.key === "replicas") {
        for (let i = 0; i < this.model.replicas.length; i++) {
          const w = this.model.replicas[i].worker;
          w.at("api").slice(-4).forEach((r, j) => positions.set(r.id, {x:428+j*47,y:208+i*252}));
          w.at("ipc").forEach((r,j) => { const p=slotPoint(j,w.config.queue,true); positions.set(r.id,{x:p.x,y:p.y+i*252}); });
          for (const [x, states] of [[790,["pending"]], [893,["prefill","decode","streaming","cancel_pending"]]]) {
            w.requests.filter(r => states.includes(r.state)).slice(-4).forEach((r, j) => positions.set(r.id, {x:x-23 + (j%2)*46, y:193 + i*252 + Math.floor(j/2)*28}));
          }
          w.requests.filter(r => TERMINAL.has(r.state)).slice(-3).forEach((r, j) => positions.set(r.id, {x:166 + j*47, y:105 + i*252}));
        }
        return positions;
      }
      const groups = [
        [["api"],195], [["pending"],686], [["prefill","decode","cancel_pending"],836],
      ];
      for (const [states, x] of groups) {
        this.requests.filter(r => states.includes(r.state)).slice(-6).forEach((r, i) => positions.set(r.id, {x:x-25 + (i%2)*50, y:195 + Math.floor(i/2)*31}));
      }
      this.model.at("ipc").forEach((r,i) => positions.set(r.id, slotPoint(i,this.model.config.queue)));
      this.requests.filter(r => r.state === "streaming").slice(-5).forEach((r, i) => positions.set(r.id, {x:940-i*50, y:418}));
      this.requests.filter(r => ["done","cancelled"].includes(r.state)).slice(-3).forEach((r, i) => positions.set(r.id, {x:175+i*50, y:418}));
      this.requests.filter(r => r.state === "rejected").slice(-5).forEach((r, i) => positions.set(r.id, {x:429+i*50, y:418}));
      return positions;
    }
    updateDots(now, snap) {
      const positions = this.positions();
      const layer = this.node.querySelector("[data-request-dots]");
      for (const [id, point] of positions) {
        const r = this.requests.find(r => r.id === id);
        let dot = this.dots.get(id);
        if (!dot) {
          const g = element("g", {class:"ss-request-dot", "data-dot-id":id, "data-ss-request":id, role:"button", tabindex:"0"});
          g.append(element("rect", {x:-21, y:-12, width:42, height:24, rx:12}), element("text", {y:4}, id));
          layer.append(g);
          dot = {g, x:snap ? point.x : 44, y:snap ? point.y : (this.key === "replicas" ? 266 : 223), tx:point.x, ty:point.y, at:now};
          this.dots.set(id, dot);
        }
        if (dot.tx !== point.x || dot.ty !== point.y) { const current = this.interpolate(dot, now); dot.x = current.x; dot.y = current.y; dot.tx = point.x; dot.ty = point.y; dot.at = now; }
        if (snap) { dot.x = point.x; dot.y = point.y; dot.at = now-1000; }
        const colors = COLORS[r.state];
        const rect = dot.g.querySelector("rect");
        rect.setAttribute("fill",colors[0]); rect.setAttribute("stroke",colors[1]);
        dot.g.querySelector("text").setAttribute("fill",colors[2]);
        dot.g.dataset.selected = String(id === this.selected);
        dot.g.setAttribute("aria-label", "Inspect " + id + ": " + STATES[r.state]);
        dot.g.setAttribute("aria-pressed", String(id === this.selected));
        dot.g.setAttribute("transform", 'translate(' + dot.x + ',' + dot.y + ')');
        const previous = this.delivered.get(id) || 0;
        if (r.delivered > previous && !snap && this.key !== "replicas") {
          const g = element("rect", {width:8,height:8,rx:2,fill:"#426f49"});
          this.node.querySelector("[data-output-dots]").append(g);
          this.tokenDots.push({g, at:now});
        }
        this.delivered.set(id, r.delivered);
      }
      for (const [id, dot] of this.dots) if (!positions.has(id)) { dot.g.remove(); this.dots.delete(id); }
      if (snap) this.animate(now+1000);
    }
    interpolate(dot, now) {
      const p = reduced.matches ? 1 : Math.min(1, Math.max(0, (now-dot.at)/650));
      const t = p*p*(3-2*p);
      return {x:dot.x + (dot.tx-dot.x)*t, y:dot.y + (dot.ty-dot.y)*t, done:p === 1};
    }
    animate(now) {
      let moving = false;
      for (const dot of this.dots.values()) {
        const p = this.interpolate(dot, now);
        dot.g.setAttribute("transform", 'translate(' + p.x.toFixed(1) + ',' + p.y.toFixed(1) + ')');
        moving ||= !p.done;
      }
      for (const token of this.tokenDots) {
        const p = Math.min(1,(now-token.at)/800);
        token.g.setAttribute("x",String(980-936*p)); token.g.setAttribute("y","346");
        if (p >= 1 || reduced.matches) token.g.remove(); else moving = true;
      }
      this.tokenDots = this.tokenDots.filter(token => token.g.isConnected);
      return moving;
    }
  }
  class ApiStory {
    constructor(node) {
      this.node = node;
      this.key = "api";
      this.mode = "normal";
      this.playing = false;
      this.nextAt = 0;
      this.dots = new Map();
      node.querySelectorAll("[data-api-controls]").forEach(el => el.hidden = false);
      node.addEventListener("click", e => {
        const mode = e.target.closest("[data-api-mode]");
        if (mode) { this.mode = mode.dataset.apiMode; this.reset(); return; }
        const action = e.target.closest("[data-api-action]")?.dataset.apiAction;
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
      this.model = new window.ServeApi.ApiProcess(this.mode);
      this.setPlaying(false);
      this.render(true);
    }
    setPlaying(value) {
      this.playing = value;
      this.node.dataset.playing = String(value);
      const button = this.node.querySelector('[data-api-action="play"]');
      button.textContent = value ? "Ⅱ Pause" : this.model.finished ? "▶ Replay" : "▶ Play";
      button.setAttribute("aria-pressed", String(value));
      this.node.querySelector("[data-current-event]").setAttribute("aria-live", value ? "off" : "polite");
      this.nextAt = performance.now() + 1300;
      if (value) startFrames();
    }
    step(snap = false) {
      this.model.next();
      this.render(snap || reduced.matches);
      if (this.model.finished) this.setPlaying(false);
    }
    render(snap = false) {
      const m = this.model, node = this.node;
      node.dataset.apiScenario = m.mode;
      node.dataset.apiStep = String(m.index);
      node.querySelector("[data-api-purpose]").textContent = m.purpose;
      node.querySelectorAll("[data-api-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.apiMode === m.mode)));
      node.querySelector("[data-api-loop-name]").textContent = m.mode === "no-loop" ? "event loop absent" : m.mode === "asyncio" ? "asyncio event loop" : "uvloop event loop";
      node.querySelector("[data-api-loop-task]").textContent = m.loopTask;
      node.querySelectorAll("[data-api-component]").forEach(el => {
        el.dataset.active = String(el.dataset.apiComponent === m.active);
        el.dataset.missing = String(el.dataset.apiComponent === m.missing);
      });
      node.querySelector('[data-api-component-state="uvicorn"]').textContent = m.missing === "uvicorn" ? "ABSENT · no HTTP service" : "receives / writes HTTP";
      node.querySelector('[data-api-component-state="fastapi"]').textContent = m.missing === "fastapi" ? "ABSENT · no MAX ASGI app" : "ASGI application";
      node.querySelector("[data-api-lock]").textContent = m.missing === "admission" ? "ABSENT · no mutual exclusion" : m.lockOwner ? m.lockOwner + " owns lock" : "lock free";
      node.querySelector("[data-api-admission-summary]").textContent = m.missing === "admission" ? "probe / send · tasks can interleave" : "probe → send · one task at a time";
      const probe = id => m.probes[id] === null ? "—" : m.probes[id] ? "✓" : "full";
      node.querySelector("[data-api-probes]").textContent = "R1 probe " + probe("R1") + " · R2 probe " + probe("R2");
      node.querySelector("[data-api-occupancy]").textContent = m.queue.length + " / 1 used";
      const slot = node.querySelector("[data-api-slot]");
      slot.dataset.occupied = String(m.queue.length > 0);
      slot.setAttribute("aria-label", "Request IPC slot: " + (m.queue[0] || "empty"));
      node.querySelector("[data-api-worker]").textContent = m.worker;
      for (const r of m.requests) node.querySelector('[data-api-state="' + r.id + '"]').textContent = r.state;
      const event = node.querySelector("[data-current-event]");
      event.textContent = m.event.text;
      event.dataset.kind = m.event.kind;
      node.querySelector("[data-api-progress]").textContent = (m.finished ? "Scene complete" : m.index ? "Step through the API" : "Ready") + " · event " + m.index;
      node.querySelector('[data-api-action="step"]').disabled = m.finished;
      node.querySelector("[data-api-transcript]").innerHTML = m.log.map(item => '<li>' + escape(item.text) + '</li>').join("");
      this.updateDots(snap);
    }
    point(request, index) {
      const y = index ? 261 : 234;
      return {
        client:{x:53, y:index ? 261 : 219}, server:{x:271, y}, app:{x:523, y},
        executor:{x:index ? 598 : 486, y:459}, probe:{x:737, y}, send:{x:865, y},
        ipc:{x:1128, y:235}, response:{x:1128, y:370},
        "client-error":{x:53, y:442}, "client-output":{x:53, y:370},
      }[request.location];
    }
    updateDots(snap) {
      const now = performance.now();
      this.model.requests.forEach((r, i) => {
        const target = this.point(r, i);
        let dot = this.dots.get(r.id);
        if (!dot) {
          dot = {g:this.node.querySelector('[data-api-request="' + r.id + '"]'), x:target.x, y:target.y, tx:target.x, ty:target.y, at:now};
          this.dots.set(r.id, dot);
        }
        const current = this.interpolate(dot, now);
        Object.assign(dot, {x:current.x, y:current.y, tx:target.x, ty:target.y, at:snap ? now-1000 : now});
        const colors = /429|RequestQueueFull/.test(r.state) ? COLORS.rejected : r.location === "ipc" ? COLORS.ipc : r.location === "client-output" ? COLORS.streaming : COLORS.api;
        dot.g.querySelector("rect").setAttribute("fill", colors[0]);
        dot.g.querySelector("rect").setAttribute("stroke", colors[1]);
        dot.g.querySelector("text").setAttribute("fill", colors[2]);
      });
      this.animate(now);
      if (!snap) startFrames();
    }
    interpolate(dot, now) {
      const p = reduced.matches ? 1 : Math.min(1, Math.max(0, (now-dot.at)/650));
      const t = p*p*(3-2*p);
      return {x:dot.x + (dot.tx-dot.x)*t, y:dot.y + (dot.ty-dot.y)*t, done:p === 1};
    }
    animate(now) {
      let moving = false;
      for (const dot of this.dots.values()) {
        const p = this.interpolate(dot, now);
        dot.g.setAttribute("transform", 'translate(' + p.x.toFixed(1) + ',' + p.y.toFixed(1) + ')');
        moving ||= !p.done;
      }
      return moving;
    }
  }
  function startFrames() { if (frame === null) frame = requestAnimationFrame(tick); }
  function tick(now) {
    frame = null;
    let work = false;
    for (const c of controllers) {
      if (c.playing && now >= c.nextAt) { c.step(); c.nextAt = now + (c.interval || (c.key === "api" ? 1300 : 850)); }
      work = c.animate(now) || c.playing || work;
    }
    if (work) startFrames();
  }
  document.documentElement.classList.add("ss-enhanced");
  document.querySelectorAll("[data-ss-legend], [data-ss-lesson]").forEach(node => cards.push(new FloatingCard(node, node.hasAttribute("data-ss-lesson"))));
  document.addEventListener("pointerdown", e => {
    if (e.target.closest("#source-dialog[open]")) return;
    cards.forEach(c => { if (!c.details.contains(e.target)) c.close(); });
  });
  document.addEventListener("keydown", e => {
    if (e.key !== "Escape" || document.querySelector("#source-dialog[open]")) return;
    const card = cards.find(c => c.details.open);
    if (card) { e.preventDefault(); card.close(true); }
  });
  window.addEventListener("resize", () => cards.forEach(c => c.position()));
  document.addEventListener("scroll", e => {
    if (document.querySelector("#source-dialog[open]") || cards.some(c => c.card.contains(e.target))) return;
    cards.forEach(c => {
      if (c.hover && c.details.open && c.mode === "pointer" && !c.details.matches(":hover")) c.close();
      else c.position();
    });
  }, {capture:true, passive:true});
  document.querySelectorAll(".ss-reference").forEach(node => node.open = false);
  document.querySelectorAll("[data-api-story]").forEach(node => controllers.push(new ApiStory(node)));
  document.querySelectorAll("[data-worker-story]").forEach(node => controllers.push(new window.ServeWorkerView(node, {reduced, startFrames, colors:COLORS, escape, element})));
  document.querySelectorAll("[data-serve-story]").forEach(node => controllers.push(new Story(node)));
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (!entry.isIntersecting) controllers.find(c => c.node === entry.target)?.setPlaying(false);
  });
  controllers.forEach(c => observer.observe(c.node));
  document.addEventListener("visibilitychange", () => { if (document.hidden) controllers.forEach(c => c.setPlaying(false)); });
  window.addEventListener("beforeprint", () => controllers.forEach(c => c.setPlaying(false)));
  reduced.addEventListener("change", () => controllers.forEach(c => c.render(true)));
  // Open enclosing reference disclosures for deep links from the existing course.
  function revealHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target) return;
    for (let el = target; el; el = el.parentElement) if (el instanceof HTMLDetailsElement) el.open = true;
    requestAnimationFrame(() => target.scrollIntoView({block:"start"}));
  }
  window.addEventListener("hashchange", revealHash);
  if (location.hash) revealHash();
  // Small, read-only bridge for browser validation of the actual rendered models.
  window.serveStorySnapshot = key => {
    const c = controllers.find(c => c.key === key);
    return c ? ["api", "worker"].includes(c.key) ? c.model.snapshot() : c.workers.map(w => w.snapshot()) : null;
  };
})();
