/* Presentation for the batch-constructor lesson; uses the shared frame loop. */
(function () {
  "use strict";
  window.ServeBatchView = class BatchStory {
    constructor(node, {reduced, startFrames, escape, element}) {
      Object.assign(this, {node, reduced, startFrames, escape, element});
      this.key = "batch"; this.mode = "mixed"; this.interval = 1200; this.playing = false;
      this.cursor = null;
      node.querySelectorAll("[data-batch-controls]").forEach(el => el.hidden = false);
      node.addEventListener("click", e => {
        const mode = e.target.closest("button[data-batch-mode]");
        if (mode) { this.mode = mode.dataset.batchMode; this.reset(); return; }
        const action = e.target.closest("[data-batch-action]")?.dataset.batchAction;
        if (action === "reset") this.reset();
        if (action === "step") { this.setPlaying(false); this.step(true); }
        if (action === "play") {
          if (this.model.finished) this.reset();
          this.setPlaying(!this.playing);
        }
      });
      node.addEventListener("change", e => { if (e.target.matches("[data-batch-option]")) this.reset(); });
      this.reset();
    }
    reset() {
      const option = key => this.node.querySelector('[data-batch-option="' + key + '"]').value;
      this.model = new window.ServeBatch.BatchConstructor(this.mode, {
        policy:option("policy"), budget:Number(option("budget")), chunking:option("chunking") === "on",
      });
      this.cursor = null;
      this.node.querySelector("[data-batch-focus]").replaceChildren();
      this.setPlaying(false); this.render(true);
    }
    setPlaying(value) {
      this.playing = value; this.node.dataset.playing = String(value);
      const button = this.node.querySelector('[data-batch-action="play"]');
      button.textContent = value ? "Ⅱ Pause" : this.model.finished ? "▶ Replay" : "▶ Play";
      button.setAttribute("aria-pressed", String(value));
      this.node.querySelector("[data-current-event]").setAttribute("aria-live", value ? "off" : "polite");
      this.nextAt = performance.now() + this.interval;
      if (value) this.startFrames();
    }
    step(snap = false) {
      this.model.next(); this.render(snap || this.reduced.matches);
      if (this.model.finished) this.setPlaying(false);
    }
    text(selector, value) { this.node.querySelector(selector).textContent = value; }
    render(snap = false) {
      const s = this.model.snapshot(), node = this.node;
      Object.assign(node.dataset, {batchMode:s.mode, batchStep:String(s.index), phase:s.phase,
        pass:String(s.pass), activeTokens:String(s.used), kvPages:String(s.pages)});
      this.text("[data-batch-purpose]", this.model.purpose);
      node.querySelectorAll("[data-batch-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.batchMode === s.mode)));
      const phase = ["complete", "fatal", "idle"].includes(s.phase) ? s.phase === "complete" ? "advance" : "readiness" : s.phase;
      node.querySelectorAll("[data-batch-phase]").forEach(el => el.dataset.current = String(el.dataset.batchPhase === phase));
      const active = {readiness:["readiness"], policy:["policy"], decode:["sets", "checks", "batch"],
        prefill:["sets", "checks", "batch"], pack:["batch"], execute:["execute"], advance:["sets", "advance"],
        complete:["sets", "advance"], fatal:["checks"], idle:["readiness"]}[s.phase] || [];
      node.querySelectorAll("[data-batch-component]").forEach(el => el.dataset.active = String(active.includes(el.dataset.batchComponent)));
      const held = s.requests.filter(r => r.phase === "HOLD");
      this.text("[data-batch-readiness]", held.length ? held.map(r => r.id + " · " + r.hold + " pending").join("    |    ") : "No held work · one Python object on the worker's loop");
      this.text("[data-batch-priority]", s.priority === "NONE" ? "no runnable work" : s.priority === "not chosen" ? "not chosen" : s.priority + (s.priority === "TG" && s.config.policy === "mixed" ? " → CE backfill" : " priority"));
      const get = id => s.requests.find(r => r.id === id);
      const ceRows = Array.from({length:3}, (_, i) => {
        const r = get(s.ce[i]);
        return {x:348, y:252+i*40, width:202, occupied:!!r, label:r ? r.id + " · " + r.processed + "/" + r.prompt + " prompt" : "empty", id:r?.id};
      });
      const tgRows = Array.from({length:2}, (_, i) => {
        const r = get(s.tg[i]);
        return {x:348, y:410+i*40, width:202, occupied:!!r,
          label:r ? r.id + " · " + r.emitted + "/" + r.target + " out" + (s.batch.some(op => op.id === r.id) ? " · in pass" : "") : "empty", id:r?.id};
      });
      this.rows("[data-batch-ce-rows]", ceRows, "state");
      this.rows("[data-batch-tg-rows]", tgRows, "state");
      const layer = node.querySelector("[data-batch-rows]"); layer.replaceChildren();
      for (let i = 0; i < s.config.slots; i++) {
        const op = s.batch[i], y = 240 + i*69;
        const g = this.element("g", {class:"ss-batch-input-row", "data-input-request":op?.id || "", "data-input-kind":op?.kind || "",
          "data-occupied":String(!!op), transform:"translate(759 " + y + ")"});
        g.append(this.element("rect", {width:222, height:58, y:-24, rx:6}));
        g.append(this.element("text", {x:111, y:-7}, op ? op.id + " · " + (op.kind === "CE" ? "P [" + op.start + ":" + (op.start+op.tokens) + "]" : "D × 1") : "free input row"));
        if (op) {
          const cellWidth = Math.min(18, 200/op.tokens);
          for (let j = 0; j < op.tokens; j++) {
            const x = 11 + j*cellWidth;
            const cell = this.element("g", {class:"ss-batch-token-cell", "data-input-token":op.kind, transform:"translate(" + x + " 4)"});
            cell.append(this.element("rect", {width:cellWidth-2, height:18, rx:3}),
              this.element("title", {}, op.id + " input " + (op.start+j)),
              this.element("text", {x:(cellWidth-2)/2, y:13}, op.kind === "CE" ? "P" : "D"));
            g.append(cell);
          }
        }
        layer.append(g);
      }
      const returned = ["pack", "execute", "advance", "complete"].includes(s.phase);
      this.text("[data-batch-inputs]", returned ? "batches = [[" + s.inputs.map(op => op.id).join(", ") + "]]" : "inputs not returned yet");
      this.text("[data-batch-execution]", s.phase === "execute" ? "pass " + s.pass + " running" : ["advance", "complete"].includes(s.phase) ? "pass " + s.pass + " returned" : "awaiting inputs");
      this.text('[data-batch-meter="seats"]', s.batch.length + " / " + s.config.slots + " selected · TG " + s.tg.length);
      this.text('[data-batch-meter="tokens"]', s.used + " / " + s.config.budget + (s.used > s.config.budget ? " · soft target exceeded" : " input tokens"));
      this.text('[data-batch-meter="kv"]', s.pages + " / " + s.config.pages + " owned · pool retained");
      this.text('[data-batch-meter="context"]', s.config.total === null ? "optional budget · OFF" : s.totalUsed + " / " + s.config.total + " page-aligned");
      this.cells("seats", s.batch.map(op => op.id), s.config.slots, 40);
      this.cells("tokens", s.batch.flatMap(op => Array(op.tokens).fill(op.kind === "CE" ? "P" : "D")), Math.max(s.config.budget, s.used), 22);
      this.cells("kv", s.requests.flatMap(r => Array(r.pages).fill(r.id)), s.config.pages, 22);
      this.cells("context", s.config.total === null ? [] : Array(Math.min(s.totalUsed, 12)).fill("·"), s.config.total === null ? 0 : Math.min(Math.max(s.config.total, s.totalUsed), 12), 22);
      const checks = node.querySelector("[data-batch-checks]"); checks.replaceChildren();
      for (const [i, [key, label]] of [["seats", "Request seat"], ["tokens", "Active tokens"], ["kv", "KV blocks"], ["context", "Context budget"]].entries()) {
        const status = s.checks[key] || (key === "context" && s.config.total === null ? "off" : "pending");
        const g = this.element("g", {class:"ss-batch-check", "data-check":key, "data-check-status":status, transform:"translate(506 " + (372+i*31) + ")"});
        g.append(this.element("rect", {x:0, y:-17, width:192, height:25, rx:4}),
          this.element("text", {x:12, y:0, "text-anchor":"start"}, (status === "pass" ? "✓ " : status === "wait" ? "… " : status === "off" ? "– " : "· ") + label));
        checks.append(g);
      }
      const results = node.querySelector("[data-batch-results]"); results.replaceChildren();
      s.requests.forEach((r, i) => results.append(this.element("text", {class:"ss-svg-muted", x:1137, y:451+i*24},
        r.id + " · " + r.emitted + " out · " + (r.phase === "DONE" ? "released" : r.phase === "TG" ? "TG" : r.phase === "HOLD" ? "held" : "CE"))));
      this.text("[data-batch-progress]", (s.finished ? "Scene complete" : "Pass " + s.pass) + " · event " + s.index);
      const event = node.querySelector("[data-current-event]"); event.textContent = s.event.text; event.dataset.kind = s.event.kind;
      node.querySelector('[data-batch-action="step"]').disabled = s.finished;
      node.querySelector("[data-batch-transcript]").innerHTML = this.model.log.map(e => "<li>" + this.escape(e.text) + "</li>").join("");
      node.querySelector("[data-batch-history]").innerHTML = s.history.slice(-8).map(h => "<div><strong>Pass " + h.pass + "</strong>" +
        (h.batch.map(op => this.escape(op.id) + " " + (op.kind === "CE" ? "P" : "D") + "×" + op.tokens).join("<br>") || "empty") + "<br>" + h.used + " input tokens</div>").join("");
      node.querySelector("[data-batch-request-states]").innerHTML = s.requests.map(r => "<span><b>" + this.escape(r.id) + " · " + this.escape(r.phase === "SELECTED_CE" ? "selected CE" : r.phase) + "</b><span>" +
        this.escape(r.phase === "HOLD" ? r.hold + " pending" : ["CE", "TG"].includes(r.phase) && /waiting|budget|preempted|pressure/.test(r.reason) ? r.reason : r.emitted + "/" + r.target + " output · " + r.pages + " KV blocks") + "</span></span>").join("");
      this.moveCursor(s, snap); this.animate(performance.now());
      if (!snap) this.startFrames();
    }
    rows(selector, rows, kind) {
      const layer = this.node.querySelector(selector); layer.replaceChildren();
      rows.forEach(row => {
        const g = this.element("g", {class:"ss-worker-slot", "data-batch-row":kind, "data-request":row.id || "",
          "data-occupied":String(row.occupied), transform:"translate(" + row.x + " " + row.y + ")"});
        g.append(this.element("rect", {x:-row.width/2, y:-15, width:row.width, height:30, rx:5}), this.element("text", {y:4}, row.label));
        layer.append(g);
      });
    }
    cells(key, labels, count, width) {
      const layer = this.node.querySelector('[data-batch-cells="' + key + '"]'); layer.replaceChildren();
      const center = 322 + ["seats", "tokens", "kv", "context"].indexOf(key)*196;
      const columns = key === "seats" ? 3 : key === "tokens" ? 7 : 6;
      const cellWidth = key === "seats" ? width : 20;
      for (let i = 0; i < count; i++) {
        const columnsThisRow = Math.min(columns, count-Math.floor(i/columns)*columns);
        const x = center + ((i%columns)-(columnsThisRow-1)/2)*width;
        const y = 592+Math.floor(i/columns)*20;
        const label = labels[i] || "";
        const g = this.element("g", {class:"ss-batch-budget-cell", "data-occupied":String(!!label), "data-budget-owner":label,
          transform:"translate(" + x + " " + y + ")"});
        g.append(this.element("rect", {x:-cellWidth/2, y:-8, width:cellWidth, height:16, rx:3}),
          this.element("title", {}, label || "free"),
          this.element("text", {y:4}, key === "kv" ? label.slice(1) : label));
        layer.append(g);
      }
    }
    moveCursor(s, snap) {
      const layer = this.node.querySelector("[data-batch-focus]");
      const phasePacket = ["pack", "execute", "advance"].includes(s.phase) && s.batch.length;
      const id = s.focus || (phasePacket ? "inputs" : null);
      if (!id) { if (this.cursor) this.cursor.g.style.visibility = "hidden"; return; }
      const now = performance.now();
      let point = {x:602, y:331};
      if (id === "inputs") point = s.phase === "execute" ? {x:1137, y:276} : s.phase === "advance" ? {x:1137,y:512} : {x:870,y:439};
      else if (s.location === "batch") point = {x:963,y:240+s.batch.findIndex(op => op.id === id)*69-7};
      if (!this.cursor) {
        const g = this.element("g", {class:"ss-request-dot ss-batch-cursor", "data-batch-cursor":""});
        g.append(this.element("rect", {x:-27, y:-12, width:54, height:24, rx:12}), this.element("text", {y:4}, id));
        layer.append(g);
        this.cursor = {g, x:348, y:252, tx:348, ty:252, at:now-1000};
      }
      const cursor = this.cursor, previous = this.position(now);
      cursor.g.style.visibility = "visible"; cursor.g.querySelector("text").textContent = id;
      if (cursor.id !== id && id !== "inputs") {
        const r = s.requests.find(r => r.id === id), ce = s.ce.indexOf(id), tg = s.tg.indexOf(id);
        previous.x = 348; previous.y = r.phase === "TG" ? 410+Math.max(0,tg)*40 : 252+Math.max(0,ce)*40;
      }
      Object.assign(cursor, {id, x:previous.x, y:previous.y, tx:point.x, ty:point.y, at:snap ? now-1000 : now});
    }
    position(now) {
      const d = this.cursor, p = this.reduced.matches ? 1 : Math.min(1, Math.max(0, (now-d.at)/650));
      const t = p*p*(3-2*p);
      return {x:d.x+(d.tx-d.x)*t, y:d.y+(d.ty-d.y)*t, done:p===1};
    }
    animate(now) {
      if (!this.cursor || this.cursor.g.style.visibility === "hidden") return false;
      const p = this.position(now);
      this.cursor.g.setAttribute("transform", "translate(" + p.x.toFixed(1) + "," + p.y.toFixed(1) + ")");
      return !p.done;
    }
  };
})();
