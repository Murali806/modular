/* Persistent SVG objects, following the request/slot pattern of serve_worker.js.
 * Motion explains a state transition; it is never a captured runtime trace. */
(function (root) {
  "use strict";
  const ns = "http://www.w3.org/2000/svg";
  const point = (x, y) => ({x, y});
  const clamp = n => Math.max(0, Math.min(1, n));
  const smooth = n => n * n * (3 - 2 * n);
  const shortDevice = d => d === "cpu" ? "CPU" : "GPU " + d.slice(3);
  function element(tag, attrs = {}, text) {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function actor(id, label, x, y, color, extra = {}) {
    return {id, label, x, y, color, opacity: 1, scale: 1, width: 70, ...extra};
  }
  function along(points, t) {
    const lengths = points.slice(1).map((b, i) => Math.hypot(b.x - points[i].x, b.y - points[i].y));
    let distance = lengths.reduce((a, b) => a + b, 0) * t;
    for (let i = 0; i < lengths.length; i++) {
      if (distance <= lengths[i] || i === lengths.length - 1) {
        const k = lengths[i] ? distance / lengths[i] : 1;
        return point(points[i].x + (points[i + 1].x - points[i].x) * k,
          points[i].y + (points[i + 1].y - points[i].y) * k);
      }
      distance -= lengths[i];
    }
    return points[0];
  }

  root.InferenceSessionVisual = class {
    constructor(node, kind, reduced) {
      this.node = node;
      this.kind = kind;
      this.reduced = reduced;
      this.layer = node.querySelector("[data-actors]");
      this.actors = new Map();
      this.frozenAt = null;
      this.started = 0;
      this.duration = 0;
    }
    slots(name, labels, missing = []) {
      const group = this.node.querySelector('[data-slots="' + name + '"]');
      if (!group) return;
      group.dataset.values = JSON.stringify(labels);
      group.querySelectorAll("[data-slot]").forEach((slot, i) => {
        const occupied = labels[i] !== undefined && labels[i] !== null;
        slot.dataset.occupied = String(occupied);
        slot.dataset.missing = String(missing.includes(i));
        slot.dataset.label = occupied ? labels[i] : "";
        slot.setAttribute("aria-label", occupied ? String(labels[i]) : missing.includes(i) ? "missing binding" : "empty slot");
        slot.querySelector("text").textContent = occupied ? "" : missing.includes(i) ? "W_b ?" : "—";
      });
    }
    text(selector, value) {
      const el = this.node.querySelector(selector);
      if (el) el.textContent = value;
    }
    render(frame, animate, speed) {
      const now = performance.now();
      const before = this.frozenAt ?? now;
      this.frozenAt = null;
      this.started = now;
      this.duration = animate && !this.reduced.matches ? 1150 / speed : 0;
      this.frame = frame;
      const targets = this.sceneTargets(frame);
      const ids = new Set(targets.map(a => a.id));
      for (const old of this.actors.values()) {
        if (!ids.has(old.id)) targets.push({...old.target, opacity: 0, scale: .7, via: []});
      }
      for (const target of targets) {
        let item = this.actors.get(target.id);
        let from;
        if (!item) {
          const g = element("g", {class: "session-actor", "data-actor": target.id});
          g.append(element("title"), element("rect", {y: -13, height: 26, rx: 6}), element("text", {y: 4}));
          this.layer.append(g);
          from = {...(target.origin || target), opacity: 1, scale: 1};
          item = {id: target.id, g};
          this.actors.set(target.id, item);
        } else from = this.sample(item, before);
        const travel = Math.hypot(target.x - from.x, target.y - from.y) > 1;
        const changing = travel || Math.abs(from.opacity - target.opacity) > .005 || Math.abs(from.scale - target.scale) > .005;
        Object.assign(item, {from, target, started: now + (target.delay || 0) / speed,
          duration: changing ? this.duration : 0, points: [from, ...(travel ? target.via || [] : []), target]});
        item.g.dataset.color = target.color;
        item.g.dataset.place = target.place || "";
        item.g.dataset.detached = String(Boolean(target.detached));
        item.g.querySelector("rect").setAttribute("width", target.width);
        item.g.querySelector("rect").setAttribute("x", -target.width / 2);
        item.g.querySelector("text").textContent = target.label;
        item.g.querySelector("title").textContent = target.description || target.label;
      }
      this.tick(now);
    }
    sample(item, now) {
      if (!item.target) return item.from;
      const t = item.duration ? clamp((now - item.started) / item.duration) : 1;
      const fade = item.target.merge ? clamp((t - .65) / .35) : smooth(t);
      return {...along(item.points, smooth(t)),
        opacity: item.from.opacity + (item.target.opacity - item.from.opacity) * fade,
        scale: item.from.scale + (item.target.scale - item.from.scale) * smooth(t), done: t === 1};
    }
    tick(now) {
      if (this.frozenAt !== null) return false;
      let moving = false;
      for (const item of this.actors.values()) {
        const p = this.sample(item, now);
        item.g.toggleAttribute("hidden", p.opacity < .005);
        item.g.setAttribute("transform", "translate(" + p.x.toFixed(2) + " " + p.y.toFixed(2) + ") scale(" + p.scale.toFixed(3) + ")");
        item.g.style.opacity = p.opacity;
        item.g.dataset.moving = String(!p.done);
        moving ||= !p.done;
      }
      const work = this.node.querySelector("[data-work-fill]");
      if (work) {
        const active = this.kind === "compilation" ? this.frame.phase === "compile" : this.frame.executing;
        const complete = this.kind === "compilation" && this.frame.artifact && this.frame.mode !== "mef";
        const p = this.duration ? clamp((now - this.started) / this.duration) : 1;
        work.setAttribute("width", (this.kind === "compilation" ? 200 : 154) * (active ? p : complete ? 1 : 0));
        moving ||= active && p < 1;
      }
      return moving;
    }
    pause() {
      if (this.frozenAt !== null || !this.frame) return;
      const now = performance.now();
      this.tick(now);
      this.frozenAt = now;
    }
    resume() {
      if (this.frozenAt === null) return;
      const elapsed = performance.now() - this.frozenAt;
      this.actors.forEach(a => a.started += elapsed);
      this.started += elapsed;
      this.frozenAt = null;
    }

    constructorScene(f) {
      const list = [];
      this.slots("input", f.input.map((d, i) => i < f.processed ? null : shortDevice(d)));
      this.slots("normalized", f.normalized.map(shortDevice));
      this.slots("registered", f.runtimeConfig ? f.normalized.map(shortDevice) : []);
      this.slots("config", f.compileConfig ? ["CFG"] : []);
      this.slots("context", f.context ? ["CTX"] : []);
      f.input.forEach((d, i) => {
        const processed = i < f.processed;
        const duplicate = f.input.indexOf(d) < i;
        const slot = f.normalized.indexOf(d);
        list.push(actor("device-input-" + i, shortDevice(d), processed ? 424 + slot * 110 : 88 + i * 62, 277,
          d === "cpu" ? "weights" : "device", {width: 48,
            opacity: processed && duplicate ? 0 : 1, scale: processed && duplicate ? .55 : 1,
            merge: duplicate, place: processed ? duplicate ? "merged" : "normalized" : "input",
            description: duplicate ? shortDevice(d) + ": duplicate reference merges with its first occurrence" : shortDevice(d) + ": one distinct device reference"}));
      });
      if (f.normalized.includes("cpu") && !f.input.includes("cpu"))
        list.push(actor("host-cpu", "CPU", 424 + f.normalized.indexOf("cpu") * 110, 277, "weights",
          {width: 48, origin: point(534, 373), place: "normalized"}));
      if (f.runtimeConfig) f.normalized.forEach((d, i) => {
        list.push(actor("native-" + d, shortDevice(d), 848 + i * 128, 206, d === "cpu" ? "weights" : "device",
          {width: 94, origin: point(424 + i * 110, 277), via: [point(734, 277), point(734, 206)],
            delay: i * 100, place: "registered", description: "Retained " + shortDevice(d) + " handle and existing context"}));
      });
      if (f.compileConfig) list.push(actor("compile-config", "CFG", 862, 389, "config",
        {width: 78, origin: point(680, 447), via: [point(736, 447), point(736, 389)], place: "compile-config"}));
      if (f.context) list.push(actor("engine-context", "CTX", 1076, 389, "runtime",
        {width: 78, origin: point(1076, 281), place: "context"}));
      const d = f.current === null ? null : f.input[f.current];
      this.text("[data-decision]", f.error ? "Stop: native S1 was never created." : d && f.input.indexOf(d) < f.current
        ? shortDevice(d) + " already kept → merge, count unchanged"
        : f.runtimeConfig ? "Same handles retained by native S1"
        : f.native ? "Final list passed into native S1"
        : f.normalized.includes("cpu") && !f.input.includes("cpu") ? "CPU appended automatically"
        : d ? shortDevice(d) + " occupies one distinct slot" : "Empty slots show what is still missing.");
      this.node.querySelectorAll(".policy-lamp").forEach((el, i) => {
        el.dataset.enabled = String(f.store === (i ? "export" : "import") || f.store === "conflict");
        el.dataset.conflict = String(f.error);
      });
      return list;
    }
    compilation(f) {
      const list = [], reuse = f.mode === "mef";
      const compiling = ["request", "compile"].includes(f.phase) && !reuse;
      this.slots("graph", f.graph ? ["IR1", "IR2", "IR3"] : []);
      this.slots("compiler", compiling ? ["IR1", "IR2", "IR3"] : []);
      this.slots("artifact", f.artifact ? ["MEF"] : []);
      this.slots("cache", reuse ? ["MEF"] : []);
      this.slots("weights", f.weightNames, f.error ? [1] : []);
      const binding = ["validate", "initialize", "model"].includes(f.phase);
      this.slots("bindings", binding ? f.weightNames : f.repaired ? ["W_a"] : [], f.error ? [1] : []);
      this.slots("model", f.initialized ? ["M1"] : []);
      this.node.querySelector("[data-init-gate]").dataset.open = String(f.validated);
      this.node.querySelector("[data-init-gate]").dataset.blocked = String(f.error);
      this.node.querySelector("[data-cache]").dataset.selected = String(reuse);
      this.node.querySelector('[data-node="compiler"]').dataset.bypassed = String(reuse);
      this.text("[data-gate-label]", f.error ? "W_b missing · initialization blocked" : f.validated ? "both bindings valid · gate open" : "waiting for artifact + weights");
      if (f.graph) for (let i = 0; i < 3; i++) {
        list.push(actor("graph-ir-" + i, "IR" + (i + 1), 86 + i * 73, 239, "graph", {width: 47, place: "graph"}));
        if (f.compileCalls && !reuse) list.push(actor("compile-ir-" + i,
          f.phase === "compile" || f.artifact ? "code" : "IR" + (i + 1), f.artifact ? 734 : 370 + i * 73, 239,
          f.phase === "compile" || f.artifact ? "artifact" : "graph",
          {width: 47, origin: point(86 + i * 73, 239), delay: i * 80,
            opacity: f.artifact ? 0 : 1, scale: f.artifact ? .6 : 1, merge: true,
            place: f.artifact ? "compiled" : "compiler"}));
      }
      if (reuse) list.push(actor("cache-mef", "MEF", 159, 442, "artifact", {width: 80, place: "cache"}));
      if (f.artifact || f.phase === "import") list.push(actor("compiled-artifact", "MEF", 734, 239, "artifact",
        {width: 80, origin: reuse ? point(159, 442) : point(443, 239),
          via: reuse ? [point(581, 442), point(581, 333), point(734, 333)] : [], place: "artifact"}));
      f.weightNames.forEach((name, i) => {
        list.push(actor("weight-source-" + name, name, 679 + i * 110, 442, "weights", {width: 76, place: "registry"}));
        if (binding || (f.repaired && i === 0)) list.push(actor("binding-" + name, name, 984 + i * 134, 239, "weights",
          {width: 92, origin: point(679 + i * 110, 442), via: [point(878, 442), point(878, 338), point(984 + i * 134, 338)],
            delay: i * 100, place: "binding"}));
      });
      if (f.initialized) list.push(actor("initialized-model", "M1", 1048, 442, "model",
        {width: 78, origin: point(1145, 283), via: [point(1145, 342), point(1048, 342)], place: "model"}));
      return list;
    }
    execution(f) {
      const list = [];
      this.slots("tokens", f.tokens.map(String));
      this.slots("history-a", f.history.map(pair => String(pair[0])));
      this.slots("history-b", f.history.map(pair => String(pair[1])));
      this.slots("invocation", f.phase === "call" ? ["B" + f.pass] : []);
      this.slots("runtime", ["runtime", "submit"].includes(f.phase) ? ["B" + f.pass] : []);
      this.slots("results", f.output ? ["logits A", "logits B"] : []);
      f.tokens.forEach((token, i) => {
        const b = i >= f.offsets[1];
        list.push(actor("input-B" + f.pass + "-" + i, String(token), 119 + i * 106, 372, b ? "weights" : "device",
          {width: 70, place: "input", origin: f.pass > 1 ? point(96 + (f.pass - 2) * 36, b ? 198 : 160) : point(175, 268),
            via: [point(330, 270), point(330, 372)], delay: i * 65}));
      });
      if (["prepare", "call", "runtime", "submit"].includes(f.phase)) {
        const target = f.phase === "prepare" ? point(573, 418) : f.phase === "call" ? point(530, 162) : point(852, 162);
        list.push(actor("invocation-B" + f.pass, "B" + f.pass, target.x, target.y, "model",
          {width: 62, origin: point(573, 418), via: f.phase === "call" ? [point(637, 418), point(637, 262), point(530, 262)] : [],
            place: f.phase === "prepare" ? "prepared" : f.phase === "call" ? "model" : "runtime",
            description: "Invocation B" + f.pass + ": logical call, not a host/device buffer copy"}));
      }
      if (f.executing) ["A", "B"].forEach((name, i) => list.push(actor("work-B" + f.pass + "-" + name,
        name + " · " + (f.tokens.length > 2 ? "P" + (i ? f.tokens.length - f.offsets[1] : f.offsets[1]) : "D1"),
        1060 + i * 88, 417, i ? "weights" : "device", {width: 73, origin: point(852, 162),
          via: [point(974, 162), point(974, 181), point(1060 + i * 88, 181)], delay: i * 90,
          place: "gpu-work", description: name + ": illustrative device work, not a CPU request object"})));
      if (f.output) ["A", "B"].forEach((name, i) => list.push(actor("result-B" + f.pass + "-" + name,
        "logits " + name, 742 + i * 118, 403, i ? "weights" : "device", {width: 84,
          origin: point(1104, 394), via: [point(974, 394), point(974, 312), point(742 + i * 118, 312)],
          delay: i * 90, place: "output-ref"})));
      f.history.forEach((pair, index) => pair.forEach((token, i) => {
        const x = 96 + index * 36, y = i ? 198 : 160;
        list.push(actor("sample-" + index + "-" + i, String(token), x, y, i ? "weights" : "device",
          {width: 26, origin: point(742 + i * 118, 403), delay: i * 130,
            via: [point(742 + i * 118, 494), point(637, 494), point(637, 264), point(327, 264), point(327, y)],
            place: "sampled", description: "Illustrative sampled token for sequence " + (i ? "B" : "A")}));
      }));
      this.text("[data-kv-label]", "KV entries · A " + f.kvCounts[0] + " / B " + f.kvCounts[1]);
      const kv = this.node.querySelector("[data-kv-tiles]");
      kv.replaceChildren();
      f.kvCounts.forEach((count, row) => {
        for (let i = 0; i < 8; i++) kv.append(element("rect", {x: 1021 + i * 21, y: 338 + row * 23,
          width: 15, height: 15, rx: 3, class: "kv-entry", "data-owner": row ? "B" : "A", "data-filled": String(i < count)}));
      });
      return list;
    }
    lifetime(f) {
      const owners = ["caller", "decoder", "sampler"], labels = ["W", "D", "S"];
      const list = [];
      this.slots("references", owners.map((key, i) => f[key] ? labels[i] : null));
      this.slots("context", f.retained ? ["CTX"] : []);
      this.text("[data-retention]", f.retained ? "A remaining owner keeps S1 alive." : "No pictured owner retains S1.");
      owners.forEach((key, i) => {
        const at = f.releaseHistory.indexOf(key);
        list.push(actor("reference-" + key, labels[i], f[key] ? 583 + i * 91 : 589 + at * 84, f[key] ? 288 : 471,
          f[key] ? "native" : "driver", {width: 54, place: f[key] ? "retaining" : "released", detached: !f[key],
            description: (f[key] ? "Retained by " : "Released by ") + key}));
        if (f[key]) list.push(actor("owner-" + key, labels[i] + " → S1", 192, 151 + i * 144,
          key === "caller" ? "session" : "model", {width: 92, place: "owner"}));
      });
      if (f.retained) list.push(actor("owned-context", "CTX", 1057, 218, "runtime", {width: 78, place: "context"}));
      list.push(actor("shared-resources", "shared", 1057, 423, "driver", {width: 94, place: "shared"}));
      return list;
    }
  };
  // "constructor" is reserved for class construction, so dispatch this scene explicitly.
  root.InferenceSessionVisual.prototype.sceneTargets = function (frame) {
    return this.kind === "constructor" ? this.constructorScene(frame) : this[this.kind](frame);
  };
})(globalThis);
