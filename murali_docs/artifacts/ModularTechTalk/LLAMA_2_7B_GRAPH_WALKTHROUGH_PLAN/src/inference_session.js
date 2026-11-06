/* Controls, accessible explanations and source explorer for the live lessons. */
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("inference-session-data").textContent);
  const Model = globalThis.InferenceSessionLesson;
  const $ = id => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const scenes = new Map();
  const card = $("explanation-card"), legend = $("legend-card"), dialog = $("source-dialog");
  let cardTrigger = null, pinned = false, closeTimer = null, suppressFocus = false;
  let sourceTrigger = null, sourceReference = null, sourceLines = [], matches = [], matchIndex = -1;
  let downloadURL = null, toastTimer = null, raf = null;
  document.documentElement.classList.add("js");
  document.querySelectorAll("[data-interactive]").forEach(node => {node.hidden = false;});

  function sourceLink(key) {
    const r = data.references[key];
    if (!r) throw new Error("Missing InferenceSession source: " + key);
    return '<a href="../sources/' + escape(r.file) + "#L" + r.line + '" data-ref="' + key + '">' +
      escape(r.label) + " <small>[" + r.alias + ":L" + r.line + "]</small></a>";
  }
  function sourceLinks(keys) { return keys.map(sourceLink).join(" · "); }
  function svgElement(tag, attributes = {}, text) {
    const el = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function statusText(text, value) {
    text.textContent = value;
    const shape = text.parentElement.querySelector(":scope > rect");
    if (!shape) return;
    const maxWidth = Number(shape.getAttribute("width")) - 16;
    if (text.getComputedTextLength() <= maxWidth) return;
    const words = String(value).split(" ");
    if (words.length < 2) return;
    let split = 1, best = Infinity;
    for (let i = 1; i < words.length; i++) {
      const difference = Math.abs(words.slice(0, i).join(" ").length - words.slice(i).join(" ").length);
      if (difference < best) {best = difference; split = i;}
    }
    text.replaceChildren(svgElement("tspan", {x: text.getAttribute("x"), dy: -11}, words.slice(0, split).join(" ")),
      svgElement("tspan", {x: text.getAttribute("x"), dy: 12}, words.slice(split).join(" ")));
  }
  function chip(label, value) {
    return '<span class="state-chip">' + escape(label) + '<strong>' + escape(value) + "</strong></span>";
  }
  function renderState(scene, frame) {
    const node = scene.node;
    const values = {...frame};
    let alive = {}, chips = "";
    if (scene.kind === "constructor") {
      alive = {pipeline: true, arguments: true, hardware: true, python: frame.python,
        normalize: frame.python && !frame.error, store: frame.python, devices: frame.runtimeConfig,
        compile: frame.compileConfig, context: frame.context};
      values.pipeline = frame.returned ? "self.session = S1" : "constructor call";
      values.python = frame.returned ? "Python wrapper retained" : frame.python ? "constructor active" : "not constructed";
      values.registered = frame.runtimeConfig ? frame.normalized.length + " distinct device references retained" : "no handles registered";
      values.hardware = frame.context ? "no decoder execution" : "selected device target";
      node.querySelector("[data-constructor-input]").textContent = "[" + frame.input.join(", ") + "]";
      node.querySelector("[data-constructor-threads]").textContent = "num_threads = " + (frame.threads === null ? "None → 0" : frame.threads);
      node.querySelector("[data-constructor-store]").textContent = "custom_extensions = " + (frame.extension ? "[example]" : "[]");
      node.querySelector("[data-boundary]").dataset.state = frame.native ? "ready" : "waiting";
      chips = chip("Native session", frame.native ? "S1" : "—") +
        chip("Device handles", frame.normalized.length) +
        chip("Native thread option", frame.native ? frame.nativeThreads : "—") +
        chip("Decoder artifacts", 0) + chip("Initialized decoders", 0);
    } else if (scene.kind === "compilation") {
      alive = {session: true, graph: frame.graph, compiler: frame.compileCalls > 0 && frame.mode !== "mef",
        artifact: frame.artifact, registry: frame.registry, init: frame.registry, model: frame.initialized};
      if (frame.mode === "mef") values.compilerStatus = "bypassed · stored MEF used";
      else if (frame.artifact) values.compilerStatus = "compile work completed";
      node.querySelector("[data-repair]").disabled = !frame.error;
      chips = chip("Session", "S1") + chip("Traced graphs", frame.graph ? 1 : 0) +
        chip("Artifacts", frame.artifact ? 1 : 0) + chip("Native init calls", frame.initCalls) +
        chip("Ready decoders", frame.initialized ? 1 : 0);
    } else if (scene.kind === "execution") {
      alive = {pipeline: true, model: true, runtime: true, driver: true, gpu: true,
        buffers: frame.tokens.length > 0, results: frame.output};
      node.querySelector("[data-another]").disabled = scene.index !== scene.frames.length - 1 || (scene.options.extraPasses || 0) >= 4;
      node.querySelector("[data-offsets]").textContent = "row_offsets = [" + frame.offsets.join(", ") + "]";
      node.querySelector("[data-gpu-work]").dataset.busy = String(frame.executing);
      node.querySelector("[data-work-label]").textContent = frame.executing ? "batch " + frame.pass : "between passes";
      node.querySelector("[data-result-tile]").toggleAttribute("hidden", !frame.output);
      chips = chip("Session", "S1") + chip("Decoder", "M1") + chip("Decoder setups", 1) +
        chip("Input tokens", frame.inputCount) + chip("execute calls", frame.calls) +
        chip("Completed passes", frame.completed);
    } else {
      alive = {caller: frame.caller, decoder: frame.decoder, sampler: frame.sampler,
        native: frame.retained, context: frame.retained, release: true};
      values.caller = frame.caller ? "retains native S1" : "wrapper released";
      values.decoder = frame.decoder ? "retains native S1" : "model released";
      values.sampler = frame.sampler ? "retains native S1" : "model released";
      values.releaseStatus = "can outlive this session";
      node.querySelector("[data-reference-count]").textContent = frame.references;
      ["caller", "decoder", "sampler"].forEach(key => {
        node.querySelector('[data-wire="' + key + '-ref"]').classList.toggle("released", !frame[key]);
        node.querySelector('[data-drop-owner="' + key + '"]').disabled = !frame[key];
      });
      node.querySelector('[data-wire="owned"]').classList.toggle("released", !frame.retained);
      chips = chip("Pictured native refs", frame.references) +
        chip("Native S1", frame.retained ? "retained" : "eligible for release") +
        chip("Underlying shared resources", "can outlive S1");
    }
    node.querySelectorAll("[data-node]").forEach(component => {
      const key = component.dataset.node;
      const error = frame.error && ((scene.kind === "constructor" && key === "store") ||
        (scene.kind === "compilation" && ["init", "registry"].includes(key)));
      component.dataset.state = error ? "error" : alive[key] ? "ready" : scene.kind === "lifetime" ? "released" : "waiting";
      component.classList.toggle("active", frame.active.includes(key));
    });
    node.querySelectorAll("[data-status]").forEach(text => {
      const value = values[text.dataset.status];
      if (value !== undefined && typeof value !== "boolean") statusText(text, String(value));
    });
    node.querySelector("[data-state-strip]").innerHTML = chips;
    node.dataset.status = frame.status;
    node.dataset.index = scene.index;
    node.dataset.frames = scene.frames.length;
    node.querySelector("[data-progress]").textContent = (scene.index + 1) + " / " + scene.frames.length;
    node.querySelector("[data-event-title]").textContent = frame.title;
    node.querySelector("[data-event-detail]").textContent = frame.detail;
    node.querySelector("[data-event]").dataset.kind = frame.kind;
    node.querySelector("[data-event-sources]").innerHTML = sourceLinks(frame.refs);
    node.querySelector('[data-action="back"]').disabled = scene.index === 0;
    node.querySelector('[data-action="step"]').disabled = scene.index === scene.frames.length - 1;
    node.querySelectorAll("[data-jump]").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.jump) === scene.index)));
    node.querySelectorAll("[data-wire]").forEach(w => {
      w.classList.toggle("active", w.dataset.wire === frame.packet?.edge);
      w.style.markerEnd = "url(#arrow-" + scene.kind + ")";
    });
    const phase = scene.kind === "constructor" ? frame.returned ? "ready" : frame.context ? "context" :
      frame.runtimeConfig ? "register" : frame.native ? "native" : frame.normalized.length ? "normalize" : "options"
      : scene.kind === "compilation" ? ({ready: "trace", trace: "trace", request: "compile", compile: "compile", import: "compile",
        artifact: "compile", collect: "bind", validate: "bind", initialize: "initialize", model: "ready"})[frame.phase]
      : scene.kind === "execution" ? ({ready: "prepare", prepare: "prepare", call: "execute", runtime: "execute",
        submit: "submit", outputs: "outputs", sample: "sample"})[frame.phase] : String(frame.references);
    node.querySelectorAll("[data-phase]").forEach(el => el.setAttribute("aria-current", String(el.dataset.phase === phase)));
  }
  function setFrame(scene, index, animate = false) {
    scene.index = Math.max(0, Math.min(scene.frames.length - 1, index));
    renderState(scene, scene.frames[scene.index]);
    scene.visual.render(scene.frames[scene.index], animate, scene.speed);
    if (animate) schedule();
  }
  function pause(scene) {
    scene.playing = false;
    scene.visual.pause();
    scene.node.dataset.playing = "false";
    const button = scene.node.querySelector('[data-action="play"]');
    button.textContent = "▶ Play";
    button.setAttribute("aria-pressed", "false");
  }
  function pauseAll() {scenes.forEach(pause);}
  function rebuildFrames(scene) {
    scene.frames = Model.framesFor(scene.kind, scene.options);
    scene.node.querySelector("[data-transcript]").innerHTML = scene.frames.map((frame, i) =>
      '<li><button data-jump="' + i + '" aria-pressed="false"><span>' + String(i + 1).padStart(2, "0") +
      "</span>" + escape(frame.title) + "</button></li>").join("");
  }
  function reset(scene) {
    pause(scene);
    const modes = [...scene.node.querySelectorAll("[data-mode]")];
    const selected = scene.options.mode || modes[0]?.dataset.mode;
    modes.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === selected)));
    rebuildFrames(scene);
    setFrame(scene, 0);
  }
  function start(scene) {
    pauseAll();
    closeCard(false);
    closeLegend(false);
    if (scene.index === scene.frames.length - 1) setFrame(scene, 0);
    scene.playing = true;
    scene.node.dataset.playing = "true";
    scene.nextAt = performance.now() + 1900 / scene.speed;
    const button = scene.node.querySelector('[data-action="play"]');
    button.textContent = "Ⅱ Pause";
    button.setAttribute("aria-pressed", "true");
    scene.visual.resume();
    schedule();
  }
  function schedule() {if (raf === null) raf = requestAnimationFrame(tick);}
  function tick(now) {
    raf = null;
    let needed = false;
    for (const scene of scenes.values()) {
      if (scene.playing && now >= scene.nextAt) {
        if (scene.index === scene.frames.length - 1) pause(scene);
        else {
          setFrame(scene, scene.index + 1, true);
          scene.nextAt = now + 1900 / scene.speed;
        }
      }
      const moving = scene.visual.tick(now);
      needed ||= scene.playing || moving;
    }
    if (needed) schedule();
  }
  for (const lesson of data.lessons) {
    const node = $(lesson.id);
    const scene = {kind: lesson.id, node, options: {}, index: 0, frames: [],
      playing: false, speed: 1, nextAt: 0,
      visual: new globalThis.InferenceSessionVisual(node, lesson.id, reduced)};
    scenes.set(lesson.id, scene);
    reset(scene);
    node.querySelector(".playback").addEventListener("click", event => {
      const button = event.target.closest("[data-action]");
      if (!button) return;
      const action = button.dataset.action;
      closeCard(false);
      if (action === "play") {if (scene.playing) pause(scene); else start(scene);}
      else {
        pause(scene);
        if (action === "reset") {
          delete scene.options.repaired; delete scene.options.extraPasses; delete scene.options.order;
          reset(scene);
        }
        else setFrame(scene, scene.index + (action === "step" ? 1 : -1), true);
      }
    });
    node.querySelector("[data-speed]").addEventListener("change", event => {
      scene.speed = Number(event.target.value);
      if (scene.playing) scene.nextAt = performance.now() + 1900 / scene.speed;
    });
    node.querySelectorAll("[data-device-case], [data-mode]").forEach(button => {
      button.addEventListener("click", () => {
        closeCard(false);
        const attr = button.hasAttribute("data-mode") ? "mode" : "deviceCase";
        node.querySelectorAll(attr === "mode" ? "[data-mode]" : "[data-device-case]")
          .forEach(b => b.setAttribute("aria-pressed", String(b === button)));
        scene.options[attr === "mode" ? "mode" : "devices"] = button.dataset[attr];
        delete scene.options.repaired; delete scene.options.extraPasses; delete scene.options.order;
        reset(scene);
      });
    });
    node.querySelectorAll("[data-option]").forEach(input => {
      input.addEventListener("change", () => {
        const option = input.dataset.option;
        scene.options[option] = option === "extension" ? input.checked :
          option === "threads" ? (input.value === "default" ? null : Number(input.value)) : input.value;
        reset(scene);
      });
    });
    node.querySelector("[data-transcript]").addEventListener("click", event => {
      const button = event.target.closest("[data-jump]");
      if (button) {pause(scene); setFrame(scene, Number(button.dataset.jump), true);}
    });
    node.querySelector("[data-repair]")?.addEventListener("click", () => {
      if (!scene.frames[scene.index].error) return;
      pause(scene); closeCard(false);
      scene.options.repaired = true;
      rebuildFrames(scene);
      setFrame(scene, scene.index + 1, true);
      start(scene);
    });
    node.querySelector("[data-another]")?.addEventListener("click", () => {
      if (scene.index !== scene.frames.length - 1 || (scene.options.extraPasses || 0) >= 4) return;
      pause(scene); closeCard(false);
      scene.options.extraPasses = (scene.options.extraPasses || 0) + 1;
      rebuildFrames(scene);
      setFrame(scene, scene.index + 1, true);
      start(scene);
    });
    node.querySelectorAll("[data-drop-owner]").forEach(button => button.addEventListener("click", () => {
      const frame = scene.frames[scene.index], owner = button.dataset.dropOwner;
      if (!frame[owner]) return;
      pause(scene); closeCard(false);
      const prefix = [...frame.releaseHistory, owner];
      scene.options.order = [...prefix, ...["caller", "decoder", "sampler"].filter(key => !prefix.includes(key))];
      node.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", "false"));
      rebuildFrames(scene);
      setFrame(scene, prefix.length, true);
    }));
  }
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) pause(scenes.get(entry.target.dataset.lesson));
    });
  }, {threshold: 0});
  scenes.forEach(scene => observer.observe(scene.node));
  document.addEventListener("visibilitychange", () => {if (document.hidden) pauseAll();});
  reduced.addEventListener("change", () => pauseAll());

  function restoreFocus(trigger) {
    if (!trigger?.isConnected) return;
    suppressFocus = true;
    trigger.focus({preventScroll: true});
    queueMicrotask(() => {suppressFocus = false;});
  }
  function closeCard(focus = false) {
    clearTimeout(closeTimer);
    const trigger = cardTrigger;
    card.hidden = true;
    pinned = false;
    if (trigger) trigger.setAttribute("aria-expanded", "false");
    cardTrigger = null;
    if (focus) restoreFocus(trigger);
  }
  function closeLegend(focus = false) {
    legend.hidden = true;
    $("legend-button").setAttribute("aria-expanded", "false");
    if (focus) restoreFocus($("legend-button"));
  }
  function positionCard(trigger) {
    const box = trigger.getBoundingClientRect();
    const width = card.getBoundingClientRect().width;
    const height = card.getBoundingClientRect().height;
    const left = Math.min(window.innerWidth - width - 14, Math.max(14, box.left));
    const top = Math.max(65, Math.min(box.bottom + 10, window.innerHeight - height - 14));
    card.style.left = left + "px";
    card.style.top = top + "px";
  }
  function openCard(trigger, explicit = false) {
    if (pinned && !explicit && cardTrigger !== trigger) return;
    clearTimeout(closeTimer);
    closeLegend();
    if (cardTrigger && cardTrigger !== trigger) cardTrigger.setAttribute("aria-expanded", "false");
    const key = trigger.dataset.lessonCard || trigger.dataset.component;
    const lesson = trigger.dataset.lessonCard && data.lessons.find(item => item.id === key);
    if (lesson) {
      $("explanation-kicker").textContent = "Lesson " + lesson.number + " · SOURCE";
      $("explanation-title").textContent = lesson.title;
      $("explanation-body").innerHTML = "<p>" + escape(lesson.detail) + '</p><p class="card-try"><b>Try it</b><br>' +
        escape(lesson.try_it) + '</p><p class="card-source">' + sourceLinks(lesson.refs) + "</p>";
    } else {
      const component = data.components[key];
      if (!component) return;
      $("explanation-kicker").textContent = "Component · " + (["driver", "gpu", "hardware"].includes(key) ? "BOUNDARY" : "SOURCE");
      $("explanation-title").textContent = component.title;
      $("explanation-body").innerHTML = "<p>" + escape(component.text) + '</p><p class="card-source">' + sourceLinks(component.refs) + "</p>";
    }
    cardTrigger = trigger;
    trigger.setAttribute("aria-expanded", "true");
    card.dataset.key = lesson ? "lesson:" + key : "component:" + key;
    card.hidden = false;
    pinned = explicit;
    card.dataset.pinned = String(explicit);
    positionCard(trigger);
    if (explicit) {pauseAll(); card.focus({preventScroll: true});}
  }
  function delayClose() {
    clearTimeout(closeTimer);
    if (!pinned) closeTimer = setTimeout(() => {
      if (!card.matches(":hover") && !card.contains(document.activeElement) &&
          !cardTrigger?.matches(":hover") && cardTrigger !== document.activeElement) closeCard();
    }, 180);
  }
  document.querySelectorAll("[data-lesson-card], [data-component]").forEach(trigger => {
    trigger.setAttribute("aria-controls", "explanation-card");
    trigger.setAttribute("aria-expanded", "false");
    if (trigger.hasAttribute("data-lesson-card")) {
      trigger.addEventListener("pointerenter", event => {if (event.pointerType !== "touch") openCard(trigger);});
      trigger.addEventListener("pointerleave", delayClose);
      trigger.addEventListener("focus", () => {if (!suppressFocus) openCard(trigger);});
      trigger.addEventListener("blur", delayClose);
    }
    trigger.addEventListener("click", event => {
      event.stopPropagation();
      if (cardTrigger === trigger && pinned && !card.hidden) closeCard(false);
      else openCard(trigger, true);
    });
    if (trigger instanceof SVGElement) trigger.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {event.preventDefault(); openCard(trigger, true);}
    });
  });
  card.addEventListener("pointerenter", () => clearTimeout(closeTimer));
  card.addEventListener("pointerleave", delayClose);
  card.addEventListener("focusout", delayClose);
  card.querySelector("[data-close-card]").addEventListener("click", () => closeCard(true));
  const legendKeys = ["pipeline", "python", "native", "compile", "context", "devices", "graph", "artifact", "registry", "model", "buffers", "runtime", "driver", "gpu"];
  $("legend-content").innerHTML = legendKeys.map(key => {
    const c = data.components[key], colors = Model.COLORS[c.color];
    return '<div class="legend-entry"><i class="swatch" style="background:' + colors[1] + ";border-color:" + colors[0] +
      '"></i><div><strong>' + escape(c.title) + "</strong><p>" + escape(c.text) + "</p></div></div>";
  }).join("");
  $("legend-button").addEventListener("click", event => {
    event.stopPropagation();
    const opening = legend.hidden;
    closeCard(false);
    if (opening) {legend.hidden = false; $("legend-button").setAttribute("aria-expanded", "true"); legend.focus({preventScroll: true});}
    else closeLegend(true);
  });
  legend.querySelector("[data-close-legend]").addEventListener("click", () => closeLegend(true));
  document.addEventListener("click", event => {
    const source = event.target.closest("[data-ref]");
    if (source) {
      event.preventDefault();
      openSource(source.dataset.ref, card.contains(source) ? cardTrigger : source);
      return;
    }
    if (!card.hidden && !card.contains(event.target) && event.target !== cardTrigger) closeCard();
    if (!legend.hidden && !legend.contains(event.target) && !$("legend-button").contains(event.target)) closeLegend();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !dialog.open) {
      if (!card.hidden) {event.preventDefault(); closeCard(true);}
      else if (!legend.hidden) {event.preventDefault(); closeLegend(true);}
    }
    if (dialog.open || !card.hidden || !legend.hidden || event.target.closest("input,select,textarea")) return;
    const scene = scenes.get(event.target.closest("[data-lesson]")?.dataset.lesson);
    if (scene && ["ArrowRight", "ArrowLeft"].includes(event.key) && !event.target.closest(".visual")) {
      event.preventDefault(); pause(scene);
      setFrame(scene, scene.index + (event.key === "ArrowRight" ? 1 : -1), true);
    }
  });

  function openSource(key, trigger) {
    pauseAll();
    sourceTrigger = trigger;
    closeCard(false);
    closeLegend(false);
    sourceReference = data.references[key];
    const r = sourceReference, file = data.files[r.file];
    sourceLines = file.text.split("\n");
    if (sourceLines[sourceLines.length - 1] === "") sourceLines.pop();
    $("source-title").textContent = r.label;
    $("source-location").textContent = r.file + " · L" + r.start + "–" + r.end + " · call/definition L" + r.line +
      " · revision " + data.revision.slice(0, 12);
    const content = document.createDocumentFragment();
    sourceLines.forEach((line, i) => {
      const n = i + 1, row = document.createElement("span");
      row.className = "source-line" + (n >= r.start && n <= r.end ? " definition-line" : "") +
        (n === r.line ? " selected-line" : "");
      row.dataset.line = n;
      const number = document.createElement("span");
      number.className = "line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = n;
      row.append(number, document.createTextNode(line));
      content.append(row);
    });
    $("source-file").querySelector("code").replaceChildren(content);
    $("source-search").value = "";
    matches = []; matchIndex = -1; $("source-count").textContent = "";
    if (downloadURL) URL.revokeObjectURL(downloadURL);
    downloadURL = URL.createObjectURL(new Blob([file.text], {type: "text/plain"}));
    $("source-download").href = downloadURL;
    $("source-download").download = r.file.split("/").pop();
    if (!dialog.open) dialog.showModal();
    requestAnimationFrame(() => {
      const selected = $("source-file").querySelector(".selected-line");
      $("source-file").scrollTop = selected.offsetTop - $("source-file").offsetTop - 48;
    });
  }
  $("source-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => restoreFocus(sourceTrigger));
  dialog.addEventListener("click", event => {
    const box = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right ||
        event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
  });
  function nextMatch(delta) {
    if (!matches.length) return;
    matchIndex = (matchIndex + delta + matches.length) % matches.length;
    $("source-count").textContent = (matchIndex + 1) + " / " + matches.length;
    matches[matchIndex].scrollIntoView({block: "center", behavior: "instant"});
  }
  $("source-search").addEventListener("input", () => {
    const query = $("source-search").value.toLowerCase();
    matches = [];
    $("source-file").querySelectorAll(".source-line").forEach((row, i) => {
      const hit = Boolean(query) && sourceLines[i].toLowerCase().includes(query);
      row.classList.toggle("match-line", hit);
      if (hit) matches.push(row);
    });
    matchIndex = -1;
    $("source-count").textContent = query ? "0 / " + matches.length : "";
    nextMatch(1);
  });
  $("source-next").addEventListener("click", () => nextMatch(1));
  $("source-previous").addEventListener("click", () => nextMatch(-1));
  $("source-search").addEventListener("keydown", event => {
    if (event.key === "Enter") {event.preventDefault(); nextMatch(event.shiftKey ? -1 : 1);}
  });
  function toast(message) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").hidden = false;
    toastTimer = setTimeout(() => {$("toast").hidden = true;}, 2400);
  }
  $("source-copy").addEventListener("click", async () => {
    const r = sourceReference;
    const code = data.files[r.file].text.split(/(?<=\n)/).slice(r.start - 1, r.end).join("");
    try {await navigator.clipboard.writeText(code); toast("Complete definition copied.");}
    catch {toast("Clipboard unavailable. Select the source text or download the file.");}
  });
  const printStates = new Map();
  window.addEventListener("beforeprint", () => {
    pauseAll();
    document.querySelectorAll(".transcript, .contract").forEach(el => {printStates.set(el, el.open); el.open = true;});
  });
  window.addEventListener("afterprint", () => {printStates.forEach((open, el) => {el.open = open;}); printStates.clear();});
  document.querySelector("[data-print]").addEventListener("click", () => window.print());
})();
