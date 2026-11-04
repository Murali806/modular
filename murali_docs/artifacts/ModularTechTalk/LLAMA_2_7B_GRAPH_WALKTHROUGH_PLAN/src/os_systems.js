/* Browser presentation: source-backed cards, guided animation and the teaching lab. */
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("os-data").textContent);
  const Lab = globalThis.OSSystemsLab;
  const $ = id => document.getElementById(id);
  const escape = text => String(text).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const number = n => Math.round(n).toLocaleString("en-US");
  const seconds = n => n === null || n === undefined ? "—" : n.toFixed(2) + " s";
  const gib = n => (n / Lab.GIB).toFixed(4);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const records = new Map([...document.querySelectorAll("[data-reading]")].map(node => [node.dataset.reading, node]));
  const card = $("reading-card");
  let cardTrigger = null;
  let cardHeld = false;
  let currentCard = null;
  let touchTrigger = null;
  let touchAt = -Infinity;
  let raf = null;
  let lastWall = null;
  let labPlaying = false;
  let labCreditMs = 0;
  let lastLabRender = 0;
  const stories = new Map();
  document.documentElement.classList.add("js");
  document.querySelectorAll("[data-interactive]").forEach(node => {node.hidden = false;});
  document.querySelectorAll("[data-card]").forEach(node => {
    node.setAttribute("aria-controls", "reading-card");
    node.setAttribute("aria-expanded", "false");
  });

  function closeCard(restoreFocus = false) {
    const trigger = cardTrigger;
    if (trigger) trigger.setAttribute("aria-expanded", "false");
    card.hidden = true;
    cardHeld = false;
    currentCard = null;
    cardTrigger = null;
    card.dataset.held = "false";
    $("card-hold").setAttribute("aria-pressed", "false");
    if (restoreFocus && trigger?.isConnected && trigger.checkVisibility()) {
      // Suppress focus's ordinary hover-card opening when dismissing a card.
      trigger.dataset.dismissFocus = "true";
      trigger.focus({preventScroll: true});
      queueMicrotask(() => {delete trigger.dataset.dismissFocus;});
    }
  }
  function evidenceFor(key) {
    if (key.startsWith("step:")) {
      const [, scene, index] = key.split(":");
      return data.scenes.find(record => record.id === scene).steps[Number(index)].evidence;
    }
    return key.startsWith("example:") ? "EXAMPLE" : "SOURCE / BOUNDARY";
  }
  function holdCard(held) {
    cardHeld = held;
    card.dataset.held = String(held);
    $("card-hold").setAttribute("aria-pressed", String(held));
    $("card-hold").textContent = held ? "Release card" : "Hold card";
  }
  function openCard(trigger, explicit = false, enter = false) {
    const key = trigger.dataset.card;
    if (!records.has(key) || (!explicit && cardHeld)) return;
    if (currentCard !== key || card.hidden) {
      if (cardTrigger) cardTrigger.setAttribute("aria-expanded", "false");
      const record = records.get(key);
      $("reading-card-title").textContent = record.querySelector(":scope > summary").textContent;
      const content = record.querySelector(":scope > .reading-body").cloneNode(true);
      content.querySelectorAll("[id]").forEach(node => node.removeAttribute("id"));
      $("reading-card-content").replaceChildren(content);
      $("reading-card-content").scrollTop = 0;
      $("card-evidence").textContent = evidenceFor(key);
      cardTrigger = trigger;
      currentCard = key;
      card.dataset.key = key;
      holdCard(false);
    }
    if (cardTrigger !== trigger) {
      if (cardTrigger) cardTrigger.setAttribute("aria-expanded", "false");
      cardTrigger = trigger;
    }
    card.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    if (enter) {holdCard(true); card.focus({preventScroll: true});}
  }
  $("card-close").addEventListener("click", () => closeCard(true));
  $("card-hold").addEventListener("click", () => holdCard(!cardHeld));
  document.addEventListener("pointerover", event => {
    if (event.pointerType === "touch") return;
    const trigger = event.target.closest("[data-card]");
    if (trigger && !trigger.contains(event.relatedTarget)) openCard(trigger);
  });
  document.addEventListener("pointerdown", event => {
    if (event.pointerType === "touch") {
      touchTrigger = event.target.closest("[data-card]");
      touchAt = performance.now();
    }
  });
  document.addEventListener("focusin", event => {
    const trigger = event.target.closest("[data-card]");
    // A touch's focus precedes click. Wait until click so the appearing card
    // cannot cover the button between finger-down and finger-up.
    if (trigger && trigger === touchTrigger && performance.now() - touchAt < 700) return;
    if (trigger && !trigger.dataset.dismissFocus) openCard(trigger);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !card.hidden && !$("source-dialog").open) {
      event.preventDefault(); closeCard(true); return;
    }
    const trigger = event.target.closest("[data-card]");
    if (trigger && (event.key === " " || (event.key === "Enter" && trigger instanceof SVGElement))) {
      event.preventDefault();
      selectStoryTrigger(trigger);
      openCard(trigger, true, true);
    }
  });
  document.addEventListener("click", event => {
    const trigger = event.target.closest("[data-card]");
    if (trigger) {
      event.preventDefault(); selectStoryTrigger(trigger); openCard(trigger, true, true);
    } else if (!card.hidden && !card.contains(event.target)) closeCard(false);
  });
  document.addEventListener("toggle", event => {
    if (event.target instanceof HTMLDetailsElement && !event.target.open) {
      const story = stories.get(event.target.dataset.scene);
      if (story) pauseStory(story);
      if (cardTrigger && event.target.contains(cardTrigger)) closeCard(false);
    }
  }, true);

  function pauseStory(story) {
    story.playing = false;
    story.packet = null;
    story.node.querySelector(".sequence-packet").classList.remove("visible");
    const play = story.node.querySelector('[data-action="play"]');
    play.textContent = "Play";
    play.setAttribute("aria-pressed", "false");
  }
  function pauseStories() {stories.forEach(pauseStory);}
  function selectStoryTrigger(trigger) {
    if (!trigger.hasAttribute("data-story-step")) return;
    const story = stories.get(trigger.closest("[data-scene]").dataset.scene);
    pauseStory(story);
    setFrame(story, Number(trigger.dataset.storyStep), true);
  }
  function setFrame(story, index, animate = false) {
    const last = story.record.steps.length - 1;
    story.index = Math.max(0, Math.min(last, index));
    const step = story.record.steps[story.index];
    story.node.querySelectorAll("[data-frame]").forEach(node => {
      const active = Number(node.dataset.frame) === story.index;
      node.classList.toggle("active", active);
      node.setAttribute("aria-hidden", String(!active));
    });
    story.node.querySelectorAll("[data-story-step]").forEach(node => node.setAttribute("aria-pressed", String(Number(node.dataset.storyStep) === story.index)));
    story.node.querySelector("[data-step-evidence]").textContent = step.evidence + " · values are EXAMPLE";
    story.node.querySelector("[data-step-title]").textContent = step.title;
    story.node.querySelector("[data-step-summary]").innerHTML = step.summary;
    story.node.querySelector("[data-step-result]").innerHTML = step.result;
    story.node.querySelector("[data-progress]").textContent = (story.index + 1) + " / " + (last + 1);
    story.node.querySelector('[data-action="previous"]').disabled = story.index === 0;
    story.node.querySelector('[data-action="next"]').disabled = story.index === last;
    story.packet = null;
    story.node.querySelector(".sequence-packet").classList.remove("visible");
    if (animate && !reduced.matches) {
      const path = story.node.querySelector('.sequence-event.active [data-motion-path]');
      story.packet = {path, length: path.getTotalLength(), started: null, duration: 1100 / story.speed};
      schedule();
    }
  }
  for (const record of data.scenes) {
    const node = $("scene-" + record.id);
    const story = {record, node, index: 0, speed: 1, playing: false, nextAt: 0, packet: null};
    stories.set(record.id, story);
    node.querySelector(".sequence-packet text").textContent = record.packet;
    node.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => {
      const action = button.dataset.action;
      if (action === "play") {
        const playing = !story.playing;
        pauseStories();
        story.playing = playing;
        if (playing) {
          if (story.index === record.steps.length - 1) setFrame(story, 0);
          setFrame(story, story.index, true);
          story.nextAt = performance.now() + 2200 / story.speed;
          button.textContent = "Pause";
          button.setAttribute("aria-pressed", "true");
          closeCard(false);
          schedule();
        }
      } else {
        pauseStory(story);
        closeCard(false);
        setFrame(story, action === "replay" ? 0 : story.index + (action === "previous" ? -1 : 1), true);
      }
    }));
    node.querySelector("[data-speed]").addEventListener("change", event => {
      story.speed = Number(event.target.value);
      story.nextAt = performance.now() + 2200 / story.speed;
    });
    setFrame(story, 0);
  }
  reduced.addEventListener("change", () => {
    if (reduced.matches) stories.forEach(story => {story.packet = null; story.node.querySelector(".sequence-packet").classList.remove("visible");});
  });

  let sourceRef = null;
  let sourceReturnFocus = null;
  let sourceRows = [];
  let sourceMatches = [];
  let sourceMatchIndex = 0;
  const sourceDialog = $("source-dialog");
  async function copyText(text) {
    try {await navigator.clipboard.writeText(text);}
    catch (_) {
      const area = document.createElement("textarea");
      area.value = text; area.setAttribute("readonly", "");
      area.style.cssText = "position:fixed;left:-10000px;top:0";
      // Modal focus containment requires the fallback field to be inside the dialog.
      (sourceDialog.open ? sourceDialog : document.body).append(area);
      const focused = document.activeElement;
      area.select();
      const copied = document.execCommand("copy");
      area.remove(); focused?.focus({preventScroll: true});
      if (!copied) {$("copy-status").textContent = "Select the source text to copy it."; $("copy-status").hidden = false; return;}
    }
    $("copy-status").textContent = "Complete definition copied.";
    $("copy-status").hidden = false;
    setTimeout(() => {$("copy-status").hidden = true;}, 2200);
  }
  function showSource(key, trigger) {
    const ref = data.references[key];
    if (!ref) return;
    sourceRef = ref;
    sourceReturnFocus = card.contains(trigger) ? cardTrigger : trigger;
    closeCard(false); pauseStories(); setLabPlaying(false);
    $("source-title").textContent = ref.symbol;
    $("source-location").textContent = ref.file + " · SOURCE L" + ref.start + "–L" + ref.end + " · selected L" + ref.line + " · " + data.revision.slice(0, 12);
    $("source-download").href = "../sources/" + ref.file;
    $("source-search").value = "";
    const lines = data.files[ref.file].text.split("\n");
    if (lines[lines.length - 1] === "") lines.pop();
    $("source-file").innerHTML = lines.map((text, index) => {
      const line = index + 1;
      const classes = "source-line" + (line >= ref.start && line <= ref.end ? " definition-line" : "") + (line === ref.line ? " selected-line" : "");
      return '<span class="' + classes + '" data-line="' + line + '"><span class="line-number" aria-hidden="true">' + line + '</span><span class="source-text">' + escape(text) + '</span></span>';
    }).join("\n");
    sourceRows = [...$("source-file").querySelectorAll(".source-line")].map(node => ({node, text: node.querySelector(".source-text").textContent}));
    searchSource();
    sourceDialog.showModal();
    requestAnimationFrame(() => $("source-file").querySelector(".selected-line").scrollIntoView({block: "center"}));
  }
  function searchSource() {
    const query = $("source-search").value.toLowerCase();
    sourceMatches = [];
    sourceMatchIndex = 0;
    for (const {node, text} of sourceRows) {
      node.classList.remove("search-line", "search-current");
      let start = 0, found = 0, output = "";
      if (query) {
        const lower = text.toLowerCase();
        while (true) {
          const at = lower.indexOf(query, start);
          if (at < 0) break;
          output += escape(text.slice(start, at)) + "<mark>" + escape(text.slice(at, at + query.length)) + "</mark>";
          start = at + query.length; found++;
        }
      }
      node.querySelector(".source-text").innerHTML = output + escape(text.slice(start));
      if (found) {node.classList.add("search-line"); sourceMatches.push(node);}
    }
    $("source-match-count").textContent = query ? sourceMatches.length + " matching lines" : "";
    $("source-prev").disabled = $("source-next").disabled = !sourceMatches.length;
    if (query && sourceMatches.length) selectSourceMatch(0);
  }
  function selectSourceMatch(delta) {
    if (!sourceMatches.length) return;
    sourceMatchIndex = (sourceMatchIndex + delta + sourceMatches.length) % sourceMatches.length;
    sourceMatches.forEach((node, index) => node.classList.toggle("search-current", index === sourceMatchIndex));
    sourceMatches[sourceMatchIndex].scrollIntoView({block: "center"});
    $("source-match-count").textContent = (sourceMatchIndex + 1) + " / " + sourceMatches.length + " lines";
  }
  document.addEventListener("click", event => {
    const link = event.target.closest("a[data-ref]");
    if (link) {event.preventDefault(); showSource(link.dataset.ref, link);}
    const copy = event.target.closest("[data-copy]");
    if (copy) copyText(data.references[copy.dataset.copy].code);
  });
  $("source-close").addEventListener("click", () => sourceDialog.close());
  sourceDialog.addEventListener("close", () => {
    if (sourceReturnFocus?.isConnected && sourceReturnFocus.checkVisibility()) {
      const trigger = sourceReturnFocus;
      trigger.dataset.dismissFocus = "true";
      trigger.focus({preventScroll: true});
      queueMicrotask(() => {delete trigger.dataset.dismissFocus;});
    }
  });
  $("source-search").addEventListener("input", searchSource);
  $("source-prev").addEventListener("click", () => selectSourceMatch(-1));
  $("source-next").addEventListener("click", () => selectSourceMatch(1));
  $("source-copy").addEventListener("click", () => sourceRef && copyText(sourceRef.code));

  let applied = {...Lab.DEFAULTS};
  let simulation = new Lab.Simulation(applied, data.example);
  let selectedRequest = "R1";
  let cachedSelected = null;
  const replicaNodes = new Map();
  const requestNodes = new Map();
  const routingNames = {round_robin: "Round robin", least_outstanding: "Least outstanding streams", token_work: "Least estimated token work", power_two: "Power of two choices"};
  const form = $("lab-form");
  function setLabPlaying(playing) {
    labPlaying = playing;
    $("lab-play").setAttribute("aria-pressed", String(playing));
    $("lab-play").textContent = playing ? "Pause" : "Play";
    lastWall = null;
    if (playing) schedule();
  }
  function writeConfig(config) {
    for (const [key, value] of Object.entries(config)) if (form.elements.namedItem(key)) form.elements.namedItem(key).value = value;
  }
  function restart(config, label) {
    setLabPlaying(false); closeCard(false);
    applied = {...config};
    simulation = new Lab.Simulation(applied, data.example);
    selectedRequest = "R1"; cachedSelected = null; labCreditMs = 0;
    replicaNodes.clear(); requestNodes.clear();
    $("replica-rows").replaceChildren(); $("request-list").replaceChildren();
    $("config-pending").hidden = true;
    renderLab();
    $("lab-status").textContent = label + ": paused at t=0. The snapshot seeded " + simulation.stats.seeded + " tokenized requests.";
  }
  form.addEventListener("input", () => {$("config-pending").hidden = false;});
  form.addEventListener("change", () => {$("config-pending").hidden = false;});
  form.addEventListener("submit", event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const config = {...applied};
    for (const [key, value] of new FormData(form)) config[key] = ["routing", "shape"].includes(key) ? value : Number(value);
    try {Lab.normalize(config); restart(config, "Custom scenario");}
    catch (error) {$("lab-status").textContent = error.message; return;}
    document.querySelectorAll("[data-preset]").forEach(button => button.setAttribute("aria-pressed", "false"));
  });
  document.querySelectorAll("[data-preset]").forEach(button => button.addEventListener("click", () => {
    const preset = Lab.PRESETS[button.dataset.preset];
    const config = {...Lab.DEFAULTS, ...preset.config};
    writeConfig(config); restart(config, preset.title);
    document.querySelectorAll("[data-preset]").forEach(node => node.setAttribute("aria-pressed", String(node === button)));
  }));
  $("lab-play").addEventListener("click", () => {
    setLabPlaying(!labPlaying);
    if (!labPlaying) renderLab();
    $("lab-status").textContent = labPlaying ? "Running the example service model. Inspect a request or change the fleet while it runs." : "Paused; request and queue states are held at the displayed time.";
  });
  $("lab-step").addEventListener("click", () => {
    setLabPlaying(false); simulation.tick(10); renderLab();
    $("lab-status").textContent = "Advanced ten 10 ms ticks; paused for inspection.";
  });
  $("lab-reset").addEventListener("click", () => restart(applied, "Restarted applied scenario"));
  $("lab-add").addEventListener("click", () => {
    const replica = simulation.addReplica();
    if (replica) $("lab-status").textContent = "Added " + replica.label + ": warms for two simulated seconds, then receives only new requests. Existing owners stay unchanged.";
    renderLab();
  });
  $("request-filter").addEventListener("change", renderLab);
  function sourceLink(key, label) {
    const ref = data.references[key];
    return '<a data-ref="' + key + '" href="../sources/' + ref.file + '#L' + ref.line + '">' + escape(label || ref.label) + '</a>';
  }
  function createReplica(replica) {
    const node = document.createElement("article");
    node.className = "replica-row"; node.dataset.replica = replica.id;
    node.innerHTML = '<div class="replica-header"><strong>Replica ' + replica.label + '</strong><span class="replica-state"></span><span class="replica-busy"></span><div class="replica-actions"><button data-card="object:replica" aria-controls="reading-card" aria-expanded="false">Objects ↗</button><button data-crash="' + replica.id + '">Crash</button><button data-replace="' + replica.id + '" hidden>Replace</button></div></div>' +
      '<div class="replica-flow"><div class="flow-node" data-flow="api"><span>API / CPU</span><strong></strong><small></small></div><div class="flow-node" data-flow="transit"><span>IPC transit N</span><strong></strong><small></small></div><div class="flow-node" data-flow="waiting"><span>CE waiting</span><strong></strong><small></small></div><div class="flow-node gpu-node" data-flow="active"><span>GPU batch</span><strong></strong><small></small></div><div class="flow-node output-node" data-flow="buffered"><span>Output</span><strong></strong><small></small></div><span class="request-mover" hidden></span></div>' +
      '<div class="replica-batch"><span>Selected IDs</span><span class="batch-ids"></span><div class="flight-progress" aria-label="Current simulated GPU pass progress"><span></span></div><span class="flight-label"></span></div><div class="replica-memory"><div class="memory-title"></div><div class="memory-bar" aria-label="Assigned pages, reserved growth headroom and null block"><span class="assigned"></span><span class="reserved"></span><span class="null"></span></div><div class="memory-labels"></div></div><div class="replica-note" hidden></div>';
    $("replica-rows").append(node); replicaNodes.set(replica.id, node);
    return node;
  }
  function renderReplica(replica, snap) {
    const node = replicaNodes.get(replica.id) || createReplica(replica);
    node.dataset.state = replica.state;
    node.querySelector(".replica-state").textContent = replica.state.toUpperCase() + " · " + replica.outstanding + " unfinished streams";
    node.querySelector(".replica-busy").textContent = "modeled busy " + Math.round(replica.busyFraction * 100) + "%";
    node.querySelector("[data-crash]").hidden = replica.state === "down";
    node.querySelector("[data-replace]").hidden = replica.state !== "down";
    const sending = snap.requests.filter(r => r.replicaId === replica.id && r.state === "sending").length;
    const values = {api: replica.api + replica.tokenizing, transit: replica.transit,
      waiting: replica.waiting, active: replica.active.length, buffered: replica.buffered};
    const captions = {api: replica.tokenizing + " encoding · " + replica.api + " waiting",
      transit: replica.transit + " / " + (applied.transitLimit || "∞") + " exact slots",
      waiting: "CE total " + replica.ce + " / M " + (applied.pendingLimit || "∞"),
      active: "ceiling " + applied.batchSize + " · owns KV pages",
      buffered: "tokens · " + sending + " generation-finished streams"};
    for (const [key, value] of Object.entries(values)) {
      const field = node.querySelector('[data-flow="' + key + '"]');
      const previous = field.dataset.value;
      field.querySelector("strong").textContent = number(value);
      field.querySelector("small").textContent = captions[key];
      if (previous !== undefined && previous !== String(value) && labPlaying && !reduced.matches) {
        field.classList.remove("pulse");
        // Re-running a short highlight represents a changed counter, not measured packet timing.
        requestAnimationFrame(() => field.classList.add("pulse"));
      }
      field.dataset.value = value;
    }
    const tracked = snap.requests.find(request => request.id === selectedRequest && request.replicaId === replica.id);
    const stageKeys = {api_wait: "api", tokenizing: "api", transit: "transit", waiting: "waiting",
      prefill: "active", decode: "active", cancel_pending: "active", sending: "buffered", completed: "buffered"};
    const mover = node.querySelector(".request-mover");
    mover.hidden = !tracked || !stageKeys[tracked.state] || replica.state !== "ready";
    if (!mover.hidden) {
      const location = node.querySelector('[data-flow="' + stageKeys[tracked.state] + '"]');
      mover.textContent = tracked.id;
      mover.title = "Selected request " + tracked.id + " at " + tracked.state + "; example state position";
      mover.dataset.request = tracked.id;
      mover.dataset.stage = tracked.state;
      mover.style.left = location.offsetLeft + location.offsetWidth / 2 + "px";
      mover.style.top = location.offsetTop - 1 + "px";
    }
    const ids = node.querySelector(".batch-ids");
    const signature = replica.active.join(",");
    if (ids.dataset.ids !== signature) {
      ids.innerHTML = replica.active.length ? replica.active.map(id => '<button class="batch-token" data-request="' + id + '">' + id + '</button>').join(" ") : "none";
      ids.dataset.ids = signature;
    }
    const progress = replica.flight ? 100 * (1 - replica.flight.remainingMs / replica.flight.durationMs) : 0;
    node.querySelector(".flight-progress>span").style.width = progress + "%";
    node.querySelector(".flight-label").textContent = replica.flight ? "B" + replica.flight.batchId + ": " + replica.flight.promptTokens + " prefill tokens + " + replica.flight.decoders + " decoders" : "no device pass";
    const total = snap.memory.poolBlocks;
    const ready = replica.state === "ready";
    node.querySelector(".memory-title").textContent = ready ? "KV backing pool " + gib(replica.allocatedBytes) + " GiB · " + total + " blocks × 64 MiB" : "KV backing pool is unavailable while " + replica.state;
    node.querySelector(".memory-bar .assigned").style.width = ready && total ? replica.assigned / total * 100 + "%" : "0%";
    node.querySelector(".memory-bar .reserved").style.width = ready && total ? (replica.reserved - replica.assigned) / total * 100 + "%" : "0%";
    node.querySelector(".memory-bar .null").style.width = ready && total ? 100 / total + "%" : "0%";
    node.querySelector(".memory-labels").innerHTML = '<span>Assigned <b>' + replica.assigned + '</b> · growth reserved <b>' + (replica.reserved - replica.assigned) + '</b></span><span>Unreserved usable <b>' + replica.reusable + '</b> · null <b>' + (ready ? snap.memory.nullBlocks : 0) + '</b></span>';
    const note = node.querySelector(".replica-note");
    note.hidden = ready;
    note.textContent = replica.state === "warming" ? "Not eligible for routing. Ready in " + seconds(Math.max(0, replica.readyAt - snap.now)) + "." : "Worker/API instance down. Its unfinished streams failed; Replace starts fresh state.";
  }
  function drawChart(id, history, key, now) {
    const width = 460, height = 150, left = 42, right = 15, top = 12, bottom = 25;
    const maxValue = Math.max(1, Math.ceil(Math.max(0, ...history.map(item => item[key])) / 5) * 5);
    const start = history.length ? history[0].at : 0;
    const end = Math.max(start + 1, now);
    const x = value => left + (value - start) / (end - start) * (width - left - right);
    const y = value => height - bottom - value / maxValue * (height - top - bottom);
    const points = history.map(item => x(item.at).toFixed(2) + "," + y(item[key]).toFixed(2));
    const path = points.length ? "M" + points.join(" L") : "";
    const area = points.length ? path + " L" + x(history[history.length - 1].at).toFixed(2) + "," + (height - bottom) + " L" + left + "," + (height - bottom) + " Z" : "";
    $(id).innerHTML = '<title>EXAMPLE ' + escape(key) + ' history; last ' + (now - start).toFixed(1) + ' simulated seconds</title><path class="axis" d="M' + left + ',' + top + ' V' + (height - bottom) + ' H' + (width - right) + '"/>' +
      '<path class="grid-line" d="M' + left + ',' + y(maxValue) + ' H' + (width - right) + ' M' + left + ',' + y(maxValue / 2) + ' H' + (width - right) + '"/><text x="3" y="' + (top + 5) + '">' + number(maxValue) + '</text><text x="15" y="' + (height - bottom + 4) + '">0</text><text x="' + left + '" y="145">' + start.toFixed(1) + ' s</text><text text-anchor="end" x="445" y="145">' + now.toFixed(1) + ' s</text><path class="plot-fill" d="' + area + '"/><path class="plot-line" d="' + path + '"/>';
  }
  function matchesFilter(request, filter) {
    if (filter === "active") return ["prefill", "decode", "cancel_pending"].includes(request.state);
    if (filter === "waiting") return ["api_wait", "tokenizing", "transit", "waiting"].includes(request.state);
    if (filter === "sending") return request.state === "sending";
    if (filter === "terminal") return Lab.TERMINAL.has(request.state);
    return true;
  }
  const stateExplanation = {
    api_wait: ["The request is waiting for an example tokenizer CPU slot. It has not been admitted to the worker and owns no KV pages.", "tokenizer_cpu"],
    tokenizing: ["The default-executor encoding stage is using CPU service. The API coroutine can await its result while the loop dispatches other work.", "tokenizer_cpu"],
    transit: ["The tokenized context passed the API admission gate. It occupies this replica's exact lab N queue until the worker's CE cap allows a drain.", "proxy_submit"],
    waiting: ["The worker owns this pending CE context. It waits for a batch slot and enough pages under the lab's conservative reservation policy.", "scheduler_pending"],
    prefill: ["The request owns a batch slot and KV reservation. Prompt work advances in chunks. Assigned pages can be allocated before their KV contents are fully populated.", "construct_batch"],
    decode: ["Prompt prefill has produced the first token. Later ordinary decode passes extend the sequence and its assigned KV pages until the effective output limit.", "scheduler_execute"],
    sending: ["Generation is finished and its KV ownership is already returned. Buffered output and the HTTP stream remain live until the client consumes the result.", "release_request"],
    cancel_pending: ["Cancellation arrived during a live device pass. The lab holds its page ownership through that completion boundary, then discards undelivered output and releases resources.", "release_request"],
    cancelled: ["Cancellation cleanup completed safely. Undelivered output was discarded; already-delivered output remains part of the client's history.", "proxy_cancel"],
    completed: ["All generated output was delivered. Proxy/output request state can be released; the client connection may be reused.", "proxy_drain"],
    rejected: ["This request was rejected before generation. A 429 models MAX's worker transport admission; other rejection reasons are explicit lab/external policy.", "http_429"],
    failed: ["Its worker/API replica failed. The lab does not migrate its KV or replay its partially delivered stream. External replacement handles future requests.", "process_ready"],
  };
  function renderRequestDetail(request, snap) {
    const detail = $("request-detail");
    if (!request) {detail.innerHTML = '<p class="inspector-help">No request yet. Step or Play to admit new arrivals, then select a request.</p>'; return;}
    const replica = snap.replicas.find(r => r.id === request.replicaId);
    const [explanation, ref] = stateExplanation[request.state];
    $("request-inspector").querySelector("h3").textContent = request.id + " · " + request.state;
    const values = {request_id: request.id, replica: replica?.label || "none",
      client: request.clientId + 1, prompt_tokens: request.promptTokens,
      requested_output: request.requestedOutput, effective_output_limit: request.outputLimit,
      prompt_processed: request.promptDone, output_generated: request.generated,
      output_delivered: request.delivered, output_buffered: request.buffered,
      output_discarded: request.discarded, reader_tokens_per_second: request.readerRate,
      KV_reserved_pages: request.reservedPages, KV_assigned_pages: request.assignedPages};
    if (detail.dataset.request !== request.id) {
      detail.dataset.request = request.id;
      detail.innerHTML = '<p class="inspector-help" data-explanation></p><div class="inspector-grid"><div><span>Owner</span><strong data-owner></strong></div><div><span>Client reader</span><strong data-reader></strong></div><div><span>First produced</span><strong data-produced></strong></div><div><span>Delivered TTFT</span><strong data-delivered></strong></div></div><pre class="object-values"></pre><p class="result" data-reason hidden></p><p class="inspector-help" data-stage-source></p><button data-cancel="' + request.id + '">Cancel ' + request.id + '</button><details><summary>Held request transition record</summary><ol></ol></details>';
    }
    detail.querySelector("[data-explanation]").textContent = explanation;
    detail.querySelector("[data-owner]").textContent = replica ? "Replica " + replica.label : "No ready replica";
    detail.querySelector("[data-reader]").textContent = request.readerRate + " tokens/s" + (request.slow ? " · slow" : "");
    detail.querySelector("[data-produced]").textContent = seconds(request.firstProducedAt === null ? null : request.firstProducedAt - request.createdAt);
    detail.querySelector("[data-delivered]").textContent = seconds(request.firstDeliveredAt === null ? null : request.firstDeliveredAt - request.createdAt);
    detail.querySelector("pre").textContent = JSON.stringify(values, null, 2);
    const reason = detail.querySelector("[data-reason]");
    reason.hidden = !request.reason;
    reason.textContent = request.reason + (request.code ? " · HTTP " + request.code : "");
    const source = detail.querySelector("[data-stage-source]");
    if (source.dataset.ref !== ref) {source.innerHTML = sourceLink(ref, "Source for this stage ↗"); source.dataset.ref = ref;}
    detail.querySelector("[data-cancel]").hidden = Lab.TERMINAL.has(request.state) || request.state === "cancel_pending";
    detail.querySelector("details ol").innerHTML = request.history.map(event => '<li>t=' + event.at.toFixed(2) + ' · ' + escape(event.message) + '</li>').join("");
  }
  function renderRequests(snap) {
    const filter = $("request-filter").value;
    const visible = snap.requests.filter(r => matchesFilter(r, filter)).sort((a, b) => {
      const aTerminal = Lab.TERMINAL.has(a.state), bTerminal = Lab.TERMINAL.has(b.state);
      return Number(aTerminal) - Number(bTerminal) || (aTerminal ? b.number - a.number : a.number - b.number);
    }).slice(0, 60);
    const ids = new Set(visible.map(r => r.id));
    for (const [id, node] of requestNodes) if (!ids.has(id)) {node.remove(); requestNodes.delete(id);}
    visible.forEach((request, index) => {
      let wrapper = requestNodes.get(request.id);
      if (!wrapper) {
        wrapper = document.createElement("div"); wrapper.setAttribute("role", "listitem");
        const button = document.createElement("button"); button.className = "request-row";
        button.dataset.request = request.id; button.setAttribute("aria-controls", "request-inspector");
        wrapper.append(button); requestNodes.set(request.id, wrapper);
      }
      const button = wrapper.firstElementChild;
      button.innerHTML = '<strong>' + request.id + '</strong><span><span class="state-chip ' + request.state + '">' + request.state + '</span> · ' + (snap.replicas.find(r => r.id === request.replicaId)?.label || '—') + '</span><span>' + request.generated + '/' + request.outputLimit + ' tok</span>';
      button.setAttribute("aria-pressed", String(request.id === selectedRequest));
      const at = $("request-list").children[index];
      if (at !== wrapper) $("request-list").insertBefore(wrapper, at || null);
    });
    const current = snap.requests.find(r => r.id === selectedRequest);
    if (current) cachedSelected = current;
    if (!cachedSelected && visible.length) {selectedRequest = visible[0].id; cachedSelected = visible[0];}
    renderRequestDetail(cachedSelected, snap);
  }
  function renderLab() {
    const snap = simulation.snapshot();
    $("lab-clock").textContent = "t = " + snap.now.toFixed(2) + " s";
    $("client-count").textContent = number(applied.clients);
    $("idle-count").textContent = number(snap.idle);
    $("unfinished-count").textContent = number(snap.unfinished);
    $("gpu-count").textContent = number(snap.counts.active);
    const queued = Math.max(0, snap.unfinished - snap.counts.active - snap.counts.sending);
    $("population-track").innerHTML = '<span class="idle" style="width:' + 100 * snap.idle / applied.clients + '%" title="' + snap.idle + ' idle"></span><span class="waiting" style="width:' + 100 * queued / applied.clients + '%" title="' + queued + ' waiting / CPU / IPC"></span><span class="working" style="width:' + 100 * snap.counts.active / applied.clients + '%" title="' + snap.counts.active + ' GPU members"></span><span class="sending" style="width:' + 100 * snap.counts.sending / applied.clients + '%" title="' + snap.counts.sending + ' sending"></span>';
    $("population-track").setAttribute("aria-label", snap.idle + " idle; " + queued + " queued or CPU/IPC; " + snap.counts.active + " GPU members; " + snap.counts.sending + " sending");
    $("metric-submitted").textContent = number(snap.stats.submitted);
    $("metric-waiting").textContent = number(snap.counts.waiting + snap.counts.transit);
    $("metric-buffered").textContent = number(snap.counts.buffered);
    $("metric-errors").textContent = number(snap.stats.rejected) + " / " + number(snap.stats.failed);
    $("metric-errors").title = snap.stats.rejected429 + " HTTP 429; " + snap.stats.cancelled + " cancelled; " + snap.stats.busyClients + " arrivals found all clients busy";
    $("metric-rate").textContent = snap.now ? snap.deliveredRate.toFixed(1) : "—";
    $("metric-ttft").textContent = seconds(snap.ttftP95);
    $("routing-label").textContent = routingNames[applied.routing];
    $("lab-add").disabled = snap.replicas.length >= 12;
    snap.replicas.forEach(replica => renderReplica(replica, snap));
    drawChart("queue-chart", snap.history, "waiting", snap.now);
    drawChart("rate-chart", snap.history, "rate", snap.now);
    renderRequests(snap);
    $("lab-events").innerHTML = [...snap.events].reverse().map(event => '<li>t=' + event.at.toFixed(2) + ' · ' + escape(event.message) + '</li>').join("");
    const m = snap.memory;
    $("memory-arithmetic").textContent = 'weight estimate = ' + gib(m.weightBytes) + ' GiB (' + m.weightBytes + ' bytes)\n' +
      'KV bytes/token = 2 × 32 layers × 32 KV heads × 128 head dim × 2 bytes = ' + m.tokenBytes + '\n' +
      'KV page = 128 tokens × 512 KiB = 64 MiB\n' +
      'budget = max(0, ' + applied.gpuGiB + ' GiB × ' + applied.utilization + ' − weights) = ' + gib(m.budgetBytes) + ' GiB\n' +
      'context-plan ceiling = ' + applied.batchSize + ' × ceil(4096/128) + 1 null = ' + m.fullContextBlocks + ' blocks\n' +
      'pool blocks = min(budget blocks, context-plan ceiling, optional cap) = ' + m.poolBlocks + '\n' +
      'allocated pool = ' + gib(m.allocatedBytes) + ' GiB; usable request pages = ' + m.usablePages + '\n' +
      'default-shaped request reservation = ceil((' + applied.promptTokens + ' + effective output ' + Math.min(applied.outputTokens, data.example.maxContext - applied.promptTokens) + ')/128) pages\n' +
      'request completion returns assigned/reserved ownership; pool backing remains allocated\n' +
      'accepted ' + snap.stats.admitted + '; completed ' + snap.stats.completed + '; HTTP 429 ' + snap.stats.rejected429 + '; cancelled ' + snap.stats.cancelled + '; busy-client demand ' + snap.stats.busyClients;
    $("lab").dataset.time = String(snap.now);
    $("lab").dataset.active = String(snap.counts.active);
  }
  document.addEventListener("click", event => {
    const request = event.target.closest("[data-request]");
    if (request) {selectedRequest = request.dataset.request; cachedSelected = null; renderLab();}
    const cancel = event.target.closest("[data-cancel]");
    if (cancel) {simulation.cancel(cancel.dataset.cancel); renderLab(); $("lab-status").textContent = "Cancellation sent for " + cancel.dataset.cancel + "; in-flight ownership is held until the pass boundary.";}
    const crash = event.target.closest("[data-crash]");
    if (crash) {simulation.crashReplica(crash.dataset.crash); renderLab(); $("lab-status").textContent = "Replica crashed. Its unfinished streams failed; other replicas continue independently.";}
    const replace = event.target.closest("[data-replace]");
    if (replace) {simulation.replaceReplica(replace.dataset.replace); renderLab(); $("lab-status").textContent = "External replacement warming for two simulated seconds; old streams remain failed.";}
  });
  function schedule() {if (raf === null) raf = requestAnimationFrame(frame);}
  function frame(wall) {
    raf = null;
    const elapsed = lastWall === null ? 0 : Math.min(100, wall - lastWall);
    lastWall = wall;
    let active = false;
    for (const story of stories.values()) {
      if (story.playing) {
        active = true;
        if (wall >= story.nextAt) {
          if (story.index === story.record.steps.length - 1) pauseStory(story);
          else {setFrame(story, story.index + 1, true); story.nextAt = wall + 2200 / story.speed;}
        }
      }
      if (story.packet) {
        active = true;
        const packet = story.packet;
        if (packet.started === null) packet.started = wall;
        const progress = Math.min(1, (wall - packet.started) / packet.duration);
        const point = packet.path.getPointAtLength(progress * packet.length);
        const marker = story.node.querySelector(".sequence-packet");
        marker.classList.add("visible"); marker.setAttribute("transform", "translate(" + point.x + "," + point.y + ")");
        if (progress === 1) {story.packet = null; marker.classList.remove("visible");}
      }
    }
    if (labPlaying) {
      active = true;
      labCreditMs += elapsed * Number($("lab-speed").value);
      const ticks = Math.min(80, Math.floor(labCreditMs / Lab.TICK_MS));
      if (ticks) {simulation.tick(ticks); labCreditMs -= ticks * Lab.TICK_MS;}
      if (wall - lastLabRender >= 100) {renderLab(); lastLabRender = wall;}
    }
    if (active) schedule(); else lastWall = null;
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {pauseStories(); setLabPlaying(false); $("lab-status").textContent = "Paused while the page is hidden.";}
  });
  function revealHash() {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!target) return;
    let node = target;
    while (node) {if (node instanceof HTMLDetailsElement) node.open = true; node = node.parentElement;}
    requestAnimationFrame(() => target.scrollIntoView({block: "start"}));
  }
  window.addEventListener("hashchange", revealHash);
  let printState = [];
  window.addEventListener("beforeprint", () => {
    closeCard(false); pauseStories(); setLabPlaying(false);
    printState = [...document.querySelectorAll("details")].map(node => [node, node.open]);
    printState.forEach(([node]) => {node.open = true;});
  });
  window.addEventListener("afterprint", () => printState.forEach(([node, open]) => {node.open = open;}));
  $("print-page").addEventListener("click", () => print());
  renderLab(); revealHash();
})();
