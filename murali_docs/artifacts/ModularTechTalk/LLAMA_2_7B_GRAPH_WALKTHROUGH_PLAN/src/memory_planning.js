"use strict";
(() => {
  const data = JSON.parse(document.querySelector("#memory-data").textContent);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const gib = 1024 ** 3;
  const mib = 1024 ** 2;
  const example = data.example;
  const $ = selector => document.querySelector(selector);
  const all = selector => [...document.querySelectorAll(selector)];
  let calculation;
  let toastTimer;

  all("[data-interactive]").forEach(node => node.hidden = false);
  function toast(message) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("visible"), 2500);
  }
  async function copyText(value) {
    try { await navigator.clipboard.writeText(value); return true; }
    catch {
      const input = document.createElement("textarea");
      input.value = value;
      input.style.cssText = "position:fixed;left:-10000px;top:0";
      document.body.append(input);
      input.select();
      const copied = document.execCommand("copy");
      input.remove();
      return copied;
    }
  }
  const toGiB = bytes => (bytes / gib).toFixed(3) + " GiB";
  const toMiB = bytes => (bytes / mib).toFixed(0) + " MiB";
  function sourceLink(key, label) {
    const ref = data.references[key];
    const a = document.createElement("a");
    a.className = "source-link";
    a.dataset.ref = key;
    a.href = "../sources/" + ref.file + "#L" + ref.line;
    a.textContent = (label || ref.label) + " [" + ref.alias + ":L" + ref.line + "]";
    return a;
  }

  const notes = {
    provenance: [
      ["CLI / MAX defaults", "Command options become PipelineArgs. PipelineConfig.from_args() builds the main MAXModelConfig and runtime policy.", "from_args"],
      ["Hugging Face checkpoint metadata", "AutoConfig supplies dimensions and the checkpoint length bound. Selected weight-file sizes come from disk/cache or Hub metadata; this pricing step does not load the state_dict.", "weights_size"],
      ["MAX's architecture declaration", "llama_arch selects Llama3Config and PagedMemoryPlanner. initialize() combines HF dimensions with MAX dtype, page, device and length settings to create architecture facts.", "initialize_config"],
      ["Live device driver", "load_devices() creates Device handles. The parent reads free_memory and per-allocation limits from those selected devices while planning.", "free_memory"],
    ],
    budget: [
      ["Read current device statistics", () => "Free memory is " + toGiB(calculation.free) + " in this hypothetical device input. MAX reads and sums Device.stats[\"free_memory\"] in the API parent.", "free_memory"],
      ["Apply the utilization policy", () => "The utilization envelope is " + toGiB(calculation.envelope) + ". The excluded fraction remains headroom. This uses free memory, not a GPU product's nominal capacity.", "plan_sizes"],
      ["Subtract weights and static reserves", () => "The example subtracts " + toGiB(example.weightBytes) + " of selected checkpoint file sizes, a zero Llama activation reservation and zero single-device signal bytes. Initial KV budget: " + toGiB(calculation.initial) + ".", "weights_size"],
      ["Respect the allocation limit", () => "Cache budget after the selected single-buffer cap is " + toGiB(calculation.capped) + ". MAX also accounts for per-layer allocation layouts and optional vision reservations; this Llama exercise has neither.", "plan_sizes"],
      ["Ask the selected KV manager", () => calculation.error || ("For this plain one-leaf full-attention example, demand including the null page is " + toGiB(calculation.required) + ". The committed byte budget is " + toGiB(calculation.committed) + "; physical allocation rounds to whole pages."), "manager_size"],
      ["Return the effective plan", () => calculation.error || ("MemoryPlan carries a " + toGiB(calculation.committed) + " cache budget and a " + calculation.plannedLength + "-token effective length. This is a separate result; PipelineConfig keeps its construction values."), "plan_type"],
    ],
    handoff: [
      ["Build PipelineConfig in the parent", "serve_api_server_and_model_worker() resolves task and configuration before creating the serving application or model worker.", "config_call"],
      ["Plan memory in retrieve_factory()", "The parent loads device handles and queries current free bytes, initializes architecture facts and returns MemoryPlan. No model worker exists yet on this route.", "memory_call"],
      ["Bind a deferred pipeline factory", "The registry returns RetrievedPipeline. Its factory is a functools.partial with pipeline_config, tokenizer, architecture class, weight adapters and memory_plan already bound.", "partial"],
      ["Enter the lifespan and spawn a child", "start_model_worker() starts a new Python process and passes the factory and config. The bound MemoryPlan crosses through serialization; it is not shared process memory.", "spawn"],
      ["Invoke the factory in ModelWorker.run()", "model_factory() constructs TextGenerationPipeline. The pipeline reads planned_max_batch_size and loads devices from the plan's DeviceSpecs.", "invoke"],
      ["Load weights and build the Llama model", "The worker resolves/downloads missing weight files, loads weights, finalizes model config and performs model graph tracing, compilation and initialization.", "finalized"],
      ["Initialize the KV manager, then scheduler", "TextGenerationPipeline calls load_kv_manager() after constructing the pipeline model, using the plan's committed cache budget. ModelWorker then creates its scheduler and signals readiness.", "pipeline_init"],
      ["Wait for readiness and IPC, then serve HTTP", "The parent waits for the ready event and worker IPC connection before reaching server.serve(). The diagram shows ordered milestones, not measured timing durations.", "ready"],
    ],
  };

  const animations = [];
  class StepAnimation {
    constructor(node) {
      this.node = node;
      this.kind = node.dataset.animation;
      this.index = 0;
      this.timer = null;
      this.playing = false;
      this.node.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => {
        switch (button.dataset.action) {
          case "previous": this.pause(); this.set(this.index - 1); break;
          case "next": this.pause(); this.set(this.index + 1); break;
          case "play": this.play(); break;
          case "pause": this.pause(); break;
          case "replay": this.pause(); this.set(0); this.play(); break;
        }
      }));
      node.querySelectorAll("button[data-frame]").forEach(button => button.addEventListener("click", () => {
        this.pause(); this.set(Number(button.dataset.frame));
      }));
      this.set(0);
    }
    set(index) {
      this.index = Math.max(0, Math.min(notes[this.kind].length - 1, index));
      this.node.querySelectorAll("[data-frame]").forEach(frame => {
        const selected = Number(frame.dataset.frame) === this.index;
        frame.classList.toggle("active", selected);
        if (frame.tagName === "BUTTON") frame.setAttribute("aria-pressed", String(selected));
      });
      this.updateNote();
      this.updateControls();
    }
    updateNote() {
      const [title, description, ref] = notes[this.kind][this.index];
      this.node.querySelector("[data-frame-title]").textContent = title;
      this.node.querySelector("[data-frame-description]").textContent = typeof description === "function" ? description() : description;
      let source = this.node.querySelector("[data-frame-source]");
      if (!source) {
        source = document.createElement("p");
        source.className = "sources";
        source.dataset.frameSource = "";
        this.node.querySelector(".frame-note").append(source);
      }
      source.replaceChildren(sourceLink(ref));
      this.node.querySelector("[data-progress]").textContent = (this.index + 1) + " / " + notes[this.kind].length;
    }
    updateControls() {
      this.node.querySelector('[data-action="previous"]').disabled = this.index === 0;
      this.node.querySelector('[data-action="next"]').disabled = this.index === notes[this.kind].length - 1;
      this.node.querySelector('[data-action="play"]').disabled = this.playing;
      this.node.querySelector('[data-action="pause"]').disabled = !this.playing;
      this.node.dataset.playing = String(this.playing);
    }
    play() {
      if (this.playing) return;
      if (this.index === notes[this.kind].length - 1) this.set(0);
      this.playing = true;
      this.updateControls();
      const tick = () => {
        if (!this.playing) return;
        this.set(this.index + 1);
        if (this.index === notes[this.kind].length - 1) this.pause();
        else this.timer = setTimeout(tick, 3000);
      };
      // Reader-controlled discrete steps also work with reduced motion.
      this.timer = setTimeout(tick, 3000);
    }
    pause() {
      this.playing = false;
      clearTimeout(this.timer);
      this.timer = null;
      this.updateControls();
    }
  }

  class InitStepCards {
    constructor() {
      this.card = $("#init-step-card");
      this.sequence = $("#initialize-sequence");
      this.trigger = null;
      this.held = false;
      this.dismissed = null;
      this.frame = null;
      all("#initialize-sequence a[data-init-step]").forEach(trigger => {
        trigger.addEventListener("pointerenter", event => {
          if (event.pointerType === "touch") return;
          this.dismissed = null;
          this.show(trigger);
        });
        trigger.addEventListener("focus", () => {
          if (this.dismissed === trigger) return;
          this.dismissed = null;
          this.show(trigger);
        });
        trigger.addEventListener("keydown", event => {
          if (event.key !== " ") return;
          event.preventDefault();
          this.show(trigger, {force: true, hold: true, focus: true});
        });
      });
      // A touch tap opens readable logic; its source links retain the code route.
      this.sequence.addEventListener("click", event => {
        const trigger = event.target.closest("a[data-init-step]");
        if (!trigger || event.pointerType !== "touch") return;
        event.preventDefault();
        event.stopPropagation();
        this.show(trigger, {force: true});
      }, true);
      $("#init-step-card-close").addEventListener("click", () => this.close(true));
      $("#init-step-card-hold").addEventListener("click", () => this.setHeld(!this.held));
      this.card.addEventListener("click", event => {
        if (event.target.closest("a")) this.close();
      });
      document.addEventListener("pointerdown", event => {
        if (!this.card.contains(event.target) && !event.target.closest("a[data-init-step]")) this.close();
      });
      document.addEventListener("keydown", event => {
        if (event.key === "Escape" && !this.card.hidden) {
          event.preventDefault();
          this.close(true);
        }
      });
      // The floating card lives outside the SVG's horizontal scroll container.
      document.addEventListener("scroll", event => {
        if (this.card.hidden || this.held || this.card.contains(event.target)) return;
        cancelAnimationFrame(this.frame);
        this.frame = requestAnimationFrame(() => this.position());
      }, true);
      window.addEventListener("resize", () => this.position(this.held));
      let parent = this.sequence;
      while (parent) {
        if (parent.tagName === "DETAILS") parent.addEventListener("toggle", event => {
          if (!event.target.open) this.close();
        });
        parent = parent.parentElement;
      }
    }
    show(trigger, {force = false, hold = false, focus = false} = {}) {
      if ($("#source-dialog").open || (this.held && this.trigger !== trigger && !force)) return;
      if (this.trigger !== trigger || this.card.hidden) {
        if (this.trigger) this.deactivate();
        const step = trigger.dataset.initStep;
        const record = $("#init-logic-" + step);
        const title = record.querySelector("h3");
        const fragment = document.createDocumentFragment();
        [...record.children].filter(node => node !== title).forEach(node => fragment.append(node.cloneNode(true)));
        $("#init-step-card-title").textContent = title.textContent.replace(/^\d+\s*·\s*/, "");
        $("#init-step-card-number").textContent = "Step " + step + " of 12 · SOURCE + EXAMPLE";
        $("#init-step-card-content").replaceChildren(fragment);
        $("#init-step-card-content").scrollTop = 0;
        this.trigger = trigger;
        this.card.dataset.step = step;
        this.card.hidden = false;
        trigger.setAttribute("aria-expanded", "true");
        trigger.closest("g[data-step]").setAttribute("data-active", "true");
        this.setHeld(hold);
        this.position();
      } else if (hold) this.setHeld(true);
      if (focus) this.card.focus({preventScroll: true});
    }
    setHeld(held) {
      this.held = held;
      this.card.dataset.held = String(held);
      $("#init-step-card-hold").setAttribute("aria-pressed", String(held));
      $("#init-step-card-hold").textContent = held ? "Release card" : "Hold card";
      $("#init-step-card-state").textContent = held ? "This step stays selected" : "Stays open until closed";
    }
    position(keepPosition = false) {
      if (this.card.hidden || !this.trigger) return;
      const margin = 12;
      const width = document.documentElement.clientWidth;
      const height = innerHeight;
      const safeTop = Math.max(margin, $(".page-header").getBoundingClientRect().bottom + margin);
      this.card.style.maxHeight = Math.max(120, height - safeTop - margin) + "px";
      if (width <= 600 && matchMedia("(hover: none)").matches) {
        this.card.style.left = margin + "px";
        this.card.style.top = safeTop + "px";
        return;
      }
      let card = this.card.getBoundingClientRect();
      const label = this.trigger.querySelector(".init-step-label").getBoundingClientRect();
      keepPosition ||= label.bottom < safeTop || label.top > height;
      let left = parseFloat(this.card.style.left) || margin;
      let top = parseFloat(this.card.style.top) || safeTop;
      if (!keepPosition) {
        // Prefer beside the label. On narrow screens, scroll the card's body
        // within the larger space above/below so the triggering label stays usable.
        if (label.right + 10 + card.width <= width - margin) {
          left = label.right + 10;
          top = label.top;
        } else if (label.left - 10 - card.width >= margin) {
          left = label.left - 10 - card.width;
          top = label.top;
        } else {
          const above = label.top - safeTop - 10;
          const below = height - label.bottom - margin - 10;
          this.card.style.maxHeight = Math.max(160, Math.max(above, below)) + "px";
          card = this.card.getBoundingClientRect();
          left = label.left;
          top = below >= above ? label.bottom + 10 : label.top - card.height - 10;
        }
      }
      this.card.style.left = Math.max(margin, Math.min(left || margin, width - card.width - margin)) + "px";
      this.card.style.top = Math.max(safeTop, Math.min(top || safeTop, height - card.height - margin)) + "px";
    }
    deactivate() {
      this.trigger.setAttribute("aria-expanded", "false");
      this.trigger.closest("g[data-step]").removeAttribute("data-active");
    }
    close(restoreFocus = false) {
      if (!this.trigger) return;
      const trigger = this.trigger;
      this.dismissed = trigger;
      this.deactivate();
      this.trigger = null;
      this.card.hidden = true;
      this.setHeld(false);
      cancelAnimationFrame(this.frame);
      if (restoreFocus && trigger.getBoundingClientRect().height) trigger.focus({preventScroll: true});
    }
  }
  const initCards = new InitStepCards();

  function calculate() {
    const batchInput = $("#candidate-batch");
    const candidate = Number(batchInput.value);
    if (!Number.isInteger(candidate) || candidate < 1 || candidate > 512) {
      batchInput.setCustomValidity("Enter a whole request count from 1 to 512.");
      return false;
    }
    batchInput.setCustomValidity("");
    const free = Number($("#free-memory").value) * gib;
    const utilization = Number($("#utilization").value);
    const length = Number($("#context-length").value);
    const width = Number($("#cache-width").value);
    const capChoice = Number($("#allocation-cap").value);
    const allocationCap = capChoice ? capChoice * gib : Infinity;
    const perToken = 2 * example.layers * example.kvHeads * example.headDim * width;
    const pageBytes = example.pageSize * perToken;
    const pagesPerRequest = Math.ceil(length / example.pageSize);
    const envelope = free * utilization;
    const initial = Math.trunc(envelope - example.weightBytes);
    const capped = Math.min(initial, allocationCap);
    const payload = candidate * pagesPerRequest * pageBytes;
    const required = payload + pageBytes;
    // Source-derived simplification: one plain full-attention leaf, DP=TP=1,
    // no scales/padding/draft/vision. Jenga saturates at the byte budget;
    // its physical slab floors that budget to whole huge blocks.
    const committed = Math.max(0, Math.min(capped, required));
    const totalBlocks = Math.floor(committed / pageBytes);
    const poolSlots = Math.max(0, totalBlocks - 1);
    const plannedLength = Math.min(length, poolSlots * example.pageSize);
    const fullContexts = Math.floor(poolSlots / pagesPerRequest);
    let error = "";
    if (example.weightBytes > free) error = "Static weight estimate exceeds free memory: MAX's planner raises.";
    else if (initial <= 0) error = "No positive KV budget remains inside the utilization envelope: MAX's planner raises.";
    else if (totalBlocks < 2) error = "The budget cannot hold both a null block and one request page: the KV manager raises.";
    calculation = {free, utilization, envelope, initial, capped, committed, required, payload, totalBlocks,
      poolSlots, plannedLength, fullContexts, pageBytes, perToken, pagesPerRequest, width, candidate, length, error,
      footprint: example.weightBytes + committed};
    return true;
  }

  function refreshCalculation() {
    if (!calculate()) return;
    const c = calculation;
    const values = {
      free: toGiB(c.free), envelope: toGiB(c.envelope), initial: toGiB(c.initial), capped: toGiB(c.capped),
      committed: c.error ? "No plan" : toGiB(c.committed), footprint: c.error ? "No plan" : toGiB(c.footprint),
      perToken: (c.perToken / 1024).toFixed(0) + " KiB", pagesPerRequest: c.pagesPerRequest,
      pageBytes: toMiB(c.pageBytes), perRequest: toGiB(c.pagesPerRequest * c.pageBytes),
      requestDemand: toGiB(c.payload), pageDemand: toGiB(c.required), poolSlots: c.poolSlots,
    };
    all("[data-value]").forEach(node => node.textContent = values[node.dataset.value]);
    all("[data-bar]").forEach(node => {
      node.style.width = Math.max(0, Math.min(100, c[node.dataset.bar] / c.free * 100)) + "%";
    });
    const partitions = {
      weights: example.weightBytes,
      cache: c.error ? 0 : c.committed,
      unused: Math.max(0, c.envelope - example.weightBytes - (c.error ? 0 : c.committed)),
      headroom: c.free - c.envelope,
    };
    all("[data-slice]").forEach(node => node.style.width = Math.max(0, partitions[node.dataset.slice] / c.free * 100) + "%");
    $("#context-length-value").textContent = c.length;
    $("#element-width").replaceChildren(document.createTextNode(c.width + " "));
    const widthLabel = document.createElement("small");
    widthLabel.textContent = "bytes / element";
    $("#element-width").append(widthLabel);
    const grid = $("#page-grid");
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < c.pagesPerRequest; index++) {
      const block = document.createElement("span");
      block.className = "page-block";
      block.textContent = String(index).padStart(2, "0");
      block.title = "Page " + index + ": " + example.pageSize + " tokens, " + toMiB(c.pageBytes);
      fragment.append(block);
    }
    grid.replaceChildren(fragment);
    grid.setAttribute("aria-label", c.pagesPerRequest + " pages for a " + c.length + "-token request");
    const limited = c.fullContexts < c.candidate || c.plannedLength < c.length;
    const budgetStatus = $("#budget-status");
    const cacheStatus = $("#cache-status");
    [budgetStatus, cacheStatus].forEach(node => {
      node.classList.toggle("capacity-limited", limited && !c.error);
      node.classList.toggle("error", Boolean(c.error));
    });
    budgetStatus.textContent = c.error || (limited
      ? "The candidate batch remains " + c.candidate + ", but the pool holds " + c.fullContexts + " contexts at " + c.length + " tokens. Effective single-request length: " + c.plannedLength + "."
      : c.candidate + " full " + c.length + "-token contexts fit in this example cache.");
    cacheStatus.textContent = c.error || (c.poolSlots + " usable request pages; " + c.fullContexts + " complete " + c.length + "-token contexts fit. Physical slab: " + toGiB(c.totalBlocks * c.pageBytes) + " including the null block.");
    animations.forEach(animation => animation.updateNote());
    // A small read-only snapshot lets the browser check arithmetic invariants.
    $("#memory-budget").dataset.calculation = JSON.stringify(calculation);
  }

  refreshCalculation();
  all("[data-animation]").forEach(node => animations.push(new StepAnimation(node)));
  all("#free-memory, #utilization, #allocation-cap, #candidate-batch, #context-length, #cache-width")
    .forEach(input => input.addEventListener("input", refreshCalculation));
  all("details").forEach(detail => detail.addEventListener("toggle", () => {
    if (!detail.open) animations.filter(animation => detail.contains(animation.node)).forEach(animation => animation.pause());
  }));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) animations.forEach(animation => animation.pause());
  });
  reducedMotion.addEventListener("change", () => animations.forEach(animation => animation.pause()));

  const sidebarToggle = $("#sidebar-toggle");
  function showSidebar(shown) {
    document.body.classList.toggle("nav-hidden", !shown);
    sidebarToggle.setAttribute("aria-expanded", String(shown));
    sidebarToggle.textContent = shown ? "Hide contents" : "Show contents";
  }
  showSidebar(!matchMedia("(max-width: 960px)").matches);
  sidebarToggle.addEventListener("click", () => showSidebar(document.body.classList.contains("nav-hidden")));
  $("#expand-explanations").addEventListener("click", () => all("details:not(.code-cell)").forEach(node => node.open = true));
  $("#collapse-explanations").addEventListener("click", () => {
    all("details:not(.root-qna)").forEach(node => node.open = false);
    $("#qna-memory").open = true;
  });
  function revealHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target) return;
    if (target.tagName === "DETAILS") target.open = true;
    let parent = target.parentElement;
    while (parent) { if (parent.tagName === "DETAILS") parent.open = true; parent = parent.parentElement; }
    requestAnimationFrame(() => target.scrollIntoView({block: "start", behavior: "auto"}));
  }
  window.addEventListener("hashchange", revealHash);
  document.addEventListener("click", event => {
    const link = event.target.closest('a[href^="#"]');
    if (link && link.getAttribute("href") === location.hash) revealHash();
  });
  if (location.hash) revealHash();

  const dialog = $("#source-dialog");
  const filePre = $("#source-file");
  const search = $("#source-search");
  let selectedReference;
  let sourceMatches = [];
  let sourceIndex = -1;
  function scrollSource(row) {
    if (row) filePre.scrollTop += row.getBoundingClientRect().top - filePre.getBoundingClientRect().top - 24;
  }
  function openSource(key) {
    const ref = data.references[key];
    if (!ref) return;
    initCards.close();
    selectedReference = ref;
    const file = data.files[ref.file];
    const fragment = document.createDocumentFragment();
    const lines = file.text.replace(/\n$/, "").split("\n");
    lines.forEach((line, index) => {
      const row = document.createElement("span");
      row.className = "code-line";
      row.dataset.line = index + 1;
      row.classList.toggle("definition-line", index + 1 >= ref.start && index + 1 <= ref.end);
      row.classList.toggle("selected-line", index + 1 === ref.line);
      const number = document.createElement("span");
      number.className = "line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = index + 1;
      const text = document.createElement("span");
      text.className = "line-text";
      text.textContent = line;
      row.append(number, text);
      fragment.append(row);
    });
    filePre.querySelector("code").replaceChildren(fragment);
    $("#source-title").textContent = ref.label;
    $("#source-provenance").textContent = ref.file + " · L" + ref.start + "–L" + ref.end + " · " +
      (ref.kind === "call site" ? "call at L" + ref.line + " · " : "") + lines.length + " complete file lines · " +
      data.revision.slice(0, 12) + " · file SHA-256 " + file.sha256;
    $("#source-download").href = "../" + file.snapshot;
    search.value = "";
    sourceMatches = [];
    sourceIndex = -1;
    $("#source-match-count").textContent = "";
    if (!dialog.open) dialog.showModal();
    requestAnimationFrame(() => {
      filePre.scrollTop = 0;
      filePre.scrollLeft = 0;
      scrollSource(filePre.querySelector('[data-line="' + ref.line + '"]'));
    });
  }
  document.addEventListener("click", event => {
    const link = event.target.closest("a[data-ref]");
    if (link && data.references[link.dataset.ref]) { event.preventDefault(); openSource(link.dataset.ref); }
    const copy = event.target.closest("button[data-copy]");
    if (copy) copyText(data.references[copy.dataset.copy].code).then(ok => toast(ok ? "Complete original definition copied" : "Use Download full file to save the source"));
  });
  $("#close-source").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  $("#copy-source").addEventListener("click", async () => {
    if (selectedReference) toast(await copyText(selectedReference.code) ? "Complete original definition copied" : "Use Download original file to save the source");
  });
  function findSource(next = false) {
    const rows = [...filePre.querySelectorAll(".code-line")];
    rows.forEach(row => row.classList.remove("file-match", "current-match"));
    const term = search.value.toLocaleLowerCase();
    if (!next) {
      sourceMatches = term ? rows.filter(row => row.querySelector(".line-text").textContent.toLocaleLowerCase().includes(term)) : [];
      sourceIndex = sourceMatches.length ? 0 : -1;
    } else if (sourceMatches.length) sourceIndex = (sourceIndex + 1) % sourceMatches.length;
    sourceMatches.forEach(row => row.classList.add("file-match"));
    $("#source-match-count").textContent = !term ? "" : sourceMatches.length ? (sourceIndex + 1) + " / " + sourceMatches.length : "No matches";
    if (sourceIndex >= 0) {
      sourceMatches[sourceIndex].classList.add("current-match");
      scrollSource(sourceMatches[sourceIndex]);
    }
  }
  search.addEventListener("input", () => findSource());
  search.addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); findSource(true); } });
  $("#source-next").addEventListener("click", () => findSource(true));

  let printState;
  window.addEventListener("beforeprint", () => {
    initCards.close();
    animations.forEach(animation => animation.pause());
    printState = all("details").map(node => [node, node.open]);
    printState.forEach(([node]) => node.open = true);
  });
  window.addEventListener("afterprint", () => (printState || []).forEach(([node, open]) => node.open = open));
})();
