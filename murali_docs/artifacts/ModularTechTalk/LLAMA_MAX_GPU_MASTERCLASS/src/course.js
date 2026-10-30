(() => {
  const dataNode = document.querySelector("#course-data");
  if (!dataNode) return;

  const course = JSON.parse(dataNode.textContent);
  const scenes = course.scenes;
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const state = {
    scene: 0,
    step: 0,
    view: "system",
    playing: false,
    timer: null,
    selectedLine: null,
  };

  function esc(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function current() {
    return scenes[state.scene];
  }

  function progressClass(index) {
    if (index === state.step - 1) return "revealed current";
    if (index < state.step) return "revealed";
    return "future";
  }

  function evidenceClass(value) {
    if (value.includes("BOUNDARY")) return "boundary";
    if (value.includes("GPU-LAB")) return "gpu-lab";
    if (value.includes("COMPILE-ONLY")) return "compile-only";
    return "source";
  }

  function renderRail() {
    const rail = $("#scene-rail");
    rail.innerHTML = scenes.map((scene, index) => {
      const unit = Math.floor(index / 2) + 1;
      const half = index % 2 ? "B" : "A";
      const status = index === state.scene ? "active" : index < state.scene ? "done" : "";
      return `<button class="scene-chip ${status}" data-scene="${index}" title="${esc(scene.title)}"><b>${String(unit).padStart(2, "0")}${half}</b><span>${esc(scene.unitTitle)}</span></button>`;
    }).join("");
    rail.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        stop();
        state.scene = Number(button.dataset.scene);
        state.step = 0;
        state.selectedLine = null;
        render();
      });
    });
    rail.querySelector(".active")?.scrollIntoView({ inline: "center", block: "nearest" });
  }

  function renderSource(scene) {
    $("#source-symbol").textContent = scene.title;
    $("#source-path").textContent = scene.file;
    $("#source-range").textContent = `L${scene.lines[0].number}-L${scene.lines.at(-1).number}`;
    const evidence = $("#source-evidence");
    evidence.textContent = scene.evidence;
    evidence.className = `evidence ${evidenceClass(scene.evidence)}`;

    $("#source-code").innerHTML = scene.lines.map((line, index) => {
      const visible = index < state.step * 2;
      const currentLine = state.step > 0 && index >= (state.step - 1) * 2 && index < state.step * 2;
      const selected = state.selectedLine === index;
      return `<button class="source-line ${visible ? "revealed" : "future"} ${currentLine ? "current" : ""} ${selected ? "selected" : ""}" data-line-index="${index}"><span>${line.number}</span><code>${esc(line.text || " ")}</code></button>`;
    }).join("");
    $("#source-code").querySelectorAll(".source-line").forEach((line) => {
      line.addEventListener("click", () => {
        state.selectedLine = Number(line.dataset.lineIndex);
        renderSource(scene);
        renderInspector(scene);
      });
    });

    const focusIndex = state.selectedLine ?? Math.max(0, state.step * 2 - 1);
    const focus = scene.lines[focusIndex];
    $("#line-explanation").innerHTML = `<code>L${focus.number}</code><span>${esc(scene.explanations[focusIndex])}</span>`;
  }

  function flowDiagram(scene, mode = "flow") {
    const vertical = mode === "sequence";
    return `<div class="flow-diagram ${vertical ? "vertical" : ""} ${mode === "boundary" ? "boundary-flow" : ""}">
      ${scene.concepts.map((concept, index) => `<div class="flow-node ${progressClass(index)}" data-detail="${esc(concept)}"><small>${String(index + 1).padStart(2, "0")}</small><strong>${esc(concept)}</strong><span>${index === 0 ? "input state" : index === scene.concepts.length - 1 ? "result state" : "transformed state"}</span></div>`).join('<i class="flow-arrow"></i>')}
    </div>`;
  }

  function irDiagram(scene) {
    return `<div class="ir-diagram">
      <section><header>BEFORE</header>${scene.concepts.slice(0, 2).map((concept, index) => `<div class="ir-op ${progressClass(index)}"><code>%${index}</code><span>${esc(concept)}</span></div>`).join("")}</section>
      <div class="pass-gate ${state.step >= 3 ? "revealed current" : "future"}"><b>PASS</b><span>legality + rewrite</span></div>
      <section><header>AFTER</header>${scene.concepts.slice(2).map((concept, local) => `<div class="ir-op ${progressClass(local + 2)}"><code>%${local + 2}</code><span>${esc(concept)}</span></div>`).join("")}</section>
    </div>`;
  }

  function memoryDiagram(scene) {
    return `<div class="memory-diagram"><header><span>OWNER / SOURCE</span><span>NEXT RESIDENCE</span></header>${scene.concepts.map((concept, index) => `<div class="memory-row ${progressClass(index)}"><div><small>${index < 2 ? "HOST" : "DEVICE"}</small><strong>${esc(concept)}</strong></div><i><span></span></i><div><small>${index < 2 ? "RUNTIME" : "COMPUTE"}</small><strong>${esc(index === scene.concepts.length - 1 ? "consumer-visible result" : scene.concepts[index + 1])}</strong></div></div>`).join("")}</div>`;
  }

  function tileCells(step, offset) {
    return Array.from({ length: 64 }, (_, index) => {
      const row = Math.floor(index / 8);
      const col = index % 8;
      const threshold = Math.max(1, step) + offset;
      const active = step > 0 && ((row + col + offset) % Math.max(2, 6 - step) === 0 || (row < threshold && col < threshold));
      return `<i class="${active ? "active" : ""}"></i>`;
    }).join("");
  }

  function tileDiagram(scene) {
    return `<div class="tile-diagram"><div class="matrix"><strong>A / INPUT</strong><div>${tileCells(state.step, 0)}</div></div><div class="tile-memory">${scene.concepts.map((concept, index) => `<span class="${progressClass(index)}">${esc(concept)}</span>`).join("")}</div><div class="matrix"><strong>B / OUTPUT</strong><div>${tileCells(state.step, 2)}</div></div></div>`;
  }

  function timelineDiagram(scene) {
    const lanes = ["HOST", "COPY / COMPILE", "QUEUE", "GPU"];
    return `<div class="timeline-diagram">${lanes.map((lane, index) => `<div class="timeline-row ${progressClass(index)}"><strong>${lane}</strong><div><span style="--offset:${index};--span:${4 - index}">${esc(scene.concepts[index])}</span></div></div>`).join("")}<i class="playhead" style="--step:${state.step}"></i></div>`;
  }

  function rooflineDiagram(scene) {
    const pointX = 64 + state.step * 45;
    const pointY = 185 - state.step * 32;
    return `<div class="roofline-diagram"><svg viewBox="0 0 360 230" role="img" aria-label="Arithmetic intensity roofline model"><path class="axis" d="M42 16V194H340"/><path class="grid" d="M42 150H340M42 106H340M42 62H340M112 16V194M182 16V194M252 16V194"/><path class="memory-roof" d="M48 188L230 42"/><path class="compute-roof" d="M230 42H334"/><circle class="roof-point" cx="${pointX}" cy="${pointY}" r="8"/><text x="110" y="218">Arithmetic intensity = FLOP / byte</text><text x="235" y="34">compute roof</text><text x="67" y="154" transform="rotate(-38 67 154)">bandwidth roof</text></svg><div class="roof-legend">${scene.concepts.map((concept, index) => `<span class="${progressClass(index)}">${esc(concept)}</span>`).join("")}</div></div>`;
  }

  function backendDiagram(scene) {
    return `<div class="backend-diagram"><div class="hal-core"><small>COMMON CONTRACT</small><strong>${esc(scene.concepts[0])}</strong><span>MAX HAL</span></div><div class="backend-fan">${scene.concepts.slice(1).map((concept, local) => `<div class="${progressClass(local + 1)}"><small>${local === 0 ? "PLUGIN" : local === 1 ? "QUEUE" : "TARGET"}</small><strong>${esc(concept)}</strong></div>`).join("")}</div><p>Vendor/native calls remain behind the marked implementation boundary.</p></div>`;
  }

  function stateDiagram(scene) {
    return `<div class="state-diagram">${scene.concepts.map((concept, index) => `<div class="state-node ${progressClass(index)}"><b>${index + 1}</b><strong>${esc(concept)}</strong><span>${index === state.step - 1 ? "current state" : index < state.step ? "completed" : "pending"}</span></div>`).join("")}</div>`;
  }

  function mathView(scene) {
    return `<div class="math-view"><section><small>MODEL</small><code>${esc(scene.formula)}</code></section><section><small>STEP INPUT</small><strong>${esc(state.step ? scene.concepts[state.step - 1] : scene.concepts[0])}</strong><p>${esc(scene.role)}</p></section><section><small>INTERPRETATION</small><p>${state.step === 0 ? "Start with the contract and identify every symbol before applying the transformation." : `Build ${state.step} exposes the state associated with ${esc(scene.concepts[state.step - 1])}. Source inspection proves structure; target timing remains a separate measurement.`}</p></section></div>`;
  }

  function evidenceView(scene) {
    return `<div class="evidence-view"><div class="evidence-stage source"><b>SOURCE</b><span>Repository lines and object contracts visible in this revision.</span></div><i></i><div class="evidence-stage boundary"><b>BOUNDARY</b><span>Native compiler, runtime plugin, vendor API, or serialized artifact implementation.</span></div><i></i><div class="evidence-stage gpu-lab"><b>GPU-LAB</b><span>Latency, occupancy, bandwidth, overlap, and achieved throughput require measurement.</span></div><p class="evidence-current">Current scene: <strong>${esc(scene.evidence)}</strong></p></div>`;
  }

  function systemDiagram(scene) {
    if (["ir"].includes(scene.diagram)) return irDiagram(scene);
    if (["memory", "buffers", "cache", "ownership"].includes(scene.diagram)) return memoryDiagram(scene);
    if (["tile"].includes(scene.diagram)) return tileDiagram(scene);
    if (["timeline", "schedule", "queue"].includes(scene.diagram)) return timelineDiagram(scene);
    if (scene.diagram === "roofline") return rooflineDiagram(scene);
    if (["backend", "dispatch"].includes(scene.diagram)) return backendDiagram(scene);
    if (["state", "tree"].includes(scene.diagram)) return stateDiagram(scene);
    return flowDiagram(scene, scene.diagram);
  }

  function renderVisualization(scene) {
    $("#diagram-kind").textContent = scene.diagram.toUpperCase();
    $("#diagram-title").textContent = state.view === "system" ? scene.unitTitle : state.view === "math" ? "Mathematical model" : "Evidence boundary";
    $("#visual-canvas").innerHTML = state.view === "system" ? systemDiagram(scene) : state.view === "math" ? mathView(scene) : evidenceView(scene);
    $$('[data-view]').forEach((button) => button.classList.toggle("active", button.dataset.view === state.view));
  }

  function renderInspector(scene) {
    const focusIndex = state.selectedLine ?? Math.max(0, state.step * 2 - 1);
    const focus = scene.lines[focusIndex];
    $("#step-number").textContent = String(state.step).padStart(2, "0");
    $("#step-title").textContent = state.step ? scene.concepts[state.step - 1] : "Contract overview";
    $("#step-summary").textContent = state.step ? scene.explanations[focusIndex] : scene.role;
    $("#object-ledger").innerHTML = scene.concepts.map((concept, index) => `<div class="object-row ${progressClass(index)}"><strong>${esc(concept)}</strong><span>${index < state.step ? "materialized" : index === state.step ? "next" : "pending"}</span></div>`).join("");
    $("#metric-grid").innerHTML = [
      [`${state.step * 2}/8`, "source lines"],
      [`${state.step}/4`, "state changes"],
      [scene.evidence.includes("GPU-LAB") ? "LAB" : "SRC", "evidence mode"],
      [scene.diagram.toUpperCase(), "visual model"],
    ].map(([value, label]) => `<div class="metric"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");
    $("#formula").textContent = scene.formula;
    $("#boundary-note").textContent = scene.evidence.includes("GPU-LAB")
      ? "The implementation structure is source-visible. Performance, overlap, cycles, occupancy, and bandwidth are not measurements in this course artifact."
      : scene.evidence.includes("BOUNDARY")
        ? "The displayed source proves the handoff and contract. Native or vendor implementation details below that handoff are explicitly treated as a boundary."
        : "The current claim is grounded in the displayed repository source and current revision.";
    $("#focus-line").textContent = `L${focus.number}`;
  }

  function renderTransport(scene) {
    $("#transport-label").textContent = state.step ? scene.concepts[state.step - 1] : scene.unitTitle;
    $("#transport-count").textContent = `${state.scene + 1}/${scenes.length} · ${state.step}/4`;
    $("#play").textContent = state.playing ? "Ⅱ" : "▶";
    $("#play").title = state.playing ? "Pause" : "Play";
    $("#step-slider").value = String(state.step);
    $$("[data-build]").forEach((button) => button.classList.toggle("active", Number(button.dataset.build) === state.step));
  }

  function updateHash() {
    history.replaceState(null, "", `#scene-${current().id}/build-${state.step}`);
  }

  function render() {
    const scene = current();
    document.documentElement.style.setProperty("--accent", `#${course.part.color}`);
    $("#part-title").textContent = course.part.title;
    $("#part-subtitle").textContent = course.part.subtitle;
    $("#scene-title").textContent = scene.title;
    renderRail();
    renderSource(scene);
    renderVisualization(scene);
    renderInspector(scene);
    renderTransport(scene);
    updateHash();
  }

  function stop() {
    state.playing = false;
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  function move(delta) {
    stop();
    if (delta > 0) {
      if (state.step < 4) state.step += 1;
      else if (state.scene < scenes.length - 1) {
        state.scene += 1;
        state.step = 0;
        state.selectedLine = null;
      }
    } else if (state.step > 0) state.step -= 1;
    else if (state.scene > 0) {
      state.scene -= 1;
      state.step = 4;
      state.selectedLine = null;
    }
    render();
  }

  function togglePlay() {
    if (state.playing) {
      stop();
      renderTransport(current());
      return;
    }
    state.playing = true;
    state.timer = setInterval(() => {
      if (state.scene === scenes.length - 1 && state.step === 4) {
        stop();
        renderTransport(current());
        return;
      }
      if (state.step < 4) state.step += 1;
      else {
        state.scene += 1;
        state.step = 0;
        state.selectedLine = null;
      }
      render();
    }, 1700);
    renderTransport(current());
  }

  function parseHash() {
    const match = location.hash.match(/^#scene-([^/]+)\/build-(\d)$/);
    if (!match) return;
    const sceneIndex = scenes.findIndex((scene) => scene.id === match[1]);
    if (sceneIndex >= 0) state.scene = sceneIndex;
    state.step = Math.max(0, Math.min(4, Number(match[2])));
  }

  $("#previous").addEventListener("click", () => move(-1));
  $("#next").addEventListener("click", () => move(1));
  $("#play").addEventListener("click", togglePlay);
  $("#reset").addEventListener("click", () => {
    stop();
    state.step = 0;
    state.selectedLine = null;
    render();
  });
  $("#step-slider").addEventListener("input", (event) => {
    stop();
    state.step = Number(event.target.value);
    render();
  });
  $$("[data-build]").forEach((button) => button.addEventListener("click", () => {
    stop();
    state.step = Number(button.dataset.build);
    render();
  }));
  $$("[data-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    renderVisualization(current());
  }));
  addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.key === "ArrowRight" || event.key === "PageDown") move(1);
    if (event.key === "ArrowLeft" || event.key === "PageUp") move(-1);
    if (event.key === " ") {
      event.preventDefault();
      togglePlay();
    }
  });

  parseHash();
  render();
})();
