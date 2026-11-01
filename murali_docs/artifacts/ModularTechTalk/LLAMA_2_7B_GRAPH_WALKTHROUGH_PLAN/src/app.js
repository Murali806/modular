"use strict";
(() => {
  const data = JSON.parse(document.querySelector("#guide-data").textContent);
  const cells = new Map(data.cells.map(c => [c.id, c]));
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const state = Object.fromEntries(Object.keys(data.stages).map(key => [key, 0]));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const behavior = reducedMotion ? "auto" : "smooth";
  let activeChapter = "1";
  let toastTimer;
  const toast = text => {
    const element = document.querySelector("#toast");
    element.textContent = text;
    element.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => element.classList.remove("visible"), 2800);
  };

  function highlight(id, start, end) {
    const cell = document.getElementById(`code-${id}`);
    cell.querySelectorAll(".code-line").forEach(line => {
      const n = Number(line.dataset.line);
      line.classList.toggle("highlight", n >= start && n <= end);
    });
    cell.querySelectorAll(".annotation").forEach(button => button.classList.toggle("active", Number(button.dataset.start) === start));
    const pane = cell.querySelector(".source-code");
    const line = cell.querySelector(`.code-line[data-line="${start}"]`);
    if (line) pane.scrollTop += line.getBoundingClientRect().top - pane.getBoundingClientRect().top - 28;
  }

  function openCell(id, annotation = null) {
    const cell = document.getElementById(`code-${id}`);
    if (!cell) return;
    cell.open = true;
    document.querySelectorAll(".code-cell.targeted").forEach(e => e.classList.remove("targeted"));
    cell.classList.add("targeted");
    cell.scrollIntoView({behavior, block:"start"});
    if (annotation !== null && cells.get(id).notes[annotation]) {
      const n = cells.get(id).notes[annotation];
      requestAnimationFrame(() => highlight(id, n.start, n.end));
    }
  }

  function renderPanel(kind, item) {
    const title = `<h4>${escape(item.title)}</h4>`;
    if (kind === "finalize") return title + `<div class="state-pair"><div><small>BEFORE THIS STEP</small><p>${escape(item.before)}</p></div><div><small>AFTER THIS STEP</small><p>${escape(item.after)}</p><code>${escape(item.field)}</code></div></div>`;
    if (kind === "compiler") return title + `<div class="compiler-contract"><div><small>INPUT</small><p>${escape(item.input)}</p></div><div><small>TRANSFORMATION</small><p>${escape(item.work)}</p></div><div><small>OUTPUT</small><p>${escape(item.output)}</p></div></div>`;
    if (kind === "cache") return title + `<div class="flow">${item.path.map(p => `<span class="flow-node">${escape(p)}</span>`).join('<span aria-hidden="true" class="flow-arrow">→</span>')}</div><p>${escape(item.text)}</p>`;
    const tokens = item.tokens.map((token, i) => `<span class="token ${i < item.split ? "sequence-a" : "sequence-b"}">${token}</span>`).join("");
    return title + `<div class="buffer-row"><strong>tokens</strong><div class="token-strip">${tokens}</div><code>int64 [${item.total}]</code></div><div class="buffer-row"><strong>row offsets</strong><code>[${item.offsets.join(", ")}]</code><code>uint32 [3]</code></div><p>${escape(item.text)}</p>`;
  }

  function updateExplorer(kind, index) {
    const items = data.stages[kind];
    index = Math.max(0, Math.min(items.length - 1, index));
    state[kind] = index;
    const item = items[index];
    document.querySelector(`#panel-${kind}`).innerHTML = renderPanel(kind, item);
    document.querySelectorAll(`[data-explorer="${kind}"]`).forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.step) === index)));
    document.querySelector(`[data-position="${kind}"]`).textContent = `${index + 1} / ${items.length}`;
    document.querySelector(`[data-previous="${kind}"]`).disabled = index === 0;
    document.querySelector(`[data-next="${kind}"]`).disabled = index === items.length - 1;
    document.querySelector(`[data-explorer-source="${kind}"]`).dataset.target = item.cell || "ragged";
  }
  for (const kind of Object.keys(state)) updateExplorer(kind, 0);

  async function copySource(id) {
    const text = cells.get(id).code;
    try {
      await navigator.clipboard.writeText(text);
      toast("Complete source copied");
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.cssText = "position:fixed;left:-10000px;top:0";
      document.body.append(textarea);
      textarea.select();
      const success = document.execCommand("copy");
      textarea.remove();
      toast(success ? "Complete source copied" : "Copy unavailable here; use Download file");
    }
  }

  const dialog = document.querySelector("#source-dialog");
  const select = document.querySelector("#file-select");
  const fullFile = document.querySelector("#full-file");
  const search = document.querySelector("#file-search");
  let matches = [];
  let matchIndex = -1;

  function scrollFileLine(number) {
    const row = fullFile.querySelector(`[data-line="${number}"]`);
    if (row) fullFile.scrollTop += row.getBoundingClientRect().top - fullFile.getBoundingClientRect().top - 30;
  }

  function renderFile(file, line = 1) {
    const source = data.files[file];
    select.value = file;
    const fragment = document.createDocumentFragment();
    const lines = source.text.split("\n");
    if (lines.at(-1) === "") lines.pop();
    lines.forEach((text, index) => {
      const row = document.createElement("span");
      row.className = "code-line";
      row.dataset.line = index + 1;
      const number = document.createElement("span");
      number.className = "line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = index + 1;
      const code = document.createElement("span");
      code.className = "line-text";
      code.textContent = text;
      row.append(number, code);
      fragment.append(row);
    });
    fullFile.querySelector("code").replaceChildren(fragment);
    document.querySelector("#file-provenance").textContent = `${lines.length} lines · ${data.revision.slice(0,12)} · SHA-256 ${source.sha256}`;
    const download = document.querySelector("#download-full");
    download.href = source.snapshot;
    download.download = file.split("/").at(-1);
    search.value = "";
    matches = [];
    matchIndex = -1;
    document.querySelector("#find-count").textContent = "";
    requestAnimationFrame(() => {
      fullFile.scrollTop = 0;
      scrollFileLine(line);
      const row = fullFile.querySelector(`[data-line="${line}"]`);
      if (row && line > 1) row.classList.add("highlight");
    });
  }

  function openFile(file, line = 1) {
    if (!data.files[file]) return;
    if (!dialog.open) dialog.showModal();
    renderFile(file, line);
  }

  function findInFile(next = false) {
    const term = search.value.toLocaleLowerCase();
    const rows = [...fullFile.querySelectorAll(".code-line")];
    rows.forEach(row => row.classList.remove("highlight", "current-match", "match"));
    if (!next) {
      matches = term ? rows.filter(row => row.querySelector(".line-text").textContent.toLocaleLowerCase().includes(term)) : [];
      matchIndex = matches.length ? 0 : -1;
    } else if (matches.length) matchIndex = (matchIndex + 1) % matches.length;
    matches.forEach(row => row.classList.add("match"));
    document.querySelector("#find-count").textContent = term ? (matches.length ? `${matchIndex + 1} / ${matches.length}` : "No matches") : "";
    if (matchIndex >= 0) {
      matches[matchIndex].classList.add("current-match");
      scrollFileLine(Number(matches[matchIndex].dataset.line));
    }
  }
  search.addEventListener("input", () => findInFile());
  search.addEventListener("keydown", e => {if (e.key === "Enter") {e.preventDefault(); findInFile(true);}});
  document.querySelector("#find-next").addEventListener("click", () => findInFile(true));
  select.addEventListener("change", () => renderFile(select.value));
  document.querySelector("#open-explorer").addEventListener("click", () => openFile(select.value));
  document.querySelector("#close-explorer").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });

  document.addEventListener("click", event => {
    const target = event.target.closest("button,a");
    if (!target) return;
    if (target.dataset.explorer) updateExplorer(target.dataset.explorer, Number(target.dataset.step));
    if (target.dataset.previous) updateExplorer(target.dataset.previous, state[target.dataset.previous] - 1);
    if (target.dataset.next) updateExplorer(target.dataset.next, state[target.dataset.next] + 1);
    if (target.dataset.explorerSource) {
      const kind = target.dataset.explorerSource;
      const item = data.stages[kind][state[kind]];
      openCell(item.cell || "ragged", item.note ?? null);
    }
    if (target.classList.contains("annotation")) highlight(target.dataset.cell, Number(target.dataset.start), Number(target.dataset.end));
    if (target.dataset.copy) copySource(target.dataset.copy);
    if (target.dataset.file) {
      event.preventDefault();
      openFile(target.dataset.file, Number(target.dataset.fileLine || 1));
    }
    const href = target.getAttribute("href");
    if (href?.startsWith("#code-")) openCell(href.slice(6));
  });

  function updateChapter() {
    const chapters = [...document.querySelectorAll(".chapter")];
    let selected = chapters[0];
    for (const chapter of chapters) if (chapter.getBoundingClientRect().top <= 170) selected = chapter;
    activeChapter = selected.dataset.chapter;
    document.querySelectorAll("[data-nav]").forEach(a => {
      if (a.dataset.nav === activeChapter) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    });
  }
  let scrollPending = false;
  window.addEventListener("scroll", () => {
    if (!scrollPending) {scrollPending = true; requestAnimationFrame(() => {updateChapter(); scrollPending = false;});}
  }, {passive:true});
  const sidebar = document.querySelector("#chapter-sidebar");
  const sidebarToggle = document.querySelector("#toggle-sidebar");
  const sidebarPreference = "max-llama-graph-guide.sidebar-collapsed";
  function setSidebarExpanded(expanded, remember = false) {
    if (!expanded && sidebar.contains(document.activeElement)) sidebarToggle.focus();
    sidebar.hidden = !expanded;
    document.querySelector(".page-shell").classList.toggle("sidebar-collapsed", !expanded);
    sidebarToggle.setAttribute("aria-expanded", String(expanded));
    sidebarToggle.querySelector("span").textContent = expanded ? "Hide sidebar" : "Show sidebar";
    sidebarToggle.title = expanded ? "Hide sidebar" : "Show sidebar";
    if (remember) {
      try { localStorage.setItem(sidebarPreference, String(!expanded)); } catch { /* Keep the toggle working when storage is unavailable. */ }
    }
    requestAnimationFrame(updateChapter);
  }
  let sidebarCollapsed = false;
  try { sidebarCollapsed = localStorage.getItem(sidebarPreference) === "true"; } catch { /* Default to visible navigation. */ }
  sidebarToggle.hidden = false;
  setSidebarExpanded(!sidebarCollapsed);
  sidebarToggle.addEventListener("click", () => setSidebarExpanded(sidebar.hidden, true));
  updateChapter();
  document.querySelector("#expand-chapter").addEventListener("click", () => {
    updateChapter();
    document.querySelectorAll(`#chapter-${activeChapter} .code-cell`).forEach(cell => cell.open = true);
    toast(`Chapter ${activeChapter} source expanded`);
  });
  document.querySelector("#collapse-code").addEventListener("click", () => document.querySelectorAll(".code-cell").forEach(cell => cell.open = false));
  function readHash() {
    if (location.hash.startsWith("#code-")) openCell(location.hash.slice(6));
    if (location.hash === "#qna-1") {
      const qna = document.getElementById("qna-1");
      qna.open = true;
      qna.scrollIntoView({behavior, block:"start"});
    }
  }
  window.addEventListener("hashchange", readHash);
  readHash();

  let printState = [];
  window.addEventListener("beforeprint", () => {
    printState = [...document.querySelectorAll("details")].map(e => [e, e.open]);
    printState.forEach(([e]) => e.open = true);
  });
  window.addEventListener("afterprint", () => printState.forEach(([e, open]) => e.open = open));
  document.querySelector("#print-guide").addEventListener("click", () => window.print());
})();
