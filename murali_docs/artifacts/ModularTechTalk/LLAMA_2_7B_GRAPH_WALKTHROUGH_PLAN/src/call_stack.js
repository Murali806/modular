"use strict";
(() => {
  const data = JSON.parse(document.querySelector("#stack-data").textContent);
  const dialog = document.querySelector("#source-dialog");
  const select = document.querySelector("#file-select");
  const full = document.querySelector("#full-file");
  const fileSearch = document.querySelector("#file-search");
  const treeSearch = document.querySelector("#tree-search");
  const treeRows = [...document.querySelectorAll(".tree-line")];
  let fileMatches = [], fileIndex = -1, treeMatches = [], treeIndex = -1;
  let toastTimer;
  const toast = message => {
    const node = document.querySelector("#toast");
    node.textContent = message;
    node.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("visible"), 2500);
  };
  function scrollSource(row) {
    if (row) full.scrollTop += row.getBoundingClientRect().top - full.getBoundingClientRect().top - 24;
  }
  function openFile(file, line = 1, ref = null) {
    const source = data.files[file];
    if (!source) return;
    select.value = file;
    const lines = source.text.replace(/\n$/, "").split("\n");
    const fragment = document.createDocumentFragment();
    lines.forEach((value, index) => {
      const row = document.createElement("span");
      row.className = "code-line";
      row.dataset.line = index + 1;
      const number = document.createElement("span");
      number.className = "line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = index + 1;
      const text = document.createElement("span");
      text.className = "line-text";
      text.textContent = value;
      row.append(number, text);
      fragment.append(row);
    });
    full.querySelector("code").replaceChildren(fragment);
    fileSearch.value = "";
    fileMatches = []; fileIndex = -1;
    document.querySelector("#find-count").textContent = "";
    const target = ref ? ref.kind + " · " + ref.label + " · " : "";
    document.querySelector("#file-provenance").textContent = target + lines.length + " complete lines · " + data.revision.slice(0, 12) + " · SHA-256 " + source.sha256;
    document.querySelector("#download-full").href = source.snapshot;
    if (!dialog.open) dialog.showModal();
    requestAnimationFrame(() => {
      full.scrollTop = 0;
      const row = full.querySelector('[data-line="' + line + '"]');
      if (row) row.classList.add("highlight");
      scrollSource(row);
    });
  }
  document.addEventListener("click", event => {
    const link = event.target.closest("a[data-ref], a[data-file]");
    if (!link) return;
    event.preventDefault();
    const ref = data.references[link.dataset.ref];
    if (ref) openFile(ref.file, ref.line, ref);
    else openFile(link.dataset.file);
  });
  select.addEventListener("change", () => openFile(select.value));
  document.querySelector("#close-explorer").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  function findFile(next = false) {
    const rows = [...full.querySelectorAll(".code-line")];
    rows.forEach(row => row.classList.remove("match", "current-match", "highlight"));
    const term = fileSearch.value.toLocaleLowerCase();
    if (!next) {
      fileMatches = term ? rows.filter(row => row.querySelector(".line-text").textContent.toLocaleLowerCase().includes(term)) : [];
      fileIndex = fileMatches.length ? 0 : -1;
    } else if (fileMatches.length) fileIndex = (fileIndex + 1) % fileMatches.length;
    fileMatches.forEach(row => row.classList.add("match"));
    document.querySelector("#find-count").textContent = !term ? "" : fileMatches.length ? (fileIndex + 1) + " / " + fileMatches.length : "No matches";
    if (fileIndex >= 0) {
      fileMatches[fileIndex].classList.add("current-match");
      scrollSource(fileMatches[fileIndex]);
    }
  }
  fileSearch.addEventListener("input", () => findFile());
  fileSearch.addEventListener("keydown", e => {if (e.key === "Enter") {e.preventDefault(); findFile(true);}});
  document.querySelector("#find-next").addEventListener("click", () => findFile(true));
  function findTree(next = false) {
    treeRows.forEach(row => row.classList.remove("stack-match", "stack-current"));
    const term = treeSearch.value.toLocaleLowerCase();
    if (!next) {
      treeMatches = term ? treeRows.filter(row => row.textContent.toLocaleLowerCase().includes(term)) : [];
      treeIndex = treeMatches.length ? 0 : -1;
    } else if (treeMatches.length) treeIndex = (treeIndex + 1) % treeMatches.length;
    treeMatches.forEach(row => row.classList.add("stack-match"));
    document.querySelector("#tree-count").textContent = !term ? "" : treeMatches.length ? (treeIndex + 1) + " / " + treeMatches.length + " matching lines" : "No matches";
    // Typing stays at the search field; Next/Enter navigates through matches.
    if (treeIndex >= 0) {
      treeMatches[treeIndex].classList.add("stack-current");
      if (next) treeMatches[treeIndex].scrollIntoView({block:"center", inline:"nearest", behavior:"auto"});
    }
  }
  treeSearch.addEventListener("input", () => findTree());
  treeSearch.addEventListener("keydown", e => {if (e.key === "Enter") {e.preventDefault(); findTree(true);}});
  document.querySelector("#tree-next").addEventListener("click", () => findTree(true));
  document.querySelector("#copy-tree").addEventListener("click", async () => {
    try {await navigator.clipboard.writeText(data.text); toast("All call-tree text copied");}
    catch {
      const area = document.createElement("textarea");
      area.value = data.text;
      area.style.cssText = "position:fixed;left:-10000px;top:0";
      document.body.append(area); area.select();
      const ok = document.execCommand("copy"); area.remove();
      toast(ok ? "All call-tree text copied" : "Use Download .txt to save the tree");
    }
  });
  let printState = [];
  window.addEventListener("beforeprint", () => {
    printState = [...document.querySelectorAll("details")].map(node => [node, node.open]);
    printState.forEach(([node]) => node.open = true);
  });
  window.addEventListener("afterprint", () => printState.forEach(([node, open]) => node.open = open));
})();
