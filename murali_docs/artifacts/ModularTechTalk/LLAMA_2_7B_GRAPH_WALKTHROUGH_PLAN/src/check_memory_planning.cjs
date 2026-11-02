#!/usr/bin/env node
"use strict";

const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const auditRenderedContrast = require("./contrast.cjs");
const TASK = path.resolve(__dirname, "..");
const OUT = path.join(TASK, "validation");
const FILE = path.join(TASK, "link_htmls", "llama2_7b_memory_planning_and_memory_plan.html");
const pageURL = pathToFileURL(FILE).href;
const sourceChecks = JSON.parse(fs.readFileSync(path.join(TASK, "memory_planning_manifest.json"), "utf8"));

(async () => {
  fs.mkdirSync(OUT, {recursive: true});
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"]});
  const errors = [];
  const checks = [];
  const contrast = [];
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, reducedMotion: "reduce",
    permissions: ["clipboard-read", "clipboard-write"]});
  await context.route(/^https?:/, route => {errors.push("Unexpected external request: " + route.request().url()); return route.abort();});
  const watch = page => {
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => {if (message.type() === "error") errors.push(message.text());});
  };
  const audit = async (page, state) => {
    const result = await page.evaluate(auditRenderedContrast);
    contrast.push({state, ...result});
    assert.deepEqual(result.failures, [], "Rendered contrast: " + state);
  };
  const calc = page => page.locator("#memory-budget").evaluate(node => JSON.parse(node.dataset.calculation));
  const open = async (page, id) => {
    await page.evaluate(id => {
      const target = document.getElementById(id);
      target.open = true;
      let parent = target.parentElement;
      while (parent) {if (parent.tagName === "DETAILS") parent.open = true; parent = parent.parentElement;}
      location.hash = id;
    }, id);
    await page.waitForFunction(id => document.getElementById(id).open, id);
  };
  let page;
  try {
    const stack = await context.newPage();
    watch(stack);
    await stack.goto(pathToFileURL(path.join(TASK, "max_serve_llama2_7b_startup_call_stack.html")).href);
    const link = stack.locator('a.tree-doc-link[href*="memory_planning_and_memory_plan"]');
    assert.equal(await link.count(), 1);
    const treeRow = await link.evaluate(node => node.parentElement.textContent);
    assert.ok(treeRow.startsWith("        |   |   +-- "));
    const neighborRows = await link.evaluate(node => [node.parentElement.previousElementSibling.textContent, node.parentElement.nextElementSibling.textContent]);
    assert.deepEqual(neighborRows, ["        |   |", "        |   |"]);
    const newPage = context.waitForEvent("page");
    await link.click();
    page = await newPage;
    watch(page);
    await page.waitForLoadState("load");
    assert.equal(page.url(), pageURL + "#input-origins");
    await stack.close();
    checks.push("Call-stack link opens the new page at input origins; connector-only spacer rows preserve the tree");

    const payload = await page.locator("#memory-data").evaluate(node => JSON.parse(node.textContent));
    assert.equal(payload.revision, sourceChecks.revision);
    assert.equal(await page.locator("details.qna").count(), 9);
    assert.equal(await page.locator("details.code-cell").count(), 23);
    assert.equal(await page.locator("details.code-cell[open]").count(), 0);
    assert.equal(await page.locator('input[type="radio"]').count(), 0);
    assert.equal(await page.evaluate(() => getComputedStyle(document.body).backgroundColor), "rgb(27, 29, 33)");
    assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), "dark");
    assert.ok(await page.locator("#input-origins").evaluate(node => node.open));
    await page.locator("#input-origins > summary").focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#input-origins").evaluate(node => node.open), false);
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#input-origins").evaluate(node => node.open), true);
    await audit(page, "Desktop / input origins / keyboard focus");
    await page.locator("#memory-planning").scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(OUT, "memory_planning_overview.png")});
    checks.push("Dark theme, nine nested QnA cells, 23 collapsed code cells, keyboard expansion and offline deep links");

    const originalWidth = await page.locator("main").evaluate(node => node.clientWidth);
    await page.locator("#sidebar-toggle").click();
    assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "false");
    assert.ok(await page.locator("main").evaluate(node => node.clientWidth) > originalWidth);
    await page.locator("#sidebar-toggle").focus();
    await page.keyboard.press("Space");
    assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "true");
    checks.push("Contents sidebar collapses and expands with pointer and keyboard");

    await open(page, "memory-budget");
    let c = await calc(page);
    assert.equal(c.perToken, 524288);
    assert.equal(c.pageBytes, 67108864);
    assert.equal(c.pagesPerRequest, 32);
    assert.equal(c.initial, 9715947126);
    assert.equal(c.committed, 8657043456);
    assert.equal(c.footprint, 22133919728);
    assert.equal(c.poolSlots, 128);
    assert.equal(c.fullContexts, 4);
    assert.equal(c.plannedLength, 4096);
    assert.equal(c.error, "");
    await audit(page, "Desktop / budget baseline");
    await page.locator('[data-animation="budget"]').screenshot({path: path.join(OUT, "memory_budget_waterfall.png"), style: ".page-header { visibility: hidden !important; }"});

    await page.locator("#allocation-cap").selectOption("4");
    c = await calc(page);
    assert.equal(c.committed, 4 * 1024 ** 3);
    assert.equal(c.poolSlots, 63);
    assert.equal(c.fullContexts, 1);
    assert.equal(c.candidate, 4);
    assert.equal(c.plannedLength, 4096);
    assert.match(await page.locator("#budget-status").innerText(), /candidate batch remains 4/);
    await audit(page, "Desktop / allocation-limited cache");
    await page.locator("#allocation-cap").selectOption("1");
    c = await calc(page);
    assert.equal(c.poolSlots, 15);
    assert.equal(c.plannedLength, 1920);

    await page.locator("#allocation-cap").selectOption("0");
    await open(page, "kv-pages");
    await page.locator("#cache-width").selectOption("4");
    c = await calc(page);
    assert.equal(c.perToken, 1048576);
    assert.equal(c.pageBytes, 134217728);
    assert.equal(c.required, 17314086912);
    assert.equal(c.committed, c.capped);
    assert.equal(c.fullContexts, 2);
    await page.locator("#free-memory").selectOption("16");
    await page.locator("#utilization").selectOption("0.8");
    c = await calc(page);
    assert.match(c.error, /null block and one request page/);
    assert.equal(await page.locator('[data-value="committed"]').innerText(), "No plan");
    await audit(page, "Desktop / cache-too-small error");

    await page.locator("#free-memory").selectOption("24");
    await page.locator("#utilization").selectOption("0.9");
    await page.locator("#cache-width").selectOption("2");
    await page.locator("#candidate-batch").fill("512");
    c = await calc(page);
    assert.equal(c.candidate, 512);
    assert.equal(c.payload, 1024 * 1024 ** 3);
    assert.ok(c.fullContexts < c.candidate);
    assert.ok(c.committed <= c.capped);
    await page.locator("#candidate-batch").fill("0");
    assert.ok(await page.locator("#candidate-batch").evaluate(node => !node.validity.valid));
    await page.locator("#candidate-batch").fill("4");
    await page.locator("#context-length").fill("2048");
    c = await calc(page);
    assert.equal(c.pagesPerRequest, 16);
    assert.equal(c.required, 4362076160);
    await page.locator("#context-length").fill("4096");
    await audit(page, "Desktop / KV pages restored baseline");
    await page.locator("#kv-pages").screenshot({path: path.join(OUT, "llama2_kv_pages.png"), style: ".page-header { visibility: hidden !important; }"});
    checks.push("Page-width/context arithmetic, committed budget, null-block capacity, explicit allocation cap and 512-request demand are consistent; too-small caches have an error state");

    for (const [kind, section, frames] of [["provenance", "input-origins", 4], ["budget", "memory-budget", 6], ["handoff", "parent-worker-handoff", 8]]) {
      await open(page, section);
      const animation = page.locator('[data-animation="' + kind + '"]');
      assert.equal(await animation.getAttribute("data-playing"), "false");
      assert.equal(await animation.locator('[data-action="previous"]').isDisabled(), true);
      await animation.locator('[data-action="next"]').click();
      assert.equal(await animation.locator("[data-progress]").innerText(), "2 / " + frames);
      await animation.locator('[data-action="previous"]').click();
      assert.equal(await animation.locator("[data-progress]").innerText(), "1 / " + frames);
      await animation.locator('[data-action="play"]').click();
      assert.equal(await animation.getAttribute("data-playing"), "true");
      await animation.locator('[data-action="pause"]').click();
      assert.equal(await animation.getAttribute("data-playing"), "false");
      await animation.locator('[data-action="replay"]').click();
      assert.equal(await animation.getAttribute("data-playing"), "true");
      await page.locator("#" + section + " > summary").click();
      await page.waitForFunction(kind => document.querySelector('[data-animation="' + kind + '"]').dataset.playing === "false", kind);
      await open(page, section);
      for (let index = 1; index < frames; index++) await animation.locator('[data-action="next"]').click();
      assert.equal(await animation.locator('[data-action="next"]').isDisabled(), true);
      await audit(page, "Desktop / " + kind + " final frame");
    }
    const provenance = page.locator('[data-animation="provenance"]');
    await provenance.locator('[data-action="replay"]').click();
    await page.waitForTimeout(3150);
    assert.equal(await provenance.locator("[data-progress]").innerText(), "2 / 4");
    await provenance.locator('[data-action="pause"]').click();
    await page.locator('[data-animation="handoff"]').screenshot({path: path.join(OUT, "memory_plan_worker_handoff.png"), style: ".page-header { visibility: hidden !important; }"});
    assert.equal(await page.locator('[data-bar="committed"]').evaluate(node => getComputedStyle(node).transitionDuration), "0s");
    checks.push("All three diagrams have bounded steps, play/pause/replay, real frame advancement and pause on collapse; reduced motion removes transitions");

    await open(page, "memory-budget");
    await page.locator("#code-memory > summary").click();
    const original = payload.references.memory.code;
    assert.deepEqual(await page.locator("#code-memory .line-text").allTextContents(), original.split("\n").slice(0, -1));
    await page.locator('#code-memory [data-copy="memory"]').click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), original);
    await page.locator("#expand-explanations").click();
    for (const key of ["memory", "memory_call", "kv_dtype", "generic_blocks", "default_batch"]) {
      await page.locator('.source-directory [data-ref="' + key + '"]').click();
      assert.ok(await page.locator("#source-dialog").evaluate(node => node.open));
      const ref = payload.references[key];
      const file = payload.files[ref.file];
      assert.equal(await page.locator("#source-file .line-text").evaluateAll(nodes => nodes.map(node => node.textContent).join("\n") + "\n"), file.text);
      assert.equal(await page.locator("#source-file .definition-line").count(), ref.end - ref.start + 1);
      assert.equal(await page.locator("#source-file .selected-line").getAttribute("data-line"), String(ref.line));
      if (key === "memory") {
        await page.locator("#copy-source").click();
        assert.equal(await page.evaluate(() => navigator.clipboard.readText()), ref.code);
        await page.locator("#source-search").fill("max_batch_size");
        assert.match(await page.locator("#source-match-count").innerText(), /^1 \/ /);
        await page.locator("#source-next").click();
        assert.match(await page.locator("#source-match-count").innerText(), /^2 \/ /);
        await page.locator("#source-search").fill("unfindable-memory-term");
        assert.equal(await page.locator("#source-match-count").innerText(), "No matches");
        await page.locator("#source-search").fill("");
        await audit(page, "Desktop / full original source and selected line");
        await page.screenshot({path: path.join(OUT, "memory_planning_source_explorer.png")});
      }
      await page.keyboard.press("Escape");
      assert.equal(await page.locator("#source-dialog").evaluate(node => node.open), false);
    }
    checks.push("Source explorer preserves complete files/definitions and call-site lines; exact copying, searching, next/no-match and Escape work offline");

    await page.locator("#collapse-explanations").click();
    assert.equal(await page.locator("details.code-cell[open]").count(), 0);
    assert.equal(await page.locator("details.qna[open]").count(), 1);
    for (const width of [390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.goto(pageURL + "#memory-budget");
      await page.reload();
      assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "false");
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Budget overflow at " + width);
      await page.locator("#expand-explanations").click();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Expanded overflow at " + width);
      assert.ok(await page.locator(".qna > summary").evaluateAll(nodes => nodes.every(node =>
        node.querySelector(".summary-main").getBoundingClientRect().width >= node.getBoundingClientRect().width * 0.7)), "Mobile section titles are cramped");
      await audit(page, "Mobile / expanded explanations / " + width);
      await page.locator("#memory-budget").screenshot({path: path.join(OUT, "memory_budget_mobile_" + width + ".png"), style: ".page-header { visibility: hidden !important; }"});
      await page.locator("#memory-budget > summary").evaluate(node => node.scrollIntoView({block: "start"}));
      await page.screenshot({path: path.join(OUT, "memory_planning_mobile_" + width + ".png")});
      await page.locator("#code-memory > summary").click();
      assert.ok(await page.locator("#code-memory pre").evaluate(node => node.scrollWidth > node.clientWidth));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Code overflow at " + width);
      await page.locator('.source-directory [data-ref="generic_blocks"]').click();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Source dialog overflow at " + width);
      await audit(page, "Mobile / complete source / " + width);
      await page.keyboard.press("Escape");
      await page.locator("#sidebar-toggle").click();
      assert.equal(await page.locator("#sidebar-toggle").getAttribute("aria-expanded"), "true");
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    }
    checks.push("390px/320px layouts, expanded diagrams/tables, code scrolling, contents toggle and source dialog fit without document overflow");

    const fallbackContext = await browser.newContext({javaScriptEnabled: false, viewport: {width: 390, height: 844}});
    const fallback = await fallbackContext.newPage();
    watch(fallback);
    await fallback.goto(pageURL);
    assert.equal(await fallback.locator(".page-tools").isVisible(), false);
    assert.equal(await fallback.locator("#page-grid .page-block").count(), 32);
    await fallback.locator("#input-origin-table > summary").click();
    assert.ok(await fallback.locator("#input-origin-table").evaluate(node => node.open));
    await fallback.locator("#code-free_memory > summary").click();
    assert.deepEqual(await fallback.locator("#code-free_memory .line-text").allTextContents(), payload.references.free_memory.code.split("\n").slice(0, -1));
    assert.ok((await fallback.locator('#code-free_memory a[download]').getAttribute("href")).startsWith("../sources/"));
    assert.ok(await fallback.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await fallbackContext.close();
    checks.push("Without JavaScript, native nested disclosures, static page drawing, original code and local source files remain readable");

    assert.deepEqual(errors, []);
    const report = {checked_at: new Date().toISOString(), result: "PASS", browser: browser.version(),
      revision: payload.revision, page: path.relative(TASK, FILE),
      command: "NODE_PATH=/tmp/max-v4-playwright/node_modules node src/check_memory_planning.cjs",
      evidence: "Browser/document validation only; no MAX model execution or GPU measurement.", checks,
      source_targets: Object.keys(payload.references).length, complete_code_cells: payload.cells.length,
      viewports: [1440, 390, 320], offline_direct_file: true, errors};
    fs.writeFileSync(path.join(OUT, "memory_planning_browser_checks.json"), JSON.stringify(report, null, 2) + "\n");
    fs.writeFileSync(path.join(OUT, "memory_planning_contrast_checks.json"), JSON.stringify(contrast, null, 2) + "\n");
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    if (page && !page.isClosed()) await page.screenshot({path: path.join(OUT, "memory_planning_failure.png")}).catch(() => {});
    fs.writeFileSync(path.join(OUT, "memory_planning_browser_failure.json"), JSON.stringify({error: String(error), errors, checks, contrast}, null, 2));
    console.error(error);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
