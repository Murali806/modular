"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const task = path.resolve(__dirname, "..");
const url = pathToFileURL(path.join(task, "link_htmls/llama2_7b_memory_planning_and_memory_plan.html")).href;

(async () => {
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"]});
  const errors = [];
  const context = await browser.newContext({viewport: {width: 1600, height: 1100}, reducedMotion: "reduce"});
  await context.route(/^https?:/, route => {errors.push(route.request().url()); return route.abort();});
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  try {
    await page.goto(url + "#initialize-sequence");
    const cell = page.locator("#initialize-sequence");
    assert.equal(await cell.evaluate(node => node.previousElementSibling.id), "code-kv_construct");
    assert.ok(await cell.evaluate(node => node.open && node.closest("#arch-initialization").open));
    assert.equal(await cell.locator("g[data-step]").count(), 12);
    const payload = await page.locator("#memory-data").evaluate(node => JSON.parse(node.textContent));
    const sourceKeys = [];
    for (let step = 1; step <= 12; step++) {
      const link = cell.locator('g[data-step="' + step + '"] a[data-ref]');
      const key = await link.getAttribute("data-ref");
      await link.locator("text").first().click();
      assert.ok(await page.locator("#source-dialog").evaluate(node => node.open));
      assert.equal(await page.locator("#source-file .selected-line").getAttribute("data-line"), String(payload.references[key].line));
      assert.equal(await page.locator("#source-file .definition-line").count(), payload.references[key].end - payload.references[key].start + 1);
      sourceKeys.push(key);
      await page.keyboard.press("Escape");
    }
    const objectKeys = ["pipeline", "model", "hf", "kv-config", "kv-params", "arch"];
    const values = {};
    for (const key of objectKeys) {
      const record = page.locator("#init-object-" + key);
      await cell.locator('a.init-object[href="#init-object-' + key + '"]').first().click();
      await page.waitForFunction(key => document.getElementById("init-object-" + key).open, key);
      assert.equal(new URL(page.url()).hash, "#init-object-" + key);
      values[key] = JSON.parse(await record.locator("pre").innerText());
    }
    assert.equal(values.hf.hidden_size, 4096);
    assert.equal(values.hf.num_hidden_layers, 32);
    assert.equal(values.hf.torch_dtype, "float16");
    assert.equal(values.model.quantization_encoding, "bfloat16");
    assert.equal(values["kv-params"].dtype, "DType.bfloat16");
    assert.equal(values["kv-params"].head_dim, values.hf.hidden_size / values.hf.num_attention_heads);
    assert.equal(values["kv-params"].page_size, 128);
    assert.equal(values["kv-params"]["n_kv_heads_per_device (property)"], 32);
    assert.equal(values.arch.max_seq_len, values.model.max_length);
    assert.equal(values.arch.kv_params, "V1");
    await page.evaluate(() => {location.hash = "initialize-sequence";});
    await cell.locator(".init-sequence-scroll").evaluate(node => node.scrollLeft = 0);
    await cell.scrollIntoViewIfNeeded();
    await page.screenshot({path: "/tmp/llama_initialize_sequence_desktop.png", fullPage: false});
    await cell.locator(".init-sequence-scroll").screenshot({path: path.join(task, "validation/initialize_sequence.png")});
    for (const width of [390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.goto(url + "#initialize-sequence");
      await page.locator("#expand-explanations").click();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Document overflow at " + width);
      assert.ok(await cell.locator(".init-sequence-scroll").evaluate(node => node.scrollWidth > node.clientWidth));
      await cell.locator(".init-sequence-scroll").evaluate(node => node.scrollLeft = node.scrollWidth);
      await cell.locator('a.init-object[href="#init-object-arch"]').first().click();
      assert.ok(await page.locator("#init-object-arch").evaluate(node => node.open));
      await page.screenshot({path: "/tmp/llama_initialize_object_" + width + ".png"});
    }
    await page.setViewportSize({width: 1600, height: 1100});
    await page.goto(url + "#length-and-batch");
    const limits = page.locator("#length-and-batch");
    assert.equal(await limits.locator(".worked-example").count(), 6);
    const exampleLinks = limits.locator(':scope > .qna-body > .table-scroll a[href^="#length-example-"]');
    assert.equal(await exampleLinks.count(), 7);
    const exampleTargets = [];
    for (let index = 0; index < await exampleLinks.count(); index++) {
      const link = exampleLinks.nth(index);
      const target = (await link.getAttribute("href")).slice(1);
      await link.click();
      await page.waitForFunction(id => document.getElementById(id).open, target);
      assert.equal(new URL(page.url()).hash, "#" + target);
      assert.ok(await page.locator("#" + target).locator(":scope > .detail-body > ol > li").count() >= 4);
      exampleTargets.push(target);
    }
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.locator("#length-example-resident > summary").focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => !document.getElementById("length-example-resident").open);
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.getElementById("length-example-resident").open);
    for (const key of ["input_token_setting", "resident_token_setting", "batch_input_clamp", "oom_search_gate", "capacity_clamp"]) {
      await limits.locator('a[data-ref="' + key + '"]').first().click();
      assert.equal(await page.locator("#source-file .selected-line").getAttribute("data-line"), String(payload.references[key].line));
      await page.keyboard.press("Escape");
    }
    await page.evaluate(() => {location.hash = "length-example-pages";});
    await page.screenshot({path: "/tmp/llama_length_examples_desktop.png"});
    for (const width of [390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.goto(url + "#length-example-caps");
      await page.locator("#expand-explanations").click();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Worked-example overflow at " + width);
      await page.locator("#length-example-caps").evaluate(node => node.scrollIntoView({block: "start"}));
      await page.screenshot({path: path.join(task, "validation/length_examples_mobile_" + width + ".png")});
      await page.evaluate(() => window.scrollBy(0, 560));
      await page.screenshot({path: "/tmp/llama_length_steps_" + width + ".png"});
    }
    const fallbackContext = await browser.newContext({javaScriptEnabled: false, viewport: {width: 390, height: 844}});
    const fallback = await fallbackContext.newPage();
    await fallback.goto(url);
    await fallback.locator("#arch-initialization > summary").click();
    await fallback.locator("#initialize-sequence > summary").click();
    await fallback.locator("#init-object-kv-params > summary").click();
    assert.equal(await fallback.locator("#initialize-sequence svg g[data-step]").count(), 12);
    assert.ok((await fallback.locator("#init-object-kv-params pre").innerText()).includes("DType.bfloat16"));
    assert.ok((await fallback.locator('#initialize-sequence a[data-ref="kv_to_params_call"]').getAttribute("href")).startsWith("../sources/"));
    await fallbackContext.close();
    assert.deepEqual(errors, []);
    const report = {checked_at: new Date().toISOString(), result: "PASS", revision: payload.revision,
      command: "NODE_PATH=/tmp/max-v4-playwright/node_modules node src/check_initialize_and_limits.cjs",
      sequence_steps: 12, source_keys: sourceKeys, object_records: objectKeys,
      viewports: [1600, 390, 320], source_navigation: true, object_navigation: true,
      offline: true, javascript_disabled: true, no_document_overflow: true, errors};
    fs.writeFileSync(path.join(task, "validation/initialize_sequence_checks.json"), JSON.stringify(report, null, 2) + "\n");
    fs.writeFileSync(path.join(task, "validation/length_examples_checks.json"), JSON.stringify({
      checked_at: report.checked_at, result: "PASS", revision: payload.revision, command: report.command,
      worked_examples: 6, table_link_targets: exampleTargets, source_targets: Object.keys(payload.references).length,
      keyboard_disclosures: true, viewports: [1600, 390, 320], no_document_overflow: true, errors
    }, null, 2) + "\n");
    console.log(JSON.stringify(report, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
