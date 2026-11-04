"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const auditRenderedContrast = require("./contrast.cjs");
const task = path.resolve(__dirname, "..");
const url = pathToFileURL(path.join(task, "link_htmls/llama2_7b_memory_planning_and_memory_plan.html")).href;

(async () => {
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"]});
  const errors = [];
  const cardContrast = [];
  const auditCard = async (page, state) => {
    const result = await page.evaluate(auditRenderedContrast);
    assert.deepEqual(result.failures, [], "Step-card contrast: " + state);
    cardContrast.push({state, ...result});
  };
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
    assert.equal(await cell.locator("[data-init-logic]").count(), 12);
    const card = page.locator("#init-step-card");
    const payload = await page.locator("#memory-data").evaluate(node => JSON.parse(node.textContent));
    const sourceKeys = [];
    for (let step = 1; step <= 12; step++) {
      const link = cell.locator('g[data-step="' + step + '"] a[data-ref]');
      const key = await link.getAttribute("data-ref");
      const label = link.locator(".init-step-label");
      await label.hover();
      await page.waitForFunction(step => {
        const card = document.getElementById("init-step-card");
        return !card.hidden && card.dataset.step === String(step);
      }, step);
      assert.equal(await card.locator("pre").innerText(), await cell.locator("#init-logic-" + step + " pre").textContent());
      assert.ok(await card.locator("ol > li").count() >= 3);
      assert.equal(await link.getAttribute("aria-expanded"), "true");
      if (step === 6) {
        await card.locator("#init-step-card-content").hover();
        await page.mouse.move(5, 5);
        await page.waitForTimeout(700);
        assert.equal(await card.isVisible(), true, "Explanation stays open after leaving the label and card");
        await card.locator("#init-step-card-hold").click();
        await cell.locator('g[data-step="1"] .init-step-label').hover();
        assert.equal(await card.getAttribute("data-step"), "6", "Held step survives other hovers");
        assert.equal(await card.locator("#init-step-card-hold").getAttribute("aria-pressed"), "true");
        await card.locator("#init-step-card-hold").click();
        await cell.locator('g[data-step="7"] .init-step-label').hover();
        assert.equal(await card.getAttribute("data-step"), "7", "Released card follows the next hover");
        await label.hover();
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await auditCard(page, "Desktop / step 6");
        await page.screenshot({path: path.join(task, "validation/initialize_step_card_desktop.png")});
        await card.locator('a[data-ref="' + key + '"]').click();
      } else await label.click();
      assert.ok(await page.locator("#source-dialog").evaluate(node => node.open));
      assert.equal(await card.isVisible(), false, "Opening source dismisses the step card");
      assert.equal(await page.locator("#source-file .selected-line").getAttribute("data-line"), String(payload.references[key].line));
      assert.equal(await page.locator("#source-file .definition-line").count(), payload.references[key].end - payload.references[key].start + 1);
      sourceKeys.push(key);
      await page.keyboard.press("Escape");
    }
    const keyboardStep = cell.locator('g[data-step="11"] a[data-init-step]');
    await keyboardStep.focus();
    await page.keyboard.press("Space");
    assert.equal(await card.getAttribute("data-step"), "11");
    assert.equal(await card.getAttribute("data-held"), "true");
    assert.equal(await page.evaluate(() => document.activeElement.id), "init-step-card");
    await page.keyboard.press("Tab");
    assert.equal(await page.evaluate(() => document.activeElement.id), "init-step-card-close");
    await page.keyboard.press("Escape");
    assert.equal(await card.isVisible(), false);
    assert.ok(await keyboardStep.evaluate(node => document.activeElement === node));
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#source-dialog").evaluate(node => node.open), true);
    await page.keyboard.press("Escape");
    await cell.locator('g[data-step="3"] .init-step-label').hover();
    await page.locator(".page-header .badge").click();
    assert.equal(await card.isVisible(), false, "Outside click closes the card");
    await cell.locator('g[data-step="3"] .init-step-label').hover();
    await cell.evaluate(node => node.open = false);
    await page.waitForFunction(() => document.getElementById("init-step-card").hidden);
    await cell.evaluate(node => node.open = true);
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
    for (const width of [390, 320]) {
      const touchContext = await browser.newContext({viewport: {width, height: 844}, hasTouch: true, isMobile: true, reducedMotion: "reduce"});
      await touchContext.route(/^https?:/, route => {errors.push(route.request().url()); return route.abort();});
      const touchPage = await touchContext.newPage();
      touchPage.on("pageerror", error => errors.push(error.message));
      await touchPage.goto(url + "#initialize-sequence");
      const touchLabel = touchPage.locator('g[data-step="6"] .init-step-label');
      await touchLabel.tap({position: {x: 8, y: 8}});
      const touchCard = touchPage.locator("#init-step-card");
      assert.equal(await touchCard.getAttribute("data-step"), "6");
      assert.equal(await touchCard.isVisible(), true);
      assert.equal(await touchPage.locator("#source-dialog").evaluate(node => node.open), false);
      await touchPage.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.ok(await touchCard.evaluate(node => {
        const bounds = node.getBoundingClientRect();
        return bounds.left >= 0 && bounds.top >= 0 && bounds.right <= innerWidth + 1 && bounds.bottom <= innerHeight + 1;
      }), "Touch card stays in the viewport at " + width);
      assert.ok(await touchCard.locator("#init-step-card-content").evaluate(node => node.scrollWidth <= node.clientWidth + 1));
      await touchPage.screenshot({path: path.join(task, "validation/initialize_step_card_mobile_" + width + ".png")});
      await auditCard(touchPage, "Touch / step 6 / " + width);
      await touchCard.locator("#init-step-card-hold").tap();
      assert.equal(await touchCard.getAttribute("data-held"), "true");
      await touchCard.locator('a[data-ref="kv_dtype"]').tap();
      assert.equal(await touchPage.locator("#source-dialog").evaluate(node => node.open), true);
      assert.equal(await touchCard.isVisible(), false);
      assert.equal(await touchPage.locator("#source-file .selected-line").getAttribute("data-line"), String(payload.references.kv_dtype.line));
      await touchPage.locator("#close-source").tap();
      await touchLabel.tap({position: {x: 8, y: 8}});
      await touchCard.locator("#init-step-card-close").tap();
      assert.equal(await touchCard.isVisible(), false);
      assert.ok(await touchPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await touchContext.close();
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
    assert.ok((await fallback.locator('#initialize-sequence a[data-init-step="9"]').getAttribute("href")).startsWith("../sources/"));
    await fallback.locator("#initialize-step-logic > summary").click();
    assert.equal(await fallback.locator("[data-init-logic]").count(), 12);
    assert.ok((await fallback.locator("#init-logic-6").innerText()).includes("float32"));
    assert.equal(await fallback.locator("#init-step-card").isVisible(), false);
    await fallbackContext.close();
    assert.deepEqual(errors, []);
    const report = {checked_at: new Date().toISOString(), result: "PASS", revision: payload.revision,
      command: "NODE_PATH=/tmp/max-v4-playwright/node_modules node src/check_initialize_and_limits.cjs",
      sequence_steps: 12, source_keys: sourceKeys, object_records: objectKeys,
      hover_cards: 12, persistent_cards: true, hold_and_release: true,
      keyboard_card_access: true, escape_focus_return: true, outside_click: true,
      card_source_navigation: true, touch_card_viewports: [390, 320],
      card_contrast_checks: cardContrast,
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
