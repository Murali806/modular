"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const auditContrast = require("./contrast.cjs");
const task = path.resolve(__dirname, "..");
const url = pathToFileURL(path.join(task, "link_htmls/max_serve_os_driver_and_scaling.html")).href;
const errors = [], network = [], audits = [], screenshots = [], checks = [];
const shot = async (page, name) => {
  await page.evaluate(() => getSelection()?.removeAllRanges());
  await page.screenshot({path: path.join(task, "validation/" + name)});
  screenshots.push(name);
};
const audit = async (page, state) => {
  const result = await page.evaluate(auditContrast);
  assert.deepEqual(result.failures, [], "Contrast: " + state);
  audits.push({state, ...result});
};
const closeCard = async page => {
  if (await page.locator("#reading-card").isVisible()) await page.keyboard.press("Escape");
  await page.mouse.move(0, 0);
};
const openAncestors = async locator => locator.evaluate(node => {
  let current = node;
  while (current) {if (current instanceof HTMLDetailsElement) current.open = true; current = current.parentElement;}
});

(async () => {
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"]});
  const setup = async options => {
    const context = await browser.newContext(options);
    await context.route(/^https?:/, route => {network.push(route.request().url()); return route.abort();});
    await context.addInitScript(() => {
      Object.defineProperty(navigator, "clipboard", {configurable: true, value: {
        writeText: async text => {window.__copiedText = text;},
      }});
    });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    return {context, page};
  };
  try {
    const {context, page} = await setup({viewport: {width: 1600, height: 1100}, reducedMotion: "reduce"});
    await page.goto(url);
    const payload = await page.locator("#os-data").evaluate(node => JSON.parse(node.textContent));
    const card = page.locator("#reading-card");
    assert.equal(await page.locator(".scene").count(), 6);
    assert.equal(await page.locator("#lab").getAttribute("data-time"), "0");
    await page.waitForTimeout(150);
    assert.equal(await page.locator("#lab").getAttribute("data-time"), "0", "no autoplay");
    assert.equal(await page.locator("#lab").getAttribute("data-active"), "4");
    assert.equal(await page.locator("#metric-waiting").innerText(), "96");
    assert.equal(await page.locator("#idle-count").innerText(), "900");
    assert.equal(await page.locator(".example-library").isVisible(), false, "no duplicate examples on screen");
    await audit(page, "Desktop initial page");
    await shot(page, "os_systems_overview.png");
    checks.push("offline load, no autoplay and exact opening counts");

    let stepCards = 0;
    for (const scene of payload.scenes) {
      const node = page.locator("#scene-" + scene.id);
      await node.evaluate(element => element.open = true);
      for (let i = 0; i < scene.steps.length; i++) {
        await closeCard(page);
        const trigger = node.locator('[data-story-step="' + i + '"]');
        await trigger.hover();
        assert.equal(await card.getAttribute("data-key"), "step:" + scene.id + ":" + i);
        assert.equal(await trigger.getAttribute("aria-expanded"), "true");
        assert.deepEqual(JSON.parse(await card.locator("pre").innerText()), scene.steps[i].inputs);
        assert.ok(await card.locator("ol>li").count() >= 3);
        assert.ok((await card.locator(".result").innerText()).length > 20);
        await trigger.click();
        assert.equal(await trigger.getAttribute("aria-pressed"), "true");
        assert.equal(await card.getAttribute("data-held"), "true");
        assert.equal(await node.locator(".sequence-event.active").getAttribute("data-frame"), String(i));
        stepCards++;
      }
      await audit(page, scene.id + " final step card");
    }
    assert.equal(stepCards, 48);
    checks.push("all 48 step cards, inputs, logic, results and selected animation frames");

    await closeCard(page);
    const startup = page.locator("#scene-startup");
    await startup.locator('[data-story-step="4"]').hover();
    await page.mouse.move(0, 0);
    await page.waitForTimeout(250);
    assert.equal(await card.isVisible(), true, "card remains after hover ends");
    await card.locator("#card-hold").click();
    await startup.locator('[data-story-step="0"]').hover();
    assert.equal(await card.getAttribute("data-key"), "step:startup:4", "hold blocks a different hover");
    await card.locator("#card-hold").click();
    await page.mouse.move(0, 0);
    await startup.locator('[data-story-step="0"]').hover();
    assert.equal(await card.getAttribute("data-key"), "step:startup:0");
    await closeCard(page);
    const keyboard = startup.locator('[data-story-step="2"]');
    await keyboard.focus();
    await page.keyboard.press("Space");
    assert.equal(await card.getAttribute("data-held"), "true");
    assert.equal(await page.evaluate(() => document.activeElement.id), "reading-card");
    await page.keyboard.press("Tab");
    assert.equal(await page.evaluate(() => document.activeElement.id), "card-close");
    await page.keyboard.press("Escape");
    assert.equal(await card.isVisible(), false);
    assert.ok(await keyboard.evaluate(node => node === document.activeElement));
    await startup.locator('[data-story-step="6"]').hover();
    await card.locator('a[data-ref="connected"]').click();
    assert.equal(await card.isVisible(), false);
    assert.equal(await page.locator("#source-dialog").evaluate(node => node.open), true);
    await page.keyboard.press("Escape");
    assert.equal(await card.isVisible(), false);
    assert.ok(await startup.locator('[data-story-step="6"]').evaluate(node => node === document.activeElement));
    await page.mouse.move(0, 0);
    checks.push("persistent hover, hold/release, Space/Tab/Escape and source focus return");

    let objects = 0;
    for (const [key, record] of Object.entries(payload.objects)) {
      await closeCard(page);
      const trigger = page.locator('[data-card="object:' + key + '"]').first();
      await openAncestors(trigger);
      await trigger.hover();
      assert.equal(await card.getAttribute("data-key"), "object:" + key);
      assert.deepEqual(JSON.parse(await card.locator("pre").innerText()), record.fields);
      objects++;
    }
    checks.push("all " + objects + " object cards and their example fields");

    for (let i = 1; i <= 6; i++) {
      await closeCard(page);
      await page.locator("#systems-example-table button").nth(i - 1).hover();
      assert.ok((await card.locator("ol>li").count()) >= 4);
      assert.ok((await card.locator("pre").innerText()).includes("="));
      await audit(page, "Example " + i);
    }
    await shot(page, "os_systems_example_card.png");
    checks.push("six detailed worked examples in cards");

    await closeCard(page);
    let sourceCells = 0;
    for (const key of payload.cells) {
      const cell = page.locator("#code-" + key);
      await cell.evaluate(node => node.open = true);
      await cell.locator(".code-tools a[data-ref]").click();
      const ref = payload.references[key];
      assert.equal(await page.locator("#source-file .selected-line").getAttribute("data-line"), String(ref.line));
      assert.equal(await page.locator("#source-file .definition-line").count(), ref.end - ref.start + 1);
      await page.locator("#source-copy").click();
      assert.equal(await page.evaluate(() => window.__copiedText), ref.code);
      await page.locator("#source-search").fill(ref.symbol.split(".").at(-1));
      assert.ok(await page.locator("#source-file mark").count() > 0);
      await page.locator("#source-next").click();
      assert.equal(await page.locator("#source-file .search-current").count(), 1);
      assert.ok((await page.locator("#source-download").getAttribute("href")).endsWith(ref.file));
      if (key === "proxy_submit") await audit(page, "Source search and selected definition");
      await page.locator("#source-close").click();
      sourceCells++;
    }
    checks.push("all " + sourceCells + " complete source cells, exact copy, full-file search and highlights");

    await page.locator('[data-preset="memory"]').click();
    assert.equal(await page.locator("#lab").getAttribute("data-active"), "3");
    await page.locator('#lab-form input[name="outputTokens"]').fill("256");
    assert.equal(await page.locator("#config-pending").isVisible(), true);
    assert.equal(await page.locator("#lab").getAttribute("data-active"), "3", "changes wait for Apply");
    await page.locator('#lab-form button[type="submit"]').click();
    assert.equal(await page.locator("#lab").getAttribute("data-active"), "2");
    assert.equal(await page.locator("#config-pending").isVisible(), false);
    await page.locator('[data-preset="connections"]').click();
    await page.locator('#request-detail [data-cancel="R1"]').click();
    assert.ok((await page.locator("#request-inspector h3").innerText()).includes("cancel_pending"));
    assert.equal(JSON.parse(await page.locator("#request-detail pre").innerText()).KV_reserved_pages, 5);
    await page.locator("#lab-step").click();
    assert.ok((await page.locator("#request-inspector h3").innerText()).includes("cancelled"));
    assert.equal(JSON.parse(await page.locator("#request-detail pre").innerText()).KV_reserved_pages, 0);
    checks.push("controls apply/restart consistently, small-pool capacity and cancellation boundary");

    await page.locator('[data-preset="connections"]').click();
    await page.locator("#lab-add").click();
    assert.equal(await page.locator('[data-replica="replica-2"]').getAttribute("data-state"), "warming");
    for (let i = 0; i < 20; i++) await page.locator("#lab-step").click();
    assert.equal(await page.locator('[data-replica="replica-2"]').getAttribute("data-state"), "ready");
    const liveId = await page.locator('[data-replica="replica-1"] .batch-token').first().getAttribute("data-request");
    await page.locator('[data-replica="replica-1"] .batch-token').first().click();
    await page.locator('[data-crash="replica-1"]').click();
    assert.equal(await page.locator('[data-replica="replica-1"]').getAttribute("data-state"), "down");
    assert.ok((await page.locator("#request-inspector h3").innerText()).includes(liveId + " · failed"));
    await page.locator('[data-replace="replica-1"]').click();
    for (let i = 0; i < 20; i++) await page.locator("#lab-step").click();
    assert.equal(await page.locator('[data-replica="replica-1"]').getAttribute("data-state"), "ready");
    assert.ok((await page.locator("#request-inspector h3").innerText()).includes("failed"));
    checks.push("new-replica readiness, crash/failure and fresh replacement without stream replay");

    await page.locator('[data-preset="overload"]').click();
    for (let i = 0; i < 20; i++) await page.locator("#lab-step").click();
    assert.ok(Number((await page.locator("#metric-errors").innerText()).split(" / ")[0]) > 0);
    await page.locator('[data-preset="connections"]').click();
    await page.locator("#request-detail details").evaluate(node => node.open = true);
    await page.locator("#lab-play").click();
    await page.waitForTimeout(350);
    await page.locator("#lab-play").click();
    assert.ok(Number(await page.locator("#lab").getAttribute("data-time")) > 0);
    assert.equal(await page.locator("#request-detail details").evaluate(node => node.open), true);
    const pausedAt = await page.locator("#lab").getAttribute("data-time");
    await page.waitForTimeout(200);
    assert.equal(await page.locator("#lab").getAttribute("data-time"), pausedAt);
    await page.locator("#lab-settings").evaluate(node => node.open = false);
    await page.locator("#lab").scrollIntoViewIfNeeded();
    await audit(page, "Desktop running lab then paused");
    await shot(page, "os_systems_scaling_lab.png");
    checks.push("overload rejections, playback/pause and held live request disclosure");

    for (const width of [390, 320]) {
      const touch = await setup({viewport: {width, height: 844}, hasTouch: true, isMobile: true, reducedMotion: "reduce"});
      await touch.page.goto(url + "#scene-gpu");
      const label = touch.page.locator('#scene-gpu [data-story-step="3"]');
      await label.tap();
      const touchCard = touch.page.locator("#reading-card");
      assert.equal(await touchCard.getAttribute("data-key"), "step:gpu:3");
      assert.equal(await touchCard.getAttribute("data-held"), "true");
      assert.ok(await touchCard.evaluate(node => {
        const r = node.getBoundingClientRect();
        return r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1;
      }));
      assert.ok(await touchCard.locator("#reading-card-content").evaluate(node => node.scrollWidth <= node.clientWidth + 1));
      await audit(touch.page, "Touch GPU driver card " + width);
      await shot(touch.page, "os_systems_card_mobile_" + width + ".png");
      await touchCard.locator("#card-close").tap();
      await touch.page.locator('#scene-gpu [data-story-step="1"]').tap();
      await touchCard.locator('a[data-ref="llama_execute"]').tap();
      assert.equal(await touch.page.locator("#source-dialog").evaluate(node => node.open), true);
      assert.equal(await touch.page.locator("#source-file .selected-line").getAttribute("data-line"), String(payload.references.llama_execute.line));
      assert.ok(await touch.page.locator("#source-dialog").evaluate(node => {
        const r = node.getBoundingClientRect();
        return r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1;
      }));
      await audit(touch.page, "Touch source explorer " + width);
      await touch.page.locator("#source-close").tap();
      await touch.page.goto(url + "#worked-examples");
      await touch.page.locator("#systems-example-table").evaluate(node => node.parentElement.scrollLeft = node.parentElement.scrollWidth);
      await touch.page.locator("#systems-example-table button").nth(2).tap();
      assert.equal(await touchCard.getAttribute("data-key"), "example:tight");
      await touchCard.locator("#card-close").tap();
      await touch.page.goto(url + "#scale-lab");
      await touch.page.locator("#lab-settings").evaluate(node => node.open = false);
      await touch.page.locator('[data-preset="memory"]').tap();
      await touch.page.locator("#lab-step").tap();
      assert.equal(await touch.page.locator("#lab").getAttribute("data-active"), "3");
      assert.ok(await touch.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "mobile page overflow " + width);
      await audit(touch.page, "Touch memory lab " + width);
      await touch.page.locator(".replica-row").first().scrollIntoViewIfNeeded();
      await shot(touch.page, "os_systems_lab_mobile_" + width + ".png");
      await touch.context.close();
    }
    checks.push("390/320 touch cards, examples, bounded layouts and memory lab");

    const nojs = await setup({viewport: {width: 390, height: 844}, javaScriptEnabled: false});
    await nojs.page.goto(url);
    assert.equal(await nojs.page.locator("#lab").isVisible(), false);
    assert.equal(await nojs.page.locator(".example-library").isVisible(), true);
    const example = nojs.page.locator("#example-tight");
    await example.locator(":scope>summary").click();
    assert.ok((await example.locator(".reading-body").innerText()).includes("15 usable pages"));
    const source = nojs.page.locator("#code-spawn");
    await source.locator(":scope>summary").click();
    assert.ok(await source.locator(".source-code").isVisible());
    assert.ok(await nojs.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await nojs.context.close();
    checks.push("no-JavaScript sequences, examples, source and mobile fallback");

    await page.goto(url + "#scene-gpu");
    await page.evaluate(() => window.dispatchEvent(new Event("beforeprint")));
    await page.emulateMedia({media: "print"});
    assert.equal(await page.locator(".example-library").isVisible(), true);
    assert.ok(await page.locator("#example-tight .reading-body").isVisible());
    assert.equal(await page.locator("#lab").isVisible(), false);
    await page.emulateMedia({media: "screen"});
    await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
    checks.push("print expands all explanations/source and suppresses simulation controls");

    const motion = await setup({viewport: {width: 1500, height: 1000}, reducedMotion: "no-preference"});
    await motion.page.goto(url + "#scene-network");
    const scene = motion.page.locator("#scene-network");
    await scene.locator('[data-action="play"]').click();
    await motion.page.waitForFunction(() => document.querySelector("#scene-network .sequence-packet").classList.contains("visible"));
    const first = await scene.locator(".sequence-packet").getAttribute("transform");
    await motion.page.waitForTimeout(150);
    const second = await scene.locator(".sequence-packet").getAttribute("transform");
    assert.notEqual(first, second, "packet moves on a sequence path");
    await scene.locator('[data-action="play"]').click();
    await motion.page.waitForTimeout(1200);
    assert.equal(await scene.locator('[data-action="play"]').getAttribute("aria-pressed"), "false");
    await scene.locator('[data-action="next"]').click();
    await motion.page.waitForFunction(() => document.querySelector("#scene-network .sequence-packet").classList.contains("visible"));
    await audit(motion.page, "Animated network packet");
    await shot(motion.page, "os_systems_network_animation.png");
    await motion.context.close();
    checks.push("native SVG packet movement and explicit play/pause");
    const reader = await context.newPage();
    reader.on("pageerror", error => errors.push(error.message));
    await reader.goto(pathToFileURL(path.join(task, "max_serve_llama2_7b_startup_call_stack.html")).href);
    await reader.locator('.stack-header-links a[href="link_htmls/max_serve_os_driver_and_scaling.html"]').click();
    assert.equal(reader.url(), url);
    assert.equal(await reader.locator("#lab").getAttribute("data-active"), "4");
    await reader.close();
    checks.push("call-stack header opens the new standalone systems lab");
    await context.close();
    assert.deepEqual(errors, [], "browser errors");
    assert.deepEqual(network, [], "page requested external runtime resources");
    const result = {checked_at: new Date().toISOString(), result: "PASS", checks,
      step_cards: stepCards, object_cards: objects, source_cells: sourceCells,
      console_errors: errors, external_runtime_requests: network, contrast: audits,
      screenshots, evidence: "CPU-RUN browser/UI checks; no model execution or GPU measurements",
      command: "NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_os_systems_browser.cjs"};
    fs.writeFileSync(path.join(task, "validation/os_systems_browser.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify({result: result.result, checks: checks.length, step_cards: stepCards,
      object_cards: objects, source_cells: sourceCells, contrast_states: audits.length,
      screenshots, errors, network}, null, 2));
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
