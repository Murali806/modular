"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const auditContrast = require("./contrast.cjs");
const Model = require("./inference_session_model.js");
const task = path.resolve(__dirname, "..");
const url = pathToFileURL(path.join(task, "link_htmls/max_inference_session_devices_runtime_and_compilation.html")).href;
const checks = [], audits = [], errors = [], network = [], screenshots = [];

function svgAudit() {
  const parse = value => {
    const parts = value.match(/[\d.]+/g)?.map(Number);
    return parts && parts.length >= 3 ? parts.slice(0, 3) : null;
  };
  const luminance = c => c.map(v => {
    v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
  }).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
  const failures = [], overflow = [];
  let count = 0, minimum = Infinity;
  for (const text of document.querySelectorAll(".visual svg text")) {
    const box = text.getBoundingClientRect();
    if (!box.width || !text.checkVisibility({checkVisibilityCSS: true})) continue;
    const style = getComputedStyle(text), ink = parse(style.fill);
    if (!ink) continue;
    let backing = [252, 253, 248];
    for (const shape of text.closest("svg").querySelectorAll("rect")) {
      if (!(shape.compareDocumentPosition(text) & Node.DOCUMENT_POSITION_FOLLOWING)) continue;
      if (!shape.checkVisibility({checkVisibilityCSS: true})) continue;
      const r = shape.getBoundingClientRect(), x = box.x + box.width / 2, y = box.y + box.height / 2;
      if (x >= r.x && x <= r.right && y >= r.y && y <= r.bottom) backing = parse(getComputedStyle(shape).fill) || backing;
    }
    const a = luminance(ink), b = luminance(backing);
    const ratio = (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
    const threshold = parseFloat(style.fontSize) >= 24 ? 3 : 4.5;
    minimum = Math.min(minimum, ratio);
    count++;
    if (ratio + .01 < threshold) failures.push({text: text.textContent, ratio, threshold});
    const owner = text.closest("[data-node]");
    if (owner) {
      const r = owner.querySelector(":scope > rect").getBoundingClientRect();
      if (box.left < r.left - 1 || box.right > r.right + 1 || box.top < r.top - 1 || box.bottom > r.bottom + 1)
        overflow.push({node: owner.dataset.node, text: text.textContent, textWidth: box.width, boxWidth: r.width});
    }
  }
  return {count, minimum_ratio: Number(minimum.toFixed(3)), failures, overflow};
}
const screenshot = async (page, name) => {
  const file = "inference_session_" + name + ".png";
  await page.screenshot({path: path.join(task, "validation", file)});
  screenshots.push(file);
};
const audit = async (page, state) => {
  const html = await page.evaluate(auditContrast);
  const svg = await page.evaluate(svgAudit);
  assert.deepEqual(html.failures, [], "HTML contrast: " + state);
  assert.deepEqual(svg.failures, [], "SVG contrast: " + state);
  assert.deepEqual(svg.overflow, [], "SVG labels fit: " + state);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "No document overflow: " + state);
  audits.push({state, html_samples: html.samples, minimum_html_ratio: html.minimum_text_ratio,
    svg_samples: svg.count, minimum_svg_ratio: svg.minimum_ratio});
};
const align = async (page, key) => {
  await page.locator("#" + key).evaluate(el => {
    window.scrollTo({top: window.scrollY + el.getBoundingClientRect().top - 64, behavior: "instant"});
  });
};
const finish = async (page, key) => {
  const scene = page.locator("#" + key);
  const total = Number(await scene.getAttribute("data-frames"));
  for (let i = Number(await scene.getAttribute("data-index")); i < total - 1; i++)
    await scene.locator('[data-action="step"]').click();
  assert.equal(await scene.locator('[data-action="step"]').isDisabled(), true);
  return scene;
};

(async () => {
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"]});
  const setup = async options => {
    const context = await browser.newContext(options);
    await context.route(/^https?:/, route => {network.push(route.request().url()); return route.abort();});
    await context.addInitScript(() => {
      Object.defineProperty(navigator, "clipboard", {configurable: true, value: {
        writeText: async text => {window.__copiedDefinition = text;}
      }});
    });
    const page = await context.newPage();
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(url);
    return {page, context};
  };
  try {
    const {page, context} = await setup({viewport: {width: 1600, height: 1200}, reducedMotion: "reduce"});
    const payload = await page.locator("#inference-session-data").evaluate(el => JSON.parse(el.textContent));
    assert.equal(await page.locator("[data-lesson]").count(), 4);
    await page.waitForTimeout(180);
    assert.equal(await page.locator("#constructor").getAttribute("data-index"), "0", "No autoplay");
    await audit(page, "Desktop initial");
    await screenshot(page, "overview");

    const ctor = page.locator("#constructor");
    for (const [key, record] of Object.entries(Model.DEVICE_CASES)) {
      await ctor.locator('[data-device-case="' + key + '"]').click();
      await finish(page, "constructor");
      const expected = Model.normalizeDevices(record.devices).map(d => d === "cpu" ? "CPU" : "GPU " + d.slice(3));
      assert.deepEqual(JSON.parse(await ctor.locator('[data-slots="normalized"]').getAttribute("data-values")), expected);
      assert.equal(await ctor.locator('[data-slots="registered"] [data-occupied="true"]').count(), expected.length);
      assert.equal(await ctor.locator('[data-actor][data-place="registered"]:not([hidden])').count(), expected.length);
      assert.equal(await ctor.locator('[data-node="context"]').getAttribute("data-state"), "ready");
      assert.match(await ctor.locator("[data-state-strip]").innerText(), /Decoder artifacts\s*0/);
      await audit(page, "Constructor: " + key);
    }
    await ctor.locator('[data-option="threads"]').selectOption("4");
    await ctor.locator('[data-option="extension"]').check();
    for (const policy of ["import", "export", "conflict"]) {
      await ctor.locator('[data-option="store"]').selectOption(policy);
      await finish(page, "constructor");
      if (policy === "conflict") {
        assert.equal(await ctor.locator("[data-boundary]").getAttribute("data-state"), "waiting");
        assert.equal(await ctor.locator('[data-slots="normalized"] [data-occupied="true"]').count(), 0);
        assert.equal(await ctor.locator("[data-event]").getAttribute("data-kind"), "error");
      } else {
        assert.match(await ctor.locator("[data-state-strip]").innerText(), /Native thread option\s*4/);
      }
      await audit(page, "MEF policy: " + policy);
    }
    await ctor.locator('[data-option="store"]').selectOption("none");
    await ctor.locator('[data-device-case="duplicates"]').click();
    await finish(page, "constructor");
    await align(page, "constructor");
    await screenshot(page, "constructor");
    checks.push("All device presets, thread/extension settings, import/export and constructor rejection displayed correctly");

    const compilation = page.locator("#compilation");
    for (const mode of ["cold", "mef", "missing"]) {
      await compilation.locator('[data-mode="' + mode + '"]').click();
      await finish(page, "compilation");
      assert.equal(await compilation.locator('[data-node="artifact"]').getAttribute("data-state"), "ready");
      assert.equal(await compilation.locator('[data-node="model"]').getAttribute("data-state"), mode === "missing" ? "waiting" : "ready");
      assert.match(await compilation.locator("[data-state-strip]").innerText(), mode === "missing" ? /Native init calls\s*0/ : /Native init calls\s*1/);
      await audit(page, "Compilation: " + mode);
    }
    await align(page, "compilation");
    await screenshot(page, "missing_weight");
    checks.push("Cold compile and MEF reuse reach a model; missing-weight validation stops native initialization");
    assert.equal(await compilation.locator('[data-init-gate]').getAttribute("data-blocked"), "true");
    assert.equal(await compilation.locator('[data-slots="bindings"] [data-missing="true"]').count(), 1);
    await compilation.locator('[data-repair]').click();
    await finish(page, "compilation");
    assert.equal(await compilation.locator('[data-init-gate]').getAttribute("data-open"), "true");
    assert.equal(await compilation.locator('[data-slots="bindings"] [data-occupied="true"]').count(), 2);
    assert.equal(await compilation.locator('[data-node="model"]').getAttribute("data-state"), "ready");
    assert.equal(await compilation.locator('[data-actor="compiled-artifact"]:not([hidden])').count(), 1);
    await audit(page, "Missing weight supplied: gate opens");
    await screenshot(page, "repaired_weight");
    checks.push("Supply W_b fills the empty binding, opens the gate and initializes M1 using the retained artifact");

    const execution = page.locator("#execution");
    await execution.locator('[data-mode="prefill_decode"]').click();
    let sawFive = false, sawTwo = false, sawOutput = false;
    for (let i = 0; i < 12; i++) {
      await execution.locator('[data-action="step"]').click();
      const count = await execution.locator('[data-slots="tokens"] [data-occupied="true"]').count();
      sawFive ||= count === 5; sawTwo ||= count === 2;
      sawOutput ||= await execution.locator("[data-result-tile]").isVisible();
      assert.match(await execution.locator("[data-state-strip]").innerText(), /Decoder setups\s*1/);
      await audit(page, "Execution event " + (i + 1));
    }
    assert.ok(sawFive && sawTwo && sawOutput);
    assert.match(await execution.locator("[data-state-strip]").innerText(), /Completed passes\s*2/);
    await align(page, "execution");
    await screenshot(page, "execution");
    await execution.locator('[data-another]').click();
    assert.deepEqual(JSON.parse(await execution.locator('[data-slots="tokens"]').getAttribute("data-values")), ["64", "12"]);
    await finish(page, "execution");
    assert.match(await execution.locator("[data-state-strip]").innerText(), /Completed passes\s*3/);
    assert.deepEqual(JSON.parse(await execution.locator('[data-slots="history-a"]').getAttribute("data-values")), ["88", "64", "25"]);
    assert.equal(await execution.locator('[data-kv-tiles] [data-owner="A"][data-filled="true"]').count(), 5);
    assert.equal(await execution.locator('[data-kv-tiles] [data-owner="B"][data-filled="true"]').count(), 4);
    assert.match(await execution.locator("[data-state-strip]").innerText(), /Decoder setups\s*1/);
    await audit(page, "Additional decode feeds previous samples");
    await execution.locator('[data-mode="decode_only"]').click();
    await finish(page, "execution");
    assert.match(await execution.locator("[data-state-strip]").innerText(), /Completed passes\s*1/);
    checks.push("Prefill/decode token tiles, row offsets, returned logits, reset and persistent decoder setup");

    const lifetime = page.locator("#lifetime");
    for (const mode of ["caller_first", "models_first"]) {
      await lifetime.locator('[data-mode="' + mode + '"]').click();
      for (const count of [2, 1, 0]) {
        await lifetime.locator('[data-action="step"]').click();
        assert.equal(await lifetime.locator("[data-reference-count]").textContent(), String(count));
        assert.equal(await lifetime.locator('[data-node="native"]').getAttribute("data-state"), count ? "ready" : "released");
        await audit(page, "Lifetime " + mode + ": " + count);
      }
    }
    checks.push("Reference edges and lifetime states follow both release orders");
    await lifetime.locator('[data-action="reset"]').click();
    for (const [i, owner] of ["sampler", "caller", "decoder"].entries()) {
      await lifetime.locator('[data-drop-owner="' + owner + '"]').click();
      assert.equal(await lifetime.locator("[data-reference-count]").textContent(), String(2 - i));
      assert.equal(await lifetime.locator('[data-drop-owner="' + owner + '"]').isDisabled(), true);
      assert.equal(await lifetime.locator('[data-actor="reference-' + owner + '"]').getAttribute("data-place"), "released");
      assert.equal(await lifetime.locator('[data-node="context"]').getAttribute("data-state"), i === 2 ? "released" : "ready");
    }
    assert.equal(await lifetime.locator('[data-actor="shared-resources"]').isVisible(), true);
    assert.equal(await lifetime.locator('[data-mode][aria-pressed="true"]').count(), 0, "Custom order does not falsely select a preset");
    await audit(page, "Custom release order leaves shared resources");
    await align(page, "lifetime");
    await screenshot(page, "lifetime");
    checks.push("Release sampler, wrapper and decoder directly: each reference detaches, context survives until the last, shared resources remain illustrated");

    await align(page, "constructor");
    const badge = ctor.locator("[data-lesson-card]");
    await badge.hover();
    assert.equal(await page.locator("#explanation-card").isVisible(), true);
    assert.equal(await page.locator("#explanation-card").getAttribute("data-key"), "lesson:constructor");
    await page.locator("#explanation-card").hover();
    await page.waitForTimeout(220);
    assert.equal(await page.locator("#explanation-card").isVisible(), true, "Hover-to-card handoff");
    await page.mouse.move(0, 0);
    await page.waitForTimeout(230);
    assert.equal(await page.locator("#explanation-card").isVisible(), false, "Hover leave closes card");
    await badge.focus();
    await page.keyboard.press("Space");
    assert.equal(await page.locator("#explanation-card").getAttribute("data-pinned"), "true");
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#explanation-card").isVisible(), false);
    assert.ok(await badge.evaluate(el => el === document.activeElement));
    await page.locator("#legend-button").hover();
    assert.equal(await page.locator("#legend-card").isVisible(), false, "Legend is click-to-open");
    await page.locator("#legend-button").click();
    assert.equal(await page.locator("#legend-card .legend-entry").count(), 14);
    await audit(page, "Component legend");
    await page.keyboard.press("Escape");
    const contextNode = ctor.locator('[data-component="context"]');
    await contextNode.focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#explanation-card").getAttribute("data-key"), "component:context");
    await page.locator('#explanation-card [data-ref="context_create"]').click();
    assert.equal(await page.locator("#source-dialog").evaluate(el => el.open), true);
    await page.keyboard.press("Escape");
    await page.waitForFunction(() => document.activeElement?.dataset.component === "context");
    assert.ok(await contextNode.evaluate(el => el === document.activeElement));
    checks.push("Numbered hover/focus cards, hover-to-card handoff, Space/Enter, Escape, click-only legend and focus restoration");

    await page.locator(".source-directory").evaluate(el => el.open = true);
    for (const [key, ref] of Object.entries(payload.references)) {
      await page.locator('.source-directory [data-ref="' + key + '"]').click();
      assert.equal(await page.locator("#source-file .selected-line").getAttribute("data-line"), String(ref.line));
      assert.equal(await page.locator("#source-file .definition-line").count(), ref.end - ref.start + 1);
      const displayed = await page.locator("#source-file .source-line").evaluateAll(rows =>
        rows.map(row => [...row.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join("")).join("\n"));
      assert.equal(displayed + "\n", payload.files[ref.file].text, "Full source unchanged: " + key);
      await page.keyboard.press("Escape");
    }
    await page.locator('.source-directory [data-ref="session"]').click();
    await page.locator("#source-copy").click();
    const ref = payload.references.session;
    assert.equal(await page.evaluate(() => window.__copiedDefinition),
      payload.files[ref.file].text.split(/(?<=\n)/).slice(ref.start - 1, ref.end).join(""));
    await page.locator("#source-search").fill("final_devices");
    assert.ok(await page.locator("#source-file .match-line").count() >= 3);
    await page.locator("#source-next").click();
    assert.match(await page.locator("#source-count").textContent(), /^2 \/ /);
    await audit(page, "Full-file source explorer");
    await screenshot(page, "source_explorer");
    await page.keyboard.press("Escape");
    checks.push("All 30 source targets open exact complete definitions; full-file content, search, copy and return focus verified");
    await page.evaluate(() => window.dispatchEvent(new Event("beforeprint")));
    await page.emulateMedia({media: "print"});
    assert.equal(await page.locator(".playback").first().isVisible(), false);
    assert.equal(await page.locator(".transcript[open]").count(), 4);
    await page.emulateMedia({media: "screen"});
    await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
    checks.push("Print view exposes portable flows and hides playback controls");
    await page.goto(pathToFileURL(path.join(task, "max_serve_llama2_7b_startup_call_stack.html")).href);
    const popupReady = page.waitForEvent("popup");
    await page.locator('a.tree-doc-link[href="link_htmls/max_inference_session_devices_runtime_and_compilation.html"]').click();
    const linked = await popupReady;
    await linked.waitForLoadState("load");
    assert.equal(linked.url(), url);
    assert.equal(await linked.locator("[data-lesson]").count(), 4);
    await linked.close();
    checks.push("The new link beneath InferenceSession opens the standalone animation from the real call-tree reader");
    await context.close();

    const motion = await setup({viewport: {width: 1440, height: 1050}, reducedMotion: "no-preference"});
    const animated = motion.page.locator("#constructor");
    await align(motion.page, "constructor");
    await animated.locator('[data-device-case="duplicates"]').click();
    await animated.locator('[data-action="step"]').click();
    await animated.locator('[data-action="step"]').click();
    const device = animated.locator('[data-actor="device-input-0"]');
    await device.evaluate(el => {window.__keptDevice = el;});
    const initialPosition = await device.getAttribute("transform");
    await animated.locator('[data-action="step"]').click();
    await motion.page.waitForTimeout(80);
    const before = await device.getAttribute("transform");
    await motion.page.waitForTimeout(140);
    assert.notEqual(await device.getAttribute("transform"), before, "The actual device card moves toward final_devices");
    assert.notEqual(before, initialPosition);
    assert.ok(await device.evaluate(el => el === window.__keptDevice), "Object identity survives movement");
    await animated.locator('[data-action="play"]').click();
    await motion.page.waitForTimeout(100);
    await animated.locator('[data-action="play"]').click();
    const frozenPosition = await device.getAttribute("transform");
    await motion.page.waitForTimeout(220);
    assert.equal(await device.getAttribute("transform"), frozenPosition, "Pause freezes in-flight objects");
    await animated.locator('[data-action="step"]').click();
    const duplicate = animated.locator('[data-actor="device-input-1"]');
    assert.equal(await duplicate.getAttribute("data-place"), "merged");
    await motion.page.waitForTimeout(1280);
    assert.equal(await duplicate.isVisible(), false, "Duplicate merges and disappears without another slot");
    assert.equal(await animated.locator('[data-slots="normalized"] [data-occupied="true"]').count(), 1);
    assert.equal(await device.isVisible(), true);
    checks.push("Persistent device identity moves into a slot; duplicate merges into it; Pause freezes the object mid-movement");
    await animated.locator('[data-action="reset"]').click();
    await animated.locator("[data-speed]").selectOption("2");
    await animated.locator('[data-action="play"]').click();
    await motion.page.waitForTimeout(1350);
    assert.ok(Number(await animated.getAttribute("data-index")) > 0, "Play advances events");
    await animated.locator('[data-action="play"]').click();
    const paused = await animated.getAttribute("data-index");
    await motion.page.waitForTimeout(200);
    assert.equal(await animated.getAttribute("data-index"), paused);
    await animated.locator('[data-action="play"]').click();
    await align(motion.page, "lifetime");
    await motion.page.waitForTimeout(140);
    assert.equal(await animated.getAttribute("data-playing"), "false", "Offscreen pause");
    await align(motion.page, "constructor");
    await animated.locator('[data-action="play"]').click();
    await motion.page.evaluate(() => {
      Object.defineProperty(document, "hidden", {configurable: true, value: true});
      document.dispatchEvent(new Event("visibilitychange"));
      delete document.hidden;
    });
    assert.equal(await animated.getAttribute("data-playing"), "false", "Hidden-document pause");
    const movingExecution = motion.page.locator("#execution");
    await align(motion.page, "execution");
    await movingExecution.locator('[data-action="step"]').click();
    const tokenPositions = await movingExecution.locator('[data-actor][data-place="input"]').evaluateAll(els => els.map(el => el.getAttribute("transform")));
    await motion.page.waitForTimeout(180);
    const shiftedPositions = await movingExecution.locator('[data-actor][data-place="input"]').evaluateAll(els => els.map(el => el.getAttribute("transform")));
    assert.equal(tokenPositions.length, 5);
    assert.ok(shiftedPositions.filter((p, i) => p !== tokenPositions[i]).length >= 3, "Several distinct input tokens move concurrently");
    for (let i = 0; i < 3; i++) await movingExecution.locator('[data-action="step"]').click();
    await motion.page.waitForTimeout(250);
    assert.equal(await movingExecution.locator('[data-actor][data-place="gpu-work"]:not([hidden])').count(), 2);
    const gauge = Number(await movingExecution.locator('[data-work-fill]').getAttribute("width"));
    assert.ok(gauge > 0 && gauge < 154, "GPU work has visible progress during the boundary illustration");
    await motion.page.waitForTimeout(1000);
    await screenshot(motion.page, "device_work");
    await motion.context.close();
    checks.push("Concurrent token and GPU work objects, activity progress, Play/Pause, speed, reset, offscreen and hidden-document pause work");

    const mobile = await setup({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true, reducedMotion: "reduce"});
    await audit(mobile.page, "Mobile 390px");
    await align(mobile.page, "constructor");
    await mobile.page.locator('#constructor [data-lesson-card]').tap();
    assert.equal(await mobile.page.locator("#explanation-card").isVisible(), true);
    await audit(mobile.page, "Mobile lesson card");
    await mobile.page.locator("[data-close-card]").tap();
    await mobile.page.locator("#legend-button").tap();
    assert.equal(await mobile.page.locator("#legend-card").isVisible(), true);
    await mobile.page.locator("[data-close-legend]").tap();
    const visual = mobile.page.locator("#constructor .visual");
    await visual.evaluate(el => {el.scrollLeft = 330;});
    assert.ok(await visual.evaluate(el => el.scrollLeft > 0));
    await mobile.page.locator('#constructor [data-action="step"]').tap();
    assert.equal(await mobile.page.locator('#constructor [data-actor][data-moving="true"]').count(), 0, "Reduced motion settles objects without travel");
    await screenshot(mobile.page, "mobile");
    await mobile.context.close();
    checks.push("Touch lesson/legend controls, contained diagram panning, 390px layout and reduced motion");

    const nojs = await setup({viewport: {width: 1280, height: 900}, javaScriptEnabled: false});
    assert.equal(await nojs.page.locator(".nojs-note").isVisible(), true);
    assert.equal(await nojs.page.locator(".playback").first().isVisible(), false);
    assert.equal(await nojs.page.locator(".transcript pre").count(), 4);
    assert.equal(await nojs.page.locator(".code-cell").count(), 11);
    await nojs.page.locator("#constructor .transcript > summary").click();
    assert.ok(await nojs.page.locator("#constructor .transcript pre").isVisible());
    await nojs.context.close();
    checks.push("No-JavaScript portable flows and complete source definitions remain readable");

    assert.deepEqual(errors, [], "No browser errors");
    assert.deepEqual(network, [], "No network dependencies");
    const result = {checked_at: new Date().toISOString(), result: "PASS", revision: payload.revision,
      command: "NODE_PATH=<Playwright installation>/node_modules node src/check_inference_session_browser.cjs",
      evidence: "CPU-RUN browser checks only; no MAX/GPU execution", checks, audits, screenshots,
      page_errors: errors, external_requests: network};
    fs.writeFileSync(path.join(task, "validation/inference_session_browser.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify({result: result.result, checks, contrast_audits: audits.length, screenshots}, null, 2));
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
