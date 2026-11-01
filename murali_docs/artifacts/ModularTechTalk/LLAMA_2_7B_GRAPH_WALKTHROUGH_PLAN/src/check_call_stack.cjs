"use strict";
const assert = require("assert/strict");
const path = require("path");
const {pathToFileURL} = require("url");

module.exports = async function checkCallStack({browser, context, base, watch, audit, checks, out}) {
  const guide = await context.newPage();
  watch(guide);
  await guide.goto(base + "/index.html#qna-1");
  const openBefore = await guide.locator("#qna-1-1").evaluate(e=>e.open);
  const popup = context.waitForEvent("page");
  await guide.locator("#open-call-stack").click();
  const page = await popup;
  watch(page);
  await page.waitForLoadState("load");
  assert.ok(page.url().endsWith("/call_stack.html"));
  assert.equal(await guide.locator("#qna-1-1").evaluate(e=>e.open), openBefore);
  assert.equal(await page.locator(".stack-section").count(), 6);
  assert.equal(await page.locator('input[type="radio"]').count(), 0);
  assert.equal(await guide.locator('input[type="radio"]').count(), 0);
  assert.match(await page.locator(".stack-commands").innerText(), /--task text_generation/);
  assert.ok((await page.locator(".stack-commands").innerText()).trim().endsWith("max serve --model meta-llama/Llama-2-7b-hf"));
  checks.push("QnA 1.1 header opens a separate six-section text-tree page without toggling the parent cell; both commands are text examples, with no radios");

  await guide.locator("#qna-1-1 > summary").click();
  assert.equal(await guide.locator("#startup-map").evaluate(e=>e.open), false);
  await guide.locator("#startup-map > summary").focus();
  await guide.keyboard.press("Enter");
  assert.ok(await guide.locator("#startup-map").evaluate(e=>e.open));
  await guide.keyboard.press("Space");
  assert.equal(await guide.locator("#startup-map").evaluate(e=>e.open), false);
  await guide.locator("#qna-1-1 > summary").screenshot({path:path.join(out,"call_stack_launch.png")});
  await audit(page, "Call-tree window: commands, closed map and text");
  await page.screenshot({path:path.join(out,"call_stack_overview.png")});
  assert.equal(await page.locator("#startup-map").evaluate(e=>e.open), false);
  await page.locator("#startup-map > summary").click();
  await page.locator("#startup-map").screenshot({path:path.join(out,"call_stack_startup_map.png")});
  await page.locator("#startup-map > summary").click();
  checks.push("Startup map is initially collapsed and expands/collapses in the guide and separate reader");

  const payload = await page.evaluate(()=>JSON.parse(document.querySelector("#stack-data").textContent));
  for (const section of payload.sections) {
    assert.equal(await page.locator("#" + section.id + " .text-tree code").textContent(), section.text);
    await page.locator('.stack-nav a[href="#' + section.id + '"]').click();
    assert.ok(await page.locator("#" + section.id).isVisible());
  }
  assert.ok(payload.sections[1].text.includes("request PUSH"));
  assert.ok(payload.sections[4].text.indexOf("model_factory()") < payload.sections[4].text.indexOf("alive.set()"));
  assert.ok(payload.sections[5].text.indexOf("Llama3Config.finalize()") < payload.sections[5].text.indexOf("compile_and_init_graphs()"));
  checks.push("All six displayed text trees exactly match their export payload; worker, IPC directions and finalize-before-compile order are present");

  for (const key of ["cli", "autoconfig", "register_all", "zmq_interface", "autotokenizer", "gen_load", "download", "finalize", "finalize_call", "graph_output"]) {
    const ref = payload.references[key];
    await page.locator('[data-ref="' + key + '"]').first().click();
    assert.equal(await page.locator("#file-select").inputValue(), ref.file);
    await page.locator('#full-file .code-line[data-line="' + ref.line + '"].highlight').waitFor();
    assert.equal(await page.locator('#full-file .code-line[data-line="' + ref.line + '"] .line-text').textContent(), ref.text);
    assert.equal(await page.locator("#full-file .line-text").evaluateAll(es=>es.map(e=>e.textContent).join("\n")+"\n"), payload.files[ref.file].text);
    if (key === "finalize") {
      await audit(page, "Call-tree window: exact source dialog");
      await page.locator("#source-dialog").screenshot({path:path.join(out,"call_stack_source.png")});
      await page.locator("#file-search").fill("state_dict");
      assert.match(await page.locator("#find-count").innerText(), /^1 \/ /);
      await page.locator("#find-next").click();
      assert.match(await page.locator("#find-count").innerText(), /^2 \/ /);
    }
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#source-dialog").evaluate(e=>e.open), false);
  }
  await page.locator(".stack-source-map > summary").click();
  await page.locator('.stack-source-map a[data-file="' + payload.references.zmq_interface.file + '"]').click();
  assert.equal(await page.locator("#file-select").inputValue(), payload.references.zmq_interface.file);
  await page.locator("#close-explorer").click();
  await page.locator(".stack-source-map > summary").click();
  checks.push("Links across all six trees and the alias source map open exact original files; definition/caller highlighting, source search and Escape work");

  await page.locator("#tree-search").fill("Llama3Config");
  assert.match(await page.locator("#tree-count").innerText(), /^1 \/ /);
  await page.locator("#tree-next").click();
  assert.match(await page.locator("#tree-count").innerText(), /^2 \/ /);
  await audit(page, "Call-tree window: matching text");
  await page.locator("#tree-search").fill("unfindable-tree-term");
  assert.equal(await page.locator("#tree-count").innerText(), "No matches");
  await page.locator("#tree-search").fill("");
  await page.locator("#copy-tree").click();
  assert.equal(await page.evaluate(()=>navigator.clipboard.readText()), payload.text);
  checks.push("Tree search/next/no-match states work; Copy all text exactly preserves the exported text");

  await page.evaluate(()=>window.dispatchEvent(new Event("beforeprint")));
  await page.emulateMedia({media:"print"});
  assert.ok(await page.locator("#startup-map").evaluate(e=>e.open));
  await audit(page, "Call-tree window: print colors");
  await page.emulateMedia({media:"screen"});
  await page.evaluate(()=>window.dispatchEvent(new Event("afterprint")));
  assert.equal(await page.locator("#startup-map").evaluate(e=>e.open), false);
  checks.push("Text-tree printing expands the map and restores disclosure state afterward");

  for (const width of [390,320]) {
    await page.setViewportSize({width,height:844});
    await page.locator('.stack-nav a[href="#stack-1-1-2"]').click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1), "Call-tree window has page overflow");
    const tree=page.locator("#stack-1-1-2 .text-tree");
    assert.ok(await tree.evaluate(e=>e.scrollWidth>e.clientWidth));
    await tree.evaluate(e=>e.scrollLeft=e.scrollWidth);
    assert.ok(await tree.evaluate(e=>e.scrollLeft>0));
    await tree.evaluate(e=>e.scrollLeft=0);
    await page.screenshot({path:path.join(out,"call_stack_mobile_" + width + ".png")});
    await page.locator(".stack-source-map > summary").click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1), "Source map has page overflow");
    await page.locator(".stack-source-map > summary").click();
    await page.locator('[data-ref="gen_load"]').first().click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1), "Source dialog has page overflow");
    await audit(page, "Call-tree window: " + width + "px source view");
    await page.keyboard.press("Escape");
  }
  checks.push("390px/320px reader preserves text indentation with horizontal scrolling inside trees; source map/dialog fit the viewport");
  await page.close();
  await guide.close();

  const offline = await context.newPage();
  watch(offline);
  await offline.goto(pathToFileURL(path.resolve(out,"..","call_stack.html")).href);
  await offline.locator('[data-ref="hf_load"]').first().click();
  assert.ok(await offline.locator("#source-dialog").evaluate(e=>e.open));
  await offline.close();
  const noJS = await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
  const fallback = await noJS.newPage();
  watch(fallback);
  await fallback.goto(base + "/call_stack.html");
  assert.equal(await fallback.locator(".stack-section").count(),6);
  await fallback.locator("#startup-map > summary").click();
  assert.ok(await fallback.locator("#startup-map").evaluate(e=>e.open));
  assert.ok((await fallback.locator('[data-ref="finalize"]').getAttribute("href")).startsWith("sources/"));
  assert.ok(await fallback.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await noJS.close();
  checks.push("Separate text-tree reader works directly from file://; without JavaScript, all trees, map and snapshot links remain readable");
};
