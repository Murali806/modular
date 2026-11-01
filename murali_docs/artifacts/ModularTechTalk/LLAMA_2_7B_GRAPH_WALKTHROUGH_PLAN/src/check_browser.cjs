#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const http = require("http");
const {pathToFileURL} = require("url");
const assert = require("assert/strict");
const auditRenderedContrast = require("./contrast.cjs");
const checkCallStack = require("./check_call_stack.cjs");
const TASK = path.resolve(__dirname, "..");
const OUT = path.join(TASK, "validation");
const TMP = path.join(OUT, ".browser-tmp");
fs.mkdirSync(TMP, {recursive:true});
process.env.TMPDIR = TMP;
const {chromium} = require("playwright");

(async () => {
  const errors = [];
  const checks = [];
  const contrast = [];
  const audit = async (page, state) => {
    const result = await page.evaluate(auditRenderedContrast);
    contrast.push({state, ...result});
    assert.deepEqual(result.failures, [], `Contrast failure: ${state}`);
  };
  const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (pathname === "/favicon.ico") {res.writeHead(204); res.end(); return;}
    const file = path.resolve(TASK, "." + (pathname === "/" ? "/index.html" : pathname));
    if (!file.startsWith(TASK + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {res.writeHead(404); res.end(); return;}
    res.setHeader("Content-Type", file.endsWith(".html") ? "text/html; charset=utf-8" : "text/plain; charset=utf-8");
    res.end(fs.readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
  const watch = page => {
    page.on("pageerror", e => errors.push(e.message));
    page.on("console", msg => {if (msg.type() === "error") errors.push(msg.text());});
    page.on("request", req => {if (/^https?:/.test(req.url()) && !req.url().startsWith(base)) errors.push(`External request: ${req.url()}`);});
  };
  try {
    const context = await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:"reduce",permissions:["clipboard-read","clipboard-write"]});
    const page = await context.newPage();
    watch(page);
    await page.goto(base + "/index.html", {waitUntil:"load"});
    assert.equal(await page.locator(".chapter").count(), 6);
    assert.equal(await page.locator(".code-cell").count(), 39);
    assert.equal(await page.locator(".code-cell[open]").count(), 0);
    checks.push("Six chapters and 39 initially collapsed complete source cells");
    assert.equal(await page.evaluate(() => getComputedStyle(document.body).backgroundColor), "rgb(27, 29, 33)");
    assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), "dark");
    await audit(page, "Desktop: initial dark theme");
    await page.screenshot({path:path.join(OUT,"overview.png")});

    const qna = page.locator("#qna-1");
    assert.equal(await page.locator(".qna-cell").count(), 1);
    assert.equal(await page.locator("#qna-1 > summary strong").innerText(), "QnA 1");
    assert.equal(await qna.locator(".qna-body h3").count(), 3);
    assert.ok(await page.evaluate(() => document.querySelector("#chapter-1 > p").nextElementSibling === document.querySelector("#qna-1")));
    assert.equal(await qna.evaluate(e=>e.open), false);
    await page.locator("#qna-1 > summary").focus();
    await page.keyboard.press("Enter");
    assert.ok(await qna.evaluate(e=>e.open));
    await audit(page, "Desktop: QnA expanded and keyboard focused");
    await qna.screenshot({path:path.join(OUT,"qna_expanded.png")});
    const links = await qna.locator("a[data-file]").evaluateAll(es=>es.map(e=>({file:e.dataset.file,line:Number(e.dataset.fileLine)})));
    assert.ok(links.length >= 15);
    for (const link of links) {
      const valid = await page.evaluate(({file,line})=>{
        const data=JSON.parse(document.querySelector("#guide-data").textContent);
        return data.files[file] && line > 0 && line <= data.files[file].text.split("\n").length;
      }, link);
      assert.ok(valid, `Invalid QnA source link: ${link.file}:${link.line}`);
    }
    await page.locator("#qna-1-1 > summary").click();
    for (const suffix of ["1", "2", "3", "4", "5", "6"]) {
      const id = `#qna-1-1-${suffix}`;
      const child = page.locator(id);
      await page.locator(`${id} > summary`).click();
      assert.ok(await child.evaluate(e=>e.open), `Nested QnA did not open: ${id}`);
      assert.ok(await child.locator(".nested-qna-body ul").count() > 0, `No call-flow list in ${id}`);
      assert.ok(await child.locator('.nested-qna-body a[data-file]').count() > 0, `No source links in ${id}`);
      if (suffix !== "2") await page.locator(`${id} > summary`).click();
    }
    checks.push("All six QnA 1.1.x cells expand and contain linked call-flow trees");
    await page.locator('#qna-1-1-2 a[data-file="max/python/max/serve/api_server.py"][data-file-line="346"]').click();
    assert.equal(await page.locator("#file-select").inputValue(), "max/python/max/serve/api_server.py");
    await page.locator('#full-file .code-line[data-line="346"].highlight').waitFor();
    checks.push("QnA 1.1.2 function links open Source Explorer at the linked definition line");
    await page.keyboard.press("Escape");
    await qna.locator('a[data-file="max/python/max/pipelines/architectures/__init__.py"]').first().click();
    assert.equal(await page.locator("#file-select").inputValue(), "max/python/max/pipelines/architectures/__init__.py");
    assert.ok((await page.locator("#full-file").innerText()).includes('_LazyArch("LlamaForCausalLM", ".llama3", "llama_arch")'));
    await audit(page, "Desktop: QnA complete-source dialog");
    await page.keyboard.press("Escape");
    await page.locator("#qna-1 > summary").focus();
    await page.keyboard.press("Space");
    assert.equal(await qna.evaluate(e=>e.open), false);
    await page.goto(base + "/index.html#qna-1", {waitUntil:"load"});
    assert.ok(await qna.evaluate(e=>e.open));
    await page.locator("#qna-1 > summary").click();
    checks.push("QnA 1 follows the first paragraph, contains three answers, expands/collapses by keyboard, and opens source/deep links");

    const fidelity = await page.evaluate(() => {
      const d=JSON.parse(document.querySelector("#guide-data").textContent);
      return d.cells.every(c=>[...document.querySelectorAll(`#code-${c.id} .line-text`)].map(x=>x.textContent).join("\n")+"\n"===c.code);
    });
    assert.ok(fidelity, "Displayed code differs from original payload");
    checks.push("All displayed function text equals the original source payload");

    await page.locator('[href="#chapter-3"][data-nav]').click();
    const summary = page.locator("#code-graph_output > summary");
    await summary.focus();
    await page.keyboard.press("Enter");
    assert.ok(await page.locator("#code-graph_output").evaluate(e=>e.open));
    await page.locator('#code-graph_output .annotation').nth(2).click();
    assert.ok(await page.locator('#code-graph_output .code-line[data-line="1459"]').evaluate(e=>e.classList.contains("highlight")));
    await page.locator('#code-graph_output').screenshot({path:path.join(OUT,"code_expanded.png")});
    await audit(page, "Desktop: expanded code and active annotation");
    checks.push("Keyboard expansion and signature annotation highlight original lines");

    await page.locator('#code-graph_output [data-copy]').click();
    const clipboard = await page.evaluate(() => navigator.clipboard.readText());
    const original = await page.evaluate(() => JSON.parse(document.querySelector("#guide-data").textContent).cells.find(c=>c.id==="graph_output").code);
    assert.equal(clipboard, original);
    checks.push("Copy preserves the entire original Graph.output function");

    await page.locator('#code-graph_output [data-file]').click();
    assert.ok(await page.locator("#source-dialog").evaluate(e=>e.open));
    const fileFidelity=await page.evaluate(()=>{
      const d=JSON.parse(document.querySelector("#guide-data").textContent);
      return [...document.querySelectorAll("#full-file .line-text")].map(x=>x.textContent).join("\n")+"\n"===d.files[document.querySelector("#file-select").value].text;
    });
    assert.ok(fileFidelity);
    await page.locator("#file-search").fill("def output(");
    assert.match(await page.locator("#find-count").innerText(), /^1 \/ [2-9]/);
    await page.locator("#find-next").click();
    assert.match(await page.locator("#find-count").innerText(), /^2 \/ /);
    await audit(page, "Desktop: source search and current match");
    await page.locator("#source-dialog").screenshot({path:path.join(OUT,"source_explorer.png")});
    await page.locator("#file-search").fill("no-symbol-by-this-name-validation");
    assert.equal(await page.locator("#find-count").innerText(), "No matches");
    await page.locator("#file-select").selectOption("GraphCompiler/lib/GraphCompiler.cpp");
    assert.ok((await page.locator("#full-file").innerText()).includes("buildMoPipeline"));
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#source-dialog").evaluate(e=>e.open), false);
    checks.push("Whole-file fidelity, source switching, text search, next match, and Escape close");

    for (const kind of ["finalize","compiler","cache","buffers"]) {
      const items = await page.evaluate(k=>JSON.parse(document.querySelector("#guide-data").textContent).stages[k], kind);
      for (let i=0;i<items.length;i++) {
        await page.locator(`[data-explorer="${kind}"][data-step="${i}"]`).click();
        assert.equal(await page.locator(`#panel-${kind} h4`).innerText(), items[i].title);
        await audit(page, `Desktop: ${kind} state ${i+1}`);
      }
      assert.equal(await page.locator(`[data-next="${kind}"]`).isDisabled(),true);
      await page.locator(`[data-previous="${kind}"]`).click();
      assert.equal(await page.locator(`#panel-${kind} h4`).innerText(), items.at(-2).title);
    }
    checks.push("Every finalization, compiler, reuse, and buffer state; previous/next bounds");
    await page.locator('[data-explorer="compiler"][data-step="4"]').click();
    await page.locator('#explorer-compiler').screenshot({path:path.join(OUT,"compiler_dark.png")});
    await page.locator('.sequence-diagram').screenshot({path:path.join(OUT,"native_sequence_dark.png")});

    await page.locator('[data-explorer="finalize"][data-step="5"]').click();
    await page.locator('[data-explorer-source="finalize"]').click();
    await page.locator('#code-graph_output .code-line[data-line="1488"].highlight').waitFor();
    assert.ok(await page.locator('#code-graph_output .code-line[data-line="1488"]').evaluate(e=>e.classList.contains("highlight")));
    await page.locator("#collapse-code").click();
    assert.equal(await page.locator(".code-cell[open]").count(), 0);
    await page.goto(base + "/index.html#code-compile_module", {waitUntil:"load"});
    assert.ok(await page.locator("#code-compile_module").evaluate(e=>e.open));
    await page.locator("#expand-chapter").click();
    assert.ok(await page.locator("#chapter-4 .code-cell").evaluateAll(es=>es.every(e=>e.open)));
    checks.push("Diagram-to-source jump, collapse all, direct code deep link, and chapter expansion");

    const offline=await context.newPage();
    watch(offline);
    await offline.goto(pathToFileURL(path.join(TASK,"index.html")).href,{waitUntil:"load"});
    await offline.locator('#qna-1 > summary').click();
    assert.ok(await offline.locator('#qna-1').evaluate(e=>e.open));
    await offline.locator('[data-explorer="compiler"][data-step="6"]').click();
    assert.equal(await offline.locator("#panel-compiler h4").innerText(), "Produce the compiled artifact");
    await offline.locator("#open-explorer").click();
    assert.ok(await offline.locator("#source-dialog").evaluate(e=>e.open));
    checks.push("Direct file:// opening works with embedded source and interactions");
    await offline.close();

    const mobileContext=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,reducedMotion:"reduce"});
    const mobile=await mobileContext.newPage();
    watch(mobile);
    await mobile.goto(base + "/index.html#qna-1",{waitUntil:"load"});
    const noOverflow=()=>mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
    assert.ok(await noOverflow(),"Mobile body has horizontal overflow");
    assert.ok(await mobile.locator('#qna-1').evaluate(e=>e.open));
    await audit(mobile, "Mobile: expanded QnA");
    await mobile.screenshot({path:path.join(OUT,"mobile_qna.png")});
    await mobile.locator("#qna-1-1 > summary").click();
    await mobile.locator("#qna-1-1-2 > summary").click();
    assert.ok(await mobile.locator("#qna-1-1-2").evaluate(e=>e.open));
    assert.ok(await noOverflow(), "Expanded QnA 1.1.2 call flow overflows a 390px viewport");
    await mobile.setViewportSize({width:320,height:760});
    assert.ok(await noOverflow(),"Expanded QnA 1.1.2 call flow overflows a 320px viewport");
    checks.push("Expanded nested call-flow tree stays within 390px and 320px mobile viewports");
    await mobile.locator("#qna-1-1-2 > summary").click();
    await mobile.setViewportSize({width:390,height:844});
    await mobile.locator('#code-graph_output > summary').click();
    assert.ok(await noOverflow(),"Expanded code causes mobile body overflow");
    await mobile.locator('[data-explorer="finalize"][data-step="2"]').click();
    await mobile.locator('#explorer-finalize').screenshot({path:path.join(OUT,"mobile_finalization.png")});
    await mobile.locator("#open-explorer").click();
    assert.ok(await noOverflow(),"Source dialog causes mobile body overflow");
    await audit(mobile, "Mobile: source dialog and placeholder");
    await mobile.locator("#close-explorer").click();
    checks.push("390px mobile layout, expanded code, and full-file dialog stay within viewport");
    checks.push("Expanded QnA stays within both 390px and 320px viewports");
    await page.goto(base + "/index.html", {waitUntil:"load"});
    await page.locator("#open-explorer").hover();
    await audit(page, "Desktop: hovered control");
    await page.evaluate(() => window.dispatchEvent(new Event("beforeprint")));
    await page.emulateMedia({media:"print"});
    assert.equal(await page.evaluate(() => getComputedStyle(document.body).backgroundColor), "rgb(255, 255, 255)");
    await audit(page, "Print: expanded guide on white paper");
    await page.emulateMedia({media:"screen"});
    await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
    assert.equal(await page.locator("#qna-1").evaluate(e=>e.open), false);
    checks.push("Print colors remain readable and disclosure states restore after printing");
    checks.push("Rendered text contrast meets WCAG text thresholds in inspected states; controls, SVG lines, and keyboard focus checked separately");
    await checkCallStack({browser, context, base, watch, audit, checks, out:OUT});
    await mobileContext.close();
    await context.close();
    assert.deepEqual(errors,[]);
    const report={result:"PASS",checked_at:new Date().toISOString(),browser:await browser.version(),checks,console_errors:errors,viewports:["1440x1000","390x844","320x760"],offline_file_url:true};
    fs.writeFileSync(path.join(OUT,"browser_checks.json"),JSON.stringify(report,null,2)+"\n");
    fs.writeFileSync(path.join(OUT,"contrast_checks.json"),JSON.stringify({result:"PASS",checked_at:report.checked_at,states:contrast,note:"Rendered color checks, not a full accessibility-conformance audit."},null,2)+"\n");
    console.log(JSON.stringify(report,null,2));
  } finally {
    await browser.close();
    await new Promise(resolve=>server.close(resolve));
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
