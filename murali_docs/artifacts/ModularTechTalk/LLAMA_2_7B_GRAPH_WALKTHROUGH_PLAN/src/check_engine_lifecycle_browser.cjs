"use strict";
const assert=require("node:assert/strict"),path=require("node:path"),fs=require("node:fs"),{pathToFileURL}=require("node:url");
const {chromium}=require("playwright"),contrast=require("./contrast.cjs");
const task=path.resolve(__dirname,".."),url=pathToFileURL(path.join(task,"link_htmls/max_engine_graph_compilation_mef_and_inference.html")).href;
(async()=>{
  const browser=await chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],external=[];
  page.on("pageerror",e=>errors.push(e.message));page.on("request",r=>{if(/^https?:/.test(r.url()))external.push(r.url());});
  await page.goto(url);await page.locator("#animation").scrollIntoViewIfNeeded();
  const snapshot=()=>page.evaluate(()=>window.engineLifecycleSnapshot());
  assert.equal((await snapshot()).index,-1);assert.equal(await page.locator("#play").getAttribute("aria-pressed"),"false");
  const badge=page.locator("#lesson-button"),card=page.locator("#hover-card");
  await badge.hover();assert.ok(await card.isVisible());await card.hover();await page.waitForTimeout(180);assert.ok(await card.isVisible());
  await page.mouse.move(4,950);await page.waitForTimeout(200);assert.equal(await card.isVisible(),false);
  await badge.focus();assert.ok(await card.isVisible());await page.keyboard.press("Escape");assert.equal(await card.isVisible(),false);
  await page.keyboard.press("Tab");
  await page.locator("#legend-button").hover();assert.equal(await page.locator("#legend-card").isVisible(),false);
  await page.locator("#legend-button").click();assert.ok(await page.locator("#legend-card").isVisible());
  const colors=await page.locator(".legend-row .swatch").evaluateAll(nodes=>nodes.map(n=>getComputedStyle(n).backgroundColor));
  assert.equal(new Set(colors).size,colors.length);
  await page.locator("#legend-filter").fill("InferenceSession");assert.ok(await page.locator('.legend-row[data-object="session"]').isVisible());assert.ok(await page.locator(".legend-row:visible").count()<colors.length);
  await page.locator('.legend-row[data-object="session"] a[data-ref="session"]').click();
  assert.ok(await page.locator("#source-dialog").evaluate(n=>n.open));
  const source=await page.locator("#source-code .source-line b").allTextContents();
  const engine=fs.readFileSync(path.join(task,"sources/max/python/max/engine/api.py"),"utf8").split("\n");assert.deepEqual(source,engine,"source explorer preserves the complete original file");
  await page.keyboard.press("Escape");assert.ok(await page.locator("#legend-card").isVisible());
  await page.keyboard.press("Escape");assert.equal(await page.locator("#legend-card").isVisible(),false);
  for(const [id,ref,file] of [
    ['add','mojo_add','max/kernels/src/graph_compiler/builtin_kernels/elementwise.mojo'],
    ['primitives','ready_counts','MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp']
  ]){
    await page.locator('#legend-button').click();await page.locator('#legend-filter').fill('');
    await page.locator(`.legend-row[data-object="${id}"] a[data-ref="${ref}"]`).click();
    assert.deepEqual(await page.locator('#source-code .source-line b').allTextContents(),fs.readFileSync(path.join(task,'sources',file),'utf8').split('\n'));
    await page.keyboard.press('Escape');await page.keyboard.press('Escape');
  }
  const maxStep=(await snapshot()).total;
  let renderedEdges=0;
  for(let i=0;i<maxStep;i++){
    await page.locator('#timeline').evaluate((n,i)=>{n.value=i;n.dispatchEvent(new Event('input',{bubbles:true}));},i);
    const s=await snapshot();
    if(s.edge){
      const source=await page.locator(`.object-node[data-object="${s.edge[0]}"]`).count();
      const target=await page.locator(`.object-node[data-object="${s.edge[1]}"]`).count();
      assert.ok(source&&target,'edge endpoints remain visible: '+s.title);
      assert.equal(await page.locator('#active-edge path.active-wire').count(),1,'active handoff has a routed arrow: '+s.title);
      renderedEdges++;
    }
  }
  let audited=0;
  for(const phase of ["graph","compiler","mojo","mef","init","inference"]){
    await page.locator(`button[data-phase="${phase}"]`).click();assert.equal((await snapshot()).phase,phase);
    const audit=await page.evaluate(contrast);assert.deepEqual(audit.failures,[],phase+": rendered contrast");audited++;
    const nodes=await page.locator(".object-node").evaluateAll(nodes=>nodes.map(n=>{const t=n.querySelectorAll("text"),b=n.querySelector("rect").getBBox();return [...t].map(text=>{const r=text.getBBox();return{label:text.textContent,fits:r.x>=b.x&&r.x+r.width<=b.x+b.width-4&&r.y+r.height<=b.y+b.height};}).flat();}).flat());
    nodes.forEach(n=>assert.ok(n.fits,phase+": node label fits: "+n.label));
  }
  await page.locator('button[data-phase="compiler"]').click();
  await page.locator('#timeline').evaluate(n=>{n.value=Number(n.value)+6;n.dispatchEvent(new Event('input',{bubbles:true}));});
  await page.screenshot({path:'/tmp/max-engine-lifecycle-compiler.png',fullPage:false});
  await page.locator('button[data-phase="mojo"]').click();
  await page.locator('#timeline').evaluate(n=>{n.value=Number(n.value)+4;n.dispatchEvent(new Event('input',{bubbles:true}));});
  await page.screenshot({path:'/tmp/max-engine-lifecycle-mojo.png',fullPage:false});
  await page.locator('#timeline').evaluate(n=>{n.value=n.max;n.dispatchEvent(new Event('input',{bubbles:true}));});
  const done=await snapshot();assert.ok(done.compiledReady&&done.initialized&&done.outputsReady);
  assert.equal(await page.locator('[data-gate][data-ready="true"]').count(),4);
  await page.locator("#reset").click();assert.equal((await snapshot()).index,-1);assert.equal(await page.locator('[data-gate][data-ready="true"]').count(),0);
  await page.locator("#play").click();assert.equal(await page.locator("#animation").getAttribute("data-playing"),"true");
  await page.locator("#play").click();const paused=(await snapshot()).index;await page.waitForTimeout(1650);assert.equal((await snapshot()).index,paused);
  await page.locator('#route-mode').selectOption('read');assert.equal((await snapshot()).index,-1);assert.equal((await snapshot()).options.route,'read');
  await page.locator('#timeline').evaluate(n=>{n.value=n.max;n.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.equal((await snapshot()).graphReady,false);assert.ok((await snapshot()).outputsReady);
  await page.locator('#initialization-mode').selectOption('virtual');
  await page.locator('#timeline').evaluate(n=>{n.value=n.max;n.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.equal((await snapshot()).error,'UNKNOWN GRAPH NAMES');
  await page.locator('#route-mode').selectOption('cold');await page.locator('#initialization-mode').selectOption('device');
  await page.locator("#ownership").evaluate(n=>n.open=true);await page.locator("#play").click();await page.evaluate(()=>window.scrollTo(0,document.body.scrollHeight));await page.waitForTimeout(250);assert.equal(await page.locator("#animation").getAttribute("data-playing"),"false");
  for(const width of [320,390]){
    const mobile=await browser.newPage({viewport:{width,height:800},hasTouch:true,isMobile:true});await mobile.goto(url);
    const dims=await mobile.evaluate(()=>({document:document.documentElement.scrollWidth,viewport:innerWidth,svg:document.querySelector(".canvas-scroll").scrollWidth,container:document.querySelector(".canvas-scroll").clientWidth}));
    assert.ok(dims.document<=dims.viewport+1,"document contains horizontal scrolling");assert.ok(dims.svg>dims.container);
    await mobile.locator("#legend-button").tap();const box=await mobile.locator("#legend-card").boundingBox();assert.ok(box.x>=0&&box.x+box.width<=width+1);
    await mobile.locator('[data-close="legend"]').tap();await mobile.locator("#lesson-button").tap();assert.ok(await mobile.locator("#hover-card").isVisible());
    await mobile.locator('[data-close="hover"]').tap();assert.equal(await mobile.locator("#hover-card").isVisible(),false);
    await mobile.close();
  }
  await page.emulateMedia({reducedMotion:"reduce"});await page.locator("#reset").click();await page.locator("#step").click();assert.equal((await snapshot()).index,0);
  const nojs=await browser.newPage({javaScriptEnabled:false});await nojs.goto(url);await nojs.locator("#text-walkthrough > summary").click();
  assert.equal(await nojs.locator(".text-phase").count(),6);assert.equal(await nojs.locator(".inventory-item").count(),40);
  await nojs.close();assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  console.log(JSON.stringify({result:"PASS",chapters:6,contrastAudits:audited,renderedEdges,mobileWidths:[320,390],checks:["no autoplay","hover/focus/touch cards","click-only legend","complete source explorer","object labels fit","graph/artifact/init/output gates","reset/seek/pause","pause offscreen","reduced motion","offline/no-JavaScript"],evidence:"CPU-RUN headless Chromium; no MAX execution"},null,2));
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1);});
