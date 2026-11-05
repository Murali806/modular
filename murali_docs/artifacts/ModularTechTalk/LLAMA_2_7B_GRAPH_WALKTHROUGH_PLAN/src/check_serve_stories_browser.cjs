"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const contrast = require("./contrast.cjs");
const task = path.resolve(__dirname,"..");
const url = page => pathToFileURL(path.join(task,"link_htmls/max_serve_os_"+page+".html")).href;
const errors=[], network=[], checks=[], audits=[];
async function inViewport(page, locator) {
  const box=await locator.boundingBox(), view=page.viewportSize();
  assert.ok(box && box.x>=0 && box.y>=0 && box.x+box.width<=view.width+1 && box.y+box.height<=view.height+1,"Card stays within the viewport");
}
async function clickLegend(page) {
  // A physical click preserves the current scene; locator.click scrolls a
  // sticky element back toward its original document position in Chromium.
  const box=await page.locator(".ss-legend-button").boundingBox();
  await page.mouse.click(box.x+box.width/2,box.y+box.height/2);
}
async function cardsAndLegend(page,key,audit) {
  const story=page.locator("#story-"+key), lesson=story.locator("[data-ss-lesson]");
  const badge=lesson.locator(":scope > summary"), card=lesson.locator(":scope > aside");
  assert.equal(await lesson.evaluate(n=>n.open),false,"Lesson explanation starts hidden");
  const layout=await story.evaluate(n=>({outer:n.getBoundingClientRect().width,animation:n.querySelector(".ss-experiment").getBoundingClientRect().width}));
  assert.equal(layout.outer,layout.animation,"The animation uses the full story width");
  await badge.hover();
  assert.equal(await lesson.evaluate(n=>n.open),true);
  await card.locator("h2").hover();
  await page.waitForTimeout(180);
  assert.equal(await lesson.evaluate(n=>n.open),true,"Moving into the explanation keeps it readable");
  await inViewport(page,card);
  await audit(page,"Lesson hover card "+key);
  await page.screenshot({path:"/tmp/max-serve-lesson-card-"+key+".png"});
  await page.mouse.move(page.viewportSize().width-4,page.viewportSize().height-4);
  await page.waitForTimeout(180);
  assert.equal(await lesson.evaluate(n=>n.open),false,"Leaving the badge and explanation dismisses it");
  await badge.focus();
  await page.waitForTimeout(80);
  assert.equal(await lesson.evaluate(n=>n.open),true,"Keyboard focus reveals the explanation even when it scrolls into view");
  await page.keyboard.press("Enter");
  assert.equal(await card.evaluate(n=>n===document.activeElement),true);
  await page.keyboard.press("Escape");
  assert.equal(await lesson.evaluate(n=>n.open),false);
  assert.equal(await badge.evaluate(n=>n===document.activeElement),true);
  await page.keyboard.press("Tab");
  assert.equal(await lesson.evaluate(n=>n.open),false);

  const legend=page.locator("[data-ss-legend]"), legendCard=page.locator("#ss-legend-card");
  const buttonBox=await page.locator(".ss-legend-button").boundingBox();
  await page.mouse.move(buttonBox.x+20,buttonBox.y+20);
  assert.equal(await legend.evaluate(n=>n.open),false,"The legend opens on click, not hover");
  const before=await page.evaluate(()=>scrollY);
  await clickLegend(page);
  assert.equal(await legend.evaluate(n=>n.open),true);
  assert.equal(await page.evaluate(()=>scrollY),before,"Opening the legend keeps the current scene in place");
  await inViewport(page,legendCard);
  const colors=await page.evaluate(()=>[...document.querySelectorAll(".ss-legend-rows > li")].map(row=>{
    const key=row.dataset.component, shape=document.querySelector('.ss-visual [data-component="'+key+'"] > .ss-component-box');
    return {key,color:getComputedStyle(row.querySelector(".ss-legend-swatch")).backgroundColor,stroke:shape&&getComputedStyle(shape).stroke};
  }));
  assert.equal(new Set(colors.map(c=>c.color)).size,colors.length,"Every component has a distinct legend color");
  colors.forEach(c=>assert.equal(c.color,c.stroke,"Animation and legend agree: "+c.key));
  assert.match(await legendCard.innerText(),/asyncio\.Lock/);
  assert.match(await legendCard.innerText(),/multiprocessing\.Event/);
  await audit(page,"Component legend "+key);
  await page.screenshot({path:"/tmp/max-serve-legend-"+key+".png"});
  const ipc=legend.locator('[data-component="ipc"]');
  if (!await ipc.locator(".ss-legend-source").evaluate(n=>n.open)) await ipc.locator("summary").click();
  await ipc.locator('a[data-ref="ipc_address"]').click();
  assert.equal(await page.locator("#source-dialog").evaluate(n=>n.open),true);
  assert.match(await page.locator("#source-file").innerText(),/ipc:\/\//);
  await page.keyboard.press("Escape");
  assert.equal(await legend.evaluate(n=>n.open),true,"The source modal can return to its legend link");
  await page.keyboard.press("Escape");
  assert.equal(await legend.evaluate(n=>n.open),false);
  await clickLegend(page);
  await page.mouse.click(4,page.viewportSize().height-4);
  assert.equal(await legend.evaluate(n=>n.open),false,"Outside click dismisses the legend");
  checks.push("Full-width "+key+" animation, numbered hover/focus card, click-only L legend, source links and matching component colors");
}
async function stepUntil(page,key,predicate,limit=200) {
  const result = await page.evaluate(({key,body,limit}) => {
    const test = new Function("snap", "return ("+body+")(snap)");
    for(let i=0;i<limit;i++) {
      const snap=window.serveStorySnapshot(key);
      if(test(snap))return {success:true,steps:i};
      document.querySelector('[data-serve-story="'+key+'"] [data-ss-action="step"]').click();
    }
    return {success:false};
  }, {key,body:String(predicate),limit});
  assert.ok(result.success,"Expected browser scenario: "+key);
}
async function stepApiUntil(page, predicate) {
  const success = await page.evaluate(body => {
    const test = new Function("s", "return ("+body+")(s)");
    for (let i=0; i<25; i++) {
      if (test(window.serveStorySnapshot("api"))) return true;
      document.querySelector('#story-api [data-api-action="step"]').click();
    }
    return false;
  }, String(predicate));
  assert.ok(success, "API scene reaches the expected visible outcome");
}
async function apiIntro(page, audit, name) {
  const api=page.locator("#story-api");
  assert.equal(await page.locator(".ss-story").first().getAttribute("id"),"story-api");
  assert.equal(await api.getAttribute("data-api-step"),"0");
  assert.equal(await api.getAttribute("data-playing"),"false");
  assert.equal(await api.locator("[data-api-mode]").count(),6);
  assert.equal(await api.locator('.ss-lesson-button').innerText(),"01");
  await api.locator('[data-api-action="step"]').focus();
  await page.keyboard.press("Enter");
  assert.equal(await api.getAttribute("data-api-step"),"1");
  await stepApiUntil(page,s=>s.lockOwner==="R1"&&s.requests[1].state==="awaiting admission lock");
  assert.match(await api.locator('[data-api-state="R2"]').innerText(),/awaiting admission lock/);
  await audit(page,"API shared loop and lock "+name);
  await api.screenshot({path:"/tmp/max-serve-api-locked-"+name+".png"});
  await stepApiUntil(page,s=>s.requests[1].state==="HTTP 429 returned");
  assert.equal(await api.locator('[data-api-slot]').getAttribute("data-occupied"),"true");
  assert.equal(await api.locator('[data-api-lock]').textContent(),"lock free");
  await stepApiUntil(page,s=>s.finished);
  assert.match(await api.locator('[data-api-state="R1"]').innerText(),/first chunk delivered/);
  assert.equal(await api.locator('[data-api-slot]').getAttribute("data-occupied"),"false");
  assert.equal(await api.locator('[data-api-action="step"]').isDisabled(),true);
  assert.match(await api.locator('[data-api-action="play"]').innerText(),/Replay/);

  for (const [mode,role] of [["no-server","uvicorn"],["no-app","fastapi"],["no-loop","loop"]]) {
    await api.locator('[data-api-mode="'+mode+'"]').click();
    await api.locator('[data-api-action="step"]').click();
    const snap=await page.evaluate(()=>window.serveStorySnapshot("api"));
    assert.equal(snap.event.kind,"blocked");
    assert.deepEqual(snap.queue,[]);
    assert.equal(await api.locator('[data-api-component="'+role+'"]').getAttribute("data-missing"),"true");
    assert.equal(await api.locator('[data-api-mode][aria-pressed="true"]').getAttribute("data-api-mode"),mode);
    await audit(page,"API absent role "+role+" "+name);
  }
  await api.locator('[data-api-mode="no-lock"]').click();
  assert.match(await api.locator('[data-api-admission-summary]').textContent(),/tasks can interleave/);
  await stepApiUntil(page,s=>s.probes.R1&&s.probes.R2&&s.queue.length===0);
  assert.match(await api.locator('[data-api-probes]').textContent(),/R1 probe ✓ · R2 probe ✓/);
  await stepApiUntil(page,s=>s.finished);
  assert.match(await api.locator('[data-api-state="R2"]').innerText(),/send can wait/);
  assert.match(await api.locator('[data-api-purpose]').textContent(),/Possible interleaving/);
  assert.equal(await api.locator('[data-api-slot]').getAttribute("data-occupied"),"true");
  await audit(page,"API possible no-lock interleaving "+name);
  await api.screenshot({path:"/tmp/max-serve-api-unlocked-"+name+".png"});

  await api.locator('[data-api-mode="asyncio"]').click();
  await stepApiUntil(page,s=>s.finished);
  assert.match(await api.locator('[data-api-loop-name]').textContent(),/^asyncio/);
  assert.match(await api.locator('[data-api-purpose]').textContent(),/configuration changes/);
  assert.match(await api.locator('[data-api-state="R1"]').innerText(),/first chunk delivered/);
  await api.locator('[data-api-mode="normal"]').click();
  assert.equal(await api.getAttribute("data-api-step"),"0");
  assert.equal(await api.locator('[data-api-component][data-missing="true"]').count(),0);
  assert.equal(await api.locator('[data-api-slot]').getAttribute("data-occupied"),"false");
  await api.locator('.ss-api-transcript>summary').click();
  assert.equal(await api.locator('.ss-api-role-table tbody tr').count(),4);
  await api.locator('.ss-api-role-table a[data-ref="proxy_submit"]').click();
  assert.equal(await page.locator('#source-dialog').evaluate(n=>n.open),true);
  assert.match(await page.locator('#source-file').innerText(),/async with self\._admission_lock/);
  await page.keyboard.press("Escape");
  await api.locator('.ss-api-transcript>summary').click();
  checks.push("API introduction on "+name+": keyboard controls, normal stream/429, three absent-role stops, no-lock backpressure, replacement loop, reset and pinned-source links");
}
async function stepWorkerUntil(page, predicate) {
  const success=await page.evaluate(body=>{
    const test=new Function("s", "return ("+body+")(s)");
    for(let i=0;i<150;i++) {
      if(test(window.serveStorySnapshot("worker")))return true;
      document.querySelector('#story-worker [data-worker-action="step"]').click();
    }
    return false;
  },String(predicate));
  assert.ok(success,"Worker close-up reaches its expected outcome");
}
async function workerIntro(page,audit,name) {
  const worker=page.locator('#story-worker');
  assert.equal(await page.locator('.ss-story').nth(1).getAttribute('id'),'story-worker');
  assert.equal(await worker.locator('.ss-lesson-button').innerText(),'02');
  assert.equal(await worker.getAttribute('data-worker-step'),'0');
  assert.equal(await worker.getAttribute('data-playing'),'false');
  assert.equal(await worker.locator('[data-worker-mode]').count(),4);
  assert.equal(await worker.locator('[data-api-thread]').count(),0,"Thread badges stay confined to the API animation");
  assert.equal(await worker.locator('[data-worker-ipc-slot][data-occupied="true"]').count(),2);
  await worker.locator('[data-worker-action="step"]').focus();
  await page.keyboard.press('Enter');
  assert.equal(await worker.getAttribute('data-worker-step'),'1');
  assert.equal(await worker.locator('[data-worker-ipc-slot][data-occupied="true"]').count(),0);
  await stepWorkerUntil(page,s=>s.phase==='advance'&&s.requests[0].promptDone===8);
  assert.match(await worker.locator('[data-worker-state="R1"]').innerText(),/prefill 8\/12 · output 0/);
  assert.equal(await worker.locator('[data-worker-page-slot][data-occupied="true"]').count(),3);
  await audit(page,'Worker partial prefill '+name);
  await worker.screenshot({path:'/tmp/max-serve-worker-prefill-'+name+'.png'});
  await stepWorkerUntil(page,s=>s.results.length===2);
  assert.match(await worker.locator('[data-worker-published="R1"]').textContent(),/R1: t1/);
  assert.match(await worker.locator('[data-worker-published="R2"]').textContent(),/R2: t1/);
  await stepWorkerUntil(page,s=>s.finished);
  assert.equal(await worker.locator('[data-worker-page-slot][data-occupied="true"]').count(),0);
  assert.equal(await worker.locator('[data-worker-page-slot]').count(),4);
  assert.equal(await worker.locator('[data-worker-action="step"]').isDisabled(),true);
  assert.match(await worker.locator('[data-worker-action="play"]').innerText(),/Replay/);

  await worker.locator('[data-worker-mode="busy"]').click();
  assert.equal(await worker.getAttribute('data-phase'),'execute');
  assert.match(await worker.locator('[data-worker-state="R2"]').innerText(),/waiting in IPC/);
  assert.match(await worker.locator('[data-worker-gpu-work]').textContent(),/R1/);
  assert.doesNotMatch(await worker.locator('[data-worker-gpu-work]').textContent(),/R2/);
  await audit(page,'Worker already executing '+name);
  await stepWorkerUntil(page,s=>s.phase==='execute'&&s.batch.some(op=>op.id==='R2'));
  assert.match(await worker.locator('[data-worker-gpu-work]').textContent(),/R2/);

  await worker.locator('[data-worker-mode="cache"]').click();
  assert.match(await worker.locator('[data-worker-state="R2"]').innerText(),/waiting for reusable KV/);
  assert.match(await worker.locator('[data-worker-batch-slots]').textContent(),/free slot/);
  assert.equal(await worker.locator('[data-worker-page-slot][data-owner="R1"]').count(),2);
  await audit(page,'Worker KV pressure '+name);
  await worker.screenshot({path:'/tmp/max-serve-worker-cache-'+name+'.png'});
  await stepWorkerUntil(page,s=>s.requests[1].selected!==null);
  assert.equal(await worker.locator('[data-worker-page-slot][data-owner="R1"]').count(),0);
  assert.equal(await worker.locator('[data-worker-page-slot][data-owner="R2"]').count(),1);

  await worker.locator('[data-worker-mode="idle"]').click();
  await stepWorkerUntil(page,s=>s.finished);
  assert.match(await worker.locator('[data-worker-loop]').textContent(),/NO_PROGRESS.*backoff/);
  assert.equal(await worker.locator('[data-worker-phase="idle"]').getAttribute('data-current'),'true');
  await audit(page,'Worker idle backoff '+name);
  await worker.locator('[data-worker-mode="normal"]').click();
  await worker.locator('.ss-worker-transcript>summary').click();
  assert.equal(await worker.locator('.ss-api-role-table tbody tr').count(),6);
  await worker.locator('.ss-api-role-table a[data-ref="construct_batch"]').click();
  assert.equal(await page.locator('#source-dialog').evaluate(n=>n.open),true);
  assert.match(await page.locator('#source-file').innerText(),/def construct_batch/);
  await page.keyboard.press('Escape');
  await worker.locator('.ss-worker-transcript>summary').click();
  checks.push('Worker close-up on '+name+': second lesson, keyboard controls, partial prefill, result maps, busy-pass membership, KV waiting/reuse, idle backoff, reset and source links');
}
(async()=>{
  const browser=await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
  const setup=async options=>{
    const context=await browser.newContext(options);
    await context.route(/^https?:/,route=>{network.push(route.request().url());return route.abort();});
    const page=await context.newPage();page.on("pageerror",error=>errors.push(error.message));
    return {context,page};
  };
  const audit=async(page,label)=>{
    const result=await page.evaluate(contrast);
    assert.deepEqual(result.failures,[],label);
    audits.push({label,...result});
  };
  try {
    const {context,page}=await setup({viewport:{width:1440,height:1100},reducedMotion:"reduce"});
    await page.goto(url("components_and_flow"));
    assert.equal(await page.locator("[data-serve-story]").count(),4);
    assert.equal(await page.locator("[data-api-story]").count(),1);
    assert.deepEqual(await page.locator('.ss-lesson-button').allTextContents(),["01","02","03","04","05","06"]);
    assert.equal(await page.locator("[data-machine]").count(),3);
    assert.equal(await page.locator("#reference-atlas").evaluate(n=>n.open),false);
    await page.waitForTimeout(250);
    assert.equal(await page.locator("#story-one").getAttribute("data-time"),"0");
    await audit(page,"Request stories: initial page");
    await page.screenshot({path:"/tmp/max-serve-request-stories.png"});
    await apiIntro(page,audit,"components");
    await cardsAndLegend(page,"api",audit);
    await workerIntro(page,audit,"components");
    await cardsAndLegend(page,"worker",audit);
    await cardsAndLegend(page,"one",audit);
    await page.locator('#story-one [data-ss-action="step"]').focus();
    await page.keyboard.press("Enter");
    assert.notEqual(await page.locator("#story-one").getAttribute("data-time"),"0");
    await stepUntil(page,"one",s=>s[0].done===1);
    assert.match(await page.locator('#story-one [data-request-detail]').innerText(),/Completed/);
    checks.push("API and worker opening lessons plus four request lessons, three preserved machines, no autoplay and keyboard stepping to completion");

    const pressure=page.locator("#story-pressure");
    await pressure.locator('[data-ss-action="send"]').click();
    await pressure.locator('[data-ss-action="step"]').click();
    await stepUntil(page,"pressure",s=>s[0].rejected>0);
    assert.match(await pressure.locator('[data-current-event]').innerText(),/HTTP 429/);
    assert.match(await pressure.locator('[data-request-detail]').innerText(),/HTTP 429/);
    assert.equal(await pressure.locator('[data-gate]').getAttribute("data-gate"),"closed");
    assert.equal(await pressure.locator('[data-ipc-slot]').count(),2);
    assert.equal(await pressure.locator('[data-ipc-slot][data-occupied="true"]').count(),2);
    const slots=await pressure.locator('[data-ipc-slot]').evaluateAll(nodes=>nodes.map(n=>({id:n.dataset.request,slot:n.getAttribute("transform"),dot:n.closest("svg").querySelector('[data-dot-id="'+n.dataset.request+'"]').getAttribute("transform")})));
    slots.forEach(s=>assert.deepEqual(s.slot.match(/[\d.]+/g).map(Number),s.dot.match(/[\d.]+/g).map(Number),"Queued request occupies its visible slot: "+s.id));
    await audit(page,"Visible saturation and rejection");
    await pressure.screenshot({path:"/tmp/max-serve-admission-rejection.png"});
    await pressure.locator('[data-ss-option="queue"]').selectOption("6");
    assert.equal(await pressure.getAttribute("data-rejected"),"0");
    assert.equal(await pressure.locator('[data-ipc-slot]').count(),6);
    assert.equal(await pressure.locator('[data-ipc-slot][data-occupied="true"]').count(),0);
    await pressure.locator('[data-ss-action="send"]').click();
    await pressure.locator('[data-ss-action="step"]').click();
    await stepUntil(page,"pressure",s=>s[0].accepted===7);
    assert.equal(await pressure.getAttribute("data-rejected"),"0");
    assert.equal(await pressure.locator('[data-ipc-slot][data-occupied="true"]').count(),Number(await pressure.getAttribute("data-ipc")));
    await stepUntil(page,"pressure",s=>s[0].done===7,700);
    assert.equal(await pressure.locator('[data-ipc-slot][data-occupied="true"]').count(),0,"Drained queue slots become empty again");
    await pressure.locator('[data-ss-action="reset"]').click();
    assert.equal(await pressure.locator('[data-ipc-slot]').count(),6);
    checks.push("Burst fills M then visible N slots; 429 selects the rejected request; N=6 redraws six slots; occupancy follows admission, drain and reset");

    const busy=page.locator("#story-busy");
    await busy.locator('[data-ss-option="batch"]').selectOption("2");
    await busy.locator('[data-ss-action="send"]').click();
    await busy.locator('[data-ss-action="step"]').click();
    await stepUntil(page,"busy",s=>s[0].requests[1].selected!==null);
    const busyState=await page.evaluate(()=>window.serveStorySnapshot("busy")[0]);
    assert.ok(busyState.requests[0].generated<busyState.requests[0].output);
    await busy.locator('[data-dot-id="R1"]').click();
    assert.match(await busy.locator('[data-request-detail]').innerText(),/R1/);
    await busy.locator('[data-dot-id="R2"]').focus();
    await page.keyboard.press("Enter");
    assert.match(await busy.locator('[data-request-detail]').innerText(),/R2/);
    await busy.locator('.ss-request-strip [data-ss-request="R2"]').click();
    assert.match(await busy.locator('[data-request-detail]').innerText(),/R2/);
    await busy.locator('[data-request-detail] a').click();
    assert.equal(await page.locator("#machine-request").isVisible(),true);
    await busy.locator('.ss-lesson-button').hover();
    await busy.locator('.ss-evidence>summary').click();
    await busy.locator('a[data-ref="proxy_submit"]').click();
    assert.equal(await page.locator("#source-dialog").evaluate(n=>n.open),true);
    assert.match(await page.locator("#source-file").innerText(),/RequestQueueFull/);
    await page.locator("#source-file").evaluate(n=>n.scrollTop=0);
    await page.waitForTimeout(180);
    assert.equal(await busy.locator('[data-ss-lesson]').evaluate(n=>n.open),true,"Scrolling the source viewer preserves its originating hover card");
    await page.keyboard.press("Escape");
    await page.waitForTimeout(80);
    assert.equal(await busy.locator('a[data-ref="proxy_submit"]').evaluate(n=>n===document.activeElement),true,"Source close returns keyboard focus to the lesson link");
    checks.push("Later batch admission, selected-request lifecycle link and exact-source explorer");

    await page.goto(url("driver_and_scaling"));
    assert.equal(await page.locator("[data-serve-story]").count(),3);
    assert.equal(await page.locator("[data-api-story]").count(),1);
    assert.deepEqual(await page.locator('.ss-lesson-button').allTextContents(),["01","02","03","04","05"]);
    await audit(page,"Systems stories: initial page");
    await apiIntro(page,audit,"systems");
    await workerIntro(page,audit,"systems");
    await cardsAndLegend(page,"clocks",audit);
    await stepUntil(page,"reader",s=>s[0].requests[0].state==="streaming"&&s[0].buffered>0);
    assert.equal(await page.locator("#story-reader").getAttribute("data-pages"),"0");
    assert.match(await page.locator('#story-reader [data-owner-lanes]').innerText(),/buffered/);
    const fleet=page.locator("#story-replicas");
    await fleet.locator('[data-ss-action="replica"]').click();
    assert.match(await fleet.locator('[data-fleet-state="1"]').textContent(),/Initializing/);
    await fleet.locator('[data-ss-action="step"]').click();
    await stepUntil(page,"replicas",s=>s[0].time>=4);
    await fleet.locator('[data-ss-action="send"]').click();
    await fleet.locator('[data-ss-action="step"]').click();
    await stepUntil(page,"replicas",s=>s[1].accepted>0);
    const fleetState=await page.evaluate(()=>window.serveStorySnapshot("replicas"));
    for(let i=0;i<2;i++) assert.equal(await fleet.locator('[data-fleet-detail="'+i+'"] [data-ipc-slot][data-occupied="true"]').count(),fleetState[i].ipc);
    assert.ok(fleetState[0].requests.some(r=>r.id==="R1"));
    assert.ok(!fleetState[1].requests.some(r=>r.id==="R1"));
    await audit(page,"Independent replica routing");
    await fleet.screenshot({path:"/tmp/max-serve-replica-story.png"});
    await fleet.locator('[data-ss-action="reset"]').click();
    assert.equal(await fleet.locator('[data-fleet-detail="1"] [data-ipc-slot][data-occupied="true"]').count(),0);
    assert.match(await fleet.locator('[data-fleet-detail="1"] [data-ipc-slots]').getAttribute("aria-label"),/not added/);
    await page.goto(url("driver_and_scaling")+"#scale-lab");
    assert.equal(await page.locator("#reference-atlas").evaluate(n=>n.open),true);
    assert.equal(await page.locator("#lab").isVisible(),true);
    checks.push("Slow-reader output outlives KV ownership; warming replicas excluded; advanced deep links preserved");
    await context.close();

    for(const width of [390,320]) {
      const mobile=await setup({viewport:{width,height:844},hasTouch:true,isMobile:true,reducedMotion:"reduce"});
      for(const name of ["components_and_flow","driver_and_scaling"]) {
        await mobile.page.goto(url(name));
        assert.ok(await mobile.page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),"No document overflow: "+width);
        const story=mobile.page.locator('[data-api-story]').first();
        await story.locator('[data-api-mode="no-lock"]').tap();
        await story.locator('[data-api-action="step"]').tap();
        assert.equal(await story.getAttribute("data-api-step"),"1");
        await story.locator('.ss-lesson-button').tap();
        assert.equal(await story.locator('[data-ss-lesson]').evaluate(n=>n.open),true);
        await inViewport(mobile.page,story.locator('.ss-lesson-card'));
        await audit(mobile.page,"Touch lesson card "+width+" "+name);
        await story.locator('[data-card-dismiss]').tap();
        assert.equal(await story.locator('[data-ss-lesson]').evaluate(n=>n.open),false);
        await mobile.page.locator('.ss-legend-button').tap();
        assert.equal(await mobile.page.locator('[data-ss-legend]').evaluate(n=>n.open),true);
        await inViewport(mobile.page,mobile.page.locator('#ss-legend-card'));
        await audit(mobile.page,"Touch legend "+width+" "+name);
        await mobile.page.locator('#ss-legend-card [data-card-dismiss]').tap();
        await audit(mobile.page,"Touch story "+width+" "+name);
        const worker=mobile.page.locator('#story-worker');
        await worker.locator('[data-worker-mode="cache"]').tap();
        await worker.locator('[data-worker-action="step"]').tap();
        assert.equal(await worker.getAttribute('data-worker-step'),'1');
        assert.match(await worker.locator('[data-worker-state="R2"]').innerText(),/waiting for reusable KV/);
        await worker.locator('.ss-lesson-button').tap();
        assert.equal(await worker.locator('[data-ss-lesson]').evaluate(n=>n.open),true);
        await inViewport(mobile.page,worker.locator('.ss-lesson-card'));
        await audit(mobile.page,'Touch worker card '+width+' '+name);
        await worker.locator('[data-card-dismiss]').tap();
        assert.equal(await worker.locator('[data-ss-lesson]').evaluate(n=>n.open),false);
        assert.ok(await mobile.page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Worker scene keeps document scrolling contained');
      }
      await mobile.page.screenshot({path:"/tmp/max-serve-story-mobile-"+width+".png"});
      await mobile.context.close();
    }
    checks.push("320/390 px touch controls, readable contrast and contained diagram scrolling");

    const nojs=await setup({viewport:{width:390,height:844},javaScriptEnabled:false});
    await nojs.page.goto(url("components_and_flow"));
    assert.equal(await nojs.page.locator('#story-api .ss-lesson-card h2').isVisible(),true);
    assert.equal(await nojs.page.locator('#story-api [data-api-controls]').first().isVisible(),false);
    await nojs.page.locator('#story-api .ss-api-transcript>summary').click();
    assert.equal(await nojs.page.locator('#story-api .ss-api-role-table').isVisible(),true);
    assert.match(await nojs.page.locator('#story-api .ss-transcript pre').innerText(),/Without lock \(possible interleaving\)/);
    assert.equal(await nojs.page.locator('#story-worker .ss-lesson-card h2').isVisible(),true);
    assert.equal(await nojs.page.locator('#story-worker [data-worker-controls]').first().isVisible(),false);
    await nojs.page.locator('#story-worker .ss-worker-transcript>summary').click();
    assert.equal(await nojs.page.locator('#story-worker .ss-api-role-table').isVisible(),true);
    assert.match(await nojs.page.locator('#story-worker .ss-transcript pre').innerText(),/NO_PROGRESS.*backoff/);
    assert.equal(await nojs.page.locator('#story-one .ss-lesson-card h2').isVisible(),true);
    await nojs.page.locator('.ss-legend-button').click();
    assert.equal(await nojs.page.locator('[data-ss-legend]').evaluate(n=>n.open),true);
    await nojs.page.locator('.ss-legend-button').click();
    assert.equal(await nojs.page.locator('#story-one [data-ss-controls]').isVisible(),false);
    await nojs.page.locator('#story-pressure .ss-transcript>summary').click();
    assert.match(await nojs.page.locator('#story-pressure .ss-transcript pre').innerText(),/HTTP 429/);
    assert.equal(await nojs.page.locator("#reference-atlas").evaluate(n=>n.open),true);
    await nojs.context.close();
    checks.push("No-JavaScript lesson text, portable flows and expanded reference atlas");

    const motion=await setup({viewport:{width:1440,height:1100},reducedMotion:"no-preference"});
    await motion.page.goto(url("components_and_flow"));
    const apiMotion=motion.page.locator('#story-api');
    await apiMotion.locator('[data-api-action="play"]').click();
    const apiDot=apiMotion.locator('[data-api-request="R1"]');
    const apiBefore=await apiDot.getAttribute("transform");
    await motion.page.waitForFunction(()=>window.serveStorySnapshot("api").index===1);
    await motion.page.waitForTimeout(250);
    assert.notEqual(await apiDot.getAttribute("transform"),apiBefore,"API request markers animate during playback");
    await apiMotion.locator('[data-api-action="play"]').click();
    const apiPaused=await apiMotion.getAttribute('data-api-step');
    await motion.page.waitForTimeout(1450);
    assert.equal(await apiMotion.getAttribute('data-api-step'),apiPaused);
    await apiMotion.locator('[data-api-action="play"]').click();
    await motion.page.locator('#state-machines').scrollIntoViewIfNeeded();
    await motion.page.waitForTimeout(150);
    assert.equal(await apiMotion.getAttribute('data-playing'),'false');
    const workerMotion=motion.page.locator('#story-worker');
    await workerMotion.locator('[data-worker-action="play"]').click();
    const workerDot=workerMotion.locator('[data-worker-request="R1"]');
    const workerBefore=await workerDot.getAttribute('transform');
    await motion.page.waitForFunction(()=>window.serveStorySnapshot('worker').index===1);
    await motion.page.waitForTimeout(250);
    assert.notEqual(await workerDot.getAttribute('transform'),workerBefore,'Worker contexts move from IPC into the process');
    await workerMotion.locator('[data-worker-action="play"]').click();
    const workerPaused=await workerMotion.getAttribute('data-worker-step');
    await motion.page.waitForTimeout(1450);
    assert.equal(await workerMotion.getAttribute('data-worker-step'),workerPaused);
    await workerMotion.locator('[data-worker-action="play"]').click();
    await motion.page.locator('#state-machines').scrollIntoViewIfNeeded();
    await motion.page.waitForTimeout(150);
    assert.equal(await workerMotion.getAttribute('data-playing'),'false');
    await motion.page.locator('#story-one [data-ss-action="send"]').click();
    const dot=motion.page.locator('#story-one [data-dot-id="R2"]');
    const before=await dot.getAttribute("transform");await motion.page.waitForTimeout(200);
    assert.notEqual(await dot.getAttribute("transform"),before,"Requests physically move between stations");
    await motion.page.locator('#story-one [data-ss-action="play"]').click();
    const time=await motion.page.locator("#story-one").getAttribute("data-time");
    await motion.page.waitForTimeout(1000);
    assert.equal(await motion.page.locator("#story-one").getAttribute("data-time"),time);
    await motion.page.locator('#story-one [data-ss-action="play"]').click();
    await motion.page.locator("#state-machines").scrollIntoViewIfNeeded();
    await motion.page.waitForTimeout(150);
    assert.equal(await motion.page.locator("#story-one").getAttribute("data-playing"),"false");
    await motion.context.close();
    checks.push("Moving request markers, stable pause and automatic pause when the lesson leaves view");
    const print=await setup({viewport:{width:1200,height:900},reducedMotion:"reduce"});
    for(const name of ["components_and_flow","driver_and_scaling"]) {
      await print.page.goto(url(name));
      await print.page.evaluate(()=>window.dispatchEvent(new Event("beforeprint")));
      await print.page.emulateMedia({media:"print"});
      assert.equal(await print.page.locator('.ss-lesson-card h2').first().isVisible(),true);
      assert.equal(await print.page.locator('#ss-legend-card').isVisible(),true);
      assert.equal(await print.page.locator('#ss-legend-card').evaluate(n=>getComputedStyle(n).position),"static");
      assert.equal(await print.page.locator('#story-api .ss-api-role-table').isVisible(),true);
      assert.equal(await print.page.locator('#story-worker .ss-api-role-table').isVisible(),true);
      assert.equal(await print.page.locator('#story-worker .ss-lesson-card h2').isVisible(),true);
      await print.page.emulateMedia({media:"screen"});
      await print.page.evaluate(()=>window.dispatchEvent(new Event("afterprint")));
      assert.equal(await print.page.locator('[data-ss-lesson]').first().evaluate(n=>n.open),false);
    }
    await print.context.close();
    checks.push("Printing expands lesson and legend content and restores the collapsed screen layout afterward");
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const result={result:"PASS",checked_at:new Date().toISOString(),browser:await browser.version(),checks,audits,errors,external_runtime_requests:network,evidence:"CPU-RUN browser validation; illustrative source-backed models, no GPU measurement"};
    fs.writeFileSync(path.join(task,"validation/serve_stories_browser.json"),JSON.stringify(result,null,2)+"\n");
    console.log(JSON.stringify({result:result.result,checks,contrast_audits:audits.length},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
