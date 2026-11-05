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
    const key=row.dataset.component, shape=document.querySelector('.ss-pipeline [data-component="'+key+'"] > rect');
    return {key,color:getComputedStyle(row.querySelector(".ss-legend-swatch")).backgroundColor,stroke:shape&&getComputedStyle(shape).stroke};
  }));
  assert.equal(new Set(colors.map(c=>c.color)).size,colors.length,"Every component has a distinct legend color");
  colors.forEach(c=>assert.equal(c.color,c.stroke,"Animation and legend agree: "+c.key));
  assert.match(await legendCard.innerText(),/asyncio\.Lock/);
  assert.match(await legendCard.innerText(),/multiprocessing\.Event/);
  await audit(page,"Component legend "+key);
  await page.screenshot({path:"/tmp/max-serve-legend-"+key+".png"});
  const ipc=legend.locator('[data-component="ipc"]');
  await ipc.locator("summary").click();
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
    assert.equal(await page.locator("[data-machine]").count(),3);
    assert.equal(await page.locator("#reference-atlas").evaluate(n=>n.open),false);
    await page.waitForTimeout(250);
    assert.equal(await page.locator("#story-one").getAttribute("data-time"),"0");
    await audit(page,"Request stories: initial page");
    await page.screenshot({path:"/tmp/max-serve-request-stories.png"});
    await cardsAndLegend(page,"one",audit);
    await page.locator('#story-one [data-ss-action="step"]').focus();
    await page.keyboard.press("Enter");
    assert.notEqual(await page.locator("#story-one").getAttribute("data-time"),"0");
    await stepUntil(page,"one",s=>s[0].done===1);
    assert.match(await page.locator('#story-one [data-request-detail]').innerText(),/Completed/);
    checks.push("Four request lessons, three preserved machines, no autoplay and keyboard stepping to completion");

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
    await audit(page,"Systems stories: initial page");
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
        const story=mobile.page.locator('[data-serve-story]').first();
        await story.locator('[data-ss-action="step"]').tap();
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
      }
      await mobile.page.screenshot({path:"/tmp/max-serve-story-mobile-"+width+".png"});
      await mobile.context.close();
    }
    checks.push("320/390 px touch controls, readable contrast and contained diagram scrolling");

    const nojs=await setup({viewport:{width:390,height:844},javaScriptEnabled:false});
    await nojs.page.goto(url("components_and_flow"));
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
