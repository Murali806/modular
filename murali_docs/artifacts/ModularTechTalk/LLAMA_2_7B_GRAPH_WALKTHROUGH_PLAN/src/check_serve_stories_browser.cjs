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
    await audit(page,"Visible saturation and rejection");
    await pressure.screenshot({path:"/tmp/max-serve-admission-rejection.png"});
    await pressure.locator('[data-ss-option="queue"]').selectOption("6");
    assert.equal(await pressure.getAttribute("data-rejected"),"0");
    await pressure.locator('[data-ss-action="send"]').click();
    await pressure.locator('[data-ss-action="step"]').click();
    await stepUntil(page,"pressure",s=>s[0].accepted===7);
    assert.equal(await pressure.getAttribute("data-rejected"),"0");
    checks.push("Burst fills M then N; 429 selects the rejected request; larger N accepts the same burst");

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
    await busy.locator('.ss-evidence>summary').click();
    await busy.locator('a[data-ref="proxy_submit"]').click();
    assert.equal(await page.locator("#source-dialog").evaluate(n=>n.open),true);
    assert.match(await page.locator("#source-file").innerText(),/RequestQueueFull/);
    await page.keyboard.press("Escape");
    checks.push("Later batch admission, selected-request lifecycle link and exact-source explorer");

    await page.goto(url("driver_and_scaling"));
    assert.equal(await page.locator("[data-serve-story]").count(),3);
    await audit(page,"Systems stories: initial page");
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
    assert.ok(fleetState[0].requests.some(r=>r.id==="R1"));
    assert.ok(!fleetState[1].requests.some(r=>r.id==="R1"));
    await audit(page,"Independent replica routing");
    await fleet.screenshot({path:"/tmp/max-serve-replica-story.png"});
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
        await audit(mobile.page,"Touch story "+width+" "+name);
      }
      await mobile.page.screenshot({path:"/tmp/max-serve-story-mobile-"+width+".png"});
      await mobile.context.close();
    }
    checks.push("320/390 px touch controls, readable contrast and contained diagram scrolling");

    const nojs=await setup({viewport:{width:390,height:844},javaScriptEnabled:false});
    await nojs.page.goto(url("components_and_flow"));
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
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const result={result:"PASS",checked_at:new Date().toISOString(),browser:await browser.version(),checks,audits,errors,external_runtime_requests:network,evidence:"CPU-RUN browser validation; illustrative source-backed models, no GPU measurement"};
    fs.writeFileSync(path.join(task,"validation/serve_stories_browser.json"),JSON.stringify(result,null,2)+"\n");
    console.log(JSON.stringify({result:result.result,checks,contrast_audits:audits.length},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
