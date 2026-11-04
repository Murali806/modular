"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const {pathToFileURL} = require("url");
const {chromium} = require("playwright");
const contrast = require("./contrast.cjs");
const task = path.resolve(__dirname,"..");
const url = pathToFileURL(path.join(task,"link_htmls/max_serve_os_components_and_flow.html")).href;
const checks=[], audits=[], screenshots=[], errors=[], network=[];

async function audit(page,state) {
  const result=await page.evaluate(contrast);
  assert.deepEqual(result.failures,[],"Rendered contrast: "+state);
  audits.push({state,...result});
}
async function closeCard(page) {
  if(await page.locator("#reading-card").isVisible())await page.keyboard.press("Escape");
  await page.mouse.move(0,0);
}
async function openAncestors(locator) {
  await locator.evaluate(n=>{for(let e=n;e;e=e.parentElement)if(e instanceof HTMLDetailsElement)e.open=true;});
}
async function shot(page,name) {
  await page.evaluate(()=>getSelection()?.removeAllRanges());
  await page.screenshot({path:path.join(task,"validation/"+name)});screenshots.push(name);
}

(async()=>{
  const browser=await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
  const setup=async options=>{
    const context=await browser.newContext(options);
    await context.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});
    await context.addInitScript(()=>Object.defineProperty(navigator,"clipboard",{configurable:true,value:{writeText:async value=>{window.__copied=value;}}}));
    const page=await context.newPage();page.on("pageerror",e=>errors.push(e.message));
    return {context,page};
  };
  try {
    const {context,page}=await setup({viewport:{width:1600,height:1100},reducedMotion:"reduce"});
    await page.goto(url);
    const data=await page.locator("#component-data").evaluate(n=>JSON.parse(n.textContent));
    const card=page.locator("#reading-card");
    assert.equal(await page.locator("[data-flow]").count(),3);
    assert.equal(await page.locator("[data-machine]").count(),3);
    assert.equal(await page.locator('input[type="number"], [data-preset], #lab').count(),0);
    await page.waitForTimeout(200);
    assert.equal(await page.locator("#flow-startup").getAttribute("data-frame"),"0");
    assert.equal(await page.locator("#machine-server").getAttribute("data-state"),"configured");
    assert.equal(await page.locator("#flow-startup").getAttribute("data-http"),"Not accepting");
    await audit(page,"Initial focused page");await shot(page,"os_components_overview.png");
    checks.push("offline focused scope, three flows/three machines and no autoplay");

    let components=0;
    for(const [key,record] of Object.entries(data.components)) {
      await closeCard(page);
      const trigger=page.locator('[data-card="component:'+key+'"]').first();
      await openAncestors(trigger);await trigger.hover();
      assert.equal(await card.getAttribute("data-key"),"component:"+key);
      assert.ok((await card.innerText()).includes(record.role));
      assert.ok((await card.innerText()).includes(record.interacts));
      assert.ok((await card.innerText()).includes(record.entry));components++;
    }
    assert.equal(components,15);
    await page.mouse.move(0,0);await page.waitForTimeout(180);
    assert.equal(await card.isVisible(),true,"hover card persists after pointer leaves");
    await card.locator("#card-hold").click();
    const heldKey=await card.getAttribute("data-key");
    await page.locator('[data-card="component:api"]').first().hover();
    assert.equal(await card.getAttribute("data-key"),heldKey);
    await card.locator("#card-hold").click();
    await page.mouse.move(0,0);await page.locator('[data-card="component:api"]').first().hover();
    assert.equal(await card.getAttribute("data-key"),"component:api");
    checks.push("all 15 roles, layer/call-stack mapping and persistent Hold/Release cards");

    await closeCard(page);
    const keyTrigger=page.locator('#flow-startup [data-story-step="1"]');
    await keyTrigger.focus();await page.keyboard.press("Space");
    assert.equal(await card.getAttribute("data-held"),"true");
    assert.equal(await page.evaluate(()=>document.activeElement.id),"reading-card");
    await page.keyboard.press("Tab");assert.equal(await page.evaluate(()=>document.activeElement.id),"card-close");
    await card.locator('a[data-ref="spawn"]').click();
    assert.equal(await page.locator("#source-dialog").evaluate(n=>n.open),true);
    await page.keyboard.press("Escape");
    await page.waitForFunction(()=>document.activeElement.matches('#flow-startup [data-story-step="1"]'));
    assert.equal(await card.isVisible(),false);
    assert.ok(await keyTrigger.evaluate(n=>n===document.activeElement));
    await keyTrigger.focus();await page.keyboard.press("Space");await page.keyboard.press("Escape");
    assert.equal(await card.isVisible(),false);assert.ok(await keyTrigger.evaluate(n=>n===document.activeElement));
    checks.push("keyboard card entry, Tab/Escape and source-dialog focus return");

    let steps=0;
    for(const flow of data.flows) {
      const node=page.locator("#flow-"+flow.id);await node.evaluate(n=>n.open=true);
      for(let i=0;i<flow.steps.length;i++) {
        await closeCard(page);
        await node.locator('[data-story-step="'+i+'"]').click();
        assert.equal(await node.getAttribute("data-frame"),String(i));
        assert.equal(await card.getAttribute("data-key"),"step:"+flow.id+":"+i);
        assert.ok((await card.innerText()).includes(flow.steps[i].owner));
        assert.ok((await card.innerText()).includes(flow.steps[i].interaction));
        assert.ok((await card.innerText()).includes(flow.steps[i].result));
        assert.equal(await node.locator(".sequence-event.active").getAttribute("data-frame"),String(i));
        if(flow.id==="startup")assert.equal(await node.getAttribute("data-http"),i===flow.steps.length-1?"Accepting":"Not accepting");
        steps++;
      }
      await audit(page,flow.id+" selected step and roles");
    }
    assert.equal(steps,26);await closeCard(page);
    await page.locator('#flow-startup [data-action="replay"]').click();
    await page.locator("#flow-startup").scrollIntoViewIfNeeded();
    await shot(page,"os_components_startup.png");
    checks.push("all 26 owner/interaction/state-change cards and readiness before HTTP serving");

    let paths=0,states=0;
    for(const machine of data.machines) {
      const node=page.locator("#machine-"+machine.id);
      for(const [key,scenario] of Object.entries(machine.scenarios)) {
        await node.locator("[data-scenario]").selectOption(key);
        for(let i=0;i<scenario.path.length;i++) {
          if(i)await node.locator('[data-action="next"]').click();
          assert.equal(await node.getAttribute("data-state"),scenario.path[i]);
          assert.equal(await node.locator(".state-node.active").count(),1);
          assert.equal(await node.locator(".state-node.active").getAttribute("data-state"),scenario.path[i]);
          assert.equal(await node.locator(".state-edge.active").count(),i?1:0);
          if(i)assert.equal(await node.locator(".state-edge.active").getAttribute("data-edge"),scenario.path[i-1]+"-"+scenario.path[i]);
        }
        assert.equal(await node.locator('[data-action="next"]').isDisabled(),true);
        await node.locator('[data-action="replay"]').click();assert.equal(await node.locator('[data-action="back"]').isDisabled(),true);
        paths++;
      }
      for(const [key,record] of Object.entries(machine.states)) {
        await closeCard(page);await node.locator('.state-node[data-state="'+key+'"]').hover();
        assert.equal(await card.getAttribute("data-key"),"state:"+machine.id+":"+key);
        assert.ok((await card.innerText()).includes(record.meaning));states++;
      }
      assert.ok(await node.locator(".state-edge").evaluateAll(es=>es.every(e=>Number.isFinite(e.getTotalLength())&&e.getTotalLength()>0)));
      await audit(page,machine.id+" state machine and card");
    }
    assert.equal(paths,8);assert.equal(states,19);
    await closeCard(page);await page.locator("#machine-request [data-scenario]").selectOption("cancel");
    for(let i=0;i<4;i++)await page.locator('#machine-request [data-action="next"]').click();
    assert.equal(await page.locator("#machine-request").getAttribute("data-state"),"cancelling");
    assert.ok((await page.locator("#machine-request [data-machine-detail]").innerText()).includes("cancellation"));
    await page.locator("#machine-server").scrollIntoViewIfNeeded();await closeCard(page);await shot(page,"os_components_state_machine.png");
    checks.push("all 8 legal state paths, 19 state meanings, failure/cancellation and decode transitions");

    let refs=0;
    await page.locator(".source-directory").evaluate(n=>n.open=true);
    for(const [key,ref] of Object.entries(data.references)) {
      await closeCard(page);
      await page.locator('.source-directory a[data-ref="'+key+'"]').click();
      assert.equal(await page.locator("#source-dialog").evaluate(n=>n.open),true);
      assert.equal(await page.locator("#source-file .selected").getAttribute("data-line"),String(ref.line));
      assert.equal(await page.locator("#source-file .source-text").evaluateAll(es=>es.map(e=>e.textContent).join("\n")+"\n"),data.files[ref.file].text);
      if(key==="session_native") {
        await page.locator("#source-copy").click();assert.equal(await page.evaluate(()=>window.__copied),ref.code);
        await page.locator("#source-search").fill("_InferenceSession");assert.match(await page.locator("#source-count").innerText(),/^1 \/ /);
        await page.locator("#source-next").click();
        await audit(page,"Exact source, selection and search");
      }
      await page.keyboard.press("Escape");
      await page.waitForFunction(key=>document.activeElement.dataset.ref===key,key);
      refs++;
    }
    assert.equal(refs,32);checks.push("all 32 exact source targets, full-file fidelity, complete copying and search");

    await page.waitForFunction(()=>document.querySelector("#copy-status").hidden);
    await page.locator('#flow-device [data-story-step="1"]').click();await audit(page,"Driver role card");
    await shot(page,"os_components_driver_card.png");await closeCard(page);
    await page.waitForFunction(()=>document.querySelector("#copy-status").hidden);
    await page.locator("#flow-request").evaluate(n=>n.open=false);
    const openBefore=await page.locator("#flow-request").evaluate(n=>n.open);
    await page.evaluate(()=>window.dispatchEvent(new Event("beforeprint")));await page.emulateMedia({media:"print"});
    assert.equal(await page.locator("#component-library").isVisible(),true);
    assert.equal(await page.locator("#flow-request").evaluate(n=>n.open),true);
    await audit(page,"Printed roles, sequences and state meanings");
    await page.emulateMedia({media:"screen"});await page.evaluate(()=>window.dispatchEvent(new Event("afterprint")));
    assert.equal(await page.locator("#flow-request").evaluate(n=>n.open),openBefore);
    checks.push("print expands complete explanations and restores disclosure state");

    for(const width of [390,320]) {
      const touch=await setup({viewport:{width,height:844},hasTouch:true,isMobile:true,reducedMotion:"reduce"});
      await touch.page.goto(url+"#flow-device");
      const trigger=touch.page.locator('#flow-device [data-story-step="1"]');await trigger.tap();
      const tc=touch.page.locator("#reading-card");
      assert.equal(await tc.getAttribute("data-held"),"true");assert.equal(await tc.getAttribute("data-key"),"step:device:1");
      assert.ok(await tc.evaluate(n=>{const r=n.getBoundingClientRect();return r.left>=0&&r.top>=0&&r.right<=innerWidth+1&&r.bottom<=innerHeight+1;}));
      assert.ok(await touch.page.locator("#card-content").evaluate(n=>n.scrollWidth<=n.clientWidth+1));
      await audit(touch.page,"Touch driver card "+width);await shot(touch.page,"os_components_card_mobile_"+width+".png");
      await tc.locator('a[data-ref="session_native"]').tap();
      assert.equal(await touch.page.locator("#source-dialog").evaluate(n=>n.open),true);
      assert.ok(await touch.page.locator("#source-dialog").evaluate(n=>{const r=n.getBoundingClientRect();return r.left>=0&&r.top>=0&&r.right<=innerWidth+1&&r.bottom<=innerHeight+1;}));
      await audit(touch.page,"Touch source explorer "+width);await touch.page.locator("#source-close").tap();
      await touch.page.goto(url+"#state-machines");await touch.page.locator('#machine-server [data-scenario]').selectOption("worker_exit");
      await touch.page.locator('#machine-server [data-action="next"]').tap();
      assert.equal(await touch.page.locator("#machine-server").getAttribute("data-state"),"spawning");
      assert.ok(await touch.page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await audit(touch.page,"Touch state machine "+width);await touch.context.close();
    }
    checks.push("390/320 touch cards, source view, state controls and bounded document layout");

    const nojs=await setup({viewport:{width:320,height:844},javaScriptEnabled:false});
    await nojs.page.goto(url);assert.equal(await nojs.page.locator("#component-library").isVisible(),true);
    const explanation=nojs.page.locator('[data-reading="component:ipc"]');await explanation.locator(":scope>summary").click();
    assert.ok((await explanation.innerText()).includes("API PUSH"));
    assert.equal(await nojs.page.locator("#flow-startup .portable").count(),1);
    await nojs.page.locator("#flow-startup .portable>summary").click();
    assert.ok((await nojs.page.locator("#flow-startup .portable pre").innerText()).includes("await alive"));
    assert.ok(await nojs.page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await nojs.context.close();checks.push("no-JavaScript role explanations and portable sequence/state fallback");

    const motion=await setup({viewport:{width:1500,height:1000},reducedMotion:"no-preference"});
    await motion.page.goto(url+"#flow-startup");await motion.page.locator('#flow-startup [data-action="next"]').click();
    await motion.page.waitForFunction(()=>document.querySelector("#flow-startup .sequence-packet").classList.contains("visible"));
    const initial=await motion.page.locator("#flow-startup .sequence-packet").getAttribute("transform");
    await motion.page.waitForTimeout(180);const later=await motion.page.locator("#flow-startup .sequence-packet").getAttribute("transform");
    assert.notEqual(initial,later);await motion.page.locator('#flow-startup [data-action="play"]').click();
    await motion.page.waitForFunction(()=>document.querySelector("#flow-startup").dataset.frame==="2");
    await motion.page.locator('#flow-startup [data-action="play"]').click();
    await motion.page.waitForTimeout(1100);const paused=await motion.page.locator("#flow-startup").getAttribute("data-frame");
    await motion.page.waitForTimeout(180);assert.equal(await motion.page.locator("#flow-startup").getAttribute("data-frame"),paused);
    await motion.page.locator('#flow-startup [data-action="next"]').click();await audit(motion.page,"Animated event marker");
    await motion.context.close();checks.push("moving event marker, controlled playback/pause and reduced-motion state operation");

    const reader=await context.newPage();reader.on("pageerror",e=>errors.push(e.message));
    await reader.goto(pathToFileURL(path.join(task,"max_serve_llama2_7b_startup_call_stack.html")).href);
    await reader.locator('.stack-header-links a[href="link_htmls/max_serve_os_components_and_flow.html"]').click();
    assert.equal(reader.url(),url);await reader.close();checks.push("call-stack header opens the focused component reader");
    await context.close();assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const report={result:"PASS",checked_at:new Date().toISOString(),browser:await browser.version(),
      command:"NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_os_components_browser.cjs",
      checks,components,flow_steps:steps,state_paths:paths,state_cards:states,source_targets:refs,
      contrast:audits,screenshots,console_errors:errors,external_runtime_requests:network,
      evidence:"CPU-RUN browser/UI validation; animation is event order, no model/device measurement"};
    fs.writeFileSync(path.join(task,"validation/os_components_browser.json"),JSON.stringify(report,null,2)+"\n");
    console.log(JSON.stringify({result:"PASS",checks:checks.length,components,steps,paths,states,source_targets:refs,contrast_states:audits.length,screenshots,errors,network},null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
