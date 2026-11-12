'use strict';
// Exercises the generated offline readers, not MAX or the GPU implementation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');
const task = path.resolve(__dirname, '..');
const names = ['max_example_04_fused_attention_online_softmax.html', 'max_example_04_gpu_artifacts_abi_and_driver.html'];
const url = name => pathToFileURL(path.join(task, 'link_htmls', name)).href;

function renderedAudit(root) {
  const failures = [], collisions = [];
  let labels = 0;
  const svg = root.querySelector('svg'), g = svg.querySelector('[data-scene-actors]');
  const actors = [...g.children].filter(el => getComputedStyle(el).display !== 'none');
  for (const actor of actors) {
    const body = actor.querySelector('[data-part="body"]');
    const width = Number(body.getAttribute('width')), height = Number(body.getAttribute('height'));
    for (const text of actor.querySelectorAll('text')) {
      if (getComputedStyle(text).display === 'none') continue;
      labels++;
      const b = text.getBBox();
      if (b.x < -1 || b.y < -1 || b.x + b.width > width + 1 || b.y + b.height > height + 1) {
        failures.push({actor:actor.dataset.actorId,text:text.textContent,bounds:[b.x,b.y,b.width,b.height],size:[width,height]});
      }
    }
  }
  for (let i=0;i<actors.length;i++) for (let j=i+1;j<actors.length;j++) {
    const a=actors[i].querySelector('[data-part="body"]').getBoundingClientRect();
    const b=actors[j].querySelector('[data-part="body"]').getBoundingClientRect();
    if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>2 && Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>2) collisions.push([actors[i].dataset.actorId,actors[j].dataset.actorId]);
  }
  return {failures,collisions,labels,documentOverflow:document.documentElement.scrollWidth>innerWidth+1};
}

async function run() {
  const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  const errors=[],network=[],checks=[],screenshots=[],scenes=[];
  async function setup(options={}) {
    const c=await browser.newContext({viewport:{width:1440,height:1050},...options});
    await c.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});
    const p=await c.newPage();p.setDefaultTimeout(8000);p.on('pageerror',e=>errors.push(e.message));
    return {c,p};
  }
  async function seek(player,index) {
    await player.locator('[data-seek]').evaluate((el,n)=>{el.value=n;el.dispatchEvent(new Event('input',{bubbles:true}));},index);
  }
  async function shot(p,filename) {
    await p.screenshot({path:path.join(task,'validation',filename)});screenshots.push(filename);
  }
  async function auditFrames(player,label) {
    const length=await player.evaluate(el=>el.__player.frames.length), signatures=new Set();
    let testedLabels=0;
    for(let i=0;i<length;i++) {
      await seek(player,i);
      const audit=await player.evaluate(renderedAudit);
      assert.deepEqual(audit.failures,[],`${label} frame ${i}: SVG text bounds`);
      assert.deepEqual(audit.collisions,[],`${label} frame ${i}: actor overlap`);
      assert.equal(audit.documentOverflow,false,`${label}: document width`);
      testedLabels+=audit.labels;
      assert.equal(await player.locator('[data-event-title]').textContent(),await player.evaluate(el=>el.__player.frames[el.__player.index].title));
      assert.ok(await player.locator('[data-evidence] a').count(),`${label} frame ${i}: source evidence`);
      signatures.add(await player.locator('[data-scene-actors]').evaluate(el=>[...el.querySelectorAll('text')].filter(t=>getComputedStyle(t).display!=='none'&&getComputedStyle(t.closest('[data-actor-id]')).display!=='none').map(t=>t.textContent).join('\n')));
    }
    assert.ok(testedLabels>length*5,`${label}: rendered labels were checked`);
    assert.ok(signatures.size>=Math.floor(length*.6),`${label}: visible mechanisms change through the steps`);
    scenes.push({label,frames:length,renderedLabels:testedLabels,distinctScenes:signatures.size});
  }
  try {
    const {c,p}=await setup({reducedMotion:'reduce'});
    for(const name of names) {
      await p.goto(url(name));
      const stages=await p.evaluate(()=>AttentionReader.data.chapters.map(c=>c.id));
      assert.equal(await p.locator('main>.chapter:visible').count(),1);
      assert.equal(await p.locator('main>.chapter:visible [data-view-panel="tree"]').isVisible(),true);
      for(const stage of stages) {
        await p.locator(`[data-stage-link="${stage}"]`).click();
        const section=p.locator(`#stage-${stage}`);
        for(const view of ['tree','why','animation']) {
          await section.locator(`[data-view="${view}"]`).click();
          assert.equal(await section.locator('[data-view-panel]:visible').count(),1);
          assert.equal(await section.locator(`[data-view-panel="${view}"]`).isVisible(),true);
        }
        const player=section.locator('.player');
        await auditFrames(player,stage);
        if(['graph','compile','init','execute','host-link','launch'].includes(stage)) {
          const last=Number(await player.locator('[data-seek]').getAttribute('max'));
          await seek(player,stage==='compile'?12:stage==='execute'?12:stage==='host-link'?7:last);
          await shot(p,`attention_stage_${stage}.png`);
        }
        await player.locator('[data-action="back"]').click();
        const index=await player.evaluate(el=>el.__player.index);assert.ok(index>=0);
        await player.locator('[data-action="reset"]').click();
        assert.equal(await player.getAttribute('data-frame'),'0');
      }
    }
    checks.push('All 11 stages and three views work offline; every main/driver frame has readable, nonoverlapping actors and source evidence.');

    await p.goto(url(names[0]));
    await p.locator('[data-stage-link="graph"]').click();
    await p.locator('#stage-graph a[data-source="graph:910"]').click();
    let source=p.locator('.source-window:visible');
    assert.equal(await source.getAttribute('data-source-line'),'910');
    assert.match(await source.locator('.source-line.is-selected').textContent(),/self\._import_kernels/);
    const lineBounds=await source.locator('.source-line.is-selected').boundingBox();
    const scrollBounds=await source.locator('.source-scroll').boundingBox();
    assert.ok(lineBounds.y>=scrollBounds.y&&lineBounds.y<scrollBounds.y+scrollBounds.height,'source line is visible without manual scrolling');
    await source.locator('[data-window-hold]').click();
    await p.evaluate(()=>AttentionReader.openSource('load:166'));
    assert.equal(await p.locator('.source-window:visible').count(),2,'held source stays at its original location');
    const newer=p.locator('.source-window:visible').last();
    await newer.locator('input[aria-label="Source line"]').fill('181');
    await newer.locator('[data-go-line]').click();
    assert.equal(await newer.getAttribute('data-source-line'),'181');
    await newer.locator('[data-window-close]').click();
    await source.locator('[data-window-close]').click();
    checks.push('Caller links open full pinned source at the exact selected line; Hold preserves a source while another is inspected.');

    const cards=await p.evaluate(()=>AttentionReader.data.cards.map(c=>c.id));
    for(const id of cards) {
      await p.evaluate(id=>AttentionReader.openCard(id),id);
      const card=p.locator(`[data-window-key="card-${id}"]`);
      assert.equal(await card.isVisible(),true);
      for(const button of ['hold','expand','close']) assert.equal(await card.locator(`[data-window-${button}]`).isVisible(),true);
      await card.locator('[data-window-expand]').click();
      await card.locator('.card-motion > summary').click();
      const player=card.locator('.player');
      await player.locator('.canvas-scroll').scrollIntoViewIfNeeded();
      await auditFrames(player,`card:${id}`);
      if(id==='attention-math') {
        await seek(player,28);await shot(p,'attention_numeric_denominator.png');
        await card.locator('[data-query]').selectOption('1');
        assert.equal(await player.getAttribute('data-frame'),'0');
        await seek(player,37);
        const values=await player.evaluate(el=>el.__player.frames.at(-1).math.O);
        assert.ok(Math.abs(values[0]-.9406738431877646)<1e-12);
      }
      await card.locator('[data-window-close]').click();
    }
    checks.push('All nine focused cards render their call trees and complete mechanism timelines; changing query row resets the arithmetic.');

    await p.evaluate(()=>AttentionReader.openCard('module'));
    const card=p.locator('[data-window-key="card-module"]');
    await card.locator('[data-window-hold]').click();
    await p.locator('#stage-graph .chapter-header').click({position:{x:20,y:10}});
    assert.equal(await card.isVisible(),true);
    await card.locator('.reader-window-header').focus();
    const before=await card.boundingBox();await p.keyboard.press('ArrowLeft');
    assert.ok((await card.boundingBox()).x<before.x,'keyboard move');
    const resize=card.locator('.reader-window-resize');await resize.focus();
    const size=await card.boundingBox();await p.keyboard.press('ArrowLeft');
    assert.ok((await card.boundingBox()).width<size.width,'keyboard resize');
    const header=await card.locator('.reader-window-header').boundingBox();
    await p.mouse.move(header.x+90,header.y+18);await p.mouse.down();await p.mouse.move(header.x+120,header.y+38);await p.mouse.up();
    assert.equal(await card.getAttribute('data-card-dragging'),'false');
    await p.evaluate(()=>AttentionReader.openCard('context'));
    await card.locator('.reader-window-header').focus();
    const z=await card.evaluate(el=>Number(getComputedStyle(el).zIndex));
    assert.ok(z>Number(await p.locator('[data-window-key="card-context"]').evaluate(el=>getComputedStyle(el).zIndex)),'focusing a held window brings it forward');
    await card.locator('[data-window-close]').click();
    await p.locator('[data-window-key="card-context"] [data-window-close]').click();
    checks.push('Hold, Close, Expand/Restore, keyboard movement/resizing, pointer dragging and held-window stacking work.');

    await p.locator('#stage-graph [data-view="animation"]').click();
    await p.locator('#stage-graph [data-fullscreen]').click();
    await p.waitForFunction(()=>document.fullscreenElement?.id==='stage-graph');
    await p.evaluate(()=>AttentionReader.openCard('module'));
    assert.equal(await card.evaluate(el=>document.fullscreenElement.contains(el)),true);
    await card.locator('[data-window-hold]').click();
    await p.locator('#stage-graph [data-fullscreen]').click();
    await p.waitForFunction(()=>!document.fullscreenElement);
    assert.equal(await card.evaluate(el=>el.parentElement===document.body),true);
    await card.locator('[data-window-close]').click();
    checks.push('Chrome fullscreen keeps stage scrolling and card controls usable, including held-card reparenting on exit.');

    await p.locator('[data-stage-link="compile"]').click();
    await p.locator('#stage-compile [data-view="animation"]').click();
    const compiler=p.locator('#stage-compile .player');
    const cold=Number(await compiler.locator('[data-seek]').getAttribute('max'));
    await compiler.locator('[data-cache]').selectOption('hit');
    assert.ok(Number(await compiler.locator('[data-seek]').getAttribute('max'))<cold);
    await compiler.locator('[data-cache]').selectOption('miss');
    await compiler.locator('[data-vendor]').selectOption('amd');
    await seek(compiler,13);
    assert.match(await compiler.locator('[data-event-action]').textContent(),/AMDGPU/);
    assert.match(await compiler.locator('[data-evidence]').textContent(),/AMDGPUBackend/);
    checks.push('Backend/cache controls update displayed transformations and source evidence together.');

    // Real requestAnimationFrame playback: pause during a transition, keep
    // the same actor node, then resume from its frozen pose.
    const live=await setup({reducedMotion:'no-preference'}),lp=live.p;
    await lp.goto(url(names[0]));
    await lp.locator('#stage-build [data-view="animation"]').click();
    const motion=lp.locator('#stage-build .player');
    await motion.evaluate(el=>{el.__identity=el.querySelector('[data-actor-id="request"]');});
    await motion.locator('[data-action="play"]').click();
    await lp.waitForFunction(()=>{let p=document.querySelector('#stage-build .player').__player;return p.phase>.15&&p.phase<.7;});
    await motion.locator('[data-action="play"]').click();
    const paused=await motion.evaluate(el=>({i:el.__player.index,t:el.__player.phase,html:el.querySelector('svg').innerHTML}));
    await lp.waitForTimeout(180);
    assert.deepEqual(await motion.evaluate(el=>({i:el.__player.index,t:el.__player.phase,html:el.querySelector('svg').innerHTML})),paused);
    await motion.locator('[data-action="play"]').click();
    await lp.waitForFunction(i=>document.querySelector('#stage-build .player').__player.index>i,paused.i);
    assert.equal(await motion.evaluate(el=>el.__identity===el.querySelector('[data-actor-id="request"]')),true);
    await lp.locator('#stage-build [data-view="tree"]').click();
    assert.equal(await motion.evaluate(el=>el.__player.playing),false);
    await lp.locator('#stage-build [data-view="animation"]').click();
    await motion.locator('[data-action="play"]').click();
    await lp.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});
    assert.equal(await motion.evaluate(el=>el.__player.playing),false);
    await live.c.close();
    checks.push('Real playback freezes/resumes mid-transition without replacing actor nodes; view changes and hidden documents stop time.');

    for(const width of [390,320]) {
      await p.setViewportSize({width,height:850});
      for(const id of ['build','graph','compile','execute']) {
        await p.locator(`[data-stage-link="${id}"]`).click();
        const s=p.locator(`#stage-${id}`);await s.locator('[data-view="tree"]').click();
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`tree width ${width}/${id}`);
        await s.locator('[data-view="animation"]').click();
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`animation width ${width}/${id}`);
        const canvas=s.locator('.canvas-scroll');
        assert.equal(await canvas.evaluate(el=>el.scrollWidth>el.clientWidth),true,'wide scene scrolls locally');
      }
      await p.evaluate(()=>AttentionReader.openCard('attention-math'));
      const small=p.locator('[data-window-key="card-attention-math"]');
      const b=await small.boundingBox();assert.ok(b.x>=0&&b.x+b.width<=width);
      for(const action of ['hold','expand','close'])assert.equal(await small.locator(`[data-window-${action}]`).isVisible(),true);
      await small.locator('[data-window-close]').click();
      await shot(p,`attention_mobile_${width}.png`);
    }
    checks.push('320/390px layouts preserve page width; large diagrams scroll locally and window controls remain reachable.');
    await c.close();

    const plain=await setup({javaScriptEnabled:false}),np=plain.p;
    await np.goto(url(names[0]));
    assert.equal(await np.locator('main>.chapter:visible').count(),6);
    assert.equal(await np.locator('#stage-build .portable li').first().isVisible(),true);
    await np.locator('#note-attention-math > summary').click();
    assert.equal(await np.locator('#note-attention-math .call-tree').isVisible(),true);
    assert.equal(await np.locator('#note-attention-math .card-motion').isVisible(),false);
    assert.equal(await np.locator('[data-action="play"]:visible').count(),0);
    await plain.c.close();
    const print=await setup({reducedMotion:'reduce'}),pp=print.p;
    await pp.goto(url(names[0]));await pp.emulateMedia({media:'print'});
    await pp.evaluate(()=>dispatchEvent(new Event('beforeprint')));
    assert.equal(await pp.locator('main>.chapter:visible').count(),6);
    assert.equal(await pp.locator('#stage-compile .portable li').first().isVisible(),true);
    assert.equal(await pp.locator('#note-attention-math .call-tree').isVisible(),true);
    assert.equal(await pp.locator('[data-action="play"]:visible').count(),0);
    await pp.evaluate(()=>dispatchEvent(new Event('afterprint')));await print.c.close();
    checks.push('No-JavaScript and print views expose complete trees, design explanations and static notes without empty playback controls.');
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const report={result:'PASS',checks,scenes,errors,externalRequests:network.length,screenshots,evidence:'Offline Chromium UI and browser arithmetic only; no Bazel/MAX/Mojo/GPU execution'};
    fs.writeFileSync(path.join(task,'validation/attention_walkthrough_browser.json'),JSON.stringify(report,null,2)+'\n');
    return report;
  } finally {await browser.close();}
}
module.exports={run};
if(require.main===module)run().then(report=>console.log(JSON.stringify(report,null,2))).catch(e=>{console.error(e);process.exitCode=1;});
