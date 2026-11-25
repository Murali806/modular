/* Offline browser validation of the architecture additions, including real playback. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url'),{chromium}=require('playwright');
const toy=require('./architecture_toy.js'),cost=require('./architecture_model.js');
const out=path.resolve(__dirname,'../../link_htmls/llama_2_7b');
const url=slug=>pathToFileURL(path.join(out,slug+'.html')).href;
const errors=[],network=[],report={checks:[],numericalFrames:0,mathPasses:0,mobilePages:0,noJavaScriptPages:0};
let browser,page;
async function context(options={}){
  const ctx=await browser.newContext({viewport:{width:1440,height:1100},offline:true,...options});
  ctx.on('page',p=>{p.on('pageerror',e=>errors.push(p.url()+': '+e.message));p.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});});return ctx;
}
async function open(slug,p=page){await p.goto(url(slug));}
async function seek(i,p=page){await p.locator('#math-seek').evaluate((el,i)=>{el.value=i;el.dispatchEvent(new Event('input',{bubbles:true}));},i);}
async function fit(p=page){const dims=await p.evaluate(()=>[document.documentElement.scrollWidth,document.documentElement.clientWidth]);assert.ok(dims[0]<=dims[1]+1,p.url()+': '+dims);}
async function stage(label){await page.selectOption('#math-stage',{label});}
const close=(a,b)=>assert.ok(Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
function arraysClose(a,b){assert.equal(a.length,b.length);a.forEach((x,i)=>Array.isArray(x)?arraysClose(x,b[i]):close(x,b[i]));}

async function calculations(){
  await open('architecture');
  assert.equal(await page.locator('.architecture-ledger tbody tr:not(.layer-band)').count(),23);
  assert.ok((await page.locator('#kv-in-one-picture').textContent()).includes('256 MiB'));
  assert.ok((await page.locator('.architecture-ledger').textContent()).includes('[512, 4096]'));
  await page.locator('.architecture-ledger a[href$="#cost-q"]').click();
  assert.ok(page.url().endsWith('architecture_01_prefill.html#cost-q'));
  assert.equal(await page.locator('#cost-stage').inputValue(),'q');
  assert.ok((await page.locator('#cost-formula').textContent()).includes('8,589,934,592 MACs'));
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().totals.mac),3350272671744);
  await page.selectOption('#setting-traffic','materialized');
  assert.equal(await page.locator('#cost-attention').count(),0);
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().totals.traffic),cost.calculate({traffic:'materialized'}).totals.traffic);
  await page.selectOption('#setting-pairs','rectangle');
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().totals.mac),3384565301248);
  await page.locator('[data-cost-stage="qk"]').click();
  assert.ok((await page.locator('#cost-formula').textContent()).includes('1,073,741,824 MACs'));
  assert.ok((await page.locator('#cost-formula').textContent()).includes('Required average BW'));
  await page.locator('#setting-budgetMs').fill('5');
  close(await page.evaluate(()=>LlamaCosts.snapshot().totals.requiredGBs),cost.calculate({traffic:'materialized',pairs:'rectangle',budgetMs:5}).totals.requiredGBs);
  await page.selectOption('#setting-traffic','boundary');
  await page.selectOption('#cost-stage','attention');
  assert.ok((await page.locator('#cost-formula').textContent()).includes('MACs already counted in 08 and 10'));
  await page.locator('#setting-T').fill('4096');await page.locator('#setting-C').fill('1');
  assert.equal(await page.evaluate(()=>LlamaCosts.valid()),false);
  assert.equal(await page.locator('#cost-summary').isVisible(),false);
  assert.equal(await page.locator('#cost-export').isDisabled(),true);
  assert.ok((await page.locator('#cost-error').textContent()).includes('4096-token'));
  await page.locator('#cost-reset').click();
  assert.equal(await page.evaluate(()=>LlamaCosts.valid()),true);
  assert.equal(await page.locator('#setting-budgetMs').inputValue(),'10');
  await page.locator('[data-cost-preset="chunk"]').click();
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().settings.C),384);
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().memory.pages),4);
  await page.locator('[data-cost-preset="decode"]').click();
  await page.selectOption('#setting-w','0.5');await page.selectOption('#setting-k','1');
  assert.ok((await page.locator('#cost-assumptions').textContent()).includes('quantization metadata'));
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().memory.usedKV),128.25*2**20);
  await page.locator('#cost-reset').click();
  const dlPromise=page.waitForEvent('download');await page.locator('#cost-export').click();
  const download=await dlPromise,data=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
  assert.equal(data.totals.mac,3350272671744);assert.ok(data.scope.includes('not measured'));
  await page.locator('.architecture-nav a[href="architecture_02_decode.html"]').click();
  await page.waitForFunction(()=>window.LlamaCosts);
  assert.equal(await page.locator('#setting-T').inputValue(),'1');assert.equal(await page.locator('#setting-C').inputValue(),'512');
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().totals.mac),6741557248);
  assert.equal(await page.locator('.real-cache-page').count(),5);
  assert.ok((await page.locator('[data-logical-page="4"]').textContent()).includes('0 retained · 1 new · 127 unused'));
  await page.selectOption('#cost-stage','kv');
  assert.ok((await page.locator('#cost-formula').textContent()).includes('16,384 bytes/layer'));
  await page.locator('#setting-B').fill('3');
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().memory.pages),15);
  assert.equal(await page.evaluate(()=>LlamaCosts.snapshot().memory.weights),13476831232);
  await page.locator('#cost-reset').click();await page.locator('#cost-calculator').scrollIntoViewIfNeeded();
  await page.screenshot({path:'/tmp/llama_architecture_costs_desktop.png'});await fit();
  report.checks.push('Architecture stage links; prefill/decode/chunk presets; per-stage MAC/byte/BW/AI formulas; materialized vs fused accounting; invalid inputs; precision controls; page rounding; JSON export');
}

async function timelines(){
  await open('architecture_math');
  assert.equal(await page.evaluate(()=>LlamaMath.snapshot().playing),false);
  await page.evaluate(()=>{window.savedCacheSlot=document.querySelector('.toy-slot');window.savedBuffer=document.querySelector('[data-buffer="weights"]');});
  for(const name of ['prefill','decode','crosspage']){
    await page.selectOption('#math-pass',name);const data=toy.route(name),frames=toy.timeline(data);
    const retained=await page.evaluate(()=>LlamaMath.snapshot().cache);
    for(const [i,frame]of frames.entries()){
      await seek(i);const state=await page.evaluate(()=>LlamaMath.snapshot()),e=data.events[frame.event];
      assert.equal(state.index,i);assert.equal(state.stage,e.stage);assert.equal(state.term,frame.term);
      // Transcendental FP64 calculations may differ by an ULP across engines.
      // Within the browser, retained prefix entries must remain bit-for-bit equal.
      for(let layer=0;layer<2;layer++)for(const kv of ['K','V']){
        arraysClose(state.cache[layer][kv],e.cache[layer][kv]);
        assert.deepEqual(state.cache[layer][kv].slice(0,data.C),retained[layer][kv]);
      }
      assert.deepEqual(state.buffers,e.buffers);
      assert.equal(await page.locator('#math-progress').textContent(),`${i+1} / ${frames.length}`);
      const lengths=await page.locator('.toy-slot').evaluateAll(nodes=>[0,1].map(layer=>nodes.filter(n=>n.dataset.layer===String(layer)&&n.dataset.state!=='empty').length));
      assert.deepEqual(lengths,e.cache.map(c=>c.K.length));
      if(state.partial){
        const operand=e.headMatrices?.[state.head]||e;
        let expected=0;for(let k=0;k<frame.term;k++)expected+=operand.left.values[state.row][k]*operand.right.values[k][state.col];
        close(state.partial.value,expected*(e.scale||1));
        const shown=Number(await page.locator('#math-result td.target button').textContent());
        assert.ok(Math.abs(shown-state.partial.value)<.00051);
      }
      if(e.written){const cell=page.locator(`.toy-slot[data-layer="${e.layer}"][data-position="${e.written.position}"]`);assert.equal(await cell.getAttribute('data-state'),'write');}
      report.numericalFrames++;
    }
    assert.equal(await page.locator('#math-step').isDisabled(),true);
    const final=await page.evaluate(()=>LlamaMath.snapshot());
    await page.locator('#math-back').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().index),final.index-1);
    await page.locator('#math-step').click();assert.deepEqual(await page.evaluate(()=>LlamaMath.snapshot()),final);
    await page.locator('#math-reset').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().index),0);
    assert.equal(await page.evaluate(()=>savedCacheSlot===document.querySelector('.toy-slot')&&savedBuffer===document.querySelector('[data-buffer="weights"]')),true);
    report.mathPasses++;
  }
  await page.selectOption('#math-pass','prefill');
  await stage('Layer 0 · 03 · Q = N₁ × W_Qᵀ');
  await page.locator('#math-result td[data-row="2"][data-col="3"] button').click();
  for(let i=0;i<4;i++)await page.locator('#math-step').click();
  const partial=await page.evaluate(()=>LlamaMath.snapshot().partial);close(partial.value,partial.expected);
  await stage('Layer 0 · 09 · Causal mask, then stable softmax');
  await page.selectOption('#math-head','1');
  await page.locator('#math-result td[data-row="1"][data-col="2"] button').click();
  assert.equal(await page.locator('#math-result td.target button').textContent(),'0');
  assert.equal(await page.locator('#math-left td[data-row="1"][data-col="2"]').textContent(),'−∞');
  await page.selectOption('#math-row','2');await page.locator('#math-online summary').click();
  assert.ok((await page.locator('#online-progress').textContent()).includes('1 / 2'));
  await page.locator('#online-step').click();assert.equal(await page.locator('#online-step').isDisabled(),true);
  assert.ok((await page.locator('#online-result').textContent()).includes('Final online result matches'));
  await page.locator('#online-back').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().onlineIndex),0);
  await page.screenshot({path:'/tmp/llama_architecture_attention_desktop.png',fullPage:true});
  await page.selectOption('#math-pass','crosspage');await stage('Layer 0 · 07 · Store position 4 in layer 0');
  assert.equal(await page.locator('[data-physical-page="5"]').getAttribute('data-used'),'true');
  assert.equal(await page.locator('.toy-slot[data-layer="0"][data-position="4"]').getAttribute('data-state'),'write');
  assert.equal(await page.locator('.toy-slot[data-layer="1"][data-position="4"]').getAttribute('data-state'),'empty');
  await page.screenshot({path:'/tmp/llama_architecture_cache_desktop.png',fullPage:true});
  report.checks.push('Every numerical microstep across prefill and both decodes; reversible seek/back/reset; unchanged old KV; stable DOM buffers/pages; selected dot products; head selection; causal mask; page-boundary append; online vector softmax');
}

async function playbackAndWindows(){
  await open('architecture_math');await page.selectOption('#math-speed','500');
  await page.locator('#math-play').click();await page.waitForFunction(()=>LlamaMath.snapshot().index>=1);
  await page.locator('#math-play').click();const paused=await page.evaluate(()=>LlamaMath.snapshot().index);
  await page.waitForTimeout(850);assert.equal(await page.evaluate(()=>LlamaMath.snapshot().index),paused);
  await page.locator('#math-play').click();await page.waitForFunction(i=>LlamaMath.snapshot().index>i,paused);
  await page.locator('#math-play').click();const total=await page.evaluate(()=>LlamaMath.snapshot().total);
  await seek(total-2);await page.locator('#math-play').click();await page.waitForFunction(()=>!LlamaMath.snapshot().playing&&LlamaMath.snapshot().index===LlamaMath.snapshot().total-1);
  await page.locator('#math-play').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().index),0);
  await page.locator('#math-play').click();await page.locator('#math-play').click();
  await page.locator('#math-transcript').evaluate(n=>n.scrollIntoView({block:'start'}));await page.waitForFunction(()=>!LlamaMath.snapshot().playing);
  await page.locator('#math-badge').hover();await page.locator('#detail-card').waitFor({state:'visible'});
  const title=await page.locator('#card-title').textContent();await page.locator('#detail-card [data-window-hold]').click();
  await page.locator('#math-legend').click();
  assert.equal(await page.locator('[data-held-window] header strong').textContent(),title);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#detail-card').isVisible(),false);
  await page.locator('[data-held-window] [data-window-close]').click();
  await stage('Layer 0 · 08 · Q′ × cached K′ᵀ / √d');
  await page.locator('#math-fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement?.id==='math-animation');
  await page.locator('#math-sources [data-source]').click();
  await page.locator('#source-body .source-line.target').waitFor({state:'visible'});
  assert.equal(await page.locator('#source-card').evaluate(n=>n.parentElement.id),'math-animation');
  await page.locator('#source-card [data-window-expand]').click();assert.equal(await page.locator('#source-card').getAttribute('data-card-expanded'),'true');
  await page.locator('#source-card [data-window-close]').click();await page.locator('#math-fullscreen').click();
  await page.waitForFunction(()=>!document.fullscreenElement);await fit();
  report.checks.push('Timed Play/Pause/resume/end/replay and offscreen pause; badge, legend, Hold/Close/Escape; fullscreen source windows with exact target line and Expand');
}

async function layouts(){
  const mobile=await context({viewport:{width:390,height:844}}),p=await mobile.newPage();
  for(const slug of ['architecture','architecture_01_prefill','architecture_02_decode','architecture_math']){
    await open(slug,p);await fit(p);report.mobilePages++;
    if(slug==='architecture_02_decode'){
      await p.locator('#cost-calculator').scrollIntoViewIfNeeded();await p.screenshot({path:'/tmp/llama_architecture_costs_mobile.png',fullPage:true});
      await p.selectOption('#cost-stage','attention');assert.ok((await p.locator('#cost-formula').textContent()).includes('Required average BW'));
    }
    if(slug==='architecture_math'){
      await p.selectOption('#math-stage',{label:'Layer 0 · 06 · Reshape heads, rotate Q and K'});await fit(p);
      await p.screenshot({path:'/tmp/llama_architecture_math_mobile.png',fullPage:true});
      await p.locator('#math-legend').click();const box=await p.locator('#detail-card').boundingBox();assert.ok(box.x>=0&&box.x+box.width<=391);
      await p.locator('#detail-card [data-window-close]').click();
      await p.locator('.math-cache').evaluate(n=>n.scrollIntoView({block:'center'}));
      const transport=await p.locator('.math-transport').boundingBox();assert.ok(transport.y>=-1&&transport.y<20,'playback controls remain available beside the lower cache view');
    }
  }
  await mobile.close();
  const reduced=await context({reducedMotion:'reduce'}),r=await reduced.newPage();await open('architecture_math',r);
  await r.selectOption('#math-pass','decode');await r.locator('#math-step').click();assert.equal(await r.evaluate(()=>LlamaMath.snapshot().index),1);await reduced.close();
  const nojs=await context({javaScriptEnabled:false}),n=await nojs.newPage();
  for(const slug of ['architecture','architecture_01_prefill','architecture_02_decode','architecture_math']){
    await open(slug,n);await fit(n);report.noJavaScriptPages++;
    if(slug==='architecture_math')assert.ok((await n.locator('#math-transcript').textContent()).includes('sampling itself writes no K/V'));
    else assert.equal(await n.locator('.architecture-ledger tbody tr:not(.layer-band)').count(),23);
    if(slug.includes('_0'))assert.ok((await n.locator('#counting-conventions').textContent()).includes('2×MAC'));
  }
  await nojs.close();await open('architecture_math');await stage('Layer 0 · 09 · Causal mask, then stable softmax');
  await page.pdf({path:'/tmp/llama_architecture_print.pdf',format:'A4',printBackground:true});
  assert.equal(await page.locator('#math-online').getAttribute('open'),null,'printing restores disclosures');
  report.checks.push('390px layouts, sticky cache controls and mobile card bounds; reduced motion; all four pages readable without JavaScript; real print and disclosure restoration');
}
(async()=>{
  browser=await chromium.launch({headless:true,args:['--no-sandbox']});page=await (await context()).newPage();
  await calculations();console.log('Calculators and navigation passed');
  await timelines();console.log(`${report.numericalFrames} numerical frames passed`);
  await playbackAndWindows();await layouts();
  assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
  Object.assign(report,{chromium:browser.version(),node:process.version,playwright:require('playwright/package.json').version,offline:true,pageErrors:errors,networkRequests:network,result:'PASS'});
  fs.writeFileSync('/tmp/llama_architecture_browser_report.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(async e=>{console.error(e);console.error({errors,network});if(browser)await browser.close();process.exit(1);});
