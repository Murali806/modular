/* Offline browser validation of the architecture additions, including real playback. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url'),{chromium}=require('playwright');
const tensor=require('./architecture_tensor.js'),cost=require('./architecture_model.js');
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
async function stage(id){await page.selectOption('#math-stage',id);}
async function number(id,v,p=page){await p.locator('#'+id).fill(String(v));await p.locator('#'+id).press('Tab');}
const close=(a,b)=>assert.ok(Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
function arraysClose(a,b){assert.equal(a.length,b.length);a.forEach((x,i)=>Array.isArray(x)?arraysClose(x,b[i]):close(x,b[i]));}

async function calculations(){
  await open('architecture');
  assert.equal(await page.locator('.architecture-ledger tbody tr:not(.layer-band)').count(),23);
  assert.ok((await page.locator('#kv-in-one-picture').textContent()).includes('256 MiB'));
  assert.ok((await page.locator('.architecture-ledger').textContent()).includes('[512, 4096]'));
  await page.locator('.architecture-ledger a[href$="#cost-q"]').click();
  await page.waitForURL('**/architecture_01_prefill.html#cost-q');
  await page.waitForFunction(()=>window.LlamaCosts);
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
  await page.evaluate(()=>{window.savedCacheSlot=document.querySelector('.cache-slot');window.savedBuffer=document.querySelector('[data-buffer="weights"]');window.savedMatrix=document.querySelector('#math-left table');});
  for(const name of ['prefill','decode','next']){
    await page.selectOption('#math-pass',name);
    for(const id of Object.keys(tensor.definitions)){
      await stage(id);
      if(tensor.block.includes(id))await page.selectOption('#math-layer','31');
      if(['rope','kv','qk','softmax','pv','merge'].includes(id))await page.selectOption('#math-head','31');
      await page.locator('#math-last-row').click();await page.locator('#math-last-col').click();
      let state=await page.evaluate(()=>LlamaMath.snapshot()),event=tensor.events[state.eventIndex],op=tensor.operation(name,event,state.head);
      const ref=tensor.probe(op,state.row,state.col);
      assert.deepEqual(state.shapes,{left:op.left.shape,right:op.right.shape,result:op.result.shape});
      const href=await page.locator('#math-cost-link a').getAttribute('href'),[file,hash]=href.split('#');
      assert.ok(fs.readFileSync(path.join(out,file),'utf8').includes('id="'+hash+'"'),href);
      for(const at of new Set([0,Math.floor(op.steps/2),op.steps-1,op.steps])){
        await seek(at);state=await page.evaluate(()=>LlamaMath.snapshot());
        assert.equal(state.step,at);assert.equal(state.inner,op.inner);
        if(op.inner&&Number.isFinite(ref.prefix[at]))close(state.partial,ref.prefix[at]);
        close(state.expected,ref.expected);
        assert.deepEqual(state.cache,tensor.cacheState(name,event,at));
        const valid=state.cache.lengths[state.layer];
        assert.equal(await page.locator('.cache-slot:not([data-state="empty"])').count(),valid);
        assert.equal(await page.locator('.layer-chip').count(),32);
        assert.equal(await page.locator('.layer-chip.active small').textContent(),String(valid));
        if(at===op.steps){
          const shown=Number(await page.locator('#math-result td.target button').textContent());
          assert.ok(Math.abs(shown-state.expected)<.00051);
        }
        // Rendering stays bounded even for [4096,32000] weights.
        assert.ok(await page.locator('.math-matrices td').count()<=45);
        assert.ok(await page.locator('body *').count()<3100);
        report.numericalFrames++;
      }
      await fit();
    }
    await page.locator('#math-reset').click();
    assert.equal(await page.evaluate(()=>LlamaMath.snapshot().eventIndex),0);
    assert.equal(await page.evaluate(()=>savedCacheSlot===document.querySelector('.cache-slot')&&savedBuffer===document.querySelector('[data-buffer="weights"]')&&savedMatrix===document.querySelector('#math-left table')),true);
    report.mathPasses++;
  }
  // Complete 11008-term reduction, exact coordinate navigation and reversible progress.
  await page.selectOption('#math-pass','prefill');await stage('down');
  await number('math-row',511);await number('math-col',4095);await seek(11007);
  assert.equal(await page.locator('#math-result td.target button').textContent(),'…');
  await page.selectOption('#math-chunk','1');await page.locator('#math-step').click();
  let final=await page.evaluate(()=>LlamaMath.snapshot());assert.equal(final.step,11008);close(final.partial,final.expected);
  await page.locator('#math-back').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().step),11007);
  await page.locator('#math-step').click();assert.deepEqual(await page.evaluate(()=>LlamaMath.snapshot()),final);
  await page.locator('#math-result .tensor-overview').click({position:{x:40,y:15}});
  let moved=await page.evaluate(()=>LlamaMath.snapshot());assert.ok(moved.row<511&&moved.col<4095);
  await number('math-row',-9);assert.equal(await page.locator('#math-row').inputValue(),'0');
  await number('math-col',999999);assert.equal(await page.locator('#math-col').inputValue(),'4095');
  await page.locator('#math-result td:not(.target) button').first().click();
  const clicked=await page.evaluate(()=>LlamaMath.snapshot());assert.notEqual(clicked.col,4095);
  await number('math-row',511);await number('math-col',4095);await seek(5632);
  await page.locator('#math-matrices').scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/llama_tensor_down_desktop.png'});

  // All 32 layers really select distinct cache slices, with last layer completion.
  await page.selectOption('#math-pass','decode');await stage('kv');
  const old=await page.locator('.cache-slot[data-position="511"]').getAttribute('data-state');assert.equal(old,'retained');
  for(let l=0;l<32;l++){
    await page.selectOption('#math-layer',String(l));
    assert.equal(await page.locator('.cache-slot[data-position="512"]').getAttribute('data-state'),'empty');
    await page.locator('#math-finish').click();
    assert.equal(await page.locator('.cache-slot[data-position="512"]').getAttribute('data-state'),'write');
    const state=await page.evaluate(()=>LlamaMath.snapshot());
    assert.equal(state.cache.lengths[l],513);assert.equal(state.cache.completed,l===31?513:512);
  }
  assert.equal(await page.evaluate(()=>LlamaMath.snapshot().cache.used),256.5*2**20);
  assert.equal(await page.evaluate(()=>LlamaMath.snapshot().cache.assigned),320*2**20);
  await number('math-cache-channel',127);await number('math-cache-position',512);
  const retained=await page.locator('#math-cache-vector').textContent();
  await page.selectOption('#math-pass','next');await stage('kv');await page.selectOption('#math-layer','31');await number('math-cache-channel',127);await number('math-cache-position',512);
  assert.equal(await page.locator('#math-cache-vector').textContent(),retained);
  await page.locator('#math-finish').click();assert.equal(await page.locator('.cache-slot[data-position="513"]').getAttribute('data-state'),'write');
  await page.locator('.cache-slot[data-position="639"]').click();
  assert.ok((await page.locator('#math-cache-vector').textContent()).includes('unwritten'));
  await number('math-cache-position',513);
  await page.locator('.math-cache').scrollIntoViewIfNeeded();await page.waitForTimeout(220);await page.screenshot({path:'/tmp/llama_tensor_cache_desktop.png'});

  // Full causal row and connected online attention, including masked tiles.
  await page.selectOption('#math-pass','prefill');await stage('softmax');await page.selectOption('#math-head','31');
  await number('math-row',0);await number('math-col',511);await page.locator('#math-finish').click();
  assert.equal(await page.locator('#math-result td.target button').textContent(),'0');
  assert.equal(await page.locator('#math-left td[data-row="0"][data-col="511"]').textContent(),'−∞');
  await number('math-row',511);assert.notEqual(await page.locator('#math-result td.target button').textContent(),'0');
  await page.locator('#math-online summary').click();await number('online-query',511);
  for(let i=0;i<3;i++)await page.locator('#online-step').click();
  assert.equal(await page.locator('#online-step').isDisabled(),true);
  assert.ok((await page.locator('#online-result').textContent()).includes('All tiles included'));
  await page.locator('#online-back').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().onlineIndex),2);
  await page.locator('#math-online').scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/llama_tensor_online_desktop.png'});
  await number('online-query',0);await page.locator('#online-step').click();
  assert.ok((await page.locator('#online-result').textContent()).includes('α = exp(old m − m′) = 1'));
  await stage('rope');await page.selectOption('#math-head','31');await page.selectOption('#math-rope-kind','K');
  await number('math-row',511);await number('math-col',127);await seek(3);
  assert.ok((await page.locator('#math-equation').textContent()).includes('pair (63, 127)'));
  await page.locator('#math-equation').scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/llama_tensor_rope_desktop.png'});
  report.checks.push('All 23 operations across three passes; actual shapes and full reductions; bounded DOM; every layer slice; complete down projection; viewports and last coordinates; stable DOM; retained KV values and 4→5 page boundary; causal mask; online 128-channel vectors; exact cost links');
}

async function playbackAndWindows(){
  await open('architecture_math');await stage('q');await page.selectOption('#math-speed','400');
  await page.locator('#math-play').click();await page.waitForFunction(()=>LlamaMath.snapshot().step>=128);
  await page.locator('#math-play').click();const paused=await page.evaluate(()=>LlamaMath.snapshot().step);
  await page.waitForTimeout(650);assert.equal(await page.evaluate(()=>LlamaMath.snapshot().step),paused);
  await page.locator('#math-play').click();await page.waitForFunction(i=>LlamaMath.snapshot().step>i,paused);
  await page.locator('#math-play').click();await stage('sample');await seek(31999);
  await page.locator('#math-play').click();await page.waitForFunction(()=>!LlamaMath.snapshot().playing&&LlamaMath.snapshot().step===32000);
  await page.locator('#math-play').click();assert.equal(await page.evaluate(()=>LlamaMath.snapshot().eventIndex),0);
  await page.locator('#math-play').click();await page.locator('#math-play').click();
  await page.locator('#math-transcript').evaluate(n=>n.scrollIntoView({block:'start'}));await page.waitForFunction(()=>!LlamaMath.snapshot().playing);
  await page.locator('#math-badge').hover();await page.locator('#detail-card').waitFor({state:'visible'});
  const title=await page.locator('#card-title').textContent();await page.locator('#detail-card [data-window-hold]').click();
  await page.locator('#math-legend').click();assert.equal(await page.locator('[data-held-window] header strong').textContent(),title);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#detail-card').isVisible(),false);
  await page.locator('[data-held-window] [data-window-close]').click();
  await stage('qk');await page.locator('#math-fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement?.id==='math-animation');
  await page.locator('#math-sources [data-source]').click();await page.locator('#source-body .source-line.target').waitFor({state:'visible'});
  assert.equal(await page.locator('#source-card').evaluate(n=>n.parentElement.id),'math-animation');
  await page.locator('#source-card [data-window-expand]').click();assert.equal(await page.locator('#source-card').getAttribute('data-card-expanded'),'true');
  await page.locator('#source-card [data-window-close]').click();await page.locator('#math-fullscreen').click();
  await page.waitForFunction(()=>!document.fullscreenElement);await fit();
  // Crossing a layer boundary updates cache and layer identity.
  await stage('add2');await page.selectOption('#math-layer','0');await page.locator('#math-next').click();
  assert.equal(await page.evaluate(()=>LlamaMath.snapshot().layer),1);assert.equal(await page.evaluate(()=>LlamaMath.snapshot().stage),'norm1');
  report.checks.push('Actual Play/Pause/resume/end/replay; offscreen pause; badge/legend/Hold/Escape; fullscreen exact source and Expand; next-layer transition');
}

async function layouts(){
  const mobile=await context({viewport:{width:390,height:844}}),p=await mobile.newPage();
  for(const slug of ['architecture','architecture_01_prefill','architecture_02_decode','architecture_math']){
    await open(slug,p);await fit(p);report.mobilePages++;
    if(slug==='architecture_02_decode'){
      await p.locator('#cost-calculator').scrollIntoViewIfNeeded();await p.screenshot({path:'/tmp/llama_architecture_costs_mobile.png',fullPage:true});
    }
    if(slug==='architecture_math'){
      await p.selectOption('#math-stage','pv');await p.locator('#math-last-row').click();await p.locator('#math-last-col').click();
      await seek(256,p);await fit(p);
      await p.locator('#math-matrices').scrollIntoViewIfNeeded();await p.screenshot({path:'/tmp/llama_tensor_math_mobile.png',fullPage:true});
      await p.locator('#math-legend').click();const box=await p.locator('#detail-card').boundingBox();assert.ok(box.x>=0&&box.x+box.width<=391);
      await p.locator('#detail-card [data-window-close]').click();
      await p.selectOption('#math-pass','decode');await p.selectOption('#math-stage','kv');await p.selectOption('#math-layer','31');await p.locator('#math-finish').click();
      await p.locator('.math-cache').scrollIntoViewIfNeeded();await p.waitForTimeout(220);await p.screenshot({path:'/tmp/llama_tensor_cache_mobile.png'});await fit(p);
    }
  }
  await mobile.close();
  const reduced=await context({reducedMotion:'reduce'}),r=await reduced.newPage();await open('architecture_math',r);
  await r.selectOption('#math-stage','q');await r.locator('#math-step').click();assert.equal(await r.evaluate(()=>LlamaMath.snapshot().step),128);await reduced.close();
  const nojs=await context({javaScriptEnabled:false}),n=await nojs.newPage();
  for(const slug of ['architecture','architecture_01_prefill','architecture_02_decode','architecture_math']){
    await open(slug,n);await fit(n);report.noJavaScriptPages++;
    if(slug==='architecture_math'){
      assert.ok((await n.locator('#math-transcript').textContent()).includes('sampling itself writes no K/V'));
      assert.ok((await n.locator('#math-transcript').textContent()).includes('11008'));
    }else assert.equal(await n.locator('.architecture-ledger tbody tr:not(.layer-band)').count(),23);
  }
  await nojs.close();await open('architecture_math');await stage('softmax');
  await page.pdf({path:'/tmp/llama_tensor_print.pdf',format:'A4',printBackground:true});
  assert.equal(await page.locator('#math-online').getAttribute('open'),null,'printing restores disclosures');
  report.checks.push('390px full-size viewports, page grid and card bounds; reduced motion; four no-JavaScript pages; real print and restored disclosures');
}
(async()=>{
  browser=await chromium.launch({headless:true,args:['--no-sandbox']});page=await (await context()).newPage();
  await calculations();console.log('Calculators and navigation passed');
  await timelines();console.log(report.numericalFrames+' actual-dimension frames passed');
  await playbackAndWindows();await layouts();
  assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
  Object.assign(report,{chromium:browser.version(),node:process.version,playwright:require('playwright/package.json').version,offline:true,pageErrors:errors,networkRequests:network,result:'PASS'});
  fs.writeFileSync('/tmp/llama_architecture_browser_report.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(async e=>{console.error(e);console.error({errors,network});if(browser)await browser.close();process.exit(1);});
