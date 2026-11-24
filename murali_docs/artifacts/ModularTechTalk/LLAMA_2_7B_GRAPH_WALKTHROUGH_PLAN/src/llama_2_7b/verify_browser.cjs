/* Real local-file browser checks. Install Playwright outside the repo if needed. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url'),{createHash}=require('node:crypto');
const {chromium}=require('playwright');
const out=path.resolve(__dirname,'../../link_htmls/llama_2_7b');
const url=(slug,suffix='')=>pathToFileURL(path.join(out,slug+'.html')).href+suffix;
const chapters=JSON.parse(fs.readFileSync(path.join(out,'chapters.json'),'utf8'));
const readData=slug=>JSON.parse(fs.readFileSync(path.join(out,slug+'.html'),'utf8').match(/<script type="application\/json" id="lesson-data">(.*?)<\/script>/s)[1]);
const copy=x=>JSON.parse(JSON.stringify(x));
const errors=[],network=[],report={checks:[],desktopStates:0,mobilePages:0,sourceFiles:0};
let browser,page;

async function context(options={}){
  const ctx=await browser.newContext({viewport:{width:1440,height:1100},offline:true,...options});
  ctx.on('page',p=>{
    p.on('pageerror',e=>errors.push(p.url()+': '+e.message));
    p.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
  });
  return ctx;
}
async function open(slug,p=page,suffix=''){
  await p.goto(url(slug,suffix));
  if(!['sources','coverage'].includes(slug))await p.waitForFunction(()=>window.LlamaReader);
}
async function seek(index,p=page){
  await p.locator('#seek').evaluate((el,i)=>{el.value=i;el.dispatchEvent(new Event('input',{bubbles:true}));},index);
}
async function domState(p=page){
  return p.locator('#scene').evaluate(scene=>Object.fromEntries([...scene.querySelectorAll('.actor')].map(a=>[a.dataset.actor,{
    state:a.querySelector('.actor-state').textContent,
    slots:[...a.querySelectorAll('.value-chip')].map(n=>({id:n.dataset.item,label:n.textContent})),
    facts:[...a.querySelectorAll('.actor-facts li')].map(n=>n.textContent)
  }])));
}
async function expectState(expected,index,total,p=page){
  assert.deepEqual(await domState(p),expected);
  assert.equal(await p.locator('#progress').textContent(),`${index+1} / ${total}`);
  assert.equal(await p.locator('#step').isDisabled(),index===total-1);
  assert.equal(await p.locator('#back').isDisabled(),index===0);
}
async function fit(p=page){
  const dimensions=await p.evaluate(()=>({scroll:document.documentElement.scrollWidth,viewport:document.documentElement.clientWidth}));
  assert.ok(dimensions.scroll<=dimensions.viewport+1,JSON.stringify({url:p.url(),...dimensions}));
}
async function sourceReady(p=page,selector='#source-body'){
  await p.locator(selector+' .source-line.target').waitFor({state:'visible'});
}

async function lessonStates(){
  for(const [n,{slug}]of chapters.entries()){
    await open(slug);
    assert.equal(await page.evaluate(()=>LlamaReader.getState().playing),false,'no autoplay');
    const data=readData(slug);
    for(const route of [{id:'main',steps:data.steps},...data.scenarios]){
      await page.selectOption('#scenario',route.id);
      const expected=Object.fromEntries(data.actors.map(a=>[a.id,{state:'not created / no input yet',slots:[],facts:[]} ])),history=[];
      for(const [i,event]of route.steps.entries()){
        Object.assign(expected,copy(event.changes));history.push(copy(expected));
        if(i)await page.locator('#step').click();
        await expectState(expected,i,route.steps.length);
        report.desktopStates++;
      }
      await page.locator('#back').click();
      await expectState(history.at(-2),history.length-2,history.length);
      const middle=Math.floor(history.length/2);
      await seek(middle);const delta=await page.locator('#delta').textContent();
      await page.locator('#reset').click();
      await expectState(history[0],0,history.length);
      await seek(middle);await expectState(history[middle],middle,history.length);
      assert.equal(await page.locator('#delta').textContent(),delta,'seek/reset preserves the event predecessor');
    }
    await fit();
    const link=page.locator('.call-tree .tree-symbol').first(),alias=await link.getAttribute('data-source'),line=await link.getAttribute('data-line');
    await link.click();await sourceReady();
    assert.equal(await page.locator('#source-body').getAttribute('data-file'),alias);
    assert.equal(await page.locator('#source-body .target').getAttribute('data-line'),line);
    await page.locator('#source-card [data-window-close]').click();
    if((n+1)%5===0)console.log(`Checked ${n+1}/${chapters.length} lessons and their routes`);
  }
  report.checks.push('All lesson/scenario states: Step, Back, Reset, seek, direct call-tree source links');
}

async function playback(){
  await open('asyncrt_workers',page,'#animation');
  await page.evaluate(()=>{window.savedChip=document.querySelector('[data-item="task"]');});
  await page.locator('#step').click();
  assert.equal(await page.evaluate(()=>savedChip===document.querySelector('.actor [data-item="task"]')),true);
  assert.equal(await page.locator('.actor [data-item="task"]').evaluate(n=>n.closest('.actor').dataset.actor),'QUEUE');
  await page.locator('#reset').click();
  await page.locator('#play').click();
  await page.waitForFunction(()=>LlamaReader.getState().index>=1);
  await page.locator('#play').click();
  const paused=await page.evaluate(()=>LlamaReader.getState().index);
  await page.waitForTimeout(2500);
  assert.equal(await page.evaluate(()=>LlamaReader.getState().index),paused,'pause must stop the timer');
  await page.locator('#play').click();
  await page.waitForFunction(i=>LlamaReader.getState().index>i,paused);
  await page.locator('#play').click();
  const last=readData('asyncrt_workers').steps.length-1;
  await seek(last-1);await page.locator('#play').click();
  await page.waitForFunction(i=>LlamaReader.getState().index===i&&!LlamaReader.getState().playing,last);
  await page.locator('#play').click();
  assert.equal(await page.evaluate(()=>LlamaReader.getState().index),0,'replay renders the first event immediately');
  await page.locator('#play').click();
  await page.locator('#play').click();
  await page.evaluate(()=>window.scrollTo(0,document.body.scrollHeight));
  await page.waitForFunction(()=>!LlamaReader.getState().playing);
  report.checks.push('Real timed Play/Pause/resume, automatic end stop, replay, offscreen pause, stable moving DOM identity');
}

async function windowsAndSources(){
  await open('session',page,'#animation');
  await page.locator('#lesson-badge').hover();
  await page.locator('#detail-card').waitFor({state:'visible'});
  const title=await page.locator('#card-title').textContent();
  await page.locator('#detail-card [data-window-hold]').click();
  await page.locator('#legend').click();
  const held=page.locator('.floating-card[data-held-window]');
  assert.equal(await held.count(),1);
  assert.equal(await held.locator('header strong').textContent(),title,'Hold preserves the previous explanation');
  assert.equal(await page.locator('#card-title').textContent(),'Objects, owners and synchronization');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#detail-card').isVisible(),false);
  assert.equal(await held.isVisible(),true,'Escape must preserve a held window');
  await held.locator('[data-window-close]').click();

  const links=page.locator('.call-tree .tree-symbol');
  await links.first().click();await sourceReady();
  const oldFile=await page.locator('#source-body').getAttribute('data-file');
  await page.locator('#source-card [data-window-hold]').click();
  const other=page.locator(`.call-tree .tree-symbol:not([data-source="${oldFile}"])`).first();
  await other.focus();await page.keyboard.press('Enter');await sourceReady();
  const copyWindow=page.locator('.source-card[data-held-window]');
  await copyWindow.locator('.source-line.target').waitFor();
  assert.equal(await copyWindow.locator('[data-source-pane]').getAttribute('data-file'),oldFile);
  assert.notEqual(await page.locator('#source-body').getAttribute('data-file'),oldFile);
  await page.locator('#source-card [data-window-close]').click();
  await copyWindow.locator('header').focus();
  const before=await copyWindow.boundingBox();
  await page.keyboard.press('ArrowRight');
  const moved=await copyWindow.boundingBox();assert.ok(moved.x>before.x);
  const header=await copyWindow.locator('header').boundingBox();
  await page.mouse.move(header.x+20,header.y+20);await page.mouse.down();
  await page.mouse.move(header.x+60,header.y+60,{steps:5});await page.mouse.up();
  const dragged=await copyWindow.boundingBox();assert.ok(dragged.y>moved.y);
  await copyWindow.locator('.reader-window-resize').focus();
  await page.keyboard.press('ArrowLeft');
  assert.ok((await copyWindow.boundingBox()).width<dragged.width);
  const size=await copyWindow.boundingBox();
  const handle=await copyWindow.locator('.reader-window-resize').boundingBox();
  await page.mouse.move(handle.x+10,handle.y+10);await page.mouse.down();
  await page.mouse.move(handle.x-40,handle.y+40,{steps:5});await page.mouse.up();
  assert.ok((await copyWindow.boundingBox()).width<size.width);
  await copyWindow.locator('[data-window-expand]').click();
  assert.equal(await copyWindow.getAttribute('data-card-expanded'),'true');
  assert.ok((await copyWindow.boundingBox()).width>=1400);
  await copyWindow.locator('[data-window-expand]').click();
  await copyWindow.locator('[data-source-all]').click();
  await page.waitForFunction(()=>document.querySelector('[data-held-window] [data-source-pane]').dataset.sourceComplete==='true');
  const lines=await copyWindow.locator('.source-line').count();
  assert.equal(lines,await page.evaluate(a=>LLAMA_SOURCES[a].lines,oldFile),'held source retains complete-file controls');
  await copyWindow.locator('.source-line.target').click();
  assert.equal(await copyWindow.locator('.source-line').count(),lines,'selecting source text must not toggle the complete-file view');
  await copyWindow.locator('[data-source-next]').click();
  assert.ok(await copyWindow.locator('.source-line').count()<=100);
  await copyWindow.locator('[data-window-close]').click();
  assert.equal(await page.locator('[data-held-window]').count(),0);

  const loaded=await page.evaluate(async()=>Promise.all(Object.keys(LLAMA_SOURCES).map(async alias=>({alias,text:await LlamaReader.loadSource(alias),expected:LLAMA_SOURCES[alias].sha256}))));
  for(const {alias,text,expected}of loaded)assert.equal(createHash('sha256').update(text).digest('hex'),expected,alias);
  report.sourceFiles=loaded.length;
  await open('sources',page,'?file=kv&line=525');
  await sourceReady(page,'#source-view');
  await page.locator('#source-search').fill('PACKED_PAGE_STRIDE = -1');
  await page.locator('#source-search').press('Enter');
  await page.waitForFunction(()=>document.querySelector('#source-view .target')?.textContent.includes('PACKED_PAGE_STRIDE = -1'));
  report.checks.push('Hold preserves content; held-source paging/all-file controls; Close/Escape; pointer/keyboard drag and resize; Expand/Restore; all source hashes loaded offline; search');
}

async function layoutsAndLabs(){
  await open('mojo_codegen',page,'#animation');await seek(5);
  const previous=await page.locator('#delta').textContent();
  await page.locator('#fullscreen').click();
  await page.waitForFunction(()=>document.fullscreenElement?.id==='animation');
  assert.equal(await page.locator('#delta').textContent(),previous);
  await page.locator('#step-sources [data-source]').first().click();await sourceReady();
  assert.equal(await page.locator('#source-card').evaluate(n=>n.parentElement.id),'animation');
  await page.locator('#source-card [data-window-close]').click();
  await page.locator('#fullscreen').click();
  await page.waitForFunction(()=>!document.fullscreenElement);

  await open('kv_cache');
  assert.ok((await page.locator('#lab-result').textContent()).includes('2.0000 GiB'));
  await page.locator('#kv-tokens').fill('130');await page.locator('#kv-batch').fill('2');
  assert.ok((await page.locator('#lab-result').textContent()).includes('0.2500 GiB'));
  const initial=Number(await page.locator('.address-lab').getAttribute('data-byte-offset'));
  await page.locator('#addr-padding').check();
  assert.equal(Number(await page.locator('.address-lab').getAttribute('data-byte-offset'))-initial,256);
  await page.locator('[data-page-token="127"]').click();
  assert.ok((await page.locator('#address-result').textContent()).includes('physical page 7'));
  await page.locator('[data-page-token="128"]').click();
  assert.ok((await page.locator('#address-result').textContent()).includes('physical page 2'));
  await page.locator('.address-lab').screenshot({path:'/tmp/llama_kv_address.png'});
  await open('attention_rope');
  assert.ok((await page.locator('#lab-result').textContent()).includes('24.205125'));
  await page.selectOption('#sm-tile','1');
  assert.ok((await page.locator('#lab-result').textContent()).includes('24.205125'));
  await page.locator('#sm-scores').fill('');
  assert.ok((await page.locator('#lab-result').textContent()).includes('finite numbers'));
  await page.locator('#sm-scores').fill('2,1,3');

  const mobileContext=await context({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  const mobile=await mobileContext.newPage();
  for(const {slug}of chapters){await open(slug,mobile);await fit(mobile);await mobile.locator('#step').click();report.mobilePages++;}
  await open('kv_cache',mobile);await mobile.locator('.address-lab').scrollIntoViewIfNeeded();
  await mobile.screenshot({path:'/tmp/llama_mobile.png'});
  await mobile.locator('.call-tree .tree-symbol').first().click();await sourceReady(mobile);
  await fit(mobile);const card=await mobile.locator('#source-card').boundingBox();
  assert.ok(card.x>=0&&card.x+card.width<=391);
  await mobileContext.close();
  const reducedContext=await context({reducedMotion:'reduce'}),reduced=await reducedContext.newPage();
  await open('asyncrt_workers',reduced,'#animation');await reduced.locator('#step').click();
  assert.equal(await reduced.locator('.chip-flying').count(),0);
  assert.equal(await reduced.evaluate(()=>document.getAnimations().length),0);
  await reducedContext.close();
  report.checks.push('Fullscreen source windows and stable deltas; numerical labs and page selection; every lesson at 390px; mobile source card; reduced motion');
}

async function noScriptAndPrint(){
  const ctx=await context({javaScriptEnabled:false}),p=await ctx.newPage();
  for(const {slug}of chapters){
    await p.goto(url(slug));await p.locator('#event-transcript>summary').click();
    assert.ok(await p.locator('#event-transcript>ol').first().isVisible());
    assert.ok(await p.locator('.call-tree pre').isVisible());
  }
  await p.goto(url('sources'));
  assert.equal(await p.locator('noscript a').count(),report.sourceFiles);
  await p.goto(url('driver'));await p.emulateMedia({media:'print'});
  assert.ok(await p.locator('#event-transcript>ol').first().isVisible(),'no-JavaScript print includes closed transcripts');
  await ctx.close();
  await open('driver');
  const original=await page.locator('details').evaluateAll(ds=>ds.map(d=>d.open));
  await page.pdf({path:'/tmp/llama_driver_print.pdf',format:'A4',printBackground:true});
  assert.deepEqual(await page.locator('details').evaluateAll(ds=>ds.map(d=>d.open)),original,'printing restores disclosures');
  await page.locator('#animation').screenshot({path:'/tmp/llama_driver.png'});
  report.checks.push('Every lesson readable without JavaScript; original source fallback links; no-JavaScript print; real PDF print/restored disclosures');
}

async function navigation(){
  const startup=path.resolve(out,'../../max_serve_llama2_7b_startup_call_stack.html');
  await page.goto(pathToFileURL(startup).href);
  const link=page.locator('a.tree-doc-link[href="link_htmls/llama_2_7b/index.html"]');
  assert.equal(await link.count(),1);
  await link.evaluate(el=>{for(let parent=el.parentElement;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;});
  const [reader]=await Promise.all([page.context().waitForEvent('page'),link.click()]);
  await reader.waitForFunction(()=>window.LlamaReader);
  assert.equal(reader.url(),url('index'));
  await reader.selectOption('#chapter-select','asyncrt_execution.html');
  await reader.waitForURL(url('asyncrt_execution'));
  await reader.waitForFunction(()=>window.LlamaReader?.data.slug==='asyncrt_execution');
  await reader.locator('.site-header a[href="coverage.html"]').click();
  await reader.waitForURL(url('coverage'));
  assert.equal(await reader.locator('.coverage-list tbody tr').count(),35);
  await reader.locator('.site-header a[href="sources.html"]').click();
  await reader.waitForURL(url('sources'));await sourceReady(reader,'#source-view');
  await reader.selectOption('#source-select','kv');
  await reader.waitForFunction(()=>document.querySelector('#source-view').dataset.file==='kv');
  await sourceReady(reader,'#source-view');
  assert.equal(new URL(reader.url()).search,'?file=kv');
  await reader.locator('#source-search').fill('PACKED_PAGE_STRIDE = -1');
  await reader.locator('#source-search').press('Enter');
  await reader.waitForFunction(()=>document.querySelector('#source-view .target')?.textContent.includes('PACKED_PAGE_STRIDE = -1'));
  await reader.close();
  report.checks.push('Real startup InferenceSession link opens index; lesson selector; coverage navigation; offline source-file selector/history/search');
}

(async()=>{
  browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  report.chromium=browser.version();report.node=process.version;report.playwright=require('playwright/package.json').version;
  const ctx=await context();page=await ctx.newPage();page.setDefaultTimeout(12000);
  const navigationOnly=process.argv.includes('--navigation-only');
  if(!navigationOnly){await lessonStates();await playback();await windowsAndSources();await layoutsAndLabs();await noScriptAndPrint();}
  await navigation();
  assert.deepEqual(errors,[],'browser errors');assert.deepEqual(network,[],'reader must make no network requests');
  report.offline=true;report.pageErrors=errors;report.networkRequests=network;report.result='PASS';
  fs.writeFileSync(navigationOnly?'/tmp/llama_2_7b_navigation_report.json':'/tmp/llama_2_7b_browser_report.json',JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(async error=>{
  if(page)await page.screenshot({path:'/tmp/llama_browser_failure.png',fullPage:false}).catch(()=>{});
  console.error(error);console.error(JSON.stringify({errors,network}));
  if(browser)await browser.close();process.exitCode=1;
});
