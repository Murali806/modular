'use strict';
const assert=require('assert/strict'),fs=require('fs'),path=require('path');
const {pathToFileURL}=require('url'),{chromium}=require('playwright');
const renderedAudit=require('./compiler_browser_audit.cjs');
const task=path.resolve(__dirname,'..'),manifest=JSON.parse(fs.readFileSync(path.join(task,'compiler_examples_manifest.json'),'utf8'));
const home=pathToFileURL(path.join(task,'link_htmls/max_mlir_mojo_compiler_and_runtime.html')).href;
const checks=[],audits=[],errors=[],network=[],screenshots=[];
async function align(p,k){await p.locator('#'+k).evaluate(el=>scrollTo({top:scrollY+el.getBoundingClientRect().top-68,behavior:'instant'}));}
async function seek(p,k,index){await p.locator('#'+k+' [data-seek]').evaluate((e,i)=>{e.value=i;e.dispatchEvent(new Event('input',{bubbles:true}));},index);}
async function state(p,k){return p.evaluate(k=>compilerExampleSnapshot(k),k);}
async function finish(p,k){await align(p,k);await seek(p,k,Number(await p.locator('#'+k).getAttribute('data-frames'))-1);return state(p,k);}
async function audit(p,label){const a=await p.evaluate(renderedAudit);assert.deepEqual(a.fails,[],'Contrast: '+label);assert.deepEqual(a.overflow,[],'Actor text bounds: '+label);assert.deepEqual(a.occlusion,[],'Unobscured labels: '+label);assert.equal(a.documentOverflow,false,label);audits.push({label,samples:a.samples,minimum:a.minimum});}
async function shot(p,k,name){const filename='compiler_example_'+name+'.png';await p.locator('#'+k).screenshot({path:path.join(task,'validation',filename)});screenshots.push(filename);}
async function readingDiagrams(p){
 const reading=p.locator('#toolchain-text');
 const reference=reading.locator('#toolchain-reference');
 if(await reference.count()&&await reference.getAttribute('open')===null)await reference.locator(':scope > summary').click();
 assert.deepEqual(await reading.locator('.qna-figure').evaluateAll(es=>es.map(e=>e.dataset.diagram)),['build','macro','ir','kernel','offload']);
 assert.equal(await reading.locator('code.language-mermaid').count(),0);
 assert.doesNotMatch(await reading.textContent(),/This Mermaid diagram exceeds the terminal renderer's size limits|\bflowchart (?:LR|TD)\b/);
 const original=reading.locator('[data-original-point="1"]');
 assert.deepEqual(await original.evaluate(e=>[...e.children].filter(n=>n.matches('pre,figure')).map(n=>n.matches('figure')?n.dataset.diagram:'text')),['text','build','text','macro','text']);
 for(const key of ['build','macro'])assert.equal(await original.locator('[data-diagram="'+key+'"] svg[role="img"]').isVisible(),true);
}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const setup=async (file,options={})=>{const context=await browser.newContext({viewport:{width:1600,height:1180},reducedMotion:'reduce',...options});await context.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});await context.addInitScript(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async value=>window.__copied=value}});});const p=await context.newPage();p.setDefaultTimeout(8000);p.on('pageerror',e=>errors.push(e.message));await p.goto(file||home);return {p,context};};
 try {
  const desktop=await setup();const p=desktop.p;
  assert.equal(await p.locator('#examples .example-link').count(),4);
  for(const e of manifest.examples){
   await p.goto(home);await p.locator('#examples a[href="'+path.basename(e.page)+'"]').click();assert.ok(p.url().endsWith(path.basename(e.page)));await p.waitForTimeout(60);
   assert.equal(await p.locator('[data-lesson]').count(),5);for(const k of ['toolchain','build','compile','run'])assert.equal((await state(p,k)).index,0);
   await audit(p,e.example.id+' initial');
   const payloadText=await p.locator('#compiler-examples-data').evaluate(e=>JSON.parse(e.textContent));
   await align(p,'toolchain');await p.locator('#toolchain-text-tab').click();
   assert.equal(await p.locator('#toolchain-animation').isVisible(),false);assert.equal(await p.locator('#toolchain-text').isVisible(),true);
   assert.equal(await p.locator('#toolchain [data-original-point]').count(),7);assert.equal(await p.locator('#toolchain .qna-figure').count(),5);
   if(e.example.id==='addition'){
    assert.equal(await p.locator('#toolchain-reference').getAttribute('open'),null);
    assert.equal(await p.locator('.call-stage').count(),4);
    for(const stage of ['graph','session','compile','init']){
     const article=p.locator('#toolchain-call-'+stage);
     assert.equal(await article.locator('.call-contracts > div').count(),3);
     assert.equal(await article.locator(':scope > .call-tree-scroll').isVisible(),true);
     await shot(p,'toolchain-call-'+stage,'addition_calltree_'+stage);
    }
    const cases=[['graph','graph_init'],['session','ct_session_native'],['compile','ct_context_compile'],['init','ct_model_setup']];
    for(const [stage,key] of cases){
     await p.locator('#toolchain-call-'+stage+' > .call-tree-scroll a[data-ref="'+key+'"]').click();
     assert.equal(await p.locator('#source-dialog').isVisible(),true);
     const r=payloadText.references[key];
     const full=await p.locator('#source-file .source-line').evaluateAll(es=>es.map(e=>[...e.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('')).join('\n'));
     assert.equal(full,payloadText.files[r.file].text);
     await p.keyboard.press('Escape');
    }
    await p.locator('#toolchain-call-graph-package > summary').click();
    await p.locator('#toolchain-call-graph-package a[data-ref="ct_package_write"]').click();
    assert.equal(await p.locator('#source-dialog').isVisible(),true);await p.keyboard.press('Escape');
    await p.locator('#toolchain-call-graph-package > summary').click();
    await p.locator('.calltree-nav a[href="#toolchain-call-init"]').click();
    assert.equal(await p.locator('#toolchain').getAttribute('data-mode'),'text');
    assert.ok(p.url().endsWith('#toolchain-call-init'));
   }
   await readingDiagrams(p);
   for(let n=2;n<=7;n++)assert.equal(await p.locator('#toolchain [data-original-point="'+n+'"]').textContent(),payloadText.toolchain.originals[n]);
   assert.equal(payloadText.toolchain.originals[1],fs.readFileSync(path.join(task,'src/compiler_toolchain_point_1.txt'),'utf8').replace(/\n+$/,''));
   await p.locator('#toolchain [data-qna-copy]').click();assert.equal(await p.evaluate(()=>window.__copied),payloadText.toolchain.text);
   if(e.example.id==='addition')for(const key of ['build','macro'])await shot(p,'toolchain-q1 [data-diagram="'+key+'"]','addition_'+key+'_diagram');
   await audit(p,e.example.id+' complete text');
   await p.locator('#toolchain-text-tab').focus();await p.keyboard.press('ArrowLeft');assert.equal(await p.locator('#toolchain-animation-tab').getAttribute('aria-selected'),'true');assert.equal(await p.locator('#toolchain-text').isVisible(),false);
   if(e.example.id==='addition')await p.locator('#toolchain-reference').evaluate(detail=>detail.open=false);
   await seek(p,'toolchain',10);const readingIndex=(await state(p,'toolchain')).index;await p.locator('#toolchain [data-read-current]').click();assert.equal(await p.locator('#toolchain').getAttribute('data-mode'),'text');assert.equal(await p.evaluate(()=>document.activeElement.id),'toolchain-q3-3-1');
   if(e.example.id==='addition')assert.equal(await p.locator('#toolchain-reference').getAttribute('open'),'');
   await p.locator('#toolchain [data-return-animation]').first().click();assert.equal((await state(p,'toolchain')).index,readingIndex);assert.equal(await p.locator('#toolchain').getAttribute('data-playing'),'false');
   for(const k of ['toolchain','build','compile','run']){await align(p,k);const total=Number(await p.locator('#'+k).getAttribute('data-frames'));for(let i=0;i<total;i++){await seek(p,k,i);await audit(p,e.example.id+' '+k+' '+i);}await shot(p,k,e.example.id+'_'+k);}
   const original=await state(p,'run');assert.equal(original.hostRead,e.example.id!=='attention');
   const middle=e.example.id==='attention'?15:e.example.id==='matmul'?8:e.example.id==='epilogue'?5:null;
   if(middle!==null){await seek(p,'run',middle);await shot(p,'run',e.example.id+'_mechanism');await finish(p,'run');}
   const values=await p.locator('#run [data-actor^="O1-"]:not([hidden]) .cell-label').allTextContents();assert.deepEqual(values,original.computed.map(n=>Number.isInteger(n)?String(n):Number(n.toFixed(4)).toString()));
   for(const target of ['cpu','rocm']){await p.locator('#target-select').selectOption(target);for(const k of ['toolchain','compile','run']){await align(p,k);const total=Number(await p.locator('#'+k).getAttribute('data-frames'));for(let i=0;i<total;i++){await seek(p,k,i);await audit(p,e.example.id+' '+target+' '+k+' '+i);}}const route=await state(p,'toolchain');assert.equal(route.selectedBackend,target==='cpu'?'HostBackend':'AMDGPUBackend');assert.equal(route.targetCode,true);if(target==='cpu'){assert.equal(await p.locator('#run [data-wire="upload"]').isVisible(),false);assert.equal(await p.locator('#run [data-actor^="A-shared"]:not([hidden])').count(),0);}}
   await p.locator('#target-select').selectOption('cuda');
   if(e.example.id==='epilogue'){await p.locator('#fusion-select').selectOption('separate');await finish(p,'compile');await audit(p,'epilogue separate compilation');for(let i=0;i<Number(await p.locator('#run').getAttribute('data-frames'));i++){await seek(p,'run',i);await audit(p,'epilogue separate '+i);}assert.equal((await state(p,'run')).traffic.reads,8);assert.equal((await state(p,'run')).traffic.writes,8);assert.equal(await p.locator('#run [data-actor="TMP"]:not([hidden])').count(),0);await seek(p,'run',6);assert.equal(await p.locator('#run [data-actor="TMP1-life"]:not([hidden])').count(),1);assert.equal(await p.locator('#run [data-actor="TMP2-life"]:not([hidden])').count(),1);await shot(p,'run','epilogue_intermediates');await p.locator('#fusion-select').selectOption('fused');await finish(p,'run');assert.equal(await p.locator('#run [data-actor="TMP"]:not([hidden])').count(),0);assert.deepEqual((await state(p,'run')).computed,original.computed);}
   if(e.example.id==='matmul'){await p.locator('#algorithm-select').selectOption('naive');await finish(p,'compile');await finish(p,'run');assert.deepEqual((await state(p,'run')).computed,[11,8,3,6]);await audit(p,'matmul naive GPU');await p.locator('#algorithm-select').selectOption('tiled');}
   if(e.example.id==='attention'){await align(p,'run');await seek(p,'run',5);const slot=p.locator('#run [data-actor="online-score-slot-0"]');await slot.evaluate(e=>e.dataset.sameStorage='retained');await seek(p,'run',10);assert.equal(await slot.isVisible(),false);await seek(p,'run',11);assert.equal(await slot.getAttribute('data-same-storage'),'retained');assert.equal(await slot.locator('text.cell-label').textContent(),'3');await shot(p,'run','attention_score_storage');await finish(p,'run');}
   await align(p,'build');const badge=p.locator('#build [data-card]');await badge.hover();assert.equal(await p.locator('#explanation-card').isVisible(),true);await p.locator('#explanation-card').hover();await p.waitForTimeout(160);assert.equal(await p.locator('#explanation-card').isVisible(),true);await p.mouse.move(1580,1120);await p.waitForTimeout(170);assert.equal(await p.locator('#explanation-card').isVisible(),false);await badge.focus();await p.keyboard.press('Escape');assert.equal(await p.locator('#explanation-card').isVisible(),false);
   await p.locator('#legend-button').hover();assert.equal(await p.locator('#legend-card').isVisible(),false);await p.locator('#legend-button').click();assert.equal(await p.locator('#legend-card').isVisible(),true);await audit(p,e.example.id+' legend');
   assert.equal(new Set(await p.locator('.legend-row .swatch').evaluateAll(es=>es.map(e=>getComputedStyle(e).backgroundColor))).size,11);
   const matching=await p.evaluate(()=>[...document.querySelectorAll('.legend-row')].every(row=>{const key=row.dataset.legendComponent,swatch=row.querySelector('.swatch'),actor=document.querySelector('.visual [data-color="'+key+'"] .actor-label, .visual [data-component="'+key+'"] .owner-title');return !actor||getComputedStyle(actor).fill===getComputedStyle(swatch).backgroundColor;}));assert.ok(matching);
   await p.locator('#legend-card a[data-ref="registration"]').first().click();const payload=await p.locator('#compiler-examples-data').evaluate(e=>JSON.parse(e.textContent)),ref=payload.references.registration;
   assert.equal(await p.locator('#source-dialog').isVisible(),true);const actual=await p.locator('#source-file .source-line').evaluateAll(es=>es.map(e=>[...e.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('')).join('\n'));assert.equal(actual,payload.files[ref.file].text);
   await p.locator('#source-copy').click();assert.equal(await p.evaluate(()=>window.__copied),payload.files[ref.file].text.split('\n').slice(ref.start-1,ref.end).join('\n'));
   await p.locator('#source-search').fill(ref.symbol);await p.locator('#source-next').click();assert.ok(await p.locator('.search-hit').count());
   const download=await Promise.all([p.waitForEvent('download'),p.locator('#source-download').click()]);assert.equal(fs.readFileSync(await download[0].path(),'utf8'),payload.files[ref.file].text);
   await p.keyboard.press('Escape');assert.equal(await p.locator('#legend-card').isVisible(),true);await p.keyboard.press('Escape');assert.equal(await p.locator('#legend-card').isVisible(),false);
  }
  checks.push('Four actual workbench links; each standalone reader starts a connected journey, target-code walkthrough and focused scenes paused');
  checks.push('Every default, CPU and AMD intermediate frame: contrast, actor bounds, label occlusion and document fit');
  checks.push('Fused/separate epilogue changes intermediate allocations; tiled/naive matmul changes visible execution');
  checks.push('Result cells agree with the model; source-specific host-copy behavior and CPU placement remain visible');
  checks.push('Hover/focus cards dismiss, L opens only on click, eleven colors match, original source/search/copy/download work');
  checks.push('Text/Animation tabs preserve the paused timeline; five offline diagrams replace Mermaid source in place; exact previous answers remain in complete copy/download');
  checks.push('Add-one Text starts with four annotated input/output/example call trees; native/package links open complete source and old step links reveal the collapsed reference');
  await desktop.context.close();
  const addition=manifest.examples.find(e=>e.example.id==='addition');
  const deep=await setup(pathToFileURL(path.join(task,addition.page)).href+'#toolchain-call-compile-kernels');
  assert.equal(await deep.p.locator('#toolchain').getAttribute('data-mode'),'text');
  assert.equal(await deep.p.locator('#toolchain-call-compile-kernels').getAttribute('open'),'');
  assert.equal(await deep.p.locator('#toolchain-call-compile-kernels .call-tree-scroll').isVisible(),true);
  await deep.p.goto(pathToFileURL(path.join(task,addition.page)).href+'#toolchain-q3-3-1');
  assert.equal(await deep.p.locator('#toolchain-reference').getAttribute('open'),'');
  assert.equal(await deep.p.locator('#toolchain-q3-3-1').isVisible(),true);
  await deep.context.close();
  for(const width of [320,390]){
   const narrow=await setup(pathToFileURL(path.join(task,addition.page)).href+'#toolchain-call-graph',{viewport:{width,height:850},isMobile:true,hasTouch:true});
   await audit(narrow.p,'addition call trees '+width);
   for(const detail of await narrow.p.locator('.addition-calltrees .call-branch').all()){
    await detail.locator(':scope > summary').tap();
    assert.equal(await detail.locator('.call-tree-scroll').isVisible(),true);
    await audit(narrow.p,'addition expanded '+await detail.getAttribute('id')+' '+width);
    await detail.locator(':scope > summary').tap();
   }
   if(width===390)await shot(narrow.p,'toolchain-call-graph','addition_calltree_mobile');
   await narrow.context.close();
  }
  checks.push('Call-tree and original-answer deep links work from a fresh file open; all nine expanded branches fit at 320/390px with local tree scrolling');
  const matmul=manifest.examples.find(e=>e.example.id==='matmul');
  const motion=await setup(pathToFileURL(path.join(task,matmul.page)).href,{reducedMotion:'no-preference'});await align(motion.p,'run');await seek(motion.p,'run',5);await motion.p.locator('#run [data-action="play"]').click();await motion.p.waitForTimeout(470);
  const actor=motion.p.locator('#run [data-actor="A-shared-0-0"]');assert.equal(await actor.count(),1);await actor.evaluate(e=>e.dataset.identityCheck='same');const start=await actor.getAttribute('transform');await motion.p.waitForTimeout(140);assert.notEqual(await actor.getAttribute('transform'),start);
  await motion.p.locator('#run [data-action="play"]').click();const frozen=await actor.getAttribute('transform'),index=(await state(motion.p,'run')).index;await motion.p.waitForTimeout(240);assert.equal(await actor.getAttribute('transform'),frozen);assert.equal((await state(motion.p,'run')).index,index);
  await motion.p.locator('#run [data-action="play"]').click();await motion.p.waitForTimeout(130);assert.notEqual(await actor.getAttribute('transform'),frozen);assert.equal(await actor.getAttribute('data-identity-check'),'same');await motion.p.locator('#build').scrollIntoViewIfNeeded();await motion.p.waitForTimeout(170);assert.equal(await motion.p.locator('#run').getAttribute('data-playing'),'false');
  await motion.p.locator('#run [data-action="reset"]').click();assert.equal((await state(motion.p,'run')).index,0);assert.equal(await motion.p.locator('#run [data-actor^="A-shared"]:not([hidden])').count(),0);await motion.context.close();checks.push('Persistent shared-tile actors move; Pause freezes pose and steps; Resume/offscreen/reset behave correctly');
  const toolMotion=await setup(pathToFileURL(path.join(task,matmul.page)).href,{reducedMotion:'no-preference'});await align(toolMotion.p,'toolchain');await seek(toolMotion.p,'toolchain',15);const op=toolMotion.p.locator('#toolchain [data-actor="OP1"]');await op.evaluate(e=>e.dataset.retainedIdentity='same');await toolMotion.p.locator('#toolchain [data-action="play"]').click();await toolMotion.p.waitForTimeout(470);const moving=await op.getAttribute('transform');await toolMotion.p.waitForTimeout(110);assert.notEqual(await op.getAttribute('transform'),moving);
  await toolMotion.p.locator('#toolchain-text-tab').click();const frozenOp=await op.getAttribute('transform'),frozenIndex=(await state(toolMotion.p,'toolchain')).index;await toolMotion.p.waitForTimeout(250);assert.equal(await op.getAttribute('transform'),frozenOp);assert.equal((await state(toolMotion.p,'toolchain')).index,frozenIndex);assert.equal(await toolMotion.p.locator('#toolchain').getAttribute('data-playing'),'false');
  await toolMotion.p.locator('#toolchain-animation-tab').click();assert.equal(await op.getAttribute('transform'),frozenOp);await toolMotion.p.locator('#toolchain [data-action="play"]').click();await toolMotion.p.waitForTimeout(120);assert.notEqual(await op.getAttribute('transform'),frozenOp);assert.equal(await op.getAttribute('data-retained-identity'),'same');await toolMotion.p.locator('#toolchain [data-action="play"]').click();await toolMotion.context.close();checks.push('Entering Text freezes a moving OP1 and semantic step; returning to Animation resumes the same actor');
  for(const width of [320,390])for(const e of manifest.examples){const mobile=await setup(pathToFileURL(path.join(task,e.page)).href,{viewport:{width,height:850},isMobile:true,hasTouch:true});await finish(mobile.p,'run');await audit(mobile.p,e.example.id+' narrow '+width);await mobile.p.locator('#run [data-card]').tap();assert.equal(await mobile.p.locator('#explanation-card').isVisible(),true);await audit(mobile.p,e.example.id+' touch card '+width);await mobile.p.locator('#run [data-card]').tap();assert.equal(await mobile.p.locator('#explanation-card').isVisible(),false);await mobile.p.locator('#legend-button').tap();await audit(mobile.p,e.example.id+' touch legend '+width);await mobile.p.keyboard.press('Escape');await align(mobile.p,'toolchain');await mobile.p.locator('#toolchain-text-tab').tap();await readingDiagrams(mobile.p);await audit(mobile.p,e.example.id+' narrow text '+width);assert.ok(await mobile.p.locator('#toolchain-q3-3-1').count());if(width===390&&e.example.id==='addition')await shot(mobile.p,'toolchain-q1 [data-diagram="build"]','addition_build_diagram_mobile');await mobile.context.close();}
  checks.push('All four readers at 320/390px contain diagram scrolling and fit touch cards/legends');
  for(const e of manifest.examples){const nojs=await setup(pathToFileURL(path.join(task,e.page)).href,{javaScriptEnabled:false,viewport:{width:1000,height:900}});assert.ok(await nojs.p.locator('noscript').innerText());assert.equal(await nojs.p.locator('.source-cell').count(),20);assert.equal(await nojs.p.locator('#toolchain-text').isVisible(),true);await readingDiagrams(nojs.p);assert.equal(await nojs.p.locator('#toolchain [data-original-point]').count(),7);assert.equal(await nojs.p.locator('#toolchain [data-transcript] > li').count(),e.toolchain.steps);await nojs.p.locator('.source-cell').first().locator('summary').click();assert.ok((await nojs.p.locator('.source-cell').first().locator('code').innerText()).includes('InferenceSession'));if(e.example.id==='addition'){assert.equal(await nojs.p.locator('.call-stage').count(),4);await nojs.p.locator('#toolchain-call-graph-package > summary').click();assert.equal(await nojs.p.locator('#toolchain-call-graph-package .call-tree-scroll').isVisible(),true);await nojs.p.locator('#toolchain-call-graph-package > summary').click();}await nojs.p.emulateMedia({media:'print'});assert.equal(await nojs.p.locator('.menu').isVisible(),false);assert.equal(await nojs.p.locator('#toolchain-text').isVisible(),true);await readingDiagrams(nojs.p);if(e.example.id==='addition'){assert.equal(await nojs.p.locator('#toolchain-call-graph-package .call-tree-scroll').isVisible(),true);await shot(nojs.p,'toolchain-q1 [data-diagram="build"]','addition_build_diagram_print');}await nojs.context.close();}
  checks.push('Offline network-free readers retain complete original sources and portable flows with JavaScript disabled and in print');
  assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
  fs.writeFileSync(path.join(task,'validation/compiler_examples_browser.json'),JSON.stringify({evidence:'CPU-RUN · Chromium teaching animation only; no MAX/GPU execution',checks,audits,screenshots,errors,network},null,2)+'\n');console.log('Example browser checks passed:',checks.length,'groups /',audits.length,'render audits');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
