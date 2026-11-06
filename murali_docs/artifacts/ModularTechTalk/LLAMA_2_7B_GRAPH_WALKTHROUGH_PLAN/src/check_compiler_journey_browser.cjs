'use strict';
const assert=require('assert/strict'),fs=require('fs'),path=require('path'),{pathToFileURL}=require('url'),{chromium}=require('playwright');
const {JourneyModel}=require('./compiler_journey_model.js'),renderedAudit=require('./compiler_browser_audit.cjs');
const task=path.resolve(__dirname,'..'),manifest=JSON.parse(fs.readFileSync(path.join(task,'compiler_examples_manifest.json'))),audits=[],errors=[],network=[],screenshots=[];
const pages=[{id:'overview',page:'link_htmls/max_mlir_mojo_compiler_and_runtime.html'},...manifest.examples.map(e=>({id:e.example.id,page:e.page}))];
async function align(p){await p.locator('#journey').evaluate(e=>scrollTo({top:scrollY+e.getBoundingClientRect().top-68,behavior:'instant'}));}
async function seek(p,i){await p.locator('#journey [data-seek]').evaluate((e,index)=>{e.value=index;e.dispatchEvent(new Event('input',{bubbles:true}));},i);}
async function state(p){return p.evaluate(()=>compilerJourneySnapshot());}
async function audit(p,label){const a=await p.evaluate(renderedAudit);assert.deepEqual(a.fails,[],label+' contrast');assert.deepEqual(a.overflow,[],label+' bounds');assert.deepEqual(a.occlusion,[],label+' occlusion');assert.equal(a.documentOverflow,false,label+' document fit');audits.push({label,minimum:a.minimum});}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const setup=async(item,options={})=>{const context=await browser.newContext({viewport:{width:1600,height:1180},reducedMotion:'reduce',...options});await context.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});const p=await context.newPage();p.on('pageerror',e=>errors.push(e.message));p.setDefaultTimeout(7000);await p.goto(pathToFileURL(path.join(task,item.page)).href);return {p,context};};
 try {
  for(const item of pages){
   const {p,context}=await setup(item);assert.equal((await state(p)).index,0);assert.equal(await p.locator('#journey').getAttribute('data-playing'),'false');
   const id=item.id==='overview'?'addition':item.id;
   for(const target of ['cuda','cpu','rocm'])for(const cache of ['cold','kernel-hit','artifact-hit']){
    await p.locator('#target-select').selectOption(target);await p.locator('#journey [data-cache]').selectOption(cache);await align(p);
    const model=new JourneyModel(id,{target,cache});
    for(let i=0;i<model.frames.length;i++){await seek(p,i);await audit(p,item.id+' '+target+' '+cache+' '+i);const s=await state(p);assert.equal(s.phase,model.frames[i].phase);
     assert.equal(await p.locator('#journey [data-actor="G1"]:not([hidden])').count(),s.graph?1:0);
     assert.equal(await p.locator('#journey [data-actor="M1"]:not([hidden])').count(),s.initialized?1:0);
     assert.equal(await p.locator('#journey [data-actor="GPU1"]:not([hidden])').count(),s.deviceCode?1:0);
     if(s.kernelIR&&!s.hostCode)assert.equal(await p.locator('#journey [data-actor="C1"]').getAttribute('data-label'),'C1 · MGP');
    }
    const s=await state(p);assert.equal(s.call,2);assert.equal(s.compileRuns,cache==='cold'?1:0);assert.equal(await p.locator('#journey [data-actor="HOST1"]:not([hidden])').count(),1);
   }
   await p.locator('#target-select').selectOption('cuda');await p.locator('#journey [data-cache]').selectOption('cold');await align(p);
   const model=new JourneyModel(id),index=phase=>model.frames.findIndex(f=>f.phase===phase);
   await seek(p,index('expand'));const main=p.locator('#journey [data-actor="op-main"]');await main.focus();await p.keyboard.press('Enter');assert.equal((await state(p)).selection,'operation');
   await p.locator('#journey [data-actor="op-load"]').click();assert.equal((await state(p)).selection,'load');assert.equal(await p.locator('#journey [data-actor="op-load"]').getAttribute('aria-pressed'),'true');
   await seek(p,index('clone'));await p.locator('#journey [data-action="nextChange"]').click();assert.equal((await state(p)).phase,'kernel_lookup');
   await p.locator('#journey [data-stage="artifact"]').click();assert.equal((await state(p)).stage,'artifact');
   for(const [name,phase] of [['ir','llvm_dialect'],['code','host_code'],['reuse','ready2']]){await seek(p,index(phase));const file='compiler_journey_'+item.id+'_'+name+'.png';await p.locator('#journey').screenshot({path:path.join(task,'validation',file)});screenshots.push(file);}
   await p.locator('#journey [data-card]').hover();assert.equal(await p.locator('#explanation-card').isVisible(),true);await p.mouse.move(1580,1100);await p.waitForTimeout(180);assert.equal(await p.locator('#explanation-card').isVisible(),false);
   await seek(p,index('expand'));await p.locator('#journey .journey-source summary').click();assert.ok((await p.locator('#journey [data-source-preview]').innerText()).length>30);await p.locator('#journey [data-source-link] a').click();assert.equal(await p.locator('#source-dialog').isVisible(),true);await p.keyboard.press('Escape');
   await context.close();console.log('Journey browser routes:',item.id);
  }
  const item=pages[1],motion=await setup(item,{reducedMotion:'no-preference'}),model=new JourneyModel('addition');await align(motion.p);const i=model.frames.findIndex(f=>f.phase==='semantics');await seek(motion.p,i-1);
  const actor=motion.p.locator('#journey [data-actor="op-main"]');await actor.evaluate(e=>e.dataset.persistentCheck='original');await motion.p.locator('#journey [data-action="play"]').click();await motion.p.waitForTimeout(450);const start=await actor.getAttribute('transform');await motion.p.waitForTimeout(130);assert.notEqual(await actor.getAttribute('transform'),start);
  await motion.p.locator('#journey [data-action="play"]').click();const frozen=await actor.getAttribute('transform'),index=(await state(motion.p)).index;await motion.p.waitForTimeout(240);assert.equal(await actor.getAttribute('transform'),frozen);assert.equal((await state(motion.p)).index,index);
  await motion.p.locator('#journey [data-action="play"]').click();await motion.p.waitForTimeout(120);assert.notEqual(await actor.getAttribute('transform'),frozen);assert.equal(await actor.getAttribute('data-persistent-check'),'original');await motion.p.locator('#run').scrollIntoViewIfNeeded();await motion.p.waitForTimeout(170);assert.equal(await motion.p.locator('#journey').getAttribute('data-playing'),'false');await motion.context.close();
  for(const width of [320,390])for(const item of pages){const {p,context}=await setup(item,{viewport:{width,height:850},isMobile:true,hasTouch:true});await align(p);await seek(p,Number(await p.locator('#journey').getAttribute('data-frames'))-1);await audit(p,item.id+' mobile '+width);await p.locator('#journey [data-card]').tap();assert.equal(await p.locator('#explanation-card').isVisible(),true);await p.locator('#journey [data-card]').tap();assert.equal(await p.locator('#explanation-card').isVisible(),false);await context.close();}
  const nojs=await setup(pages[1],{javaScriptEnabled:false});assert.ok((await nojs.p.locator('#journey .transcript pre').textContent()).includes('execute(B2)'));await nojs.p.emulateMedia({media:'print'});assert.equal(await nojs.p.locator('#journey .playback').isVisible(),false);await nojs.context.close();
  assert.deepEqual(errors,[]);assert.deepEqual(network,[]);fs.writeFileSync(path.join(task,'validation/compiler_journey_browser.json'),JSON.stringify({evidence:'CPU-RUN · offline Chromium teaching animations; no MAX compilation/inference or GPU measurements',audits,screenshots,errors,network,checks:['All five pages × three targets × three cache paths','Visible original graph / plan / LLVM module / initialized model distinctions','Pointer/keyboard selection and meaningful-change seeking','Persistent actor motion, frozen pause, resume and offscreen pause','Hover/touch cards, source explorer, 320/390px fit, no-JS and print']},null,2)+'\n');console.log('Journey browser passed:',audits.length,'render audits');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
