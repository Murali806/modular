'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url'),{chromium}=require('playwright');
const audit=require('./compiler_browser_audit.cjs');
const task=path.resolve(__dirname,'..'),errors=[],requests=[],checks=[],screenshots=[];
const pages={addition:'max_example_01_add_one_compilation_and_execution.html',attention:'max_example_04_fused_attention_online_softmax.html'};
const url=id=>pathToFileURL(path.join(task,'link_htmls',pages[id])).href;
const closeCard=async page=>{const c=page.locator('#call-explanation-card');if(await c.isVisible())await c.locator('[data-window-close]').click();};
const openCard=async(page,id)=>{await closeCard(page);const a=page.locator('#toolchain-callstack [data-call-explanation="'+id+'"]').first();await a.scrollIntoViewIfNeeded();await a.click();const c=page.locator('#call-explanation-card');assert.equal(await c.isVisible(),true,id);return c;};
const seek=async(page,id,index)=>page.locator('#'+id+' [data-seek]').evaluate((el,n)=>{el.value=n;el.dispatchEvent(new Event('input',{bubbles:true}));},index);
const visibleFrame=async c=>c.locator('[data-attention-mechanism]').evaluate(el=>attentionMechanismSnapshot(el));
const closeSource=async page=>page.locator('#source-dialog #source-close').click();

(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 try{
  const ctx=await browser.newContext({viewport:{width:1440,height:1050},reducedMotion:'reduce'});
  await ctx.route(/^https?:/,r=>{requests.push(r.request().url());return r.abort();});
  const p=await ctx.newPage();p.setDefaultTimeout(10000);p.on('pageerror',e=>errors.push(e.message));
  for(const id of ['addition','attention']){
   await p.goto(url(id));
   assert.equal(await p.locator('#toolchain-text-tab,#toolchain-text').count(),0);
   assert.deepEqual(await p.locator('#toolchain [role="tab"]').allTextContents(),['Animation','Call Stack Tree']);
   await p.locator('#toolchain-callstack-tab').click();
   const data=JSON.parse(await p.locator('#compiler-examples-data').textContent()),stack=p.locator('#toolchain-callstack');
   const expected=id==='attention'?['graph-constructor-copy','session-constructor','session-compile','session-init','model-execute','attention-kernel']:['graph-constructor-copy','session-constructor','session-compile','session-init','model-execute'];
   for(const tree of expected){assert.equal(await stack.locator('[data-call-tree-id="'+tree+'"]').count(),1);assert.ok(await stack.locator('[data-call-tree-id="'+tree+'"] a[data-ref]').count()>20);}
   assert.equal(await stack.locator('#toolchain-call-example-launch').isVisible(),true);
   for(const cardId of data.toolchain.calltrees.cards){
    const trigger=stack.locator('[data-call-explanation="'+cardId+'"]').first();
    if(!await trigger.count())continue; // two retained add-one cards belonged only to its removed Text view
    const c=await openCard(p,cardId);assert.equal(await c.locator('[data-window-hold],[data-window-expand],[data-window-close]').count(),3);
    assert.equal(await c.locator('[data-attention-mechanism],[data-card-conversion]').count(),1);
    const src=c.locator('a[data-ref]').first(),key=await src.getAttribute('data-ref'),line=Number(await src.getAttribute('data-line')||data.references[key].line);
    await src.click();const dialog=p.locator('#source-dialog');assert.equal(await dialog.isVisible(),true);
    assert.equal(Number(await dialog.locator('#source-file .source-line.exact').getAttribute('data-line')),line);
    await closeSource(p);await closeCard(p);
   }
   checks.push(id+': two-view layout, separate call trees, E retained, every inline card and exact source destination opens');
   await p.locator('#toolchain-animation-tab').click();
   await p.locator('#toolchain [data-read-current]').click();assert.equal(await stack.isVisible(),true);
   await closeCard(p);await p.locator('#toolchain-animation-tab').click();
   await p.locator('#toolchain .experiment-top [data-fullscreen-toggle]').click();await p.waitForFunction(()=>document.fullscreenElement?.id==='toolchain-animation');
   const fullscreen=await p.evaluate(()=>{const panel=document.fullscreenElement,svg=panel.querySelector('svg'),a=svg.getBoundingClientRect(),b=panel.getBoundingClientRect();return {panel:[b.x,b.y,b.width,b.height],viewport:[innerWidth,innerHeight],fits:a.width<=b.width+1&&a.height<=b.height+1};});
   assert.deepEqual(fullscreen.panel,[0,0,...fullscreen.viewport]);assert.equal(fullscreen.fits,true);
   await p.locator('#toolchain-animation [data-fullscreen-toggle]').click();await p.waitForFunction(()=>!document.fullscreenElement);
   checks.push(id+': Read-current routes into retained call stack; Chrome fullscreen fits and exits');
  }
  await p.goto(url('attention'));await p.locator('#toolchain-callstack-tab').click();
  const stack=p.locator('#toolchain-callstack'),graph=stack.locator('#graph-constructor-copy');
  const graphText=await graph.textContent();assert.ok(graphText.indexOf('Insert mo.GraphOp')<graphText.indexOf('_import_kernels'));assert.ok(graphText.indexOf('_import_kernels')<graphText.indexOf('results = ops.custom'));
  for(const line of await graph.locator('.call-stack-copy-line').all())if((await line.textContent()).replace(/[│\s]/g,''))assert.ok(await line.locator('a[data-ref]').count()>0);
  for(const [key,line] of [['entry',41],['entry',56],['entry',57],['entry',60],['graph_init',856],['graph_init',864],['custom_api',114],['ct_graph_op_build',849]]){
   const a=graph.locator('a[data-ref="'+key+'"][data-line="'+line+'"]').first();await a.click();assert.equal(await p.locator('#source-dialog #source-file .source-line.exact').getAttribute('data-line'),String(line));await closeSource(p);
  }
  checks.push('attention: source-ordered shell → package → with-body and exact call-site links on graph tree rows');
  let c=await openCard(p,'toolchain-call-kernel-online');await c.locator('[data-window-expand]').click();
  const numeric=c.locator('[data-attention-mechanism="online"]');
  const cardSeek=async(el,index)=>el.locator('[data-am-slider]').evaluate((r,n)=>{r.value=n;r.dispatchEvent(new Event('input',{bubbles:true}));},index);
  for(const query of ['actual','query1']){
   await numeric.locator('[data-am-scenario]').selectOption(query);
   let frame=await visibleFrame(c);assert.equal(frame.index,0);assert.equal(frame.scene.m,null);
   let alphaReached=false,maxReached=false;
   for(let i=1;i<frame.length;i++){
    await cardSeek(numeric,i);frame=await visibleFrame(c);assert.equal(frame.index,i);
    assert.equal(await numeric.locator('[data-am-step-action]').textContent(),frame.action);
    if(frame.refs[0].line===349&&frame.scene.tile===1){assert.equal(frame.scene.mNew,query==='actual'?3:1);maxReached=true;}
    if(frame.refs[0].line===355&&frame.scene.tile===1){assert.ok(Math.abs(frame.scene.alpha-(query==='actual'?Math.exp(-2):1))<1e-12);alphaReached=true;}
    if(frame.refs[0].line===358)assert.deepEqual(frame.values.E,frame.values.S);
    if(frame.refs[0].line===344){assert.equal(frame.scene.tileMax,null);assert.equal(frame.scene.dotTerms,null);}
    if(frame.refs[0].line===360)frame.values.E.forEach(x=>assert.ok(x>0&&x<=1));
    if([370,371,372].includes(frame.refs[0].line)){
     const scale=numeric.locator('[data-am-output-scale]');assert.equal(await scale.count(),1);
     assert.equal(await scale.getAttribute('data-am-changed'),'true');
     assert.equal(await numeric.locator('[data-am-m-slot]').getAttribute('data-am-changed'),'false');
     const visible=await scale.locator('b').textContent();
     if(typeof frame.values.delta==='number')assert.ok(Math.abs(Number(visible)-frame.values.delta)<1e-6);else assert.equal(visible,frame.values.delta);
    }
    if(frame.refs[0].line===373)frame.values.o.forEach((x,d)=>assert.ok(Math.abs(x-frame.scene.oldO[d]*frame.scene.alpha*frame.scene.oldL/frame.scene.lNew)<1e-12));
    const refs=await numeric.locator('[data-am-sources] a').evaluateAll(a=>a.map(x=>[x.dataset.ref,Number(x.dataset.line)]));
    assert.deepEqual(refs,frame.refs.map(r=>[r.key,r.line]));
   }
   assert.equal(maxReached&&alphaReached,true);assert.equal(frame.scene.match,true);
   assert.match(await numeric.locator('[data-am-reference]').textContent(),/MATCH/);
   const q=query==='actual'?[1,0]:[0,1],keys=[[1,0],[0,1],[3,0],[2,1]],values=[[1,0],[0,2],[2,1],[1,3]],scores=keys.map(k=>q.reduce((t,x,d)=>t+x*k[d],0)),max=Math.max(...scores),exps=scores.map(x=>Math.exp(x-max)),denom=exps.reduce((a,b)=>a+b,0);
   const ref=[0,1].map(d=>values.reduce((sum,v,j)=>sum+exps[j]/denom*v[d],0));frame.values.o.forEach((x,d)=>assert.ok(Math.abs(x-ref[d])<1e-12));
   await numeric.locator('[data-am-action="back"]').click();assert.equal((await visibleFrame(c)).index,frame.length-2);
   await numeric.locator('[data-am-action="reset"]').click();assert.deepEqual((await visibleFrame(c)).values.o,[0,0]);
  }
  checks.push('attention: every source-order recurrence operation, growing/unchanged maxima, bounded exponentials, visible output-scale scratch with identity-based highlighting, exact prior-output rescaling and two independent query references pass');
  await numeric.locator('[data-am-scenario]').selectOption('actual');await cardSeek(numeric,42);
  await c.screenshot({path:path.join(task,'validation/compiler_attention_online_card.png')});screenshots.push('compiler_attention_online_card.png');
  await c.locator('[data-window-hold]').click();assert.equal(await c.getAttribute('data-card-held'),'true');
  await c.locator('.reader-window-header').focus();await p.keyboard.press('ArrowRight');
  const heldIndex=(await visibleFrame(c)).index;
  const next=stack.locator('[data-call-explanation="toolchain-call-graph-custom-verify"]').first();await next.scrollIntoViewIfNeeded();await next.click();
  const held=p.locator('[data-held-window="true"]');assert.equal(await held.count(),1);assert.equal((await held.locator('[data-attention-mechanism]').evaluate(el=>attentionMechanismSnapshot(el))).index,heldIndex);
  assert.match(await c.locator('#call-explanation-title').textContent(),/Kernel validation/);await held.locator('[data-window-close]').click();await closeCard(p);
  checks.push('attention: Expand, keyboard movement, Hold preserves detailed scene/index while another card opens, and Close removes it');
  for(const id of JSON.parse(await p.locator('#compiler-examples-data').textContent()).toolchain.calltrees.cards){
   c=await openCard(p,id);await c.locator('[data-window-expand]').click();const motion=c.locator('[data-attention-mechanism]');
   const cases=await motion.locator('[data-am-scenario] option').evaluateAll(a=>a.map(x=>x.value));
   for(const key of cases){
    await motion.locator('[data-am-scenario]').selectOption(key);
    const initial=await visibleFrame(c);
    for(const i of [...new Set([0,Math.floor(initial.length/2),initial.length-1])]){
     await cardSeek(motion,i);const frame=await visibleFrame(c);
     assert.equal(await motion.locator('[data-am-step-input]').textContent(),frame.input);
     assert.equal(await motion.locator('[data-am-step-action]').textContent(),frame.action);
     assert.equal(await motion.locator('[data-am-step-output]').textContent(),frame.output);
     assert.equal(await motion.locator('[data-am-step-why]').textContent(),frame.why);
     for(const field of frame.focus)assert.equal(await motion.locator('[data-am-node="'+field+'"]').getAttribute('data-am-changed'),'true');
     for(const field of await motion.locator('[data-am-field]').evaluateAll(a=>a.map(x=>({keys:x.dataset.amField.split(' '),changed:x.dataset.amChanged}))))assert.equal(field.changed,String(field.keys.some(k=>frame.focus.includes(k))));
    }
   }
   assert.ok(await c.locator('[data-call-tree-id] a[data-caller-source]').count()>0,id);
   if(['toolchain-call-graph-native-builder','toolchain-call-graph-retain-body','toolchain-call-kernel-mma','toolchain-call-kernel-launch','toolchain-call-kernel-storage','toolchain-call-compile-kernels','toolchain-call-init-gates'].includes(id)){
    await motion.locator('[data-am-scenario]').selectOption('actual');const shotFrame=await visibleFrame(c);await cardSeek(motion,Math.floor(shotFrame.length/2));
    const name='compiler_attention_'+id.replace('toolchain-call-','')+'.png';await c.screenshot({path:path.join(task,'validation',name)});screenshots.push(name);
   }
   await closeCard(p);
  }
  checks.push('attention: every card and alternate case exposes consistent IN/action/OUT/why, retained objects, changed fields and a source-linked internal tree');
  c=await openCard(p,'toolchain-call-graph-custom-verify');await c.locator('[data-window-expand]').click();const validation=c.locator('[data-attention-mechanism]');
  await validation.locator('[data-am-scenario]').selectOption('invalid');let vf=await visibleFrame(c);await cardSeek(validation,vf.length-1);vf=await visibleFrame(c);assert.match(vf.values.gate,/ERROR.*BX/);await closeCard(p);
  c=await openCard(p,'toolchain-call-init-gates');await c.locator('[data-window-expand]').click();const initGate=c.locator('[data-attention-mechanism]');await initGate.locator('[data-am-scenario]').selectOption('error');let ig=await visibleFrame(c);await cardSeek(initGate,ig.length-1);ig=await visibleFrame(c);assert.match(ig.values.execute,/init failed; M1 not returned/);assert.match(ig.values.setup,/failed init AsyncValue/);await closeCard(p);
  c=await openCard(p,'toolchain-call-kernel-mma');await c.locator('[data-window-expand]').click();const mma=c.locator('[data-attention-mechanism]');let mf=await visibleFrame(c);await cardSeek(mma,mf.length-1);mf=await visibleFrame(c);assert.equal(mf.scene.acc,20);assert.deepEqual(mf.scene.partials,[8,16,4,-8]);await closeCard(p);
  c=await openCard(p,'toolchain-call-kernel-launch');await c.locator('[data-window-expand]').click();const launch=c.locator('[data-attention-mechanism]');await launch.locator('[data-am-scenario]').selectOption('other-block');let lf=await visibleFrame(c);await cardSeek(launch,lf.length-1);lf=await visibleFrame(c);assert.deepEqual(lf.scene.blockXY,[1,1]);assert.deepEqual(lf.scene.grid,[4,2]);assert.equal(lf.scene.stored,true);await closeCard(p);
  checks.push('attention: invalid parameter stops verification, four MMA partials accumulate to20, and alternate output block retains disjoint ownership');
  const expected=[0,1].map(j=>[1,0,3,2].reduce((s,x,i)=>s+Math.exp(x-3)/[1,0,3,2].reduce((t,y)=>t+Math.exp(y-3),0)*[[1,0],[0,2],[2,1],[1,3]][i][j],0));
  await p.locator('#toolchain-animation-tab').click();
  for(const target of ['cuda','rocm','cpu']){
   await p.selectOption('#target-select',target);for(const lesson of ['toolchain','build','compile','run'])await seek(p,lesson,Number(await p.locator('#'+lesson).getAttribute('data-frames'))-1);
   const state=await p.evaluate(()=>compilerExampleSnapshot('run'));state.computed.forEach((x,j)=>assert.ok(Math.abs(x-expected[j])<1e-12));assert.equal(state.params.keyTile,target==='cpu'?4:8);
   assert.equal(state.hostRead,false);assert.equal(state.ready,true);
  }
  const inspected=await p.evaluate(audit);assert.equal(inspected.documentOverflow,false);assert.deepEqual(inspected.fails,[]);assert.deepEqual(inspected.overflow,[]);
  checks.push('attention: CPU/NVIDIA/AMD illustrations reach correct tiny output; source tile sizes and no implicit CPU copy retained');
  await ctx.close();
  for(const width of [320,390]){
   const mobile=await browser.newContext({viewport:{width,height:844},isMobile:true,hasTouch:true,reducedMotion:'reduce'});const m=await mobile.newPage();m.on('pageerror',e=>errors.push(e.message));await m.goto(url('attention'));await m.locator('#toolchain-callstack-tab').tap();
   assert.equal(await m.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
   const mc=await openCard(m,'toolchain-call-kernel-online');await mc.locator('[data-window-expand]').tap();
   const r=await mc.boundingBox();assert.ok(r.x>=0&&r.x+r.width<=width+1);assert.equal(await m.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
   if(width===390){await mc.screenshot({path:path.join(task,'validation/compiler_attention_cards_mobile.png')});screenshots.push('compiler_attention_cards_mobile.png');}
   await mobile.close();
  }
  checks.push('attention: 320/390px trees scroll internally and expanded cards remain inside the mobile viewport');
  const live=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'no-preference'}),lp=await live.newPage();lp.on('pageerror',e=>errors.push(e.message));await lp.goto(url('attention'));await lp.locator('#toolchain-callstack-tab').click();
  const lc=await openCard(lp,'toolchain-call-graph-custom-insert');await lc.locator('[data-window-expand]').click();const lm=lc.locator('[data-attention-mechanism]');
  await lm.locator('[data-am-speed]').selectOption('1600');
  await lm.locator('[data-am-action="step"]').click();await lm.locator('[data-am-action="play"]').click();await lp.waitForTimeout(110);await lm.locator('[data-am-action="play"]').click();
  const pose=()=>lm.locator('.attention-changed').first().evaluate(el=>getComputedStyle(el).transform),paused=await pose(),pausedIndex=(await visibleFrame(lc)).index;
  await lp.waitForTimeout(230);assert.equal(await pose(),paused);assert.equal((await visibleFrame(lc)).index,pausedIndex);assert.equal((await visibleFrame(lc)).playing,false);
  await lm.locator('[data-am-action="play"]').click();await lp.waitForTimeout(110);assert.notEqual(await pose(),paused);await lp.waitForTimeout(1650);assert.ok((await visibleFrame(lc)).index>pausedIndex);
  await closeCard(lp);await lp.waitForTimeout(60);assert.equal(await lc.isVisible(),false);await live.close();
  checks.push('attention: Play/Pause freezes/resumes field-transform motion and semantic time; hidden cards stop playback');
  const plain=await browser.newContext({javaScriptEnabled:false,viewport:{width:1200,height:1000}}),np=await plain.newPage();await np.goto(url('attention'));
  assert.equal(await np.locator('#toolchain-callstack').isVisible(),true);const offline=np.locator('#toolchain-call-kernel-online');await offline.locator(':scope > summary').click();assert.match(await offline.textContent(),/alpha\*old_l/);assert.equal(await offline.locator('[data-am-action="play"]').isVisible(),false);await plain.close();
  checks.push('attention: source trees, native explanation disclosures and portable transcripts remain readable without JavaScript');
  const printContext=await browser.newContext({viewport:{width:1200,height:1000},reducedMotion:'reduce'}),printPage=await printContext.newPage();await printPage.goto(url('attention'));await printPage.emulateMedia({media:'print'});
  const printCard=printPage.locator('#toolchain-call-kernel-online'),printTranscript=printCard.locator('.attention-motion-transcript');assert.equal(await printCard.isVisible(),true);assert.equal(await printTranscript.locator('li').first().isVisible(),true);assert.equal(await printCard.locator('[data-am-action="play"]').isVisible(),false);assert.equal(await printCard.locator('[data-call-tree-id]').isVisible(),true);await printContext.close();
  checks.push('attention: print exposes source trees and portable per-step transcripts while hiding interactive controls');
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  const report={result:'PASS',checks,errors,externalRequests:requests.length,screenshots,evidence:'Browser/source illustrations only; no MAX/Mojo build or GPU workload'};
  fs.writeFileSync(path.join(task,'validation/compiler_example_readers_browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
