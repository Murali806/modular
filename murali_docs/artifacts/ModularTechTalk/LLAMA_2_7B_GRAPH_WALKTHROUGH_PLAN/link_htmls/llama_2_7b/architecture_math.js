/* Full-size tensor viewports; arithmetic/state are supplied by architecture_tensor.js. */
(function(){
  'use strict';
  const $=id=>document.getElementById(id),root=$('math-animation');if(!root)return;
  const model=window.LlamaTensor,{spec,events,definitions}=model,sources=JSON.parse($('math-source-data').textContent);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=n=>Number(n).toLocaleString('en-US'),f=(n,d=5)=>n===-Infinity?'−∞':!Number.isFinite(n)?'—':Number(n.toFixed(d)).toString();
  const size=n=>n>=2**20?f(n/2**20,3)+' MiB':n>=1024?f(n/1024,3)+' KiB':num(n)+' B';
  const shape=t=>'['+t.shape.join(', ')+']',windows=window.CompilerCardWindows,reduce=matchMedia('(prefers-reduced-motion: reduce)');
  let pass='prefill',eventIndex=0,layer=0,head=0,row=0,col=0,step=0,timer=null,op,probe,opKey='',probeKey='',cachePos=0,cacheChannel=0,onlineIndex=0,onlineKey='',onlineData;
  const views=new Map(),pageViews=[],layerViews=[],bufferViews=new Map();
  const colors={weights:'#6c49a5',residual:'#096649',norm:'#08737d',qkv:'#9b5016',attention:'#a52262',cache:'#245b97'};
  const bufferInfo={weights:['E / W / γ','Model lifetime · 32 sets of block weights + untied head'],residual:['X / H','Current pass · preserve until residual add'],norm:['N / G / U / Z','Current pass · norm and FFN scratch'],qkv:['Q′ / K′ / V','Current pass · projections; Q′ stays scratch'],attention:['S / P / context','S/P are mathematical views of fused attention'],cache:['K′ / V pages','Request lifetime · 32 separate layer slices']};
  const choices=(el,n)=>{el.innerHTML=Array.from({length:n},(_,i)=>`<option value="${i}">${i}</option>`).join('');};
  choices($('math-layer'),32);choices($('math-head'),32);
  $('math-stage').innerHTML=Object.entries(definitions).map(([id,[title]])=>`<option value="${id}">${esc(title)}</option>`).join('');
  $('math-pipeline').innerHTML=Object.entries(definitions).map(([id,[title,,label]])=>`<button data-stage="${id}" title="${esc(title)}">${esc(label)}</button>`).join('');
  $('math-pipeline').addEventListener('click',e=>{const b=e.target.closest('[data-stage]');if(b)selectStage(b.dataset.stage);});
  for(let l=0;l<32;l++){
    const b=document.createElement('button');b.className='layer-chip';b.dataset.layer=l;b.innerHTML=`<b>${l}</b><small>0</small>`;
    b.onclick=()=>selectLayer(l);$('math-layers').append(b);layerViews.push(b);
  }
  for(const [id,[name,owner]]of Object.entries(bufferInfo)){
    const el=document.createElement('article');el.className='buffer-card';el.dataset.buffer=id;el.style.setProperty('--buffer-color',colors[id]);
    el.innerHTML=`<b>${name}</b><small>${owner}</small><div class="buffer-value"></div><small class="buffer-status"></small>`;
    $('math-buffer-shelf').append(el);bufferViews.set(id,el);
  }
  spec.pages.forEach((physical,logical)=>{
    const el=document.createElement('article');el.className='full-cache-page';el.dataset.physicalPage=physical;
    el.innerHTML=`<h4>Page ${physical} <small>logical ${logical}</small></h4><div class="page-range">positions ${logical*128}…${logical*128+127}</div><div class="full-page-slots"></div><p class="page-count"></p>`;
    const slots=Array.from({length:128},(_,slot)=>{
      const b=document.createElement('button');b.className='cache-slot';b.dataset.position=logical*128+slot;b.dataset.slot=slot;
      b.onclick=()=>{stop();cachePos=logical*128+slot;renderCache();};el.querySelector('.full-page-slots').append(b);return b;
    });
    $('math-cache-pages').append(el);pageViews.push({el,slots,logical,physical});
  });
  function buildView(id,role){
    const el=$(id);el.innerHTML='<h3></h3><div class="tensor-shape"></div><p class="matrix-subtitle"></p><svg class="tensor-overview" viewBox="0 0 220 54" preserveAspectRatio="none" role="img"><rect x="1" y="1" width="218" height="52" class="extent"/><path d="M55 1V53M110 1V53M165 1V53M1 18H219M1 36H219" class="overview-grid"/><path class="overview-mask"/><rect class="overview-row"/><rect class="overview-column"/><rect class="overview-tile"/></svg><div class="viewport-caption"></div><div class="matrix-scroll"><table class="matrix"><thead><tr><th></th></tr></thead><tbody></tbody></table></div><p class="tensor-foot"></p>';
    const table=el.querySelector('table'),headers=Array.from({length:5},()=>{const th=document.createElement('th');th.scope='col';table.querySelector('thead tr').append(th);return th;});
    const cells=Array.from({length:3},()=>{const tr=document.createElement('tr'),label=document.createElement('th');label.scope='row';tr.append(label);const cells=Array.from({length:5},()=>{const td=document.createElement('td');if(role==='result'){const b=document.createElement('button');b.type='button';b.onclick=()=>{stop();row=Number(td.dataset.row);col=Number(td.dataset.col);probeKey='';render();};td.append(b);}tr.append(td);return td;});table.querySelector('tbody').append(tr);return {tr,label,cells};});
    const svg=el.querySelector('svg');svg.addEventListener('click',e=>{
      const b=svg.getBoundingClientRect(),t=op[role],rr=Math.min(t.rows-1,Math.floor((e.clientY-b.y)/b.height*t.rows)),cc=Math.min(t.cols-1,Math.floor((e.clientX-b.x)/b.width*t.cols));stop();
      if(role==='result'){row=rr;col=cc;}
      else if(op.kind==='dot'){if(role==='left'){row=rr;step=cc+1;}else{step=rr+1;col=cc;}}
      else return;
      probeKey='';render();
    });
    views.set(id,{el,table,headers,cells,svg,role});
  }
  buildView('math-left','left');buildView('math-right','right');buildView('math-result','result');
  function loadOperation(){
    const e=events[eventIndex];if(e.stage==='merge')head=Math.floor(col/128);
    const key=[pass,eventIndex,head,$('math-rope-kind').value].join(':');
    if(key!==opKey){op=model.operation(pass,e,head,$('math-rope-kind').value);opKey=key;probeKey='';}
    row=Math.max(0,Math.min(row,op.result.rows-1));col=Math.max(0,Math.min(col,op.result.cols-1));step=Math.max(0,Math.min(step,op.steps));op.query=row;op.mergeHead=Math.floor(col/128);
    const pk=key+':'+row+':'+col;if(pk!==probeKey){probe=model.probe(op,row,col);probeKey=pk;}
  }
  const begin=(at,total,width)=>Math.max(0,Math.min(Math.floor(at/width)*width,total-width));
  function drawMatrix(id){
    const view=views.get(id),{el,role}=view,t=op[role],k=Math.max(0,Math.min(step-1,op.inner-1));let r=row,c=col;
    if(role==='left'&&op.kind==='dot')c=k;
    else if(role==='right'&&op.kind==='dot')r=k;
    else if(role==='left'&&op.kind==='norm')c=k;
    else if(role==='right'&&op.kind==='norm')r=0;
    else if(role==='right'&&op.stage==='embedding'){r=model.token(op.C+row);c=col;}
    else if(role==='right'&&op.stage==='rope'){r=col%64;c=0;}
    else if(role==='right'&&op.stage==='merge'){r=Math.floor(col/128);c=col%128;}
    else if(role==='left'&&op.stage==='merge'){r=row;c=col%128;}
    else if(role==='left'&&op.stage==='gather')r=op.T-1;
    else if(op.kind==='argmax'){r=0;c=role==='result'?0:Math.max(0,step-1);}
    else if(op.kind==='store'&&role!=='result'){r=Math.max(0,Math.min(op.T-1,row-op.C));c=col;}
    r=Math.min(r,t.rows-1);c=Math.min(c,t.cols-1);
    const r0=begin(r,t.rows,3),c0=begin(c,t.cols,5),nr=Math.min(3,t.rows),nc=Math.min(5,t.cols);
    el.dataset.tensor=t.name;el.style.setProperty('--buffer-color',role==='result'?colors.residual:role==='right'?colors.weights:colors.cache);
    el.querySelector('h3').textContent=t.name;el.querySelector('.tensor-shape').textContent=shape(t);
    el.querySelector('.matrix-subtitle').textContent=t.axes+(t.shape.length===3?` · 2D slice [${t.rows}, ${t.cols}]`:'');
    el.querySelector('.viewport-caption').textContent=`rows ${r0}…${r0+nr-1} / ${t.rows} · cols ${c0}…${c0+nc-1} / ${t.cols}`;
    el.querySelector('.tensor-foot').textContent=role==='result'?(op.kind==='store'?'Live cache slice; unwritten rows have no valid values.':'Selected cell is pending until complete; other cells are references.'):op.kind==='dot'?'Click overview to choose another row/column or reduction index.':'Viewport values use the absolute indices above.';
    view.svg.setAttribute('aria-label',`${t.name}, full shape ${shape(t)}, viewport row ${r0}, column ${c0}. Overview geometry is schematic.`);
    const rect=(selector,x,y,w,h)=>{const n=el.querySelector(selector);for(const [key,val]of Object.entries({x,y,width:w,height:h}))n.setAttribute(key,val);};
    const mask=el.querySelector('.overview-mask'),masked=(op.stage==='softmax'||op.stage==='qk')&&(role==='result'||op.stage==='softmax');
    mask.setAttribute('d',masked?`M${1+218*(op.C+1)/op.L} 1 H219 V53 L${1+218*(op.C+op.T)/op.L} 53 Z`:'');
    rect('.overview-row',1,1+52*r/t.rows,218,Math.max(1,52/t.rows));rect('.overview-column',1+218*c/t.cols,1,Math.max(1,218/t.cols),52);
    rect('.overview-tile',1+218*c0/t.cols,1+52*r0/t.rows,Math.max(2,218*nc/t.cols),Math.max(2,52*nr/t.rows));
    view.headers.forEach((h,j)=>{h.hidden=j>=nc;h.textContent=c0+j;});
    view.cells.forEach((vr,i)=>{
      vr.tr.hidden=i>=nr;vr.label.textContent=r0+i;
      vr.cells.forEach((td,j)=>{
        td.hidden=j>=nc;if(i>=nr||j>=nc){td.classList.remove('target','hot','masked');delete td.dataset.row;delete td.dataset.col;return;}const ri=r0+i,cj=c0+j,selected=role==='result'&&ri===row&&cj===col;
        const pending=role==='result'&&(op.kind==='store'?ri>=op.C+step:selected&&step<op.steps),value=t.get(ri,cj);
        td.dataset.row=ri;td.dataset.col=cj;td.classList.toggle('target',selected);
        td.classList.toggle('hot',role!=='result'&&ri===r&&cj===c&&step>0);td.classList.toggle('masked',value===-Infinity||(op.stage==='softmax'&&cj>op.C+ri));
        const node=role==='result'?td.firstElementChild:td;node.textContent=pending?'…':f(value,3);
        td.title=`${t.name}[${ri}, ${cj}] = ${f(value,10)}${pending?' (reference; selected output pending)':''}`;
        if(role==='result')node.setAttribute('aria-label',`Inspect row ${ri}, column ${cj}: ${pending?'pending':f(value,6)}`);
      });
    });
  }
  function equation(){
    const done=step===op.steps,k=Math.max(0,step-1),prefix=probe.prefix,expected=probe.expected;let body='',title='Phases',note='Step follows the selected operation; animation speed is not hardware time.';
    if(op.kind==='dot'){
      title=`Complete reduction · k = 0…${op.inner-1}`;const start=begin(Math.max(0,step-1),op.inner,4),end=Math.min(op.inner,start+4);
      body=`<b>Output [${row}, ${col}] · ${num(step)} / ${num(op.inner)} products included</b><div class="dot-terms">`;
      for(let j=start;j<end;j++)body+=`<span class="${j<step?'done':''} ${j===step-1?'current':''}">k=${j}: ${f(op.left.get(row,j),4)} × ${f(op.right.get(j,col),4)} = ${f(op.left.get(row,j)*op.right.get(j,col),6)}</span>`;
      body+=`</div><div class="accumulator-path"><span>full row × full column</span><b>→ Σ<sub>k&lt;${step}</sub></b><strong>${f(prefix[step],9)}</strong><b>${op.scale===1?'→':'÷ √128 →'}</b><span>${done?'WRITE '+f(expected,9):'output pending'}</span></div><small>${done?'All terms have contributed.':'The accumulator includes every earlier term, including terms outside the viewport.'} Completed reference = ${f(expected,9)}.</small>`;
      note=`Slider addresses every term. A visible tile never changes the reduction extent: ${num(op.inner)}. ${op.stage==='qk'&&col>op.C+row?'This future score will be masked at softmax.':''}`;
    }else if(op.kind==='norm'){
      title='RMSNorm · sum all 4096 squared channels';body=`<b>Row ${row} · ${num(step)} / 4096 channels included</b><p>Σx² so far = <b>${f(prefix[step],9)}</b> · complete mean(x²) = ${f(op.mean(row),9)}</p><code>y[${row},${col}] = ${f(op.left.get(row,col),6)} × ${f(op.right.get(0,col),6)} / √(${f(op.mean(row),9)} + 0.00001) = ${f(expected,9)}</code><p>${done?'WRITE normalized channel.':'Output waits for the complete row reduction.'}</p>`;
    }else if(op.kind==='softmax'){
      const a=op.attention.row(row);title=`Stable softmax · key axis 0…${op.L-1}`;
      body=`<b>Query row ${row} → absolute position ${op.C+row}; allowed keys 0…${op.C+row}</b><p>1. Maximum over allowed keys = ${f(a.max,9)}. 2. Accumulate exp(S−max): ${step}/${op.L} keys, sum = <b>${f(prefix[step],9)}</b>.</p><code>3. P[${row},${col}] = ${f(op.right.get(row,col),9)} / ${f(a.denom,9)} = ${f(expected,9)}</code><p>${col>op.C+row?'Selected key is in the future: exp(−∞) = 0. ':''}${done?'Complete normalized row sums to 1.':'Normalization waits for the full denominator.'}</p>`;
    }else if(op.kind==='rope'){
      const pair=col%64,pos=op.C+row,angle=pos*Math.pow(10000,-2*pair/128),a=op.left.get(row,pair),b=op.left.get(row,pair+64),c=Math.cos(angle),s=Math.sin(angle);
      title='RoPE · read pair → angle → products → rotate';
      body=`<b>${op.ropeKind}, layer ${layer}, head ${head}, absolute position ${pos} · pair (${pair}, ${pair+64})</b><div class="rope-pair"><span class="${step>=1?'done':''}">① [${f(a)}, ${f(b)}]</span><span class="${step>=2?'done':''}">② θ=${pos} × 10000^(−${2*pair}/128) = ${f(angle,8)}<br>cos=${f(c)} · sin=${f(s)}</span><span class="${step>=3?'done':''}">③ x·cos − y·sin<br>x·sin + y·cos</span><span class="${step>=4?'done':''}">④ [${f(a*c-b*s)}, ${f(a*s+b*c)}]</span></div><small>Squared pair norm ${f(a*a+b*b,8)} is preserved. V is unchanged. All 64 pairs follow this rule.</small>`;
    }else if(op.kind==='store'){
      title=`Append active rows · ${op.T} new positions in layer ${layer}`;
      body=`<b>${step} / ${op.T} token rows stored · K′ and V, 32 heads × 128 channels each</b><p>New range: ${op.C}…${op.L-1}. Prefix ${op.C?'0…'+(op.C-1)+' is retained':'is empty'}. The page grid below fills as Step advances.</p><code>this layer: ${num(step)} × 2(K/V) × 32 × 128 × 2 bytes = ${size(step*16384)} written</code>`;
    }else if(op.kind==='argmax'){
      title='Greedy selection · all 32000 vocabulary logits';body=`<b>${num(step)} / 32000 logits compared</b><p>Current largest logit = ${f(prefix[step],8)} at ID ${step?probe.ids[step]:'—'}. ${done?'Final illustrative ID: '+expected: 'Continue the full vocabulary scan.'}</p><p>The next forward must process the selected ID before it has K/V. These logits are an independent generated sampling probe.</p>`;
    }else if(op.kind==='silu'||op.kind==='add'){
      const a=op.left.get(row,col),b=op.right.get(row,col);title=op.kind==='silu'?'Read gate/up → elementwise SiLU × up':'Read residual/update → elementwise add';
      body=`<b>Matching channel [${row}, ${col}]</b><p><code>${op.kind==='silu'?`${f(a,7)} / (1 + exp(−${f(a,7)})) × ${f(b,7)}`:`${f(a,7)} + ${f(b,7)}`} = ${f(expected,9)}</code></p><p>${done?'WRITE this channel.':'Operands ready; Step computes the channel.'} No reduction across channels.</p>`;
    }else if(op.stage==='merge')body=`<b>Head ${Math.floor(col/128)}, channel ${col%128} → hidden channel ${col}</b><p><code>hidden = head × 128 + channel = ${Math.floor(col/128)} × 128 + ${col%128}</code></p>All 32 heads contribute to [${op.T},4096]. This operation rearranges coordinates; it performs no MACs.`;
    else if(op.stage==='embedding')body=`<b>Token row ${row}: ID ${model.token(op.C+row)} selects one of 32000 embedding rows.</b><p><code>X[${row},${col}] = E[${model.token(op.C+row)},${col}] = ${f(expected,9)}</code></p>Copy 4096 channels per selected token; the complete embedding table is [32000,4096].`;
    else body=`<b>Gather row ${op.T-1} after layer 31</b><p><code>last[0,${col}] = final_hidden[${op.T-1},${col}] = ${f(expected,9)}</code></p>Only this row enters final norm and the independent [32000,4096] head weight.`;
    $('math-equation').innerHTML=body;$('math-reduction-title').textContent=title;$('math-reduction-note').textContent=note;
    $('math-progress').textContent=`${num(step)} / ${num(op.steps)}${op.inner?' terms':op.kind==='store'?' rows':' phases'}`;
    $('math-seek').max=op.steps;$('math-seek').value=step;
    const count=Math.min(32,op.steps),bands=$('math-reduction-bands');
    if(bands.children.length!==count)bands.innerHTML=Array.from({length:count},()=>'<i></i>').join('');
    [...bands.children].forEach((n,i)=>{const begin=i*op.steps/count,end=(i+1)*op.steps/count;n.style.setProperty('--fill',Math.max(0,Math.min(1,(step-begin)/(end-begin)))*100+'%');n.title=`extent ${Math.floor(begin)}…${Math.ceil(end)-1}`;});
  }
  function renderCache(){
    const state=model.cacheState(pass,events[eventIndex],step),selectedLayer=events[eventIndex].layer??layer,valid=state.lengths[selectedLayer],r=model.route(pass);
    cachePos=Math.min(cachePos,state.pages*128-1);$('math-cache-position').max=state.pages*128-1;$('math-cache-position').value=cachePos;$('math-cache-channel').value=cacheChannel;
    layerViews.forEach((el,l)=>{el.querySelector('small').textContent=state.lengths[l];el.classList.toggle('active',l===selectedLayer);el.dataset.valid=state.lengths[l];el.setAttribute('aria-label',`Layer ${l}: ${state.lengths[l]} valid KV positions`);el.setAttribute('aria-pressed',String(l===selectedLayer));});
    for(const page of pageViews){
      page.el.hidden=page.logical>=state.pages;let old=0,fresh=0;
      page.slots.forEach((el,slot)=>{
        const pos=page.logical*128+slot,stored=pos<valid,retained=stored&&pos<r.C;
        const written=op.stage==='kv'&&stored&&pos>=r.C&&pos>=r.C+Math.max(0,step-Number($('math-chunk').value));
        const status=!stored?'empty':written?'write':retained?'retained':'new';el.dataset.state=status;el.classList.toggle('selected',pos===cachePos);
        el.title=`Layer ${selectedLayer}, position ${pos}, slot ${slot}: ${status}`;el.setAttribute('aria-label',el.title);if(stored){if(retained)old++;else fresh++;}
      });
      page.el.querySelector('.page-count').textContent=`${old} retained · ${fresh} new · ${128-old-fresh} empty`;
    }
    $('math-cache-state').innerHTML=`<b>Layer ${selectedLayer}: ${valid} valid positions</b> · KV prefix stored in every layer: ${state.completed}. Current valid payload across layers: <b>${size(state.used)}</b>; assigned ${state.pages} × 64 MiB = <b>${size(state.assigned)}</b>. <span class="kv-key">Blue = retained · amber = new · outline = empty.</span> Each slot stores both K′ and V for 32 heads; grid shows the selected layer. 2 bytes/value; physical IDs are illustrative.`;
    const a=model.address(cachePos,selectedLayer,head,cacheChannel,0),v=model.address(cachePos,selectedLayer,head,cacheChannel,1),stored=cachePos<valid,start=begin(cacheChannel,128,6),indices=Array.from({length:6},(_,i)=>start+i);
    $('math-cache-vector').innerHTML=`<div><b>position ${cachePos} → page ${a.physical}, slot ${a.slot}</b><br><small>layer ${selectedLayer}, head ${head} · channels ${start}…${start+5} of 128 · ${stored?'stored':'unwritten'}</small></div><div><b>K′</b> <code>${stored?indices.map(c=>f(model.rotated('K',selectedLayer,head,cachePos,c),5)).join(' | '):'— — — — — —'}</code><br><b>V&nbsp;</b> <code>${stored?indices.map(c=>f(model.raw('V',selectedLayer,head,cachePos,c),5)).join(' | '):'— — — — — —'}</code></div>`;
    $('math-cache-address').innerHTML=`<b>Selected coordinate: position ${cachePos}, layer ${selectedLayer}, head ${head}, channel ${cacheChannel}</b><p><code>logical = ${cachePos} // 128 = ${a.logical}; lookup[${a.logical}] = ${a.physical}; slot = ${a.slot}<br>page stride = 2 × 32 × 128 × 32 × 128 = ${num(a.stride)} elements<br>within K = ((((0 × 32 + ${selectedLayer}) × 128 + ${a.slot}) × 32 + ${head}) × 128 + ${cacheChannel}) = ${num(a.within)}<br>K byte offset = (${a.physical} × ${num(a.stride)} + ${num(a.within)}) × 2 = ${num(a.byte)}<br>V uses kv=1: byte offset = ${num(v.byte)}</code></p>Offsets are relative to the illustrated backing buffer; page holes do not count as this request’s assigned capacity. Unwritten capacity has no valid K/V yet.`;
  }
  function buffers(){
    const s=op.stage,stage=model.block.indexOf(s),kv=model.block.indexOf('kv');
    const values={weights:'Persistent E + 32 blocks + W_head; γ per norm',residual:`[${op.T},4096] · ${['norm2','gate','up','swiglu','down','add2'].includes(s)?'H':'X'}`,norm:['gate','up','swiglu','down'].includes(s)?`G/U/Z [${op.T},11008]`:`N [${op.T},4096]`,qkv:`each [${op.T},32,128] · ${stage>kv?'K′/V retained in pages':'K′/V awaiting storage'}`,attention:`S/P conceptual [32,${op.T},${op.L}]; context [${op.T},4096]`,cache:`[pages,2,32,128,32,128] · 64 MiB/page`};
    const reads={embedding:['weights'],norm1:['weights','residual'],q:['weights','norm'],k:['weights','norm'],v:['weights','norm'],rope:['qkv'],kv:['qkv'],qk:['qkv','cache'],softmax:['attention'],pv:['attention','cache'],merge:['attention'],o:['attention','weights'],add1:['residual','attention'],norm2:['residual','weights'],gate:['norm','weights'],up:['norm','weights'],swiglu:['norm'],down:['norm','weights'],add2:['residual','norm'],gather:['residual'],finalnorm:['residual','weights'],head:['norm','weights'],sample:[]};
    const writes={embedding:'residual',norm1:'norm',q:'qkv',k:'qkv',v:'qkv',rope:'qkv',kv:'cache',qk:'attention',softmax:'attention',pv:'attention',merge:'attention',o:'attention',add1:'residual',norm2:'norm',gate:'norm',up:'norm',swiglu:'norm',down:'norm',add2:'residual',gather:'residual',finalnorm:'norm'};
    const live={weights:true,cache:true,residual:s!=='sample',norm:['norm1','q','k','v','norm2','gate','up','swiglu','down','add2','finalnorm','head'].includes(s),qkv:stage>=1&&stage<=8,attention:stage>=6&&stage<=11};
    for(const [id,el]of bufferViews){const read=reads[s]?.includes(id),write=writes[s]===id;el.dataset.state=write?'write':read?'read':live[id]?'retained':'empty';el.querySelector('.buffer-value').textContent=live[id]||write?values[id]:'not live in this mathematical schedule';el.querySelector('.buffer-status').textContent=write?(read?'READ → ':'')+'WRITE':read?'READ':live[id]?'RETAINED':'EMPTY / RELEASED';}
  }
  function online(){
    if(!$('math-online').open)return;
    const r=model.route(pass),query=Math.max(0,Math.min(r.T-1,Math.floor(Number($('online-query').value)||0))),l=events[eventIndex].layer??layer;
    $('online-query').max=r.T-1;$('online-query').value=query;
    const key=[pass,l,head,query].join(':');if(key!==onlineKey){onlineData=model.online(model.attention(pass,l,head),query);onlineKey=key;onlineIndex=0;}
    onlineIndex=Math.max(0,Math.min(onlineIndex,onlineData.length-1));const t=onlineData[onlineIndex],channel=cacheChannel,ref=onlineData.at(-1).output[channel];
    $('online-progress').textContent=`tile ${onlineIndex+1}/${onlineData.length} · keys ${t.start}…${t.end-1}`;
    $('online-back').disabled=onlineIndex===0;$('online-step').disabled=onlineIndex===onlineData.length-1;
    $('online-result').innerHTML=`<div class="online-grid"><article><b>Read Q [1,128] and K/V [${t.end-t.start},128]</b><p>layer ${l}, head ${head}, query ${query}; allowed keys 0…${r.C+query}. Tile keys ${t.start}…${t.end-1}.</p><div class="online-tile-map">${onlineData.map((x,i)=>`<span class="${i<=onlineIndex?'done':''}">${x.start}…${x.end-1}</span>`).join('')}</div><p>Before: m=${f(t.old.m,8)}, ℓ=${f(t.old.l,8)}<br>After: m′=${f(t.m,8)}, ℓ′=${f(t.l,8)}<br>α = exp(old m − m′) = ${f(t.alpha,8)}</p></article><article><b>128-channel numerator · inspect channel ${channel}</b><p>old u[${channel}] = ${f(t.old.u[channel],9)}<br>u′ = α·u + Σ exp(S−m′)·V = ${f(t.u[channel],9)}<br>u′/ℓ′ = <b>${f(t.output[channel],9)}</b></p><p>Complete attention reference: ${f(ref,9)}. ${onlineIndex===onlineData.length-1?'All tiles included.':'Continue to include remaining key tiles.'}</p><small>m and ℓ are scalars; u has 128 entries. The inspector computes every channel.</small></article></div>`;
  }
  const diagram=$('model-map');
  diagram.addEventListener('click',e=>{
    const block=e.target.closest('[data-model-stage]');
    if(!block)return;
    e.preventDefault();selectStage(block.dataset.modelStage);
    $('math-event').scrollIntoView({block:'start'});
  });
  document.querySelector('.model-return').addEventListener('click',()=>{
    $('model-overview').open=true;
  });
  function renderDiagram(){
    const r=model.route(pass),l=events[eventIndex].layer??layer,cache=model.cacheState(pass,events[eventIndex],step);
    const labels={
      active:`[${r.T},4096]`,input:`Xₗ [${r.T},4096]`,
      layer:`EXPANDED DECODER · layer ${l} of 0…31`,
      'heads-rope':`each [${r.T},32,128] · absolute positions C+i`,
      'cache-shape':`target each [${r.L},32,128]`,
      'cache-valid':`layer ${l} · ${cache.lengths[l]} / ${r.L} stored`,
      scores:`[32,${r.T},${r.L}]`,probabilities:`P [32,${r.T},${r.L}]`,
      'v-shape':`V [${r.T},32,128]`,'next-layer':l<31?`Next: block ${l+1}`:'Next: last-row gather',
      heads:`[${r.T},32,128]`,hidden:`H [${r.T},4096] → FFN input`,
      gate:`G [${r.T},11008]`,up:`U [${r.T},11008]`,
      output:`Xₗ₊₁ [${r.T},4096]`
    };
    for(const n of diagram.querySelectorAll('[data-model-text]')){
      const text=labels[n.dataset.modelText];if(text!==undefined&&n.textContent!==text)n.textContent=text;
    }
    for(const n of diagram.querySelectorAll('[data-model-stage]')){
      const active=n.dataset.modelStage===op.stage;n.classList.toggle('active',active);
      if(active)n.setAttribute('aria-current','step');else n.removeAttribute('aria-current');
    }
    for(const n of diagram.querySelectorAll('[data-model-group]'))n.classList.toggle('in-progress',model.block.includes(op.stage));
    for(const n of diagram.querySelectorAll('[data-model-edge]'))n.classList.toggle('active',n.dataset.modelEdge===op.stage);
    const context=`${pass==='prefill'?'Prefill':'Decode'} · T=${r.T} active, C=${r.C} cached, L=${r.L} after append`;
    if($('model-map-context').textContent!==context)$('model-map-context').textContent=context;
    const selected=`Selected: ${events[eventIndex].layer===null?'':`layer ${l} · `}${op.title}`;
    if($('model-map-selected').textContent!==selected)$('model-map-selected').textContent=selected;
  }
  function render(){
    loadOperation();const e=events[eventIndex];layer=e.layer??(e.stage==='embedding'?0:31);
    $('math-stage').value=e.stage;$('math-layer').value=layer;$('math-head').value=head;
    $('math-layer').disabled=e.layer===null;$('math-head').disabled=!['rope','kv','qk','softmax','pv','merge'].includes(e.stage);
    $('math-rope-control').hidden=e.stage!=='rope';
    for(const n of $('math-pipeline').children){n.classList.toggle('active',n.dataset.stage===e.stage);n.setAttribute('aria-pressed',String(n.dataset.stage===e.stage));}
    $('math-event').querySelector('strong').textContent=(e.layer===null?'':`Layer ${layer} / 31 · `)+op.title;
    $('math-event').querySelector('p').textContent=op.note;
    $('math-shape-equation').textContent=op.kind==='dot'?`[${op.left.rows}, ${op.inner}] × [${op.inner}, ${op.right.cols}] → [${op.result.rows}, ${op.result.cols}]${op.scale!==1?' ÷ √128':''}`:`${shape(op.left)} → ${shape(op.result)}`;
    $('math-operator').textContent=op.kind==='dot'?'×':op.kind==='add'?'+':op.kind==='silu'?'⊙':'→';
    $('math-row').max=op.result.rows-1;$('math-row').value=row;$('math-col').max=op.result.cols-1;$('math-col').value=col;
    $('math-selection').textContent=`Rows 0…${op.result.rows-1}; columns 0…${op.result.cols-1}. Overviews show the full extent; tables show a viewport.`;
    for(const id of views.keys())drawMatrix(id);
    equation();renderCache();buffers();online();renderDiagram();
    const src=sources[op.source];$('math-sources').innerHTML=`<div class="source-refs">SOURCE · <a class="source-link" href="sources.html?file=${src.file}&amp;line=${src.line}" data-source="${src.file}" data-line="${src.line}">${esc(src.label)} <small>L${src.line}</small></a> · viewport values are generated operation probes.</div>`;
    const costStage={merge:'pv',gather:'gather',swiglu:'swiglu'}[op.stage]||op.stage;
    $('math-cost-link').innerHTML=`<a href="architecture_${pass==='prefill'?'01_prefill':'02_decode'}.html#cost-${costStage}">MACs, bytes moved, bandwidth and arithmetic intensity for this stage ↗</a>${pass==='next'?' · Set cached length C=513 in the calculator.':''}`;
    $('math-back').disabled=eventIndex===0&&step===0;$('math-step').disabled=eventIndex===events.length-1&&step===op.steps;$('math-next').disabled=eventIndex===events.length-1;$('math-finish').disabled=step===op.steps;
  }
  function stop(){if(timer!==null)clearInterval(timer);timer=null;$('math-play').textContent='Play';$('math-play').setAttribute('aria-pressed','false');}
  function eventAt(stage,l=layer){return events.findIndex(e=>e.stage===stage&&(e.layer===null||e.layer===l));}
  function selectStage(stage){stop();eventIndex=eventAt(stage);step=0;row=stage==='kv'?model.route(pass).C:0;col=0;cachePos=model.route(pass).C;render();}
  function selectLayer(l){stop();layer=l;const s=events[eventIndex].layer===null?'norm1':events[eventIndex].stage;eventIndex=eventAt(s,l);step=0;render();}
  function advance(direction){
    const chunk=op.inner||op.kind==='store'?Number($('math-chunk').value):1;
    if(direction>0){if(step<op.steps)step=Math.min(op.steps,step+chunk);else if(eventIndex<events.length-1){eventIndex++;step=0;}}
    else if(step>0)step=Math.max(0,step-chunk);else if(eventIndex>0){eventIndex--;opKey='';loadOperation();step=op.steps;}
    if(events[eventIndex].stage==='kv')cachePos=op.C+Math.max(0,step-1);
    render();
  }
  $('math-step').onclick=()=>{stop();advance(1);};$('math-back').onclick=()=>{stop();advance(-1);};
  $('math-next').onclick=()=>{stop();eventIndex=Math.min(events.length-1,eventIndex+1);step=0;render();};
  $('math-finish').onclick=()=>{stop();step=op.steps;if(op.stage==='kv')cachePos=op.L-1;render();};
  $('math-reset').onclick=()=>{stop();eventIndex=0;layer=head=row=col=step=cacheChannel=onlineIndex=0;cachePos=model.route(pass).C;onlineKey='';$('online-query').value=model.route(pass).T-1;$('math-rope-kind').value='Q';$('math-chunk').value='128';render();};
  $('math-stage').onchange=()=>selectStage($('math-stage').value);$('math-layer').onchange=()=>selectLayer(Number($('math-layer').value));
  $('math-head').onchange=()=>{stop();head=Number($('math-head').value);if(op.stage==='merge')col=head*128+col%128;render();};
  $('math-pass').onchange=()=>{stop();pass=$('math-pass').value;eventIndex=0;layer=head=row=col=step=0;cachePos=model.route(pass).C;onlineKey='';$('online-query').value=model.route(pass).T-1;render();};
  $('math-seek').oninput=()=>{stop();step=Number($('math-seek').value);if(op.stage==='kv')cachePos=op.C+Math.max(0,step-1);render();};
  const input=(id,set,max)=>{$(id).onchange=()=>{stop();set(Math.max(0,Math.min(max(),Math.floor(Number($(id).value)||0))));render();};};
  input('math-row',v=>row=v,()=>op.result.rows-1);input('math-col',v=>col=v,()=>op.result.cols-1);
  input('math-cache-position',v=>cachePos=v,()=>Math.ceil(model.route(pass).L/128)*128-1);input('math-cache-channel',v=>cacheChannel=v,()=>127);
  $('math-last-row').onclick=()=>{stop();row=op.result.rows-1;render();};$('math-last-col').onclick=()=>{stop();col=op.result.cols-1;render();};
  $('math-rope-kind').onchange=()=>{stop();render();};$('math-chunk').onchange=()=>{stop();render();};$('math-speed').onchange=stop;
  $('math-play').onclick=()=>{
    if(timer!==null){stop();return;}if(eventIndex===events.length-1&&step===op.steps){eventIndex=0;step=0;render();}
    $('math-play').textContent='Pause';$('math-play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{advance(1);if(eventIndex===events.length-1&&step===op.steps)stop();},Number($('math-speed').value));
  };
  $('math-online').ontoggle=online;$('online-query').onchange=online;$('online-reset').onclick=()=>{onlineIndex=0;online();};$('online-back').onclick=()=>{onlineIndex--;online();};$('online-step').onclick=()=>{onlineIndex++;online();};
  let preview=false,dismiss;const card=$('detail-card');
  function detail(title,body,temporary=false){clearTimeout(dismiss);preview=temporary;windows?.opening(card);card.hidden=false;windows?.positioned(card);$('card-title').textContent=title;$('card-body').innerHTML=body;}
  const lesson=()=>detail('03 · Full dimensions, one operation at a time','<p>Choose any of 32 layers, 32 heads and all valid row/column coordinates. Large tensors are represented by their full extents and a bounded viewport; displayed indices are absolute.</p><p>Step advances the full reduction in a chosen chunk. Finish operation reveals the selected output. Other visible output cells are completed references. Click a matrix overview or enter coordinates to move. These are mathematical tiles, not GPU launch tiles.</p><p>Generated operands keep the page small and offline. Each operation is a separate probe. RoPE, KV, QKᵀ, softmax and P·V share their generated attention values. No checkpoint or full model inference is run.</p>',true);
  const closeLater=()=>{dismiss=setTimeout(()=>{if(preview&&!windows?.isHeld(card)){card.hidden=true;windows?.closed(card);}},250);};
  $('math-badge').onmouseenter=lesson;$('math-badge').onfocus=lesson;$('math-badge').onclick=lesson;$('math-badge').onmouseleave=closeLater;$('math-badge').onblur=closeLater;card.addEventListener('mouseenter',()=>clearTimeout(dismiss));card.addEventListener('mouseleave',closeLater);
  $('math-legend').onclick=()=>detail('Tensor roles and ownership','<p>Matrix panels: blue = left operand, purple = right operand, green = result. The buffer shelf below uses the component colors listed here.</p>'+Object.entries(bufferInfo).map(([id,[label,owner]])=>`<p><span class="legend-swatch" style="background:${colors[id]}"></span><b>${label}</b><br>${owner}</p>`).join('')+'<p>Amber means a new cache write; blue means retained prefix; outlined slots are empty. Thin overview bands mark the selected row/column. The green box marks the viewport. Matrix overview geometry is schematic; all coordinates and dimensions are literal.</p>');
  $('math-fullscreen').onclick=async()=>{if(document.fullscreenElement)await document.exitFullscreen();else await root.requestFullscreen();};
  document.addEventListener('fullscreenchange',()=>{$('math-fullscreen').textContent=document.fullscreenElement?'Exit full screen':'Full screen';});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe(root);
  window.addEventListener('beforeprint',stop);reduce.addEventListener('change',stop);
  window.LlamaMath={stop,snapshot:()=>({pass,eventIndex,totalEvents:events.length,stage:op.stage,layer,head,row,col,step,steps:op.steps,inner:op.inner,playing:timer!==null,shapes:{left:op.left.shape,right:op.right.shape,result:op.result.shape},partial:probe.prefix.length>1?probe.prefix[step]:null,expected:probe.expected,cache:model.cacheState(pass,events[eventIndex],step),cachePos,cacheChannel,onlineIndex})};
  render();
})();
