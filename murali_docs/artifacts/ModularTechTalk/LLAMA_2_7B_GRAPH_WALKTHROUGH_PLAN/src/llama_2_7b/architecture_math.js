/* Deterministic mathematical playback. State comes only from architecture_toy.js. */
(function(){
  'use strict';
  const $=id=>document.getElementById(id),root=$('math-animation');if(!root)return;
  const toy=window.LlamaToy,sources=JSON.parse($('math-source-data').textContent);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const f=(n,d=3)=>n===-Infinity?'−∞':!Number.isFinite(n)?'—':Math.abs(n)<.5*10**-d?'0':Number(n.toFixed(d)).toString();
  const colors={weights:'#6c49a5',residual:'#096649',norm:'#08737d',qkv:'#9b5016',q:'#9b5016',attention:'#a52262',context:'#a52262',ffn:'#08737d',logits:'#586976',token:'#245b97',cache:'#245b97'};
  const windows=window.CompilerCardWindows,reduce=matchMedia('(prefers-reduced-motion: reduce)');
  let pass='prefill',data=toy.route(pass),frames=toy.timeline(data),index=0,head=0,row=0,col=0,timer=null,onlineIndex=0,onlineKey='';
  const matrices=new Map(),bufferEls=new Map(),slotEls=new Map();
  for(const [id,[label,owner]]of Object.entries(toy.bufferInfo)){
    const el=document.createElement('article');el.className='buffer-card';el.dataset.buffer=id;el.style.setProperty('--buffer-color',colors[id]);
    el.innerHTML=`<b>${esc(label)}</b><small>${esc(owner)}</small><div class="buffer-value"></div><small class="buffer-status"></small>`;
    $('math-buffer-shelf').append(el);bufferEls.set(id,el);
  }
  const pages=document.createElement('div');pages.className='toy-pages';$('math-cache-pages').append(pages);
  toy.spec.pages.forEach((physical,logical)=>{
    const el=document.createElement('article');el.className='toy-page';el.dataset.physicalPage=physical;
    el.innerHTML=`<h4>Physical page ${physical}</h4><small>lookup[${logical}] = ${physical} · logical positions ${logical*2}, ${logical*2+1}</small>`;
    for(let layer=0;layer<2;layer++){
      const pane=document.createElement('div');pane.className='toy-page-layer';pane.innerHTML=`<div class="toy-layer">Layer ${layer} · selected head</div>`;
      for(let slot=0;slot<2;slot++){
        const position=logical*2+slot,cell=document.createElement('div');cell.className='toy-slot';cell.dataset.layer=layer;cell.dataset.position=position;cell.dataset.slot=slot;
        cell.innerHTML=`<span class="slot-position">p${position}<br>s${slot}</span><span class="slot-values"></span>`;
        pane.append(cell);slotEls.set(layer+':'+position,cell);
      }
      el.append(pane);
    }
    pages.append(el);
  });
  const pipeline=[['embedding','E[IDs]'],['norm1','RMS'],['q','Q/K/V'],['rope','RoPE'],['kv','KV store'],['qk','QKᵀ'],['softmax','softmax'],['pv','P·V'],['o','W_O'],['add1','+ X'],['norm2','RMS'],['gate','gate/up'],['swiglu','SiLU ⊙'],['down','down'],['add2','+ H'],['gather','last'],['finalnorm','norm'],['head','head'],['sample','ID']];
  $('math-pipeline').innerHTML=pipeline.map(([id,label])=>`<span data-stage="${id}">${esc(label)}</span>`).join('');
  function sourceLink(key){const r=sources[key];return `<a class="source-link" href="sources.html?file=${r.file}&amp;line=${r.line}" data-source="${r.file}" data-line="${r.line}">${esc(r.label)} <small>L${r.line}</small></a>`;}
  function stageOptions(){
    $('math-stage').innerHTML=data.events.map((e,i)=>`<option value="${i}">${e.layer===null?'':`Layer ${e.layer} · `}${esc(e.title)}</option>`).join('');
    $('math-seek').max=frames.length-1;
  }
  function chooseRange(el,length,value){
    if(el.options.length!==length)el.innerHTML=Array.from({length},(_,i)=>`<option value="${i}">${i}</option>`).join('');
    el.value=value;
  }
  function operands(event){return event.headMatrices?.[head]||event;}
  function partial(event,ops){
    if(event.kind!=='matmul')return null;
    const terms=ops.left.values[row].map((x,k)=>x*ops.right.values[k][col]);
    const sum=terms.slice(0,frames[index].term).reduce((s,x)=>s+x,0);
    return {terms,sum,value:sum*(event.scale||1),expected:ops.result.values[row][col]};
  }
  function drawMatrix(id,tensor,role,event,term,part){
    const el=$(id),a=tensor.values,m=a.length,n=a[0]?.length||0;
    let view=matrices.get(id);
    if(!view||view.m!==m||view.n!==n){
      el.innerHTML='<h3></h3><p class="matrix-subtitle"></p>';
      const table=document.createElement('table');table.className='matrix';table.setAttribute('aria-label',tensor.label);const body=document.createElement('tbody');
      const header=document.createElement('tr');header.innerHTML='<th></th>'+Array.from({length:n},(_,j)=>`<th scope="col">${j}</th>`).join('');body.append(header);
      const cells=Array.from({length:m},(_,i)=>{const tr=document.createElement('tr');tr.innerHTML=`<th scope="row">${i}</th>`;const cells=Array.from({length:n},(_,j)=>{const td=document.createElement('td');td.dataset.row=i;td.dataset.col=j;if(role==='result'){const b=document.createElement('button');b.type='button';b.onclick=()=>{stop();row=i;col=j;onlineIndex=0;render();};td.append(b);}tr.append(td);return td;});body.append(tr);return cells;});
      table.append(body);el.append(table);view={m,n,cells,table};matrices.set(id,view);
    }
    el.querySelector('h3').textContent=tensor.label;view.table.setAttribute('aria-label',tensor.label);
    el.querySelector('.matrix-subtitle').textContent=`[${m}, ${n}] · ${role==='result'&&part?'selected cell accumulates; other cells are reference values':role==='result'?'computed result':'operand'}`;
    el.dataset.tensor=tensor.label;
    el.style.setProperty('--buffer-color',role==='result'?'#096649':role==='right'&&event.kind==='matmul'?'#6c49a5':'#245b97');
    a.forEach((r,i)=>r.forEach((v,j)=>{
      const td=view.cells[i][j],selected=role==='result'&&i===row&&j===col;
      const active=event.kind==='matmul'&&term>0&&((role==='left'&&i===row&&j===term-1)||(role==='right'&&i===term-1&&j===col));
      td.classList.toggle('hot',active);td.classList.toggle('target',selected);td.classList.toggle('masked',!Number.isFinite(v)||(event.stage==='softmax'&&role==='result'&&j>event.C+i));
      const value=selected&&part?part.value:v;
      const content=role==='result'?td.firstElementChild:td;content.textContent=f(value);td.title=`${tensor.label}[${i},${j}] = ${f(v,10)}${selected&&part?`; partial = ${f(part.value,10)}`:''}`;
      if(role==='result')content.setAttribute('aria-label',`Inspect row ${i}, column ${j}: ${f(v,6)}`);
    }));
  }
  function realShape(stage){
    const T=pass==='prefill'?512:1,C=pass==='prefill'?0:pass==='decode'?512:513,L=T+C,A=`[${T},4096]`;
    const dims={embedding:`[${T}] → ${A}`,norm1:`${A} → ${A}`,q:`${A} × [4096,4096]`,k:`${A} × [4096,4096]`,v:`${A} × [4096,4096]`,rope:`Q,K each [${T},32,128]`,kv:`K,V each [${L},32,128] per layer`,qk:`32 heads: [${T},128] × [128,${L}]`,softmax:`[32,${T},${L}] conceptual probabilities`,pv:`32 heads: [${T},${L}] × [${L},128]`,merge:`[${T},32,128] → ${A}`,o:`${A} × [4096,4096]`,add1:`${A} + ${A}`,norm2:A,gate:`${A} × [4096,11008]`,up:`${A} × [4096,11008]`,swiglu:`[${T},11008] elementwise`,down:`[${T},11008] × [11008,4096]`,add2:`${A} + ${A}`,gather:`${A} → [1,4096]`,finalnorm:'[1,4096]',head:'[1,4096] × [4096,32000]',sample:'[1,32000] → one ID'};
    return 'Real-size shape reference: '+dims[stage]+'. Use the calculators for real-size costs.';
  }
  function equation(event,ops,part){
    const {term}=frames[index];let body='';
    if(part){
      const a=ops.left.values[row],b=ops.right.values;
      body=`<b>Output [${row},${col}] · ${term}/${a.length} multiply–accumulates exposed</b><div class="dot-terms">${a.map((v,k)=>`<span class="${k<term?'done':''} ${k===term-1?'current':''}">${f(v)} × ${f(b[k][col])} = ${f(part.terms[k],5)}</span>`).join('')}</div><code>acc = ${f(part.sum,8)}${event.scale?` → acc / √${toy.spec.d} = ${f(part.value,8)}`:''}; final reference = ${f(part.expected,8)}</code>`;
    }else if(event.kind==='norm'){
      const x=ops.left.values[row],gamma=ops.right.values[0][col],ss=x.reduce((n,x)=>n+x*x,0)/x.length;
      body=`<b>Normalize row ${row}, channel ${col}</b><br><code>mean(x²) = (${x.map(x=>f(x)+'²').join(' + ')}) / ${x.length} = ${f(ss,7)}<br>y = ${f(x[col])} × 1/√(${f(ss,7)} + 0.00001) × ${f(gamma)} = ${f(ops.result.values[row][col],8)}</code>`;
    }else if(event.kind==='softmax'){
      const v=ops.left.values[row],max=Math.max(...v),e=v.map(x=>Math.exp(x-max)),sum=e.reduce((s,x)=>s+x,0);
      body=`<b>Query row ${row}, absolute position ${event.C+row}</b><br><code>allowed keys: 0…${event.C+row}; max = ${f(max,7)}<br>exp(S−max) = [${e.map(x=>f(x,5)).join(', ')}]; sum = ${f(sum,8)}<br>P[${row},${col}] = ${f(e[col],8)} / ${f(sum,8)} = ${f(ops.result.values[row][col],8)}</code>`;
    }else if(event.kind==='silu'){
      const g=ops.left.values[row][col],u=ops.right.values[row][col];body=`<b>Elementwise gate at [${row},${col}]</b><br><code>SiLU(${f(g,6)}) × ${f(u,6)} = [${f(g,6)} / (1 + exp(−${f(g,6)}))] × ${f(u,6)} = ${f(ops.result.values[row][col],8)}</code>`;
    }else if(event.kind==='add')body=`<b>Residual at [${row},${col}]</b><br><code>${f(ops.left.values[row][col],7)} + ${f(ops.right.values[row][col],7)} = ${f(ops.result.values[row][col],8)}</code>`;
    else if(event.kind==='rope'){
      const c=ops.right.values[row][0],s=ops.right.values[row][1],[qx,qy,kx,ky]=ops.left.values[row];
      body=`<b>Head ${head} · absolute position ${event.C+row}</b><br><code>cos = ${f(c,6)}, sin = ${f(s,6)}<br>Q′ = [${f(qx)}·cos − ${f(qy)}·sin, ${f(qx)}·sin + ${f(qy)}·cos]<br>K′ = [${f(kx)}·cos − ${f(ky)}·sin, ${f(kx)}·sin + ${f(ky)}·cos]</code><br>Rotation preserves each pair’s squared length. V is unchanged.`;
    }else if(event.kind==='store')body=`<b>Write K′ and V into retained storage</b><br>Position ${event.written.position} → logical page ${Math.floor(event.written.position/2)} → physical page ${toy.spec.pages[Math.floor(event.written.position/2)]}, slot ${event.written.position%2}, layer ${event.layer}. Both vectors are stored; Q′ stays in current-pass scratch.`;
    else if(event.kind==='sample')body=`<b>argmax(logits) = ID ${data.next}</b><br>Valid cache lengths remain ${data.C+data.T} per layer. Select the next decode pass to process this ID and create its K/V.`;
    else if(event.stage==='embedding')body=`<b>Gather only the requested rows</b><br><code>X[${row},${col}] = E[token_ids[${row}]=${data.tokens[row]}, ${col}] = ${f(ops.result.values[row][col],8)}</code>`;
    else if(event.stage==='gather')body=`<b>Last active row = ${data.T-1}</b><br>The final norm and head operate on this row. Prior rows have already contributed through causal attention and their retained K/V.`;
    else body='<b>Concatenate the head channels</b><br>Head 0 supplies columns 0,1; head 1 supplies columns 2,3. Each token row keeps its identity.';
    $('math-equation').innerHTML=body;
  }
  function drawBuffers(event){
    for(const [id,el]of bufferEls){
      const b=event.buffers[id],reads=event.read.includes(id),writes=event.write.includes(id);
      el.dataset.state=writes?'write':reads?'read':b.occupied?'retained':'empty';
      el.querySelector('.buffer-value').textContent=b.value;
      el.querySelector('.buffer-status').textContent=(reads?'READ':'')+(reads&&writes?' → ':'')+(writes?'WRITE':'')+(!b.occupied&&reads?' · then released':!reads&&!writes?(b.occupied?'RETAINED':'EMPTY'):'');
    }
    for(const [key,cell]of slotEls){
      const [layer,pos]=key.split(':').map(Number),k=event.cache[layer].K[pos],v=event.cache[layer].V[pos];
      const written=event.written?.layer===layer&&event.written.position===pos;
      cell.dataset.state=!k?'empty':written?'write':pos<data.C?'old':'new';
      cell.querySelector('.slot-values').innerHTML=k?`<small>K′ [${k.slice(head*2,head*2+2).map(x=>f(x)).join(', ')}]</small><small>V &nbsp;[${v.slice(head*2,head*2+2).map(x=>f(x)).join(', ')}]</small>`:'empty · K′ / V unwritten';
    }
    for(const page of pages.children)page.dataset.used=String([...page.querySelectorAll('.toy-slot')].some(s=>s.dataset.state!=='empty'));
    const lengths=event.cache.map(c=>c.K.length),completed=Math.min(...lengths);
    $('math-cache-state').textContent=`Head ${head} shown. Layer 0: ${lengths[0]} valid positions; layer 1: ${lengths[1]}. ${event.layer===null?'':`Current block: layer ${event.layer}. `}All layers have processed at least ${completed} positions. Navy entries were retained before this pass; amber entries were written in this pass.`;
    const pos=event.written?.position??Math.min(data.C+Math.min(row,data.T-1),data.C+data.T-1),logical=Math.floor(pos/2),physical=toy.spec.pages[logical],slot=pos%2,layer=event.layer??1;
    const offset=((((0*2+layer)*2+slot)*2+head)*2),pageStride=2*2*2*2*2;
    $('math-cache-address').innerHTML=`<b>Address example · K′ channel 0</b><br><code>position ${pos} // 2 = logical page ${logical}; lookup[${logical}] = ${physical}; slot = ${pos}%2 = ${slot}<br>packed page stride = 2(K/V) × 2 layers × 2 slots × 2 heads × 2 channels = ${pageStride} elements<br>element offset = ${physical} × ${pageStride} + ((((0 × 2 + ${layer}) × 2 + ${slot}) × 2 + ${head}) × 2 + 0) = ${physical*pageStride+offset}</code><br>V uses kv=1. Multiply an element offset by the chosen payload bytes. Toy values are computed in browser FP64; real memory sizes belong to the cost pages.`;
  }
  function online(){
    const event=data.events[frames[index].event],layer=event.layer??toy.spec.N-1,ctx=data.contexts[layer],qr=Math.min(row,ctx.T-1),key=`${pass}:${layer}:${head}:${qr}`;
    if(key!==onlineKey){onlineIndex=0;onlineKey=key;}
    const result=toy.online(toy.head(ctx.q,head)[qr],toy.head(ctx.k,head),toy.head(ctx.v,head),ctx.C+qr,2);
    onlineIndex=Math.min(onlineIndex,result.tiles.length-1);const tile=result.tiles[onlineIndex];
    $('online-progress').textContent=`${onlineIndex+1} / ${result.tiles.length} · layer ${layer}, head ${head}, query ${qr}`;
    $('online-back').disabled=onlineIndex===0;$('online-step').disabled=onlineIndex===result.tiles.length-1;
    const vector=a=>'['+a.map(x=>f(x,7)).join(', ')+']';
    $('online-result').innerHTML=`<div class="online-grid"><article><b>Read key/value positions ${tile.start}…${tile.end-1}</b><pre>K = ${tile.k.map(vector).join('\n    ')}\nV = ${tile.v.map(vector).join('\n    ')}\nscores = ${vector(tile.scores)}\np = exp(scores − new_m) = ${vector(tile.p)}</pre></article><article><b>Retain only m, ℓ and vector u</b><pre>old m=${f(tile.old.m,7)}, ℓ=${f(tile.old.l,7)}, u=${vector(tile.old.u)}\nnew m=max(old m,tile max)=${f(tile.m,7)}\nα=exp(old m−new m)=${f(tile.alpha,7)}\nnew ℓ=α×old ℓ+Σp=${f(tile.l,7)}\nnew u=α×old u+ΣpV=${vector(tile.u)}\nu/ℓ=${vector(tile.output)}</pre></article></div><p><b>Full stable-softmax reference:</b> ${vector(result.full)}. ${onlineIndex===result.tiles.length-1?'Final online result matches to floating-point tolerance.':'This is an intermediate tile result; continue to include the remaining allowed keys.'} Positions greater than ${ctx.C+qr} are masked. The tile size 2 is a teaching choice.</p>`;
  }
  function render(){
    const frame=frames[index],event=data.events[frame.event],ops=operands(event);
    row=Math.min(row,ops.result.values.length-1);col=Math.min(col,(ops.result.values[0]?.length||1)-1);
    const part=partial(event,ops);
    $('math-stage').value=frame.event;$('math-seek').value=index;
    $('math-progress').textContent=`${index+1} / ${frames.length}`;$('math-back').disabled=index===0;$('math-step').disabled=index===frames.length-1;
    $('math-event').querySelector('strong').textContent=(event.layer===null?'':`Layer ${event.layer} · `)+event.title;
    $('math-event').querySelector('p').textContent=event.detail+' '+realShape(event.stage);
    const group={k:'q',v:'q',merge:'pv',up:'gate'}[event.stage]||event.stage;
    for(const n of $('math-pipeline').children)n.classList.toggle('active',n.dataset.stage===group);
    chooseRange($('math-row'),ops.result.values.length,row);chooseRange($('math-col'),ops.result.values[0].length,col);
    drawMatrix('math-left',ops.left,'left',event,frame.term,part);drawMatrix('math-right',ops.right,'right',event,frame.term,part);drawMatrix('math-result',ops.result,'result',event,frame.term,part);
    $('math-operator').textContent=event.kind==='matmul'?'×':event.kind==='add'?'+':event.kind==='silu'?'⊙':'→';
    equation(event,ops,part);drawBuffers(event);online();
    $('math-sources').innerHTML='<div class="source-refs">SOURCE · corresponding full-size operation: '+sourceLink(event.source)+'</div>';
  }
  function stop(){if(timer!==null)clearInterval(timer);timer=null;$('math-play').textContent='Play';$('math-play').setAttribute('aria-pressed','false');}
  $('math-step').onclick=()=>{stop();index=Math.min(index+1,frames.length-1);render();};
  $('math-back').onclick=()=>{stop();index=Math.max(0,index-1);render();};
  $('math-reset').onclick=()=>{stop();index=0;row=0;col=0;head=0;onlineIndex=0;$('math-head').value='0';render();};
  $('math-stage').onchange=()=>{stop();index=frames.findIndex(f=>f.event===Number($('math-stage').value));row=0;col=0;render();};
  $('math-seek').oninput=()=>{stop();index=Number($('math-seek').value);render();};
  $('math-pass').onchange=()=>{stop();pass=$('math-pass').value;data=toy.route(pass);frames=toy.timeline(data);index=0;row=0;col=0;onlineIndex=0;stageOptions();render();};
  $('math-head').onchange=()=>{head=Number($('math-head').value);onlineIndex=0;render();};
  $('math-row').onchange=()=>{row=Number($('math-row').value);onlineIndex=0;render();};
  $('math-col').onchange=()=>{col=Number($('math-col').value);render();};
  $('math-play').onclick=()=>{
    if(timer!==null){stop();return;}if(index===frames.length-1){index=0;render();}
    $('math-play').textContent='Pause';$('math-play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{index++;render();if(index===frames.length-1)stop();},Number($('math-speed').value));
  };
  $('math-speed').onchange=stop;
  $('online-reset').onclick=()=>{onlineIndex=0;online();};$('online-back').onclick=()=>{onlineIndex=Math.max(0,onlineIndex-1);online();};$('online-step').onclick=()=>{onlineIndex++;online();};
  let preview=false,dismiss;
  const card=$('detail-card');
  function detail(title,body,temporary=false){clearTimeout(dismiss);preview=temporary;windows?.opening(card);card.hidden=false;windows?.positioned(card);$('card-title').textContent=title;$('card-body').innerHTML=body;}
  const lesson=()=>detail('03 · Read one numerical operation',`<p>The tiny decoder computes its matrices from invented weights. Stage controls follow embedding, two full blocks and an untied output head. A matrix stage uses several events to expand one selected dot product; other cells show completed reference values.</p><p>Colors label buffer roles. K′/V page identities remain stable as later passes reuse them. Select a result cell, a head or another pass to change the numbers. The real-size shape reference and separate calculators use Llama 2 7B dimensions.</p><p>Step, Back, seek and Reset reproduce the same state. Play has no autoplay; reduced motion uses the same numerical updates.</p>`,true);
  const closeLater=()=>{dismiss=setTimeout(()=>{if(preview&&!windows?.isHeld(card)){card.hidden=true;windows?.closed(card);}},250);};
  $('math-badge').onmouseenter=lesson;$('math-badge').onfocus=lesson;$('math-badge').onclick=lesson;$('math-badge').onmouseleave=closeLater;$('math-badge').onblur=closeLater;card.addEventListener('mouseenter',()=>clearTimeout(dismiss));card.addEventListener('mouseleave',closeLater);
  $('math-legend').onclick=()=>detail('Buffer roles, ownership and lifetime',[
    ['weights','Weights','Model lifetime; E, two independent blocks and untied head. Reads reuse these tensors.'],
    ['residual','Residual','Current pass; X or H must survive until its residual add consumes it.'],
    ['norm','Norm / FFN','Current pass; normalization and gated expansion scratch.'],
    ['qkv','Q/K/V','New rows for this pass. Q′ is scratch; K′ and V are appended to cache.'],
    ['attention','Attention math','Conceptual S/P, context and update. MAX uses fused tiles instead of full S/P storage.'],
    ['cache','KV pages','Request lifetime, separate layer slices. Blue retained prefix, amber newly written, dashed empty.']
  ].map(([id,title,body])=>`<p><span class="legend-swatch" style="background:${colors[id]}"></span><b>${title}</b><br>${body}</p>`).join('')+'<p>The scene models mathematical data dependencies; it makes no GPU synchronization or timing claim.</p>');
  $('math-fullscreen').onclick=async()=>{if(document.fullscreenElement)await document.exitFullscreen();else await root.requestFullscreen();};
  document.addEventListener('fullscreenchange',()=>{$('math-fullscreen').textContent=document.fullscreenElement?'Exit full screen':'Full screen';});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe(root);
  window.addEventListener('beforeprint',stop);reduce.addEventListener('change',stop);
  window.LlamaMath={snapshot:()=>{
    const frame=frames[index],event=data.events[frame.event],ops=operands(event);
    return {pass,index,total:frames.length,eventIndex:frame.event,stage:event.stage,layer:event.layer,term:frame.term,
      head,row,col,playing:timer!==null,cache:toy.clone(event.cache),buffers:toy.clone(event.buffers),partial:partial(event,ops),
      next:data.next,C:data.C,T:data.T,onlineIndex};
  },stop};
  stageOptions();render();
})();
