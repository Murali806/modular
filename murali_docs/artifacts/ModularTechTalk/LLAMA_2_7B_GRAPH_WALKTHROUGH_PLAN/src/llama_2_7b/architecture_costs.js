/* Offline controls for the explicit analytical model. */
(function () {
  'use strict';
  const root=document.querySelector('[data-cost-mode]'); if(!root)return;
  const model=window.LlamaArchitecture, $=id=>document.getElementById(id);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=(n,d=3)=>n.toLocaleString('en-US',{maximumFractionDigits:d});
  function bytes(n){for(const [u,f] of [['GiB',2**30],['MiB',2**20],['KiB',1024]])if(n>=f)return number(n/f)+' '+u;return number(n)+' B';}
  function count(n){for(const [u,f]of [['T',1e12],['G',1e9],['M',1e6]])if(n>=f)return number(n/f,4)+' '+u;return number(n);}
  let report=null, selected='q';
  const metric=(label,value,detail='')=>`<article><b>${esc(label)}</b><strong>${esc(value)}</strong>${detail?`<small>${esc(detail)}</small>`:''}</article>`;
  function inputs(){return Object.fromEntries(new FormData($('cost-form')));}
  function fill(values){for(const [key,value]of Object.entries(values)){const control=$('cost-form').elements.namedItem(key);if(control)control.value=String(value);}}
  function inspector(){
    const r=report.rows.find(r=>r.id===selected)||report.rows.find(r=>r.id==='q');selected=r.id;
    $('cost-stage').value=selected;
    for(const row of $('cost-rows').children)row.classList.toggle('selected',row.id==='cost-'+selected);
    const {budgetMs,bandwidth,tflops}=report.settings;
    const grouped=r.traffic===null&&['qk','softmax','pv'].includes(r.id);
    const byteText=r.traffic===null?(grouped?'Inspect group 08–10 for the shared fused boundary.':'Policy / host work is outside the byte subtotal.'):
      `Read ${number(r.read)} B + write ${number(r.write)} B = ${number(r.traffic)} B (${bytes(r.traffic)}).`;
    const intensity=r.traffic!==null&&r.mac
      ? `Contraction AI = (2 × ${number(r.mac)}) / ${number(r.traffic)} = ${number(r.ai,6)} FLOP/byte.`
      :r.traffic===null?'No independent byte boundary is assigned to this row.':'0 contraction FLOP/byte; this row has scalar or data-movement work.';
    const bw=r.traffic===null?'Bandwidth is not estimated for this individual row.':
      `Required average BW = ${number(r.traffic)} / (${number(budgetMs,6)} ms × 10⁶) = ${number(r.requiredGBs,6)} GB/s.`;
    const floor=r.traffic===null?'Use a row with a defined byte boundary to compare traffic and contraction work.':
      `At the assumed rates: transfer term = ${number(r.traffic)} / (${number(bandwidth)} × 10⁶) = ${number(r.memoryMs,6)} ms; contraction term = ${number(2*r.mac)} / (${number(tflops)} × 10⁹) = ${number(r.computeMs,6)} ms. Restricted bound = ${number(Math.max(r.memoryMs,r.computeMs),6)} ms. Scalar work and overhead are excluded.`;
    $('cost-formula').innerHTML=`<h3>${esc(r.num+' · '+r.title)} <small>${r.repeat===32?'one layer; repeats ×32':'once per pass'}</small></h3><p><code>${esc(r.ins)} → ${esc(r.out)}</code></p><p>${esc(r.operation)}</p><ol>${r.formula.map(f=>`<li>${esc(f)}</li>`).join('')}<li><b>${number(r.mac)} MACs · ${number(2*r.mac)} contraction FLOPs</b>${r.countMac?'':' (group display; MACs already counted in 08 and 10)'}</li></ol><p>${esc(r.scalar)}</p><div class="formula-results"><p>${esc(byteText)}</p><p>${esc(intensity)}</p><p>${esc(bw)}</p><p>${esc(floor)}</p></div><p>Resident stage weights: <b>${bytes(r.weight)}</b>. Named live activations at this stage: <b>${bytes(r.live)}</b>. Persistent KV is counted separately in the memory ledger.</p>`;
  }
  function memory(){
    const m=report.memory,s=report.settings,pages=Math.ceil(report.L/s.P), shown=pages<=8?Array.from({length:pages},(_,i)=>i):[0,1,2,-1,pages-3,pages-2,pages-1];
    const cards=shown.map(i=>{
      if(i===-1)return '<div class="cache-ellipsis">…<br>middle pages</div>';
      const start=i*s.P,old=Math.max(0,Math.min(s.P,s.C-start)),fresh=Math.max(0,Math.min(start+s.P,report.L)-Math.max(start,s.C));
      const free=s.P-old-fresh;
      return `<article class="real-cache-page" data-logical-page="${i}"><b>Logical page ${i}</b><small>physical = lookup[request, ${i}]</small><div class="page-occupancy" role="img" aria-label="${old} retained, ${fresh} new, ${free} unused slots"><span class="old" style="width:${100*old/s.P}%"></span><span class="fresh" style="width:${100*fresh/s.P}%"></span></div><small>${old} retained · ${fresh} new · ${free} unused</small><code>positions ${start}…${Math.min(start+s.P-1,report.L-1)}</code></article>`;
    }).join('');
    const ledger=[['All model weights',`[2VD+32(4D²+3DF)]w + 65Da`,m.weights],
      ['Old valid K+V',`B × C × 2 × 32 × 4096 × k`,m.beforeKV],
      ['New K+V writes',`B × T × 2 × 32 × 4096 × k`,m.addedKV],
      ['All valid K+V after this pass',`old + new`,m.usedKV],
      ['Assigned KV capacity',`${m.pages} pages × ${s.P} slots × ${number(m.kvPerToken)} bytes`,m.allocatedKV],
      ['Unused assigned KV',`assigned − valid`,m.paddingKV],
      ['Largest named activation set',`max(stage live activations); not sum(stage outputs)`,m.logicalPeak],
      ['Modeled storage subtotal',`all weights + assigned KV + largest named activation set`,m.modeledResident]];
    $('cost-memory-result').innerHTML=`<p><b>Request 0 · all 32 layers share each ordinary page allocation.</b> Navy = retained prefix; amber = new entries; white = unused slots. Physical page IDs come from the runtime lookup table.</p><div class="real-cache-pages">${cards}</div><p>${s.B} equal-length request(s): assigned pages <b>${m.pagesBefore} → ${m.pages}</b>, assigned bytes <b>${bytes(m.allocatedBefore)} → ${bytes(m.allocatedKV)}</b>. Valid entries <b>${s.C} → ${report.L}</b> per request. One whole page across all layers is ${bytes(m.pageBytes)}.</p><div class="table-scroll"><table><thead><tr><th>Storage quantity</th><th>Calculation</th><th>Payload</th></tr></thead><tbody>${ledger.map(([label,f,n])=>`<tr><th scope="row">${esc(label)}</th><td><code>${esc(f)}</code></td><td title="${number(n)} bytes">${bytes(n)}</td></tr>`).join('')}</tbody></table></div><p><b>Separate on-chip illustration:</b> one head tile with Bq=64, Bk=128 needs Q + K + V + FP32 score tile + FP32 output accumulator + two row statistics: <code>64×128×a + 2×128×128×k + 64×128×4 + 64×128×4 + 2×64×4</code> = ${bytes(m.exampleTile)}. This is a chosen tile’s arithmetic, not a MAX tile size, register/shared-memory allocation or occupancy claim; it is excluded from the storage subtotal.</p><p>Subtotal excludes allocator reservation, metadata, padding, full frequency tables, sampler and backend workspace. See <a href="#counting-conventions">counting conventions</a> before using it as a capacity estimate.</p>`;
  }
  function render(){
    try {report=model.calculate(inputs());}
    catch(error){root.dataset.valid='false';$('cost-error').hidden=false;$('cost-error').textContent=error.message+' Results are hidden until the inputs are valid.';$('cost-export').disabled=true;return;}
    root.dataset.valid='true';$('cost-error').hidden=true;$('cost-export').disabled=false;
    const s=report.settings,t=report.totals;
    $('cost-assumptions').textContent=`T=${s.T}, C=${s.C}, L=${report.L}, B=${s.B}; ${s.pairs==='causal'?'causal useful':'rectangular'} MACs. Device-memory traffic uses ${s.traffic==='boundary'?'an ideal fused attention boundary':'materialized FP32 scores/probabilities'}, separate projection stages, and KV read multiplier ${s.kvReads}. ${s.w<2||s.k<2?'Packed payload setting: quantization metadata/conversion costs and support are not established. ':''}GB/s and TFLOP/s are decimal; KiB/MiB/GiB are binary.`;
    $('cost-summary').innerHTML=metric('Whole-pass MACs',count(t.mac),number(t.mac)+' exactly')+metric('Contraction FLOPs',count(t.flops),'2 × MAC; scalar work separate')+metric('Modeled transfer bytes',bytes(t.traffic),number(t.traffic)+' B · read + write')+metric('Contraction intensity',number(t.ai,3)+' FLOP/B','same whole-pass byte boundary')+metric('BW required in '+number(s.budgetMs,6)+' ms',number(t.requiredGBs)+' GB/s','bytes ÷ the whole-pass window')+metric('Restricted roofline term',number(Math.max(t.computeMs,t.memoryMs),4)+' ms','ideal terms; not measured latency');
    if(!report.rows.some(r=>r.id===selected))selected='qk';
    $('cost-stage').innerHTML=report.rows.map(r=>`<option value="${r.id}">${esc(r.num+' '+r.title)}</option>`).join('');
    $('cost-rows').innerHTML=report.rows.map(r=>{
      const traffic=r.traffic===null?(['qk','softmax','pv'].includes(r.id)?'group 08–10':'outside ledger'):bytes(r.traffic);
      const ai=r.ai!==null&&r.mac?number(r.ai,3):'—';
      return `<tr id="cost-${r.id}"><th scope="row"><button type="button" data-cost-stage="${r.id}">${esc(r.num+' '+r.title)}</button></th><td>${r.repeat}</td><td title="${number(r.mac)} MACs">${count(r.mac)}</td><td>${traffic}</td><td>${ai}</td><td>${bytes(r.weight)}</td><td>${bytes(r.live)}</td></tr>`;
    }).join('');
    inspector();memory();
  }
  function choose(id,scroll=false){if(!report?.rows.some(r=>r.id===id))return;selected=id;inspector();history.replaceState(null,'','#cost-'+id);if(scroll)$('cost-inspector').scrollIntoView({block:'start'});}
  $('cost-form').addEventListener('submit',e=>e.preventDefault());
  $('cost-form').addEventListener('input',render);
  $('cost-stage').addEventListener('change',()=>choose($('cost-stage').value));
  root.addEventListener('click',e=>{
    const stage=e.target.closest('[data-cost-stage]');if(stage)choose(stage.dataset.costStage,true);
    const preset=e.target.closest('[data-cost-preset]');if(preset){fill({B:1,...({prefill:{T:512,C:0},decode:{T:1,C:512},chunk:{T:128,C:384}}[preset.dataset.costPreset])});render();}
  });
  $('cost-reset').onclick=()=>{fill({...model.DEFAULTS,...(root.dataset.costMode==='decode'?{T:1,C:512}:{})});render();};
  $('cost-export').onclick=()=>{
    if(root.dataset.valid!=='true')return;
    const blob=new Blob([JSON.stringify({scope:'Analytical payload/traffic model, not measured MAX performance. See counting-conventions on the originating page.',...report},null,2)+'\n'],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`llama2-T${report.settings.T}-C${report.settings.C}-costs.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  window.LlamaCosts={snapshot:()=>JSON.parse(JSON.stringify(report)),valid:()=>root.dataset.valid==='true'};
  render();
  const target=location.hash.replace('#cost-','');if(report.rows.some(r=>r.id===target))choose(target,true);
})();
