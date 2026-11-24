/* Small, offline calculations. These are illustrative inputs, never GPU results. */
(function(){
  'use strict';
  const lab=document.querySelector('[data-lab]');
  if(!lab)return;
  const $=s=>lab.querySelector(s),model=window.LlamaModel;
  const controls=$('#lab-controls'),result=$('#lab-result');
  const integer=(id,min,max)=>Math.max(min,Math.min(max,Math.floor(Number($(id).value))||min));
  const number=n=>n.toLocaleString('en-US');

  if(lab.dataset.lab==='kv'){
    controls.innerHTML=`
      <label>Tokens / request <input type="number" id="kv-tokens" min="1" max="4096" value="4096"></label>
      <label>Requests <input type="number" id="kv-batch" min="1" max="32" value="1"></label>
      <label>Page tokens <select id="kv-page"><option>128</option><option>256</option></select></label>
      <label>Bytes / element <select id="kv-bytes"><option value="2">2 · BF16-compatible arithmetic</option><option value="4">4 · FP32 arithmetic</option></select></label>`;
    function capacity(){
      const r=model.kv(integer('#kv-tokens',1,4096),integer('#kv-batch',1,32),Number($('#kv-page').value),Number($('#kv-bytes').value));
      result.innerHTML=`<p><b>${r.perToken/1024} KiB/token</b> across 32 layers · ${r.pages} logical pages · payload ${(r.logical/2**30).toFixed(4)} GiB · page-rounded ${(r.reserved/2**30).toFixed(4)} GiB.</p>
      <p class="notice">Illustrative payload arithmetic. Excludes weights, workspace, metadata, allocator reserve, shared prefixes and physical slab padding.</p>`;
    }
    controls.addEventListener('input',capacity);capacity();

    const address=document.createElement('section');
    address.className='address-lab';address.setAttribute('aria-label','KV page address explorer');
    address.innerHTML=`<h3>Address one cache element</h3>
      <p>Independent two-page example: <code>P=128</code>, <code>lookup[A]=[7,2]</code>, 32 layers, 32 heads, 128 channels, two bytes/element. Select a token cell or change its coordinates. The numbers below are offsets, not live pointers.</p>
      <div id="address-controls">
        <label>Token position <input type="number" id="addr-token" min="0" max="255" value="129"></label>
        <label>Layer <input type="number" id="addr-layer" min="0" max="31" value="5"></label>
        <label>Head <input type="number" id="addr-head" min="0" max="31" value="3"></label>
        <label>Channel <input type="number" id="addr-channel" min="0" max="127" value="4"></label>
        <label>Cache <select id="addr-kind"><option value="0">K</option><option value="1">V</option></select></label>
        <label><input type="checkbox" id="addr-padding"> Add 128 bytes of illustrative padding per physical page</label>
      </div><div id="address-pages" class="page-buffers"></div>
      <div id="address-result" aria-live="polite"></div>
      <p class="notice">Changing a head or layer selects a different vector inside the same page. Page ownership stays with the cache manager. <a href="#q1">Inspect the element-stride and packed-sentinel contract.</a></p>`;
    lab.append(address);
    const packed=2*32*128*32*128,table=[7,2];
    function showAddress(){
      const input={token:integer('#addr-token',0,255),layer:integer('#addr-layer',0,31),head:integer('#addr-head',0,31),channel:integer('#addr-channel',0,127),kind:Number($('#addr-kind').value),pageStride:$('#addr-padding').checked?packed+64:-1,table};
      const r=model.kvAddress(input);
      $('#address-pages').innerHTML=table.map((physical,logical)=>`<article class="page-buffer ${logical===r.logicalPage?'active-page':''}">
        <b>Logical page ${logical} → physical page ${physical}</b><small>Request A · token positions ${logical*128}–${logical*128+127}</small>
        <div class="token-grid">${Array.from({length:128},(_,slot)=>{
          const token=logical*128+slot;
          return `<button type="button" data-page-token="${token}" aria-label="Token ${token}, slot ${slot}, physical page ${physical}" aria-pressed="${token===input.token}" title="Token ${token}; slot ${slot}">${slot}</button>`;
        }).join('')}</div></article>`).join('');
      $('#address-result').innerHTML=`<ol class="address-steps">
        <li><code>${input.token} ÷ 128</code> → logical page <b>${r.logicalPage}</b>, slot <b>${r.slot}</b>.</li>
        <li><code>lookup[A,${r.logicalPage}]</code> → physical page <b>${r.physicalPage}</b>.</li>
        <li><code>[${input.kind===0?'K':'V'}, layer ${input.layer}, slot ${r.slot}, head ${input.head}, channel ${input.channel}]</code> → within-page offset <b>${number(r.withinPageElements)} elements</b>.</li>
        <li>CPU stride input <b>${number(input.pageStride)}</b> → resolved stride <b>${number(r.stride)} elements/page</b>.</li>
        <li><code>(${r.physicalPage} × ${number(r.stride)} + ${number(r.withinPageElements)}) × 2</code> → byte offset <b>${number(r.byteOffset)}</b> from pool base.</li></ol>`;
      address.dataset.byteOffset=r.byteOffset;
    }
    $('#address-controls').addEventListener('input',showAddress);
    $('#address-pages').addEventListener('click',e=>{
      const button=e.target.closest('[data-page-token]');if(!button)return;
      $('#addr-token').value=button.dataset.pageToken;showAddress();
    });
    showAddress();
  }else if(lab.dataset.lab==='softmax'){
    controls.innerHTML=`<label>Scores <input id="sm-scores" value="2,1,3"></label>
      <label>Value scalar <input id="sm-values" value="10,20,30"></label>
      <label>Tile <select id="sm-tile"><option>2</option><option>1</option><option>3</option></select></label>`;
    function parse(id){return $(id).value.split(',').map(x=>x.trim()===''?NaN:Number(x));}
    function update(){
      const s=parse('#sm-scores'),v=parse('#sm-values');
      if(s.length!==v.length||!s.length||s.length>16||s.some(x=>!Number.isFinite(x))||v.some(x=>!Number.isFinite(x))){
        result.textContent='Enter equally sized lists of up to 16 finite numbers.';return;
      }
      const rows=model.softmaxTiles(s,v,Number($('#sm-tile').value));
      if(rows.some(r=>!Number.isFinite(r.output))){result.textContent='Use smaller value magnitudes so the scalar calculation stays finite.';return;}
      const fixed=x=>Number.isFinite(x)?x.toFixed(5):'−∞';
      result.innerHTML='<table><thead><tr><th>Tile starts</th><th>old (m,l,a)</th><th>rescale α</th><th>new (m,l,a)</th><th>a/l</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${r.start}</td><td>${[r.old.m,r.old.l,r.old.a].map(fixed).join(', ')}</td><td>${fixed(r.alpha)}</td><td>${[r.m,r.l,r.a].map(fixed).join(', ')}</td><td>${r.output.toFixed(6)}</td></tr>`).join('')+'</tbody></table>';
      const m=Math.max(...s),p=s.map(x=>Math.exp(x-m)),l=p.reduce((a,b)=>a+b,0);
      result.innerHTML+='<div class="probabilities" aria-label="Final normalized attention weights">'+p.map((x,i)=>`<div><span>key ${i}: ${(x/l).toFixed(6)}</span><div class="probability-track"><span style="width:${100*x/l}%"></span></div></div>`).join('')+'</div><p class="notice">Scalar calculation using JavaScript numbers. Compare tile widths: the stable recurrence describes the same normalized sum. These are illustrative scores and values, not model activations or GPU numerical results.</p>';
    }
    controls.addEventListener('input',update);update();
  }
})();
