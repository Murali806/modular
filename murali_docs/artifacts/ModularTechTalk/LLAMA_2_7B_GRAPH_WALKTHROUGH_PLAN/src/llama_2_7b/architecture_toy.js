/* Computed miniature decoder. Invented weights, FP64 math, explicit KV state. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.LlamaToy=api;})(globalThis,function(){
  'use strict';
  const spec=Object.freeze({D:4,F:6,H:2,d:2,N:2,V:6,P:2,pages:[3,1,5],epsilon:1e-5});
  const clone=x=>structuredClone(x);
  const matrix=(m,n,fn)=>Array.from({length:m},(_,i)=>Array.from({length:n},(_,j)=>fn(i,j)));
  const transpose=a=>a[0].map((_,j)=>a.map(row=>row[j]));
  function matmul(a,b){
    if(!a.length||!b.length||a[0].length!==b.length)throw new Error('Incompatible matrix product');
    return matrix(a.length,b[0].length,(i,j)=>a[i].reduce((s,v,k)=>s+v*b[k][j],0));
  }
  const add=(a,b)=>a.map((r,i)=>r.map((v,j)=>v+b[i][j]));
  const scale=(a,s)=>a.map(r=>r.map(v=>v*s));
  const normalize=(a,g)=>a.map(r=>{const inv=1/Math.sqrt(r.reduce((s,x)=>s+x*x,0)/r.length+spec.epsilon);return r.map((x,j)=>x*inv*g[j]);});
  const silu=x=>x/(1+Math.exp(-x));
  function rotate(a,start){return a.map((r,i)=>{const c=Math.cos(start+i),s=Math.sin(start+i);return r.flatMap((x,j)=>j%2?[]:[x*c-r[j+1]*s,x*s+r[j+1]*c]);});}
  const head=(a,h)=>a.map(r=>r.slice(h*spec.d,(h+1)*spec.d));
  const learned=(m,n,seed)=>matrix(m,n,(i,j)=>Math.sin((i+1)*1.3+(j+1)*.7+seed)*.45);
  const embedding=learned(spec.V,spec.D,.1),outputWeight=learned(spec.V,spec.D,9.8);
  const weights=Array.from({length:spec.N},(_,l)=>({
    q:learned(4,4,1+l),k:learned(4,4,2.2+l),v:learned(4,4,3.5+l),o:learned(4,4,4.1+l),
    gate:learned(6,4,5.3+l),up:learned(6,4,6.7+l),down:learned(4,6,7.2+l),
    norm1:[1,1.1,.9,1.05],norm2:[.95,1,1.05,1.1]
  })),finalGamma=[1.05,.95,1.1,1];
  function softmax(a){return a.map(row=>{const m=Math.max(...row);if(!Number.isFinite(m))return row.map(()=>0);const e=row.map(x=>Math.exp(x-m)),sum=e.reduce((n,x)=>n+x,0);return e.map(x=>x/sum);});}
  const emptyCache=()=>Array.from({length:spec.N},()=>({K:[],V:[]}));
  const tensor=(label,values,note='')=>({label,values:clone(values),note});
  const shape=a=>`[${a.length}, ${a[0]?.length||0}]`;
  const bufferInfo={weights:['W','model lifetime · all weights'],residual:['X / H','current residual'],norm:['N','normalized active rows'],qkv:['Q, K, V','new projections'],q:['Q′','current rotated query'],attention:['S / P','conceptual attention scratch'],context:['O','attention update'],ffn:['G / U / Z','current FFN'],logits:['logits','sampler input'],token:['IDs','active / sampled token IDs']};
  function forward(tokens,previous=emptyCache(),collect=true){
    if(!tokens.length)throw new Error('At least one active token required');
    const cache=clone(previous),C=cache[0].K.length,T=tokens.length;
    if(cache.some(c=>c.K.length!==C||c.V.length!==C))throw new Error('A forward requires equal completed cache lengths across layers');
    if(C+T>spec.P*spec.pages.length)throw new Error('Toy cache capacity exceeded');
    const events=[],contexts=[],buffers=Object.fromEntries(Object.entries(bufferInfo).map(([id,[label,owner]])=>[id,{label,owner,value:'empty',occupied:false}]));
    const set=(id,value)=>Object.assign(buffers[id],{occupied:true,value});
    const clear=(...ids)=>ids.forEach(id=>Object.assign(buffers[id],{occupied:false,value:'released / not yet produced'}));
    set('weights','E + 2 independent blocks + untied W_head');set('token','input IDs '+tokens.join(', '));
    let layer=null;
    function emit(stage,title,detail,source,kind,left,right,result,read=[],write=[],extra={}){
      if(!collect)return;
      events.push({stage,title,detail,source,kind,left,right,result,read,write,layer,C,T,buffers:clone(buffers),cache:clone(cache),...extra});
    }
    let x=tokens.map(id=>embedding[id].slice());
    set('residual','X_input '+shape(x));
    emit('embedding','01 · Gather embedding rows','Token IDs select E rows. They are not numeric hidden features.','embedding','gather',tensor('token IDs',tokens.map(id=>[id])),tensor('E [vocab, hidden]',embedding),tensor('X',x),['weights','token'],['residual']);
    for(layer=0;layer<spec.N;layer++){
      const w=weights[layer],name=`layer ${layer}`,norm=normalize(x,w.norm1);
      set('norm','N₁ '+shape(norm));
      emit('norm1','02 · Pre-attention RMSNorm',`${name}: keep X for the attention residual. Normalize each active row.`, 'norm','norm',tensor('X',x),tensor('γ₁',[w.norm1]),tensor('N₁',norm),['residual','weights'],['norm']);
      const q=matmul(norm,transpose(w.q));set('qkv','Q '+shape(q));
      emit('q','03 · Q = N₁ × W_Qᵀ',`${name}: each output feature is a row–column dot product.`, 'qkv','matmul',tensor('N₁',norm),tensor('W_Qᵀ [in,out]',transpose(w.q)),tensor('Q',q),['norm','weights'],['qkv']);
      const k=matmul(norm,transpose(w.k));set('qkv','Q, K: each '+shape(k));
      emit('k','04 · K = N₁ × W_Kᵀ',`${name}: keys are projected only for the active tokens.`, 'qkv','matmul',tensor('N₁',norm),tensor('W_Kᵀ [in,out]',transpose(w.k)),tensor('K',k),['norm','weights'],['qkv']);
      const v=matmul(norm,transpose(w.v));set('qkv','Q, K, V: each '+shape(v));
      emit('v','05 · V = N₁ × W_Vᵀ',`${name}: values carry the information attention will mix.`, 'qkv','matmul',tensor('N₁',norm),tensor('W_Vᵀ [in,out]',transpose(w.v)),tensor('V',v),['norm','weights'],['qkv']);
      const qr=rotate(q,C),kr=rotate(k,C);clear('norm');set('q','Q′ '+shape(qr));set('qkv','K′, V pending cache write');
      const ropeHeads=Array.from({length:spec.H},(_,h)=>({
        left:tensor('Q | K · selected head',head(q,h).map((r,i)=>r.concat(head(k,h)[i]))),
        right:tensor('cos(position) | sin(position)',matrix(T,2,(i,j)=>j?Math.sin(C+i):Math.cos(C+i))),
        result:tensor('Q′ | K′ · selected head',head(qr,h).map((r,i)=>r.concat(head(kr,h)[i])))
      }));
      emit('rope','06 · Reshape heads, rotate Q and K',`${name}: absolute positions ${C}…${C+T-1}. V stays unchanged.`, 'rope','rope',ropeHeads[0].left,ropeHeads[0].right,ropeHeads[0].result,['qkv'],['q','qkv'],{headMatrices:ropeHeads});
      for(let i=0;i<T;i++){
        cache[layer].K.push(kr[i].slice());cache[layer].V.push(v[i].slice());
        const cacheHeads=Array.from({length:spec.H},(_,h)=>({
          left:tensor('new K′ · selected head',[head(kr,h)[i]]),right:tensor('new V · selected head',[head(v,h)[i]]),
          result:tensor('retained K′ | V',head(cache[layer].K,h).map((r,j)=>r.concat(head(cache[layer].V,h)[j])))
        }));
        emit('kv',`07 · Store position ${C+i} in layer ${layer}`,`Only this layer’s K′ and V are appended. Earlier positions keep exactly the same values.`, 'rope','store',cacheHeads[0].left,cacheHeads[0].right,cacheHeads[0].result,['qkv'],[],{written:{layer,position:C+i},headMatrices:cacheHeads});
      }
      clear('qkv');
      const S=[],P=[],O=[];
      for(let h=0;h<spec.H;h++){
        S[h]=scale(matmul(head(qr,h),transpose(head(cache[layer].K,h))),1/Math.sqrt(spec.d));
        P[h]=softmax(S[h].map((r,i)=>r.map((v,j)=>j>C+i?-Infinity:v)));
        O[h]=matmul(P[h],head(cache[layer].V,h));
      }
      contexts.push({q:clone(qr),k:clone(cache[layer].K),v:clone(cache[layer].V),scores:S,probabilities:P,output:O,C,T});
      set('attention','S per head '+shape(S[0])+' · mathematical view');
      const scoreHeads=Array.from({length:spec.H},(_,h)=>({left:tensor('Q′ · selected head',head(qr,h)),right:tensor('K′cacheᵀ',transpose(head(cache[layer].K,h))),result:tensor('S = dot / √2',S[h])}));
      emit('qk','08 · Q′ × cached K′ᵀ / √d',`${name}: active queries compare with every processed key. Future keys will be masked at the next stage.`, 'attention','matmul',scoreHeads[0].left,scoreHeads[0].right,scoreHeads[0].result,['q'],['attention'],{scale:1/Math.sqrt(spec.d),headMatrices:scoreHeads});
      set('attention','P per head '+shape(P[0])+' · mathematical view');
      const softHeads=Array.from({length:spec.H},(_,h)=>{
        const masked=S[h].map((r,i)=>r.map((v,j)=>j>C+i?-Infinity:v));
        const stats=masked.map(r=>{const max=Math.max(...r);return [max,r.reduce((s,x)=>s+Math.exp(x-max),0)];});
        return {left:tensor('S after causal mask',masked),right:tensor('row max | Σ exp(S−max)',stats),result:tensor('P · row sums = 1',P[h])};
      });
      emit('softmax','09 · Causal mask, then stable softmax',`${name}: query i sees key positions j ≤ C+i. Masked entries become probability 0.`, 'attention','softmax',softHeads[0].left,softHeads[0].right,softHeads[0].result,['attention'],['attention'],{headMatrices:softHeads});
      set('context','per-head context '+shape(O[0]));
      const valueHeads=Array.from({length:spec.H},(_,h)=>({left:tensor('P · selected head',P[h]),right:tensor('V cache',head(cache[layer].V,h)),result:tensor('weighted value',O[h])}));
      emit('pv','10 · Probabilities × cached values',`${name}: a weighted sum of the retained values; both old and new positions can contribute.`, 'attention','matmul',valueHeads[0].left,valueHeads[0].right,valueHeads[0].result,['attention'],['context'],{headMatrices:valueHeads});
      const merged=matrix(T,spec.D,(i,j)=>O[Math.floor(j/spec.d)][i][j%spec.d]);
      clear('q','attention');set('context','concat head 0 | head 1 '+shape(merged));
      emit('merge','10b · Concatenate heads','Place head 0’s channels followed by head 1’s channels. This reshape/concatenation has no contraction MACs.', 'attention','concat',tensor('head 0',O[0]),tensor('head 1',O[1]),tensor('merged context',merged),['context'],['context']);
      const projected=matmul(merged,transpose(w.o));set('context','attention output '+shape(projected));
      emit('o','11 · Output projection',`${name}: mix the head channels back into the residual width.`, 'qkv','matmul',tensor('merged context',merged),tensor('W_Oᵀ',transpose(w.o)),tensor('attention output',projected),['context','weights'],['context']);
      const h=add(x,projected);set('residual','H = X + attention '+shape(h));clear('context');
      emit('add1','12 · First residual add',`${name}: retain H until the second residual add; the attention update is consumed.`, 'residual','add',tensor('X',x),tensor('attention output',projected),tensor('H',h),['residual','context'],['residual']);
      const n2=normalize(h,w.norm2);set('norm','N₂ '+shape(n2));
      emit('norm2','13 · Post-attention RMSNorm',`${name}: normalize H for the feed-forward branch.`, 'norm','norm',tensor('H',h),tensor('γ₂',[w.norm2]),tensor('N₂',n2),['residual','weights'],['norm']);
      const gate=matmul(n2,transpose(w.gate));set('ffn','G '+shape(gate));
      emit('gate','14 · FFN gate projection',`${name}: expansion from width 4 to width 6; real Llama expands 4096 → 11008.`, 'mlp','matmul',tensor('N₂',n2),tensor('W_gateᵀ',transpose(w.gate)),tensor('G',gate),['norm','weights'],['ffn']);
      const up=matmul(n2,transpose(w.up));set('ffn','G + U, each '+shape(up));
      emit('up','15 · FFN up projection',`${name}: the up branch is linear; SiLU is applied to G.`, 'mlp','matmul',tensor('N₂',n2),tensor('W_upᵀ',transpose(w.up)),tensor('U',up),['norm','weights'],['ffn']);
      const z=gate.map((r,i)=>r.map((x,j)=>silu(x)*up[i][j]));clear('norm');set('ffn','Z = SiLU(G) × U '+shape(z));
      emit('swiglu','16 · SiLU(gate) ⊙ up',`${name}: elementwise gating, not a matrix product.`, 'mlp','silu',tensor('G',gate),tensor('U',up),tensor('Z',z),['ffn'],['ffn']);
      const down=matmul(z,transpose(w.down));set('ffn','FFN update '+shape(down));
      emit('down','17 · FFN down projection',`${name}: reduce the intermediate width to the residual width.`, 'mlp','matmul',tensor('Z',z),tensor('W_downᵀ',transpose(w.down)),tensor('FFN update',down),['ffn','weights'],['ffn']);
      x=add(h,down);set('residual',`X after layer ${layer} `+shape(x));clear('ffn');
      emit('add2','18 · Second residual add',layer===0?'Continue with layer 1’s independent matrices and cache.':'Both toy layers are complete. Real Llama repeats the decoder block 32 times.', 'residual','add',tensor('H',h),tensor('FFN update',down),tensor('X_next',x),['residual','ffn'],['residual']);
    }
    layer=null;
    const last=[x.at(-1).slice()];set('residual','last row [1,4]');
    emit('gather','19 · Gather last active row','LAST_TOKEN selects the last row before final normalization; only this row predicts the next token.','logits','gather',tensor('X after all layers',x),tensor('selected row index',[[T-1]]),tensor('last row',last),['residual'],['residual']);
    const normalized=normalize(last,finalGamma);clear('residual');set('norm','final norm [1,4]');
    emit('finalnorm','20 · Final RMSNorm','One row per request is normalized on this output path.','norm','norm',tensor('last row',last),tensor('γ_final',[finalGamma]),tensor('normalized last',normalized),['residual','weights'],['norm']);
    const logits=matmul(normalized,transpose(outputWeight));set('logits','[1,6] toy vocabulary scores');
    emit('head','21 · Independent LM head','W_head is a separate learned matrix from E. Real MAX returns [B,32000] FP32 logits.','logits','matmul',tensor('normalized last',normalized),tensor('W_headᵀ [hidden,vocab]',transpose(outputWeight)),tensor('logits',logits),['norm','weights'],['logits']);
    const next=logits[0].indexOf(Math.max(...logits[0]));clear('norm');set('token',`sampled ID ${next}; KV pending until next pass`);
    emit('sample','22 · Greedy sample → next input',`Toy argmax selects ID ${next}. Cache lengths remain ${C+T}; the sampled token has not been processed yet. Real sampling policy is configurable.`, 'sampling','sample',tensor('logits',logits),tensor('maximum score',[[logits[0][next]]]),tensor('sampled ID',[[next]]),['logits'],['token']);
    return {tokens:tokens.slice(),C,T,cache:clone(cache),hidden:clone(x),logits:clone(logits),next,contexts:clone(contexts),events};
  }
  function route(name){
    const prefill=forward([0,1,2]);if(name==='prefill')return {...prefill,name};
    const decode=forward([prefill.next],prefill.cache);if(name==='decode')return {...decode,name,prefix:[0,1,2]};
    if(name==='crosspage'){const cross=forward([decode.next],decode.cache);return {...cross,name,prefix:[0,1,2,prefill.next]};}
    throw new Error('Unknown pass');
  }
  function timeline(data){return data.events.flatMap((event,index)=>Array.from({length:event.kind==='matmul'?event.right.values.length+1:1},(_,term)=>({event:index,term})));}
  function online(q,k,v,position,tileSize=2){
    if(!Number.isInteger(tileSize)||tileSize<1)throw new Error('Positive integer tile size required');
    if(!Number.isInteger(position)||position<0||position>=k.length)throw new Error('Query position must have a valid cache entry');
    const scores=k.map((key,j)=>j>position?-Infinity:key.reduce((s,x,i)=>s+x*q[i],0)/Math.sqrt(q.length));
    let m=-Infinity,l=0,u=v[0].map(()=>0);const tiles=[];
    for(let start=0;start<=position;start+=tileSize){
      const end=Math.min(position+1,start+tileSize),ss=scores.slice(start,end),old={m,l,u:u.slice()};
      const nextM=Math.max(m,...ss),alpha=Number.isFinite(m)?Math.exp(m-nextM):0,p=ss.map(x=>Math.exp(x-nextM));
      u=u.map((x,j)=>alpha*x+p.reduce((s,pv,i)=>s+pv*v[start+i][j],0));
      l=alpha*l+p.reduce((s,x)=>s+x,0);m=nextM;
      tiles.push({start,end,scores:ss,k:clone(k.slice(start,end)),v:clone(v.slice(start,end)),old,m,l,u:u.slice(),alpha,p,output:u.map(x=>x/l)});
    }
    const probabilities=softmax([scores])[0],full=matmul([probabilities],v)[0];
    return {tiles,probabilities,full,output:u.map(x=>x/l)};
  }
  return {spec,bufferInfo,clone,transpose,matmul,normalize,rotate,softmax,head,emptyCache,forward,route,timeline,online,
    parameters:()=>clone({embedding,weights,outputWeight,finalGamma})};
});
