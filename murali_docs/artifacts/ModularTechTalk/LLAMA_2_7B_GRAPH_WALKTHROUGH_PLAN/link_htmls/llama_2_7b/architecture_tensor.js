/* Actual Llama 2 7B extents; lazy, synthetic operation probes, never checkpoint inference. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.LlamaTensor=api;})(globalThis,function(){
  'use strict';
  const spec=Object.freeze({D:4096,F:11008,H:32,d:128,N:32,V:32000,P:128,bytes:2,epsilon:1e-5,pages:Object.freeze([3,1,5,2,7])});
  const block=['norm1','q','k','v','rope','kv','qk','softmax','pv','merge','o','add1','norm2','gate','up','swiglu','down','add2'];
  const definitions={
    embedding:['01 · Embedding gather','embedding','E[IDs]'],norm1:['02 · Pre-attention RMSNorm','norm','RMS₁'],
    q:['03 · Q projection','qkv','Q'],k:['04 · K projection','qkv','K'],v:['05 · V projection','qkv','V'],
    rope:['06 · Reshape heads + RoPE','rope','RoPE'],kv:['07 · Append rotated K and V','rope','KV store'],
    qk:['08 · Q′ × cached K′ᵀ / √128','attention','QKᵀ'],softmax:['09 · Causal stable softmax','attention','softmax'],
    pv:['10 · Probabilities × cached V','attention','P·V'],merge:['11 · Merge 32 heads','attention','merge'],
    o:['12 · Output projection','qkv','W_O'],add1:['13 · Attention residual','residual','+ X'],
    norm2:['14 · Pre-FFN RMSNorm','norm','RMS₂'],gate:['15 · FFN gate projection','mlp','gate'],
    up:['16 · FFN up projection','mlp','up'],swiglu:['17 · SiLU(gate) ⊙ up','mlp','SiLU ⊙'],
    down:['18 · FFN down projection','mlp','down'],add2:['19 · FFN residual','residual','+ H'],
    gather:['20 · Select last active row','logits','last'],finalnorm:['21 · Final RMSNorm','norm','RMS'],
    head:['22 · Untied LM head','logits','head'],sample:['23 · Greedy sampling example','sampling','ID']
  };
  const route=name=>{if(!['prefill','decode','next'].includes(name))throw Error('Unknown pass');const C=name==='prefill'?0:name==='decode'?512:513,T=C?1:512;return {name,C,T,L:C+T};};
  const events=[{stage:'embedding',layer:null},...Array.from({length:32},(_,layer)=>block.map(stage=>({stage,layer}))).flat(),...['gather','finalnorm','head','sample'].map(stage=>({stage,layer:null}))];
  // Integer hashing permits random access without allocating matrices or storing weights.
  function value(seed,i,j){let x=Math.imul(seed+1,0x9e3779b1)^Math.imul(i+1,0x85ebca6b)^Math.imul(j+1,0xc2b2ae35);x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return ((x^(x>>>16))>>>0)/2147483648-1;}
  const tensor=(name,rows,cols,get,axes='row × channel',shape=[rows,cols])=>({name,rows,cols,get,axes,shape});
  const token=position=>(position*61+17)%32000;
  const raw=(kind,layer,head,position,channel)=>value(100+layer*97+head*3+({Q:0,K:1,V:2}[kind]),position,channel);
  function rotated(kind,layer,head,position,channel){
    const pair=channel%64,angle=position*Math.pow(10000,-2*pair/128),a=raw(kind,layer,head,position,pair),b=raw(kind,layer,head,position,pair+64);
    return channel<64?a*Math.cos(angle)-b*Math.sin(angle):a*Math.sin(angle)+b*Math.cos(angle);
  }
  function attention(pass,layer,head){
    const {C,T,L}=route(pass),rows=new Map();
    const q=(i,c)=>rotated('Q',layer,head,C+i,c),k=(j,c)=>rotated('K',layer,head,j,c),v=(j,c)=>raw('V',layer,head,j,c);
    function row(i){
      if(!rows.has(i)){
        const scores=new Float64Array(L),p=new Float64Array(L);let max=-Infinity;
        for(let j=0;j<L;j++){let sum=0;for(let c=0;c<128;c++)sum+=q(i,c)*k(j,c);scores[j]=j<=C+i?sum/Math.sqrt(128):-Infinity;max=Math.max(max,scores[j]);}
        let denom=0;for(let j=0;j<L;j++){p[j]=Math.exp(scores[j]-max);denom+=p[j];}for(let j=0;j<L;j++)p[j]/=denom;
        rows.set(i,{scores,p,max,denom});
      }
      return rows.get(i);
    }
    return {C,T,L,q,k,v,row};
  }
  function operation(pass,event,head=0,ropeKind='Q'){
    const {C,T,L}=route(pass),{stage}=event,layer=event.layer??31,seed=2000+events.findIndex(e=>e.stage===stage&&e.layer===event.layer)*17;
    const op={...event,head,ropeKind,C,T,L,title:definitions[stage][0],source:definitions[stage][1],kind:'map',steps:1,inner:0,scale:1};
    const generated=(name,m,n,s=0,scale=1)=>tensor(name,m,n,(i,j)=>value(seed+s,C+i,j)*scale);
    function dense(input,weight,output,m,n,k,offset){
      op.kind='dot';op.inner=k;op.steps=k;
      op.left=generated(input,m,k,offset);op.right=tensor(weight+'ᵀ',k,n,(i,j)=>value(seed+offset+1,j,i)/Math.sqrt(k),'input channel × output channel');
      op.weightShape=[n,k];op.result=tensor(output,m,n,(i,j)=>sum(i,j));
    }
    const sums=new Map();
    function sum(i,j){const key=i+':'+j;if(!sums.has(key)){let s=0;for(let k=0;k<op.inner;k++)s+=op.left.get(i,k)*op.right.get(k,j);sums.set(key,s*op.scale);}return sums.get(key);}
    if(['q','k','v','o','gate','up','down','head'].includes(stage)){
      const n=['gate','up'].includes(stage)?11008:stage==='head'?32000:4096,k=stage==='down'?11008:4096,m=stage==='head'?1:T;
      const labels={q:['N₁','W_Q','Q'],k:['N₁','W_K','K'],v:['N₁','W_V','V'],o:['context','W_O','attn update'],gate:['N₂','W_gate','G'],up:['N₂','W_up','U'],down:['Z','W_down','FFN update'],head:['last normalized','W_head','logits']};
      dense(...labels[stage],m,n,k,3);op.note='Synthetic operands for this projection; every selected output includes the entire reduction. Stored weight [out,in] is transposed on the right.';
    }else if(['norm1','norm2','finalnorm'].includes(stage)){
      const m=stage==='finalnorm'?1:T,means=new Map();op.kind='norm';op.inner=4096;op.steps=4096;
      op.left=generated(stage==='norm1'?'X':stage==='norm2'?'H':'last hidden',m,4096);
      op.right=tensor('γ',1,4096,(i,j)=>1+value(seed+1,0,j)*.1,'broadcast scale');
      op.mean=i=>{if(!means.has(i)){let s=0;for(let j=0;j<4096;j++)s+=op.left.get(i,j)**2;means.set(i,s/4096);}return means.get(i);};
      op.result=tensor('normalized',m,4096,(i,j)=>op.left.get(i,j)*op.right.get(0,j)/Math.sqrt(op.mean(i)+1e-5));
      op.note='Each row reduces all 4096 squared channels. Keep the residual row for its later add.';
    }else if(['rope','kv','qk','softmax','pv','merge'].includes(stage)){
      const a=attention(pass,layer,head);op.attention=a;
      if(stage==='rope'){
        op.kind='rope';op.steps=4;
        op.left=tensor(ropeKind+' · head '+head,T,128,(i,j)=>raw(ropeKind,layer,head,C+i,j),'active row × head channel',[T,32,128]);
        op.right=tensor('cos θ | sin θ',64,2,(i,j)=>(j?Math.sin:Math.cos)((C+(op.query??0))*Math.pow(10000,-2*i/128)),'pair × cos/sin');
        op.result=tensor(ropeKind+'′ · head '+head,T,128,(i,j)=>rotated(ropeKind,layer,head,C+i,j),'active row × head channel',[T,32,128]);
        op.note='Safetensors split-half RoPE: channel c pairs with c+64. Rotate Q and K at absolute position C+i; V is unchanged. These generated vectors feed the attention and cache views.';
      }else if(stage==='kv'){
        op.kind='store';op.steps=T;
        op.left=tensor('new K′ · head '+head,T,128,(i,j)=>a.k(C+i,j),'active row × head channel',[T,32,128]);
        op.right=tensor('new V · head '+head,T,128,(i,j)=>a.v(C+i,j),'active row × head channel',[T,32,128]);
        op.result=tensor('cache K′ · head '+head,L,128,a.k,'absolute position × channel',[L,32,128]);
        op.note='Append K′ and V for this layer. Only active positions C…C+T−1 are written. Page cells below retain earlier values and show actual 128-slot capacity.';
      }else if(stage==='qk'){
        op.kind='dot';op.steps=op.inner=128;op.scale=1/Math.sqrt(128);
        op.left=tensor('Q′ · head '+head,T,128,a.q);op.right=tensor('cached K′ᵀ · head '+head,128,L,(i,j)=>a.k(j,i),'head channel × absolute key');
        op.result=tensor('S · head '+head,T,L,sum,'query row × absolute key',[32,T,L]);
        op.note='One score reduces 128 channel products, then divides by √128. Future scores are masked before softmax. S is a mathematical view of fused attention.';
      }else if(stage==='softmax'){
        op.kind='softmax';op.steps=L;op.inner=L;
        op.left=tensor('scaled S + causal mask',T,L,(i,j)=>a.row(i).scores[j],'query × key',[32,T,L]);
        op.right=tensor('exp(S − row max)',T,L,(i,j)=>Math.exp(a.row(i).scores[j]-a.row(i).max),'query × key',[32,T,L]);
        op.result=tensor('P · head '+head,T,L,(i,j)=>a.row(i).p[j],'query × key',[32,T,L]);
        op.note='Key j is allowed only when j ≤ C+i. Reduce over the whole key axis, using zero for masked keys; normalize each query row to sum to one.';
      }else if(stage==='pv'){
        op.kind='dot';op.steps=op.inner=L;
        op.left=tensor('P · head '+head,T,L,(i,j)=>a.row(i).p[j],'query × absolute key');op.right=tensor('cached V · head '+head,L,128,a.v,'absolute key × channel');
        op.result=tensor('context · head '+head,T,128,sum,'active row × head channel',[T,32,128]);
        op.note='Mix all L value rows. Causally masked probabilities contribute zero. The 128 output channels from each of 32 heads concatenate into 4096.';
      }else{
        op.left=tensor('context · selected head',T,128,(i,j)=>value(seed+i,op.mergeHead??head,j),'active row × head channel',[T,32,128]);
        op.right=tensor('head/channel → hidden',32,128,(i,j)=>i*128+j,'head × head channel');
        op.result=tensor('merged context',T,4096,(i,j)=>value(seed+i,Math.floor(j/128),j%128));
        op.note='Reshape/concatenate, with no contraction: hidden channel = head × 128 + channel. Values for this layout probe are generated independently.';
      }
    }else if(stage==='embedding'){
      op.kind='gather';op.left=tensor('token IDs',T,1,i=>token(C+i),'active row × ID',[T]);
      op.right=tensor('E · untied embedding',32000,4096,(i,j)=>value(3,i,j),'vocabulary ID × channel');
      op.result=tensor('X',T,4096,(i,j)=>op.right.get(token(C+i),j));op.note='Gather one full 4096-channel embedding row per token ID. IDs select rows; no dense matrix multiplication is needed.';
    }else if(stage==='gather'){
      op.kind='gather';op.left=generated('final hidden',T,4096);op.right=tensor('last active row index',1,1,()=>T-1);
      op.result=tensor('last hidden',1,4096,(i,j)=>op.left.get(T-1,j));op.note='LAST_TOKEN route: select active row T−1 after all 32 layers. Final norm and the untied head process this one row.';
    }else if(stage==='sample'){
      op.kind='argmax';op.steps=op.inner=32000;op.left=generated('illustrative logits',1,32000);op.right=tensor('vocabulary ID',1,32000,(i,j)=>j);
      let max=-Infinity,id=0;for(let j=0;j<32000;j++){const x=op.left.get(0,j);if(x>max){max=x;id=j;}}
      op.result=tensor('chosen ID',1,1,()=>id);op.note='Greedy argmax over all 32000 independently generated logits illustrates one sampler choice. This is not a generated Llama prediction. Sampling writes no KV.';
    }else{
      const gated=stage==='swiglu',n=gated?11008:4096;op.kind=gated?'silu':'add';op.steps=2;
      op.left=generated(gated?'G':stage==='add1'?'X':'H',T,n,1);op.right=generated(gated?'U':stage==='add1'?'attention update':'FFN update',T,n,2);
      op.result=tensor(gated?'Z':'next residual',T,n,(i,j)=>{const x=op.left.get(i,j),y=op.right.get(i,j);return gated?x/(1+Math.exp(-x))*y:x+y;});
      op.note=gated?'Elementwise SiLU applies to the gate; multiply by the up branch. No reduction or channel mixing occurs here.':'Add matching channels. The earlier residual buffer must remain live until this sum.';
    }
    return op;
  }
  function probe(op,row,col){
    const n=op.inner,prefix=new Float64Array(n+1),ids=op.kind==='argmax'?new Int32Array(n+1):null;
    if(ids)prefix[0]=-Infinity;
    for(let k=0;k<n;k++){
      if(op.kind==='argmax'){const x=op.left.get(0,k),take=x>prefix[k];prefix[k+1]=take?x:prefix[k];ids[k+1]=take?k:ids[k];}
      else prefix[k+1]=prefix[k]+(op.kind==='dot'?op.left.get(row,k)*op.right.get(k,col):op.kind==='norm'?op.left.get(row,k)**2:op.right.get(row,k));
    }
    return {prefix,ids,expected:op.result.get(row,col)};
  }
  function cacheState(pass,event,step){
    const {C,T,L}=route(pass),layer=event.layer,at=block.indexOf(event.stage),kv=block.indexOf('kv');
    const lengths=Array.from({length:32},(_,l)=>event.stage==='embedding'?C:layer===null?L:l<layer?L:l>layer?C:C+(at<kv?0:at===kv?Math.min(T,step):T));
    const used=lengths.reduce((a,b)=>a+b,0)*2*32*128*2,assigned=Math.ceil(L/128)*64*2**20;
    return {lengths,completed:Math.min(...lengths),used,assigned,pages:Math.ceil(L/128)};
  }
  function address(position,layer,head,channel,kv=0){
    if(![position,layer,head,channel,kv].every(Number.isInteger)||position<0||position>=640||layer<0||layer>=32||head<0||head>=32||channel<0||channel>=128||kv<0||kv>1)throw Error('Coordinate outside illustrated page table');
    const logical=Math.floor(position/128),slot=position%128,physical=spec.pages[logical],stride=2*32*128*32*128;
    const within=((((kv*32+layer)*128+slot)*32+head)*128+channel),element=physical*stride+within;
    return {logical,slot,physical,stride,within,element,byte:element*2};
  }
  function online(a,query,tileSize=128){
    const tiles=[],u=new Float64Array(128);let m=-Infinity,l=0;
    const scores=a.row(query).scores;
    for(let start=0;start<a.L;start+=tileSize){
      const end=Math.min(a.L,start+tileSize),old={m,l,u:Array.from(u)};let tileMax=-Infinity;
      for(let j=start;j<end;j++)tileMax=Math.max(tileMax,scores[j]);
      const next=Math.max(m,tileMax),alpha=m===-Infinity?0:Math.exp(m-next);l*=alpha;
      for(let c=0;c<128;c++)u[c]*=alpha;
      for(let j=start;j<end;j++){const p=scores[j]===-Infinity?0:Math.exp(scores[j]-next);l+=p;for(let c=0;c<128;c++)u[c]+=p*a.v(j,c);}
      m=next;tiles.push({start,end,old,m,l,alpha,u:Array.from(u),output:Array.from(u,x=>l?x/l:0)});
    }
    return tiles;
  }
  return {spec,block,definitions,events,route,value,token,raw,rotated,attention,operation,probe,cacheState,address,online};
});
