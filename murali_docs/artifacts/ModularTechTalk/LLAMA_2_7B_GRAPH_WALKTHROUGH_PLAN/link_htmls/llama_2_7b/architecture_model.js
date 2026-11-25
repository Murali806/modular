/* Analytical tensor ledger. No hardware timings, model weights, or MAX execution. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LlamaArchitecture = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const MODEL = Object.freeze({D:4096, F:11008, H:32, d:128, N:32, V:32000, context:4096});
  const DEFAULTS = Object.freeze({T:512, C:0, B:1, w:2, a:2, k:2, P:128,
    pairs:'causal', traffic:'boundary', kvReads:1, budgetMs:10, bandwidth:1000, tflops:100});
  const integer = (x, lo, hi, name) => {
    if (!Number.isInteger(x) || x < lo || x > hi) throw new Error(`${name} must be an integer from ${lo} to ${hi}.`);
    return x;
  };
  function settings(input={}) {
    const s = {...DEFAULTS, ...input};
    for (const key of ['T','C','B','w','a','k','P','kvReads','budgetMs','bandwidth','tflops']) s[key] = Number(s[key]);
    integer(s.T,1,4096,'Active tokens T'); integer(s.C,0,4095,'Cached tokens C'); integer(s.B,1,64,'Requests B');
    if (s.T+s.C>4096) throw new Error('C + T must fit the 4096-token model context.');
    if (![0.5,1,2,4].includes(s.w) || ![2,4].includes(s.a) || ![1,2,4].includes(s.k)) throw new Error('Unsupported teaching payload size.');
    if (![16,32,64,128,256].includes(s.P)) throw new Error('Choose a page size listed in the calculator.');
    integer(s.kvReads,1,128,'KV read multiplier');
    for (const key of ['budgetMs','bandwidth','tflops']) if (!Number.isFinite(s[key]) || s[key]<1e-6 || s[key]>1e6) throw new Error(`${key} must be between 0.000001 and 1,000,000.`);
    if (!['causal','rectangle'].includes(s.pairs) || !['boundary','materialized'].includes(s.traffic)) throw new Error('Unknown accounting model.');
    return s;
  }
  function calculate(input={}) {
    const s=settings(input), {T,C,B,w,a,k,P}=s, {D,F,H,d,N,V}=MODEL;
    const R=B*T, L=C+T, X=R*D*a, I=R*F*a, small=B*D*a;
    const score=B*H*T*L*4, cache=B*L*D*k, fresh=R*D*k;
    const validPairs=B*(T*C+T*(T+1)/2), rectanglePairs=B*T*L;
    const pairs=s.pairs==='causal'?validPairs:rectanglePairs, attMAC=pairs*H*d;
    const rows=[], fmt=n=>n.toLocaleString('en-US'), shape=(...v)=>'['+v.join(', ')+']';
    const act=shape(R,D), wide=shape(R,F), heads=shape(R,H,d), scores=shape(B,H,T,L);
    const M=(id,num,title,ins,out,operation,weight,mac,read,write,live,formula,scalar='',repeat=N) => {
      const traffic=read===null?null:read+write;
      const row={id,num,title,ins,out,operation,weight,mac,read,write,traffic,live,formula,scalar,repeat,
        ai:traffic?2*mac/traffic:null, requiredGBs:traffic===null?null:traffic/(s.budgetMs*1e6),
        computeMs:2*mac/(s.tflops*1e9), memoryMs:traffic===null?null:traffic/(s.bandwidth*1e6),
        countMac:true, countTraffic:true};
      rows.push(row); return row;
    };
    const dense=(id,num,title,ins,out,m,n,p,live,weightName)=>M(id,num,title,ins,out,
      `${weightName}: [${p}, ${n}]; Y = A × ${weightName}ᵀ`,n*p*w,m*n*p,
      m*n*a+n*p*w,m*p*a,live,
      [`MAC = rows × input width × output width = ${fmt(m)} × ${fmt(n)} × ${fmt(p)}`,
       `Read = input + weight = ${fmt(m)} × ${fmt(n)} × ${a} + ${fmt(n)} × ${fmt(p)} × ${w} bytes`,
       `Write = output = ${fmt(m)} × ${fmt(p)} × ${a} bytes`]);
    M('tokenizer','00','Tokenizer / input IDs',C?'prepared token IDs':'text',shape(R),
      'Host tokenization prepares prompt IDs once. Decode supplies sampled IDs; a later prompt chunk supplies pending prompt IDs.',0,0,null,null,R*8,
      ['Transformer contraction MACs = 0. CPU tokenizer work and transfers are outside this tensor ledger.'],
      'Variable-length text processing; host work not estimated.',1);
    M('embedding','01','Embedding',shape(R),act,'X = E[token_ids]; E [32000, 4096]',V*D*w,0,R*8+R*D*w,X,R*8+X,
      [`Read = ${fmt(R)} × 8 ID bytes + ${fmt(R)} × ${D} × ${w} selected weight bytes`,
       `Write = ${fmt(R)} × ${D} × ${a} activation bytes; full E remains resident.`],
      'Indexed gather; no matrix contraction. Repeated IDs are counted once per lookup (no lookup reuse).',1);
    const norm=(id,num,title,count,live,repeat=N)=>M(id,num,title,shape(count,D),shape(count,D),
      'x × rsqrt(mean(x²) + 1e−5) × γ; γ [4096]',D*a,0,count*D*a+D*a,count*D*a,live,
      [`Read = (${fmt(count)} × ${D} + ${D}) × ${a} bytes; write = ${fmt(count)} × ${D} × ${a} bytes`,
       'γ uses the activation payload size in this model. Reduction accumulators and casts are additional implementation work.'],
      `${fmt(3*count*D+count)} multiplies + ${fmt(count*D)} adds + ${fmt(count)} rsqrt (one direct formula); 0 contraction MACs.`,repeat);
    norm('norm1','02','Pre-attention RMSNorm',R,2*X);
    dense('q','03','Q projection',act,act,R,D,D,3*X,'W_Q');
    dense('k','04','K projection',act,act,R,D,D,4*X,'W_K');
    dense('v','05','V projection',act,act,R,D,D,5*X,'W_V');
    // The semantic reference spells out two rotated tensors. MAX fuses RoPE/split/store.
    const trig=T*d*a;
    M('rope','06','Heads + RoPE',`Q, K, V: ${act}`,`Q′, K′, V: ${heads}`,
      'View 32 heads × 128 channels. Rotate Q and K at position C + row. V is unchanged.',0,0,2*X+trig,2*X,6*X+trig,
      [`Read Q and K = 2 × ${fmt(X)} bytes; shared cos/sin slice = T × d × a = ${fmt(trig)} bytes`,
       `Write Q′ and K′ = 2 × ${fmt(X)} bytes; reshape is a logical view.`,
       'Equal-position requests share the illustrative trig slice. Full frequency-table residency is outside the weight subtotal.'],
      `${fmt(4*R*D)} multiplies + ${fmt(2*R*D)} adds for both Q/K; precomputed frequencies assumed.`);
    M('kv','07','Append this layer’s KV',`K′, V: each ${heads}`,`cache K,V: each ${shape(B,L,H,d)}`,
      `Write positions ${C}…${L-1}. Keep earlier positions. Cache contains K′ and V only.`,0,0,2*X,2*fresh,4*X,
      [`Read current K′ and V = 2 × ${fmt(R)} × ${D} × ${a} bytes`,
       `Write NEW cache entries = 2 × ${fmt(R)} × ${D} × ${k} = ${fmt(2*fresh)} bytes/layer`,
       `Resident used K+V = 2 × B × (C+T) × D × k = ${fmt(2*cache)} bytes/layer; old entries are retained, not rewritten.`],
      'Addressing + stores; dtype conversion/quantization work, if any, is not estimated.');
    const pairFormula=s.pairs==='causal'
      ?`B × [T×C + T(T+1)/2] = ${B} × [${T}×${C} + ${T}×${T+1}/2] = ${fmt(pairs)} query–key pairs`
      :`B × T × (C+T) = ${B} × ${T} × ${L} = ${fmt(pairs)} query–key pairs`;
    M('qk','08','Attention scores',`Q′ ${heads}; K cache ${shape(B,L,H,d)}`,scores,
      'S[b,h] = Q′[T,128] × K′cacheᵀ[128,L] / √128; conceptual score tensor.',0,attMAC,
      s.traffic==='materialized'?X+cache*s.kvReads:null,s.traffic==='materialized'?score:null,2*X+score,
      [pairFormula,`MAC = pairs × H × d = ${fmt(pairs)} × ${H} × ${d}`,
       `Reference read = Q′ + K cache × read multiplier; write = B×H×T×L×4 = ${fmt(score)} score bytes.`],
      'Score scaling adds one multiply per evaluated score. Tiled kernels can do padded/masked work beyond useful MACs.');
    M('softmax','09','Causal mask + softmax',scores,scores,
      'j > C+i → −∞. P = exp(S − row_max) / row_sum. Each valid row sums to 1.',0,0,
      s.traffic==='materialized'?score:null,s.traffic==='materialized'?score:null,2*X+2*score,
      [`FP32 reference score/probability size = B × H × T × L × 4 = ${fmt(score)} bytes each`,
       `Valid scalar entries E = H × B × [T×C + T(T+1)/2] = ${fmt(H*validPairs)}; rows = B×H×T = ${fmt(B*H*T)}`],
      `Direct stable softmax: E subtracts + E exp + E divides; E−rows additions and E−rows max comparisons. Mask/scaling extra; 0 contraction MACs.`);
    M('pv','10','Weighted values + merge',`P ${scores}; V cache ${shape(B,L,H,d)}`,act,
      'Per head: P[T,L] × Vcache[L,128]. Concatenate 32 heads → [B×T,4096].',0,attMAC,
      s.traffic==='materialized'?score+cache*s.kvReads:null,s.traffic==='materialized'?X:null,3*X+score,
      [pairFormula,`MAC = pairs × H × d = ${fmt(pairs)} × ${H} × ${d}`,
       `Reference read = probabilities + V cache × read multiplier; write = ${fmt(X)} context bytes.`]);
    if(s.traffic==='boundary') {
      const group=M('attention','08–10','Fused attention boundary',`Q′ + all K′/V cache`,act,
        'One boundary estimate for scores, softmax and weighted values. No full S/P in device memory.',0,2*attMAC,
        X+2*cache*s.kvReads,X,3*X,
        [`MAC = QK + PV = 2 × ${fmt(attMAC)}; counted once via rows 08 and 10.`,
         `Read = Q′ + cache K,V × ${s.kvReads} = ${fmt(X)} + 2 × ${fmt(cache)} × ${s.kvReads} bytes`,
         `Write = context = ${fmt(X)} bytes. One KV read is an ideal reuse assumption across active queries.`,
         'On-chip tile state, repeated Q reads, split reductions, page metadata and layout traffic are excluded.']);
      group.countMac=false;
      for(const row of rows.filter(x=>['qk','softmax','pv'].includes(x.id))) row.live=3*X;
    }
    dense('o','11','Output projection',act,act,R,D,D,3*X,'W_O');
    const residual=(id,num,title,op)=>M(id,num,title,`two ${act} buffers`,act,op,0,0,2*X,X,3*X,
      [`Read = 2 × ${fmt(X)} bytes; write = ${fmt(X)} bytes.`, 'Keep the original residual until its add consumes it.'],`${fmt(R*D)} scalar adds; 0 contraction MACs.`);
    residual('add1','12','Attention residual','H = X + attention_output');
    norm('norm2','13','Post-attention RMSNorm',R,2*X);
    dense('gate','14','FFN gate',act,wide,R,D,F,2*X+I,'W_gate');
    dense('up','15','FFN up',act,wide,R,D,F,2*X+2*I,'W_up');
    M('swiglu','16','SiLU(gate) × up',`G, U: each ${wide}`,wide,
      'Z = [G / (1 + exp(−G))] ⊙ U; SiLU belongs to the gate branch.',0,0,2*I,I,X+3*I,
      [`Read G,U = 2 × ${fmt(I)} bytes; write Z = ${fmt(I)} bytes.`,
       'One combined elementwise pass is assumed; a separate SiLU temporary would add traffic/storage.'],
      `${fmt(R*F)} each of exp, negate, add, divide and multiply (one direct formula); 0 contraction MACs.`);
    dense('down','17','FFN down',wide,act,R,F,D,2*X+I,'W_down');
    residual('add2','18','FFN residual','X_next = H + Z × W_downᵀ; repeat with next layer’s weights/cache');
    M('gather','19','Select last active row',act,shape(B,D),
      'LAST_TOKEN: input_row_offsets[1:] − 1; gather BEFORE final norm and head.',0,0,small,small,X+small,
      [`Read selected rows = B × D × a = ${fmt(small)} bytes; write same.`, 'Row-offset metadata is excluded.'],
      'Gather/indexing; 0 contraction MACs.',1);
    norm('finalnorm','20','Final RMSNorm',B,2*small,1);
    const head=dense('head','21','Untied LM head + cast',shape(B,D),shape(B,V),B,D,V,small+B*V*4,'W_head');
    head.repeat=1; head.write=B*V*4; head.traffic=head.read+head.write;
    head.operation='Separate W_head [32000,4096]; logits = FP32(norm_last × W_headᵀ).';
    head.formula[2]=`Write final FP32 logits = ${B} × ${V} × 4 bytes. Head + output cast share one ideal boundary; an unfused cast adds traffic.`;
    M('sample','22','Sampling → next input',shape(B,V),shape(B),
      'Separate sampler graph: greedy or configured policy. Generated IDs have NO KV until their next forward.',0,0,null,null,B*V*4+B*8,
      [`Minimum result payload = ${B} × 8 token-ID bytes. Sampler policies/scratch and host transfers are not estimated.`],
      'Greedy needs B×(V−1) comparisons; temperature/top-k/top-p add policy-dependent work. Model-device sampling is the default route.',1);
    for(const row of rows) {
      row.ai=row.traffic?2*row.mac/row.traffic:null;
      row.requiredGBs=row.traffic===null?null:row.traffic/(s.budgetMs*1e6);
      row.computeMs=2*row.mac/(s.tflops*1e9);
      row.memoryMs=row.traffic===null?null:row.traffic/(s.bandwidth*1e6);
    }
    const sum=(field,predicate=()=>true)=>rows.filter(predicate).reduce((n,r)=>n+(r[field]||0)*r.repeat,0);
    const mac=sum('mac',r=>r.countMac), traffic=sum('traffic',r=>r.countTraffic);
    const weights=(2*V*D+N*(4*D*D+3*D*F))*w+(2*N+1)*D*a;
    const kvPerToken=2*N*D*k, usedKV=B*L*kvPerToken, beforeKV=B*C*kvPerToken;
    const pagesBefore=B*Math.ceil(C/P), pages=B*Math.ceil(L/P), allocatedKV=pages*P*kvPerToken;
    const logicalPeak=Math.max(...rows.map(r=>r.live));
    return {settings:s,model:MODEL,rows,R,L,X,I,score,validPairs,rectanglePairs,
      totals:{mac,flops:2*mac,traffic,read:sum('read'),write:sum('write'),ai:2*mac/traffic,
        requiredGBs:traffic/(s.budgetMs*1e6),computeMs:2*mac/(s.tflops*1e9),memoryMs:traffic/(s.bandwidth*1e6)},
      memory:{weights,kvPerToken,beforeKV,usedKV,addedKV:usedKV-beforeKV,pagesBefore,pages,
        allocatedKV,allocatedBefore:pagesBefore*P*kvPerToken,pageBytes:P*kvPerToken,paddingKV:allocatedKV-usedKV,
        logicalPeak,modeledResident:weights+allocatedKV+logicalPeak,
        // A single head tile only. Not multiplied by an invented GPU concurrency.
        exampleTile:64*d*a+2*128*d*k+64*128*4+64*d*4+2*64*4},
      layer:{mac:rows.filter(r=>r.repeat===N&&r.countMac).reduce((n,r)=>n+r.mac,0),
        traffic:rows.filter(r=>r.repeat===N).reduce((n,r)=>n+(r.traffic||0),0)}};
  }
  return {MODEL,DEFAULTS,settings,calculate};
});
