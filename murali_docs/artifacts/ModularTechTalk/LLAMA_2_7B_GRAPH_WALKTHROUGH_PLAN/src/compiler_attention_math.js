/* Deterministic teaching arithmetic. This module runs without a DOM or MAX. */
(function(root,factory){const m=factory();if(typeof module==='object'&&module.exports)module.exports=m;else root.AttentionMath=m;})(typeof globalThis!=='undefined'?globalThis:this,()=>{
'use strict';
const Q=[[1,0],[0,1]],K=[[1,0],[0,1],[3,0],[2,1]],V=[[1,0],[0,2],[2,1],[1,3]];
const copy=x=>JSON.parse(JSON.stringify(x));
const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
function reference(q){
 const scores=K.map(k=>dot(q,k)),m=Math.max(...scores),E=scores.map(x=>Math.exp(x-m)),l=E.reduce((a,b)=>a+b,0),p=E.map(x=>x/l);
 return {scores,m,l,p,O:V[0].map((_,d)=>V.reduce((sum,v,j)=>sum+p[j]*v[d],0))};
}
function onlineFrames(queryIndex=0){
 const q=Q[queryIndex],frames=[],values={q:copy(q),k:copy(K),v:copy(V),m:null,l:0,o:[0,0],tile:null,S:null,E:null,prior:null,delta:null,PV:null},scene={queryIndex,q:copy(q),K:copy(K),V:copy(V),tile:-1,tileMax:null,scores:[null,null,null,null],exponents:[null,null,null,null],weights:[null,null,null,null],contributions:[null,null,null,null],m:null,l:0,O:[0,0],oldM:null,oldL:0,oldO:[0,0],mNew:null,lNew:null,alpha:null,priorMass:null,tileMass:null,oldContribution:null,newContribution:null,reference:null,match:null};
 const push=(title,source,line,input,action,output,why,patch={},changes={})=>{
  const before=copy(values);Object.assign(values,copy(patch));Object.assign(scene,copy(changes));
  frames.push({title,refs:[[source,line]],input,action,output,why,detail:why,phase:action,fields:copy(values),values:copy(values),before_values:before,focus:Object.keys(patch),before:input,after:output,scene:copy(scene)});
 };
 push('Select one query row and two-key teaching tiles','attention_tiles',333,'Q/K/V tiny teaching matrices','select query row; key rows per teaching tile=2',`q${queryIndex}=${JSON.stringify(q)}`,'This tiny arithmetic uses the same recurrence, not the actual GPU launch shape. Each query normalizes across all four keys independently.');
 push('Initialize the retained row state','attention_tiles',335,'no keys visited','m₁=MIN; l₁=0; Oᵢ=0','m=−∞; l=0; O=[0,0]','−∞ is the mathematical version of the source MIN sentinel; zero prior mass means no earlier value contribution.',{m:null,l:0,o:[0,0]});
 let m=null,l=0,O=[0,0];
 for(let t=0;t<2;t++){
  const start=t*2,indices=[start,start+1],oldM=m,oldL=l,oldO=copy(O),S=indices.map(j=>dot(q,K[j])),tileMax=Math.max(...S),mNew=Math.max(m===null?-Infinity:m,tileMax),alpha=m===null?0:Math.exp(m-mNew),E=S.map(x=>Math.exp(x-mNew)),priorMass=alpha*l,tileMass=E.reduce((a,b)=>a+b,0),lNew=priorMass+tileMass,P=E.map(x=>x/lNew);
  push(`Tile ${t+1}: select K and V rows`,'attention_tiles',344,`rows ${start},${start+1}`, 'K.tile and V.tile views', `K=${JSON.stringify(indices.map(j=>K[j]))}; V=${JSON.stringify(indices.map(j=>V[j]))}`,'Views select matching key/value rows. Earlier O/m/l survive; temporary S/E/PV are reused.',{tile:`rows ${start}–${start+1}`,S:null,E:null,prior:null,delta:null,PV:null},{tile:t,tileMax:null,dotTerms:null,oldM,oldL,oldO,mNew:null,lNew:null,alpha:null,priorMass:null,tileMass:null,oldContribution:null,newContribution:null,exponents:[null,null,null,null],weights:[null,null,null,null],contributions:[null,null,null,null],activeKeys:indices});
  const scores=copy(scene.scores);indices.forEach((j,r)=>scores[j]=S[r]);
  push('Compute the two dot-product scores','attention_tiles',346,`q=${JSON.stringify(q)}; selected K rows`,'S = Q_tile × K_tileᵀ',`S=${JSON.stringify(S)}`,'Each score sums products across feature d. S columns correspond to key rows, not output features.',{S}, {scores,dotTerms:indices.map(j=>q.map((x,d)=>({q:x,k:K[j][d],product:x*K[j][d]})))});
  push('Reduce the maximum across this key tile','attention_max',348,`S=${JSON.stringify(S)}`,'max[axis=1](S,m₂)',`m₂=${tileMax}`,'axis=1 means across key columns for this query. It does not mix query rows.',{delta:`tile max ${tileMax}`},{tileMax});
  push('Merge with the previous running maximum','attention_max',349,`tile max=${tileMax}; old max=${m===null?'−∞':m}`,'m₂.max(m₁)',`m₂=${mNew}`,'Old and new exponentials must use the same reference maximum before their masses can be added.',{delta:`m_new=${mNew}`},{mNew});
  push('Copy the previous maximum into denominator scratch','attention_denominator',353,`m₁=${m===null?'−∞':m}`,'l₂.copy_from(m₁)',`l₂ scratch=${m===null?'−∞':m}`,'l₂ is scratch at this point; it is not the new denominator yet.',{prior:m===null?"−∞":m});
  push('Subtract the new maximum','attention_denominator',354,`old m=${m===null?'−∞':m}; new m=${mNew}`,'l₂ -= m₂',`l₂ scratch=${m===null?'−∞':m-mNew}`,'If the maximum grows, the old score scale must shrink.',{prior:m===null?"−∞":m-mNew});
  push('Exponentiate the scale difference','attention_denominator',355,`m_old−m_new=${m===null?'−∞':m-mNew}`,'l₂.exp()',`alpha=${alpha}`,'exp(m_old−m_new) maps all earlier exponential mass to the new maximum scale.',{prior:alpha},{alpha});
  push('Scale the old denominator mass','attention_denominator',356,`alpha=${alpha}; l_old=${l}`,'l₂ *= l₁',`prior mass=${priorMass}`,'Mass from already-visited keys contributes to the final denominator. It must not be discarded.',{prior:priorMass},{priorMass});
  push('Copy current scores into E scratch','attention_denominator',358,`S=${JSON.stringify(S)}`,'E.copy_from(S)',`E scratch=${JSON.stringify(S)}`,'S is the dot-product tile; E will be reused as exponentials then normalized weights.',{E:S});
  const shifted=S.map(x=>x-mNew);
  push('Shift each score by the shared row maximum','attention_denominator',359,`S=${JSON.stringify(S)}; m_new=${mNew}`,'E -= m₂',`E scratch=${JSON.stringify(shifted)}`,'All shifted scores are ≤0, so their exponentials are bounded by 1.',{E:shifted});
  const exponents=[null,null,null,null];indices.forEach((j,r)=>exponents[j]=E[r]);
  push('Compute stable current exponentials','attention_denominator',360,`shifted scores=${JSON.stringify(shifted)}`,'E.exp()',`E=${JSON.stringify(E)}`,'Only the current tile is stored; earlier scores are represented by m/l/O.',{E},{exponents});
  push('Sum exponential mass across current keys','attention_denominator',362,`E=${JSON.stringify(E)}`,'sum[axis=1](E,l₂_sum)',`tile mass=${tileMass}`,'This row sum is the current tile contribution to the denominator.',{delta:`tile mass=${tileMass}`},{tileMass});
  push('Add prior and current mass','attention_denominator',363,`prior=${priorMass}; current=${tileMass}`,'l₂ += l₂_sum',`l_new=${lNew}`,'The denominator now includes every key visited so far, on one consistent maximum scale.',{prior:lNew},{lNew});
  const weights=[null,null,null,null];indices.forEach((j,r)=>weights[j]=P[r]);
  push('Normalize the current tile weights','attention_output',366,`E=${JSON.stringify(E)}; l_new=${lNew}`,'E /= l₂',`P_tile=${JSON.stringify(P)}`,'These weights use the cumulative denominator; a later tile can still change their final scale.',{E:P},{weights});
  let rescaled=O.map(x=>x*l);
  push('Undo the previous output normalization','attention_output',367,`O_old=${JSON.stringify(O)}; l_old=${l}`,'Oᵢ *= l₁',`O scratch=${JSON.stringify(rescaled)}`,'O_i is stored normalized. Multiplying by its old denominator reconstructs its old numerator.',{o:rescaled},{O:rescaled,oldContribution:null});
  rescaled=rescaled.map(x=>x/lNew);
  push('Apply the new denominator to the prior output','attention_output',368,`old numerator=${JSON.stringify(values.o)}; l_new=${lNew}`,'Oᵢ /= l₂',`O scratch=${JSON.stringify(rescaled)}`,'The prior output must use the same cumulative denominator as current weights.',{o:rescaled},{O:rescaled});
  push('Copy the old maximum into output-scale scratch','attention_output',370,`m_old=${m===null?'−∞':m}`,'m₁_delta.copy_from(m₁)',`delta scratch=${m===null?'−∞':m}`,'The source recomputes the exponential scale for output rescaling; we show that actual operation order.',{delta:m===null?"−∞":m});
  push('Subtract the new maximum again','attention_output',371,`m_old and m_new=${mNew}`,'m₁_delta -= m₂',`delta=${m===null?'−∞':m-mNew}`,'The output and denominator need the same old-to-new maximum conversion.',{delta:m===null?"−∞":m-mNew});
  push('Exponentiate the output scale','attention_output',372,`delta=${m===null?'−∞':m-mNew}`,'m₁_delta.exp()',`alpha=${alpha}`,'When the maximum is unchanged, alpha=1; when it grows, alpha<1.',{delta:alpha});
  rescaled=rescaled.map(x=>x*alpha);
  push('Finish rescaling the retained output','attention_output',373,`O scratch=${JSON.stringify(values.o)}; alpha=${alpha}`,'Oᵢ *= m₁_delta',`prior O contribution=${JSON.stringify(rescaled)}`,'The complete prior factor is alpha·l_old/l_new. This preserves all earlier keys without storing their scores.',{o:rescaled},{O:rescaled,oldContribution:rescaled});
  const contributions=[null,null,null,null];indices.forEach((j,r)=>contributions[j]=V[j].map(x=>x*P[r]));const PV=V[0].map((_,d)=>indices.reduce((sum,j,r)=>sum+P[r]*V[j][d],0));
  push('Compute the weighted current values','attention_output',374,`P_tile=${JSON.stringify(P)}; V_tile=${JSON.stringify(indices.map(j=>V[j]))}`,'PV = P_tile × V_tile',`PV=${JSON.stringify(PV)}`,'Each output feature is a sum across current key rows. This is the second matrix product.',{PV},{contributions,newContribution:PV});
  O=rescaled.map((x,d)=>x+PV[d]);
  push('Merge previous and current output contributions','attention_output',375,`prior=${JSON.stringify(rescaled)}; PV=${JSON.stringify(PV)}`,'Oᵢ += PV',`O_new=${JSON.stringify(O)}`,'The output is normalized over all keys visited so far.',{o:O},{O,newO:O});
  push('Retain the new maximum for the next tile','attention_output',376,`m₂=${mNew}`,'m₁.copy_from(m₂)',`m₁=${mNew}`,'m/l/O must remain consistent across iterations.',{m:mNew},{m:mNew});m=mNew;
  push('Retain the new denominator for the next tile','attention_output',377,`l₂=${lNew}`,'l₁.copy_from(l₂)',`l₁=${lNew}`,'Only row state and the output tile survive; S/E/PV scratch may be reused.',{l:lNew},{l:lNew});l=lNew;
 }
 const ref=reference(q),match=O.every((x,d)=>Math.abs(x-ref.O[d])<1e-12);
 push('Store final O and check against independent full softmax','attention_output',378,`final row state m=${m}; l=${l}; O=${JSON.stringify(O)}`,'O.tile.copy_from(Oᵢ)',`O=${JSON.stringify(O)} · ${match?'MATCH':'MISMATCH'}`,'The independent browser reference computes all scores at once solely to verify the teaching arithmetic. The fused kernel never needs that persistent full matrix.',{o:O},{reference:ref,match,activeKeys:[],finalWeights:ref.p});
 return frames;
}
return {Q,K,V,dot,reference,onlineFrames};
});
