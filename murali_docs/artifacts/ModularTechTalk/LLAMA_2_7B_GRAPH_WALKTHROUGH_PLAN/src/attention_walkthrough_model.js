/* Pure teaching model. No MAX execution, device access or browser globals. */
(function (root) {
  'use strict';
  const Q=[[1,0],[0,1]], K=[[1,0],[0,1],[2,0],[0,2]], V=[[1,0],[0,2],[3,1],[1,4]];
  const clone=value=>Array.isArray(value)?value.map(clone):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).map(([k,v])=>[k,clone(v)])):value;
  const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
  const sum=values=>values.reduce((a,b)=>a+b,0);
  const fmt=value=>value==null?'—':value===-Infinity?'−∞':Math.abs(value)<1e-12?'0':Number(value.toFixed(6)).toString();
  function reference(query=0) {
    const scores=K.map(row=>dot(Q[query],row)), m=Math.max(...scores), ex=scores.map(s=>Math.exp(s-m)), l=sum(ex), p=ex.map(x=>x/l);
    return {scores,m,l,p,O:[0,1].map(j=>sum(p.map((x,i)=>x*V[i][j])))};
  }
  function mathFrames(query=0) {
    if(!Number.isInteger(query)||query<0||query>=Q.length)throw new Error('Unknown query');
    const result=[],state={query,Q:clone(Q),K:clone(K),V:clone(V),tile:-1,m:-Infinity,l:0,O:[0,0],scores:null,products:null,shifted:null,E:null,P:null,alpha:null,oldMass:null,newMass:null,newM:null,newL:null,scaledO:null,PV:null,seen:0,part:'initial'};
    function add(title,operation,outgoing,why,part,line,code='') {
      state.part=part;
      result.push({title,incoming:`Query q${query}=[${Q[query]}]; ${state.seen} keys committed`,operation,outgoing,why,
        refs:`[[kernel:${line}]]`,code,mark:part,math:clone(state)});
    }
    add('Start an independent query row','Initialize m=−∞, l=0 and O=[0,0].','No keys have contributed.','Each query row has its own normalization state.','initial',335);
    for(let tile=0;tile<2;tile++) {
      const indexes=[tile*2,tile*2+1], kt=indexes.map(i=>K[i]),vt=indexes.map(i=>V[i]);
      state.tile=tile;state.indexes=indexes;state.kt=clone(kt);state.vt=clone(vt);
      for(const key of ['scores','products','shifted','E','P','alpha','oldMass','newMass','newM','newL','scaledO','PV'])state[key]=null;
      const oldM=state.m,oldL=state.l,oldO=state.O.slice();
      state.oldM=oldM;state.oldL=oldL;state.oldO=oldO.slice();
      add(`Select key tile ${tile+1}`,'Read two K rows and matching V rows; keep Q fixed.',`Rows ${indexes[0]} and ${indexes[1]} become current.`,'Reuse tile scratch; retain m/l/O from earlier tiles.','select',344);
      state.products=kt.map(row=>Q[query].map((q,j)=>q*row[j]));
      state.scores=state.products.map(p=>p[0]);
      add('Multiply the first feature','For each current key, multiply q[0] × k[0].',`Partial scores = [${state.scores.map(fmt)}].`,'A dot product sums feature contributions.','dot-first',346);
      state.scores=state.products.map(sum);
      add('Accumulate the second feature','Add q[1] × k[1] to each partial score.',`S = [${state.scores.map(fmt)}].`,'These are unscaled QK scores, not softmax weights yet.','scores',346);
      state.newM=Math.max(oldM,...state.scores);
      add('Choose the new running maximum','m_new = max(m_old, max(S)).',`m_new=${fmt(state.newM)}.`,'Subtracting a common maximum stabilizes exponentials.','maximum',348);
      state.alpha=Math.exp(oldM-state.newM);
      add('Compute the old-scale conversion','alpha = exp(m_old − m_new).',`alpha=${fmt(state.alpha)}.`,'Previous exponentials used a different maximum.','alpha',355);
      state.oldMass=state.alpha*oldL;
      add('Rescale the previous denominator','old_mass = alpha × l_old.',`old_mass=${fmt(state.oldMass)}.`,'Old keys must use the same exponential reference as new keys.','old-mass',356);
      state.shifted=state.scores.map(s=>s-state.newM);
      add('Shift current scores','Copy S; subtract m_new from each entry.',`S − m_new = [${state.shifted.map(fmt)}].`,'The largest current/past score is now at most zero.','shift',359);
      state.E=state.shifted.map(Math.exp);
      add('Exponentiate the current tile','E = exp(S − m_new).',`E = [${state.E.map(fmt)}].`,'Positive unnormalized masses can be added across tiles.','exponentials',360);
      state.newMass=sum(state.E);
      add('Sum the current denominator mass','new_mass = sum(E).',`new_mass=${fmt(state.newMass)}.`,'This tile adds mass from both of its keys.','new-mass',362);
      state.newL=state.oldMass+state.newMass;
      add('Combine old and current masses','l_new = old_mass + new_mass.',`l_new=${fmt(state.newL)}.`,'One denominator normalizes all keys seen so far.','denominator',363);
      state.P=state.E.map(e=>e/state.newL);
      add('Normalize current weights','P = E / l_new.',`P = [${state.P.map(fmt)}].`,'These weights already account for earlier key tiles.','weights',366);
      state.scaledO=oldO.map(o=>o*oldL);
      add('Undo the old output normalization','O_work = O_old × l_old.',`O_work=[${state.scaledO.map(fmt)}].`,'The stored O was normalized by l_old.','old-times-l',367);
      state.scaledO=state.scaledO.map(o=>o/state.newL);
      add('Apply the new denominator','O_work /= l_new.',`O_work=[${state.scaledO.map(fmt)}].`,'Previous contributions now use the new total mass.','old-divide-l',368);
      state.scaledO=state.scaledO.map(o=>o*state.alpha);
      add('Apply the maximum-change factor','O_work *= exp(m_old − m_new).',`Retained contribution=[${state.scaledO.map(fmt)}].`,'Both the denominator and exponential reference have changed.','old-scaled',373);
      state.PV=[sum(state.P.map((p,i)=>p*vt[i][0])),null];
      add('Accumulate the first V feature','PV[0] = P[0]V[0,0] + P[1]V[1,0].',`PV[0]=${fmt(state.PV[0])}.`,'The same normalized weights combine value features.','pv-first',374);
      state.PV[1]=sum(state.P.map((p,i)=>p*vt[i][1]));
      add('Accumulate the second V feature','PV[1] = P[0]V[0,1] + P[1]V[1,1].',`PV=[${state.PV.map(fmt)}].`,'Each output feature receives its weighted V contribution.','pv',374);
      state.O=state.scaledO.map((o,j)=>o+state.PV[j]);
      add('Merge old and current output','O_new = rescaled O_old + PV.',`O_new=[${state.O.map(fmt)}].`,'This is the normalized result for all keys seen so far.','merge',375);
      state.m=state.newM;state.l=state.newL;state.seen=(tile+1)*2;
      add('Commit row state; reuse scratch','Retain m_new, l_new and O_new.',`m=${fmt(state.m)}, l=${fmt(state.l)}, O=[${state.O.map(fmt)}].`,'The next tile needs only this state, not all old scores.','commit',376);
    }
    state.expected=reference(query).O;
    add('Store the completed output row','All four keys contributed; compare with full stable softmax.',`O=[${state.O.map(fmt)}].`,'The tiled recurrence preserves the full attention result.','store',378);
    return result;
  }
  function frames(data,key,options={}) {
    if(key==='math')return mathFrames(Number(options.query||0));
    let values=(data.timelines[key]||[]).map(clone);
    const vendor=options.vendor||'nvidia';
    if(options.cache==='hit'&&['compile','pipeline','artifacts'].includes(key)) {
      values=values.filter(f=>!['specialize','llvm','device-code','host-code'].includes(f.mark));
      values=values.map(f=>f.mark==='cache'?{...f,mark:'cache-hit',title:'Reuse prepared native-code resources',operation:'A matching code-cache entry supplies A1/D1 resources and symbol names.',outgoing:'Target compilation is skipped; native loading still occurs during init.',code:'cache hit → A1 + D1 bytes / symbol names\nno callable address yet'}:f);
    }
    if(vendor==='amd') values=values.map(f=>{
      if(f.mark==='device-code')return {...f,title:'Emit the AMD device image D1',operation:'AMDGPU backend emits an object and links device libraries into HSA ELF.',outgoing:'D1: AMD device code-object bytes in host memory.',code:'amdgcn-amd-amdhsa + chosen gfx architecture\nobject + device libraries → HSA code object',refs:'[[amdgpu:109]] [[amdgpu:142]]',source_html:null};
      if(f.mark==='attributes')return {...f,title:'Keep HIP load options with the function',operation:'The shown CUDA function-attribute calls are NVIDIA-specific; the HIP path retains its own function/module state.',outgoing:'HIP DeviceFunction with its module/context references.',code:'HIP loadFunction → retained function and module\nCUDA-only attribute setters are not called here',refs:'[[hip:1972]]',source_html:null};
      return {...f,operation:f.operation.replace('cuModuleLoadDataEx (HIP: hipModuleLoadData)','hipModuleLoadData').replace('cuModuleGetFunction (HIP equivalent)','hipModuleGetFunction').replace('NVIDIA: LLVM emits PTX; CUDA compiler obtains cubin.','AMDGPU emits and links a device ELF code object.'),code:f.code.replaceAll('cuModuleLoadDataEx','hipModuleLoadData').replaceAll('cuModuleGetFunction','hipModuleGetFunction')};
    });
    return values;
  }
  function clampIndex(index,total){return Math.min(Math.max(0,index),Math.max(0,total-1));}
  const api={Q,K,V,clone,fmt,reference,mathFrames,frames,clampIndex};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.AttentionModel=api;
})(typeof window==='object'?window:globalThis);
