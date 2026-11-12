/* Distinct attention mechanisms; card-local playback and preserved Hold state. */
(()=>{
'use strict';
const states=new WeakMap(),active=new Set(),reduced=matchMedia('(prefers-reduced-motion: reduce)');
const E=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x===null||x===undefined?'—':typeof x==='number'?(Number.isInteger(x)?String(x):Number(x.toFixed(6)).toString()):Array.isArray(x)?'['+x.map(fmt).join(', ')+']':typeof x==='object'?JSON.stringify(x):String(x);
const value=x=>x===null||x===undefined?'not yet shown':fmt(x);
const cell=(label,text,cls='',attrs='')=>`<div class="attention-value ${cls}" ${attrs}><span>${E(label)}</span><b>${E(value(text))}</b></div>`;
const fieldCell=(label,text,field,cls='',attrs='')=>cell(label,text,cls,attrs+' data-am-field="'+E(field)+'"');
const panel=(title,body,cls='')=>`<section class="attention-panel ${cls}"><h4>${E(title)}</h4>${body}</section>`;
const code=x=>'<pre class="attention-diagram">'+E(x)+'</pre>';
const arrow=(label='reference / conversion')=>`<div class="attention-reference-arrow">→ <span>${E(label)}</span></div>`;
function objects(s,f){
 const rows=s.scenario.nodes.map(n=>{
  const val=f.values[n.id],changed=f.focus.includes(n.id),before=f.before_values[n.id];
  return `<div class="attention-object ${changed?'attention-changed':''}" data-am-node="${E(n.id)}" data-am-changed="${changed}" data-am-value="${E(fmt(val))}"><span class="attention-object-kind">${E(n.kind)} · ${E(n.owner)}</span><b>${E(n.label)}</b><pre>${E(value(val))}</pre>${changed&&before!==null&&before!==undefined?`<small>previous: ${E(fmt(before))}</small>`:''}</div>`;
 });
 return '<details class="attention-object-register" data-am-disclosure="objects"><summary>Objects and fields at this step · changed fields outlined</summary><div class="attention-objects">'+rows.join('')+'</div></details>';
}
function ir(s,f){
 const v=f.values,isModule=s.root.dataset.amLesson==='graph-module-setup';
 let body;
 if(isModule){
  body=v.module?`<div class="attention-nest" data-am-native="MODULE1"><b>MODULE1 · builtin.module</b>${v.region?`<div class="attention-nest"><b>Rmodule · body region</b>${v.block?`<div class="attention-nest"><b>Bmodule · body block</b>${v.sibling?'<div class="attention-op">@encoder · retained sibling</div>':''}${v.graph?'<div class="attention-nest attention-changed"><b>G1 · mo.graph @fused_attention</b><div class="attention-nest"><b>R0 · graph region</b><div class="attention-nest"><b>B0(%q,%k,%v)</b><span>graph operations inserted here later</span></div></div></div>':'<span>no graph definition inserted yet</span>'}</div>`:''}</div>`:''}</div>`:code('module selection / creation is the next step');
 }else{
  body=`<div class="attention-nest"><b>G1 · mo.graph ${E(v.name||'pending symbol')}</b>${v.region?`<div class="attention-nest" data-am-native="R0"><b>R0 · region created by builder</b>${v.block?`<div class="attention-nest" data-am-native="B0"><b>B0 · ONE entry block</b><div class="attention-port-list">${String(v.args||'').split(',').filter(x=>x.trim()).map((a,i)=>cell('argument '+i,a,'attention-port')).join('')||'<span>arguments not added yet</span>'}</div><span>empty operation sequence at this constructor stage</span></div>`:'<span>block not created yet</span>'}</div>`:'<span>region not created yet</span>'}</div>`;
 }
 return panel('Operation → region → block → contained operations',body)+panel('Current native structure',code(f.scene.ir||v.fn||v.module||'builder arguments being assembled'));
}
function insertion(s,f){
 const v=f.values,custom=Boolean(v.call),graph=Boolean(v.graph),inside=String(v.cursor||'').includes('B0');
 return panel('Two different ordered blocks',`<div class="attention-block-row"><b>Bmodule</b><div class="attention-op-row">${graph?'<span class="attention-op">G1</span>':'<span>empty</span>'}${!inside&&v.cursor?'<b class="attention-cursor">▼ IP1 · module end</b>':''}</div></div><div class="attention-block-row"><b>G1.R0.B0</b><div class="attention-op-row">${v.body?'<span class="attention-op">CH0</span>':'<span>body not retained yet</span>'}${custom?'<span class="attention-op attention-changed">%o=CUST1(%q,%k,%v)</span>':''}${inside?'<b class="attention-cursor">▼ IP2 · graph end</b>':''}</div></div>`)+panel('Insertion point carries block + position',code(f.scene.ir||v.cursor||'waiting for block iterator'));
}
function context(s,f){
 const v=f.values;
 if('default' in v)return panel('One shared object; separate thread-local stacks',fieldCell('shared fallback object',v.default,'default')+`<div class="attention-two-columns">${panel('Main thread',code(v.main||'current state not entered'))}${panel('Background worker',code(v.worker||'worker not introduced'))}</div>`+fieldCell('guard / wrapper',v.guard,'guard')+fieldCell('constructed IR',v.ir,'ir'));
 return panel('Recording scope and reset token',`<div class="attention-two-columns">${fieldCell('active compiler context',v.context,'context')}${fieldCell('saved prior Graph',v.prior,'prior')}</div>`+arrow('CURRENT_GRAPH.set(G1)')+fieldCell('Graph.current',v.current,'current')+fieldCell('reset token',v.token,'token')+fieldCell('consumer operations',v.ops,'ops'));
}
function location(s,f){
 const v=f.values;return panel('Provenance is attached metadata',fieldCell('Python frame summaries',v.stack,'stack')+arrow('frame_loc or Location.unknown')+fieldCell('LOC1',v.loc,'loc')+`<div class="attention-two-columns">${fieldCell('MODULE1.location',v.module,'module')}${fieldCell('GraphOp + entry argument locations',v.graph,'graph')}</div>`+code('Scope closes → attached location metadata survives\nContext ≠ Location ≠ tensor address'));
}
function type(s,f){
 const v=f.values;return panel('Four fields become one native type',`<div class="attention-type-fields">${fieldCell('Shape dimension expression',v.dim,'dim')}${fieldCell('MOSH shape',v.shape,'shape')}${fieldCell('element dtype',v.dtype,'dtype')}${fieldCell('device kind / id',v.device,'device')}${fieldCell('optional metadata',v.metadata,'metadata')}</div>`+arrow('mo.TensorType(shape,dtype,device,metadata)')+fieldCell('native result type',v.native,'native')+code('32:si64 is a SIZE attribute\nfloat32 describes future numerical ELEMENTS\ngpu:0 is PLACEMENT; no matrix data converted'));
}
function attributes(s,f){
 const v=f.values;return panel('Fields on CUSTOM1',`<div class="attention-attribute-record">${fieldCell('symbol',v.symbol,'symbol')}${fieldCell('device',v.device,'device')}<div class="attention-two-columns">${fieldCell('BD',v.bd,'bd')}${fieldCell('BN',v.bn,'bn')}</div>${fieldCell('parameters dictionary',v.dict,'dict')}</div>`+code('Name/device values first → insert operation\nthen parameters/device attachment → kernel verifier'));
}
function contracts(s,f){
 const v=f.values,hasTypes=('q' in v)&&!('native' in v);return panel('Symbolic type contracts, not numerical buffers',hasTypes?`<div class="attention-three-columns">${fieldCell('TQ · queries',v.q,'q')}${fieldCell('TK · comparison vectors',v.k,'k')}${fieldCell('TV · content vectors',v.v,'v')}</div>`+arrow('one attention call')+fieldCell('TO',v.output,'output'):fieldCell('Python TO',v.python,'python')+arrow('native MO result type')+fieldCell('native TO',v.native,'native'))+panel('Attention dimensions',code(f.scene.equation||v.equation||'Q[N,D] × Kᵀ[D,N] → scores[N,N]\nrow-softmax(scores) × V[N,D] → O[N,D]'));
}
function parameters(s,f){
 const v=f.values;return panel('Dimension-name discovery versus kernel specialization',fieldCell('all input shapes',v.shapes,'shapes')+arrow('collect only SymbolicDim.name; deduplicate')+fieldCell('unique names',v.names,'names')+arrow('native parameter expressions/declarations')+`<div class="attention-two-columns">${fieldCell('G1._params',v.params,'params')}${fieldCell('Graph input declarations',v.decls,'decls')}</div>`+fieldCell('separate mo.custom tile parameters',v.tiles,'tiles'));
}
function capi(s,f){
 const v=f.values;return panel('Same native operation identity through pointer adapters',`<div class="attention-capi-route">${fieldCell('MAX wrapper',v.max,'max')}${arrow('_CAPIPtr: export pointer')}${fieldCell('named capsule',v.capsule,'capsule')}${arrow('extract operation-specific pointer')}${fieldCell('C ABI handle',v.c,'c')}${arrow('recover context; forOperation')}${fieldCell('MLIR Python wrapper',v.py,'py')}</div><div class="attention-nest"><b>OP1 · native GraphOp (pointer P)</b>${fieldCell('context reference',v.context,'context')}${v.region?'<div class="attention-nest"><b>R0 · existing region</b>'+fieldCell('body aliases',v.body,'body')+'</div>':''}</div>`+code('P is illustrative identity, not a measured address\nNo bytecode, graph clone or new tensor allocation'));
}
function deviceMap(s,f){
 const v=f.values;return panel('Placement and target description',fieldCell('tensor DeviceRef',v.tensor,'tensor')+arrow('look up by device label at compilation')+`<div class="attention-two-columns">${fieldCell('host descriptor',v.cpu,'cpu')}${fieldCell('accelerator descriptor',v.gpu,'gpu')}</div>`+fieldCell('MODULE1.mo.device_info_mapping',v.dict,'dict')+code('DeviceSpec/context HANDLES are supplied by session later\nMetadata attachment does not launch a kernel'));
}
function chains(s,f){
 const v=f.values;return panel('Dependency graph and retained references',fieldCell('chain / token',v.token||v.cpu,'token cpu')+`<div class="attention-two-columns">${fieldCell('CPU chain map entry',v.map||v.cpu,'map cpu')}${fieldCell('GPU lazy entry',v.gpu,'gpu')}</div>`+fieldCell('operation list / inputs',v.body||v.inputs,'body inputs')+fieldCell('readiness / chain-input mode',v.ready||v.flag,'ready flag')+code('A chain orders effects; it is not a user tensor\nSharing a token reference is not copying a matrix'));
}
function ports(s,f){
 const v=f.values;let diagram;
 if('q' in v)diagram=`<div class="attention-port-list">${['q','k','v'].map((key,i)=>panel('B0 argument '+i+' → '+key,cell('wrapper for existing SSA value',v[key]),'attention-port')).join('')}</div>`;
 else if('wrapper' in v)diagram=fieldCell('native SSA result',v.native,'native')+arrow('value pointer bridge + typed wrapper')+fieldCell('Graph API wrapper',v.wrapper,'wrapper')+fieldCell('full type fields',v.type,'type')+fieldCell('Python list',v.list,'list');
 else diagram=fieldCell('entry arguments',v.args,'args')+arrow('inspect last native type')+fieldCell('last type',v.last,'last')+`<div class="attention-two-columns">${fieldCell('chain-input flag',v.flag,'flag')}${fieldCell('adopted dependency',v.chain,'chain')}</div>`+fieldCell('public inputs',v.public,'public');
 return panel('One block, distinct argument/result identities',diagram)+fieldCell('numeric storage',v.data||v.buffer||'no numerical data in graph wrappers','data buffer');
}
function custom(s,f){
 const v=f.values,isOutput='signature' in v;
 const body=isOutput?v.body:v.body;
 return panel('G1.R0.B0 · symbolic operation sequence',code(body||'Existing graph entry values; body changes shown when reached'))+(!isOutput?panel('CUSTOM1 operand/result ports',`<div class="attention-three-columns">${cell('query operand','%q → input 0')}${cell('key operand','%k → input 1')}${cell('value operand','%v → input 2')}</div>`+arrow('native handles consume existing SSA values')+fieldCell('custom call',v.op,'op')+arrow('one SSA result')+fieldCell('%o wrapper',v.result,'result')):panel('Final callable contract',fieldCell('signature',v.signature,'signature')+fieldCell('public names',v.names,'names')+fieldCell('verification gate',v.verified,'verified')));
}
function validation(s,f){
 const v=f.values;return panel('Declaration lookup and compatibility gates',fieldCell('call contract',v.call,'call')+arrow('loaded path set → analysis cache')+fieldCell('selected declaration analysis',v.analysis,'analysis')+arrow('registration + device')+fieldCell('registry',v.registry,'registry')+fieldCell('selected candidate',v.candidate,'candidate')+`<div class="attention-two-columns">${fieldCell('INPUT mapping',v.inputs,'inputs')}${fieldCell('OUTPUT destination mapping',v.outputs,'outputs')}</div>`+fieldCell('compile-time parameter names/types',v.params,'params')+fieldCell('outcome',v.gate,'gate', String(v.gate||'').startsWith('ERROR')?'attention-error':'attention-pass'));
}
function ownership(s,f){
 const groups=new Map();s.scenario.nodes.forEach(n=>{if(!groups.has(n.owner))groups.set(n.owner,[]);groups.get(n.owner).push(n);});
 return panel('Owners and retained objects · arrows in values are references',`<div class="attention-owner-groups">${[...groups].map(([owner,nodes])=>panel(owner,nodes.map(n=>fieldCell(n.label,f.values[n.id],n.id)).join(''),'attention-owner')).join('')}</div>`);
}
function compiler(s,f){
 const v=f.values,target=s.scenario.view==='compiler-target',stages=target?['target setup','Mojo','library IR','cache','elaborate','device backend','device LLVM','device artifact','host LLVM dialect','host LLVM IR','host artifact','runtime plan']:['API','MO','MAP','MOGG','MGP','JIT'];
 return panel('Compiler representations · current stage highlighted',`<div class="attention-stage-rail">${stages.map(st=>`<b data-am-stage="${E(st)}" class="${f.scene.stage===st?'attention-active-stage':''}">${E(st)}</b>`).join('')}</div>`+`<div class="attention-two-columns">${target?fieldCell('input declarations / imports',v.source,'source'):fieldCell('retained original graph',v.original,'original')}${target?fieldCell('input specialization values',v.params,'params'):fieldCell('changing compiler clone C1',v.clone,'clone')}</div>`+(target?`${fieldCell('host and device backend selection',v.target,'target')}<div class="attention-two-columns">${panel('Device offload D1',fieldCell('device compiler representation',v.deviceIR,'deviceIR')+fieldCell('emitted GPU payload',v.deviceCode,'deviceCode'))}${panel('Host routine module',fieldCell('MLIR LLVM dialect',v.llvmDialect,'llvmDialect')+arrow('translateModuleToLLVMIR')+fieldCell('host llvm::Module',v.llvmIR,'llvmIR'))}</div>${fieldCell('host archive / model resource',v.archive,'archive')}`:fieldCell('registry / conversion state',v.registry,'registry')+fieldCell('routine / resource',v.routine,'routine')+fieldCell('init / execute plan',v.plan,'plan')));
}
function gates(s,f){
 const v=f.values;return panel('Worker checks resources before publishing M1',`<div class="attention-gate-route">${fieldCell('1 · weight interface',v.weights,'weights')}${arrow('setup runs on CPU worker')}${fieldCell('2 · setup owner / task result',v.setup,'setup')}${arrow('await fixed and extra init results')}${fieldCell('3 · initialization dependency',v.chain,'chain')}${fieldCell('diagnostic gate',v.error,'error')}${arrow('publish checked executable; native load awaits it')}${fieldCell('later execute admission',v.execute,'execute')}</div>`);
}
function buffers(s,f){
 const v=f.values;return panel('Retained code + fresh numerical tensor storage',fieldCell('model/resources reused',v.model,'model')+`<div class="attention-three-columns">${fieldCell('Q allocation',v.q,'q')}${fieldCell('K allocation',v.k,'k')}${fieldCell('V allocation',v.v,'v')}</div>`+arrow('runtime wrappers retain the SAME storage')+fieldCell('native ABI arguments',v.args,'args')+arrow('host routine submits device work')+fieldCell('device handles/stream',v.device,'device')+fieldCell('native completion chain',v.chain,'chain')+arrow('drop internal chain; preserve output device')+fieldCell('Python result',v.output,'output'));
}
function geometry(s,f){
 const a=f.scene,bx=a.blockXY?.[0]||0,by=a.blockXY?.[1]||0,hasGrid=Boolean(f.values.config),tiles=[];
 for(let y=0;y<2;y++)for(let x=0;x<4;x++)tiles.push(`<div class="attention-grid-tile ${hasGrid&&x===bx&&y===by?'attention-owned-tile':''}" data-am-block="${x},${y}"><b>block (${x},${y})</b><span>rows ${y*16}–${y*16+15}</span><span>cols ${x*8}–${x*8+7}</span>${x===bx&&y===by&&a.stored?'<strong>stored</strong>':''}</div>`);
 return panel('O[32,32] · eight disjoint [16,8] output tiles',`<div class="attention-output-grid">${tiles.join('')}</div><p>Each block: 32 threads. Logical ownership; no measured scheduling.</p>`)+panel('Selected block sees ALL keys',`<div class="attention-key-rail">${[0,1,2,3].map(j=>`<div class="${a.keyTile===j?'attention-current-tile':''}"><b>K/V tile ${j+1}</b><span>rows ${j*8}–${j*8+7}</span></div>`).join('')}</div>`+fieldCell('query rows + output columns',f.values.tile,'tile')+fieldCell('host argument slots / driver',f.values.args,'args')+fieldCell('stream/function',f.values.stream,'stream'));
}
function mma(s,f){
 const a=f.scene,loadedA=Boolean(a.aValues),loadedB=Boolean(a.bValues),rows=[],line=f.refs[0].line;
 for(let i=0;i<8;i++)rows.push(`<tr><th>${(a.chunk||0)*8+i}</th><td data-am-field="a">${loadedA?fmt(a.aValues[i]):'—'}</td><td data-am-field="b">${loadedB?fmt(a.bValues[i]):'—'}</td><td>${loadedB?fmt(a.terms[i]):'—'}</td></tr>`);
 return panel(`${a.product||'QK'} · one illustrative output cell inside [16,8] tile`,code(`A[M,K] × B[K,N] → C[M,N]\nBK=8 · ${a.K||32}/8 reduction slices`)+`<div class="attention-table-scroll"><table class="attention-matrix-table"><thead><tr><th>feature/key index</th><th>A loaded</th><th>B loaded</th><th>A×B</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`+`<div class="attention-two-columns">${fieldCell('retained accumulator',a.acc,line===305?'':'c')}${fieldCell('MMA pending result',a.pendingAcc,line===305?'c':'')}</div>`+fieldCell('slice partial sums',a.partials,line===306?'c':''))+panel('Fragment → shared result handoff',fieldCell('c_reg / d_reg state',f.values.c,'c')+arrow('store_d maps fragment layout')+fieldCell('out_sram .SHARED',f.values.shared,'shared')+fieldCell('barrier before read',f.values.barrier,'barrier')+arrow('copy complete logical result')+fieldCell('caller result',f.values.result,'result'));
}
function storage(s,f){
 const a=f.scene;return panel('Logical allocations and lifetimes',`<div class="attention-memory-pools">${panel('Full tensor storage',fieldCell('Q/K/V/O',f.values.global,'global'),'attention-global-pool')}${panel('Retained across key iterations',fieldCell('m/l/O_i',f.values.state,'state'),'attention-retained-pool')}${panel('Current iteration scratch',fieldCell('S/E',f.values.score,'score')+fieldCell('PV',f.values.pv,'pv'),'attention-scratch-pool')}${panel('matmul helper declarations',fieldCell('.LOCAL fragments / .SHARED staging',f.values.helper,'helper'),'attention-helper-pool')}</div>`+`<div class="attention-lifetime"><b>key iteration →</b>${[0,1,2,3].map(j=>`<span class="${a.keyTile===j?'attention-current-tile':''}">tile ${j+1}</span>`).join('')}<b class="attention-lived-state">m/l/O_i retained across all four</b><span class="attention-live-scratch">S/E/PV reused for current tile</span></div>`+fieldCell('views / final store',f.values.store||f.values.views,'store views')+code('S/E/PV: [16,8] current tile, not persistent [32,32]\n.GENERIC default ≠ proof of physical register residency\nLogical bytes ≠ measured peak GPU allocation'));
}
function matrix(label,rows,activeRows=[]){
 return panel(label,`<div class="attention-table-scroll"><table class="attention-matrix-table"><thead><tr><th>row</th>${rows[0].map((_,d)=>`<th>d${d}</th>`).join('')}</tr></thead><tbody>${rows.map((r,i)=>`<tr class="${activeRows.includes(i)?'attention-current-row':''}"><th>${i}</th>${r.map(x=>`<td>${fmt(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
}
function online(s,f){
 const a=f.scene,rows=[],max=x=>x===null?'−∞':fmt(x),line=f.refs[0].line;
 for(let i=0;i<4;i++)rows.push(`<tr data-am-key="${i}" class="${a.activeKeys?.includes(i)?'attention-current-row':''}"><th>key ${i}</th><td>${fmt(a.scores[i])}</td><td>${fmt(a.exponents[i])}</td><td>${fmt(a.weights[i])}</td><td>${fmt(a.V[i])}</td><td>${fmt(a.contributions[i])}</td></tr>`);
 const old=a.priorMass,newM=a.tileMass,total=(old||0)+(newM||0),terms=a.dotTerms;
 let eq='';if(terms)eq=panel('Current dot products across features',terms.map((t,r)=>code(`score for key ${(a.activeKeys||[])[r]}: `+t.map(p=>`${p.q}×${p.k}`).join(' + ')+' = '+t.reduce((sum,p)=>sum+p.product,0))).join('')+fieldCell('S · current score tile',f.values.S,'S'));
 const inputScene=panel('Tiny teaching matrices · each query has its own softmax',`<div class="attention-three-columns">${matrix('selected Q row'+a.queryIndex,[a.q],[0])}${matrix('K · compare with query',a.K,a.activeKeys)}${matrix('V · content to mix',a.V,a.activeKeys)}</div>`)+eq;
 const stateScene=panel('Retained state versus current scratch',`<div class="attention-state-values">${fieldCell('m₁ retained',max(a.m),'m','','data-am-m-slot')}${fieldCell('l₁ retained',a.l,'l','','data-am-l-slot')}${fieldCell('O_i (can be scratch during rescale)',a.O,'o','','data-am-o-slot')}</div><div class="attention-state-values">${fieldCell('m₂ current maximum',a.mNew??a.tileMax,line===348||line===349?'delta':'')}${fieldCell('l₂ scratch',f.values.prior,'prior')}${fieldCell('E scratch / normalized weights',f.values.E,'E')}</div>`);
 const historyScene=panel('Reading history · scores shown here are not retained full-kernel storage',`<div class="attention-table-scroll"><table class="attention-numeric-table"><thead><tr><th>key</th><th>Q·K</th><th>exp(S−m₂)</th><th>E/l₂</th><th>V row</th><th>weighted V</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`);
 const massScene=panel('Combine old and current mass on one maximum scale',`<div class="attention-two-columns">${fieldCell('alpha = exp(old_m−new_m)',a.alpha,line===355?'prior':'','attention-prior')}${fieldCell('new cumulative denominator',a.lNew,line===363?'prior':'','attention-current')}</div><div class="attention-mass-bar"><span class="attention-old-mass" style="width:${total?100*(old||0)/total:0}%"></span><span class="attention-new-mass" style="width:${total?100*(newM||0)/total:0}%"></span></div><div class="attention-two-columns">${fieldCell('prior mass (pink)',old,line===356?'prior':'','attention-prior')}${fieldCell('current tile mass (green)',newM,line===362?'delta':'','attention-current')}</div>`);
 const outputScratch=line>=370&&line<=378?fieldCell('m₁_delta · output-scale scratch',f.values.delta,'delta','','data-am-output-scale')+code('copy old m → subtract new m → exp → multiply O_i'):'';
 const outputScene=panel('Rescaled prior output + current weighted values',outputScratch+`<div class="attention-output-equation">${cell('old_O × alpha × old_l / new_l',a.oldContribution,'attention-prior')}<b>+</b>${fieldCell('current P_tile × V_tile',a.newContribution,'PV','attention-current')}<b>=</b>${fieldCell('O_i at this exact source step',a.O,'o')}</div><p class="attention-reference" data-am-reference>${a.reference?`Independent stable full-softmax output ${E(fmt(a.reference.O))} · ${a.match?'MATCH':'MISMATCH'} (browser only)`:'Reference output is checked after both key tiles.'}</p>`);
 const moreInputs='<details data-am-disclosure="inputs"><summary>Tiny Q/K/V matrices and dot products</summary>'+inputScene+'</details>';
 if(line>=366&&line<=378)return stateScene+outputScene+massScene+historyScene+moreInputs;
 if(line>=352&&line<=363)return stateScene+massScene+historyScene+moreInputs;
 return inputScene+stateScene+historyScene+massScene+outputScene;
}
const renderers={ir,insertion,context,location,type,attributes,contracts,parameters,capi,'device-map':deviceMap,chains,ports,'ir-insertion':custom,validation,ownership,compiler,'compiler-target':compiler,gates,buffers,geometry,mma,storage,online,dispatch:(s,f)=>panel('Compile-time specialization selects one implementation',fieldCell('registered interface',f.values.contract,'contract')+fieldCell('concrete parameters',f.values.params,'params')+fieldCell('rank assertion',f.values.rank,'rank')+arrow('same-storage tensor views')+fieldCell('views',f.values.views,'views')+arrow('comptime target branch')+fieldCell('selected implementation',f.values.branch,'branch')+fieldCell('work geometry',f.values.work,'work'))};
function shown(s){if(!s.root.isConnected||document.hidden||!s.root.checkVisibility({checkVisibilityCSS:true}))return false;const a=s.root.querySelector('[data-am-canvas]').getBoundingClientRect(),b=s.root.closest('.call-explanation-card-body')?.getBoundingClientRect()||{top:0,bottom:innerHeight};return a.bottom>Math.max(0,b.top)&&a.top<Math.min(innerHeight,b.bottom);}
function get(root){
 if(states.has(root))return states.get(root);
 const scenarios=JSON.parse(root.dataset.amScenarios),key=root.dataset.amCase||'actual',scenario=scenarios[key];
 const s={root,scenarios,key,scenario,index:Math.min(Number(root.dataset.amIndex)||0,scenario.frames.length-1),playing:false,motions:[],remaining:3200,last:performance.now()};
 states.set(root,s);render(s,false);return s;
}
function sourceLinks(s,f){
 const out=s.root.querySelector('[data-am-sources]');out.replaceChildren();
 f.refs.forEach((r,i)=>{if(i)out.append(' · ');const a=document.createElement('a');a.dataset.ref=r.key;a.dataset.line=r.line;a.href='../sources/'+r.file+'#L'+r.line;a.textContent=r.label+' · L'+r.line;out.append(a);});
}
function play(s,on){
 if(on===s.playing)return;
 if(on){s.playing=true;s.last=performance.now();s.motions.forEach(m=>m.play());active.add(s);schedule();}
 else{s.remaining=Math.max(0,s.remaining-(performance.now()-s.last));s.playing=false;s.motions.forEach(m=>{const time=m.currentTime;m.pause();m.currentTime=time;});active.delete(s);}
 s.root.dataset.amPlaying=String(on);s.root.querySelector('[data-am-action="play"]').textContent=on?'Pause':'Play';
}
function render(s,animate){
 const f=s.scenario.frames[s.index],el=s.root;el.dataset.amIndex=String(s.index);el.dataset.amCase=s.key;
 s.motions.forEach(m=>m.cancel());s.motions=[];
 const canvas=el.querySelector('[data-am-canvas]'),open=new Set([...canvas.querySelectorAll('[data-am-disclosure][open]')].map(d=>d.dataset.amDisclosure));canvas.innerHTML=(renderers[s.scenario.view]||ownership)(s,f)+objects(s,f);canvas.querySelectorAll('[data-am-disclosure]').forEach(d=>{d.open=open.has(d.dataset.amDisclosure);});
 // Bind highlights to field identity: equal values can belong to different objects.
 canvas.querySelectorAll('[data-am-field]').forEach(node=>{
  const changed=node.dataset.amField.split(' ').some(key=>f.focus.includes(key));
  node.dataset.amChanged=String(changed);if(changed)node.classList.add('attention-changed');
 });
 el.querySelector('[data-am-title]').textContent=`${s.index+1}. ${f.title}`;
 for(const key of ['input','action','output','why'])el.querySelector('[data-am-step-'+key+']').textContent=f[key];
 el.querySelector('[data-am-note]').textContent=s.scenario.note;sourceLinks(s,f);
 el.querySelector('[data-am-count]').textContent=`${s.index+1} / ${s.scenario.frames.length}`;
 el.querySelector('[data-am-action="back"]').disabled=s.index===0;el.querySelector('[data-am-action="step"]').disabled=s.index===s.scenario.frames.length-1;
 const slider=el.querySelector('[data-am-slider]');slider.max=s.scenario.frames.length-1;slider.value=s.index;slider.setAttribute('aria-label',f.title);
 el.querySelector('[data-am-scenario]').value=s.key;
 if(animate&&!reduced.matches){canvas.querySelectorAll('.attention-changed,.attention-current-row,.attention-current-tile,.attention-active-stage,.attention-cursor').forEach(node=>{
  const m=node.animate([{opacity:0.45,transform:'translateY(8px)'},{opacity:1,transform:'translateY(0)'}],{duration:780,fill:'both',easing:'ease-out'});s.motions.push(m);
 });}
 s.remaining=Number(el.querySelector('[data-am-speed]').value);s.last=performance.now();
}
let scheduled=false;
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(tick);}}
function tick(now){
 scheduled=false;for(const s of active){
  if(!shown(s)){play(s,false);continue;}
  if(now-s.last>=s.remaining){if(s.index===s.scenario.frames.length-1){play(s,false);continue;}s.index++;render(s,true);}
 }
 if(active.size)schedule();
}
function go(s,index,key=s.key){
 play(s,false);if(key!==s.key){s.key=key;s.scenario=s.scenarios[key];}
 s.index=Math.max(0,Math.min(s.scenario.frames.length-1,index));render(s,true);
}
document.addEventListener('click',e=>{
 const button=e.target.closest('[data-am-action],[data-am-seek]');if(!button)return;
 const root=button.closest('[data-attention-mechanism]');if(!root)return;
 const s=get(root),action=button.dataset.amAction;
 if(action==='play'){if(s.index===s.scenario.frames.length-1)go(s,0);play(s,!s.playing);return;}
 if(button.dataset.amSeek!==undefined)go(s,Number(button.dataset.amSeek),button.dataset.amCase);
 else go(s,action==='reset'?0:s.index+(action==='back'?-1:1));
});
document.addEventListener('input',e=>{if(e.target.matches('[data-am-slider]'))go(get(e.target.closest('[data-attention-mechanism]')),Number(e.target.value));});
document.addEventListener('change',e=>{
 if(e.target.matches('[data-am-scenario]'))go(get(e.target.closest('[data-attention-mechanism]')),0,e.target.value);
 if(e.target.matches('[data-am-speed]')){const s=get(e.target.closest('[data-attention-mechanism]'));s.remaining=Number(e.target.value);s.last=performance.now();}
});
new MutationObserver(()=>{
 document.querySelectorAll('.reader-card-window:not([hidden]) [data-attention-mechanism]').forEach(root=>get(root));
 for(const s of active)if(!shown(s))play(s,false);
}).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['hidden']});
document.addEventListener('visibilitychange',()=>{if(document.hidden)for(const s of active)play(s,false);});
document.addEventListener('toggle',()=>{for(const s of active)if(!shown(s))play(s,false);},true);
window.attentionMechanismSnapshot=root=>{
 const el=typeof root==='string'?document.querySelector(root):root;if(!el)return null;
 const s=get(el);return structuredClone({...s.scenario.frames[s.index],index:s.index,case:s.key,length:s.scenario.frames.length,playing:s.playing});
};
})();
