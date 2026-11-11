/* Deterministic teaching state. No compiler, driver or GPU is invoked. */
(function(root) {
  'use strict';
  const TARGETS={
    cuda:{target:'NVIDIA CUDA',IMAGE:'cubin bytes',MODULE_API:'cuModuleLoadDataEx',FUNCTION_API:'cuModuleGetFunction',LAUNCH_API:'cuLaunchKernelEx',MODULE_TYPE:'CUmodule',FUNCTION_TYPE:'CUfunction'},
    hip:{target:'AMD HIP',IMAGE:'hsaco bytes',MODULE_API:'hipModuleLoadData',FUNCTION_API:'hipModuleGetFunction',LAUNCH_API:'hipModuleLaunchKernel',MODULE_TYPE:'hipModule_t',FUNCTION_TYPE:'hipFunction_t'},
    cpu:{target:'CPU',IMAGE:'no GPU image',MODULE_API:'no GPU module load',FUNCTION_API:'no GPU function lookup',LAUNCH_API:'host invocation',MODULE_TYPE:'none',FUNCTION_TYPE:'none'}
  };
  const clone=x=>JSON.parse(JSON.stringify(x));
  const LENSES=[
    ['bytes','1 · Object bytes'],['namespace','2 · Register / reuse'],
    ['linking','3 · Link / patch'],['symbols','4 · Name → address'],
    ['invoke','5 · Host call'],['device','6 · GPU handles'],
    ['launch','7 · Launch arguments'],['arithmetic','8 · Load / +1 / store'],
    ['completion','9 · Completion / copy']
  ];
  const hex=n=>'0x'+n.toString(16).padStart(16,'0');
  const FIXUP=(()=>{
    const target=0x00007f1210000400n,addend=0x30n,value=target+addend;
    return {kind:'Pointer64',target:'runtime_helper',targetAddress:hex(target),
      addend:'0x30',slotAddress:'0x00007f1200200020',value:hex(value),
      before:Array(8).fill('00'),
      after:Array.from({length:8},(_,i)=>Number((value>>BigInt(i*8))&255n).toString(16).padStart(2,'0'))};
  })();
  function lens(e){
    if(['start','caller','arguments','hash','copy'].includes(e.id))return 'bytes';
    if(['lookup','mangle','query-ready','resolve','append','next-symbol','append-second','missing'].includes(e.id))return 'symbols';
    if(e.section==='2.8')return 'linking';
    if(['ready','lifetime','execute','host-call','repeat','repeat-cache'].includes(e.id))return 'invoke';
    if(['pack','launch','repeat-pack','repeat-launch'].includes(e.id))return 'launch';
    if(['cpu-read','cpu-add','cpu-result','work-read','work-add','device-work','repeat-read','repeat-add','repeat-store'].includes(e.id))return 'arithmetic';
    if(['completion','transfer','repeat-completion','repeat-transfer'].includes(e.id))return 'completion';
    if(['2.13','2.14'].includes(e.section))return 'device';
    return 'namespace';
  }
  function detail(s,o,e){
    const call=s.repeat?2:1,input=s.repeat?[10,11,12,13]:[0,1,2,3],
      loaded=['loaded','added','stored'].includes(s.computeStage),
      added=['added','stored'].includes(s.computeStage),stored=s.computeStage==='stored',
      output=input.map(x=>x+1),gpu=o.target!=='cpu',
      inputAddress=call===1?'0x80010000':'0x80030000',
      outputAddress=call===1?'0x80020000':'0x80040000';
    return {
      lens:lens(e),object:{size:12288,magic:['7f','45','4c','46'],copy:s.copy,
        requestedNames:['add_one_entry',o.route==='missing'?'missing_entry':'shape_entry'],
        retained:s.bufferRetained,copyReleased:s.copyReleased,sections:['.text','.rodata','.symtab','.rela.rodata']},
      orc:{state:s.error?'missing':s.symbolStage,required:'Ready',namespace:s.namespace,
        mutex:s.mutex,reused:s.reused,query:s.queryReady,
        canReturn:!s.error&&(s.symbolStage==='Ready'||o.route==='debug'&&s.generator),
        interned:s.interned,name:s.lookup||'add_one_entry'},
      link:{graph:s.linkGraph,pruned:s.pruned,allocated:s.memory!=='empty',
        copied:s.sectionCopied||o.route==='reuse',external:s.externalResolved||o.route==='reuse',
        patched:s.relocationPatched||o.route==='reuse',
        finalized:s.finalized||o.route==='reuse',fixup:clone(FIXUP),
        textAddress:'0x00007f1200100000',dataAddress:'0x00007f1200200000',
        textProtection:s.finalized||o.route==='reuse'?'R-X':'RW-',
        dataProtection:s.finalized||o.route==='reuse'?'R--':'RW-'},
      table:{entries:['add_one_entry','shape_entry'].slice(0,s.table).map((name,i)=>({
        index:i,name,address:i?'0x00007f1200100080':'0x00007f1200100000'})),
        nextIndex:s.table,selected:s.executing?0:null},
      device:{active:gpu&&s.deviceImage,key:s.deviceKeyChecked,
        module:s.deviceModule,function:s.deviceFunction,cacheHit:o.deviceCache==='hit'||s.repeat,
        cached:s.deviceCached||o.deviceCache==='hit',
        image:TARGETS[o.target].IMAGE,
        keyFields:['module=demo_gpu_module','function=add_one_gpu','debug=""','max_shared=-1','optimization=-1']},
      launch:{packed:s.launchPacked,submitted:s.submitted,complete:s.complete,
        call,grid:[1,1,1],block:[32,1,1],shared:0,stream:'S0',
        arguments:[
          {index:0,slot:'arg0',slotAddress:'0x00007fff00100100',value:inputAddress,referent:'B'+call},
          {index:1,slot:'arg1',slotAddress:'0x00007fff00100108',value:outputAddress,referent:'O'+call}]},
      data:{call,inputId:'B'+call,outputId:'O'+call,hostId:'H'+call,
        input,registers:loaded?(added?output:input):[null,null,null,null],
        output:stored?output:[null,null,null,null],expected:output,
        loaded,added,stored,complete:s.complete,hostCopy:s.hostCopy,
        inputAddress,outputAddress,location:gpu?'GPU device memory':'CPU host memory'},
      resources:{bytes:s.bufferRetained||o.route==='reuse',namespace:s.namespace,
        code:s.symbolStage==='Ready'||s.generator,
        table:s.table,device:s.deviceFunction,call:s.executing?call:0}
    };
  }
  function substitute(value,options){
    if(typeof value==='string')return value.replace(/@([A-Z_]+)@/g,(_,key)=>key==='SECOND_SYMBOL'?(options.route==='missing'?'missing_entry':'shape_entry'):TARGETS[options.target][key]||key);
    if(Array.isArray(value))return value.map(x=>substitute(x,options));
    if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,substitute(v,options)]));
    return value;
  }
  function include(e,o){const valid=o.route!=='missing',gpu=valid&&o.target!=='cpu';return {
    all:true,ordinary:o.route!=='debug',cold:o.route==='cold'||o.route==='missing',reuse:o.route==='reuse',debug:o.route==='debug',
    valid,missing:o.route==='missing',cpu:valid&&o.target==='cpu',gpu,'device-hit':gpu&&o.deviceCache==='hit','device-miss':gpu&&o.deviceCache==='miss'
  }[e.route]===true;}
  function frames(data,options={}){
    const o={route:'cold',target:'cuda',deviceCache:'miss',...options};
    if(!['cold','reuse','debug','missing'].includes(o.route)||!TARGETS[o.target]||!['miss','hit'].includes(o.deviceCache))throw Error('Invalid loader scenario');
    let state={phase:'before initialization',primitive:false,arguments:false,hash:false,copy:false,mutex:false,namespace:o.route==='reuse',dependencies:o.route==='reuse',bufferRetained:false,interface:o.route==='reuse',unit:o.route==='reuse',materialized:false,memory:o.route==='reuse'?'linked':'empty',gate:o.route==='reuse'?'prior code retained':'not entered',lookup:'',interned:false,resolved:0,table:0,loadStatus:'pending',error:false,lifetime:false,executing:false,hostInvoked:false,driver:'inactive',deviceImage:false,deviceModule:!!(o.target!=='cpu'&&o.deviceCache==='hit'),deviceFunction:!!(o.target!=='cpu'&&o.deviceCache==='hit'),launchPacked:false,submitted:false,deviceWritten:false,result:false,complete:false,repeat:false,reused:false,debugFile:false,debugShared:false,generator:false,
      copyReleased:false,linkGraph:false,pruned:false,sectionCopied:false,externalResolved:false,
      relocationPatched:false,relocationComputed:false,finalized:false,symbolStage:o.route==='reuse'?'Ready':'unregistered',
      queryReady:false,deviceKeyChecked:false,deviceCached:false,computeStage:'none',hostCopy:false};
    const result=[];
    const make=(e)=>{const event=substitute(e,o);if(o.target==='hip'&&event.source==='cuda'){const hipRow=event.tree.find(row=>row[1]==='hip');if(hipRow){event.source='hip';event.line=hipRow[2];}}return {event,state:clone(state),options:clone(o),detail:detail(state,o,event)};};
    result.push(make({id:'start',section:'2.1',title:'Ready to follow compiled bytes into callable entries',source:'load',line:85,inputs:'A0 compiled object bytes; context callable table is empty.',outputs:'No loading steps advanced.',why:'Keep host-code loading, GPU-code loading and per-call computation distinct.',tree:[],focus:'input',evidence:'SOURCE'}));
    for(const raw of data.events){if(!include(raw,o))continue;state={...state,...substitute(raw.changes,o)};
      // The debug branch has no objectBuffers retain: its local BufferRef dies
      // when load returns. Its loaded shared-library code remains available.
      if(raw.id==='ready'&&o.route==='debug')state.copyReleased=true;
      result.push(make(raw));}
    return result;
  }
  class Timeline{
    constructor(data,options){this.data=data;this.reset(options);}
    reset(options){this.options={route:'cold',target:'cuda',deviceCache:'miss',...options};this.frames=frames(this.data,this.options);this.index=0;return this.snapshot();}
    seek(index){this.index=Math.max(0,Math.min(this.frames.length-1,Number(index)));return this.snapshot();}
    next(){return this.seek(this.index+1);}
    back(){return this.seek(this.index-1);}
    snapshot(){return {...clone(this.frames[this.index]),index:this.index,total:this.frames.length-1,terminal:this.index===this.frames.length-1};}
  }
  const api={frames,Timeline,TARGETS,LENSES,FIXUP};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.KernelLoaderModel=api;
})(typeof window==='undefined'?globalThis:window);
