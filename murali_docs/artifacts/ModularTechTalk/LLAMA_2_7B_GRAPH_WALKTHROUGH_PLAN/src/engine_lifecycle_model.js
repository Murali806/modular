/* Deterministic source-order illustration; does not run MAX or model code. */
(function(root) {
  'use strict';
  class EngineLifecycleModel {
    constructor(data, options = {}) {
      this.data = data;
      this.options = {route:'cold', initialization:'device', ...options};
      const {route, initialization} = this.options;
      const lowerRoute = ['cold','cache','export','decline','prejit'].includes(route);
      const success = !['missing','prejit'].includes(route);
      const knownNames = !['read','import'].includes(route);
      const match = condition => !condition || ({
        trace:route !== 'read', read:route === 'read', 'artifact-route':!lowerRoute,
        'lower-route':lowerRoute, cache:route === 'cache', lower:lowerRoute && route !== 'cache',
        jit:lowerRoute && !['cache','prejit'].includes(route), 'skip-jit':!lowerRoute || route === 'cache',
        prejit:route === 'prejit', decline:route === 'decline', import:route === 'import',
        missing:route === 'missing', export:route === 'export', success,
        'lower-success':lowerRoute && success, 'existing-success':['read','import'].includes(route),
        'bad-layout':success && initialization === 'bad-layout',
        'virtual-known':success && initialization === 'virtual' && knownNames,
        'virtual-unknown':success && initialization === 'virtual' && !knownNames,
        real:success && initialization === 'device'
      })[condition] === true;
      this.events = data.events.filter(e => match(e.when));
      const terminal = this.events.findIndex(e => e.halt);
      if (terminal !== -1) this.events = this.events.slice(0, terminal + 1);
      this.index = -1;
    }
    step() { return this.seek(this.index + 1); }
    previous() { return this.seek(this.index - 1); }
    reset() { return this.seek(-1); }
    seek(index) { this.index=Math.max(-1,Math.min(this.events.length-1,Number(index))); return this.snapshot(); }
    jump(phase) { const i=this.events.findIndex(e=>e.phase===phase); return i<0?this.snapshot():this.seek(i); }
    snapshot() {
      const state = {graphState:'NOT CREATED', graphReady:false, artifactState:'NOT CREATED', compiledReady:false,
        compileStage:'NOT STARTED', mojoStage:'NOT STARTED', lockState:'FREE', modelState:'NOT INITIALIZED',
        initialized:false, exported:false, inferenceStage:'NOT STARTED', outputsReady:false, error:null};
      const created = new Set();
      for (const e of this.events.slice(0,this.index+1)) {
        e.create.forEach(id=>created.add(id));
        Object.keys(state).forEach(k=>{if(Object.hasOwn(e,k))state[k]=e[k];});
      }
      const current = this.events[this.index] || {phase:'graph',title:'Ready to trace the engine lifecycle',
        detail:'Use Step or Play. The default route builds and compiles a symbolic graph before initialization and inference.', active:[], refs:[]};
      return {...state, created:[...created], active:current.active, edge:current.edge || null,
        phase:current.phase,title:current.title,detail:current.detail,refs:current.refs,
        index:this.index,total:this.events.length,done:this.index===this.events.length-1,
        options:{...this.options},history:this.events.slice(0,this.index+1).map(e=>e.title)};
    }
  }
  root.EngineLifecycleModel=EngineLifecycleModel;
  if(typeof module!=='undefined' && module.exports)module.exports={EngineLifecycleModel};
})(typeof window==='undefined'?globalThis:window);
