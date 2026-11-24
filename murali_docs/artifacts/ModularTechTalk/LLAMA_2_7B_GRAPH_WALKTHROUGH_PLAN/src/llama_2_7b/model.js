/* Pure, deterministic teaching state. No MAX execution is simulated as measured. */
(function(root){'use strict';
  function route(data,id){return id==='main'?data.steps:(data.scenarios.find(s=>s.id===id)?.steps||data.steps);}
  function initial(data){return Object.fromEntries(data.actors.map(a=>[a.id,{state:'not created / no input yet',slots:[],facts:[]} ]));}
  function snapshot(data,id,index){
    const events=route(data,id),state=initial(data);
    for(let i=0;i<=Math.max(0,Math.min(index,events.length-1));i++)for(const [key,value]of Object.entries(events[i].changes))state[key]=JSON.parse(JSON.stringify(value));
    return state;
  }
  function kv(tokens,batch,pageSize,bytes){if(!Number.isInteger(tokens)||tokens<0||[batch,pageSize,bytes].some(n=>!Number.isInteger(n)||n<1))throw new RangeError('Invalid cache dimensions');const rounded=Math.ceil(tokens/pageSize)*pageSize;return {perToken:2*32*32*128*bytes,logical:tokens*batch*2*32*32*128*bytes,reserved:rounded*batch*2*32*32*128*bytes,pages:Math.ceil(tokens/pageSize)*batch};}
  function kvAddress({token,layer,head,channel,kind,pageSize=128,bytes=2,pageStride=-1,table=[7,2]}){
    const packedPageElements=2*32*pageSize*32*128;
    const bounds=[[token,table.length*pageSize],[layer,32],[head,32],[channel,128],[kind,2]];
    if(bounds.some(([n,max])=>!Number.isInteger(n)||n<0||n>=max)||!Number.isInteger(pageSize)||pageSize<1||!Number.isInteger(bytes)||bytes<1||table.some(n=>!Number.isInteger(n)||n<0)||!Number.isInteger(pageStride)||(pageStride!==-1&&pageStride<packedPageElements))throw new RangeError('Invalid illustrative cache coordinates');
    const logicalPage=Math.floor(token/pageSize),slot=token%pageSize,physicalPage=table[logicalPage],stride=pageStride===-1?packedPageElements:pageStride;
    const withinPageElements=(((kind*32+layer)*pageSize+slot)*32+head)*128+channel;
    const elementOffset=physicalPage*stride+withinPageElements;
    return {logicalPage,physicalPage,slot,stride,packedPageElements,withinPageElements,elementOffset,byteOffset:elementOffset*bytes};
  }
  function softmaxTiles(scores,values,tile){if(!scores.length||scores.length!==values.length||!Number.isInteger(tile)||tile<1||[...scores,...values].some(x=>!Number.isFinite(x)))throw new RangeError('Invalid softmax example');let m=-Infinity,l=0,a=0;const history=[];for(let start=0;start<scores.length;start+=tile){const old={m,l,a};const s=scores.slice(start,start+tile),v=values.slice(start,start+tile),next=Math.max(m,...s),alpha=Number.isFinite(m)?Math.exp(m-next):0;const p=s.map(x=>Math.exp(x-next));l=alpha*l+p.reduce((x,y)=>x+y,0);a=alpha*a+p.reduce((sum,x,i)=>sum+x*v[i],0);m=next;history.push({start,old,m,l,a,alpha,p,output:a/l});}return history;}
  const api={route,initial,snapshot,kv,kvAddress,softmaxTiles};root.LlamaModel=api;if(typeof module!=='undefined')module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
