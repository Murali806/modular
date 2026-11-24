/* Run with Node alone: no MAX, browser, model data or network required. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const model=require('./model.js'),out=path.resolve(__dirname,'../../link_htmls/llama_2_7b');
let snapshots=0,addressCases=0,softmaxCases=0;
const copy=x=>JSON.parse(JSON.stringify(x));
for(const file of fs.readdirSync(out).filter(f=>f.endsWith('.html'))){
  const match=fs.readFileSync(path.join(out,file),'utf8').match(/<script type="application\/json" id="lesson-data">(.*?)<\/script>/s);
  if(!match)continue;
  const data=JSON.parse(match[1]),original=JSON.stringify(data);
  for(const route of [{id:'main',steps:data.steps},...data.scenarios]){
    const expected=Object.fromEntries(data.actors.map(a=>[a.id,{state:'not created / no input yet',slots:[],facts:[]} ]));
    const history=[];
    for(const event of route.steps){Object.assign(expected,copy(event.changes));history.push(copy(expected));}
    for(const i of [...history.keys(),...history.keys()].reverse()){
      assert.deepEqual(model.snapshot(data,route.id,i),history[i],`${file}/${route.id}/${i}`);
      snapshots++;
    }
    const returned=model.snapshot(data,route.id,0);
    returned[data.actors[0].id].state='mutated caller copy';
    assert.deepEqual(model.snapshot(data,route.id,0),history[0],'snapshots must not share mutable state');
  }
  assert.equal(JSON.stringify(data),original,'reading must preserve lesson data');
}

const parameters=2*32000*4096+32*(4*4096**2+3*4096*11008+2*4096)+4096;
assert.equal(parameters,6738415616);
assert.equal(parameters*2,13476831232);
assert.deepEqual(model.kv(4096,1,128,2),{perToken:524288,logical:2147483648,reserved:2147483648,pages:32});
assert.deepEqual(model.kv(130,2,128,2),{perToken:524288,logical:136314880,reserved:268435456,pages:4});
assert.equal(model.kv(4096,1,128,4).logical,2**32);
assert.equal(model.kv(0,1,128,2).pages,0);
assert.throws(()=>model.kv(-1,1,128,2),RangeError);
assert.throws(()=>model.kv(1,1,0,2),RangeError);

const shape=[2,32,128,32,128],packed=shape.reduce((a,b)=>a*b),table=[7,2];
for(const token of [0,1,127,128,129,255])for(const layer of [0,5,31])for(const head of [0,3,31])for(const channel of [0,4,127])for(const kind of [0,1])for(const padding of [0,64]){
  const r=model.kvAddress({token,layer,head,channel,kind,pageStride:padding?packed+padding:-1});
  const coordinates=[kind,layer,token%128,head,channel];
  const raveled=coordinates.reduce((acc,n,i)=>acc*shape[i]+n,0);
  assert.equal(r.withinPageElements,raveled);
  assert.equal(r.physicalPage,table[Math.floor(token/128)]);
  assert.equal(r.byteOffset,(r.physicalPage*(packed+padding)+raveled)*2);
  addressCases++;
}
assert.throws(()=>model.kvAddress({token:256,layer:0,head:0,channel:0,kind:0}),RangeError);
assert.throws(()=>model.kvAddress({token:0,layer:0,head:0,channel:0,kind:0,pageStride:1}),RangeError);

function reference(scores,values){
  const max=Math.max(...scores),weights=scores.map(s=>Math.exp(s-max)),sum=weights.reduce((a,b)=>a+b,0);
  return weights.reduce((total,p,i)=>total+(p/sum)*values[i],0);
}
function close(actual,expected){assert.ok(Math.abs(actual-expected)<=1e-11*Math.max(1,Math.abs(expected)),`${actual} differs from ${expected}`);}
const cases=[[[2,1,3],[10,20,30]],[[1000,1001,-1000],[-4,8,1]],[[-1001,-1000],[2,-3]],[[7],[42]],[[1,1,1],[3,2,1]]];
let seed=19;
const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
for(let n=1;n<=16;n++)for(let j=0;j<5;j++)cases.push([Array.from({length:n},()=>random()*100-50),Array.from({length:n},()=>random()*20-10)]);
for(const [scores,values]of cases)for(let tile=1;tile<=Math.min(6,scores.length+1);tile++){
  const result=model.softmaxTiles(scores,values,tile),expected=reference(scores,values);
  close(result.at(-1).output,expected);
  for(const row of result){
    close(row.output,reference(scores.slice(0,row.start+tile),values.slice(0,row.start+tile)));
    assert.ok(row.l>0&&row.alpha>=0&&row.alpha<=1&&row.m>=row.old.m);
  }
  softmaxCases++;
}
close(model.softmaxTiles([2,1,3],[10,20,30],2).at(-1).output,24.20512484720024);
assert.throws(()=>model.softmaxTiles([1],[1],0),RangeError);
assert.throws(()=>model.softmaxTiles([1,2],[1],1),RangeError);
assert.throws(()=>model.softmaxTiles([NaN],[1],1),RangeError);
console.log(JSON.stringify({snapshots,addressCases,softmaxCases,parameters,result:'PASS'},null,2));
