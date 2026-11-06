"use strict";
const assert=require("assert/strict"),fs=require("fs"),path=require("path"),crypto=require("crypto");
const {WorkbenchModel,TARGETS}=require("./compiler_workbench_model.js");
const task=path.resolve(__dirname,".."), checks=[];
const check=(name,fn)=>{fn();checks.push(name);};
const last=m=>m.seek(m.frames.length-1);
check("Graph contract is a prerequisite, not execution",()=>{
 const m=new WorkbenchModel("graph",{seal:false});let s=last(m);assert.equal(s.sealed,false);assert.equal(s.compileCalls,0);
 s=m.intervene("seal");assert.equal(s.contract,true);s=last(m);assert.equal(s.sealed,true);assert.equal(s.compileCalls,0);
});
check("Original module retained; DCE precedes Mojo emission",()=>{
 for(const fusion of [true,false]){const m=new WorkbenchModel("lower",{fusion});let s=last(m);assert.equal(s.clone,true);assert.equal(s.fused,fusion);
 assert.ok(m.frames.findIndex(f=>f.stage==="dce")<m.frames.findIndex(f=>f.stage==="mojo"));
 assert.ok(m.frames.findIndex(f=>f.stage==="device-finalize")<m.frames.findIndex(f=>f.stage==="dce"));assert.equal(s.deviceFinalized,true);
 assert.deepEqual(s.history,["MO","MAP","MOGG","MGP","Mojo emission"]);}
});
check("SIMD specialization changes grouping, preserves results",()=>{
 for(const width of [1,4,8]){const s=last(new WorkbenchModel("mojo",{width}));assert.equal(s.bundles,8/width);assert.deepEqual(s.results,[11,12,13,14,15,16,17,18]);assert.equal(s.concrete,true);}
});
check("Conflicting borrow blocks; repair mutates same allocation and moves owner",()=>{
 const m=new WorkbenchModel("mojo",{lab:"ownership"});let s=last(m);assert.equal(s.blocked,true);assert.deepEqual(s.values,[1,2,3,4]);
 m.intervene("release");s=last(m);assert.equal(s.blocked,false);assert.deepEqual(s.values,[11,2,3,4]);assert.equal(s.owner,"callee");assert.equal(s.borrow,null);
});
check("Target-dependent emission, inspection and failure gates",()=>{
 for(const target of Object.keys(TARGETS))for(const emission of ["binary","assembly"]){const m=new WorkbenchModel("target",{target,emission});let s=last(m);assert.equal(s.binary,emission==="binary");
 assert.equal(s.object,emission==="binary"&&target!=="cuda");assert.equal(s.assembly,target==="cuda"||emission==="assembly");
 if(s.binary){s=m.intervene("inspect");assert.equal(s.binary,true);assert.equal(s.inspect,true);assert.equal(s.target.binary,TARGETS[target].binary);}}
 for(const failure of ["invalidArch","missingToolkit"]){const m=new WorkbenchModel("target",{target:"cuda",[failure]:true});const s=last(m);assert.equal(s.kind,"error");assert.equal(s.binary,false);assert.ok(s.error);}
});
check("Weights gate initialization independently of MEF export",()=>{
 for(const exported of [true,false]){const m=new WorkbenchModel("package",{missingWeight:true,export:exported});let s=last(m);assert.equal(s.mef,true);assert.equal(s.exported,exported);assert.equal(s.initialized,false);assert.equal(s.weights,1);
 m.intervene("repair");s=last(m);assert.equal(s.weights,2);assert.equal(s.initialized,true);assert.equal(s.codeLoaded,true);}
});
check("Repeated calls reuse resident code and count only explicit transfers",()=>{
 for(const target of Object.keys(TARGETS))for(const resident of [true,false])for(const copyResult of [true,false]){
 const m=new WorkbenchModel("runtime",{target,resident,copyResult,calls:3});const s=last(m),cpu=target==="cpu";
 assert.equal(s.copies,cpu?0:3*((resident?0:1)+(copyResult?1:0)));assert.equal(s.dispatches,3);assert.equal(s.model,"M1");assert.equal(s.session,"S1");
 assert.equal(s.codeResident,true);assert.equal(s.weightsResident,true);assert.deepEqual(s.results,[26,30,34,38]);assert.equal(s.hostResults.length,!cpu&&copyResult?4:0);
 m.intervene("again");const next=last(m);assert.equal(next.dispatches,4);assert.equal(next.model,s.model);
 }
});
check("Seek/back/reset and scenario rebuilding are deterministic",()=>{
 for(const kind of ["graph","lower","mojo","target","package","runtime"]){const m=new WorkbenchModel(kind);const first=m.snapshot();for(let i=0;i<m.frames.length;i++){const s=m.seek(i);m.reset();assert.deepEqual(m.seek(i),s);if(i){m.back();assert.equal(m.snapshot().index,i-1);}}
 assert.deepEqual(m.reset(),first);m.configure({target:"cpu"});assert.equal(m.snapshot().index,0);}
});
check("Scenario reset clears repairs, inspection and repeated-call history",()=>{
 const m=new WorkbenchModel("target");last(m);m.intervene("inspect");m.configure({target:"cpu"});assert.equal(last(m).inspect,false);
 const run=new WorkbenchModel("runtime");last(run);run.intervene("again");assert.equal(last(run).dispatches,2);run.configure({resident:true});assert.equal(last(run).dispatches,1);
 const pkg=new WorkbenchModel("package",{missingWeight:true});last(pkg);pkg.intervene("repair");pkg.configure({missingWeight:true});assert.equal(last(pkg).initialized,false);
});
const manifest=JSON.parse(fs.readFileSync(path.join(task,"compiler_workbench_manifest.json")));
check("Embedded full files and complete definition spans match pinned snapshots",()=>{
 const html=fs.readFileSync(path.join(task,manifest.page),"utf8"),data=JSON.parse(html.match(/<script id="compiler-workbench-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
 const sha=text=>crypto.createHash("sha256").update(text).digest("hex");
 for(const f of manifest.files){const raw=fs.readFileSync(path.join(task,"sources",f.path));assert.equal(sha(raw),f.sha256);assert.equal(data.files[f.path].text,raw.toString());}
 for(const r of Object.values(manifest.references)){const lines=data.files[r.file].text.split(/(?<=\n)/);assert.equal(sha(lines.slice(r.start-1,r.end).join("")),r.sha256);assert.ok(r.start<=r.line&&r.line<=r.end);}
 for(const kind of ["graph","lower","mojo","target","package","runtime"])for(const frame of new WorkbenchModel(kind).frames)for(const ref of frame.refs)assert.ok(data.references[ref],ref);
});
fs.writeFileSync(path.join(task,"validation/compiler_workbench_model.json"),JSON.stringify({evidence:"CPU-RUN · JavaScript teaching model only",revision:manifest.revision,checks},null,2)+"\n");
console.log("Compiler model/source checks passed:",checks.length);
