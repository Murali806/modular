/* Reuse the workbench's stable SVG actors and pausable interpolation. */
(function(root) {
  'use strict';
  const {fmt,A,B,TA,TB,Q,K,V}=root.CompilerExamplesModel;
  const point=(x,y)=>({x,y});
  const actor=(id,kind,label,x,y,w=160,h=90,color='artifact',extra={})=>({id,kind,label,x,y,w,h,color,opacity:1,...extra});
  class ExampleVisual extends root.CompilerWorkbenchVisual {
    targets(s) {
      const list=[],add=(...args)=>list.push(actor(...args)),lines=m=>m.map(row=>'['+row.map(fmt).join(', ')+']');
      if(this.kind==='build') {
        add('python','file',s.filename,45,139,240,207,'python',{tag:'ORIGINAL SOURCE',lines:[s.id==='epilogue'?'(A @ B + 1) × 2':s.id==='addition'?'X + 1':s.id==='matmul'?'A @ B; algorithm param':'softmax(Q Kᵀ) V','Graph + custom_extensions','out_types supplied']});
        if(s.graph) add('G1','file','G1 · '+(s.verified?'verified':'recording'),351,139,450,267,'mlir',{tag:'SYMBOLIC · NOT EXECUTABLE',lines:s.signature,origin:point(165,236),place:'caller graph'});
        if(s.ops) {
          if(s.id==='epilogue'&&s.ops===3) {
            add('matmul-op','tile','A @ B',374,277,119,45,'mlir',{origin:point(175,238),place:'G1'});
            add('custom-op','tile','add_one',515,277,136,45,'mojo',{origin:point(530,300),place:'G1'});
            add('multiply-op','tile','× 2',673,277,105,45,'backend',{origin:point(182,241),place:'G1'});
          } else add('custom-op','tile',s.name,374,277,405,45,'mojo',{origin:point(185,243),place:'G1'});
        }
        if(s.graph) add('signature','handle','out_types · explicit shape / dtype / device',374,338,405,40,'engine',{origin:point(270,327),place:'G1 output contract'});
        if(s.verified)add('verified','stamp','Graph.output → VERIFIED',477,414,300,39,'sync',{origin:point(760,385),place:'output contract'});
        add('mojo-source','file',s.id==='matmul'?'MatrixMultiplication':s.id==='attention'?'FusedAttention':s.id==='epilogue'?'AddOneElementwise':'AddOne',878,139,273,225,'mojo',{tag:s.package?'P1 · .MOJOC / MLIR BYTECODE':'ORIGINAL .MOJO',lines:['@register(name)',s.routine,s.registered?'imported before custom call':'parameters retained'],place:'registered source'});
        if(s.registered)add('name-link','handle','same registration string',878,402,273,47,'mojo',{origin:point(569,299),place:'registration match'});
        if(s.id==='matmul'&&s.inputs)add('early-buffers','crate','A / B buffers already created',45,380,240,69,'buffer',{tag:'BEFORE GRAPH',place:'caller owns data'});
      } else if(this.kind==='compile') {
        add('G1','file','G1 · retained',45,140,207,215,'python',{tag:'VERIFIED GRAPH',lines:[s.signature[0],s.signature[1],'caller-owned source IR'],place:'caller'});
        add('mojo-source','file','Original custom Mojo',45,380,207,69,'mojo',{tag:'REGISTERED SOURCE',place:'caller source'});
        add('S1','handle','S1 · retained context',506,407,196,43,'engine',{place:'session'});
        if(s.clone) {
          const levels=['MO · symbolic op','MAP · bodies','MOGG · kernel calls','MGP · buffer plan','MGP · finalized graph'];
          if(s.level<5) {
            add('C1','file','C1 · '+levels[Math.min(4,s.level)],326,140,376,243,'mlir',{tag:'COMPILER-OWNED CLONE',lines:[s.level<3?'types / def-use preserved':'buffers + dependency operands'],origin:point(149,241),place:'compiler'});
            if(s.id==='epilogue'&&s.level===0) {
              add('op-matmul','tile','MATMUL',343,240,106,42,'mlir',{origin:point(173,226),place:'C1'});
              add('op-add','tile','+ 1',461,240,97,42,'mojo',{origin:point(173,254),place:'C1'});
              add('op-mul','tile','× 2',571,240,106,42,'backend',{origin:point(173,270),place:'C1'});
            } else if(s.id==='epilogue'&&s.fusion) {
              add('fused-body','crate',s.level<2?'MATMUL | epilogue +1, ×2':'call: MATMUL + epilogue',343,235,335,68,'mojo',{tag:'ELIGIBLE FUSION SKETCH',origin:point(515,261),place:'grouped body'});
            } else {
              add('body','tile',s.id==='epilogue'?s.level===1?'MATMUL body → intermediate':s.level===2?'kernel call: MATMUL':'primitive: MATMUL':s.level===0?s.name:s.level===1?'body: '+s.routine:s.level===2?'kernel call → custom body':'primitive → custom kernel',343,237,335,43,'mojo',{origin:point(153,282),place:'C1 operation'});
              if(s.id==='epilogue')add('separate','tile','separate epilogue +1 / ×2',343,289,335,40,'backend',{place:'separate-body illustration'});
            }
            if(s.level>=3)add('plan-slots','handle','input refs | dependency | output ref',343,335,335,33,'buffer',{origin:point(270,330),place:'MGP resource plan'});
          } else {
            add('C1','file','C1 · MGP plan',326,388,166,65,'mlir',{tag:'RETAINED DEPENDENCIES',place:'compiled plan'});
            add('KIR1','file',s.level===5?'Specialized Mojo MLIR':'LLVM dialect / MLIR',326,140,376,230,'mojo',{tag:s.level===5?'LIT / KGEN / POP':'LOWERED KERNEL IR',lines:[s.routine,'target = '+(s.cpu?'cpu':'gpu'),s.id==='matmul'?'algorithm = '+s.algorithm:s.id==='attention'?`BN=${s.params.BN}; BD=${s.params.BD}`:'dtype / width specialized'],origin:point(499,259),place:'specialized kernel representation'});
            add('params','tile',s.id==='matmul'?s.params.grid:s.id==='attention'?s.params.grid:'typed native operation',343,312,335,38,'backend',{place:'target specialization'});
          }
        }
        if(s.level>=6) {
          add('LLVM1','file',s.code?'LLVM IR retained':'LLVM IR module',771,s.code?388:143,187,s.code?65:164,'mlir',{tag:s.code?'TRANSLATED':'DISTINCT LLVM MODULE',lines:s.code?[]:['native codegen input','not MLIR dialect'],origin:point(534,227),place:'LLVM representation'});
        }
        if(s.code) {
          add('target-tool','tool',s.cpu?'HostBackend':s.target==='cuda'?'NVPTXBackend':'AMDGPUBackend',771,143,187,83,'backend',{tag:'SELECTED BACKEND',place:'target code generation'});
          add('BIN1','binary',s.cpu?'.o → native code':s.target==='cuda'?'PTX → cubin':'.o → ELF / hsaco',771,256,187,99,'artifact',{tag:'COMPILED BYTES',lines:['code / symbols'],origin:point(862,178),place:'compiled target code'});
        }
        if(s.mef)add('CM1','crate','CM1 · MEF bytes',1023,142,136,166,'artifact',{tag:'PREPARED ARTIFACT',lines:['code + plan','in memory','no disk export'],origin:point(862,306),via:[point(991,315)],place:'compiled artifact'});
        if(s.loading&&!s.initialized)add('H1','handle','H1 · callable refs',1023,343,136,85,'runtime',{tag:'LOADING MODULES',origin:point(1090,272),place:'initialization gate'});
        if(s.initialized)add('M1','chip','M1 · executable',1023,347,136,107,'engine',{tag:'INITIALIZED',lines:['S1 retained'],origin:point(1090,272),place:'initialized model'});
      } else {
        if(s.id==='epilogue')this.node.querySelector('[data-temporary-edges]').toggleAttribute('hidden',s.fusion||!s.traffic);
        add('M1','chip','M1 / S1',369,142,225,89,'engine',{tag:'INITIALIZED / RETAINED',lines:['same compiled code'],place:'resident model'});
        add('BIN1','binary','BIN1 · resident code',369,243,225,34,'artifact',{place:'resident compiled code'});
        if(s.job)add('JOB1','descriptor','execute(input refs)',369,290,225,76,'runtime',{tag:'DEPENDENCY DISPATCH',origin:point(284,219),place:'native execution'});
        if(s.job)add('device-interface','tool',s.cpu?'Native CPU execution':'DeviceContext / driver API',369,384,225,42,s.cpu?'runtime':'driver',{origin:point(483,335),place:s.cpu?'CPU native call':'host device interface'});
        if(s.ready)add('result-ref','handle','O1 · ready Buffer',369,441,225,67,'runtime',{tag:'RESULT REFERENCE',origin:point(1000,519),via:[point(1128,566),point(606,566)],place:'returned buffer handle'});
        this.node.querySelector('[data-wire="upload"]').toggleAttribute('hidden',s.cpu);
        this.slots('host-output',s.hostRead?s.computed:[]);this.slots('target-output',s.computed);
        this.text('[data-kernel-label]',s.id==='attention'?'LOCAL SCORE / RUNNING STATE':s.id==='matmul'&&!s.cpu&&s.algorithm==='tiled'?'BLOCK SHARED TILES':'NATIVE ARITHMETIC');
        if(s.hostInputs) {
          if(s.id==='addition') {
            [0,.25,.5,.75].forEach((v,i)=>add('H-X-'+i,'cell',fmt(v),49+i*62,180,54,40,'buffer',{place:'host input'}));
          } else if(s.id==='attention') {
            add('H-Q','file','Q1 · [1, 0]',45,164,245,76,'buffer',{tag:'ONE QUERY · TINY',place:'host input'});
            add('H-K','file','K1',45,261,115,146,'buffer',{tag:'FOUR KEYS',lines:['[1,0]; [0,1]','[3,0]; [2,1]'],place:'host input'});
            add('H-V','file','V1',175,261,115,146,'buffer',{tag:'FOUR VALUES',lines:['[1,0]; [0,2]','[2,1]; [1,3]'],place:'host input'});
          } else {
            const a=s.id==='matmul'?TA:A,b=s.id==='matmul'?TB:B;
            add('H-A','file','A1 · '+(s.id==='matmul'?'2×4':'2×2'),45,164,245,117,'buffer',{tag:'TINY HOST ARRAY',lines:lines(a),place:'host input'});
            add('H-B','file','B1 · '+(s.id==='matmul'?'4×2':'2×2'),45,299,245,117,'buffer',{tag:'TINY HOST ARRAY',lines:s.id==='matmul'?['[1,2]; [0,1]','[2,0]; [1,1]']:lines(b),place:'host input'});
          }
        }
        if(s.inputs) {
          const extra={tag:s.cpu?'CPU VIEW / SAME DATA':'DEVICE INPUTS',origin:s.cpu?point(680,270):point(174,208),via:s.cpu?[]:[point(331,567),point(652,567)],place:s.cpu?'ref(host buffers)':'accelerator input storage'};
          add('D-inputs','crate',s.cpu?'ref(input buffers)':'Device buffers',680,169,140,119,'buffer',{...extra,lines:[s.id==='addition'?'X1':s.id==='attention'?'Q1 / K1 / V1':'A1 / B1',s.cpu?'no GPU transfer':'explicit .to(device)']});
        }
        if(s.id==='addition') {
          if(s.lane>=0) {
            (s.operands||[]).forEach((v,i)=>add('load-'+i,'cell',fmt(v),871+i*125,204,92,48,'buffer',{origin:point(745,234),place:'SIMD operand'}));
            add('operation','tool','SIMD group · + 1',870,299,253,69,'mojo',{tag:'TWO LANES · ILLUSTRATIVE',place:'native body'});
          }
        } else if(s.id==='epilogue') {
          if(s.acc.length) {
            add('acc-title','handle',s.fusion?'Register Y['+Math.floor(s.cell/2)+','+s.cell%2+']':s.phase==='matmul'?'MATMUL → TMP1':s.phase==='add_one'?'+ 1 → TMP2':'× 2 → Y',859,176,284,53,'mojo',{place:'epilogue body'});
            if(s.fusion)(s.operands||[]).forEach((v,i)=>add('operand-'+i,'cell',fmt(v),860+i*72,266,63,42,'buffer',{origin:point(745,248),place:'dot operands'}));
            s.acc.forEach((v,i)=>add('acc-'+i,'cell',fmt(v),s.fusion?935:860+i*72,345,s.fusion?134:63,48,'runtime',{origin:point(1003,288),place:s.fusion?'native accumulator':'native computation'}));
            add('epilogue-work','tile',s.fusion?'acc → +1 → ×2 → store':'separate body / store',859,402,284,32,'mojo',{place:'operation chain'});
          }
          if(s.temporary.length)add('TMP','crate',s.tempName,680,321,140,116,'artifact',{tag:'INTERMEDIATE MATRIX',lines:['['+s.temporary.slice(0,2).map(fmt)+']','['+s.temporary.slice(2).map(fmt)+']'],origin:point(1003,367),place:'temporary allocation'});
        } else if(s.id==='matmul'&&!s.cpu&&s.algorithm==='tiled') {
          if(s.shared) {
            const t=s.tile;
            add('shared-A-label','tile','A tile · k='+t*2+','+(t*2+1),855,166,139,25,'buffer',{place:'shared tile label'});
            add('shared-B-label','tile','B tile · k='+t*2+','+(t*2+1),1008,166,139,25,'buffer',{place:'shared tile label'});
            for(const [name,matrix,ox] of [['A',TA,855],['B',TB,1015]]) {
              for(let i=0;i<4;i++) {
                const r=Math.floor(i/2),c=i%2,v=name==='A'?matrix[r][t*2+c]:matrix[t*2+r][c];
                add(name+'-shared-'+t+'-'+i,'cell',fmt(v),ox+c*59,198+r*54,48,44,'buffer',{origin:point(724,250),place:'shared '+name+' tile '+t});
              }
            }
            add('copy-gate','stamp',s.copyReady?'COPIES READY':'COPIES PENDING',1008,317,139,34,'sync',{place:'async copy completion'});
            add('block-gate','stamp',s.blockReady?'BLOCK READY':'BLOCK WAITING',1008,364,139,34,'sync',{place:'block barrier'});
          }
          if(s.acc.length)add('register-label','tile','Registers',858,300,129,22,'runtime',{place:'per-thread accumulator label'});
          s.acc.forEach((v,i)=>add('register-'+i,'cell',fmt(v),858+(i%2)*69,322+Math.floor(i/2)*57,60,44,'runtime',{place:'register accumulator'}));
          if(s.tile>=0&&!s.shared)add('reuse-gate','stamp','SHARED REUSABLE',1008,335,139,45,'sync',{place:'barrier before overwrite'});
        } else if(s.id==='matmul') {
          if(s.cell>=0) {
            add('cell-title','tile','C['+Math.floor(s.cell/2)+','+s.cell%2+'] · k='+s.k,860,182,283,51,'mojo',{place:'naive loop'});
            (s.operands||[]).forEach((v,i)=>add('operand-'+i,'cell',fmt(v),873+i*133,280,95,43,'buffer',{origin:point(742,228),place:'global operand'}));
            if(s.acc.length)add('naive-acc','cell',fmt(s.acc[0]),949,359,108,46,'runtime',{place:'accumulator'});
          }
        } else if(s.id==='attention'&&s.phase!=='resident'&&s.phase!=='arrays'&&s.phase!=='upload'&&s.phase!=='execute') {
          add('max-state','handle','m = '+(s.m===null?'MIN':fmt(s.m)),855,261,139,73,'mojo',{tag:'RUNNING MAXIMUM',place:'arithmetic state'});
          add('sum-state','handle','l = '+fmt(s.l),1008,261,139,73,'mojo',{tag:'DENOMINATOR',place:'arithmetic state'});
          if(s.local) {
            add('S-label','tile','S · local scores',855,165,139,18,'buffer',{place:'local score label'});
            s.scores.forEach((v,i)=>add('S-slot-'+i,'cell',fmt(v),855+i*75,188,63,42,'buffer',{origin:point(746,234),place:'reusable local score slot · tile '+s.tile}));
          }
          if(s.scores.length&&s.phase!=='score')add('alpha','tile','α = '+fmt(s.alpha),1009,193,139,36,'mojo',{place:'rescaling factor'});
          if(s.local&&s.E.length) {
            add('E-label','tile','E · exp(S − m)',855,357,139,18,'buffer',{place:'local exponent label'});
            s.E.forEach((v,i)=>add('E-slot-'+i,'cell',fmt(v),855+i*75,380,63,42,'buffer',{origin:point(886+i*75,211),place:'reusable local exponent slot · tile '+s.tile}));
          }
          add('running-O-label','tile','O_i · normalized',1008,335,139,20,'mojo',{place:'running output label'});
          s.O.forEach((v,i)=>add('O-running-'+i,'cell',fmt(v),1008,355+i*46,139,38,'mojo',{place:'normalized running output'}));
          add('attention-input-tiles','crate','K1 / V1 tiles',680,327,140,106,'buffer',{tag:'INPUTS REMAIN',lines:['next key tile','Q1 reused'],place:'resident inputs'});
        }
        s.computed.forEach((v,i)=>add('O1-'+i,'cell',fmt(v),852+i*74,491,64,40,'runtime',{origin:point(1000,383),place:'target result',delay:i*60}));
        if(s.hostRead)s.computed.forEach((v,i)=>add('H-O1-'+i,'cell',fmt(v),49+i*62,444,54,40,'runtime',{origin:s.cpu?point(49+i*62,444):point(884+i*74,510),via:s.cpu?[]:[point(1158,570),point(310,570)],place:s.cpu?'CPU result view':'explicit result.to(CPU())',delay:i*60}));
        if(s.id==='attention'&&s.ready)add('print-result','handle','print(output) · Buffer handles',49,444,241,56,'python',{origin:point(481,475),place:'caller output printing'});
        if(s.id==='epilogue'&&s.traffic) {
          if(s.fusion) {
            add('fusion-chain','binary','accumulator → +1 → ×2 → final store',266,717,667,78,'mojo',{tag:'ELIGIBLE FUSION ILLUSTRATION',place:'epilogue comparison'});
          } else {
            add('producer-body','tile','MATMUL',49,720,172,59,'mlir',{place:'comparison producer'});
            if(s.temporary.length&&s.tempName==='TMP1'||s.tempLive===2)add('TMP1-life','crate','TMP1 · 4 values',241,713,166,80,'buffer',{tag:'LIVE INTERMEDIATE',place:'TMP1 live allocation'});
            add('add-body','tile','+ 1',427,720,130,59,'mojo',{place:'comparison consumer'});
            if(s.temporary.length&&s.tempName==='TMP2')add('TMP2-life','crate','TMP2 · 4 values',577,713,166,80,'buffer',{tag:'LIVE INTERMEDIATE',origin:point(260,716),place:'TMP2 live allocation'});
            add('multiply-body','tile','× 2 → final Y',764,720,220,59,'backend',{place:'comparison final consumer'});
          }
          add('intermediate-read-count','handle','Intermediate reads: '+s.traffic.reads,49,813,268,38,'buffer',{place:'logical scalar access count'});
          add('intermediate-write-count','handle','Intermediate writes: '+s.traffic.writes,339,813,277,38,'buffer',{place:'logical scalar access count'});
          add('output-write-count','handle','Final writes: '+s.traffic.output,638,813,226,38,'runtime',{place:'logical output access count'});
          add('live-count','handle','Live TMPs: '+s.tempLive,886,813,268,38,'artifact',{place:'intermediate lifetime count'});
        }
        if(s.id==='attention') {
          const fullScores=K.map(row=>Q.reduce((total,q,i)=>total+q*row[i],0));
          const full=s.ready?[]:s.tile>=0?fullScores.slice(0,(s.tile+1)*2):[];
          this.slots('full-scores',full);this.slots('online-scores',s.local?s.scores:[]);
          full.forEach((v,i)=>add('full-score-slot-'+i,'cell',fmt(v),49+i*128,729,108,40,'artifact',{origin:point(883,209),place:'hypothetical full score storage'}));
          if(s.local)s.scores.forEach((v,i)=>add('online-score-slot-'+i,'cell',fmt(v),627+i*128,729,108,40,'buffer',{origin:point(883+i*75,209),place:'same online score slots · tile '+s.tile}));
          add('full-score-size','handle','Full score matrix: '+s.params.N*s.params.N+' elements',49,815,484,35,'artifact',{place:'N × N storage comparison'});
          add('online-score-size','handle','Local scores: '+s.params.BN*s.params.keyTile+' elements / tile',627,815,510,35,'buffer',{place:'queryTile × keyTile source score storage'});
          if(s.tile>=0||s.phase==='initialize') {
            add('retain-m','handle','m = '+(s.m===null?'MIN':fmt(s.m)),627,775,111,30,'mojo',{place:'retained maximum'});
            add('retain-l','handle','l = '+fmt(s.l),759,775,111,30,'mojo',{place:'retained denominator'});
            add('retain-O','handle','O = ['+s.O.map(fmt)+']',891,775,246,30,'mojo',{place:'retained normalized output'});
            add('full-score-life','handle',s.ready?'Row complete: score storage released':'Retain scores until full-row reduction',49,775,484,30,'artifact',{place:'comparison score lifetime'});
          }
        }
      }
      return list;
    }
  }
  root.CompilerExamplesVisual=ExampleVisual;
})(window);
