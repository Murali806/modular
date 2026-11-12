"""Attention-specific tile, arithmetic, storage and device-launch lessons."""
import json
from pathlib import Path
import subprocess

from compiler_attention_lessons import NAME, Story, lesson, objects


def kernel_lessons():
    cards=[]
    def add(*a,**kw):
        cards.append(lesson(*a,**kw))

    nodes=objects(("contract","execute interface","destination + 3 inputs + ctx","registered routine"),
                  ("params","compile-time specialization","inferred/explicit parameters","Mojo compiler"),
                  ("rank","rank gate","comptime assertion","execute"),
                  ("views","Q/K/V/O views","TileTensor metadata","existing buffers"),
                  ("branch","target branch","selected code path","specialization"),
                  ("work","attention work","CPU function or GPU launch","runtime"))
    def dispatch(cpu=False):
        bn,bd,n=(4,4,8) if cpu else (16,8,32)
        s=Story(nodes,label="Actual CPU fallback" if cpu else "Actual GPU specialization",view="dispatch")
        s.step("Resolve the registered execute interface","attention_registration",67,NAME,"@register maps FusedAttention","output,query,key,value,ctx",
               "The Graph has three tensor operands and one result. Runtime glue supplies the mutable output destination and context.",patch={"contract":"mut output; read query/key/value; runtime ctx"})
        s.step("Infer dtype and rank from tensor types","attention_registration",75,"Q/K/V and output types","dtype/rank inferred before // separator","dtype=float32; rank=2",
               "All four tensor interfaces must satisfy the declared shared dtype/rank.",patch={"params":"inferred dtype=f32,rank=2"})
        s.step("Bind explicit tile parameters and target","attention_registration",78,f"BD={bd},BN={bn}; graph device","specialize BN/BD; infer target","target=cpu" if cpu else "target=gpu",
               "These are compile-time choices; target does not mean a runtime string comparison for each tensor element.",patch={"params":f"dtype=f32,rank=2,BN={bn},BD={bd},target={'cpu' if cpu else 'gpu'}"})
        s.step("Check supported rank","attention_registration",88,"rank=2","comptime assert rank==2","rank gate PASS",
               "This source implements rank-2 attention, with no head/batch dimensions or causal mask.",patch={"rank":"PASS: rank==2"})
        for key,line in [("Q",91),("K",93),("V",95),("O",97)]:
            s.step(f"Create the {key} TileTensor view","attention_registration",line,f"existing {key} tensor buffer","to_tile_tensor().as_unsafe_any_origin()",f"{key} view:[{n},{n}] over existing storage",
                   "A view carries pointer/layout/origin metadata. It does not duplicate the tensor allocation.",patch={"views":f"{key}: pointer p{key}, shape=[{n},{n}], layout/strides · same storage"})
        s.step("Select one compile-time branch","attention_registration",99,"target specialization","comptime if target=='cpu'","fused_attention_cpu" if cpu else "fused_attention_gpu",
               "Host Python/MAX still run on CPU; GPU specialization adds a separate device kernel.",patch={"branch":"CPU scalar/tiled helper" if cpu else "GPU host wrapper + device kernel"})
        s.step("Call the selected implementation","attention_registration",101 if cpu else 105,f"Q/K/V/O views; BN={bn},BD={bd}","direct CPU call" if cpu else "dev_ctx copy; GPU wrapper enqueue", "two key tiles per output tile" if cpu else "eight output blocks; four key tiles per block",
               "Both routes use stable online softmax, but their tile geometry and execution mechanisms differ.",patch={"work":f"CPU Q rows4 × output cols4; key rows4" if cpu else "GPU Q rows16 × output cols8; key rows8"})
        return s
    add("kernel-dispatch","Kernel · Inferred parameters and CPU/GPU dispatch","dispatch","How do Graph values become a specialized destination-passing routine?",
        "Registered declaration, tensor types, explicit BN/BD, target and runtime ctx", "Typed Q/K/V/O views and one selected implementation",
        """**The execute interface has four tensor roles, but only three Graph inputs.** Output is a mutable destination corresponding to the Graph result. Query/key/value are the three input operands. ctx is supplied by native runtime glue.

dtype/rank are inferred; BN/BD come from custom-op parameters; the selected graph target determines cpu/gpu specialization. The `comptime` rank check and branch are compile-time operations. Views retain the same storage with layout/origin metadata.

CPU calls its helper directly with BN=BD=4. GPU uses BN=16,BD=8 and a host DeviceContext wrapper that submits a separate device kernel. **A GPU graph still needs CPU orchestration and runtime services.**""",
        """caller: generated routine reaches FusedAttention.execute [LINK:attention_registration:74]
├─ @register name; dtype/rank inferred; explicit BN/BD; target specialized [LINK:attention_registration:67] [LINK:attention_registration:75] [CALLER:entry|60]
├─ comptime assert rank==2 [LINK:attention_registration:88] [CALLER:attention_registration|74]
├─ query/key/value/output → TileTensor views over existing storage [LINK:attention_registration:91] [LINK:attention_registration:93] [LINK:attention_registration:95] [LINK:attention_registration:97] [CALLER:attention_registration|74]
└─ comptime target branch [LINK:attention_registration:99]
   ├─ CPU → fused_attention_cpu[4,4](Q,K,V,O) [LINK:attention_registration:101] [CALLER:attention_registration|101]
   └─ GPU → fused_attention_gpu[16,8](ctx,Q,K,V,O) [LINK:attention_registration:105] [CALLER:attention_registration|105]""",
        dispatch(),alternatives={"cpu":dispatch(True)},source="attention_registration",line=74)

    nodes=objects(("kernel","kernel_func","concrete device-function specialization","GPU wrapper"),
                  ("module","device module / function","loaded driver handles","ctxGPU"),
                  ("args","argument slots","device-passable encoding","host launch wrapper"),
                  ("config","grid/block configuration","launch geometry","host submission"),
                  ("stream","stream and driver call","runtime handles/API","host thread"),
                  ("tile","selected output tile","logical block ownership","GPU kernel"))
    def launch(amd=False,by=0,bx=0):
        s=Story(nodes,label="AMD driver illustration" if amd else "Block (x=1,y=1)" if bx else "NVIDIA · block (x=0,y=0)",view="geometry")
        geo=dict(N=32,D=32,BN=16,BD=8,keyRows=8,grid=[4,2],block=[32,1,1],blockXY=[bx,by])
        s.step("Specialize the device function from view types","attention_launch",391,"Q/K/V/O dtype/layouts; BN=16,BD=8","fused_attention_kernel[...types...,BN,BD]","kernel_func",
               "The specialization fixes the code and tensor layouts. It does not contain the runtime matrix values.",patch={"kernel":"fused_attention_kernel · concrete layouts + f32 + BN16/BD8"},**geo)
        s.step("Compute the two-dimensional launch grid","attention_launch",408,"O rows32,columns32; BN16,BD8","grid=(D//BD,N//BN)","grid=(4,2,1)",
               "x partitions output columns; y partitions query rows. Eight blocks cover all O cells without overlapping output tiles.",patch={"config":"grid=(4,2,1); block=(32,1,1)"})
        s.step("Check dimensions and choose the function wrapper","attention_enqueue",162,"kernel_func; grid; block","_check_dim; compile_function[func](...)","checked DeviceFunction wrapper",
               "compile_function's host path obtains the compiler-produced device payload/handle; it is distinct from CompiledKernelsContext host symbol lookup.",patch={"kernel":"DeviceFunction · checked device argument signature"})
        s.step("Load or resolve the device function handle","attention_load_function",2994,"module name, function name, compiled payload, ctxGPU","_load_function → AsyncRT_DeviceContext_loadFunctionWithDebugName","DEVICE_FN1 handle",
               "The payload is device code. DEVICE_FN1 is a driver handle, not a host callable address or a tensor pointer.",patch={"module":"MODULE_GPU1 → DEVICE_FN1 · device context retained"})
        for key in ["Q","K","V","O"]:
            s.step(f"Encode the {key} view in its launch slot","attention_pack",3639,f"{key}: pointer p{key}, layout/shape metadata","DevicePassable._to_device_type; aligned slot","packed device argument slot",
                   "Host wrapper types become the target ABI's device-passable representation. The slot carries the buffer address/metadata, not all matrix elements.",patch={"args":f"{key} slot → device pointer p{key} + encoded view metadata"})
        s.step("Submit packed arguments to native runtime","attention_packed_tail",2947,"DEVICE_FN1; four slot addresses; grid/block","ctx.enqueue(...)","native enqueue request",
               "Error handling retains source location/function labels for diagnostics.",patch={"args":"slots [Q,K,V,O] ready · same tensor allocations", "stream":"native runtime enqueue · selected stream"})
        s.step("Cross the native driver binding","attention_driver_enqueue",542,"ctxGPU/DEVICE_FN1/config/slots","AsyncRT_DeviceContext_enqueueFunctionDirect","selected device stream route",
               "The binding verifies matching device identity and dispatches through the runtime device implementation.",patch={"stream":"ctxGPU.enqueueFunctionExecDirect → device stream"})
        s.step("Call the vendor's kernel-launch API","attention_hip_launch" if amd else "attention_cuda_launch",1382 if amd else 1610,"DEVICE_FN1 + stream + packed slots","hipModuleLaunchKernel" if amd else "cuLaunchKernelEx","device work submitted",
               "Submission does not establish measured completion or timing. CUDA/HIP details are a source/driver boundary in this offline reader.",patch={"stream":"HIP stream → hipModuleLaunchKernel" if amd else "CUstream → cuLaunchKernelEx"})
        s.step("Select the block's query/output tile","attention_tiles",333,f"block_idx=(x={bx},y={by})","Q.tile[16,32](y,0); O.tile[16,8](y,x)",f"query rows {by*16}–{by*16+15}; O cols {bx*8}–{bx*8+7}",
               "Different x blocks for the same y reuse the same query rows but own different output columns.",patch={"tile":f"O rows[{by*16}:{by*16+16}],cols[{bx*8}:{bx*8+8}]"},activeBlock=[bx,by])
        for j in range(4):
            s.step(f"Visit key tile {j+1} of four","attention_tiles",344,f"key rows {j*8}–{j*8+7}","K.tile[8,32](j,0); V.tile[8,8](j,x)","current score/value tile for owned O cells",
                   "Every owned query/output column sees all key rows; tile boundaries do not truncate the attention denominator.",patch={"tile":f"O block({bx},{by}) · key rows[{j*8}:{j*8+8}]"},keyTile=j)
        s.step("Store only the owned output cells","attention_output",378,"retained O_i[16,8] after all keys","O.tile[16,8](y,x).copy_from(O_i)","one completed O tile",
               "Other blocks write disjoint output tiles. The logical grid is not a claim about simultaneous GPU block scheduling.",patch={"tile":"owned O tile stored; all other tiles have separate owners"},stored=True)
        return s
    add("kernel-launch","Launch · Function handles, argument slots and output tiles","geometry","How do host launch arguments reach the driver, and what does each GPU block own?",
        "Concrete kernel_func; Q/K/V/O views; BN16/BD8; ctxGPU", "Grid (4,2),32-thread blocks; disjoint [16,8] O tiles",
        """**grid.x=4 partitions output columns; grid.y=2 partitions query rows.** Each block writes a [16,8] O tile, computes [16,8] scores for four successive key tiles, and sees all 32 keys. x does not partition the softmax denominator.

| Object | Example | Kind |
|---|---|---|
| Host routine | HOST_FN1 | CPU callable pointer |
| Device function | DEVICE_FN1 in MODULE_GPU1 | CUDA/HIP driver handle |
| Q buffer | pQ,4096 bytes | Device tensor storage address |
| Argument slot | encoded TileTensor pointer/layout | Host ABI slot submitted to driver |
| Stream | CUstream/HIP stream | Device work queue handle |

The source route checks dimensions, obtains a checked function, encodes DevicePassable arguments, then calls native runtime enqueue. NVIDIA uses `cuLaunchKernelEx`; AMD uses `hipModuleLaunchKernel`. Select block(1,1) to see rows16–31/columns8–15 and how K/V tile views change. Physical scheduling and exact residency/timing remain unmeasured.""",
        """caller: ctx.enqueue_function[kernel_func](Q,K,V,O,grid_dim=(4,2),block_dim=32) [LINK:attention_launch:403]
├─ specialize kernel function with dtypes/layouts/BN/BD [LINK:attention_launch:391] [CALLER:attention_launch|403]
├─ DeviceContext enqueue mixin: check dimensions and compile_function [LINK:attention_enqueue:162] [LINK:attention_enqueue:171] [CALLER:attention_launch|403]
│  └─ _load_function → native loadFunctionWithDebugName [LINK:attention_load_function:2994] [LINK:attention_driver_load:504] [CALLER:attention_enqueue|171]
├─ checked argument pack [LINK:attention_pack:3503]
│  ├─ validate target argument types/offsets [LINK:attention_pack:3536] [CALLER:attention_enqueue|179]
│  ├─ _to_device_type writes aligned slots [LINK:attention_pack:3639] [CALLER:attention_enqueue|179]
│  └─ _enqueue_packed_checked → ctx.enqueue [LINK:attention_packed_tail:2947] [CALLER:attention_pack|3664]
├─ native enqueueFunctionDirect [LINK:attention_driver_enqueue:542]
│  ├─ NVIDIA → CUDADeviceFunction::cudaCallKernelDirect → cuLaunchKernelEx [LINK:attention_cuda_launch:1573] [LINK:attention_cuda_launch:1610] [CALLER:attention_driver_enqueue|554]
│  └─ AMD → HIPDeviceFunction::hipCallKernelDirect → hipModuleLaunchKernel [LINK:attention_hip_launch:1365] [LINK:attention_hip_launch:1382] [CALLER:attention_driver_enqueue|554]
└─ block(y,x): Q rows[y*16:(y+1)*16]; O cols[x*8:(x+1)*8] [LINK:attention_tiles:333]
   ├─ four key tiles: all K rows; matching V rows and owned columns [LINK:attention_tiles:344] [CALLER:attention_tiles|343]
   └─ store owned O tile after the loop [LINK:attention_output:378] [CALLER:attention_launch|403]""",
        launch(),alternatives={"other-block":launch(by=1,bx=1),"amd":launch(True)},source="attention_launch",line=403)

    nodes=objects(("views","matrix operands","TileTensor views","matmul caller"),
                  ("tensorcore","TensorCore descriptor","fragment layouts + operation","GPU helper"),
                  ("a","A fragment","loaded operand fragment","MMA step"),
                  ("b","B fragment","loaded operand fragment","MMA step"),
                  ("c","c_reg",".LOCAL accumulation fragment","matmul helper"),
                  ("shared","out_sram",".SHARED result staging","thread block"),
                  ("barrier","barrier gate","block synchronization","GPU helper"),
                  ("result","res","logical result tile","caller"))
    def mma(pv=False):
        s=Story(nodes,label="PV product · one reduction slice" if pv else "QKᵀ · four feature slices",view="mma")
        M,N,K=(16,8,8) if pv else (16,8,32)
        s.step("Select matrix dimensions and result shape","attention_mma",264,f"E[16,8] × V[8,8]" if pv else "Q[16,32] × K[8,32]ᵀ","M=lhs.rows; N=rhs.rows; K=lhs.cols","result[16,8]",
               "The source's actual PV RHS tile is square [8,8]. Transposition changes operand loading, not a full K-matrix allocation.",patch={"views":"P_tile[16,8] × V_tile[8,8]" if pv else "Q_tile[16,32] × K_tile[8,32]ᵀ"},product="PV" if pv else "QK",M=M,N=N,K=K)
        s.step("Allocate explicitly shared result staging","attention_mma",268,"M16,N8","stack_allocation(...address_space=.SHARED)","out_sram[16,8]",
               "The helper will store accumulation fragments here before copying into the logical result.",patch={"shared":"out_sram[16,8] · .SHARED · initially unspecified"})
        s.step("Define the reduction slice and fragment layouts","attention_mma",272,"BK=8; K divisible by8","TensorCore[dtype,dtype,Index(16,8,8),transpose_b]","TensorCore descriptor",
               "MMA means matrix multiply-accumulate. Fragment layouts distribute participating matrix elements across lanes; they are not necessarily row-major arrays.",patch={"tensorcore":f"TensorCore(16,8,8), transpose_b={not pv}; K/BK={K//8}"})
        s.step("Initialize the accumulation fragment","attention_mma",278,"TensorCore.c_fragment_layout",".LOCAL stack allocation.fill(0)","c_reg=0",
               "Accumulation must start at zero; every reduction slice adds to this same fragment.",patch={"c":"c_reg · all accumulation elements zero"},acc=0,partials=[],chunk=-1)
        s.step("Rebind the B view if element dtypes differ","attention_mma",284,"rhs view; lhs dtype","rebind TileTensor dtype metadata","matching native load dtypes",
               "This is a view type adjustment in the helper, not a numerical transpose or new full matrix copy.",patch={"views":"operand views rebound for compatible native loads"})
        acc=0;partial=[1] if pv else [8,16,4,-8]
        for i,p in enumerate(partial):
            aVal=0.125 if pv else 1
            bVal=1 if pv else [1,2,0.5,-1][i]
            terms=[aVal*bVal]*8
            s.step(f"Slice {i+1}: load A's eight reduction elements","attention_mma",296,f"feature/key indices {i*8}–{i*8+7}","load_a(lhs.tile[M,BK](0,k_i))","a_reg fragment",
                   "One scalar example uses eight A values; actual fragments cover the tile across participating lanes.",patch={"a":f"example A values=[{aVal}]×8"},chunk=i,terms=terms,aValues=[aVal]*8,bValues=None,pendingAcc=None)
            s.step(f"Slice {i+1}: load B with correct orientation","attention_mma",301 if not pv else 298,"K row feature slice" if not pv else "V key-row/output-column slice","transpose load_b(K.tile[N,BK])" if not pv else "load_b(V.tile[BK,N])","b_reg fragment",
                   "QKᵀ compares a query row with a key row across features; PV compares normalized key weights with one value column across keys.",patch={"b":f"example B values=[{bVal}]×8"},bValues=[bVal]*8)
            old=acc;acc+=p
            s.step(f"Slice {i+1}: multiply and accumulate","attention_mma",305,f"A/B fragments; previous c={old}","d_reg=mma_op(a_reg,b_reg,c_reg)",f"example output cell: {old}+{p}={acc}",
                   "The scalar sum shows the mathematics. One MMA operation processes fragments collectively, not a scalar Python loop.",patch={"c":f"d_reg cell={acc}; c_reg cell still {old} until copy"},priorAcc=old,pendingAcc=acc,partial=p)
            s.step(f"Slice {i+1}: retain the updated fragment","attention_mma",306,"d_reg","c_reg.copy_from(d_reg)",f"c cell={acc}",
                   "The next reduction slice starts from the new accumulation, preserving earlier feature contributions.",patch={"c":f"c_reg example cell={acc}"},acc=acc,partials=partial[:i+1],pendingAcc=None)
        s.step("Store accumulation fragments into shared staging","attention_mma",307,"completed c_reg","mma_b_t.store_d(out_sram,c_reg)","out_sram contains result tile",
               "Fragment storage is mapped back into the shared result layout.",patch={"shared":f"out_sram[16,8] populated; example cell={acc}"})
        s.step("Synchronize before consuming shared results","attention_mma",309,"participants wrote shared tile","barrier()","shared staging safe to read",
               "A block barrier orders participating writes before reads. It is not a whole-GPU or Python synchronization.",patch={"barrier":"PASS · block writers/readers ordered"})
        s.step("Copy the completed logical tile to the caller","attention_mma",310,"out_sram after barrier","res.copy_from(out_sram)","S[16,8]" if not pv else "PV[16,8]",
               "The caller now uses S for row softmax or PV for the output merge. Exact registers/spills/instruction scheduling remain compiler/target dependent.",patch={"result":f"{'PV' if pv else 'S'} tile[16,8]; example cell={acc}"})
        return s
    add("kernel-mma","MMA · Operand fragments, accumulation and barrier","mma","How does QKᵀ/PV traverse fragments, and why are c_reg and out_sram separate?",
        "Q_tile/K_tile or normalized E/V_tile", "S or PV logical [16,8] tile",
        """**QKᵀ has four BK=8 reduction slices because D=32. PV has one because the current key tile has eight rows.** The TensorCore descriptor determines fragment layout/load/MMA/store operations; a fragment is the lane-distributed representation of part of a matrix.

The animation follows one illustrative output cell: Q values=1 across features and key groups=1,2,0.5,−1 yield partial sums 8,16,4,−8 and final score20. This is a teaching input, not the seeded random caller matrix. The PV selector uses eight weights0.125 and value components1 to produce1.

c_reg is explicitly .LOCAL fragment storage. out_sram is explicitly .SHARED staging. After storing fragments, a block barrier precedes the result copy. These declarations **do not establish measured register allocation, spills or physical timing**. The source does not implement FA3-style asynchronous overlap here.""",
        """caller: S=matmul['gpu',transpose_b=True](Q_tile,K_tile) [LINK:attention_tiles:346]
├─ M/N/K; out_sram .SHARED; BK=8 divisibility check [LINK:attention_mma:264] [LINK:attention_mma:268] [CALLER:attention_tiles|346]
├─ TensorCore descriptor; c_reg .LOCAL fragment layout filled0 [LINK:attention_mma:276] [LINK:attention_mma:278] [CALLER:attention_tiles|346]
├─ rhs_typed view rebind for compatible load dtype [LINK:attention_mma:284] [CALLER:attention_tiles|346]
├─ for k_i in K//BK [LINK:attention_mma:295]
│  ├─ load_a current lhs tile [LINK:attention_mma:296] [CALLER:attention_mma|295]
│  ├─ load_b; transposed K load uses [N,BK] orientation [LINK:attention_mma:298] [LINK:attention_mma:301] [CALLER:attention_mma|295]
│  ├─ d_reg=mma_op(a_reg,b_reg,c_reg) [LINK:attention_mma:305] [CALLER:attention_mma|295]
│  └─ c_reg.copy_from(d_reg) [LINK:attention_mma:306] [CALLER:attention_mma|295]
├─ store_d(out_sram,c_reg); barrier(); res.copy_from(out_sram) [LINK:attention_mma:307] [LINK:attention_mma:309] [LINK:attention_mma:310] [CALLER:attention_tiles|346]
└─ later PV=matmul['gpu'](E,V_tile), with K=8 [LINK:attention_output:374] [CALLER:attention_output|374]""",
        mma(),alternatives={"pv":mma(True)},source="attention_tiles",line=346)

    nodes=objects(("global","global Q/K/V/O","full tensor allocations","caller/runtime"),
                  ("views","current Q/K/V views","logical slices of existing storage","block"),
                  ("state","m₁/l₁/Oᵢ","retained row state/output tile","block lifetime"),
                  ("score","S / E","temporary score/weight tiles","current key iteration"),
                  ("pv","PV","temporary value-product tile","current key iteration"),
                  ("helper","c_reg / out_sram","explicit .LOCAL / .SHARED","matmul helper"),
                  ("store","O tile store","final output write","block"))
    s=Story(nodes,view="storage")
    geo=dict(N=32,D=32,BN=16,BD=8,keyRows=8,grid=[4,2],blockXY=[0,0])
    s.step("Retain full input/output allocations","attention_registration",91,"Q/K/V/O runtime buffers","create TileTensor views","four [32,32] allocations",
           "For f32, each full tensor has4096 logical bytes. The views reuse storage.",patch={"global":"Q/K/V/O each[32,32] · 4096 bytes"},**geo,keyTile=-1)
    s.step("Select Q once for this block","attention_tiles",333,"block y=0","Q.tile[16,32](0,0)","Q view[16,32]",
           "This view is reused for four key iterations and different x blocks have their own output column ranges.",patch={"views":"Q rows0–15; all32 features"})
    s.step("Allocate persistent per-block row state","attention_tiles",335,"BN16; BD8","m₁[16],l₁[16],Oᵢ[16,8]","retained state initialized",
           "Logical f32 sizes are64+64+512=640 bytes. Default stack allocation is .GENERIC, so physical placement/spills are not proven.",patch={"state":"m₁[16]=MIN; l₁[16]=0; Oᵢ[16,8]=0 · .GENERIC default"})
    for t in range(4):
        s.step(f"Iteration {t+1}: select eight K/V rows","attention_tiles",344,f"tile_n_idx={t}","K.tile[8,32]; V.tile[8,8]","views into full K/V",
               "K keeps all feature columns; V keeps the output columns owned by this block.",patch={"views":f"Q[16,32] retained; K rows{t*8}–{t*8+7}[8,32]; V matching rows[8,8]", "score":"S pending; prior scratch reusable", "pv":"PV pending"},keyTile=t,liveScore=False,liveE=False,livePV=False)
        s.step("Create the temporary score tile","attention_tiles",346,"Q/K views","matmul → S[16,8]","512 logical bytes for this tile",
               "No persistent [32,32] score allocation is declared by this fused loop.",patch={"score":"S[16,8] · 512 logical bytes", "helper":"matmul: c_reg .LOCAL fragments; out_sram .SHARED[16,8]"},liveScore=True)
        s.step("Reuse tile-shaped exponent/weight storage","attention_denominator",358,"S and new row maxima","E=copy(S); subtract max; exp; normalize","E/P[16,8]",
               "E is overwritten from scores to exponentials to normalized weights. It represents only the current eight keys.",patch={"score":"S[16,8]; E[16,8] reused as P_tile · 512 bytes each"},liveE=True)
        s.step("Merge current values into retained O_i","attention_output",374,"P_tile×V_tile; prior O rescaled","PV[16,8]; O_i+=PV; retain new m/l","state covers keys visited so far",
               "Current S/E/PV can be reused on the next key tile; row state/O persist.",patch={"pv":"PV[16,8] · 512 logical bytes; consumed by O_i", "state":f"m₁/l₁/Oᵢ now cover keys0–{t*8+7}"},livePV=True)
    s.step("Write the output only after all keys","attention_output",378,"final O_i[16,8]","copy into O.tile[16,8](y,x)","owned output cells written",
           "Logical temporary sizes are not total GPU resource usage: fragment staging/other scratch and compiler spills also matter.",patch={"store":"O rows0–15,cols0–7 written once after four key tiles"},stored=True)
    add("kernel-storage","Storage · Full tensors, temporary tiles and retained state","storage","Which values survive a key iteration, and where is the full score matrix avoided?",
        "Global Q/K/V/O and block coordinates", "Reusable S/E/PV tiles; retained m/l/O_i; one final O tile write",
        """**Global tensor storage, logical views, retained row state and temporary arithmetic tiles have different lifetimes.** Q/K/V/O are full [32,32] buffers. Each GPU block views Q[16,32], current K[8,32] and matching V[8,8], retaining m/l[16] and O_i[16,8].

S/E/PV are [16,8] current-iteration tiles. E is reused as shifted scores, exponentials and normalized weights. Earlier keys survive through m/l/O_i, so the fused route has no persistent full N×N score/weight allocation.

**Source address spaces:** matmul c_reg=.LOCAL and out_sram=.SHARED. The general `stack_allocation` helper defaults to .GENERIC; m/l/O_i/E/PV logical stack arrays do not automatically prove hardware-register residency or absence of spills. Logical byte counts illustrate tensor size, not measured GPU memory/resource utilization.""",
        """caller: fused_attention_kernel receives Q/K/V/O views [LINK:kernel_file:313]
├─ Q_tile selected once; initialize m₁,l₁,O_i [LINK:attention_tiles:333] [LINK:attention_tiles:335] [CALLER:attention_launch|403]
│  └─ stack_allocation default address_space=.GENERIC [LINK:attention_stack:3942] [CALLER:attention_tiles|335]
├─ four key iterations with BN_1=8 [LINK:attention_tiles:341] [LINK:attention_tiles:343]
│  ├─ K[8,32],V[8,8] views [LINK:attention_tiles:344] [CALLER:attention_tiles|343]
│  ├─ S=QKᵀ[16,8] [LINK:attention_tiles:346] [CALLER:attention_tiles|343]
│  ├─ E[16,8]: score copy → exp → normalized weights [LINK:attention_denominator:358] [LINK:attention_output:366] [CALLER:attention_tiles|343]
│  ├─ temporary PV[16,8] → O_i; retain m/l [LINK:attention_output:374] [LINK:attention_output:376] [CALLER:attention_tiles|343]
│  └─ only m/l/O_i survive as the next iteration's recurrence state [LINK:attention_output:377] [CALLER:attention_tiles|343]
└─ final O tile copy after loop [LINK:attention_output:378] [CALLER:attention_launch|403]""",
        s,source="kernel_file",line=313)

    nodes=objects(("q","selected q row","tiny query vector","teaching arithmetic"),
                  ("k","K rows","tiny key matrix","teaching inputs"),
                  ("v","V rows","tiny value matrix","teaching inputs"),
                  ("m","m₁","retained running maximum","row state"),
                  ("l","l₁","retained denominator","row state"),
                  ("o","Oᵢ","normalized output/scratch","row state"),
                  ("tile","key tile iterator","current K/V row selection","key loop"),
                  ("S","S","current dot-product scores","key iteration"),
                  ("E","E / P_tile","reused exponential/weight tile","key iteration"),
                  ("prior","l₂ scratch","old/new denominator mass","key iteration"),
                  ("delta","m₁_delta / row reduction","scale scratch","key iteration"),
                  ("PV","PV","weighted current values","key iteration"))
    def online(row):
        s=Story(nodes,label="Query1 · maximum unchanged on second tile" if row else "Query0 · maximum grows 1→3",view="online")
        code="process.stdout.write(JSON.stringify(require('./compiler_attention_math.js').onlineFrames(Number(process.argv[1]))))"
        s.frames=json.loads(subprocess.check_output(["node","-e",code,str(row)],cwd=Path(__file__).parent))
        s.note="Tiny arithmetic uses two keys per tile; actual GPU uses eight. Each query's row normalization is independent."
        return s
    add("kernel-online","Online softmax · Every recurrence operation","online","Why does earlier output need rescaling, and how do two tiles equal full softmax?",
        "Tiny Q/K/V; initial m=MIN,l=0,O=0", "Final normalized O matching an independent all-key reference",
        """**Scores decide weights; values decide the content being mixed.** For query0=[1,0], K rows=[1,0],[0,1],[3,0],[2,1] yield scores1,0,3,2. V rows=[1,0],[0,2],[2,1],[1,3]. Query1=[0,1] yields scores0,1,0,1 and demonstrates an unchanged maximum.

```text
new_m = max(old_m, row_max(S))
alpha = exp(old_m-new_m)
E = exp(S-new_m)
new_l = alpha*old_l + row_sum(E)
P_tile = E/new_l
new_O = old_O * alpha*old_l/new_l + P_tile @ V_tile
```

**Why rescale?** Old_O is already normalized by old_l on the old maximum scale. Multiplying by old_l recovers its numerator; dividing by new_l and multiplying by alpha moves it to the new scale. Current PV uses that same new denominator. Adding them preserves the contribution of every earlier key without keeping its scores.

The animation follows the source's actual copy/subtract/exp/multiply/divide order, including l₂ and m₁_delta scratch. It hides unavailable values until computed. Pink shows prior-key mass/output; green shows current-key mass/output. Softmax reduces across keys **per query**, not across query rows. The final check is browser arithmetic, not a MAX/GPU run. Mathematical −∞ represents the source's finite MIN sentinel.""",
        """caller: fused_attention_kernel key loop [LINK:attention_tiles:343]
├─ Q tile; initialize m₁=MIN,l₁=0,O_i=0 [LINK:attention_tiles:333] [LINK:attention_tiles:335] [CALLER:attention_launch|403]
├─ current K/V tile; S=matmul(Q_tile,K_tileᵀ) [LINK:attention_tiles:344] [LINK:attention_tiles:346] [CALLER:attention_tiles|343]
├─ m₂=row_max(S); m₂=max(m₂,m₁) [LINK:attention_max:348] [LINK:attention_max:349] [CALLER:attention_tiles|343]
├─ denominator scratch l₂ [LINK:attention_denominator:352]
│  ├─ copy m₁; subtract m₂; exp; multiply old l₁ [LINK:attention_denominator:353] [LINK:attention_denominator:354] [LINK:attention_denominator:355] [LINK:attention_denominator:356] [CALLER:attention_tiles|343]
│  ├─ E=copy(S); E-=m₂; E.exp() [LINK:attention_denominator:358] [LINK:attention_denominator:359] [LINK:attention_denominator:360] [CALLER:attention_tiles|343]
│  └─ row_sum(E); l₂+=sum → cumulative denominator [LINK:attention_denominator:362] [LINK:attention_denominator:363] [CALLER:attention_tiles|343]
├─ E/=l₂ → current normalized weights [LINK:attention_output:366] [CALLER:attention_tiles|343]
├─ rescale old O_i: *=old l₁; /=new l₂ [LINK:attention_output:367] [LINK:attention_output:368] [CALLER:attention_tiles|343]
├─ m₁_delta=copy(m₁); -=m₂; exp; O_i*=delta [LINK:attention_output:370] [LINK:attention_output:371] [LINK:attention_output:372] [LINK:attention_output:373] [CALLER:attention_tiles|343]
├─ PV=matmul(E,V_tile); O_i+=PV [LINK:attention_output:374] [LINK:attention_output:375] [CALLER:attention_tiles|343]
├─ retain m₁←m₂ and l₁←l₂ [LINK:attention_output:376] [LINK:attention_output:377] [CALLER:attention_tiles|343]
└─ store final O_i after every key tile has contributed [LINK:attention_output:378] [CALLER:attention_launch|403]""",
        online(0),alternatives={"query1":online(1)},source="attention_tiles",line=343)
    return cards
