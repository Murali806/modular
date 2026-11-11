# Native loader animation: research and applied decisions

Researched November 11, 2026. Scope: **CompiledKernelsContext::load**, its caller,
and the **later** compiled-host-routine route into GPU module/function loading
and execution.

MAX implementation authority:
610f7111836d9a950d45ba52ec9b3e0ed589fe3f.
The same checkout selects LLVM
6941494e2ae550d753728ba13454311b344d82cf in
bazel/public-patches/llvm_source.bzl:5.
Nine complete, unchanged upstream files and their hashes are stored in
[the LLVM snapshot manifest](references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/manifest.json).
The reader embeds these files in its offline Source Explorer.

## Primary research → visible mechanism

| Reference | Mechanism checked | Animation decision |
| --- | --- | --- |
| [LLVM ORCv2](https://llvm.org/docs/ORCv2.html) | Symbol namespace, materialization and readiness | Keep JD1/MU1 separate from linked code; show the query requiring Ready |
| [LLVM JITLink](https://llvm.org/docs/JITLink.html) | LinkGraph and linking stages | Show sections → blocks/symbols/edges → working memory → fixups → finalization |
| [Pinned ObjectLinkingLayer](references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/llvm/lib/ExecutionEngine/Orc/ObjectLinkingLayer.cpp) | Object bytes decoded into a LinkGraph | Decode already compiled objects, without inventing a Mojo or MLIR compilation step |
| [Pinned JITLinkGeneric](references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/llvm/lib/ExecutionEngine/JITLink/JITLinkGeneric.cpp) | Prune, allocate, publish addresses, resolve externals, fix up, finalize | Follow the actual phase order; Resolved leaves the context table empty |
| [Pinned BasicLayout](references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/llvm/lib/ExecutionEngine/JITLink/JITLinkMemoryManager.cpp) | Block address/content assignment and byte copy | Preserve A1 while new working-content actors receive copies |
| [Pinned x86-64 fixups](references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/llvm/include/llvm/ExecutionEngine/JITLink/x86_64.h) | Absolute Pointer64 equation and little-endian store | Compute an example equation and show all eight changed bytes |
| [Pinned MemoryMapper](references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/llvm/lib/ExecutionEngine/Orc/MemoryMapper.cpp) | Requested memory protection and executable instruction-cache invalidation | Working RW- slots become illustrative R-X / R-- segments |
| [CUDA module management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MODULE.html) | Image → module handle → named function handle | Keep cubin bytes, DM1 and DF1 distinct; leave code-residency timing to the vendor |
| [CUDA launch API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html) | Pointers to argument-value slots, dimensions and stream | Show args[0] → host arg0 slot → B1 device address; keep launch and completion separate |
| [HIP module API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html) | HIP image/module/function/launch counterparts | AMD selection changes image/handle/API names and exact source links |

These design choices interpret the mechanisms for teaching. Documentation,
an animation and source review are not hardware execution evidence.
No MAX/Mojo build, inference, OS linker or GPU workload was run.

## Concrete example and limits

The object cutaway is an **ELF64/x86-64 teaching example**, not a dump of the
addition example. StaticArchiveLayer in this MAX path reads an object-file
interface; the animation does not invent archive-member extraction.

~~~text
A0 caller bytes --byte copy--> A1 owned bytes
                              |
                              +-- ref retained by object layer
                              |
                              +-- decode --> LinkGraph --> working-content copy
                                                             |
  R0 = 0x00007f1210000400                                     |
  addend = 0x30                                              |
  Pointer64 = target + addend = 0x00007f1210000430 -------------+
  little-endian bytes: 30 04 00 10 12 7f 00 00
~~~

The word is at illustrative D0 + 0x20. Other relocation kinds use different
equations. The helper/symbol names, addresses, sizes and unreachable Tdead
block are examples. The shape_entry spelling does not prove compilation
of AddOne's unimplemented optional shape function.

~~~text
NeverSearched → Materializing → Resolved → Emitted → Ready
                                |                     |
                           address known          safe access
                                |                     |
                          table still empty      lookup returns P0
~~~

The fixed stages show prerequisites rather than measured elapsed time.
Finalization follows allocation-group protections; .text R-X and .rodata
R-- are the illustrated groups.

~~~text
Later P0 invocation
  → GPU image metadata → device-function cache → DM1 / DF1
  → launch arguments → stream submission → load / add / store
  → completion → explicit result.to(CPU()) copy

args[0] = &arg0     arg0 = 0x80010000 ----> B1 device allocation
args[1] = &arg1     arg1 = 0x80020000 ----> O1 device allocation
~~~

This two-pointer device pack is an illustration, distinct from P0's async-value
host ABI and the generated AddOne routine's complete argument list. The host
function pointer and opaque GPU function handle have different roles.

The numerical view uses four float32 values while the source example is
float32[5,10]. Working-value slots do not assert physical register or
warp assignments. Loads and stores copy values; buffers retain their identities.

The second teaching call retains M1/P0/DM1/DF1, uses new B2/O2 and pointer
values, and follows each computation step again. The original example invokes
execute once. On CPU, completed O2 is already CPU-readable and no GPU transfer
is depicted.

## Reader and validation

[The reader](link_htmls/max_compiled_kernels_context_load_and_driver.html) has
nine focused cutaways on one deterministic timeline; the original ownership
map remains available in a disclosure. It retains the exact prior answer,
source-linked trees, grey theme, bold pink/green highlights, movable
Hold/Expand/Close cards, conditional routes and no-autoplay controls.

[Validation](validation/COMPILED_KERNEL_LOADER_VALIDATION.md) records the
source/state and browser checks. Screenshots and playback checks prove
browser teaching-model behavior, not compiler or accelerator execution.
