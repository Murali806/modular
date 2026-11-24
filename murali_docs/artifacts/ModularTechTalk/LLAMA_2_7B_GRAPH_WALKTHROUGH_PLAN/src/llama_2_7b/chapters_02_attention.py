from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('attention_rope','Attention: QKV, RoPE, paged reads and online softmax',
 'Trace the actual registered Llama operations down to target dispatch, then inspect a numerical softmax tile without confusing it with a GPU trace.',
 'Model and construction',[
 A('QKV','QKV projection','activation','current layer execution','Projects normalized hidden rows to query, key and value channels.'),
 A('ROPE','Rotary position step','computation','fused rope/split/store operation','Rotates Q and K using request-relative absolute positions; V is not rotated.'),
 A('KV','Paged K/V','persistent mutable buffer','cache manager; written by layer','Maps logical sequence positions to physical cache pages.'),
 A('Q','Query heads','activation','current attention execution','Q [T,32,128] is returned by the fused store path.'),
 A('MHA','Attention accumulator','register/shared/workspace concept','selected target kernel','Computes causal scaled scores and a stable normalized weighted sum.'),
 A('O','Output projection','activation','current layer','Flattens heads, projects to width 4096 and feeds the residual add.')],[
 E('Project Q, K and V','For five active tokens, the semantic QKV result is [5,12288]. The 32×128 query, key and value portions each occupy 4096 channels.',{'QKV':B('projected',('qkv','QKV [5,12288]'))},[R('attention','qkv = self.qkv_proj(x)')]),
 E('Find each token’s position','Offsets [0,3,5] assign rows to A and B. Add each request’s old cache length to its local row position before applying rotary frequencies.',{'ROPE':B('positions selected','A positions 0,1,2','B positions 0,1')},[R('attention','xq = rope_split_store_ragged('),R('rope_kernel','input_row_offsets')],edges=[('QKV','ROPE')]),
 E('Rotate Q/K and write K/V','The fused mo.rope_split_store.ragged.paged operation returns Q and mutates the layer’s paged K/V. The safetensors route uses the split-half rotary convention.',{'Q':B('rotated query',('q','Q [5,32,128]')),'KV':B('new rows stored',('cache','layer l: A[0..2], B[0..1] K/V')),'QKV':B('consumed',facts=['QKV temporary lifetime follows its consumers'])},[R('kernels','mo.rope_split_store.ragged.paged'),R('attention','interleaved=self.rope.interleaved')],transfer=['QKV','fused split / rotate / store','KV'],edges=[('ROPE','Q'),('ROPE','KV')]),
 E('Bind the paged attention contract','The Python wrapper requires rank-3 Q, matching cache dtype, uint32 offsets/layer index and dispatch metadata. Scale is a CPU FP32 scalar.',{'MHA':B('validated operands','Q + row_offsets + KV leaves','layer_idx uint32; scale FP32','dispatch metadata')},[R('kernels','def flash_attention_ragged('),R('kernels','Expected attention_dispatch_metadata')],edges=[('Q','MHA'),('KV','MHA')]),
 E('Apply request-local causality','For A query position 1, visible keys are A0 and A1. A2 and every B key are excluded, even though all new K/V rows may already have been stored.',{'MHA':B('visible key set','A q1 → {A k0,A k1}','score = dot(Q,K) / sqrt(128)')},[R('ragged_kernel','return dispatch_mask['),R('mha','def flash_attention_dispatch[')]),
 E('Accumulate a stable tile','Maintain maximum m, denominator l and weighted numerator a. Rescale the previous tile when a new maximum arrives; final output is a/l.',{'MHA':B('illustrative first tile','scores [2,1]; values [10,20]','m=2; l=1+exp(-1)','a=10+20·exp(-1)')},[R('mha','def flash_attention_dispatch[')],edges=[('KV','MHA')]),
 E('Rescale on the next tile','For the teaching third score 3 with value 30, m becomes 3. Multiply both old l and old a by exp(2−3), then add the new contribution.',{'MHA':B('illustrative second tile','m=3; α=exp(-1)','l=α·l_old+1; a=α·a_old+30','output a/l ≈ 24.2051')},[R('mha','def flash_attention_dispatch[')]),
 E('Return attention output and project','The output has Q’s [T,32,128] shape. Reshape to [T,4096], apply O projection, then add the residual outside this attention module.',{'O':B('attention branch output',('o','[5,4096] → residual add')),'Q':B('consumed',facts=['Q is execution-local; K/V persist'])},[R('attention','return self.o_proj(attn_out)')],edges=[('MHA','O')])],[
 Q('The exact operation route','''
```text
AttentionWithRope.__call__
 ├─ qkv_proj(x)
 ├─ rope_split_store_ragged
 │   └─ mo.rope_split_store.ragged.paged
 ├─ reshape Q → [T,32,128]
 ├─ flash_attention_ragged (Python)
 │   └─ mo.mha.ragged.paged (registered Mojo operation)
 │       └─ _execute_mha_ragged_paged_scalar_args
 │           ├─ unmarshal dispatch metadata
 │           ├─ generic_get_paged_cache
 │           └─ generic_flash_attention_kv_cache_ragged
 │               └─ _flash_attention_dispatch → get K/V for layer
 │                   └─ dispatch_mask → GPU flash_attention[ragged=True]
 │                       └─ target / dtype / prefill-decode dispatch
 └─ reshape → o_proj
```

This route takes a mutable rank-6 KV buffer plus metadata. It does not replace the cache with the dense rank-3 K and V demo inputs. The model’s wrapper is explicitly paged; the similarly named non-cache ragged overload is a different overload.
''',[R('attention','qkv = self.qkv_proj(x)'),R('builtin_attention','@extensibility.register("mo.mha.ragged.paged")'),R('builtin','def _execute_mha_ragged_paged_scalar_args['),R('ragged_kernel','def generic_flash_attention_kv_cache_ragged[')],open=True),
 Q('Rotary coordinates and write ordering','''
For the safetensors split-half convention, pair head channels `i` and `i+64`, for `0 ≤ i < 64`. At position `p`, with frequency `θ_i`, rotate the pair as:

```text
q'_i    = q_i cos(pθ_i) − q_(i+64) sin(pθ_i)
q'_(i+64) = q_i sin(pθ_i) + q_(i+64) cos(pθ_i)
```

Apply the same positional rotation to K, not V. The selected rotary object supplies its actual frequency/scaling policy; the saved Llama 2 configuration has no Llama 3 scaling. Old cache length plus local prompt row gives the absolute position. Decode at old length 3 writes the newly processed token at position 3.

The default attention module fuses split, rotary transformation and KV store. Q is an explicit result; K/V are mutations to the cache. Compiler/runtime dependencies must order reads after the corresponding writes, including when outputs do not directly carry a tensor representing every mutated byte.

GGUF interleaving, Q/K normalization, sliding windows and quantized cache branches must be selected by their own conditions. They are not automatically part of this safetensors Llama 2 route.
''',[R('attention','_fuse_rope_and_store: bool = True'),R('attention','fuse=self._fuse_rope_and_store'),R('rope_kernel','interleaved'),R('config','interleaved_rope_weights =')],open=True),
 Q('Attention ABI: which operands cross the custom-op boundary?','''
| Operand | Contract / purpose |
|---|---|
| output, Q | rank-3 tensors `[T,32,128]`; output shape matches Q |
| input_row_offsets | uint32 rank 1; CSR-style request boundaries |
| kv_blocks | mutable rank-6 cache; contains all physical pages |
| page_stride | CPU int64 tensor; physical stride can exceed packed payload |
| cache_lengths | uint32 per request; old cached token counts |
| lookup_table | uint32 rank 2; logical page to physical block ID |
| max_prompt_length, max_cache_length | host-visible control values |
| layer_idx | uint32 scalar identifying layer’s cache view |
| scale | FP32 scalar, normally 1/√128 ≈ 0.08838835 |
| decode dispatch metadata | CPU int64 `[4]`: batch size, maximum active prompt width, resolved partition count, maximum valid cache length |
| DeviceContext | compiler-supplied call context for device operations |

Python first constructs symbolic values and parameters. MOGG call conformance later converts them to the implementation signature. At runtime the native/Mojo boundary unwraps actual buffer descriptors. These are three representations of the same contract, not three copies of every activation.
''',[R('kernels','values: MutableSequence[Value[Any]]'),R('builtin','def _execute_mha_ragged_paged_scalar_args['),R('attn_metadata','class MHAAttnKey')]),
 Q('Decode-dispatch metadata: four host integers, with one owner','''
```text
MHAAttnKey(batch_size, max_prompt_length, num_partitions)
                   + current max_cache_valid_length
                   ↓ pack_into
CPU int64 [ batch, prompt_width, partitions, cache_length ]
                   ↓ attention_dispatch_metadata input
registered paged MHA wrapper → unpack → target/decode dispatch
```

`DispatchMetadataSpec` is shared by the symbolic graph declaration and the runtime buffer producer, so rank, dtype and placement agree. `MHAAttnKey` declares shape `(4,)`, dtype `int64`, `on_device=False`. It stays on the CPU even though Q and KV live on the GPU; the generated host path reads this control information when selecting work.

The frozen key carries batch size, prompt width and an already resolved partition count. Current cache length is supplied to `pack_into`, so changing only that length does not create a different key identity. This is not proof of one fixed partition count or captured CUDA graph for every Llama request. Other attention families declare other metadata layouts; their accelerator-resident buffers must not be substituted for MHA's four host integers.
''',[R('attn_metadata','class DispatchMetadataSpec'),R('attn_metadata','class MHAAttnKey'),R('attn_metadata','def dispatch_metadata_buffers('),R('kv','def _attn_metadata_buffer(')]),
 Q('Paged reads and a precise causal mask','''
For a query with local active-row index `r` and old cache length `c`, its absolute position is `c+r`. Causal keys satisfy `0 ≤ k ≤ c+r` for the **same request**. The lookup table resolves each key position’s logical page; the slot chooses its row inside that page. Layer and head indices choose the corresponding vector.

```text
flattened Q row → request b via offsets → local row r
                                       ↓
                         absolute query position c[b]+r
                                       ↓
                         visible logical key positions
                                       ↓
                      lookup[b, key//P] → physical page
                                       ↓
                      page + layer + (key%P) + head → K/V
```

Padding is avoided in the flattened Q stream. Paging avoids requiring each request’s retained context to occupy one contiguous allocation. Neither mechanism removes the need for correct causal masking or makes attention across requests legal.
''',[R('ragged_kernel','kv_cache.get_key_cache'),R('kv_kernel','def generic_get_paged_cache[')]),
 Q('Online softmax: old values must survive the rescale','''
For each head/query row, process visible score/value tiles. Keep `m` (maximum), `l` (sum of normalized exponentials) and vector `a` (weighted numerator). Initially `m=-∞`, `l=0`, `a=0`.

```text
m_new = max(m_old, max(scores_tile))
α     = exp(m_old − m_new), or 0 for the empty prefix
p_j   = exp(score_j − m_new)
l_new = α·l_old + Σ p_j
a_new = α·a_old + Σ p_j·V_j
O     = a_new / l_new
```

The interactive calculation below the scene keeps the old tile’s operands visible. For actual Llama, `a` has 128 components per head/query, and the kernel’s accumulation precision, tiling, masking, partitioning and reduction strategy come from the selected implementation. The scalar calculator demonstrates the recurrence; it is not proof of a particular GPU instruction sequence, exact accumulation dtype or numerical parity.

Paged attention need not materialize the whole `[B,32,Q,K]` score matrix in global memory. A full logits matrix over the vocabulary is a different tensor produced by the output head.
''',[R('mha','def flash_attention_dispatch[')]),
 Q('Target dispatch, prefill and decode are real branches','''
The GPU path uses compile-time target/dtype/head-layout checks and runtime length/dispatch information. It can distinguish prefill from token generation and use target-specific implementations, partitions and fallback paths. A single-token decode has very different query parallelism from a long prefill; retained K/V length can still be large.

The source contains NVIDIA and AMD dispatch branches and multiple hardware-generation specializations. This course does not pick a tile size, grid, tensor-core variant or kernel count without fixing the target and observing the selected branch. Example 04’s `BN=16`, `BD=8`, grid `(4,2)` and block size 32 belong to its custom teaching kernel, not to Llama’s production attention.
''',[R('mha','def _flash_attention_kv_cache['),R('mha','var is_token_generation ='),R('mha','def flash_attention_dispatch[')])],'''
normalized hidden [T,4096]
└─ QKV projection → [T,12288]
   └─ fused rotary / split / KV-store
      ├─ Q [T,32,128]
      └─ K,V → layer-specific cache pages
         └─ paged causal attention(Q, offsets, lengths, lookup, layer, metadata)
            └─ [T,32,128] → reshape [T,4096] → O projection
''',[R('attention','xq = rope_split_store_ragged('),R('kernels','def flash_attention_ragged('),R('builtin','def _execute_mha_ragged_paged_scalar_args[')],lab='softmax')

C('kv_cache','KV pages: layout, addressing, capacity and ownership',
 'Follow one request’s logical positions into physical pages across all 32 layers, then calculate the cost without confusing payload with a memory plan.',
 'Model and construction',[
 A('REQ','Request positions','logical state','batch constructor / request context','Tracks processed tokens, active tokens and cache ownership.'),
 A('TABLE','Page lookup','metadata buffer','cache manager','Maps request logical page indices to physical IDs.'),
 A('POOL','Physical cache pool','GPU allocation','cache manager / allocator','Holds persistent K/V pages independently of per-execution activation workspace.'),
 A('LAYER','Layer view','borrowed view','current attention operation','Chooses K or V, layer, token slot, head and channel.'),
 A('PLAN','Memory budget','planning result','parent planner → worker','Committed cache budget controls allocation; illustrative arithmetic does not replace this plan.'),
 A('FREE','Reusable pages','allocator state','cache manager','A page can be reused only when ownership and asynchronous-use rules allow it.')],[
 E('Plan payload per cached token','One token stores both K and V for 32 layers × 32 heads × 128 channels. With two bytes per element this is 524,288 bytes.',{'PLAN':B('illustrative payload','2×32×32×128×2 = 512 KiB/token')},[R('kv','def shape_per_block('),R('kv','def bytes_per_value_block(')]),
 E('Reserve logical pages for a request','Using an illustrative page size P=128, a 130-token request needs two logical pages. Real allocation follows the committed plan and page geometry.',{'REQ':B('130 positions','logical page 0: positions 0…127','logical page 1: positions 128…129'),'TABLE':B('mapping assigned','A page 0 → physical 7','A page 1 → physical 2')},[R('kv','page_size'),R('planner','class PagedMemoryPlanner')],edges=[('REQ','TABLE')]),
 E('Use a rank-6 buffer view','The ordinary full-cache view is [pages,2,layers,P,KVheads,head_dim]. Physical page stride is passed separately and may include padding.',{'POOL':B('allocated by cache manager',('page7','physical page 7'),('page2','physical page 2'),facts=['[pages,2,32,128,32,128] example'])},[R('kv','def shape_per_block('),R('kv','def page_stride_of(')],edges=[('TABLE','POOL')]),
 E('Resolve position 129 for layer 5','129//128=1 and 129%128=1. Lookup A[1] gives physical page 2; select K/V, layer 5, slot 1 and a head/channel.',{'LAYER':B('borrowed address','page=2; slot=1; layer=5','K or V; head h; channel d')},[R('kv_kernel','def generic_get_paged_cache['),R('kv_types','page')],edges=[('POOL','LAYER')]),
 E('Append without reallocating all previous tokens','Decode writes the new token’s K/V at the next logical position in each layer. Existing prefix entries remain available to attention reads.',{'REQ':B('new active token','old cached length=130','write position=130; slot=2'),'LAYER':B('append address','page 2 / slot 2 / layer 0…31')},[R('rope_kernel','k_cache.cache_length(bi)'),R('pipeline','self._kv_manager.step(ctx)')],edges=[('REQ','LAYER')]),
 E('Advance lengths after execution/context update','The pipeline updates contexts before kv_manager.step. Sampled output becomes a future decoder input; do not mark its K/V as already computed by sampling.',{'REQ':B('processed-length update','processed input rows reflected in cache lengths')},[R('pipeline','update_context_and_prepare_responses('),R('pipeline','self._kv_manager.step(ctx)')]),
 E('Return page ownership only when safe','Request completion/cancellation triggers the cache lifecycle in the batch-construction layer. Prefix-cache references or pending uses can keep pages live.',{'FREE':B('eligible after last ownership/use','page IDs return to pool when safe'),'POOL':B('pool retained','allocation survives request release')},[R('pipeline','Primary and extra KV cache lifecycle'),R('kv','enable_prefix_caching')])],[
 Q('Logical shape and physical stride','''
The ordinary, unquantized, all-layers buffer has logical shape:

```text
[physical_pages, K_or_V=2, layers=32, tokens_per_page=P, KV_heads=32, head_dim=128]
```

For a packed row-major illustration, element offset within a page is `((((kv*32 + layer)*P + slot)*32 + head)*128 + channel)`. The physical page base adds `physical_page_id × resolved_page_stride` **in elements**, then multiplication by dtype bytes converts the total to a byte offset.

The CPU stride input uses `PACKED_PAGE_STRIDE = -1` as the packed-layout sentinel; it is not a negative memory address. The cache constructor derives the packed product for that sentinel. For a padded slab, use the explicit element stride. `contiguous_page_view_and_stride` presents the contiguous logical view required by model execution while separately passing the physical distance between pages. Its backing arena retains the full physical span even when the packed view is shorter.

The `per_layer_buffers` branch uses a layer extent of one and separate buffers. Quantized caches add scale leaves and possibly packed head storage. These are explicit alternatives; the two-byte full-cache arithmetic here selects none of them.
''',[R('kv','def shape_per_block('),R('kv','PACKED_PAGE_STRIDE = -1'),R('kv','def page_stride_of('),R('kv','def contiguous_page_view_and_stride(')],open=True),
 Q('Capacity arithmetic that can be checked','''
| Quantity | Two-byte example |
|---|---:|
| K and V per token per layer | 2 × 32 × 128 × 2 = 16,384 bytes |
| per token over 32 layers | **524,288 bytes = 512 KiB** |
| 128-token page over 32 layers | **67,108,864 bytes = 64 MiB** |
| 4096-token request, payload only | **2,147,483,648 bytes = 2 GiB** |
| B requests of S tokens | `B × S × 524288` bytes, before page rounding |
| page-rounded reservation | `B × ceil(S/P) × P × 524288`, without sharing |

These numbers exclude lookup tables, lengths, dispatch metadata, alignment/padding, weights, temporary activations, workspace, runtime reserve and any extra cache features. The page-size choices in the calculator are illustrative, not assertions of this invocation’s resolved default.

Llama 2 7B has 32 KV heads. Substituting a modern model’s eight KV heads understates this cache payload by four times.
''',[R('kv','def bytes_per_value_block(')],open=True),
 Q('Memory plan is not current occupancy','''
The planner combines model policies, device memory information and serving limits to produce the worker’s memory plan. `available_cache_memory` is the committed cache budget passed to KV manager construction. Free device bytes, an initial candidate budget, the capped allocation and occupied pages are different values.

Request admission can be limited by sequences, active input tokens, resident KV tokens and available pages. A maximum request count does not guarantee every request can simultaneously occupy the full 4096-token context. Prefix sharing, when enabled and valid, changes ownership/accounting; it does not reduce the per-token vector dimensions.
''',[R('pipeline','available_cache_memory = memory_plan.available_cache_memory'),R('pipeline','self._kv_manager = load_kv_manager('),R('planner','class PagedMemoryPlanner')]),
 Q('Mutation, ownership and prefix correctness','''
A compiled graph receives a borrowed view of live cache buffers. The cache manager owns allocation and page ownership; the attention kernels read/write the selected layer’s regions. MGP’s temporary-buffer reuse pass cannot independently hand a persistent request KV page to an unrelated activation.

For two requests, equal physical page IDs require deliberate valid sharing; equality is not an automatic consequence of equal tokens. A shared prefix must remain stable while referenced. New writes require the cache policy’s ownership/copy rules. Completion and cancellation must release logical references while respecting pending asynchronous accesses.

The pipeline’s `release()` docstring explicitly assigns primary/extra KV lifecycle to the batch constructor. Do not teach `TextGenerationPipeline.release` as freeing every KV allocation itself.
''',[R('pipeline','Primary and extra KV cache lifecycle'),R('kv','enable_prefix_caching'),R('kernels','mo.rope_split_store.ragged.paged')])],'''
MemoryPlan.available_cache_memory
└─ load_kv_manager → physical page pool
   ├─ request ownership → logical page IDs → lookup table
   └─ graph inputs: kv_blocks + page_stride + lengths + lookup + metadata
      └─ layer_idx → K/V views → write new rows → causal reads

request progress → context update → kv_manager.step
request completion/cancel → batch-constructor cache lifecycle → reusable pages
''',[R('pipeline','self._kv_manager = load_kv_manager('),R('kv','def shape_per_block('),R('pipeline','self._kv_manager.step(ctx)')],lab='kv')

C('inputs_shapes','Input leaves, symbolic shapes and staging',
 'Match each model.execute argument to its dtype, device, shape, producer and lifetime.',
 'Model and construction',[
 A('HOST','Host batch description','request metadata','batch processor','Contains active token rows, return-logit policies and cache metadata.'),
 A('GPU','Device batch buffers','input allocations','prepared batch','Contains token IDs, offsets and the required device KV metadata.'),
 A('CPU','Host control tensors','input buffers','prepared batch','Small CPU values such as return_n_logits and page stride remain CPU inputs.'),
 A('TREE','Named input tree','schema','Llama3 / GraphSpec','Defines leaf order and group meaning.'),
 A('ARGS','Flat engine arguments','borrowed buffers','M1.execute','Must satisfy positional rank/dtype/device/static-dimension contracts.'),
 A('SYM','Shape bindings','runtime shape values','execution validation / generated host code','Binds symbolic T, batch and page counts for this execution.')],[
 E('Start with two request rows','Illustrative prompts have lengths 3 and 2. Batch preparation constructs one flattened token stream and prefix-sum offsets.',{'HOST':B('two requests','A: [A0,A1,A2]','B: [B0,B1]')},[R('batch','def prepare_initial_token_inputs(')]),
 E('Stage token and offset buffers','Tokens use int64; row offsets use uint32. Both are placed on the model device. Their values are not the symbolic input types.',{'GPU':B('device batch inputs',('tok','tokens int64 [5]'),('off','offsets uint32 [0,3,5]'))},[R('batch','def prepare_initial_token_inputs('),R('batch_types','def ragged_batch_inputs(')],edges=[('HOST','GPU')]),
 E('Keep CPU control leaves on the CPU','return_n_logits, page_stride and maximum-length controls use their declared host placement. Small controls still participate in validation and dependencies.',{'CPU':B('host controls','return_n_logits int64','page_stride int64 [1]','max_prompt/max_cache uint32 [1]')},[R('kv','page_stride=TensorType('),R('batch_types','return_n_logits')],edges=[('HOST','CPU')]),
 E('Read the schema’s leaf order','The named tree groups batch and KV inputs. Single-device Llama excludes allreduce signal buffers. Optional KV leaves exist only under their feature conditions.',{'TREE':B('ordered schema','batch → tokens, row_offsets, return_n_logits','kv → configured cache leaves')},[R('batch_types','def ragged_kv_symbolic_input_tree('),R('llama','include_signal_buffers=False')]),
 E('Bind a flat buffer list to M1','Llama3Model.execute expands model_inputs.buffers. Native validation checks count and each expected contract before preparing AsyncValues.',{'ARGS':B('flat positional inputs','*model_inputs.buffers'),'SYM':B('this execution','T=5; B=2','offset count=B+1=3')},[R('model','self.model.execute('),R('session_native','prepareInput')],edges=[('GPU','ARGS'),('CPU','ARGS'),('TREE','ARGS')]),
 E('Next decode changes values and dynamic dimensions','With one active token per request, T becomes 2 and offsets become [0,1,2]. Old cache lengths are [3,2]. The model’s static head/hidden dimensions stay fixed.',{'GPU':B('next decode inputs',('tok','tokens int64 [2]'),('off','offsets uint32 [0,1,2]')),'SYM':B('rebound for decode','T=2; B=2','old cache_lengths=[3,2]')},[R('batch','def prepare_initial_token_inputs('),R('pipeline','self._kv_manager.step(ctx)')])],[
 Q('The input contract table','''
| Leaf / group | Dtype | Shape meaning | Placement / origin |
|---|---|---|---|
| batch.tokens | int64 | `[T]` | GPU; active token IDs |
| batch.row_offsets | uint32 | `[B+1]` | GPU; flattened row boundaries |
| batch.return_n_logits | int64 | symbolic `[return_n_logits]`; this preparation binds a one-element array | CPU; scalar count stored in that array |
| kv.kv_blocks | cache dtype | `[pages,2,32,P,32,128]` ordinary example | GPU; persistent cache pool |
| kv.page_stride | int64 | `[1]` | CPU; actual bound buffer stride |
| kv.cache_lengths | uint32 | `[B]` | GPU; old processed lengths |
| kv.lookup_table | uint32 | `[B,max_num_pages]` | GPU; physical page IDs |
| kv.max_prompt_length | uint32 | `[1]` | CPU; batch control |
| kv.max_cache_length | uint32 | `[1]` | CPU; batch control |
| kv.attention_dispatch_metadata | int64 | `[4]`: batch, prompt width, partitions, cache length | CPU; `MHAAttnKey.pack_into` |

Quantized-cache scales, speculative metadata and per-layer buffers appear only when configured. The exact instantiated tree is authoritative; copying a hard-coded argument count from another model or MAX revision is unsafe. Layer index and attention scale are constructed inside the traced model, not extra top-level user token inputs.
''',[R('batch_types','def ragged_batch_inputs('),R('batch_types','np.array([return_n_logits], dtype=np.int64)'),R('kv','page_stride=TensorType('),R('attn_metadata','class MHAAttnKey')],open=True),
 Q('Flatten/unflatten preserves a schema, not just a list','''
```text
named tree → tree.leaves → Graph input_types / engine argument positions
                              ↓
trace forward(tokens, offsets, return_n, *kv_inputs)
                              ↓
unflatten_kv_inputs → KVCacheInputsPerDevice → attention helpers
```

GraphSpec preserves the input tree definition. Serving consumers inspect group kinds, so the decoder’s inputs must be named under recognized groups. Output tree preservation solves the reverse mapping for decoder outputs. The binding contract is independent of whichever Python variable happened to hold a buffer.
''',[R('batch_types','def ragged_kv_symbolic_input_tree('),R('llama','self.kv_params.unflatten_kv_inputs'),R('spec','def _check_inputs(')],open=True),
 Q('Static dimension, symbolic dimension and data-dependent value','''
`4096`, `32` heads and `128` channels are static model dimensions. `T` and batch size vary per execution. Offsets, token IDs and page IDs are tensor contents. A shape expression such as `B+1` constrains a buffer’s length without specifying its contents.

Shape inference propagates relationships; constant folding may compute parameter-only expressions. Runtime host code still handles shapes that depend on this batch. A constant dimension folded on the host does not mean all token values are known at compile time. An int64 shape parameter does not change BF16 activation elements into int64.

For prefill `[3,2]`, the last-token rows are `offsets[1:]-1=[2,4]`; for decode `[1,1]`, they are `[0,1]`. This concrete relation connects graph shapes to semantic request boundaries.
''',[R('graph','self._params = dict.fromkeys('),R('transformer','input_row_offsets[1:] - 1'),R('compiler','buildShapeInferencePipeline')]),
 Q('Buffer visibility and boundary validation','''
Staging allocates or reuses host/device buffers according to the preparation path. A `.to(device)` transfer is a runtime operation with ordering/lifetime requirements, not the same operation as constructing a `DeviceRefAttr` in G1.

Native execution validates dtype, fixed shape constraints and device placement, wraps storage and prepares inputs as AsyncValues. It releases the GIL around execution because native ownership paths may interact with Python/DLPack deleters. Output Buffer creation and host-readable tensor data remain separate gates, explained in the driver and serving lessons.
''',[R('batch','def prepare_initial_token_inputs('),R('session_native','prepareInput'),R('session_native','gil_scoped_release')])],'''
request contexts → Llama3BatchProcessor.prepare_initial_token_inputs
├─ token IDs / row offsets → model-device buffers
├─ host return_n_logits control
└─ KV manager binds persistent pages + lengths + lookup + metadata
   └─ model_inputs.buffers (schema order)
      └─ Llama3Model.execute → M1.execute(*buffers)
         ├─ native argument validation
         └─ prepareInput → AsyncValues / shape bindings
''',[R('batch','def prepare_initial_token_inputs('),R('batch_types','def ragged_kv_symbolic_input_tree('),R('model','self.model.execute(')])
