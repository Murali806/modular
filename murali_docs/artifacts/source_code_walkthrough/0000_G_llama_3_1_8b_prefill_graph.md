<a id="llama-3-1-8b-prefill-graph"></a>

# Llama 3.1 8B Prefill Graph in MAX

This document explains only the **prefill architecture** for:

```text
meta-llama/Llama-3.1-8B-Instruct
```

It follows the stage-by-stage format of the referenced Llama/QNN prefill
summaries, but uses Llama 3.1 8B dimensions and the MAX attention operations.

Concrete example:

```text
batch size                 B = 1
prompt length              S = 512 tokens
previous cached tokens     P = 0
returned vocabulary rows     = 1, for prompt position 511
```

Sections 0-4 cover prefill execution. Section 5 is a startup appendix that
traces how MAX builds, compiles, and initializes this prefill graph. Scheduling,
decode iterations, and sampling remain out of scope.

<details>
<summary><strong>0. Model architecture</strong></summary>

```text
hidden dimension                 D   = 4096
transformer layers               L   = 32
query heads                      Hq  = 32
key/value heads                  Hkv = 8
head dimension                   hd  = 128
query heads per KV head              = 4
MLP intermediate dimension       F   = 14336
vocabulary size                  V   = 128256
RMSNorm epsilon                      = 1e-5
RoPE theta                           = 500000
maximum positions                    = 131072
```

The important Llama 3.1 difference is grouped-query attention (GQA):

```text
Q width = 32 x 128 = 4096
K width =  8 x 128 = 1024
V width =  8 x 128 = 1024
```

Four query heads share one K head and one V head.

</details>

<details>
<summary><strong>1. Complete prefill pass</strong></summary>

```text
Prompt text
    |
    +-- CPU tokenizer
    |     text -> token IDs [512]
    |
    +-- MAX Llama prefill graph
          |
          +-- token embedding
          |     [512] -> [512,4096]
          |
          +-- repeat for layers 0..31
          |     |
          |     +-- pre-attention RMSNorm
          |     +-- Q/K/V projection
          |     +-- RoPE on Q and K
          |     +-- write K and V into the paged KV cache
          |     +-- causal flash attention
          |     +-- attention output projection
          |     +-- residual add
          |     +-- post-attention RMSNorm
          |     +-- SwiGLU MLP
          |     +-- residual add
          |
          +-- select prompt position 511
          +-- final RMSNorm
          +-- LM-head projection
          |
          v
      logits [1,128256]

Side output:
    512 valid K/V positions in every transformer layer
```

Prefill processes all 512 prompt tokens in one model invocation. It produces
the first next-token logits and creates the KV-cache state used by decode.

</details>

<details>
<summary><strong>2. Stage-by-stage tensor flow</strong></summary>

The primary table remains five columns wide. Each stage is immediately
followed by an expandable `-> xx details` row containing its compute
derivation, logical-memory calculation, and MAX execution reality.

The calculations use `S=512`, BF16 activations/weights/KV values, and FP32
final logits. Memory values are ideal logical reads and writes, not measured
HBM traffic; fusion, caching, tiling, and quantization can reduce physical
traffic.

| Stage | Input | Formula | Output | Purpose |
|---|---|---|---|---|
| 00 Tokenizer | prompt text; tokenizer vocabulary/configuration | `ids = tokenizer(text)` | token IDs `[512]` | Convert text to Llama vocabulary IDs on the CPU. |
|  | <details><summary><strong>-> 00 details</strong></summary><strong>COMPUTE</strong><br>There is no fixed matrix-MAC formula. Work depends on prompt bytes, tokenizer rules, and the number of produced tokens.<br><br><strong>LOGICAL MEMORY</strong><br>Write IDs: <code>512*8 = 4096 bytes = 4 KiB</code>.<br><strong>Logical tensor traffic:</strong> 4 KiB of token output plus prompt/tokenizer reads.<br><strong>Fused-kernel traffic:</strong> not applicable; this is CPU preprocessing.<br><strong>Actual HBM traffic:</strong> none at this stage; measure host-DRAM traffic if needed.<br><br><strong>MAX REALITY</strong><br>Tokenization runs on the CPU before the MAX model graph. Its implementation, vocabulary lookup structure, and cache behavior determine the cost.</details> |  |  |  |
| 01 Embedding | `ids [512]`; embedding weight `E [128256,4096]` | `X = E[ids]` | `X [512,4096]` | Gather one learned 4096-value embedding row per token. |
|  | <details><summary><strong>-> 01 details</strong></summary><strong>COMPUTE</strong><br>No dense MACs.<br><code>512</code> indexed row lookups, each returning <code>4096</code> values.<br><br><strong>LOGICAL MEMORY</strong><br>Read selected rows: <code>512*4096*2 = 4,194,304 bytes = 4 MiB</code>.<br>Write X: <code>4 MiB</code>.<br>Total: <code>8 MiB</code>.<br><strong>Logical tensor traffic:</strong> selected rows plus X output.<br><strong>Fused-kernel traffic:</strong> X may remain device-resident for the next norm.<br><strong>Actual HBM traffic:</strong> depends on embedding dtype, cache hits, and duplicate IDs.<br><br><strong>MAX REALITY</strong><br>The model gathers only requested embedding rows. Quantized embeddings may add unpacking and dequantization, while repeated token IDs may benefit from cache reuse.</details> |  |  |  |
| 02 Pre-attention RMSNorm | `X [512,4096]`; `gamma_attn [4096]`; `eps` | `Xnorm = (X / sqrt(mean(X^2) + eps)) * gamma_attn` | `Xnorm [512,4096]` | Normalize each token row before attention. |
|  | <details><summary><strong>-> 02 details</strong></summary><strong>COMPUTE</strong><br>Elements: <code>S*D = 512*4096 = 2,097,152</code>.<br>Approximate FLOPs: <code>(4 to 5)*2,097,152 = 8.4 to 10.5 MFLOPs</code>.<br>Also: <code>512</code> reciprocal-square-roots.<br><br><strong>LOGICAL MEMORY</strong><br>Read X: <code>4 MiB</code>.<br>Read gamma: <code>4096*2 = 8 KiB</code>.<br>Write Xnorm: <code>4 MiB</code>.<br>Total: <code>8.008 MiB</code>.<br><strong>Logical tensor traffic:</strong> one X read and one Xnorm write.<br><strong>Fused-kernel traffic:</strong> the Xnorm write/read can potentially be avoided.<br><strong>Actual HBM traffic:</strong> backend and fusion dependent; verify with a profiler.<br><br><strong>MAX REALITY</strong><br>RMSNorm is a reduction followed by row-wise scaling. A fused implementation can retain reduction data and may feed Xnorm directly into QKV projection.</details> |  |  |  |
| 03 Q projection | `Xnorm [512,4096]`; `Wq [4096,4096]` | `Qflat = Xnorm @ Wq^T`; `Q = reshape(Qflat)` | `Q [512,32,128]` | Create 32 query heads. |
|  | <details><summary><strong>-> 03 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*4096*4096 = 8,589,934,592</code>.<br>FLOPs: <code>2*MACs = 17,179,869,184 = 17.18 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Xnorm: <code>4 MiB</code>.<br>Read Wq: <code>32 MiB</code>.<br>Write Q: <code>4 MiB</code>.<br>Total: <code>40 MiB</code>.<br><strong>Logical tensor traffic:</strong> 40 MiB for standalone Q.<br><strong>Fused-kernel traffic:</strong> combined QKV minimum is 58 MiB across stages 03-05 instead of 66 MiB separately.<br><strong>Actual HBM traffic:</strong> depends on weight encoding, caches, and kernel tiling.<br><br><strong>MAX REALITY</strong><br>MAX can execute Q/K/V through one combined projection. In that case Xnorm is read once and the combined output is split logically.</details> |  |  |  |
| 04 K projection | `Xnorm [512,4096]`; `Wk [1024,4096]` | `Kflat = Xnorm @ Wk^T`; `K = reshape(Kflat)` | `K [512,8,128]` | Create 8 shared key heads. |
|  | <details><summary><strong>-> 04 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*4096*1024 = 2,147,483,648</code>.<br>FLOPs: <code>2*MACs = 4,294,967,296 = 4.29 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Xnorm: <code>4 MiB</code>.<br>Read Wk: <code>8 MiB</code>.<br>Write K: <code>1 MiB</code>.<br>Total: <code>13 MiB</code>.<br><strong>Logical tensor traffic:</strong> 13 MiB standalone.<br><strong>Fused-kernel traffic:</strong> Xnorm is shared with Q/V in combined QKV.<br><strong>Actual HBM traffic:</strong> depends on combined-kernel reuse and weight encoding.<br><br><strong>MAX REALITY</strong><br>K is commonly produced by the same combined QKV projection as Q and V. Its smaller width comes from eight KV heads rather than 32 query heads.</details> |  |  |  |
| 05 V projection | `Xnorm [512,4096]`; `Wv [1024,4096]` | `Vflat = Xnorm @ Wv^T`; `V = reshape(Vflat)` | `V [512,8,128]` | Create 8 shared value heads. |
|  | <details><summary><strong>-> 05 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*4096*1024 = 2,147,483,648</code>.<br>FLOPs: <code>2*MACs = 4,294,967,296 = 4.29 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Xnorm: <code>4 MiB</code>.<br>Read Wv: <code>8 MiB</code>.<br>Write V: <code>1 MiB</code>.<br>Total: <code>13 MiB</code>.<br><strong>Logical tensor traffic:</strong> 13 MiB standalone.<br><strong>Fused-kernel traffic:</strong> Xnorm is shared with Q/K in combined QKV.<br><strong>Actual HBM traffic:</strong> depends on combined projection and direct cache-store behavior.<br><br><strong>MAX REALITY</strong><br>V is generated with Q and K in the combined projection and is written to the KV cache without RoPE.</details> |  |  |  |
| 06 RoPE and KV write | Q, K, V; token positions; RoPE frequencies; K/V-cache buffers and page metadata | `Qrot = RoPE(Q,pos)`; `Krot = RoPE(K,pos)`; `Kcache[P:P+S] = Krot`; `Vcache[P:P+S] = V` | rotated Q plus cached K/V | Encode positions in Q/K and append this layer's K/V values. For this example, `P=0`. |
|  | <details><summary><strong>-> 06 details</strong></summary><strong>COMPUTE</strong><br>Rotated values: <code>S*(4096+1024) = 2,621,440</code>.<br>Rotation pairs: <code>1,310,720</code>.<br>At about six FLOPs per pair: <code>7,864,320 = 7.86 MFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Q/K/V: <code>6 MiB</code>.<br>Write Qrot/Kcache/Vcache: <code>6 MiB</code>.<br>Read RoPE data: about <code>128 KiB</code>.<br>Total: <code>12.125 MiB</code>.<br><strong>Logical tensor traffic:</strong> split, rotate, and cache writes.<br><strong>Fused-kernel traffic:</strong> avoids separate split, RoPE-output, and cache-copy passes.<br><strong>Actual HBM traffic:</strong> depends on page layout and on-chip residency.<br><br><strong>MAX REALITY</strong><br><code>rope_split_store_ragged</code> splits the combined QKV result, rotates Q/K, and writes K/V directly into paged cache storage.</details> |  |  |  |
| 07a QK score | `Qrot [512,32,128]`, so `Qrot_h [512,128]`; logical `Kcache [512,8,128]`, so `Kcache_g(h) [512,128]`; scalar scale `1/sqrt(128)` | `g(h) = floor(h/4)`; `Scores_h = Qrot_h @ Kcache_g(h)^T / sqrt(128)` | `Scores_h [512,512]` for each Q head | Compare every query position with the permitted key positions. Four Q heads share each KV head. |
|  | <details><summary><strong>-> 07a details</strong></summary><strong>COMPUTE</strong><br>Per-head MACs: <code>512*512*128 = 33,554,432</code>.<br>Per-head FLOPs: <code>67,108,864 = 67.11 MFLOPs</code>.<br>All-head FLOPs: <code>32*67,108,864 = 2,147,483,648 = 2.15 GFLOPs</code> dense upper bound.<br><br><strong>LOGICAL MEMORY</strong><br>Per head: read Qrot_h <code>128 KiB</code>; read Kcache_g(h) <code>128 KiB</code>; write Scores_h <code>512 KiB</code> BF16 or <code>1 MiB</code> FP32.<br><strong>Logical tensor traffic:</strong> full scores would occupy 16 MiB BF16 or 32 MiB FP32 across heads.<br><strong>Fused-kernel traffic:</strong> score tiles remain on chip and feed softmax directly.<br><strong>Actual HBM traffic:</strong> depends on KV-head reuse, tile size, and cache residency.<br><br><strong>MAX REALITY</strong><br><code>flash_attention_ragged</code> computes QK scores in tiles. GQA allows four query heads to reuse one KV head, and the complete Scores tensor is normally never written to HBM.</details> |  |  |  |
| 07b Causal mask and softmax | `Scores_h [512,512]`; logical causal mask `[512,512]`, shared across heads | `Prob_h = softmax(causal_mask(Scores_h))` | `Prob_h [512,512]` for each Q head | Remove future positions and convert the remaining scores into attention probabilities. |
|  | <details><summary><strong>-> 07b details</strong></summary><strong>COMPUTE</strong><br>Valid score elements are causally triangular, while the dense upper bound is <code>32*512*512 = 8,388,608</code> elements.<br>Using about five scalar operations per element: <code>41,943,040 = 41.94 MFLOPs</code>. Exponential cost is implementation-dependent.<br><br><strong>LOGICAL MEMORY</strong><br>Per head: read Scores_h and write Prob_h, each <code>512 KiB</code> BF16 or <code>1 MiB</code> FP32.<br>Logical bool mask: <code>512*512 = 256 KiB</code>.<br><strong>Logical tensor traffic:</strong> Scores read, mask application, and Prob write.<br><strong>Fused-kernel traffic:</strong> Scores, mask, and Prob remain tiled/on-chip rather than becoming full tensors.<br><strong>Actual HBM traffic:</strong> normally excludes these intermediates; profile the selected flash kernel.<br><br><strong>MAX REALITY</strong><br>The causal condition is derived from token positions and row offsets. Flash attention applies it while processing each score tile, then performs row-wise max, exponentiation, sum, and normalization.</details> |  |  |  |
| 07c Probability-value product | `Prob_h [512,512]`; logical `Vcache [512,8,128]`, so `Vcache_g(h) [512,128]` | `A_h = Prob_h @ Vcache_g(h)` | `A_h [512,128]`; combined `A [512,32,128]` | Compute the weighted value-vector output for every query head. |
|  | <details><summary><strong>-> 07c details</strong></summary><strong>COMPUTE</strong><br>Per-head MACs: <code>512*512*128 = 33,554,432</code>.<br>Per-head FLOPs: <code>67,108,864 = 67.11 MFLOPs</code>.<br>All-head FLOPs: <code>2.15 GFLOPs</code> dense upper bound.<br><br><strong>LOGICAL MEMORY</strong><br>Per head: read Prob_h <code>512 KiB</code> BF16 or <code>1 MiB</code> FP32; read Vcache_g(h) <code>128 KiB</code>; write A_h <code>128 KiB</code>.<br><strong>Logical tensor traffic:</strong> full Prob read plus V read and A write.<br><strong>Fused-kernel traffic:</strong> eliminates the full Prob write/read; only tiled V reads and final A writes remain.<br><strong>Actual HBM traffic:</strong> depends on GQA V reuse and cache locality.<br><br><strong>MAX REALITY</strong><br>Flash attention multiplies each on-chip probability tile by the matching V tile and accumulates A_h. Four query heads can reuse the same V head under GQA.</details> |  |  |  |
| 08 Output projection | `A [512,32,128]`; `Wo [4096,4096]` | `AttnOut = reshape(A,[512,4096]) @ Wo^T` | `AttnOut [512,4096]` | Mix the 32 attention-head outputs. |
|  | <details><summary><strong>-> 08 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*4096*4096 = 8,589,934,592</code>.<br>FLOPs: <code>17,179,869,184 = 17.18 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read A: <code>4 MiB</code>.<br>Read Wo: <code>32 MiB</code>.<br>Write AttnOut: <code>4 MiB</code>.<br>Total: <code>40 MiB</code>.<br><strong>Logical tensor traffic:</strong> A, Wo, and AttnOut.<br><strong>Fused-kernel traffic:</strong> reshape requires no physical copy; other fusion is backend-dependent.<br><strong>Actual HBM traffic:</strong> depends on Wo encoding and cache/tiling behavior.<br><br><strong>MAX REALITY</strong><br>The reshape from 32 heads to width 4096 is normally a logical view. The output-projection matmul performs the actual mixing of head channels.</details> |  |  |  |
| 09 Residual add | `X [512,4096]`; `AttnOut [512,4096]` | `H = X + AttnOut` | `H [512,4096]` | Preserve the layer input around attention. |
|  | <details><summary><strong>-> 09 details</strong></summary><strong>COMPUTE</strong><br>Additions: <code>512*4096 = 2,097,152</code>.<br>FLOPs: <code>2.10 MFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read X: <code>4 MiB</code>.<br>Read AttnOut: <code>4 MiB</code>.<br>Write H: <code>4 MiB</code>.<br>Total: <code>12 MiB</code>.<br><strong>Logical tensor traffic:</strong> two reads and one write.<br><strong>Fused-kernel traffic:</strong> H may be consumed without a separate HBM round trip.<br><strong>Actual HBM traffic:</strong> compiler and lifetime dependent.<br><br><strong>MAX REALITY</strong><br>This element-wise add may be fused with an adjacent operation, such as output handling or the following RMSNorm, depending on compiler decisions.</details> |  |  |  |
| 10 Post-attention RMSNorm | `H [512,4096]`; `gamma_mlp [4096]`; `eps` | `Hnorm = (H / sqrt(mean(H^2) + eps)) * gamma_mlp` | `Hnorm [512,4096]` | Normalize before the MLP. |
|  | <details><summary><strong>-> 10 details</strong></summary><strong>COMPUTE</strong><br>Elements: <code>2,097,152</code>.<br>Approximate FLOPs: <code>8.4 to 10.5 MFLOPs</code>.<br>Also: <code>512</code> reciprocal-square-roots.<br><br><strong>LOGICAL MEMORY</strong><br>Read H: <code>4 MiB</code>.<br>Read gamma: <code>8 KiB</code>.<br>Write Hnorm: <code>4 MiB</code>.<br>Total: <code>8.008 MiB</code>.<br><strong>Logical tensor traffic:</strong> one H read and one Hnorm write.<br><strong>Fused-kernel traffic:</strong> Hnorm write/read may be avoided before Gate/Up.<br><strong>Actual HBM traffic:</strong> backend and fusion dependent.<br><br><strong>MAX REALITY</strong><br>The operation is structurally the same as stage 02. Hnorm may feed a combined Gate/Up projection without a long-lived intermediate.</details> |  |  |  |
| 11 Gate projection | `Hnorm [512,4096]`; `Wgate [14336,4096]` | `Gate = Hnorm @ Wgate^T` | `Gate [512,14336]` | Produce the SwiGLU gate branch. |
|  | <details><summary><strong>-> 11 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*4096*14336 = 30,064,771,072</code>.<br>FLOPs: <code>60,129,542,144 = 60.13 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Hnorm: <code>4 MiB</code>.<br>Read Wgate: <code>112 MiB</code>.<br>Write Gate: <code>14 MiB</code>.<br>Total: <code>130 MiB</code>.<br><strong>Logical tensor traffic:</strong> 130 MiB standalone.<br><strong>Fused-kernel traffic:</strong> combined Gate/Up traffic is 256 MiB rather than 260 MiB.<br><strong>Actual HBM traffic:</strong> depends on weight encoding and combined-linear tiling.<br><br><strong>MAX REALITY</strong><br><code>StackedMLP</code> can calculate Gate and Up through one combined linear projection, sharing the Hnorm read and producing a combined 28 MiB output.</details> |  |  |  |
| 12 Up projection | `Hnorm [512,4096]`; `Wup [14336,4096]` | `Up = Hnorm @ Wup^T` | `Up [512,14336]` | Produce the SwiGLU value branch. |
|  | <details><summary><strong>-> 12 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*4096*14336 = 30,064,771,072</code>.<br>FLOPs: <code>60,129,542,144 = 60.13 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Hnorm: <code>4 MiB</code>.<br>Read Wup: <code>112 MiB</code>.<br>Write Up: <code>14 MiB</code>.<br>Total: <code>130 MiB</code>.<br><strong>Logical tensor traffic:</strong> 130 MiB standalone.<br><strong>Fused-kernel traffic:</strong> Hnorm is read once for combined Gate/Up.<br><strong>Actual HBM traffic:</strong> depends on stacked-weight layout, quantization, and kernel tiling.<br><br><strong>MAX REALITY</strong><br>Up is the second half of the combined Gate/Up projection when stacked MLP weights are used. It is consumed element-wise with the activated Gate branch.</details> |  |  |  |
| 13a SwiGLU activation | `Gate [512,14336]`; `Up [512,14336]` | `Activated = SiLU(Gate) * Up` | `Activated [512,14336]` | Apply the gated nonlinear activation. |
|  | <details><summary><strong>-> 13a details</strong></summary><strong>COMPUTE</strong><br>Elements: <code>512*14336 = 7,340,032</code>.<br>Using about six scalar operations per element: <code>44,040,192 = 44.04 MFLOPs</code>.<br>The exponential implementation changes the exact cost.<br><br><strong>LOGICAL MEMORY</strong><br>Read Gate: <code>14 MiB</code>.<br>Read Up: <code>14 MiB</code>.<br>Write Activated: <code>14 MiB</code>.<br>Total: <code>42 MiB</code>.<br><strong>Logical tensor traffic:</strong> two 14 MiB reads and one 14 MiB write.<br><strong>Fused-kernel traffic:</strong> fusion with stage 13b can avoid the 14 MiB write and following 14 MiB read.<br><strong>Actual HBM traffic:</strong> depends on whether Activated is materialized.<br><br><strong>MAX REALITY</strong><br>The graph expresses SiLU followed by element-wise multiplication. The compiler can fuse these element-wise operations and may feed their result directly into the down projection.</details> |  |  |  |
| 13b Down projection | `Activated [512,14336]`; `Wdown [4096,14336]` | `MlpOut = Activated @ Wdown^T` | `MlpOut [512,4096]` | Project the activated MLP representation back to hidden width 4096. |
|  | <details><summary><strong>-> 13b details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>512*14336*4096 = 30,064,771,072</code>.<br>FLOPs: <code>60,129,542,144 = 60.13 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read Activated: <code>14 MiB</code>.<br>Read Wdown: <code>112 MiB</code>.<br>Write MlpOut: <code>4 MiB</code>.<br>Total: <code>130 MiB</code>.<br><strong>Logical tensor traffic:</strong> 130 MiB standalone.<br><strong>Fused-kernel traffic:</strong> stages 13a+13b require 144 MiB rather than 172 MiB when Activated is not written and reread.<br><strong>Actual HBM traffic:</strong> depends on activation fusion and Wdown encoding.<br><br><strong>MAX REALITY</strong><br>The down projection is the large matmul that returns the MLP width from 14336 to 4096. Its input may be produced directly by a fused SwiGLU operation.</details> |  |  |  |
| 14 Residual add | `H [512,4096]`; `MlpOut [512,4096]` | `Xnext = H + MlpOut` | `Xnext [512,4096]` | Produce the next layer's residual stream. |
|  | <details><summary><strong>-> 14 details</strong></summary><strong>COMPUTE</strong><br>Additions: <code>512*4096 = 2,097,152</code>.<br>FLOPs: <code>2.10 MFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read H: <code>4 MiB</code>.<br>Read MlpOut: <code>4 MiB</code>.<br>Write Xnext: <code>4 MiB</code>.<br>Total: <code>12 MiB</code>.<br><strong>Logical tensor traffic:</strong> two reads and one write.<br><strong>Fused-kernel traffic:</strong> Xnext may be consumed directly by the next RMSNorm.<br><strong>Actual HBM traffic:</strong> lifetime and fusion dependent.<br><br><strong>MAX REALITY</strong><br>The add may be fused with MLP output handling or the next layer's normalization. Xnext remains the residual stream for the following layer.</details> |  |  |  |
| 15 Repeat | `Xnext`; the next layer's weights, norm scales, and KV-cache region | `X_(l+1) = Layer_l(X_l)`, for `l=0..31` | final `X32 [512,4096]` | Run stages 02-14 for all 32 layers. |
|  | <details><summary><strong>-> 15 details</strong></summary><strong>COMPUTE</strong><br>Per layer, dominant work is QKV <code>25.77G</code> + attention <code>4.34G</code> + Wo <code>17.18G</code> + MLP <code>180.44G</code> = about <code>227.7 GFLOPs</code>.<br>All layers: <code>32*227.7G = about 7.29 TFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>BF16 matrices per layer: attention <code>80 MiB</code> + MLP <code>336 MiB</code> = <code>416 MiB</code>.<br>Across layers: <code>32*416 MiB = 13 GiB</code>.<br><strong>Logical tensor traffic:</strong> matrix payload plus all activation/cache traffic.<br><strong>Fused-kernel traffic:</strong> reduces intermediate activation transfers, not the required learned parameters.<br><strong>Actual HBM traffic:</strong> depends on quantization, residency, parallelism, and cache reuse; profile the complete prefill.<br><br><strong>MAX REALITY</strong><br>Each layer uses distinct weights and its own KV-cache region. MAX executes the compiled layer operations with selected matmul, norm, RoPE, cache, and flash-attention kernels.</details> |  |  |  |
| 16 Last-token gather | `X32 [512,4096]`; final row offset/index `511` | `LastH = X32[511:512,:]` | `LastH [1,4096]` | Select prompt position 511. |
|  | <details><summary><strong>-> 16 details</strong></summary><strong>COMPUTE</strong><br>No arithmetic MACs or FLOPs; this is an indexed row copy/view operation.<br><br><strong>LOGICAL MEMORY</strong><br>Read one row: <code>4096*2 = 8 KiB</code>.<br>Write LastH: <code>8 KiB</code>.<br>Total: <code>16 KiB</code>.<br><strong>Logical tensor traffic:</strong> one row read and one row write.<br><strong>Fused-kernel traffic:</strong> a view or direct consumer can avoid the copy.<br><strong>Actual HBM traffic:</strong> depends on gather implementation and tensor residency.<br><br><strong>MAX REALITY</strong><br>MAX derives the final row index from the ragged input-row offsets and gathers only the hidden state needed for next-token logits.</details> |  |  |  |
| 17 Final RMSNorm | `LastH [1,4096]`; `gamma_final [4096]`; `eps` | `FinalH = (LastH / sqrt(mean(LastH^2) + eps)) * gamma_final` | `FinalH [1,4096]` | Normalize the selected hidden state. |
|  | <details><summary><strong>-> 17 details</strong></summary><strong>COMPUTE</strong><br>Elements: <code>4096</code>.<br>Approximate FLOPs: <code>4*4096 to 5*4096 = 16.4 to 20.5 KFLOPs</code>.<br>Also: one reciprocal-square-root.<br><br><strong>LOGICAL MEMORY</strong><br>Read LastH: <code>8 KiB</code>.<br>Read gamma: <code>8 KiB</code>.<br>Write FinalH: <code>8 KiB</code>.<br>Total: <code>24 KiB</code>.<br><strong>Logical tensor traffic:</strong> two reads and one write.<br><strong>Fused-kernel traffic:</strong> FinalH may feed the LM head without a long-lived intermediate.<br><strong>Actual HBM traffic:</strong> negligible relative to the LM-head weight read.<br><br><strong>MAX REALITY</strong><br>This is the same RMSNorm algorithm as earlier stages, but it processes only one selected token row.</details> |  |  |  |
| 18 LM head | `FinalH [1,4096]`; `Wlm [128256,4096]` | `Logits = FinalH @ Wlm^T` | logits `[1,128256]` | Score every vocabulary token. |
|  | <details><summary><strong>-> 18 details</strong></summary><strong>COMPUTE</strong><br>MACs: <code>4096*128256 = 525,336,576</code>.<br>FLOPs: <code>1,050,673,152 = 1.05 GFLOPs</code>.<br><br><strong>LOGICAL MEMORY</strong><br>Read FinalH: <code>8 KiB</code>.<br>Read BF16 Wlm: <code>1002 MiB</code>.<br>Write FP32 logits: <code>128256*4 = 501 KiB</code>.<br>Total: about <code>1002.5 MiB</code>.<br><strong>Logical tensor traffic:</strong> dominated by Wlm.<br><strong>Fused-kernel traffic:</strong> final norm/output handling may fuse, but Wlm still dominates.<br><strong>Actual HBM traffic:</strong> primarily determined by LM-head weight encoding and cache behavior.<br><br><strong>MAX REALITY</strong><br>The LM head computes only the selected final token's vocabulary row. Llama 3.1 uses a separate untied LM-head matrix. Quantized weights substantially reduce its payload.</details> |  |  |  |

The `Input` column includes runtime activations/state and learned parameters.
Projection biases are absent because this Llama configuration uses bias-free
linear layers.

Formula notation:

```text
@     matrix multiplication
*     element-wise multiplication
W^T   transpose of a weight stored as [output_dimension,input_dimension]
```

The three attention projections are independent:

```text
Q = Xnorm @ Wq^T
K = Xnorm @ Wk^T
V = Xnorm @ Wv^T

Wq [4096,4096]
Wk [1024,4096]
Wv [1024,4096]
```

K is not computed from Q, and V is not computed from K.

Sampling the next token is a separate operation and is not part of this
prefill graph.

</details>

<details>
<summary><strong>3. MAX source mapping</strong></summary>

```text
Llama3.__call__(...)
    |
    +-- embed_tokens(tokens)
    |
    +-- Transformer._process_hidden_states(...)
    |     |
    |     +-- call 32 TransformerBlock layers
    |           |
    |           +-- RMSNorm
    |           +-- AttentionWithRope
    |           |     +-- rope_split_store_ragged
    |           |     +-- flash_attention_ragged
    |           +-- MLP
    |           +-- residual additions
    |
    +-- logits_postprocess(...)
          +-- gather last prompt position
          +-- final RMSNorm
          +-- LM head
          +-- cast logits to float32
```

Relevant files:

```text
max/python/max/pipelines/architectures/llama3/llama3.py
max/python/max/pipelines/architectures/llama3/model.py
max/python/max/pipelines/architectures/llama3/model_config.py
max/python/max/nn/transformer/transformer.py
max/python/max/nn/attention/attention_with_rope.py
```

</details>

<details>
<summary><strong>4. Input, structures, logic, and output</strong></summary>

```text
IN
    token IDs [512]
    input_row_offsets [0,512]
    return_n_logits [1]
    request-start KV-cache metadata with current length 0
    trained Llama 3.1 8B weights

STRUCTURES
    token embedding with hidden width 4096
    32 transformer layers
    GQA with 32 query heads and 8 KV heads
    RoPE positional encoding
    SwiGLU MLP with width 14336
    paged KV cache
    vocabulary projection with size 128256

LOGIC
    embed all prompt tokens
    run causal attention and MLP processing through 32 layers
    store every prompt token's K/V values in each layer
    select the final prompt token's hidden state
    project that state to vocabulary logits

OUT
    float32 logits [1,128256]
    512 valid KV-cache positions in each of 32 layers
```

</details>

<details>
<summary><strong>5. GraphPipelineModelWithKVCache.load_model() call tree</strong></summary>

Evidence: `SOURCE`. Native compiler/runtime internals are marked `BOUNDARY`.

`Llama3Model` inherits the concrete startup template from
`GraphPipelineModelWithKVCache`; it does not use the non-KV
`GraphPipelineModel.load_model()` method.

```text
LlamaModelBase.__init__(...)                                      [SOURCE]
  max/python/max/pipelines/architectures/llama3/model.py:135
    |
    +-- IN
    |     pipeline_config, InferenceSession, one or more Device objects,
    |     KVCacheConfig, Weights, MemoryPlan, return-logit policy
    |
    +-- [CALL] PipelineModelWithKVCache.__init__(...)
    |     |
    |     +-- establish pipeline/model/device configuration
    |     +-- construct self.kv_params through Llama3Config.construct_kv_params(...)
    |           |
    |           +-- KVCacheConfig.to_params(...)
    |                 +-- dtype and optional quantization
    |                 +-- 8 KV heads, head dimension 128, 32 layers
    |                 +-- device/data-parallel topology
    |                 +-- paged-cache block and speculative-decoding settings
    |                 +-- OUT: KVCacheParams
    |
    +-- [CALL] self.load_model(session)
          |
          v
GraphPipelineModelWithKVCache.load_model(session)                  [SOURCE]
  max/python/max/pipelines/lib/interfaces/pipeline_model.py:948
    |
    +-- [CALL] self._load_state_dict()
    |     max/python/max/pipelines/lib/interfaces/pipeline_model.py:903
    |     |
    |     +-- [CALL] parse_state_dict_from_weights(...)
    |           max/python/max/pipelines/lib/utils.py:122
    |           |
    |           +-- [BRANCH] adapter is configured
    |           |     |
    |           |     +-- materialize dict(weights.items())
    |           |     +-- select explicit hf_config or pipeline model config
    |           |     +-- [CALL] adapter(...)
    |           |           +-- rename/reshape/repack architecture weights
    |           |           +-- [LEAF] return dict[str, WeightData]
    |           |
    |           +-- [BRANCH] no adapter
    |                 |
    |                 +-- iterate weights.items()
    |                 +-- [CALL] WeightData loader value.data() for each key
    |                 +-- [LEAF] return dict[str, WeightData]
    |     |
    |     +-- OUT: state_dict containing checkpoint weight metadata/loaders
    |
    +-- [CALL] self._create_model_config(state_dict)
    |     max/python/max/pipelines/architectures/llama3/model.py:191
    |     |
    |     +-- [CALL] Llama3Config.initialize(pipeline_config, max_seq_len)
    |     |     max/python/max/pipelines/architectures/llama3/model_config.py:224
    |     |     |
    |     |     +-- require Hugging Face config
    |     |     +-- [CALL] Llama3Config.initialize_from_config(...)
    |     |           max/python/max/pipelines/architectures/llama3/model_config.py:247
    |     |           |
    |     |           +-- _select_quantization_encoding(...)
    |     |           +-- supported_encoding_dtype(...)
    |     |           +-- cache_dtype_for_encoding(...)
    |     |           +-- convert DeviceSpec values to DeviceRef values
    |     |           +-- read embedding/residual multipliers
    |     |           +-- parse flat llama3 or longrope scaling configuration
    |     |           +-- calculate attention scale, normally 1/sqrt(head_dim)
    |     |           +-- gptq_quant_config(...) when applicable
    |     |           +-- Llama3Config.construct_kv_params(...)
    |     |           |     +-- KVCacheConfig.to_params(...)
    |     |           |     +-- [LEAF] KVCacheParams
    |     |           +-- [LEAF] construct Llama3Config
    |     |                 hidden=4096, layers=32, Q heads=32, KV heads=8,
    |     |                 MLP=14336, vocabulary=128256, devices, dtypes,
    |     |                 RoPE, quantization, KV-cache, and parallelism fields
    |     |
    |     +-- [CALL] model_config.finalize(...)
    |           max/python/max/pipelines/architectures/llama3/model_config.py:397
    |           |
    |           +-- inspect language_model./model. prefixes through a normalized view
    |           +-- _parse_quant_config(...) -> parse_quant_config(...)
    |           +-- detect normalization weight dtype
    |           +-- detect tied versus separate embedding/LM-head weights
    |           +-- read RMSNorm epsilon
    |           +-- detect packed gate_up_proj -> stacked_mlp
    |           +-- detect packed qkv_proj -> stacked_qkv
    |           +-- store norm, attention-bias, logits, and hidden-state policies
    |           +-- [LEAF] mutate and finalize the Llama3Config instance
    |     |
    |     +-- OUT: finalized model_config: Llama3Config
    |
    +-- [CALL] self._init_distributed_runtime(session, model_config)
    |     max/python/max/pipelines/lib/interfaces/pipeline_model.py:978
    |     |
    |     +-- Llama3 does not override the base implementation
    |     +-- [LEAF] discard arguments and return None
    |
    +-- [ENTER] CompilationTimer("model")
    |     max/python/max/experimental/nn/_compilation_timer.py:45
    |     |
    |     +-- time.perf_counter()
    |     +-- start 60-second progress-reminder thread
    |     +-- install local CompilationStats in a ContextVar
    |     |
    |     +-- [CALL] self._build_graph_for_compile(...)
    |     |     max/python/max/pipelines/architectures/llama3/model.py:205
    |     |     |
    |     |     +-- [BRANCH] model_config.data_parallel_degree > 1
    |     |     |     +-- create_data_parallel_graph(...)
    |     |     |     +-- [LEAF FOR THIS DOCUMENT] separate data-parallel path
    |     |     |
    |     |     +-- [BRANCH] len(self.devices) > 1
    |     |     |     +-- _build_tensor_parallel_graph_for_compile(...)
    |     |     |     +-- DistributedLlama3(...)
    |     |     |     +-- shard model state and add signal-buffer graph inputs
    |     |     |     +-- [LEAF FOR THIS DOCUMENT] separate multi-device path
    |     |     |
    |     |     +-- [BRANCH] one device, used by the example below
    |     |           |
    |     |           +-- [CALL] _build_single_device_graph_for_compile(...)
    |     |                 max/python/max/pipelines/architectures/llama3/model.py:269
    |     |                 |
    |     |                 +-- [CALL] Llama3(model_config)
    |     |                 |     max/python/max/pipelines/architectures/llama3/llama3.py:76
    |     |                 |     |
    |     |                 |     +-- assert exactly one device
    |     |                 |     |
    |     |                 |     +-- [CALL] create_rope_embedding(...)
    |     |                 |     |     model_config.py:62
    |     |                 |     |     |
    |     |                 |     |     +-- longrope configured
    |     |                 |     |     |     +-- LongRoPERotaryEmbedding(...)
    |     |                 |     |     |
    |     |                 |     |     +-- otherwise
    |     |                 |     |           +-- [LEAF] Llama3RotaryEmbedding(...)
    |     |                 |     |
    |     |                 |     +-- select normalization constructor
    |     |                 |     |     +-- RMSNorm(4096, dtype, eps), normally
    |     |                 |     |     +-- ConstantLayerNorm(...) otherwise
    |     |                 |     |
    |     |                 |     +-- select linear constructor
    |     |                 |     |     +-- GPTQLinear for GPTQ configuration
    |     |                 |     |     +-- Linear with optional QuantConfig otherwise
    |     |                 |     |
    |     |                 |     +-- select attention implementation
    |     |                 |     |     +-- GPTQAttentionWithRope for GPTQ
    |     |                 |     |     +-- GGUFQAttentionWithRope for GGUF quantization
    |     |                 |     |     +-- AttentionWithRope otherwise
    |     |                 |     |
    |     |                 |     +-- [LOOP] construct 32 TransformerBlock objects
    |     |                 |     |     |
    |     |                 |     |     +-- AttentionWithRope(...)
    |     |                 |     |     |     attention_with_rope.py:60
    |     |                 |     |     |     |
    |     |                 |     |     |     +-- calculate Q width 4096
    |     |                 |     |     |     +-- calculate K/V width 1024 each
    |     |                 |     |     |     +-- StackedLinear(...)
    |     |                 |     |     |     |     +-- packed qkv_proj Weight, or
    |     |                 |     |     |     |     +-- q_proj/k_proj/v_proj Linear children
    |     |                 |     |     |     +-- o_proj Linear(4096 -> 4096)
    |     |                 |     |     |     +-- [LEAF] attention Module tree
    |     |                 |     |     |
    |     |                 |     |     +-- MLP(...) or StackedMLP(...)
    |     |                 |     |     |     +-- gate/up: 4096 -> 14336 each,
    |     |                 |     |     |     |   or packed 4096 -> 28672
    |     |                 |     |     |     +-- down: 14336 -> 4096
    |     |                 |     |     |     +-- [LEAF] MLP Module tree
    |     |                 |     |     |
    |     |                 |     |     +-- attention RMSNorm + MLP RMSNorm
    |     |                 |     |     +-- [LEAF] TransformerBlock Module tree
    |     |                 |     |
    |     |                 |     +-- Embedding(128256, 4096)
    |     |                 |     +-- LM-head Linear(4096, 128256)
    |     |                 |     +-- optionally share embedding weight with LM head
    |     |                 |     +-- Transformer.__init__(...)
    |     |                 |           +-- register layers, final norm, embedding,
    |     |                 |               LM head, RoPE, KV params, and output policy
    |     |                 |           +-- [LEAF] single_model: Llama3
    |     |                 |
    |     |                 +-- [CALL] single_model.load_state_dict(...)
    |     |                 |     max/python/max/nn/layer/layer.py:378
    |     |                 |     |
    |     |                 |     +-- recursively _iter_named_weights()
    |     |                 |     +-- [LOOP] each fully qualified Module weight
    |     |                 |     |     +-- look up checkpoint WeightData
    |     |                 |     |     +-- _array_from_weight_loader(...), or
    |     |                 |     |     |   _validate_weight_value(...)
    |     |                 |     |     +-- override quantization metadata when requested
    |     |                 |     |     +-- force alignment=1 and _check_alignment(...)
    |     |                 |     |     +-- register DLPack-compatible value by full name
    |     |                 |     +-- strict=False tolerates the unused rope_freqs key
    |     |                 |     +-- [LEAF] Module weight-value registry populated
    |     |                 |
    |     |                 +-- [CALL] single_model.state_dict()
    |     |                 |     max/python/max/nn/layer/layer.py:480
    |     |                 |     |
    |     |                 |     +-- iterate raw_state_dict() names
    |     |                 |     +-- use loaded DLPack arrays
    |     |                 |     +-- zero-initialize any absent Module weight
    |     |                 |     +-- [LEAF] weights_registry: dict[str, DLPackArray]
    |     |                 |
    |     |                 +-- [CALL] single_model.input_types(self.kv_params)
    |     |                 |     llama3.py:230
    |     |                 |     |
    |     |                 |     +-- tokens: int64 [total_seq_len] on model device
    |     |                 |     +-- input_row_offsets: uint32 [rows+1] on model device
    |     |                 |     +-- return_n_logits: int64 [N] on CPU
    |     |                 |     +-- kv_params.get_symbolic_inputs()
    |     |                 |           cache_params.py:1888
    |     |                 |           |
    |     |                 |           +-- [LOOP] replica and device
    |     |                 |           +-- build KVCacheInputsPerDevice types
    |     |                 |                 +-- paged KV block BufferType
    |     |                 |                 +-- page stride
    |     |                 |                 +-- cache lengths
    |     |                 |                 +-- page lookup table
    |     |                 |                 +-- max prompt/cache lengths
    |     |                 |                 +-- attention dispatch metadata
    |     |                 |                 +-- optional quantization/speculation fields
    |     |                 |                 +-- [LEAF] symbolic KV input tree
    |     |                 |
    |     |                 +-- [ENTER] Graph("llama3", input_types=...)
    |     |                 |     max/python/max/graph/graph.py:682
    |     |                 |     |
    |     |                 |     +-- create top-level mo.graph and entry block
    |     |                 |     +-- create symbolic block arguments from input types
    |     |                 |     +-- initialize device chains and KernelLibrary
    |     |                 |     +-- Graph.__enter__ sets CURRENT_GRAPH ContextVar
    |     |                 |     +-- [LEAF] active MLIR graph insertion context
    |     |                 |
    |     |                 +-- graph.inputs
    |     |                 |     +-- wrap MLIR block arguments as Value objects
    |     |                 |     +-- unpack tokens, offsets, return_n_logits, KV leaves
    |     |                 |
    |     |                 +-- [CALL] self._unflatten_kv_inputs(rest)
    |     |                 |     pipeline_model.py:867
    |     |                 |     |
    |     |                 |     +-- kv_params.unflatten_kv_inputs(iterator)
    |     |                 |     |     +-- tree.unflatten(cached symbolic TreeDef, ...)
    |     |                 |     +-- tree.leaves(..., leaf=KVCacheInputsPerDevice)
    |     |                 |     +-- [LEAF] list[PagedCacheValues]
    |     |                 |
    |     |                 +-- [CALL] single_model(symbolic inputs)
    |     |                       llama3.py -> Transformer.__call__ at transformer.py:570
    |     |                       |
    |     |                       +-- IMPORTANT
    |     |                       |     These are symbolic TensorValue objects.
    |     |                       |     Calls below insert graph operations; they do not
    |     |                       |     execute the 512-token prompt during startup.
    |     |                       |
    |     |                       +-- [CALL] Embedding.__call__(tokens)
    |     |                       |     max/python/max/nn/embedding.py:110
    |     |                       |     |
    |     |                       |     +-- move symbolic indices/weight to device
    |     |                       |     +-- ops.gather(weight, indices, axis=0)
    |     |                       |     +-- optional ops.dequantize(...)
    |     |                       |     +-- [GRAPH-OP LEAF] hidden [total_seq_len,4096]
    |     |                       |
    |     |                       +-- optional embedding-multiplier graph op
    |     |                       |
    |     |                       +-- [CALL] Transformer._process_hidden_states(...)
    |     |                             transformer.py:530
    |     |                             |
    |     |                             +-- read rope.freqs_cis symbolic weight/value
    |     |                             |
    |     |                             +-- [LOOP] layer idx = 0..31
    |     |                             |     |
    |     |                             |     +-- ops.constant(idx, uint32, CPU)
    |     |                             |     |
    |     |                             |     +-- [CALL] TransformerBlock.__call__(...)
    |     |                             |           transformer.py:271
    |     |                             |           |
    |     |                             |           +-- input_layernorm(x)
    |     |                             |           |     RMSNorm.__call__
    |     |                             |           |     +-- validate final dimension 4096
    |     |                             |           |     +-- cast/move gamma
    |     |                             |           |     +-- [GRAPH-OP LEAF] ops.rms_norm(...)
    |     |                             |           |
    |     |                             |           +-- self_attn(...)
    |     |                             |           |     AttentionWithRope.__call__
    |     |                             |           |     attention_with_rope.py:498
    |     |                             |           |     |
    |     |                             |           |     +-- qkv_proj(x_norm)
    |     |                             |           |     |     StackedLinear.__call__
    |     |                             |           |     |     stacked_linear.py:403
    |     |                             |           |     |     |
    |     |                             |           |     |     +-- packed weight, or concatenate
    |     |                             |           |     |     |   q_proj/k_proj/v_proj weights
    |     |                             |           |     |     +-- linear(...)
    |     |                             |           |     |           +-- ops.qmatmul for encoded weights
    |     |                             |           |     |           +-- quantized_matmul for QuantConfig
    |     |                             |           |     |           +-- x @ weight.T otherwise
    |     |                             |           |     |           +-- [GRAPH-OP LEAF] flat QKV
    |     |                             |           |     |               [tokens,4096+1024+1024]
    |     |                             |           |     |
    |     |                             |           |     +-- optional Q/K per-head RMSNorm branch
    |     |                             |           |     |     +-- ops.split / reshape / RMSNorm / concat
    |     |                             |           |     |
    |     |                             |           |     +-- cast/move RoPE frequencies
    |     |                             |           |     |
    |     |                             |           |     +-- rope_split_store_ragged(...)
    |     |                             |           |     |     max/python/max/nn/kernels.py:286
    |     |                             |           |     |     |
    |     |                             |           |     |     +-- validate ranks and dtypes
    |     |                             |           |     |     +-- choose fused/unfused or QK-norm form
    |     |                             |           |     |     +-- flatten paged-KV operands
    |     |                             |           |     |     +-- [GRAPH-OP LEAF] ops.inplace_custom(
    |     |                             |           |     |           "mo.rope_split_store.ragged.paged", ...)
    |     |                             |           |     |           OUT: rotated Q; side effect: K/V cache write
    |     |                             |           |     |
    |     |                             |           |     +-- reshape Q to [tokens,32,128]
    |     |                             |           |     |
    |     |                             |           |     +-- flash_attention_ragged(...)
    |     |                             |           |     |     kernels.py:4200
    |     |                             |           |     |     |
    |     |                             |           |     |     +-- validate Q/KV/offset metadata
    |     |                             |           |     |     +-- select causal-mask op parameters
    |     |                             |           |     |     +-- flatten KV and dispatch metadata
    |     |                             |           |     |     +-- [GRAPH-OP LEAF] ops.inplace_custom(
    |     |                             |           |     |           "mo.mha.ragged.paged", ...)
    |     |                             |           |     |           OUT: attention [tokens,32,128]
    |     |                             |           |     |
    |     |                             |           |     +-- ops.reshape -> [tokens,4096]
    |     |                             |           |     +-- o_proj Linear.__call__
    |     |                             |           |           +-- linear(...)
    |     |                             |           |           +-- [GRAPH-OP LEAF] attention output
    |     |                             |           |
    |     |                             |           +-- optional residual multiplier
    |     |                             |           +-- [GRAPH-OP LEAF] h = x + attn_out
    |     |                             |           |
    |     |                             |           +-- post_attention_layernorm(h)
    |     |                             |           |     +-- [GRAPH-OP LEAF] ops.rms_norm(...)
    |     |                             |           |
    |     |                             |           +-- mlp(h_norm)
    |     |                             |           |     |
    |     |                             |           |     +-- standard MLP.__call__ branch
    |     |                             |           |     |     +-- fused gate/up linear, or two Linear calls
    |     |                             |           |     |     +-- ops.split(...)
    |     |                             |           |     |     +-- activation_function(gate), normally SiLU
    |     |                             |           |     |     +-- gate * up
    |     |                             |           |     |     +-- down_proj Linear
    |     |                             |           |     |     +-- [GRAPH-OP LEAF] MLP output [tokens,4096]
    |     |                             |           |     |
    |     |                             |           |     +-- packed StackedMLP.__call__ branch
    |     |                             |           |           +-- gate_up_proj(x)
    |     |                             |           |           +-- slice gate and up halves
    |     |                             |           |           +-- ops.silu(gate) * up
    |     |                             |           |           +-- down_proj(...)
    |     |                             |           |           +-- [GRAPH-OP LEAF] MLP output
    |     |                             |           |
    |     |                             |           +-- optional residual multiplier
    |     |                             |           +-- [GRAPH-OP LEAF] return h + mlp
    |     |                             |
    |     |                             +-- [CALL] _postprocess_logits(...)
    |     |                                   transformer.py:437
    |     |                                   |
    |     |                                   +-- logits_postprocess(...)
    |     |                                         transformer.py:320
    |     |                                         |
    |     |                                         +-- input_row_offsets[1:] - 1
    |     |                                         +-- ops.gather final token rows
    |     |                                         +-- final RMSNorm
    |     |                                         +-- LM-head Linear
    |     |                                         +-- ops.cast(..., float32)
    |     |                                         +-- optional VARIABLE/ALL-logits branches
    |     |                                         +-- optional logits scaling
    |     |                                         +-- extract_hs(...) policy
    |     |                                         +-- [GRAPH-OP LEAF] output tuple
    |     |                                             normally logits [batch,128256]
    |     |
    |     +-- [CALL] graph.output(*outputs)
    |     |     max/python/max/graph/graph.py:1372
    |     |     |
    |     |     +-- normalize outputs to Value objects
    |     |     +-- pack device chains when present
    |     |     +-- convert result types to MLIR types
    |     |     +-- terminate/finalize the mo.graph body
    |     |     +-- [LEAF] graph: Graph plus weights_registry
    |     |
    |     +-- [CALL] timer.mark_build_complete()
    |     |     +-- time.perf_counter()
    |     |     +-- log Python/symbolic graph-build duration
    |     |
    |     +-- self.state_dict = weights_registry
    |     |
    |     +-- [CALL] session.load(graph, weights_registry=...)
    |           max/python/max/engine/api.py:768
    |           |
    |           +-- [CALL] session.load_all(...)
    |           |     max/python/max/engine/api.py:824
    |           |     |
    |           |     +-- [CALL] compile_reusing_mefs(graph, ...)
    |           |     |     max/python/max/engine/api.py:883
    |           |     |     |
    |           |     |     +-- no MEF store -> compile(graph)
    |           |     |     +-- import mode -> claim matching precompiled MEF,
    |           |     |     |   otherwise compile graph
    |           |     |     +-- export mode -> compile, export MEF, write manifest
    |           |     |     |
    |           |     |     +-- [CALL] compile(graph)
    |           |     |           max/python/max/engine/api.py:1048
    |           |     |           |
    |           |     |           +-- record compile_seconds
    |           |     |           +-- [CALL] compile_async(graph)
    |           |     |           |     max/python/max/engine/api.py:955
    |           |     |           |     |
    |           |     |           |     +-- process custom extensions
    |           |     |           |     +-- graph.module -> MLIR Module wrapper
    |           |     |           |     +-- capture expected weight/device mapping
    |           |     |           |     +-- seed kernel declarations from KernelLibrary
    |           |     |           |     +-- [CALL] _compile_module(...)
    |           |     |           |           max/python/max/engine/api.py:1262
    |           |     |           |           |
    |           |     |           |           +-- acquire process-global compile lock
    |           |     |           |           +-- [CALL] self._impl.compile(
    |           |     |           |           |     module.mlir_module._CAPIPtr,
    |           |     |           |           |     custom_extensions,
    |           |     |           |           |     pipeline_name,
    |           |     |           |           |     tile_based_fusion,
    |           |     |           |           |   )
    |           |     |           |           |     |
    |           |     |           |           |     +-- [BOUNDARY] native MAX graph compiler
    |           |     |           |           |         lowers/optimizes the MLIR module,
    |           |     |           |           |         resolves kernels, and produces an
    |           |     |           |           |         asynchronous compiled artifact
    |           |     |           |           +-- callback releases compile lock
    |           |     |           |           +-- [LEAF] _AsyncValue[_CompiledModels]
    |           |     |           |     +-- wrap handle in CompiledModel
    |           |     |           |
    |           |     |           +-- handle.wait()
    |           |     |           +-- raise synchronous/asynchronous compile failures
    |           |     |           +-- [LEAF] CompiledModel, not executable yet
    |           |     |
    |           |     +-- [CALL] init_all(compiled, weights_registry)
    |           |           max/python/max/engine/api.py:1146
    |           |           |
    |           |           +-- [BRANCH] virtual-device mode
    |           |           |     +-- return CompileOnlyModel stand-ins
    |           |           |
    |           |           +-- [BRANCH] real device
    |           |                 +-- normalize missing registry to {}
    |           |                 +-- validate every expected weight is present
    |           |                 +-- validate expected CPU/device placement via DLPack
    |           |                 +-- _raise_if_not_contiguous(...) for each weight
    |           |                 +-- optionally wrap a resolved compiled artifact
    |           |                 +-- record init_seconds
    |           |                 +-- [CALL] self._impl._load_all(
    |           |                 |     compiled_handle, weights_registry)
    |           |                 |     |
    |           |                 |     +-- [BOUNDARY] native MAX runtime/device init
    |           |                 |         associates compiled code with weights,
    |           |                 |         prepares runtime/device resources, and
    |           |                 |         returns executable Model handles
    |           |                 +-- key Model objects by graph name
    |           |                 +-- reject duplicate graph names
    |           |                 +-- [LEAF] dict[str, Model]
    |           |
    |           +-- require exactly one model
    |           +-- [LEAF] return executable Model for graph "llama3"
    |
    +-- [EXIT] CompilationTimer
    |     +-- collect compile/init instrumentation
    |     +-- stop reminder thread
    |     +-- log build, compile, init, and total durations
    |
    +-- [CALL] self._maybe_release_host_weights(model)
    |     pipeline_model.py:490
    |     |
    |     +-- MODULAR_MAX_RELEASE_HOST_WEIGHTS != "1" -> [LEAF] no-op
    |     +-- otherwise, GPU-only release path
    |           +-- model.release_weights() when supported
    |           +-- clear retained state dictionaries
    |           +-- weights.close() when supported
    |           +-- [LEAF] host weight references/file mappings released
    |
    +-- [CALL] self._wire_batch_processor(model, model_config)
    |     pipeline_model.py:921
    |     |
    |     +-- no batch processor -> [LEAF] return
    |     +-- no bind_ep_comm_initializer method -> [LEAF] return
    |     +-- otherwise bind the optional EP communication initializer
    |
    +-- OUT: executable max.engine.Model assigned to self.model
          |
          +-- later runtime path: self.model.execute(*model_inputs.buffers)
          +-- startup graph construction and compilation are now complete
```

The practical boundary is:

```text
Python source visible in this repository
    Llama modules -> symbolic TensorValue ops -> MLIR Graph
        |
        v
self._impl.compile(...)                                      [BOUNDARY]
    native graph lowering, optimization, kernel selection, code generation
        |
        v
self._impl._load_all(...)                                    [BOUNDARY]
    native weight/runtime/device initialization
        |
        v
executable max.engine.Model
```

</details>
