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

<details>
<summary><strong>checkpoint: official Hugging Face safetensors - what does this format contain?</strong></summary>

This means the model weights come directly from Meta's Hugging Face repository
in the `safetensors` format. They are the original BF16 checkpoint tensors, not
a converted GGUF checkpoint or GPTQ/other quantized representation.

A `.safetensors` file is a binary container for named, dense tensors. It stores
model parameter data, not executable model code or the model computation graph.

```text
model-00001-of-00004.safetensors
    |
    +-- 8-byte little-endian header length
    |
    +-- UTF-8 JSON header
    |     +-- tensor name
    |     +-- dtype
    |     +-- shape
    |     +-- byte offsets into the data region
    |
    +-- contiguous raw tensor bytes
```

The JSON header conceptually contains entries such as:

```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [0, 33554432]
  },
  "model.layers.0.self_attn.k_proj.weight": {
    "dtype": "BF16",
    "shape": [1024, 4096],
    "data_offsets": [33554432, 41943040]
  }
}
```

The data offsets select regions in the raw data area:

```text
raw tensor data
    |
    +-- q_proj.weight
    |     4096 * 4096 * 2 bytes = 32 MiB
    |
    +-- k_proj.weight
    |     1024 * 4096 * 2 bytes = 8 MiB
    |
    +-- subsequent tensor bytes
```

`BF16` occupies two bytes per element. Safetensors stores the tensor shape and
dtype, but does not describe how the tensor participates in Llama inference;
MAX obtains that architecture information from `config.json` and its Llama 3
implementation.

The 8B checkpoint is divided into multiple files:

```text
meta-llama/Llama-3.1-8B-Instruct
    |
    +-- config.json
    +-- tokenizer.json and tokenizer configuration
    +-- model.safetensors.index.json
    +-- model-00001-of-00004.safetensors
    +-- model-00002-of-00004.safetensors
    +-- model-00003-of-00004.safetensors
    +-- model-00004-of-00004.safetensors
```

`model.safetensors.index.json` contains a `weight_map` that identifies the
shard containing each named tensor:

```text
model.layers.0.self_attn.q_proj.weight
    -> model-00001-of-00004.safetensors

model.layers.20.mlp.down_proj.weight
    -> model-00003-of-00004.safetensors
```

The companion `config.json` supplies architectural values such as hidden size,
layer count, Q/KV head counts, intermediate size, vocabulary size, RMSNorm
epsilon, and RoPE settings. Tokenizer files define how input text becomes token
IDs. The safetensors shards contain only named parameter arrays.

Safetensors is called "safe" because it cannot embed executable Python objects,
unlike pickle-based checkpoint formats. A loader parses the bounded JSON header
and exposes tensor byte regions. Files can be memory-mapped, allowing a loader
to locate tensors without deserializing the complete checkpoint, although a
later device transfer or dtype conversion can still copy the data.

For this MAX path:

```text
safetensors shards
    |
    +-- Weights object reads names and tensor regions
    +-- convert_safetensor_state_dict strips the "model." prefix
    +-- value.data() creates WeightData metadata and array access
    +-- load_state_dict validates names, shapes, dtypes, and alignment
    +-- underlying DLPack-compatible arrays enter weights_registry
    +-- session.load(...) binds them to the compiled Llama graph
```

The official checkpoint provides separate BF16 tensors such as `q_proj`,
`k_proj`, `v_proj`, `gate_proj`, `up_proj`, and `down_proj`; it does not provide
the packed tensors and quantization scales used by GPTQ or GGUF variants.

</details>

<details>
<summary><strong>Q: What does "selected Linear implementation: Linear, not GPTQLinear" mean?</strong></summary>

The official checkpoint contains ordinary BF16 weight matrices, so MAX uses
the standard `Linear` layer rather than the GPTQ-specific layer.

```text
official BF16 safetensors
    |
    +-- GPTQ quantization_config=None
    +-- scaled quant_config=None
    |
    v
Linear
```

For a standard projection, `Linear` represents the mathematical operation:

```text
Y = X @ W.T
```

For example, the Llama Q projection is:

```text
X  [S,4096] BF16
Wq [4096,4096] BF16
         |
         +-- X @ Wq.T
         v
Q  [S,4096] BF16
```

`Linear` registers the complete BF16 `Weight`. During graph construction MAX
inserts the matrix-multiplication operation; during compilation it can still
select optimized GPU tensor-core kernels and fuse surrounding operations.
Using `Linear` therefore does not mean that the execution is unoptimized.

`GPTQLinear` is selected only when the checkpoint and configuration describe
GPTQ-quantized weights:

```text
GPTQ checkpoint
    |
    +-- packed low-bit qweight
    +-- quantization scales
    +-- optional perm_idx for activation ordering
    |
    v
GPTQLinear
    |
    +-- rebuild the packed operand representation
    +-- optionally permute the input channels
    +-- ops.qmatmul(...)
```

GPTQ avoids storing the complete BF16 matrix directly. The official Llama 3.1
BF16 checkpoint instead provides normal matrices for Q/K/V/O, MLP, and LM-head
projections, so the selected implementation is `Linear`.

</details>

<details>
<summary><strong>Q: What does "selected attention: AttentionWithRope" mean? Is there attention without RoPE?</strong></summary>

`AttentionWithRope` is the MAX attention module selected for the normal BF16
Llama 3.1 path. It combines Q/K/V projection, Llama positional rotation, paged
KV-cache writes, causal attention, and the output projection.

```text
Input X
    |
    +-- Q = X @ Wq.T
    +-- K = X @ Wk.T
    +-- V = X @ Wv.T
    |
    +-- Qrot = RoPE(Q, token_position)
    +-- Krot = RoPE(K, token_position)
    |
    +-- store Krot and V in the paged KV cache
    +-- Attention(Qrot, Kcache, Vcache)
    +-- output projection
```

RoPE is not the attention calculation itself. It rotates pairs of Q and K
features according to token position before the dot products. This gives the
attention scores relative-position information. V is not rotated.

The Llama constructor selects among RoPE-capable implementations according to
the checkpoint encoding:

```text
BF16 weights           -> AttentionWithRope
GPTQ weights           -> GPTQAttentionWithRope
GGUF quantized weights -> GGUFQAttentionWithRope
```

There is no class literally named `AttentionWithoutRope` in this source tree.
There are attention families that do not apply RoPE internally:

- `RaggedAttention` projects Q/K/V, stores unrotated K/V in the paged cache,
  and invokes ragged flash attention.
- `MultiheadAttention` provides more general multi-head attention without RoPE
  built into that module.

Conceptually, the non-RoPE ragged path is:

```text
Q, K, V = project(X)
store unrotated K and V
Attention(Q, Kcache, Vcache)
```

That path cannot simply replace `AttentionWithRope` for Llama 3.1. The
checkpoint was trained with Llama 3 RoPE; removing it changes the positional
relationships used by every attention layer and produces incorrect output.

</details>

<details>
<summary><strong>Q: What do "stacked_qkv=False" and "stacked_mlp=False" mean?</strong></summary>

These flags describe how projection weights are stored in the checkpoint.
They do not, by themselves, determine how many GPU kernels MAX executes.

With `stacked_qkv=False`, the checkpoint stores three separate tensors:

```text
q_proj.weight [4096,4096]
k_proj.weight [1024,4096]
v_proj.weight [1024,4096]
```

With `stacked_qkv=True`, the checkpoint would already contain one packed
projection tensor:

```text
qkv_proj.weight [6144,4096]
                  |
                  +-- 4096 Q rows
                  +-- 1024 K rows
                  +-- 1024 V rows
```

Although the official checkpoint has `stacked_qkv=False`, MAX's
`StackedLinear` can concatenate the three symbolic weights and construct one
QKV matrix multiplication:

```text
Wqkv = concat(Wq, Wk, Wv)
QKV  = X @ Wqkv.T
```

With `stacked_mlp=False`, the checkpoint similarly stores separate MLP
projections:

```text
gate_proj.weight [14336,4096]
up_proj.weight   [14336,4096]
down_proj.weight [4096,14336]
```

Their logical computation is:

```text
Gate      = X @ Wgate.T
Up        = X @ Wup.T
Activated = SiLU(Gate) * Up
Output    = Activated @ Wdown.T
```

With `stacked_mlp=True`, the checkpoint would contain a pre-packed gate/up
weight:

```text
gate_up_proj.weight [28672,4096]
                      |
                      +-- 14336 gate rows
                      +-- 14336 up rows
```

For the unquantized BF16 path, the normal `MLP` can still concatenate the
separate gate and up weights while constructing the graph:

```text
Wgate_up = concat(Wgate, Wup)
GateUp   = X @ Wgate_up.T
Gate, Up = split(GateUp)
```

The distinction is therefore:

```text
stacked=False -> separate tensors in the checkpoint
stacked=True  -> a packed tensor already exists in the checkpoint

runtime/compiler fusion -> a separate optimization decision
```

</details>

```text
CONCRETE PATH USED BY THIS TREE
    |
    +-- model: meta-llama/Llama-3.1-8B-Instruct
    +-- checkpoint: official Hugging Face safetensors
    +-- target: one GPU
    +-- request: 512-token prefill
    +-- no explicit --quantization-encoding override
    +-- no explicit KV-cache dtype override
    |
    +-- resolved model encoding: bfloat16
    +-- model/weight compute dtype: BF16
    +-- KV-cache dtype: BF16
    +-- model QuantizationEncoding: None
    +-- GPTQ QuantizationConfig: None
    +-- scaled QuantConfig: None
    +-- selected Linear implementation: Linear, not GPTQLinear
    +-- selected attention: AttentionWithRope
    +-- selected normalization: RMSNorm
    +-- selected RoPE: Llama3RotaryEmbedding
    +-- official separate weights: stacked_qkv=False, stacked_mlp=False
    +-- tie_word_embeddings=False
    |
    +-- NOTE: q4_k is Llama3Config.DEFAULT_ENCODING, but it is only the
        fallback when checkpoint/config inspection cannot infer an encoding.
        The official BF16 safetensors resolve to bfloat16 before that fallback.

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
    |           +-- [CALL] Llama3Config.get_head_dim(hf_config)
    |           |     +-- hidden_size / num_attention_heads
    |           |     +-- 4096 / 32 = 128 elements per Q/K/V head
    |           |
    |           +-- [CALL] cache_dtype_for_encoding("bfloat16", None)
    |           |     +-- no explicit kv_cache_format override
    |           |     +-- encoding-to-cache map selects DType.bfloat16
    |           |     +-- [LEAF] each cached scalar occupies 2 bytes
    |           |
    |           +-- [CALL] KVCacheConfig.to_params(...)
    |                 max/python/max/pipelines/kv_cache/config.py:317
    |                 |
    |                 +-- IN
    |                 |     dtype=BF16, n_kv_heads=8, head_dim=128,
    |                 |     num_layers=32, devices=[gpu:0], DP degree=1,
    |                 |     speculative_method=None, num_draft_tokens=0
    |                 |
    |                 +-- resolve page_size=128 tokens by default
```

<details>
<summary><strong>|                 +-- resolve page_size=128 tokens by default   => can you explain why page_size is  128 tokens? tokens means (embedded tokens which is of size hidden_dimension=4096 - correct?, dtype?)</strong></summary>

`page_size=128 tokens` means one KV-cache page holds 128 sequence positions. It does not store 128 embedding vectors of size 4096.

For each input token:

```text
token ID: scalar int64
    |
    +-- embedding lookup
    |
    v
hidden state: [4096] BF16
    |
    +-- K projection -> [8,128] BF16
    +-- V projection -> [8,128] BF16
```

The KV cache stores only K and V, not the original `[4096]` embedding or Q.

For one token in one layer:

```text
K: 8 heads * 128 elements * 2 bytes = 2 KiB
V: 8 heads * 128 elements * 2 bytes = 2 KiB

K + V = 4 KiB per layer
```

Across 32 layers:

```text
4 KiB * 32 = 128 KiB per token
```

Therefore, one 128-token page is:

```text
128 tokens * 128 KiB/token = 16 MiB
```

Its physical tensor shape is:

```text
[2, 32, 128, 8, 128]
 |   |    |   |   |
 |   |    |   |   +-- head dimension
 |   |    |   +------ KV heads
 |   |    +---------- token slots per page
 |   +--------------- transformer layers
 +------------------- K and V
```

The dtype is BF16 for this path, so each element occupies two bytes.

`128` is a MAX paged-KV-cache implementation default, not a Llama architectural property. The current implementation requires the page size to be at least 128 and a multiple of 128. It provides a tradeoff between:

- Fewer page-table entries and more contiguous kernel work.
- Allocation granularity and unused space in partially filled pages.
- Efficient paged-attention kernel dispatch.

For a 512-token prompt:

```text
512 / 128 = 4 KV-cache pages
```

A page table maps logical token positions to those physical cache pages:

```text
positions 0-127   -> page 0
positions 128-255 -> page 1
positions 256-383 -> page 2
positions 384-511 -> page 3
```

So “token” here means a cached sequence position whose projected K/V values are stored across all layers.

</details>

```text
    |                 +-- slots_per_page=None -> 128 physical token slots
    |                 +-- tensor_parallel_degree = devices / DP = 1 / 1 = 1
    |                 +-- n_kv_heads_per_device = 8 / TP = 8
    |                 +-- MHA kv_dim=2: index 0 stores K; index 1 stores V
    |                 +-- no KV-cache quantization scale buffer
    |                 +-- no speculative draft-cache fields
    |                 |
    |                 +-- [LEAF] MHAKVCacheParams.shape_per_block
    |                 |     [kv_dim, layers, slots, KV heads/device, head_dim]
    |                 |     = [2, 32, 128, 8, 128]
    |                 |
    |                 +-- [LEAF] raw value bytes per 128-token page
    |                 |     2 * 32 * 128 * 8 * 128 * 2 bytes
    |                 |     = 16 MiB across the one-device TP group
    |                 |
    |                 +-- [LEAF] raw K/V bytes per cached token
    |                 |     2 * 32 * 8 * 128 * 2 bytes = 128 KiB
    |                 |
    |                 +-- [LEAF] raw 512-token prefill K/V payload
    |                       512 * 128 KiB = 64 MiB
    |                       excluding page tables, allocator padding, reserved
    |                       free pages, dispatch metadata, and prefix-cache data
    |
    |           +-- OUT: self.kv_params: MHAKVCacheParams
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
    |           +-- [UPSTREAM ADAPTER SELECTION BEFORE MODEL CONSTRUCTION]
    |           |     +-- model_config.resolved_weight_paths()
    |           |     +-- resolve/download official *.safetensors shards
    |           |     +-- weights_format(paths) -> WeightsFormat.safetensors
    |           |     +-- architecture registry supplies
    |           |         convert_safetensor_state_dict as self.adapter
    |           |
    |           +-- [BRANCH] adapter is configured: ACTIVE HERE
    |           |     |
    |           |     +-- materialize dict(weights.items())
    |           |     +-- select explicit hf_config or pipeline model config
    |           |     +-- [CALL] convert_safetensor_state_dict(...)
    |           |           weight_adapters.py:115
    |           |           |
    |           |           +-- _select_quantization_encoding(...)
    |           |           |     +-- infer bfloat16 from official weights/config
    |           |           |     +-- do not use fallback q4_k
    |           |           |
    |           |           +-- _select_dtype_cast(...)
    |           |           |     +-- native BF16 -> (None, None)
    |           |           |     +-- no load-time dtype conversion
    |           |           |
    |           |           +-- [LOOP] each safetensor weight
    |           |           |     +-- strip leading "model."
    |           |           |     |     model.layers.0.self_attn.q_proj.weight
    |           |           |     |       -> layers.0.self_attn.q_proj.weight
    |           |           |     +-- rename GPTQ g_idx -> perm_idx if present
    |           |           |     +-- value.data() -> WeightData
    |           |           |
    |           |           +-- [INACTIVE GPTQ ADAPTER WORK]
    |           |           |     +-- argsort permutation indices
    |           |           |     +-- cast selected FP16 tensors to BF16
    |           |           |     +-- remove unused bias/qzeros/perm_idx keys
    |           |           |
    |           |           +-- [LEAF] dict[str, WeightData]
    |           |                 each value carries data, name, dtype, shape,
    |           |                 and optional quantization metadata
    |           |
    |           +-- [BRANCH] no adapter: INACTIVE FOR SAFETENSORS
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
    |     |           |     +-- OUT: SupportedEncoding "bfloat16"
    |     |           +-- supported_encoding_dtype("bfloat16")
    |     |           |     +-- OUT: config.dtype = DType.bfloat16
    |     |           +-- supported_encoding_quantization("bfloat16")
    |     |           |     +-- OUT: model_quantization_encoding = None
    |     |           +-- cache_dtype_for_encoding("bfloat16", None)
    |     |           |     +-- OUT: KV dtype = DType.bfloat16
    |     |           +-- convert DeviceSpec values to DeviceRef values
```

<details>
<summary><strong>|     |           +-- convert DeviceSpec values to DeviceRef values  => ??</strong></summary>

`DeviceSpec` and `DeviceRef` represent the same device at different stages.

```text
DeviceSpec -> configuration-time device selection
DeviceRef  -> symbolic graph-time device placement
Device     -> actual runtime driver device
```

### DeviceSpec

`DeviceSpec` is a frozen Python dataclass:

```python
@dataclass(frozen=True)
class DeviceSpec:
    id: int
    device_type: Literal["cpu", "gpu"]
```

It records what the user or configuration requested:

```text
--devices gpu:0
    |
    v
DeviceSpec(id=0, device_type="gpu")
```

It is used for configuration validation and device discovery. It does not represent a tensor placement inside a MAX graph.

### DeviceRef

`DeviceRef` is the symbolic device representation used by `max.graph`:

```python
class DeviceRef:
    device_type: DeviceKind
    id: int
```

The Llama configuration performs:

```python
device_refs = [
    DeviceRef(spec.device_type, spec.id)
    for spec in device_specs
]
```

For example:

```text
DeviceSpec(id=0, device_type="gpu")
    |
    v
DeviceRef(DeviceKind.GPU, id=0)
    |
    v
"gpu:0"
```

`DeviceRef` is attached to symbolic graph objects:

```python
TensorType(
    DType.bfloat16,
    shape=["total_seq_len", 4096],
    device=DeviceRef.GPU(0),
)

Weight(
    "weight",
    DType.bfloat16,
    [4096, 4096],
    device=DeviceRef.GPU(0),
)
```

During graph lowering, it becomes an MLIR device attribute:

```text
DeviceRef.GPU(0)
    |
    +-- DeviceRef.to_mlir()
    v
MLIR device attribute representing gpu:0
```

### Actual Device

A third type, `Device`, represents the real runtime driver object:

```text
DeviceSpec("gpu", 0)
    |
    +-- load_device(...)
    v
Accelerator(0): Device
```

This actual `Device` is used by the inference session and buffers:

```text
Device
    +-- allocate memory
    +-- create device queues
    +-- copy buffers
    +-- synchronize work
    +-- execute compiled operations
```

The complete flow is:

```text
CLI/configuration
--devices gpu:0
    |
    v
DeviceSpec("gpu", 0)
    |  configuration and validation
    |
    +-----------------------------+
    |                             |
    v                             v
DeviceRef("gpu", 0)          Accelerator(0)
symbolic graph placement     actual driver device
    |                             |
    +-------------+---------------+
                  |
                  v
        InferenceSession compilation
        and runtime execution
```

Converting `DeviceSpec` to `DeviceRef` does not load the GPU, allocate memory, or move tensors. It only tells the graph compiler that particular tensors and operations are intended for `gpu:0`.

</details>

```text
    |     |           +-- read embedding/residual multipliers
    |     |           +-- parse flat llama3 RoPE configuration
    |     |           |     +-- theta=500000
    |     |           |     +-- factor=8, low_freq_factor=1,
    |     |           |     |   high_freq_factor=4
    |     |           |     +-- original_max_position=8192
    |     |           |     +-- configured maximum positions=131072 unless
    |     |           |         runtime max_seq_len is explicitly constrained
    |     |           +-- safetensors -> interleaved_rope_weights=False
    |     |           +-- attention scale = 1/sqrt(head_dim)
    |     |           |     = 1/sqrt(128)
    |     |           +-- gptq_quant_config("bfloat16", ...)
    |     |           |     +-- [LEAF] None
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
    |           |     +-- no scaled FP8/FP4 metadata -> quant_config=None
    |           +-- detect normalization weight dtype -> BF16 checkpoint value
    |           +-- tie_word_embeddings=False and lm_head.weight exists
    |           |     +-- keep embedding and LM-head weights separate
    |           +-- read rms_norm_eps=1e-5
    |           +-- no layers.0.mlp.gate_up_proj.weight
    |           |     +-- stacked_mlp=False; use gate/up/down checkpoint tensors
    |           +-- no layers.0.self_attn.qkv_proj.weight
    |           |     +-- stacked_qkv=False; use q_proj/k_proj/v_proj tensors
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
    |     |     |     +-- [INACTIVE FOR THE ONE-GPU EXAMPLE]
    |     |     |     +-- _build_tensor_parallel_graph_for_compile(...)
    |     |     |     |     |
    |     |     |     |     +-- [CALL] DistributedLlama3(model_config)
    |     |     |     |     |     distributed_llama.py:39
    |     |     |     |     |     |
    |     |     |     |     |     +-- reject GPTQ and packed stacked-MLP TP
    |     |     |     |     |     +-- Q/K/V projection outputs split by heads
    |     |     |     |     |     +-- o_proj input channels split; partial
    |     |     |     |     |     |   attention outputs combined by allreduce
    |     |     |     |     |     +-- gate/up output channels split
```

<details>
<summary><strong>Tensor-parallel attention and MLP split details</strong></summary>

The lines describe tensor parallelism: one transformer layer is divided across multiple GPUs. Consider Llama 3.1 8B running on two GPUs.

## 1. Initial Hidden State

At the beginning of the attention block, both GPUs have the same hidden state:

```text
GPU 0: X [S,4096]
GPU 1: X [S,4096]
```

`S` is the number of tokens currently processed.

```text
Replicated X [S,4096]
        |
        +-----------------------------+-----------------------------+
        | GPU 0                       | GPU 1
        | X [S,4096]                  | X [S,4096]
        +-----------------------------+-----------------------------+
```

RMSNorm is also replicated, so each GPU independently computes the same:

```text
Xnorm = RMSNorm(X) -> [S,4096]
```

## 2. Q/K/V Projection Outputs Split By Heads

Llama 3.1 8B has:

```text
32 Q heads
8 K heads
8 V heads
head_dim = 128
```

The full projection weights are:

```text
Wq [4096,4096]
Wk [1024,4096]
Wv [1024,4096]
```

With two GPUs, MAX splits the projection output rows by heads:

```text
Full Wq [4096,4096]
    +-- GPU 0: Wq0 [2048,4096] -> 16 Q heads
    +-- GPU 1: Wq1 [2048,4096] -> 16 Q heads

Full Wk [1024,4096]
    +-- GPU 0: Wk0 [512,4096] -> 4 K heads
    +-- GPU 1: Wk1 [512,4096] -> 4 K heads

Full Wv [1024,4096]
    +-- GPU 0: Wv0 [512,4096] -> 4 V heads
    +-- GPU 1: Wv1 [512,4096] -> 4 V heads
```

Each GPU performs its local projections:

```text
GPU 0:
    Q0 = Xnorm @ Wq0.T -> [S,2048] -> [S,16,128]
    K0 = Xnorm @ Wk0.T -> [S, 512] -> [S, 4,128]
    V0 = Xnorm @ Wv0.T -> [S, 512] -> [S, 4,128]

GPU 1:
    Q1 = Xnorm @ Wq1.T -> [S,2048] -> [S,16,128]
    K1 = Xnorm @ Wk1.T -> [S, 512] -> [S, 4,128]
    V1 = Xnorm @ Wv1.T -> [S, 512] -> [S, 4,128]
```

```text
Replicated Xnorm [S,4096]
        |
        +-----------------------------+-----------------------------+
        | GPU 0                       | GPU 1
        | Q heads 0-15                | Q heads 16-31
        | K heads 0-3                 | K heads 4-7
        | V heads 0-3                 | V heads 4-7
        |                             |
        | Q0 [S,16,128]               | Q1 [S,16,128]
        | K0 [S, 4,128]               | K1 [S, 4,128]
        | V0 [S, 4,128]               | V1 [S, 4,128]
        +-----------------------------+-----------------------------+
```

That is what "Q/K/V projection outputs split by heads" means.

## 3. Local RoPE, KV Cache And Attention

Each GPU applies RoPE only to its local Q and K heads:

```text
GPU 0:
    Qrot0 = RoPE(Q0)
    Krot0 = RoPE(K0)

GPU 1:
    Qrot1 = RoPE(Q1)
    Krot1 = RoPE(K1)
```

Each GPU stores its local K/V heads in its local KV-cache shard:

```text
GPU 0 KV cache: Krot0 and V0
GPU 1 KV cache: Krot1 and V1
```

Because this is grouped-query attention:

```text
16 local Q heads / 4 local KV heads
    = 4 Q heads sharing each KV head
```

Each GPU computes only its local attention:

```text
GPU 0: A0 = Attention(Qrot0, Kcache0, Vcache0)
GPU 1: A1 = Attention(Qrot1, Kcache1, Vcache1)
```

Shapes:

```text
GPU 0: A0 [S,16,128] -> [S,2048]
GPU 1: A1 [S,16,128] -> [S,2048]
```

```text
        +-----------------------------+-----------------------------+
        | GPU 0                       | GPU 1
        | RoPE(Q0,K0)                 | RoPE(Q1,K1)
        | local KV cache K0/V0        | local KV cache K1/V1
        |                             |
        | local attention A0          | local attention A1
        | A0 [S,2048]                 | A1 [S,2048]
        +-----------------------------+-----------------------------+
```

The logical complete attention value would be:

```text
A = concat(A0, A1) -> [S,4096]
```

MAX does not need to create this concatenated value before `o_proj`.

## 4. `o_proj` Input Channels Split

The complete output projection is:

```text
Y = A @ Wo.T

A  [S,4096]
Wo [4096,4096]
Y  [S,4096]
```

Because `A` is divided between GPUs, `Wo` is split along its input-channel dimension:

```text
Wo = [Wo0 | Wo1]

Wo0 [4096,2048]
Wo1 [4096,2048]
```

Each GPU multiplies its local attention channels by the corresponding weight columns:

```text
GPU 0:
    Y0 = A0 @ Wo0.T
    Y0 [S,4096]

GPU 1:
    Y1 = A1 @ Wo1.T
    Y1 [S,4096]
```

`Y0` and `Y1` both have the final output shape, but each contains only one GPU's contribution. This is why they are called partial outputs.

Mathematically:

```text
Y = concat(A0,A1) @ [Wo0 | Wo1].T

  = A0 @ Wo0.T + A1 @ Wo1.T

  = Y0 + Y1
```

## 5. Attention Allreduce

Allreduce sums the partial values and returns the result to every GPU:

```text
GPU 0 has Y0 ----+
                 +-- allreduce SUM --+--> Y0 + Y1 -> GPU 0
GPU 1 has Y1 ----+                   +--> Y0 + Y1 -> GPU 1
```

Overall attention flow:

```text
Replicated X [S,4096]
        |
        +-----------------------------+-----------------------------+
        | GPU 0                       | GPU 1
        | local Q/K/V heads           | local Q/K/V heads
        | local RoPE and KV cache     | local RoPE and KV cache
        | local attention A0          | local attention A1
        | partial o_proj Y0           | partial o_proj Y1
        +--------------+--------------+--------------+--------------+
                       |                             |
                       +-------- allreduce SUM ------+
                                      |
                                      v
                         complete Y [S,4096]
                         replicated on both GPUs
                                      |
                                      v
                            H = X + Y residual
```

After allreduce:

```text
GPU 0: Y [S,4096]
GPU 1: Y [S,4096]
```

Both GPUs can independently perform:

```text
H = X + Y
```

## 6. MLP Input

The post-attention RMSNorm is replicated:

```text
GPU 0: Hnorm = RMSNorm(H) -> [S,4096]
GPU 1: Hnorm = RMSNorm(H) -> [S,4096]
```

The Llama MLP has:

```text
gate_proj [14336,4096]
up_proj   [14336,4096]
down_proj [4096,14336]
```

## 7. Gate And Up Projections Split

The `14336` intermediate channels are divided across GPUs:

```text
Full Wgate [14336,4096]
    +-- GPU 0: Wgate0 [7168,4096]
    +-- GPU 1: Wgate1 [7168,4096]

Full Wup [14336,4096]
    +-- GPU 0: Wup0 [7168,4096]
    +-- GPU 1: Wup1 [7168,4096]
```

Each GPU computes half of the intermediate MLP channels:

```text
GPU 0:
    Gate0 = Hnorm @ Wgate0.T -> [S,7168]
    Up0   = Hnorm @ Wup0.T   -> [S,7168]
    Z0    = SiLU(Gate0) * Up0

GPU 1:
    Gate1 = Hnorm @ Wgate1.T -> [S,7168]
    Up1   = Hnorm @ Wup1.T   -> [S,7168]
    Z1    = SiLU(Gate1) * Up1
```

```text
Replicated Hnorm [S,4096]
        |
        +-----------------------------+-----------------------------+
        | GPU 0                       | GPU 1
        | Gate0 [S,7168]              | Gate1 [S,7168]
        | Up0   [S,7168]              | Up1   [S,7168]
        |                             |
        | Z0 = SiLU(Gate0) * Up0      | Z1 = SiLU(Gate1) * Up1
        +-----------------------------+-----------------------------+
```

No communication is needed for SiLU or multiplication because these are elementwise operations on independent intermediate channels.

Logically:

```text
Z = concat(Z0, Z1) -> [S,14336]
```

## 8. Down Projection Split

The down projection converts `14336` intermediate channels back to hidden size `4096`:

```text
M = Z @ Wdown.T
```

`Wdown` is split along its input-channel dimension:

```text
Wdown = [Wdown0 | Wdown1]

Wdown0 [4096,7168]
Wdown1 [4096,7168]
```

Each GPU calculates a partial full-shaped output:

```text
GPU 0:
    M0 = Z0 @ Wdown0.T
    M0 [S,4096]

GPU 1:
    M1 = Z1 @ Wdown1.T
    M1 [S,4096]
```

Mathematically:

```text
M = concat(Z0,Z1) @ [Wdown0 | Wdown1].T

  = Z0 @ Wdown0.T + Z1 @ Wdown1.T

  = M0 + M1
```

## 9. MLP Allreduce

```text
GPU 0 has M0 ----+
                 +-- allreduce SUM --+--> M0 + M1 -> GPU 0
GPU 1 has M1 ----+                   +--> M0 + M1 -> GPU 1
```

Overall MLP flow:

```text
Replicated Hnorm [S,4096]
        |
        +-----------------------------+-----------------------------+
        | GPU 0                       | GPU 1
        | local Gate0 and Up0         | local Gate1 and Up1
        | local activation Z0         | local activation Z1
        | partial down_proj M0        | partial down_proj M1
        +--------------+--------------+--------------+--------------+
                       |                             |
                       +-------- allreduce SUM ------+
                                      |
                                      v
                         complete M [S,4096]
                         replicated on both GPUs
                                      |
                                      v
                            Output = H + M
```

## 10. Complete Transformer Block

```text
Replicated X [S,4096]
        |
        +-- replicated input RMSNorm
        |
        +-- split Q/K/V projection outputs by heads
        +-- local RoPE
        +-- local KV-cache writes
        +-- local attention
        +-- split o_proj input channels
        +-- partial attention outputs
        +-- allreduce SUM
        |
        +-- attention residual: H = X + Y
        +-- replicated post-attention RMSNorm
        |
        +-- split gate/up output channels
        +-- local SiLU(Gate) * Up
        +-- split down_proj input channels
        +-- partial MLP outputs
        +-- allreduce SUM
        |
        +-- MLP residual: Output = H + M
        |
        v
Replicated block output [S,4096] on both GPUs
```

Attention partitions the Q/K/V heads. The MLP partitions the `14336` intermediate channels. Both paths project their local values back to `[S,4096]`, produce partial contributions, and use allreduce to reconstruct the complete result on every GPU.

</details>

<details>
<summary><strong>Which structure holds Wq0 and Wq1 after graph compilation?</strong></summary>

The actual GPU bytes are held in native runtime-managed device allocations owned by the initialized `max.engine.Model`.

They are not stored inside the Python `Weight` object.

```text
self.model: max.engine.Model
    |
    +-- native MAX runtime model handle
          |
          +-- GPU 0 device allocation containing Wq0
          +-- GPU 1 device allocation containing Wq1
```

There are three different structures at different stages:

| Stage | Structure | Holds actual bytes? |
|---|---|---:|
| Checkpoint loading | `weights_registry: dict[str, DLPackArray]` | Yes, full host `Wq` |
| Graph construction | `max.graph.Weight` shard objects | No, symbolic descriptors |
| Model initialization/runtime | native allocations owned by `max.engine.Model` | Yes, GPU shards |

## During Graph Construction

The Python objects are approximately:

```text
TensorParallelAttentionWithRope
    |
    +-- list_of_attentions[0]
    |     |
    |     +-- qkv_proj.q_proj.weight
    |           max.graph.Weight
    |           shard_idx=0
    |           device=gpu:0
    |           -> symbolic Wq0
    |
    +-- list_of_attentions[1]
          |
          +-- qkv_proj.q_proj.weight
                max.graph.Weight
                shard_idx=1
                device=gpu:1
                -> symbolic Wq1
```

These `Weight` objects contain metadata:

```text
name
dtype
shape
DeviceRef
shard index
sharding strategy
reference to the original host Weight
```

They do not contain the actual GPU memory.

## Host Weight Structure

The complete checkpoint tensor is held through:

```python
weights_registry: dict[str, DLPackArray]
```

For example:

```text
weights_registry[
    "layers.0.self_attn.q_proj.weight"
]
    |
    v
full Wq [4096,4096] BF16
```

The underlying `DLPackArray` may be a safetensors-backed array, NumPy array, or MAX `Buffer`.

## MLIR Graph Structure

When the symbolic shard is used, MAX inserts graph values conceptually equivalent to:

```text
external constant Wq
    |
    +-- shard/slice for shard_idx=0
    |       |
    |       +-- transfer to gpu:0
    |              |
    |              v
    |          TensorValue Wq0
    |
    +-- shard/slice for shard_idx=1
            |
            +-- transfer to gpu:1
                   |
                   v
               TensorValue Wq1
```

`TensorValue` and the underlying MLIR SSA value describe data flow. They still do not own persistent GPU bytes.

## Runtime Structure

Initialization crosses this native boundary:

```python
models = self._impl._load_all(
    compiled._compiled,
    weights_registry,
)
```

`_load_all()` returns native model handles:

```text
max._core.engine.Model
    exposed as
max.engine.Model
```

That initialized model internally references the persistent device allocations:

```text
max.engine.Model
    |
    +-- runtime parameter/resource table
    |     |
    |     +-- gpu:0 allocation -> Wq0
    |     +-- gpu:1 allocation -> Wq1
    |
    +-- compiled executable kernels
    +-- device queues and execution state
```

The Python code does not expose this parameter table as something like:

```python
model.gpu0_weights["Wq0"]
```

The storage is opaque and managed by the native MAX engine.

So the precise answer is:

```text
Wq0 actual bytes
    -> GPU 0 VRAM allocation
    -> owned/referenced by native max.engine.Model

Wq1 actual bytes
    -> GPU 1 VRAM allocation
    -> owned/referenced by native max.engine.Model
```

The per-device Python `max.graph.Weight` objects only tell the compiler which shard each native runtime allocation must contain.

</details>

```text
    |     |     |     |     |     +-- down_proj input channels split; partial
    |     |     |     |     |     |   MLP outputs combined by allreduce
    |     |     |     |     |     +-- RMSNorm weights replicated on every GPU
    |     |     |     |     |     +-- embedding vocabulary rows sharded
    |     |     |     |     |     +-- LM-head vocabulary output range sharded;
    |     |     |     |     |     |   logits combined by allgather
    |     |     |     |     |     +-- KV heads divided across TP devices
    |     |     |     |     |     +-- [LEAF] sharded Module/Weight hierarchy
    |     |     |     |     |
    |     |     |     |     +-- dist_model.load_state_dict(state_dict, ...)
    |     |     |     |     |     +-- each sharded Weight records the slice of
    |     |     |     |     |     |   the original checkpoint tensor it owns
    |     |     |     |     |     +-- registry values remain DLPack-compatible
    |     |     |     |     |     +-- [LEAF] per-device model state registry
    |     |     |     |     |
    |     |     |     |     +-- DistributedLlama3.input_types(kv_params)
    |     |     |     |           |
    |     |     |     |           +-- normal token/offset/return-logit inputs
    |     |     |     |           +-- Signals(devices).input_types()
    |     |     |     |           |     +-- one mutable uint8 BufferType/GPU
    |     |     |     |           |     +-- barrier counters and communication
    |     |     |     |           |     |   scratch used by P2P collectives
    |     |     |     |           |     +-- graph inputs because these buffers
    |     |     |     |           |         are runtime state, not weights
    |     |     |     |           +-- flattened per-device KV-cache inputs
    |     |     |     |           +-- [LEAF] TP graph input signature
    |     |     |     |
    |     |     |     +-- runtime Signals.allocate(...)
    |     |     |     |     +-- enable peer access when available
    |     |     |     |     +-- allocate/zero one signal buffer per GPU
    |     |     |     |     +-- initialize Lamport communication region
    |     |     |     |     +-- synchronize devices before first collective
    |     |     |     |
    |     |     |     +-- NOTE
    |     |     |     |     These signal buffers are unrelated to the paged
    |     |     |     |     KV cache. They coordinate allreduce/allgather and
    |     |     |     |     device-to-device communication.
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
    |     |                 |     |     +-- [BRANCH] longrope configured
    |     |                 |     |     |     +-- LongRoPERotaryEmbedding(...)
    |     |                 |     |     |           |
    |     |                 |     |     |           +-- INACTIVE for Llama 3.1
    |     |                 |     |     |           +-- consume short_factor[] and
    |     |                 |     |     |           |   long_factor[] arrays
    |     |                 |     |     |           +-- compute separately scaled
    |     |                 |     |     |           |   inverse frequencies
    |     |                 |     |     |           +-- positions below original
    |     |                 |     |     |           |   limit use short factors
    |     |                 |     |     |           +-- later positions use long
    |     |                 |     |     |           |   factors
    |     |                 |     |     |           +-- stitch both cos/sin tables
    |     |                 |     |     |           +-- optionally adjust attention
    |     |                 |     |     |               scale for context extension
    |     |                 |     |     |
    |     |                 |     |     +-- [BRANCH] otherwise: ACTIVE HERE
    |     |                 |     |           +-- [CALL] Llama3RotaryEmbedding(...)
    |     |                 |     |                 rotary_embedding.py:362
    |     |                 |     |                 |
    |     |                 |     |                 +-- dim=4096, n_heads=32
    |     |                 |     |                 +-- head_dim=4096/32=128
    |     |                 |     |                 +-- theta=500000
    |     |                 |     |                 +-- interleaved=False
    |     |                 |     |                 +-- Llama3 scaling parameters
    |     |                 |     |                 |     factor=8, low=1, high=4,
    |     |                 |     |                 |     original positions=8192
    |     |                 |     |                 |
    |     |                 |     |                 +-- base inverse frequency i
    |     |                 |     |                 |     inv_freq[i]
    |     |                 |     |                 |       = 1 / theta^(2i/128)
    |     |                 |     |                 |       for i=0..63
    |     |                 |     |                 |
    |     |                 |     |                 +-- classify each wavelength
    |     |                 |     |                 |     short/high-frequency:
    |     |                 |     |                 |       retain base frequency
    |     |                 |     |                 |     long/low-frequency:
    |     |                 |     |                 |       divide frequency by 8
    |     |                 |     |                 |     middle range:
    |     |                 |     |                 |       smoothly interpolate
    |     |                 |     |                 |
    |     |                 |     |                 +-- outer(position, inv_freq)
    |     |                 |     |                 +-- stack cos and sin values
    |     |                 |     |                 +-- reshape frequency table to
    |     |                 |     |                 |   [2*max_seq_len, 128]
    |     |                 |     |                 +-- [LEAF] symbolic freqs_cis
    |     |                 |     |                     later rotates Q and K only;
    |     |                 |     |                     V is not position-rotated
    |     |                 |     |
    |     |                 |     +-- select normalization constructor
    |     |                 |     |     +-- [ACTIVE] RMSNorm(4096, BF16, 1e-5,
    |     |                 |     |     |     multiply_before_cast=False)
    |     |                 |     |     |     rms_norm.py:55
    |     |                 |     |     |     |
    |     |                 |     |     |     +-- learned gamma shape [4096]
    |     |                 |     |     |     +-- rms = sqrt(mean(x^2) + eps)
    |     |                 |     |     |     +-- y = (x / rms) * gamma
    |     |                 |     |     |     +-- no mean subtraction and no beta
    |     |                 |     |     |     +-- used before attention, before MLP,
    |     |                 |     |     |         and once before the LM head
    |     |                 |     |     |
    |     |                 |     |     +-- [INACTIVE] ConstantLayerNorm(...)
    |     |                 |     |           layer_norm.py:154
    |     |                 |     |           +-- constant gamma=1 and beta=0
    |     |                 |     |           +-- normalize in FP32
    |     |                 |     |           +-- cast result back to input dtype
    |     |                 |     |
    |     |                 |     +-- select linear constructor
    |     |                 |     |     +-- [INACTIVE] GPTQLinear for GPTQ
    |     |                 |     |     |     linear.py:748
    |     |                 |     |     |     +-- packed uint8 qweight
    |     |                 |     |     |     +-- packed scales
    |     |                 |     |     |     +-- optional int32 perm_idx for
    |     |                 |     |     |     |   activation-order GPTQ
    |     |                 |     |     |     +-- reconstruct packed operand layout
    |     |                 |     |     |     +-- ops.qmatmul(...)
    |     |                 |     |     |
    |     |                 |     |     +-- [ACTIVE] Linear(quant_config=None)
    |     |                 |     |           +-- BF16 Weight tensors
    |     |                 |     |           +-- no quantization scales
    |     |                 |     |           +-- graph emits x @ weight.T
    |     |                 |     |
    |     |                 |     +-- select attention implementation
    |     |                 |     |     +-- GPTQAttentionWithRope for GPTQ
    |     |                 |     |     +-- GGUFQAttentionWithRope for GGUF quantization
    |     |                 |     |     +-- [ACTIVE] AttentionWithRope otherwise
    |     |                 |     |
    |     |                 |     +-- [LOOP] construct 32 TransformerBlock objects
    |     |                 |     |     |
    |     |                 |     |     +-- AttentionWithRope(...)
    |     |                 |     |     |     attention_with_rope.py:60
    |     |                 |     |     |     |
    |     |                 |     |     |     +-- q_weight_dim
    |     |                 |     |     |     |     = 32 Q heads * 128 = 4096
    |     |                 |     |     |     +-- kv_weight_dim
    |     |                 |     |     |     |     = 8 KV heads * 128 = 1024
    |     |                 |     |     |     +-- flat QKV output width
    |     |                 |     |     |     |     = 4096 + 1024 + 1024 = 6144
    |     |                 |     |     |     +-- scale=1/sqrt(128)
    |     |                 |     |     |     +-- mask_variant=CAUSAL_MASK
    |     |                 |     |     |     +-- StackedLinear(...)
    |     |                 |     |     |     |     +-- stacked_qkv=False
    |     |                 |     |     |     |     +-- q_proj weight [4096,4096]
    |     |                 |     |     |     |     +-- k_proj weight [1024,4096]
    |     |                 |     |     |     |     +-- v_proj weight [1024,4096]
    |     |                 |     |     |     |     +-- may concatenate these symbolic
    |     |                 |     |     |     |         weights for one QKV matmul
    |     |                 |     |     |     +-- o_proj weight [4096,4096]
    |     |                 |     |     |     +-- rope reference supplies freqs_cis
    |     |                 |     |     |     +-- kv_params supplies paged-cache
    |     |                 |     |     |     |   dimensions and storage contract
    |     |                 |     |     |     +-- [LEAF] attention Module tree
    |     |                 |     |     |
    |     |                 |     |     +-- MLP(...) or StackedMLP(...)
    |     |                 |     |     |     +-- [ACTIVE] MLP with separate
    |     |                 |     |     |     |   checkpoint weights
    |     |                 |     |     |     |     gate_proj [14336,4096]
    |     |                 |     |     |     |     up_proj   [14336,4096]
    |     |                 |     |     |     |     down_proj [4096,14336]
    |     |                 |     |     |     |
    |     |                 |     |     |     +-- graph-time fused gate/up option
    |     |                 |     |     |     |     quantization is None, therefore
    |     |                 |     |     |     |     concatenate gate/up weights to
    |     |                 |     |     |     |     [28672,4096] and use one matmul
    |     |                 |     |     |     |
    |     |                 |     |     |     +-- Hnorm [S,4096]
    |     |                 |     |     |     +-- Gate = Hnorm @ Wgate.T
    |     |                 |     |     |     |     -> [S,14336]
    |     |                 |     |     |     +-- Up = Hnorm @ Wup.T
    |     |                 |     |     |     |     -> [S,14336]
    |     |                 |     |     |     +-- Activated = SiLU(Gate) * Up
    |     |                 |     |     |     |     -> [S,14336]
    |     |                 |     |     |     +-- MlpOut = Activated @ Wdown.T
    |     |                 |     |     |     |     -> [S,4096]
    |     |                 |     |     |     +-- [INACTIVE] StackedMLP checkpoint
    |     |                 |     |     |         has gate_up_proj [28672,4096]
    |     |                 |     |     |     +-- [LEAF] MLP Module tree
    |     |                 |     |     |
    |     |                 |     |     +-- attention RMSNorm + MLP RMSNorm
    |     |                 |     |     +-- [LEAF] TransformerBlock Module tree
    |     |                 |     |
    |     |                 |     +-- Embedding(128256, 4096)
    |     |                 |     +-- LM-head Linear(4096, 128256)
    |     |                 |     +-- tie_word_embeddings=False
    |     |                 |     |     +-- do not share embedding and LM-head Weight
    |     |                 |     +-- Transformer.__init__(...)
    |     |                 |           transformer.py:488
    |     |                 |           |
    |     |                 |           +-- self.layers = LayerList(32 blocks)
    |     |                 |           +-- self.norm = final RMSNorm(4096)
    |     |                 |           +-- self.lm_head = separate vocabulary projection
    |     |                 |           +-- self.embed_tokens = token embedding
    |     |                 |           +-- self.kv_params = paged-cache layout contract
    |     |                 |           +-- self.rope = cached frequency-table producer
    |     |                 |           +-- return_logits = LAST_TOKEN normally
    |     |                 |           +-- return_hidden_states = NONE normally
    |     |                 |           +-- embedding_multiplier=1.0
    |     |                 |           +-- logits_scaling=1.0
    |     |                 |           |
    |     |                 |           +-- Module.__setattr__(...)
    |     |                 |           |     +-- child Module values enter _sublayers
    |     |                 |           |     +-- Weight values enter _layer_weights
    |     |                 |           |     +-- recursive attribute names form stable
    |     |                 |           |     |   names such as
    |     |                 |           |     |   layers.0.self_attn.q_proj.weight
    |     |                 |           |     +-- registration builds module/weight
    |     |                 |           |         ownership; it executes no inference
    |     |                 |           |
    |     |                 |           +-- [LEAF] single_model: Llama3 Module tree
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
    |     |                 |     |           |
    |     |                 |     |           +-- DLPackArray is a runtime-checkable
    |     |                 |     |           |   Python Protocol, not one concrete
    |     |                 |     |           |   array implementation
    |     |                 |     |           +-- requires __dlpack__() and
    |     |                 |     |           |   __dlpack_device__()
    |     |                 |     |           +-- protocol communicates tensor pointer,
    |     |                 |     |           |   dtype, shape/strides, and device metadata
    |     |                 |     |           +-- NumPy/framework arrays, MAX Buffer,
    |     |                 |     |           |   and WeightData can satisfy it
    |     |                 |     |           +-- often enables zero-copy sharing, but
    |     |                 |     |           |   DLPack itself neither owns nor copies
    |     |                 |     |           |   the underlying memory
    |     |                 |     |           +-- producer lifetime, device placement,
    |     |                 |     |               alignment, and contiguity still matter
    |     |                 |     |           +-- _array_from_weight_loader(...).data
    |     |                 |     |               unwraps WeightData and stores the
    |     |                 |     |               underlying DLPackArray in the Module
    |     |                 |     +-- strict=False tolerates the unused rope_freqs key
    |     |                 |     +-- [LEAF] Module weight-value registry populated
    |     |                 |
    |     |                 +-- [CALL] single_model.state_dict()
    |     |                 |     max/python/max/nn/layer/layer.py:480
    |     |                 |     |
    |     |                 |     +-- iterate raw_state_dict() names
    |     |                 |     +-- return the loaded underlying DLPack arrays
    |     |                 |     |     +-- the earlier WeightData envelope carried
    |     |                 |     |         name, dtype, shape, and optional
    |     |                 |     |         quantization metadata during validation
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
    |     |                 |     +-- collect symbolic dimension parameters
    |     |                 |     |     total_seq_len, input_row_offsets_len,
    |     |                 |     |     return_n_logits, and symbolic KV dimensions
    |     |                 |     |
    |     |                 |     +-- create or reuse builtin.ModuleOp
    |     |                 |     |     +-- top-level MLIR container that can own
    |     |                 |     |         this graph and optional subgraphs
```

<details>
<summary><strong>Is device information saved in the mo.graph operation?</strong></summary>

No. `mo.device_info_mapping` is saved on the enclosing MLIR `builtin.module`, not directly on the `mo.graph` operation.

The code does:

```python
module = self._mlir_op.block.owner

module.attributes["mo.device_info_mapping"] = ...
```

The resulting MLIR structure is conceptually:

```text
builtin.module
    attributes {
        mo.device_info_mapping = {
            "cpu": {
                label = "cpu",
                api = ...,
                arch = ...,
                model = ...,
                tile_based_fusion = false
            },
            "gpu": {
                label = "gpu",
                api = "cuda" or "hip",
                arch = ...,
                model = ...,
                tile_based_fusion = false
            }
        }
    }
    |
    +-- mo.graph @llama3(...)
          |
          +-- graph inputs
          +-- weights
          +-- computation operations
          +-- outputs
```

Device placement is represented separately inside `mo.graph`. For example:

```text
TensorType(..., device=gpu:0)
Weight(..., device=gpu:0)
ops executing on gpu:0
```

These become MLIR types or attributes containing a `DeviceRefAttr`:

```text
gpu:0
cpu:0
```

The relationship is:

```text
MLIR module
    |
    +-- mo.device_info_mapping
    |     describes available device kinds:
    |     API, architecture and model information
    |
    +-- mo.graph @llama3
          |
          +-- DeviceRef gpu:0
          |     identifies where an operation/value is placed
          |
          +-- DeviceRef cpu:0
                identifies host placement
```

When compiling, MAX passes the complete MLIR module to the native compiler:

```text
builtin.module
    +-- mo.device_info_mapping
    +-- mo.graph @llama3
    +-- kernel declarations
          |
          v
native MAX compiler
```

So:

```text
mo.device_info_mapping
    -> module-level device capability metadata

DeviceRef attributes inside mo.graph
    -> per-tensor and per-operation placement

mo.graph
    -> model computation and data flow
```

The compiler reads both together while lowering the graph for the target CPU or GPU.

</details>

```text
    |     |                 |     |
    |     |                 |     +-- [CALL] _mo.GraphOp(name="llama3", ...)
    |     |                 |     |     +-- convert every TensorType/BufferType
    |     |                 |     |     |   into its native MLIR type
    |     |                 |     |     +-- create the mo.graph operation
    |     |                 |     |     +-- create its graph region
    |     |                 |     |     +-- create the region's entry block
    |     |                 |     |     +-- create one block argument per input
    |     |                 |     |     +-- [LEAF] block arguments are symbolic
    |     |                 |     |         MLIR SSA values, not request data
    |     |                 |     |
    |     |                 |     +-- _populate_device_info_mapping()
    |     |                 |     |     +-- attach visible CPU/GPU labels, API,
    |     |                 |     |     |   architecture, and model metadata
    |     |                 |     |     +-- compiler later uses placement metadata
    |     |                 |     |         while lowering device operations
    |     |                 |     |
    |     |                 |     +-- initialize _DeviceChainMap
    |     |                 |     |     +-- host/CPU chain orders orchestration ops
    |     |                 |     |     +-- lazily create a chain for each device
    |     |                 |     |     +-- a new device chain starts from the host chain
    |     |                 |     |     +-- collectives merge host and participating
    |     |                 |     |     |   device chains before communication
    |     |                 |     |     +-- chains express asynchronous ordering
    |     |                 |     |         dependencies without forcing every GPU
    |     |                 |     |         operation to execute synchronously
```

<details>
<summary><strong>In case of Llama 3.1 8B model =&gt; how above happens (explain above answer with example)</strong></summary>

Assume Llama 3.1 8B runs prefill with:

```text
S = prompt token count
hidden size = 4096
32 Q heads, 8 KV heads, head dimension = 128
MLP intermediate size = 14336
2 GPUs using tensor parallelism
```

### 1. Initial Chain State

When the graph is created:

```text
host chain = H0
GPU0 chain = H0
GPU1 chain = H0
```

The GPU chains start from `H0` because host-side setup must finish first:

```text
load weights
create KV-cache buffers
create allreduce signal buffers
prepare input tensors
```

### 2. Vocabulary-Parallel Embedding

The vocabulary of 128,256 entries is divided:

```text
GPU0: embedding rows     0 .. 64,127
GPU1: embedding rows 64,128 .. 128,255
```

Tokens are broadcast to both GPUs:

```text
tokens on GPU0
      |
      +-- distributed broadcast --> GPU0 tokens
      |                         -> GPU1 tokens
```

Each GPU performs its local embedding lookup:

```text
GPU0 -> E0[S,4096]
GPU1 -> E1[S,4096]
```

Only one GPU has a nonzero result for each token. An allreduce combines them:

```text
merge(H0, GPU0-chain, GPU1-chain) -> M0
M0 -> allreduce(E0, E1) -> E[S,4096] on both GPUs
```

Afterward:

```text
host = E_chain
GPU0 = E_chain
GPU1 = E_chain
```

### 3. One Transformer Layer

This process repeats for all 32 layers.

#### Local Attention Work

Each GPU independently processes half the heads:

```text
GPU0:
    Q0[S,16,128]
    K0[S, 4,128]
    V0[S, 4,128]
    local KV cache for 4 KV heads

GPU1:
    Q1[S,16,128]
    K1[S, 4,128]
    V1[S, 4,128]
    local KV cache for 4 KV heads
```

The chains can progress independently:

```text
GPU0: E_chain -> QKV/Attention/KV-write -> A0_chain
GPU1: E_chain -> QKV/Attention/KV-write -> A1_chain
```

Therefore GPU0 and GPU1 may execute their local attention simultaneously.

Each local output projection produces a partial hidden result:

```text
GPU0: Y0[S,4096]
GPU1: Y1[S,4096]
```

They must be summed:

```text
merge(host, A0_chain, A1_chain) -> M_attn
M_attn -> allreduce(Y0, Y1) -> Y[S,4096] on both GPUs
```

The allreduce output chain replaces all participating chains:

```text
host = ATTENTION_DONE
GPU0 = ATTENTION_DONE
GPU1 = ATTENTION_DONE
```

### 4. Local MLP Work

The MLP intermediate dimension is divided:

```text
GPU0:
    Gate0[S,7168]
    Up0[S,7168]
    Activated0 = SiLU(Gate0) * Up0
    Down0[S,4096] partial result

GPU1:
    Gate1[S,7168]
    Up1[S,7168]
    Activated1 = SiLU(Gate1) * Up1
    Down1[S,4096] partial result
```

The two MLP paths may overlap:

```text
GPU0: ATTENTION_DONE -> local MLP -> MLP0_chain
GPU1: ATTENTION_DONE -> local MLP -> MLP1_chain
```

The partial down-projection results are combined:

```text
merge(host, MLP0_chain, MLP1_chain) -> M_mlp
M_mlp -> allreduce(Down0, Down1) -> MLP[S,4096]
```

Afterward:

```text
host = LAYER_0_DONE
GPU0 = LAYER_0_DONE
GPU1 = LAYER_0_DONE
```

Layer 1 starts from `LAYER_0_DONE`, and this repeats through layer 31.

### Important Distinction

Tensor SSA dependencies order normal operations such as matmul, RoPE and RMSNorm. Device chains principally order asynchronous transfers, collectives, cache-related stateful operations and orchestration.

The allreduce `signal_buffers` are real GPU communication buffers. The `!mo.chain` values contain no data; they only ensure that the communication starts after both GPUs have produced their partial results.

Thus, `_DeviceChainMap` lets GPU0 and GPU1 work concurrently between collectives while enforcing synchronization at embedding, attention-output and MLP-output allreduces.

</details>

```text
    |     |                 |     |
    |     |                 |     +-- create always-ready chain
    |     |                 |     |     +-- used for work safe to schedule without
    |     |                 |     |         normal per-device ordering, such as staging
    |     |                 |     |
    |     |                 |     +-- [CALL] KernelLibrary()
    |     |                 |     |     graph.py:214
    |     |                 |     |     |
    |     |                 |     |     +-- obtain default MLIR context
    |     |                 |     |     +-- register MAX dialect/context support
    |     |                 |     |     +-- create native kernel Analysis object
    |     |                 |     |     +-- inspect built-in/default kernel overlays
    |     |                 |     |     +-- load optional compiled .mojoc packages
    |     |                 |     |     +-- compile/load optional Mojo source packages
    |     |                 |     |     +-- later seed referenced kernel declarations
    |     |                 |     |         into the module before native compilation
    |     |                 |     |
    |     |                 |     +-- [CALL] Graph.__enter__()
    |     |                 |           graph.py:1110
    |     |                 |           |
    |     |                 |           +-- ensure the default MLIR context is active,
    |     |                 |           |   including background-thread construction
```

<details>
<summary><strong>What is the meaning of the default MLIR context?</strong></summary>

The **default MLIR context** is the shared in-memory environment MAX uses while constructing MLIR operations.

It is created once:

```python
_DEFAULT_MLIR_CONTEXT = mlir.Context()
```

It owns or manages:

```text
MLIR Context
    |
    +-- registered dialects
    |     +-- builtin
    |     +-- mo
    |     +-- kgen
    |     +-- other MAX dialects
    |
    +-- MLIR types
    |     +-- tensor types
    |     +-- buffer types
    |     +-- !mo.chain
    |
    +-- attributes and locations
    +-- operation definitions
    +-- diagnostic/error handlers
    +-- type/attribute uniquing and internal caches
```

For the Llama graph:

```text
with Graph("llama3") as graph:
    create TensorType
    create mo.graph
    create matmul/RMSNorm/RoPE operations
    create device chains
    create collective operations
```

All these MLIR objects must belong to an active `mlir.Context`.

### Meaning of “Active”

The Python MLIR API keeps the current context in thread-local state:

```python
mlir.Context.current
```

Conceptually:

```text
Python thread
    |
    +-- active MLIR context
          |
          +-- newly created MLIR types
          +-- newly created attributes
          +-- newly inserted operations
```

Without an active context, operations such as creating an MLIR location or type can fail:

```text
RuntimeError: Can't create location: No MLIR context active
```

### Background Threads

The current context is thread-local. A background thread does not automatically inherit the main thread’s context:

```text
Main thread
    mlir.Context.current = default context

Background thread
    mlir.Context.current = None
```

Therefore `Graph.__enter__()` does:

```python
with ensure_default_mlir_context():
    ...
```

This temporarily activates the shared context on that thread:

```text
Background thread
    |
    +-- enter _DEFAULT_MLIR_CONTEXT
    |
    +-- construct Llama MLIR graph
    |
    +-- leave thread-local context scope
```

“including background-thread construction” therefore means:

> If the Llama graph is being built from a worker thread, make the shared MLIR context current on that thread before creating MLIR operations.

It does **not** mean a CUDA context, GPU execution context, FastAPI request context, or `InferenceSession`. It is specifically the environment used to construct and manage the MLIR representation before compilation.

</details>

<details>
<summary><strong>Can you give me an example =&gt; what goes into this MLIR context =&gt; in this Llama 3.1 8B example?</strong></summary>

The MLIR context does **not** receive actual prompt data or GPU weights. It is the environment in which the symbolic Llama graph, its types, attributes and operations are created.

Assume Llama 3.1 8B, BF16, two GPUs and prompt length `S`.

## 1. Dialect Definitions

The context knows the operation languages used by MAX:

```text
MLIR Context
    |
    +-- builtin dialect
    |     +-- module
    |     +-- function/region/block structures
    |
    +-- MO dialect
    |     +-- graph operations
    |     +-- device transfers
    |     +-- chains
    |     +-- distributed collectives
    |
    +-- KGEN dialect
          +-- kernel declarations
          +-- kernel-generation attributes
```

This lets MLIR understand operations such as `mo.graph`, `mo.chain.create` and distributed allreduce.

## 2. Llama Types

When MAX constructs graph inputs, intermediate values and outputs, their types are created in this context:

```text
tokens:
    int64 [total_seq_len] on GPU0

input_row_offsets:
    uint32 [number_of_requests + 1] on GPU0

return_n_logits:
    int64 [N] on CPU

hidden state:
    bfloat16 [S, 4096]

local GPU0 Q:
    bfloat16 [S, 16, 128]

local GPU0 K/V:
    bfloat16 [S, 4, 128]

MLP intermediate:
    bfloat16 [S, 7168] per GPU

device chain:
    !mo.chain

KV-cache and allreduce buffers:
    !mo.buffer<...>
```

Equivalent types are uniqued. Repeated uses of `bfloat16 [S,4096]` can refer to the same canonical MLIR type representation.

## 3. `graph.inputs` and Symbolic Block Arguments

The graph inputs are also visible at this point:

```text
    |     |                 +-- graph.inputs
    |     |                 |     +-- read the entry block's symbolic arguments
    |     |                 |     +-- wrap MLIR block arguments as Value objects
    |     |                 |     |     +-- TensorType -> value.tensor
    |     |                 |     |     +-- BufferType -> value.buffer
    |     |                 |     +-- unpack tokens, offsets, return_n_logits, KV leaves
```

When `_mo.GraphOp` is created, MAX converts every declared `TensorType` and
`BufferType` into an MLIR type and creates one symbolic SSA block argument for
each graph input:

```text
input_types
    |
    +-- tokens: TensorType(int64, [total_seq_len], gpu:0)
    +-- input_row_offsets: TensorType(uint32, [rows+1], gpu:0)
    +-- return_n_logits: TensorType(int64, [N], cpu:0)
    +-- KV-cache inputs: BufferType/TensorType values
    |
    v
mo.graph @llama3 entry block
    |
    +-- %arg0: symbolic tokens
    +-- %arg1: symbolic input-row offsets
    +-- %arg2: symbolic return-n-logits value
    +-- %arg3...: symbolic KV-cache leaves
```

For the two-GPU tensor-parallel graph, signal buffers are inserted before the
KV-cache leaves:

```text
%arg0       tokens
%arg1       input_row_offsets
%arg2       return_n_logits
%arg3       GPU0 allreduce signal buffer
%arg4       GPU1 allreduce signal buffer
%arg5...    GPU0/GPU1 KV-cache input leaves
```

The `graph.inputs` property reads those existing MLIR block arguments and wraps
them in Python `Value` objects:

```text
MLIR block argument
    |
    v
Value wrapper
    |
    +-- TensorType -> value.tensor -> TensorValue
    +-- BufferType -> value.buffer -> BufferValue
```

Therefore:

```text
MLIR module associated with the context contains
    -> typed symbolic SSA block arguments

graph.inputs returns
    -> Python wrappers referring to those MLIR values

graph.inputs does not contain
    -> actual prompt tokens, KV-cache bytes or allreduce-buffer contents
```

At request execution time, the runtime supplies concrete token, offset,
KV-cache and signal-buffer values for these compiled symbolic arguments.

## 4. Device Attributes

Device placement information is also created using the context:

```text
DeviceRefAttr(cpu:0)
DeviceRefAttr(gpu:0)
DeviceRefAttr(gpu:1)
```

For example:

```text
tokens                 -> gpu:0
return_n_logits        -> cpu:0
GPU0 KV-cache          -> gpu:0
GPU1 KV-cache          -> gpu:1
GPU0 attention result  -> gpu:0
GPU1 attention result  -> gpu:1
```

The module also receives `mo.device_info_mapping`, describing the visible CPU and GPUs.

## 5. Llama Graph Operations

As Python executes the model construction code, operations are inserted into an MLIR module associated with the context:

```text
builtin.module
    |
    +-- mo.graph @llama3
          |
          +-- vocabulary-parallel embedding
          +-- distributed broadcast
          +-- embedding allreduce
          |
          +-- transformer layer 0
          |     +-- RMSNorm
          |     +-- Q/K/V projections
          |     +-- RoPE
          |     +-- KV-cache write
          |     +-- causal attention
          |     +-- output projection
          |     +-- attention allreduce
          |     +-- residual add
          |     +-- RMSNorm
          |     +-- gate/up projections
          |     +-- SiLU and multiply
          |     +-- down projection
          |     +-- MLP allreduce
          |     +-- residual add
          |
          +-- transformer layers 1...31
          +-- final RMSNorm
          +-- LM head
          +-- logits outputs
```

A simplified conceptual representation is:

```mlir
builtin.module {
    mo.graph @llama3(%tokens, %offsets, %kv_cache, %signals) {
        %hidden = embedding %tokens

        %norm = rms_norm %hidden
        %q, %k, %v = qkv_projection %norm
        %q_rot, %k_rot = rope %q, %k
        %attn_partial = paged_attention %q_rot, %k_rot, %v, %kv_cache
        %attn = distributed_allreduce %attn_partial

        %mlp_partial = llama_mlp %attn
        %mlp = distributed_allreduce %mlp_partial

        mo.output %mlp
    }
}
```

This is conceptual syntax, not the exact printed MAX MLIR.

## 6. Attributes and Metadata

The context also manages attributes attached to these operations:

```text
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
intermediate_size = 14336
number_of_layers = 32
RoPE theta and scaling values
tensor dtype = bfloat16
device placement = gpu:0 / gpu:1
collective participant devices
symbolic dimensions such as total_seq_len
source-code locations for diagnostics
```

## What Does Not Go Into It

```text
Actual prompt token values                 No
Actual KV-cache contents                   No
Actual GPU allocations                     No
CUDA streams                               No
Loaded weight bytes                        No
Compiled GPU machine code                  No
Runtime request state                      No
```

The weight operations contain symbolic references such as weight names, shapes and types. The actual weight arrays remain in `weights_registry` and are bound when `session.load(...)` initializes the compiled model.

Conceptually:

```text
MLIR Context
    -> understands and manages the graph representation

MLIR Module
    -> contains the symbolic Llama computation

weights_registry
    -> provides actual checkpoint weight arrays

session.load(...)
    -> compiles the graph and binds runtime weights

GPU runtime
    -> executes it using real prompt and KV-cache data
```

</details>

```text
    |     |                 |           +-- CURRENT_GRAPH.set(self)
    |     |                 |           |     +-- CURRENT_GRAPH is a Python ContextVar,
    |     |                 |           |     |   not global GPU/runtime state
    |     |                 |           |     +-- nested ops.* calls resolve
    |     |                 |           |     |   Graph.current from this context
    |     |                 |           |     +-- context-local state avoids one graph
    |     |                 |           |         leaking into another async Python task
    |     |                 |           +-- set current insertion target to the end
    |     |                 |           |   of the graph entry block
    |     |                 |           +-- each ops.* call
    |     |                 |           |     +-- unwrap Python Value/Type wrappers
    |     |                 |           |     +-- construct and insert an MLIR op
    |     |                 |           |     +-- verify the inserted operation
    |     |                 |           |     +-- wrap MLIR SSA results as
    |     |                 |           |         Value/TensorValue objects
    |     |                 |           +-- Graph.__exit__ resets the ContextVar token
    |     |                 |           +-- [LEAF] active MLIR insertion context
    |     |                 |
    |     |                 +-- graph.inputs
    |     |                 |     +-- read the entry block's symbolic arguments
    |     |                 |     +-- wrap MLIR block arguments as Value objects
    |     |                 |     |     +-- TensorType -> value.tensor
    |     |                 |     |     +-- BufferType -> value.buffer
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
    |     |                             |           |     +-- x [S,4096], gamma [4096]
    |     |                             |           |     +-- cast gamma to x dtype and move
    |     |                             |           |     |   it to x.device
    |     |                             |           |     +-- symbolic formula per token
    |     |                             |           |     |     rms = sqrt(mean(x*x) + 1e-5)
    |     |                             |           |     |     x_norm = (x / rms) * gamma
    |     |                             |           |     +-- [GRAPH-OP LEAF] ops.rms_norm(...)
    |     |                             |           |         OUT: x_norm [S,4096] BF16
    |     |                             |           |
    |     |                             |           +-- self_attn(...)
    |     |                             |           |     AttentionWithRope.__call__
    |     |                             |           |     attention_with_rope.py:498
    |     |                             |           |     |
    |     |                             |           |     +-- qkv_proj(x_norm)
    |     |                             |           |     |     StackedLinear.__call__
    |     |                             |           |     |     stacked_linear.py:403
    |     |                             |           |     |     |
    |     |                             |           |     |     +-- [ACTIVE] concatenate separate
    |     |                             |           |     |     |   q/k/v weights along output rows
    |     |                             |           |     |     |     Wqkv [6144,4096] BF16
    |     |                             |           |     |     +-- [INACTIVE] directly use packed
    |     |                             |           |     |     |   qkv_proj checkpoint weight
    |     |                             |           |     |     +-- linear(...)
    |     |                             |           |     |           +-- GPTQ -> ops.qmatmul
    |     |                             |           |     |           +-- scaled quant -> quantized_matmul
    |     |                             |           |     |           +-- [ACTIVE] x_norm @ Wqkv.T
    |     |                             |           |     |           +-- [GRAPH-OP LEAF] flat QKV
    |     |                             |           |     |               [S,6144] BF16
    |     |                             |           |     |               for S=512: [512,6144]
    |     |                             |           |     |
    |     |                             |           |     +-- [INACTIVE FOR LLAMA 3.1]
    |     |                             |           |     |   optional Q/K per-head RMSNorm branch
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
    |     |                             |           |     |     +-- logically split flat QKV into
    |     |                             |           |     |     |     Q [S,32,128]
    |     |                             |           |     |     |     K [S, 8,128]
    |     |                             |           |     |     |     V [S, 8,128]
    |     |                             |           |     |     +-- derive each token's position from
    |     |                             |           |     |     |   ragged row offsets/cache metadata
    |     |                             |           |     |     +-- apply Llama3 cos/sin rotation to
    |     |                             |           |     |     |   Q and K pairs; leave V unchanged
    |     |                             |           |     |     +-- write rotated K and unrotated V into
    |     |                             |           |     |     |   this layer's paged-cache slots
    |     |                             |           |     |     +-- for S=512, create 512 valid K/V
    |     |                             |           |     |     |   positions in each of 32 layers
    |     |                             |           |     |     +-- [GRAPH-OP LEAF] ops.inplace_custom(
    |     |                             |           |     |           "mo.rope_split_store.ragged.paged", ...)
    |     |                             |           |     |           OUT: Qrot [S,4096]; side effect:
    |     |                             |           |     |           Kcache/Vcache page writes
    |     |                             |           |     |
    |     |                             |           |     +-- reshape Q to [tokens,32,128]
    |     |                             |           |     |
    |     |                             |           |     +-- flash_attention_ragged(...)
    |     |                             |           |     |     kernels.py:4200
    |     |                             |           |     |     |
    |     |                             |           |     |     +-- validate Q/KV/offset metadata
    |     |                             |           |     |     +-- select causal-mask op parameters
    |     |                             |           |     |     +-- flatten KV and dispatch metadata
    |     |                             |           |     |     +-- grouped-query mapping
    |     |                             |           |     |     |     32 Q heads / 8 KV heads
    |     |                             |           |     |     |     = 4 Q heads share each KV head
    |     |                             |           |     |     +-- for query position t, read cached
    |     |                             |           |     |     |   K/V positions 0..t only
    |     |                             |           |     |     +-- logical attention per Q head
    |     |                             |           |     |     |     Scores = Qrot @ Kcache.T / sqrt(128)
    |     |                             |           |     |     |     Prob = softmax(causal_mask(Scores))
    |     |                             |           |     |     |     A = Prob @ Vcache
    |     |                             |           |     |     +-- flash implementation tiles these
    |     |                             |           |     |     |   operations and avoids materializing the
    |     |                             |           |     |     |   full [32,S,S] score/probability tensors
    |     |                             |           |     |     +-- [GRAPH-OP LEAF] ops.inplace_custom(
    |     |                             |           |     |           "mo.mha.ragged.paged", ...)
    |     |                             |           |     |           OUT: attention [S,32,128] BF16
    |     |                             |           |     |
    |     |                             |           |     +-- ops.reshape -> [tokens,4096]
    |     |                             |           |     +-- o_proj Linear.__call__
    |     |                             |           |           +-- A [S,4096] @ Wo.T [4096,4096]
    |     |                             |           |           +-- [GRAPH-OP LEAF] attention output
    |     |                             |           |               AttnOut [S,4096] BF16
    |     |                             |           |
    |     |                             |           +-- optional residual multiplier
    |     |                             |           +-- [GRAPH-OP LEAF] h = x + attn_out
    |     |                             |           |
    |     |                             |           +-- post_attention_layernorm(h)
    |     |                             |           |     +-- same RMSNorm formula and gamma [4096]
    |     |                             |           |     +-- [GRAPH-OP LEAF] h_norm [S,4096]
    |     |                             |           |
    |     |                             |           +-- mlp(h_norm)
    |     |                             |           |     |
    |     |                             |           |     +-- [ACTIVE] standard MLP.__call__ branch
    |     |                             |           |     |     linear.py:1037
    |     |                             |           |     |     |
    |     |                             |           |     |     +-- _can_used_fused_mlp() -> True
    |     |                             |           |     |     |     no model QuantizationEncoding and
    |     |                             |           |     |     |     quant_config is None
    |     |                             |           |     |     +-- concatenate Wgate and Wup
    |     |                             |           |     |     |     [14336,4096] + [14336,4096]
    |     |                             |           |     |     |       -> Wgate_up [28672,4096]
    |     |                             |           |     |     +-- GateUp = h_norm @ Wgate_up.T
    |     |                             |           |     |     |     -> [S,28672]
    |     |                             |           |     |     +-- ops.split ->
    |     |                             |           |     |     |     Gate [S,14336], Up [S,14336]
    |     |                             |           |     |     +-- GateAct = SiLU(Gate)
    |     |                             |           |     |     |     = Gate * sigmoid(Gate)
    |     |                             |           |     |     +-- Activated = GateAct * Up
    |     |                             |           |     |     |     -> [S,14336]
    |     |                             |           |     |     +-- MlpOut = Activated @ Wdown.T
    |     |                             |           |     |     |     Wdown [4096,14336]
    |     |                             |           |     |     |     -> [S,4096]
    |     |                             |           |     |     +-- [GRAPH-OP LEAF] MLP output [tokens,4096]
    |     |                             |           |     |
    |     |                             |           |     +-- [INACTIVE] packed StackedMLP.__call__ branch
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
```

<details>
<summary><strong>What is MEF? What does weight/device mapping mean? What does seed kernel declarations from KernelLibrary mean?</strong></summary>

## 1. What Is MEF?

**MEF** means **Modular Executable Format**. It is MAX’s serialized compiled-model artifact format.

```text
Llama MLIR graph
    |
    +-- graph compiler
    +-- optimization
    +-- kernel selection
    +-- CPU/GPU code generation
    |
    v
CompiledModel
    |
    +-- export_mef(...)
    |
    v
llama3-<signature>.mef
```

A MEF can contain:

```text
compiled graph structure
selected kernels
host executable code
GPU binaries such as CUDA cubins
input/output signatures
device-target information
model names and initialization metadata
```

External Llama checkpoint weights are normally not embedded by this path. They are supplied later through `weights_registry`.

`compile_reusing_mefs()` supports three paths:

```text
No MEF store
    -> compile the Llama graph normally

Import/precompiled mode
    -> match graph name and input/output signature
    -> load an existing .mef
    -> avoid graph recompilation

Export mode
    -> compile the graph
    -> export the result as .mef
    -> record it in manifest.json
```

MEFs reduce subsequent model startup compilation time, but compiled MEFs are generally target-platform-specific.

## 2. Expected Weight/Device Mapping

The compiler records:

```python
expected_weights = {
    name: weight.value.device
    for name, weight in model._weights.items()
}
```

This produces a mapping such as:

```text
layers.0.self_attn.q_proj.weight -> cpu:0
layers.0.self_attn.k_proj.weight -> cpu:0
layers.0.self_attn.v_proj.weight -> cpu:0
layers.0.mlp.gate_proj.weight    -> cpu:0
model.embed_tokens.weight        -> cpu:0
```

It means:

> For every external weight name referenced by the graph, remember whether the supplied DLPack weight is expected to be host-resident or accelerator-resident.

During initialization:

```text
CompiledModel
    |
    +-- expected weight names/devices
    |
    +-- weights_registry
          name -> actual DLPack array
    |
    v
validate
    +-- is every expected weight present?
    +-- is its CPU-versus-accelerator device type correct?
    +-- is the array contiguous?
```

For example:

```text
Graph expects:
    q_proj.weight -> CPU

weights_registry supplies:
    q_proj.weight -> CPU BF16 array

Result:
    accepted
```

This mapping is not necessarily the final tensor-parallel placement:

```text
checkpoint Wq on CPU
    |
    +-- graph initialization/sharding
    |
    +-- Wq0 allocation on GPU0
    +-- Wq1 allocation on GPU1
```

The graph’s device operations determine final GPU placement. The captured mapping mainly validates external weight inputs at initialization. The current validation distinguishes CPU from non-CPU; it does not enforce an exact GPU ordinal.

## 3. What Is `KernelLibrary`?

`KernelLibrary` is a graph-level registry and resolver for kernels and custom operations available to the graph.

```text
KernelLibrary
    |
    +-- built-in MAX kernel packages
    +-- compiled Mojo .mojoc packages
    +-- optional Mojo source packages
    +-- registered kernel symbol names
    +-- kernel input/output declarations
    +-- shape functions
    +-- opaque-type mappings
    +-- custom-operation verification
```

For a Llama graph, it can resolve kernels used for operations such as fused projections, normalization, paged attention, KV-cache operations and collectives.

The exact selected kernels depend on device, dtype, quantization and compiler decisions.

## 4. What Are Kernel Declarations?

A kernel declaration describes how the compiler may call a kernel:

```text
kernel symbol/name
input types
output types
compile-time parameters
device requirements
shape function
linkage/package information
```

Conceptually:

```text
declare kernel @paged_attention(
    Q,
    KV-cache,
    row_offsets,
    attention_parameters
) -> attention_output
```

It is comparable to a function prototype. It identifies the callable kernel and its interface; it is not the complete kernel implementation.

The implementation resides in a built-in package, `.mojoc` package or another registered kernel library.

## 5. What Does “Seed Kernel Declarations” Mean?

“Seed” means **prepopulate the MLIR module with the declarations known by the graph’s `KernelLibrary`**:

```text
Graph's KernelLibrary
    |
    +-- kernel declarations
    +-- opaque-type mapping
    |
    | seed_kernel_decls(...)
    v
Llama MLIR module
    |
    +-- declarations inserted before compilation
    |
    v
graph compiler can resolve referenced kernel symbols
```

This does not execute or compile the kernels. It makes their names and interfaces available to the compiler before lowering begins.

It also avoids repeatedly reloading and analyzing kernel packages for every graph compilation:

```text
KernelLibrary construction
    -> load/analyze packages once

compile Llama graph
    -> copy required declarations into module
    -> compiler skips repeated package-import work
```

</details>

```text
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
```

<details>
<summary><strong>What is this batch processor?</strong></summary>

A **batch processor** converts the requests selected by the scheduler into the exact buffers expected by the compiled model graph.

It does not choose which requests enter the batch. The scheduler does that.

```text
Scheduler
    |
    +-- selects request contexts
    |
    v
Llama3BatchProcessor
    |
    +-- combine tokens
    +-- create row offsets
    +-- attach KV-cache inputs
    +-- attach signal buffers
    +-- create Llama3Inputs
    |
    v
compiled Llama graph
    |
    v
raw output buffers
    |
    +-- Llama3BatchProcessor.process_outputs(...)
    |
    v
ModelOutputs
```

## Llama Ragged-Batch Example

Suppose the scheduler selects two requests:

```text
Request A active tokens: [101, 102, 103]
Request B active tokens: [201, 202]
```

`Llama3BatchProcessor` creates:

```text
tokens:
    [101, 102, 103, 201, 202]

input_row_offsets:
    [0, 3, 5]
```

The offsets describe the sequence boundaries:

```text
Request A = tokens[0:3]
Request B = tokens[3:5]
```

Therefore no padding is required:

```text
Padded representation:
    A = [101, 102, 103]
    B = [201, 202, PAD]

Ragged representation:
    tokens  = [101, 102, 103, 201, 202]
    offsets = [0, 3, 5]
```

It then constructs:

```text
Llama3Inputs
    |
    +-- tokens
    +-- input_row_offsets
    +-- return_n_logits
    +-- signal_buffers
    +-- KV-cache input buffers
    +-- optional data_parallel_splits
```

These become positional graph inputs:

```python
self.model.execute(*model_inputs.buffers)
```

## Main Responsibilities

`Llama3BatchProcessor` handles:

```text
Input processing
    +-- flatten contexts from replicas
    +-- concatenate active token windows
    +-- calculate cumulative row offsets
    +-- stage host token data into device buffers
    +-- attach paged KV-cache metadata
    +-- attach multi-GPU allreduce signal buffers
    +-- calculate data-parallel splits
    +-- construct Llama3Inputs

Output processing
    +-- interpret the raw model output buffers
    +-- extract logits
    +-- extract optional hidden states
    +-- process KV-related outputs
    +-- return ModelOutputs
```

It also supplies symbolic input types while building the graph:

```text
Llama3BatchProcessor.get_symbolic_inputs(...)
    |
    +-- tokens TensorType
    +-- input_row_offsets TensorType
    +-- return_n_logits TensorType
    +-- signal BufferType values for multiple GPUs
    +-- symbolic KV-cache inputs
```

This keeps graph construction and runtime input preparation consistent.

## Meaning of `_wire_batch_processor()`

The batch processor is constructed earlier in `PipelineModel.__init__()`. `_wire_batch_processor()` performs optional post-compilation wiring:

```text
_wire_batch_processor(...)
    |
    +-- batch_processor is None
    |     -> architecture has no batch processor
    |
    +-- no bind_ep_comm_initializer method
    |     -> normal processor needs no late wiring
    |
    +-- method exists
          -> provide the EP communication initializer
```

**EP** means **Expert Parallelism**, used by Mixture-of-Experts models. Its communication buffers may only become available during distributed model initialization.

For ordinary dense Llama 3.1 8B:

```text
batch processor = Llama3BatchProcessor
bind_ep_comm_initializer = absent
result = no additional wiring required
```

Llama 3.1 8B is not an MoE model, so this specific `_wire_batch_processor()` call effectively returns without binding EP resources. The batch processor still performs its normal token, row-offset, KV-cache and output-processing work for every inference batch.

</details>

```text
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
