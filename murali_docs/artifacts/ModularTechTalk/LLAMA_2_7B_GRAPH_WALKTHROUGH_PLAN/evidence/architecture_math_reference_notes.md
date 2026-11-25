# Llama 2 7B architecture and mathematics: reference reconciliation

Date: November 25, 2026. Model: `meta-llama/Llama-2-7b-hf`.
Implementation revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Evidence: **SOURCE** for model contracts and operation ordering. **CPU-RUN**
applies only to the deterministic miniature JavaScript model and document
validation. Payload, traffic and roofline numbers are analytical teaching
models. No MAX inference, accelerator measurement or quantized deployment was
performed for this change.

## User-provided references

The public URLs returned HTTP 404 through an independent raw-content request;
the web reader could not fetch them. The user's local copies were available
under `/prj/adsp-platform/mur/Murali/aisw/Qgen/QNN/LLM/` and were reviewed.
Their stage-table organization and row-by-row calculation approach informed
the new pages. Their hardware assignments, performance claims and conflicting
model details were not imported as MAX evidence.

| File / supplied location | Local bytes | Local SHA-256 |
|---|---:|---|
| `01_zA02_Llama_2_7B_prefill_summary.md` — `https://github.com/Murali806/aisw/blob/main/Qgen/QNN/LLM/01_zA02_Llama_2_7B_prefill_summary.md` | 13,761 | `c4052dc3a12c954f84be8126601d9ec192d63e00aee7e5347c01b4726c5aee55` |
| `000_LLM_Llama7B_HTP_Prefill_Summary.txt` — `https://github.com/Murali806/aisw/blob/main/Qgen/QNN/LLM/000_LLM_Llama7B_HTP_Prefill_Summary.txt` | 40,014 | `f781e1d3ff254fddaa7a425b9c98a7c073d4e1e5afe8005cde782ff635e0da3a` |

## Source-grounded choices

| Topic | New walkthrough |
|---|---|
| Hardware | MAX's pinned source route; no HVX/HMX/VTCM placement inferred from Qualcomm notes |
| Checkpoint dimensions | 32 layers; 4096 hidden; 11008 MLP; 32 Q and 32 KV heads; head dimension 128; vocabulary 32000 |
| Encoding | Two-byte examples are compatible-encoding arithmetic. Saved FP16 metadata is not proof of accepted runtime encoding or automatic conversion |
| Weights | Embedding and LM head are independent; `tie_word_embeddings=false` |
| Norm order | `LAST_TOKEN` gathers last active rows before final RMSNorm and head |
| MLP | `SiLU(gate) * up`; three separate semantic matrices, stackable gate/up execution |
| KV | Retain rotated K and V per layer. Each new processed token writes all layers; sampling alone writes no KV |
| Attention matrix | Mathematical S/P shapes; fused MAX attention does not require their full materialization in device memory |
| Sampling | Separate sampler graph, model device unless `sample_on_host`; miniature model uses explicitly chosen greedy argmax |
| First generated token | Produced by prefill's final logits and sampling. It becomes the next forward's input. A separate decode is not required merely to sample it |
| Bias | These projections have no bias in this model. A general per-vocabulary bias would not be a uniform softmax shift, so no such rationale is repeated |
| Units | KiB/MiB/GiB for binary payload sizes; decimal GB/s and TFLOP/s; FLOPs are a work count, not a rate |

Source links resolve to complete offline files in the existing source explorer:
`llama3.py`, `model_config.py`, `attention_with_rope.py`, `transformer.py`,
`linear.py`, `cache_params.py` and `text_generation.py`. Source hashes remain
checked against the pinned snapshot by the existing validator.

Saved model metadata is in `llama2_7b_public_mirror_config.json` and
`llama2_7b_memory_planning_provenance.json`. The public mirror is
`NousResearch/Llama-2-7b-hf` at
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`; it does not substitute for a
successful original serving command.

For the online softmax mechanism, the primary research reference reviewed was
Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness*, arXiv `2205.14135v2`. The demonstration maintains row maximum,
normalization sum and a vector weighted-value accumulator; the pinned MAX
source establishes the serving operation, not a presumed identical kernel
schedule or tile size.

## Independent reference arithmetic

One MAC means one multiply–accumulate; conventional contraction FLOPs are
twice the MAC count. Elementwise math, reductions, exponentials, addressing,
casts and sampling have separate operation descriptions.

For one initial 512-token prefill:

```text
Q/K/V/O, each:       512 × 4096 × 4096 = 8,589,934,592 MACs
gate/up/down, each:  512 × 4096 × 11008 = 23,085,449,216 MACs
causal key pairs:    512 × 513 / 2 = 131,328
QK or PV, useful:    131,328 × 32 × 128 = 537,919,488 MACs
per layer:          4×8,589,934,592 + 3×23,085,449,216 + 2×537,919,488
                  = 104,691,924,992 MACs
last-row head:       4096 × 32000 = 131,072,000 MACs
all contractions:   32×104,691,924,992 + 131,072,000
                  = 3,350,272,671,744 MACs
                  = 6,700,545,343,488 conventional FLOPs
full rectangle:     3,384,565,301,248 MACs (masked/padded work convention)
```

For one decode with 512 previously processed positions:

```text
current active rows = 1; visible keys/values = 513
per-layer dense contractions = 4×4096² + 3×4096×11008 = 202,375,168 MACs
QK or PV = 513×4096 = 2,101,248 MACs
total = 32×(202,375,168 + 2×2,101,248) + 131,072,000
      = 6,741,557,248 MACs
```

At two-byte weights/activations/cache:

```text
parameters = 6,738,415,616; payload = 13,476,831,232 bytes
one Q/K/V/O matrix = 32 MiB; one gate/up/down = 86 MiB
embedding = 250 MiB; independent head = 250 MiB; one norm scale = 8 KiB
all-layer K+V per token = 2×32×32×128×2 = 524,288 bytes = 512 KiB
512 positions = 256 MiB valid KV; 513 = 256.5 MiB valid KV
P=128: 4 pages = 256 MiB; 5 pages = 320 MiB; unused at 513 = 63.5 MiB
```

The calculator names its transfer boundary, exposes a KV-read multiplier,
includes writes, and keeps payload storage distinct from bytes transferred.
Its restricted roofline is an editable analytical comparison, not a device
benchmark. Packed-byte controls omit quantization metadata and conversion
costs and do not assert encoding support. The two reference notes' approximate
7.4-T prefill total, hardware timing numbers and memory placements are not used
as numerical or runtime validation.
