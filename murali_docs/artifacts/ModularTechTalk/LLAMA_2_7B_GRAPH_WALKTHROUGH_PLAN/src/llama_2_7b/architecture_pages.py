"""One-screen stage ledger, focused prefill/decode pages, and the math lab."""
import html
import json
import subprocess
from functools import lru_cache

from catalog import HERE, OUT, resolve
from architecture_diagram import model_diagram

ASSETS = ('architecture.css', 'architecture_model.js', 'architecture_costs.js',
          'architecture_tensor.js', 'architecture_math.js')


@lru_cache(maxsize=1)
def reports():
    script = "const m=require('./architecture_model.js');console.log(JSON.stringify([m.calculate(),m.calculate({T:1,C:512})]));"
    return json.loads(subprocess.check_output(['node', '-e', script], cwd=HERE))


def esc(text):
    return html.escape(str(text))


def size(n):
    for unit, factor in [('GiB', 2**30), ('MiB', 2**20), ('KiB', 2**10)]:
        if n >= factor:
            return f'{n/factor:,.3f}'.rstrip('0').rstrip('.') + ' ' + unit
    return f'{n:,} B'


def count(n):
    if n >= 10**12: return f'{n/10**12:.4f} T'
    if n >= 10**9: return f'{n/10**9:.4f} G'
    if n >= 10**6: return f'{n/10**6:.4f} M'
    return f'{n:,}'


def navigation(current='architecture'):
    links = [('architecture','Overview'), ('architecture_01_prefill','01 · Prefill + costs'),
             ('architecture_02_decode','02 · Decode + costs'), ('architecture_math','03 · Animate the math')]
    return '<nav class="architecture-nav" aria-label="Architecture views">' + ''.join(
        f'<a href="{slug}.html" {"aria-current=page" if slug == current else ""}>{title}</a>'
        for slug, title in links) + '</nav>'


def source_refs(sources, refs_html):
    specs = [('llama','embedding_layer = Embedding('), ('attention','qkv = self.qkv_proj(x)'),
             ('attention','xq = rope_split_store_ragged('), ('attention','attn_out = flash_attention_ragged('),
             ('transformer','h = x + attn_out'), ('linear','gate_out = self.activation_function(gate_out)'),
             ('transformer','last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)'),
             ('pipeline','self._sampler_device: Device = ('), ('kv','PACKED_PAGE_STRIDE = -1')]
    return refs_html([resolve(sources, alias, needle) for alias, needle in specs])


def ledger(report, mode='prefill'):
    rows = []
    destination = f'architecture_0{1 if mode == "prefill" else 2}_{mode}.html'
    for row in report['rows']:
        if row['id'] == 'attention': continue  # This is a cost-accounting group, not another model operation.
        if row['id'] == 'norm1':
            rows.append('<tr class="layer-band"><th colspan="4">Repeat stages 02–18 × 32 decoder layers · independent weights and KV slice per layer</th></tr>')
        if row['id'] == 'gather':
            rows.append('<tr class="layer-band"><th colspan="4">End of decoder stack · LAST_TOKEN output path</th></tr>')
        weight = f'<small>Resident weight payload: {size(row["weight"])}.</small>' if row['weight'] else ''
        rows.append(f'<tr><th scope="row"><a href="{destination}#cost-{row["id"]}" title="Calculate this stage">{row["num"]} {esc(row["title"])}</a></th>'
                    f'<td><code>{esc(row["ins"])}</code></td><td><code>{esc(row["out"])}</code></td>'
                    f'<td>{esc(row["operation"])}{weight}</td></tr>')
    return '<div class="table-scroll"><table class="architecture-ledger"><caption>Every stage in one pass · select a stage for its calculation</caption><thead><tr><th>Stage</th><th>Input shape</th><th>Output shape</th><th>Key weight / mathematical operation</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def intro(mode):
    if mode == 'decode':
        return '''<div class="architecture-context"><b>One request · first decode after a 512-token prefill</b><br>
<code>T=1</code> new input ID · <code>C=512</code> cached positions · <code>L=C+T=513</code> visible keys/values.
Every projection processes one new row; attention reads the retained prefix. Prefill already sampled the first generated token.</div>'''
    return '''<div class="architecture-context"><b>One request · 512-token initial prefill</b><br>
<code>T=512</code> active token IDs · <code>C=0</code> cached positions · <code>L=C+T=512</code> keys/values.
<code>D=4096</code> · <code>F=11008</code> · <code>32 heads × 128</code> · <code>32 layers</code> · <code>32,000 vocabulary entries</code>.</div>'''


def compatibility():
    return '''<p class="architecture-note"><b>Payload convention:</b> table sizes use 2 bytes/value (BF16-compatible arithmetic); logits are FP32.
Weights are untied, and SiLU applies to the gate. Hardware scheduling follows the MAX source links below.
The saved checkpoint says FP16; this pinned architecture’s accepted encodings omit FP16. This is the existing compatible-encoding teaching route.
<a href="build_startup.html#q2">Encoding gate</a>.</p>'''


def cache_explainer(mode):
    now = '512 → 513' if mode == 'decode' else '0 → 512'
    detail = '''<b>First decode:</b> input is the token sampled by prefill. Each layer writes position 512;
K/V then span 513 positions. K+V grow by <b>512 KiB across all layers</b>. With P=128, a fifth page
raises assigned payload from 256 to <b>320 MiB</b>: 256.5 MiB used, 63.5 MiB empty.''' if mode == 'decode' else '''<b>Initial prefill:</b> write positions 0…511 at all 32 layers. Used K+V =
<code>2 × 32 × 512 × 32 × 128 × 2 = 268,435,456 bytes = 256 MiB</code>.
Each layer contributes 8 MiB. At page size P=128, four pages are exactly full.'''
    return f'''<section class="kv-explanation" id="kv-in-one-picture"><h2>KV cache: what survives this pass?</h2>
<div class="kv-ownership"><article><b>Model lifetime · weights</b><p>E, 32 sets of decoder matrices/norms, untied W_head. Same weights for prefill and every decode.</p></article>
<article><b>Request lifetime · K′ and V</b><p>32 distinct layer slices. Retain each processed position’s rotated key and unrotated value. Length {now}.</p></article>
<article><b>Current pass · activations</b><p>Q, residuals, attention results and FFN values change each call. Full scores/probabilities are a mathematical view of fused attention.</p></article></div>
<pre class="math-flow">current token IDs → embedding → layer 0 → … → layer 31 → last row → norm → head
                                      │                                    │
                        K′, V → cache[layer, position]                     ↓
                            retained for later queries               sampled token ID
                            Q′ reads all causal keys                       │
                                                               NEXT forward creates its KV</pre>
<p>{detail}</p><p><b>Why cache?</b> Reusing old K/V avoids projecting the whole prefix again.
For current query i, allowed key positions satisfy <code>j ≤ C+i</code>. The new token attends to its own K/V too.</p>
<details><summary>Physical pages, payload arithmetic and lifetime</summary><div class="section-body">
<p>Logical shape per request/layer: <code>K,V each [L,32,128]</code>. Ordinary physical backing:
<code>[physical_pages, 2, 32 layers, P, 32 KV heads, 128]</code>.</p>
<pre>logical page = position // P          slot = position % P
physical page = lookup[request, logical page]
within-page element offset = ((((kv × 32 + layer) × P + slot) × 32 + head) × 128 + channel)
byte offset = (physical page × page_stride_elements + within-page offset) × dtype_bytes</pre>
<p>Here <code>kv=0</code> selects K, <code>kv=1</code> selects V. CPU stride sentinel <code>−1</code>
means derive the packed stride; explicit strides can include padding. A logical sequence can occupy nonadjacent physical pages.</p>
<table><thead><tr><th>At 2 bytes/value</th><th>Calculation</th><th>Payload</th></tr></thead><tbody>
<tr><td>One K or V vector, all heads</td><td>32 × 128 × 2</td><td>8 KiB / token / layer</td></tr>
<tr><td>K+V across all layers</td><td>2 × 32 layers × 8 KiB</td><td>512 KiB / token</td></tr>
<tr><td>One 128-token page, all layers</td><td>128 × 512 KiB</td><td>64 MiB</td></tr>
<tr><td>4096 processed positions</td><td>4096 × 512 KiB</td><td>2 GiB / request</td></tr></tbody></table>
<p>Assigned pages, valid entries, newly written bytes and bytes read are separate quantities. The manager may reserve a larger pool.
No prefix sharing, offload, eviction, page padding or quantization metadata is included in these examples.
<a href="kv_cache.html">Explore cache ownership and exact offsets →</a></p></div></details></section>'''


def overview(report, sources, refs_html):
    return (navigation() + '<section id="architecture-at-a-glance"><h2>The full architecture, in one pass</h2>' +
            intro('prefill') + ledger(report) +
            '<p class="architecture-note">The table shows semantic stages. MAX can stack Q/K/V and gate/up projections; '
            'RoPE, split and KV store are fused on this route. Paged attention consumes tiles without requiring the full score tensor in device memory.</p>' +
            compatibility() + cache_explainer('prefill') + source_refs(sources, refs_html) + '</section>')


def options(values, selected):
    return ''.join(f'<option value="{esc(v)}" {"selected" if v == selected else ""}>{esc(label)}</option>' for v,label in values)


def numeric(label, key, value, minimum, maximum, step=1):
    return f'<label>{label}<input id="setting-{key}" name="{key}" type="number" value="{value}" min="{minimum}" max="{maximum}" step="{step}" required></label>'


def select(label, key, values, selected):
    return f'<label>{label}<select id="setting-{key}" name="{key}">{options(values,selected)}</select></label>'


def cost_rows(report):
    body=''
    for r in report['rows']:
        traffic = size(r['traffic']) if r['traffic'] is not None else ('group 08–10' if r['id'] in ('qk','softmax','pv') else 'outside ledger')
        ai = f'{r["ai"]:.3f}' if r['ai'] is not None and r['mac'] else '—'
        body += f'<tr id="cost-{r["id"]}"><th scope="row"><button type="button" data-cost-stage="{r["id"]}">{r["num"]} {esc(r["title"])}</button></th><td>{r["repeat"]}</td><td>{count(r["mac"])}</td><td>{traffic}</td><td>{ai}</td><td>{size(r["weight"])}</td><td>{size(r["live"])}</td></tr>'
    return body


def calculator(report, mode):
    s=report['settings']
    fields = numeric('Active tokens / request · T','T',s['T'],1,4096) + numeric('Already cached / request · C','C',s['C'],0,4095) + numeric('Equal-length requests · B','B',1,1,64)
    fields += select('Matrix weight payload · bytes/value','w',[(2,'2 · BF16 example'),(4,'4 · FP32 example'),(1,'1 · hypothetical packed payload'),(0.5,'0.5 · hypothetical INT4 payload')],2)
    fields += select('Activations + norm scales · bytes/value','a',[(2,'2 bytes'),(4,'4 bytes')],2)
    fields += select('KV payload · bytes/value','k',[(2,'2 bytes'),(4,'4 bytes'),(1,'1 · hypothetical packed payload')],2)
    fields += select('Tokens / page · P','P',[(n,str(n)) for n in (16,32,64,128,256)],128)
    fields += select('Attention MAC convention','pairs',[('causal','Causal useful pairs'),('rectangle','Full T × L rectangle')],'causal')
    fields += select('Attention traffic convention','traffic',[('boundary','Fused boundary · no full S/P'),('materialized','Reference · FP32 S/P materialized')],'boundary')
    fields += numeric('KV read multiplier (per call)','kvReads',1,1,128)
    fields += numeric('Evaluation window · Δt (ms)','budgetMs',10,0.000001,1000000,'any')
    fields += numeric('Assumed memory BW (GB/s)','bandwidth',1000,0.000001,1000000,'any')
    fields += numeric('Assumed contraction rate (TFLOP/s)','tflops',100,0.000001,1000000,'any')
    return f'''<section class="cost-workbench" id="cost-calculator" data-cost-mode="{mode}"><h2>Calculate each stage</h2>
<p><b>Analytical model · editable assumptions.</b> Counts are per pass; stage rows 02–18 are for <b>one layer</b>.
The total applies ×32. Each MAC is one multiply–accumulate; conventional contraction FLOPs = 2 × MACs.
The MAC column counts matrix products; scalar operations are listed separately for each stage.
The architecture table above is the named example; changing inputs here updates this calculator and the selected stage’s shapes.</p>
<form id="cost-form"><div class="cost-controls">{fields}</div><div class="cost-presets"><button type="button" data-cost-preset="prefill">512-token prefill</button><button type="button" data-cost-preset="decode">Decode at C=512</button><button type="button" data-cost-preset="chunk">Chunk: C=384, T=128</button><button type="button" id="cost-reset">Reset all assumptions</button><button type="button" id="cost-export">Export exact ledger (JSON)</button></div></form>
<p id="cost-error" role="alert" hidden></p><p class="architecture-note" id="cost-assumptions"></p>
<div id="cost-summary" class="metric-grid" aria-live="polite"><article><b>Contraction MACs</b><strong>{count(report['totals']['mac'])}</strong></article><article><b>Used KV payload</b><strong>{size(report['memory']['usedKV'])}</strong></article></div>
<section id="cost-inspector" class="cost-inspector"><label><b>Show the calculation for</b> <select id="cost-stage">{options([(r['id'],r['num']+' '+r['title']) for r in report['rows']],'q')}</select></label><div id="cost-formula" aria-live="polite"></div></section>
<div class="table-scroll"><table class="cost-table"><caption>Per-stage ledger · click a stage to see substituted formulas, bandwidth and intensity</caption><thead><tr><th>Stage</th><th>Repeat</th><th>Contraction MACs / occurrence</th><th>Read + write bytes / occurrence</th><th>2×MAC / byte</th><th>Resident weight</th><th>Named live activations</th></tr></thead><tbody id="cost-rows">{cost_rows(report)}</tbody></table></div>
<section id="cost-memory"><h3>Memory buffers and KV pages</h3><div id="cost-memory-result"></div></section>
<noscript><p>The table above is the default {'512-token prefill' if mode=='prefill' else 'C=512, T=1 decode'} calculation. Enable JavaScript for interactive assumptions and per-stage substitution. The formulas and complete counting conventions follow below.</p></noscript></section>'''


def methodology(md):
    return '<details class="architecture-methods" id="counting-conventions" open><summary>Read the numbers correctly · formulas, traffic boundary and memory scope</summary><div class="section-body">'+md('''
`T` = active rows per request; `C` = previous cached positions; `L=C+T`; `B` = equal-length requests; `R=B×T`.
`D=4096`, `F=11008`, `H=32`, `d=128`, `N=32`, `V=32000`. `w/a/k` are bytes per matrix weight / activation (and norm scale) / cache element.

| Quantity | Exact equation in this calculator |
|---|---|
| Dense projection `[M,K] × [K,N]` | `M×K×N` MACs; `2×M×K×N` conventional FLOPs |
| One attention projection Q/K/V/O | `R×D²` MACs |
| One MLP matrix gate/up/down | `R×D×F` MACs |
| Allowed query–key pairs | `B×[T×C + T(T+1)/2]` |
| Full rectangular pairs | `B×T×(C+T)` |
| QK or PV | `pairs×H×d`; causal useful count omits masked pairs |
| Whole pass, contractions | `32×[4R D² + 3R D F + 2×pairs×H×d] + B D V` MACs |
| Dense projection traffic | `M K a + K N w + M N a` bytes (input + weight reads + output write) |
| Arithmetic intensity | `2×MAC / (read_bytes + write_bytes)`; contraction-only numerator |
| Required average bandwidth | `bytes / (Δt_ms × 10⁶)` in **GB/s**, for the selected occurrence OR the whole pass |
| Ideal contraction time | `2×MAC / (assumed_TFLOP/s × 10⁹)` ms |
| Ideal transfer time | `bytes / (assumed_GB/s × 10⁶)` ms |
| Restricted roofline bound | `max(ideal contraction time, ideal transfer time)`; not a measured latency |
| Weights | `[2VD + 32(4D² + 3DF)]w + 65Da` bytes |
| Used KV | `2×32×B×L×H×d×k` bytes |
| KV newly written | `2×32×B×T×H×d×k` bytes |
| Assigned KV | `B×ceil(L/P)×P×(2×32×H×d×k)` bytes, no shared prefixes |

**Two attention traffic views.** The reference view materializes FP32 scores and probabilities: each is `B×H×T×L×4` bytes, including masked positions. QK writes S; softmax reads S and writes P; PV reads P. The fused boundary view accounts once for Q, K/V reads and the context write in a separate **08–10 group**. Individual QK/softmax/PV byte and intensity cells then point to that group. The group repeats the MACs for inspection but does **not** add them again to the total.

**Traffic is an explicit model of device-memory transfers.** Dense stages read each operand once per stage and write their outputs once. Weights are reused across the `B×T` rows of that stage. Projection rows are semantically separate: actual stacked QKV and gate/up can reuse input reads. RoPE/store, norms, residuals and casts can fuse or alias. Cache hierarchy, tiling, layouts and dispatch can also add transfers. The fused attention boundary assumes one complete K/V read by default; its multiplier illustrates rescanning. No measured HBM/DDR traffic or guaranteed overall IO lower bound is claimed. CPU tokenization, sampler policy, host transfers and metadata are outside the traffic subtotal. Norms and other scalar stages still move bytes even when their **contraction MACs are zero**; their scalar work is listed in the inspector.

**Intensity depends on the boundary.** Weight-only GEMM intensity is `2×R/w` FLOP/byte (about `4R` for ideal INT4 payload). Including activations, output writes and KV changes the denominator. Hypothetical byte-size controls change payload arithmetic only: they do not establish a MAX-supported encoding, dequantization cost, quality, scales or zero-point storage. Norm scales use `a`, not the low-bit matrix setting. A zero contraction intensity for normalization/softmax is not zero arithmetic.

**Time inputs are illustrative, not device specifications.** 100 TFLOP/s, 1000 GB/s and 10 ms are editable teaching values. A time window applied to each row is an independent question, not an allocation of that same window across 32 layers. Total bandwidth uses total bytes divided by the whole-pass window. Real rooflines need the actual dtype’s achievable compute and memory rates; scalar math, launch/synchronization, incomplete utilization and overlap require measurement. FLOPs are a work count; FLOP/s is a rate.

**Memory is not traffic.** Resident weights count each parameter once. Embedding traffic reads selected rows, although the entire embedding table occupies storage. Used KV, assigned page capacity and an allocator’s larger reserved pool are different. Named live activations use an explicit out-of-place mathematical schedule: residuals stay live for their adds; Q/K/V coexist before cache storage; the FFN can hold `H+G+U+Z`. The largest row is a teaching working set, not a proven peak allocation. Weights + assigned KV + that maximum is a **modeled storage subtotal**; it excludes the allocator pool, metadata, full RoPE tables, quantization metadata, compiler/runtime objects, backend workspace and on-chip tiles. Native lifetimes/aliasing can alter it. Do not sum per-stage working sets into peak memory.

**Ragged serving.** Equal lengths make the interactive equations readable. For unequal requests, replace `R` with `ΣTᵢ`, pairs with `Σ[TᵢCᵢ + Tᵢ(Tᵢ+1)/2]`, assigned pages with `Σceil((Cᵢ+Tᵢ)/P)`, and head rows with the number of requests. The model itself uses ragged row offsets.
''')+'</div></details>'


def math_shell():
    return '''<section class="math-workbench" id="math-animation" aria-label="Actual-dimension Llama tensor and KV animation">
<div class="scene-top"><button class="lesson-badge" id="math-badge" aria-label="Explain the tensor explorer">03</button><strong>Llama 2 7B · actual tensor dimensions</strong><span class="scene-top-right"><button id="math-fullscreen">Full screen</button><button class="legend" id="math-legend" aria-label="Buffer legend">L</button></span></div>
<p class="math-scope"><b>SOURCE dimensions · CPU-RUN operation probes.</b> 32 layers · 4096 hidden · 32 heads × 128 · 11008 FFN · 32000 vocabulary. Select any coordinate; only a small viewport is drawn. Values are deterministic generated operands in browser FP64. Each operation is a separate numerical probe; RoPE, KV, scores, softmax and P·V share one consistent attention example. This is a math explorer, not a checkpoint forward.</p>
<div class="math-transport"><div class="controls"><button id="math-reset">Reset</button><button id="math-back">← Back</button><button id="math-play" aria-pressed="false">Play</button><button id="math-step">Step →</button><button id="math-finish">Finish operation</button><button id="math-next">Next stage ⇥</button><label>Pass <select id="math-pass"><option value="prefill">Prefill · 512 tokens</option><option value="decode">Decode · 512 → 513</option><option value="next">Decode · 513 → 514</option></select></label></div>
<div class="controls"><label>Stage <select id="math-stage"></select></label><label>Layer <select id="math-layer" aria-label="Decoder layer"></select></label><label>Head <select id="math-head" aria-label="Attention head"></select></label><label>Step size <select id="math-chunk"><option value="128">128 terms / slots</option><option value="32">32 terms / slots</option><option value="1">1 term / slot</option><option value="512">512 terms / slots</option></select></label><label>Speed <select id="math-speed"><option value="1100">Normal</option><option value="400">Fast</option><option value="2200">Slow</option></select></label></div></div>
''' + model_diagram() + '''
<p class="math-layer-caption">32 decoder layers · small numbers show valid KV positions per layer</p><div class="math-layer-rail" id="math-layers" aria-label="All 32 decoder layer slices"></div>
<a class="model-return" href="#model-overview">↑ Full model diagram</a><div class="event" id="math-event" aria-live="polite"><strong>Explore a full-size operation.</strong><p>Choose a stage, then use Step to follow the reduction. Finish operation jumps to its result.</p></div>
<div class="math-pipeline" id="math-pipeline" aria-label="Select a mathematical operation"></div>
<div class="math-shape-equation" id="math-shape-equation"></div>
<div class="controls matrix-choices"><label>Result row <input type="number" id="math-row" min="0" value="0" aria-label="Result row"></label><label>Result column <input type="number" id="math-col" min="0" value="0" aria-label="Result column"></label><button id="math-last-row">Last row</button><button id="math-last-col">Last column</button><label id="math-rope-control" hidden>Rotate <select id="math-rope-kind"><option value="Q">Q</option><option value="K">K</option></select></label><span id="math-selection"></span></div>
<div class="math-matrices" id="math-matrices"><article id="math-left"></article><b class="math-operator" id="math-operator">×</b><article id="math-right"></article><b class="math-operator">→</b><article id="math-result"></article></div>
<div class="math-reduction"><div class="reduction-heading"><b id="math-reduction-title">Reduction</b><output id="math-progress"></output></div><input type="range" min="0" max="1" value="0" id="math-seek" aria-label="Terms or phases completed"><div id="math-reduction-bands" class="reduction-bands"></div><small id="math-reduction-note"></small></div>
<div class="math-equation" id="math-equation" aria-live="polite"></div>
<div id="math-cost-link" class="math-cost-link"></div>
<details class="math-buffers-detail"><summary>Buffer ownership · what survives this operation?</summary><div id="math-buffer-shelf" class="math-buffer-shelf"></div></details>
<section class="math-cache"><div class="cache-heading"><h3>KV memory · one request, all 32 layer slices</h3><label>Inspect absolute position <input type="number" min="0" max="511" value="0" id="math-cache-position"></label><label>Channel <input type="number" min="0" max="127" value="0" id="math-cache-channel"></label></div><p id="math-cache-state"></p><div id="math-cache-pages" class="full-cache-pages"></div><div id="math-cache-vector" class="cache-vector"></div><details><summary>Resolve this position to its physical K/V byte addresses</summary><div id="math-cache-address" class="math-equation"></div></details></section>
<details id="math-online"><summary>Attention without a full score buffer · 128-key online tiles</summary><div class="online-body"><p>Same generated Q/K/V, selected layer and head. Accumulate all 128 output channels with a running maximum, denominator and numerator. Tile size 128 is an illustrative teaching schedule, not a measured kernel configuration.</p><div class="controls"><label>Query row <input id="online-query" type="number" min="0" max="511" value="511"></label><button id="online-reset">First tile</button><button id="online-back">← Previous tile</button><button id="online-step">Next tile →</button><output id="online-progress"></output></div><div id="online-result"></div></div></details>
<div id="math-sources"></div><noscript><p>The full-size sequence, equations and cache map are available in the transcript below. Enable JavaScript for coordinate selection and computed operation probes.</p></noscript></section>'''


def build_companions(nav, sources, header, footer, md, refs_html):
    prefill, decode = reports()
    refs=source_refs(sources,refs_html)
    for mode, report, number in [('prefill',prefill,'01'),('decode',decode,'02')]:
        slug=f'architecture_{number}_{mode}'
        title='Prefill: process 512 tokens, retain their K/V' if mode=='prefill' else 'Decode: process one token, reuse the prefix K/V'
        text=header(title,'Shapes first. Then exact MACs, explicit byte traffic, bandwidth assumptions and memory buffers.','architecture',nav)
        text=text.replace('</head>','<link rel="stylesheet" href="architecture.css"></head>')
        text+=navigation(slug)+intro(mode)+ledger(report,mode)+compatibility()+cache_explainer(mode)
        text+=calculator(report,mode)+methodology(md)+refs
        text+='<p><a href="../../evidence/architecture_math_reference_notes.md">Reference notes, corrections and provenance</a> · <a href="architecture_math.html">Animate the numerical operations →</a></p>'
        text+=footer()
        text=text.replace('</body>','<script src="architecture_model.js"></script><script src="architecture_costs.js"></script></body>')
        (OUT/(slug+'.html')).write_text(text)
    text=header('Animate the math: matrices, residuals and KV memory','Explore full-size tensor coordinates, complete reductions and the 512 → 513 token cache boundary.','architecture',nav)
    text=text.replace('</head>','<link rel="stylesheet" href="architecture.css"></head>')
    text+=navigation('architecture_math')+math_shell()
    text+='<section id="math-transcript"><h2>The mathematical sequence · portable / print</h2>'+md('''
```text
IDs → E[IDs] → X
for each layer:
  N = RMSNorm(X)
  Q = N W_Qᵀ; K = N W_Kᵀ; V = N W_Vᵀ
  reshape heads; (Q′,K′) = RoPE(Q,K, absolute positions)
  cache[layer,C:C+T] ← (K′,V)       old cache entries survive
  S = Q′ K′cacheᵀ / sqrt(head_dim)
  S[i,j] = −infinity when j > C+i  causal, within this request
  P = exp(S − max(S)) / sum(exp(S − max(S)))
  context = concat_heads(P Vcache)
  H = X + context W_Oᵀ
  N2 = RMSNorm(H)
  G = N2 W_gateᵀ; U = N2 W_upᵀ
  Z = SiLU(G) ⊙ U; X = H + Z W_downᵀ
last = gather_last(X) → RMSNorm → independent W_headᵀ → logits
sample ID → next pass input        sampling itself writes no K/V
```

**Actual sizes, bounded viewport.** Prefill uses T=512, C=0, L=512. Decode uses T=1, C=512, L=513; the following decode uses C=513, L=514. All 32 layers are selectable. Q/K/V have 32 heads of 128 channels. A viewport shows a few entries with their **absolute row and column indices**; the overview shows where that tile sits in the complete tensor. Click its overview to move, enter an exact coordinate, or choose the last row/column.

**Complete reductions.** Q/K/V/O and the gate/up/head projections reduce 4096 products for each selected output. The down projection reduces 11008. QKᵀ reduces 128, then scales by 1/√128. P·V reduces L value rows. Step reveals 1, 32, 128 or 512 terms; the seek control addresses every term, including the final partial tile. A running accumulator changes while the output cell remains pending until its reduction finishes. Other viewport cells are completed references. No discarded channels or rescaled miniature matrices are involved.

**Numerical scope.** Deterministically generated operands avoid downloading model weights. Each operation is a separate FP64 probe; dense-projection values are not propagated through a full 32-layer forward. The attention probes are connected: generated Q/K/V → RoPE → retained K′/V → scores → softmax → P·V. Their values agree across these stages and across pass boundaries for the same absolute positions. This proves the illustrated arithmetic, not Llama output parity. Model extents and operation order follow the pinned sources. Payload memory uses an explicit 2-byte assumption; browser computation is FP64.

**Norm and rotary detail.** RMSNorm uses all 4096 channels: yⱼ=xⱼγⱼ/√(Σx²/4096+1e−5). A safetensors split-half head pairs channel c with c+64, for c=0…63. Angle θ=position×10000^(−2c/128) gives (x cosθ−y sinθ, x sinθ+y cosθ). Q and K rotate; V does not. The pair inspector exposes both outputs and preserves pair norm. The absolute position is C+i.

**Attention and the mask.** For head h, [T,128] × [128,L] gives [T,L] scores. Mask key j when j>C+i, then compute exp(S−row maximum)/Σexp(S−row maximum). All 32 heads have their own scores and values. [T,L] × [L,128] gives the head context; merging maps hidden channel = h×128+c. S/P are mathematical matrices; fused attention need not allocate them in full. The online inspector uses m′=max(m,tile max), α=exp(m−m′), ℓ′=αℓ+Σexp(S−m′), u′=αu+Σexp(S−m′)V and output=u′/ℓ′, for all 128 channels. Fully masked tiles preserve the accumulated result.

**KV occupancy and lifetime.** The page grid shows all 128 slots of each illustrated physical page for the selected layer. Old entries remain READ/RETAINED; Step at KV store fills only new slots. Earlier layers can have longer valid prefixes while the current forward runs. The 32 layer indicators expose this; the minimum length advances only when all layers finish storing. Named scratch buffers are illustrative mathematical lifetimes, not a captured compiler allocation schedule. Sampling itself writes no K/V.

**Concrete page boundary.** P=128 is a chosen paging setting. Logical pages 0,1,2,3,4 map to illustrative physical IDs 3,1,5,2,7. Each page covers 128 positions, both K/V, all 32 layers, all 32 heads and 128 channels. Four pages hold 512 tokens. The first decode assigns a fifth page and writes position 512 to slot 0; the next writes position 513 to slot 1. IDs stay fixed across routes.
At 2 bytes/value: K+V=16 KiB/token/layer, 512 KiB/token across all layers, and 64 MiB/page. Completed prefill uses 256 MiB. Completed first decode uses 256.5 MiB in 320 MiB assigned capacity; the remaining 63.5 MiB is unused page capacity. This excludes any larger allocator pool and metadata.

```text
logical_page = absolute_position // 128
slot         = absolute_position % 128
physical     = lookup[logical_page]
backing      = [physical_pages, 2, 32, 128, 32, 128]
page_stride  = 2 × 32 × 128 × 32 × 128 = 33,554,432 elements
within_page  = ((((kv × 32 + layer) × 128 + slot) × 32 + head) × 128 + channel)
byte_address = (physical × page_stride + within_page) × 2
kv = 0 for rotated K; kv = 1 for V
```

''')+'</section>'+refs
    text+='<p><a href="architecture_01_prefill.html#cost-calculator">Real-size prefill calculations</a> · <a href="architecture_02_decode.html#cost-calculator">Real-size decode calculations</a> · <a href="attention_rope.html">Source-level attention and RoPE lesson</a></p>'
    text+='<script type="application/json" id="math-source-data">'+json.dumps({key:resolve(sources,*spec) for key,spec in {
        'embedding':('transformer','self.embed_tokens(tokens)'), 'norm':('norm','class RMSNorm'),
        'qkv':('attention','qkv = self.qkv_proj(x)'), 'rope':('attention','xq = rope_split_store_ragged('),
        'attention':('attention','attn_out = flash_attention_ragged('), 'residual':('transformer','h = x + attn_out'),
        'mlp':('linear','gate_out = self.activation_function(gate_out)'),
        'logits':('transformer','last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)'),
        'sampling':('pipeline','self._sampler_device: Device = (')}.items()},ensure_ascii=False).replace('</','<\\/')+'</script>'
    text+=footer()
    text=text.replace('</body>','<script src="architecture_tensor.js"></script><script src="architecture_math.js"></script></body>')
    (OUT/'architecture_math.html').write_text(text)
