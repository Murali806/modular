"""Small inline SVG: complete model, expanded decoder, and exact math-stage links."""
from html import escape


def model_diagram():
    parts = ['''<details class="model-overview" id="model-overview" open>
<summary>Full model block diagram · token IDs → 32 decoder layers → next token</summary>
<div class="model-map-toolbar"><p>Choose a block to open its calculation. <b id="model-map-context">Prefill · T=512 active, C=0 cached, L=512 total</b></p><a href="#math-event">Current calculation ↓</a></div>
<div class="model-map-scroll" tabindex="0" aria-label="Full model diagram; scroll horizontally on narrow screens">
<svg id="model-map" viewBox="0 0 1380 890" role="group" aria-labelledby="model-map-title model-map-description">
<title id="model-map-title">Llama 2 7B: full model and one expanded decoder block</title>
<desc id="model-map-description">Prompt tokenization feeds embeddings, 32 distinct decoder layers, last-row selection, final RMSNorm, an untied language-model head and sampling. The sampled ID returns to embedding on the next decode. Each layer has pre-norm attention, Q/K/V projections, RoPE on Q and K, retained K/V, causal softmax, a value mixture and an attention residual. Its pre-norm feed-forward branch uses SiLU on the gate, multiplication by the up branch, a down projection and a second residual.</desc>
<defs><marker id="model-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs>
<rect class="model-whole" x="16" y="12" width="1348" height="198" rx="9"/>
<text x="36" y="39" class="model-section-title">ONE REQUEST · LAST_TOKEN serving route</text>
<text x="1343" y="39" text-anchor="end" class="model-caption">D=4096 · 32 heads × 128 · FFN=11008 · vocab=32000</text>
<rect class="model-decoder-frame" x="16" y="233" width="1348" height="640" rx="9" data-model-group="decoder"/>
<text x="36" y="259" class="model-section-title" data-model-text="layer">EXPANDED DECODER · layer 0 of 0…31</text>
<text x="1343" y="259" text-anchor="end" class="model-caption">Repeat this structure 32 times · distinct weights and KV slice in each layer</text>
<rect class="model-attention-region" x="38" y="293" width="652" height="492" rx="8"/>
<rect class="model-ffn-region" x="775" y="293" width="560" height="492" rx="8"/>
<text x="96" y="317" class="model-region-title">ATTENTION</text><text x="791" y="317" class="model-region-title">FEED-FORWARD / SwiGLU</text>''']

    def label(x, y, text, key=None, cls='model-caption', anchor='middle'):
        dynamic = f' data-model-text="{key}"' if key else ''
        parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{dynamic}>{escape(text)}</text>')

    def wire(path, kind='', stage=None):
        active = f' data-model-edge="{stage}"' if stage else ''
        parts.append(f'<path class="model-wire {kind}" d="{path}" marker-end="url(#model-arrow)"{active}/>')

    def box(node, x, y, w, h, title, lines=(), stage=None, kind='math', group=None):
        attrs = f' data-model-stage="{stage}"' if stage else ''
        attrs += f' data-model-group="{group}"' if group else ''
        tag = 'a' if stage else 'g'
        href = f' href="#math-transcript" aria-label="Inspect {escape(title)}"' if stage else ''
        parts.append(f'<{tag} id="model-{node}" class="model-node model-{kind}"{attrs}{href}>')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>')
        label(x+w/2, y+22, title, cls='model-node-title')
        for i, line in enumerate(lines):
            text, key = line if isinstance(line, tuple) else (line, None)
            label(x+w/2, y+42+i*16, text, key, cls='model-node-detail')
        parts.append(f'</{tag}>')

    # Main route. The decode feedback enters embedding directly, after tokenization.
    for left, right in [(176,210),(355,390),(550,585),(720,755),(885,920),(1090,1125)]:
        wire(f'M{left} 103 H{right}')
    wire('M1222 142 V183 H282 V142', 'model-feedback', 'sample')
    label(755,176,'NEXT DECODE · sampled ID → embedding; append its K/V during the next forward')
    wire('M470 142 V173 M470 193 V222 H700 V233', 'model-expansion')
    box('tokenizer',36,66,140,76,'Tokenizer',['prompt → IDs','prompt entry only'],kind='io')
    box('embedding',210,66,145,76,'Embedding E',[('[512,4096]','active'),'E [32000,4096]'],'embedding','weights')
    box('stack',390,66,160,76,'32 decoder layers',['blocks 0 → 31',('[512,4096]','active')],'norm1','residual','decoder')
    box('gather',585,66,135,76,'Last active row',['row T−1','[1,4096]'],'gather','io')
    box('finalnorm',755,66,130,76,'Final RMSNorm',['[1,4096]','γ [4096]'],'finalnorm','norm')
    box('head',920,66,170,76,'Untied LM head',['W [32000,4096]','logits [1,32000]'],'head','weights')
    box('sample',1125,66,195,76,'Sampler',['logits → one token ID','ID → detokenized text'],'sample','io')

    # Attention: residual X bypasses the whole normalized branch.
    label(342,287,'Xₗ [512,4096]', 'input')
    wire('M342 290 V330', stage='norm1')
    wire('M342 290 H72 V728 H100', 'model-residual-wire', 'add1')
    label(83,553,'keep Xₗ',cls='model-skip-label',anchor='start')
    wire('M342 380 V389 H177 V404', stage='q')
    wire('M342 389 H362 V404', stage='k')
    wire('M362 389 H547 V404', stage='v')
    wire('M177 460 V480', stage='rope')
    wire('M362 460 V480', stage='rope')
    wire('M440 508 H465', 'model-cache-wire', 'kv')
    wire('M547 460 V492', 'model-cache-wire', 'kv')
    label(560,479,'V [512,32,128]','v-shape',anchor='start')
    wire('M145 538 V600', stage='qk')
    label(158,573,'Q′',anchor='start')
    wire('M465 558 H270 V581 H215 V600', 'model-cache-wire', 'qk')
    label(365,576,'read cached K′')
    wire('M565 582 V600', 'model-cache-wire', 'pv')
    wire('M255 630 H285', stage='softmax')
    wire('M440 630 H470', stage='pv')
    wire('M547 662 V701', stage='merge')
    wire('M470 728 H440', stage='o')
    wire('M285 728 H255', stage='add1')
    box('norm1',247,330,190,50,'Pre-attn RMSNorm',[('[512,4096]','active')],'norm1','norm')
    for key,x in [('q',100),('k',285),('v',470)]:
        box(key,x,404,155,56,key.upper()+' projection',[('W [4096,4096]',None)],key,'weights')
    box('rope',100,480,340,58,'Split 32 heads · RoPE on Q and K',[('each [512,32,128] · absolute positions C+i','heads-rope')],'rope','attention')
    box('kv',465,492,205,90,'Retained K′ / V cache',[('target each [512,32,128]','cache-shape'),('layer 0 · 0 / 512 stored','cache-valid'),'append new K′ and V'],'kv','cache')
    box('qk',100,600,155,66,'Q′ × K′ᵀ / √128',[('[32,512,512]','scores'),'128 MACs / score'],'qk','attention')
    box('softmax',285,600,155,66,'Mask + softmax',[('P [32,512,512]','probabilities'),'allow key j ≤ C+i'],'softmax','attention')
    box('pv',470,600,155,66,'P × cached V',[('[512,32,128]','heads'),'mix all L value rows'],'pv','attention')
    box('merge',470,701,155,56,'Merge 32 heads',[('[512,4096]','active')],'merge','attention')
    box('o',285,701,155,56,'W_O projection',['W [4096,4096]'],'o','weights')
    box('add1',100,701,155,56,'＋ Add X',['H = Xₗ + attn'],'add1','residual')

    label(1280,317,'W_gate/up [11008,4096]',anchor='end')

    # H joins the second pre-norm branch and survives the second residual.
    wire('M177 757 V807 H735 V282 H1045 V330', 'model-residual-wire', 'norm2')
    label(466,799,'H [512,4096] → FFN input','hidden')
    wire('M1045 282 H1300 V745 H1150', 'model-residual-wire', 'add2')
    label(1278,615,'keep H',cls='model-skip-label',anchor='end')
    wire('M1045 380 V393 H927 V414', stage='gate')
    wire('M1045 393 H1172 V414', stage='up')
    wire('M927 470 V493', stage='swiglu')
    wire('M927 547 V568 H1000 V586', stage='swiglu')
    wire('M1172 470 V568 H1100 V586', stage='swiglu')
    wire('M1050 642 V664', stage='down')
    wire('M1050 720 V731', stage='add2')
    wire('M1050 779 V824', 'model-residual-wire')
    box('norm2',930,330,230,50,'Pre-FFN RMSNorm',[('[512,4096]','active')],'norm2','norm')
    box('gate',835,414,185,56,'Gate projection',[('G [512,11008]','gate')],'gate','weights')
    box('up',1080,414,185,56,'Up projection',[('U [512,11008]','up')],'up','weights')
    box('silu',835,493,185,54,'SiLU on gate',['G / (1 + exp(−G))'],'swiglu','norm')
    box('swiglu',950,586,200,56,'Elementwise multiply',[('Z = SiLU(G) ⊙ U','product')],'swiglu','norm')
    box('down',950,664,200,56,'Down projection',['W [4096,11008]'],'down','weights')
    box('add2',950,731,200,52,'＋ FFN residual',['Xₗ₊₁ = H + FFN'],'add2','residual')
    label(1050,840,'Xₗ₊₁ [512,4096]','output',cls='model-output')
    label(1050,860,'Next: block 1','next-layer')
    label(55,836,'After block 31 → last-row gather → final norm → LM head',anchor='start')
    parts.append(r'''</svg></div>
<div class="model-map-footer"><p id="model-map-selected" aria-live="polite">Selected: embedding · open a block to inspect its math.</p><p><span class="model-key residual-key"></span>Residual bypass <span class="model-key cache-key"></span>Retained K/V <span class="model-key active-key"></span>Selected operation</p></div>
<p class="model-map-note">T = active tokens; C = cached prefix; L = C+T. Q/K/V split into 32 heads of 128 channels; RoPE rotates Q/K. Cache shapes show the target after this layer stores its new rows. <b>SOURCE:</b> these are mathematical dependencies; MAX can fuse QKV projection, RoPE/cache storage and attention. <a href="attention_rope.html">Inspect the attention source →</a></p>
<details class="model-portable"><summary>Portable block diagram / print</summary><pre>prompt → tokenizer → token IDs ────────────────────────────────────┐
                                                                 v
                    sampled ID ── next decode ───────────────→ embedding
                                                                 |
                  REPEAT ×32: distinct weights and KV per layer   |
         ┌───────────────────────────────────────────────────────┘
         v
         X ────────────────────────────────────────────────┐
         |                                                |
       RMSNorm                                            |
         ├──────────────┬──────────────┐                  |
         v              v              v                  |
         Q              K              V                  |
         |              |              |                  |
       RoPE           RoPE             |                  |
         |              └── append K′/V to layer cache    |
         |                         |                      |
         Q′ × cached K′ᵀ / √128    |                      |
         |                         |                      |
       causal mask → softmax P    |                      |
         |                         |                      |
         P × cached V ←────────────┘                      |
         |                                                |
       merge 32 heads → W_O → attention update → ADD ←─────┘
                                                   |
                                                   H ────────────┐
                                                   |             |
                                                 RMSNorm         |
                                               /         \       |
                                             gate         up     |
                                               |          |      |
                                             SiLU         |      |
                                               └─── ⊙ ────┘      |
                                                    |            |
                                                  down → ADD ←───┘
                                                          |
                                      next layer ← X_next ┘

after layer 31 → gather last row → final RMSNorm → untied LM head
                                                        |
                                                 logits [1,32000]
                                                        |
                                                      sampler
                                                        |
                                            token ID → output text

D=4096; F=11008; 32 layers; 32 heads × 128 channels; vocabulary=32000.
Prefill: T=512, C=0, L=512. First decode: T=1, C=512, L=513.
Retain rotated K and V per layer. Sampling itself writes no K/V.</pre></details>
</details>''')
    return ''.join(parts)
