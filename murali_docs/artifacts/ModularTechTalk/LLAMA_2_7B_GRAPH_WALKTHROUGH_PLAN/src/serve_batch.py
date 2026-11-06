"""Batch-constructor close-up, shared by the two standalone Serve readers."""

from html import escape

BATCH_REFS = {
    "batch_sets", "batch_enqueue", "batch_priority", "batch_replica", "batch_ce",
    "batch_tg", "batch_budgets", "batch_advance", "batch_requeue", "batch_reset",
    "active_budget", "context_budget", "construct_batch", "scheduler_iteration",
    "scheduler_execute", "release_request", "preempt_request",
}

BATCH_ROLES = [
    ("Request state", "ReplicaRequests owns ordered CE and TG collections inside TextBatchConstructor. Selecting CE pops its context; selecting TG keeps it tracked.", "batch_sets"),
    ("Ready work", "Poll KV transfers, readmit completed onloads and promote grammar-ready contexts. Held work does not consume a model input row.", "construct_batch"),
    ("Priority", "Per-replica priority considers queue state, configuration, cadence and KV pressure. In-flight TG can backfill CE; a global override suppresses that backfill.", "batch_replica"),
    ("Request seats", "Selected CE plus active TG is bounded by max_batch_size. Spare token capacity does not create a seat.", "batch_ce"),
    ("Active tokens", "Check an active input window. With chunking OFF the active target is soft; with chunking ON a prompt can be split. Checks do not charge usage until the context is accepted.", "active_budget"),
    ("Context tokens", "Optional separate budget for page-aligned context length. Chunking the active window does not shrink the full resident context. The budget collection stops at the first non-AVAILABLE status.", "context_budget"),
    ("KV allocation", "TG checks token budget before allocating; CE allocates before the detailed token check. TG shortages can preempt the newest remaining candidate; transient CE shortages retry, irrecoverable shortages raise.", "batch_tg"),
    ("Inputs + feedback", "Return lists of the selected context references. The pipeline prepares tensors and executes. advance_requests returns partial prefill to CE, promotes completed prefill to TG, and the caller releases terminated requests.", "batch_advance"),
]

BATCH_ASCII = """MODEL WORKER PROCESS · one scheduler loop
  scheduler: enqueue_new_request(context); construct_batch()
       |
       v
 +------------- TextBatchConstructor: one Python object --------------+
 | poll transfers / onloads / grammar readiness                       |
 |                                                                   |
 | ReplicaRequests         choose priority         next input batch   |
 | CE [prefill contexts] -> CE or TG -----------> [context references] |
 | TG [decode contexts]      |                                        |
 |                         request seats                             |
 |                         active-token window                       |
 |                         optional total-context budget             |
 |                         KV alloc / readiness                      |
 |                                                                   |
 | TG: token check -> KV alloc -> select                              |
 |     no blocks -> preempt newest remaining TG candidate -> CE reset |
 | CE: seat + remaining budget -> KV alloc -> chunk/check -> select    |
 |     transient shortage -> retry; irrecoverable shortage -> error   |
 |                                                                   |
 | per-replica in-flight ON: TG first -> optional CE backfill          |
 | per-replica in-flight OFF: CE first -> TG if CE selected nothing    |
 | global TG override: TG only while a TG priority exists             |
 +-----------------------------------|-------------------------------+
                                     v
                       TextGenerationInputs(batches=[[...]])
                                     |
                       pipeline prepares / executes / samples
                                     |
                       advance_requests + caller releases finished work
                       partial prompt -> CE head; full prefill -> TG
                       empty inputs -> NO_PROGRESS -> worker backoff
 SOURCE: Python branches and ownership. BOUNDARY: native execution.
 One ordinary replica; tiny page costs, tokens and readiness are illustrative.
 Repeated request IDs refer to the same context, not duplicated requests."""


def batch_svg():
    panels = [
        ("pending", "sets", 232, 158, 232, 350, "Owned request state", "ReplicaRequests · ordered contexts"),
        ("scheduler", "policy", 490, 158, 224, 80, "Choose priority", "per replica / global override"),
        ("scheduler", "checks", 490, 261, 224, 247, "Candidate checks", "one request at a time"),
        ("scheduler", "batch", 740, 158, 260, 350, "Next input batch", "references · one fixed pass"),
    ]
    boxes = "".join(
        f'<g data-component="{color}" data-batch-component="{key}"><rect class="ss-component-box" x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>'
        f'<text class="ss-node-label" x="{x+w/2}" y="{y+24}">{title}</text><text class="ss-svg-muted" x="{x+w/2}" y="{y+44}">{subtitle}</text></g>'
        for color, key, x, y, w, h, title, subtitle in panels
    )
    budgets = "".join(
        f'<g data-component="{color}"><rect class="ss-component-box" x="{232+i*196}" y="533" width="180" height="88" rx="7"/>'
        f'<text class="ss-node-label" x="{322+i*196}" y="554">{label}</text><text class="ss-svg-muted" x="{322+i*196}" y="575" data-batch-meter="{key}">{value}</text>'
        f'<g data-batch-cells="{key}"></g></g>'
        for i, (color, key, label, value) in enumerate([
            ("scheduler", "seats", "Request seats", "0 / 3 selected"),
            ("scheduler", "tokens", "Active input tokens", "0 / 8 · per pass"),
            ("kv", "kv", "KV block ownership", "1 / 12 · retained pool"),
            ("scheduler", "context", "Total-context tokens", "optional · OFF"),
        ])
    )
    return '''<svg class="ss-batch-diagram" viewBox="0 0 1240 657" role="group" aria-labelledby="ss-batch-title" aria-describedby="ss-batch-desc">
<title id="ss-batch-title">Inside TextBatchConstructor: owned request state, policy, budgets, KV eligibility, inputs and feedback</title>
<desc id="ss-batch-desc">One Python object on the model-worker scheduler loop owns CE and TG request sets. Ready work is evaluated against request seats, active-token and optional context budgets, and KV allocation. Repeated request IDs show references to the same context. Selected input-token tiles are labeled P for prefill or D for decode. The constructor returns inputs to the caller; pipeline execution and sampling happen outside it. After execution, request advancement and release feed the next pass. Controls and a transcript provide a text equivalent.</desc>
<defs><marker id="ss-batch-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#73877a"/></marker></defs>
<g data-component="scheduler"><rect class="ss-component-box ss-batch-owner" x="208" y="22" width="816" height="616" rx="14"/><text class="ss-process-label" x="616" y="45">INSIDE THE BATCH CONSTRUCTOR</text><text class="ss-svg-muted" x="616" y="63">Python object · owned by the model-worker process</text></g>
<text class="ss-svg-muted" x="102" y="44">MODEL WORKER</text><text class="ss-svg-muted" x="102" y="62">scheduler loop</text>
<g data-component="worker"><rect class="ss-component-box" x="18" y="158" width="166" height="160" rx="9"/><text class="ss-node-label" x="101" y="183">Caller: scheduler</text><text class="ss-svg-muted" x="101" y="209">enqueue_new_request()</text><text class="ss-svg-muted" x="101" y="234">construct_batch()</text><text class="ss-svg-muted" x="101" y="274">already-admitted</text><text class="ss-svg-muted" x="101" y="294">request contexts</text></g>
<g data-component="loop" data-batch-component="readiness"><rect class="ss-component-box" x="232" y="74" width="768" height="61" rx="8"/><text class="ss-node-label" x="616" y="96">Poll transfers · readmit landed onloads · promote grammar-ready requests</text><text class="ss-svg-muted" x="616" y="119" data-batch-readiness>No held work · ordinary replica · same worker loop</text></g>
''' + boxes + '''
<text class="ss-node-label" x="250" y="228" text-anchor="start">CE · needs prefill</text><g data-batch-ce-rows data-component="pending"></g>
<text class="ss-node-label" x="250" y="383" text-anchor="start">TG · ready to decode</text><g data-batch-tg-rows data-component="pending"></g>
<text class="ss-svg-muted" x="348" y="493">IDs persist across iterations</text>
<text class="ss-count" x="602" y="220" data-batch-priority>not chosen</text>
<rect class="ss-batch-focus-plate" x="548" y="316" width="108" height="30" rx="6"/>
<g data-batch-focus></g><g data-batch-checks></g><g data-batch-rows></g>
<text class="ss-svg-muted" x="870" y="468" data-batch-inputs>inputs not returned yet</text>
<text class="ss-svg-muted" x="870" y="489">P = prefill · D = decode input</text>
''' + budgets + '''
<g data-component="pipeline" data-batch-component="execute"><rect class="ss-component-box" x="1050" y="158" width="174" height="172" rx="9"/><text class="ss-node-label" x="1137" y="184">Model pipeline</text><text class="ss-svg-muted" x="1137" y="206">outside constructor</text><text class="ss-svg-muted" x="1137" y="237">prepare tensors</text><text class="ss-svg-muted" x="1137" y="257">execute → sample</text><text class="ss-count" x="1137" y="299" data-batch-execution>awaiting inputs</text></g>
<g data-component="worker" data-batch-component="advance"><rect class="ss-component-box" x="1050" y="374" width="174" height="178" rx="9"/><text class="ss-node-label" x="1137" y="398">After execution</text><text class="ss-svg-muted" x="1137" y="421">advance / release</text><g data-batch-results></g><text class="ss-svg-muted" x="1137" y="537">partial → CE · full → TG</text></g>
<g data-component="kv"><rect class="ss-component-box" x="18" y="374" width="166" height="143" rx="9"/><text class="ss-node-label" x="101" y="400">KV manager</text><text class="ss-svg-muted" x="101" y="422">alloc / release</text><text class="ss-svg-muted" x="101" y="449">CPU bookkeeping</text><text class="ss-svg-muted" x="101" y="470">device block pool</text><text class="ss-svg-muted" x="101" y="495">source of page grants</text></g>
<path class="ss-wire" d="M184 234 H232 M464 277 H490 M714 331 H740 M1000 277 H1050 M1137 330 V374" marker-end="url(#ss-batch-arrow)"/>
<path class="ss-worker-kv-wire" d="M184 445 H208 V649 H698 V621" marker-end="url(#ss-batch-arrow)"/>
<path class="ss-wire" d="M1050 514 H1035 V145 H348 V158" marker-end="url(#ss-batch-arrow)"/>
<text class="ss-svg-muted" x="1115" y="579">feedback → next pass</text>
<text class="ss-svg-muted" x="1126" y="613">same worker process</text><text class="ss-svg-muted" x="1126" y="633">no new thread / IPC</text>
</svg>'''


def render_batch_intro():
    modes = "".join(f'<button data-batch-mode="{key}" aria-pressed="{str(key == "mixed").lower()}">{label}</button>' for key, label in [
        ("mixed", "Mixed requests"), ("seats", "Batch full"), ("cache", "KV wait"),
        ("preempt", "Preempt + recompute"), ("context", "Context budget"),
        ("readiness", "Not ready"), ("impossible", "Cannot fit"), ("idle", "No requests"),
    ])
    phases = "".join(f'<li data-batch-phase="{key}"><span>{i}</span>{label}</li>' for i, (key, label) in enumerate([
        ("readiness", "Ready work"), ("policy", "Priority"), ("decode", "TG selection"),
        ("prefill", "CE selection"), ("pack", "Return inputs"), ("execute", "Execute"),
        ("advance", "Advance / release"),
    ], 1))
    rows = "".join(f'<tr><th scope="row">{escape(name)}</th><td>{escape(why)}</td><td>{{{{ref:{ref}}}}}</td></tr>' for name, why, ref in BATCH_ROLES)
    refs = " · ".join("{{ref:" + ref + "}}" for ref in sorted(BATCH_REFS))
    return '''<section class="ss-story ss-batch-story" id="story-batch" data-batch-story aria-label="Inside the batch constructor">
<div class="ss-experiment"><div class="ss-experiment-top">
<details class="ss-lesson" data-ss-lesson open><summary class="ss-lesson-button" aria-label="Lesson 03: Inside the batch constructor" aria-controls="ss-lesson-batch" title="Hover or tap for this lesson">03</summary>
<aside class="ss-prose ss-floating-card ss-lesson-card" id="ss-lesson-batch" role="region" aria-labelledby="ss-heading-batch" tabindex="-1">
<div class="ss-card-heading"><p class="ss-kicker">03 · INSIDE THE BATCH CONSTRUCTOR</p><button data-card-dismiss aria-label="Close lesson 03">×</button></div>
<h2 id="ss-heading-batch">Turn tracked work into the next pass.</h2>
<p>The request sets live inside the constructor. It chooses context references for one model pass; the model pipeline prepares and executes the tensors afterward.</p>
<p>Watch each candidate cross the seat, token and KV checks. CE selection pops a context from its queue; TG stays tracked while its reference joins the next batch.</p>
<p class="ss-try"><strong>Try it</strong> Step through Mixed requests. Compare per-replica in-flight ON/OFF, then disable chunking. KV wait, preemption and context budgets explain why spare batch space can stay unused.</p>
<details class="ss-evidence"><summary>SOURCE · follow the code</summary><p>''' + refs + '''</p></details>
</aside></details><span class="ss-signal"><i></i> BATCH DECISIONS</span><span class="ss-timing-note">SOURCE branches · illustrative tokens / blocks</span></div>
<div class="ss-api-modes" data-batch-controls hidden><span>Choose a situation</span>''' + modes + '''</div>
<p class="ss-api-purpose" data-batch-purpose>One constructor owns request state and forms the next input batch.</p>
<p class="ss-pan-hint">Scroll sideways to follow candidate checks → selected inputs → feedback</p>
<div class="ss-visual" tabindex="0" aria-label="Batch constructor diagram; scroll horizontally on a narrow screen">''' + batch_svg() + '''</div>
<div class="ss-controls" data-batch-controls hidden><button class="ss-play" data-batch-action="play" aria-pressed="false">▶ Play</button><button data-batch-action="step">Step</button><button data-batch-action="reset">↺ Reset</button><span class="ss-control-gap"></span>
<label class="ss-option">Policy<select data-batch-option="policy"><option value="mixed">Per replica · in-flight ON</option><option value="prefill">Per replica · in-flight OFF</option><option value="strict">Global TG override</option></select></label>
<label class="ss-option">Active target<select data-batch-option="budget"><option value="8">8 tokens</option><option value="12">12 tokens</option></select></label>
<label class="ss-option">Chunking<select data-batch-option="chunking"><option value="on">ON</option><option value="off">OFF · soft active target</option></select></label>
<span class="ss-api-progress" data-batch-progress>Pass 0 · event 0</span></div>
<ol class="ss-phases" aria-label="Batch-constructor decision path">''' + phases + '''</ol>
<p class="ss-event" data-current-event role="status" aria-live="polite">Ready. Select a situation, then press Play or Step.</p>
<div class="ss-batch-history" data-batch-history aria-label="Input compositions across passes"></div>
<div class="ss-api-requests" data-batch-request-states aria-label="Request state and why it waits"></div>
<details class="ss-transcript ss-api-transcript ss-batch-transcript"><summary>Decision transcript, roles and portable flow</summary>
<div class="ss-api-role-table"><table><thead><tr><th>Part</th><th>Logic / ownership</th><th>SOURCE</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p>SOURCE: pinned Python branches and ownership. This close-up projects one ordinary replica with no LoRA, speculative drafts, graph padding or overlapping passes. Per-replica in-flight ON permits CE backfill; cadence/backfill are open here. The KV-pressure example uses an illustrative 90% threshold. Alignment, minimum chunk size and per-request active caps are disabled. Global priority voting and DP planning are linked in the source, outside this single-replica scene.</p>
<p>BOUNDARY: execution is schematic. Tiny four-token KV pages, page grants, sequence lengths and readiness steps are teaching inputs. P/D tiles are input work, not output tokens. Existing tokens survive preemption but KV is recomputed. An irrecoverable allocation error is distinct from waiting and from the API admission gate.</p>
<p>Budget checks return AVAILABLE, REACHED or EXHAUSTED. REACHED accepts this context and stops that selection loop. The real composite returns the first non-AVAILABLE result; the Context budget scenario demonstrates its total-context branch while active capacity remains available. Chunking OFF can exceed the soft active target.</p>
<ol data-batch-transcript></ol><pre>''' + escape(BATCH_ASCII) + '''</pre></details>
</div></section>'''


def batch_intro_markdown():
    return [
        "### 03 · Inside the batch constructor",
        "The third animation places ReplicaRequests inside TextBatchConstructor and follows readiness, priority, request seats, token checks, KV allocation, context references, execution and feedback. Repeated IDs refer to the same request context.",
        "Play, Step and Reset compare Mixed requests, Batch full, KV wait, Preempt + recompute, Context budget, Not ready, Cannot fit and No requests. Policy, active-target and chunking controls compare per-replica in-flight ON/OFF and global TG override.",
        "SOURCE: a one-replica projection of the pinned Python branches. BOUNDARY: tiny page costs, inputs, readiness and native execution are illustrative. No adapters, speculative drafts, DP padding or overlap. In this scene cadence/backfill are open, KV pressure uses a 90% teaching threshold, and chunk alignment/floors/per-request caps are disabled.",
        "The optional total-context budget counts page-aligned full context, not the current prefill chunk. Budget checks short-circuit at the first non-AVAILABLE result. With chunking OFF the active-token target is soft. KV preemption resets existing tokens into a fresh prompt for recomputation; it does not discard the token history.",
        "Transient allocation shortages can wait; an irrecoverable shortage with no other work or transfer raises InsufficientBlocksError. Empty inputs let the ordinary caller return NO_PROGRESS and back off.",
        "\n".join(["| Part | Logic / ownership |", "| --- | --- |"] + [f"| {name} | {why} |" for name, why, _ in BATCH_ROLES]),
        "~~~text\n" + BATCH_ASCII + "\n~~~",
    ]
