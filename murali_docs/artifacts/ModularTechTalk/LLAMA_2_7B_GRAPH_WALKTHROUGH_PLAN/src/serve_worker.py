"""Model-worker close-up shared by the two offline MAX Serve readers."""

from html import escape


WORKER_REFS = {
    "worker_entry", "worker_run", "worker_queues", "scheduler_pending",
    "scheduler_iteration", "scheduler_execute", "construct_batch",
    "pipeline_execute", "manager_load", "release_request", "preempt_request",
}

WORKER_ROLES = [
    ("Worker loop", "Runs the scheduler repeatedly; yields with backoff when it reports no progress.",
     "Without the loop, queued work never reaches another scheduling iteration.", "worker_run"),
    ("Request IPC + request sets", "Pulls admitted contexts into the worker's CE/prefill and TG/decode collections. A configured M limits the prefill backlog.",
     "Without the channel, contexts cannot cross from the API. Without request state, the worker cannot track prompt or generation progress.", "scheduler_pending"),
    ("Batch constructor", "Selects eligible request work using batch, token and cache constraints.",
     "Without batch construction, the pipeline has no valid selection of inputs for its next pass.", "construct_batch"),
    ("KV manager", "Tracks each request's cached attention state and reusable blocks in a retained device pool.",
     "Without available blocks, a new prefill can wait. Removing the manager itself breaks this paged-cache pipeline.", "manager_load"),
    ("Model pipeline + runtime", "Prepares inputs, executes the compiled model and produces per-request results through the native device boundary.",
     "Without an executable pipeline/runtime, selected work cannot produce model output.", "pipeline_execute"),
    ("Advance, release + publish", "Moves finished prefill into decode, releases terminated requests and sends results keyed by request ID.",
     "Without advancement or release, work and cache ownership cannot progress correctly. Without publication, the API keeps waiting for results.", "scheduler_execute"),
]

WORKER_ASCII = """API PROCESS (offstage)
    | tokenized, admitted request contexts
    v
[request IPC slots] --> +---- MODEL WORKER: separate OS process -----+
                       | worker uvloop: repeat run_iteration()      |
                       |                                            |
                       | pull within M -> CE/TG request sets         |
                       |                        |                   |
                       |                 construct next batch       |
                       |                 slots + tokens + KV        |
                       |                        |                   |
                       | KV manager <------> model pipeline         |
                       | request ownership      | native runtime ---+--> OS driver / GPU
                       |                        | <--- results ------+---- weights + KV tensors
                       | advance CE -> TG; release finished work    |
[response IPC] <-------| publish {request ID: token/result/status}   |
    |                  |                        |                   |
    v                  | next iteration <-------+                   |
API routes / streams   | no eligible work -> NO_PROGRESS -> backoff |
                       +--------------------------------------------+
Busy pass: its membership stays fixed; a later request waits for another pass.
Partial prefill: finish the whole prompt before emitting its first token.
KV pressure: already-admitted work waits; HTTP 429 is an API admission decision.
Release: request block ownership becomes reusable; the backing pool stays allocated.
One ordinary non-overlap pipeline; tiny queues, budgets and reservations are teaching choices."""


def worker_svg():
    return '''<svg class="ss-worker-diagram" viewBox="0 0 1240 584" role="group" aria-labelledby="ss-worker-title" aria-describedby="ss-worker-desc">
<title id="ss-worker-title">Inside the model worker: request sets, batch selection, KV ownership, execution and results</title>
<desc id="ss-worker-desc">A separate model-worker process owns its uvloop, scheduler, request contexts, batch constructor, KV manager and model pipeline. CPU request contexts move from request IPC into sets and selected batches. The native runtime crosses the OS driver boundary to the GPU, which holds weights and KV tensors. Advance and release update request ownership; response IPC returns a map keyed by request ID. Two requests and a small block pool illustrate an ordinary non-overlap pipeline. The event transcript and role table provide a text equivalent.</desc>
<defs><marker id="ss-worker-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#73877a"/></marker></defs>
<g data-component="worker"><rect class="ss-component-box ss-process" x="210" y="22" width="792" height="546" rx="15"/><text class="ss-process-label" x="606" y="48">MODEL WORKER · SEPARATE PROCESS</text></g>
<g data-component="loop" data-worker-component="loop"><rect class="ss-component-box" x="234" y="68" width="740" height="42" rx="8"/><text class="ss-node-label" x="604" y="94">Worker uvloop · repeat scheduler.run_iteration()</text></g>
<text class="ss-svg-muted" x="604" y="133" data-worker-loop>One loop thread · ready for the next iteration</text>
<text class="ss-node-label" x="102" y="94">API offstage</text><text class="ss-svg-muted" x="102" y="114">already admitted</text>
<g data-component="ipc" data-worker-component="ipc"><rect class="ss-component-box" x="16" y="148" width="172" height="170" rx="9"/><text class="ss-node-label" x="102" y="173">Request IPC</text><text class="ss-svg-muted" x="102" y="194">ZeroMQ · contexts</text><g data-worker-ipc-slots><g class="ss-worker-slot" data-occupied="true" transform="translate(72 234)"><rect x="-26" y="-15" width="52" height="30" rx="5"/><text y="4">R1</text></g><g class="ss-worker-slot" data-occupied="true" transform="translate(134 234)"><rect x="-26" y="-15" width="52" height="30" rx="5"/><text y="4">R2</text></g></g><text class="ss-count" x="102" y="279" data-worker-ipc-count>2 / 2 used</text><text class="ss-svg-muted" x="102" y="303">2 teaching slots</text></g>
<g data-component="pending" data-worker-component="sets"><rect class="ss-component-box" x="242" y="148" width="180" height="170" rx="9"/><text class="ss-node-label" x="332" y="173">Request sets</text><text class="ss-svg-muted" x="332" y="194">CE: prefill · TG: decode</text><g data-worker-set-slots></g><text class="ss-svg-muted" x="332" y="303" data-worker-set-count>CPU contexts · M = 2</text></g>
<g data-component="scheduler" data-worker-component="batch"><rect class="ss-component-box" x="450" y="148" width="224" height="170" rx="9"/><text class="ss-node-label" x="562" y="173">Batch constructor</text><text class="ss-svg-muted" x="562" y="194">slots + tokens + KV</text><g data-worker-batch-slots></g><text class="ss-svg-muted" x="562" y="303" data-worker-budget>0 / 8 input tokens</text></g>
<g data-component="pipeline" data-worker-component="pipeline"><rect class="ss-component-box" x="702" y="148" width="268" height="109" rx="9"/><text class="ss-node-label" x="836" y="173">Model pipeline</text><text class="ss-svg-muted" x="836" y="195">prepare inputs → execute → sample</text><text class="ss-count" x="836" y="227" data-worker-pipeline>awaiting a batch</text><text class="ss-svg-muted" x="836" y="247">Python → compiled model</text></g>
<g data-component="driver" data-worker-component="runtime"><rect class="ss-component-box" x="702" y="278" width="268" height="58" rx="8"/><text class="ss-node-label" x="836" y="300">MAX runtime / user driver</text><text class="ss-svg-muted" x="836" y="321">native submission + completion</text></g>
<g data-component="kernel"><rect class="ss-component-box" x="1044" y="148" width="180" height="81" rx="8"/><text class="ss-node-label" x="1134" y="177">OS GPU driver</text><text class="ss-svg-muted" x="1134" y="202">kernel boundary</text></g>
<g data-component="gpu" data-worker-component="gpu"><rect class="ss-component-box" x="1044" y="272" width="180" height="239" rx="10"/><text class="ss-node-label" x="1134" y="298">GPU device</text><text class="ss-svg-muted" x="1134" y="320">weights + KV tensors</text><text class="ss-count" x="1134" y="348" data-worker-gpu>idle</text><g data-worker-gpu-work></g><text class="ss-svg-muted" x="1134" y="482">selected model work</text><text class="ss-svg-muted" x="1134" y="499">hardware · not a process</text></g>
<g data-component="kv" data-worker-component="kv"><rect class="ss-component-box" x="242" y="385" width="240" height="126" rx="9"/><text class="ss-node-label" x="362" y="410">KV manager</text><text class="ss-svg-muted" x="362" y="430">request block ownership</text><g data-worker-pages></g><text class="ss-svg-muted" x="362" y="499" data-worker-page-count>0 / 4 owned · pool retained</text></g>
<g data-component="worker" data-worker-component="results"><rect class="ss-component-box" x="512" y="385" width="458" height="126" rx="9"/><text class="ss-node-label" x="741" y="410">Advance / release / publish</text><text class="ss-svg-muted" x="741" y="430">prefill → decode · finished → release KV ownership</text><text class="ss-svg-muted" x="645" y="470" data-worker-result="R1">R1: no output yet</text><text class="ss-svg-muted" x="866" y="470" data-worker-result="R2">R2: no output yet</text><text class="ss-svg-muted" x="741" y="499">results stay correlated by request ID</text></g>
<g data-component="ipc" data-worker-component="response"><rect class="ss-component-box" x="16" y="385" width="172" height="126" rx="9"/><text class="ss-node-label" x="102" y="410">Response IPC</text><text class="ss-svg-muted" x="102" y="430">last result map</text><rect class="ss-worker-result-map" x="30" y="439" width="144" height="55" rx="5"/><text class="ss-svg-muted" x="102" y="459" data-worker-published="R1">—</text><text class="ss-svg-muted" x="102" y="482" data-worker-published="R2">—</text></g>
<path class="ss-wire" d="M188 234 H242 M422 234 H450 M674 234 H702 M836 257 V278 M970 307 H1016 V188 H1044 M1134 229 V272 M1044 446 H970" marker-end="url(#ss-worker-arrow)"/>
<path class="ss-wire" d="M741 385 V356 H332 V318" marker-end="url(#ss-worker-arrow)"/><text class="ss-svg-muted" x="490" y="350">next iteration</text>
<path class="ss-worker-kv-wire" d="M482 420 H495 V330 H562 V318 M512 482 H482" marker-end="url(#ss-worker-arrow)"/>
<path class="ss-return" data-component="ipc" d="M741 511 V540 H102 V511" marker-end="url(#ss-worker-arrow)"/><text class="ss-svg-muted" x="424" y="559">token IDs + status → API routes / streams</text>
<text class="ss-svg-muted" x="1134" y="552">BOUNDARY · illustrative</text>
<g data-worker-dots aria-hidden="true"></g><g data-worker-token-dots aria-hidden="true"></g>
</svg>'''


def render_worker_intro():
    choices = [("normal", "Two requests"), ("busy", "GPU busy"), ("cache", "KV pressure"), ("idle", "No requests")]
    buttons = ''.join(f'<button data-worker-mode="{key}" aria-pressed="{str(key == "normal").lower()}">{label}</button>' for key, label in choices)
    rows = ''.join(f'<tr><th scope="row">{name}</th><td>{why}</td><td>{missing}</td><td>{{{{ref:{ref}}}}}</td></tr>' for name, why, missing, ref in WORKER_ROLES)
    phases = ''.join(f'<li data-worker-phase="{key}">{label}</li>' for key, label in [("drain", "Pull"), ("select", "Build batch"), ("execute", "Execute"), ("advance", "Advance / release"), ("publish", "Publish"), ("idle", "Idle / backoff")])
    return '''<section class="ss-story ss-worker-story" id="story-worker" data-worker-story aria-label="Inside the model worker process">
<div class="ss-experiment"><div class="ss-experiment-top">
<details class="ss-lesson" data-ss-lesson open><summary class="ss-lesson-button" aria-label="Lesson 02: Inside the model worker" aria-controls="ss-lesson-worker" title="Hover or tap for this lesson">02</summary>
<aside class="ss-prose ss-floating-card ss-lesson-card" id="ss-lesson-worker" role="region" aria-labelledby="ss-heading-worker" tabindex="-1"><div class="ss-card-heading"><p class="ss-kicker">02 · Inside the model worker</p><button data-card-dismiss aria-label="Close lesson 02">×</button></div><h2 id="ss-heading-worker">A batch, then another.</h2>
<p>The API has already admitted these requests. Inside the separate worker process, one loop calls the scheduler: <b>pull → build → execute → advance / release → publish</b>. No eligible work produces <code>NO_PROGRESS</code>, then an awaited backoff.</p>
<p><b>CE</b> is prompt processing (prefill); <b>TG</b> is continuation (decode). The constructor picks eligible work within batch, token and KV constraints. A running pass keeps its selected members. A partial prefill returns for another pass before it can produce its first token.</p>
<p>The pipeline calls the native runtime and GPU. Request contexts stay in CPU memory; GPU tiles represent the selected work. Results are returned by request ID. Finished requests release block ownership; the backing KV pool remains allocated.</p>
<p class="ss-try"><strong>Try it</strong> Compare <b>GPU busy</b> with <b>KV pressure</b>. Watch R2's waiting location, the unused batch slot, and the blocks becoming reusable. <b>No requests</b> shows the idle path.</p>
<p class="ss-setting-note">One ordinary non-overlap pipeline; small synthetic requests and an illustrative decode-first, chunked policy. The toy cache reserves prompt plus output blocks at entry. Actual allocation, preemption, overlap and priorities depend on configuration. This close-up begins after admission; cache waits are not HTTP 429.</p>
<details class="ss-evidence"><summary>SOURCE · follow the code</summary><p>{{ref:worker_entry}} · {{ref:worker_run}} · {{ref:scheduler_pending}} · {{ref:construct_batch}} · {{ref:scheduler_execute}} · {{ref:release_request}}</p></details>
</aside></details><span class="ss-signal"><i></i> INSIDE THE MODEL WORKER</span><span class="ss-timing-note">one process · one pass at a time</span></div>
<div class="ss-api-modes ss-worker-modes" data-worker-controls hidden role="group" aria-label="Choose a model-worker scenario"><span>Explore the worker</span>''' + buttons + '''</div>
<p class="ss-api-purpose" data-worker-purpose>Follow two admitted requests through batch selection, execution and results.</p>
<p class="ss-pan-hint">Scroll sideways: IPC → worker components → GPU</p>
<div class="ss-visual" tabindex="0" aria-label="Model-worker diagram; scroll horizontally on a narrow screen">''' + worker_svg() + '''</div>
<div class="ss-controls" data-worker-controls hidden><button class="ss-play" data-worker-action="play" aria-pressed="false">▶ Play</button><button data-worker-action="step">Step</button><button data-worker-action="reset">↺ Reset</button><span class="ss-control-gap"></span><span class="ss-api-progress" data-worker-progress>Ready · event 0</span></div>
<ol class="ss-phases" aria-label="Model-worker iteration">''' + phases + '''</ol>
<div class="ss-api-requests" aria-label="Worker request states"><span><b>R1</b> <span data-worker-state="R1">admitted · waiting in IPC</span></span><span><b>R2</b> <span data-worker-state="R2">admitted · waiting in IPC</span></span></div>
<p class="ss-event" data-current-event role="status" aria-live="polite">Ready. Play or Step to pull the admitted contexts into the worker.</p>
<details class="ss-transcript ss-api-transcript ss-worker-transcript"><summary>Why each part exists · if missing · source and portable flow</summary>
<p><b>SOURCE</b> The pinned worker calls <code>run_iteration()</code> on its own loop. The scheduler drains contexts, constructs a batch, executes it, advances state, releases finished requests and publishes a result map. <b>BOUNDARY</b> The diagram splits this ordinary call into teaching steps; it is not a GPU trace or a claim that these Python stages run on separate threads.</p>
<div class="ss-api-role-table" tabindex="0" aria-label="Worker components and missing-role outcomes"><table><thead><tr><th>Part</th><th>Why it exists</th><th>If unavailable</th><th>Source</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p>Request and response channels: {{ref:worker_queues}}. Iteration and idle outcome: {{ref:scheduler_iteration}}. Actual cache/preemption policy: {{ref:construct_batch}} · {{ref:preempt_request}}. Missing-role outcomes assume no replacement implementation; they are not MAX switches.</p>
<p>All four scenarios reuse the same teaching worker as the later stories. N = 2, M = 2, batch ceiling = 2 and input budget = 8; KV ownership uses four illustrative blocks, or two under pressure. P marks prefill input tokens; D marks decode input tokens. Pool sizes, reservations, selected priorities and event durations are illustrative. Prefix reuse, multi-device execution, grammar/adapter constraints and overlap are outside this close-up.</p>
<ol data-worker-transcript></ol><pre>''' + escape(WORKER_ASCII) + '''</pre></details>
</div></section>'''


def worker_intro_markdown():
    return [
        '### 02 · Inside the model worker',
        'The second animation isolates the separate worker process after API admission: request IPC, CE/TG request sets, batch construction, KV ownership, the model pipeline, native runtime, OS driver, GPU and response IPC. Hover/focus/tap 02 for the lesson; L explains the component colors and ownership.',
        'Play, Step and Reset compare Two requests, GPU busy, KV pressure and No requests. The loop highlights pull → build → execute → advance/release → publish, with NO_PROGRESS/backoff when no work is eligible. P and D show prefill/decode input work. The CPU request markers and GPU work tiles represent different things.',
        'SOURCE: An ordinary worker calls the scheduler on its own loop. It publishes results keyed by request ID and releases finished request/cache ownership. BOUNDARY: The close-up reuses the existing teaching worker, including its illustrative decode-first chunking, exact two-slot request queue and conservative prompt-plus-output block reservation. Timings, capacities and starting snapshots are not measurements. The backing KV pool remains allocated when request ownership is released.',
        'In the busy scenario, R2 waits in IPC until a subsequent worker iteration. Under cache pressure, R2 is already admitted and waits even though a batch slot is available; it starts after R1 releases blocks. Partial prefill emits no token before the full prompt is consumed. The idle scenario checks for work and then yields with backoff.',
        '\n'.join(['| Part | Why it exists | If unavailable |', '| --- | --- | --- |'] + [f'| {name} | {why} | {missing} |' for name, why, missing, _ in WORKER_ROLES]),
        '```text\n' + WORKER_ASCII + '\n```',
    ]
