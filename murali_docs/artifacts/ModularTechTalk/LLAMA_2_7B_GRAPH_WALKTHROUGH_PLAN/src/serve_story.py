"""Small interactive essays shared by the two MAX Serve HTML readers."""

from html import escape

from serve_api import API_REFS, api_intro_markdown, render_api_intro


LEGEND = [
    ("api-process", "API process", "OS process", "Owns the HTTP server, app, event loop and tokenizer executor.", ["serve"]),
    ("uvicorn", "Uvicorn / Server", "Python object", "Uvicorn's Server object handles HTTP and invokes the FastAPI ASGI app in the API process.", ["uvicorn_cfg"]),
    ("fastapi", "FastAPI app", "ASGI application", "Routes and validates requests. Async request tasks share the API loop thread.", ["app"]),
    ("loop", "uvloop event loop", "Event-loop thread", "Runs ready callbacks and coroutines. The API and model worker each own a loop in their own process.", ["eventloop", "worker_run"]),
    ("tokenizer", "Tokenizer executor", "CPU thread pool", "CPU tokenization runs via run_in_executor(None, …) on executor threads in the API process.", ["tokenizer_executor"]),
    ("ipc", "ZeroMQ channels", "IPC · Unix sockets", "Local ipc:// PUSH/PULL channels carry requests, responses and cancellation. Each drawn slot holds one request context; N is exact only in this teaching model.", ["ipc_address", "ipc_config"]),
    ("worker", "Model worker / scheduler", "Spawned OS process", "Owns pending work, batch selection and model execution. The scheduler runs on the worker's loop thread; it selects the next pass.", ["spawn", "worker_run"]),
    ("driver", "MAX runtime / user driver", "Native code · worker", "Submits compiled model work and manages device completion. Internal native threads sit behind this boundary.", ["session_native"]),
    ("kernel", "OS GPU driver", "Kernel boundary", "The operating system's device driver connects the user-space runtime to the GPU.", []),
    ("gpu", "GPU device", "Hardware", "Executes the selected model pass and holds device data, including model weights and KV cache.", []),
]


def render_legend():
    rows = []
    for key, title, kind, description, refs in LEGEND:
        evidence = " · ".join('{{ref:' + ref + '}}' for ref in refs)
        source = f'<details class="ss-legend-source"><summary>SOURCE</summary>{evidence}</details>' if refs else '<a class="ss-boundary-link" href="#reference-atlas">BOUNDARY · reference atlas ↗</a>'
        rows.append(f'<li data-component="{key}"><span class="ss-legend-swatch" aria-hidden="true"></span><div><strong>{title}</strong><span class="ss-kind">{kind}</span></div><div><p>{description}</p>{source}</div></li>')
    return '''<details class="ss-legend" data-ss-legend>
<summary class="ss-legend-button" aria-label="Component legend" aria-controls="ss-legend-card" title="Component colors and ownership">L</summary>
<aside class="ss-floating-card ss-legend-card" id="ss-legend-card" role="region" aria-labelledby="ss-legend-title" tabindex="-1">
<div class="ss-card-heading"><div><p class="ss-kicker">COLOR → COMPONENT → OWNER</p><h2 id="ss-legend-title">Read the architecture</h2></div><button data-card-dismiss aria-label="Close component legend">×</button></div>
<p class="ss-legend-intro">Colors match the animations. Boxes nest inside their owning process; arrows carry calls or messages.</p>
<ul class="ss-legend-rows">''' + ''.join(rows) + '''</ul>
<div class="ss-sync-key"><h3>What synchronizes the handoff?</h3>
<div data-component="admission"><b>◇ Admission gate</b><p><code>asyncio.Lock</code> · coroutine mutual exclusion around the writable probe and send, inside the API process. It is released before awaiting model output; it is not another event loop.</p><details class="ss-legend-source"><summary>SOURCE</summary>{{ref:proxy_init}} · {{ref:proxy_submit}}</details></div>
<div><b>◎ Worker readiness</b><p><code>multiprocessing.Event</code> · a process-shared readiness signal during startup. The API also awaits the channel connection.</p><details class="ss-legend-source"><summary>SOURCE</summary>{{ref:start_worker}} · {{ref:connected}}</details></div>
</div><p class="ss-legend-footnote">Component colors identify owners; request colors indicate request state. SOURCE: pinned Python control flow. BOUNDARY: native runtime, OS and device. One ordinary, non-overlap worker per replica.</p>
</aside></details>'''


STORIES = {
    "components": [
        dict(id="one", number="02", label="Start with one request", title="Accepted. Then scheduled.",
             intro="A valid request R1 arrives at a ready MAX Serve instance. Follow it through tokenization, the admission gate, prefill and repeated decode passes.",
             lesson="Admission gives R1 a place on the request channel. The worker later chooses when it can execute. Output tokens return to the client while generation continues.",
             try_it="Press Play. Click a request to follow its state; Step advances one event.",
             action="Send another request", refs=["proxy_submit", "scheduler_iteration", "first_chunk"],
             ascii="Client → API/tokenize → admission → request IPC → worker pending\n                                                  ↓\nClient ← TCP/SSE ← API ← response IPC ← GPU/native ← selected batch\n                                      ↑ prefill, then repeated decode"),
        dict(id="busy", number="03", label="Now the worker is busy", title="There is room to wait.",
             intro="R1 is already generating. Send R2 while the worker is inside a model pass. Watch where R2 waits and when it joins a batch.",
             lesson="With one batch slot, R2 waits for R1 to finish. With two slots and enough KV space, R2 can join a later pass while R1 is still decoding. A running pass keeps its selected members.",
             try_it="Send R2, then replay with two batch slots. Compare the waiting location.",
             action="Send R2", refs=["proxy_submit", "scheduler_pending", "construct_batch"],
             option=("Batch slots", "batch", [(1, "1 slot"), (2, "2 slots")]),
             ascii="API accepts R2 ──→ IPC  | R1's selected pass is running\nnext worker iteration ─→ pending → batch eligibility\nB=1: R2 waits for a slot   B=2: R2 may join a subsequent pass"),
        dict(id="pressure", number="04", label="Keep sending work", title="Watch backpressure reach the gate.",
             intro="Keep R1 decoding, then send a burst. The worker's pending-prefill queue M fills. It stops pulling from IPC. The request channel N then fills too.",
             lesson="When the API proxy finds that connected channel unwritable, it raises RequestQueueFull. The HTTP handler returns 429 with Retry-After: 1 before streaming begins. Already admitted requests stay in the system.",
             try_it="Send a burst of six. Increase N and replay the same burst to see more waiting room.",
             action="Send a burst of 6", refs=["queue_settings", "scheduler_pending", "proxy_submit", "http_429"],
             option=("IPC waiting room N", "queue", [(2, "2 requests"), (6, "6 requests")]),
             ascii="Long decode holds batch slot\n       ↓\nM full → stop pulling → N fills → API probe fails → HTTP 429\nWork completes → pending drains → IPC drains → admission has room again"),
        dict(id="scheduler", number="05", label="Look inside the loop", title="A batch is one pass of work.",
             intro="R1 is decoding a response. R2 has a longer prompt. The scheduler builds each pass from eligible work, within batch, token and KV limits.",
             lesson="This example gives decode one input token per request and uses the remaining budget for chunked prefill. A partial prompt returns for another pass. Completing prefill yields the first output token; ordinary decode then adds one token per selected request per pass.",
             try_it="Step through the five highlighted phases. Compare the pass history below the diagram.",
             action="Add a short prompt", refs=["worker_run", "scheduler_iteration", "scheduler_execute", "construct_batch", "preempt_request"],
             option=("Input tokens per pass", "budget", [(8, "8 tokens"), (16, "16 tokens")]),
             ascii="repeat: pull within M → construct batch → pipeline.execute()\n             ↑                                  ↓\n       publish by request ID ← advance state / release finished requests\nno eligible work → NO_PROGRESS → yield/backoff → check again"),
    ],
    "systems": [
        dict(id="clocks", number="02", label="Put the owners side by side", title="Busy belongs to an owner.",
             intro="While the model worker executes a pass, the API can tokenize and admit another request. The OS schedules runnable CPU threads; MAX chooses model work; the GPU executes submitted operations.",
             lesson="An HTTP task waiting for output is suspended until data is available. It does not require a dedicated CPU thread. GPU completion and socket readiness are different events. The lanes below show logical ownership, not a measured thread trace.",
             try_it="Send another request during Execute. Compare the API lane with the worker and device lanes.",
             action="Send another request", refs=["worker_run", "eventloop", "tokenizer_cpu", "scheduler_execute", "session_native"],
             ascii="OS:     gives CPU time to runnable threads in both processes\nAPI:    tokenize → admit → await output → SSE\nWorker: drain → select → execute → advance → publish → repeat\nGPU:                     submitted compute → result\nLinux socket readiness wakes I/O; native runtime manages device dependencies."),
        dict(id="reader", number="03", label="Separate generation from delivery", title="The GPU finishes. The stream remains.",
             intro="Make the client read slowly. Output tokens collect after generation, while the worker can release that request's KV ownership and start other work.",
             lesson="Response IPC, API output queues and TCP transport each have buffering. This animation groups them into one output buffer. A slow client can raise egress memory and latency; an admitted stream does not turn into a new HTTP 429 response.",
             try_it="Play until R1 has generated all five tokens. Compare KV ownership with buffered output.",
             action="Send another request", refs=["proxy_responses", "proxy_drain", "release_request", "sse_response"],
             option=("Client reads", "reader", [(12, "Slowly"), (1, "Normally")]),
             ascii="GPU completes → release request KV ownership (pool remains allocated)\n       ↓\nresponse IPC → API output buffer → TCP → slow client\n              tokens can remain after generation finishes"),
        dict(id="replicas", number="04", label="Give new work another destination", title="Another replica, another worker.",
             intro="An external router can send new requests to independent MAX Serve replicas. Each replica has its own API, worker, queues, model allocation and KV pool.",
             lesson="Add B, wait for readiness, then send more requests. Existing streams keep their original owner. This round-robin router and its warmup duration are teaching choices. Adding HTTP connections alone does not add model compute.",
             try_it="Add replica B. After it becomes ready, send a burst and watch which replica owns each request.",
             action="Send a burst of 4", refs=["ready", "connected", "worker_run"],
             ascii="Clients → external round-robin router → ready replica A (API + worker + GPU/KV)\n                                     → ready replica B (API + worker + GPU/KV)\nInitializing replicas receive no new requests. Existing work stays with its owner."),
    ],
}


def story_refs(page):
    return {key for story in STORIES[page] for key in story["refs"]} | {
        "queue_settings", "proxy_submit", "scheduler_pending", "construct_batch",
        "http_429", "worker_run", "scheduler_execute", "release_request",
        "proxy_init", "start_worker", "connected",
    } | {ref for row in LEGEND for ref in row[4]} | API_REFS


def pipeline_svg(key):
    nodes = [(686, "Pending prefill", "worker cap M", "pending"), (836, "Batch members", "slots + KV", "batch")]
    boxes = "".join(f'<g class="ss-station" data-component="worker" data-station="{kind}"><rect class="ss-component-box" x="{x-66}" y="142" width="132" height="160" rx="9"/><text x="{x}" y="163" class="ss-node-label">{title}</text><text x="{x}" y="325" class="ss-svg-muted" data-cap="{kind}">{subtitle}</text><text x="{x}" y="287" class="ss-count" data-count="{kind}">0</text></g>' for x, title, subtitle, kind in nodes)
    return f'''<svg class="ss-pipeline" viewBox="0 0 1240 448" role="group" aria-labelledby="ss-title-{key}" aria-describedby="ss-desc-{key}">
<title id="ss-title-{key}">Animated requests from API admission to worker batches and client output</title>
<desc id="ss-desc-{key}">Colored components nest in their owning process. Uvicorn's Server and FastAPI share the API event loop. CPU tokenization uses executor threads. ZeroMQ message slots cross to a separately spawned worker, its scheduler and native runtime. The OS driver and GPU are separate boundaries. Use L for the legend, or the current event and request details for a text equivalent.</desc>
<defs><marker id="ss-arrow-{key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#73877a"/></marker></defs>
<g data-component="api-process"><rect class="ss-component-box ss-process" x="92" y="26" width="332" height="305" rx="14"/><text class="ss-process-label" x="258" y="50">API PROCESS</text></g>
<g data-component="worker"><rect class="ss-component-box ss-process" x="602" y="26" width="464" height="305" rx="14"/><text class="ss-process-label" x="834" y="50">MODEL WORKER · SEPARATE PROCESS</text></g>
<g data-component="uvicorn"><rect class="ss-component-box" x="110" y="72" width="132" height="55" rx="7"/><text class="ss-node-label" x="176" y="94">Uvicorn</text><text class="ss-svg-muted" x="176" y="113">Server object</text></g>
<path class="ss-wire" d="M69 223 H83 V100 H110 M242 100 H258" marker-end="url(#ss-arrow-{key})"/>
<g data-component="fastapi"><rect class="ss-component-box" x="258" y="72" width="148" height="55" rx="7"/><text class="ss-node-label" x="332" y="94">FastAPI app</text><text class="ss-svg-muted" x="332" y="113">ASGI · request tasks</text></g>
<g data-component="loop"><rect class="ss-component-box" x="276" y="146" width="130" height="47" rx="7"/><text class="ss-node-label" x="341" y="166">uvloop</text><text class="ss-svg-muted" x="341" y="183">API loop thread</text></g>
<path class="ss-wire ss-call" d="M332 127 V146 M276 169 H268" marker-end="url(#ss-arrow-{key})"/>
<g data-component="loop"><rect class="ss-component-box" x="620" y="72" width="428" height="48" rx="7"/><text class="ss-node-label" x="834" y="92">uvloop · worker loop thread</text><text class="ss-svg-muted" x="834" y="110">scheduler → execute → publish → repeat</text></g>
<path class="ss-wire" d="M268 223 H920" marker-end="url(#ss-arrow-{key})"/>
<circle class="ss-client" cx="44" cy="223" r="25"/><path class="ss-client-icon" d="M30 229 Q44 214 58 229 M44 206 v4"/>
<text class="ss-node-label" x="44" y="182">Client</text>
<g class="ss-station" data-component="tokenizer" data-station="api"><rect class="ss-component-box" x="122" y="142" width="146" height="160" rx="9"/><text class="ss-node-label" x="195" y="163">Tokenize</text><text class="ss-count" x="195" y="287" data-count="api">0</text><text class="ss-svg-muted" x="195" y="321">executor threads</text></g>
<g class="ss-gate" data-gate="open"><path d="M345 200 L363 229 L345 258 L327 229 Z"/><text x="345" y="234">✓</text></g>
<text class="ss-node-label" x="345" y="281">admission</text><text class="ss-svg-muted" x="345" y="300">asyncio.Lock</text>
<g class="ss-station" data-component="ipc" data-station="ipc"><rect class="ss-component-box" x="446" y="142" width="132" height="160" rx="9"/><text class="ss-node-label" x="512" y="163">Request IPC</text><g data-ipc-slots aria-label="Request message slots">{queue_slots_svg()}</g><text class="ss-count" x="512" y="287" data-count="ipc">0 / 2</text><text class="ss-svg-muted" x="512" y="325">ZeroMQ · ipc://</text></g>
{boxes}
<g data-component="driver"><rect class="ss-component-box" x="920" y="175" width="128" height="111" rx="9"/><text class="ss-node-label" x="984" y="199">MAX runtime</text><text class="ss-node-label" x="984" y="220">+ user driver</text><text class="ss-svg-muted" x="984" y="249">native code</text><text class="ss-svg-muted" x="984" y="268">in worker</text></g>
<path class="ss-boundary" d="M1085 56 V330"/><text class="ss-svg-muted" x="1164" y="84">OS / DEVICE</text>
<path class="ss-wire" d="M1048 229 H1074 V135 H1110 M1164 158 V190" marker-end="url(#ss-arrow-{key})"/>
<g data-component="kernel"><rect class="ss-component-box" x="1100" y="112" width="128" height="46" rx="7"/><text class="ss-node-label" x="1164" y="132">OS GPU driver</text><text class="ss-svg-muted" x="1164" y="149">kernel</text></g>
<g class="ss-gpu" data-component="gpu"><rect class="ss-component-box" x="1110" y="190" width="108" height="112" rx="11"/><text class="ss-node-label" x="1164" y="216">GPU device</text><text class="ss-count" x="1164" y="247" data-gpu>Idle</text><text class="ss-svg-muted" x="1164" y="279">hardware</text></g>
<path class="ss-return" data-component="ipc" d="M984 286 V350 H44 V250" marker-end="url(#ss-arrow-{key})"/>
<text class="ss-svg-muted" x="620" y="372">worker publishes → response IPC → API queue → SSE / TCP → client</text>
<text class="ss-svg-muted" x="94" y="422">delivered</text><text class="ss-svg-muted" x="349" y="422">429 rejected</text><text class="ss-svg-muted" x="823" y="395">output still buffered</text>
<g data-request-dots></g><g data-output-dots aria-hidden="true"></g>
</svg>'''


def fleet_svg():
    rows = []
    for i, name in enumerate(("A", "B")):
        rows.append(f'''<g data-fleet-replica="{i}" transform="translate(0 {i*252})">
<rect class="ss-replica-box" x="385" y="26" width="843" height="225" rx="13"/><text class="ss-node-label" x="444" y="53">Replica {name}</text><text class="ss-svg-muted" x="932" y="53" data-fleet-state="{i}">{'Ready · its own worker and GPU' if i == 0 else 'Not added'}</text>
<g data-fleet-detail="{i}">
<g data-component="api-process"><rect class="ss-component-box ss-process" x="398" y="82" width="202" height="161" rx="9"/><text class="ss-process-label" x="499" y="101">API PROCESS</text></g>
<g data-component="uvicorn"><rect class="ss-component-box" x="410" y="113" width="87" height="30" rx="5"/><text class="ss-svg-muted" x="454" y="132">Uvicorn</text></g>
<g data-component="fastapi"><rect class="ss-component-box" x="505" y="113" width="83" height="30" rx="5"/><text class="ss-svg-muted" x="547" y="132">FastAPI</text></g>
<g data-component="loop"><rect class="ss-component-box" x="410" y="152" width="87" height="30" rx="5"/><text class="ss-svg-muted" x="454" y="171">uvloop</text></g>
<g data-component="tokenizer"><rect class="ss-component-box" x="505" y="152" width="83" height="30" rx="5"/><text class="ss-svg-muted" x="547" y="171">tokenizer</text></g>
<text class="ss-svg-muted" x="499" y="236" data-fleet-api>0 requests</text>
<path class="ss-wire" d="M600 195 H1094"/>
<g data-component="ipc"><rect class="ss-component-box" x="614" y="130" width="102" height="113" rx="7"/><text class="ss-node-label" x="665" y="153">IPC · N</text><g data-ipc-slots aria-label="Replica request slots">{queue_slots_svg(fleet=True)}</g><text class="ss-svg-muted" x="665" y="234" data-fleet-ipc>0 / 2</text></g>
<g data-component="worker"><rect class="ss-component-box ss-process" x="730" y="82" width="328" height="161" rx="9"/><text class="ss-process-label" x="894" y="101">MODEL-WORKER PROCESS</text></g>
<g data-component="loop"><rect class="ss-component-box" x="744" y="113" width="300" height="30" rx="5"/><text class="ss-svg-muted" x="894" y="132">uvloop · worker loop thread</text></g>
<g data-component="worker"><rect class="ss-component-box" x="744" y="157" width="92" height="75" rx="6"/><text class="ss-svg-muted" x="790" y="175" data-fleet-pending>Pending · 0</text></g>
<g data-component="worker"><rect class="ss-component-box" x="847" y="157" width="92" height="75" rx="6"/><text class="ss-svg-muted" x="893" y="175" data-fleet-batch>Batch · 0</text></g>
<g data-component="driver"><rect class="ss-component-box" x="953" y="157" width="91" height="75" rx="6"/><text class="ss-svg-muted" x="999" y="185">MAX runtime</text><text class="ss-svg-muted" x="999" y="207">+ user driver</text></g>
<path class="ss-wire" d="M1044 195 H1080 V131 H1094 M1152 150 V169"/>
<g data-component="kernel"><rect class="ss-component-box" x="1094" y="112" width="117" height="38" rx="6"/><text class="ss-svg-muted" x="1152" y="136">OS GPU driver</text></g>
<g class="ss-gpu" data-component="gpu"><rect class="ss-component-box" x="1110" y="169" width="87" height="63" rx="7"/><text class="ss-node-label" x="1154" y="190">GPU</text><text class="ss-svg-muted" x="1154" y="217" data-fleet-gpu>Idle</text></g>
</g></g>''')
    return '''<svg class="ss-fleet" viewBox="0 0 1240 548" role="group" aria-label="An external router sends new requests to ready, independent replicas A and B. Each has colored API and model-worker processes, IPC slots, a native runtime and GPU.">
<path class="ss-wire" d="M119 266 H279 V195 H398 M279 266 V447 H398"/>
<circle class="ss-client" cx="83" cy="266" r="36"/><text class="ss-node-label" x="83" y="270">Clients</text>
<rect class="ss-process" x="184" y="228" width="174" height="77" rx="10"/><text class="ss-node-label" x="271" y="258">External router</text><text class="ss-svg-muted" x="271" y="284">round robin · example</text>
''' + ''.join(rows) + '''<g data-request-dots></g>
<text class="ss-svg-muted" x="806" y="533">Each request keeps one replica owner throughout its stream.</text></svg>'''


def queue_slots_svg(fleet=False):
    # Static empty slots remain understandable without JavaScript.
    x, y = (638, 195) if fleet else (486, 227)
    return ''.join(f'<g class="ss-message-slot" data-ipc-slot="{i}" data-occupied="false" transform="translate({x+i*52} {y})"><rect x="-23" y="-15" width="46" height="30" rx="5"/><text y="4">{i+1}</text></g>' for i in range(2))


def render_stories(page):
    result = [render_api_intro()]
    for story in STORIES[page]:
        key = story["id"]
        option = ""
        if "option" in story:
            label, name, choices = story["option"]
            option = f'<label class="ss-option">{label}<select data-ss-option="{name}">' + "".join(f'<option value="{value}">{text}</option>' for value, text in choices) + '</select></label>'
        extra = '<button data-ss-action="replica">＋ Add replica B</button>' if key == "replicas" else ''
        phases = "".join(f'<li data-phase="{phase}"><span>{n}</span>{label}</li>' for n, (phase, label) in enumerate([
            ("drain", "Pull requests"), ("select", "Build batch"), ("execute", "Execute"),
            ("advance", "Advance / release"), ("publish", "Publish")], 1))
        refs = " · ".join('{{ref:' + ref + '}}' for ref in story["refs"])
        lanes = '<div class="ss-lanes" data-owner-lanes></div>' if key in ("clocks", "reader") else ''
        history = '<div class="ss-batch-history" data-batch-history aria-label="Recent batch compositions"></div>' if key in ("scheduler", "busy") else ''
        diagram = fleet_svg() if key == "replicas" else pipeline_svg(key)
        setting_note = ('<p class="ss-setting-note">N = <code>--max-queue-size</code><br>M = <code>--max-pending-requests</code><br>Both default to unbounded in this pinned checkout. This scene sets finite example limits.</p>'
                        if key == 'pressure' else '')
        result.append(f'''<section class="ss-story" id="story-{key}" data-serve-story="{key}" aria-label="{escape(story['label'])}">
<div class="ss-experiment"><div class="ss-experiment-top">
<details class="ss-lesson" data-ss-lesson open><summary class="ss-lesson-button" aria-label="Lesson {story['number']}: {escape(story['title'])}" aria-controls="ss-lesson-{key}" title="Hover or tap for this lesson">{story['number']}</summary>
<aside class="ss-prose ss-floating-card ss-lesson-card" id="ss-lesson-{key}" role="region" aria-labelledby="ss-heading-{key}" tabindex="-1"><div class="ss-card-heading"><p class="ss-kicker">{story['number']} · {story['label']}</p><button data-card-dismiss aria-label="Close lesson {story['number']}">×</button></div><h2 id="ss-heading-{key}">{story['title']}</h2><p>{story['intro']}</p><p>{story['lesson']}</p><p class="ss-try"><strong>Try it</strong> {story['try_it']}</p>{setting_note}<details class="ss-evidence"><summary>SOURCE · follow the code</summary><p>{refs}</p></details></aside></details>
<span class="ss-signal"><i></i> LIVE EXPLANATION</span><span class="ss-timing-note">Illustrative events · no measured timings</span></div>
<p class="ss-pan-hint">Scroll the diagram sideways to follow the worker →</p>
<div class="ss-visual" tabindex="0" aria-label="{escape(story['title'])} diagram; scroll horizontally on a narrow screen">{diagram}</div>
<div class="ss-controls" data-ss-controls hidden><button class="ss-play" data-ss-action="play" aria-pressed="false">▶ Play</button><button data-ss-action="step">Step</button><button data-ss-action="reset" aria-label="Replay {escape(story['title'])}">↺ Reset</button><span class="ss-control-gap"></span>{option}<button class="ss-send" data-ss-action="send">{story['action']}</button>{extra}</div>
<div class="ss-meters" data-meters><span>Accepted <b>0</b></span><span>Waiting <b>0</b></span><span>In batch <b>0</b></span><span>Rejected <b>0</b></span></div>
{('<ol class="ss-phases" aria-label="Worker loop">' + phases + '</ol>') if key != 'replicas' else ''}
<p class="ss-event" data-current-event role="status" aria-live="polite">Ready. Press Play or Step to follow the next event.</p>
{lanes}{history}
<div class="ss-inspection"><div class="ss-request-strip" data-request-list aria-label="Choose a request"></div><div data-request-detail class="ss-request-detail">Request details appear here as the scene runs.</div></div>
<details class="ss-transcript"><summary>Event transcript and portable flow</summary><ol data-transcript></ol><pre>{escape(story['ascii'])}</pre></details>
</div></section>''')
    return '\n'.join(result)


def story_contract():
    return '''<details class="ss-contract" id="animation-contract"><summary>What these animations represent</summary>
<p><b>SOURCE</b> Admission, the N/M backpressure chain, the worker loop and request cleanup follow this checkout's pinned source. <b>BOUNDARY</b> Device execution and OS lanes describe ownership; they are not GPU or Linux traces.</p>
<p>The opening API-process lesson uses two valid requests and one connected IPC slot. Component removals and the alternative event loop are thought experiments. The no-lock trace selects a possible interleaving where two probes see room before either sends; it does not claim a captured race or queue overflow. Normal playback stops after the first output chunk.</p>
<p>The diagrams use exact, small N-slot queues, artificial event durations, tiny token counts, one ordinary non-overlap worker and a simple chunked, decode-first policy. Real ZeroMQ high-water marks are approximate. MAX's batch strategy, token budgets, grammar readiness, adapters, cache availability and overlap settings can change the selected work and when results arrive.</p>
<p>The toy worker conservatively reserves prompt plus output KV pages at batch entry. MAX allocates through its cache manager and can preempt/recompute or use configured transfer paths. A request waiting for cache space is already admitted; this simulation never converts that wait into an HTTP 429.</p>
<p>M is checked against <code>all_ce_reqs</code> at the start of the iteration. The drawings keep unfinished prefill visible with waiting prefill. Decoders are separate. Both queue limits default to <code>None</code> (unbounded) in the pinned source; finite values here are teaching settings.</p>
<p>The replica scene uses an external example router, one independent GPU allocation per replica and illustrative readiness delays. It does not model tensor parallelism or automatic scaling. Output buffering is grouped across response IPC, API and transport.</p>
</details>'''


def story_markdown(page):
    lines = ['## Interactive request stories',
             'SOURCE-backed control flow; illustrative event durations, exact toy queues and conservative KV reservations. No GPU measurements.',
             'Each animation uses the full story width. Hover or focus its numbered badge for the lesson card; leaving the badge/card dismisses it. Touch taps toggle the card. Escape or the close button dismisses it.',
             'Click the circled **L** at the right edge of the sticky section menu for component colors and ownership. Outside click, Escape or Close dismisses this legend. Lesson and legend content remain available without JavaScript and when printing.',
             '### Component legend']
    lines += ['\n'.join(['| Component | Kind / owner | Meaning |', '| --- | --- | --- |'] +
                        [f'| {title} | {kind} | {description} |' for _, title, kind, description, _ in LEGEND])]
    lines += ['Admission uses an `asyncio.Lock` around probe/send in the API process. Worker startup uses a `multiprocessing.Event` readiness signal and awaits channel connection.',
              'Dashed boxes show process ownership. Request IPC has visible empty/occupied message slots, redrawn when N changes. Native runtime, kernel driver and hardware are distinct boundaries. Request-pill colors describe lifecycle state separately from component colors.']
    lines += api_intro_markdown()
    for story in STORIES[page]:
        lines += ['### ' + story['title'], story['intro'], story['lesson'],
                  'Try it: ' + story['try_it'], '```text\n' + story['ascii'] + '\n```']
    return lines
