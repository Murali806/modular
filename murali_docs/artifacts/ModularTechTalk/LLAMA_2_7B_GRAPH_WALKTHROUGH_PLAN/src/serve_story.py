"""Small interactive essays shared by the two MAX Serve HTML readers."""

from html import escape


STORIES = {
    "components": [
        dict(id="one", number="01", label="Start with one request", title="Accepted. Then scheduled.",
             intro="A valid request R1 arrives at a ready MAX Serve instance. Follow it through tokenization, the admission gate, prefill and repeated decode passes.",
             lesson="Admission gives R1 a place on the request channel. The worker later chooses when it can execute. Output tokens return to the client while generation continues.",
             try_it="Press Play. Click a request to follow its state; Step advances one event.",
             action="Send another request", refs=["proxy_submit", "scheduler_iteration", "first_chunk"],
             ascii="Client → API/tokenize → admission → request IPC → worker pending\n                                                  ↓\nClient ← TCP/SSE ← API ← response IPC ← GPU/native ← selected batch\n                                      ↑ prefill, then repeated decode"),
        dict(id="busy", number="02", label="Now the worker is busy", title="There is room to wait.",
             intro="R1 is already generating. Send R2 while the worker is inside a model pass. Watch where R2 waits and when it joins a batch.",
             lesson="With one batch slot, R2 waits for R1 to finish. With two slots and enough KV space, R2 can join a later pass while R1 is still decoding. A running pass keeps its selected members.",
             try_it="Send R2, then replay with two batch slots. Compare the waiting location.",
             action="Send R2", refs=["proxy_submit", "scheduler_pending", "construct_batch"],
             option=("Batch slots", "batch", [(1, "1 slot"), (2, "2 slots")]),
             ascii="API accepts R2 ──→ IPC  | R1's selected pass is running\nnext worker iteration ─→ pending → batch eligibility\nB=1: R2 waits for a slot   B=2: R2 may join a subsequent pass"),
        dict(id="pressure", number="03", label="Keep sending work", title="Watch backpressure reach the gate.",
             intro="Keep R1 decoding, then send a burst. The worker's pending-prefill queue M fills. It stops pulling from IPC. The request channel N then fills too.",
             lesson="When the API proxy finds that connected channel unwritable, it raises RequestQueueFull. The HTTP handler returns 429 with Retry-After: 1 before streaming begins. Already admitted requests stay in the system.",
             try_it="Send a burst of six. Increase N and replay the same burst to see more waiting room.",
             action="Send a burst of 6", refs=["queue_settings", "scheduler_pending", "proxy_submit", "http_429"],
             option=("IPC waiting room N", "queue", [(2, "2 requests"), (6, "6 requests")]),
             ascii="Long decode holds batch slot\n       ↓\nM full → stop pulling → N fills → API probe fails → HTTP 429\nWork completes → pending drains → IPC drains → admission has room again"),
        dict(id="scheduler", number="04", label="Look inside the loop", title="A batch is one pass of work.",
             intro="R1 is decoding a response. R2 has a longer prompt. The scheduler builds each pass from eligible work, within batch, token and KV limits.",
             lesson="This example gives decode one input token per request and uses the remaining budget for chunked prefill. A partial prompt returns for another pass. Completing prefill yields the first output token; ordinary decode then adds one token per selected request per pass.",
             try_it="Step through the five highlighted phases. Compare the pass history below the diagram.",
             action="Add a short prompt", refs=["worker_run", "scheduler_iteration", "scheduler_execute", "construct_batch", "preempt_request"],
             option=("Input tokens per pass", "budget", [(8, "8 tokens"), (16, "16 tokens")]),
             ascii="repeat: pull within M → construct batch → pipeline.execute()\n             ↑                                  ↓\n       publish by request ID ← advance state / release finished requests\nno eligible work → NO_PROGRESS → yield/backoff → check again"),
    ],
    "systems": [
        dict(id="clocks", number="01", label="Put the owners side by side", title="Busy belongs to an owner.",
             intro="While the model worker executes a pass, the API can tokenize and admit another request. The OS schedules runnable CPU threads; MAX chooses model work; the GPU executes submitted operations.",
             lesson="An HTTP task waiting for output is suspended until data is available. It does not require a dedicated CPU thread. GPU completion and socket readiness are different events. The lanes below show logical ownership, not a measured thread trace.",
             try_it="Send another request during Execute. Compare the API lane with the worker and device lanes.",
             action="Send another request", refs=["worker_run", "eventloop", "tokenizer_cpu", "scheduler_execute", "session_native"],
             ascii="OS:     gives CPU time to runnable threads in both processes\nAPI:    tokenize → admit → await output → SSE\nWorker: drain → select → execute → advance → publish → repeat\nGPU:                     submitted compute → result\nLinux socket readiness wakes I/O; native runtime manages device dependencies."),
        dict(id="reader", number="02", label="Separate generation from delivery", title="The GPU finishes. The stream remains.",
             intro="Make the client read slowly. Output tokens collect after generation, while the worker can release that request's KV ownership and start other work.",
             lesson="Response IPC, API output queues and TCP transport each have buffering. This animation groups them into one output buffer. A slow client can raise egress memory and latency; an admitted stream does not turn into a new HTTP 429 response.",
             try_it="Play until R1 has generated all five tokens. Compare KV ownership with buffered output.",
             action="Send another request", refs=["proxy_responses", "proxy_drain", "release_request", "sse_response"],
             option=("Client reads", "reader", [(12, "Slowly"), (1, "Normally")]),
             ascii="GPU completes → release request KV ownership (pool remains allocated)\n       ↓\nresponse IPC → API output buffer → TCP → slow client\n              tokens can remain after generation finishes"),
        dict(id="replicas", number="03", label="Give new work another destination", title="Another replica, another worker.",
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
    }


def pipeline_svg(key):
    nodes = [(200, "API", "tokenize", "api"), (420, "Request IPC", "waiting room N", "ipc"),
             (586, "Pending prefill", "worker cap M", "pending"), (736, "Batch members", "slots + KV", "batch")]
    boxes = "".join(f'<g class="ss-station" data-station="{kind}"><rect x="{x-58}" y="108" width="116" height="137" rx="10"/><text x="{x}" y="129" class="ss-node-label">{title}</text><text x="{x}" y="265" class="ss-svg-muted" data-cap="{kind}">{subtitle}</text><text x="{x}" y="232" class="ss-count" data-count="{kind}">0</text></g>' for x, title, subtitle, kind in nodes)
    return f'''<svg class="ss-pipeline" viewBox="0 0 980 392" role="group" aria-labelledby="ss-title-{key}" aria-describedby="ss-desc-{key}">
<title id="ss-title-{key}">Animated requests from API admission to worker batches and client output</title>
<desc id="ss-desc-{key}">Request IDs move through separate queues. Read the current event and request details below for an equivalent text explanation.</desc>
<defs><marker id="ss-arrow-{key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#81918b"/></marker></defs>
<rect class="ss-process" x="126" y="46" width="212" height="241" rx="14"/><text class="ss-process-label" x="232" y="73">API PROCESS</text>
<rect class="ss-process" x="510" y="46" width="300" height="241" rx="14"/><text class="ss-process-label" x="660" y="73">MODEL-WORKER PROCESS</text>
<path class="ss-wire" d="M83 176 H876" marker-end="url(#ss-arrow-{key})"/>
<circle class="ss-client" cx="62" cy="176" r="27"/><path class="ss-client-icon" d="M48 182 Q62 167 76 182 M62 159 v4"/>
<text class="ss-node-label" x="62" y="125">Client</text>
{boxes}
<g class="ss-gate" data-gate="open"><path d="M294 146 L311 176 L294 206 L277 176 Z"/><text x="294" y="181">✓</text></g>
<text class="ss-svg-muted" x="294" y="225">gate</text>
<g class="ss-gpu"><rect x="846" y="112" width="117" height="130" rx="12"/><text class="ss-node-label" x="904" y="139">Native / GPU</text><text class="ss-count" x="904" y="180" data-gpu>Idle</text><text class="ss-svg-muted" x="904" y="209">selected pass</text></g>
<path class="ss-return" d="M904 244 V304 H62 V207" marker-end="url(#ss-arrow-{key})"/>
<text class="ss-svg-muted" x="498" y="329">response IPC → API / SSE → TCP → client</text>
<text class="ss-svg-muted" x="88" y="363">delivered</text><text class="ss-svg-muted" x="285" y="363">429 rejected</text>
<g data-request-dots></g><g data-output-dots aria-hidden="true"></g>
</svg>'''


def fleet_svg():
    return '''<svg class="ss-fleet" viewBox="0 0 980 392" role="group" aria-label="An external router sends new requests to ready, independent replicas A and B">
<path class="ss-wire" d="M128 190 H282 V123 H395 M282 190 V285 H395"/>
<circle class="ss-client" cx="92" cy="190" r="36"/><text class="ss-node-label" x="92" y="194">Clients</text>
<rect class="ss-process" x="190" y="152" width="169" height="77" rx="10"/><text class="ss-node-label" x="274" y="182">External router</text><text class="ss-svg-muted" x="274" y="208">round robin · example</text>
<g data-fleet-replica="0"><rect class="ss-process" x="395" y="48" width="548" height="133" rx="13"/><text class="ss-node-label" x="478" y="77">Replica A</text><text class="ss-svg-muted" x="760" y="77" data-fleet-state="0">Ready · its own worker and GPU</text></g>
<g data-fleet-replica="1"><rect class="ss-process" x="395" y="213" width="548" height="133" rx="13"/><text class="ss-node-label" x="478" y="244">Replica B</text><text class="ss-svg-muted" x="760" y="244" data-fleet-state="1">Not added</text></g>
<g data-fleet-detail="0"></g><g data-fleet-detail="1"></g><g data-request-dots></g>
<text class="ss-svg-muted" x="663" y="374">Every request keeps one replica owner throughout its stream.</text>
</svg>'''


def render_stories(page):
    result = []
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
        result.append(f'''<section class="ss-story" id="story-{key}" data-serve-story="{key}">
<div class="ss-prose"><p class="ss-kicker"><span>{story['number']}</span> {story['label']}</p><h2>{story['title']}</h2><p>{story['intro']}</p><p>{story['lesson']}</p><p class="ss-try"><strong>Try it</strong> {story['try_it']}</p>{setting_note}<details class="ss-evidence"><summary>SOURCE · follow the code</summary><p>{refs}</p></details></div>
<div class="ss-experiment"><div class="ss-experiment-top"><span class="ss-signal"><i></i> LIVE EXPLANATION</span><span>Illustrative events · no measured timings</span></div>
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
<p>The diagrams use exact, small N-slot queues, artificial event durations, tiny token counts, one ordinary non-overlap worker and a simple chunked, decode-first policy. Real ZeroMQ high-water marks are approximate. MAX's batch strategy, token budgets, grammar readiness, adapters, cache availability and overlap settings can change the selected work and when results arrive.</p>
<p>The toy worker conservatively reserves prompt plus output KV pages at batch entry. MAX allocates through its cache manager and can preempt/recompute or use configured transfer paths. A request waiting for cache space is already admitted; this simulation never converts that wait into an HTTP 429.</p>
<p>M is checked against <code>all_ce_reqs</code> at the start of the iteration. The drawings keep unfinished prefill visible with waiting prefill. Decoders are separate. Both queue limits default to <code>None</code> (unbounded) in the pinned source; finite values here are teaching settings.</p>
<p>The replica scene uses an external example router, one independent GPU allocation per replica and illustrative readiness delays. It does not model tensor parallelism or automatic scaling. Output buffering is grouped across response IPC, API and transport.</p>
</details>'''


def story_markdown(page):
    lines = ['## Interactive request stories',
             'SOURCE-backed control flow; illustrative event durations, exact toy queues and conservative KV reservations. No GPU measurements.']
    for story in STORIES[page]:
        lines += ['### ' + story['title'], story['intro'], story['lesson'],
                  'Try it: ' + story['try_it'], '```text\n' + story['ascii'] + '\n```']
    return lines
