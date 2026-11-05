"""API-process introduction for the shared offline MAX Serve readers."""

from html import escape


API_REFS = {
    "app", "server", "uvicorn_cfg", "eventloop", "http", "connected",
    "tokenizer_executor", "proxy_init", "proxy_submit", "queue_writable",
    "queue_put", "http_429", "proxy_responses", "proxy_drain", "first_chunk",
}

API_ROLES = [
    ("Uvicorn / Server", "Turns HTTP traffic into ASGI calls and writes the response to the connection.",
     "Without an HTTP/ASGI server, this configuration has no HTTP service to accept requests.", "server"),
    ("FastAPI app", "Provides MAX routes, request validation and the handlers that call the pipeline.",
     "Without an ASGI app, the configured MAX service cannot be assembled; its routes and handlers are unavailable.", "app"),
    ("uvloop event loop", "Runs ready callbacks and request coroutines on the API loop thread; resumes them when awaited work is ready.",
     "Without a running event loop, async startup, request handling and response delivery cannot progress.", "eventloop"),
    ("Admission asyncio.Lock", "Lets one coroutine at a time check channel writability and send its request context.",
     "Without this guard, two probes can observe the same space. A later send can face backpressure, losing the immediate-rejection guarantee.", "proxy_submit"),
]

API_ASCII = """Client HTTP
    |
+---v---------------- API OS PROCESS ---------------------------+
| T1: uvloop runs ready callbacks / async request tasks          |
|                                                              |
| Uvicorn / Server -> FastAPI route -> admission asyncio.Lock    |
|      T1                 T1                    T1              |
|   HTTP <-> ASGI       |              [probe -> send] ----------+--> request IPC
|                      | await                 | release        |    [one slot]
|                      v                       v                |    worker offstage
|              tokenizer executor       await response          |
|              T2 / T3: example CPU workers in this process      |
|                                                              |
| client <- HTTP/SSE <- resume request task <--------------------+--- response IPC
+--------------------------------------------------------------+
With lock: R1 checks + sends; R2 then sees full -> HTTP 429.
Without lock (possible interleaving): R1 checks; R2 checks;
R1 sends; R2's later send can wait for room.
An await can let another coroutine run on the SAME loop thread T1.
Only the CPU tokenization jobs go to T2/T3; request tasks resume on T1.
Thread IDs and executor assignments are illustrative; pool sizing/reuse vary.
The admission lock is released before waiting for model output."""


def thread_badge(thread, x, y):
    description = ("Shared API loop thread: Uvicorn, FastAPI and admission"
                   if thread == "T1" else "Example tokenizer executor thread; assignment and pool size are illustrative")
    return f'<g class="ss-api-thread-badge" data-api-thread="{thread}" transform="translate({x} {y})" role="img" aria-label="{thread}: {description}"><title>{thread}: {description}</title><rect x="-14" y="-10" width="28" height="20" rx="4"/><text y="4">{thread}</text></g>'


def api_svg():
    return f'''<svg class="ss-api-diagram" viewBox="0 0 1240 526" role="group" aria-labelledby="ss-api-title" aria-describedby="ss-api-desc">
<title id="ss-api-title">Inside the API process: HTTP server, ASGI app, event loop and admission lock</title>
<desc id="ss-api-desc">One API process contains Uvicorn's Server, FastAPI and the shared uvloop thread T1. Repeated T1 badges identify the same thread. The event loop surrounds the async work; it is not another request station. T2 and T3 illustrate executor threads running CPU tokenization jobs while request tasks await and later resume on T1. These are teaching IDs and example assignments; real pool sizing and reuse vary. R1 and R2 identify requests. The admission lock covers only the writability probe and send to the connected request IPC channel. The model worker is outside this view. Use the controls and current event for an equivalent text explanation.</desc>
<defs><marker id="ss-api-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#73877a"/></marker></defs>
<g data-component="api-process"><rect class="ss-component-box ss-process" x="112" y="22" width="876" height="484" rx="15"/><text class="ss-process-label" x="550" y="48">API PROCESS · ONE OS PROCESS</text></g>
<g data-component="loop" data-api-component="loop"><rect class="ss-component-box ss-api-loop-box" x="136" y="68" width="828" height="271" rx="12"/>{thread_badge('T1', 166, 94)}<text class="ss-node-label ss-api-loop-name" x="302" y="97" data-api-loop-name>uvloop event loop</text><text class="ss-svg-muted" x="720" y="97">ONE LOOP THREAD · READY ASYNC TASKS TAKE TURNS</text><circle class="ss-api-pulse" cx="586" cy="318" r="4"/><text class="ss-svg-muted ss-api-loop-task" x="762" y="322" data-api-loop-task>Ready to run · awaiting your first step</text></g>
<path class="ss-wire" d="M82 226 H158 M384 226 H406 M640 226 H662 M942 226 H1040" marker-end="url(#ss-api-arrow)"/>
<circle class="ss-client" cx="53" cy="163" r="25"/><path class="ss-client-icon" d="M39 169 Q53 154 67 169 M53 146 v4"/>
<text class="ss-node-label" x="53" y="121">Client</text><text class="ss-svg-muted" x="53" y="295">HTTP</text>
<g data-component="uvicorn" data-api-component="uvicorn"><title>Uvicorn's Server object handles HTTP and invokes the ASGI app on T1.</title><rect class="ss-component-box" x="158" y="126" width="226" height="170" rx="9"/>{thread_badge('T1', 361, 147)}<text class="ss-node-label" x="271" y="150">Uvicorn / Server</text><text class="ss-svg-muted" x="271" y="171">HTTP ↔ ASGI · Python object</text><text class="ss-svg-muted" x="271" y="192" data-api-component-state="uvicorn">receives / writes HTTP</text></g>
<g data-component="fastapi" data-api-component="fastapi"><title>FastAPI supplies MAX's routes, validation and async request handlers on T1.</title><rect class="ss-component-box" x="406" y="126" width="234" height="170" rx="9"/>{thread_badge('T1', 617, 147)}<text class="ss-node-label" x="523" y="150">FastAPI app</text><text class="ss-svg-muted" x="523" y="171">routes · validates · calls handler</text><text class="ss-svg-muted" x="523" y="192" data-api-component-state="fastapi">ASGI application</text></g>
<g data-component="admission" data-api-component="admission"><title>asyncio.Lock coordinates request tasks on T1 around the writable probe and send. It is neither a second event loop nor a cross-process lock.</title><rect class="ss-component-box" x="662" y="126" width="280" height="170" rx="9"/>{thread_badge('T1', 919, 147)}<text class="ss-node-label" x="802" y="150">Admission · asyncio.Lock</text><text class="ss-svg-muted" x="802" y="171" data-api-admission-summary>probe → send · one task at a time</text><text class="ss-svg-muted" x="802" y="192" data-api-lock>lock free</text><path class="ss-wire" d="M775 235 H820" marker-end="url(#ss-api-arrow)"/><text class="ss-svg-muted" x="737" y="215">probe</text><text class="ss-svg-muted" x="865" y="215">send</text><text class="ss-svg-muted" x="802" y="282" data-api-probes>R1 probe — · R2 probe —</text></g>
<g data-component="ipc"><rect class="ss-component-box" x="1040" y="135" width="176" height="189" rx="9"/><text class="ss-node-label" x="1128" y="159">Request IPC</text><text class="ss-svg-muted" x="1128" y="181">ZeroMQ · PUSH / PULL</text><g class="ss-message-slot" data-api-slot data-occupied="false" transform="translate(1128 235)"><rect x="-25" y="-17" width="50" height="34" rx="5"/><text y="4">empty</text></g><text class="ss-count" x="1128" y="282" data-api-occupancy>0 / 1 used</text><text class="ss-svg-muted" x="1128" y="305">one teaching slot</text></g>
<text class="ss-svg-muted" x="1128" y="82">IPC BOUNDARY</text><text class="ss-svg-muted" x="1128" y="348">worker offstage</text><text class="ss-svg-muted" x="1128" y="411" data-api-worker>connected · not pulling yet</text>
<path class="ss-return" data-component="ipc" d="M1128 324 V370 H271 V296" marker-end="url(#ss-api-arrow)"/><text class="ss-svg-muted" x="780" y="389">response IPC → resume task → SSE / TCP</text>
<path class="ss-wire" d="M510 296 V396 M536 396 V296" marker-end="url(#ss-api-arrow)"/>
<g data-component="tokenizer" data-api-component="tokenizer"><rect class="ss-component-box" x="406" y="396" width="234" height="81" rx="8"/><text class="ss-node-label" x="523" y="417">Tokenizer executor</text><text class="ss-svg-muted" x="523" y="436">CPU threads · same process</text><rect class="ss-api-thread-lane" x="416" y="444" width="102" height="29" rx="5"/><rect class="ss-api-thread-lane" x="528" y="444" width="102" height="29" rx="5"/>{thread_badge('T2', 436, 459)}{thread_badge('T3', 548, 459)}</g>
<text class="ss-svg-muted" x="523" y="496">CPU job → T2 / T3 · result → T1</text>
<text class="ss-node-label" x="271" y="418">When a task awaits…</text><text class="ss-svg-muted" x="271" y="440">the loop can run another ready task.</text><text class="ss-svg-muted" x="802" y="440">The lock ends after send.</text><text class="ss-svg-muted" x="802" y="460">It does not cover generation.</text>
<text class="ss-svg-muted" x="53" y="397">output</text><text class="ss-svg-muted" x="53" y="473">429</text>
<g data-api-dots aria-hidden="true"><g class="ss-request-dot ss-api-request" data-api-request="R1" transform="translate(53 219)"><rect x="-21" y="-12" width="42" height="24" rx="12" fill="#edf0e2" stroke="#778662"/><text y="4" fill="#414f33">R1</text></g><g class="ss-request-dot ss-api-request" data-api-request="R2" transform="translate(53 261)"><rect x="-21" y="-12" width="42" height="24" rx="12" fill="#edf0e2" stroke="#778662"/><text y="4" fill="#414f33">R2</text></g></g>
</svg>'''


def render_api_intro():
    choices = [("normal", "All present"), ("no-server", "No Uvicorn"),
               ("no-app", "No FastAPI"), ("no-loop", "No event loop"),
               ("no-lock", "No admission lock"), ("asyncio", "Use asyncio loop")]
    buttons = ''.join(f'<button data-api-mode="{key}" aria-pressed="{str(key == "normal").lower()}">{label}</button>' for key, label in choices)
    rows = ''.join(f'<tr><th scope="row">{name}</th><td>{why}</td><td>{missing}</td><td>{{{{ref:{ref}}}}}</td></tr>' for name, why, missing, ref in API_ROLES)
    return '''<section class="ss-story ss-api-story" id="story-api" data-api-story aria-label="Start inside the API process">
<div class="ss-experiment"><div class="ss-experiment-top">
<details class="ss-lesson" data-ss-lesson open><summary class="ss-lesson-button" aria-label="Lesson 01: Inside the API process" aria-controls="ss-lesson-api" title="Hover or tap for this lesson">01</summary>
<aside class="ss-prose ss-floating-card ss-lesson-card" id="ss-lesson-api" role="region" aria-labelledby="ss-heading-api" tabindex="-1"><div class="ss-card-heading"><p class="ss-kicker">01 · Start inside the API process</p><button data-card-dismiss aria-label="Close lesson 01">×</button></div><h2 id="ss-heading-api">One process. Different jobs.</h2>
<p>Uvicorn speaks HTTP. FastAPI supplies MAX's routes and handlers. The uvloop event loop runs their ready async work on the same loop thread. CPU tokenization uses executor threads.</p>
<p><b>T1</b> marks that shared loop thread in every component. <b>T2 / T3</b> mark example tokenizer executor threads. Only CPU jobs go to those threads; async request tasks resume on T1. <b>R1 / R2</b> remain request IDs. Thread IDs and executor assignments are illustrative; real pool sizing and reuse vary.</p>
<p>The admission guard is an <code>asyncio.Lock</code>: coroutine mutual exclusion around <b>probe → send</b>. It is a synchronization primitive, not another event loop or a process-shared OS lock. An <code>await</code> can let a second task run even on one thread.</p>
<p class="ss-try"><strong>Try it</strong> Play with all roles present. Then choose <b>No admission lock</b> and compare R2. Remove the other roles to see why the API service needs them.</p>
<p class="ss-setting-note">Two valid requests, a connected worker and one illustrative IPC slot. The worker initially stops pulling. Native/GPU execution is outside this view.</p>
<details class="ss-evidence"><summary>SOURCE · follow the code</summary><p>{{ref:server}} · {{ref:app}} · {{ref:eventloop}} · {{ref:tokenizer_executor}} · {{ref:proxy_init}} · {{ref:proxy_submit}} · {{ref:queue_writable}} · {{ref:queue_put}} · {{ref:http_429}} · {{ref:proxy_drain}}</p></details>
</aside></details><span class="ss-signal"><i></i> INSIDE THE API PROCESS</span><span class="ss-timing-note">2 requests · 1 teaching slot</span></div>
<div class="ss-api-modes" data-api-controls hidden role="group" aria-label="Remove or replace an API component"><span>Try one change</span>''' + buttons + '''</div>
<p class="ss-api-purpose" data-api-purpose>All roles present. Follow two requests sharing one event-loop thread.</p>
<p class="ss-pan-hint">Scroll sideways to follow HTTP → app → admission → IPC</p>
<div class="ss-visual" tabindex="0" aria-label="API-process diagram; scroll horizontally on a narrow screen">''' + api_svg() + '''</div>
<div class="ss-controls" data-api-controls hidden><button class="ss-play" data-api-action="play" aria-pressed="false">▶ Play</button><button data-api-action="step">Step</button><button data-api-action="reset">↺ Reset</button><span class="ss-control-gap"></span><span class="ss-api-progress" data-api-progress>Ready · event 0</span></div>
<div class="ss-api-requests" aria-label="Request states"><span><b>R1</b> <span data-api-state="R1">at client</span></span><span><b>R2</b> <span data-api-state="R2">at client</span></span></div>
<p class="ss-event" data-current-event role="status" aria-live="polite">Ready. Press Play or Step. The worker is connected, but is not pulling requests yet.</p>
<details class="ss-transcript ss-api-transcript"><summary>Why each role exists · missing-role outcomes · source and portable flow</summary>
<p><b>SOURCE</b> The normal path follows the pinned Python control flow. <b>BOUNDARY</b> Removing or replacing a component is a thought experiment, not a MAX configuration switch. The no-lock scene shows one possible interleaving; it does not claim a measured race or queue overflow. Real ZeroMQ high-water marks are approximate.</p>
<div class="ss-api-role-table" tabindex="0" aria-label="API roles and missing-role outcomes"><table><thead><tr><th>Role</th><th>Why it exists</th><th>If absent, with no replacement</th><th>Source</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p><b>Implementations can be replaced.</b> Another HTTP/ASGI server or ASGI app can fill the first two roles if integrated appropriately. A compatible asyncio loop can replace uvloop with the needed startup/configuration changes. This checkout explicitly calls <code>uvloop.run(serve())</code> and selects <code>loop="uvloop"</code>; uninstalling uvloop alone does not enable a fallback. {{ref:http}} · {{ref:uvicorn_cfg}}</p>
<p>Request connectivity is established before HTTP serving: {{ref:connected}}. Returned worker output wakes the API response path: {{ref:proxy_responses}} · {{ref:proxy_drain}} · {{ref:first_chunk}}.</p>
<p>BOUNDARY · Official references: <a href="https://docs.python.org/3.13/library/asyncio-eventloop.html" target="_blank" rel="noopener">Python event loop</a> · <a href="https://docs.python.org/3.13/library/asyncio-sync.html#lock" target="_blank" rel="noopener">asyncio.Lock</a> · <a href="https://uvicorn.dev/concepts/event-loop/" target="_blank" rel="noopener">Uvicorn loop implementations</a>.</p>
<ol data-api-transcript></ol><pre>''' + escape(API_ASCII) + '''</pre></details>
</div></section>'''


def api_intro_markdown():
    return [
        '### 01 · Inside the API process',
        'The first animation isolates the API process. Play, Step or Reset two requests sharing one event-loop thread and one illustrative IPC slot. Choose All present, No Uvicorn, No FastAPI, No event loop, No admission lock, or Use asyncio loop. Hover/focus/tap 01 for its lesson; the existing L legend explains ownership.',
        'Thread badges appear only in this opening animation: T1 is repeated on uvloop, Uvicorn, FastAPI and admission to show their shared API loop thread. T2/T3 identify example executor threads running CPU tokenization jobs; the async request tasks resume on T1. R1/R2 continue to identify requests. Thread IDs and executor assignments are illustrative; real pool sizing and reuse vary.',
        '\n'.join(['| Role | Why it exists | If absent, with no replacement |', '| --- | --- | --- |'] +
                  [f'| {name} | {why} | {missing} |' for name, why, missing, _ in API_ROLES]),
        'SOURCE: The API loop runs ready coroutines; CPU tokenization uses executor threads in the same process. Admission uses `asyncio.Lock` around the writability probe and send, then releases it before awaiting output. The model worker is offstage and connected before HTTP serving.',
        'BOUNDARY: Role removals and replacement are thought experiments, not supported MAX flags. The no-lock trace is one possible interleaving, not a measured failure or queue overflow. With the lock, the second request observes a full channel and gets HTTP 429. Without it, two earlier positive probes can leave the later send waiting for room. The exact one-slot capacity is illustrative.',
        'A compatible asyncio loop can replace uvloop with startup/configuration changes. This pinned checkout calls `uvloop.run(serve())` and selects `loop="uvloop"`; simply removing the uvloop package does not provide a fallback. HTTP-server and ASGI-app implementations can also be replaced if their roles are supplied.',
        'The normal animation stops after R1 delivers its first output chunk; subsequent chunks repeat the response wake/resume/send cycle. It does not simulate the offstage worker or GPU in this lesson.',
        '```text\n' + API_ASCII + '\n```',
    ]
