# MAX Serve interactive stories: validation

Date: November 5, 2026 (Asia/Colombo). Machine-readable reports use UTC.

Artifacts: [requests and scheduling](../link_htmls/max_serve_os_components_and_flow.html)
and [workers, drivers and scaling](../link_htmls/max_serve_os_driver_and_scaling.html).
Source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

`CPU-RUN`: JavaScript model, headless Chromium, source and document checks.
Model/device execution: none. The source context is the Llama 2 7B walkthrough;
the stories use small synthetic requests, not model weights or real inference.

## Template and scope

Both pages open with an API-process lesson, followed by the seven existing
request and systems experiments (five lessons on components, four on systems).
Requests visibly move through API tokenization, the admission gate,
IPC, pending prefill and active batches. Output tokens use a separate return
path. Each experiment uses the full story width. Its numbered badge reveals a
lesson card on hover or focus; leaving the badge/card dismisses it. Touch taps
toggle it. The circled L at the right edge of the sticky menu opens a separate
color legend. It identifies process ownership, shared event-loop threads,
tokenizer executor threads, ZeroMQ Unix IPC, the admission coroutine lock,
the startup process Event, native code, the OS driver and GPU hardware.
N slots visibly fill, drain and resize with the queue setting. Colors match
across the legend, pipeline diagrams and independent-replica diagram.
Moving requests and their ID buttons support pointer and keyboard
selection. Playback pauses when the story leaves the viewport or the tab hides.

The API introduction contains Uvicorn/Server, FastAPI, an enclosing event-loop
thread, tokenizer executor threads and the admission `asyncio.Lock`. Its six
modes compare all roles present, missing HTTP server, missing ASGI app, no
running event loop, a possible no-lock interleaving, and a compatible asyncio
loop replacement. The worker is offstage. Source links include the startup
calls, writable probe, asynchronous send and lock-protected admission.
Only this first animation has thread badges: T1 repeats on the loop, Uvicorn,
FastAPI and admission; T2/T3 label two example tokenizer executor lanes. Request
markers enter those lanes during CPU tokenization and return to the API path.
The lesson card, SVG description and portable flow distinguish thread IDs from
R1/R2 request IDs and state that executor assignments and pool sizing vary.

The three server/request/OS-thread state machines remain intact. Detailed maps,
source cells, cards and the advanced simulator are available in the reference
atlas. Existing fragment links expand that atlas. Without JavaScript, the
atlas starts open and portable flows and lesson text remain readable.

The interaction approach was informed by [Sam Who's Load Balancing](https://samwho.dev/load-balancing/).
The MAX-specific diagrams, layout and simulation are original. The source
article's initial busy-server drop rule is not used for MAX admission.

## Simulation contract

- The API scene starts two valid requests with one connected teaching slot.
  With the lock, R2 checks after R1 sends and gets HTTP 429; the lock ends before
  R1 awaits output. Without the lock, the selected possible interleaving lets
  both probes see room, after which R2's send can wait. This is not a measured
  race or a claim of queue overflow. Normal playback stops after R1's first
  output chunk, not after the complete stream.
- Missing roles and replacement are BOUNDARY thought experiments, not supported
  MAX flags. A missing ASGI app prevents assembly of this configured service;
  the example does not invent a 404/500. A compatible asyncio loop requires
  startup/configuration changes because this revision explicitly imports and
  calls uvloop; uninstalling the package alone is not a fallback.
- Each playback step is a logical event. Beats are not seconds; no timing,
  throughput, memory-size or GPU-utilization prediction is made.
- N is an exact toy queue. MAX's ZeroMQ HWM is approximate. Both N and M default
  to `None` in the pinned source; the pressure scene explicitly sets finite limits.
- M is checked against unfinished CE/prefill work at the worker's drain point.
  A selected pass finishes before this ordinary worker checks for another batch.
- Batch selection uses one simple chunked, decode-first policy. MAX also has
  configurable priorities, grammar/adapter constraints, overlap and other paths.
- The toy cache conservatively reserves prompt plus output pages. MAX's real
  per-pass allocation, preemption and transfer algorithms are not reproduced.
- Response IPC, API and transport buffering are grouped. Pending publication
  is separate from readable output, so output cannot be delivered before publish.
- The replica scene uses an external round-robin router and illustrative warmup.
  It models separate workers and GPU allocations, not tensor parallelism or an
  automatic scaling controller.

## Commands and results

Run from the repository root. The Playwright installation is local to this host.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/validate.py
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_stories.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_api.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_stories_browser.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_os_components_browser.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_os_systems_browser.cjs
git diff --check
```

Behavioral checks: **PASS**, eight scenario groups. Admission continues during
execution; a later request never joins an already selected pass. A second batch
slot allows R2 to join before R1 finishes. M fills before the overload example
rejects at N; increasing N admits the same burst. Chunked prefill obeys the token
budget and produces no token until its prompt completes. Slow output survives KV
release. Cancellation waits for safe ownership release. KV pressure delays
admitted work without becoming HTTP 429. Routing excludes initializing replicas
and preserves existing request ownership.

API model checks: **PASS**, five groups. They verify suspended tokenization,
lock ownership through probe/send, release before waiting for output, R2's 429,
R1's first chunk, missing-role stops without invented HTTP codes, the possible
no-lock interleaving without queue overflow, compatible-loop outcomes, and
stable snapshots and replay isolation.

Browser checks: **PASS**, thirteen groups and thirty-two rendered-contrast
audits. They cover the new API lesson on both pages, the seven existing stories,
hover/click disclosures and rendered contrast.
Coverage includes controls, moving dots, request selection, 429 inspection,
queue-setting changes, lifecycle/source links, slow readers, replica readiness,
320/390-pixel layouts, touch, keyboard, no JavaScript, reduced motion, pause and
offscreen pause. Additional checks cover full-width layout, dismissing hover
cards, keyboard focus after scrolling, click-only legend opening without
changing the scroll position, source-modal focus return, matching component
colors, N-slot occupancy and position, queue drain/reset, mobile card bounds,
and print expansion/restoration. The new API checks exercise all six modes,
the actual rendered slot and request states, keyboard/touch controls, replay,
source links, movement, pause and the no-JavaScript outcome table.
The machine-readable report records the
completed group and contrast-audit counts, errors and runtime requests.

Existing reference regressions: **PASS**. Component coverage includes all 15
components, 26 handoffs, 19 states, eight paths and 45 source targets. Systems
coverage includes 48 step cards, 22 object cards, 58 source targets and 11
complete source cells. The three browser suites report no page errors or
external runtime requests.

Machine-readable evidence: [new browser checks](serve_stories_browser.json),
[API model checks](SERVE_API_MODEL_CHECKS.json),
[component regressions](os_components_browser.json),
[systems regressions](os_systems_browser.json) and [source checks](source_checks.json).
Browser screenshots for the new request, overload, replica and mobile stories
are written to `/tmp/max-serve-*.png` for local visual review.
