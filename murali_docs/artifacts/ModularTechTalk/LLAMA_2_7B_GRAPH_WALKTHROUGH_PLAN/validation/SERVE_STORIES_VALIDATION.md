# MAX Serve interactive stories: validation

Date: November 5, 2026 (Asia/Colombo). Machine-readable reports use UTC.

Artifacts: [requests and scheduling](../link_htmls/max_serve_os_components_and_flow.html)
and [workers, drivers and scaling](../link_htmls/max_serve_os_driver_and_scaling.html).
Source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

`CPU-RUN`: JavaScript model, headless Chromium, source and document checks.
Model/device execution: none. The source context is the Llama 2 7B walkthrough;
the stories use small synthetic requests, not model weights or real inference.

## Template and scope

Seven numbered experiments replace the maps and large lab as the first reading
experience. Requests visibly move through API tokenization, the admission gate,
IPC, pending prefill and active batches. Output tokens use a separate return
path. Each experiment keeps its explanation, controls and request inspector
together. Moving requests and their ID buttons support pointer and keyboard
selection. Playback pauses when the story leaves the viewport or the tab hides.

The three server/request/OS-thread state machines remain intact. Detailed maps,
source cells, cards and the advanced simulator are available in the reference
atlas. Existing fragment links expand that atlas. Without JavaScript, the
atlas starts open and portable flows and lesson text remain readable.

The interaction approach was informed by [Sam Who's Load Balancing](https://samwho.dev/load-balancing/).
The MAX-specific diagrams, layout and simulation are original. The source
article's initial busy-server drop rule is not used for MAX admission.

## Simulation contract

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

Browser checks: **PASS**, seven groups and eight rendered-contrast audits.
Coverage includes controls, moving dots, request selection, 429 inspection,
queue-setting changes, lifecycle/source links, slow readers, replica readiness,
320/390-pixel layouts, touch, keyboard, no JavaScript, reduced motion, pause and
offscreen pause. Zero page errors and zero external runtime requests.

Existing reference regressions: **PASS**. Component coverage includes all 15
components, 26 handoffs, 19 states, eight paths and 37 source targets. Systems
coverage includes 48 step cards, 22 object cards and 11 complete source cells.

Machine-readable evidence: [new browser checks](serve_stories_browser.json),
[component regressions](os_components_browser.json),
[systems regressions](os_systems_browser.json) and [source checks](source_checks.json).
Browser screenshots for the new request, overload, replica and mobile stories
are written to `/tmp/max-serve-*.png` for local visual review.
