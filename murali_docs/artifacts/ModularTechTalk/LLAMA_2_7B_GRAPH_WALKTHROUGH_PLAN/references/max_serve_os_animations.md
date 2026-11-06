# MAX Serve OS animations: requirements and accepted decisions

Consolidated November 6, 2026 from the user's prompts, accepted revisions,
current authoring sources and recorded validation. Primary target:
[max_serve_os_driver_and_scaling.html](../link_htmls/max_serve_os_driver_and_scaling.html).
Shared lessons also appear in
[max_serve_os_components_and_flow.html](../link_htmls/max_serve_os_components_and_flow.html).

Explicit user requirements describe the desired experience. Accepted design
choices describe how the current artifact satisfies them. Source invariants
describe the pinned implementation. Illustrative scenarios and numbers remain
changeable teaching choices. A later user instruction can revise the design;
recheck source invariants when changing the source revision.

## User requirements and interaction

- Teach through animation. The original dense readers were not helpful;
  their "Three different kinds of state" transitions were helpful and should
  remain available. Show when requests are admitted/rejected, what waits when
  the worker is busy, and how the scheduler chooses and repeats work.
- Use Sam Who's `https://samwho.dev/load-balancing/` as the requested
  interaction inspiration: moving requests, visible queues, decisions and
  consequences. MAX's own implementation determines the behavior shown.
- Use the accepted light article theme for these readers. Give the animation
  the available width and remove the persistent right explanation column.
- Inside each animation, keep only a compact numbered lesson badge such as
  01. Hover or keyboard focus reveals its explanation card; leaving the badge
  and card closes it. Moving from the badge into the card keeps it readable.
  Support touch toggling, a Close button and Escape dismissal.
- Place a circled, slightly larger **L** button at the right edge of the sticky
  top/menu bar. Clicking opens the component legend card; outside click,
  Close or Escape dismisses it. The prompt's "clock L" was implemented and
  accepted as "click L". Hover alone does not open the legend.
- Give each component a distinct color, consistent between its diagram and
  legend entry. Keep request lifecycle colors separate. Include compact
  ownership/type descriptions and real synchronization or IPC mechanisms.
- Show queue slots with empty/occupied state, occupancy/capacity and draining.
  Reset restores queue and request state. Visible slot counts are teaching
  settings, not a promise of exact ZeroMQ high-water-mark enforcement.

Accepted supporting interactions: Play/Pause, Step and Reset/Replay; no
autoplay; focused scenarios/settings; deterministic request state; pause when
offscreen or the document is hidden; respect reduced motion. Keep hover/focus
source cards, a readable event transcript and portable ASCII equivalents.
Support direct-file/offline reading, no-JavaScript reading, printing, keyboard
navigation and touch. On narrow screens, contain horizontal SVG scrolling
within the animation rather than overflowing the document.

Keep changing status/event text easy to read. The accepted local styling uses
larger/bold event text and tinted backgrounds fitted to the content, including
the API and worker loop-status backplates. Keep explanatory prose compact;
details belong in cards or the collapsed reference material.

## Architecture and legend semantics

Draw components nested in their actual owner. Arrows distinguish function
calls, async wakeups, IPC messages and device submission. The event loop runs
ready work; it is not a station that every request physically passes through.

| Component | Kind / ownership to describe in its card |
| --- | --- |
| API process | OS process containing HTTP serving, app, API loop and tokenizer executor |
| Uvicorn / Server | Uvicorn's Python `Server` object; HTTP/ASGI role inside the API process |
| FastAPI app | ASGI application, routes, validation and async handlers on the API loop |
| uvloop | Event loop; API and worker each own one in their separate process |
| Admission | Async coroutine logic guarded by `asyncio.Lock` on the API loop |
| Tokenizer executor | CPU thread pool in the API process; synchronous jobs submitted to the default executor |
| Request/response/cancel IPC | Local ZeroMQ PUSH/PULL channels over Unix `ipc://` sockets; visible message slots |
| Model worker / scheduler | Separately spawned OS process; scheduler repeats on its worker loop |
| CE / TG request state | Ordered worker-owned Python collections, owned through `ReplicaRequests` by the constructor |
| Batch constructor | Python object selecting eligible contexts for one model pass on the worker loop |
| KV manager | Host bookkeeping plus device cache pool; request block ownership and reuse |
| Model pipeline | Prepares selected inputs, invokes the compiled model and returns results |
| MAX runtime / user driver | Native code in the worker; device submission/completion boundary |
| OS GPU driver | Kernel/device-driver boundary |
| GPU device | Hardware execution and device data, including weights and KV cache |

Uvicorn and its Server object may share a legend entry; do not invent another
server process. Distinguish a dashed OS-process boundary from the constructor's
solid Python-object boundary. Show synchronization separately: `asyncio.Lock`
coordinates API coroutines around probe/send; `multiprocessing.Event` signals
worker startup readiness. KV ownership is resource management, not a mutex.
Native internal threads/locks and OS/device scheduling are `BOUNDARY` unless
there is actual evidence. Do not invent a semaphore, mutex or native thread ID.

## Shared opening lessons

### 01 — Inside the API process

User requirement: isolate the API block, explain FastAPI, Uvicorn/Server,
uvloop and admission async logic, and show why each role is needed or what
fails when it is absent. This is the only animation with small two-character
thread badges such as T1, T2 and T3.

Accepted behavior:

- Repeat T1 on Uvicorn, FastAPI, uvloop and admission to mean the **same API
  loop thread**. T2/T3 illustrate tokenizer executor lanes in the same process;
  R1/R2 remain request IDs. Executor assignments and pool size are illustrative.
- Show `run_in_executor(None, …)` handing a synchronous tokenizer job to the
  default executor. The request coroutine awaits and resumes on T1; reusable
  executor workers have no permanent request assignment.
- The tokenizer hover/focus/touch card explains the pool, handoff, reuse and
  version-dependent count. The current Python 3.13 example is
  `min(32, (os.process_cpu_count() or 1) + 4)`: eight available CPUs allow up to
  twelve workers, created on demand. Preserve its version qualifier and
  official-source attribution; it is not a MAX-configured two-thread pool.
- Cover All present, No Uvicorn, No FastAPI, No event loop, No admission lock,
  and Use asyncio loop. These are teaching thought experiments, not MAX CLI
  switches. Compatible server/app/loop implementations can fill those roles
  with integration/configuration changes. Removing uvloop alone does not
  enable a fallback in the pinned checkout.
- Admission is async logic with an `asyncio.Lock`, not another loop or
  process-shared OS lock. Lock only the writable probe and send; release the
  lock before awaiting model output.
- The no-lock mode illustrates possible stale-probe interleaving and a send
  waiting for capacity. It is not a captured race or guaranteed overflow/429.
  Do not fabricate an HTTP response for an absent server/app/loop role.

### 02 — Inside the model worker · separate process

User requirement: add a second focused animation like 01 for the worker's
internal work. Include its own uvloop, request IPC, scheduler, CE/TG request
state, constructor, KV manager, pipeline, runtime/driver/GPU and result path.
Thread badges are confined to 01.

Accepted scenarios are Two requests, GPU busy, KV pressure and No requests.
Show pull → construct batch → execute → advance/release → publish by request
ID → repeat. No eligible work returns `NO_PROGRESS` and causes caller
backoff/yield. Membership of an executing pass stays fixed; a late request
waits for a later pass. Partial prefill produces no output token until the
prompt is complete. KV pressure on already-admitted work means a scheduling
wait, not a new HTTP rejection. Releasing a request makes its cache blocks
reusable while the backing device pool remains allocated.

Use one ordinary worker without overlap. Tiny capacities, conservative KV
reservations and event durations are illustrative, not measured GPU timing.

### Request sets versus batch constructor: clarification

The user asked why these need separate boxes and whether they could be one,
with "no edits allowed" for that explanation turn. The distinction is
pedagogical: tracked work across passes versus selected work for this pass.
It does not introduce another thread, process or IPC hop.

The source `TextBatchConstructor` owns `ReplicaRequests`, including ordered
`ce_reqs` and `tg_reqs`; nesting them reflects that ownership. The later
explicit request authorized building 03 with this combined ownership view.
The existing 02 boxes remain teaching views. Do not infer that the earlier
question authorized a separate redesign of 02 or permanently banned edits.

### 03 — Inside the batch constructor

User requirement: a complete third animation of the constructor's work and
logic. Show a solid Python-object boundary in the worker context, with owned
CE/TG sets inside it. The constructor chooses context references; the pipeline
outside it prepares tensors, executes and samples. It introduces no new
thread or IPC hop.

Accepted selection flow and invariants:

- Ordered request state → readiness → priority → candidate seat/token/KV
  checks → `TextGenerationInputs` → caller/pipeline execution → feedback.
  Grammar/KV-onload readiness can hold work outside the selected inputs.
- Selecting CE pops that context until feedback returns partial prefill to
  CE or promotes finished prefill to TG. TG contexts remain tracked. Repeated
  request IDs indicate references to the same context, not duplicate requests.
- Show P/D input-token tiles for prefill/decode, per-pass history and each
  candidate's state/reason. Input tiles are not generated output tokens.
- Seats count active TG, even if unselected, plus selected CE against
  `max_batch_size`. Extra token or KV capacity cannot create a request seat.
- Show policy choices: per-replica in-flight mode ON prioritizes TG with
  optional CE backfill; OFF prioritizes CE, then falls back to TG when no CE
  was selected. The illustrated global TG override suppresses CE backfill.
- TG checks token budget before KV allocation; CE allocates before detailed
  active-token checks. Checks and committed budget usage are distinct.
- Active input-token target and optional page-aligned total-context budget
  are separate. Chunking reduces the active window, not the full context.
  When chunking is OFF the active target is soft: a whole long prompt may
  exceed it. Do not turn the active target into a universal hard limit.
- Budget collection short-circuits on the first non-`AVAILABLE` status.
  `REACHED` accepts the candidate and stops selection. Preserve source order;
  do not promise a universally strict aggregate bound the code does not enforce.
- Under the shown KV-pressure path, preempt the newest remaining TG
  candidate, release its cache and reset its **existing prompt plus generated
  tokens** as a fresh prompt for recomputation. Preserve already-emitted tokens
  and history; recomputation is not losing the user's generated output.
- Distinguish transient waiting from irrecoverable `InsufficientBlocksError`
  when nothing can relieve the shortage. Already-admitted waiting is not 429.
- Keep execution membership fixed. Empty inputs lead the caller to
  `NO_PROGRESS`/backoff. Advance partial/finished requests correctly and
  release terminated ownership through the caller's completion path.

Current accepted examples: Mixed requests, Batch full, KV wait, Preempt +
recompute, Context budget, Not ready, Cannot fit and No requests; active-token
targets 8/12 and chunking ON/OFF. These names and values describe the current
lesson, not required limits on every future animation.

Current illustration scope is a one-replica projection with cadence/backfill
gates open, four-token pages and illustrative 90% pressure. Alignment,
minimum-chunk and per-request active caps are disabled in the teaching model.
LoRA, speculative drafts, DP padding/plan, external KV stores and overlap are
outside this projection. Link the real broader paths rather than presenting
the projection as every supported MAX scheduling mode.

## Later lessons and preserved state views

On the systems reader, current later lessons are:

| Badge / stable fragment | Teaching purpose |
| --- | --- |
| 04 / `#story-clocks` | OS CPU scheduling, API async tasks, MAX batch selection and GPU execution; identify which owner is busy |
| 05 / `#story-reader` | Generation/KV ownership may finish while output remains buffered in response IPC, API or TCP |
| 06 / `#story-replicas` | Illustrative external router, warming/readiness and independent replica resources; existing requests keep their owner |

A waiting HTTP coroutine does not own a dedicated CPU thread. Slow reading
does not turn an accepted stream into HTTP 429. The replica story is not
automatic autoscaling or tensor parallelism: each ready replica owns its API,
worker, queues, model/GPU and KV allocation; a warming replica is excluded.
Keep the advanced reference atlas collapsed and retain existing fragment IDs,
including `#scale-lab`, when renumbering lessons.

The components companion retains its one-request, busy-worker, pressure and
scheduler stories and useful state machines. Server readiness, request
lifecycle and OS-thread scheduling are different kinds of state; show their
own transitions rather than combining them into one overloaded status.

## Source-backed admission and execution baseline

Current pinned source revision:
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. Resolve symbol/call/line references
through [the systems manifest](../os_systems_manifest.json) and
[the components manifest](../os_components_manifest.json); retain complete,
unchanged source definitions and Source Explorer links. Recheck the relevant
definitions when changing revision or behavior.

- N is `--max-queue-size`; M is `--max-pending-requests`. Both default to
  `None`/unbounded in this revision. Finite teaching settings are explicit.
- The worker checks M against `all_ce_reqs` at iteration start. Unfinished CE
  and TG work are distinct; M is not a total active-stream or GPU-busy flag.
- Filled pending CE capacity stops IPC draining; a bounded channel can then
  become non-writable. After connection, a false API writable probe raises
  `RequestQueueFull`; the API maps it to HTTP 429 with `Retry-After: 1`
  before streaming. Worker/GPU busy alone does not imply rejection.
- Accepted ≠ selected/executing ≠ generated ≠ delivered. Keep response
  correlation by request ID and buffering visible.
- Spawn readiness via `multiprocessing.Event`, awaiting the IPC connection
  and HTTP-ready are separate milestones.
- Mark repository-proven control flow `SOURCE`. Simulated JavaScript behavior
  may be checked on CPU but is not MAX/GPU execution evidence. Native runtime,
  OS driver and device internals remain `BOUNDARY` without runtime evidence.

Primary source snapshots: [worker proxy](../sources/max/python/max/serve/worker_interface/zmq_interface.py),
[ZeroMQ sockets](../sources/max/python/max/serve/worker_interface/_zmq_queue.py),
[scheduler](../sources/max/python/max/serve/scheduler/text_generation_scheduler.py),
[constructor and owned sets](../sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py),
[token budgets](../sources/max/python/max/serve/scheduler/batch_constructor/token_budget.py)
and [tokenizer executor](../sources/max/python/max/pipelines/lib/tokenizer.py).

## Authoring map and scoped validation

Paths below are relative to the task root. Edit sources, then generate pages;
do not hand-edit generated HTML. Shared 01–03 edits belong in both readers;
page-specific changes need not modify unrelated lessons.

| Area | Authoring / behavior files |
| --- | --- |
| Shared layout, cards, legend, later stories | `src/serve_story.py`, `.css`, `.js`, `src/serve_story_model.js` |
| 01 API | `src/serve_api.py`, `src/serve_api_model.js`; view/controller in `src/serve_story.js` |
| 02 worker | `src/serve_worker.py`, `.js`, `src/serve_worker_model.js` |
| 03 constructor | `src/serve_batch.py`, `.js`, `src/serve_batch_model.js` |
| Systems page/reference atlas | `src/os_systems.py`, `.html`, `.css`, `.js`, `src/os_systems_lab.js`, `src/os_systems_content.py` |
| Components page | `src/os_components.py`, `.html`, `.css`, `.js`, `src/os_components_content.py` |
| Build / links / source spans | `src/build.py`, `src/validate.py`, manifests and pinned `sources/` |

Use the build command in [agent.md](../agent.md). For changed behavior, run
the relevant `node src/check_serve_api.cjs`, `check_serve_worker.cjs`,
`check_serve_batch.cjs` or `check_serve_stories.cjs` from the task root.
For shared UI changes run `node src/check_serve_stories_browser.cjs` on both
readers, then `python3 src/validate.py` for generated links/source spans.
Use page/lab-specific checks when those areas change. Browser checks require
Playwright; discover the available installation rather than assuming a
host-specific temporary path.

Check meaningful behavior: admission/backpressure, queue reset, request
ownership, budget/KV/preemption invariants, partial-prefill output, pass
membership and completion. Inspect hover-to-card movement, focus/touch,
dismissal, legend color/type matching, readable dynamic text, no autoplay,
pause/reduced motion, narrow-screen overflow, print and offline fallback.

[Recorded constructor validation](../validation/BATCH_CONSTRUCTOR_VALIDATION.md)
documents the November 6, 2026 baseline: 96 scenario/policy/budget combinations,
72 browser contrast audits and 2,066 links. These are historical results,
not required future counts or proof of a later edit.

For skill/requirements-only edits, validate `SKILL.md` frontmatter, changed
reference links and `git diff --check`. Do not regenerate pages or rerun
model/browser workloads solely to update instructions. Follow the scoped Git
workflow in [agent.md](../agent.md), preserving unrelated workspace edits.
