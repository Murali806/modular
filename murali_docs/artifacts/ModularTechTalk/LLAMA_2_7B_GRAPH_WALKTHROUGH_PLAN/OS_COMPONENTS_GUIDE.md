# MAX Serve: OS components, roles and high-level flow

[Interactive focused page](link_htmls/max_serve_os_components_and_flow.html) · SOURCE revision `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Linux + NVIDIA. Source calls and external OS/device mechanisms are distinguished. Animation expresses event order.

```text
USER SPACE: API + uvloop <-- Unix IPC --> worker + MAX scheduler + native backend
KERNEL:     processes/threads | sockets/epoll | memory/files | GPU kernel driver
HARDWARE:   NIC | host RAM/storage | GPU/VRAM
```

## Interactive request stories

SOURCE-backed control flow; illustrative event durations, exact toy queues and conservative KV reservations. No GPU measurements.

Each animation uses the full story width. Hover or focus its numbered badge for the lesson card; leaving the badge/card dismisses it. Touch taps toggle the card. Escape or the close button dismisses it.

Click the circled **L** at the right edge of the sticky section menu for component colors and ownership. Outside click, Escape or Close dismisses this legend. Lesson and legend content remain available without JavaScript and when printing.

### Component legend

| Component | Kind / owner | Meaning |
| --- | --- | --- |
| API process | OS process | Owns the HTTP server, app, event loop and tokenizer executor. |
| Uvicorn / Server | Python object | Uvicorn's Server object handles HTTP and invokes the FastAPI ASGI app in the API process. |
| FastAPI app | ASGI application | Routes and validates requests. Async request tasks share the API loop thread. |
| uvloop event loop | Event-loop thread | Runs ready callbacks and coroutines. The API and model worker each own a loop in their own process. |
| Tokenizer executor | CPU thread pool | CPU tokenization runs via run_in_executor(None, …) on executor threads in the API process. |
| ZeroMQ channels | IPC · Unix sockets | Local ipc:// PUSH/PULL channels carry requests, responses and cancellation. Each drawn slot holds one request context; N is exact only in this teaching model. |
| Model worker / scheduler | Spawned OS process | Owns pending work, batch selection and model execution. The scheduler runs on the worker's loop thread; it selects the next pass. |
| CE / TG request sets | CPU collections · worker | CE holds prefill work; TG holds decoding contexts. These are worker-owned Python collections, not IPC channels or separate threads. |
| Batch constructor | Python object · worker loop | Chooses eligible work within batch, token and KV limits. It is called on the worker's loop thread. P = prefill; D = decode; work-tile counts are input tokens. |
| Model pipeline | Python → compiled model | Prepares inputs and calls the native runtime for selected model work. It returns per-request results; it is not another OS process. |
| KV manager | Host bookkeeping + device pool | Tracks request attention-cache ownership and reusable blocks. Releasing a finished request makes its blocks reusable while the backing pool remains allocated. This is resource management, not a mutex. |
| MAX runtime / user driver | Native code · worker | Submits compiled model work and manages device completion. Internal native threads sit behind this boundary. |
| OS GPU driver | Kernel boundary | The operating system's device driver connects the user-space runtime to the GPU. |
| GPU device | Hardware | Executes the selected model pass and holds device data, including model weights and KV cache. |

Admission uses an `asyncio.Lock` around probe/send in the API process. Worker startup uses a `multiprocessing.Event` readiness signal and awaits channel connection.

Dashed boxes show process ownership. Request IPC has visible empty/occupied message slots, redrawn when N changes. Native runtime, kernel driver and hardware are distinct boundaries. Request-pill colors describe lifecycle state separately from component colors.

### 01 · Inside the API process

The first animation isolates the API process. Play, Step or Reset two requests sharing one event-loop thread and one illustrative IPC slot. Choose All present, No Uvicorn, No FastAPI, No event loop, No admission lock, or Use asyncio loop. Hover/focus/tap 01 for its lesson; hover or focus Tokenizer executor for a color-coded explanation of the default pool, T1 handoff and illustrative thread count. Its changing loop-status line is larger and bold; the existing L legend explains ownership.

Thread badges appear only in this opening animation: T1 is repeated on uvloop, Uvicorn, FastAPI and admission to show their shared API loop thread. T2/T3 identify example executor threads running CPU tokenization jobs; the async request tasks resume on T1. R1/R2 continue to identify requests. Thread IDs and executor assignments are illustrative; real pool sizing and reuse vary.

| Role | Why it exists | If absent, with no replacement |
| --- | --- | --- |
| Uvicorn / Server | Turns HTTP traffic into ASGI calls and writes the response to the connection. | Without an HTTP/ASGI server, this configuration has no HTTP service to accept requests. |
| FastAPI app | Provides MAX routes, request validation and the handlers that call the pipeline. | Without an ASGI app, the configured MAX service cannot be assembled; its routes and handlers are unavailable. |
| uvloop event loop | Runs ready callbacks and request coroutines on the API loop thread; resumes them when awaited work is ready. | Without a running event loop, async startup, request handling and response delivery cannot progress. |
| Admission asyncio.Lock | Lets one coroutine at a time check channel writability and send its request context. | Without this guard, two probes can observe the same space. A later send can face backpressure, losing the immediate-rejection guarantee. |

SOURCE: The API loop runs ready coroutines; CPU tokenization uses executor threads in the same process. Admission uses `asyncio.Lock` around the writability probe and send, then releases it before awaiting output. The model worker is offstage and connected before HTTP serving.

BOUNDARY: Role removals and replacement are thought experiments, not supported MAX flags. The no-lock trace is one possible interleaving, not a measured failure or queue overflow. With the lock, the second request observes a full channel and gets HTTP 429. Without it, two earlier positive probes can leave the later send waiting for room. The exact one-slot capacity is illustrative.

A compatible asyncio loop can replace uvloop with startup/configuration changes. This pinned checkout calls `uvloop.run(serve())` and selects `loop="uvloop"`; simply removing the uvloop package does not provide a fallback. HTTP-server and ASGI-app implementations can also be replaced if their roles are supplied.

The normal animation stops after R1 delivers its first output chunk; subsequent chunks repeat the response wake/resume/send cycle. It does not simulate the offstage worker or GPU in this lesson.

```text
Client HTTP
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
The admission lock is released before waiting for model output.
```

### 02 · Inside the model worker

The second animation isolates the separate worker process after API admission: request IPC, CE/TG request sets, batch construction, KV ownership, the model pipeline, native runtime, OS driver, GPU and response IPC. Hover/focus/tap 02 for the lesson; L explains the component colors and ownership.

Play, Step and Reset compare Two requests, GPU busy, KV pressure and No requests. The loop highlights pull → build → execute → advance/release → publish, with NO_PROGRESS/backoff when no work is eligible. P and D show prefill/decode input work. The CPU request markers and GPU work tiles represent different things.

SOURCE: An ordinary worker calls the scheduler on its own loop. It publishes results keyed by request ID and releases finished request/cache ownership. BOUNDARY: The close-up reuses the existing teaching worker, including its illustrative decode-first chunking, exact two-slot request queue and conservative prompt-plus-output block reservation. Timings, capacities and starting snapshots are not measurements. The backing KV pool remains allocated when request ownership is released.

In the busy scenario, R2 waits in IPC until a subsequent worker iteration. Under cache pressure, R2 is already admitted and waits even though a batch slot is available; it starts after R1 releases blocks. Partial prefill emits no token before the full prompt is consumed. The idle scenario checks for work and then yields with backoff.

| Part | Why it exists | If unavailable |
| --- | --- | --- |
| Worker loop | Runs the scheduler repeatedly; yields with backoff when it reports no progress. | Without the loop, queued work never reaches another scheduling iteration. |
| Request IPC + request sets | Pulls admitted contexts into the worker's CE/prefill and TG/decode collections. A configured M limits the prefill backlog. | Without the channel, contexts cannot cross from the API. Without request state, the worker cannot track prompt or generation progress. |
| Batch constructor | Selects eligible request work using batch, token and cache constraints. | Without batch construction, the pipeline has no valid selection of inputs for its next pass. |
| KV manager | Tracks each request's cached attention state and reusable blocks in a retained device pool. | Without available blocks, a new prefill can wait. Removing the manager itself breaks this paged-cache pipeline. |
| Model pipeline + runtime | Prepares inputs, executes the compiled model and produces per-request results through the native device boundary. | Without an executable pipeline/runtime, selected work cannot produce model output. |
| Advance, release + publish | Moves finished prefill into decode, releases terminated requests and sends results keyed by request ID. | Without advancement or release, work and cache ownership cannot progress correctly. Without publication, the API keeps waiting for results. |

```text
API PROCESS (offstage)
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
One ordinary non-overlap pipeline; tiny queues, budgets and reservations are teaching choices.
```

### 03 · Inside the batch constructor

The third animation places ReplicaRequests inside TextBatchConstructor and follows readiness, priority, request seats, token checks, KV allocation, context references, execution and feedback. Repeated IDs refer to the same request context.

Play, Step and Reset compare Mixed requests, Batch full, KV wait, Preempt + recompute, Context budget, Not ready, Cannot fit and No requests. Policy, active-target and chunking controls compare per-replica in-flight ON/OFF and global TG override.

SOURCE: a one-replica projection of the pinned Python branches. BOUNDARY: tiny page costs, inputs, readiness and native execution are illustrative. No adapters, speculative drafts, DP padding or overlap. In this scene cadence/backfill are open, KV pressure uses a 90% teaching threshold, and chunk alignment/floors/per-request caps are disabled.

The optional total-context budget counts page-aligned full context, not the current prefill chunk. Budget checks short-circuit at the first non-AVAILABLE result. With chunking OFF the active-token target is soft. KV preemption resets existing tokens into a fresh prompt for recomputation; it does not discard the token history.

Transient allocation shortages can wait; an irrecoverable shortage with no other work or transfer raises InsufficientBlocksError. Empty inputs let the ordinary caller return NO_PROGRESS and back off.

| Part | Logic / ownership |
| --- | --- |
| Request state | ReplicaRequests owns ordered CE and TG collections inside TextBatchConstructor. Selecting CE pops its context; selecting TG keeps it tracked. |
| Ready work | Poll KV transfers, readmit completed onloads and promote grammar-ready contexts. Held work does not consume a model input row. |
| Priority | Per-replica priority considers queue state, configuration, cadence and KV pressure. In-flight TG can backfill CE; a global override suppresses that backfill. |
| Request seats | Selected CE plus active TG is bounded by max_batch_size. Spare token capacity does not create a seat. |
| Active tokens | Check an active input window. With chunking OFF the active target is soft; with chunking ON a prompt can be split. Checks do not charge usage until the context is accepted. |
| Context tokens | Optional separate budget for page-aligned context length. Chunking the active window does not shrink the full resident context. The budget collection stops at the first non-AVAILABLE status. |
| KV allocation | TG checks token budget before allocating; CE allocates before the detailed token check. TG shortages can preempt the newest remaining candidate; transient CE shortages retry, irrecoverable shortages raise. |
| Inputs + feedback | Return lists of the selected context references. The pipeline prepares tensors and executes. advance_requests returns partial prefill to CE, promotes completed prefill to TG, and the caller releases terminated requests. |

~~~text
MODEL WORKER PROCESS · one scheduler loop
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
 Repeated request IDs refer to the same context, not duplicated requests.
~~~

### Accepted. Then scheduled.

A valid request R1 arrives at a ready MAX Serve instance. Follow it through tokenization, the admission gate, prefill and repeated decode passes.

Admission gives R1 a place on the request channel. The worker later chooses when it can execute. Output tokens return to the client while generation continues.

Try it: Press Play. Click a request to follow its state; Step advances one event.

```text
Client → API/tokenize → admission → request IPC → worker pending
                                                  ↓
Client ← TCP/SSE ← API ← response IPC ← GPU/native ← selected batch
                                      ↑ prefill, then repeated decode
```

### There is room to wait.

R1 is already generating. Send R2 while the worker is inside a model pass. Watch where R2 waits and when it joins a batch.

With one batch slot, R2 waits for R1 to finish. With two slots and enough KV space, R2 can join a later pass while R1 is still decoding. A running pass keeps its selected members.

Try it: Send R2, then replay with two batch slots. Compare the waiting location.

```text
API accepts R2 ──→ IPC  | R1's selected pass is running
next worker iteration ─→ pending → batch eligibility
B=1: R2 waits for a slot   B=2: R2 may join a subsequent pass
```

### Watch backpressure reach the gate.

Keep R1 decoding, then send a burst. The worker's pending-prefill queue M fills. It stops pulling from IPC. The request channel N then fills too.

When the API proxy finds that connected channel unwritable, it raises RequestQueueFull. The HTTP handler returns 429 with Retry-After: 1 before streaming begins. Already admitted requests stay in the system.

Try it: Send a burst of six. Increase N and replay the same burst to see more waiting room.

```text
Long decode holds batch slot
       ↓
M full → stop pulling → N fills → API probe fails → HTTP 429
Work completes → pending drains → IPC drains → admission has room again
```

### A batch is one pass of work.

R1 is decoding a response. R2 has a longer prompt. The scheduler builds each pass from eligible work, within batch, token and KV limits.

This example gives decode one input token per request and uses the remaining budget for chunked prefill. A partial prompt returns for another pass. Completing prefill yields the first output token; ordinary decode then adds one token per selected request per pass.

Try it: Step through the five highlighted phases. Compare the pass history below the diagram.

```text
repeat: pull within M → construct batch → pipeline.execute()
             ↑                                  ↓
       publish by request ID ← advance state / release finished requests
no eligible work → NO_PROGRESS → yield/backoff → check again
```

## Component roles

### API process · User space

Owns the HTTP frontend, tokenizer and worker proxy.

Interacts with: Linux sockets, its event loop and the model worker through IPC.

Call stack: `cli_serve → serve_api_server_and_model_worker → lifespan`.

### Model-worker process · User space

Owns the initialized model pipeline, native session and request scheduler.

Interacts with: The API through IPC; files and RAM; the native device backend.

Call stack: `proc.start(ModelWorker()) → ModelWorker.run → model_factory`.

### Linux processes and CPU threads · Kernel

Provide isolation and decide which runnable OS threads receive CPU time.

Interacts with: Both Python processes, executor threads and native-runtime threads.

Call stack: `multiprocessing.get_context("spawn"); executor and event-loop work`.

### uvloop / libuv event loop · User space

Runs ready callbacks and resumes Python tasks after awaited work becomes available.

Interacts with: Linux epoll, network/IPC readiness, timers and completed executor work.

Call stack: `uvloop.run(serve()); uvloop executes ModelWorker.run`.

### epoll and file descriptors · Kernel

Report which registered I/O resources are ready for an operation.

Interacts with: libuv, TCP sockets and local Unix-domain sockets.

Call stack: `uvloop's Linux I/O backend; proc lifecycle add_reader`.

### TCP listener and connected sockets · Kernel

Maintain connection state and receive/send byte buffers.

Interacts with: The NIC/network stack and Uvicorn's event loop.

Call stack: `await server.serve(sockets=sockets)`.

### NIC and Linux network driver · Hardware + kernel

Move network packets between the link and host buffers, then feed the TCP stack.

Interacts with: DMA-accessible host RAM, NAPI/interrupt processing and Linux sockets.

Call stack: `The network boundary behind server.serve and HTTP traffic`.

### ZeroMQ and Unix-domain IPC · User space + kernel

Carry request contexts, responses and cancellation between the two processes.

Interacts with: API proxy PUSH/PULL endpoints and worker PUSH/PULL endpoints.

Call stack: `ZmqModelWorkerInterface → model_worker_queues / model_worker_proxy`.

### Readiness Event and process lifecycle · User space + OS support

Let the parent wait for a usable worker and notice worker exit or timeout.

Interacts with: Worker alive.set(), API executor wait and lifecycle FD monitoring.

Call stack: `await proc.ready(alive) → wait_until_connected()`.

### Host virtual memory and RAM · Kernel + hardware

Back process heaps, native buffers, file pages and socket buffers.

Interacts with: API and worker address spaces, Linux memory mappings and device transfers.

Call stack: `TextGenerationPipeline.__init__; load_weights; device buffers`.

### Filesystem and page cache · Kernel

Supply cached checkpoint and executable data to model initialization.

Interacts with: Worker file access, host RAM and storage I/O on a cache miss.

Call stack: `TextGenerationPipeline → load_weights → load_graphs`.

### MAX request scheduler · User space

Admit queued work, select a batch and call the initialized pipeline.

Interacts with: Worker IPC queues, request state, KV ownership and the native model pipeline.

Call stack: `load_scheduler → scheduler.run_iteration`.

### MAX native session and device backend · User space / native boundary

Execute loaded graphs and manage device-buffer operations through the hardware backend.

Interacts with: The worker's pipeline, GPU user-space driver and device resources.

Call stack: `InferenceSession → load_graphs → pipeline.execute`.

### GPU user-space and kernel drivers · User space + kernel boundary

Provide device access, contexts, memory mappings and work-submission mechanisms.

Interacts with: The native backend, Linux device resources and GPU hardware.

Call stack: `The device boundary below InferenceSession and MAX buffers`.

### GPU execution and VRAM · Hardware

Run submitted compute and retain model data and active request cache state.

Interacts with: The native backend's streams/dependencies, transfer engines and device memory.

Call stack: `LlamaModelBase.load_graphs; scheduler pipeline execution`.

## Startup: two processes, then HTTP readiness

The API starts waiting immediately after spawn. The worker initializes on its own stack while that wait is pending.

```text
API PROCESS             LINUX / FILES        MODEL WORKER           NATIVE / GPU
build app + config
spawn ----------------> fresh interpreter -> ModelWorker.run()
await alive ...                              InferenceSession ----> backend
                                             read weights <------ file cache
                                             load_graphs ---------> device state
                                             warmup + queues + scheduler
observe alive <------------------------------ alive.set(); loop continues
connect request PUSH <----------------------> request PULL
lifespan yields
server.serve() --------> HTTP sockets / epoll
API handles HTTP; worker independently continues scheduler.run_iteration().
```

### 1. Construct serving objects

Owner: API process

Resolve configuration, build FastAPI and Uvicorn configuration, and enter MAX's lifespan for storage, telemetry and worker setup.

State change: App and server controller exist; HTTP request handling has not begun.

### 2. Spawn the worker; start the readiness wait

Owner: API → Linux → worker

multiprocessing spawn starts a fresh interpreter. The API immediately awaits proc.ready(alive); its executor performs the Event wait while the child initializes.

State change: Two process address spaces exist. The parent wait and child initialization overlap.

### 3. Create the pipeline and native session

Owner: Model worker → MAX native backend

model_factory() executes the deferred partial. TextGenerationPipeline creates InferenceSession, which delegates to native device code.

State change: The worker owns a session for model initialization; backend resource details cross the native boundary.

### 4. Obtain model data through files and RAM

Owner: Worker → Linux filesystem / memory

After session creation, the pipeline resolves cached weight paths and loads model data. Linux supplies file pages from RAM or storage as needed.

State change: Checkpoint data becomes host-accessible input to model construction.

### 5. Prepare executable and device state

Owner: MAX native backend → driver → GPU

load_graphs prepares executable graphs and initialized weights. Device buffers, required transfers and the backing KV pool are prepared through the backend.

State change: The model has reusable executable and device state for inference.

### 6. Finish preparation; create queues and scheduler

Owner: Model worker

Configured warmup/graph-capture preparation completes. The worker establishes request PULL, response PUSH and cancel PULL queues, then loads the scheduler.

State change: The initialized worker is about to enter its scheduler loop.

### 7. Signal alive on the first scheduler iteration

Owner: Worker → readiness Event → API

alive.set() satisfies the parent's readiness Event. ProcessManager.ready() observes and clears it; the worker continues its scheduling loop.

State change: The readiness wait completes. Request-channel connectivity still has to be established.

### 8. Connect IPC and construct the frontend pipeline

Owner: API proxy ↔ worker request channel

wait_until_connected() waits for request PUSH/PULL connectivity. MAX then constructs TokenGeneratorPipeline and the lifespan yields to serve().

State change: The API has a connected path to the initialized worker.

### 9. Enter Uvicorn serving

Owner: API → Linux socket / epoll path

MAX now awaits server.serve(sockets=sockets). Uvicorn handles HTTP using uvloop, while the worker continues independently.

State change: HTTP requests can reach the initialized model through the connected IPC proxy.

## One request: network → API → worker → GPU → response

Begin with an already-ready server. The arrows show ownership and handoff; the visible animation speed represents event order.

```text
CLIENT / NIC -> LINUX TCP socket -> epoll readiness -> UVLOOP / API
                                                         |
                                                   parse + tokenize
                                                         |
                  API request PUSH -> local Unix IPC -> WORKER request PULL
                                                         |
                                                   MAX selects work
                                                         |
                                                native backend -> GPU
                                                         |
                  API response PULL <- local Unix IPC <- WORKER response PUSH
                         |
                  SSE -> TCP send buffers -> NIC -> CLIENT
Repeat output/decode while needed. Cancellation uses its own API PUSH -> worker PULL channel.
```

### 1. Receive bytes through the network path

Owner: NIC / network driver → Linux TCP

The NIC receives packets; a typical Linux driver path uses DMA, interrupts/NAPI and TCP processing to place bytes in the connection's receive state.

State change: A connected socket has data available to read.

### 2. Report socket readiness to the event loop

Owner: Linux epoll → libuv / uvloop → Uvicorn

epoll reports ready I/O. libuv dispatches the callback, and Uvicorn reads and interprets HTTP bytes for the FastAPI application.

State change: The application's request handling becomes runnable.

### 3. Prepare a request context

Owner: API frontend and tokenizer

FastAPI reaches TokenGeneratorPipeline. The tokenizer creates the request context; its encoding helper uses the default executor for CPU work.

State change: The API has encoded model input and a request identity for worker communication.

### 4. Send the context on the request channel

Owner: API request PUSH → local Unix IPC

ZeroMQ encodes and sends a message through ipc://. Linux supplies the local Unix-domain socket transport between the processes.

State change: Request data crosses the process boundary without making an HTTP or NIC hop.

### 5. Select queued model work

Owner: Worker request PULL → MAX scheduler

The worker receives contexts. MAX's scheduler chooses eligible work and calls the initialized pipeline; Linux schedules the CPU threads executing that code.

State change: Selected requests become an inference pass; other requests remain waiting.

### 6. Execute through the native device backend

Owner: Worker → native session → GPU driver / device

pipeline.execute() uses loaded executable state and device buffers. The native backend submits required transfers and compute with device dependencies.

State change: The GPU has work to execute for the selected requests.

### 7. Make completed output available

Owner: GPU completion → native backend → worker

The backend observes required completion dependencies before consuming results. Prefill/decode passes continue the model's generation work using retained state.

State change: The worker can publish output for the corresponding request.

### 8. Return output on the response channel

Owner: Worker response PUSH → IPC → API response PULL

The API proxy drains worker responses and associates each response with its request's output stream.

State change: The API has a chunk to send; it can still be waiting for later chunks.

### 9. Stream bytes using transport flow control

Owner: API / Uvicorn → TCP buffers → NIC → client

The API serializes streamed output. Socket write readiness and transport backpressure determine when more bytes can be sent to a slow reader.

State change: The client receives output incrementally while later inference passes may continue.

### 10. Finish or cancel request ownership

Owner: MAX scheduler and API request lifecycle

Generation completion releases scheduler membership and request KV ownership. Cancellation travels on its own IPC channel and is handled by the worker; cleanup must respect live device work.

State change: Request resources can be reused safely. A retained backing pool and remaining buffered output have separate lifetimes.

## Device boundary: access, buffers, submission and completion

Linux + a discrete NVIDIA GPU is the reference model. Resource setup can persist across many requests; the Python source ends at the native backend boundary.

```text
WORKER / MAX NATIVE -> GPU USER-SPACE DRIVER
                             |
                    device access / mappings
                             v
                      LINUX GPU KERNEL DRIVER <-> GPU resources

HOST RAM === driver-managed transfer / DMA ===> GPU VRAM
native backend -> queued device work -> GPU compute -> required completion
worker consumes results; later passes reuse device state
request ends -> return KV ownership; server ends -> release owned device resources

CPU virtual addresses, physical/DMA addresses and GPU device addresses are distinct domains.
```

### 1. Reach the vendor device interface

Owner: MAX native session / device backend

InferenceSession and MAX buffer abstractions delegate to the native backend, which uses the selected hardware interface and device context.

State change: The application reaches a vendor device API through MAX's hardware abstraction.

### 2. Establish access and mappings when needed

Owner: GPU user-space driver ↔ Linux GPU kernel driver

Driver components cooperate on device access, memory mapping and context/resource management. This is resource setup and management, not a promised syscall on every inference launch.

State change: Authorized device resources and mappings can be retained for later use.

### 3. Move required data to device buffers

Owner: Backend transfer operation / device DMA

Ordinary host buffers and GPU VRAM occupy different address domains. The backend transfers required weights or inputs, subject to buffer and stream dependencies.

State change: Required data is available to the device when its dependencies are satisfied.

### 4. Submit work with execution dependencies

Owner: Native backend / GPU user-space interface

Device operations are submitted to execution streams and dependencies. Submission can complete on the CPU before the corresponding GPU operation completes.

State change: The device has queued work; submitted and finished are different states.

### 5. Execute compute using resident device state

Owner: GPU hardware

GPU hardware runs the submitted compute and uses its VRAM buffers. Model weights and request cache state are reused across inference passes.

State change: A device operation produces results and advances toward completion.

### 6. Observe the completion required by the consumer

Owner: Device completion → native runtime → worker

Streams/events or backend synchronization express when output can be consumed. Host access may require a copy or synchronization; the exact backend mechanism remains a native boundary.

State change: The consumer can safely read the required result. epoll socket readiness is a separate mechanism from GPU completion.

### 7. Respect request and server resource lifetimes

Owner: MAX request owner / serving-process owner

Request cleanup returns cache-page ownership after safe completion. The backing pool and model buffers can remain for the server's lifetime; process/lifespan cleanup releases serving resources on exit.

State change: One request ending and the server releasing all device state are separate lifecycle transitions.

## Server readiness and shutdown

Teaching states for the complete serving instance. Each readiness gate belongs to a different owner.

```text
Configured -> Starting worker -> Worker ready -> IPC connected -> Serving
                  |                 |                              |
               failure           IPC failure                    stop / signal
                  v                 v                              v
              Failure observed -------------------------------> Stopping -> Stopped
                                                                  ^
Serving -- worker exit --> Failure observed -----------------------+
```

- **Configured** (API process): Serving objects exist. MAX has not entered the HTTP serving loop.

- **Starting worker** (API + child): The worker initializes while the API awaits its readiness Event.

- **Worker ready** (Worker Event): Initialization reached the worker scheduler loop and the parent observed alive.

- **IPC connected** (API proxy): The API's request channel is connected to the worker.

- **Serving** (API + worker): Uvicorn handles HTTP; the worker independently schedules model work.

- **Failure observed** (Lifecycle manager): Startup failed, timed out, or a worker exited. The owning lifecycle unwinds.

- **Stopping** (Serving lifecycle): Exit from serve/lifespan tears down owned worker and frontend resources.

- **Stopped** (Process/resource owners): This instance no longer accepts requests or executes its scheduler loop.

## Request waiting, execution and streaming

Teaching states group API and worker operations. Streaming can overlap later decode passes; cancellation is a cleanup request.

```text
API handling -> Worker queued -> Executing pass -> Output streaming -> Complete
      |             |                ^    |                |
   queue full       |                |    |                |
      v             |                +----|-- next decode -+
   Rejected         +-- cancel ------+----v----------------+
                                    Cancel requested -> safe cleanup -> Cancelled
GPU request ownership can end before the final output buffer has drained.
```

- **API handling** (API / tokenizer): HTTP input is parsed and encoded before worker admission.

- **Worker queued** (IPC / scheduler): The request was admitted and waits for model work selection.

- **Executing pass** (Worker / native GPU): The selected pass owns device buffers and request cache state.

- **Output streaming** (Worker → API → TCP): Produced output travels back to the client. More decode work may still be needed.

- **Complete** (Request lifecycle): Generation and response delivery have completed. Request ownership has been released.

- **Rejected** (API admission): A full connected request channel can return HTTP 429 before SSE response headers.

- **Cancel requested** (API → worker): The API sends cancellation; cleanup must respect any live device work.

- **Cancelled** (Worker / request owner): Cancellation has been processed and request ownership is safely released.

## One Linux thread: runnable, running, waiting

A simplified OS-thread state machine. A suspended Python task and a sleeping OS thread belong to different layers.

```text
Runnable -- CPU dispatch --> Running -- blocking wait --> Waiting
   ^                            |                          |
   +------ preempt / yield ------+                          |
   +---------------------- event / I/O wake ----------------+

Python task: await -> suspended -> resumed by its event loop
OS thread:  may keep running another callback; waits only when appropriate
GPU work:   queued -> executing -> completion; separate device dependencies
```

- **Runnable** (Linux scheduler): The OS thread can run and is waiting for CPU scheduling.

- **Running** (CPU thread): The thread executes callbacks, Python code or native work on a CPU.

- **Waiting** (Kernel I/O / wait): A thread is blocked on an event or I/O operation. Other threads can run.

## Many clients on this one instance

```text
many connected sockets
          |
    ready I/O callbacks
          |
   API tasks / local IPC
          |
  queued model requests
          |
 selected inference passes
          |
 output -> API/TCP -> clients

Linux schedules OS threads; uvloop dispatches ready tasks; MAX selects model work.
A connected client does not require its own spawned model-worker process.
```

## Evidence and references

[Source/evidence manifest](os_components_manifest.json)

[Validation record](validation/OS_COMPONENTS_VALIDATION.md)

- [NGINX: event-driven server architecture](https://blog.nginx.org/blog/inside-nginx-how-we-designed-for-performance-scale)

- [libuv: design of the I/O event loop](https://docs.libuv.org/en/v1.x/design.html)

- [Linux epoll: interest and ready lists](https://man7.org/linux/man-pages/man7/epoll.7.html)

- [Linux NAPI: network-driver event processing](https://docs.kernel.org/networking/napi.html)

- [Linux DMA API: CPU, physical and device addresses](https://docs.kernel.org/core-api/dma-api-howto.html)

- [Linux memory-management concepts](https://docs.kernel.org/admin-guide/mm/concepts.html)

- [Linux getrlimit: open-file-descriptor limits](https://man7.org/linux/man-pages/man2/getrlimit.2.html)

- [Python multiprocessing: spawn and process lifecycle](https://docs.python.org/3/library/multiprocessing.html)

- [Uvicorn: flow control and server limits](https://uvicorn.dev/server-behavior/)

- [ZeroMQ ipc transport: local Unix-domain sockets](https://libzmq.readthedocs.io/en/latest/zmq_ipc.html)

- [NVIDIA CUDA: asynchronous execution, streams and events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

- [NVIDIA CUDA: driver API and context address spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html)

- [NVIDIA Linux GPU kernel modules and user-space components](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/main/README.md)
