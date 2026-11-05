# MAX Serve: Linux, GPU drivers and serving at scale

[Interactive systems lab](link_htmls/max_serve_os_driver_and_scaling.html). SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Linux + NVIDIA. Source calls and public-reference mechanisms accompany an explicitly illustrative, deterministic lab.

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

### Busy belongs to an owner.

While the model worker executes a pass, the API can tokenize and admit another request. The OS schedules runnable CPU threads; MAX chooses model work; the GPU executes submitted operations.

An HTTP task waiting for output is suspended until data is available. It does not require a dedicated CPU thread. GPU completion and socket readiness are different events. The lanes below show logical ownership, not a measured thread trace.

Try it: Send another request during Execute. Compare the API lane with the worker and device lanes.

```text
OS:     gives CPU time to runnable threads in both processes
API:    tokenize → admit → await output → SSE
Worker: drain → select → execute → advance → publish → repeat
GPU:                     submitted compute → result
Linux socket readiness wakes I/O; native runtime manages device dependencies.
```

### The GPU finishes. The stream remains.

Make the client read slowly. Output tokens collect after generation, while the worker can release that request's KV ownership and start other work.

Response IPC, API output queues and TCP transport each have buffering. This animation groups them into one output buffer. A slow client can raise egress memory and latency; an admitted stream does not turn into a new HTTP 429 response.

Try it: Play until R1 has generated all five tokens. Compare KV ownership with buffered output.

```text
GPU completes → release request KV ownership (pool remains allocated)
       ↓
response IPC → API output buffer → TCP → slow client
              tokens can remain after generation finishes
```

### Another replica, another worker.

An external router can send new requests to independent MAX Serve replicas. Each replica has its own API, worker, queues, model allocation and KV pool.

Add B, wait for readiness, then send more requests. Existing streams keep their original owner. This round-robin router and its warmup duration are teaching choices. Adding HTTP connections alone does not add model compute.

Try it: Add replica B. After it becomes ready, send a burst and watch which replica owns each request.

```text
Clients → external round-robin router → ready replica A (API + worker + GPU/KV)
                                     → ready replica B (API + worker + GPU/KV)
Initializing replicas receive no new requests. Existing work stays with its owner.
```

## Start two processes, then become ready

Follow the startup call stack across its OS boundaries. The parent begins waiting as soon as it spawns the worker; the animation expands what the worker does during that wait. PIDs and sizes shown in cards are example values.

```text
API PID 4100             LINUX / FILES             WORKER PID 4132        GPU
  build Settings + app
  spawn ----------------> fresh interpreter ----> ModelWorker.run()
  wait for readiness ...                         create native session -> driver
                                                 read checkpoint <---- page cache
                                                 load graphs / weights -> VRAM
                                                 set up IPC + scheduler
  readiness wait <------------------------------- alive.set()
  connect request PUSH <------------------------> worker request PULL
  enter TokenGeneratorPipeline lifespan
  server.serve() -------> listen / accept HTTP
  [worker continues scheduler iterations while API serves]
Ownership: two process address spaces; explicitly shared IPC resources only.
```


### 1. Build the app and server controller

Click calls MAX orchestration. MAX resolves configuration and constructs the ASGI app, Uvicorn config and server controller.

```json
{
  "Settings.host": "0.0.0.0",
  "Settings.port": 8000,
  "config.lifespan": "off",
  "app": "FastAPI",
  "HTTP_accepting": false
}
```

1. PipelineConfig.from_args() resolves the model and serving inputs; ServingTokenGeneratorSettings carries the factory, tokenizer and memory plan.

2. fastapi_app() registers routes and middleware. fastapi_config() reads the same Settings object and selects lifespan="off".

3. Server(config) creates a controller. MAX will enter its own lifespan before awaiting server.serve().


Result: The API has Python objects for serving; its HTTP accept loop has not started.

At scale: Startup object construction must be distinguished from readiness when a load balancer decides where to send traffic.


[serve_api_server_and_model_worker()](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60) · [call site](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L96) · [ServingTokenGeneratorSettings(...)](sources/max/python/max/serve/api_server.py#L102) · [fastapi_app(settings, pipeline_settings)](sources/max/python/max/serve/api_server.py#L346) · [fastapi_config(app, server_settings=settings)](sources/max/python/max/serve/api_server.py#L482) · [await server.serve(sockets=sockets)](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L147)


### 2. Spawn a fresh worker process

MAX selects Python's spawn context. Linux supplies the new process and protected virtual address space.

```json
{
  "parent_pid": 4100,
  "child_pid": 4132,
  "start_method": "spawn",
  "target": "ModelWorker()",
  "parent_state": "waiting for readiness"
}
```

1. MAX passes the deferred model factory, configuration, readiness event and IPC configuration to the worker entry point.

2. Python starts a fresh interpreter using OS process-creation machinery. The exact syscall sequence is Python/platform dependent; the child is not a continuation of the parent's Python stack.

3. The child has its own PID, heap and FD table. Linux schedules runnable threads in both processes. The parent's asynchronous readiness wait lets its event loop do other work.


Result: Two processes exist; the child starts model initialization while the parent waits.

At scale: A model worker is a process per serving instance, not a process created for every connected client.


[proc.start(ModelWorker(), alive, factory, config, ...)](sources/max/python/max/serve/pipelines/model_worker.py#L801) · [ProcessManager.ready()](sources/max/python/max/serve/process_control.py#L201) · [_read_ready()](sources/max/python/max/serve/process_control.py#L56)


### 3. Create the native engine and device context

The Python pipeline creates an InferenceSession; its implementation crosses into MAX's native bindings and the device backend.

```json
{
  "python_session": "InferenceSession",
  "native_impl": "_InferenceSession",
  "device": "GPU 0",
  "device_address_space": "context-owned"
}
```

1. The Python session hands devices, thread/runtime settings and options to _InferenceSession.

2. MAX's native engine and GPU backend create or use device resources. NVIDIA's user-space driver cooperates with kernel modules for access, mappings and device management.

3. The diagram labels this as a boundary: the saved Python sources do not reveal every internal runtime thread, driver call or kernel ioctl.


Result: The worker owns a session able to load executable graphs and use the selected device.

At scale: Extra replicas consume native resources and GPU memory; extra client coroutines do not each create another model context.


[TextGenerationPipeline.__init__()](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L208) · [Python session delegates to the native engine](sources/max/python/max/engine/api.py#L707) · [load_devices()](sources/max/python/max/driver/driver.py#L103)


### 4. Read checkpoint and graph artifacts

After creating and configuring its session, the text pipeline resolves weight paths and loads model data through filesystem and host-memory mechanisms.

```json
{
  "checkpoint": "meta-llama/Llama-2-7b-hf",
  "cached_weight_bytes": 13476876272,
  "host_buffer": "process virtual address",
  "file_cache": "Linux page cache"
}
```

1. model_factory() executes the deferred partial. TextGenerationPipeline creates its session before resolving weight paths and passing load_weights(weight_paths) to model construction.

2. Opening and reading or mapping cached files crosses the filesystem boundary. Linux can serve file pages from RAM; a cache miss may require storage I/O. A first download also involves a network path.

3. Host pointers belong to the worker's virtual address space. A CPU page-table translation connects them to physical RAM. File-backed pages and Python/native allocations have different lifetimes.


Result: The model loader obtains checkpoint/graph data in host-accessible memory.

At scale: Warm filesystem caches can reduce storage work, but each GPU replica still needs its executable state and device-resident model data.


[model_factory()](sources/max/python/max/serve/pipelines/model_worker.py#L444) · [TextGenerationPipeline.__init__()](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L208) · [load_weights(weight_paths)](sources/max/python/max/graph/weights/load.py#L24)


### 5. Load weights and retain a KV pool

Graph loading, compilation or cache reuse and weight initialization prepare the model. Device allocation and copies cross the driver boundary.

```json
{
  "weight_estimate_GiB": 12.551319,
  "KV_dtype": "bfloat16",
  "KV_bytes_per_token": 524288,
  "page_tokens": 128,
  "example_pool_GiB": 8.0625
}
```

1. The Llama graph route prepares executable graphs and initializes weights. Existing compilation caches may change the cold-start work.

2. Native/device code allocates GPU buffers and moves required host data to them. GPU copies can use DMA; useful overlap depends on buffers, streams and dependencies.

3. The paged cache manager retains a GPU KV pool. This lesson's batch-4/full-context example allocates 129 × 64 MiB, including one null block. Its exact sizes come from the saved model metadata and the memory lesson.


Result: Weights, executable state and a reusable KV pool are ready for requests.

At scale: GPU VRAM is a separate budget from Linux RAM and socket buffers; one limit cannot stand in for the other.


[load_weights(weight_paths)](sources/max/python/max/graph/weights/load.py#L24) · [load_kv_manager()](sources/max/python/max/pipelines/kv_cache/registry.py#L164) · [MemoryEstimator.plan_from_sizes()](sources/max/python/max/pipelines/lib/memory_estimation.py#L383) · [KVCacheParams.bytes_per_block (property)](sources/max/python/max/nn/kv_cache/cache_params.py#L1923)


### 6. Signal readiness after initialization

The worker sets up queues and the scheduler, finishes configured preparation, then signals alive on its first scheduler iteration.

```json
{
  "alive_event": "set",
  "worker_state": "initialized + scheduler loop",
  "API_readiness_wait": "unblocked",
  "HTTP_accepting": false
}
```

1. Model construction and configured warmup/graph-capture preparation precede entering the worker's scheduling loop.

2. The worker creates its request PULL, response PUSH and cancel PULL queues and loads the scheduler with the initialized pipeline and memory plan.

3. Its first alive.set() satisfies the parent's readiness wait. Later sets are heartbeats. This event is one condition; the API-side IPC handshake must also finish.


Result: The parent knows the worker reached its serving loop.

At scale: A live PID alone does not prove that the model, memory or queues are initialized.


[ModelWorker.run()](sources/max/python/max/serve/pipelines/model_worker.py#L338) · [alive.set()](sources/max/python/max/serve/pipelines/model_worker.py#L650) · [await proc.ready(alive, timeout=...)](sources/max/python/max/serve/pipelines/model_worker.py#L815)


### 7. Connect the API's IPC proxy

The API creates its ZmqModelWorkerProxy and waits for request PUSH/PULL connectivity.

```json
{
  "request_endpoint": "ipc:///tmp/6a82c6df53f7448c91",
  "API_request": "PUSH",
  "worker_request": "PULL",
  "connected": true
}
```

1. MAX generates unique local IPC endpoint paths. On Linux, libzmq's ipc:// transport uses Unix-domain sockets.

2. wait_until_connected() probes request-channel writability. The API establishes this once at startup so later unwritability represents overload.

3. The lifespan constructs TokenGeneratorPipeline with tokenizer and connected proxy, then yields back to the serving coroutine.


Result: The API frontend can hand request contexts to the initialized worker.

At scale: An IPC endpoint is a local message channel, not an extra HTTP listener or a connection to a remote GPU server.


[generate_zmq_ipc_path()](sources/max/python/max/serve/worker_interface/_zmq_queue.py#L63) · [ZmqModelWorkerProxy.wait_until_connected()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L87) · [lifespan()](sources/max/python/max/serve/api_server.py#L117)


### 8. Enter Uvicorn's HTTP serving loop

Only after the lifespan yields does MAX await server.serve(). Uvicorn performs HTTP socket serving on uvloop.

```json
{
  "listener": "0.0.0.0:8000",
  "API_loop": "uvloop / libuv",
  "worker_loop": "scheduler.run_iteration()",
  "ready_for_requests": true
}
```

1. Uvicorn sets up or uses the provided server sockets and starts accepting HTTP work. Prebound socket details can differ by deployment.

2. libuv uses Linux's I/O readiness facility, epoll. When no callbacks need work, the loop can wait for an I/O event or timer instead of busy-polling every client.

3. The worker independently continues scheduling queued inference work. Linux can run its threads while the API is waiting for network I/O.


Result: The serving instance accepts requests after model readiness and IPC connectivity.

At scale: Thousands of mostly idle sockets can coexist with a small GPU batch because connection readiness and model execution are separate stages.


[await server.serve(sockets=sockets)](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L147) · [uvloop.run(serve())](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L176) · [alive.set()](sources/max/python/max/serve/pipelines/model_worker.py#L650)


## Turn network packets into a ready socket

This scene zooms into one Linux host receiving a normal TCP/HTTP request. A reverse proxy may terminate TLS before this hop. The NIC ring, TCP accept queue and model-worker request queue are different objects.

```text
peer -> wire -> NIC -> DMA into host RX buffers
                         |
                    IRQ / NAPI poll (kernel)
                         |
                    IP / TCP processing
                         |
 new connection: SYN state -> completed accept queue -> accept() -> connection FD
 existing stream: TCP payload -> connection receive buffer -> readable FD
                         |
                   epoll ready list -> libuv callback -> HTTP parser -> ASGI task
Queues: RX descriptors != TCP listen queues != socket bytes != model request N.
```


### 1. Receive frames at the NIC

A client or upstream proxy sends TCP packets carrying connection setup or HTTP bytes to the server's network interface.

```json
{
  "peer": "proxy:53122",
  "destination": "server:8000",
  "packet": "TCP payload",
  "existing_connection": true
}
```

1. The NIC receives link-layer frames, validates the parts supported by its offload features and places work in its receive queues.

2. A single HTTP request may span several packets, and a packet does not necessarily correspond to one complete request.

3. Receive-side steering and multiple NIC queues can spread work over CPU cores; the exact queue/core arrangement is driver and host configuration dependent.


Result: Hardware has packet data to hand to the host networking stack.

At scale: Network packet rate and byte rate are independent load dimensions; counting inference requests alone misses them.





### 2. DMA packet bytes into host RX buffers

The driver prepares receive descriptors; the NIC DMA engine writes packet data to mapped host-memory buffers.

```json
{
  "RX_descriptor": "buffer DMA address + length",
  "buffer_capacity_example": "2048 bytes",
  "location": "host RAM",
  "GPU_VRAM_used": false
}
```

1. The driver allocates/recycles buffers and gives the NIC device-visible DMA addresses. These can differ from CPU virtual and physical addresses.

2. An IOMMU may translate those DMA addresses to host physical pages. The device performs the byte transfer without the CPU executing a copy instruction for each byte.

3. CPU work is still needed to maintain descriptors, consume packets and run the protocol stack. Ordinary HTTP arrival does not DMA prompt text directly into the model's KV cache.


Result: Host RX buffers contain bytes that kernel networking can inspect.

At scale: DMA reduces CPU copying work at this boundary; it does not remove buffer, memory-bandwidth or packet-processing costs.





### 3. Use interrupts and NAPI to drain packets

An interrupt can announce receive work; NAPI allows a driver to process a budgeted group of packets by polling.

```json
{
  "IRQ": "RX work notification",
  "NAPI_poll_budget_example": 64,
  "execution": "usually softirq; threaded modes also exist"
}
```

1. A typical driver schedules NAPI when receive work arrives, then suppresses repeated interrupts for that queue while polling drains it.

2. A poll method consumes descriptors up to a budget. Continued work can trigger more polling; the driver rearms notifications when appropriate.

3. This balances latency and interrupt overhead under load. The exact IRQ moderation, poll budget and CPU affinity are host/driver policy.


Result: The stack receives batches of packets rather than requiring a separate application wakeup for every frame.

At scale: At high packet rates, kernel CPU time and receive-ring drops may become bottlenecks before the model worker.





### 4. Reassemble TCP and identify the socket

The kernel validates and routes IP/TCP data, tracks sequence numbers and places payload into the matching TCP receive path.

```json
{
  "connection_key": "source/destination IP + ports",
  "TCP_sequence": "ordered byte stream",
  "socket_receive_bytes_example": 4096
}
```

1. TCP tracks connection state, acknowledgements, retransmission and receive windows. The application observes an ordered byte stream, not raw RX descriptors.

2. For an established connection, received payload can make its socket readable. Request boundaries are subsequently recognized by the HTTP parser.

3. Kernel socket buffers consume host memory and may retain data while the API is busy or its transport pauses reading.


Result: The right connection has ordered bytes available for userspace.

At scale: A slow event loop can grow socket buffers even if the GPU is idle; this is a CPU/network bottleneck.





### 5. Accept new connections and create FDs

For a new connection, TCP setup precedes an accepted socket. The listener and each accepted connection have distinct file descriptors.

```json
{
  "listener_fd_example": 9,
  "connection_fd_example": 57,
  "TCP_state": "ESTABLISHED",
  "FD_owner": "API PID 4100"
}
```

1. Linux maintains incomplete handshake state and a completed connection queue for the listening socket. Backlog settings affect this connection-admission layer.

2. Uvicorn/libuv accepts a completed connection, obtains a new socket FD and attaches protocol state to it.

3. The FD is a small integer indexing this process's FD table. RLIMIT_NOFILE limits open descriptors; sockets, pipes and opened files all contribute.


Result: The API owns a socket for this connection and can register its I/O readiness.

At scale: A larger TCP backlog does not enlarge the GPU batch or the model-worker request queue. They protect different resources.





### 6. Add the readable FD to epoll's ready list

epoll tracks an interest list and reports ready descriptors. It lets an event loop wait on many connections at once.

```json
{
  "interest_list_example": "1,000 connection FDs + listener + IPC + timers",
  "ready_list_example": [
    57,
    83
  ],
  "wait": "epoll_wait / equivalent"
}
```

1. libuv registers sockets with Linux's readiness mechanism. When a socket's state satisfies its interest, epoll can report it ready.

2. A sleeping loop wakes for ready events or a timer. Only runnable callbacks consume the loop thread's CPU time.

3. The loop dispatches nonblocking reads and must handle partial data and EAGAIN. A ready FD is permission to attempt I/O, not proof that an entire HTTP request is present.


Result: The event loop has callbacks to run for the ready connections.

At scale: 1,000 connected clients can yield only two ready socket callbacks in one poll cycle; this does not require 1,000 Python threads.





### 7. Parse HTTP and create an ASGI request task

Uvicorn's protocol code consumes bytes and invokes MAX's FastAPI application once the HTTP request can be handled.

```json
{
  "method": "POST",
  "path": "/v1/chat/completions",
  "body": "JSON messages + token limit",
  "request_task": "one coroutine's state"
}
```

1. The HTTP implementation parses headers/body and produces ASGI receive/send events. The concrete parser depends on Uvicorn configuration.

2. FastAPI routing and MAX middleware resolve the endpoint, request identity and checks. Many such coroutines can share one event-loop thread.

3. A coroutine awaiting body bytes or worker output yields control. A synchronous CPU-heavy callback, by contrast, occupies the thread until it yields or returns.


Result: MAX's API handler receives a structured HTTP request.

At scale: Async I/O handles waiting efficiently; CPU work still needs CPU time and careful scheduling.


[fastapi_app(settings, pipeline_settings)](sources/max/python/max/serve/api_server.py#L346) · [Tokenize, submit and drain token chunks](sources/max/python/max/serve/pipelines/llm.py#L392)


### 8. Keep connection pressure distinct from inference pressure

Server flow control, connection limits and MAX worker admission act at different points in the path.

```json
{
  "open_connections_example": 1000,
  "submitted_requests_example": 100,
  "GPU_batch_example": 4,
  "idle_clients_example": 900
}
```

1. Idle keep-alive connections still occupy socket/HTTP state. A streaming request can stay connected while waiting for model output.

2. Uvicorn transport flow control can pause reading/writing at buffer thresholds. Optional Uvicorn concurrency rejection has its own semantics; it is not MAX's 429 worker gate.

3. The lab opens a conceptual population of clients and counts request states separately. Its connection counts and buffer sizes are illustrative, not a captured Linux trace.


Result: The lab separates client connections, HTTP tasks, queued contexts and active model sequences.

At scale: Capacity planning needs FD, RAM, CPU, egress, GPU compute and KV-memory budgets together.


[Settings.max_queue_size and max_pending_requests](sources/max/python/max/serve/config.py#L158) · [ZmqModelWorkerProxy.stream()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L105) · [_request_queue_full_exception_handler()](sources/max/python/max/serve/api_server.py#L323)


## Hand a request from the API to the worker

Track request R1 through CPU tokenization and three local message channels. MAX's request high-water mark N and worker pending-prefill limit M default to unbounded in this pinned source; the lab chooses finite example caps.

```text
HTTP JSON -> API validation -> CPU executor tokenization -> TextContext(R1)
                                      |
 API proxy: writable? -- no --> RequestQueueFull --> HTTP 429 (before SSE starts)
           | yes
           +--> request PUSH == local Unix-domain IPC ==> request PULL
                    in transit: N                         pending CE: M
                                                           |
                                                        batch selector
 API output queue(R1) <== response PULL <== IPC <== response PUSH [R1, result]
 API disconnect -------> cancel PUSH   ==> IPC ==> cancel PULL [R1]
Large NumPy payloads may use out-of-band/shared-memory encoding intra-node.
```


### 1. Validate inputs and establish request identity

The API converts request fields into the text-generation contract and carries a request ID through the serving path.

```json
{
  "request_id": "R1",
  "messages": [
    {
      "role": "user",
      "content": "Explain a TCP socket."
    }
  ],
  "max_completion_tokens": 64,
  "stream": true
}
```

1. The route parses and validates the JSON contract and resolves the requested model/task. Invalid input can produce an HTTP error before generation.

2. The tokenizer context carries a request identity and generation limits. A request ID identifies results across processes; it is not an OS PID or a socket FD.

3. The text-generation pipeline uses this context to connect tokenization, worker submission and output decoding.


Result: R1 has validated inputs and an identity suitable for worker/result correlation.

At scale: Logs and traces need the request ID to follow one stream among many clients and batch iterations.


[TextTokenizer.new_context()](sources/max/python/max/pipelines/lib/tokenizer.py#L927) · [Tokenize, submit and drain token chunks](sources/max/python/max/serve/pipelines/llm.py#L392) · [_start_stream()](sources/max/python/max/serve/router/openai_routes.py#L189)


### 2. Run tokenizer encoding in the default executor

TextTokenizer._encode_text() awaits run_with_default_executor so tokenizer CPU work can execute outside the event-loop callback.

```json
{
  "text": "example prompt after formatting",
  "prompt_token_count_example": 512,
  "executor": "default CPU executor",
  "API_coroutine": "awaiting encoding"
}
```

1. The method wraps tokenizer encoding as a callable and submits it with MAX's run_with_default_executor helper.

2. The coroutine waits for the result while its event loop can dispatch other ready work. Linux schedules the executor's runnable CPU threads.

3. Executor capacity, CPU quota and tokenizer implementation determine throughput. Python/native work and the GIL affect how much useful CPU parallelism is possible; async syntax alone does not establish that.


Result: The API receives prompt token IDs without holding its event-loop callback for the whole encoding call.

At scale: Long prompts can create a tokenizer queue even when the network loop and GPU are healthy.


[TextTokenizer._encode_text()](sources/max/python/max/pipelines/lib/tokenizer.py#L769)


### 3. Create TextContext and cap generation length

The tokenizer assembles token IDs and generation settings into the worker's context type.

```json
{
  "TextContext.request_id": "R1",
  "prompt_tokens": 512,
  "requested_completion_tokens": 64,
  "model_context_limit": 4096,
  "effective_completion_tokens": 64
}
```

1. The context's prompt IDs, length and generation settings become inputs to batching and KV allocation.

2. MAX uses max_tokens_to_generate to determine the effective completion allowance. The request's desired output may be capped by the available context length.

3. For a 4,080-token prompt and a 64-token request in a 4,096-token context, an example effective allowance is 16 tokens. Do not assume every oversized requested completion causes rejection.


Result: R1 carries a 512-token prompt and a 64-token effective generation limit in this example.

At scale: Routing and memory estimates should use token lengths and effective limits, not only request counts.


[TextTokenizer.new_context()](sources/max/python/max/pipelines/lib/tokenizer.py#L927)


### 4. Probe admission and register R1's response queue

The proxy checks request-channel writability under an admission lock before pushing a context.

```json
{
  "request_id": "R1",
  "N_example": 32,
  "channel_writable": true,
  "pending_out_queues[R1]": "asyncio.Queue()"
}
```

1. The request channel was connected during startup. An unwritable channel at this point signals capacity pressure.

2. The admission lock makes the writability probe and subsequent push a single protected submission step. A full queue raises RequestQueueFull immediately.

3. After a successful probe the proxy creates R1's output queue and pushes its context. A failed send removes the registration and initiates cleanup.


Result: R1 is admitted, with an API-side queue ready to receive its results.

At scale: N is an approximate ZeroMQ transport high-water mark in MAX. The lab uses an exact N-slot queue to make pressure visible.


[ZmqModelWorkerProxy.stream()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L105) · [Settings.max_queue_size and max_pending_requests](sources/max/python/max/serve/config.py#L158) · [_request_queue_full_exception_handler()](sources/max/python/max/serve/api_server.py#L323)


### 5. Send the context over local IPC

The request PUSH/PULL channel crosses the process boundary using libzmq's ipc transport.

```json
{
  "endpoint": "ipc:///tmp/6a82c6df53f7448c91",
  "payload": "serialized TextContext R1",
  "API_direction": "PUSH",
  "worker_direction": "PULL"
}
```

1. libzmq serializes and buffers messages around a Unix-domain socket. Both processes remain on this host.

2. The kernel provides local socket transport and access control. A socket pathname identifies a communication endpoint, not a file containing prompt bytes.

3. The intra-node NumPy encoder can choose an out-of-band/shared-memory path for large arrays. Therefore, the diagram's arrow represents message ownership transfer and metadata as well as possible socket bytes.


Result: The worker can receive R1's context without sharing the API's ordinary Python heap.

At scale: Large payload encoding, host-memory limits and IPC buffers matter; not every object is fully copied through socket bytes.


[generate_zmq_ipc_path()](sources/max/python/max/serve/worker_interface/_zmq_queue.py#L63) · [ZmqConfig.__init__()](sources/max/python/max/serve/worker_interface/_zmq_queue.py#L250)


### 6. Drain only within the pending-prefill cap M

The scheduler moves admitted contexts from transport into its prefill/context-encoding set, respecting max_pending_requests when configured.

```json
{
  "M_example": 128,
  "existing_CE_contexts_example": 127,
  "available_drain_slots": 1,
  "contexts_left_in_transport": "beyond that slot"
}
```

1. _retrieve_pending_requests() computes available capacity from M - len(all_ce_reqs).

2. If no slot remains, it stops draining. Otherwise it limits the number pulled and enqueues those contexts with the batch constructor.

3. CE includes contexts awaiting or undergoing prefill, not every decoding request. A full CE set allows transport pressure to grow until the API's N gate rejects new submissions.


Result: R1 becomes scheduler-owned pending prefill work when M has room.

At scale: N and M protect consecutive stages. Setting only N while the worker drains endlessly does not bound all worker-owned pending contexts.


[TokenGenerationScheduler._retrieve_pending_requests()](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L244) · [Settings.max_queue_size and max_pending_requests](sources/max/python/max/serve/config.py#L158) · [TextBatchConstructor.construct_batch()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L2254)


### 7. Correlate responses by request ID and batch ID

The worker sends result maps; the proxy routes each result to the correct request output queue.

```json
{
  "response": {
    "R1": {
      "token_example": 29871,
      "is_done": false,
      "batch_id": 17
    }
  },
  "API_output_queue": "pending_out_queues[R1]"
}
```

1. A batch can return results for several request IDs. The proxy's response worker dispatches each result into that request's queue.

2. _drain_responses() awaits queue items and can combine already-available outputs while preserving order.

3. The batch ID identifies the producing forward pass for trace correlation. The API then decodes output and exposes token chunks to the route.


Result: R1's coroutine receives its results even when it shared the GPU batch with R2–R4.

At scale: One GPU batch does not merge four clients' HTTP streams; IDs keep the outputs separate.


[ZmqModelWorkerProxy.response_worker()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L247) · [ZmqModelWorkerProxy._drain_responses()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L167) · [Tokenize, submit and drain token chunks](sources/max/python/max/serve/pipelines/llm.py#L392)


### 8. Send cancellation on its own channel

Abandoning a stream triggers proxy cleanup and a cancel message carrying the request ID.

```json
{
  "disconnecting_request": "R1",
  "cancel_payload": [
    "R1"
  ],
  "API_cancel": "PUSH",
  "worker_cancel": "PULL"
}
```

1. The response generator's cleanup detects an abandoned, unfinished stream and cancels worker work for that ID.

2. The separate cancellation channel lets the scheduler observe cancellation independently of new request messages.

3. Scheduler cleanup releases request resources at its execution boundaries. An in-flight operation cannot safely have its live KV pages reused by another request before that operation finishes.


Result: R1 becomes eligible for cancellation cleanup, while other request streams continue.

At scale: Cancellation is a lifecycle transition with ownership cleanup, not a guarantee that already-submitted GPU instructions disappear immediately.


[ZmqModelWorkerProxy.cancel()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L236) · [ZmqModelWorkerProxy._drain_responses()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L167) · [TokenGenerationScheduler.run_iteration()](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L304) · [TextBatchConstructor.release_request()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1066)


## Submit a batch through the runtime and GPU driver

Now expand the InferenceSession and Llama execution boundary. MAX's Python call sites are source facts; the NVIDIA layers describe external mechanisms. The animation is a conceptual dependency order, not a recorded kernel or syscall trace.

```text
MAX scheduler -> select R1..R4 + token budget -> KV page tables
       |
 pipeline.execute -> Llama executable -> native MAX engine -> user-space GPU driver
                                                               |
                              control: device access / allocation / mappings
                                                               v
                                                       Linux kernel modules
                                                               |
     data / commands: host IDs -- copy --> GPU; kernels use weights + device KV
                                                               v
                                    stream order: copy -> prefill -> decode / sample
                                     completion dependency -> safe host token access
       <---------------------------------------------------------- results
No fresh allocation of all weights or all KV memory for each token.
```


### 1. Select sequences and claim KV page ownership

The batch constructor chooses eligible prefill/decode work and obtains cache resources for the selected requests.

```json
{
  "batch_id": 17,
  "request_ids": [
    "R1",
    "R2",
    "R3",
    "R4"
  ],
  "batch_ceiling_example": 4,
  "prompt_tokens_each": 512,
  "prompt_pages_each": 4
}
```

1. Batch selection considers request state, input-token limits and cache availability. A configured batch ceiling is not a promise that every iteration has that many requests.

2. For BF16 Llama 2 7B, 128 tokens occupy a 64 MiB KV page in this example. A 512-token prompt needs four pages per request.

3. A request's page table maps its logical token positions to pages in a retained pool. Page ownership, model context length and GPU buffer allocation are distinct concepts.


Result: Selected requests have the cache resources and metadata required for execution.

At scale: Plenty of connected clients does not help if compute or KV pages limit the eligible batch.


[TextBatchConstructor.construct_batch()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L2254) · [plan_jenga_geometry()](sources/max/python/max/pipelines/kv_cache/paged_kv_cache/jenga_block_pool.py#L253) · [KVCacheParams.bytes_per_block (property)](sources/max/python/max/nn/kv_cache/cache_params.py#L1923)


### 2. Execute the packed model inputs

The text pipeline prepares inputs and calls the Llama executable through the native engine.

```json
{
  "packed_prompt_IDs_example": "R1[512] | R2[512] | R3[512] | R4[512]",
  "row_offsets_example": [
    0,
    512,
    1024,
    1536,
    2048
  ],
  "KV_metadata": "cache inputs + page tables"
}
```

1. The scheduler calls the pipeline's execute path. The model's execute method selects its executable and forwards prepared tensors and KV-related inputs.

2. Packed token inputs and offsets let one forward pass handle several sequences while preserving their individual positions and ownership.

3. Python crosses to compiled/native execution. The saved Python source proves this handoff; it does not specify each kernel's launch parameters.


Result: The native model executable receives one batch's tensors and cache metadata.

At scale: Batch construction is MAX policy. Linux's thread scheduler cannot choose which prompts should share an LLM forward pass.


[Execute, advance requests and release completed requests](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L487) · [TextGenerationPipeline._execute()](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L594) · [LlamaModelBase.execute()](sources/max/python/max/pipelines/architectures/llama3/model.py#L201) · [Python session delegates to the native engine](sources/max/python/max/engine/api.py#L707)


### 3. Enqueue work through the GPU backend

MAX's native backend submits device work through NVIDIA's user-space driver interfaces.

```json
{
  "command_examples": [
    "copy token IDs",
    "execute compiled kernels",
    "record completion dependency"
  ],
  "stream": "ordered device work",
  "host_return": "may precede GPU completion"
}
```

1. The engine binds input/output buffers and arranges executable work through its GPU backend.

2. The user-space driver manages device-context resources and submission mechanisms. CUDA streams express ordering among copies and kernel operations.

3. An asynchronous enqueue may return before execution starts or finishes. The engine must observe dependencies before it exposes results that require completion.


Result: Device commands are submitted with their dependencies.

At scale: A fast host return does not measure GPU execution time; completion-aware timing is needed for real profiling.





### 4. Use the kernel driver for privileged control

Device access, allocation/mapping and resource management require cooperation with Linux's GPU kernel modules.

```json
{
  "control_examples": [
    "device handles",
    "memory mappings",
    "context resources"
  ],
  "kernel_components_example": [
    "nvidia",
    "nvidia-uvm"
  ],
  "driver_version": "deployment dependent"
}
```

1. User-space code cannot independently establish all privileged device mappings or access controls. The kernel driver supplies protected mechanisms.

2. Device initialization, allocation and certain control operations can involve opened device nodes, ioctls and memory mappings. The exact path depends on the driver and runtime.

3. Do not model every GPU kernel launch as a new syscall. Established channels and mapped submission resources may support later submissions without repeating the entire setup.


Result: The process has authorized device resources and mappings for its work.

At scale: This is a control path alongside command/data submission; sending an HTTP request does not open a new GPU device context for that client.





### 5. Move needed data, retaining device-resident state

The GPU uses resident weights and KV data. Host/device copies move inputs or outputs when the executable needs them.

```json
{
  "host_input": "token IDs + metadata",
  "resident_GPU_data": "weights + KV pool",
  "link_example": "PCIe",
  "copy_engine": "DMA-capable device hardware"
}
```

1. A copy command can move host data into device buffers. Pinned/page-locked host memory and suitable dependencies enable useful asynchronous transfer patterns.

2. The GPU reads resident weights and the relevant KV pages during execution. It does not reload the entire checkpoint for each generated token.

3. A CUDA stream orders copies and dependent kernels. Multiple streams can expose possible overlap, but hardware capacity and dependencies decide whether overlap occurs.


Result: The device can read the required input tensors and persistent model state.

At scale: PCIe traffic, host pinning and GPU-memory bandwidth are separate costs from network packet arrival.





### 6. Prefill prompt tokens, then decode incrementally

Prefill/context encoding populates KV for the prompt; later decode iterations consume past KV and extend each live sequence.

```json
{
  "R1_prompt_tokens": 512,
  "after_prefill": "prompt KV + first generated token",
  "next_decode_input": "latest token + cached positions",
  "output_limit": 64
}
```

1. The model evaluates prompt tokens, potentially in chunks chosen by scheduling policy, and writes their K/V data into assigned pages.

2. A completed prefill can produce the first output token. Decode then advances a live autoregressive sequence, normally producing the next token per sequence per ordinary iteration.

3. GPU hardware schedules thread blocks/warps to execute submitted kernels. Linux manages host threads, while MAX decides the request membership and token work for the next pass.


Result: Selected sequences advance; one request's completion can free a slot for another request in a later batch.

At scale: Continuous batching avoids waiting for every member of a static batch to finish, but actual policy and achievable rates require source configuration and measurement.


[Execute, advance requests and release completed requests](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L487) · [TextBatchConstructor.construct_batch()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L2254) · [LlamaModelBase.execute()](sources/max/python/max/pipelines/architectures/llama3/model.py#L201)


### 7. Observe completion before reading token results

The runtime makes device results safe for their consumer before host code reads them.

```json
{
  "device_result_example": "next-token ID",
  "host_consumer": "scheduler / API result path",
  "required_dependency": "producing work and necessary copy complete"
}
```

1. Streams/events or another engine dependency mechanism establish when output data is ready. This diagram does not assert which internal synchronization primitive MAX uses for every operation.

2. The public Buffer host-conversion helper documents device-to-host availability and synchronization behavior for that interface.

3. Correct ownership must last through execution and required transfers. A request's KV pages cannot be recycled while a live operation can still access them.


Result: The pipeline can return safe, correlated token results to the scheduler.

At scale: Unnecessary host/device synchronization can reduce overlap; removing a required dependency would make results incorrect.


[_to_numpy()](sources/max/python/max/driver/buffer.py#L186) · [inplace_copy_from()](sources/max/python/max/driver/buffer.py#L143) · [LlamaModelBase.execute()](sources/max/python/max/pipelines/architectures/llama3/model.py#L201)


### 8. Release completed requests to the retained pool

After execution, the scheduler advances request states and releases finished requests before sending their results.

```json
{
  "R1_generated_tokens": 64,
  "R1_state": "generation complete",
  "R1_page_ownership": "returned",
  "GPU_pool_allocation": "retained for reuse"
}
```

1. The scheduler updates contexts using pipeline results and identifies completions or cancelled requests at its safe scheduling boundaries.

2. TextBatchConstructor.release_request() releases request/cache ownership. Freed pages become available for subsequent work.

3. The backing GPU pool remains allocated. In the lab R1 can continue in sending because a slow client still has buffered tokens even after its KV ownership is gone.


Result: Request pages can be reused while API delivery of final output continues.

At scale: Allocated VRAM staying flat after a request finishes is compatible with healthy page reuse; the ownership counter is the useful distinction.


[Execute, advance requests and release completed requests](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L487) · [TextBatchConstructor.release_request()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1066) · [load_kv_manager()](sources/max/python/max/pipelines/kv_cache/registry.py#L164)


## Stream tokens, reject overload and clean up cancellation

Streaming connects two independent rates: the worker produces results and the client consumes bytes. This source uses unbounded per-request asyncio output queues, so transport flow control alone does not bound all API output memory.

```text
worker results -> API Queue(R1) -> decoded SSE -> Uvicorn transport -> TCP -> client
   80 tok/s         grows if            send may await drain              1 tok/s
                    consumer slows

new context -> pending CE M full -> worker stops drain -> IPC N full -> HTTP 429

generation done -> return request KV pages to retained pool
               -> buffered SSE can still be draining to client
disconnect -> cancel channel -> safe scheduler boundary -> cleanup / page reuse
Already-committed SSE status cannot be replaced with a fresh HTTP status.
```


### 1. Enqueue correlated token results

The worker's response channel delivers results to the API proxy, which routes them to R1's output queue.

```json
{
  "request_id": "R1",
  "produced_tokens_example": 20,
  "consumed_tokens_example": 3,
  "queued_tokens_example": 17,
  "asyncio_queue_maxsize": 0
}
```

1. Response maps can contain several request IDs from the same batch. Each ID has an independent consumer queue.

2. The proxy creates asyncio.Queue() without a maxsize, so this per-request output layer is unbounded in the pinned source.

3. The lab counts buffered tokens as a teaching unit. Real RAM usage also includes result objects, text/JSON framing and buffers at several layers.


Result: R1's consumer can await queued results independently of other streams.

At scale: Many slow consumers can accumulate API RAM even if the GPU and KV allocator continue making progress.


[ZmqModelWorkerProxy.stream()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L105) · [ZmqModelWorkerProxy.response_worker()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L247) · [ZmqModelWorkerProxy._drain_responses()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L167)


### 2. Resolve the first chunk before committing SSE

The route pulls the first chunk before returning its EventSourceResponse.

```json
{
  "first_chunk": "result or early error",
  "HTTP_status_before_response": "not committed",
  "normal_stream_status": 200
}
```

1. _start_stream() asks the generator for its first item. Early errors can still be returned as ordinary HTTP responses.

2. Successful admission and an output chunk allow the route to return EventSourceResponse for streaming.

3. Once response headers are committed, later problems require stream error/termination semantics; the server cannot replace an existing 200 stream with a new 429 status.


Result: The client starts receiving an SSE response after the early error/admission decision.

At scale: This boundary explains why MAX's worker-queue rejection happens before streaming starts.


[_start_stream()](sources/max/python/max/serve/router/openai_routes.py#L189) · [Return the SSE response after admission and first chunk](sources/max/python/max/serve/router/openai_routes.py#L2615) · [_request_queue_full_exception_handler()](sources/max/python/max/serve/api_server.py#L323)


### 3. Send SSE bytes with transport flow control

Uvicorn writes response bytes and can suspend ASGI send while its transport write buffer is above its high-water mark.

```json
{
  "frame_example": "data: {\"choices\":[...]}\n\n",
  "write_buffer": "userspace transport",
  "TCP_send_buffer": "kernel memory",
  "ASGI_send": "may await drain"
}
```

1. The route decodes tokens and frames JSON/text as SSE. Framing, HTTP chunks and TCP packets are different boundaries.

2. The transport buffers pending writes; the kernel socket send buffer holds bytes awaiting network progress. Both live in host memory.

3. Uvicorn documents a high/low watermark mechanism: ASGI send waits when necessary until the transport drains below its low watermark.


Result: The stream can yield while the network catches up.

At scale: Transport flow control protects that layer; upstream unbounded application queues can still grow.





### 4. Let a slow reader reveal buffered output

Client/network consumption can be slower than token production, leaving output waiting after generation completes.

```json
{
  "generated_tokens": 64,
  "delivered_tokens_example": 8,
  "buffered_tokens": 56,
  "client_read_rate_example": "1 token/s",
  "generation_state": "done"
}
```

1. TCP flow/congestion control and application read speed determine when bytes leave buffers and reach the consumer.

2. The lab collapses several output layers into a token buffer and advances its reader with a configured tokens/s rate.

3. With 56 tokens left at one token/s, the lab needs another 56 seconds to drain them. This arithmetic is an example, not a measured MAX or network performance claim.


Result: The request remains in sending while GPU work for it is already complete.

At scale: Open streams and egress backlog can remain high while active GPU batch membership falls.





### 5. Reject a new request when N fills behind M

A full pending-prefill set stops worker draining; the request transport eventually becomes unwritable and MAX returns HTTP 429 for a new submission.

```json
{
  "M_example": 8,
  "CE_count": 8,
  "N_example": 4,
  "transit_count_example": 4,
  "new_request": "R101",
  "HTTP_result": 429
}
```

1. _retrieve_pending_requests() does not drain contexts once M has no room.

2. ZmqModelWorkerProxy.stream() checks writability before registration/submission and raises RequestQueueFull when capacity is unavailable.

3. The API handler maps this to an OpenAI-compatible 429 response with a Retry-After hint. Real ZeroMQ high-water-mark enforcement is approximate; the lab's slots are exact.


Result: The new request is rejected before SSE headers, while existing admitted requests keep their lifecycle.

At scale: Finite queues bound waiting work at a stage; they cannot increase sustained GPU service capacity. Clients need bounded retry/backoff policy.


[TokenGenerationScheduler._retrieve_pending_requests()](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L244) · [ZmqModelWorkerProxy.stream()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L105) · [_request_queue_full_exception_handler()](sources/max/python/max/serve/api_server.py#L323) · [Settings.max_queue_size and max_pending_requests](sources/max/python/max/serve/config.py#L158)


### 6. Cancel work at a safe execution boundary

A disconnect/abandoned generator sends cancellation. The lab holds in-flight page ownership until its current pass completes.

```json
{
  "request": "R1",
  "GPU_pass": "in flight",
  "state": "cancel_pending",
  "assigned_pages_example": 5,
  "reuse_now": false
}
```

1. The API's generator cleanup sends the request ID over the cancel channel; the scheduler processes cancellation during its iteration logic.

2. The simulation marks a live pass's request cancel_pending. It suppresses further output and waits for that pass's completion before releasing pages.

3. This models the ownership invariant. It does not claim a particular native cancellation primitive or that an in-flight kernel is preempted immediately.


Result: Cancellation returns resources safely; already-consumed tokens remain part of the client's history.

At scale: Reusing live buffers early risks cross-request corruption. Keeping cancelled work forever wastes scarce slots and memory.


[ZmqModelWorkerProxy.cancel()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L236) · [ZmqModelWorkerProxy._drain_responses()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L167) · [TokenGenerationScheduler.run_iteration()](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L304) · [TextBatchConstructor.release_request()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1066)


### 7. Return KV pages before all output is delivered

Finished generation releases request-owned cache pages; the API can continue draining its final results.

```json
{
  "R1_generation_done": true,
  "R1_buffered_tokens_example": 56,
  "R1_assigned_KV_pages": 0,
  "pool_GiB": 8.0625,
  "new_owner": "R101 may reuse pages"
}
```

1. The scheduler's execution path updates contexts, releases completed requests and then sends outputs.

2. The cache manager retains its backing pool and marks request pages available. Subsequent requests can obtain those page identities.

3. Socket/output state has a different owner and lifetime. Page reuse does not require waiting until the remote client renders the last token.


Result: KV availability can improve while buffered-output memory is still occupied.

At scale: Track allocated pool bytes, assigned pages, active sequences and output backlog separately.


[Execute, advance requests and release completed requests](sources/max/python/max/serve/scheduler/text_generation_scheduler.py#L487) · [TextBatchConstructor.release_request()](sources/max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1066) · [load_kv_manager()](sources/max/python/max/pipelines/kv_cache/registry.py#L164)


### 8. Finish the stream and release API ownership

When generation and delivery finish, the stream's generator cleanup removes proxy output registration and the HTTP request completes.

```json
{
  "R1_generated": 64,
  "R1_delivered": 64,
  "R1_buffered": 0,
  "pending_out_queues[R1]": "removed",
  "TCP_connection": "can remain keep-alive"
}
```

1. The consumer observes the completion result and exits its response-draining loop; cleanup removes that request's output queue.

2. The HTTP/SSE response ends. Depending on protocol and connection policy, the socket may close or remain available for another HTTP request.

3. The lab marks this as completed only after buffered tokens have been consumed. It separately records first-produced and first-delivered times to explain client-visible latency.


Result: R1 no longer owns worker KV, a proxy result queue or an unfinished response task.

At scale: A successful GPU pass, generation completion and client-visible completion are three distinct milestones.


[ZmqModelWorkerProxy._drain_responses()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L167)


## Scale a fleet of independent serving replicas

This scene adds an external deployment layer around the single-instance MAX path. Routing, autoscaling and replacement policies here are teaching examples, not claims that this max serve invocation implements a fleet controller.

```text
1,000 clients -> external TLS / L7 proxy -> ready replica A: API -> worker -> GPU A
                            |             ready replica B: API -> worker -> GPU B
                            |             warming replica C: excluded until ready
                         routing
                RR / least outstanding / estimated token work / power of two

Control plane: desired replicas -> start -> initialize -> ready -> receive new work
              worker crash -> API teardown -> remove replica -> external replacement
Streams stay with their selected replica; scale-out does not move existing KV/streams.
Replicas duplicate model state; tensor-parallel shards are a different architecture.
```


### 1. Route requests through an external frontend

A production frontend can terminate TLS, authenticate, apply limits and route HTTP requests to ready model replicas.

```json
{
  "front_connections_example": 1000,
  "ready_replicas": [
    "A",
    "B"
  ],
  "routing_layer": "external reverse proxy / gateway",
  "streaming": "SSE"
}
```

1. A frontend separates public connections from internal serving endpoints. HTTP/TLS policy and backend connection pooling depend on deployment.

2. It chooses a backend before sending this request. The resulting SSE stream is tied to that backend for its lifetime.

3. The lab treats each replica as an independent API/worker pair with one GPU and a separate model/KV pool.


Result: New requests have a selected ready serving instance.

At scale: L7 request routing and L4 connection routing have different reuse/affinity behavior; this lab uses request-level routing.





### 2. See how equal counts can hide unequal work

Round robin distributes requests in turn, but long prompts and outputs make LLM requests unequal.

```json
{
  "A_requests_example": [
    {
      "prompt": 256,
      "output": 32
    }
  ],
  "B_requests_example": [
    {
      "prompt": 3072,
      "output": 384
    }
  ],
  "request_count_each": 1
}
```

1. Both replicas have one request, yet B must encode twelve times as many prompt tokens and can generate twelve times as many output tokens.

2. Count-only policies do not see that difference. Estimated remaining token work can consider prompt work still to encode and completion tokens still to generate.

3. Token work is still an estimate: attention/context costs, prefix reuse, heterogeneous GPUs and future output lengths can invalidate a simplistic linear model.


Result: Equal request counts need not mean equal queueing delay or GPU work.

At scale: Compare policies with the same workload and assumptions, then measure real SLO outcomes before adopting a production policy.





### 3. Choose among four illustrative routing policies

The lab lets the frontend use round robin, least outstanding, least estimated token work, or power of two choices.

```json
{
  "round_robin": "next ready replica",
  "least_outstanding": "fewest unfinished streams",
  "token_work": "smallest estimated remaining GPU seconds",
  "power_of_two": "two candidates; fewest unfinished streams"
}
```

1. Round robin needs minimal load information. Least outstanding uses unfinished request counts, including slow streams still sending.

2. Estimated token work uses this lab's linear prefill/decode assumptions and treats generation-finished streams as zero remaining GPU work.

3. Power of two samples two ready replicas and compares their outstanding counts. The lab uses a fixed random seed so repeated experiments are reproducible.


Result: The selected policy assigns only new requests; active requests retain their existing owner.

At scale: Different policies optimize different signals. GPU-work routing can ignore an egress/RAM problem caused by slow consumers.





### 4. Report several capacity signals

A deployment controller should distinguish readiness and load, and observe more than client count.

```json
{
  "signals": [
    "pending CE + IPC depth",
    "oldest / p95 TTFT",
    "active batch",
    "assigned KV pages",
    "output backlog",
    "CPU / RAM / egress"
  ],
  "open_connections": "not GPU batch size"
}
```

1. Readiness decides whether new traffic may reach a replica. Load and SLO signals decide whether the fleet needs more capacity.

2. Queue growth plus latency can indicate an arrival/service imbalance. GPU activity by itself cannot reveal tokenizer starvation or slow-client buffers.

3. The lab displays simulated queue depth, device busy fraction, delivered token rate and delivered TTFT. Production measurements require the actual workload, versions and timing method.


Result: The controller has signals that explain which stage is under pressure.

At scale: Scale the limiting resource; extra GPUs do not cure a saturated frontend or network link.





### 5. Warm a new replica before routing to it

Scale-out starts another serving instance and waits for its initialization and readiness.

```json
{
  "new_replica": "C",
  "initial_state": "warming",
  "lab_warmup_seconds": 2,
  "real_warmup": "workload / hardware / cache dependent",
  "receives_new_requests": false
}
```

1. The external deployment starts the instance. It follows the same configuration, graph/weight loading, queue and readiness path as scene 1.

2. A warming replica is excluded from routing. The lab's two-second warmup compresses this lifecycle for teaching and is not a MAX startup measurement.

3. Once ready, C owns its own model state and KV pool. Replication increases aggregate capacity by adding a GPU; it does not shard one model over A, B and C.


Result: C becomes eligible for new requests only after its readiness gate.

At scale: Cold-start time makes scale-out lag demand; warm capacity and bounded admission can absorb the gap.


[proc.start(ModelWorker(), alive, factory, config, ...)](sources/max/python/max/serve/pipelines/model_worker.py#L801) · [await proc.ready(alive, timeout=...)](sources/max/python/max/serve/pipelines/model_worker.py#L815) · [ZmqModelWorkerProxy.wait_until_connected()](sources/max/python/max/serve/worker_interface/zmq_interface.py#L87) · [await server.serve(sockets=sockets)](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L147)


### 6. Keep existing requests attached during scale-out

New traffic can use the added replica, while already queued or streaming requests remain on their original one.

```json
{
  "R1_owner": "A",
  "R1_page_table": "GPU A",
  "new_R101_owner": "C",
  "migrate_existing_requests": false
}
```

1. A owns its pending contexts, request IDs, device KV and connection response path. Moving them requires an explicit migration protocol.

2. The lab adds C for subsequent routing decisions and leaves A's queues intact, so old backlog may drain slowly even after scale-out.

3. This is why fleet arrival capacity can improve sooner than the latency of requests that were already waiting on an overloaded instance.


Result: Scale-out changes future assignments; it does not teleport existing work.

At scale: A replica-local queue can preserve imbalance. Shared queues or migration systems have different designs and costs.





### 7. Handle a worker crash and fail its streams

This MAX lifecycle propagates a worker failure and tears down the serving context. External systems remove and replace the failed replica.

```json
{
  "replica": "A",
  "worker_state": "exited",
  "API_effect": "serving context tears down",
  "ongoing_streams": "failed",
  "transparent_replay": false
}
```

1. MAX watches process exit/readiness/heartbeat state. A worker error surfaces through process-control task cancellation and serving teardown.

2. The external frontend/controller observes failure and stops choosing A. The lab's Crash action fails that replica's unfinished requests and returns no invented continuation tokens.

3. A client retry is a new request with application-specific replay policy. Already-delivered SSE output and KV state are not automatically migrated.


Result: The failed instance stops receiving traffic; other replicas remain independent.

At scale: More replicas limit a failure's scope, but retry storms and non-idempotent surrounding actions need deployment/application policy.


[_read_ready()](sources/max/python/max/serve/process_control.py#L56) · [ProcessManager.ready()](sources/max/python/max/serve/process_control.py#L201) · [serve_api_server_and_model_worker()](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60) · [start_model_worker()](sources/max/python/max/serve/pipelines/model_worker.py#L759)


### 8. Replace, initialize and rejoin the fleet

The lab's Replace action models an external supervisor starting a replacement API/worker pair.

```json
{
  "replica": "A",
  "lifecycle": [
    "down",
    "warming",
    "ready"
  ],
  "old_streams": "stay failed",
  "new_requests": "eligible after ready"
}
```

1. The external policy creates fresh serving state and repeats initialization. Model/cache availability affects the real recovery time.

2. A becomes ready again and may receive future traffic. The lab retains failed request records for inspection without replaying them.

3. Capacity accounting includes each ready replica's GPU, CPU, RAM and egress. Independent replicas duplicate model memory; tensor/pipeline parallelism must be designed separately.


Result: The fleet restores capacity without pretending the failed streams resumed.

At scale: Serving at scale combines fast data-plane loops with slower deployment, health, scaling and recovery control loops.
