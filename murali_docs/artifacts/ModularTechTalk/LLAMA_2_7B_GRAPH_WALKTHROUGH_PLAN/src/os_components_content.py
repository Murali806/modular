"""Focused OS roles, interactions and teaching state machines for one MAX server."""


def component(name, layer, role, interacts, entry, details, refs=(), docs=()):
    return dict(name=name, layer=layer, role=role, interacts=interacts, entry=entry,
                details=list(details), refs=list(refs), docs=list(docs))


def event(sender, receiver, title, owner, interaction, result, refs=(), docs=(), states=None):
    return dict(sender=sender, receiver=receiver, title=title, owner=owner,
                interaction=interaction, result=result, refs=list(refs),
                docs=list(docs), states=states or {})


COMPONENTS = {
    "api": component(
        "API process", "User space",
        "Owns the HTTP frontend, tokenizer and worker proxy.",
        "Linux sockets, its event loop and the model worker through IPC.",
        "cli_serve → serve_api_server_and_model_worker → lifespan",
        ["FastAPI is the ASGI application; Uvicorn is the HTTP server. Creating their Python objects prepares the frontend.",
         "ServingTokenGeneratorSettings carries the model factory, configuration, tokenizer, task and memory plan. These are application inputs.",
         "After worker readiness and IPC connectivity, TokenGeneratorPipeline connects HTTP handling to worker communication."],
        ("serve", "bundle", "app", "uvicorn_cfg", "lifespan", "http")),
    "worker": component(
        "Model-worker process", "User space",
        "Owns the initialized model pipeline, native session and request scheduler.",
        "The API through IPC; files and RAM; the native device backend.",
        "proc.start(ModelWorker()) → ModelWorker.run → model_factory",
        ["Python spawn starts a fresh interpreter with its own process address space and file-descriptor table.",
         "The worker initializes the pipeline once and repeatedly calls scheduler.run_iteration(). It serves requests through this long-lived loop.",
         "Its first alive.set() announces readiness after initialization; later signals can support heartbeat monitoring."],
        ("spawn", "invoke", "worker_run", "alive"), ("spawn",)),
    "threads": component(
        "Linux processes and CPU threads", "Kernel",
        "Provide isolation and decide which runnable OS threads receive CPU time.",
        "Both Python processes, executor threads and native-runtime threads.",
        "multiprocessing.get_context(\"spawn\"); executor and event-loop work",
        ["A PID identifies a process. Several OS threads can share that process's memory while having separate execution state.",
         "Linux schedules CPU threads. MAX's model scheduler separately chooses requests for an inference pass.",
         "A Python coroutine can suspend while its event-loop thread continues another callback. One process can contain threads in different states."],
        ("spawn", "process_ready", "tokenizer_cpu"), ("spawn", "nginx", "libuv")),
    "loop": component(
        "uvloop / libuv event loop", "User space",
        "Runs ready callbacks and resumes Python tasks after awaited work becomes available.",
        "Linux epoll, network/IPC readiness, timers and completed executor work.",
        "uvloop.run(serve()); uvloop executes ModelWorker.run",
        ["uvloop implements Python's asyncio event-loop interface using libuv. The API and worker run separate loops in separate processes.",
         "On Linux, libuv uses epoll for I/O readiness. Callbacks handle ready descriptors; idle connections retain state without continuous polling by Python.",
         "CPU work that occupies an event-loop thread can delay other callbacks. The tokenizer and readiness wait use executor helpers in this source."],
        ("eventloop", "worker_run", "tokenizer_cpu", "process_ready"), ("libuv", "epoll", "nginx")),
    "epoll": component(
        "epoll and file descriptors", "Kernel",
        "Report which registered I/O resources are ready for an operation.",
        "libuv, TCP sockets and local Unix-domain sockets.",
        "uvloop's Linux I/O backend; proc lifecycle add_reader",
        ["A file descriptor is a process-local handle to a kernel resource. epoll maintains an interest list and a ready list.",
         "Socket readability or writability wakes the I/O path. The application still performs reads/writes and protocol handling.",
         "The process lifecycle helper uses add_reader for an FD becoming readable. The separate alive Event wait runs in an executor."],
        ("fd_wait", "process_ready", "connected"), ("epoll", "libuv", "fdlimit")),
    "socket": component(
        "TCP listener and connected sockets", "Kernel",
        "Maintain connection state and receive/send byte buffers.",
        "The NIC/network stack and Uvicorn's event loop.",
        "await server.serve(sockets=sockets)",
        ["The listener accepts connections; connected sockets carry request and response bytes. A deployment may pass prebound sockets to Uvicorn.",
         "The source starts Uvicorn serving after MAX's lifespan has yielded. An earlier Server(config) constructor prepares a controller.",
         "A socket can exist while the client is idle, waiting for inference or reading output. Socket counts and active GPU requests have different lifetimes."],
        ("uvicorn_cfg", "http"), ("uvicorn", "epoll", "fdlimit")),
    "nic": component(
        "NIC and Linux network driver", "Hardware + kernel",
        "Move network packets between the link and host buffers, then feed the TCP stack.",
        "DMA-accessible host RAM, NAPI/interrupt processing and Linux sockets.",
        "The network boundary behind server.serve and HTTP traffic",
        ["A typical receive path uses NIC descriptors and DMA into host buffers. Linux's network driver and NAPI process received work.",
         "The TCP stack associates bytes with a connection, making its socket ready for user-space processing.",
         "The API-to-worker ipc:// transport is local Unix IPC. Its messages take a local kernel path rather than travelling through the NIC."],
        ("http", "ipc_address"), ("napi", "dma", "zmq")),
    "ipc": component(
        "ZeroMQ and Unix-domain IPC", "User space + kernel",
        "Carry request contexts, responses and cancellation between the two processes.",
        "API proxy PUSH/PULL endpoints and worker PUSH/PULL endpoints.",
        "ZmqModelWorkerInterface → model_worker_queues / model_worker_proxy",
        ["Requests: API PUSH → worker PULL. Responses: worker PUSH → API PULL. Cancellation: API PUSH → worker PULL.",
         "libzmq implements message framing and queues in user space. On Linux, its ipc:// transport uses Unix-domain sockets.",
         "The API waits for request-channel connectivity during startup. Each process retains its own ordinary Python objects; IPC transfers encoded information."],
        ("ipc_address", "connected", "proxy_submit", "proxy_drain", "proxy_cancel"), ("zmq",)),
    "ready": component(
        "Readiness Event and process lifecycle", "User space + OS support",
        "Let the parent wait for a usable worker and notice worker exit or timeout.",
        "Worker alive.set(), API executor wait and lifecycle FD monitoring.",
        "await proc.ready(alive) → wait_until_connected()",
        ["The worker sets a multiprocessing Event after initialization. ProcessManager.ready() waits through run_in_executor and clears the Event after observing it.",
         "The wait checks for timeout or an exited subprocess. First readiness and optional later heartbeat monitoring have different purposes.",
         "Readiness is followed by the request-channel handshake. Only then does MAX return from lifespan startup and enter Uvicorn serving."],
        ("alive", "ready", "process_ready", "fd_wait", "connected"), ("spawn",)),
    "memory": component(
        "Host virtual memory and RAM", "Kernel + hardware",
        "Back process heaps, native buffers, file pages and socket buffers.",
        "API and worker address spaces, Linux memory mappings and device transfers.",
        "TextGenerationPipeline.__init__; load_weights; device buffers",
        ["CPU pointers belong to a process virtual address space. Linux page tables map ordinary process pages to host RAM.",
         "The API and worker have separate heaps. Sharing memory requires an explicit mechanism; an IPC message alone does not share an entire heap.",
         "GPU VRAM and GPU device pointers belong to the device side. A driver-managed transfer or mapping connects the two domains when required."],
        ("pipeline_init", "load_weights", "buffer_copy"), ("memory", "dma", "cuda_driver")),
    "files": component(
        "Filesystem and page cache", "Kernel",
        "Supply cached checkpoint and executable data to model initialization.",
        "Worker file access, host RAM and storage I/O on a cache miss.",
        "TextGenerationPipeline → load_weights → load_graphs",
        ["The text pipeline creates its session before resolving weight paths and passing loaded weights into model construction.",
         "Reads or mappings of locally cached files use the filesystem and can be satisfied from Linux's page cache.",
         "The native backend separately prepares executable state and device-resident data. Media storage and telemetry setup also use ordinary lifecycle-managed I/O."],
        ("pipeline_init", "load_weights", "load_graphs", "lifespan"), ("memory",)),
    "scheduler": component(
        "MAX request scheduler", "User space",
        "Admit queued work, select a batch and call the initialized pipeline.",
        "Worker IPC queues, request state, KV ownership and the native model pipeline.",
        "load_scheduler → scheduler.run_iteration",
        ["This scheduler chooses model work and advances requests. Linux separately schedules the CPU threads that run that code.",
         "A selected inference pass uses the worker's initialized model. Waiting requests remain in application/IPC state until selected.",
         "Completion or cancellation releases request ownership of cache pages; the backing KV pool can remain allocated for later requests."],
        ("scheduler", "scheduler_iteration", "scheduler_execute", "release_request")),
    "native": component(
        "MAX native session and device backend", "User space / native boundary",
        "Execute loaded graphs and manage device-buffer operations through the hardware backend.",
        "The worker's pipeline, GPU user-space driver and device resources.",
        "InferenceSession → load_graphs → pipeline.execute",
        ["Python's InferenceSession delegates to a native _InferenceSession implementation.",
         "max.driver exposes MAX device/buffer abstractions. Vendor GPU drivers provide the lower device interface used by the backend.",
         "The saved Python source identifies this boundary. Exact backend threads, submission operations and synchronization depend on the native implementation."],
        ("session_native", "load_graphs", "scheduler_execute", "buffer_copy", "buffer_host"), ("cuda_driver", "cuda_async")),
    "driver": component(
        "GPU user-space and kernel drivers", "User space + kernel boundary",
        "Provide device access, contexts, memory mappings and work-submission mechanisms.",
        "The native backend, Linux device resources and GPU hardware.",
        "The device boundary below InferenceSession and MAX buffers",
        ["The CUDA user-space driver exposes context, memory and execution APIs. Linux GPU kernel modules supply privileged device-management mechanisms.",
         "Resource creation and memory mapping involve driver cooperation. Resources established during initialization can be reused across many inference passes.",
         "Submission and completion are separate events. The diagrams show their roles without assigning an invented syscall to every kernel launch."],
        ("session_native", "buffer_copy", "buffer_host"), ("cuda_driver", "cuda_async", "nvidia_kernel")),
    "gpu": component(
        "GPU execution and VRAM", "Hardware",
        "Run submitted compute and retain model data and active request cache state.",
        "The native backend's streams/dependencies, transfer engines and device memory.",
        "LlamaModelBase.load_graphs; scheduler pipeline execution",
        ["GPU hardware executes submitted work using device resources; Linux CPU scheduling is a separate mechanism.",
         "Weights and a backing KV pool can outlive one request. Request completion returns ownership for reuse within that retained allocation.",
         "The backend observes required completion dependencies before consuming results or reusing live buffers."],
        ("load_graphs", "scheduler_execute", "buffer_host", "release_request"), ("cuda_async", "cuda_driver")),
}


FLOWS = [
    dict(id="startup", title="Startup: two processes, then HTTP readiness",
         intro="The API starts waiting immediately after spawn. The worker initializes on its own stack while that wait is pending.",
         actors=[("api", "API process", "api"), ("linux", "Linux", "threads"),
                 ("worker", "Model worker", "worker"), ("files", "Files / RAM", "files"),
                 ("native", "Native / driver", "native"), ("gpu", "GPU / VRAM", "gpu")],
         ascii="""API PROCESS             LINUX / FILES        MODEL WORKER           NATIVE / GPU
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
API handles HTTP; worker independently continues scheduler.run_iteration().""",
         steps=[
             event("api", "api", "Construct serving objects", "API process",
                   "Resolve configuration, build FastAPI and Uvicorn configuration, and enter MAX's lifespan for storage, telemetry and worker setup.",
                   "App and server controller exist; HTTP request handling has not begun.",
                   ("serve", "config_call", "bundle", "app", "uvicorn_cfg", "lifespan"),
                   states={"API": "Startup", "Worker": "Absent", "HTTP": "Not accepting"}),
             event("api", "worker", "Spawn the worker; start the readiness wait", "API → Linux → worker",
                   "multiprocessing spawn starts a fresh interpreter. The API immediately awaits proc.ready(alive); its executor performs the Event wait while the child initializes.",
                   "Two process address spaces exist. The parent wait and child initialization overlap.",
                   ("spawn", "ready", "process_ready"), ("spawn", "libuv"),
                   {"API": "Awaiting readiness", "Worker": "Starting", "HTTP": "Not accepting"}),
             event("worker", "native", "Create the pipeline and native session", "Model worker → MAX native backend",
                   "model_factory() executes the deferred partial. TextGenerationPipeline creates InferenceSession, which delegates to native device code.",
                   "The worker owns a session for model initialization; backend resource details cross the native boundary.",
                   ("invoke", "pipeline_init", "session_native"), ("cuda_driver",),
                   {"API": "Awaiting readiness", "Worker": "Creating session", "HTTP": "Not accepting"}),
             event("worker", "files", "Obtain model data through files and RAM", "Worker → Linux filesystem / memory",
                   "After session creation, the pipeline resolves cached weight paths and loads model data. Linux supplies file pages from RAM or storage as needed.",
                   "Checkpoint data becomes host-accessible input to model construction.",
                   ("pipeline_init", "load_weights"), ("memory",),
                   {"API": "Awaiting readiness", "Worker": "Loading model data", "HTTP": "Not accepting"}),
             event("native", "gpu", "Prepare executable and device state", "MAX native backend → driver → GPU",
                   "load_graphs prepares executable graphs and initialized weights. Device buffers, required transfers and the backing KV pool are prepared through the backend.",
                   "The model has reusable executable and device state for inference.",
                   ("load_graphs", "session_native", "buffer_copy"), ("cuda_driver", "cuda_async"),
                   {"API": "Awaiting readiness", "Worker": "Initializing device state", "HTTP": "Not accepting"}),
             event("worker", "worker", "Finish preparation; create queues and scheduler", "Model worker",
                   "Configured warmup/graph-capture preparation completes. The worker establishes request PULL, response PUSH and cancel PULL queues, then loads the scheduler.",
                   "The initialized worker is about to enter its scheduler loop.",
                   ("worker_run", "scheduler"), (),
                   {"API": "Awaiting readiness", "Worker": "Queues + scheduler ready", "HTTP": "Not accepting"}),
             event("worker", "api", "Signal alive on the first scheduler iteration", "Worker → readiness Event → API",
                   "alive.set() satisfies the parent's readiness Event. ProcessManager.ready() observes and clears it; the worker continues its scheduling loop.",
                   "The readiness wait completes. Request-channel connectivity still has to be established.",
                   ("alive", "ready", "process_ready"), (),
                   {"API": "Worker readiness observed", "Worker": "Scheduler loop", "HTTP": "Not accepting"}),
             event("api", "worker", "Connect IPC and construct the frontend pipeline", "API proxy ↔ worker request channel",
                   "wait_until_connected() waits for request PUSH/PULL connectivity. MAX then constructs TokenGeneratorPipeline and the lifespan yields to serve().",
                   "The API has a connected path to the initialized worker.",
                   ("connected", "ipc_address", "lifespan"), ("zmq",),
                   {"API": "IPC connected", "Worker": "Scheduler loop", "HTTP": "Not accepting"}),
             event("api", "linux", "Enter Uvicorn serving", "API → Linux socket / epoll path",
                   "MAX now awaits server.serve(sockets=sockets). Uvicorn handles HTTP using uvloop, while the worker continues independently.",
                   "HTTP requests can reach the initialized model through the connected IPC proxy.",
                   ("http", "eventloop", "worker_run"), ("uvicorn", "epoll", "libuv"),
                   {"API": "Handling HTTP", "Worker": "Scheduler loop", "HTTP": "Accepting"}),
         ]),
    dict(id="request", title="One request: network → API → worker → GPU → response",
         intro="Begin with an already-ready server. The arrows show ownership and handoff; the visible animation speed represents event order.",
         actors=[("client", "Client / NIC", "nic"), ("linux", "Sockets / epoll", "socket"),
                 ("api", "API process", "api"), ("ipc", "Local IPC", "ipc"),
                 ("worker", "Worker / scheduler", "scheduler"), ("device", "Native / GPU", "gpu")],
         ascii="""CLIENT / NIC -> LINUX TCP socket -> epoll readiness -> UVLOOP / API
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
Repeat output/decode while needed. Cancellation uses its own API PUSH -> worker PULL channel.""",
         steps=[
             event("client", "linux", "Receive bytes through the network path", "NIC / network driver → Linux TCP",
                   "The NIC receives packets; a typical Linux driver path uses DMA, interrupts/NAPI and TCP processing to place bytes in the connection's receive state.",
                   "A connected socket has data available to read.", ("http",), ("napi", "dma")),
             event("linux", "api", "Report socket readiness to the event loop", "Linux epoll → libuv / uvloop → Uvicorn",
                   "epoll reports ready I/O. libuv dispatches the callback, and Uvicorn reads and interprets HTTP bytes for the FastAPI application.",
                   "The application's request handling becomes runnable.", ("eventloop", "http", "app"), ("epoll", "libuv", "uvicorn")),
             event("api", "api", "Prepare a request context", "API frontend and tokenizer",
                   "FastAPI reaches TokenGeneratorPipeline. The tokenizer creates the request context; its encoding helper uses the default executor for CPU work.",
                   "The API has encoded model input and a request identity for worker communication.", ("lifespan", "tokenizer_cpu", "proxy_submit")),
             event("api", "ipc", "Send the context on the request channel", "API request PUSH → local Unix IPC",
                   "ZeroMQ encodes and sends a message through ipc://. Linux supplies the local Unix-domain socket transport between the processes.",
                   "Request data crosses the process boundary without making an HTTP or NIC hop.", ("proxy_submit", "ipc_address"), ("zmq",)),
             event("ipc", "worker", "Select queued model work", "Worker request PULL → MAX scheduler",
                   "The worker receives contexts. MAX's scheduler chooses eligible work and calls the initialized pipeline; Linux schedules the CPU threads executing that code.",
                   "Selected requests become an inference pass; other requests remain waiting.", ("scheduler", "scheduler_iteration", "scheduler_execute")),
             event("worker", "device", "Execute through the native device backend", "Worker → native session → GPU driver / device",
                   "pipeline.execute() uses loaded executable state and device buffers. The native backend submits required transfers and compute with device dependencies.",
                   "The GPU has work to execute for the selected requests.", ("scheduler_execute", "session_native", "load_graphs"), ("cuda_driver", "cuda_async")),
             event("device", "worker", "Make completed output available", "GPU completion → native backend → worker",
                   "The backend observes required completion dependencies before consuming results. Prefill/decode passes continue the model's generation work using retained state.",
                   "The worker can publish output for the corresponding request.", ("scheduler_execute", "buffer_host"), ("cuda_async",)),
             event("worker", "api", "Return output on the response channel", "Worker response PUSH → IPC → API response PULL",
                   "The API proxy drains worker responses and associates each response with its request's output stream.",
                   "The API has a chunk to send; it can still be waiting for later chunks.", ("proxy_drain", "proxy_submit"), ("zmq",)),
             event("api", "client", "Stream bytes using transport flow control", "API / Uvicorn → TCP buffers → NIC → client",
                   "The API serializes streamed output. Socket write readiness and transport backpressure determine when more bytes can be sent to a slow reader.",
                   "The client receives output incrementally while later inference passes may continue.", ("http", "proxy_drain"), ("uvicorn", "epoll")),
             event("worker", "worker", "Finish or cancel request ownership", "MAX scheduler and API request lifecycle",
                   "Generation completion releases scheduler membership and request KV ownership. Cancellation travels on its own IPC channel and is handled by the worker; cleanup must respect live device work.",
                   "Request resources can be reused safely. A retained backing pool and remaining buffered output have separate lifetimes.", ("release_request", "proxy_cancel", "scheduler_execute"), ("cuda_async",)),
         ]),
    dict(id="device", title="Device boundary: access, buffers, submission and completion",
         intro="Linux + a discrete NVIDIA GPU is the reference model. Resource setup can persist across many requests; the Python source ends at the native backend boundary.",
         actors=[("worker", "Worker / MAX", "native"), ("ram", "Host RAM", "memory"),
                 ("user", "GPU user driver", "driver", "User space"), ("kernel", "GPU kernel driver", "driver", "Kernel"),
                 ("gpu", "GPU / VRAM", "gpu")],
         ascii="""WORKER / MAX NATIVE -> GPU USER-SPACE DRIVER
                             |
                    device access / mappings
                             v
                      LINUX GPU KERNEL DRIVER <-> GPU resources

HOST RAM === driver-managed transfer / DMA ===> GPU VRAM
native backend -> queued device work -> GPU compute -> required completion
worker consumes results; later passes reuse device state
request ends -> return KV ownership; server ends -> release owned device resources

CPU virtual addresses, physical/DMA addresses and GPU device addresses are distinct domains.""",
         steps=[
             event("worker", "user", "Reach the vendor device interface", "MAX native session / device backend",
                   "InferenceSession and MAX buffer abstractions delegate to the native backend, which uses the selected hardware interface and device context.",
                   "The application reaches a vendor device API through MAX's hardware abstraction.", ("session_native", "buffer_copy"), ("cuda_driver",)),
             event("user", "kernel", "Establish access and mappings when needed", "GPU user-space driver ↔ Linux GPU kernel driver",
                   "Driver components cooperate on device access, memory mapping and context/resource management. This is resource setup and management, not a promised syscall on every inference launch.",
                   "Authorized device resources and mappings can be retained for later use.", ("session_native",), ("cuda_driver", "nvidia_kernel")),
             event("ram", "gpu", "Move required data to device buffers", "Backend transfer operation / device DMA",
                   "Ordinary host buffers and GPU VRAM occupy different address domains. The backend transfers required weights or inputs, subject to buffer and stream dependencies.",
                   "Required data is available to the device when its dependencies are satisfied.", ("buffer_copy", "load_graphs"), ("dma", "cuda_async")),
             event("user", "gpu", "Submit work with execution dependencies", "Native backend / GPU user-space interface",
                   "Device operations are submitted to execution streams and dependencies. Submission can complete on the CPU before the corresponding GPU operation completes.",
                   "The device has queued work; submitted and finished are different states.", ("scheduler_execute", "session_native"), ("cuda_async", "cuda_driver")),
             event("gpu", "gpu", "Execute compute using resident device state", "GPU hardware",
                   "GPU hardware runs the submitted compute and uses its VRAM buffers. Model weights and request cache state are reused across inference passes.",
                   "A device operation produces results and advances toward completion.", ("load_graphs", "scheduler_execute"), ("cuda_async",)),
             event("gpu", "worker", "Observe the completion required by the consumer", "Device completion → native runtime → worker",
                   "Streams/events or backend synchronization express when output can be consumed. Host access may require a copy or synchronization; the exact backend mechanism remains a native boundary.",
                   "The consumer can safely read the required result. epoll socket readiness is a separate mechanism from GPU completion.", ("buffer_host", "scheduler_execute"), ("cuda_async",)),
             event("worker", "gpu", "Respect request and server resource lifetimes", "MAX request owner / serving-process owner",
                   "Request cleanup returns cache-page ownership after safe completion. The backing pool and model buffers can remain for the server's lifetime; process/lifespan cleanup releases serving resources on exit.",
                   "One request ending and the server releasing all device state are separate lifecycle transitions.", ("release_request", "lifespan", "serve"), ("cuda_async", "cuda_driver")),
         ]),
]


def state(label, owner, meaning, x, y, refs=(), docs=()):
    return dict(label=label, owner=owner, meaning=meaning, x=x, y=y,
                refs=list(refs), docs=list(docs))


def edge(start, end, label, detail, refs=(), docs=()):
    return dict(start=start, end=end, label=label, detail=detail,
                refs=list(refs), docs=list(docs))


MACHINES = [
    dict(id="server", title="Server readiness and shutdown", width=1080, height=320,
         intro="Teaching states for the complete serving instance. Each readiness gate belongs to a different owner.",
         states={
             "configured": state("Configured", "API process", "Serving objects exist. MAX has not entered the HTTP serving loop.", 110, 70, ("app", "uvicorn_cfg")),
             "spawning": state("Starting worker", "API + child", "The worker initializes while the API awaits its readiness Event.", 325, 70, ("spawn", "ready")),
             "ready": state("Worker ready", "Worker Event", "Initialization reached the worker scheduler loop and the parent observed alive.", 540, 70, ("alive", "process_ready")),
             "connected": state("IPC connected", "API proxy", "The API's request channel is connected to the worker.", 755, 70, ("connected",)),
             "serving": state("Serving", "API + worker", "Uvicorn handles HTTP; the worker independently schedules model work.", 970, 70, ("http", "worker_run")),
             "failed": state("Failure observed", "Lifecycle manager", "Startup failed, timed out, or a worker exited. The owning lifecycle unwinds.", 325, 245, ("process_ready", "serve")),
             "stopping": state("Stopping", "Serving lifecycle", "Exit from serve/lifespan tears down owned worker and frontend resources.", 755, 245, ("serve", "lifespan")),
             "stopped": state("Stopped", "Process/resource owners", "This instance no longer accepts requests or executes its scheduler loop.", 970, 245, ("serve", "lifespan")),
         },
         edges=[
             edge("configured", "spawning", "spawn", "Start the child, then immediately wait for its readiness Event.", ("spawn", "ready")),
             edge("spawning", "ready", "alive observed", "The parent observes the first alive signal after model initialization.", ("alive", "process_ready")),
             edge("ready", "connected", "request channel connected", "The API completes the ZeroMQ request-channel handshake.", ("connected",)),
             edge("connected", "serving", "lifespan yields", "MAX resumes serve() and enters server.serve().", ("lifespan", "http")),
             edge("spawning", "failed", "error / timeout", "An unusable worker prevents startup from reaching HTTP serving.", ("process_ready",)),
             edge("ready", "failed", "connection failure", "A failed or timed-out IPC connection prevents the startup handoff.", ("connected",)),
             edge("serving", "failed", "worker exits", "The worker lifecycle failure tears down the running server task.", ("serve",)),
             edge("failed", "stopping", "unwind", "The owning context exits and performs cleanup.", ("serve", "lifespan")),
             edge("serving", "stopping", "normal stop / signal", "The serving coroutine exits; MAX's explicit lifespan handles teardown.", ("serve", "lifespan")),
             edge("stopping", "stopped", "cleanup completes", "Frontend and worker lifetime ends for this instance.", ("lifespan",)),
         ],
         scenarios={
             "normal": dict(label="Start, serve, then stop", path=["configured", "spawning", "ready", "connected", "serving", "stopping", "stopped"]),
             "startup_failure": dict(label="Worker fails during startup", path=["configured", "spawning", "failed", "stopping", "stopped"]),
             "worker_exit": dict(label="Worker exits while serving", path=["configured", "spawning", "ready", "connected", "serving", "failed", "stopping", "stopped"]),
         },
         ascii="""Configured -> Starting worker -> Worker ready -> IPC connected -> Serving
                  |                 |                              |
               failure           IPC failure                    stop / signal
                  v                 v                              v
              Failure observed -------------------------------> Stopping -> Stopped
                                                                  ^
Serving -- worker exit --> Failure observed -----------------------+"""),
    dict(id="request", title="Request waiting, execution and streaming", width=1080, height=330,
         intro="Teaching states group API and worker operations. Streaming can overlap later decode passes; cancellation is a cleanup request.",
         states={
             "received": state("API handling", "API / tokenizer", "HTTP input is parsed and encoded before worker admission.", 110, 80, ("tokenizer_cpu", "proxy_submit")),
             "queued": state("Worker queued", "IPC / scheduler", "The request was admitted and waits for model work selection.", 325, 80, ("proxy_submit", "scheduler_iteration")),
             "executing": state("Executing pass", "Worker / native GPU", "The selected pass owns device buffers and request cache state.", 540, 80, ("scheduler_execute",)),
             "streaming": state("Output streaming", "Worker → API → TCP", "Produced output travels back to the client. More decode work may still be needed.", 755, 80, ("proxy_drain", "http")),
             "complete": state("Complete", "Request lifecycle", "Generation and response delivery have completed. Request ownership has been released.", 970, 80, ("release_request", "proxy_submit")),
             "rejected": state("Rejected", "API admission", "A full connected request channel can return HTTP 429 before SSE response headers.", 110, 250, ("proxy_submit", "http_429")),
             "cancelling": state("Cancel requested", "API → worker", "The API sends cancellation; cleanup must respect any live device work.", 540, 250, ("proxy_cancel", "release_request"), ("cuda_async",)),
             "cancelled": state("Cancelled", "Worker / request owner", "Cancellation has been processed and request ownership is safely released.", 755, 250, ("proxy_cancel", "release_request")),
         },
         edges=[
             edge("received", "queued", "admission + IPC", "Encode and submit the context; the worker accepts queued work.", ("proxy_submit",)),
             edge("queued", "executing", "select pass", "MAX selects eligible work for pipeline execution.", ("scheduler_execute",)),
             edge("executing", "streaming", "output available", "Required device output becomes a worker response, then an API stream chunk.", ("scheduler_execute", "proxy_drain")),
             edge("streaming", "executing", "next decode", "The next pass can execute while earlier output is still being delivered.", ("scheduler_execute",)),
             edge("streaming", "complete", "generation + delivery done", "Request device ownership may already have ended while the final buffered output was draining.", ("release_request", "proxy_submit")),
             edge("received", "rejected", "request channel full", "The connected request channel fails immediate admission before the SSE response begins.", ("proxy_submit", "http_429")),
             edge("queued", "cancelling", "cancel", "Cancellation can remove work that is still waiting.", ("proxy_cancel", "release_request")),
             edge("executing", "cancelling", "cancel", "Cancellation requests cleanup while a pass may still own live device buffers.", ("proxy_cancel", "release_request"), ("cuda_async",)),
             edge("streaming", "cancelling", "disconnect / cancel", "The API sends cancellation when the request stream ends early.", ("proxy_cancel", "proxy_submit")),
             edge("cancelling", "cancelled", "safe cleanup", "The worker processes cancellation and releases owned request state safely.", ("release_request",), ("cuda_async",)),
         ],
         scenarios={
             "normal": dict(label="Generate and stream", path=["received", "queued", "executing", "streaming", "executing", "streaming", "complete"]),
             "cancel": dict(label="Client cancels while streaming", path=["received", "queued", "executing", "streaming", "cancelling", "cancelled"]),
             "reject": dict(label="Admission is full", path=["received", "rejected"]),
         },
         ascii="""API handling -> Worker queued -> Executing pass -> Output streaming -> Complete
      |             |                ^    |                |
   queue full       |                |    |                |
      v             |                +----|-- next decode -+
   Rejected         +-- cancel ------+----v----------------+
                                    Cancel requested -> safe cleanup -> Cancelled
GPU request ownership can end before the final output buffer has drained."""),
    dict(id="thread", title="One Linux thread: runnable, running, waiting", width=1000, height=230,
         intro="A simplified OS-thread state machine. A suspended Python task and a sleeping OS thread belong to different layers.",
         states={
             "runnable": state("Runnable", "Linux scheduler", "The OS thread can run and is waiting for CPU scheduling.", 170, 110, (), ("nginx", "libuv")),
             "running": state("Running", "CPU thread", "The thread executes callbacks, Python code or native work on a CPU.", 500, 110, ("eventloop", "worker_run")),
             "waiting": state("Waiting", "Kernel I/O / wait", "A thread is blocked on an event or I/O operation. Other threads can run.", 830, 110, ("fd_wait", "process_ready"), ("epoll", "libuv")),
         },
         edges=[
             edge("runnable", "running", "CPU dispatch", "Linux selects the runnable OS thread to execute."),
             edge("running", "waiting", "blocking wait", "An idle event-loop thread can wait for I/O; an executor thread can wait for an Event.", ("process_ready",), ("epoll", "libuv")),
             edge("waiting", "runnable", "event / I/O arrives", "The awaited event makes the thread eligible to execute again.", (), ("epoll", "libuv")),
             edge("running", "runnable", "preempt / yield CPU", "Linux may schedule another runnable thread while this one retains its execution state."),
         ],
         scenarios={
             "io": dict(label="Wait for I/O, then resume", path=["runnable", "running", "waiting", "runnable", "running"]),
             "preempt": dict(label="CPU time shared between threads", path=["runnable", "running", "runnable", "running"]),
         },
         ascii="""Runnable -- CPU dispatch --> Running -- blocking wait --> Waiting
   ^                            |                          |
   +------ preempt / yield ------+                          |
   +---------------------- event / I/O wake ----------------+

Python task: await -> suspended -> resumed by its event loop
OS thread:  may keep running another callback; waits only when appropriate
GPU work:   queued -> executing -> completion; separate device dependencies"""),
]


CLIENTS_NOTE = [
    ("Connected socket", "Linux TCP + a process FD", "Can remain idle or wait while retaining connection state."),
    ("Python task", "API event loop", "Resumes when network, IPC or executor work is ready."),
    ("Waiting inference request", "ZeroMQ / MAX scheduler", "Keeps application state until model work is selected."),
    ("Active inference pass", "Worker + native device backend", "Uses the initialized model and selected request buffers."),
    ("Output waiting to send", "API / transport / TCP buffers", "Progress depends on the client and transport readiness."),
]

CLIENTS_ASCII = """many connected sockets
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
A connected client does not require its own spawned model-worker process."""
