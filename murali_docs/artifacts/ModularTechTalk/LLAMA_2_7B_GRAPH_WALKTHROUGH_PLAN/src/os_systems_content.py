"""Authored systems lessons. SOURCE calls, external mechanisms and examples stay distinct."""


def step(sender, receiver, title, evidence, summary, inputs, logic, result, scale, refs=(), docs=()):
    return dict(**{"from": sender, "to": receiver}, title=title, evidence=evidence,
                summary=summary, inputs=inputs, logic=logic, result=result, scale=scale,
                refs=list(refs), docs=list(docs))


def obj(title, intro, fields, logic, refs=(), docs=()):
    return dict(title=title, intro=intro, fields=fields, logic=logic,
                refs=list(refs), docs=list(docs))


SCENES = [
    dict(id="startup", title="Start two processes, then become ready",
         subtitle="Process creation · address spaces · files · GPU initialization · readiness",
         packet="init",
         intro="Follow the startup call stack across its OS boundaries. The parent begins waiting as soon as it spawns the worker; the animation expands what the worker does during that wait. PIDs and sizes shown in cards are example values.",
         actors=[("api", "API process", "PID 4100 · Python", "api"),
                 ("os", "Linux", "processes + memory", "linux"),
                 ("disk", "Files / page cache", "checkpoint + graph", "files"),
                 ("worker", "Model worker", "PID 4132 · Python", "worker"),
                 ("driver", "Native / driver", "MAX + NVIDIA", "driver"),
                 ("gpu", "GPU", "device memory", "gpu")],
         ascii="""API PID 4100             LINUX / FILES             WORKER PID 4132        GPU
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
Ownership: two process address spaces; explicitly shared IPC resources only.""",
         steps=[
             step("api", "api", "Build the app and server controller", "SOURCE",
                  "Click calls MAX orchestration. MAX resolves configuration and constructs the ASGI app, Uvicorn config and server controller.",
                  {"Settings.host": "0.0.0.0", "Settings.port": 8000, "config.lifespan": "off", "app": "FastAPI", "HTTP_accepting": False},
                  ["<code>PipelineConfig.from_args()</code> resolves the model and serving inputs; <code>ServingTokenGeneratorSettings</code> carries the factory, tokenizer and memory plan.",
                   "<code>fastapi_app()</code> registers routes and middleware. <code>fastapi_config()</code> reads the same Settings object and selects <code>lifespan=\"off\"</code>.",
                   "<code>Server(config)</code> creates a controller. MAX will enter its own lifespan before awaiting <code>server.serve()</code>."],
                  "The API has Python objects for serving; its HTTP accept loop has not started.",
                  "Startup object construction must be distinguished from readiness when a load balancer decides where to send traffic.",
                  ("serve", "config_call", "bundle", "app", "uvicorn_cfg", "http")),
             step("api", "os", "Spawn a fresh worker process", "SOURCE + BOUNDARY",
                  "MAX selects Python's spawn context. Linux supplies the new process and protected virtual address space.",
                  {"parent_pid": 4100, "child_pid": 4132, "start_method": "spawn", "target": "ModelWorker()", "parent_state": "waiting for readiness"},
                  ["MAX passes the deferred model factory, configuration, readiness event and IPC configuration to the worker entry point.",
                   "Python starts a fresh interpreter using OS process-creation machinery. The exact syscall sequence is Python/platform dependent; the child is not a continuation of the parent's Python stack.",
                   "The child has its own PID, heap and FD table. Linux schedules runnable threads in both processes. The parent's asynchronous readiness wait lets its event loop do other work."],
                  "Two processes exist; the child starts model initialization while the parent waits.",
                  "A model worker is a process per serving instance, not a process created for every connected client.",
                  ("spawn", "process_ready", "fd_wait"), ("spawn", "memory")),
             step("worker", "driver", "Create the native engine and device context", "SOURCE + BOUNDARY",
                  "The Python pipeline creates an InferenceSession; its implementation crosses into MAX's native bindings and the device backend.",
                  {"python_session": "InferenceSession", "native_impl": "_InferenceSession", "device": "GPU 0", "device_address_space": "context-owned"},
                  ["The Python session hands devices, thread/runtime settings and options to <code>_InferenceSession</code>.",
                   "MAX's native engine and GPU backend create or use device resources. NVIDIA's user-space driver cooperates with kernel modules for access, mappings and device management.",
                   "The diagram labels this as a boundary: the saved Python sources do not reveal every internal runtime thread, driver call or kernel ioctl."],
                  "The worker owns a session able to load executable graphs and use the selected device.",
                  "Extra replicas consume native resources and GPU memory; extra client coroutines do not each create another model context.",
                  ("pipeline_init", "session_native", "load_devices"), ("cuda_driver", "nvidia_kernel")),
             step("worker", "disk", "Read checkpoint and graph artifacts", "SOURCE + BOUNDARY",
                  "After creating and configuring its session, the text pipeline resolves weight paths and loads model data through filesystem and host-memory mechanisms.",
                  {"checkpoint": "meta-llama/Llama-2-7b-hf", "cached_weight_bytes": 13476876272, "host_buffer": "process virtual address", "file_cache": "Linux page cache"},
                  ["<code>model_factory()</code> executes the deferred partial. <code>TextGenerationPipeline</code> creates its session before resolving weight paths and passing <code>load_weights(weight_paths)</code> to model construction.",
                   "Opening and reading or mapping cached files crosses the filesystem boundary. Linux can serve file pages from RAM; a cache miss may require storage I/O. A first download also involves a network path.",
                   "Host pointers belong to the worker's virtual address space. A CPU page-table translation connects them to physical RAM. File-backed pages and Python/native allocations have different lifetimes."],
                  "The model loader obtains checkpoint/graph data in host-accessible memory.",
                  "Warm filesystem caches can reduce storage work, but each GPU replica still needs its executable state and device-resident model data.",
                  ("invoke", "pipeline_init", "load_weights"), ("memory",)),
             step("driver", "gpu", "Load weights and retain a KV pool", "SOURCE + BOUNDARY",
                  "Graph loading, compilation or cache reuse and weight initialization prepare the model. Device allocation and copies cross the driver boundary.",
                  {"weight_estimate_GiB": 12.551319, "KV_dtype": "bfloat16", "KV_bytes_per_token": 524288, "page_tokens": 128, "example_pool_GiB": 8.0625},
                  ["The Llama graph route prepares executable graphs and initializes weights. Existing compilation caches may change the cold-start work.",
                   "Native/device code allocates GPU buffers and moves required host data to them. GPU copies can use DMA; useful overlap depends on buffers, streams and dependencies.",
                   "The paged cache manager retains a GPU KV pool. This lesson's batch-4/full-context example allocates 129 × 64 MiB, including one null block. Its exact sizes come from the saved model metadata and the memory lesson."],
                  "Weights, executable state and a reusable KV pool are ready for requests.",
                  "GPU VRAM is a separate budget from Linux RAM and socket buffers; one limit cannot stand in for the other.",
                  ("load_weights", "manager_load", "plan_sizes", "bytes_per_block"), ("dma", "cuda_async")),
             step("worker", "api", "Signal readiness after initialization", "SOURCE",
                  "The worker sets up queues and the scheduler, finishes configured preparation, then signals alive on its first scheduler iteration.",
                  {"alive_event": "set", "worker_state": "initialized + scheduler loop", "API_readiness_wait": "unblocked", "HTTP_accepting": False},
                  ["Model construction and configured warmup/graph-capture preparation precede entering the worker's scheduling loop.",
                   "The worker creates its request PULL, response PUSH and cancel PULL queues and loads the scheduler with the initialized pipeline and memory plan.",
                   "Its first <code>alive.set()</code> satisfies the parent's readiness wait. Later sets are heartbeats. This event is one condition; the API-side IPC handshake must also finish."],
                  "The parent knows the worker reached its serving loop.",
                  "A live PID alone does not prove that the model, memory or queues are initialized.",
                  ("worker_run", "alive", "ready")),
             step("api", "worker", "Connect the API's IPC proxy", "SOURCE + BOUNDARY",
                  "The API creates its ZmqModelWorkerProxy and waits for request PUSH/PULL connectivity.",
                  {"request_endpoint": "ipc:///tmp/6a82c6df53f7448c91", "API_request": "PUSH", "worker_request": "PULL", "connected": True},
                  ["MAX generates unique local IPC endpoint paths. On Linux, libzmq's <code>ipc://</code> transport uses Unix-domain sockets.",
                   "<code>wait_until_connected()</code> probes request-channel writability. The API establishes this once at startup so later unwritability represents overload.",
                   "The lifespan constructs <code>TokenGeneratorPipeline</code> with tokenizer and connected proxy, then yields back to the serving coroutine."],
                  "The API frontend can hand request contexts to the initialized worker.",
                  "An IPC endpoint is a local message channel, not an extra HTTP listener or a connection to a remote GPU server.",
                  ("ipc_address", "connected", "lifespan"), ("zmq",)),
             step("api", "os", "Enter Uvicorn's HTTP serving loop", "SOURCE + BOUNDARY",
                  "Only after the lifespan yields does MAX await server.serve(). Uvicorn performs HTTP socket serving on uvloop.",
                  {"listener": "0.0.0.0:8000", "API_loop": "uvloop / libuv", "worker_loop": "scheduler.run_iteration()", "ready_for_requests": True},
                  ["Uvicorn sets up or uses the provided server sockets and starts accepting HTTP work. Prebound socket details can differ by deployment.",
                   "libuv uses Linux's I/O readiness facility, epoll. When no callbacks need work, the loop can wait for an I/O event or timer instead of busy-polling every client.",
                   "The worker independently continues scheduling queued inference work. Linux can run its threads while the API is waiting for network I/O."],
                  "The serving instance accepts requests after model readiness and IPC connectivity.",
                  "Thousands of mostly idle sockets can coexist with a small GPU batch because connection readiness and model execution are separate stages.",
                  ("http", "eventloop", "alive"), ("libuv", "epoll")),
         ]),
    dict(id="network", title="Turn network packets into a ready socket",
         subtitle="NIC rings · DMA · interrupts / NAPI · TCP queues · FDs · epoll",
         packet="pkt",
         intro="This scene zooms into one Linux host receiving a normal TCP/HTTP request. A reverse proxy may terminate TLS before this hop. The NIC ring, TCP accept queue and model-worker request queue are different objects.",
         actors=[("client", "Client / proxy", "TCP peer", "client"),
                 ("nic", "NIC", "hardware RX ring", "nic"),
                 ("driver", "NIC driver / NAPI", "kernel networking", "napi"),
                 ("tcp", "TCP + sockets", "kernel buffers", "socket"),
                 ("loop", "epoll / uvloop", "ready I/O", "eventloop"),
                 ("api", "Uvicorn / API", "HTTP + ASGI", "api")],
         ascii="""peer -> wire -> NIC -> DMA into host RX buffers
                         |
                    IRQ / NAPI poll (kernel)
                         |
                    IP / TCP processing
                         |
 new connection: SYN state -> completed accept queue -> accept() -> connection FD
 existing stream: TCP payload -> connection receive buffer -> readable FD
                         |
                   epoll ready list -> libuv callback -> HTTP parser -> ASGI task
Queues: RX descriptors != TCP listen queues != socket bytes != model request N.""",
         steps=[
             step("client", "nic", "Receive frames at the NIC", "BOUNDARY",
                  "A client or upstream proxy sends TCP packets carrying connection setup or HTTP bytes to the server's network interface.",
                  {"peer": "proxy:53122", "destination": "server:8000", "packet": "TCP payload", "existing_connection": True},
                  ["The NIC receives link-layer frames, validates the parts supported by its offload features and places work in its receive queues.",
                   "A single HTTP request may span several packets, and a packet does not necessarily correspond to one complete request.",
                   "Receive-side steering and multiple NIC queues can spread work over CPU cores; the exact queue/core arrangement is driver and host configuration dependent."],
                  "Hardware has packet data to hand to the host networking stack.",
                  "Network packet rate and byte rate are independent load dimensions; counting inference requests alone misses them.",
                  docs=("napi",)),
             step("nic", "driver", "DMA packet bytes into host RX buffers", "BOUNDARY",
                  "The driver prepares receive descriptors; the NIC DMA engine writes packet data to mapped host-memory buffers.",
                  {"RX_descriptor": "buffer DMA address + length", "buffer_capacity_example": "2048 bytes", "location": "host RAM", "GPU_VRAM_used": False},
                  ["The driver allocates/recycles buffers and gives the NIC device-visible DMA addresses. These can differ from CPU virtual and physical addresses.",
                   "An IOMMU may translate those DMA addresses to host physical pages. The device performs the byte transfer without the CPU executing a copy instruction for each byte.",
                   "CPU work is still needed to maintain descriptors, consume packets and run the protocol stack. Ordinary HTTP arrival does not DMA prompt text directly into the model's KV cache."],
                  "Host RX buffers contain bytes that kernel networking can inspect.",
                  "DMA reduces CPU copying work at this boundary; it does not remove buffer, memory-bandwidth or packet-processing costs.",
                  docs=("dma", "napi")),
             step("driver", "driver", "Use interrupts and NAPI to drain packets", "BOUNDARY",
                  "An interrupt can announce receive work; NAPI allows a driver to process a budgeted group of packets by polling.",
                  {"IRQ": "RX work notification", "NAPI_poll_budget_example": 64, "execution": "usually softirq; threaded modes also exist"},
                  ["A typical driver schedules NAPI when receive work arrives, then suppresses repeated interrupts for that queue while polling drains it.",
                   "A poll method consumes descriptors up to a budget. Continued work can trigger more polling; the driver rearms notifications when appropriate.",
                   "This balances latency and interrupt overhead under load. The exact IRQ moderation, poll budget and CPU affinity are host/driver policy."],
                  "The stack receives batches of packets rather than requiring a separate application wakeup for every frame.",
                  "At high packet rates, kernel CPU time and receive-ring drops may become bottlenecks before the model worker.",
                  docs=("napi",)),
             step("driver", "tcp", "Reassemble TCP and identify the socket", "BOUNDARY",
                  "The kernel validates and routes IP/TCP data, tracks sequence numbers and places payload into the matching TCP receive path.",
                  {"connection_key": "source/destination IP + ports", "TCP_sequence": "ordered byte stream", "socket_receive_bytes_example": 4096},
                  ["TCP tracks connection state, acknowledgements, retransmission and receive windows. The application observes an ordered byte stream, not raw RX descriptors.",
                   "For an established connection, received payload can make its socket readable. Request boundaries are subsequently recognized by the HTTP parser.",
                   "Kernel socket buffers consume host memory and may retain data while the API is busy or its transport pauses reading."],
                  "The right connection has ordered bytes available for userspace.",
                  "A slow event loop can grow socket buffers even if the GPU is idle; this is a CPU/network bottleneck.",
                  docs=("epoll", "uvicorn")),
             step("tcp", "api", "Accept new connections and create FDs", "BOUNDARY",
                  "For a new connection, TCP setup precedes an accepted socket. The listener and each accepted connection have distinct file descriptors.",
                  {"listener_fd_example": 9, "connection_fd_example": 57, "TCP_state": "ESTABLISHED", "FD_owner": "API PID 4100"},
                  ["Linux maintains incomplete handshake state and a completed connection queue for the listening socket. Backlog settings affect this connection-admission layer.",
                   "Uvicorn/libuv accepts a completed connection, obtains a new socket FD and attaches protocol state to it.",
                   "The FD is a small integer indexing this process's FD table. <code>RLIMIT_NOFILE</code> limits open descriptors; sockets, pipes and opened files all contribute."],
                  "The API owns a socket for this connection and can register its I/O readiness.",
                  "A larger TCP backlog does not enlarge the GPU batch or the model-worker request queue. They protect different resources.",
                  docs=("fdlimit", "uvicorn", "epoll")),
             step("tcp", "loop", "Add the readable FD to epoll's ready list", "BOUNDARY",
                  "epoll tracks an interest list and reports ready descriptors. It lets an event loop wait on many connections at once.",
                  {"interest_list_example": "1,000 connection FDs + listener + IPC + timers", "ready_list_example": [57, 83], "wait": "epoll_wait / equivalent"},
                  ["libuv registers sockets with Linux's readiness mechanism. When a socket's state satisfies its interest, epoll can report it ready.",
                   "A sleeping loop wakes for ready events or a timer. Only runnable callbacks consume the loop thread's CPU time.",
                   "The loop dispatches nonblocking reads and must handle partial data and <code>EAGAIN</code>. A ready FD is permission to attempt I/O, not proof that an entire HTTP request is present."],
                  "The event loop has callbacks to run for the ready connections.",
                  "1,000 connected clients can yield only two ready socket callbacks in one poll cycle; this does not require 1,000 Python threads.",
                  docs=("epoll", "libuv")),
             step("loop", "api", "Parse HTTP and create an ASGI request task", "SOURCE + BOUNDARY",
                  "Uvicorn's protocol code consumes bytes and invokes MAX's FastAPI application once the HTTP request can be handled.",
                  {"method": "POST", "path": "/v1/chat/completions", "body": "JSON messages + token limit", "request_task": "one coroutine's state"},
                  ["The HTTP implementation parses headers/body and produces ASGI receive/send events. The concrete parser depends on Uvicorn configuration.",
                   "FastAPI routing and MAX middleware resolve the endpoint, request identity and checks. Many such coroutines can share one event-loop thread.",
                   "A coroutine awaiting body bytes or worker output yields control. A synchronous CPU-heavy callback, by contrast, occupies the thread until it yields or returns."],
                  "MAX's API handler receives a structured HTTP request.",
                  "Async I/O handles waiting efficiently; CPU work still needs CPU time and careful scheduling.",
                  ("app", "token_chunks"), ("nginx", "libuv")),
             step("api", "tcp", "Keep connection pressure distinct from inference pressure", "SOURCE + BOUNDARY",
                  "Server flow control, connection limits and MAX worker admission act at different points in the path.",
                  {"open_connections_example": 1000, "submitted_requests_example": 100, "GPU_batch_example": 4, "idle_clients_example": 900},
                  ["Idle keep-alive connections still occupy socket/HTTP state. A streaming request can stay connected while waiting for model output.",
                   "Uvicorn transport flow control can pause reading/writing at buffer thresholds. Optional Uvicorn concurrency rejection has its own semantics; it is not MAX's 429 worker gate.",
                   "The lab opens a conceptual population of clients and counts request states separately. Its connection counts and buffer sizes are illustrative, not a captured Linux trace."],
                  "The lab separates client connections, HTTP tasks, queued contexts and active model sequences.",
                  "Capacity planning needs FD, RAM, CPU, egress, GPU compute and KV-memory budgets together.",
                  ("queue_settings", "proxy_submit", "http_429"), ("uvicorn", "fdlimit")),
         ]),
    dict(id="ipc", title="Hand a request from the API to the worker",
         subtitle="Validation · tokenization · CPU executor · local IPC · IDs · N and M",
         packet="R1",
         intro="Track request R1 through CPU tokenization and three local message channels. MAX's request high-water mark N and worker pending-prefill limit M default to unbounded in this pinned source; the lab chooses finite example caps.",
         actors=[("api", "API handler", "ASGI coroutine", "api"),
                 ("cpu", "CPU executor", "tokenizer work", "executor"),
                 ("proxy", "Worker proxy", "API-side queues", "ipc"),
                 ("kernel", "Unix sockets", "local IPC bytes", "socket"),
                 ("worker", "Worker scheduler", "pending contexts", "worker")],
         ascii="""HTTP JSON -> API validation -> CPU executor tokenization -> TextContext(R1)
                                      |
 API proxy: writable? -- no --> RequestQueueFull --> HTTP 429 (before SSE starts)
           | yes
           +--> request PUSH == local Unix-domain IPC ==> request PULL
                    in transit: N                         pending CE: M
                                                           |
                                                        batch selector
 API output queue(R1) <== response PULL <== IPC <== response PUSH [R1, result]
 API disconnect -------> cancel PUSH   ==> IPC ==> cancel PULL [R1]
Large NumPy payloads may use out-of-band/shared-memory encoding intra-node.""",
         steps=[
             step("api", "api", "Validate inputs and establish request identity", "SOURCE",
                  "The API converts request fields into the text-generation contract and carries a request ID through the serving path.",
                  {"request_id": "R1", "messages": [{"role": "user", "content": "Explain a TCP socket."}], "max_completion_tokens": 64, "stream": True},
                  ["The route parses and validates the JSON contract and resolves the requested model/task. Invalid input can produce an HTTP error before generation.",
                   "The tokenizer context carries a request identity and generation limits. A request ID identifies results across processes; it is not an OS PID or a socket FD.",
                   "The text-generation pipeline uses this context to connect tokenization, worker submission and output decoding."],
                  "R1 has validated inputs and an identity suitable for worker/result correlation.",
                  "Logs and traces need the request ID to follow one stream among many clients and batch iterations.",
                  ("token_context", "token_chunks", "first_chunk")),
             step("api", "cpu", "Run tokenizer encoding in the default executor", "SOURCE + BOUNDARY",
                  "TextTokenizer._encode_text() awaits run_with_default_executor so tokenizer CPU work can execute outside the event-loop callback.",
                  {"text": "example prompt after formatting", "prompt_token_count_example": 512, "executor": "default CPU executor", "API_coroutine": "awaiting encoding"},
                  ["The method wraps tokenizer encoding as a callable and submits it with MAX's <code>run_with_default_executor</code> helper.",
                   "The coroutine waits for the result while its event loop can dispatch other ready work. Linux schedules the executor's runnable CPU threads.",
                   "Executor capacity, CPU quota and tokenizer implementation determine throughput. Python/native work and the GIL affect how much useful CPU parallelism is possible; async syntax alone does not establish that."],
                  "The API receives prompt token IDs without holding its event-loop callback for the whole encoding call.",
                  "Long prompts can create a tokenizer queue even when the network loop and GPU are healthy.",
                  ("tokenizer_cpu",), ("libuv", "cgroup")),
             step("cpu", "api", "Create TextContext and cap generation length", "SOURCE",
                  "The tokenizer assembles token IDs and generation settings into the worker's context type.",
                  {"TextContext.request_id": "R1", "prompt_tokens": 512, "requested_completion_tokens": 64, "model_context_limit": 4096, "effective_completion_tokens": 64},
                  ["The context's prompt IDs, length and generation settings become inputs to batching and KV allocation.",
                   "MAX uses <code>max_tokens_to_generate</code> to determine the effective completion allowance. The request's desired output may be capped by the available context length.",
                   "For a 4,080-token prompt and a 64-token request in a 4,096-token context, an example effective allowance is 16 tokens. Do not assume every oversized requested completion causes rejection."],
                  "R1 carries a 512-token prompt and a 64-token effective generation limit in this example.",
                  "Routing and memory estimates should use token lengths and effective limits, not only request counts.",
                  ("token_context",)),
             step("api", "proxy", "Probe admission and register R1's response queue", "SOURCE",
                  "The proxy checks request-channel writability under an admission lock before pushing a context.",
                  {"request_id": "R1", "N_example": 32, "channel_writable": True, "pending_out_queues[R1]": "asyncio.Queue()"},
                  ["The request channel was connected during startup. An unwritable channel at this point signals capacity pressure.",
                   "The admission lock makes the writability probe and subsequent push a single protected submission step. A full queue raises <code>RequestQueueFull</code> immediately.",
                   "After a successful probe the proxy creates R1's output queue and pushes its context. A failed send removes the registration and initiates cleanup."],
                  "R1 is admitted, with an API-side queue ready to receive its results.",
                  "N is an approximate ZeroMQ transport high-water mark in MAX. The lab uses an exact N-slot queue to make pressure visible.",
                  ("proxy_submit", "queue_settings", "http_429")),
             step("proxy", "kernel", "Send the context over local IPC", "SOURCE + BOUNDARY",
                  "The request PUSH/PULL channel crosses the process boundary using libzmq's ipc transport.",
                  {"endpoint": "ipc:///tmp/6a82c6df53f7448c91", "payload": "serialized TextContext R1", "API_direction": "PUSH", "worker_direction": "PULL"},
                  ["libzmq serializes and buffers messages around a Unix-domain socket. Both processes remain on this host.",
                   "The kernel provides local socket transport and access control. A socket pathname identifies a communication endpoint, not a file containing prompt bytes.",
                   "The intra-node NumPy encoder can choose an out-of-band/shared-memory path for large arrays. Therefore, the diagram's arrow represents message ownership transfer and metadata as well as possible socket bytes."],
                  "The worker can receive R1's context without sharing the API's ordinary Python heap.",
                  "Large payload encoding, host-memory limits and IPC buffers matter; not every object is fully copied through socket bytes.",
                  ("ipc_address", "ipc_config"), ("zmq", "memory")),
             step("kernel", "worker", "Drain only within the pending-prefill cap M", "SOURCE",
                  "The scheduler moves admitted contexts from transport into its prefill/context-encoding set, respecting max_pending_requests when configured.",
                  {"M_example": 128, "existing_CE_contexts_example": 127, "available_drain_slots": 1, "contexts_left_in_transport": "beyond that slot"},
                  ["<code>_retrieve_pending_requests()</code> computes available capacity from <code>M - len(all_ce_reqs)</code>.",
                   "If no slot remains, it stops draining. Otherwise it limits the number pulled and enqueues those contexts with the batch constructor.",
                   "CE includes contexts awaiting or undergoing prefill, not every decoding request. A full CE set allows transport pressure to grow until the API's N gate rejects new submissions."],
                  "R1 becomes scheduler-owned pending prefill work when M has room.",
                  "N and M protect consecutive stages. Setting only N while the worker drains endlessly does not bound all worker-owned pending contexts.",
                  ("scheduler_pending", "queue_settings", "construct_batch")),
             step("worker", "proxy", "Correlate responses by request ID and batch ID", "SOURCE",
                  "The worker sends result maps; the proxy routes each result to the correct request output queue.",
                  {"response": {"R1": {"token_example": 29871, "is_done": False, "batch_id": 17}}, "API_output_queue": "pending_out_queues[R1]"},
                  ["A batch can return results for several request IDs. The proxy's response worker dispatches each result into that request's queue.",
                   "<code>_drain_responses()</code> awaits queue items and can combine already-available outputs while preserving order.",
                   "The batch ID identifies the producing forward pass for trace correlation. The API then decodes output and exposes token chunks to the route."],
                  "R1's coroutine receives its results even when it shared the GPU batch with R2–R4.",
                  "One GPU batch does not merge four clients' HTTP streams; IDs keep the outputs separate.",
                  ("proxy_responses", "proxy_drain", "token_chunks")),
             step("api", "worker", "Send cancellation on its own channel", "SOURCE",
                  "Abandoning a stream triggers proxy cleanup and a cancel message carrying the request ID.",
                  {"disconnecting_request": "R1", "cancel_payload": ["R1"], "API_cancel": "PUSH", "worker_cancel": "PULL"},
                  ["The response generator's cleanup detects an abandoned, unfinished stream and cancels worker work for that ID.",
                   "The separate cancellation channel lets the scheduler observe cancellation independently of new request messages.",
                   "Scheduler cleanup releases request resources at its execution boundaries. An in-flight operation cannot safely have its live KV pages reused by another request before that operation finishes."],
                  "R1 becomes eligible for cancellation cleanup, while other request streams continue.",
                  "Cancellation is a lifecycle transition with ownership cleanup, not a guarantee that already-submitted GPU instructions disappear immediately.",
                  ("proxy_cancel", "proxy_drain", "scheduler_iteration", "release_request")),
         ]),
    dict(id="gpu", title="Submit a batch through the runtime and GPU driver",
         subtitle="KV ownership · native boundary · control / data paths · streams · prefill / decode",
         packet="B17",
         intro="Now expand the InferenceSession and Llama execution boundary. MAX's Python call sites are source facts; the NVIDIA layers describe external mechanisms. The animation is a conceptual dependency order, not a recorded kernel or syscall trace.",
         actors=[("sched", "MAX scheduler", "request / token work", "batch"),
                 ("kv", "KV manager", "page ownership", "kv"),
                 ("native", "Native engine", "compiled executable", "native"),
                 ("user", "GPU driver", "user-space backend", "driver"),
                 ("kernel", "Linux GPU driver", "access + mappings", "kernel-driver"),
                 ("gpu", "GPU", "copy + compute", "gpu")],
         ascii="""MAX scheduler -> select R1..R4 + token budget -> KV page tables
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
No fresh allocation of all weights or all KV memory for each token.""",
         steps=[
             step("sched", "kv", "Select sequences and claim KV page ownership", "SOURCE",
                  "The batch constructor chooses eligible prefill/decode work and obtains cache resources for the selected requests.",
                  {"batch_id": 17, "request_ids": ["R1", "R2", "R3", "R4"], "batch_ceiling_example": 4, "prompt_tokens_each": 512, "prompt_pages_each": 4},
                  ["Batch selection considers request state, input-token limits and cache availability. A configured batch ceiling is not a promise that every iteration has that many requests.",
                   "For BF16 Llama 2 7B, 128 tokens occupy a 64 MiB KV page in this example. A 512-token prompt needs four pages per request.",
                   "A request's page table maps its logical token positions to pages in a retained pool. Page ownership, model context length and GPU buffer allocation are distinct concepts."],
                  "Selected requests have the cache resources and metadata required for execution.",
                  "Plenty of connected clients does not help if compute or KV pages limit the eligible batch.",
                  ("construct_batch", "geometry", "bytes_per_block")),
             step("sched", "native", "Execute the packed model inputs", "SOURCE + BOUNDARY",
                  "The text pipeline prepares inputs and calls the Llama executable through the native engine.",
                  {"packed_prompt_IDs_example": "R1[512] | R2[512] | R3[512] | R4[512]", "row_offsets_example": [0, 512, 1024, 1536, 2048], "KV_metadata": "cache inputs + page tables"},
                  ["The scheduler calls the pipeline's execute path. The model's execute method selects its executable and forwards prepared tensors and KV-related inputs.",
                   "Packed token inputs and offsets let one forward pass handle several sequences while preserving their individual positions and ownership.",
                   "Python crosses to compiled/native execution. The saved Python source proves this handoff; it does not specify each kernel's launch parameters."],
                  "The native model executable receives one batch's tensors and cache metadata.",
                  "Batch construction is MAX policy. Linux's thread scheduler cannot choose which prompts should share an LLM forward pass.",
                  ("scheduler_execute", "pipeline_execute", "llama_execute", "session_native")),
             step("native", "user", "Enqueue work through the GPU backend", "BOUNDARY",
                  "MAX's native backend submits device work through NVIDIA's user-space driver interfaces.",
                  {"command_examples": ["copy token IDs", "execute compiled kernels", "record completion dependency"], "stream": "ordered device work", "host_return": "may precede GPU completion"},
                  ["The engine binds input/output buffers and arranges executable work through its GPU backend.",
                   "The user-space driver manages device-context resources and submission mechanisms. CUDA streams express ordering among copies and kernel operations.",
                   "An asynchronous enqueue may return before execution starts or finishes. The engine must observe dependencies before it exposes results that require completion."],
                  "Device commands are submitted with their dependencies.",
                  "A fast host return does not measure GPU execution time; completion-aware timing is needed for real profiling.",
                  docs=("cuda_driver", "cuda_async")),
             step("user", "kernel", "Use the kernel driver for privileged control", "BOUNDARY",
                  "Device access, allocation/mapping and resource management require cooperation with Linux's GPU kernel modules.",
                  {"control_examples": ["device handles", "memory mappings", "context resources"], "kernel_components_example": ["nvidia", "nvidia-uvm"], "driver_version": "deployment dependent"},
                  ["User-space code cannot independently establish all privileged device mappings or access controls. The kernel driver supplies protected mechanisms.",
                   "Device initialization, allocation and certain control operations can involve opened device nodes, ioctls and memory mappings. The exact path depends on the driver and runtime.",
                   "Do not model every GPU kernel launch as a new syscall. Established channels and mapped submission resources may support later submissions without repeating the entire setup."],
                  "The process has authorized device resources and mappings for its work.",
                  "This is a control path alongside command/data submission; sending an HTTP request does not open a new GPU device context for that client.",
                  docs=("nvidia_kernel", "cuda_driver", "dma")),
             step("user", "gpu", "Move needed data, retaining device-resident state", "BOUNDARY",
                  "The GPU uses resident weights and KV data. Host/device copies move inputs or outputs when the executable needs them.",
                  {"host_input": "token IDs + metadata", "resident_GPU_data": "weights + KV pool", "link_example": "PCIe", "copy_engine": "DMA-capable device hardware"},
                  ["A copy command can move host data into device buffers. Pinned/page-locked host memory and suitable dependencies enable useful asynchronous transfer patterns.",
                   "The GPU reads resident weights and the relevant KV pages during execution. It does not reload the entire checkpoint for each generated token.",
                   "A CUDA stream orders copies and dependent kernels. Multiple streams can expose possible overlap, but hardware capacity and dependencies decide whether overlap occurs."],
                  "The device can read the required input tensors and persistent model state.",
                  "PCIe traffic, host pinning and GPU-memory bandwidth are separate costs from network packet arrival.",
                  docs=("cuda_async", "dma")),
             step("gpu", "gpu", "Prefill prompt tokens, then decode incrementally", "SOURCE + BOUNDARY",
                  "Prefill/context encoding populates KV for the prompt; later decode iterations consume past KV and extend each live sequence.",
                  {"R1_prompt_tokens": 512, "after_prefill": "prompt KV + first generated token", "next_decode_input": "latest token + cached positions", "output_limit": 64},
                  ["The model evaluates prompt tokens, potentially in chunks chosen by scheduling policy, and writes their K/V data into assigned pages.",
                   "A completed prefill can produce the first output token. Decode then advances a live autoregressive sequence, normally producing the next token per sequence per ordinary iteration.",
                   "GPU hardware schedules thread blocks/warps to execute submitted kernels. Linux manages host threads, while MAX decides the request membership and token work for the next pass."],
                  "Selected sequences advance; one request's completion can free a slot for another request in a later batch.",
                  "Continuous batching avoids waiting for every member of a static batch to finish, but actual policy and achievable rates require source configuration and measurement.",
                  ("scheduler_execute", "construct_batch", "llama_execute"), ("batching",)),
             step("gpu", "native", "Observe completion before reading token results", "SOURCE + BOUNDARY",
                  "The runtime makes device results safe for their consumer before host code reads them.",
                  {"device_result_example": "next-token ID", "host_consumer": "scheduler / API result path", "required_dependency": "producing work and necessary copy complete"},
                  ["Streams/events or another engine dependency mechanism establish when output data is ready. This diagram does not assert which internal synchronization primitive MAX uses for every operation.",
                   "The public Buffer host-conversion helper documents device-to-host availability and synchronization behavior for that interface.",
                   "Correct ownership must last through execution and required transfers. A request's KV pages cannot be recycled while a live operation can still access them."],
                  "The pipeline can return safe, correlated token results to the scheduler.",
                  "Unnecessary host/device synchronization can reduce overlap; removing a required dependency would make results incorrect.",
                  ("buffer_host", "buffer_copy", "llama_execute"), ("cuda_async",)),
             step("sched", "kv", "Release completed requests to the retained pool", "SOURCE",
                  "After execution, the scheduler advances request states and releases finished requests before sending their results.",
                  {"R1_generated_tokens": 64, "R1_state": "generation complete", "R1_page_ownership": "returned", "GPU_pool_allocation": "retained for reuse"},
                  ["The scheduler updates contexts using pipeline results and identifies completions or cancelled requests at its safe scheduling boundaries.",
                   "<code>TextBatchConstructor.release_request()</code> releases request/cache ownership. Freed pages become available for subsequent work.",
                   "The backing GPU pool remains allocated. In the lab R1 can continue in <code>sending</code> because a slow client still has buffered tokens even after its KV ownership is gone."],
                  "Request pages can be reused while API delivery of final output continues.",
                  "Allocated VRAM staying flat after a request finishes is compatible with healthy page reuse; the ownership counter is the useful distinction.",
                  ("scheduler_execute", "release_request", "manager_load")),
         ]),
    dict(id="pressure", title="Stream tokens, reject overload and clean up cancellation",
         subtitle="SSE · output queues · socket flow control · slow readers · 429 · reuse",
         packet="tok",
         intro="Streaming connects two independent rates: the worker produces results and the client consumes bytes. This source uses unbounded per-request asyncio output queues, so transport flow control alone does not bound all API output memory.",
         actors=[("gpu", "GPU / scheduler", "token producer", "batch"),
                 ("proxy", "API output queue", "per-request results", "output"),
                 ("api", "SSE / Uvicorn", "ASGI send", "sse"),
                 ("socket", "TCP send buffer", "kernel + transport", "socket"),
                 ("client", "Client", "token consumer", "client"),
                 ("kv", "KV pool", "reusable pages", "kv")],
         ascii="""worker results -> API Queue(R1) -> decoded SSE -> Uvicorn transport -> TCP -> client
   80 tok/s         grows if            send may await drain              1 tok/s
                    consumer slows

new context -> pending CE M full -> worker stops drain -> IPC N full -> HTTP 429

generation done -> return request KV pages to retained pool
               -> buffered SSE can still be draining to client
disconnect -> cancel channel -> safe scheduler boundary -> cleanup / page reuse
Already-committed SSE status cannot be replaced with a fresh HTTP status.""",
         steps=[
             step("gpu", "proxy", "Enqueue correlated token results", "SOURCE",
                  "The worker's response channel delivers results to the API proxy, which routes them to R1's output queue.",
                  {"request_id": "R1", "produced_tokens_example": 20, "consumed_tokens_example": 3, "queued_tokens_example": 17, "asyncio_queue_maxsize": 0},
                  ["Response maps can contain several request IDs from the same batch. Each ID has an independent consumer queue.",
                   "The proxy creates <code>asyncio.Queue()</code> without a maxsize, so this per-request output layer is unbounded in the pinned source.",
                   "The lab counts buffered tokens as a teaching unit. Real RAM usage also includes result objects, text/JSON framing and buffers at several layers."],
                  "R1's consumer can await queued results independently of other streams.",
                  "Many slow consumers can accumulate API RAM even if the GPU and KV allocator continue making progress.",
                  ("proxy_submit", "proxy_responses", "proxy_drain")),
             step("proxy", "api", "Resolve the first chunk before committing SSE", "SOURCE",
                  "The route pulls the first chunk before returning its EventSourceResponse.",
                  {"first_chunk": "result or early error", "HTTP_status_before_response": "not committed", "normal_stream_status": 200},
                  ["<code>_start_stream()</code> asks the generator for its first item. Early errors can still be returned as ordinary HTTP responses.",
                   "Successful admission and an output chunk allow the route to return <code>EventSourceResponse</code> for streaming.",
                   "Once response headers are committed, later problems require stream error/termination semantics; the server cannot replace an existing 200 stream with a new 429 status."],
                  "The client starts receiving an SSE response after the early error/admission decision.",
                  "This boundary explains why MAX's worker-queue rejection happens before streaming starts.",
                  ("first_chunk", "sse_response", "http_429")),
             step("api", "socket", "Send SSE bytes with transport flow control", "BOUNDARY",
                  "Uvicorn writes response bytes and can suspend ASGI send while its transport write buffer is above its high-water mark.",
                  {"frame_example": "data: {\"choices\":[...]}\n\n", "write_buffer": "userspace transport", "TCP_send_buffer": "kernel memory", "ASGI_send": "may await drain"},
                  ["The route decodes tokens and frames JSON/text as SSE. Framing, HTTP chunks and TCP packets are different boundaries.",
                   "The transport buffers pending writes; the kernel socket send buffer holds bytes awaiting network progress. Both live in host memory.",
                   "Uvicorn documents a high/low watermark mechanism: ASGI send waits when necessary until the transport drains below its low watermark."],
                  "The stream can yield while the network catches up.",
                  "Transport flow control protects that layer; upstream unbounded application queues can still grow.",
                  docs=("uvicorn",)),
             step("socket", "client", "Let a slow reader reveal buffered output", "BOUNDARY + EXAMPLE",
                  "Client/network consumption can be slower than token production, leaving output waiting after generation completes.",
                  {"generated_tokens": 64, "delivered_tokens_example": 8, "buffered_tokens": 56, "client_read_rate_example": "1 token/s", "generation_state": "done"},
                  ["TCP flow/congestion control and application read speed determine when bytes leave buffers and reach the consumer.",
                   "The lab collapses several output layers into a token buffer and advances its reader with a configured tokens/s rate.",
                   "With 56 tokens left at one token/s, the lab needs another 56 seconds to drain them. This arithmetic is an example, not a measured MAX or network performance claim."],
                  "The request remains in sending while GPU work for it is already complete.",
                  "Open streams and egress backlog can remain high while active GPU batch membership falls.",
                  docs=("uvicorn",)),
             step("gpu", "api", "Reject a new request when N fills behind M", "SOURCE",
                  "A full pending-prefill set stops worker draining; the request transport eventually becomes unwritable and MAX returns HTTP 429 for a new submission.",
                  {"M_example": 8, "CE_count": 8, "N_example": 4, "transit_count_example": 4, "new_request": "R101", "HTTP_result": 429},
                  ["<code>_retrieve_pending_requests()</code> does not drain contexts once M has no room.",
                   "<code>ZmqModelWorkerProxy.stream()</code> checks writability before registration/submission and raises <code>RequestQueueFull</code> when capacity is unavailable.",
                   "The API handler maps this to an OpenAI-compatible 429 response with a Retry-After hint. Real ZeroMQ high-water-mark enforcement is approximate; the lab's slots are exact."],
                  "The new request is rejected before SSE headers, while existing admitted requests keep their lifecycle.",
                  "Finite queues bound waiting work at a stage; they cannot increase sustained GPU service capacity. Clients need bounded retry/backoff policy.",
                  ("scheduler_pending", "proxy_submit", "http_429", "queue_settings")),
             step("client", "gpu", "Cancel work at a safe execution boundary", "SOURCE + EXAMPLE",
                  "A disconnect/abandoned generator sends cancellation. The lab holds in-flight page ownership until its current pass completes.",
                  {"request": "R1", "GPU_pass": "in flight", "state": "cancel_pending", "assigned_pages_example": 5, "reuse_now": False},
                  ["The API's generator cleanup sends the request ID over the cancel channel; the scheduler processes cancellation during its iteration logic.",
                   "The simulation marks a live pass's request <code>cancel_pending</code>. It suppresses further output and waits for that pass's completion before releasing pages.",
                   "This models the ownership invariant. It does not claim a particular native cancellation primitive or that an in-flight kernel is preempted immediately."],
                  "Cancellation returns resources safely; already-consumed tokens remain part of the client's history.",
                  "Reusing live buffers early risks cross-request corruption. Keeping cancelled work forever wastes scarce slots and memory.",
                  ("proxy_cancel", "proxy_drain", "scheduler_iteration", "release_request")),
             step("gpu", "kv", "Return KV pages before all output is delivered", "SOURCE",
                  "Finished generation releases request-owned cache pages; the API can continue draining its final results.",
                  {"R1_generation_done": True, "R1_buffered_tokens_example": 56, "R1_assigned_KV_pages": 0, "pool_GiB": 8.0625, "new_owner": "R101 may reuse pages"},
                  ["The scheduler's execution path updates contexts, releases completed requests and then sends outputs.",
                   "The cache manager retains its backing pool and marks request pages available. Subsequent requests can obtain those page identities.",
                   "Socket/output state has a different owner and lifetime. Page reuse does not require waiting until the remote client renders the last token."],
                  "KV availability can improve while buffered-output memory is still occupied.",
                  "Track allocated pool bytes, assigned pages, active sequences and output backlog separately.",
                  ("scheduler_execute", "release_request", "manager_load")),
             step("api", "client", "Finish the stream and release API ownership", "SOURCE + BOUNDARY",
                  "When generation and delivery finish, the stream's generator cleanup removes proxy output registration and the HTTP request completes.",
                  {"R1_generated": 64, "R1_delivered": 64, "R1_buffered": 0, "pending_out_queues[R1]": "removed", "TCP_connection": "can remain keep-alive"},
                  ["The consumer observes the completion result and exits its response-draining loop; cleanup removes that request's output queue.",
                   "The HTTP/SSE response ends. Depending on protocol and connection policy, the socket may close or remain available for another HTTP request.",
                   "The lab marks this as completed only after buffered tokens have been consumed. It separately records first-produced and first-delivered times to explain client-visible latency."],
                  "R1 no longer owns worker KV, a proxy result queue or an unfinished response task.",
                  "A successful GPU pass, generation completion and client-visible completion are three distinct milestones.",
                  ("proxy_drain",), ("uvicorn",)),
         ]),
    dict(id="fleet", title="Scale a fleet of independent serving replicas",
         subtitle="External routing · cost skew · warmup · readiness · failure domains · scale-out",
         packet="R42",
         intro="This scene adds an external deployment layer around the single-instance MAX path. Routing, autoscaling and replacement policies here are teaching examples, not claims that this max serve invocation implements a fleet controller.",
         actors=[("client", "Client population", "1,000 connections", "client"),
                 ("lb", "Load balancer", "external L7 policy", "balancer"),
                 ("r1", "Replica A", "API + worker + GPU", "replica"),
                 ("r2", "Replica B", "API + worker + GPU", "replica"),
                 ("ctl", "Deployment", "readiness + scaling", "control")],
         ascii="""1,000 clients -> external TLS / L7 proxy -> ready replica A: API -> worker -> GPU A
                            |             ready replica B: API -> worker -> GPU B
                            |             warming replica C: excluded until ready
                         routing
                RR / least outstanding / estimated token work / power of two

Control plane: desired replicas -> start -> initialize -> ready -> receive new work
              worker crash -> API teardown -> remove replica -> external replacement
Streams stay with their selected replica; scale-out does not move existing KV/streams.
Replicas duplicate model state; tensor-parallel shards are a different architecture.""",
         steps=[
             step("client", "lb", "Route requests through an external frontend", "BOUNDARY + EXAMPLE",
                  "A production frontend can terminate TLS, authenticate, apply limits and route HTTP requests to ready model replicas.",
                  {"front_connections_example": 1000, "ready_replicas": ["A", "B"], "routing_layer": "external reverse proxy / gateway", "streaming": "SSE"},
                  ["A frontend separates public connections from internal serving endpoints. HTTP/TLS policy and backend connection pooling depend on deployment.",
                   "It chooses a backend before sending this request. The resulting SSE stream is tied to that backend for its lifetime.",
                   "The lab treats each replica as an independent API/worker pair with one GPU and a separate model/KV pool."],
                  "New requests have a selected ready serving instance.",
                  "L7 request routing and L4 connection routing have different reuse/affinity behavior; this lab uses request-level routing.",
                  docs=("nginx", "ray", "balancing")),
             step("lb", "r1", "See how equal counts can hide unequal work", "BOUNDARY + EXAMPLE",
                  "Round robin distributes requests in turn, but long prompts and outputs make LLM requests unequal.",
                  {"A_requests_example": [{"prompt": 256, "output": 32}], "B_requests_example": [{"prompt": 3072, "output": 384}], "request_count_each": 1},
                  ["Both replicas have one request, yet B must encode twelve times as many prompt tokens and can generate twelve times as many output tokens.",
                   "Count-only policies do not see that difference. Estimated remaining token work can consider prompt work still to encode and completion tokens still to generate.",
                   "Token work is still an estimate: attention/context costs, prefix reuse, heterogeneous GPUs and future output lengths can invalidate a simplistic linear model."],
                  "Equal request counts need not mean equal queueing delay or GPU work.",
                  "Compare policies with the same workload and assumptions, then measure real SLO outcomes before adopting a production policy.",
                  docs=("balancing", "batching")),
             step("lb", "r2", "Choose among four illustrative routing policies", "BOUNDARY + EXAMPLE",
                  "The lab lets the frontend use round robin, least outstanding, least estimated token work, or power of two choices.",
                  {"round_robin": "next ready replica", "least_outstanding": "fewest unfinished streams", "token_work": "smallest estimated remaining GPU seconds", "power_of_two": "two candidates; fewest unfinished streams"},
                  ["Round robin needs minimal load information. Least outstanding uses unfinished request counts, including slow streams still sending.",
                   "Estimated token work uses this lab's linear prefill/decode assumptions and treats generation-finished streams as zero remaining GPU work.",
                   "Power of two samples two ready replicas and compares their outstanding counts. The lab uses a fixed random seed so repeated experiments are reproducible."],
                  "The selected policy assigns only new requests; active requests retain their existing owner.",
                  "Different policies optimize different signals. GPU-work routing can ignore an egress/RAM problem caused by slow consumers.",
                  docs=("balancing",)),
             step("r1", "ctl", "Report several capacity signals", "BOUNDARY + EXAMPLE",
                  "A deployment controller should distinguish readiness and load, and observe more than client count.",
                  {"signals": ["pending CE + IPC depth", "oldest / p95 TTFT", "active batch", "assigned KV pages", "output backlog", "CPU / RAM / egress"], "open_connections": "not GPU batch size"},
                  ["Readiness decides whether new traffic may reach a replica. Load and SLO signals decide whether the fleet needs more capacity.",
                   "Queue growth plus latency can indicate an arrival/service imbalance. GPU activity by itself cannot reveal tokenizer starvation or slow-client buffers.",
                   "The lab displays simulated queue depth, device busy fraction, delivered token rate and delivered TTFT. Production measurements require the actual workload, versions and timing method."],
                  "The controller has signals that explain which stage is under pressure.",
                  "Scale the limiting resource; extra GPUs do not cure a saturated frontend or network link.",
                  docs=("ray",)),
             step("ctl", "r2", "Warm a new replica before routing to it", "SOURCE + BOUNDARY",
                  "Scale-out starts another serving instance and waits for its initialization and readiness.",
                  {"new_replica": "C", "initial_state": "warming", "lab_warmup_seconds": 2, "real_warmup": "workload / hardware / cache dependent", "receives_new_requests": False},
                  ["The external deployment starts the instance. It follows the same configuration, graph/weight loading, queue and readiness path as scene 1.",
                   "A warming replica is excluded from routing. The lab's two-second warmup compresses this lifecycle for teaching and is not a MAX startup measurement.",
                   "Once ready, C owns its own model state and KV pool. Replication increases aggregate capacity by adding a GPU; it does not shard one model over A, B and C."],
                  "C becomes eligible for new requests only after its readiness gate.",
                  "Cold-start time makes scale-out lag demand; warm capacity and bounded admission can absorb the gap.",
                  ("spawn", "ready", "connected", "http"), ("ray",)),
             step("lb", "r2", "Keep existing requests attached during scale-out", "BOUNDARY + EXAMPLE",
                  "New traffic can use the added replica, while already queued or streaming requests remain on their original one.",
                  {"R1_owner": "A", "R1_page_table": "GPU A", "new_R101_owner": "C", "migrate_existing_requests": False},
                  ["A owns its pending contexts, request IDs, device KV and connection response path. Moving them requires an explicit migration protocol.",
                   "The lab adds C for subsequent routing decisions and leaves A's queues intact, so old backlog may drain slowly even after scale-out.",
                   "This is why fleet arrival capacity can improve sooner than the latency of requests that were already waiting on an overloaded instance."],
                  "Scale-out changes future assignments; it does not teleport existing work.",
                  "A replica-local queue can preserve imbalance. Shared queues or migration systems have different designs and costs.",
                  docs=("ray", "balancing")),
             step("r1", "ctl", "Handle a worker crash and fail its streams", "SOURCE + BOUNDARY",
                  "This MAX lifecycle propagates a worker failure and tears down the serving context. External systems remove and replace the failed replica.",
                  {"replica": "A", "worker_state": "exited", "API_effect": "serving context tears down", "ongoing_streams": "failed", "transparent_replay": False},
                  ["MAX watches process exit/readiness/heartbeat state. A worker error surfaces through process-control task cancellation and serving teardown.",
                   "The external frontend/controller observes failure and stops choosing A. The lab's Crash action fails that replica's unfinished requests and returns no invented continuation tokens.",
                   "A client retry is a new request with application-specific replay policy. Already-delivered SSE output and KV state are not automatically migrated."],
                  "The failed instance stops receiving traffic; other replicas remain independent.",
                  "More replicas limit a failure's scope, but retry storms and non-idempotent surrounding actions need deployment/application policy.",
                  ("fd_wait", "process_ready", "serve", "start_worker"), ("ray",)),
             step("ctl", "r1", "Replace, initialize and rejoin the fleet", "BOUNDARY + EXAMPLE",
                  "The lab's Replace action models an external supervisor starting a replacement API/worker pair.",
                  {"replica": "A", "lifecycle": ["down", "warming", "ready"], "old_streams": "stay failed", "new_requests": "eligible after ready"},
                  ["The external policy creates fresh serving state and repeats initialization. Model/cache availability affects the real recovery time.",
                   "A becomes ready again and may receive future traffic. The lab retains failed request records for inspection without replaying them.",
                   "Capacity accounting includes each ready replica's GPU, CPU, RAM and egress. Independent replicas duplicate model memory; tensor/pipeline parallelism must be designed separately."],
                  "The fleet restores capacity without pretending the failed streams resumed.",
                  "Serving at scale combines fast data-plane loops with slower deployment, health, scaling and recovery control loops.",
                  docs=("ray", "nginx")),
         ]),
]


OBJECTS = {
    "api": obj("API process / ASGI request state",
               "SOURCE + BOUNDARY. The API process owns HTTP routing, tokenizer access, request coroutines and the worker proxy. PIDs, FDs and queue contents below are examples.",
               {"pid": 4100, "event_loop": "uvloop", "listener": "0.0.0.0:8000", "connection_fd": 57, "request_id": "R1", "worker_proxy": "ZmqModelWorkerProxy"},
               ["A process has an address space and FD table. Threads share those process resources; Python coroutines are resumable program state scheduled by the event loop.",
                "An HTTP task can wait for tokenization or results while the loop handles another socket. Runnable CPU work still needs a Linux-scheduled thread.",
                "Each live request has its own identity, response state and output consumer. One listener accepts many connections."],
               ("app", "lifespan", "token_chunks"), ("libuv", "epoll")),
    "linux": obj("Linux: runnable threads, virtual memory and FDs",
                 "BOUNDARY. Linux isolates processes, schedules CPU threads and provides protected I/O and memory mechanisms.",
                 {"API_pid": 4100, "worker_pid": 4132, "API_thread_example": "event loop", "worker_thread_example": "scheduler/native runtime", "shared_between_processes": "only explicit resources"},
                 ["Linux schedules runnable threads onto CPUs. It does not inspect prompt lengths to decide an LLM batch.",
                  "Each process uses virtual addresses translated through page tables. Shared mappings are explicit; identical numeric pointers in two processes need not refer to the same bytes.",
                  "FDs reference open kernel objects. Socket readiness lets a waiting event loop become runnable; a GPU wait and a socket wait are different conditions."],
                 docs=("memory", "spawn", "epoll", "cgroup")),
    "files": obj("Checkpoint files, storage and Linux page cache",
                "SOURCE + BOUNDARY. Weight loading starts in the worker; filesystem pages, native allocations and device copies belong to different layers.",
                {"model": "meta-llama/Llama-2-7b-hf", "selected_weight_bytes": 13476876272, "weight_estimate_GiB": 12.551319, "host_pages": "cached / mapped / copied as needed"},
                ["Cached model files may be read or mapped into the process. The Linux page cache can satisfy repeated file access from RAM.",
                 "Physical RAM backs resident CPU pages; disk provides durable file content. A virtual mapping alone does not prove all file bytes are resident.",
                 "GPU weights are a separate resident copy/state. A filesystem cache hit does not by itself initialize a new GPU replica."],
                ("load_weights",), ("memory",)),
    "worker": obj("Model worker and pending context ownership",
                 "SOURCE. A separately spawned worker owns the initialized model pipeline, scheduler, request contexts and device cache management.",
                 {"pid": 4132, "entry": "ModelWorker.__call__ -> run", "queues": {"request": "PULL", "response": "PUSH", "cancel": "PULL"}, "M_example": 128},
                 ["The worker performs model construction and preparation before its first readiness signal. It remains in the scheduling loop while the API serves.",
                  "Pending CE contexts, active decode contexts and IPC in-transit messages are distinct sets. M bounds the CE set when configured.",
                  "The scheduler retrieves work, selects batches, executes, advances contexts, releases completed work and sends results."],
                 ("spawn", "worker_run", "scheduler_pending", "scheduler_iteration", "scheduler_execute")),
    "eventloop": obj("epoll, libuv and uvloop",
                    "BOUNDARY. epoll reports kernel I/O readiness; libuv implements the event loop; uvloop exposes an asyncio-compatible loop used by MAX.",
                    {"interest_example": "listener + connections + IPC / process-watch FDs", "ready_example": [57, 83], "loop_thread": "one thread per loop", "tasks": "many Python coroutines"},
                    ["The kernel's interest/ready lists avoid requiring one sleeping thread per socket. Readiness is permission to attempt nonblocking I/O.",
                     "The loop dispatches ready callbacks, timers and resumed coroutines. Long synchronous work delays all callbacks on that loop thread.",
                     "MAX's process-control helper uses <code>loop.add_reader()</code> to await a child's exit sentinel without a blocking join in the ordinary event loop."],
                    ("eventloop", "fd_wait"), ("epoll", "libuv", "nginx")),
    "executor": obj("Tokenizer CPU executor",
                   "SOURCE + BOUNDARY. TextTokenizer._encode_text() delegates an encoding callable to run_with_default_executor.",
                   {"request": "R1", "prompt_tokens_example": 512, "executor_threads_lab": 2, "tokenizer_rate_lab": "20,000 prompt tokens/s per thread"},
                   ["The API awaits the callable's result while its loop can handle other ready work. This does not make the tokenizer's CPU cost vanish.",
                    "The lab uses two example tokenizer servers with linear per-token service time. Actual native/Python work, GIL behavior, CPU quota and executor sizing require measurement.",
                    "If tokenization is the bottleneck, requests collect in the API stage before reaching the GPU's admission and batch queues."],
                   ("tokenizer_cpu",), ("libuv", "cgroup")),
    "client": obj("Connections, requests and client readers",
                 "BOUNDARY + EXAMPLE. The lab uses a fixed population of clients with at most one unfinished request each; idle connections stay visible.",
                 {"connections": 1000, "requests_submitted_at_snapshot": 100, "idle_clients": 900, "batch_members": 4, "reader_rates_lab": {"normal": "120 tokens/s", "slow": "1 token/s"}},
                 ["A connection can be idle, sending a request or consuming a long SSE response. These states have different CPU, network and model costs.",
                  "The lab assigns one unfinished stream per client and chooses a replica at request admission. An L7 proxy's backend connection pool may differ from this illustrative mapping.",
                  "Reader rates represent consumption of complete token units in the simulation, not an actual socket bandwidth measurement."],
                 docs=("nginx", "uvicorn")),
    "nic": obj("NIC RX ring and DMA buffer addresses",
              "BOUNDARY. The receive ring is hardware/driver packet state in host memory; it is unrelated to model KV pages.",
              {"descriptor_example": {"DMA_address": "device-visible mapping", "buffer_length": 2048}, "packet_buffer": "host RAM", "notification": "IRQ / NAPI"},
              ["The driver prepares descriptor/buffer ownership, and the NIC writes received packet bytes through DMA.",
               "CPU virtual, CPU physical and DMA addresses can differ. An IOMMU can translate device-visible addresses to host physical pages.",
               "Packet handling still consumes CPU time and memory bandwidth. Ordinary HTTP input follows the host network stack before tokenization and model input preparation."],
              docs=("dma", "napi")),
    "napi": obj("NIC driver, IRQ and NAPI poll work",
               "BOUNDARY. NAPI coordinates receive notification and budgeted polling so high packet rates do not require unbounded interrupt work.",
               {"trigger": "receive queue work", "poll_budget_example": 64, "usual_context": "softirq", "variants": "threaded NAPI / busy polling exist"},
               ["An IRQ can schedule NAPI; the driver polls descriptors and passes packet data to networking, subject to a budget.",
                "Interrupt moderation and queue/CPU affinity are driver/OS settings. This page does not claim a particular production setting.",
                "Kernel packet work shares host CPUs with application threads. CPU saturation at this layer can delay requests despite unused GPU slots."],
               docs=("napi",)),
    "socket": obj("Socket FD, kernel buffers and listen queues",
                 "BOUNDARY. A socket is an OS object; its FD is a process-local reference. Connection setup queues and established-stream byte buffers have separate lifetimes.",
                 {"listener_fd_example": 9, "connected_fd_example": 57, "receive_buffer": "incoming bytes", "send_buffer": "outgoing bytes", "RLIMIT_NOFILE": "per-process descriptor limit"},
                 ["TCP provides an ordered byte stream, retransmission and flow control. The HTTP parser above it identifies request/response boundaries.",
                  "The listener's incomplete/complete connection queues precede accept. An accepted socket has its own buffers and event-loop registration.",
                  "Socket/transport buffers use host memory. Increasing connection backlog or FD limits does not create GPU KV pages."],
                 docs=("epoll", "fdlimit", "uvicorn")),
    "ipc": obj("Three local ZeroMQ channels and response registration",
              "SOURCE + BOUNDARY. This single-instance path generates ipc:// endpoint names and connects API and worker with local Unix-domain sockets.",
              {"request": "API PUSH -> worker PULL", "response": "worker PUSH -> API PULL", "cancel": "API PUSH -> worker PULL", "N_default": None, "M_default": None, "R1_output_queue": "unbounded asyncio.Queue"},
              ["N configures the request transport's approximate high-water mark. M bounds the worker's CE/prefill set and can stop transport draining.",
               "The API gates submission under a lock and raises RequestQueueFull when the connected request channel is unwritable.",
               "Intra-node encoding can use shared-memory/out-of-band NumPy payloads. Remote/disaggregated paths are separate designs."],
              ("ipc_address", "ipc_config", "proxy_submit", "scheduler_pending", "queue_settings"), ("zmq",)),
    "batch": obj("MAX request/token batch selection",
                "SOURCE. The scheduler chooses model work; a batch ceiling, context length and input-token budget describe different constraints.",
                {"batch_ceiling_lab": 4, "input_tokens_per_pass_lab": 1024, "active_ids": ["R1", "R2", "R3", "R4"], "prefill": "prompt work", "decode": "live sequence work"},
                ["An eligible request must have required model/cache resources. A batch may be smaller than the ceiling when queues, memory or state limit it.",
                 "The lab splits prompt work into chunks and includes one next-token step for ordinary decoders. Its simplified linear rates and reservation policy are not the actual MAX scheduling implementation.",
                 "Completing a request can free a slot for new work on a later iteration, even while other sequences remain in decode."],
                ("construct_batch", "scheduler_execute", "preempt_request"), ("batching",)),
    "kv": obj("Allocated KV pool versus request page ownership",
             "SOURCE + EXAMPLE. BF16 Llama 2 7B uses 512 KiB of K+V per token in this one-device geometry; a 128-token block is 64 MiB.",
             {"layers": 32, "KV_heads": 32, "head_dim": 128, "dtype_bytes": 2, "K_plus_V": 2, "token_bytes": 524288, "page_bytes": 67108864, "pool_blocks_example": 129, "usable_pages": 128},
             ["Compute K+V bytes as 2 × layers × KV heads × head dimension × dtype bytes. GPU KV blocks are token-oriented model allocations, not Linux 4 KiB virtual-memory pages.",
              "The pool retains its backing device allocation. Request page tables assign portions of that pool and release ownership at safe boundaries.",
              "The lab reserves pages for prompt plus effective completion when it admits a GPU member, while assigned pages grow with actual tokens. MAX has its own allocation/preemption logic.",
              "At 512 prompt + 64 output, conservative reservation is ceil(576/128) = 5 pages. Initially only four prompt pages per request are assigned."],
             ("geometry", "bytes_per_block", "manager_load", "release_request", "preempt_request")),
    "native": obj("MAX native execution boundary",
                 "SOURCE + BOUNDARY. Python builds and invokes the pipeline; native MAX executes compiled graphs and manages backend resources.",
                 {"Python": "InferenceSession / pipeline / model execute", "native": "_InferenceSession and executable bindings", "backend": "NVIDIA GPU in this lesson"},
                 ["The Python call site provides model inputs and devices. The session stores a native implementation object.",
                  "The compiled executable can dispatch many kernels and memory operations for one forward pass; one Python call is not one GPU instruction.",
                  "Internal native threads, stream choices and driver submissions are boundary mechanisms here. Real traces are needed to label their exact times and counts."],
                 ("session_native", "llama_execute", "pipeline_execute"), ("cuda_driver", "cuda_async")),
    "driver": obj("NVIDIA user-space driver and context resources",
                 "BOUNDARY. The native backend reaches user-space GPU driver interfaces, which cooperate with kernel modules and device firmware.",
                 {"example_library": "libcuda.so", "context": "device-associated resources/address space", "commands": "copy / kernels / dependencies", "privileged_control": "kernel-driver cooperation"},
                 ["The CUDA driver API exposes lower-level device/context/resource control. This page does not assert which specific API MAX uses at every boundary.",
                  "Established submission resources can be reused. Allocation/control and command execution are not the same path or the same frequency.",
                  "Asynchronous work needs dependency tracking before outputs are safely read or buffers recycled."],
                 docs=("cuda_driver", "cuda_async", "nvidia_kernel")),
    "kernel-driver": obj("Linux GPU kernel modules and protected control",
                        "BOUNDARY. Kernel components participate in device access, memory mappings, resource isolation and management.",
                        {"example_modules": ["nvidia", "nvidia-uvm"], "possible_interfaces": ["device node", "ioctl", "mmap"], "submission_path": "driver/version dependent"},
                        ["User-space GPU code uses protected interfaces to establish and manage device resources. Device mappings also involve OS memory and address translation.",
                         "NVIDIA's open-kernel-module reference distinguishes kernel modules, firmware and matching user-space driver components.",
                         "Avoid equating every kernel launch with one ioctl, or a GPU device reset with a Linux thread context switch."],
                        docs=("nvidia_kernel", "dma", "cuda_driver")),
    "gpu": obj("GPU compute, copy engines and device memory",
              "BOUNDARY. The GPU executes submitted commands using its own compute scheduling and device address space.",
              {"device_example": "GPU 0", "VRAM_available_lab_GiB": 24, "weight_estimate_GiB": 12.551319, "persistent_state": "weights + KV pool", "work": "prefill / decode kernels + copies"},
              ["GPU hardware schedules kernel blocks/warps over its compute resources. Linux schedules the host threads that submit work and handle I/O.",
               "Streams express operation order and events/dependencies express readiness. Multiple streams permit possible concurrency without guaranteeing it.",
               "Compute, GPU-memory bandwidth, transfer bandwidth and capacity can each limit serving. The lab's busy fraction is simulated service occupancy, not NVIDIA utilization telemetry."],
              docs=("cuda_async", "cuda_driver")),
    "output": obj("Per-request output buffering",
                 "SOURCE + EXAMPLE. The proxy's output queues are unbounded in the pinned source. The lab combines application, transport and kernel output into a visible token counter.",
                 {"R1_generated_example": 64, "R1_delivered_example": 8, "R1_buffered": 56, "R1_KV_owned": 0, "request_state": "sending"},
                 ["Worker responses enter the appropriate asyncio queue even while a slow consumer is blocked further down the response path.",
                  "Uvicorn's transport flow control applies to its own buffers. It is not a global bound on every upstream result queue.",
                  "The simulation conserves generated tokens as delivered + buffered + discarded. Cancellation/failure can discard undelivered tokens, which is different from client delivery."],
                 ("proxy_submit", "proxy_responses", "proxy_drain"), ("uvicorn",)),
    "sse": obj("SSE framing and HTTP status commitment",
              "SOURCE + BOUNDARY. The API frames decoded text as event-stream data and uses Uvicorn's ASGI send path.",
              {"content_type": "text/event-stream", "status": 200, "frame_example": "data: {\"choices\":[...]}\n\n", "early_worker_overload": "HTTP 429 before headers"},
              ["The route resolves the first generator item before returning EventSourceResponse, preserving ordinary HTTP errors at the early boundary.",
               "Once headers are sent, errors have in-stream/termination semantics. HTTP status cannot be rewritten for an already-started response.",
               "Output chunks can combine tokens. SSE frames, tokens, HTTP chunks and network packets need not align one-to-one."],
              ("first_chunk", "sse_response", "http_429"), ("uvicorn",)),
    "balancer": obj("External request routing",
                   "BOUNDARY + EXAMPLE. This page surrounds MAX instances with an illustrative L7 frontend; max serve itself is the per-instance server here.",
                   {"policies": ["round robin", "least outstanding", "estimated token work", "power of two"], "eligible": "ready replicas only", "stream_affinity": "selected replica"},
                   ["Round robin spreads counts; least outstanding considers unfinished streams; token work estimates remaining model effort; power of two compares a sampled pair.",
                    "Request costs vary with prompt/output length and cache state. A count of open connections can hide idle clients and expensive generation.",
                    "The lab uses deterministic routing and simplified work estimates. A real frontend also needs health, draining, timeout and connection-pooling policy."],
                   docs=("balancing", "nginx", "ray")),
    "replica": obj("Independent MAX replica / one GPU",
                  "SOURCE + BOUNDARY. A lab replica contains one API process, one model worker, a model/KV allocation and one conceptual GPU.",
                  {"A": {"API_pid_example": 4100, "worker_pid_example": 4132, "GPU_example": 0}, "queues": "replica local", "states": ["warming", "ready", "down"]},
                  ["Each instance follows the pinned MAX startup path and has local request/result/cancel channels.",
                   "Adding a lab replica adds a GPU plus its associated CPU/RAM and model state. It does not split existing requests or one model's tensor operations across GPUs.",
                   "Requests stay attached to their chosen replica. A crashed instance's unfinished streams fail; replacement is an external policy."],
                  ("spawn", "ready", "connected", "http"), ("ray",)),
    "control": obj("External deployment, readiness and recovery",
                  "BOUNDARY + EXAMPLE. A slower control plane starts/removes replicas and responds to readiness, failures and capacity signals.",
                  {"lifecycle": "start -> warming -> ready -> down -> replacement", "lab_warmup_seconds": 2, "autoscaling_input_examples": "queues + latency + resource pressure"},
                  ["Readiness decides traffic eligibility; health/failure detection removes broken instances. Scaling chooses desired capacity.",
                   "Cold initialization delays useful new capacity. Requests already queued elsewhere are not automatically reassigned.",
                   "MAX propagates worker exit through its serving lifecycle. The lab's Add/Crash/Replace actions model external fleet policy, not an embedded MAX autoscaler."],
                  ("serve", "process_ready", "fd_wait"), ("ray",)),
}


def systems_map():
    return """
<div class="ownership-map" aria-label="Ownership and resource boundaries">
  <div class="map-lane outside"><div class="lane-title">Outside this instance <span class="badge">BOUNDARY</span></div>
    <div class="map-row"><button data-card="object:client"><strong>Clients</strong><small>connections + readers</small></button><span class="map-arrow">HTTP / SSE ⇄</span><button data-card="object:balancer"><strong>Gateway / load balancer</strong><small>external fleet policy</small></button></div></div>
  <div class="map-lane userspace"><div class="lane-title">Host user space <span class="badge">SOURCE</span></div>
    <div class="map-row"><button data-card="object:api"><strong>API process</strong><small>Uvicorn → FastAPI → tokenizer → proxy</small></button><span class="map-arrow">local IPC ⇄</span><button data-card="object:worker"><strong>Model worker</strong><small>scheduler → pipeline → native engine</small></button></div></div>
  <div class="map-boundary"><span>socket reads / writes ↓</span><span>device control + submission ↓</span></div>
  <div class="map-lane kernelspace"><div class="lane-title">Linux kernel <span class="badge">BOUNDARY</span></div>
    <div class="map-row"><button data-card="object:socket"><strong>Network + local sockets</strong><small>TCP · Unix IPC · epoll · FDs</small></button><button data-card="object:linux"><strong>CPU + virtual memory</strong><small>threads · address spaces · files · cgroups</small></button><button data-card="object:kernel-driver"><strong>GPU kernel modules</strong><small>device access · allocation · mappings</small></button></div></div>
  <div class="map-lane hardware"><div class="lane-title">Hardware <span class="badge">BOUNDARY</span></div>
    <div class="map-row"><button data-card="object:nic"><strong>NIC ⇄ host RAM</strong><small>RX / TX DMA + CPU processing</small></button><span class="map-arrow">host/device copies ⇄</span><button data-card="object:gpu"><strong>GPU + VRAM</strong><small>weights · KV pages · compute / copy engines</small></button></div></div>
</div>
<p class="map-caption">The kernel provides protected control and mappings. Established GPU command/data paths can be reused; a per-token launch does not repeat the entire initialization path. Click a component for its ownership and example objects.</p>
<div class="scheduler-grid">
 <button data-card="object:linux"><span class="eyebrow">01 · Linux scheduler</span><strong>Runs CPU threads</strong><small>Which runnable host thread gets CPU time?</small></button>
 <button data-card="object:eventloop"><span class="eyebrow">02 · uvloop / libuv</span><strong>Dispatches ready I/O</strong><small>Which callback or coroutine can proceed?</small></button>
 <button data-card="object:batch"><span class="eyebrow">03 · MAX scheduler</span><strong>Chooses model work</strong><small>Which requests and tokens form this batch?</small></button>
 <button data-card="object:gpu"><span class="eyebrow">04 · GPU hardware</span><strong>Executes device commands</strong><small>Which kernel blocks/warps run on the device?</small></button>
</div>
<details class="text-fallback"><summary>Portable component map</summary><pre>Clients -- HTTP/SSE -- external frontend -- MAX API process
                                               |
                                      local request/result/cancel IPC
                                               |
                                          model worker
                                               |
                                      native MAX / user driver
                                               |
                                     GPU commands + device data
                                               v
                                           GPU / VRAM
Linux: CPU thread scheduling + virtual memory + files + sockets + epoll
       + NIC driver/NAPI + GPU kernel-driver access/control/mappings</pre></details>"""


DEEP_DIVES = """
<details class="deep-dive" id="queues"><summary>01 · Which queue is full?</summary>
<p>Read the queue name and its unit before changing a limit. Connection slots, bytes, request contexts and KV token pages cannot substitute for one another.</p>
<div class="table-scroll"><table><thead><tr><th>Layer / owner</th><th>What waits</th><th>Pressure / limit</th><th>Follow it</th></tr></thead><tbody>
<tr><td>NIC / kernel driver</td><td>Packet descriptors and host buffers</td><td>Ring capacity, packet CPU budget; drops possible</td><td><button data-card="object:napi">Driver card</button></td></tr>
<tr><td>TCP listener</td><td>Connection handshakes and completed accepts</td><td>Listen backlog / kernel policy</td><td><button data-card="object:socket">Socket card</button></td></tr>
<tr><td>Established TCP + transport</td><td>Incoming/outgoing bytes</td><td>Buffer watermarks, flow control, RAM</td><td>{{doc:uvicorn|Transport flow control}}</td></tr>
<tr><td>API / tokenizer</td><td>HTTP tasks and CPU encoding</td><td>CPU/executor capacity; optional external admission</td><td>{{ref:tokenizer_cpu}}</td></tr>
<tr><td>Request ZeroMQ transport N</td><td>Admitted contexts in transit</td><td>Approximate HWM; defaults to unbounded</td><td>{{ref:proxy_submit}}</td></tr>
<tr><td>Worker CE/prefill set M</td><td>Contexts waiting for / undergoing prefill</td><td>Configured max_pending_requests; stops drain</td><td>{{ref:scheduler_pending}}</td></tr>
<tr><td>GPU batch / KV manager</td><td>Selected sequences and token work</td><td>Batch ceiling, input budget, pages, compute</td><td>{{ref:construct_batch}}</td></tr>
<tr><td>API per-request outputs</td><td>Generated result objects</td><td>Unbounded asyncio.Queue in this source</td><td>{{ref:proxy_submit}}</td></tr>
</tbody></table></div>
<pre class="flow-pre">M fills -> worker stops pulling -> N fills -> new submission fails -> HTTP 429
slow client -> transport send waits -> API output can accumulate -> RAM pressure</pre>
</details>
<details class="deep-dive" id="memory-spaces"><summary>02 · Which memory does this page belong to?</summary>
<div class="memory-spaces"><div><strong>Linux host RAM</strong><p>Python heaps, tokenizer IDs, checkpoint page cache, IPC/shared buffers and sockets. Virtual addresses map to resident physical pages.</p><small>OS pages: architecture dependent; 4 KiB is common.</small></div><div><strong>GPU device memory</strong><p>Weights, executable buffers, temporary workspace and the retained KV pool. Native/driver code controls the device allocation.</p><small>KV page here: 128 tokens × 512 KiB = 64 MiB.</small></div></div>
<pre class="flow-pre">host virtual pointer -> CPU page tables -> physical RAM
device DMA address   -> possible IOMMU  -> mapped RAM
GPU KV logical token -> request page table -> retained VRAM page</pre>
<p>A Linux page and a model KV page use different units and mappings. Releasing request page ownership returns space inside the existing pool; it does not necessarily lower allocated VRAM.</p>
<p>{{ref:host_memory_limit}} · {{ref:bytes_per_block}} · {{ref:release_request}} · {{doc:memory}} · {{doc:dma}}</p>
</details>
<details class="deep-dive" id="latency"><summary>03 · Where does client-visible latency come from?</summary>
<pre class="flow-pre">request start
  -> frontend/network -> API CPU queue + tokenization -> IPC / pending queue
  -> prefill + first token produced -> output / network -> first token delivered
                                                          [client TTFT]
  -> later decode / output -> generation complete -> final bytes delivered
                                                     [end-to-end latency]</pre>
<p>First-produced time is the model side's milestone. TTFT in the lab uses first-delivered time and includes its queues and reader delay. Queues grow when arrival work exceeds service capacity; more waiting slots postpone rejection but do not make a GPU faster.</p>
<div class="equation">Little's law, for a stable boundary: average in-system requests L ≈ accepted request rate λ × average time W.</div>
<p>For an illustrative 20 accepted requests/s and a five-second average lifetime, L ≈ 100 unfinished requests. Count idle sockets separately. This relationship uses consistent averages over a stable interval; it does not predict a growing overload queue or p95 latency.</p>
<p>The lab uses 10 ms time steps, fixed linear service rates, one serialized conceptual GPU service timeline per replica, finite example N/M and a conservative reservation policy. It omits real kernel mixes, GPU occupancy, bandwidth contention, prefix reuse, speculative execution, TLS cost and network retransmission. Charts show the model's behavior under those assumptions.</p>
</details>
<details class="deep-dive" id="host-limits"><summary>04 · Why can a server slow down while its GPU has room?</summary>
<div class="table-scroll"><table><thead><tr><th>Observed stage</th><th>Resource to inspect</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>API/encoding backlog</td><td>Host CPU, executor work, cgroup cpu.max</td><td>Tokenizer or event-loop work can starve GPU submission.</td></tr>
<tr><td>Connection failures</td><td>FD usage/limits, listen queues, proxy limits</td><td>The host can run out of connection resources before model admission.</td></tr>
<tr><td>Many sending requests</td><td>Result buffers, transport/TCP bytes, egress, readers</td><td>Model generation may be finished while delivery still consumes RAM/sockets.</td></tr>
<tr><td>Pending work with small batches</td><td>KV availability, token budgets, scheduler policy</td><td>A batch ceiling is a maximum, not a requirement that the device be full.</td></tr>
<tr><td>Replica death under RAM pressure</td><td>cgroup memory.max, Linux OOM events</td><td>Host RAM limits do not describe the GPU KV budget.</td></tr>
</tbody></table></div>
<p>{{doc:cgroup}} · {{doc:fdlimit}} · {{doc:uvicorn}} · {{ref:host_memory_limit}}</p>
</details>
<details class="deep-dive" id="production"><summary>05 · What changes for thousands of clients?</summary>
<pre class="flow-pre">public clients -> gateway / limits / TLS -> ready MAX replicas -> GPU work
                      |                       |
                    egress             per-replica N / M / KV
                      +----- metrics / traces / health -----+
                                          |
                                 deployment / scaling / recovery</pre>
<div class="table-scroll"><table><thead><tr><th>Design decision</th><th>Reason</th></tr></thead><tbody>
<tr><td>Reuse event-driven connections</td><td>Many clients spend time waiting. Allocate CPU to ready work and budget connection RAM/FDs.</td></tr>
<tr><td>Bound admitted waiting work</td><td>Separate transport, pending-model and output memory. Reject before unbounded queues destroy latency.</td></tr>
<tr><td>Use token-aware workload estimates</td><td>Prompt/output length and resident KV make requests unequal; raw client count hides cost.</td></tr>
<tr><td>Warm and health-gate replicas</td><td>New GPU capacity becomes useful only after weights, executables, queues and readiness.</td></tr>
<tr><td>Plan stream affinity and failure policy</td><td>Long-lived responses stay with their owner. A failed stream needs an explicit retry/recovery contract.</td></tr>
<tr><td>Measure client SLOs and stage pressure</td><td>TTFT/ITL, queues, CPU, RAM, egress and GPU memory explain different bottlenecks.</td></tr>
</tbody></table></div>
<p>{{doc:nginx}} supplies the event-driven connection model; {{doc:balancing}} supplies interactive queue/routing intuition; {{doc:batching}} supplies iteration-level batching intuition; {{doc:ray}} supplies proxy/replica/control-plane examples. Their architecture and historical performance claims are not imported as MAX benchmark results.</p>
</details>
"""


WORKED_EXAMPLES = """
<div class="table-scroll"><table id="systems-example-table"><thead><tr><th>Question</th><th>Inputs</th><th>Result to notice</th><th>Open the arithmetic</th></tr></thead><tbody>
<tr><td>How can 1,000 clients coexist with batch 4?</td><td>1,000 connections; 100 already-tokenized contexts</td><td>900 idle; 96 waiting; four GPU members</td><td><button data-card="example:connections">Example 1 ↗</button></td></tr>
<tr><td>How much KV does four-request work own?</td><td>512 prompt + 64 output; 128 tokens/page</td><td>20 reserved pages; 16 prompt pages initially assigned</td><td><button data-card="example:memory">Example 2 ↗</button></td></tr>
<tr><td>Why is batch 4 only batch 3 with a 1 GiB pool?</td><td>16 pool blocks; one null; five-page reservations</td><td>15 usable pages fit three reservations</td><td><button data-card="example:tight">Example 3 ↗</button></td></tr>
<tr><td>What if arrivals exceed service?</td><td>10 requests/s arrive; illustrative capacity 6/s</td><td>Waiting work grows about four requests/s until a cap</td><td><button data-card="example:overload">Example 4 ↗</button></td></tr>
<tr><td>Why can VRAM be reusable while a stream stays open?</td><td>64 generated; eight delivered; reader one token/s</td><td>56 tokens left to send; request KV ownership already zero</td><td><button data-card="example:slow">Example 5 ↗</button></td></tr>
<tr><td>What does adding a replica change?</td><td>Old 96-request backlog on A; add warming B</td><td>B receives future traffic when ready; A's old queue stays</td><td><button data-card="example:fleet">Example 6 ↗</button></td></tr>
</tbody></table></div>
<div class="reading-library example-library">
<details data-reading="example:connections" id="example-connections"><summary>Example 1 · 1,000 connections and four GPU members</summary><div class="reading-body">
<p>Imagine 1,000 clients have a live connection to the frontend. Exactly 100 have submitted a request, and their tokenization/IPC admission has already finished. The other 900 are waiting to send a request.</p>
<pre class="object-values">connections = 1,000
already-tokenized requests = 100
batch ceiling = 4
pending CE cap M = 128
900 idle + 96 waiting + 4 prefill members = 1,000 clients</pre>
<ol><li>The 900 idle connections need socket/protocol state, but they have no prompt context or KV ownership.</li><li>Of the 100 submitted requests, four obtain a GPU batch slot. The remaining 96 stay in the worker's CE set. All 100 fit the example M=128.</li><li>Four prefill members still count in CE until prefill advances them to decode. This is why the label “M” is not just a count of the visible waiting tiles.</li><li>The starting snapshot bypasses earlier stages because those requests already passed them. Subsequent arrivals traverse API encoding and IPC normally.</li></ol>
<p class="result">Connection concurrency = 1,000; unfinished requests = 100; selected GPU sequences = 4. These are three useful counts with different resource costs.</p><p>{{ref:scheduler_pending}} · {{ref:construct_batch}} · {{doc:nginx}}</p>
</div></details>
<details data-reading="example:memory" id="example-memory"><summary>Example 2 · Calculate pool, reservation and assigned pages</summary><div class="reading-body">
<p>Use the saved BF16 Llama 2 7B geometry on one GPU. Each token stores K and V across 32 layers, 32 KV heads and 128 values per head, at two bytes per value.</p>
<pre class="object-values">bytes/token = 2 × 32 × 32 × 128 × 2 = 524,288 = 512 KiB
bytes/page = 128 × 512 KiB = 64 MiB
budget = 24 GiB × 0.9 − 12.551319 GiB = 9.048681 GiB
full-context plan cap = 4 × ceil(4096/128) + 1 null = 129 blocks
allocated pool = 129 × 64 MiB = 8.0625 GiB
usable request pages = 129 − 1 = 128</pre>
<ol><li>The example budget subtracts the saved weight-size estimate from 90% of the available-memory input. It is a planning illustration; a live run also has runtime allocations and reports actual free memory.</li><li>The budget could fund more blocks, but the illustrated full-context batch-4 plan needs at most 129, including the null block. The allocated pool therefore remains 8.0625 GiB.</li><li>The lab conservatively reserves ceil((512+64)/128)=5 pages per admitted GPU member. Four members reserve 20 pages, or 1.25 GiB inside the pool.</li><li>Initially each prompt needs ceil(512/128)=4 assigned pages. Four requests assign 16 pages = 1 GiB; four extra reserved pages remain unassigned headroom for generated tokens.</li><li>Reservation and assignment are counters inside the retained allocation. Completion returns ownership for reuse, while the 8.0625 GiB backing pool stays allocated.</li></ol>
<p class="result">Allocated: 129 blocks. Reserved: 20 request pages. Initially assigned: 16 request pages. Unreserved usable pages: 128−20=108.</p><p>{{ref:geometry}} · {{ref:bytes_per_block}} · {{ref:plan_sizes}} · {{ref:release_request}}</p>
</div></details>
<details data-reading="example:tight" id="example-tight"><summary>Example 3 · A 1 GiB KV pool admits only three five-page reservations</summary><div class="reading-body">
<p>Keep the same 512-token prompt, 64-token output and configured batch ceiling four, but cap the teaching pool at 1 GiB.</p>
<pre class="object-values">pool blocks = floor(1 GiB / 64 MiB) = 16
usable pages = 16 − 1 null block = 15
reservation/request = ceil(576/128) = 5 pages
GPU members = min(4, floor(15/5)) = 3</pre>
<ol><li>One null block is unavailable for ordinary request ownership, leaving 15 usable pages.</li><li>The first three members reserve 3×5=15 pages. A fourth would need 20 total reserved pages, so it waits even though the batch ceiling has one unused slot.</li><li>The three prompts initially assign 3×4=12 pages. Three reserved pages remain growth headroom; they are not free to admit a fourth five-page reservation under this lab policy.</li><li>When a member completes, its five-page reservation is returned. The next waiting request can claim that capacity without reallocating the whole GPU pool.</li></ol>
<p class="result">The lab runs at most three such GPU members. This is a deliberately conservative admission example; MAX's actual allocation/preemption can follow a different policy.</p><p>{{ref:bytes_per_block}} · {{ref:preempt_request}} · {{ref:release_request}}</p>
</div></details>
<details data-reading="example:overload" id="example-overload"><summary>Example 4 · Queue growth and the limits of scale-out</summary><div class="reading-body">
<p>For easy arithmetic, assume an independent example server can sustainably finish six requests/s for this workload while ten requests/s arrive. These are assumed rates, not outputs of the lab's default GPU settings.</p>
<pre class="object-values">arrival rate = 10 requests/s
service capacity = 6 requests/s
net queue growth ≈ 10 − 6 = 4 requests/s
after 10 s without a cap: about 40 extra waiting requests</pre>
<ol><li>A queue temporarily absorbs the four-request/s imbalance. It grows until arrival slows, capacity rises or a finite gate rejects work.</li><li>If pending CE reaches M, the worker stops draining. The IPC transport fills toward N, after which MAX can return 429 for new submissions before SSE begins.</li><li>More queue slots increase tolerated backlog, not the six-request/s sustainable capacity. Waiting time still rises.</li><li>Two identical ready replicas would have a combined illustrative ceiling of 12 requests/s, but only if routing, CPU, memory and network resources support that workload. A warming replica contributes no ready capacity yet.</li><li>Old requests already assigned to the first replica remain there in this lab. Its backlog can persist while the new replica takes future arrivals.</li></ol>
<p class="result">Finite admission controls overload; additional ready service capacity addresses sustained work imbalance.</p><p>{{ref:scheduler_pending}} · {{ref:http_429}} · {{doc:balancing}}</p>
</div></details>
<details data-reading="example:slow" id="example-slow"><summary>Example 5 · Generation is done, delivery needs another 56 seconds</summary><div class="reading-body">
<p>R1 has produced all 64 output tokens. Its slow reader has consumed eight. Use one token/s for the remaining conceptual output buffer.</p>
<pre class="object-values">generated = 64
delivered = 8
buffered = 64 − 8 = 56 tokens
remaining reader time = 56 / 1 = 56 s
request KV assigned/reserved pages = 0 / 0</pre>
<ol><li>The scheduler can finish R1's generation and return its request KV ownership immediately at the safe completion boundary.</li><li>Its final results still occupy API/transport/network state. The request remains sending while 56 tokens drain.</li><li>A new request can reuse the KV pages during those 56 seconds. Allocated KV pool bytes remain unchanged.</li><li>If 100 such slow streams each retain 56 tokens, the lab shows 5,600 buffered tokens. Real byte/RAM cost includes objects, text, framing and network buffers, so multiplying by KV bytes/token would be incorrect.</li><li>If the client disconnects, undelivered tokens are discarded and request cleanup runs. Already delivered tokens are not undone.</li></ol>
<p class="result">GPU/KV pressure and API/egress pressure can move independently.</p><p>{{ref:proxy_submit}} · {{ref:scheduler_execute}} · {{ref:release_request}} · {{doc:uvicorn}}</p>
</div></details>
<details data-reading="example:fleet" id="example-fleet"><summary>Example 6 · Add a replica without moving existing work</summary><div class="reading-body">
<p>Replica A owns 96 waiting requests and four GPU members. At lab time t=0, add B with a compressed two-second initialization period.</p>
<pre class="object-values">t=0 s: B warming; ready replicas = [A]
t=2 s: B ready; ready replicas = [A, B]
old 100 requests: owner remains A
new requests after readiness: policy may choose A or B</pre>
<ol><li>B does not receive requests during warming, because it has not completed the serving readiness lifecycle.</li><li>Once B is ready, the frontend considers it for new traffic. A retains its pending contexts, active KV and ongoing output streams.</li><li>Least-outstanding or estimated-work routing can choose the emptier B for new requests; round robin keeps alternating even if A has a large backlog.</li><li>If A crashes, A's unfinished streams fail. The lab's replacement is a fresh instance; the old streams stay failed.</li><li>Each replica has its own GPU/model/KV state. This example scales independent instances and does not explain tensor-parallel execution of one request.</li></ol>
<p class="result">Scale-out improves future assignment capacity after readiness. It does not automatically migrate queued or partially generated requests.</p><p>{{ref:ready}} · {{ref:connected}} · {{ref:http}} · {{doc:ray}}</p>
</div></details>
</div>
"""
