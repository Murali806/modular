from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('initialization','C1 → M1: weights, host loading and model setup',
 'Follow init_all down to the init invokable, its worker-thread requirement, retained DLPack ownership and the setup wait.',
 'Initialization and runtime',[
 A('C1','C1 · compiled artifact','code + metadata','CompiledCallable','Carries graph names, expected weight placement and compiled resources.'),
 A('W','Weight registry','names + retained pointers','native ModelHandle','DLPack wrappers preserve producer ownership; read-only NumPy borrowing has a caller lifetime requirement.'),
 A('TASK','Setup task','AsyncRT work item','engine context / CPUDevice','Runs setup with the required CPUDevice in TLS.'),
 A('RES','Initialized resources','AsyncValues / handles','ExecutableModel','Init chain, MGP context, weight-registry reference and device/extra context bundles.'),
 A('H1','Host callable symbols','resolved code','CompiledKernelsContext','Loaded from compiled archives during resource setup.'),
 A('M1','M1 · reusable Model','initialized handle','pipeline / compiled callable','Retains compiled models, session, executable state and weight ownership.')],[
 E('Validate the registry against C1','init_all checks required weight names and CPU/non-CPU placement metadata when captured from a Graph, then checks contiguous registry values.',{'C1':B('ready for init','compiled graph names','expected weight devices'),'W':B('registry validated','name → DLPack-compatible value')},[R('engine','def init_all('),R('engine','_raise_if_not_contiguous(weight_value)')],edges=[('C1','W')]),
 E('Build native name/pointer ownership','DLPack-managed tensors are kept in a shared vector. Read-only NumPy arrays can use borrowed pointers whose backing storage must outlive model use.',{'W':B('native retained registry',('w','WeightsRegistry + managed tensor owners'))},[R('session_native','buildWeightsFromDict('),R('session_native','localManagedTensors.emplace_back(')]),
 E('Await compiled models and queue setup','loadAll waits for pending compilation, then loads each model. Engine::Context::initModel schedules setup onto the CPUDevice work queue.',{'TASK':B('queued',('setup','setup(model, weights, devices)'))},[R('session_native','InferenceSession::loadAll('),R('engine_context','Engine::Context::initModel(const M::Model *model')],edges=[('W','TASK'),('C1','TASK')]),
 E('Run Model::setup on a registered worker','Map DeviceRefs to runtime contexts, ensure configured streams and invoke the model init function. TLS is required by runtime buffer borrowing.',{'TASK':B('running',('setup','worker TLS CPUDevice → Model::setup')),'RES':B('initializing','borrowed registry reference','mapped device contexts')},[R('mef','Model::setup(CPUDevice'),R('engine_context','Run on a cpuDevice worker thread')],edges=[('TASK','RES')]),
 E('Resolve resources and host callable code','Init invokes the compiled resource-loading plan. Host symbols become H1; resources remain AsyncValues until their own initialization completes.',{'H1':B('host symbols resolved',('h1','H1 host function pointer')),'RES':B('resource AsyncValues','init chain','MGP context','weights registry / extraArgs')},[R('host_loader','CompiledKernelsContext::load'),R('mef','initFn->invoke(initArgs, cpuDevice)')],edges=[('C1','H1'),('H1','RES')]),
 E('Wait for setup and propagate errors','createExecutableModel waits for chain, context, weights and extraArgs. Native load releases the Python GIL around its await.',{'TASK':B('completed'),'RES':B('setup complete','chain / context / resources available')},[R('mef','ExecutableModel::waitForModelSetup()'),R('engine_context','executableModel->waitForModelSetup()'),R('session_native','Release the GIL during model initialization')]),
 E('Return M1 with lifetime anchors','The ModelHandle retains C1, S1, executable model, registry and shared managed tensors. Input/output device vectors are cached at load time.',{'M1':B('initialized and retained',('m1','M1 ready for repeated execute'))},[R('session_native','ModelHandle handle(compiledModels.copy()'),R('session_native','handle.inputDevicesPy =')],edges=[('W','M1'),('RES','M1'),('C1','M1')])],[
 Q('Three checks before native setup','''
Python validates the weight-name/device contract only when C1 captured `_expected_weights` from a Graph. It checks whether each expected name is present and whether the actual CPU/non-CPU device category matches. It also rejects non-contiguous registered values. These are specific checks; do not relabel them as a universal full checkpoint compatibility proof.

Native `_load_all` accepts either already resolved or pending compiled models and builds the registry. GraphSpec’s merged registry must keep any shared names unambiguous. The resulting dictionary is keyed by actual model symbols; a sole cached/imported model can have a different name from the current spec.
''',[R('engine','compiled._expected_weights is not None'),R('engine','_raise_if_not_contiguous(weight_value)'),R('spec','if len(by_name) == 1:')],open=True),
 Q('DLPack and borrowed arrays: exact lifetime implications','''
```text
ModelHandle M1
 ├─ compiledModels reference → C1 resources
 ├─ InferenceSession reference → S1 / contexts
 ├─ loadedModel → ExecutableModel
 ├─ WeightsRegistry → names / data pointers
 └─ shared managedTensors → DLPack ownership / deleters
```

`buildWeightsFromDict` asks DLPack-capable producers for managed tensors and retains their wrappers so producer storage stays valid. The special read-only NumPy route borrows a pointer; the source explicitly requires backing storage to outlive model execution. Dropping a Python state dictionary is safe only when the underlying required storage remains owned appropriately.

The model constructor can release eligible host weight state after initialization. This is different from destroying the live initialized model’s weight buffers or registry. The model may use transformed/staged representations with their own owners.
''',[R('session_native','buildWeightsFromDict('),R('session_native','Take a shared owning reference'),R('model','def load_graphs(')],open=True),
 Q('Why setup is scheduled on AsyncRT','''
`Engine::Context::initModel` creates an AsyncValue for the future executable and queues a task. The source explains that Model::setup reaches `MGP_RT_BufferBorrowV2`, which needs a current CPUDevice in TLS. An arbitrary Python ThreadPoolExecutor thread does not necessarily have that registration.

The overload starting from pending compiled models uses `andThenAsync` for the same reason. Merely running the continuation immediately on a foreign completion thread would not establish the required worker context. This is an actual mechanism where the synchronous-versus-asynchronous continuation choice affects correctness.
''',[R('engine_context','Run on a cpuDevice worker thread'),R('engine_context','Use andThenAsync to ensure createExecutableModels')]),
 Q('Model::setup and the setup wait','''
Setup checks CPUDevice consistency, wraps a borrowed registry reference, maps graph DeviceRefs to runtime devices, ensures configured streams and invokes the MEF init function. This source’s stream-count default is two (base plus one side), with configuration controlling the actual value. That is not a promise that every operation uses both streams.

The executable retains the init chain, MGP context, weight registry and extra argument bundles. `waitForModelSetup` awaits these values and checks each for errors before the high-level load succeeds. A partially constructed handle is not silently returned as a working M1.

Device image/function loading may still have first-use sites later. “Init returned” establishes the setup contract, not a measured statement that all GPU code pages were eagerly loaded.
''',[R('mef','Model::setup(CPUDevice'),R('mef','configAsInt("stream-count", 2)'),R('mef','ExecutableModel::waitForModelSetup()')]),
 Q('Virtual mode is explicitly compile-only','''
In virtual-device mode, init_all creates CompileOnlyModel stand-ins when graph names are known. They deliberately refuse work requiring a device. Path-compiled artifacts without known graph names are rejected by that branch.

This distinction prevents a cross-compile/documentation environment from presenting successful virtual initialization as GPU execution evidence. None of the current course’s artifact checks invoke init_all or allocate model/device buffers.
''',[R('engine','CompileOnlyModel(name, compiled)'),R('engine','Cannot initialize a path-compiled artifact')])],'''
GraphSpec → session.init_all(C1, weights_registry)
├─ expected names/device categories + contiguity
├─ native loadAll → buildWeightsFromDict → await C1
└─ for each model: load → Engine::Context::initModel
   ├─ queue worker setup task (CPUDevice TLS)
   ├─ Model::setup → map devices → initFn.invoke
   │  └─ resource setup / host code loading → init results
   ├─ waitForModelSetup → errors or initialized executable
   └─ ModelHandle(C1, S1, executable, registry, managed tensors) → M1
''',[R('engine','def init_all('),R('session_native','InferenceSession::loadAll('),R('mef','Model::setup(CPUDevice')])

C('asyncrt_values','AsyncRT I: AsyncValues, continuations and readiness',
 'Make unresolved values, waiter installation, errors, indirect results and retained ownership visible.',
 'Initialization and runtime',[
 A('PROD','Producer','host task / primitive','executor or engine task','Owns a result reference while preparing its value or error.'),
 A('AV','AV1 · AsyncValue','reference-counted value','producer/consumers','Stores terminal payload/error state and continuation bookkeeping.'),
 A('WAIT','Continuation list','waiter storage','AsyncValue','Holds dependent actions while a value is unresolved.'),
 A('QUEUE','AsyncRT work queue','host scheduler','CPUDevice','Receives asynchronous continuations; synchronous ready continuations can run immediately.'),
 A('CONS','Consumer','dependent task','MEF executor / API','Can inspect the payload only after successful readiness; errors are also ready terminal results.'),
 A('IND','Indirect result slot','forwarding AsyncValue','out-of-line primitive','Lets the caller hold a stable result reference before the producing task finishes.')],[
 E('Allocate an unresolved value','AV1 initially has no constructed successful payload. Its state includes continuation-management information, not just a Boolean ready flag.',{'PROD':B('owns pending result',('owner','producer reference')),'AV':B('kUnconstructed',('av','AV1: payload unavailable'))},[R('async','kUnconstructed = 0'),R('async','AsyncValue::allocate')]),
 E('Attach a dependent continuation','andThenSync on an unresolved value registers a waiter. Internal installation and out-of-line slot states coordinate concurrent registration/publication.',{'WAIT':B('waiting for AV1',('cont','consumer continuation')),'AV':B('unconstructed + waiter bookkeeping',('av','AV1: pending'))},[R('async','kUnconstructedInitializingInlineWaiter'),R('async','kUnconstructed1ValidOOLWaiterSlots')],edges=[('AV','WAIT')]),
 E('Use an indirect result for out-of-line work','mgp_generic_execute allocates indirect result slots before enqueueing work so downstream code has references even while local results do not yet exist.',{'IND':B('unresolved forwarding slot',('ind','result ref → no target yet'))},[R('primitive','outResults.allocateIndirectResults()')]),
 E('Construct payload, then publish terminal readiness','The producer emplaces the concrete payload and transitions to available, releasing waiter actions through the runtime’s publication rules.',{'AV':B('kAvailable',('av','AV1: successful payload')),'PROD':B('producer completed')},[R('async','AsyncValue::emplaceAndDecRef'),R('async','notifyReadyAndDecRef(State::kAvailable')],edges=[('PROD','AV')]),
 E('Choose continuation execution semantics','For a ready value, andThenSync can execute immediately. andThenAsync routes through its out-of-line path, which is required by model setup’s worker/TLS contract.',{'WAIT':B('detached after publication'),'QUEUE':B('asynchronous continuation queued',('cont','consumer continuation'))},[R('async','andThenAsyncOutOfLine'),R('async','runWaiterNow(')],transfer=['WAIT','andThenAsync scheduling','QUEUE']),
 E('Resolve the public result slot','The primitive publishes local results to indirect outputs only after the host kernel call and its required context-reachable cleanup have finished.',{'IND':B('resolved forwarding slot',('ind','result ref → AV1')),'CONS':B('eligible to continue','read success payload / propagate error')},[R('primitive','std::move(output).resolveIndirect(std::move(local))')],edges=[('AV','IND'),('IND','CONS')]),
 E('Drop references only after ownership ends','AsyncValue reference counting protects the payload and waiter/resource lifetimes. Readiness does not mean no owner remains; dropping one reference may not destroy anything.',{'AV':B('ready, possibly still retained',('av','AV1: consumers retain payload')),'QUEUE':B('work consumed')},[R('async','AsyncValue::dropRef('),R('async','destroyWithRefCountZero()')])],[
 Q('The real state machine has waiter substates','''
```text
unconstructed
  ├─ transient inline-waiter installation
  ├─ 1 / 2 / 3 / 4 valid out-of-line waiter slots
  └─ publish ──> available(payload)  [terminal, ready]
             └─ error(diagnostic)   [terminal, ready]
```

The native enum includes `kUnconstructed`, `kUnconstructedInitializingInlineWaiter`, four valid-OOL-slot states, `kAvailable` and `kError`. Pending states encode synchronization/storage details for waiters. An error is ready because waiters must wake and propagate it; it is not a successful tensor payload.

Concrete AsyncValues hold typed payload storage. Indirect AsyncValues forward a stable reference to another result. The latter is useful when an asynchronous primitive must return result references before its queued work creates the local values.
''',[R('async','enum class State'),R('async','class IndirectAsyncValue'),R('async','return state == State::kAvailable || state == State::kError')],open=True),
 Q('andThenSync and andThenAsync are different contracts','''
| Call / condition | Consequence |
|---|---|
| andThenSync, already ready | run waiter immediately through the synchronous path |
| andThenSync, pending | register continuation; publication later makes it runnable |
| andThenAsync | use the out-of-line continuation mechanism |
| engine init continuation | uses asynchronous scheduling to establish registered CPUDevice worker context |

“Sync” does not mean “block until ready” here. It describes continuation execution policy. Blocking/await behavior is a different API discussed in the worker lesson. Immediate ready callbacks are why consumers such as MEFExecutor need construction guards before registering dependencies.
''',[R('async','void AsyncValue::andThen(Waiter &&waiter)'),R('engine_context','Use andThenAsync to ensure createExecutableModels')],open=True),
 Q('Publication and reference ownership','''
The producer must finish payload construction before successful readiness is published. Waiter installation and notification coordinate with atomics; a consumer must not read an unconstructed payload merely because it owns an AsyncValue pointer.

Reference increments can use relaxed ordering because obtaining a new reference from an existing one assumes the transfer already supplies necessary synchronization. Reference decrements handle last-owner destruction with the documented acquire/release requirements. This is lifetime management, separate from the semantic dependency graph.

The out-of-line MGP primitive captures operands/results/resources and a device context. It calls H1, clears captured operands and the pinned device context, then resolves output slots last. Publishing too early could expose downstream completion while task cleanup still touches context-reachable objects.
''',[R('async','AsyncValue::addRef(uint32_t count)'),R('async','AsyncValue::dropRef(uint32_t count)'),R('primitive','Results are published into `outResults`')]),
 Q('Three readiness questions for Llama','''
| Question | Mechanism |
|---|---|
| Is the symbolic cache read ordered after its write? | G1 chain/effect relation, lowered into the runtime plan |
| Does the host runtime have this value/resource result? | AsyncValue availability/error and MEF dependencies |
| Has the GPU finished the memory operation needed by the next access? | driver stream/event tracking and the access-specific wait |

A Buffer descriptor can become available while its device work is tracked asynchronously. A runtime Chain carries an ordering contract; a CUDA event reports device progress. The pipeline’s tracked token-copy `to_numpy()` supplies the final host-readability gate needed for sampled IDs.
''',[R('graph','self._always_ready_chain ='),R('async','bool isValueAvailable()'),R('pipeline','generated_tokens_host.to_numpy()')]),
 Q('Error and cancellation boundaries','''
An AsyncValue error propagates diagnostics to waiters/API boundaries. It must not be interpreted as valid logits or a token. A cancelled serving request also needs cache/response lifecycle handling; AsyncValue terminal error alone is not a universal GPU cancellation command.

Already submitted GPU work and retained input buffers must obey their completion/lifetime rules even if the client no longer wants the result. This is why request state, runtime value state and device event state are taught separately.
''',[R('async','kError = 7'),R('pipeline','Primary and extra KV cache lifecycle')])],'''
producer → allocate AV1 (pending)
consumer → andThenSync / andThenAsync → waiter bookkeeping
producer → construct payload OR error → publish readiness
 ├─ synchronous waiter may run immediately when already ready
 └─ asynchronous waiter → CPUDevice work queue

out-of-line primitive:
allocate indirect outputs → retain captures → invoke H1 → cleanup
                         → resolve indirect outputs last → dependents
''',[R('async','enum class State'),R('async','void AsyncValue::andThen(Waiter &&waiter)'),R('primitive','outResults.allocateIndirectResults()')],scenarios=[dict(id='error',title='Producer publishes an error',steps=[
 E('Start with a pending value and waiter','The consumer owns a reference but has no successful payload to read.',{'AV':B('unconstructed',('av','pending')),'WAIT':B('registered',('cont','dependent continuation'))},[R('async','kUnconstructed = 0')]),
 E('Publish a terminal diagnostic','kError counts as ready and wakes continuations. Consumers must take the diagnostic path instead of treating this as logits.',{'AV':B('kError',('av','error diagnostic')),'CONS':B('error continuation','propagate failure')},[R('async','kError = 7'),R('async','return state == State::kAvailable || state == State::kError')],edges=[('AV','CONS')])])])

C('asyncrt_workers','AsyncRT II: queues, await donation and the wakeup race',
 'Watch a work item move between queue and worker, then compare registered-worker waits with foreign Python-thread waits.',
 'Initialization and runtime',[
 A('CALLER','Calling thread','OS thread','Python/native caller','May be foreign to AsyncRT; the Python CPU runtime uses a main non-donation policy.'),
 A('QUEUE','General / affinity queues','task storage','ThreadPoolWorkQueue','General lock-free queue, per-worker affinity queues and overflow/spill paths.'),
 A('WORKER','Registered worker','OS thread + TLS','CPUDevice work queue','Runs host tasks with runtime TLS; can donate work while awaiting.'),
 A('WAIT','Await completion counter','atomic + semaphore','await call scope','Counts unresolved values and signals when the last becomes ready.'),
 A('VALUE','Awaited AsyncValues','dependency results','producer / consumers','Already-ready values take the fast path; errors are terminal-ready too.'),
 A('SLEEP','Suspension handshake','atomic worker state','work queue','Coordinates queue publication, suspended bits and wakeups without a lost notification.')],[
 E('Submit a work item from the caller','Engine tasks or continuations call addTask. An ordinary general task is enqueued and a suspended worker is woken.',{'CALLER':B('submitting',('task','setup / compiled-call work item')),'QUEUE':B('available task queue')},[R('queue','void ThreadPoolWorkQueue::addTask(')]),
 E('Transfer ownership into the queue','A non-affinity task enters the global queue if possible; overflow uses a mutex-protected list. addTask does not execute immediately on the caller’s stack.',{'CALLER':B('task handed off'),'QUEUE':B('queued',('task','work item'))},[R('queue','if (taskList.enqueue('),R('queue','overflowTaskList.emplace_back')],transfer=['CALLER','enqueue','QUEUE']),
 E('A foreign caller awaits results','The ready fast path checks all values. For unresolved values and no registered owning worker, await uses its semaphore path; this caller does not donate a worker loop.',{'WAIT':B('foreign-thread wait','numRemaining = 1','local semaphore'),'VALUE':B('pending','result AV')},[R('queue','void ThreadPoolWorkQueue::await('),R('queue','The caller is a \'foreign\' thread')],edges=[('CALLER','WAIT')]),
 E('A worker dequeues and runs the task','The registered worker supplies runtime TLS. It may execute setup or a generated host call that submits GPU commands; GPU SM threads are separate.',{'QUEUE':B('dequeued'),'WORKER':B('running',('task','work item'))},[R('queue','doWork<'),R('engine_context','Run on a cpuDevice worker thread')],transfer=['QUEUE','dequeue','WORKER']),
 E('Publish the result and signal the waiter','The last continuation decrements numRemaining to zero and posts the semaphore, allowing the foreign caller to resume.',{'VALUE':B('ready','result AV available/error'),'WAIT':B('signaled','numRemaining = 0'),'WORKER':B('task completed')},[R('queue','numRemaining.fetch_sub('),R('queue','Semaphore sema;')],edges=[('WORKER','VALUE'),('VALUE','WAIT')]),
 E('Before sleeping, mark and recheck','A worker marks itself suspended, then rechecks affinity/global/spill/overflow work. The enqueuer publishes work then checks suspended workers with matching fences.',{'SLEEP':B('safe sleep protocol','mark suspended → fence → queue recheck','enqueue → fence → wake suspended worker')},[R('queue','Dekker handshake'),R('queue','Dropping either one is the GEX-4343 hang')]),
 E('Return to the caller only after its values are ready','The API resumes; a host buffer still requires the appropriate device-tracking wait before data access. Await readiness alone does not globally synchronize all GPU streams.',{'CALLER':B('resumed','await contract fulfilled'),'WAIT':B('scope can finish'),'SLEEP':B('workers remain available','idle or next queued work')},[R('queue','void ThreadPoolWorkQueue::await('),R('pipeline','generated_tokens_host.to_numpy()')])],[
 Q('Queue routing, including full queues','''
| Submission | Destination / fallback | Wake behavior |
|---|---|---|
| `taskId ≥ 0` | target worker affinity queue or local spill queue | wake that worker / semaphore route |
| ordinary `addTask` | global lock-free queue | wake a suspended worker |
| global queue full | mutex-protected overflow list | wake a worker; overflow pump participates in pre-sleep checks |
| `addLocalTask`, registered worker/main | local work list | handled by owning worker loop |
| `addLocalTask`, foreign caller | falls back to `addTask` | normal enqueue/wakeup |

The source explicitly rejects running a newly submitted general task immediately on the caller’s stack as an overflow fallback. That avoids recursive stack growth and preserves addTask’s contract. The queue does not promise strict FIFO fairness for every overflow/spill interaction.
''',[R('queue','void ThreadPoolWorkQueue::addTask('),R('queue','void ThreadPoolWorkQueue::addLocalTask(')],open=True),
 Q('Await has three paths; donation is conditional','''
```text
await(values)
 ├─ all ready? → return
 └─ unresolved
    ├─ getOwningWorkQueueThread() succeeds
    │  └─ register countdown waiters → runItemsOnOwningThread until zero
    └─ foreign/unregistered thread
       └─ register countdown waiters → semaphore wait → resume
```

The registered worker/main branch donates its thread to execute work while awaiting. The foreign branch sleeps while other workers make progress. The Python CPU driver route uses `withMainWillNotDonate`; therefore “Python await always steals/runs tasks” is wrong. The source’s actual registration state decides the branch.

The countdown is initialized before continuations are attached because already-ready values can invoke synchronous waiters immediately. The final decrement wakes the awaiting worker or foreign semaphore so completion is not missed while the waiter is asleep.
''',[R('queue','void ThreadPoolWorkQueue::await('),R('device_native','withMainWillNotDonate')],open=True),
 Q('The lost-wakeup race, step by step','''
| Worker | Enqueuer | Required outcome |
|---|---|---|
| observes no work | preparing a task | neither observation alone proves idle |
| sets suspended bit | publishes task to queue | operations must become visible in a coordinated order |
| sequentially consistent fence; rechecks queues | matching fence; reads suspended bits | at least one side observes the other |
| sees task and continues, or sleeps | wakes a marked suspended worker | task cannot remain queued with every worker asleep |

The source calls this a Dekker handshake and identifies removing either fence as a hang. Release enqueue plus relaxed/acquire queue checks alone do not establish the required either-side observation.

A global dequeue attempt can also fail while work may remain due to optimistic concurrent claims. The pre-suspend path requires agreement between a failed dequeue and zero approximate queue size; it retries otherwise. `size_approx()` is not a general exact queue length or a guarantee that the next dequeue succeeds.
''',[R('queue','Dekker handshake'),R('queue','try_dequeue can fail while the queue is not empty')]),
 Q('Task affinity is not GPU thread placement','''
The runtime primitive derives a device hint and maps it to a task ID using the work queue’s parallelism. An asynchronous kernel or wrong-worker affinity schedules out of line; an eligible synchronous call can run inline. This is CPU host-work routing.

GPU grid/block configuration, SM scheduling and stream ordering belong to later host/driver/device stages. Assigning a task to worker 2 does not imply GPU 2, CUDA stream 2 or warp 2. The worker binds the required device context for the host call and unbinds afterward.
''',[R('primitive','MLRT::taskIdForDevice('),R('primitive','shouldRunInlineForTask(taskId)'),R('primitive','bindToCurrentThread()')]),
 Q('Shutdown and no hidden measurement claims','''
CPUDevice teardown shuts down/drains work before invalidating runtime resources. Captured input/result/context references must remain valid during that work. Destroying the visible Python variable does not authorize premature teardown while native owners still exist.

The scene is a deterministic concurrency explanation. It does not measure thread counts, queue fairness, CPU occupancy or wakeup latency. Real scheduling interleavings require instrumentation; the source establishes the invariant that the explanation models.
''',[R('cpu','CPUDevice::~CPUDevice'),R('primitive','Captures keep the inputs')])],'''
addTask(work, taskId)
├─ affinity: worker queue → spill if needed → wake target
└─ general: lock-free queue → overflow if needed → wake worker

await(values)
├─ all ready → return
├─ registered worker/main → countdown + donate runItems loop
└─ foreign thread → countdown + semaphore wait

worker idle → mark suspended + fence → recheck queues → sleep if safe
enqueuer    → publish work + fence → inspect suspended bits → wake
''',[R('queue','void ThreadPoolWorkQueue::addTask('),R('queue','void ThreadPoolWorkQueue::await('),R('queue','Dekker handshake')],scenarios=[dict(id='worker',title='Registered worker donates while awaiting',steps=[
 E('Worker awaits an unresolved value','This scenario selects a registered worker, unlike the primary foreign Python-thread path.',{'WORKER':B('awaiting result','owning worker recognized'),'VALUE':B('pending','wanted AV'),'QUEUE':B('ready work',('other','independent work item')),'WAIT':B('countdown','numRemaining=1')},[R('queue','if (awaitingWorker) {')]),
 E('Donate the thread to useful work','runItemsOnOwningThread can execute available work while checking the early-stop predicate.',{'QUEUE':B('dequeued'),'WORKER':B('running while awaiting',('other','independent work item'))},[R('queue','awaitingWorker->runItemsOnOwningThread(')],transfer=['QUEUE','donating worker runs item','WORKER']),
 E('Last awaited value becomes ready','The countdown reaches zero and the semaphore is posted in case the worker slept; the early-stop predicate ends the await loop.',{'VALUE':B('ready','wanted AV available'),'WAIT':B('complete','numRemaining=0'),'WORKER':B('await returns')},[R('queue','numRemaining.load(std::memory_order_seq_cst) == 0')])])])
