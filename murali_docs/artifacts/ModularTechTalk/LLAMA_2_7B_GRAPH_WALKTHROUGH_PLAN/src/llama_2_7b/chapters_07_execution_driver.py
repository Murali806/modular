from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('asyncrt_execution','AsyncRT III: one Model.execute through MEF readiness',
 'Follow argument validation, per-primitive dependency counters, the async host ABI, result publication and Python Buffer reconstruction.',
 'Initialization and runtime',[
 A('INPUT','Prepared input buffers','borrowed values','batch processor / native binding','Must satisfy model positional contracts; wrapped into runtime values.'),
 A('EXEC','E1 · MEF executor','per-invocation state','MEFInvokable','Retains the MEF file, register table, primitive table and pending dependencies.'),
 A('COUNT','Primitive readiness','atomic counter','executor primitive record','Counts required operand readiness plus a constructor guard.'),
 A('TASK','Compiled host work','work item / inline call','mgp_generic_execute','Captures operands, output slots, constant resources and device context as required.'),
 A('H1','H1 · generated host callable','native code','CompiledKernelsContext','Receives pointers to AsyncValueRef slots and resource entries via the selected async ABI.'),
 A('RESULT','Result Buffer wrappers','runtime/API outputs','native driver extraction','Waits runtime results, drops the leading chain and preserves aliased tensor ownership.')],[
 E('Validate and prepare input storage','The binding checks count/type/static shape and device compatibility, then prepares each input storage as a runtime AsyncValue.',{'INPUT':B('validated model inputs','tokens / offsets / host controls / KV leaves')},[R('session_native','validateInputSpecList('),R('session_native','Driver::prepareInput(')]),
 E('Invoke with retained model resources','ExecutableModel prepends its init chain, MGP context, weight registry and extra resources to the new inputs, establishes CPUDevice TLS and invokes executeFn.',{'EXEC':B('invocation E1','retained model resources + new inputs')},[R('mef','ExecutableModel::execute('),R('mef','passedArgs.emplace_back(weightsRegistry.copy())')],edges=[('INPUT','EXEC')]),
 E('Initialize readiness before callbacks can fire','For a teaching primitive with two readiness operands, the counter starts at 2+1=3. The extra constructor count prevents early execution while tables are still being built.',{'COUNT':B('illustrative two-operand primitive','count=3: Q + chain + constructor guard')},[R('invokable','numOperandsNotReady(numOperandsBeforeExecuting + 1)')]),
 E('Observe one ready operand and finish construction','Q is already ready: 3→2. Dropping the constructor guard gives 2→1. The remaining cache-write dependency still prevents this primitive from running.',{'COUNT':B('one dependency still pending','count=1: cache-write chain'),'EXEC':B('registers and primitive records complete')},[R('invokable','We initialized each PrimitiveInfo::numOperandsNotReady'),R('invokable','registerVal->andThenSync(')]),
 E('Last dependency becomes ready','The final decrement 1→0 builds a runnable primitive frame. Error operands are propagated automatically instead of invoking normal compute.',{'COUNT':B('eligible','count=0'),'TASK':B('primitive dispatched',('work','mgp_generic_execute frame'))},[R('invokable','numOperandsNotReady.fetch_sub(1)'),R('invokable','If any operand is an error')],edges=[('COUNT','TASK')]),
 E('Select async ABI and schedule appropriately','mgp_generic_execute selects CompiledKernelAsyncABIFn in both scheduling branches. Async or wrong-affinity work is queued with indirect output slots and retained captures.',{'TASK':B('host task retained',('work','operands + results + resources + DeviceContext')),'H1':B('selected ABI','void** → AsyncValueRef slots / resources')},[R('primitive','CompiledKernelAsyncABIFn *kernelPtr'),R('primitive','shouldRunInlineForTask(taskId)')],edges=[('TASK','H1')]),
 E('Bind context, invoke H1, publish outputs last','The task packs the ABI, binds the device context, calls H1, unbinds, clears captured context-reachable references, then resolves the indirect outputs.',{'TASK':B('task complete'),'RESULT':B('runtime results published','output AsyncValues available / error')},[R('primitive','kernelPtr(reinterpret_cast<void **>(kernelInputs.data()))'),R('primitive','std::move(output).resolveIndirect(std::move(local))')],transfer=['H1','result slots after cleanup','RESULT']),
 E('Reconstruct API output Buffers','extractDriverResults awaits runtime values, drops the first chain result, preserves aliasing via a buffer cache and wraps tensor storage. Host-readability still follows buffer/device tracking.',{'RESULT':B('Python Buffers returned','logits Buffer + output metadata','device completion/readback is a separate gate')},[R('execution','extractDriverResults('),R('execution','results.drop_front()')],edges=[('RESULT','INPUT')])],[
 Q('The full source order for one decoder invocation','''
```text
Llama3Model.execute(*model_inputs.buffers)
 → ModelHandle.executeDeviceTensors
   → validateInputSpecList / verifyInputDevice / getInputStorage
   → prepareInput → AsyncValues
 → executePreparedDeviceTensors [GIL released]
   → executeFromDriver → ExecutableModel.execute
     → fixed init resources + extraArgs + current model inputs
     → MEFInvokable.invoke → MEFExecutor
       → operand readiness → primitive frame
       → mgp_generic_execute → host H1 → generated driver work
   → extractDriverResults → drop chain / preserve aliases
 → extractPyBufferResults → returned Buffers
```

This is a local native execution route within the model worker. It does not send an RPC to the GPU. MAX Serve’s request IPC boundary is earlier in the serving process structure.
''',[R('model','self.model.execute('),R('session_native','ModelHandle::executeDeviceTensors('),R('execution','executeFromDriver('),R('mef','ExecutableModel::execute(')],open=True),
 Q('Readiness tables and the constructor guard','''
Each executor owns register and primitive records. PrimitiveInfo contains an atomic remaining-operand count and a primitive offset. Counts start at `numOperandsBeforeExecuting + 1`, and constructor completion drops the extra guard.

Ready inputs can invoke callbacks immediately. Without a guard, a primitive could execute before the executor finished installing its register/use tables. Later decrements dispatch only when the previous count was one. The two-operand animation is illustrative; actual paged-attention implementation inputs are more numerous.

The executor is reference-counted and retains its MEF file during asynchronous work. Multiple executors can be active for the same invokable, so per-execution state is distinct from the reusable model artifact.
''',[R('invokable','struct PrimitiveInfo'),R('invokable','class MEFExecutor final'),R('invokable','numOperandsNotReady.fetch_sub(1)')],open=True),
 Q('Strict dependencies, non-strict operands and errors','''
The count represents operands required before the primitive may start. Source code also supports non-strict primitives whose frame can contain unresolved indirect operands. Therefore “all operands of every primitive must be concrete first” is too broad.

When constructing a frame, the executor processes operand registers and use counts. If any resolved operand is an error, it propagates that error to results and short-circuits normal primitive execution, while still processing operands needed for correct reference accounting. This avoids leaking other inputs merely because one input failed.
''',[R('invokable','The operand register may not be available for non-strict primitives'),R('invokable','If any operand is an error')]),
 Q('The actual host ABI packet','''
`getKernelInputs` packs, in order:

1. addresses of input `AnyAsyncValueRef` slots;
2. addresses of output `AnyAsyncValueRef` slots;
3. each constant resource’s address and encoded byte-count entry.

The resulting `void**` is passed to the selected `CompiledKernelAsyncABIFn`. These pointers address **host runtime reference slots**, not directly one contiguous array of CUDA device tensor pointers. Generated host code unwraps descriptors/resources and later constructs device launch arguments.

Even the inline scheduling branch uses this selected async ABI. “Synchronous kernel” in the branch label describes scheduling choice and does not select the unrelated two-argument ABI declaration.
''',[R('primitive','getKernelInputs(OperandTy &operands'),R('primitive','kernelInputs.push_back(reinterpret_cast<void *>(byteCount))'),R('primitive','CompiledKernelAsyncABIFn *kernelPtr')]),
 Q('Out-of-line capture, context binding and last publication','''
Queued execution allocates indirect output results because the task outlives the primitive call. Captured operands, output references, constant resources and DeviceContext keep native state alive. Before H1 the task binds its device context; afterward it unbinds and verifies consumed operand slots are reset.

The task clears operand/context references before resolving public indirect outputs. The code comment calls out the ordering: no downstream work should see completion before the host kernel call and context-reachable cleanup finish. Device-side operations still obey their own stream/event tracking contract; publishing a runtime buffer does not authorize an unsynchronized CPU read.
''',[R('primitive','Captures keep the inputs'),R('primitive','Results are published into `outResults`'),R('primitive','Release everything context-reachable BEFORE')]),
 Q('GIL release, output aliases and failure reporting','''
The native binding releases the GIL while executing so another thread can perform needed Python/DLPack decrefs; holding it across native waits could deadlock those ownership paths. This does not remove input validation or exception propagation.

Result extraction awaits runtime errors, maps output DeviceRefs to configured contexts and drops the leading chain. A buffer cache preserves aliased tensors using a shallow borrowed tensor representation, avoiding double moves. The wrapper then returns driver Buffers with tensor specifications. A returned GPU logits Buffer remains device data until a suitable consumer or host-copy operation handles it.
''',[R('session_native','This situation can cause a deadlock.'),R('execution','Tensors may be aliasing'),R('execution','tensor.borrow()')])],'''
M1.execute(inputs)
├─ native validation / prepareInput
├─ fixed model resources + execution inputs → MEFInvokable
│  └─ per-execution tables + dependency countdown
│     └─ ready primitive (or error propagation)
│        └─ mgp_generic_execute
│           ├─ inline if eligible, otherwise queue with retained captures
│           └─ CompiledKernelAsyncABIFn → H1
└─ await runtime results → drop chain → preserve buffer aliases → Python Buffers
''',[R('invokable','numOperandsNotReady(numOperandsBeforeExecuting + 1)'),R('primitive','mgp_generic_execute('),R('execution','extractDriverResults(')])

C('driver','Driver: launch ABI, streams, events and host visibility',
 'Follow host code into CUDA, then distinguish submission, device completion, AsyncRT continuation and safe token readback.',
 'Initialization and runtime',[
 A('H1','Generated host callable','CPU code','runtime work item','Reads runtime descriptors and creates the device argument storage.'),
 A('ABI','Device argument packet','host bytes','launch caller','Contains device-layout scalar/pointer values referenced by driver argument pointers.'),
 A('STREAM','CUDA stream','ordered device queue','Driver DeviceContext','Orders submitted copies, kernel launches and event records within its contract.'),
 A('GPU','KD1 + device buffers','device execution','CUDA module / buffer owners','Uses retained code plus weight, KV and activation addresses.'),
 A('EVENT','Driver event / handler','completion state','DeviceContext','Reports completion and releases deferred host/runtime actions.'),
 A('HOST','Tracked host staging','host-visible storage','pipeline / Buffer tracking','Host reads wait for the relevant copy/usage completion.')],[
 E('Unwrap runtime values in H1','The first ABI delivered AsyncValueRef/resource slots. H1 obtains buffer descriptors and the device context needed for this kernel call.',{'H1':B('host work executing','AsyncValue/resource slots → descriptors')},[R('primitive','getKernelInputs(OperandTy &operands')]),
 E('Pack device-layout arguments','The second ABI carries scalar values and device addresses. Host argument storage must remain valid for the launch API’s consumption, and device buffers must outlive their queued use.',{'ABI':B('device launch packet',('args','Q/KV/output device pointers + controls'))},[R('cuda','void **args,'),R('cuda','cuFuncGetParamInfo')],edges=[('H1','ABI')]),
 E('Submit cuLaunchKernelEx','Fill grid/block dimensions, dynamic shared bytes, stream and attributes in CUlaunchConfig; call the loaded KD1 handle with argument pointers.',{'STREAM':B('kernel queued',('launch','launch KD1(args)')),'GPU':B('code / data retained','KD1 function handle','weights + KV + activation buffers')},[R('cuda','CUlaunchConfig config;'),R('cuda','cuLaunchKernelEx(')],transfer=['ABI','driver launch submission','STREAM']),
 E('Device work progresses in its stream','Submission success does not mean the kernel has completed. Dependencies must order consumers across streams explicitly; a stream-wait event is distinct from blocking the host.',{'STREAM':B('commands in flight',('launch','queued/running KD1')),'EVENT':B('recorded after work','event pending')},[R('cuda','cuEventRecord(event, stream_)'),R('cuda','cuStreamWaitEvent(')],edges=[('STREAM','GPU'),('STREAM','EVENT')]),
 E('Completion is observed through events/tracking','Event query reports pending or complete; synchronization waits. DeviceContext pending handlers can bridge completed device work to host/runtime actions.',{'EVENT':B('completion observed','event ready → registered handler'),'STREAM':B('relevant work complete')},[R('cuda','cuEventQuery(event_)'),R('device_context','enqueueEventHandler(')]),
 E('Queue sampled token copy into staging memory','The pipeline copies generated IDs into a tracked staging Buffer. The accelerator association chooses the correct device/stream even though staging storage is host-accessible.',{'HOST':B('copy pending',('tokens','host staging: bytes not yet safe to read'))},[R('pipeline','generated_tokens_host.inplace_copy_from('),R('buffer_native','bool Python::Buffer::isHostHazardTracked()')],edges=[('GPU','HOST')]),
 E('Read only after the tracked completion gate','generated_tokens_host.to_numpy() waits for the copy. Request contexts can now read token IDs and construct responses.',{'HOST':B('safe host read',('tokens','host-visible generated token IDs'))},[R('pipeline','generated_tokens_host.to_numpy()')],transfer=['EVENT','tracked copy completion','HOST'])],[
 Q('Two ABIs, side by side','''
| Boundary | Caller → callee | Packet contents |
|---|---|---|
| runtime / host callable | mgp_generic_execute → H1 | addresses of AsyncValueRef input/output slots; resource addresses and sizes |
| host / GPU driver | generated device launch → CUDA KD1 | device-layout argument values in host storage, referenced by `void** args` |

A device pointer value is stored in a host argument slot; the driver reads that slot to pass the address to the GPU function. The slot address and the device address it contains are different addresses in different roles. Dtype/rank/layout wrappers can expand into implementation-specific fields; the source signature and generated lowering determine the exact layout.

`cuFuncGetParamInfo` is used to query function parameter sizes for launch tracing. That metadata is not a direct description of the Python Graph argument list: generated host/device wrappers may have transformed it.
''',[R('primitive','getKernelInputs(OperandTy &operands'),R('cuda','cuFuncGetParamInfo'),R('cuda','CUDADeviceFunction::cudaCallKernelDirect(')],open=True),
 Q('The launch configuration is concrete, but its values are target-selected','''
The CUDA call fills `gridDimX/Y/Z`, `blockDimX/Y/Z`, `sharedMemBytes`, `hStream`, `attrs` and `numAttrs`. It then invokes `cuLaunchKernelEx` with the loaded function, argument pointers and a null extra packet.

This source proves the launch mechanism, not the actual grid/block values for a Llama layer. Those depend on the selected production kernel, dimensions and target. Example 04’s tiny demo launch geometry cannot be copied here. Optional tracing records launch metadata; optional sync/debug modes can alter waiting behavior and should not be presented as the default fast path.
''',[R('cuda','CUlaunchConfig config;'),R('cuda','cuLaunchKernelEx('),R('cuda','MAYBE_SYNC(CUDADeviceContext::cast(context()))')],open=True),
 Q('Streams, event record/query/wait and AsyncRT','''
```text
AsyncRT worker: H1 → enqueue copy/kernel → return according to host contract
CUDA stream:   earlier work → kernel → record event
Device:        commands progress → event reaches completion
Driver:        query/sync/pending handlers → release deferred actions
Host consumer: tracked read waits if needed → accesses completed data
```

Recording an event inserts an ordered milestone; querying checks it without pretending every query succeeds immediately. Event synchronization waits for that milestone. `cuStreamWaitEvent` orders work on another stream without being the same operation as a CPU semaphore wait.

AsyncRT can schedule the host work that submits or responds to device operations. It does not schedule each CUDA thread instruction. A runtime AsyncValue and a CUDA event can participate in one dependency chain while remaining separate objects.
''',[R('cuda','CUDADeviceEvent::synchronize()'),R('cuda','cuEventQuery(event_)'),R('cuda','cuStreamWaitEvent('),R('device_context','enqueueEventHandler(')]),
 Q('Transfers and staging ownership','''
H2D copies stage input/control data to the model device where required. D2H copies make selected results available to the host. The NVIDIA implementation uses asynchronous copy calls on streams for these transfer paths. Storage owners and tracking must outlive pending copies.

`Usage.STAGING` on an accelerator-associated Buffer chooses pinned/host staging behavior. Adding `UNTRACKED` changes hazard tracking; callers then carry explicit synchronization responsibilities. The final generated-token copy in TextGenerationPipeline deliberately uses tracked staging, so `to_numpy()` waits before request contexts inspect bytes.

Do not assume all 32000 logits are copied to the host for default device sampling. The pipeline can run the sampler on the model device and transfer generated token IDs; `sample_on_host` is a separate configuration branch.
''',[R('cuda','cuMemcpyHtoDAsync_v2(dst, src'),R('cuda','cuMemcpyDtoHAsync_v2(dst'),R('buffer_native','bool Python::Buffer::isHostHazardTracked()'),R('pipeline','d2h_generated_tokens')]),
 Q('Driver resources and external boundaries','''
Loaded CUDA module/function resources are retained and eventually unloaded by their owner. Weight, KV, scratch and staging allocations have separate ownership paths. DeviceContext has pending-event handling and deferred resource cleanup; freeing memory at the wrong host milestone can race its last device use.

**BOUNDARY:** vendor driver internals, OS command transport, physical GPU scheduling, paging and memory placement are not fully exposed by these MAX call sites. This reader does not infer a particular syscall or exact hardware queue packet from `cuLaunchKernelEx`.

Discovery through NVML, CUDA context creation, MIG/visible ordinals and process startup are upstream concerns. The serving worker must use valid process-local contexts/handles; no context-sharing-by-fork shortcut is assumed. The parent startup reader supplies the actual worker creation route.
''',[R('cuda','CUDAModuleFunction::~CUDAModuleFunction()'),R('device_context','enqueueEventHandler('),R('device_native','DeviceContextPair')])],'''
mgp_generic_execute → H1 (host AsyncValue/resource ABI)
└─ descriptors + device context → GPU argument packet
   └─ CUDADeviceFunction.cudaCallKernelDirect
      ├─ CUlaunchConfig(grid,block,shared memory,stream,attributes)
      └─ cuLaunchKernelEx(KD1,args)
         └─ stream work → record/query/wait event → pending handlers
            └─ tracked staging copy → to_numpy wait → safe host token read
''',[R('primitive','getKernelInputs(OperandTy &operands'),R('cuda','CUDADeviceFunction::cudaCallKernelDirect('),R('pipeline','generated_tokens_host.to_numpy()')])
