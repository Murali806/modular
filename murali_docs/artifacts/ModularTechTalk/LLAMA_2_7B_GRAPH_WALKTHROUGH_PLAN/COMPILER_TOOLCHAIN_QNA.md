# Add-one: Graph, session, compile and init call trees

SOURCE · repository revision 610f7111836d9a950d45ba52ec9b3e0ed589fe3f. Numerical values are illustrative; no MAX execution.

Reading order: A.1 kernels/*.mojo → P1 (.mojoc); A.2 P1 + Graph(...) → G1 (MO); B InferenceSession(...) → S1; C compile(G1) → C1: MO → MAP → MOGG → MGP → CM1; D init(CM1) → M1.

## A · kernels/ → package P1 → recorded graph G1 (MO)

graph = Graph("addition", forward=..., input_types=..., custom_extensions=[mojo_kernels])

INPUT: kernels/__init__.mojo + referenced kernels/*.mojo; name="addition"; symbolic X: float32[5,10]
OUTPUT: P1: one precompiled .mojoc package; G1: verified caller-owned MO graph using P1
EXAMPLE: parse/check package → P1; forward records custom("add_one", [X]) → symbolic Y; MO → MAP is in C

Package preparation is shown first. Graph.__init__ creates its empty scaffold earlier and records the custom operation after P1 is ready; A.2 shows that source order. C performs MO → MAP on clone C1.

```text
_import_kernels(...): prepare package P1 [max/python/max/graph/graph.py:L1703] ↗ caller [max/python/max/graph/graph.py:L910]
├─ KernelLibrary.load_paths([kernels/]) [max/python/max/graph/graph.py:L269]
│  └─ _build_mojo_source_package(kernels/) [Mojo/python/mojo/paths.py:L141]
│     ├─ kernels/__init__.mojo [max/examples/custom_ops/kernels/__init__.mojo:L1] + referenced child modules, including add_one.mojo ↗ caller [Mojo/python/mojo/paths.py:L141]
│     ├─ separate compiler process → Mojo executable + mojo precompile kernels/ -o staging.mojoc
│     │  └─ parse → resolve → semantic/lifetime checks → MLIR bytecode ↗ caller [Mojo/python/mojo/paths.py:L141]
│     └─ atomic publication → P1: one .mojoc package ↗ caller [Mojo/python/mojo/paths.py:L141]
├─ existing .mojoc path → load that package directly ↗ caller [max/python/max/graph/graph.py:L1703]
└─ package phase complete → A.1 · parser, checks, file format and subprocess ↗ caller [max/python/max/graph/graph.py:L1703]

2 · Record and finish G1 with P1 available [max/examples/custom_ops/addition.py:L33]
├─ @in_default_mlir_context [max/python/max/_mlir_context.py:L70] → activate an MLIR context on this thread → Full explanation · is this a global variable? → A.0 ↗ caller [max/examples/custom_ops/addition.py:L33]
├─ reuse the constructor's empty builtin.ModuleOp + mo.GraphOp("addition") scaffold
│  └─ input X: float32[5,10] + device metadata + ordering chains ↗ caller [max/examples/custom_ops/addition.py:L33]
├─ forward(symbolic X)
│  └─ ops.custom(name="add_one", values=[X], out_types=[Y_type], device=DeviceRef.from_device(device)) [max/python/max/graph/ops/custom.py:L69]
│     ├─ insert mo.custom → symbolic Y ↗ caller [max/python/max/graph/ops/custom.py:L69]
│     └─ verify_custom_op [max/python/max/graph/graph.py:L307] → verify against P1 ↗ caller [max/python/max/graph/ops/custom.py:L69]
├─ Graph.output(Y) [max/python/max/graph/graph.py:L1396] → mo.output + output signature + graph verification ↗ caller [max/examples/custom_ops/addition.py:L33]
└─ return G1: caller-owned MO graph → A.3 · registration and validation ↗ caller [max/examples/custom_ops/addition.py:L33]
   Next: create S1 (B); compile a clone C1: MO → MAP → MOGG → MGP (C).
```

### A.0 · What is the default MLIR context? Is it a global variable?

**Yes. `_DEFAULT_MLIR_CONTEXT` is a module-level global variable holding an `mlir.Context` object.** MAX creates it when this module is imported:

```python
_DEFAULT_MLIR_CONTEXT = mlir.Context()
```

The same default object is reused within that Python interpreter. See [_mlir_context.py:26](sources/max/python/max/_mlir_context.py#L26).

An **MLIR context** provides the shared infrastructure for representing compiler IR:

- **Dialects:** definitions of operations and their types.
- **Types:** shared representations such as `f32` and tensor types.
- **Attributes:** metadata attached to operations and types.

Multiple graphs can use the same context while keeping their own operations and graph structure. ([mlir.llvm.org](https://mlir.llvm.org/doxygen/classmlir_1_1MLIRContext.html))

These three names have different roles: one is a context **object**, one is the **per-thread active-context lookup**, and one is a **thread-pool executor** that scopes tasks with a context.

| Name | What it is | The question it answers | Lifetime / behavior |
|---|---|---|---|
| [`_DEFAULT_MLIR_CONTEXT`](sources/max/python/max/_mlir_context.py#L26) | One module-level `mlir.Context` object in this Python interpreter. | “Which context should MAX use as its fallback?” | Created at module import. MAX enters it on the importing thread and exits it at interpreter shutdown. Worker tasks can temporarily enter this same object. |
| [`mlir.Context.current`](sources/max/python/max/_mlir_context.py#L27) | A thread-local property; it is not another context object or a module global. | “Which context, if any, is active on **this OS thread right now**?” | Can be `None`, `_DEFAULT_MLIR_CONTEXT`, or another context. One thread's current value is not automatically copied to another thread. |
| [`MLIRThreadPoolExecutor`](sources/max/python/max/_mlir_context.py#L89) | MAX's subclass of Python's `ThreadPoolExecutor`. It is an executor, not a context. | “How should tasks submitted to this pool get a usable MLIR context?” | Its `submit()` wraps each task with `call_with_default_mlir_context`. On the worker thread, that wrapper keeps an already-current context or temporarily enters `_DEFAULT_MLIR_CONTEXT` if the worker has none. It does not create a context per task or copy the submitting thread's current context. |

The implementation and its thread-local note are in [_mlir_context.py:26–67](sources/max/python/max/_mlir_context.py#L26); the executor wrapper is in [_mlir_context.py:89](sources/max/python/max/_mlir_context.py#L89).

```text
Shared object: _DEFAULT_MLIR_CONTEXT
Main thread: module import enters it → mlir.Context.current on main points to this object
Worker task: MLIRThreadPoolExecutor.submit → call_with_default_mlir_context(fn)
  If current on that worker is None → temporarily enter the shared default → run fn → leave it
  If current on that worker is already set → preserve it → run fn
```

So “current” is the thread's active-context slot; “default” is the shared object that MAX can put in that slot when it is empty; the executor runs tasks on worker threads and applies that fallback around each submitted task. The executor is not a per-context mutex or semaphore. The conditional behavior is implemented in [ensure_default_mlir_context:55](sources/max/python/max/_mlir_context.py#L55) and [call_with_default_mlir_context:44](sources/max/python/max/_mlir_context.py#L44). **Entering a context makes it active for IR-building APIs; entering alone does not compile or execute the graph.** ([MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/))

For the `Graph(...)` call tree, the decorator applies the same rule directly around construction:

```text
Graph(...)
└─ @in_default_mlir_context
   ├─ This thread already has an active context
   │  └─ Keep that context; call Graph.__init__
   └─ This thread has no active context
      └─ Temporarily enter _DEFAULT_MLIR_CONTEXT
         └─ Call Graph.__init__
```

### A.0.1 · What does `@in_default_mlir_context` mean?

It is **Python decorator syntax**. `in_default_mlir_context` is a Python function; `@` applies it to the `__init__` function when the class body is created. It is not a Python attribute on `Graph` or on a graph instance.

This syntax:

```python
@in_default_mlir_context
def __init__(self, ...):
    ...
```

is equivalent in effect to:

```python
def __init__(self, ...):
    ...

__init__ = in_default_mlir_context(__init__)
```

The decorator returns a wrapper. When `Graph(...)` calls the wrapped constructor, the wrapper enters `ensure_default_mlir_context()`, calls the original constructor, and leaves the context manager when construction returns or raises. If this thread already has an active MLIR context, the helper keeps it instead of entering another one. See the [decorator implementation](sources/max/python/max/_mlir_context.py#L70).

### A.0.2 · Why mention a background thread?

A **background thread** (bg-thread) is another OS thread in the same Python process, often a worker created by a thread pool. It is not another process. `mlir.Context.current` is thread-local: the main thread's active-context setting is not automatically copied to a worker thread. MAX's source states this directly in [`ensure_default_mlir_context`](sources/max/python/max/_mlir_context.py#L54).

For example, a caller may construct a graph from a worker:

```python
from concurrent.futures import ThreadPoolExecutor

def make_graph():
    return Graph("addition", input_types=[...], forward=...)

with ThreadPoolExecutor() as workers:
    graph = workers.submit(make_graph).result()
```

When `Graph.__init__` runs on that worker, its decorator notices whether that worker has a current MLIR context. If not, it temporarily enters `_DEFAULT_MLIR_CONTEXT` so `builtin.ModuleOp`, `mo.GraphOp`, and related MLIR bindings run with the intended context. The scope ends with the constructor call. MAX also provides an [`MLIRThreadPoolExecutor`](sources/max/python/max/_mlir_context.py#L89) that wraps submitted tasks with the same context guard. The module comments say this scoped worker handling also avoids nanobind GIL/TLS teardown crashes ([context setup and comments](sources/max/python/max/_mlir_context.py#L26)).

### A.1 · Parse kernels/*.mojo → package P1 (.mojoc)

```text
_build_mojo_source_package(kernels_dir) [Mojo/python/mojo/paths.py:L141]  [parent Python process]
├─ input: absolute directory with kernels/__init__.mojo [max/examples/custom_ops/kernels/__init__.mojo:L1] ↗ caller [Mojo/python/mojo/paths.py:L141]
├─ hash = MD5(absolute path string) → a stable filename, not a source-content hash ↗ caller [Mojo/python/mojo/paths.py:L141]
├─ final: <tempdir>/.modular_<uid>/mojo_pkg/mojo_pkg_<hash>.mojoc ↗ caller [Mojo/python/mojo/paths.py:L141]
├─ staging: .staging.<pid>/mojo_pkg_<hash>.mojoc ↗ caller [Mojo/python/mojo/paths.py:L141]
├─ subprocess_run_mojo(["precompile", kernels_dir, "-o", staging], check=True) [Mojo/python/mojo/run.py:L57]
│  ├─ SDK default: <MAX package root>/bin/mojo; ambient path override [Mojo/python/mojo/run.py:L22] ↗ caller [Mojo/python/mojo/run.py:L57]
│  ├─ MODULAR_MOJO_MAX_DRIVER_PATH → otherwise shutil.which("mojo") on PATH ↗ caller [Mojo/python/mojo/run.py:L57]
│  └─ subprocess.run(...) → Python waits for the separate compiler process
│     └─ mojo precompile: precompile(State) [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209]
│        ├─ validate source package / names; create compiler context; open output ↗ caller [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209]
│        ├─ invokeMojoParser(...)
│        │  └─ LIT::importMojoPackage(kernels_dir, "kernels", ...) [Mojo/lib/MojoParser/EntryPoint.cpp:L391]
│        │     └─ DeclResolver::resolveAllReferencedFrom(package) [Mojo/lib/MojoParser/DeclResolver.cpp:L1756]
│        │        ├─ parse the package + child modules, including add_one.mojo [max/examples/custom_ops/kernels/add_one.mojo:L1] ↗ caller [Mojo/lib/MojoParser/DeclResolver.cpp:L1756]
│        │        └─ resolve referenced dependency declarations and bodies ↗ caller [Mojo/lib/MojoParser/DeclResolver.cpp:L1756]
│        ├─ buildPrecompiledPackage(...) [Mojo/lib/Precompile/Precompile.cpp:L146] → serializable post-parse IR clone ↗ caller [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209]
│        ├─ KGENCompiler::runCheckLITPipeline(parsed_module) [Mojo/lib/Compiler/KGENCompiler.cpp:L1119] → semantic/lifetime checks ↗ caller [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209]
│        └─ writePrecompiledFile(built_package, stream) [Mojo/lib/Support/MojoPrecompiledFile.cpp:L128]
│           └─ LIT lit.package MLIR bytecode → zstd payload + MPKG V2 header → staging .mojoc ↗ caller [Mojo/lib/Support/MojoPrecompiledFile.cpp:L128]
│
├─ compiler succeeds → os.replace(staging, final) → atomic publication ↗ caller [max/examples/custom_ops/addition.py:L33]
├─ finally → remove staging directory ↗ caller [max/examples/custom_ops/addition.py:L33]
└─ return final Path → P1; addition.py stays a separate Python program ↗ caller [max/examples/custom_ops/addition.py:L33]
   The source-directory branch invokes precompile again; it has no exists() skip.
```

### A.1.1 · Is `mojo` an executable? Where is it?

**Yes.** Here, `mojo` is the Mojo compiler's command-line driver executable. Python starts it as a separate child process with `subprocess.run`; `mojo` is not a Python class and is not just a shell alias. The actual argument vector is:

```text
[resolved-mojo-executable, "precompile", <kernels-directory>, "-o", <staging-.mojoc-path>]
```

There is **no single absolute path for every install**. The [SDK environment setup](sources/Mojo/python/mojo/run.py#L22) chooses `<get_package_root()>/bin/mojo` when MAX can find its installed package root. An ambient `MODULAR_MOJO_MAX_DRIVER_PATH` overrides that default. If the package root is unavailable (the helper calls this the Bazel case) and there is no override, [the launcher falls back to `shutil.which("mojo")`](sources/Mojo/python/mojo/run.py#L57), which searches `PATH`.

For a wheel install, the bundled path is therefore typically shaped like `<venv>/lib/pythonX.Y/site-packages/max/bin/mojo`; in a Bazel or source-run environment, it can be a runfiles/toolchain path or whatever `PATH` resolves. To print the path selected by the installed `mojo` Python helper in the current environment:

```bash
python -c 'from mojo.run import _mojo_env; import shutil; e = _mojo_env(); print(e.get("MODULAR_MOJO_MAX_DRIVER_PATH") or shutil.which("mojo"))'
```

The helper verifies that the resolved path exists, prepends it to the `precompile` arguments, and waits for that compiler process to finish before publishing P1.

### A.1.2 · What does the `add_one` MLIR bytecode look like?

First, distinguish the **file bytes** from a **readable MLIR listing**. The `.mojoc` file is an MPKG V2 package header plus a zstd-compressed MLIR bytecode payload; literal byte values are binary and do not display as a function listing. During this precompile step the payload is a `lit.package` containing post-parse **LIT-dialect** IR, including function bodies after semantic/lifetime checks. It has not yet been specialized for a concrete `target`/`width`, lowered through the kernel JIT, or emitted as CPU/GPU machine code. See the [repository's Mojo package walkthrough](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L612) and the [package writer](sources/Mojo/lib/Support/MojoPrecompiledFile.cpp#L128).

The exact source expression is in [`add_one.mojo`](sources/max/examples/custom_ops/kernels/add_one.mojo#L32). `execute` has a compile-time `target`; its helper has a compile-time `width`:

```mojo
def elementwise_add_one[
    width: Int
](idx: Coord) {var} -> SIMD[x.dtype, width]:
    return x.load[width](idx) + 1

foreach[target=target](elementwise_add_one, output, ctx)
```

The package's readable structure is conceptually:

```text
lit.package @kernels
  lit.struct.decl @AddOne
    lit.fn "execute[target: StaticString]"  [body retained]
      helper "elementwise_add_one[width: Int]"  [body retained]
        load x[idx] as SIMD → add 1 → return SIMD
      foreach[target=target](elementwise_add_one, output, ctx)
```

This is a **structural sketch, not a captured compiler dump**: the repository documents `lit.package`, `lit.struct.decl`, and `lit.fn` as the package IR shape, while the arrows summarize the source-level math rather than claiming exact LIT operation spellings. The actual `.mojoc` payload is binary. A literal byte dump or exact decoded `add_one` IR would require running this repository revision's compiler and capturing its output. See the [documented package shape](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L647).

### _location() · What is loc?

**SOURCE · loc is an MLIR Location: metadata describing where an IR operation was created.** The Python variable <code>loc</code> holds the Location returned by <code>_location()</code>. The active MLIR Context owns its interned metadata; source locations help diagnostics and source-aware debugging. See [the helper](sources/max/python/max/graph/graph.py#L438) and [the native frame binding](sources/max/python/max/_core/max_license/modules/graph.cpp#L162).

**Inputs → decisions → output, in source order**

```text
Graph.__init__: with _location() as loc [LINK:graph_init:851]
└─ _location(ignore_frames=1, attach_profile_scopes=True) [LINK:ct_graph_location:438]
   ├─ Require an active mlir.Context.current [LINK:ct_graph_location:442]
   │  └─ absent → raise RuntimeError [LINK:ct_graph_location:443]
   ├─ Read the current source_tracebacks setting [LINK:ct_graph_location:449]
   │  ├─ disabled → Location.unknown() [LINK:ct_graph_location:450]
   │  └─ enabled → traceback.extract_stack() [LINK:ct_graph_location:457]
   │     ├─ Keep frame summaries; trim _location + ignore_frames callers [LINK:ct_graph_location:457]
   │     ├─ nonempty → _graph.frame_loc(current_context, tb) [LINK:ct_graph_location:459]
   │     │  └─ PythonFrameAttr::getOpaque → MLIR Location [LINK:ct_graph_frame_loc:165]
   │     └─ empty → Location.unknown() [LINK:ct_graph_location:461]
   ├─ If attach_profile_scopes: wrap active name/color scopes [LINK:ct_graph_location:464]
   │  └─ _graph.profile_scope_location(context, name, location, color) [LINK:ct_graph_location:472]
   └─ Return location → the variable loc [LINK:ct_graph_location:475]
```

**What <code>with</code> does next**

The Location context manager makes <code>loc</code> current for the block. MAX also passes that same object explicitly to [<code>builtin.ModuleOp(location=loc)</code>](sources/max/python/max/graph/graph.py#L856) and [<code>_mo.GraphOp(builder, loc, ...)</code>](sources/max/python/max/graph/graph.py#L860). The native [GraphOp builder](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) attaches its operation location to every input block argument. Exiting <code>with</code> restores the surrounding current Location; the created IR retains its attached metadata. ([MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/))

**Add-one example · a Python stack becomes IR provenance**

The graph is requested at [addition.py:33](sources/max/examples/custom_ops/addition.py#L33). With source tracebacks enabled, the helper collects the stack and removes its final two frames with the default <code>ignore_frames=1</code>:

~~~text
Before trimming, illustrative frame summaries:
  ... → addition.py:33 → ... → Graph.__init__:851 → _location:457

After trimming:
  ... → addition.py:33 → ...

frame_loc(context, remaining summaries) → loc
  loc → builtin.ModuleOp
  loc → mo.GraphOp("addition") → symbolic input argument X
~~~

The ellipses allow surrounding Python/decorator frames. This is a source-derived example of retained provenance, rather than an executed stack capture or an exact printed MLIR dump. The native wrapper stores the summaries in an opaque MLIR source Location. Profiling scopes can wrap either the stack-derived or unknown Location, even when source tracebacks are disabled. See [frame_loc](sources/max/python/max/_core/max_license/modules/graph.cpp#L162) and [scope wrapping](sources/max/python/max/graph/graph.py#L464).

### MLIR Module · Reuse or create the IR container

**SOURCE · self._module holds the selected builtin.ModuleOp: the IR container for this graph.** A public Python <code>Module</code> wrapper exposes that operation through <code>module.mlir_module</code>. Its body can hold several named graph definitions, compiled together. See [Module's contract](sources/max/python/max/graph/graph.py#L562).

**Inputs → branch → retained module, in source order**

```text
Graph.__init__(..., module=module): inside with _location() as loc [LINK:graph_init:851]
└─ self._module = module.mlir_module if module else builtin.ModuleOp(location=loc) [LINK:graph_init:855]
   ├─ module supplied → read its mlir_module attribute [LINK:ct_graph_module:619]
   │  └─ retain the same typed builtin.ModuleOp as self._module [LINK:ct_graph_module:637]
   └─ module omitted → builtin.ModuleOp(location=loc) [LINK:graph_init:856]
      └─ nanobind ModuleOp constructor(location, name=None) [LINK:ct_builtin_module:62]
         └─ mlir::ModuleOp::create(loc, name) [LINK:ct_builtin_module:65]
            └─ new builtin.module → body region → initially empty body block [LINK:ct_builtin_module:68]
```

<code>loc</code> comes from the preceding <code>_location()</code> call. The fresh-module branch passes it to the native constructor. The reuse branch keeps the existing module and its attached location; the upcoming graph definition receives the current <code>loc</code> separately. See [the conditional assignment and GraphOp call](sources/max/python/max/graph/graph.py#L855).

**Add-one · no module argument**

The [example's Graph call](sources/max/examples/custom_ops/addition.py#L33) omits <code>module=</code>. At this row, the newly created module's body is empty. The next two source actions position the builder and insert the graph definition:

~~~text
At this row                         After GraphOp insertion
self._module                        self._module
  builtin.module                      builtin.module
    body region                         body region
      body block: []                      body block
                                            mo.graph @addition
                                              graph region / block
                                                symbolic argument X
~~~

These are schematic IR structures before forward records the computation. The module body contains graph definitions; the graph's own body contains its computation. See [GraphOp insertion](sources/max/python/max/graph/graph.py#L860) and [native graph-body creation](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L847).

**Reuse example · encoder and decoder share one compilation unit**

Excerpt of the [Module documentation example](sources/max/python/max/graph/graph.py#L588); graph-output code is omitted here:

~~~python
module = Module()
with Graph("encoder", input_types=[], module=module) as encoder:
    ...  # record encoder output
with Graph("decoder", input_types=[], module=module) as decoder:
    ...  # record decoder output
~~~

~~~text
module.mlir_module ────────┐
encoder._module ──────────┼── same builtin.module
decoder._module ──────────┘     body block
                                 mo.graph @encoder
                                 mo.graph @decoder
~~~

Both Graph objects retain the shared enclosing operation and their own graph definitions. Constructing the decoder appends a definition to the existing module; it preserves the encoder definition. [Module's documented example](sources/max/python/max/graph/graph.py#L609) hands this unit to <code>InferenceSession.load_all(module)</code>.

### OpBuilder · Insert at module.body.end

**SOURCE · OpBuilder is a native MLIR operation builder, positioned after the module body's existing operations.** The Python constructor creates a C++ <code>mlir::OpBuilder</code>; its insertion point identifies a block and a position within that block. See [the MAX call site](sources/max/python/max/graph/graph.py#L858) and [the native constructor](sources/max/python/max/_core/max_license/_core.cpp#L612).

**Evaluate the expression left to right**

```text
builder = OpBuilder(self._module.body.end) [LINK:graph_init:858]
├─ self._module → selected builtin.ModuleOp [LINK:graph_init:855]
├─ .body → native ModuleOp body binding [LINK:ct_builtin_module:68]
│  └─ &self.getBodyRegion().front() → the module body block [LINK:ct_builtin_module:69]
├─ .end → native Block.end binding [LINK:ct_block_end:408]
│  ├─ cast the Python block view to mlir::Block* [LINK:ct_block_end:410]
│  └─ OpBuilder::InsertPoint(block, block->end()) [LINK:ct_block_end:411]
└─ OpBuilder(insert_point) → nanobind constructor [LINK:ct_op_builder:612]
   └─ std::make_unique<mlir::OpBuilder>(ip.getBlock(), ip.getPoint()) [LINK:ct_op_builder:614]
      └─ OUT: builder ready to insert operations in this module block [LINK:ct_op_builder:615]
```

<code>body.end</code> is a position after the last operation, including the empty-block case. Creating the builder sets this position; the next [<code>_mo.GraphOp(builder, loc, ...)</code>](sources/max/python/max/graph/graph.py#L860) call creates the graph at that position. The native [GraphOp builder](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820) fills the graph signature, region and input block arguments.

**Fresh add-one module · the first insertion**

~~~text
Before builder creation          Builder ready                 After GraphOp call
module body: []                  module body: []               module body:
                                ↑ insertion point at end        mo.graph @addition
~~~

**Shared module · append the decoder after the encoder**

~~~text
Before decoder construction         After decoder GraphOp insertion
module body                          module body
  mo.graph @encoder                    mo.graph @encoder
  ↑ end insertion point                mo.graph @decoder
~~~

The arrow marks the position after <code>@encoder</code>. The selected block is the outer module body, so the new graph is a sibling of the encoder. The graph's computation is recorded later in its own body through [<code>with self</code> and the forward callback](sources/max/python/max/graph/graph.py#L921).

**Input → output:** the existing module's body block and its end iterator become a native builder stored in the local Python variable <code>builder</code>. That builder is passed immediately to GraphOp construction. These diagrams describe symbolic IR construction, with no compilation or tensor execution at this step.

### to_mlir() · Python input type → MLIR tensor type

**SOURCE · t.to_mlir() turns a Python type description into the native MO tensor type used by the graph's IR.** In add-one, <code>t</code> is the caller's <code>TensorType</code> describing float32 elements, shape <code>[5, 10]</code>, and a selected device. The result is a <code>mo.TensorType</code> handle. See [the symbolic-type contract](sources/max/python/max/graph/type.py#L358) and [the conversion method](sources/max/python/max/graph/type.py#L430).

**Inputs → conversions → native type, in source order**

```text
Graph.__init__: input_types=[t.to_mlir() for t in input_types] [LINK:graph_init:864]
└─ TensorType.to_mlir(self=t) [LINK:ct_tensor_to_mlir:430]
   ├─ metadata = []; optionally append the layout attribute [LINK:ct_tensor_to_mlir:436]
   │  └─ add-one has _layout=None → empty metadata [LINK:ct_tensor_description:404]
   └─ return mo.TensorType(...) [LINK:ct_tensor_to_mlir:439]
      ├─ argument 1: self.shape.to_mlir() [LINK:ct_tensor_to_mlir:440]
      │  └─ Shape.to_mlir() → mosh.ShapeAttr(dim attributes, mosh.ShapeType()) [LINK:ct_shape_to_mlir:36]
      │     └─ for 5 and 10: StaticDim.to_mlir() [LINK:ct_static_dim_to_mlir:562]
      │        └─ signed i64 IntegerAttr → kgen.CastFromBuiltinAttr [LINK:ct_static_dim_to_mlir:570] [CALLER:ct_shape_to_mlir|39]
      ├─ argument 2: self.dtype → existing DType.float32 value [LINK:ct_tensor_to_mlir:441] [CALLER:ct_tensor_to_mlir|441]
      ├─ argument 3: self.device.to_mlir() [LINK:ct_tensor_to_mlir:442]
      │  └─ DeviceRef.to_mlir() → m.DeviceRefAttr(kind string, device id) [LINK:ct_device_ref_to_mlir:181] [CALLER:ct_tensor_to_mlir|442]
      └─ keyword metadata: builtin.DictionaryAttr(metadata) [LINK:ct_tensor_to_mlir:443] [CALLER:ct_tensor_to_mlir|443]
```

The method is called on each Python input type. For this example there is one input, so GraphOp receives a one-element list of native types. The native MO type builders derive their MLIR Context from the shape attribute and retain shape, dtype, device reference and metadata as the tensor type's fields. See [the context-inferred builder](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L324) and [the base type builder](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L304).

**Add-one example · before and after**

Illustrative Python using the same float32 <code>[5, 10]</code> input declaration as [addition.py](sources/max/examples/custom_ops/addition.py#L51), with CPU selected explicitly:

~~~python
t = TensorType(DType.float32, shape=[5, 10], device=DeviceRef.CPU())
ir_type = t.to_mlir()
input_types_for_graph_op = [ir_type]
~~~

| Python description | Native type field |
| --- | --- |
| <code>shape=[5, 10]</code> | <code>mosh.ShapeAttr</code> containing typed attributes for 5 and 10 |
| <code>dtype=DType.float32</code> | dtype field; displayed as <code>f32</code> in MLIR assembly |
| <code>device=DeviceRef.CPU()</code> | <code>m.DeviceRefAttr("cpu", 0)</code> |
| <code>_layout=None</code> | empty <code>builtin.DictionaryAttr</code> metadata |

~~~text
Python type description t
  TensorType(float32, [5, 10], cpu:0)
                ↓ t.to_mlir()
Native mo.TensorType handle ir_type
  shape + dtype + device reference + metadata
                ↓ passed in input_types=[ir_type]
mo.GraphOp("addition")
  input block argument X has this type
~~~

**Schematic IR view of that typed input** — this is an illustration, not a captured compiler dump:

~~~mlir
// CPU: the default device can be omitted in printed tensor types.
mo.graph @addition(%x: !mo.tensor<[5, 10], f32>) -> () {
  // Construction snapshot: forward/output have not been recorded yet.
}

// If the same description uses DeviceRef.GPU(0), the device field changes:
// %x: !mo.tensor<[5, 10], f32, gpu:0>
~~~

The [native tensor-type printer](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L219) prints these fields; [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) gives each input block argument its declared type. This operation converts type metadata: tensor numbers enter later through execution buffers, while <code>t</code> remains the Python description. Conversion itself does not emit a <code>.mojoc</code> package or run graph-compilation passes.

**1 · Shape: what happens to the numbers 5 and 10?**

The Python sizes describe the number of rows and columns. The compiler needs typed attributes for those sizes, not raw Python integers. [StaticDim.to_mlir](sources/max/python/max/graph/dim.py#L570) constructs each attribute; [Shape.to_mlir](sources/max/python/max/graph/shape.py#L39) collects them.

~~~text
StaticDim(5)                                   StaticDim(10)
  ↓ self.dim = 5                                 ↓ self.dim = 10
IntegerType(64, signed) = si64                   IntegerType(64, signed) = si64
  ↓                                              ↓
IntegerAttr(si64, 5)                            IntegerAttr(si64, 10)
  ↓                                              ↓
kgen.CastFromBuiltinAttr(the typed 5)           kgen.CastFromBuiltinAttr(the typed 10)
  └──────────────────────┬───────────────────────┘
                         ↓
mosh.ShapeAttr([dimension-expression 5, dimension-expression 10], mosh.ShapeType())
                         ↓
shape field of mo.TensorType → printed shape [5, 10]
~~~

**Read it as:** <code>si64</code> means a signed 64-bit integer <em>type</em>; <code>IntegerAttr(si64, 5)</code> is an immutable typed constant describing size 5; <code>CastFromBuiltinAttr</code> wraps that builtin constant for KGEN's dimension-expression representation. It does not cast the tensor's float32 elements to integers. This structural illustration is not the exact serialized bytecode or printed attribute spelling. A tensor with 5 rows still has float32 elements.

**2 · Dtype: why is self.dtype passed directly?**

~~~python
# Illustrative values already stored in the Python type description:
t.dtype                 # DType.float32
# TensorType.to_mlir supplies this existing value as argument 2:
mo.TensorType(shape_attr, t.dtype, device_attr, metadata=metadata_attr)
~~~

[The conversion](sources/max/python/max/graph/type.py#L441) passes the dtype value already chosen by the caller. The [native type builder](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L304) uses it for the element type; the [printer](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L219) displays float32 as <code>f32</code>. For <code>[5, 10]</code>, this describes 50 float32 elements when a tensor is supplied later. Choosing <code>DType.int32</code> instead would describe integer elements; this method does not convert any actual tensor data.

| Field | Meaning in this example |
| --- | --- |
| <code>IntegerAttr(si64, 5)</code> | Row-count metadata: the size is 5 |
| <code>DType.float32</code> / <code>f32</code> | Element datatype: each tensor value is float32 |

**3 · Device: what becomes cpu:0 or gpu:0?**

[DeviceRef.to_mlir](sources/max/python/max/graph/type.py#L181) takes the symbolic device kind and index:

~~~text
DeviceRef.CPU()   → self.device_type="cpu", self.id=0 → m.DeviceRefAttr("cpu", 0)
DeviceRef.GPU(0)  → self.device_type="gpu", self.id=0 → m.DeviceRefAttr("gpu", 0)
DeviceRef.GPU(1)  → self.device_type="gpu", self.id=1 → m.DeviceRefAttr("gpu", 1)
~~~

The attribute labels the graph tensor's device. A <code>gpu:0</code> label is not a GPU buffer or a driver handle; forming it does not allocate or transfer tensor data. The same <code>[5, 10]</code> / <code>f32</code> type can therefore carry either a CPU or GPU reference. The tensor-type printer omits the default <code>cpu:0</code> field and prints a nondefault GPU field.

**4 · Metadata: a concrete convolution layout example**

**Add-one:** <code>_layout=None → metadata=[] → builtin.DictionaryAttr([])</code>. There is no convolution-filter layout attached to its input. The optional layout path is useful for convolution filters; [FilterLayout](sources/max/python/max/graph/type.py#L38) describes their memory layout.

~~~python
# Illustrative internal type description for a convolution filter:
filter_type = TensorType(
    DType.float32, [3, 3, 16, 32], device=DeviceRef.CPU(),
    _layout=FilterLayout.RSCF,
)
# to_mlir() builds:
layout_attr = FilterLayout.RSCF.to_mlir()  # mo.LayoutAttr(format_str="RSCF")
metadata = [NamedAttribute("layout", layout_attr)]
metadata_attr = builtin.DictionaryAttr(metadata)
# metadata_attr is argument metadata=... of mo.TensorType(...)
~~~

[conv2d's documented order](sources/max/python/max/graph/ops/conv.py#L44) is <code>R, S, C, F = height, width, input channels / groups, output channels</code>. With groups=1, this example is a 3×3 filter with 16 input channels and 32 output channels. The <code>layout</code> key tells compiler operations how to interpret the axes, separately from element dtype and device.

~~~text
Python _layout=FilterLayout.RSCF
  → mo.LayoutAttr("RSCF")
  → DictionaryAttr {layout: the RSCF attribute}
  → metadata field on native mo.TensorType
  → TensorType::getLayoutAttr() reads metadata["layout"]
  → filter-packing shape calculation reads the format:
       RSCF: F is last axis → [3, 3, 16, 32] → F = 32
       FCRS: F is first axis → [32, 16, 3, 3] → F = 32
~~~

The [native metadata reader](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L454) obtains this key; the [filter-packing shape function](sources/GraphCompiler/lib/MODialect/PackingShapeFunctions.cpp#L279) uses the layout to choose the F axis. [conv2d](sources/max/python/max/graph/ops/conv.py#L141) attaches the requested filter layout through <code>filter._with_layout(filter_layout)</code>. Merely attaching metadata does not transpose a buffer. RSCF and FCRS examples describe different axis orders; a real reorder requires the appropriate operation.

**Small extension · a symbolic dimension**

For <code>TensorType(DType.float32, ["batch", 10], device=DeviceRef.CPU())</code>, the shape remains rank two. Instead of fixing the first size, [SymbolicDim.to_mlir](sources/max/python/max/graph/dim.py#L286) creates <code>kgen.ParamDeclRefAttr("batch", si64)</code>. The second dimension still follows the static-size conversion above. The graph later declares its named dimension parameters from these types.

### GraphOp · Region, block arguments and signature

**SOURCE · GraphOp creates one region and one body block; each input adds an argument to that same block.** The region is created by the builder, not passed in as existing region information. See [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820).

**Native calls in source order**

```text
Graph.__init__ → mo.GraphOp(builder, loc, name, input_types, result_types=[]) [LINK:graph_init:860]
└─ GraphOp::build(builder, state, name, inputTypes, resultTypes, isSubgraph) [LINK:ct_graph_op_build:820]
   ├─ Set symbol name: "addition" [LINK:ct_graph_op_build:825] [CALLER:ct_graph_op_build|825]
   ├─ builder.getFunctionType(inputTypes, resultTypes) → ordered input/output types [LINK:ct_graph_op_build:831] [CALLER:ct_graph_op_build|831]
   ├─ Wrap function type in KGEN::FuncTypeGeneratorType; attach signature and functionType [LINK:ct_graph_op_build:832] [CALLER:ct_graph_op_build|832]
   ├─ Collect named dimension parameters from input/result shapes [LINK:ct_graph_op_build:839] [CALLER:ct_graph_op_build|839]
   ├─ state.addRegion() → new empty body region owned by GraphOp [LINK:ct_graph_op_build:846] [CALLER:ct_graph_op_build|846]
   ├─ bodyRegion->emplaceBlock() → ONE block inside that region [LINK:ct_graph_op_build:847] [CALLER:ct_graph_op_build|847]
   └─ for each input type: body.addArgument(input, state.location) [LINK:ct_graph_op_build:848]
      └─ append a typed argument to the SAME block; attach loc [LINK:ct_graph_op_build:849] [CALLER:ct_graph_op_build|849]
```

**Where does the region come from?** <code>OperationState</code> is the operation being assembled. <code>state.addRegion()</code> creates its nested body container; <code>emplaceBlock()</code> creates the block that will hold the computation. The later forward callback inserts operations into that body. Input types provide the arguments' types; they do not provide a region or a block per input.

~~~text
Example with TWO user inputs (schematic, not an executed dump):
inputTypes = [float32[5,10], float32[5,10]]
                         ↓
builtin.module
└─ mo.graph @pair                 ← graph operation
   └─ body region                ← created ONCE by addRegion()
      └─ entry block             ← created ONCE by emplaceBlock()
         ├─ block argument %x : float32[5,10]  ← first addArgument()
         ├─ block argument %y : float32[5,10]  ← second addArgument()
         └─ operations inserted later: use %x and %y in this block
~~~

**Why one argument per input?** Each argument gives the computation a separate symbolic value for an input position. For a two-input addition, the body needs both <code>%x</code> and <code>%y</code>. Arguments are values, not blocks. A block contains an ordered sequence of operations; splitting the inputs into two blocks would not represent the simple two-input computation shown here. In the actual add-one example, there is only one declared tensor input, so the native builder initially adds one tensor argument.

**Why keep a signature?** It is the ordered type contract for entering and leaving the graph. It tells the compiler the number and types of inputs/results, including shapes, dtypes and devices carried by those types. The builder stores a function type and its KGEN generator signature; the signature is metadata about the computation, while the body contains the computation itself.

~~~text
Add-one lifecycle (T = !mo.tensor<[5, 10], f32>):
Shell construction:    inputTypes=[T], resultTypes=[] → function type (T) → ()
Forward recording:     %x : T → mo.custom "add_one" → %result : T
Graph.output(result):  update output contract        → function type (T) → (T)
                       attach generator signature + mo.output terminator
~~~

[Graph.output](sources/max/python/max/graph/graph.py#L1455) constructs the function type from the body-argument types and output-value types, then updates <code>signature</code> and <code>functionType</code> before inserting the output operation. The empty result list is a construction-stage snapshot, not the final add-one output contract.

~~~mlir
// Schematic two-input computation, after forward/output recording.
// Block arguments are printed in the graph header; ^entry is implicit here.
mo.graph @pair(%x: !mo.tensor<[5, 10], f32>,
               %y: !mo.tensor<[5, 10], f32>) -> !mo.tensor<[5, 10], f32> {
  // Recorded operations consume %x and %y and produce %sum.
  // The final mo.output returns %sum using the result type in the signature.
}
~~~

These examples illustrate IR structure and contracts, not executed tensor arithmetic. The source-linked tree above shows the exact native construction calls.

### Shape parameters · Declare symbolic dimensions

**SOURCE · Declare named shape parameters, not extra tensor inputs.** The constructor first collects <code>SymbolicDim</code> names from input shapes, then declares those names on GraphOp. See [discovery](sources/max/python/max/graph/graph.py#L840) and [declaration](sources/max/python/max/graph/graph.py#L868).

```text
Graph.__init__(input_types=...) [LINK:graph_init:840]
├─ scan TensorType/BufferType shapes; collect only SymbolicDim.name [LINK:graph_init:843] [CALLER:graph_init|840]
├─ dict.fromkeys(names) → unique names in first-seen order [LINK:graph_init:840] [CALLER:graph_init|840]
├─ si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True)) [LINK:graph_init:868] [CALLER:graph_init|868]
└─ op.input_parameters = ParamDeclArrayAttr([ParamDeclAttr(p, si64) for p in self._params]) [LINK:graph_init:870] [CALLER:graph_init|870]
```

~~~text
Static add-one input:        TensorType(float32, [5, 10])
  discovered names          []
  self._params              {}
  op.input_parameters       empty ParamDeclArrayAttr
  tensor block arguments    one: %x : float32[5,10]

Illustrative dynamic input: TensorType(float32, ["batch", 10])
  discovered names          ["batch"]
  self._params              {"batch": None}
  op.input_parameters       [ParamDeclAttr("batch", si64)]
  tensor block arguments    one: %x : float32[batch,10]
~~~

**Input → output:** a named shape description becomes a typed parameter declaration; the actual size is not chosen by this constructor. If a later input tensor has shape <code>[8, 10]</code>, 8 is the concrete batch-size example. The declaration itself is not a second tensor or a data allocation. The KGEN <code>si64</code> here is a one-lane signed 64-bit parameter type; it does not change float32 tensor elements.

[SymbolicDim.to_mlir](sources/max/python/max/graph/dim.py#L286) creates the corresponding named parameter reference in the tensor shape. Declaring <code>batch</code> gives that reference a name/type definition. For two tensor types <code>["batch", 10]</code> and <code>["batch", 20]</code>, <code>dict.fromkeys</code> declares <code>batch</code> once while both shapes refer to it. Static sizes 5 and 10 remain dimension constants.

### Graph handles · Retain the operation and body block

**SOURCE · `_CAPICreate(op._CAPIPtr)` bridges two Python binding families around the same native GraphOp.** `op` is MAX's `_core` GraphOp wrapper. `mlir.Operation` is the MLIR Python Operation class [re-exported by MAX](sources/max/python/max/mlir/__init__.py#L19). The bridge creates a Python wrapper, not another graph, region, block, or IR copy.

**CAPI = C Application Programming Interface.** MLIR exposes C-compatible handles and functions around its C++ implementation. `MlirOperation` is a small opaque handle containing `void *ptr`; it is not the serialized graph or its tensor data. A Python capsule carries that pointer with a type-specific name so the receiving binding can extract the correct kind of handle. [Opaque structs](sources/external/llvm-project/mlir/include/mlir-c/IR.h#L45), [capsule contract](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L88).

**Exact constructor line, expanded in evaluation order**

```text
self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr) [LINK:graph_init:874]
├─ Evaluate op._CAPIPtr first: MAX _core property getter [LINK:ct_capi_export:607]
│  ├─ self.getOperation() → native mlir::Operation * for the existing mo.graph [LINK:ct_capi_export:609] [CALLER:ct_capi_export|609]
│  ├─ wrap(native pointer) → MlirOperation{ptr = same pointer} [LINK:capi_wrap:25] [CALLER:ct_capi_export|609]
│  └─ mlirPythonOperationToCapsule(handle) → PyCapsule_New(handle.ptr, operation capsule name, NULL) [LINK:capi_capsule:330] [CALLER:ct_capi_export|609]
├─ Call mlir.Operation._CAPICreate(capsule) [LINK:capi_factory_bind:4486]
│  └─ PyOperation::createFromCapsule(capsule) [LINK:capi_factory:1207]
│     ├─ mlirPythonCapsuleToOperation → PyCapsule_GetPointer(capsule, expected operation name) [LINK:capi_capsule:338] [CALLER:capi_factory|1208]
│     ├─ null extracted handle? throw nb::python_error() [LINK:capi_factory:1209] [CALLER:capi_factory|1209]
│     ├─ mlirOperationGetContext(rawOperation) → wrap(unwrap(op)->getContext()) [LINK:capi_context_get:695] [CALLER:capi_factory|1211]
│     └─ forOperation(PyMlirContext::forContext(rawCtxt), rawOperation) [LINK:capi_factory:1212]
│        ├─ createInstance → allocate Python PyOperation wrapper around the same handle [LINK:capi_wrapper_create:948] [CALLER:capi_wrapper_create|964]
│        └─ releaseObject() → return Python object; not release/destroy native IR [LINK:capi_factory:1213] [CALLER:capi_factory|1212]
├─ assign returned Python wrapper to self._mlir_op [LINK:graph_init:874] [CALLER:graph_init|874]
├─ self._current_block = self._mlir_op.regions[0].blocks[0] [LINK:graph_init:875] [CALLER:graph_init|875]
└─ self._graph_body = self._current_block [LINK:graph_init:876] [CALLER:graph_init|876]
```

**A small pointer example** — `P` is a symbolic address label, not a measured address:

~~~text
Native IR already exists:
MODULE1: builtin.module
└─ OP1: mo.graph @addition at address P
   └─ region[0] → block[0] → input %x

MAX _core GraphOp wrapper op ──→ OP1 at P
           ↓ op._CAPIPtr
PyCapsule {name=<qualified operation name>, pointer=P, destructor=NULL}
           ↓ mlir.Operation._CAPICreate(capsule)
MLIR Python PyOperation wrapper ──→ OP1 at the SAME P
           ↓ assigned to G1._mlir_op
G1._current_block and G1._graph_body ──→ the SAME block[0]

Native graph count: 1 throughout. Binding wrappers differ; graph identity does not.
~~~

The property name `_CAPIPtr` is defined by [MLIR's interop header](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L97). Its value is a capsule, not a Python integer address. `_CAPICreate` is a bound static factory method, [mapped to createFromCapsule](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L4486). These are Python extension calls in the same process; no subprocess, HTTP request, compiler invocation or machine-code lowering occurs here.

**What the C API looks like internally** — these short structural examples expand the actual macros/helpers:

~~~cpp
// mlir-c/IR.h exposes an opaque C handle:
struct MlirOperation { void *ptr; };
// CAPI/Wrap.h + CAPI/IR.h generate pointer adapters:
MlirOperation wrap(mlir::Operation *p) { return MlirOperation{p}; }
mlir::Operation *unwrap(MlirOperation h) {
  return static_cast<mlir::Operation *>(h.ptr);
}
// Python Interop.h carries that same pointer across binding families:
PyCapsule_New(h.ptr, MLIR_PYTHON_CAPSULE_OPERATION, NULL);
PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_OPERATION);
~~~

[wrap/unwrap macro](sources/external/llvm-project/mlir/include/mlir/CAPI/Wrap.h#L25), [Operation specialization](sources/external/llvm-project/mlir/include/mlir/CAPI/IR.h#L29), [capsule export/import](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L330). The qualified capsule name includes the binding package prefix; the source defines its suffix as `ir.Operation._CAPIPtr`. The producer and consumer must agree on that name. A wrong-kind capsule fails extraction and raises a Python error; a plain Python integer is not a correctly named capsule. The name check does not prove pointer liveness; the native operation must still be valid.

**Lifetime and identity**

| Item | What it retains or owns here |
| --- | --- |
| `MODULE1` / enclosing module | The attached native GraphOp and its region/block |
| MAX `_core` wrapper `op` | A binding view of the existing native GraphOp |
| `_CAPIPtr` capsule | Pointer + name; the export supplies no destructor or keep-alive owner |
| MLIR `PyOperation` wrapper | The C API operation handle plus a Python MLIR-context reference |
| `G1._mlir_op` | A Python reference to the returned wrapper |

**The capsule itself does not keep the native operation alive.** Keep the enclosing module/graph alive while using the wrappers. This constructor already retains `self._module`. [createFromCapsule](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1207) does not clone, erase, detach, or rename the GraphOp. [PyOperation's destructor](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L921) distinguishes an attached operation from a detached operation owned by Python. Do not generalize the lifetime rules of this attached GraphOp bridge to other factories such as Context/Module: the interop header gives those factories their own ownership contract.

**What the retained body is used for next**

`self._mlir_op.regions[0].blocks[0]` retrieves the existing graph entry block through the new wrapper. Both `_current_block` and `_graph_body` initially reference it. [_add_op_generated](sources/max/python/max/graph/graph.py#L1258) inserts later generated operations at the current block's end; [Graph.inputs](sources/max/python/max/graph/graph.py#L942) wraps the body arguments as symbolic Python values. None of those lookups executes tensor arithmetic.

**Evidence:** MAX export and constructor sources are pinned to the page's MAX revision. The complete upstream MLIR files embedded in Source Explorer are separately pinned to LLVM `6941494e2ae550d753728ba13454311b344d82cf`, the [LLVM_COMMIT selected by that MAX revision](sources/bazel/public-patches/llvm_source.bzl#L5). Source Explorer labels LLVM spans with their dependency revision. These are source-derived implementation examples, not proof that a particular installed wheel was rebuilt or executed.

### Device metadata · Attach the module mapping

**SOURCE · Add target-description metadata to the enclosing MLIR module.** This is separate from the device label on each tensor type and from the session's runtime devices. See [_populate_device_info_mapping](sources/max/python/max/graph/graph.py#L1366).

```text
Graph.__init__ → self._populate_device_info_mapping() [LINK:graph_init:877]
└─ Graph._populate_device_info_mapping() [LINK:ct_graph_device:1366]
   ├─ module = self._mlir_op.block.owner → enclosing builtin.module [LINK:ct_graph_device:1368] [CALLER:ct_graph_device|1368]
   ├─ existing mo.device_info_mapping? return immediately [LINK:ct_graph_device:1369] [CALLER:ct_graph_device|1369]
   ├─ devices = [CPU()]; if accelerator_count() > 0, append Accelerator() [LINK:ct_graph_device:1371] [CALLER:ct_graph_device|1371]
   ├─ read architecture_name and model_name; each falls back to "unknown" on error [LINK:ct_graph_device:1377] [CALLER:ct_graph_device|1375]
   ├─ _DeviceInfoAttr(label, api, arch, model, tile_based_fusion=False) [LINK:ct_graph_device:1385] [CALLER:ct_graph_device|1385]
   └─ entries keyed by dev.label → module attribute mo.device_info_mapping = mlir.DictAttr.get(entries) [LINK:ct_graph_device:1393] [CALLER:ct_graph_device|1393]
```

~~~text
Illustrative mapping, with placeholder values rather than measured hardware:
builtin.module attributes
└─ mo.device_info_mapping
   ├─ "cpu:0" → {label, api, arch="<host architecture>", model="<host model>",
   │             tile_based_fusion=false}
   └─ "gpu:0" → {label, api, arch="<detected architecture>", model="<GPU model>",
                 tile_based_fusion=false}   ← only if an accelerator is detected
~~~

| Item | What it describes |
| --- | --- |
| Tensor <code>DeviceRefAttr("gpu", 0)</code> | This tensor's symbolic device reference |
| Module <code>mo.device_info_mapping</code> | Device labels mapped to target-description attributes |
| <code>InferenceSession(devices=[device])</code> | Runtime devices selected for that session |

**Important source limit:** this helper starts with CPU and adds one default <code>Accelerator()</code> when any accelerator is available; it does not enumerate every GPU in a multi-GPU machine. A supplied shared module with an existing mapping takes the early return. This method records metadata; it does not compile a kernel or submit graph execution.

### Weights and chains · Initialize graph bookkeeping

**SOURCE · Initialize two separate kinds of graph bookkeeping: symbolic weights and dependency chains.** See [constructor initialization](sources/max/python/max/graph/graph.py#L879).

```text
Graph.__init__ [LINK:graph_init:879]
├─ self._weights = {} → initially no named weight declarations [LINK:graph_init:879] [CALLER:graph_init|879]
├─ self._has_chain_input = False → no adopted incoming chain yet [LINK:graph_init:880] [CALLER:graph_init|880]
└─ self.device_chains = _DeviceChainMap(self) [LINK:graph_init:881]
   ├─ OrderedDict.__init__() → initially empty mapping [LINK:ct_device_chain_map:127] [CALLER:ct_device_chain_map|127]
   └─ retain self._graph = graph for later lazy insertion [LINK:ct_device_chain_map:129] [CALLER:ct_device_chain_map|129]
```

~~~text
Immediately after these lines:
_weights          = {}
_has_chain_input  = False
device_chains     = {}        ← next constructor step seeds the host chain
~~~

**Weights example:** add-one needs no model weights, so its registry stays empty. A different graph could register a symbolic weight named <code>projection.weight</code>. [Graph.add_weight](sources/max/python/max/graph/graph.py#L1619) checks the name, records a <code>mo.constant_external</code> declaration, and stores <code>_GraphWeight(weight, weight_tensor)</code> under that name. Reusing the same weight object returns an existing declaration; a different object with the same name raises an error. This Python map describes graph weights; it is not the runtime <code>WeightsRegistry</code> populated during model initialization.

**Lazy chains example (after the next step has seeded CPU):**

~~~text
Stored map:                  {cpu:0 → CH0}
Read device_chains[gpu:0]:    missing key → _DeviceChainMap.__getitem__()
Normal add-one graph:        _has_chain_input=False
                             → reuse current CPU chain as the seed
Stored map becomes:          {cpu:0 → CH0, gpu:0 → CH0}
~~~

[The lazy lookup](sources/max/python/max/graph/graph.py#L135) registers a missing device on access. In a graph with incoming chains at its top-level body, it can instead add a chain block argument. Merely constructing the map creates no GPU chain, thread, queue, lock or GPU allocation. The map holds symbolic ordering values for later graph operations.

### Ordering chain · Create the always-ready seed

**SOURCE · Create an initially ready IR chain with no input dependencies.** A chain is an internal value used to express ordering of side effects. It is not tensor data, a mutex or a semaphore.

```text
Graph.__init__: create always-ready chain [LINK:graph_init:886]
├─ _add_op_generated(ChainCreateOp, result=ChainType(), inputs=[], attach_profile_scopes=False) [LINK:graph_init:887]
│  ├─ builder at self._current_block.end → insert and verify the op [LINK:ct_generated_op:1258] [CALLER:ct_generated_op|1258]
│  └─ return wrappers for op.results; constructor selects result [0] [LINK:ct_generated_op:1277] [CALLER:graph_init|886]
├─ _ChainValue(result[0]) → self._always_ready_chain = CH0 [LINK:graph_init:886] [CALLER:graph_init|886]
└─ _update_chain(CH0) → device_chains[DeviceRef.CPU()] = CH0 [LINK:ct_update_chain:1064] [CALLER:graph_init|894]
```

~~~text
IR dependencies (schematic):
[] ──→ mo.chain.create ──→ CH0 : !mo.chain
                          ├─ _always_ready_chain retains CH0
                          └─ device_chains[cpu:0] initially points to CH0

Later side-effect ordering example:
CH0 → side effect A → CH1 → side effect B → CH2
      device_chains[cpu:0] can advance to the latest chain
      _always_ready_chain remains CH0
~~~

**Why ready?** [ChainCreateOp's definition](sources/GraphCompiler/include/GraphCompiler/MODialect/MOOps.td#L4804) says its result depends on all input values. Here <code>inputs=[]</code>, so this initial chain has no producer dependencies to wait for. Graph construction records that property; it does not perform the side effects now.

**Why keep CH0 separately?** The [constructor comment](sources/max/python/max/graph/graph.py#L883) says it is never advanced by the graph and is used for operations safe to schedule without per-device ordering constraints, such as host-to-device staging transfers. The mutable host-chain map can advance independently. An always-ready dependency does not make a transfer itself complete or guarantee hardware timing.

**General ChainCreate example:** with <code>inputs=[CH1, tensor_result]</code>, the produced chain depends on both producers. This expresses a join of IR dependencies. The constructor's empty-input version is the initial seed. <code>attach_profile_scopes=False</code> avoids attaching an active profiling scope to this bookkeeping operation.

### Chain input · Recognize and adopt an incoming dependency

**SOURCE · If the final declared input is a chain, adopt it as the host ordering input.** The constructor checks the last block argument's type; it does not add a trailing chain to add-one.

```text
Graph.__init__: examine graph-body arguments [LINK:graph_init:896]
├─ no arguments? skip the check [LINK:graph_init:896] [CALLER:graph_init|896]
├─ wrap self._graph_body.arguments[-1] as a native Value [LINK:graph_init:897] [CALLER:graph_init|897]
├─ _is_chain_value(value) → isinstance(value.type, mo.ChainType) [LINK:ct_is_chain:119] [CALLER:graph_init|900]
└─ if True: _has_chain_input=True; convert with _ChainValue.from_mlir and _update_chain [LINK:graph_init:901] [CALLER:graph_init|901]
```

~~~text
Actual add-one:
body arguments = [%x : float32[5,10]]
last argument  = %x → tensor type → not a chain
_has_chain_input stays False
device_chains[cpu:0] stays CH0 (the new empty-input chain)

Subgraph-style example:
body arguments = [%x : float32[5,10], %incoming : !mo.chain]
last argument  = %incoming → chain type → adopt it
_has_chain_input becomes True
device_chains[cpu:0] = %incoming
_always_ready_chain still holds CH0
~~~

**Where can an incoming chain come from?** [Graph.add_subgraph](sources/max/python/max/graph/graph.py#L1027) constructs the child with <code>input_types=[*input_types, _ChainType()]</code>. The extra block argument lets the child carry an ordering dependency from its caller. The child graph's construction recognizes that chain; construction does not execute the caller or child.

**Why separate it from tensor inputs?** [Graph.inputs](sources/max/python/max/graph/graph.py#L943) excludes internal trailing chain values before exposing user inputs to the forward callback. In the simple child example, the lambda gets symbolic <code>x</code>, not the extra chain. Per-device chain bookkeeping can extend this pattern; the host chain is seeded from the explicit trailing chain instead of adding a duplicate host-chain argument.

### with self · Activate graph construction

**SOURCE · `with self` enters Python's context-manager protocol and makes G1 the active graph.** It does not execute a compiled model. [Constructor call](sources/max/python/max/graph/graph.py#L921).

```text
with self [LINK:graph_init:921]
├─ Graph.__enter__ → push self._enter() context manager and enter it [LINK:ct_graph_context_manager:1135] [CALLER:graph_init|921]
└─ Graph._enter() [LINK:ct_graph_enter:1142]
   ├─ ensure_default_mlir_context() → make MLIR construction context available on this thread [LINK:ct_graph_enter:1144] [CALLER:ct_graph_enter|1144]
   ├─ token = CURRENT_GRAPH.set(self) → active graph is G1 [LINK:ct_graph_enter:1145] [CALLER:ct_graph_enter|1145]
   ├─ yield self → constructor runs its forward-building body [LINK:ct_graph_enter:1147] [CALLER:ct_graph_enter|1147]
   └─ finally CURRENT_GRAPH.reset(token) → restore earlier graph context [LINK:ct_graph_enter:1149] [CALLER:ct_graph_enter|1149]
```

~~~text
Before with G1:   CURRENT_GRAPH = previous graph or unset
Inside with G1:   CURRENT_GRAPH = G1 → ops.custom finds G1
After with G1:    previous graph or unset is restored
~~~

`Graph.current` is a Python class property backed by a ContextVar, not a newly constructed graph. [Its getter](sources/max/python/max/graph/graph.py#L1215) reads `CURRENT_GRAPH`; an unset value raises `LookupError("No graph found")`. The active graph selects the destination for operations. The MLIR Context instead owns/types IR objects. Nested graph contexts restore the previous graph when they exit, including exceptional exits.

### self.inputs · Block argument → symbolic TensorValue

**SOURCE · Wrap the graph's declared block arguments as Python symbolic values.** No matrix data is supplied at this step.

```text
forward(*self.inputs, *args, **kwargs) [LINK:graph_init:922]
└─ cached Graph.inputs property [LINK:ct_graph_inputs:935]
   ├─ body_args = self._graph_body.arguments [LINK:ct_graph_inputs:942] [CALLER:ct_graph_inputs|942]
   ├─ omit internal trailing chain values when the graph has chain inputs [LINK:ct_graph_inputs:943] [CALLER:ct_graph_inputs|943]
   └─ Value.from_mlir(_Value._from_cmlir(arg)) for each argument → tuple of symbolic values [LINK:ct_graph_inputs:949] [CALLER:ct_graph_inputs|949]
```

~~~text
G1 body:  %arg0 : !mo.tensor<[5,10], f32>
                         ↓ native value wrapped by Python Value.from_mlir
self.inputs = (TensorValue referring to %arg0,)
                         ↓ tuple expanded as forward(*self.inputs)
lambda parameter x refers to that same symbolic input
~~~

**Example:** `x.dtype` is float32 and `x.tensor.shape` describes `[5,10]`. There are still no 50 matrix values in `x`. The tuple has one element because add-one declares one user input; two declared tensor inputs would give two symbolic arguments to a two-parameter callback. The `cached_property` retains the result of this input-wrapper lookup.

### forward(x) · Execute Python to record IR

**SOURCE · The constructor calls the Python lambda immediately with symbolic x.** This executes graph-building Python, not the add-one Mojo kernel. [Lambda](sources/max/examples/custom_ops/addition.py#L37), [call](sources/max/python/max/graph/graph.py#L922).

```text
Graph.__init__ → result = forward(*self.inputs, *args, **kwargs) [LINK:graph_init:922]
└─ addition.py forward=lambda x: (...) [LINK:entry:37]
   ├─ prepare ops.custom arguments: name, device, values=[x] [LINK:entry:39] [CALLER:entry|38]
   ├─ TensorType(dtype=x.dtype, shape=x.tensor.shape, device=DeviceRef.from_device(device)) [LINK:entry:43] [CALLER:entry|43]
   └─ call ops.custom(...)[0].tensor → one symbolic result returned to constructor [LINK:entry:49] [CALLER:entry|38]
```

~~~text
INPUT x: symbolic float32[5,10], selected device
  → output Python TensorType: float32[5,10], same selected device
  → out_types=[that type] passed to ops.custom
  → type.to_mlir(): native MO result type
  → future result %r : float32[5,10]
~~~

The output TensorType is a declaration of the result contract. Copying x's shape/dtype does not allocate an output tensor or add 1 to its elements. `ops.custom` uses the declared types to construct a symbolic call. The registered Mojo implementation becomes executable through later compilation and initialization; real data enters through model execution.


**Concrete CPU example · values available while Python records G1**

| Expression | Concrete description | Comes from |
| --- | --- | --- |
| `device` | `CPU()`; symbolic reference `cpu:0` | Example device selection |
| `x` | Python TensorValue X1 referring to native block argument `%arg0` | `self.inputs[0]` |
| `x.dtype` | `DType.float32` | Input type supplied to Graph |
| `x.tensor.shape` | Shape containing static dimensions 5 and 10 | Input shape `[rows, columns]` |
| `DeviceRef.from_device(device)` | CPU reference, id 0 | Same selected device |
| `T_out` | `TensorType(DType.float32, [5,10], device=cpu_ref)` | Output declaration in the lambda |
| `out_types` | `[T_out]` | One expected output |

```python
# Expanded form of the original lambda, for explanation.
# x is a symbolic TensorValue supplied by Graph.__init__.
cpu_ref = DeviceRef.from_device(device)
T_out = TensorType(dtype=x.dtype, shape=x.tensor.shape, device=cpu_ref)
outputs = ops.custom(name="add_one", device=cpu_ref,
                     values=[x], out_types=[T_out])
y = outputs[0].tensor
```

```text
forward(X1) [LINK:graph_init:922]
├─ x.dtype → float32; x.tensor.shape → static [5,10] [LINK:entry:44] [CALLER:entry|43]
├─ DeviceRef.from_device(CPU()) → cpu reference 0 [LINK:entry:46] [CALLER:entry|46]
├─ TensorType(...) → T_out; declares type, allocates no output buffer [LINK:entry:43] [CALLER:entry|43]
├─ ops.custom(..., out_types=[T_out]) → [Y1 referring to %r] [LINK:entry:38] [CALLER:entry|38]
└─ outputs[0].tensor → Y1 → constructor's result [LINK:entry:49] [CALLER:graph_init|922]
```

For a selected accelerator, the same shape/dtype is declared with a GPU device reference (for example `gpu:0`). Neither case puts actual matrix elements in X1 or Y1. Later `model.execute` can receive, for example, a matrix whose first row starts `[0.0, 1.0, 2.0, ...]`; its output then starts `[1.0, 2.0, 3.0, ...]`. Those numbers belong to execution, not this forward callback.

### ops.custom · Name, device and output attributes

**SOURCE · Convert the Python call's name and device into attributes, and its output descriptions into native types.** [ops.custom](sources/max/python/max/graph/ops/custom.py#L69).

```text
ops.custom("add_one", device, values=[x], out_types=[T]) [LINK:custom_api:69]
├─ graph = Graph.current; context = default_mlir_context() [LINK:custom_api:94] [CALLER:custom_api|94]
├─ symbol_attr = StringAttr.get(name, context) → "add_one" [LINK:custom_api:96] [CALLER:custom_api|96]
├─ device = DeviceRef.from_device(device) [LINK:custom_api:97] [CALLER:custom_api|97]
├─ reject buffers/opaque values for this functional custom-op path; normalize inputs [LINK:custom_api:99] [CALLER:custom_api|99]
├─ [t.to_mlir() for t in out_types] → declared native result types [LINK:custom_api:110] [CALLER:custom_api|109]
└─ after insertion: custom_op.device = bridge(device.to_mlir()) [LINK:custom_api:122] [CALLER:custom_api|122]
```

~~~text
Python name "add_one"      → MLIR StringAttr → custom op symbol
Python selected device    → DeviceRef      → DeviceRefAttr → custom op device
Python out_types=[T]      → [T.to_mlir()]   → one native MO result type
Python parameters=None   → no extra kernel-parameter dictionary in add-one
~~~

The string names the kernel registered by the Mojo decorator; it does not embed the Python lambda or Mojo function body in the MO call. The device attribute is assigned after operation insertion and before kernel verification. Optional `parameters` are attached only when supplied; this example supplies none.


**Concrete attributes · CPU and GPU examples**

```text
name = "add_one" [LINK:entry:39]
└─ StringAttr.get("add_one", context) [LINK:custom_api:96]
   └─ MLIR string attribute whose value is add_one → symbol field of CUST1 [LINK:custom_api:110] [CALLER:custom_api|96]

device = DeviceRef.from_device(CPU()) [LINK:entry:40]
└─ DeviceRef.from_device(...) → DeviceRef(kind=cpu, id=0) [LINK:custom_api:97]
   └─ DeviceRef.to_mlir() → m.DeviceRefAttr("cpu", 0) [LINK:ct_device_ref_to_mlir:183]
      └─ attribute C API bridge → custom_op.device [LINK:custom_api:122] [CALLER:custom_api|122]

GPU illustration: DeviceRef(kind=gpu, id=0) [LINK:ct_device_ref_to_mlir:183]
└─ m.DeviceRefAttr("gpu", 0) → GPU-labelled call [LINK:ct_device_ref_to_mlir:183] [CALLER:custom_api|122]

out_types = [T_out: float32[5,10] on cpu:0] [LINK:entry:42]
└─ T_out.to_mlir() → native MO tensor type with shape/dtype/device [LINK:ct_tensor_to_mlir:430]
   └─ result_types=[that native type] → one native SSA result %r [LINK:custom_api:110] [CALLER:custom_api|109]
```

The device examples above show constructor arguments and attribute contents, not an asserted textual MLIR printer syntax. A DeviceRefAttr identifies a device kind and index; it does not contain the driver, stream, GPU memory, or compiled kernel. The registration string selects P1's kernel declaration; device metadata guides subsequent compatibility checks.

### mo.custom · Insert the call into G1

**SOURCE · Insert a new MO call operation into the active graph body, preserving the input value reference.** Python x remains a wrapper for the existing block argument.

```text
ops.custom → graph._add_op_get_op_with_results(mo.custom, result_types, values, symbol=...) [LINK:custom_api:109]
└─ Graph._add_op_get_op_with_results [LINK:ct_insert_result:1286]
   ├─ unwrap Python Value/Type/Attribute handles to MLIR binding handles [LINK:ct_insert_result:1297] [CALLER:ct_insert_result|1297]
   ├─ choose insertion point _ip or mlir.InsertionPoint(self._body) [LINK:ct_insert_result:1312] [CALLER:ct_insert_result|1312]
   ├─ enter insertion point + _location(); call mo.custom with converted arguments [LINK:ct_insert_result:1320] [CALLER:ct_insert_result|1320]
   ├─ find newly inserted operation and verify its IR structure [LINK:ct_insert_result:1328] [CALLER:ct_insert_result|1328]
   └─ return symbolic result wrappers and staged_op [LINK:ct_insert_result:1364] [CALLER:ct_insert_result|1364]
```

~~~text
Before: G1 body has %arg0 : float32[5,10] and chain seed CH0
             ↓ insert mo.custom using %arg0 as its operand
After:  G1 body retains %arg0 and CH0
        + custom call CUST1:
            symbol = "add_one"
            operand = %arg0
            result = %r : float32[5,10]
~~~

This is an operation-structure illustration, not an exact assembly dump. The input reference is connected to a new result-producing operation; the input data is not copied or computed. The generic builder verifies the inserted operation's IR structure before `ops.custom` finishes device/parameter attachment and explicitly verifies it against the loaded kernel library.


**Complete add-one call · source order and concrete arguments**

```text
ops.custom(name="add_one", device=cpu_ref, values=[X1], out_types=[T_out]) [LINK:entry:38]
├─ Graph.current → active G1; default_mlir_context() → IR context [LINK:custom_api:94] [CALLER:custom_api|94]
├─ symbol_attr = StringAttr.get("add_one", context) [LINK:custom_api:96] [CALLER:custom_api|96]
├─ normalize selected device → CPU DeviceRef, id=0 [LINK:custom_api:97] [CALLER:custom_api|97]
├─ reject Buffer/opaque arguments; normalize tensor-like inputs [LINK:custom_api:99] [CALLER:custom_api|99]
├─ convert T_out → native tensor type float32[5,10] on cpu:0 [LINK:custom_api:110] [CALLER:custom_api|109]
├─ G1._add_op_get_op_with_results(mo.custom, [T_native], [X1], symbol=symbol_attr) [LINK:custom_api:109]
│  ├─ recursively unwrap [X1] → [mlir.Value referring to %arg0] [LINK:ct_insert_result:1299] [CALLER:ct_insert_result|1309]
│  ├─ bridge native type/attribute handles; retain symbol="add_one" [LINK:ct_insert_result:1303] [CALLER:ct_insert_result|1310]
│  ├─ ip = mlir.InsertionPoint(G1._body) [LINK:ct_insert_result:1312] [CALLER:ct_insert_result|1312]
│  ├─ with ip, _location(...): call mo.custom([T_native], [%arg0], symbol=...) [LINK:ct_insert_result:1317] [CALLER:ct_insert_result|1320]
│  ├─ find inserted CUST1; staged_op = staged_op.opview [LINK:ct_insert_result:1323] [CALLER:ct_insert_result|1328]
│  ├─ G1._verify_op(staged_op) → structural IR verification [LINK:ct_insert_result:1329] [CALLER:ct_insert_result|1329]
│  ├─ attach output parameter declarations; wrap native %r as Y1 [LINK:ct_insert_result:1351] [CALLER:ct_insert_result|1360]
│  └─ return ([Y1], CUST1) to ops.custom [LINK:ct_insert_result:1364] [CALLER:custom_api|109]
├─ parameters=None → skip optional parameter dictionary assignment [LINK:custom_api:113] [CALLER:custom_api|113]
├─ attach device attribute m.DeviceRefAttr("cpu",0) to CUST1 [LINK:custom_api:122] [CALLER:custom_api|122]
├─ verify CUST1 against the loaded P1 kernel library [LINK:custom_api:127] [CARD:toolchain-call-graph-custom-verify|Kernel validation · P1 and MO] [CALLER:custom_api|127]
└─ return [Y1] → caller selects [0].tensor [LINK:custom_api:129] [CALLER:entry|49]
```

**Graph body before / after · schematic IR structure, not an exact compiler dump**

~~~text
T_cpu = native MO tensor type: float32, shape=[5,10], device=cpu:0

BEFORE custom call
builtin.module MODULE1
└─ mo.graph @addition : (T_cpu) → ()       output not finished yet
   └─ region R0 / block B0
      ├─ block argument %arg0 : T_cpu      X1 refers here
      └─ CH0: initial ordering-chain operation

AFTER insertion + device attachment + validation
builtin.module MODULE1
└─ mo.graph @addition : (T_cpu) → ()       output still not finished
   └─ region R0 / block B0
      ├─ block argument %arg0 : T_cpu
      ├─ CH0: initial ordering-chain operation
      └─ CUST1: mo.custom
         ├─ symbol attribute: "add_one"
         ├─ device attribute: kind=cpu, id=0
         ├─ operand 0: %arg0 : T_cpu
         └─ result 0: %r : T_cpu           Y1 refers here

AFTER Graph.output(Y1)                    later constructor continuation
mo.graph @addition : (T_cpu) → (T_cpu)
└─ same B0: CH0, CUST1, then mo.output(%r)
~~~

At insertion, the generic helper verifies the operation's IR structure. The call's device attachment and P1 kernel-contract verification follow in `ops.custom`. P1 remains a separate implementation package throughout; the body here contains a symbolic call, not `x.load + 1` or the Mojo source.


**Concrete attributes · CPU and GPU examples**

```text
name = "add_one" [LINK:entry:39]
└─ StringAttr.get("add_one", context) [LINK:custom_api:96]
   └─ MLIR string attribute whose value is add_one → symbol field of CUST1 [LINK:custom_api:110] [CALLER:custom_api|96]

device = DeviceRef.from_device(CPU()) [LINK:entry:40]
└─ DeviceRef.from_device(...) → DeviceRef(kind=cpu, id=0) [LINK:custom_api:97]
   └─ DeviceRef.to_mlir() → m.DeviceRefAttr("cpu", 0) [LINK:ct_device_ref_to_mlir:183]
      └─ attribute C API bridge → custom_op.device [LINK:custom_api:122] [CALLER:custom_api|122]

GPU illustration: DeviceRef(kind=gpu, id=0) [LINK:ct_device_ref_to_mlir:183]
└─ m.DeviceRefAttr("gpu", 0) → GPU-labelled call [LINK:ct_device_ref_to_mlir:183] [CALLER:custom_api|122]

out_types = [T_out: float32[5,10] on cpu:0] [LINK:entry:42]
└─ T_out.to_mlir() → native MO tensor type with shape/dtype/device [LINK:ct_tensor_to_mlir:430]
   └─ result_types=[that native type] → one native SSA result %r [LINK:custom_api:110] [CALLER:custom_api|109]
```

The device examples above show constructor arguments and attribute contents, not an asserted textual MLIR printer syntax. A DeviceRefAttr identifies a device kind and index; it does not contain the driver, stream, GPU memory, or compiled kernel. The registration string selects P1's kernel declaration; device metadata guides subsequent compatibility checks.

### Kernel validation · MO call ↔ Mojo package P1

**SOURCE · Check the symbolic MO call against the kernel definitions loaded from P1.** The existing Mojo package supplies the registered implementation contract.

```text
ops.custom → graph._kernel_library.verify_custom_op(custom_op) [LINK:custom_api:127]
└─ KernelLibrary.verify_custom_op [LINK:ct_graph_verify:307]
   ├─ library paths → frozenset cache key [LINK:ct_graph_verify:315] [CALLER:ct_graph_verify|315]
   ├─ cached Analysis exists? reuse it and verify this operation [LINK:ct_graph_verify:319] [CALLER:ct_graph_verify|319]
   └─ otherwise verify with current Analysis; cache it when paths exist [LINK:ct_graph_verify:322] [CALLER:ct_graph_verify|322]
```

~~~text
P1 (.mojoc): @extensibility.register("add_one") → AddOne.execute
                                           ↘ registered contract
MO call CUST1: "add_one", operand %x, result type T, device
                                           ↗ validation
valid symbolic call → Python result wrappers may be returned
invalid contract    → exception; caller does not receive a successful result
~~~

[Native Analysis::verifyCustomOp](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L69) bridges the operation to compiler analysis. P1 remains an implementation package; validation does not inline its Mojo body as MO arithmetic or run the kernel. Add-one supplies output types explicitly; its optional unimplemented shape function is not needed to infer these declared outputs.

### Result wrappers · %r → [TensorValue] → lambda result

**SOURCE · Return a Python list of wrappers for the custom operation's symbolic results.** Add-one has one declared result type, so the list contains one result.

```text
Graph._add_op_get_op_with_results [LINK:ct_insert_result:1356]
├─ native result Value → Value.from_mlir(_Value._from_cmlir(result)) [LINK:ct_insert_result:1358] [CALLER:ct_insert_result|1358]
└─ return results, staged_op → ops.custom returns results [LINK:custom_api:129] [CALLER:ct_insert_result|1364]
   └─ addition lambda selects results[0].tensor [LINK:entry:49] [CALLER:entry|38]
```

~~~text
MO result %r : float32[5,10]
    ↑ referenced by Python TensorValue wrapper
results = [wrapper(%r)]
    → results[0]          selects that wrapper
    → results[0].tensor   exposes the TensorValue interface
    → constructor receives symbolic result, not 50 computed numbers
~~~

The return is a list because `out_types` can declare several results; each result has its own native value and Python wrapper. `.tensor` does not execute or transfer data. `Graph.output(result)` records which symbolic result will be returned when the compiled model is eventually run.


**Complete returned values · named wrappers and native referents**

~~~text
X1 = Python TensorValue wrapper
     native referent = block argument %arg0
     dtype = DType.float32
     shape = static [5,10]
     device = cpu:0

CUST1 = native mo.custom operation owned by G1 body B0
        symbol="add_one", operands=[%arg0], results=[%r]

Y1 = Python TensorValue wrapper
     native referent = CUST1 result %r
     dtype = DType.float32
     shape = static [5,10]
     device = cpu:0

helper return:      ([Y1], staged_op=CUST1)
ops.custom return:  [Y1]                 one-element Python list
...[0]:            Y1                   selects the first wrapper
...[0].tensor:     Y1 tensor interface   refers to the same %r
forward return:    Y1
Graph.output(Y1):  record mo.output(%r), finish output signature
~~~

The X1/Y1 labels are teaching names, not actual Python `repr` output or pointer addresses. The list contains a symbolic TensorValue, not a Buffer, NumPy array, or 50 calculated numbers. Its declared shape has 50 elements, but their values become available only through later execution. A native result is identified by its producing operation and result index; the wrapper preserves that identity.

```text
native result %r → _Value._from_cmlir(result) [LINK:ct_insert_result:1360]
└─ Value.from_mlir(...) → tensor-valued wrapper Y1 [LINK:ct_insert_result:1360] [CALLER:ct_insert_result|1360]
   ├─ helper returns results=[Y1] and staged_op=CUST1 [LINK:ct_insert_result:1364] [CALLER:ct_insert_result|1364]
   ├─ ops.custom verifies CUST1, then returns only results=[Y1] [LINK:custom_api:127] [CALLER:custom_api|129]
   └─ original lambda selects [0].tensor → constructor receives Y1 [LINK:entry:49] [CALLER:graph_init|922]
```

### Finish forward · Output contract and context exit

**SOURCE · Finish the symbolic graph and restore the previous active graph.** [Constructor continuation](sources/max/python/max/graph/graph.py#L932).

```text
Graph.__init__ → Graph.output(result) [LINK:graph_init:932]
├─ flatten result structure and collect native output values/types [LINK:graph_output:1433] [CALLER:graph_output|1433]
├─ construct function type (body input types) → (output types) and generator signature [LINK:graph_output:1458] [CALLER:graph_output|1458]
├─ insert mo.OutputOp, record argument/result names and verify completed G1 [LINK:graph_output:1467] [CALLER:graph_output|1467]
└─ leave with self → __exit__ → CURRENT_GRAPH.reset(token) [LINK:ct_graph_enter:1149] [CALLER:ct_graph_context_manager|1138]
```

~~~text
Shell:     (%x : T) → ()
Forward:   %x → CUST1("add_one") → %r : T
Output:    signature becomes (T) → (T); mo.output returns %r
Exit:      previous Graph.current restored; populated G1 retained
Next:      session.compile(G1), then init, then execute with actual data
~~~

The output operation terminates the graph body and establishes its result contract. Verification checks the completed symbolic IR; no matrix arithmetic has happened during this recording sequence.

### A.2 · ModuleOp + GraphOp → record graph G1

**`builtin.ModuleOp` is the outer MLIR container; `mo.GraphOp("addition")` is the named computation inside it.** Python’s `Graph(...)` constructor creates and retains both. See the [construction code](sources/max/python/max/graph/graph.py#L851).

**Source order inside this constructor:**

```text
@in_default_mlir_context
→ create empty builtin.ModuleOp + mo.GraphOp scaffold
→ import kernels/; prepare P1 (.mojoc)
→ forward(symbolic X); record mo.custom("add_one")
→ Graph.output(Y); return completed graph G1
```

The reader opens with the **package-preparation phase A.1**, then follows the completed **MO graph G1 in A.2/A.3**. The empty scaffold precedes package loading in the source; the custom operation is recorded after loading. See [kernel import](sources/max/python/max/graph/graph.py#L909) and [forward recording](sources/max/python/max/graph/graph.py#L918).

`Op` means **operation**. `builtin` is MLIR’s builtin dialect; `mo` is MAX’s graph dialect, defined in [MODialect.td](sources/GraphCompiler/include/GraphCompiler/MODialect/MODialect.td#L25). ([MLIR builtin dialect](https://mlir.llvm.org/docs/Dialects/Builtin/))

**1. `builtin.ModuleOp`: the container**

It represents an MLIR operation normally printed as `module { ... }`. Its body contains other operations—including your graph definition. MLIR organizes this nesting as **operation → region → block → operations**. ([MLIR operation structure](https://mlir.llvm.org/docs/LangRef/))

A **module is a compilation unit that groups definitions**. MAX’s Python `Module` wrapper owns the enclosing `builtin.module`; each `Graph` contributes a named computation inside it. The module’s body gives the compiler one place to find and process those definitions. See [Module’s documented role](sources/max/python/max/graph/graph.py#L562).

```text
Graph.__init__(..., module=maybe_module) [LINK:graph_init:851]
├─ caller supplied a Module wrapper
│  └─ reuse module.mlir_module → self._module [LINK:graph_init:855]
│     └─ Module exposes/stores the typed builtin.ModuleOp [LINK:ct_graph_module:637]
└─ no module supplied
   └─ builtin.ModuleOp(location=loc) [LINK:graph_init:856]
      └─ nanobind ModuleOp constructor [LINK:ct_builtin_module:62]
         └─ mlir::ModuleOp::create(loc, name=None) [LINK:ct_builtin_module:65]
            └─ builtin.module → body region → body block
               └─ initially empty; GraphOp will be inserted here
```

The native constructor and `.body` binding are in [builtin.cpp](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62).

When you supply `Graph(..., module=existing_module)`, MAX reuses that module. This allows several named graphs to share one module—for example, encoder and decoder graphs. See [the Python `Module` wrapper](sources/max/python/max/graph/graph.py#L562).

**For the addition example · immediately after GraphOp insertion**

At this instant, GraphOp has one typed input and no result types yet (<code>result_types=[]</code>). It has created the graph region and its entry block; <code>%arg0</code> is the symbolic input X, not a tensor buffer containing numbers. This is a temporary construction snapshot before the chain, custom operation, and <code>mo.output</code> are added. Location annotations are omitted from this schematic:

~~~mlir
module {
  mo.graph @addition(%arg0: !mo.tensor<[5, 10], f32>) -> () {
    // Entry block contains %arg0; no computation or mo.output yet.
  }
}
~~~

**Structure:** the outer module body contains the <code>mo.graph @addition</code> operation; that graph owns one region and one block, whose argument list contains <code>%arg0</code>. The selected-device mapping is attached separately immediately afterward by <code>Graph._populate_device_info_mapping()</code> ([graph.py](sources/max/python/max/graph/graph.py#L877)).

**1.1 Example · two graphs in one module**

**Illustrative Python using the repository’s APIs.** Both graphs take two float32 vectors of length four:

```python
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, ops

device = CPU()
t = TensorType(DType.float32, shape=[4],
               device=DeviceRef.from_device(device))
unit = Module()

g_add = Graph("add", module=unit, input_types=[t, t],
              forward=lambda a, b: ops.add(a, b))
g_mul = Graph("mul", module=unit, input_types=[t, t],
              forward=lambda a, b: ops.mul(a, b))
```

`unit` owns one IR module containing **two graph definitions**. `g_add` and `g_mul` retain their respective graph operations. Calls to `ops.add` and `ops.mul` first use **RMO builders** for relaxed shape handling; RMO later lowers to strict MO. See [add’s builder](sources/max/python/max/graph/ops/elementwise.py#L73), [mul’s builder](sources/max/python/max/graph/ops/elementwise.py#L353), and [the RMO dialect](sources/GraphCompiler/include/GraphCompiler/RMODialect/RMODialect.td#L27).

**Recorded module structure before compilation:**

```text
unit.mlir_module → builtin.module
├─ g_add → mo.graph @add → relaxed rmo.add computation
└─ g_mul → mo.graph @mul → relaxed rmo.mul computation
```

The Python calls build symbolic definitions. **C.1** follows their compilation through strict MO and MAP; **D.1** initializes the resulting named models.

**2. `mo.GraphOp("addition")`: the computation definition**

Here, **`"addition"` names the graph**. The operation that performs your computation is separately referenced as **`"add_one"`** in [addition.py](sources/max/examples/custom_ops/addition.py#L33).

The actual constructor takes more than the name:

```python
op = _mo.GraphOp(
    builder,
    loc,
    name="addition",
    input_types=[t.to_mlir() for t in input_types],
    result_types=[],
)
```

Its native construction expands as follows:

```text
Graph.__init__ [LINK:graph_init:851]
└─ builder = OpBuilder(self._module.body.end) [LINK:graph_init:858]
   ├─ evaluate self._module.body
   │  └─ ModuleOp.body binding returns the module body block [LINK:ct_builtin_module:68]
   ├─ evaluate .end
   │  └─ choose an insertion point after the block's current top-level operations
   ├─ construct the MLIR Python OpBuilder at that insertion point
   │  └─ underlying native mlir::OpBuilder is positioned in this module block
   └─ pass builder to _mo.GraphOp(builder, loc, ...) [LINK:graph_init:860]
      └─ C++ GraphOp::build(OpBuilder &, OperationState, ...) [LINK:ct_graph_op_build:820]
         ├─ sym_name = addition → identifies this graph in the module
         ├─ function type → input float32[5,10]; outputs initially empty
         ├─ KGEN signature + symbolic-dimension parameter attributes
         └─ create body region → entry block → typed symbolic input X
```

The builder positions the new graph at the end of the module body, so adding a graph to a shared module leaves earlier top-level graphs in place. The MLIR Python binding supplies the OpBuilder and block insertion-point API; the MAX call site is [graph.py:858](sources/max/python/max/graph/graph.py#L858), and the module-body binding is in [builtin.cpp:68–70](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L68).

These steps come directly from [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820). The operation’s definition requires a module parent, a single graph-body block, and a terminating `mo.output`; see [MOOps.td](sources/GraphCompiler/include/GraphCompiler/MODialect/MOOps.td#L211).

**3. How the initially empty graph becomes your addition graph**

The constructor calls `forward(...)` with symbolic inputs. Your `ops.custom(name="add_one", ...)` records the computation and produces a symbolic output. Then `Graph.output(...)` establishes the output signature and inserts `mo.output`. See [forward construction](sources/max/python/max/graph/graph.py#L918) and [output finalization](sources/max/python/max/graph/graph.py#L1458).

The resulting structure is **schematic**, with internal chain and metadata operations omitted:

```text
Python Graph object
│
├─ _module → builtin.module
│  └─ body region / block
│     └─ mo.graph @addition
│        │
│        ├─ input signature:  float32[5,10]
│        ├─ output signature: float32[5,10]
│        │
│        └─ body region / block
│           ├─ X = symbolic input
│           ├─ custom computation "add_one"(X) → symbolic Y
│           └─ mo.output(Y)
│
└─ _mlir_op → another wrapper around that same mo.graph
```

The `_mlir_op` assignment wraps the existing operation through its C API pointer; it does not create another graph. See [graph.py](sources/max/python/max/graph/graph.py#L874).

**3.1 Your addition example · Python call → MO operation**

**Equivalent explicit construction** for `addition.py`: `Y_type` is its declared float32 `[5,10]` tensor type and `mojo_kernels` is its `kernels/` path. The original example uses a `forward` callback to record the same custom operation:

```python
with Graph("addition", input_types=[Y_type],
           custom_extensions=[mojo_kernels]) as graph:
    X = graph.inputs[0]
    y = ops.custom(name="add_one",
                   device=DeviceRef.from_device(device),
                   values=[X], out_types=[Y_type])[0].tensor
    graph.output(y)
```

**Schematic CPU MO view** of the recorded computation, with extra metadata omitted:

```mlir
module {
  mo.graph @addition(%x: !mo.tensor<[5, 10], f32>)
      -> !mo.tensor<[5, 10], f32> {
    %y = mo.custom [symbol = "add_one",
                   device = #M.device_ref<"cpu", 0>](%x)
        : (!mo.tensor<[5, 10], f32>)
          -> !mo.tensor<[5, 10], f32>
    mo.output %y : !mo.tensor<[5, 10], f32>
  }
}
```

| Python construction | Change in the IR |
|---|---|
| `Graph("addition", input_types=[...])` | Insert `mo.graph @addition`; create symbolic `%x`. |
| `ops.custom(name="add_one", ...)` | Insert `mo.custom`; attach kernel name and device; create symbolic `%y`. |
| `graph.output(y)` | Set the graph’s result signature and insert `mo.output %y`. |

The **Python builder runs to create IR nodes**; the kernel’s numerical work happens later. This does not translate the entire `addition.py` program into MLIR. See [custom-op construction](sources/max/python/max/graph/ops/custom.py#L114), [Graph.output](sources/max/python/max/graph/graph.py#L1458), and [the original example](sources/max/examples/custom_ops/addition.py#L33).

**Actual numbers arrive later**, through `model.execute(x)`. For example, an input element `2.5` produces `3.5` in this example. Compilation, initialization, buffer creation, and execution occur after graph construction in [addition.py](sources/max/examples/custom_ops/addition.py#L67).

The prepared **P1 `.mojoc` package** supplies kernel implementation bodies. **G1** contains the symbolic MO graph. In **C.1**, compilation rewrites a clone **C1** through MO → MAP; G1 remains with the caller. See [package loading](sources/max/python/max/graph/graph.py#L269).

### A.3 · Custom-op registration → verified graph G1

```text
ops.custom("add_one", values=[X], out_types=[Y_type], device) [max/python/max/graph/ops/custom.py:L69]
├─ explicit out_types → output dtype/shape/device is already supplied
│  └─ this path bypasses AddOne's optional unimplemented shape function ↗ caller [max/python/max/graph/ops/custom.py:L69]
├─ insert mo.custom with name="add_one" → link to @extensibility.register("add_one") / AddOne [max/examples/custom_ops/kernels/add_one.mojo:L32] ↗ caller [max/python/max/graph/ops/custom.py:L69]
├─ KernelLibrary.verify_custom_op(operation) [max/python/max/graph/graph.py:L307]
│  ├─ lookup reusable native Analysis by frozenset(package_paths) ↗ caller [max/python/max/graph/graph.py:L307]
│  └─ Analysis::verifyCustomOp(operation) [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L69]
│     └─ MojoRegistry → check registered kernel, device, arguments and results ↗ caller [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L69]
├─ return symbolic custom result → callback returns its .tensor ↗ caller [max/python/max/graph/ops/custom.py:L69]
└─ Graph.output(Y) [max/python/max/graph/graph.py:L1396]
   ├─ flatten result tree; infer output types/names; write graph signature ↗ caller [max/python/max/graph/graph.py:L1396]
   ├─ record mo.output(Y, required chains) ↗ caller [max/python/max/graph/graph.py:L1396]
   └─ verify G1 → Y describes a future output; it contains no computed numbers ↗ caller [max/python/max/graph/graph.py:L1396]
```

## B · InferenceSession(devices=[device]) → S1

session = InferenceSession(devices=[device])

INPUT: existing Device handle(s); num_threads=None; session custom_extensions=None
OUTPUT: S1: Python session → native session → Engine::Context + host CPU runtime
EXAMPLE: CPU → [CPU]; GPU → [Accelerator, CPU]. G1 keeps its own package P1.

```text
InferenceSession.__init__(devices=[device]) [max/python/max/engine/api.py:L664] → S1
├─ normalize devices → deduplicate while preserving first occurrence ↗ caller [max/python/max/engine/api.py:L664]
├─ append CPU() if absent → host runtime is present for both target routes ↗ caller [max/python/max/engine/api.py:L664]
├─ optional import/export MEF policy → neither selected in addition.py ↗ caller [max/python/max/engine/api.py:L664]
├─ session extensions = []; num_threads or 0 → automatic runtime choice ↗ caller [max/python/max/engine/api.py:L664]
└─ _InferenceSession(final_devices, [], 0) [max/python/max/_core/max_license/modules/engine.cpp:L787]  [nanobind → C++]
   └─ Core::InferenceSession::InferenceSession(...) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
      ├─ populateCompileConfig([]) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L615] → session compilation settings ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
      ├─ retain Device handles; runtimeConfig.addDevice(DeviceSpec, driver context) ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
      ├─ runtimeConfig → numThreads=0; API language="python" ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
      └─ Engine::Context::create(runtimeConfig) [max/lib/max_license/API/c/EngineContext.cpp:L169]
         ├─ getOrCreateCPUDeviceOptions(config) [max/lib/max_license/API/c/EngineContext.cpp:L50] → host scheduling options ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L169]
         ├─ Init::getOrCreateContext("max-engine", options) [Init/lib/Init.cpp:L116]
         │  ├─ matching global context exists → retain it ↗ caller [Init/lib/Init.cpp:L116]
         │  └─ otherwise → createContextImpl(...) [Init/lib/Init.cpp:L40] → MAX context + CPUDevice ↗ caller [Init/lib/Init.cpp:L116]
         ├─ Engine::Context(config, maxContext) [max/lib/max_license/API/c/EngineContext.cpp:L214] → retain runtime references ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L169]
         └─ native session owns this Engine::Context → S1 ready for compile/init → B.1 ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L169]
```

### B.1 · Devices → native context and host runtime → S1

```text
Constructor input → normalized output [max/python/max/engine/api.py:L664]
├─ [CPU()]                  → [CPU()] ↗ caller [max/python/max/engine/api.py:L664]
├─ [Accelerator()]          → [Accelerator(), CPU()] ↗ caller [max/python/max/engine/api.py:L664]
└─ [device, device, CPU()]  → [device, CPU()]  (if device is an accelerator) ↗ caller [max/python/max/engine/api.py:L664]

S1 [Python InferenceSession]
└─ _impl → Core::InferenceSession [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541] [C++ object in the same host process]
   ├─ devices → retained existing driver handles / contexts ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
   ├─ sessionCompileConfig → compiler settings / optional extension paths ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
   └─ runtimeContext → Engine::Context [max/lib/max_license/API/c/EngineContext.cpp:L214] [owned native object]
      ├─ config → runtime settings / device descriptions ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L214]
      ├─ maxContext → shared MAX ContextRef [Init/lib/Init.cpp:L116]
      │  └─ creation/reuse protected by std::mutex ↗ caller [Init/lib/Init.cpp:L116]
      └─ cpuDevice → AsyncRT host runtime / worker scheduling ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L214]

Thread policy → num_threads=None becomes 0: automatic selection [max/lib/max_license/API/c/EngineContext.cpp:L50]
└─ context/runtime can be reused → do not infer a new private pool per S1 ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L50]
Constructor result → S1 configuration + ownership; G1 and its prepared P1 are supplied later to compile.
In this example, session custom_extensions=None → package preparation remains in A.
```



**InferenceSession does more than normalize devices.** It retains compilation settings, device handles, runtime configuration, and a native engine context. Its methods later compile graphs and initialize executable models; it also configures diagnostics/profiling. Constructor completion alone produces neither compiled code nor a Model.

**Example object snapshot · CPU add-one, immediately after construction**

~~~text
S1: Python InferenceSession
├─ num_threads = None                     Python argument, native policy becomes 0
└─ _impl → N1: Core::InferenceSession
   ├─ devices = [CPU0]                    retained driver handle
   ├─ sessionCompileConfig = CFG1: CompilationRequest
   │  ├─ modelSource = monostate          no model supplied yet
   │  ├─ pipelineName = None              chosen later for compilation
   │  └─ passthroughConfigs = []          no constructor extensions
   ├─ runtimeConfig
   │  ├─ numThreads = 0                   automatic selection, not zero workers
   │  ├─ apiLanguage = "python"
   │  └─ configured devices = [{spec=CPU0.deviceSpec, context=CPU0.ctx}]
   └─ runtimeContext → E1: Engine::Context (owned by N1)
      ├─ config = runtimeConfig copy
      ├─ maxContext = ContextRef(CX1)     reference to shared MAX M::Context
      ├─ cpuDevice = reference to HRT1    host AsyncRT CPUDevice
      └─ crashRecoveryContext            configured stack/crash recovery

CX1: MAX M::Context (reference-counted native service container)
├─ cpuDeviceRef → HRT1                   retained runtime reference
└─ typed service storage
   ├─ M::Config                          loaded process configuration
   ├─ Init::Options                      stored options used for reuse checks
   ├─ Telemetry::TelemetryContext        initialized with programName="max-engine"
   └─ GML::MaxEngineTelemetryContext     emplaced by Engine::Context::create

GPU variant: devices=[GPU0,CPU0]; configured-device pairs include both.
E1 and CX1 are still host-side native objects, not GPU memory allocations.
~~~

These labels are an explanatory snapshot, not Python `repr`, captured addresses, or an exhaustive list of process-dependent services. Native members are declared in [InferenceSession.h](sources/max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h#L95). The [Engine::Context constructor](sources/max/lib/max_license/API/c/EngineContext.cpp#L217) copies runtime config and retains the MAX context and CPU runtime reference. [M::Context](sources/Support/include/Support/Context.h#L30) is a typed service container; it is distinct from the MLIR Context that owns graph IR.

**Created or reused · same shared object, separate owned engine wrappers**

~~~text
First session:
  getOrCreateContext("max-engine", options)
    → no current global MAX context → create CX1 → retain HRT1 + services
  N1 owns E1; E1.maxContext refers to CX1

Later compatible session:
  getOrCreateContext("max-engine", same options)
    → current CX1 exists → ContextRef::copy(existing)
  N2 owns E2; E2.maxContext also refers to CX1

Different Init::Options with an existing global context:
  → fatal configuration mismatch; no silent incompatible reuse
~~~

The exact reuse decision is in [Init::getOrCreateContext](sources/Init/lib/Init.cpp#L116). A configuration-supplied MAX context follows the [earlier direct-retain branch](sources/max/lib/max_license/API/c/EngineContext.cpp#L175). Shared CX1 does not imply that every session has the same runtime device list: E1/E2 retain their own runtime configuration copies.

**populateCompileConfig · concrete before and after values**

| Moment | CompilationRequest contents |
| --- | --- |
| Original constructor input | `customExtensions=[]` |
| Default CFG1 | `modelSource=monostate`, `pipelineName=None`, `passthroughConfigs=[]` |
| `populateCompileConfig([])` | Copies the default request; adds no kernel-library entries |
| Optional extension illustration | `customExtensions=["/tmp/addon.mojoc"]` (must already exist) |
| Request with that extension | `passthroughConfigs=[KernelLibraryDriverConfig(path="/tmp/addon.mojoc")]`; source/pipeline still unset |
| Later compile of original G1 | Copies session CFG1 and adds G1's P1 path plus model/pipeline settings |

An extension is a **Mojo implementation package path**, not a device, Python class, or machine-code kernel. For example, `/tmp/addon.mojoc` can contain registered custom kernels. This path is illustrative; no such file is claimed to exist here. The [native extension validator](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L600) rejects missing or unsupported package paths. [CompilationRequest](sources/max/include/max/max_license/API/c/CompiledModels.h#L65) copies source/pipeline fields and clones its driver configs; [addKernelLibraryPath](sources/max/include/max/max_license/API/c/CompiledModels.h#L135) stores a KernelLibraryDriverConfig.

The original add-one call supplies no session extensions. Its Graph already owns P1's package paths, which [compile_async](sources/max/python/max/engine/api.py#L1022) supplies later. A separate optional API example is `InferenceSession(devices=[device], custom_extensions=[Path("/tmp/addon.mojoc")])`; Python processes these extensions before entering the native constructor. Session construction prepares the ownership/configuration needed for later work; it does not execute or specialize those kernels.

## C · session.compile(G1) → compiled artifact CM1

compiled = session.compile(graph)

INPUT: S1 + G1 module + package P1 + graph target metadata; expected_weights={}
OUTPUT: CM1: CompiledModel with resolved native artifact, MEF bytes and uninitialized model descriptor(s)
EXAMPLE: name="addition"; one X/Y float32[5,10]; AddOne specializes for CPU or GPU

```text
InferenceSession.compile(G1) [max/python/max/engine/api.py:L1062]  [Python caller]
├─ compile_async(G1) [max/python/max/engine/api.py:L969]
│  ├─ module = G1.module; extensions += G1.kernel_libraries_paths → P1 ↗ caller [max/python/max/engine/api.py:L969]
│  ├─ expected_weights = {} → add-one has no external weight bindings ↗ caller [max/python/max/engine/api.py:L969]
│  ├─ Analysis::seedKernelDecls(module) [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L75] → add reusable declarations to caller module ↗ caller [max/python/max/engine/api.py:L969]
│  ├─ _compile_module(module, extensions, tile_based_fusion=False) [max/python/max/engine/api.py:L1276]
│  │  ├─ acquire process-global compilation lock ↗ caller [max/python/max/engine/api.py:L1276]
│  │  ├─ _impl.compile(module._CAPIPtr, extensions, pipeline_name, False) [max/python/max/_core/max_license/modules/engine.cpp:L824]
│  │  │  └─ Core::InferenceSession::compile(capsule, ...) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L654]
│  │  │     ├─ CompilationRequest → source module pointer + graph format + options ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L654]
│  │  │     └─ Engine::Context::compileModel(config) [max/lib/max_license/API/c/EngineContext.cpp:L225]
│  │  │        ├─ allocate pending AsyncValue<CompiledModels> ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L225]
│  │  │        └─ addTask(cpuDevice, ...) → compilation on a host runtime worker
│  │  │           └─ compileModelSync(config) [max/lib/max_license/API/c/EngineContext.cpp:L269]
│  │  │              └─ CompiledModels::create(context, compileCfg, runtimeCfg) [max/lib/max_license/API/c/CompiledModels.cpp:L100]
│  │  │                 └─ MOFContext::compileToBinary(source, format) [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
│  │  │                    ├─ clone module → compiler working graph C1 ↗ caller [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
│  │  │                    ├─ graph-artifact cache hit → reuse prepared MEF bytes ↗ caller [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
│  │  │                    └─ miss/disabled → MO → MAP → MOGG → MGP → C.1 ↗ caller [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
│  │  │                       → kernel JIT / LLVM / targets C.2 → MEF C.3
│  │  └─ done callback → release compilation lock when the task resolves ↗ caller [max/examples/custom_ops/addition.py:L67]
│  └─ wrap pending handle → CompiledModel(handle, expected_weights={}) [max/python/max/engine/api.py:L414] → CM1 ↗ caller [max/examples/custom_ops/addition.py:L67]
│
├─ handle.wait() → caller waits for completion; exception() → raise on failure ↗ caller [max/examples/custom_ops/addition.py:L67]
└─ return resolved CM1 → graph names=("addition",); executable state comes from init ↗ caller [max/examples/custom_ops/addition.py:L67]
   G1 remains with the caller; native lowering transforms the clone C1.
```

### C.1 · Clone G1 → MO → MAP → MOGG → MGP

```text
MOFContext::compileToBinary [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377] → owns the compilation route for C1
└─ cache miss → convertMOToMEF(C1, options) [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356]
   └─ MO::runGCPipeline(C1, options) [GraphCompiler/lib/GraphCompiler.cpp:L638]
      ├─ create mlir::PassManager; gcPipelineImpl builds passes; pm.run(C1) executes them ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L638]
      ├─ buildMoPipeline [GraphCompiler/lib/GraphCompiler.cpp:L219] → MO = Modular Operations
      │  ├─ graph/tensors/shapes/device placement/custom op contracts ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L219]
      │  └─ seeded kernel declarations → skip redundant package import ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L219]
      ├─ MOToMAPPass [GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L764] → MAP = Modular Access Primitives
      │  └─ expose accesses and computations for fusion / implementation decisions ↗ caller [GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L764]
      ├─ buildMapPipeline [GraphCompiler/lib/GraphCompiler.cpp:L338] → optimize access patterns / eligible fusion ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L638]
      ├─ MAPToMOGGPass [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L53] → MOGG = Modular Operation Generator Graph
      │  └─ MojoRegistry + MAPToMOGGState → select implementation / glue calls ↗ caller [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L53]
      ├─ buildMoggPipeline [GraphCompiler/lib/GraphCompiler.cpp:L370] → refine Mojo-oriented functions and calls ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L638]
      ├─ MOGGToMGPPass [GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:L2406] → MGP = Modular Glue Primitives
      │  └─ MOGraphLowering → init/execute regions, buffers, state slots and chains ↗ caller [GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:L2406]
      ├─ buildMgpPipeline [GraphCompiler/lib/GraphCompiler.cpp:L382]
      │  ├─ FinalizeDeviceGraphsPass [GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:L375] → finalize marked device graphs when present ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L382]
      │  └─ EmitMojo [GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:L147] → emitted source for native routines / package imports ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L382]
      ├─ JitCompileKernelsPass [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529] → compile/reuse native routines (C.2) ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L638]
      └─ buildJitPipeline [GraphCompiler/lib/GraphCompiler.cpp:L75]
         ├─ LowerCompiledRoutinesPass [GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:L102] → ModuleLoad / ExecuteGeneric ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L75]
         └─ inline/clean chains → OutlineModelsToCallables → initialization + execution plan ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L75]
   Dialect names and roles → GraphCompiler.md [GraphCompiler/docs/GraphCompiler.md:L1]

Add-one ancestry [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L53] [schematic representation; not a captured compiler dump]
└─ mo.custom("add_one", X) → implementation-oriented accesses/call ↗ caller [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L53]
   → selected AddOne routine → buffer/state execution plan → compiled callable
```

[Open the detailed C.1 compiler route · examples and why for every stage](max_graph_compiler_pipeline_examples.html)

**Shared-module example · compilation belongs to C.1**

After **B** has created `session` (S1), compile the `unit` built in A.2:

```python
compiled_unit = session.compile(unit)
```

This accepts a `Module` containing multiple graphs and returns one compiled artifact. See [compile’s Module input](sources/max/python/max/engine/api.py#L1062).

**Schematic MO view after RMO → MO.** Extra chains and device metadata are omitted. The graph bodies use the spelling in the repository’s [two-graph compiler test](sources/GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir#L14):

```mlir
module {
  mo.graph @add(%a: !mo.tensor<[4], f32>,
                %b: !mo.tensor<[4], f32>)
      -> !mo.tensor<[4], f32> {
    %r = mo.add(%a, %b) : !mo.tensor<[4], f32>
    mo.output %r : !mo.tensor<[4], f32>
  }
  mo.graph @mul(%a: !mo.tensor<[4], f32>,
                %b: !mo.tensor<[4], f32>)
      -> !mo.tensor<[4], f32> {
    %r = mo.mul(%a, %b) : !mo.tensor<[4], f32>
    mo.output %r : !mo.tensor<[4], f32>
  }
}
```

| Python / IR item | Meaning in this example |
|---|---|
| `unit` / outer `module { ... }` | Owns both definitions as one compilation unit. |
| `g_add` / `mo.graph @add` | Describes elementwise `A + B`. |
| `g_mul` / `mo.graph @mul` | Describes elementwise `A * B`. |
| `%a`, `%b`, `%r` | Symbolic values connected by data dependencies. |
| `!mo.tensor<[4], f32>` | Tensor type: four float32 elements; the numbers arrive at execution. |


**G1 → compiler clone C1 → MO → MAP**

`G1` is the caller-owned Python `Graph`; its `builtin.module` contains the `mo.graph` operation. `session.compile(G1)` gives the compiler a working clone, **C1**. MAX runs graph passes on C1, leaving G1 available to the caller.

```text
G1 / C1 at the MO boundary
module {
  mo.graph @addition(%x: !mo.tensor<[5, 10], f32>) -> !mo.tensor<[5, 10], f32> {
    %y = mo.custom [symbol = "add_one"](%x)
         : (!mo.tensor<[5, 10], f32>) -> !mo.tensor<[5, 10], f32>
    mo.output %y : !mo.tensor<[5, 10], f32>
  }
}

MOToMAP · registered custom-op pattern · schematic for add_one
map.iter.opaque
├─ inputs (X); outputs (Y)
└─ map.call.opaque @...AddOne
   └─ kernelName = "add_one"
```

**Custom-op evidence:** the [MOToMAP custom-kernel test](sources/GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels.mlir#L30) shows the same registered opaque-op form: `mo.custom` becomes `map.iter.opaque`, which contains `map.call.opaque`. Its test-only symbol is `map_lower_one_fused_output`; the drawing applies that lowering pattern to `add_one` and is schematic, not a captured MAX dump. The package P1 supplies the registered `AddOne` implementation; its source uses `foreach` to define the elementwise work ([AddOne](sources/max/examples/custom_ops/kernels/add_one.mojo#L32)).

**Built-in-op comparison:** Python `ops.add(A, B)` initially records relaxed RMO. After RMO → MO, the [elementwise-add test](sources/GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-add-static.mlir#L6) shows `mo.add` becoming `map.iter.elementwise` with indexed loads, `map.call.elementwise @...Add`, and a store. That is a distinct MAP lowering shape from the `add_one` opaque custom call.

The enclosing module groups definitions while compiler passes rewrite C1's contents. The [graph-stage test](sources/GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir#L28) later expects `mo.graph @add` and `mo.graph @mul` to become `mgp.model @add` and `mgp.model @mul` in the runtime-plan representation.

### C.2 · Kernel specialization → LLVM → target code

```text
compileMojoToBinary(C1, emitted Mojo, ...) [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529]
├─ emitted routine + P1 → parser/checks/LIT lowering → separate kernel IR KIR1 ↗ caller [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529]
├─ compiled-kernel lookup → checkCache(state, ...) [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L213]
│  ├─ hit → handleCacheHits [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L507] → reuse compatible native archives ↗ caller [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L213]
│  └─ miss → handleCacheMisses [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L435]
│     ├─ KGENCompiler::runElaborationPipeline [Mojo/lib/Compiler/KGENCompiler.cpp:L1140]
│     │  ├─ populateElaborateModulePasses [Mojo/lib/Compiler/KGENCompiler.cpp:L967] → concrete parameter specialization ↗ caller [Mojo/lib/Compiler/KGENCompiler.cpp:L1140]
│     │  ├─ buildPostElaborationPipeline [Mojo/lib/Compiler/Pipeline/Pipeline.cpp:L154] → concrete IR optimization ↗ caller [Mojo/lib/Compiler/KGENCompiler.cpp:L1140]
│     │  └─ offloads encountered → compileOffloads [Mojo/lib/Compiler/KGENCompiler.cpp:L516] → group/slice IR by TargetInfo ↗ caller [Mojo/lib/Compiler/KGENCompiler.cpp:L1140]
│     └─ ObjectCompiler::emitArchive [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L1023] → native archive with callable symbols
│        └─ ObjectCompiler::lowerAllFuncsToLLVM [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L816]
│           ├─ buildLowerToLLVMPipeline [Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp:L26] → KGEN/POP/control flow → LLVM dialect (MLIR) ↗ caller [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L816]
│           └─ MLIR-to-LLVM translation [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L796] → llvm::Module → LLVM1 (LLVM IR) ↗ caller [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L816]
│
├─ target options guide compilation → TargetBackendRegistry::getBackend [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L114] uses the triple
│  ├─ host target → HostBackend [Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:L54] → CPU object / host routines ↗ caller [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L114]
│  ├─ nvptx64-nvidia-cuda, e.g. sm_90a → NVPTXBackend [Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:L1] → PTX → cubin ↗ caller [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L114]
│  └─ amdgcn-amd-amdhsa, e.g. gfx942 → AMDGPUBackend [Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:L77] → GPU object → ELF image ↗ caller [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L114]
│     Target examples are source examples, not detected hardware on this host.
├─ LLVM TargetRegistry + TargetMachine [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L63] → in-process LLVM code generation [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L277] ↗ caller [max/examples/custom_ops/addition.py:L67]
└─ native compiled resources + C1 runtime plan → continue MEF preparation (C.3) ↗ caller [max/examples/custom_ops/addition.py:L67]

JIT = just-in-time compilation → required specialized code is prepared as the program runs.
└─ here: addition.py → session.compile(G1) invokes compiler APIs in the host process ↗ caller [max/examples/custom_ops/addition.py:L67]
   → AddOne's dtype/layout/target specialize; compiled code can be reused for later inputs.
   P1 precompilation is the earlier subprocess; kernel JIT is a later compilation phase.
```

### C.3 · MEF preparation → compiled artifact CM1

```text
convertMOToMEF(C1) [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356]  [after graph passes + native routine JIT]
└─ MO::convertMGPToMEF(lowered_module) [GraphCompiler/lib/GraphCompiler.cpp:L692]
   ├─ clone for serialization → prepare device-specific runtime primitive names ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L692]
   ├─ translateMLIRToMEF → serialized graph callables / attrs / native resources ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L692]
   └─ WriteableBuffer → prepared MEF bytes in memory ↗ caller [GraphCompiler/lib/GraphCompiler.cpp:L692]

CompiledModels::create(...) [max/lib/max_license/API/c/CompiledModels.cpp:L100] receives those bytes
└─ CompiledModels::read(buffer, frameworkCtx, sourceModel) [max/lib/max_license/API/c/CompiledModels.cpp:L80]
   └─ loadModelsFromMEF(buffer) [max/lib/max_license/API/c/CompiledModels.cpp:L29]
      ├─ MEFFile::open → validate/open the prepared container ↗ caller [max/lib/max_license/API/c/CompiledModels.cpp:L29]
      ├─ loadMGPMEFFile(mef) [GraphCompiler/lib/MEFModel/MEFModel.cpp:L901] → register runtime primitives; read Model descriptors ↗ caller [max/lib/max_license/API/c/CompiledModels.cpp:L29]
      └─ retain buffer + MEF handle + model descriptors → native CompiledModels ↗ caller [max/lib/max_license/API/c/CompiledModels.cpp:L29]
         → resolve async handle → Python CM1: CompiledModel

Cache routes → MOFContext graph-artifact cache [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
├─ artifact hit → skip graph lowering / kernel JIT; read retained MEF bytes/descriptors ↗ caller [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
├─ kernel hit only → reuse routine code; still lower/prepare the graph's runtime plan ↗ caller [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
└─ cold path → both compilation phases run ↗ caller [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]

On disk → CM1.export_mef(path) [max/python/max/engine/api.py:L414] explicitly exports the compiled bytes
└─ session.compile(G1) itself returns CM1; it does not require a user-named .mef file. ↗ caller [max/python/max/engine/api.py:L414]
```

## D · session.init(CM1) → executable model M1

model = session.init(compiled)

INPUT: S1 + resolved CM1 + real device contexts; weights_registry omitted → {}
OUTPUT: M1: initialized ModelHandle retaining executable state, CM1 and native S1
EXAMPLE: init_all → {"addition": M1}; init → M1. The input Buffer is supplied later.

```text
InferenceSession.init(CM1) [max/python/max/engine/api.py:L1126]
├─ init_all(CM1, weights_registry=None) [max/python/max/engine/api.py:L1160]
│  ├─ registry = {}; expected_weights = {} → no weight-binding validation failures → D.2 ↗ caller [max/python/max/engine/api.py:L1160]
│  ├─ _impl._load_all(compiled_handle, registry) [max/python/max/_core/max_license/modules/engine.cpp:L787] → share artifact handle
│  │  └─ Core::InferenceSession::loadAll(handle, weightsDict) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L289]
│  │     ├─ buildWeightsFromDict({}) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L214] → empty native registry + managed-tensor list ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L289]
│  │     ├─ await compiled handle (release GIL while waiting) → descriptor "addition" ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L289]
│  │     └─ for each model descriptor → Core::InferenceSession::load(model, registry) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L275]
│  │        └─ Engine::Context::initModel(model, registry) [max/lib/max_license/API/c/EngineContext.cpp:L288]
│  │           └─ addTask(cpuDevice, ...) → initializer runs on a host runtime worker
│  │              └─ createExecutableModel(model, cpuDevice, config, registry) [max/lib/max_license/API/c/EngineContext.cpp:L79]
│  │                 ├─ DeviceMap from session DeviceSpecs + retained driver contexts ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L79]
│  │                 ├─ Model::setup(cpuDevice, registry, deviceMap) [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184] → D.1
│  │                 │  ├─ mapDeviceRefsToDevices [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24] → synchronize/resolve contexts ↗ caller [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184]
│  │                 │  ├─ initArgs = registry pointer + model device contexts ↗ caller [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184]
│  │                 │  ├─ initFn.invoke(initArgs) → run compiled initialization callable ↗ caller [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184]
│  │                 │  └─ construct ExecutableModel from chain/context/registry/extra args ↗ caller [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184]
│  │                 └─ waitForModelSetup() [GraphCompiler/lib/MEFModel/MEFModel.cpp:L387] → initialization results must be ready ↗ caller [max/lib/max_license/API/c/EngineContext.cpp:L79]
│  │        → ModelHandle(..., native_session_ref, executable, registry, index) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L194]
│  │          → cache input/output devices → M1
│  └─ name-index results → {"addition": M1} ↗ caller [max/examples/custom_ops/addition.py:L68]
├─ exactly one model? yes → return sole model M1 ↗ caller [max/examples/custom_ops/addition.py:L68]
└─ next source step → numeric input allocation, model.execute(x) (outside init) ↗ caller [max/examples/custom_ops/addition.py:L68]
```

### D.1 · Load runtime resources → ready executable M1

```text
Model::setup [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184] → compiled initFn.invoke(registry + devices)
├─ mapDeviceRefsToDevices [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24]
│  ├─ synchronize supplied driver contexts before first resource use ↗ caller [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24]
│  ├─ CPU DeviceRef → create legacy runtime CPU context ↗ caller [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24]
│  └─ GPU DeviceRef → reuse supplied context; fallback create by device id if absent ↗ caller [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24]
├─ CUDA/HIP contexts → ensure configured stream count before capture (default: 2) ↗ caller [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184]
├─ invoke the compiled init region → runtime/state initialization and native routine loading
│  └─ ModuleLoad planned by LowerCompiledRoutinesPass [GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:L102]
│     └─ mgp_module_load_cpu [GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp:L55] → CompiledKernelsContext in a runtime state slot
│        └─ CompiledKernelsContext::load(archive_bytes, symbols) [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L85] Loader deep dive · all steps + GPU driver → max_compiled_kernels_context_load_and_driver.html
│           ├─ ordinary route: archive hash → execution engine addIfAbsent<StaticArchiveLayer> ↗ caller [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L85]
│           └─ lookup named symbols → resident native callable pointers ↗ caller [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L85]
│              (debug/profiler route can link/load a shared library)
├─ initializer results → [chain, MGP context, weight-registry reference, extra state/device args] ↗ caller [max/examples/custom_ops/addition.py:L68]
├─ waitForModelSetup [GraphCompiler/lib/MEFModel/MEFModel.cpp:L387] → await every required value; propagate errors ↗ caller [max/examples/custom_ops/addition.py:L68]
└─ executable M1 → reuse retained state and routines for fresh execution inputs ↗ caller [max/examples/custom_ops/addition.py:L68]

Boundaries
├─ .mojoc P1: portable post-parse package for later specialization ↗ caller [max/examples/custom_ops/addition.py:L68]
├─ CM1: compiled bytes + descriptors; still needs runtime setup ↗ caller [max/examples/custom_ops/addition.py:L68]
└─ M1: ready executable handle; numeric X arrives at execute, not at constructor/init ↗ caller [max/examples/custom_ops/addition.py:L68]
```

**Shared-module example · runtime initialization belongs to D.1**

Use the compiled artifact from C.1:

```python
models = session.init_all(compiled_unit)
add_model = models["add"]
mul_model = models["mul"]
# Supply input buffers to add_model.execute(A, B) or mul_model.execute(A, B) later.
```

`init_all` returns a model dictionary keyed by each graph’s `sym_name`; the compiled artifact is ready for execution after successful runtime initialization. See [init_all’s named results](sources/max/python/max/engine/api.py#L1160). The single-graph addition example uses `session.init(CM1)` to obtain one M1.

### D.2 · Initialization gates, weights and retained ownership

```text
init_all initialization gates [max/python/max/engine/api.py:L1160]
├─ add-one: expected weights {} + supplied registry {} → pass ↗ caller [max/python/max/engine/api.py:L1160]
├─ weighted graph: missing declared name → ValueError before native initialization ↗ caller [max/python/max/engine/api.py:L1160]
├─ supplied weight: device type must match; array must be contiguous ↗ caller [max/python/max/engine/api.py:L1160]
├─ actual device/context/setup failure → native initialization raises ↗ caller [max/python/max/engine/api.py:L1160]
├─ virtual-device mode → CompileOnlyModel stand-in; execute refuses hardware work ↗ caller [max/python/max/engine/api.py:L1160]
└─ init [max/python/max/engine/api.py:L1126] requires exactly one result; multi-model artifacts use init_all ↗ caller [max/python/max/engine/api.py:L1160]

M1 [native ModelHandle exposed as the Python Model]
├─ loadedModel → initialized ExecutableModel and persistent execution state ↗ caller [max/examples/custom_ops/addition.py:L68]
├─ compiledModels → shared CM1 native artifact / model descriptors / bytes ↗ caller [max/examples/custom_ops/addition.py:L68]
├─ inferenceSession → retained native S1 reference [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L194]
│  └─ Engine::Context → MAX context / CPU runtime / session device settings ↗ caller [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L194]
└─ weightsRegistry + managedTensors → retained bindings (empty in add-one) ↗ caller [max/examples/custom_ops/addition.py:L68]

Repeated initialization → _load_all uses compiled.copy() [max/python/max/_core/max_license/modules/engine.cpp:L787]
└─ same CM1 can initialize multiple handles; each handle retains what it needs. ↗ caller [max/python/max/_core/max_license/modules/engine.cpp:L787]
```

## E · Input and output example

```text
addition.py: execution happens after all four preparation calls [max/examples/custom_ops/addition.py:L1]
├─ x_values = np.random.uniform(size=(5,10)).astype(np.float32) → host array ↗ caller [max/examples/custom_ops/addition.py:L71]
├─ x = Buffer.from_numpy(x_values).to(device) → B1: input Buffer ↗ caller [max/examples/custom_ops/addition.py:L71]
├─ result = model.execute(x)[0] → O1: output Buffer ↗ caller [max/examples/custom_ops/addition.py:L71]
└─ result.to(CPU()).to_numpy() → CPU-readable numbers; expected = x_values + 1 ↗ caller [max/examples/custom_ops/addition.py:L71]

Illustrative values in the same [5,10] shape [not an executed MAX result]
├─ input row 0:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ↗ caller [max/examples/custom_ops/addition.py:L71]
├─ output row 0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ↗ caller [max/examples/custom_ops/addition.py:L71]
└─ G1 / S1 / CM1 / M1 were prepared without these numbers; B1/O1 hold per-call data. ↗ caller [max/examples/custom_ops/addition.py:L71]
```


---

# From Bazel’s build graph to a selected device backend

## 1. Previous answer — unchanged

For ./bazelw build //max/examples/custom_ops:addition, add_one.mojo is runtime source data. Bazel does not compile
  that custom kernel into CPU or GPU machine code in this target. The Python example precompiles the kernel package
  later, when it runs and constructs its Graph.

  ### 1.1 What the addition build produces

  Bazel analysis identifies the addition target as a py_binary. Its resolved files include:

  • bazel-out/k8-dbg/bin/max/examples/custom_ops/addition — the Python launcher.
  • max/examples/custom_ops/addition.py — the Python entry-point source, available through runfiles.
  • bazel-out/k8-dbg/bin/max/examples/custom_ops/addition.mypy_stdout — an auxiliary output; I haven’t inspected its
    contents.

  The macro supplies :kernel_sources as data, so the matching kernels/*.mojo files are available at runtime. They are
  not compiled into kernels.mojoc as part of the :addition target. A separate :kernels target produces that package;
  shared Mojo dependencies such as layout, extensibility, and std have their own .mojoc outputs. See the BUILD file
  (max/examples/custom_ops/BUILD.bazel:9) and example macro (max/examples/custom_ops/custom_op_example.bzl:12).

  These paths came from Bazel analysis. I started the actual build, but interrupted its cold dependency build before
  it finished, so I can’t report a successful build result.

  ### 1.2 Where the Mojo dependencies come from

  They are Bazel targets and generated package files, not Python classes or objects.

   Label in the example macro    Target it aliases to                       Analyzed .mojoc output
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   //max:layout                  //max/kernels/src/layout:layout            bazel-out/k8-dbg/bin/max/kernels/src/
                                                                            layout/layout.mojoc
  ────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────────────
   //max:extensibility           //max/kernels/src/graph_compiler/          bazel-out/k8-dbg/bin/max/kernels/src/
                                 extensibility:extensibility                graph_compiler/extensibility/
                                                                            extensibility.mojoc
  ────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────────────
   @mojo//:std                   //Mojo/stdlib/std:std                      bazel-out/k8-dbg/bin/Mojo/stdlib/std/
                                                                            std.mojoc

  The //max:* aliases are created by max_aliases() in max/BUILD.bazel (max/BUILD.bazel) using the mappings in
  mojo_aliases.bzl (bazel/mojo_aliases.bzl:14). The source packages are declared in their respective BUILD files:
  layout (max/kernels/src/layout/BUILD.bazel), extensibility (max/kernels/src/graph_compiler/extensibility/
  BUILD.bazel), and stdlib (Mojo/stdlib/std/BUILD.bazel).

  A .mojoc is a precompiled Mojo package: it stores a lit.package operation serialized as MLIR bytecode, including
  declarations and bodies in the package’s post-parse IR. It has not yet been elaborated for concrete parameter
  values or turned into target machine code. The repository’s Mojo compiler walkthrough (Mojo/docs/compiler/
  MojoCompilerWalkthrough.md:599) describes this format; the official Mojo docs also describe .mojoc as a precompiled
  package for importing into other Mojo code. (mojolang.org)

  ### 1.3.1 How BUILD.bazel connects the targets

  The :kernel_sources filegroup is used in two distinct ways:

  This Mermaid diagram exceeds the terminal renderer's size limits.
  flowchart LR
      BUILD["BUILD.bazel"] --> MACRO["custom_op_example_py_binary(name = addition)"]
      MACRO --> PY["py_binary :addition"]
      ADDPY["addition.py · srcs"] --> PY
      SOURCES["filegroup :kernel_sources<br/>glob kernels/*.mojo"] -->|"runtime data / runfiles"| PY
      MOJODEPS["layout · extensibility · std<br/>mojo_deps"] --> ENV["Mojo runtime environment"]
      ENV --> PY

      MACRO --> TESTMACRO["modular_run_binary_test"]
      TESTMACRO --> TEST["binary_test :addition.example-test"]
      TEST -->|"runs this binary when tested"| PY

      SOURCES -->|"srcs"| LIB["mojo_library :kernels"]
      LIB -->|"dependency"| BENCH["mojo_binary :benchmarks"]

  There is no :addition → :kernels dependency edge. The Python target receives the Mojo sources as runtime data.
  Separately, :kernels precompiles them, and :benchmarks depends on that library. The rules and edges are in
  BUILD.bazel (max/examples/custom_ops/BUILD.bazel:9).

  #### 1.3.1.1 What each BUILD construct means

   Construct                      What it is and how this file uses it
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   mojo_binary                    A Bazel rule for a Mojo executable. Here, :benchmarks is a Mojo program that
                                  depends on :kernels. The local wrapper forwards to the upstream Mojo rule.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   mojo_library                   A Bazel rule that precompiles Mojo sources into a .mojoc package. Here, :kernels
                                  uses :kernel_sources as srcs. The local wrapper also adds documentation targets.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   custom_op_example_py_binary    A local Starlark macro, not a Bazel rule. It expands an example declaration into a
                                  Python binary and its automatic example test.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   filegroup                      A Bazel target that groups files. :kernel_sources collects kernels/*.mojo; the
                                  filegroup itself does not compile them.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   load(...)                      A Starlark directive that imports symbols from a .bzl file into the BUILD file’s
                                  scope. It does not create a build target.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   package(...)                   Package-level metadata. Here, default_visibility makes targets private unless they
                                  specify otherwise; it does not create a Python or Mojo package artifact.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   target_compatible_with         A rule attribute containing platform constraints. It can make a target
                                  incompatible with selected OS or hardware configurations; it does not select a
                                  compiler backend at runtime.

  The request listed mojo_binary twice; it is covered once above.

  #### 1.3.1.2 How they work together here

  • load makes mojo_binary, mojo_library, requirement, and the local Python-example macro available to BUILD.bazel.
  • filegroup(:kernel_sources) collects the .mojo files.
  • custom_op_example_py_binary(name = "addition") creates :addition with addition.py as its source, the kernel
    filegroup as run-time data, Python dependencies, and Mojo dependencies.

  • mojo_library(:kernels) compiles the same source group into a separate .mojoc package.
  • mojo_binary(:benchmarks) consumes :kernels.
  • package(default_visibility = ...) supplies the package’s default visibility. target_compatible_with, when
    provided, constrains the compatibility of the generated targets.

  ### 1.3.2 How the macro expands

  The custom_op_example_py_binary macro in custom_op_example.bzl (max/examples/custom_ops/custom_op_example.bzl:5)
  calls two other macros:

  This Mermaid diagram exceeds the terminal renderer's size limits.
  flowchart TD
      CALL["BUILD call:<br/>custom_op_example_py_binary(name = addition)"]
      CALL -->|"creates"| MPY["modular_py_binary(name = addition)"]
      MPY --> ENV["mojo_test_environment<br/>with layout, extensibility, std"]
      MPY --> PY["py_binary :addition<br/>addition.py + Python deps<br/>kernel sources in data/runfiles"]
      MPY --> HELPERS["Supporting targets<br/>Mojo deps, venv, debug, pydeps"]

      CALL -->|"also creates"| RUNTEST["modular_run_binary_test<br/>name = addition.example-test"]
      RUNTEST --> BT["binary_test :addition.example-test"]
      BT -->|"symlinks executable and merges runfiles"| PY

  modular_py_binary prepares the Mojo environment and then invokes py_binary; it also defines supporting targets such
  as the virtual environment and debug entry point. See modular_py_binary.bzl (bazel/internal/
  modular_py_binary.bzl:53).

  modular_run_binary_test creates a sibling test target named :addition.example-test. It configures the test
  environment and calls binary_test. That rule symlinks the binary’s executable and merges its runfiles. See
  modular_run_binary_test.bzl (bazel/internal/modular_run_binary_test.bzl:6) and binary_test.bzl (bazel/internal/
  binary_test.bzl:3).

  So:

  • ./bazelw build //max/examples/custom_ops:addition builds the Python target and its dependencies; it does not run
    addition.py.

  • ./bazelw run //max/examples/custom_ops:addition runs the Python example.
  • ./bazelw test //max/examples/custom_ops:addition.example-test runs the generated test target.

### Portable build and macro diagrams

```text
BUILD.bazel
  +-- custom_op_example_py_binary(name="addition")
  |     +-- modular_py_binary -> :addition Python launcher / runfiles
  |     |     +-- addition.py + Python dependencies
  |     |     +-- :kernel_sources -> raw kernels/*.mojo runtime data
  |     |     +-- mojo_test_environment -> shared .mojoc packages / tools
  |     +-- modular_run_binary_test -> :addition.example-test
  |           +-- binary_test -> runs :addition with its runfiles
  +-- :kernel_sources -> mojo_library :kernels -> kernels.mojoc
        +-- mojo_binary :benchmarks consumes the separate :kernels library

No :addition -> :kernels dependency edge.
```

### 2. Graph(custom_extensions=[kernels_dir]) packages the Mojo package in kernels/; it does not package
  addition.py

  At runtime, Graph passes the extension path to KernelLibrary.load_paths. If the path is a Mojo source package
  directory, MAX calls _build_mojo_source_package, which invokes mojo precompile and writes a .mojoc file in a
  temporary package cache. addition.py remains a separate Python program. See graph.py (max/python/max/graph/
  graph.py:269) and paths.py (Mojo/python/mojo/paths.py:141).

  For this example, the extension path is kernels/, a Mojo package rooted at kernels/__init__.mojo. Its source
  modules, including add_one.mojo, are part of that package. addition.py is not. The runtime package is cached under
  a path shaped like /tmp/.modular_<uid>/mojo_pkg/mojo_pkg_<hash>.mojoc.


### 2.1 Start the Python application {#toolchain-q2-1}

**2.1.1 Build and run are distinct actions.** The Python launcher resolves `addition.py`, its Python dependencies, the raw kernel source files and the Mojo runtime environment. The application begins when the launcher runs.

~~~bash
# From the modular/ repository root:
./bazelw build //max/examples/custom_ops:addition
./bazelw run //max/examples/custom_ops:addition

# The generated sibling test also runs the example:
./bazelw test //max/examples/custom_ops:addition.example-test
~~~

**2.1.2 The binary's environment supplies the native tools.** `modular_py_binary` creates a `mojo_test_environment`, collects transitive `MojoInfo` packages and expands compiler, import-path, shared-library and linker variables. The configured source-built Mojo tool is `//Mojo/tools/mojo`; the LLD dependency is `@llvm-project//lld:lld`.

| Runtime variable | What the Python/native compiler can locate |
| --- | --- |
| `MODULAR_MOJO_MAX_DRIVER_PATH` | Mojo compiler driver |
| `MODULAR_MOJO_MAX_IMPORT_PATH` | Directories containing the precompiled Mojo dependencies |
| `MODULAR_MOJO_MAX_LLD_PATH` | Configured LLD executable |
| `MODULAR_MOJO_MAX_LINKER_DRIVER` | Host C++ linker driver |
| `MODULAR_MOJO_MAX_SHARED_LIBS` / `...SYSTEM_LIBS` | Native libraries and system-link settings |

These are launch settings derived from the build toolchain. The kernel's LLVM target is selected later from target information. [Python binary rule stages the Mojo runtime environment (bazel/internal/modular_py_binary.bzl:53)](sources/bazel/internal/modular_py_binary.bzl#L53) · [Mojo toolchain and native runtime files collected for execution (bazel/internal/mojo_test_environment.bzl:78)](sources/bazel/internal/mojo_test_environment.bzl#L78) · [Mojo compiler, LLD and C++ linker settings exposed to the process (bazel/internal/mojo_test_environment.bzl:178)](sources/bazel/internal/mojo_test_environment.bzl#L178) · [Source-built Mojo compiler and LLVM LLD toolchain (bazel/internal/BUILD.bazel:185)](sources/bazel/internal/BUILD.bazel#L185)

### 2.2 Prepare the source package at Graph construction {#toolchain-q2-2}

**2.2.1 Call path.** `Graph(custom_extensions=[mojo_kernels])` uses `KernelLibrary.load_paths`. A source-package directory is passed to `mojo.paths._build_mojo_source_package`; an existing `.mojoc` path can be added directly.

**2.2.2 Actual subprocess arguments.** The helper constructs the following argument list; the configured Mojo driver is supplied by `subprocess_run_mojo`:

~~~text
mojo precompile <absolute kernels/ source path> -o <private staging package path>
~~~

The compiler receives the **Mojo source package directory**. `addition.py` remains the Python entry point. The package includes its modules such as `add_one.mojo`; this does not mean every packaged function is emitted as machine code.

**2.2.3 Actual output naming and publication.** The helper hashes the absolute source path and constructs:

~~~text
<tempdir>/.modular_<uid>/mojo_pkg/
    .staging.<pid>/mojo_pkg_<path_hash>.mojoc   # private compiler output
    mojo_pkg_<path_hash>.mojoc                 # published path
~~~

It runs precompilation into the staging directory, then `os.replace` atomically publishes the complete package. The private file uses the final basename because the compiler embeds the module name. This prevents another process from observing a half-written package.

**2.2.4 This hash identifies a source path.** It is not a hash of the kernel's contents, selected GPU architecture or final machine code. The helper shown here invokes precompilation; the presence of that pathname alone is not proof of a skipped compile. [Import source or precompiled Mojo packages (max/python/max/graph/graph.py:269)](sources/max/python/max/graph/graph.py#L269) · [Actual mojo precompile command and atomic package publication (Mojo/python/mojo/paths.py:141)](sources/Mojo/python/mojo/paths.py#L141) · [Custom kernel package root (max/examples/custom_ops/kernels/__init__.mojo:1)](sources/max/examples/custom_ops/kernels/__init__.mojo#L1)

### 2.3 What is inside .mojoc? {#toolchain-q2-3}

**2.3.1 Representation.** A precompiled Mojo package serializes a `lit.package` operation as MLIR bytecode. It retains declarations, implementation bodies and package/dependency information in post-parse IR. The compiler walkthrough describes semantic/lifetime checking during package preparation and later loading of those bodies.

**2.3.2 Why bodies remain.** `AddOne.execute` still has a compile-time `target`; its nested function still has a compile-time `width`. Concrete instantiation needs those bodies later. This package is useful compiler input rather than a final CPU/GPU image.

**2.3.3 Two .mojoc routes exist.** The standalone `:kernels` Bazel library and the runtime source-package helper can precompile the same source group into different output locations. The shared `layout`, `extensibility` and `std` packages are separate dependencies. The dependency edges in point 1 determine which packages Bazel builds.

~~~text
addition.py ── Python program; Graph builder and execute calls
kernels/    ── Mojo source package
  __init__.mojo
  add_one.mojo
  other declared .mojo modules
         │ mojo precompile
         ▼
P1 · .mojoc ── post-parse MLIR bytecode, declarations and bodies
         │ import + concrete specialization later
         ▼
target-specific routines / device images
~~~

[Mojo package bytecode and deferred specialization (Mojo/docs/compiler/MojoCompilerWalkthrough.md:569)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L569) · [Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1) · [Complete custom-op BUILD graph (max/examples/custom_ops/BUILD.bazel:1)](sources/max/examples/custom_ops/BUILD.bazel#L1)

### 2.4 Register implementations and record G1 {#toolchain-q2-4}

**2.4.1 Name linkage.** Python records `ops.custom(name="add_one", ...)`. Mojo declares `@extensibility.register("add_one")` on `AddOne`. The matching string connects the graph operation to its registered implementation.

**2.4.2 Input/output contract.** The example records float32 tensors of shape `[5,10]`, the chosen `DeviceRef` and an explicit output `TensorType`. Its custom shape function raises `NotImplemented`; the explicit output contract makes that optional function unnecessary in this example.

**2.4.3 Recording is symbolic.** G1 stores operations, types, shapes, device placement and def-use relationships. The example creates its random input values later. `Graph.output` completes/verifies the graph output contract.

**2.4.4 Compiler declarations can be seeded once.** The Graph branch of `InferenceSession.compile_async` seeds the MLIR module from the Graph's kernel analysis before native compilation. The compiler recognizes the mapping attribute and can avoid repeating package import. [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Create and immediately verify a custom graph operation (max/python/max/graph/ops/custom.py:69)](sources/max/python/max/graph/ops/custom.py#L69) · [Create MLIR graph structure and optionally trace a Python callable. (max/python/max/graph/graph.py:819)](sources/max/python/max/graph/graph.py#L819) · [Finish the graph's output contract and verify the completed graph. (max/python/max/graph/graph.py:1396)](sources/max/python/max/graph/graph.py#L1396) · [Prepare a Graph, Module, or artifact path for asynchronous native compilation. (max/python/max/engine/api.py:969)](sources/max/python/max/engine/api.py#L969)

### 3. InferenceSession.compile runs MAX’s graph compiler pipeline

  The MAX graph compilation path uses MLIR pass managers in the GraphCompiler library. The source builds the pipeline
  through MO, MAP, MOGG, and MGP stages, then can proceed to JIT compilation. These are compiler-library passes
  invoked by the runtime path; they are not a sequence of standalone shell commands such as mlir-opt. See
  GraphCompiler.cpp (GraphCompiler/lib/GraphCompiler.cpp:199).

  Conceptually:

  Python Graph + custom package
    → MAX C++ / MLIR GraphCompiler
    → MO → MAP → MOGG → MGP
    → device-graph finalization and kernel JIT


### 3.1 Which objects own and run the transformations? {#toolchain-q3-1}

**3.1.1 Entry.** The Python `InferenceSession` passes the Graph's module to the native engine. `MOFContext::compileToBinary` clones the MLIR module and gathers `GCPipelineOptions`. G1 remains the caller's graph; C1 is the compiler-owned clone.

**3.1.2 Execution mechanism.** Native `mlir::PassManager`/`mlir::OpPassManager` objects run passes assembled by `M::MO::buildGCPipeline` and `gcPipelineImpl`. The dialect classes provide operations/types; the conversion pass objects rewrite the module.

**3.1.3 Relation to MAX Serve.** These are MAX/Mojo compiler objects reached through the engine API. A Serve pipeline can call the same engine while initializing a model. Its API server, worker scheduler and batch constructor do not implement the MO/MAP/MOGG/MGP conversions. The four Python examples call the engine directly.

| Layer / object | Responsibility |
| --- | --- |
| Python `Graph`, `Module`, `KernelLibrary` | Symbolic graph and imported kernel declarations |
| Python `InferenceSession` + native engine/context | Compilation request and selected-device runtime context |
| Native `MOFContext` | Clone, compiler configuration, artifact lookup and conversion |
| `GCPipelineOptions`, MLIR pass managers | Configure and execute the ordered compiler transformations |
| `MOToMAPPass`, `MAPToMOGGPass`, `MOGGToMGPPass` | Perform the three named dialect conversions |
| `KGENCompiler`, `ObjectCompiler` | Specialize kernel implementations and emit code |

[Prepare a Graph, Module, or artifact path for asynchronous native compilation. (max/python/max/engine/api.py:969)](sources/max/python/max/engine/api.py#L969) · [Clone the graph module, configure lowering, and consult the native MEF cache. (GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:377)](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)

### 3.2 MO — Modular Operations {#toolchain-q3-2}

**3.2.1 Format and class.** `M::MO::MODialect` is the framework-level graph dialect. Operations carry typed tensor values and attributes. The source dialect definition calls it the frontend input to the graph compiler.

**3.2.2 Work at this level.** `buildMoPipeline` schedules shape/layout inference, symbolic optimization, constant folding when enabled, cleanup, package/kernel verification and device validation. These passes increase the information needed for implementation selection.

**3.2.3 Add-one teaching view.** C1 still describes “apply the registered add_one operation to float32 X[5,10] on the selected device; return Y with the declared type.” It contains tensor contracts and symbolic relationships rather than random X values.

**3.2.4 Artifact distinction.** MO is an in-memory representation. A `.mlir` file can be a diagnostic dump of it. It is not a mandatory file produced by `bazelw build :addition`. [MO dialect definition (GraphCompiler/include/GraphCompiler/MODialect/MODialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MODialect/MODialect.td#L1) · [Prepare and optimize high-level graph operations before MAP lowering. (GraphCompiler/lib/GraphCompiler.cpp:219)](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [Dialect names and current MAP fusion design (GraphCompiler/docs/GraphCompiler.md:1)](sources/GraphCompiler/docs/GraphCompiler.md#L1)

### 3.3 MO → MAP — Modular Access Primitives {#toolchain-q3-3}

#### 3.3.1 The conversion pass {#toolchain-q3-3-1}

`gcPipelineImpl` schedules `MO::createMarkStableTensorsPass()`, then `MO::createMOToMAP()`. `MOToMAPPass::runOnOperation` creates a `MOGG::MojoRegistry`, collects lowerable operations and calls `lowerOp`. Unsupported ranks or failed lowering signal pass failure.

The result exposes implementation-oriented accesses and kernel structure. It lets later passes reason about how computation loads values and stores outputs. [MOToMAPPass walks and lowers graph operations (GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:764)](sources/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp#L764) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)

#### 3.3.2 MAP's operation families and fusion {#toolchain-q3-3-2}

| MAP family | Information made visible |
| --- | --- |
| `map.iter.opaque` | A hand-written Mojo kernel and a store epilogue/fusion interface |
| `map.iter.elementwise` | Per-index loads, elementwise calls and stores |
| `map.load`, `map.store`, `map.call.elementwise` | The data accesses and arithmetic used by an explicit body |

`buildMapPipeline` schedules `MAP::createFuseKernels`, optionally `createFuseMutableLoads`, accumulation-buffer allocation and structured-body outlining. Fusion eligibility depends on implementation interfaces and data relationships.

In the epilogue example, `AddOneElementwise(ElementwiseUnaryOp)` explicitly supplies a fusible `x + 1` body beside matmul and multiplication. The simple `AddOne.execute` supplies its hand-written `foreach` routine. The animations show these structures schematically; they are not captured MAP dumps or proof of a particular fusion result. [MAP dialect and access-operation families (GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPDialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPDialect.td#L1) · [Fuse eligible structured computations and outline kernel bodies. (GraphCompiler/lib/GraphCompiler.cpp:338)](sources/GraphCompiler/lib/GraphCompiler.cpp#L338) · [Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1)

### 3.4 MAP → MOGG — Modular Operation Generator Graph {#toolchain-q3-4}

**3.4.1 Conversion owner.** `MAP::createMAPToMOGG(options)` constructs `MAPToMOGGPass`. Its pass creates `MojoRegistry` and `MAPToMOGGState` objects to lower outlined opaque kernels and shape-function invocations.

**3.4.2 New representation.** MOGG expresses generator/kernel choices, calls, parameter information and implementation interfaces. The graph now has code-generation-oriented routine boundaries while keeping its input/output dependencies.

**3.4.3 Finishing passes.** `buildMoggPipeline` legalizes rowwise kernels, conforms call operands, outlines routines and adds trace wrapping. These are compiler-generated call interfaces around the authored implementation.

**3.4.4 Current pipeline evidence.** The pipeline performs advanced kernel fusion in MAP before this conversion. Some MOGG dialect prose contains older fusion motivation; the ordered pipeline and the current GraphCompiler design document establish the current placement of that work. [MAPToMOGGPass uses MojoRegistry and MAPToMOGGState (GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:53)](sources/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp#L53) · [Legalize kernel calls and outline routines. (GraphCompiler/lib/GraphCompiler.cpp:370)](sources/GraphCompiler/lib/GraphCompiler.cpp#L370) · [MOGG dialect definition (GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGDialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGDialect.td#L1) · [Dialect names and current MAP fusion design (GraphCompiler/docs/GraphCompiler.md:1)](sources/GraphCompiler/docs/GraphCompiler.md#L1)

### 3.5 MOGG → MGP — Modular Glue Primitives {#toolchain-q3-5}

**3.5.1 Conversion owner.** `MGP::createMOGGToMGP(options)` constructs `MOGGToMGPPass`. `runOnOperation` uses `SymbolTableAnalysis`, orders graphs by their call dependencies and invokes `MOGraphLowering(...).run()`. A dependency cycle stops the pass.

The representation becomes runtime glue: initialized context/state, buffer references, calls, allocation and execution dependencies.

#### 3.5.2 MGP objects and memory/dependency planning {#toolchain-q3-5-2}

| MGP representation | Runtime meaning recorded by the compiler |
| --- | --- |
| `mgp.model` / `MGP::ModelOp` | Model initialization and execution invokables |
| `!mgp.context` | `MGP::Context` with reusable `StateSlot` values |
| `!mgp.buffer` | Byte storage represented by an MLRT `BufferRef` |
| `!mgp.tensor`, tensor spec | Shape/dtype metadata associated with buffer storage |
| Chains and call operands | Dependencies and values used to coordinate runtime work |

`buildMgpPipeline` runs buffer reuse, dynamic memory planning, deferred allocations, model-argument validation, chain simplification, slot globalization and primitive lowering. A slot in this plan is a compiler/runtime resource; it is not a CUDA thread, an IPC queue seat or a physical GPU register.

For add-one, the teaching view records X's input reference, an output resource and the routine/dependency needed to produce Y. Numerical X/Y contents arrive during execution. [MOGGToMGPPass and dependency-ordered MOGraphLowering (GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2406)](sources/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp#L2406) · [MGP model, context, buffers and state slots (GraphCompiler/include/GraphCompiler/MGPDialect/Dialect/MGPDialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MGPDialect/Dialect/MGPDialect.td#L1) · [Plan runtime resources, lower primitives, and emit Mojo routines. (GraphCompiler/lib/GraphCompiler.cpp:382)](sources/GraphCompiler/lib/GraphCompiler.cpp#L382)

### 3.6 Device-graph finalization {#toolchain-q3-6}

**3.6.1 Ordered preparation.** The MGP pipeline stages device graphs, partitions collectives, computes/simplifies dependencies, generates build stubs and schedules `FinalizeDeviceGraphsPass`.

**3.6.2 Actual guard.** The finalization pass reads `ModelOp` and returns immediately when `getIsDeviceGraph()` is false. For a device-graph model, it finalizes the `DeviceGraphCollectiveOp` operations in its execute block. Scheduling this pass does not establish that every ordinary example has a nonempty device graph.

**3.6.3 Milestones.** Python `Graph.output` verifies the symbolic output contract; this later pass handles compiler device-graph structure; model initialization loads prepared runtime resources; execution submits work. The animation keeps these milestones separate. [Plan runtime resources, lower primitives, and emit Mojo routines. (GraphCompiler/lib/GraphCompiler.cpp:382)](sources/GraphCompiler/lib/GraphCompiler.cpp#L382) · [Device-graph finalization guard and collective audit (GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:375)](sources/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp#L375)

### 3.7 Emit routines, then enter kernel JIT {#toolchain-q3-7}

**3.7.1 Generated source.** `MOGG::createEmitMojo` imports the required standard/custom packages and emits Mojo routine scaffolding for the lowered graph. The emitted source is a compiler resource consumed by the JIT path. The precompiled custom package supplies the authored bodies that the scaffolding references.

**3.7.2 Separate compiler modules.** C1 retains the MGP runtime plan. The JIT entry creates a separate parsed Mojo/KGEN module, KIR1. Rewriting KIR1 toward LLVM does not turn the MGP buffer/dependency plan into GPU assembly.

**3.7.3 Pipeline tail.** On the normal JIT route, `gcPipelineImpl` adds `createJitCompileKernels`, strips kernel declarations and calls `buildJitPipeline`. Final runtime-call lowering and MEF preparation follow compiled-kernel work.



~~~mermaid
flowchart LR
    G1["Python Graph G1 · retained"] -->|clone| C1["C1 · MO"]
    C1 -->|MOToMAPPass| MAP["C1 · MAP"]
    MAP -->|MAPToMOGGPass| MOGG["C1 · MOGG"]
    MOGG -->|MOGGToMGPPass| MGP["C1 · MGP plan"]
    MGP --> FINAL["Conditional device-graph finalization"]
    FINAL --> KIR["EmitMojo + JIT · separate KIR1"]
    KIR --> CODE["compiled host / device routines"]
    CODE --> CALLS["runtime call lowering + MEF bytes"]
    CALLS --> M1["session.init · executable M1"]
~~~

~~~text
G1 retained
  └─ clone C1: MO → MAP → MOGG → MGP
                                 ├─ buffer / dependency / init / execute plan
                                 └─ EmitMojo → separate KIR1 → target code
                         runtime calls + code resources → CM1 → init → M1
~~~

[EmitMojo imports packages and generates routine source (GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:147)](sources/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp#L147) · [MAX kernel JIT entry: parse, LIT lowering, compiler objects and cache routes (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:529)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494) · [Lower compiled loads/calls to runtime operations (GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:102)](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [Finalize runtime chains and outline callables (GraphCompiler/lib/GraphCompiler.cpp:75)](sources/GraphCompiler/lib/GraphCompiler.cpp#L75)

### 4. Mojo specializes kernels and lowers them through its compiler pipeline

  The Mojo compiler’s KGEN pipeline performs specialization and lowering. The LLVM dialect is still MLIR; later, the
  compiler translates that representation to LLVM IR and uses the target backend to emit the target artifact. For
  MAX’s JIT path, this happens through compiler APIs in the process rather than a separate mojo build command for
  each kernel.

  The mojo precompile subprocess above is a separate package-preparation step: it creates .mojoc from source before
  the graph’s compilation pipeline consumes that package.


### 4.1 What JIT means in this example {#toolchain-q4-1}

**4.1.1 Definition.** JIT means **just-in-time compilation**: the running application prepares native code from compiler representations for the required implementation/target. The timing can be explicit preparation before an inference call.

**4.1.2 Concrete MAX path.** In `addition.py`, the order is `Graph(...)`, `InferenceSession(...)`, `session.compile(graph)`, `session.init(compiled)`, creation of numerical X, then `model.execute(x)`. Kernel JIT is reached during graph compilation. This example does not depend on first-execute lazy compilation.

**4.1.3 Actual entry and objects.** `JitCompileKernelsPass::compileMojoToBinary` retrieves target information, parses generated Mojo, runs LIT-to-KGEN work, creates a `KGENCompiler` and `ObjectCompiler`, and stores them with the MGP/KGEN modules in `CompilationState`.

**4.1.4 Miss/hit behavior.** `handleCacheMisses` calls `compiler->runElaborationPipeline`, then `objectCompiler->emitArchive`. It records a compiled-load resource. `handleCacheHits` loads compatible compiled archives. The surrounding graph/runtime plan still needs preparation. A graph-artifact hit is a separate, broader cache route.

~~~text
Running Python process
  session.compile(G1)
    native GraphCompiler passes on C1
      EmitMojo
        compileMojoToBinary
          parse → LIT-to-KGEN → compiled-kernel lookup
          miss: KGENCompiler.runElaborationPipeline
                ObjectCompiler.emitArchive → compiled resources
          hit:  load compatible compiled resources
      runtime-call lowering → prepared CM1
  session.init(CM1) → M1
  model.execute(Buffer X) → result Buffer Y
~~~

[Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [MAX kernel JIT entry: parse, LIT lowering, compiler objects and cache routes (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:529)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [Elaborate and compile cache-miss routines (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:435)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L435) · [Load compatible compiled-kernel archives (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:507)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L507) · [Native graph-artifact cache hit and miss (GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:377)](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377)

### 4.2 Which code performs specialization? {#toolchain-q4-2}

**4.2.1 Compiler pipeline.** `Mojo/lib/Compiler/KGENCompiler.cpp` implements `KGENCompiler::runElaborationPipeline`. It configures an MLIR pass manager, supplies target information and invokes `populateElaborateModulePasses`. That builder calls `buildElaborateModulePipeline` and `buildPostElaborationPipeline`.

The elaboration machinery instantiates generator functions/types with concrete compile-time values. The compiler walkthrough identifies `Mojo/lib/Elaborator/Elaborator.cpp` and its IR evaluators; the KGEN compiler supplies the `compileOffloads` callback used when specialization encounters device offloads.

#### 4.2.2 Add-one specialization example {#toolchain-q4-2-2}

| Authored parameter / operation | Concrete information used during preparation |
| --- | --- |
| `AddOne.execute[target]` | `"cpu"` or `"gpu"` implementation route |
| `InputTensor[dtype=output.dtype, rank=output.rank, ...]` | Output/input dtype and rank; float32, rank 2 in this example |
| `elementwise_add_one[width]` | A width selected by the implementation and target |
| `SIMD[x.dtype, width]` | Concrete scalar/vector type |
| `x.load[width](idx) + 1` | Typed load and native arithmetic |
| `foreach[target=target]` | The chosen target's iteration/dispatch machinery |

The animation's small SIMD width is illustrative. The full graph has 50 elements; a source walkthrough alone does not establish a captured vector width, launch count or instruction sequence.

**4.2.3 IR result.** Parametric `kgen.generator`/`kgen.struct.generator` forms become concrete `kgen.func`/`kgen.struct.instance` forms. Compile-time parameter expressions are evaluated; runtime tensor values remain runtime inputs. [KGENCompiler runs the elaboration pass manager (Mojo/lib/Compiler/KGENCompiler.cpp:1140)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) · [Elaboration and post-elaboration pipeline builders (Mojo/lib/Compiler/KGENCompiler.cpp:967)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L967) · [Mojo compile-time specialization (Mojo/docs/compiler/MojoCompilerWalkthrough.md:428)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L428) · [Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1)

### 4.3 Post-elaboration optimization {#toolchain-q4-3}

**4.3.1 Actual code.** `Mojo/lib/Compiler/Pipeline/Pipeline.cpp` implements `buildPostElaborationPipeline`. It applies floating-point policy, verifies concrete IR, updates materialized compile-time storage, then invokes the first and late optimization pipelines.

**4.3.2 Purpose.** This stage simplifies concrete code, lowers argument/calling conventions and enables optimizations such as scalar replacement, promotion and inlining. It operates on the specialized implementation; it does not invent the tiled matmul algorithm authored in the example or execute the tensor's arithmetic in Python. [Concrete-IR verification, materialization and optimization (Mojo/lib/Compiler/Pipeline/Pipeline.cpp:154)](sources/Mojo/lib/Compiler/Pipeline/Pipeline.cpp#L154) · [Mojo compile-time specialization (Mojo/docs/compiler/MojoCompilerWalkthrough.md:428)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L428)

### 4.4 Which code lowers to LLVM dialect? {#toolchain-q4-4}

**4.4.1 Caller.** `ObjectCompiler::lowerAllFuncsToLLVM` configures a pass manager and calls `buildLowerToLLVMPipeline`.

**4.4.2 Exact builder.** `Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp` schedules:

| Pass / action | Representation change |
| --- | --- |
| `createLowerKGENToLLVM` | KGEN functions/types toward LLVM-compatible forms |
| Runtime-closure lowering and POP legalization | Make closures and primitive operations lowerable |
| `createLowerGlobalPOPToLLVM`, `createLowerPOPToLLVM` | Primitive operations and globals into LLVM dialect |
| `createLowerControlFlow` | Lower remaining control-flow forms |
| Cast reconciliation, suspension-point lowering, cleanup | Finish legal LLVM-compatible MLIR |
| `createTargetSpecificLLVMLowering` | Late target-specific MLIR lowering |
| Debug-info lowering | Translate debug representation for LLVM export |

**4.4.3 Format.** LLVM dialect is still **MLIR**: MLIR operations, types, regions and an MLIR context. For example the conceptual SIMD float32 type can become an LLVM-compatible vector representation. A `.pre-llvm-ir.mlir` diagnostic dump is still MLIR. [MLIR lowering, optional dumps and LLVM-module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:816)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816) · [Exact KGEN / POP / control-flow to LLVM-dialect passes (Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp:26)](sources/Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp#L26) · [KGEN / POP → LLVM dialect (Mojo/docs/compiler/MojoCompilerWalkthrough.md:542)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L542)

### 4.5 Which code translates to LLVM IR? {#toolchain-q4-5}

**4.5.1 Exact API.** The file-local `translateModuleToLLVMIR` helper in `ObjectCompiler.cpp` invokes `mlir::translateModuleToLLVMIR(module, ctx, moduleName)`. It produces a separate `std::unique_ptr<llvm::Module>` using an LLVM context.

**4.5.2 Boundary between IR frameworks.** KIR1 before translation is an MLIR module containing LLVM-dialect operations. LLVM1 after translation is an `llvm::Module`. Textual LLVM IR can be exported as `.ll`; LLVM bitcode can be exported as `.bc`. These exports are optional views of compiler data.

**4.5.3 Code generation.** LLVM optimization and `runLlcPasses` use the selected `llvm::TargetMachine`. The helper creates/runs LLVM code-generation passes in the process. The C++ callback name `runLlc` does not establish that a standalone `llc` executable is launched for each kernel.



~~~mermaid
flowchart LR
    PKG["P1 · .mojoc / post-parse MLIR"] --> KGEN["KIR1 · parametric KGEN"]
    KGEN -->|KGENCompiler + elaborator| CONCRETE["concrete KGEN / POP"]
    CONCRETE -->|buildLowerToLLVMPipeline| DIALECT["LLVM dialect · MLIR"]
    DIALECT -->|mlir::translateModuleToLLVMIR| LLVM["LLVM1 · llvm::Module"]
    LLVM -->|LLVM target codegen| BYTES["object / PTX bytes"]
~~~

~~~text
post-parse package + generated routines
  → parametric KGEN → elaboration → concrete KGEN / POP
  → LLVM dialect (MLIR module)
  → translation → LLVM IR (llvm::Module)
  → target machine / backend → compiled bytes
~~~

[MLIR → LLVM module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:796)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L796) · [MLIR lowering, optional dumps and LLVM-module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:816)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816) · [In-process LLVM machine-code pass manager (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:277)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L277)

### 5. TargetBackendRegistry is a compiler-side C++ registry

  It is declared in TargetBackend.h (Mojo/include/Mojo/Compiler/Target/TargetBackend.h:338) and implemented in
  TargetBackend.cpp (Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:108). It holds registered
  TargetBackend implementations and looks one up using an LLVM target triple.

  Backends register themselves during static initialization. Depending on the build, these include the host backend
  and available device backends such as NVPTX and AMDGPU; other backends can also be present. It is a registry of
  backend implementation objects in the compiler, not a Python dictionary of devices.


### 5.1 Where is the registry, and what does it contain? {#toolchain-q5-1}

**5.1.1 Declaration.** `Mojo/include/Mojo/Compiler/Target/TargetBackend.h` declares `M::KGEN::TargetBackendRegistry`. Its private storage is `std::vector<std::unique_ptr<TargetBackend>> Backends`: owned C++ implementation objects.

**5.1.2 Initialization.** A `RegisterTargetBackend<BackendT>` static object constructs a backend and calls `TargetBackendRegistry::get().add(...)`. Host, NVPTX and AMDGPU source files contain these registrations. Available implementations depend on which backends the compiler build includes.

**5.1.3 Lookup.** `Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp` iterates the registered implementations and calls `resolve(triple)`. It prefers an explicitly resolved backend over a direct self-match. A missing supported implementation produces an error. This is compiler dispatch over target triples.

| Compiler implementation | Source / registration evidence |
| --- | --- |
| `HostBackend` | `Target/Host/HostBackend.h` and `.cpp` |
| `NVPTXBackend` | `Target/max_license/NVPTX/NVPTXBackend.cpp` |
| `AMDGPUBackend` | `Target/max_license/AMDGPU/AMDGPUBackend.cpp` |
| Other compiled-in backends | Resolved by their registered implementation; this table focuses on the requested CPU/NVIDIA/AMD routes |

[Owned backend vector and static registration helper (Mojo/include/Mojo/Compiler/Target/TargetBackend.h:339)](sources/Mojo/include/Mojo/Compiler/Target/TargetBackend.h#L339) · [TargetBackendRegistry resolves the LLVM triple (Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:114)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp#L114) · [Host backend interface and target resolution (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.h:1)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.h#L1) · [Object emission and linking (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:54)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp#L54) · [NVIDIA backend registration, PTX and CUDA compilation (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:1)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L1) · [AMDGPU target and ELF linking (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:77)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77)

### 5.2 How is the backend selected? {#toolchain-q5-2}

**5.2.1 Target record.** `TargetInfoAttr` describes a triple, architecture, features, ABI, relocation model and data layout. `compileOffloads` groups device work by target, copies those fields to `CompilationOptions` and compiles a module sliced for that target.

**5.2.2 Mojo backend selection.** `ObjectCompiler::create` calls `TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple))`. The backend is resolved before its lowering/emission hooks are used.

**5.2.3 LLVM machine selection.** `defaultCreateTargetMachine` then uses the separate LLVM `TargetRegistry::lookupTarget` and `createTargetMachine(triple, targetCpu, targetFeatures, ...)`. ABI, relocation, code model and optimization settings also feed machine construction. The Mojo backend controls the artifact route; the LLVM target machine supplies the target's instruction/code-generation machinery.

| Illustrative target record | Selected backend / role |
| --- | --- |
| `x86_64-unknown-linux-gnu` + a compatible CPU/features | `HostBackend` for Linux host code |
| `nvptx64-nvidia-cuda` + `sm_90a` | `NVPTXBackend` for an NVIDIA device image |
| `amdgcn-amd-amdhsa` + `gfx942` | `AMDGPUBackend` for an AMD device code object |

The NVIDIA/AMD records come from the repository's cross-codegen test. The animation does not report that those devices are present or that the test was executed. [Per-target triple, architecture and backend lookup (Mojo/lib/Compiler/KGENCompiler.cpp:213)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L213) · [Compile each offload for its selected target triple (Mojo/lib/Compiler/KGENCompiler.cpp:516)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L516) · [Select target backend (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:93)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L93) · [LLVM TargetRegistry and triple / CPU / features / ABI (Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:63)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp#L63) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1)

### 5.3 NVIDIA: LLVM NVPTX → PTX → cubin {#toolchain-q5-3}

#### 5.3.1 Emit PTX {#toolchain-q5-3-1}

With the `nvptx64-nvidia-cuda` triple and a selected SM architecture, `NVPTXBackend::emitObject` calls `ctx.runLlc(..., createObjectFile=false)`. The LLVM NVPTX backend emits **PTX text** into a byte buffer. PTX is the virtual device instruction representation consumed by NVIDIA's compilation tooling.

#### 5.3.2 Compile PTX for the same SM {#toolchain-q5-3-2}

The backend reads `ctx.tm.getTargetCPU()`, verifies that it starts with `sm_`, creates `Driver::CompilationDevice` with CUDA API and that architecture, then calls `compileFunction(ptx, debugLevel, optimizationLevel)`. The returned image is a **cubin**, containing native device code for that target.

The source explicitly ties cubin preparation to the SM used to emit PTX. Its Linux branch documents the NVPTXCompiler API and a configurable compiler-path override. The source boundary is the `CompilationDevice` call; it does not require every run to launch `nvcc` or a standalone `ptxas` process.

**5.3.3 Concrete source example.** The cross-codegen test requests `default-device-target-triple=nvptx64-nvidia-cuda` and `default-device-target-arch=sm_90a`, then inspects a generated `.ptx` diagnostic file for the LLVM NVPTX marker. This is a compiler-test specification, not a measured inference run.

~~~text
Mojo/lib/Compiler/ObjectCompiler/
  ObjectCompiler.cpp
  Target/TargetBackend.cpp
  Target/max_license/NVPTX/NVPTXBackend.cpp

TargetInfo(nvptx64-nvidia-cuda, sm_90a, features)
  → NVPTXBackend + LLVM TargetMachine
  → PTX bytes → CUDA CompilationDevice.compileFunction
  → cubin bytes → later module loading / kernel launch
~~~

[LLVM PTX → CUDA CompilationDevice → cubin (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:152)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L152) · [NVIDIA backend registration, PTX and CUDA compilation (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:1)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L1) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1)

### 5.4 AMD: LLVM AMDGPU → object → linked ELF / hsaco {#toolchain-q5-4}

**5.4.1 Target and implementation.** `AMDGPUBackend` inherits the host backend's object-emission mechanism and supplies AMD target traits, offload behavior, shared-memory address-space information and device-library/link hooks. A target such as `amdgcn-amd-amdhsa` + `gfx942` selects this route.

**5.4.2 Device-library preparation.** Its `linkRuntimeLibraries` constructs a `ROCDLTargetAttr` and uses `AMDGPUModuleLinker` for required ROCm bitcode. The shown ASAN case requests OCKL/OCML and `/opt/rocm/amdgcn/bitcode/asanrtl.bc`; those ASAN libraries are conditional rather than universal inputs for add-one.

**5.4.3 Machine code and link.** LLVM AMDGPU code generation emits target object bytes. The link step resolves the device object into a loadable ELF code object, commonly called an `hsaco`. The AMD backend adds `--no-undefined` to catch unresolved symbols during the link.

**5.4.4 Concrete source example.** The repository test selects `default-device-target-triple=amdgcn-amd-amdhsa`, `default-device-target-arch=gfx942` and checks an `.amdgcn` assembly diagnostic for its target marker.

~~~text
Mojo/lib/Compiler/ObjectCompiler/
  ObjectCompiler.cpp
  Target/TargetBackend.cpp
  Target/Host/HostBackend.cpp
  Target/max_license/AMDGPU/AMDGPUBackend.cpp

TargetInfo(amdgcn-amd-amdhsa, gfx942, features)
  → AMDGPUBackend + LLVM TargetMachine
  → required device bitcode linked into LLVM module
  → AMDGPU target .o bytes → device ELF link
  → loadable code object / hsaco → later device launch
~~~

[AMDGPU target and ELF linking (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:77)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77) · [Object emission and linking (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:54)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp#L54) · [In-process LLVM machine-code pass manager (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:277)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L277) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1)

### 5.5 CPU route and file-format comparison {#toolchain-q5-5}

`HostBackend::emitObject` calls `ctx.runLlc(..., createObjectFile=true)` and `ctx.linkObject`. Host objects are linked into loadable native code. The JIT archive path also uses its MC-linking flow; the backend's `createArchive` hook itself is marked unwired in this revision.

| Representation | What it contains / when it appears |
| --- | --- |
| `.mojo` | Source code |
| `.mojoc` | Precompiled post-parse Mojo package / MLIR bytecode |
| MO / MAP / MOGG / MGP | In-memory MAX graph dialect stages; optional `.mlir` dumps |
| LLVM dialect | MLIR representation suitable for LLVM translation |
| LLVM IR / `.ll` / `.bc` | LLVM module / optional text or bitcode export |
| CPU `.o` → native linked code | Host instruction bytes and link resources |
| PTX → cubin | NVIDIA virtual instruction text → native device image |
| AMDGPU `.o` → ELF / hsaco | AMD device object → loadable device image |
| MEF / compiled artifact | Runtime plan and code/resources prepared for initialization |

Object emission does not require writing an assembly file first. Text assembly is an optional output; NVIDIA's PTX route explicitly uses a textual intermediate. The file sheets in the animation represent in-memory compiler data unless a dump/export is named. [Object emission and linking (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:54)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp#L54) · [LLVM PTX → CUDA CompilationDevice → cubin (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:152)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L152) · [AMDGPU target and ELF linking (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:77)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77) · [CompiledModel (max/python/max/engine/api.py:414)](sources/max/python/max/engine/api.py#L414)

### 6. The selected graph device determines the kernel’s target

  In addition.py, the example chooses CPU() when no accelerator is available and otherwise uses an Accelerator(). It
  assigns that device to the graph and session. The graph’s device target information then guides kernel compilation
  and backend selection.

  The Python process, MAX runtime, and device driver remain host-side. For a CPU target the host backend emits host
  code; for a GPU target a device backend emits GPU-target code, which the host runtime can submit with its buffers
  to the device. The example’s Mojo kernel has a compile-time target parameter and passes it to foreach, so its
  implementation specializes for the chosen target.


### 6.1 Select graph placement {#toolchain-q6-1}

**6.1.1 Add-one source choice.** The unchanged example uses:

~~~python
device = CPU() if accelerator_count() == 0 else Accelerator()
~~~

It supplies `DeviceRef.from_device(device)` to the custom operation and tensor types, then constructs `InferenceSession(devices=[device])`. The compiler's MO device-validation/configuration work supplies the corresponding target information.

**6.1.2 One selected arithmetic device.** This script applies add-one on its chosen CPU or accelerator. A GPU route still has a CPU application and host launch/control routines. A graph can describe multiple placements; this example's single operation has one explicit placement. [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Prepare and optimize high-level graph operations before MAP lowering. (GraphCompiler/lib/GraphCompiler.cpp:219)](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)

### 6.2 Separate host routines from device offloads {#toolchain-q6-2}

**6.2.1 Authored dispatch.** `AddOne.execute[target]` passes its compile-time target to `foreach`. CPU iteration and GPU dispatch are target-specific implementations of that operation.

**6.2.2 Compiler mechanism.** Mojo elaboration can encounter device offload compilation. `compileOffloads` receives a map keyed by `TargetInfoAttr`, slices the relevant generators into a target-specific module, applies that group's options and compiles the group. The host module retains control/launch code and references to the resulting device image.

**6.2.3 Why the Python program stays on the host.** The compiler only sends selected native routine IR to a device backend. Python constructs the graph and supplies buffers; native host routines interact with device APIs. Device image bytes contain the selected kernels, with the target's ABI, address spaces and entry metadata.



~~~mermaid
flowchart TD
    APP["CPU process · Python + MAX"] --> HOST["host routine IR / HostBackend"]
    HOST --> LAUNCH["host driver launch + Buffer references"]
    HOST -->|offload generator + TargetInfo| DEV["separate device IR"]
    DEV --> NV["nvptx64 / sm → PTX → cubin"]
    DEV --> AMD["amdgcn / gfx → object → ELF"]
    NV --> IMAGE["resident compatible device image"]
    AMD --> IMAGE
    IMAGE --> GPU["physical GPU executes kernels"]
    LAUNCH --> GPU
~~~

~~~text
CPU process: Python → MAX host routines → driver API
                    │
                    ├─ host code / runtime state stays on CPU
                    └─ selected offload IR → GPU backend → device image
                                            │
host launch(kernel handle, device Buffers) ───┴─→ GPU execution
~~~

[Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1) · [Compile each offload for its selected target triple (Mojo/lib/Compiler/KGENCompiler.cpp:516)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L516) · [Per-target triple, architecture and backend lookup (Mojo/lib/Compiler/KGENCompiler.cpp:213)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L213) · [Driver execution (max/lib/max_license/API/c/ModelExecution.cpp:192)](sources/max/lib/max_license/API/c/ModelExecution.cpp#L192)

### 6.3 Finish the runtime call interface {#toolchain-q6-3}

**6.3.1 Prepared placeholders.** Kernel compilation creates compiled-load resources and compiled routine calls in the lowered module.

**6.3.2 Lowering owner.** `LowerCompiledRoutinesPass` converts these into module loading and runtime execution/call operations such as `ModuleLoad` and `ExecuteGeneric`. `buildJitPipeline` then inlines/cleans up MGP calls, simplifies chains and outlines model callables.

**6.3.3 Resulting contract.** Initialization knows which modules/callables to load. Each execution knows which input references and dependencies feed those callables. This is the graph's runtime interface around compiled routines. [Lower compiled loads/calls to runtime operations (GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:102)](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [Finalize runtime chains and outline callables (GraphCompiler/lib/GraphCompiler.cpp:75)](sources/GraphCompiler/lib/GraphCompiler.cpp#L75)

### 6.4 Prepare the compiled artifact / MEF bytes {#toolchain-q6-4}

**6.4.1 Artifact owner.** The engine API exposes a `CompiledModel` wrapper for prepared compilation. Its runtime plan, code resources and metadata belong to CM1; G1 remains the symbolic graph.

**6.4.2 In-memory preparation.** The example assigns `compiled = session.compile(graph)`. It does not explicitly export a `.mef` disk file. A prepared byte buffer is not yet the initialized executable model.

**6.4.3 Data separation.** There are no model weights to bind in these four tensor-input examples. Their numerical arrays are passed to execution. [CompiledModel (max/python/max/engine/api.py:414)](sources/max/python/max/engine/api.py#L414) · [Serialize MEF (GraphCompiler/lib/GraphCompiler.cpp:692)](sources/GraphCompiler/lib/GraphCompiler.cpp#L692) · [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1)

### 6.5 Initialize M1 {#toolchain-q6-5}

**6.5.1 Actual call.** `model = session.init(compiled)` initializes the prepared artifact with the session's runtime/device context.

**6.5.2 Runtime consequences.** Module-load and callable initialization resolve resident code/resources and executable handles. M1 can then accept input Buffers. CM1's compiled bytes, M1's initialized handles and S1's session context are separate identities in the scene.

**6.5.3 Reuse.** The initialized model can be retained for repeated executions with fresh inputs. The matmul example's helper specifically compiles/initializes a new model per helper invocation while its main function reuses the session; the diagram does not change that source ordering. [Validate weight buffers and initialize models in a compiled artifact. (max/python/max/engine/api.py:1160)](sources/max/python/max/engine/api.py#L1160) · [Lower compiled loads/calls to runtime operations (GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:102)](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [Complete Python example (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1)

### 6.6 Supply buffers and execute {#toolchain-q6-6}

**6.6.1 Input creation.** For add-one, `Buffer.from_numpy(x_values).to(device)` supplies numerical storage to the selected device. Wrapping/copying data and compilation are distinct operations.

**6.6.2 Actual call.** `model.execute(x)` hands Buffer references to native execution. MLRT's model/executor dependency machinery coordinates ready operations and result values. This is native graph-runtime work, independent of Serve's request batching scheduler.

**6.6.3 Host/device action.** A CPU routine operates on CPU storage. A GPU routine's host launch references resident device code and accelerator Buffers. Actual driver queues, scheduling overlap and elapsed time are beyond this browser teaching model. [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Model.execute (max/python/max/engine/api.py:118)](sources/max/python/max/engine/api.py#L118) · [Native input preparation (max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:451)](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L451) · [Driver execution (max/lib/max_license/API/c/ModelExecution.cpp:192)](sources/max/lib/max_license/API/c/ModelExecution.cpp#L192)

### 6.7 Obtain result readiness and host-readable values {#toolchain-q6-7}

**6.7.1 Readiness.** The runtime returns/coordinates result Buffers through dependency/readiness state. The executor tracks operands that are not ready and triggers dependent work as their values become available.

**6.7.2 Add-one read.** The Python source calls `result.to(CPU())`, then `result.to_numpy()`. GPU values must become CPU-readable through that explicit operation. A returned Buffer handle by itself is not a copied NumPy array.

**6.7.3 Verification limits in the originals.** Addition prints result and expected values without a numerical equality assertion. The epilogue and matmul scripts contain numerical checks. The attention script prints output Buffers without an explicit host copy/parity assertion. The browser's small arithmetic checks verify the teaching model rather than execution of MAX on a GPU. [Ready primitive worklist (MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp:559)](sources/MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp#L559) · [Await / extract outputs (max/lib/max_license/API/c/ModelExecution.cpp:100)](sources/max/lib/max_license/API/c/ModelExecution.cpp#L100) · [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Complete Python example (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1)

### 7. No standalone mlir-opt command is part of the example path

  The example BUILD rules do not declare mlir-opt as a step for compiling add_one.mojo. The graph compiler uses in-
  process MLIR pipelines, while package preparation invokes mojo precompile. The repository has compiler utilities
  for development and debugging, but they are separate from building or running this custom-op example.


### 7.1 Which tool runs at each boundary? {#toolchain-q7-1}

| Step | Tool / implementation actually responsible | Invocation boundary |
| --- | --- | --- |
| Build dependency graph | `./bazelw` → Bazel and declared rules | Build orchestration |
| Shared Mojo packages | Configured Mojo compiler through `mojo_library` | Bazel build action producing `.mojoc` |
| Runtime source-package preparation | `mojo.paths._build_mojo_source_package` → `mojo precompile` | Explicit subprocess in the Python application's package-preparation path |
| MAX MO/MAP/MOGG/MGP transformations | GraphCompiler C++ library + MLIR pass managers | Native compiler APIs in the application process |
| Kernel elaboration | `KGENCompiler` + elaboration pipeline/evaluators | Native compiler APIs and their work queues |
| KGEN/POP → LLVM dialect | `ObjectCompiler` + `buildLowerToLLVMPipeline` | MLIR passes in the compiler |
| LLVM dialect → LLVM IR | `mlir::translateModuleToLLVMIR` | MLIR export API producing `llvm::Module` |
| Target code generation | LLVM `TargetMachine`, `runLlcPasses`, selected backend | LLVM code-generation libraries |
| NVIDIA PTX preparation | CUDA `CompilationDevice::compileFunction` | Driver/compiler implementation boundary; configurable tooling |
| Host / AMD object linking | Object-compiler/link hooks and configured link tools | Link step; precise invocation follows target/tool configuration |
| Initialization / inference | MAX native runtime and device API | Loaded artifact + ready input/result resources |

`mo-opt`, `kgen-opt` and `mlir-opt` are useful compiler development tools. Their existence does not make them required subprocesses in `:addition`. Likewise, `mojo build` is a standalone program-build interface; MAX kernel JIT invokes the corresponding compiler machinery through native APIs.

[Custom-op binary: source data, Mojo libraries and Python dependencies (max/examples/custom_ops/custom_op_example.bzl:12)](sources/max/examples/custom_ops/custom_op_example.bzl#L12) · [Bazel Mojo library precompiles source into a .mojoc package (bazel/internal/mojo_library.bzl:35)](sources/bazel/internal/mojo_library.bzl#L35) · [Actual mojo precompile command and atomic package publication (Mojo/python/mojo/paths.py:141)](sources/Mojo/python/mojo/paths.py#L141) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494) · [KGENCompiler runs the elaboration pass manager (Mojo/lib/Compiler/KGENCompiler.cpp:1140)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) · [Exact KGEN / POP / control-flow to LLVM-dialect passes (Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp:26)](sources/Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp#L26) · [MLIR → LLVM module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:796)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L796) · [In-process LLVM machine-code pass manager (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:277)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L277)

### 7.2 How can each representation be inspected? {#toolchain-q7-2}

**7.2.1 Read the actual definitions first.** Source cards and the offline Source Explorer expose complete unchanged source files, the selected function/pass span, its path and pinned revision.

**7.2.2 Diagnostic files are optional.** Graph-stage dumps can expose MO, MAP, MOGG and MGP views. ObjectCompiler's temporary-output policy can write pre-LLVM-dialect/pre-LLVM-IR `.mlir` views and LLVM/native diagnostics. These are separate from the ordinary example's required build outputs.

**7.2.3 Target examples are source tests.** `GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir` supplies the explicit NVIDIA/AMD triples and architectures shown in point 5. `gc-pipeline-dump.mlir` specifies stage-dump checks. Their `RUN` lines describe diagnostic compiler-test invocations; they have not been executed as part of this HTML update.

**7.2.4 Evidence categories.**

| Evidence shown here | What it establishes |
| --- | --- |
| Pinned repository source | Call ordering, pass construction, target selection and representation ownership |
| Inspected compiler-test source | Expected cross-target/dump behavior specified by the test |
| Deterministic browser state and arithmetic | The animation's transitions, controls and small numerical illustrations |
| Actual MAX / GPU execution | Requires running the real example and recording its outputs on the selected device; no such execution is claimed by this page |

The previous point 1 explicitly records an interrupted cold Bazel build. This update preserves that statement and does not replace it with a successful build claim.

[Source test for graph-stage diagnostic dumps (GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir:1)](sources/GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir#L1) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1) · [MLIR lowering, optional dumps and LLVM-module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:816)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816)
