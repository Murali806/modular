**The original `Graph.__init__` runs at `fn(*args, **kwargs)`. `yield` pauses the context helper so execution can return to the wrapper and reach that call.** The context becomes active earlier, when `with _DEFAULT_MLIR_CONTEXT` invokes its native `__enter__()` method. [Wrapper source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:81)

There are two different times involved: **creating the wrapper function** and **calling that wrapper later**.

```python
def in_default_mlir_context(fn):
    def wrapper(*args, **kwargs):
        with ensure_default_mlir_context():
            return fn(*args, **kwargs)

    return wrapper
```

When Python calls `in_default_mlir_context(original_init)`, the inner `def wrapper` creates a function object. Its body has not run yet. `return wrapper` returns that function object; there are no parentheses to call it. The returned function retains access to the outer variable `fn`. This retained access is called a **closure**. [Decorator source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:70)

The relationship is:

```text
When Python defines the Graph class
│
├─ Create original constructor function F
│
├─ Apply decorator: in_default_mlir_context(F)
│  ├─ fn refers to F
│  ├─ def wrapper creates another function W
│  ├─ W retains access to fn → F
│  └─ return wrapper → return W
│
└─ Store W as Graph.__init__


Later: graph = Graph(...)
│
└─ Python calls W(self, ...)
   └─ W's body now runs
      ├─ with ensure_default_mlir_context():
      └─ return fn(self, ...)
         └─ Call F: the original constructor body
```

`F` and `W` are explanatory names for two function objects. Replacing the class’s `__init__` attribute with `W` does not change the original function object retained through `fn`. See [the decorator attachment](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/graph/graph.py:819) and [the returned wrapper](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:86).

**`with` automatically calls entry and exit methods.** For this code:

```python
with ensure_default_mlir_context():
    return fn(*args, **kwargs)
```

Python obtains a context-manager object, calls its `__enter__()`, executes the indented body, and calls its `__exit__()` before leaving the block—even when leaving through `return`. Because this manager was created using `@contextlib.contextmanager`, its entry method advances the generator to `yield`; its normal exit method resumes the generator after `yield`. [Python context-manager implementation](/usr/lib/python3.10/contextlib.py:123)

Here is the exact handoff for a thread whose current context starts as `None`:

| Step | Code running | What happens |
|---|---|---|
| 1 | `wrapper` | Calls `ensure_default_mlir_context()` and receives a context-manager object. |
| 2 | Manager’s `__enter__()` | Starts the helper’s generator. |
| 3 | Helper: `with _DEFAULT_MLIR_CONTEXT:` | Activates the default context on this thread. |
| 4 | Helper: `yield` | Pauses the helper and lets the manager’s `__enter__()` finish. |
| 5 | Wrapper: `fn(*args, **kwargs)` | **Calls the original `Graph.__init__`.** |
| 6 | Original constructor | Initializes `self` and returns `None`. |
| 7 | Manager’s `__exit__()` | Resumes the helper after `yield`; the helper exits its inner `with`, removing the temporary context scope. |
| 8 | Wrapper | Returns the constructor’s `None`. |

The MAX statements are at [_mlir_context.py:63](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:63) and [83–84](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:83); Python’s generator handoff is in [contextlib.__enter__ and __exit__](/usr/lib/python3.10/contextlib.py:130).

**`yield` does not know about `fn`.** Notice that `ensure_default_mlir_context()` receives no function argument:

```python
@contextlib.contextmanager
def ensure_default_mlir_context() -> Generator[None]:
    if mlir.Context.current is None:
        with _DEFAULT_MLIR_CONTEXT:
            yield
    else:
        yield
```

Its responsibility is to keep the context scope active while the wrapper executes its own block. The suspended generator remains inside `with _DEFAULT_MLIR_CONTEXT` until it resumes. This happens synchronously on the calling thread. [Source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:54)

Your observation about the missing assignment is the key: **`mlir.Context.current` is a read-only property backed by a native thread-local stack.** Entering a context pushes a frame onto that stack. Reading `Context.current` looks up the context in the top frame. There is no Python assignment such as `mlir.Context.current = ...`. [Native property binding](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:3445)

This is the native call path hidden behind the inner `with`:

```text
with _DEFAULT_MLIR_CONTEXT:
│
└─ Python automatically invokes its __enter__()
   └─ PyMlirContext::contextEnter(context)
      └─ PyThreadContextEntry::pushContext(context)
         └─ push(FrameKind::Context, context, ...)
            ├─ getStack()
            │  └─ Return this thread's native stack
            └─ stack.emplace_back(..., context, ...)
               └─ Store a reference to the context in a new frame
```

The source locations are [the `__enter__` binding](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:3439), [contextEnter](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:496), [pushContext](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:653), and [the actual stack insertion](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:598).

The stack is explicitly declared in C++:

```cpp
std::vector<PyThreadContextEntry> &PyThreadContextEntry::getStack() {
  static thread_local std::vector<PyThreadContextEntry> stack;
  return stack;
}
```

**`thread_local` means each thread has its own stack.** The entries can refer to the same shared context object. [Source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:586)

For example, call the shared context object `CTX0`:

```text
Python module:
    _DEFAULT_MLIR_CONTEXT ──────────────────────► CTX0

Worker thread, before entering:
    native context stack = []
    mlir.Context.current → None

Worker thread, inside with _DEFAULT_MLIR_CONTEXT:
    native context stack = [
        frame { context: reference to CTX0 }
    ]
    mlir.Context.current → CTX0

Worker thread, after exiting:
    native context stack = []
    mlir.Context.current → None

The module-level variable still refers to CTX0.
```

The module-level reference is created at [_mlir_context.py:26](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:26). Exit calls `popContext`, which removes the stack frame; it does not remove that module-level reference. [Exit implementation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:660)

When constructor code needs the active context, the lookup goes back through that stack:

```text
MAX native binding: current_context()
└─ Read max.mlir.Context.current
   └─ PyThreadContextEntry::getDefaultContext()
      └─ getTopOfStack()
         └─ Read the top frame's context → CTX0
```

See [MAX’s lookup helper](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_core/max_license/lib/Graph/Bindings.cpp:24) and [the native stack lookup](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:638). The decorator therefore does not need to cache the active-context selection in `Graph`; `Graph` separately retains its module and operation handles. [Graph storage](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/graph/graph.py:855)

One final distinction explains the return values:

```python
with _DEFAULT_MLIR_CONTEXT as x:
    # x is the context object: its native __enter__ returns that object.
    ...

with ensure_default_mlir_context() as x:
    # x is None: this manager's generator uses a bare yield.
    ...
```

Both can activate the same context, but their entry return values differ. Neither return value is needed by the actual wrapper, which uses no `as x`. [Native context entry return](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/bazel-modular/external/+llvm_source+llvm-raw/mlir/lib/Bindings/Python/IRCore.cpp:653) · [MAX helper’s bare yield](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:65)