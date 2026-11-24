**The call comes from the decorator wrapper around `Graph.__init__`.** That wrapper activates an MLIR context, calls the original constructor, then exits the context scope. See [the decorator on Graph.__init__](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/graph/graph.py:819) and [the wrapper implementation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:81).

When Python defines the class, this:

```python
@in_default_mlir_context
def __init__(self, ...):
    ...
```

is conceptually equivalent to:

```python
def __init__(self, ...):
    ...

__init__ = in_default_mlir_context(__init__)
```

The decorator receives the original constructor as `fn` and returns a new function, `wrapper`. Python stores that wrapper as `Graph.__init__`. This happens when the class is defined; it does not construct a graph yet. ([docs.python.org](https://docs.python.org/3/reference/compound_stmts.html))

For this synchronous constructor, the relevant decorator logic is equivalent to:

```python
def in_default_mlir_context(fn):
    def wrapper(*args, **kwargs):
        with ensure_default_mlir_context():
            return fn(*args, **kwargs)

    return wrapper
```

Here, **`fn` still refers to the original constructor**. Therefore, `fn(*args, **kwargs)` enters its original body rather than recursively calling the wrapper. The real implementation also uses `functools.wraps(fn)` to preserve function metadata. [Source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:70)

The executable body of `ensure_default_mlir_context()` is:

```python
@contextlib.contextmanager
def ensure_default_mlir_context() -> Generator[None]:
    if mlir.Context.current is None:
        with _DEFAULT_MLIR_CONTEXT:
            yield
    else:
        yield
```

[Source: _mlir_context.py:54](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:54)

**`yield` is where execution pauses while the constructor runs.** The context-manager machinery resumes the generator after the surrounding `with` block finishes. A bare `yield` yields `None`; a hypothetical `with ensure_default_mlir_context() as value:` would therefore bind `value` to `None`. The actual wrapper has no `as` clause. ([docs.python.org](https://docs.python.org/3/library/contextlib.html))

For a thread with no active MLIR context, the source-ordered flow is:

```text
caller: graph = Graph(..., forward=None)
│
└─ Python invokes the decorated Graph.__init__(self, ...)
   └─ wrapper(self, ...)
      │
      ├─ Evaluate ensure_default_mlir_context()
      │  └─ @contextmanager supplies a context-manager object
      │     The generator body has not executed yet.
      │
      ├─ Enter that context manager: __enter__()
      │  └─ Advance its generator to the first yield
      │     ├─ Read mlir.Context.current → None
      │     ├─ Enter with _DEFAULT_MLIR_CONTEXT
      │     │  └─ Make that context current on this thread
      │     └─ yield → supplies None and pauses the generator
      │        The default context remains active.
      │
      ├─ Execute return fn(self, ...)
      │  └─ Original Graph.__init__(self, ...)
      │     ├─ Collect symbolic dimensions
      │     ├─ Construct the module, GraphOp and other graph state
      │     └─ Finish initialization → return None
      │
      ├─ Before wrapper can return, exit its with block
      │  └─ Context-manager __exit__() resumes the generator
      │     ├─ Continue immediately after yield
      │     ├─ Exit with _DEFAULT_MLIR_CONTEXT
      │     │  └─ Restore the thread's previous context state
      │     └─ Generator finishes
      │
      └─ wrapper returns the constructor's None
```

The wrapper and branching behavior come directly from [lines 63–84](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:63); the pause/resume behavior is supplied by Python’s `contextmanager`. ([docs.python.org](https://docs.python.org/3/library/contextlib.html))

There are several different “return values” here:

| Expression or function | What it returns |
|---|---|
| `in_default_mlir_context(original_init)` | The **wrapper function**, installed during class definition. |
| `ensure_default_mlir_context()` | A **context-manager object**. |
| That manager’s `__enter__()` | **`None`**, supplied by the bare `yield`. |
| Original `Graph.__init__(...)` | **`None`**, after initializing `self`. |
| The decorator’s `wrapper(...)` | That same **`None`**, after context cleanup. |

These are visible in the [decorator/context-manager source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:54) and [constructor body](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/graph/graph.py:820).

**The MLIR context reaches the constructor through the thread’s active context state—not through a returned argument.** Consequently, the two branches behave like this: [source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:63)

```text
No context active:
    None → DEFAULT active → original constructor runs → None

An existing context C already active:
    C    → C remains active → original constructor runs → C
```

The wrapper runs on the thread that called `Graph(...)`; it does not create a worker thread or submit work to a thread pool. [Source](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_mlir_context.py:81)
