**`CompiledKernelsContext::load` connects compiled binary code to the runtime’s callable table.** It takes binary bytes, registers them with LLVM’s execution engine, resolves requested symbols, and stores their addresses for later execution.

Below, names, sizes, hashes, and addresses are **illustrative values**, not a dump from running the addition example.

**2.1 Inputs and initial state**

The implementation starts at [CompiledKernelsContext.cpp:85](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Support/CompiledKernelsContext.cpp:85):

```cpp
ErrorOrSuccess CompiledKernelsContext::load(
    ArrayRef<uint8_t> binary,
    ArrayRef<StringRef> kernelNames,
    size_t kernelIndex
)
```

Example:

```text
binary          = 12,288 bytes of compiled object code
kernelNames     = ["add_one_entry", "shape_entry"]
kernelIndex     = 0
loadDebuginfo   = false

Before loading:
  context.kernels     = []
  context.symbolTable = []
  context.engine      = pointer to the runtime's ExecutionEngine
```

`ArrayRef` describes an array using a pointer and a length. The argument itself does not own the supplied bytes.

The requested names must match symbols in the compiled binary. `"add_one_entry"` here represents a generated native entry point; the actual compiler-generated name may differ.

**2.2 Compute the binary’s namespace name**

```text
load(binary, names, 0)
└─ llvm::xxh3_64bits(binary)
   └─ llvm::utohexstr(hash, LowerCase=true, Width=16)
```

Example values:

```text
IN:
  binary = the complete 12,288-byte binary

Intermediate:
  hash = illustrative 64-bit value 0x8a3f10c492be760d

OUT:
  dylibName = "8a3f10c492be760d"
```

This name identifies the binary inside the execution engine. It is not a filesystem path. See [CompiledKernelsContext.cpp:88](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Support/CompiledKernelsContext.cpp:88).

Why needed:

```text
Binary A:
  namespace "8a3f10c492be760d"
  symbol "add_one_entry"

Binary B:
  namespace "71e5a84203fc9b66"
  symbol "add_one_entry"
```

The symbols have identical names, but lookup specifies which namespace to search.

**2.3 Copy the bytes into owned storage**

```text
load(...)
└─ StringRef(binary.data(), binary.size())
   └─ Buffer::get(StringRef)
      └─ BufferRef::create(data)
         └─ owned, reference-counted binary storage
```

Example:

```text
IN:
  caller-owned bytes at address B0
  length = 12,288

OUT:
  buf = BufferRef
  buf contains a copy of those 12,288 bytes
```

This copy matters because later LLVM materialization can use the bytes after this function’s local variables have gone away.

The ownership and copying behavior are documented directly in [Buffer.h:47](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Support/include/Support/Buffer.h:47).

**2.4 Register the binary once: `addIfAbsent`**

For the ordinary route:

```text
load(...)
└─ engine->addIfAbsent<StaticArchiveLayer>(dylibName, move(buf))
   ├─ findLayer<StaticArchiveLayer>()
   ├─ acquire engine mutex
   ├─ executionSession->getJITDylibByName(dylibName)
   │
   ├─ Namespace already exists
   │  └─ return success without registering another object
   │
   └─ Namespace absent
      └─ StaticArchiveLayer::add(dylibName, buf)
```

Example:

```text
First initialization:
  engine has no "8a3f10c492be760d"
  → register the object

Second initialization using identical binary bytes:
  engine already has "8a3f10c492be760d"
  → reuse existing registration
  → still perform this context's requested symbol lookups
```

The mutex prevents two threads from simultaneously registering the same namespace through this route. See [ExecutionEngine.h:153](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/include/Mojo/ExecutionEngine/ExecutionEngine.h:153).

**2.5 Create the `JITDylib` and connect runtime dependencies**

```text
StaticArchiveLayer::add(name, buf)
└─ MaterializationLayer::getOrCreateDylib(name)
   ├─ session.getJITDylibByName(name)
   └─ if absent:
      ├─ session.createJITDylib(name)
      └─ ExecutionEngine::addToSearchOrder(name, dylib)
         ├─ add CompilerRT library to its link order
         ├─ add mlirc library to its link order
         └─ add this dylib to engine search order
```

A **JITDylib is LLVM ORC’s symbol namespace and linking unit**. In this ordinary route, creating one does not create a `.so` file.

Example state:

```text
JITDylib "8a3f10c492be760d"
├─ definitions supplied by our object
│  ├─ add_one_entry
│  └─ shape_entry
└─ link-order dependencies
   ├─ CompilerRT library
   └─ mlirc library
```

If the object calls a runtime helper, its dependency link order gives the linker places to resolve that helper.

Sources: [MaterializationLayer.cpp:40](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/JIT/MaterializationLayer.cpp:40) and [ExecutionEngine.cpp:410](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/ExecutionEngine.cpp:410).

**2.6 Read the object’s interface and register a materialization unit**

```text
StaticArchiveLayer::add(...)
├─ objectBuffers.push_back(object.copy())
│    Retain another reference to the owned bytes
│
├─ dylib->getDefaultResourceTracker()
│    Obtain the resource tracker for this registration
│
├─ object->getMemBufferRef()
│    Create a view of the retained binary bytes
│
├─ llvm::orc::getObjectFileInterface(session, objectBuffer)
│    Read the symbols/interface supplied by the object
│
└─ dylib->define(
       StaticArchiveObjectMaterializationUnit(
           objectLayer, objectBuffer, objectInterface),
       resourceTracker
   )
```

Conceptual example:

```text
Object interface:
  "add_one_entry" → definition available in this object
  "shape_entry"   → definition available in this object

Registered materialization unit:
  objectBuffer = view of the 12,288 retained bytes
  genLayer     = execution engine's ObjectLinkingLayer
```

At this stage, ORC knows **where the definitions can come from**. Registering that information does not execute either function.

Despite its name, this implementation of `StaticArchiveLayer` calls an **object-file interface reader**; this code does not show a loop extracting `.a` archive members.

See [StaticArchiveLayer.cpp:81](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/JIT/StaticArchiveLayer.cpp:81).

**2.7 Look up the first requested symbol**

```text
load(...)
└─ engine->lookup("8a3f10c492be760d", "add_one_entry")
   ├─ executionSession->getJITDylibByName(namespace)
   └─ lookupWithSearchOrder([that JITDylib], symbol)
      ├─ mangleAndIntern(symbol)
      │  ├─ llvm::Mangler::getNameWithPrefix(..., dataLayout)
      │  └─ executionSession->intern(mangledName)
      └─ executionSession->lookup(searchOrder, internedName)
```

Example name conversion:

```text
Requested name: "add_one_entry"

Typical ELF target:
  linker name → "add_one_entry"

A target with "_" as its global prefix:
  linker name → "_add_one_entry"
```

The engine’s `DataLayout` determines the prefix. Interning puts the spelling into ORC’s shared symbol-string pool.

The output is either a resolved symbol address or an error. See [ExecutionEngine.cpp:373](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/ExecutionEngine.cpp:373) and [ExecutionEngine.cpp:425](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/ExecutionEngine.cpp:425).

**2.8 Lookup can trigger object materialization and linking**

The callback implemented in this repository is:

```text
ORC lookup needs a registered definition
└─ StaticArchiveObjectMaterializationUnit::materialize(responsibility)
   └─ genLayer.emit(
          responsibility,
          MemoryBuffer::getMemBuffer(objectBuffer))
      └─ LLVM ObjectLinkingLayer processes the object
```

See [StaticArchiveLayer.cpp:53](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/JIT/StaticArchiveLayer.cpp:53).

The linking work below `emit` can be understood with this illustrative object:

```text
Before linking:
  .text:
    add_one_entry machine instructions
    an instruction referring to runtime_helper

  relocation:
    patch that instruction using runtime_helper's resolved address

After linking:
  add_one_entry → executable address P0
  runtime_helper reference → resolved dependency address R0
```

Conceptually, the linker places code/data in memory, resolves dependencies, applies relocations, and makes the definitions available for lookup. This is **linking previously compiled machine code**, rather than rerunning the Mojo parser or MLIR lowering pipeline.

The engine constructs an LLVM `ObjectLinkingLayer` with a JIT-link memory manager in [ExecutionEngine.cpp:197](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/lib/ExecutionEngine/ExecutionEngine.cpp:197).

**2.9 Convert the resolved address into a stored callable**

```text
ExecutionEngine::lookupWithSearchOrder(...)
└─ ORC returns ExecutorSymbolDef
   └─ symbol.getAddress().toPtr<void*>()
      └─ CompiledFunc(pointer)

Back in load(...)
└─ CompiledFunc::getFunctionPointer()
   └─ addKernel(name, kernelIndex, pointer)
      ├─ assert kernels.size() == kernelIndex
      ├─ kernels.push_back(pointer)
      └─ symbolTable.push_back(name)
```

Example after the first symbol:

```text
Resolved address P0 = illustrative 0x00007f1200100000

kernels     = [P0]
symbolTable = ["add_one_entry"]

kernelIndex changes:
  0 → 1
```

After the second:

```text
Resolved address P1 = illustrative 0x00007f1200100080

kernels     = [P0, P1]
symbolTable = ["add_one_entry", "shape_entry"]

kernelIndex:
  1 → 2
```

The starting index must equal the current table size. For example, appending to a table containing two entries requires `kernelIndex=2`.

Sources: [ExecutionEngine.h:71](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/Mojo/include/Mojo/ExecutionEngine/ExecutionEngine.h:71) and [CompiledKernelsContext.cpp:165](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Support/CompiledKernelsContext.cpp:165).

**2.10 Later execution retrieves the pointer using an index**

Loading ends with:

```text
load(...) → success

No tensor computation has occurred yet.
```

Later:

```text
runtime requests function index 0
└─ getKernelAsyncABI(0)
   ├─ read kernels[0] → P0
   └─ reinterpret_cast<CompiledKernelAsyncABIFn*>(P0)
      └─ callable signature: void(void **)
```

The other supported accessor returns a callable with signature:

```cpp
void(void **args, ssize_t *dims)
```

For an illustrative synchronous tensor routine:

```text
args = [input-buffer pointer, output-buffer pointer]
dims = [] when no dynamic dimension values are required
```

The exact argument list depends on the generated routine’s ABI. The cast does not construct arguments or validate their types; the caller must supply the matching ABI.

See [CompiledKernelsContext.h:76](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/Support/CompiledKernelsContext.h:76). The MGP execution path retrieves the async callable in [MGPPrimitives.cpp:176](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Primitives/MGPPrimitives.cpp:176).

**2.11 Alternative route: `loadDebuginfo=true`**

This replaces the ordinary object-registration branch:

```text
load(...)
├─ createJITDylib(hashName)
├─ llvm::object::createBinary(buffer)
│    Parse/recognize the supplied binary
├─ writeTempFile(...".o", objectBytes)
├─ keep the temporary object file
├─ createUniqueFile(...".so")
├─ construct linker command
├─ system(command)
├─ DynamicLibrarySearchGenerator::Load(sharedLibraryPath, globalPrefix)
├─ dylib.addGenerator(generator)
└─ engine.addToSearchOrder(hashName, dylib)

Then the same symbol-lookup/table-population loop runs.
```

Illustrative values:

```text
objectFile = "/tmp/8a3f10c492be760d-k7p2.o"
outPath    = "/tmp/8a3f10c492be760d-m4q8.so"

Command:
  <configured GENERICML_LINKER_CMD> -shared <objectFile> -o <outPath>
```

The linker executable comes from `GENERICML_LINKER_CMD`; this function does not hard-code `clang` or `ld`.

The shared-library generator makes the library’s exported symbols available to ORC lookup. The source’s stated purpose is to preserve symbol/debug information for profiling. Unlike the ordinary branch, this branch directly calls `createJITDylib`; it does not use the atomic `addIfAbsent` reuse route.

See [CompiledKernelsContext.cpp:98](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Support/CompiledKernelsContext.cpp:98).

**2.12 Errors and lifetimes**

Example failure:

```text
kernelNames = ["add_one_entry", "missing_entry"]

First lookup:
  succeeds → P0 appended

Second lookup:
  missing symbol → return error
```

The function does not contain a rollback that removes the first table entry. A successful return means all requested entries were resolved.

The retained object bytes survive through the layer’s references. The execution engine owns the loaded-code infrastructure; `CompiledKernelsContext` borrows its engine pointer, and the header requires that engine to outlive the context and outstanding runtime tasks. See [CompiledKernelsContext.h:141](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/Support/CompiledKernelsContext.h:141).
