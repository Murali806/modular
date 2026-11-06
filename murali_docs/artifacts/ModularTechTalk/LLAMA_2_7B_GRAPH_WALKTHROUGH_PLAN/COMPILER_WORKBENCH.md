# MAX / MLIR / Mojo compiler workbench

[Open the animation](link_htmls/max_mlir_mojo_compiler_and_runtime.html).

SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. IR views, filenames, lane counts and device motion are illustrations. No MAX compilation or hardware execution was run.

## 01 · From Python to a verified graph

Watch a definition become named operators, then seal its output contract.

```text
Python builder → G1 / native MLIR → Graph.output signature + verify
Verified Graph ≠ compiled artifact ≠ initialized Model
```

## 02 · A module changes its language

Follow C1 through structured bodies, kernel calls, buffers and generated Mojo.

```text
G1 retained → compiler clone C1
MO → MAP → MOGG → MGP → symbol DCE → emitted Mojo
Enclosing pipeline: kernel JIT → final runtime-call lowering
```

## 03 · Readable code. Native control.

Change the SIMD width, or try a conflicting borrow and watch the legality gate.

```text
Mojo → LIT → lifetime/control-flow checks → KGEN
Parameters → elaborated functions → LLVM dialect
Handle/reference movement ≠ copying allocation cells
```

## 04 · Pick a target. Change the binary.

Change the target, choose assembly or a binary, then inspect its disassembly.

```text
LLVM dialect / MLIR → LLVM module
CPU: LLVM MC → .o → link → .so
NVIDIA: LLVM NVPTX → PTX → CUDA CompilationDevice → cubin
AMD: LLVM AMDGPU → .o → ELF link → hsaco
Binary → disassembly is an inspection branch
```

## 05 · Pack the artifact. Bind the values.

A cutaway archive separates compiled code, its plan and the weight-binding rack.

```text
Compiled routines + MGP plan/resources → prepared primitives → MEF bytes
Optional export → model.mef
CompiledModel + weight registry → init_all → executable M1
```

## 06 · Code stays. Buffers move.

Queue new inputs while the same code, weights, M1 and S1 remain resident.

```text
Ready M1/S1 + inputs → optional explicit upload
Descriptor references code/buffers → native driver → hardware
AsyncValue ready → result buffers → optional explicit host copy
Next call reuses code/model/session
```
