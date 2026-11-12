# Fresh attention reader

The user explicitly rejected the expanded attention page as crowded and
requested deletion/recreation at the same filename on November 12, 2026.
This replaces the previous attention layout; preserve the add-one reference
and unrelated readers.

## Teaching structure

- Show one stage at a time: **01 build**, **02.1 Graph G1**, **02.2 session
  S1**, **02.3 compile C1**, **02.4 init M1**, **02.5 execute**. Start with the
  source-ordered call tree. Each stage has one coherent animation and a
  detailed explanation of why its boundaries and structures exist.
- **C1 is the public CompiledModel**. Call the mutable compiler clone
  **IR1**, keeping G1, P1, S1, M1, host resource A1, device image D1, host
  callable H1 and device function KD1 distinct.
- Use focused inline cards for unfamiliar mechanisms. Their internal trees
  need concrete values, source links and meaningful state changes. Detailed
  ELF, host linking, two ABIs and device loading belong in the separate
  attention GPU-artifact guide linked from compile/init/execute.
- Keep grey surfaces and bold pink/green word highlights. Give the active
  diagram enough width and progressively disclose long explanations. More
  cards, simultaneous panels and repeated prose do not establish depth.
- Keep movable/resizable Hold/Expand/Close windows and the offline complete
  source explorer. Open exact call sites, not the start of an unrelated
  enclosing function. Preserve connectors when tree lines wrap.

## Non-obvious source distinctions

- Bazel builds a Python launcher with raw custom Mojo sources in runfiles.
  Shared Mojo library targets can precompile at build time; G1 prepares its
  source extension P1 with a separate `mojo precompile` subprocess at runtime.
  A deterministic package path is not a source-package cache-hit branch.
- G1 creates its module/GraphOp scaffold before importing P1, then the
  explicit with-block records the custom operation. This example has no
  forward lambda, and uses three rank-2 float32 GPU inputs, N=D=32, BN=16,
  BD=8, key tile 8 and grid (4,2), block 32.
- Put concrete BN/BD/type/device choices on G1's call and the specialization
  state. P1 remains the parametric package. In the numerical animation, keep
  the old tile's m/l/O snapshot separate from the newly committed values so
  the rescaling equation continues to display its actual operands.
- `.mojoc`, mutable MLIR, MEF, host object bytes, GPU executable bytes,
  function handles and tensor allocations have separate roles. Do not call
  tensor payloads ELF `.data` or imply physical placement measured on a GPU.
- `mgp_generic_execute` actually selects `CompiledKernelAsyncABIFn` and
  passes addresses of AsyncValueRef slots plus resource entries. The header
  also declares a different two-argument ABI; do not substitute that
  declaration for the primitive's actual call. A second pack encodes the
  GPU launch arguments into device-layout values in host storage.
- Normal host loading uses ORC/JITLink. A debug/profiler branch creates and
  loads a temporary `.so`; installed CUDA/HIP libraries have different
  origins. The JITDylib symbol namespace does not itself imply dlopen.
- Init waits for setup before returning M1. The generated device load/check
  site is reached through enqueue, potentially on cold first execution;
  do not claim that every GPU code page is uploaded during Python init.
- Result AsyncValue readiness, device command completion and host buffer
  accessibility are different gates. The script prints output Buffers with
  no explicit readback or parity assertion. Any added host read is labeled
  as a teaching extension.
- GPU launch is local device offload, not MAX Serve RPC. The native driver
  boundary does not reveal every vendor transport, queue or paging detail.

## Authoring and checking

The replacement is authored in `src/attention_walkthrough*`, with focused
content in `attention_driver_content.py` and `attention_walkthrough_cards.py`.
`compiler_examples.py` routes example 04 to the new builder. Rebuild only
this reader and its companion to preserve independent changes:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_examples.py --examples attention --readers-only
PYTHONDONTWRITEBYTECODE=1 python3 src/check_attention_walkthrough.py
node src/check_attention_walkthrough_model.cjs
NODE_PATH=/tmp/codex-inference-session-playwright/node_modules node src/check_attention_walkthrough_browser.cjs
```

The browser test dependency path records this host's existing installation;
use the available Playwright installation elsewhere. Check actual controls,
source links, retained actor nodes, pause/resume, cache/vendor route changes,
numeric prefix/final outputs, cards, fullscreen, narrow layouts and offline
reading. These are documentation checks, not MAX compilation or GPU parity.
The shared source/model checks and `check_compiler_example_readers_browser.cjs`
route this layout to the focused attention validators and retain add-one
coverage. The older generic three-phase browser checker predates this layout.

Primary LLVM/NVIDIA/ELF documentation informs the external boundaries.
Pinned repository source establishes the example route. Preserve this scope
when adding detail; do not turn a design choice for this reader into a
universal requirement for every page.
