# Custom-op example readers: accepted decisions

This reference consolidates the user's iterative requirements through
November 11, 2026. Read it when editing the four custom-op readers or making
another reader similar to
[the current add-one page](../link_htmls/max_example_01_add_one_compilation_and_execution.html).
The latest request determines the scope; these decisions do not authorize
unrelated changes or require a fixed number of cards or animation steps.

## Layout and preserved content

- Keep 00 Build & run, 01 From Bazel's build graph to a selected device backend,
  the operation journey and the focused graph/compiler/kernel animations.
- In add-one's 01, remove the Text · answers 1–7 tab and its reading panel.
  Retain Animation and Call Stack Tree, their controls, and all working
  explanation/source cards. Readers requested to match add-one use this
  current layout. Do not recreate a removed text view or menu under a new name.
- Keep the call trees for Graph construction, InferenceSession construction,
  compile, init and execute separately titled. Put the exact example call
  before each tree; make constructor recording style match the source
  (`forward=` versus `with Graph(...)`).
- The user removed A/A.0/A.1/A.2/A.3 and B/C/D menu-like card links; the
  explanations remain inline triggers in the trees. Retain the useful E
  input/execution example trigger. Avoid another duplicate navigation menu.
- Preserve user-supplied text word for word when requested. Explanations and
  diagrams may be added alongside it. Display rendered diagrams directly;
  do not show Mermaid source or terminal size-limit messages as the diagram.
- Portable flow transcripts use readable, larger text and include each
  transformation, input/output handoff, source evidence and relevant branch.

## Trees and source links

- Follow executable source order. In Graph.__init__, the empty module/GraphOp
  scaffold precedes package import; package P1 is available before recording
  and validating the custom call. MO → MAP occurs during compilation, not
  while constructing Graph or the session.
- Link functions to their definitions and callers to the exact invocation
  line. Include a caller link on the first invocation and on every terminal
  branch. Avoid linking a later statement to the constructor's opening line.
- Source links open complete pinned files in the offline Source Explorer and
  highlight the selected statement/function. A file path mentioned in an
  explanation should also have a working source link.
- Preserve connected tree branches around explanations and spacer rows.
  Use compact IN/OUT examples beside the relevant call; name retained graph,
  package, session, compiled artifact, model and per-call buffer objects.
- Use one card per concept at its actual tree row. Examples include context,
  _location(), MLIR Module Reuse or create, builder insertion point, to_mlir,
  graph signature/body arguments, devices, chains and kernel validation.
- Use the user's wording where applicable: **MLIR.module is IR container**.
  For _location(), keep the explanation that it provides stack trace metadata
  for mo.GraphOp and builtin.ModuleOp(location=loc), with an illustrative
  Python call-site location.

## What the cards explain

Each card answers how the operation works: expand the internal call path,
show concrete input/output values, state why it is needed, and animate the
relevant representation or ownership change. A caption listing stages alone
does not replace the mechanism.

| Concept | Distinction or example to retain |
| --- | --- |
| MLIR context | Current is thread-local active context state; DEFAULT_MLIR_CONTEXT is a shared module-level instance; MLIRThreadPoolExecutor runs work with context entry. Explain background threads and the Python decorator. |
| Location | Provenance metadata, optionally derived from the Python stack; attached operations retain it after leaving the scope. |
| Module + GraphOp | Operation → region → block → operations; one graph body block contains one argument per input. Show the graph after insertion and after output finalization. |
| to_mlir | Static/symbolic shape attributes, DType.float32, device kind/id, optional layout dictionary; describe their values rather than only listing type names. |
| CAPI bridge | MAX wrapper → named capsule → MlirOperation → MLIR Python wrapper around the same native operation; no IR copy. Link export/import and internal pointer adapters. |
| Custom call | Exact registration name, device attribute, compile-time parameters, operand/result mapping, package validation and symbolic result wrappers. |
| Session | Configurations and retained runtime context as well as devices; give example members and extension paths. |
| Compilation | Clone C1 preserves G1; MO → MAP → MOGG → MGP transforms graph/runtime plans; separate Mojo specialization → MLIR LLVM dialect → LLVM IR → backend artifacts. |
| Initialization | Resource loading, state slots, weight/device/context references, CPU worker task and initialized M1; explain why asynchronous setup and readiness are needed. |
| Execute | Numeric input buffers arrive here; show argument preparation, compiled-call invocation, result extraction, readiness and explicit host-copy behavior of the actual example. |

## Interaction and visual treatment

- All explanation/legend/source cards support move, resize/expand, small Hold
  and Close buttons, touch and keyboard use. Hold retains the card until
  closed. Opening another card should preserve a held card and its contents.
- Use bold text with clearly distinguishable font colors. The native-loader
  supplement uses grey with bold pink/green; reuse this accepted contrast for
  similar detailed cards. Do not add colored word backgrounds merely to
  compensate for weak font contrast.
- Keep the viewport usable in Chrome fullscreen. Cards and source dialogs
  must remain visible and bounded; long trees scroll within their containers
  on 320/390px screens.
- Animation changes show actual payload/state transformations: source/type
  fields becoming MO fields, symbols resolving, pointer indirection, numeric
  loads/arithmetic/stores, fresh per-call inputs and retained code/handles.
  Use focused views rather than one unreadable diagram.
- No autoplay. Step, Back, Reset, Play/Pause and replay must agree with the
  portable transcript; respect reduced motion and pause hidden/closed cards.

## Evidence and adapting another example

- Recheck exact source inputs, graph name, kernel registration, parameters,
  outputs and CPU/GPU paths before adapting the add-one reader. Avoid an
  add-one-shaped explanation with renamed headings.
- Bazel's Python example packages raw Mojo source as runfiles; shared Mojo
  dependencies are precompiled packages. A separate mojo precompile process
  prepares a source extension package at runtime. Distinguish the executable
  path, package cache and .mojoc MLIR bytecode from target machine code.
- CompiledKernelsContext::load links already compiled host objects; later
  host-routine execution reaches device-module/function loading and driver
  launch. Keep native host addresses, GPU handles and tensor addresses distinct.
- Use primary web sources when researching additional mechanisms. Record
  applied decisions, pin external implementation files to the checkout's
  selected dependency revision, and keep source snapshots available offline.
- Schematics are examples, not compiler dumps. Browser arithmetic is not MAX
  or GPU execution. Keep numerical scheduling, vendor residency and hardware
  timing marked BOUNDARY unless measured.
- For fused attention, verify rank-2 unscaled Q/K/V → softmax(QKᵀ)V, three
  graph inputs, BN/BD parameters, CPU/GPU tile differences and actual launch
  geometry. Explain running max/denominator/output rescaling with a tile
  example whose second maximum grows. Show temporary tile storage and
  retained state without implying a full N×N score allocation in the provided
  fused implementation.

Author in src/, rebuild only the requested readers, validate retained source
links/cards and live controls, and inspect intermediate frames and narrow
layouts. Preserve unrelated working-tree changes and earlier Git checkpoints.

## Current fused-attention implementation

[The attention reader](../link_htmls/max_example_04_fused_attention_online_softmax.html)
now uses the two-view layout with separate graph/session/compile/init/execute
trees and a kernel tree. Its authoring source is
[compiler_attention_calltrees.py](../src/compiler_attention_calltrees.py).
The common source-link renderer is
[compiler_calltree_reader.py](../src/compiler_calltree_reader.py); the graph
and kernel payloads remain example-specific.

Use [compiler_attention_motion.js](../src/compiler_attention_motion.js) for
deterministic numerical and representation frames,
[compiler_attention_mechanisms.py](../src/compiler_attention_mechanisms.py)
for embedded offline scenes, and
[compiler_attention_mechanisms.js](../src/compiler_attention_mechanisms.js)
for per-card controls. The recurrence card shows only the terms reached at
the current step: prior-output rescaling precedes the new PV contribution.
Keep matrix data, symbolic values, native callables and GPU handles distinct.
Pause freezes the current packet pose; resume continues from that pose.

Scoped commands from this folder:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_examples.py --examples addition attention --readers-only
PYTHONDONTWRITEBYTECODE=1 python3 src/check_compiler_examples.py --examples addition attention
NODE_PATH=/tmp/codex-inference-session-playwright/node_modules node src/check_compiler_example_readers_browser.cjs
```

The NODE_PATH above records this host's installed browser-check dependency;
use the available Playwright installation on another host. The checks cover
exact links/cards, recurrence values, Hold/Expand/Close, real Play/Pause,
Chrome fullscreen, 320/390px layouts, and reading without JavaScript.
These checks do not execute the MAX/Mojo examples or prove GPU parity.
