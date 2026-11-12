# Detailed attention cards

**Superseded layout:** the user subsequently rejected the expanded page as
crowded and requested a fresh stage-led reader. Use
[the replacement design](attention_stage_reader.md) for example 04. The
mechanism/source-research distinctions below remain useful background; the
earlier card count, layout and implementation list are not acceptance criteria.

The user reassessed the attention reader on November 12, 2026: matching
add-one's appearance and adding a short animation to every card did not
provide enough explanation. Apply these decisions to complex attention
readers when the request calls for this depth.

## The failure to correct

The first attention implementation had 34 cards, but most lacked an internal
call tree. Several different concepts shared the same four-stage scene.
Changing captions on a source-to-result packet did not show module nesting,
native handles, parameter inference, buffer ownership or tile arithmetic.
Card count and playback controls alone are insufficient.

## Design each card around its question

- Give each card a distinct source-ordered internal tree with precise
  definition and caller links, concrete inputs/outputs, and the reason for
  each operation. Include normal and relevant alternate branches; a
  comparison is explicitly separate from the actual example.
- Use a lesson specific to that card. Every step identifies its current
  input, exact operation, visible result, consumer and reason. Show fields
  and values as they become available. Explain unfamiliar words where used.
- Select a visual form for the mechanism: thread/context stacks, nested
  operation-region-block structure, insertion cursor, type/attribute
  builders, named-pointer adapters, registry candidate/role mapping,
  retained versus transformed compiler representations, object ownership,
  readiness gates, output tile grids, fragment accumulation or weighted
  matrix arithmetic. Avoid the same generic traveling box for all of them.
- Use stable object and storage identities. Compare before/after values
  and make changes observable. A new Python wrapper is not a graph clone,
  a view is not an input copy, and runtime state is not a compiler context.
- For attention, draw actual small Q/K/V matrices and follow highlighted
  query/key/feature cells through dot products, stable exponentials,
  denominator mass, normalized weights and output components. Expand
  maximum growth, alpha, old-output rescaling and PV into separate steps.
  Include a second query so row-wise normalization is unambiguous.
- Keep global Q/K/V/O, current S/E/PV tiles and retained per-row m/l/O_i
  distinguishable. Draw key iteration and output-block ownership separately.
  Label logical storage shapes versus source-declared address spaces.
- Keep the actual source's normalized-output recurrence. An unnormalized
  numerator or log-sum-exp state can explain a comparison, but must not
  silently replace O_i or claim the source uses another implementation.
- Reuse the accepted grey surface, bold pink/green words, movable
  Hold/Expand/Close windows, exact offline source explorer and two page views.
  All lessons support deterministic Step/Back/seek/Reset and real
  Play/Pause. Pause freezes visible motion as well as semantic stepping.
- Keep the current source operation and IN/DO/OUT/WHY above the canvas.
  Put the relevant arithmetic first: dot-product cells during QK, mass/state
  during denominator updates, and prior-output/PV terms during output
  rescaling. Inputs and the complete object register can remain expandable
  rather than pushing the active transformation below the viewport.
- Bind highlighted values to field identity, not matching display text. The
  old maximum and output-scale scratch can both contain 1 while only the
  scratch changes. Show that scratch through copy/subtract/exp and clear
  stale tile maxima/dot products before the next tile computes them.
- Recheck whether a cache branch actually exists. In this checkout,
  `_build_mojo_source_package` creates a deterministic destination but always
  invokes precompile for a source directory. Supplying an existing `.mojoc`
  skips that subprocess; `KernelLibrary._analysis_cache` reuses analysis;
  kernel code caches reuse native routine resources. Do not infer a
  source-package file cache hit from the destination name.
- Audit the exact linked statement, not merely the enclosing function span.
  A link can pass a bounds check while landing on a docstring, error branch or
  neighboring call. Compare each frame's operation with the pinned source;
  use the actual invocation for caller links and the actual return/binding
  when explaining a returned value.
- Preserve constructor bookkeeping exactly: `dict.fromkeys` initializes
  `_params` values to None. Shape conversion creates dimension expressions;
  do not attribute that work to the name-collection statement.
- In JIT cards, show host backend selection before routine-cache lookup.
  Cache misses elaborate generators and discover device offloads. Those
  receive separate target modules/options/backends before host archive
  lowering completes. A GPU architecture override does not replace the
  ordinary host CPU triple. Cached archive bytes/symbol names are not yet
  resident callable addresses; initialization resolves those addresses.
- Follow deferred initialization inside `createExecutableModel`: construct
  the owner, await fixed and extra init results, check errors, then publish
  the executable. Native load waits before Python init returns M1. A failed
  init stops the caller before its execute line; do not animate execution as
  the first place that waits for the initial resource chain.

## Primary research applied

Readings inform original teaching diagrams; pinned MAX source establishes
the implementation. Do not reproduce copyrighted figures or make a serving
implementation part of this standalone rank-2 example.

| Primary source | Useful idea | Applied distinction |
| --- | --- | --- |
| [Modular GPU kernels / FlashAttention](https://handbook.modular.com/kernel-optimization/flashattention/) | Layout indexing, tiling, stable softmax, output normalization | Draw logical tiles and address calculations; verify the actual helper and recurrence in this checkout |
| [FlashAttention paper, authors' publication](https://arxiv.org/abs/2205.14135) | Compute attention in tiles and retain row state | Animate S/E reuse and the changing maximum/denominator; exclude a persistent full-score allocation from the fused route |
| [FlashInfer Cascade Attention](https://flashinfer.ai/2024/02/02/cascade-inference.html) | A partial attention result can be combined with another using compatible normalization | Draw the prior mass, current mass and weighted-output merge; use the source's m/l/normalized-O state |
| [vLLM PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/) | Query/key products, block-local softmax and value accumulation | Use cell-level Q/K/V arithmetic; paged KV and token-serving behavior remain comparisons |
| [SGLang team's introduction](https://lmsys.org/blog/2024-01-17-sglang/) | Serving exposes distinct reuse and execution scopes | Keep graph/code reuse separate from fresh tensor data; do not insert RadixAttention into this kernel |
| [FlashAttention-3 authors' explanation](https://tridao.me/blog/2024/flash3/) | Hardware pipelining and overlap are separate optimizations | Show loads, MMA and barriers individually; do not claim this source implements FA3 warp specialization or asynchronous overlap |

This research was inspected on November 12, 2026. Numerical values, example
addresses and schematic IR are teaching material. Browser arithmetic and
source checks do not establish MAX execution, GPU parity or measured speed.

The web tool returned no usable page body in this session. Primary pages were
retrieved directly over HTTPS; vLLM's documentation endpoint returned 403, so
its first-party `docs/design/paged_attention.md` on GitHub was read instead.
The Modular handbook also discusses newer FlashAttention variants. Those
capabilities do not change what the pinned example implements.

## Implemented source and checks

The lesson schema and source-ordered frames live in
[compiler_attention_lessons.py](../src/compiler_attention_lessons.py), with
separate [constructor](../src/compiler_attention_graph_lessons.py),
[recording](../src/compiler_attention_recording_lessons.py),
[runtime](../src/compiler_attention_runtime_lessons.py), and
[kernel](../src/compiler_attention_kernel_lessons.py) lesson data.

[compiler_attention_math.js](../src/compiler_attention_math.js) is the
DOM-independent numerical model. It follows each source copy/subtract/exp/
multiply/divide, preserves normalized O, and supplies two independent query
cases. The [renderer](../src/compiler_attention_mechanisms.py) embeds complete
frames and portable transcripts. The [controller](../src/compiler_attention_mechanisms.js)
uses mechanism-specific diagrams and preserves the timeline in held windows.

Validation checks prefix outputs as well as final softmax, growing/unchanged
maxima, scratch and retained states, source links, every card's alternate
cases, real animation pause/resume, card controls, fullscreen, narrow/offline
reading and print. `check_attention_math.cjs` runs only browser-style numeric
teaching arithmetic; it does not build or execute MAX/Mojo.

## Verify depth as well as behavior

Review every card at intermediate frames. Check source-order and exact
links, available/unknown fields, retained identities, role mappings,
parameter/type changes, and prerequisites. Independently compare every
query's final tiny output with a stable full-softmax reference. Check growing
and unchanged maxima, CPU/GPU tile geometry, S/E/PV storage lifetimes and
partial fragment sums. Test real Hold/Expand/Close, Play/Pause/resume,
fullscreen, 320/390px layouts, offline/no-JavaScript and print behavior.

Keep this guidance focused on the demonstrated depth problem. Do not turn
the current number of cards/frames or the teaching matrix size into a quota.
