# MAX engine animation validation

Date: November 6, 2026.

Deliverable: [MAX engine / Mojo animation](../link_htmls/max_engine_graph_compilation_mef_and_inference.html).

SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
The built-in Llama Graph API route anchors graph tracing; `meta-llama/Llama-2-7b-hf`
is the parent walkthrough's model ID. Engine artifact alternatives are shown
separately. No model was loaded, compiled, initialized or executed during this
work. Runtime device for validation: CPU host with headless Chromium; no MAX
accelerator run.

Six chapters cover session/graph ownership, native graph passes, Mojo kernel
registration and compiler architecture, MEF preparation/reuse, initialization,
and dependency-driven inference. Forty components retain distinct colors;
legend cards describe object/IR/thread/synchronization ownership. The call-stack
supplement link sits immediately below `InferenceSession`.

Run these from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_engine_lifecycle.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_engine_lifecycle_browser.cjs
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/validate.py --report murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/validation/engine_lifecycle_source_checks.json
```

- [Animation model](engine_lifecycle_model.json): PASS, 24 combinations and
  445 transitions. Checked compilation/initialization/output gates, cache and
  artifact bypasses, export, virtual-device refusal, failure termination, and
  deterministic seek/reset.
- [Browser](engine_lifecycle_browser.json): PASS, six chapter contrast audits,
  all 32 default-route handoff arrows, full Python/C++/Mojo source display,
  labels within cards, no autoplay, hover/focus/touch cards, click-only legend,
  pause/reset/seek, offscreen pause, reduced motion, offline loading and
  JavaScript-disabled fallback. Mobile widths: 320 and 390 pixels.
- [Source/link validation](engine_lifecycle_source_checks.json): PASS, 47 engine
  source targets; all embedded files match saved snapshots and pinned revision;
  all Python/C++ definitions and Mojo/document selections have verified spans;
  2,261 local links in the combined artifacts; no external runtime dependencies.

Evidence is `SOURCE` for the technical explanation and `CPU-RUN` for the
animation/browser checks. Mojo phase views describe the repository compiler
architecture rather than a captured kernel JIT trace. Device launch details,
GPU overlap, timing and memory traffic remain `BOUNDARY` / future `GPU-LAB`
evidence. The driver result path visibly awaits runtime result values; the
animation does not imply a device-wide barrier or an automatic host copy.
