# InferenceSession animation validation

Validated November 6, 2026.

[Open the standalone animation](../link_htmls/max_inference_session_devices_runtime_and_compilation.html).
It is linked immediately below the session ownership explanation in the
[startup call tree](../max_serve_llama2_7b_startup_call_stack.html).

| Check | Result |
| --- | --- |
| Pinned MAX source | 610f7111836d9a950d45ba52ec9b3e0ed589fe3f |
| Model context | meta-llama/Llama-2-7b-hf; standard worker, built-in Graph API |
| Runtime evidence | CPU-RUN browser/JavaScript checks; no MAX model or GPU execution |
| Standalone page | Four full-width animated lessons; no external runtime dependencies |
| Constructor cases | 160 combinations of devices, threads, MEF policy and extensions |
| State checks | 1,549 frames; initialization dependencies, batch reuse and reference ownership |
| Source provenance | 30 exact targets in 10 complete, hash-verified source files |
| Browser checks | Desktop, 390px touch viewport, no JavaScript and printing |
| Contrast | 34 audits; SVG text checked against its actual drawn backing |
| Interactions | Moving packets, Play/Pause, Step/Back, Reset, speed and step index |
| Readability | Numbered lesson cards, component legend, source search/copy and focus return |
| Motion policy | No autoplay; reduced motion; pause offscreen and when the document is hidden |
| Integration | Real call-tree link opens the standalone page in a new window |
| Guide-wide links | 2,281 local links validated; no external runtime dependencies |

The device combinations, weight names, token IDs, session/model labels and
pictured reference counts are teaching inputs. Native driver/GPU motion is
marked BOUNDARY. Setup and reference counts are not measured hardware data.

Commands run from the repository root:

~~~sh
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_inference_session.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_inference_session_browser.cjs
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/validate.py
~~~

The Playwright location above is specific to this host; use an available
installation on another machine.

Detailed results:

- [State and source checks](inference_session_model.json)
- [Browser checks and contrast audits](inference_session_browser.json)
- [Constructor screenshot](inference_session_constructor.png)
- [Missing-weight branch](inference_session_missing_weight.png)
- [Repeated execution](inference_session_execution.png)
- [Source explorer](inference_session_source_explorer.png)
- [Mobile layout](inference_session_mobile.png)

All listed checks passed. Source-highlight line-number contrast was corrected
during browser review. The final diagrams route weight bindings to initialization
and runtime configuration to the context without crossing unrelated component
boxes.
