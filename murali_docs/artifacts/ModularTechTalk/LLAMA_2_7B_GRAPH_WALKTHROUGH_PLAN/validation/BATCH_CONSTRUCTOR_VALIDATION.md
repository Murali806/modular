# Batch constructor animation validation

Date: November 6, 2026.
SOURCE revision: 610f7111836d9a950d45ba52ec9b3e0ed589fe3f.
Model ID: none; no checkpoint loaded. Device: CPU / headless Chromium.

The two standalone readers include 03 after the API and worker close-ups.
Request sets are drawn inside the constructor's solid Python-object boundary.
Later lesson badges are renumbered; their existing fragment IDs are preserved.

Commands run from the repository root:

~~~bash
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_batch.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_worker.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_api.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_stories.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_serve_stories_browser.cjs
python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/validate.py
git diff --check
~~~

| Evidence | Result |
| --- | --- |
| CPU-RUN constructor model | PASS: 96 combinations of eight scenarios, three policies, two active targets and chunking ON/OFF |
| CPU-RUN behavioral invariants | PASS: request seats, KV capacity, CE/TG ownership, token preservation, partial-prefill output, immutable execution membership, readiness and completion |
| CPU-RUN existing API / worker / request models | PASS |
| CPU-RUN browser | PASS on both readers: candidate movement, controls, hover/focus cards, source links, touch at 320/390 px, pause, print and no-JavaScript fallback |
| CPU-RUN visual audit | PASS: 72 contrast audits; occupied request rows inherit component tint and ink |
| SOURCE / local links | PASS: 2,066 local links; source spans and snapshots match the pinned revision |

The staged generators were also exported to a temporary directory, both pages
were rebuilt from those exact source files, and the constructor and browser
checks passed there. Pre-existing workspace edits were excluded from the commit.

BOUNDARY: GPU execution, page grants, token counts and readiness events are
illustrative. This is a one-replica branch projection with no adapters,
speculative drafts, DP padding or overlap; it is not an inference benchmark.
