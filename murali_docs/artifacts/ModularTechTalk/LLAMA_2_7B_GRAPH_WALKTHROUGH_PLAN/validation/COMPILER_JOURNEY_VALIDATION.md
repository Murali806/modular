# Connected compiler journey validation

Date: November 7, 2026. SOURCE revision:
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Artifact: [compiler workbench](../link_htmls/max_mlir_mojo_compiler_and_runtime.html)
and its four linked repository-example readers. The connected 01 shares a
deterministic model, persistent SVG actor map and controller. The existing
focused scenes and their fragment IDs remain usable.

## Evidence and results

| Check | Result | Evidence |
| --- | --- | --- |
| Journey model | 36 example/target/cache routes pass | [compiler_journey_model.json](compiler_journey_model.json) |
| Journey browser | 1,220 frame audits pass across five pages | [compiler_journey_browser.json](compiler_journey_browser.json) |
| Focused example arithmetic/lifecycle | 18 routes pass against independent tiny arithmetic | [compiler_examples_model.json](compiler_examples_model.json) |
| Focused example browser | 464 render audits pass | [compiler_examples_browser.json](compiler_examples_browser.json) |
| Existing workbench model/source | 10 check groups pass | `src/check_compiler_workbench.cjs` |
| Existing workbench browser | 72 render audits pass | [compiler_workbench_browser.json](compiler_workbench_browser.json) |
| Example bytes/spans/navigation | 22 original files, 160 spans, 276 local links pass; no runtime network dependencies | [compiler_examples_sources.json](compiler_examples_sources.json) |
| Skill | Frontmatter validator and local reference paths pass | Local skill/references and quick_validate.py |

CPU-RUN means JavaScript arithmetic/state and offline Chromium on this host.
SOURCE means checked original repository bytes and complete-definition spans.
The parent embeds 49 source targets from 24 original files. Browser motion,
register labels, code shapes, crate ingredients and device timings are schematic.
No MAX/Mojo compilation, inference, model loading, benchmark or GPU measurement
was performed. Compiler inspection tools were unavailable on PATH; no captured
IR/assembly dump is claimed.

## Mechanisms verified

Package import precedes custom-call construction. G1 stays caller-owned while
C1 is compiled. Python output-contract verification differs from later device
graph finalization. The MGP plan remains visible while KIR1 and LLVM1 change.
Runtime-call lowering, module loading and initialized M1 are separate milestones.

Compatible graph-artifact hits bypass new graph/kernel lowering. Compiled-kernel
hits retain graph planning and runtime-call preparation while skipping fresh
kernel code generation. CPU/NVIDIA/AMD routes produce different code roles.
Two illustrative calls keep S1/M1/code and exchange input/result identities.
The matmul helper's original fresh-model-per-invocation behavior is labeled.

Pointer and keyboard selection, Next change filtering, stage jumps, source
ancestry and original-file explorer work. The same SVG node moves, Pause freezes
its pose and semantic index, Resume continues, and offscreen sections pause.
No autoplay. Reduced-motion, hover/focus/touch cards, click-only L, 320/390px
layout, no-JavaScript transcripts and print were checked without HTTP requests.

Epilogue fused/separate arithmetic agrees. Intermediate counters exclude A/B
loads and hardware traffic; TMP1/TMP2 lifetimes end while output remains.
Attention's online recurrence agrees with independently evaluating full softmax.
The same local score slot is released/reused across key tiles while m/l/O
remain. Full-score storage is a hypothetical comparison, not the source kernel.

Live checks caught and repaired a transcript-click selector that accidentally
matched the enclosing section and immediately paused Play. A mobile long-card
case also caught the card covering its triggering badge; cards now fit above or
below the badge with internal scrolling.

## Commands

Run from repository root. `task_root` below is the task directory; the local
Playwright installation supplies Chromium, not a runtime dependency of the HTML.

```sh
task_root=murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN
PYTHONDONTWRITEBYTECODE=1 python3 "$task_root/src/compiler_workbench.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$task_root/src/compiler_examples.py"
node "$task_root/src/check_compiler_journey.cjs"
node "$task_root/src/check_compiler_examples.cjs"
node "$task_root/src/check_compiler_workbench.cjs"
PYTHONDONTWRITEBYTECODE=1 python3 "$task_root/src/check_compiler_examples.py"
NODE_PATH=/tmp/max-os-playwright/node_modules node "$task_root/src/check_compiler_journey_browser.cjs"
NODE_PATH=/tmp/max-os-playwright/node_modules node "$task_root/src/check_compiler_examples_browser.cjs"
NODE_PATH=/tmp/max-os-playwright/node_modules node "$task_root/src/check_compiler_workbench_browser.cjs"
python3 /local/mnt/workspace/MyTools/.codex/skills/.system/skill-creator/scripts/quick_validate.py "$task_root"
git diff --check
```

## Rendered review

Reviewed intermediate representation, target-code and second-call states, plus
the epilogue/attention memory mechanisms. Representative screenshots:

- [LLVM dialect before/after](compiler_journey_addition_ir.png)
- [Distinct host/device code](compiler_journey_addition_code.png)
- [Retained executable, second input/result](compiler_journey_addition_reuse.png)
- [Epilogue intermediate lifetimes](compiler_example_epilogue_intermediates.png)
- [Attention score storage/reuse](compiler_example_attention_score_storage.png)

[Research decisions](../COMPILER_WORKBENCH_RESEARCH.md) and
[retrieval evidence](../evidence/compiler_deep_research.json) document the source
material used for the redesign. They do not establish measured compiler timing.
