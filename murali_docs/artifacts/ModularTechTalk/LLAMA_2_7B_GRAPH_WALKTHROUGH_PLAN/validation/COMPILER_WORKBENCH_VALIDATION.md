# Compiler workbench validation

Date: November 6, 2026. Host: Linux CPU; browser: headless Chromium through
Playwright. SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
[Open the animation](../link_htmls/max_mlir_mojo_compiler_and_runtime.html).

From the task directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_workbench.py --refresh-session
node src/check_compiler_workbench.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_compiler_workbench_browser.cjs
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py --report validation/compiler_workbench_sources.json
```

The isolated builder generated six lessons with 37 source targets and 19
complete pinned source files. The existing session reader retains its four
lessons and adds the compiler-workbench link. The startup call-stack reader and
portable companions were rebuilt separately from their existing verified payload.
The broader guide/Serve pages were not regenerated for this task.

CPU-RUN results:

- [Model/source checks](compiler_workbench_model.json): 10 groups passed,
  including finalization order, SIMD results, borrow repair, target output/error
  branches, named-weight gate, cumulative explicit copies, repeated residency,
  scenario reset and embedded source hashes.
- [Browser checks](compiler_workbench_browser.json): 11 groups passed, with
  72 rendered audits of intermediate/default/scenario/card states. Actor labels
  fit their own backings and remain unobscured; tested text contrast is at least
  the applicable 4.5:1/3:1 threshold. Controls/wires were checked at 3:1.
- [Document/source checks](compiler_workbench_sources.json): passed, including
  pinned compiler files and 2,362 local links. Skill frontmatter validation and
  82 linked skill/reference paths passed; `git diff --check` was clean.
- Persistent actor identity and real motion, frozen in-flight Pause, resume,
  reset/seek, offscreen pause, no autoplay, reduced motion, 320/390 px internal
  scrolling, hover/focus/touch/Escape, click-only circled L, matching component
  colors, source search/copy/exact original text, no external runtime requests,
  no-JavaScript reading and print behavior passed.

Representative reviewed frames:
[graph](compiler_workbench_graph.png),
[dialects](compiler_workbench_lower.png),
[Mojo](compiler_workbench_mojo.png),
[borrow conflict](compiler_workbench_borrow_gate.png),
[target code](compiler_workbench_target.png),
[MEF/init](compiler_workbench_package.png),
[runtime/reuse](compiler_workbench_runtime.png).

These checks execute the teaching model and browser UI. They do not execute
MAX, compile a Mojo example, generate real IR/object/MEF artifacts, perform
inference or measure accelerator/driver timing. Repository tests are inspected
SOURCE/TEST evidence, not locally executed compiler or GPU tests.
