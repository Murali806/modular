---
name: max-serve-visual-walkthrough
description: Maintain this source-linked Llama 2 / MAX Serve walkthrough and its interactive OS readers, including API, worker and batch-constructor animations. Use for requirements capture, teaching-page changes and artifact validation in LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN.
---

# MAX Serve visual walkthrough

Keep this walkthrough accurate to its pinned MAX source and useful through
visible request movement, state transitions and component ownership.

Read [agent.md](agent.md) for the task's existing scope, main-guide conventions,
evidence rules and authoring workflow. This skill lives in the user-selected
artifact folder and uses its existing builders and source snapshots.

For either MAX Serve OS reader, read
[the animation requirements](references/max_serve_os_animations.md). They
consolidate the user's prompts and accepted implementation decisions through
November 6, 2026 for:

- [Driver, OS and scaling reader](link_htmls/max_serve_os_driver_and_scaling.html).
- [Components and request-flow reader](link_htmls/max_serve_os_components_and_flow.html).

Preserve the accepted opening sequence: 01 inside the API process, 02 inside
the separate model-worker process, 03 inside the batch-constructor object.
Use full-width animations, numbered hover/focus lesson cards and the circled
L legend at the right edge of the top bar. Thread badges T1/T2/T3 apply only
to 01. Component colors convey ownership; request colors convey lifecycle.

Before changing behavior, resolve the relevant source definitions and existing
animation model. Keep admission separate from batch selection, generation
separate from delivery, and Python objects separate from OS processes. Treat
tiny queues, page costs, thread examples and animation timing as illustrations.
Apply shared lesson changes to both readers; preserve fragment IDs.

For page changes, edit `src/` and rebuild through the task's builder. Select
the behavioral and browser checks described in the reference for the changed
lesson, then validate generated links and source spans. For requirement-only
changes, validate skill metadata and reference links; no page rebuild is needed.

When recording new decisions, distinguish explicit user requirements, accepted
implementation choices and source-backed behavior. Keep examples configurable
and preserve the existing guide and memory-planning instructions. Follow the
task's scoped Git workflow while retaining unrelated workspace changes.
